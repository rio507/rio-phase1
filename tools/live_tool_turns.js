/* live_tool_turns.js — the browser half of a real drive, outside a browser.
 *
 * Driven by tools/live_tool_turns.py, which owns the two sockets this cannot
 * open in node 18: the live session and the dialogue relay. Everything on the
 * PAGE'S side of those sockets is the real thing and is loaded from static/ --
 * the controller that decides what to say, the sink that turns words into
 * scheduled audio, the navigation panel that answers the route tools, and the
 * real /realtime/tool endpoint over HTTP for the ones the server owns.
 *
 * It exists because three live failures were all in the space between a tool
 * result and a sound, and nothing could reach that space. The unit tests drive
 * the controller with events a test wrote; the recording tool (voice_demo.py)
 * drives the real session with a REIMPLEMENTATION of the controller, which is
 * how it once had the very bug it was recording. This runs the shipped code
 * against the shipped server and writes down what came out of the speaker.
 *
 * TWO MOUTHS, ONE RECORDER. Under VOICE_BACKEND=elevenlabs the session writes
 * and the dialogue sink speaks, so the audio arrives on the relay socket.
 * Under openai_realtime the session SPEAKS -- audio comes back on the session
 * socket itself, as `response.output_audio.delta` -- and there is no relay at
 * all. Which one is live is read off the minted session, never assumed, and
 * both end up in the same recorder on the same clock, so the WAV means the
 * same thing either way: what a listener would have heard, when.
 *
 * Line-delimited JSON on stdin and stdout. In:
 *   {k:'ev', ev}          one event off the live session
 *   {k:'wire', m}         one message off the dialogue relay
 *   {k:'gps', lat, lng}   where the car is
 *   {k:'wav', path}       write everything heard so far and say so
 * Out:
 *   {k:'send', obj}       something for the live session
 *   {k:'wire', obj}       something for the relay
 *   {k:'note', ...}       an observation worth putting in the report
 */
'use strict';

const path = require('path');
const fs = require('fs');
const REPO = path.join(__dirname, '..');

const base = process.argv[2];
const sessionId = process.argv[3];

function out(o) { process.stdout.write(JSON.stringify(o) + '\n'); }

/* ---------------------------------------------------------------------------
   A page, enough of one for the panel to load into.
   --------------------------------------------------------------------------- */
global.window = global;
const stubNode = () => ({
  style: {}, textContent: '', innerHTML: '', appendChild: () => {},
  querySelector: () => stubNode(), addEventListener: () => {},
  setAttribute: () => {},
});
const docHandlers = {};
global.document = {
  addEventListener: (t, f) => { (docHandlers[t] = docHandlers[t] || []).push(f); },
  getElementById: () => null, createElement: stubNode,
  createTextNode: () => ({}), head: { appendChild: () => {} },
};
/* THE ELEMENT THE DETERMINISTIC LINES ARE PLAYED THROUGH, and it has to be a
   real speaker rather than a stub.
 *
 * Under the ElevenLabs backend RIO does NOT dictate a turn call to the live
 * session — realtime.py turns that off deliberately, because /nav/voice
 * already synthesises on the same voice id and is faster to first byte. The
 * call therefore arrives here, as MP3, through an <audio> element. Stubbed
 * out, every navigation line in the drive is silently discarded and the WAV
 * shows a car that never said anything, which is the exact failure this
 * recording exists to disprove.
 *
 * So the element decodes what it is given (ffmpeg, to the same rate the sink
 * runs at) and hands it to the same recorder the conversation goes through,
 * at the moment playback started. One timeline, both voices, in the order a
 * listener would have heard them. */
const blobs = new Map();
let blobSeq = 0;
/* THE TWO BLOB METHODS, ADDED TO `URL` RATHER THAN INSTEAD OF IT.
 *
 * This used to REPLACE global.URL with an object carrying only these two
 * functions, and on a node without a global `fetch` that was harmless: the
 * shim below builds its own `URL` out of require('url'), so nothing noticed.
 *
 * Node 18.19 HAS a global fetch, so the shim is not installed and undici's is
 * used -- and undici resolves every request through `new URL(...)`. Against a
 * plain object that is "URL is not a constructor", which undici reports as
 * `TypeError: Failed to parse URL from http://127.0.0.1:8888/...` with the
 * real cause one level down. What that looked like from the report was RIO
 * failing every tool in the drive: six turns, six `unreachable`s, and her
 * saying she could not see the camera or reach the car. None of it was hers.
 *
 * So the constructor is kept and the two browser-only methods are hung off a
 * subclass of it. A stub that removes a standard global is not a stub, it is a
 * different runtime. */
class HarnessURL extends URL {}
HarnessURL.createObjectURL = function (blob) {
  const key = 'blob:' + (++blobSeq);
  blobs.set(key, blob && blob.__bytes);
  return key;
};
HarnessURL.revokeObjectURL = function (key) { blobs.delete(key); };
global.URL = HarnessURL;
global.AbortController = global.AbortController || function () {
  return { signal: null, abort() {} };
};

/* AN ELEMENT THAT CAN BE PRELOADED, because one of them now is.
 *
 * The old stub decoded on `src =` and emitted the audio there, with play() a
 * no-op. That models a blob being fetched and played once, which is what every
 * caller did -- and it models a PRELOADED element as silence, because a
 * preloaded element is never given a new src: it is rewound and played. The
 * imminent turn call is exactly that now, so a harness with the old stub would
 * have recorded the junction as silent and sent this back for a fix that was
 * not needed.
 *
 * So loading and playing are separate here, as they are in a browser. `src =`
 * decodes and holds; play() emits what is held, at the moment it was asked
 * for. The gap between those two is the thing worth measuring, and it is
 * reported as `wait_ms`: on a preloaded clip it is zero, and on a blob that is
 * still being fetched it is the fetch.
 */
global.Audio = function (initialSrc) {
  const el = {
    preload: '', muted: false, currentTime: 0,
    onended: null, onerror: null,
    pause: () => {},
  };
  let src = '';
  let decoded = null;            // Promise<Float32Array|null>, once loaded

  function decode(bytes) {
    return new Promise((resolve) => {
      require('child_process').execFile(
        'ffmpeg', ['-hide_banner', '-loglevel', 'error', '-i', 'pipe:0',
                   '-f', 's16le', '-ar', String(RATE), '-ac', '1', 'pipe:1'],
        { encoding: 'buffer', maxBuffer: 64 * 1024 * 1024 },
        (err, pcm) => {
          if (err || !pcm || !pcm.length) return resolve(null);
          const n = pcm.length >> 1;
          const data = new Float32Array(n);
          for (let i = 0; i < n; i++) data[i] = pcm.readInt16LE(i * 2) / 32768;
          resolve(data);
        }).stdin.end(bytes);
    });
  }

  function load(v) {
    // A blob the page just fetched...
    const bytes = blobs.get(v);
    if (bytes && bytes.length) { decoded = decode(bytes); return; }
    // ...or a file the page is serving, which is what a preloaded clip holds.
    const m = /\/static\/(.+)$/.exec(String(v || ''));
    if (m) {
      try {
        decoded = decode(fs.readFileSync(path.join(REPO, 'static', m[1])));
        return;
      } catch (e) { /* falls through to the null below */ }
    }
    decoded = Promise.resolve(null);
  }

  Object.defineProperty(el, 'src', {
    get: () => src,
    set(v) { src = v; load(v); },
  });

  el.play = () => {
    if (!decoded) {
      setTimeout(() => { if (el.onerror) el.onerror(); }, 0);
      return Promise.resolve();
    }
    // WHEN a listener started hearing it, taken before the decode is awaited
    // so the time ffmpeg costs this harness does not appear in the recording
    // as RIO being late. `wait_ms` reports it separately instead.
    const at = ctx.currentTime;
    return decoded.then((data) => {
      if (!data) {
        out({ k: 'note', note: 'nav_audio_failed', src: src });
        if (el.onerror) el.onerror();
        return;
      }
      heard.push({ at: at, buf: { length: data.length,
                                  getChannelData: () => data } });
      out({ k: 'note', note: 'nav_audio', at: at, src: src,
            seconds: data.length / RATE,
            // Zero on an element that was already holding this file, which is
            // the whole claim the preloading makes.
            wait_ms: Math.round((ctx.currentTime - at) * 1000) });
      if (el.onended) el.onended();
    });
  };

  /* THE ATTRIBUTE HALF OF AN AUDIO ELEMENT, and it is here because leaving it
     out cost a whole navigation acceptance run.
     rio_nav.js:unlockOne() saves the real clip off the element, swaps in the
     silent buffer to unlock it on iOS, and puts the clip back:
         var real = a.getAttribute('src') || '';
         ... a.src = silent; a.load(); ... a.src = real; a.load();
     A stub with only a `src` PROPERTY throws "a.getAttribute is not a
     function" on the first line of that, inside the panel's own route tool, so
     start_navigation came back ok=false with a DOM TypeError for a note and
     the recording caught RIO saying "it didn't start. The route failed to
     load." That is the harness failing, not the car: the same unlock path is
     green in a real browser (tools/output_bus_selftest.py, 27/27).
     Backed by the same `src` state as the property, so the two cannot
     disagree -- which is the bug a second copy would have introduced. */
  el.getAttribute = (name) => (String(name).toLowerCase() === 'src' ? (src || null) : null);
  el.setAttribute = (name, v) => {
    if (String(name).toLowerCase() === 'src') { src = String(v); load(src); }
  };
  el.removeAttribute = (name) => {
    if (String(name).toLowerCase() === 'src') { src = ''; decoded = null; }
  };
  // load() re-reads whatever src currently holds, exactly as a browser does.
  // Without it the restore half of unlockOne silently keeps the silent buffer
  // on the element and the junction call is a clip of nothing.
  el.load = () => { if (src) load(src); };

  if (initialSrc) { src = initialSrc; load(initialSrc); }
  return el;
};
/* THE ONE BROWSER GLOBAL THE SINK CANNOT DO WITHOUT. Audio arrives off the
   relay base64-encoded and rio_voice_eleven decodes it with atob, which node
   12 does not have. Without this the sink throws on the FIRST chunk of speech
   and takes the whole node process down with it -- and since node is where
   every decision in this harness is made, the drive carries on with nobody
   answering: one turn spoke and the remaining five read as silent, which is
   indistinguishable in the report from the product failure this file exists
   to catch. */
global.atob = (b64) => Buffer.from(b64, 'base64').toString('binary');

/* ...AND THE OTHER ONE, for the same reason and with a worse disguise. Every
   tool in this harness reaches the server through fetch — the panel's routing
   call, the camera, the reasoning model — and node 12 has none. Without this
   a local tool throws before it does anything and the catch below files it as
   'panel error', which reads in the report as RIO's route failing rather than
   as this file never having asked for one. */
/* THIS SHIM, ALWAYS — not only when node has no `fetch` of its own.
 *
 * The condition used to be `if (typeof global.fetch !== 'function')`, which on
 * node 18 means never, and that quietly changed what a `blob()` is. The
 * fallback voice path is `fetch('/nav/voice') -> r.blob() ->
 * URL.createObjectURL(blob)`, and the object store two hundred lines up reads
 * `blob.__bytes` because that is what this shim returns. Node's own fetch
 * returns a real Blob, which has no `__bytes`, so every fallback line was
 * fetched, decoded to nothing and played as silence -- a nav drive reporting
 * `tts: 8, silent: 8` while the server had synthesised all eight.
 *
 * The browser has one `fetch` and one `Blob` and they agree with each other.
 * This file has to supply both halves or neither, so it supplies both. */
{
  const http = require('http');
  const { URL } = require('url');
  global.fetch = function (url, opts) {
    opts = opts || {};
    return new Promise((resolve, reject) => {
      // RELATIVE PATHS RESOLVE AGAINST THE PAGE, and in a browser that is
      // free. rio_speak asks for '/nav/voice?...' exactly as the page does;
      // node's URL throws on it, the fetch rejects, and a turn call comes out
      // as one more line the recorder never got — which is how seven of them
      // were lost while every event said they had been spoken.
      const u = new URL(url, base);
      const req = http.request({
        hostname: u.hostname, port: u.port, path: u.pathname + u.search,
        method: opts.method || 'GET', headers: opts.headers || {},
      }, (res) => {
        let body = '';
        const chunks = [];
        res.on('data', (d) => { body += d; chunks.push(Buffer.from(d)); });
        res.on('end', () => resolve({
          ok: res.statusCode < 400, status: res.statusCode,
          headers: { get: (k) => res.headers[String(k).toLowerCase()] || null },
          text: () => Promise.resolve(body),
          json: () => Promise.resolve(JSON.parse(body)),
          // The synthesised line comes back as MP3, so this one keeps bytes.
          blob: () => Promise.resolve({ __bytes: Buffer.concat(chunks) }),
        }));
      });
      req.on('error', reject);
      if (opts.body) req.write(opts.body);
      req.end();
    });
  };
}
const gps = { sink: null, at: { lat: 34.0219, lng: -118.4814 } };
global.navigator = {
  geolocation: {
    getCurrentPosition: (okc) => okc({ coords: {
      latitude: gps.at.lat, longitude: gps.at.lng, accuracy: 8 } }),
  },
};
const RIO = global.RIO || (global.RIO = {});
RIO.sessionId = sessionId;
RIO.url = (p) => base + p + (p.indexOf('?') >= 0 ? '&' : '?') +
                 'session_id=' + encodeURIComponent(sessionId);
RIO.headway = { startWatch: () => {}, onPosition: (fn) => { gps.sink = fn; } };
/* THE REAL rio_speak.js, not a stub.
 *
 * A turn call is not synthesised on its own any more — it is DICTATED to the
 * live session, word for word, and comes out of the same mouth as the
 * conversation. Stubbing that out was fine while this file only tested tool
 * turns; it makes navigation inaudible, which is the one thing a recording of
 * a navigating drive is for. RIO.realtime.active() is set further down, once
 * there is a controller for it to answer with. */

require(path.join(REPO, 'static', 'rio_speech.js'));
require(path.join(REPO, 'static', 'rio_speak.js'));
require(path.join(REPO, 'static', 'rio_navcore.js'));
require(path.join(REPO, 'static', 'rio_navplan.js'));
require(path.join(REPO, 'static', 'rio_nav.js'));
const rt = require(path.join(REPO, 'static', 'rio_realtime.js'));
const eleven = require(path.join(REPO, 'static', 'rio_voice_eleven.js'));
(docHandlers.DOMContentLoaded || []).forEach((f) => f());

/* ---------------------------------------------------------------------------
   A LOUDSPEAKER THAT WRITES DOWN WHAT IT PLAYED.
   The sink schedules buffers against a context clock; this is that context,
   on a real clock, keeping every buffer at the offset it was scheduled for.
   The WAV it writes is therefore not "the audio that was generated" -- it is
   the audio in the order and at the times a listener would have heard it,
   which is the only version that can show a gap.
   --------------------------------------------------------------------------- */
const RATE = 24000;
const heard = [];               // { at, data: Float32Array }
const t0 = Date.now();
const gainNode = {
  gain: { value: 0, cancelScheduledValues() {}, setValueAtTime() {},
          linearRampToValueAtTime() {} },
  connect() {},
};
const ctx = {
  get currentTime() { return (Date.now() - t0) / 1000; },
  state: 'running', destination: {},
  createGain: () => gainNode,
  createBuffer(channels, length, rate) {
    const data = new Float32Array(length);
    return { duration: length / rate, length, sampleRate: rate,
             numberOfChannels: channels, getChannelData: () => data };
  },
  createBufferSource() {
    const src = { buffer: null, connect() {},
                  start(at) { heard.push({ at, buf: src.buffer }); },
                  stop() { src.stopped = true; } };
    return src;
  },
  resume() {}, close() {},
};

function writeWav(file) {
  if (!heard.length) { fs.writeFileSync(file, Buffer.alloc(0)); return 0; }
  const end = Math.max.apply(null, heard.map(
    (h) => h.at + h.buf.length / RATE));
  const total = Math.max(1, Math.ceil(end * RATE));
  const mix = new Float32Array(total);
  for (const h of heard) {
    const off = Math.max(0, Math.round(h.at * RATE));
    const d = h.buf.getChannelData(0);
    for (let i = 0; i < d.length && off + i < total; i++) mix[off + i] += d[i];
  }
  const pcm = Buffer.alloc(total * 2);
  for (let i = 0; i < total; i++) {
    let v = Math.max(-1, Math.min(1, mix[i]));
    pcm.writeInt16LE(Math.round(v * 32767), i * 2);
  }
  const head = Buffer.alloc(44);
  head.write('RIFF', 0); head.writeUInt32LE(36 + pcm.length, 4);
  head.write('WAVE', 8); head.write('fmt ', 12);
  head.writeUInt32LE(16, 16); head.writeUInt16LE(1, 20);
  head.writeUInt16LE(1, 22); head.writeUInt32LE(RATE, 24);
  head.writeUInt32LE(RATE * 2, 28); head.writeUInt16LE(2, 32);
  head.writeUInt16LE(16, 34); head.write('data', 36);
  head.writeUInt32LE(pcm.length, 40);
  fs.writeFileSync(file, Buffer.concat([head, pcm]));
  return total / RATE;
}

/* The session the browser was given, and the one thing about it that decides
   how this file behaves: whose mouth RIO has. */
const session = JSON.parse(process.argv[4] || '{}');
const textMode = (session.output_modalities || ['audio'])[0] === 'text';

/* The relay socket, as the sink's `transport`: send() goes to python, and
   python pushes what came back into onmessage. */
const transport = {
  readyState: 1,
  send(text) { out({ k: 'wire', obj: JSON.parse(text) }); },
  close() {},
};
const sink = textMode
  ? eleven.createSink({ transport, context: ctx, sampleRate: RATE,
                        onEvent: (ev) => out({ k: 'note', note: 'voice', ev }) })
  : null;
if (sink) sink.open().catch(() => {});

/* ---------------------------------------------------------------------------
   THE SPEECH-TO-SPEECH MOUTH.

   In the car this is an <audio> element carrying a WebRTC track: the session
   speaks, the browser plays it, and muting the element is how RIO is silenced
   the instant the driver starts talking. Over a WebSocket there is no track --
   the same audio arrives as base64 PCM on the session socket -- so this is the
   speaker, and it writes down what it played for the same reason the sink
   does.

   A SPEAKER, NOT A BUFFER, and that is the whole reason for `playHead`. The
   model generates faster than real time: a four-second answer lands in about a
   second. Stamping each chunk at the moment it ARRIVED would compress the
   answer into a third of its length and make every gap in the recording look
   smaller than it was. So chunks queue behind each other exactly as they would
   in a speaker, and the recording runs at the speed a person would hear.

   MUTED AUDIO IS NOT RECORDED, for the same reason: a muted element is silence
   in the cabin. The clock still advances through it, because the session is
   still speaking and the element is still playing -- it is just inaudible.
   That is the difference between a barge-in (heard as RIO stopping) and a
   cancel (heard as RIO stopping AND the rest never existing), and a recording
   that could not tell them apart would be no use for the thing this exists
   for. */
let muted = false;
let playHead = 0;
const audioMouth = {
  mute() { muted = true; },
  unmute() { muted = false; },
  push(b64) {
    const raw = Buffer.from(b64 || '', 'base64');
    const n = raw.length >> 1;
    if (!n) return;
    const at = Math.max(ctx.currentTime, playHead);
    playHead = at + n / RATE;
    if (muted) return;
    const data = new Float32Array(n);
    for (let i = 0; i < n; i++) data[i] = raw.readInt16LE(i * 2) / 32768;
    heard.push({ at, buf: { length: n, getChannelData: () => data } });
  },
};
const mouth = sink || audioMouth;

/* ---------------------------------------------------------------------------
   The controller, wired exactly as static/index.html wires it.
   --------------------------------------------------------------------------- */
const controller = rt.createController({
  arbiter: require(path.join(REPO, 'static', 'rio_speech.js')).makeArbiter(),
  send: (obj) => out({ k: 'send', obj }),
  tool: (name, args) => {
    const t = Date.now();
    const finish = (r) => {
      out({ k: 'note', note: 'tool', name, args, ok: !!(r && r.ok),
            path: (r && r.path) || null, direct: !!(r && r.speak_directly),
            note_text: (r && (r.note || r.error)) || null,
            speech: (r && (r.speech || r.scene)) || null,
            ms: Date.now() - t });
      return r;
    };
    if (rt.localTools && rt.localTools[name]) {
      /* `.then(finish)`, not `finish(...)`: startNavigation returns a PROMISE
         of a route and the others return objects, and reporting on the
         promise says every route failed in one millisecond. The controller
         was always getting the right result -- this line was the part that
         lied about it. */
      try { return Promise.resolve(rt.localTools[name](args)).then(finish); }
      catch (e) { return Promise.resolve(finish({ ok: false, note: 'panel error' })); }
    }
    return fetch(RIO.url('/realtime/tool'), {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, arguments: args,
                             where: { lat: gps.at.lat, lng: gps.at.lng,
                                      accuracy_m: 8, age_s: 0.2 },
                             spoken: controller.state().spoken_this_turn || '' }),
    }).then((r) => r.json()).then(finish)
      .catch(() => finish({ ok: false, note: 'unreachable' }));
  },
  audio: { mute: () => mouth.mute(), unmute: () => mouth.unmute() },
  // `voice` is what puts the controller in text mode. Absent under
  // openai_realtime, which is what makes it forward audio instead of words.
  voice: sink,
  // The voice the SESSION carried, and no default behind it: a harness that
  // quietly substitutes a voice is a harness that would have recorded the
  // cedar/marin mix-up as a clean drive.
  liveVoice: session.live_voice,
  // The tools the session ships with, and the ones that wait for a route.
  // Without these the recording is of a drive where "avoid the freeway" had
  // no tool to reach for -- which is what the first one was.
  toolSchemas: session.tool_schemas,
  conditionalTools: session.conditional_tools,
  verbatimInstruction: session.verbatim_instruction,
  resumeInstruction: session.resume_instruction,
  speakTimeoutMs: session.speak_timeout_ms,
  directSpeechTimeoutMs: session.direct_speech_timeout_ms,
  lookAnswerMaxTokens: session.look_answer_max_tokens,
  bargeSustainMs: session.barge_sustain_ms,
  bargeConfirmMs: session.barge_confirm_ms,
  maxResumes: session.max_resumes,
  onEvent: (ev) => out({ k: 'note', note: 'live', ev }),
});

/* WHAT rio_speak.js LOOKS FOR. index.html publishes the same two functions on
   the same name; a navigation call finds them here and is spoken by the
   session rather than fetched from /nav/voice, which is what makes it her
   voice and what puts it in the WAV. */
RIO.realtime = {
  active: () => ({
    speak: (text, o) => controller.speak(text, o),
    /* THE PAGE'S OWN RULE, copied rather than simplified. Under the
       ElevenLabs backend the server sets speech_enabled false — a turn call
       is synthesised by /nav/voice on the same voice id instead of being
       dictated through a socket built for conversation — and a harness that
       says "yes" here would be recording a path the car does not take. */
    speechEnabled: (channel) => {
      if (session.speech_enabled === false) return false;
      const chans = session.speech_channels || {};
      return chans[channel] !== false;
    },
    /* ...and the same per-line budget the page resolves. Copied rather than
       simplified, for the same reason speechEnabled is: a harness that made
       up its own deadline would be recording a path the car does not take. */
    speakTimeout: (channel, callType) => {
      const table = session.speak_timeout_ms_by_channel || {};
      const byCall = table[channel];
      if (!byCall) return session.speak_timeout_ms;
      return byCall[callType] || byCall._default || session.speak_timeout_ms;
    },
  }),
};

/* ...and the tools that only exist while a route does, attached and taken
   away by the same route events the panel uses. No channel to wait for here:
   `send` is a pipe to python and is open before any of this runs. */
controller.watchToolConditions(RIO.bus);

/* Every navigation call, as the driver would have heard it ordered. */
RIO.bus.on('*', (ev) => {
  if (/^NAV_/.test(ev.type) && ev.type !== 'NAV_PROGRESS') {
    out({ k: 'note', note: 'nav', ev: ev });
  }
});

/* ---------------------------------------------------------------------------
   ONE EVENT OFF THE SESSION, INTO THE PAGE.

   Two things happen here that do not happen in a browser, and both are
   TRANSPORT differences rather than behaviour ones -- the same difference this
   file's header already documents for `output_audio_buffer.clear`.

   THE AUDIO. In the car RIO's voice arrives on a WebRTC media track and never
   passes through the controller at all; the browser plays the track and the
   controller only ever sees events. Over a WebSocket the audio IS an event, so
   it is peeled off here and handed to the speaker. The controller is given the
   event too, unchanged, because it is entitled to see everything the session
   sent.

   THE START AND STOP OF A SPOKEN LINE. `output_audio_buffer.started` and
   `.stopped` are emitted by the server for WebRTC sessions ONLY. The
   controller uses `started` for exactly one thing and it is not cosmetic: it
   is how a DICTATED line -- a turn call, a headway warning -- learns that it
   is actually being spoken, so the fallback armed against it never starting
   can stand down. Without that event every dictated line in this harness would
   sit out its whole budget and then be re-spoken by the synthesiser, and the
   recording would show a drive where dictation never worked. That would be a
   fault entirely in this file, and the last time this harness invented one of
   those it was believed for a week.

   So the two events are SYNTHESISED from the audio, and only if the real ones
   did not arrive -- if the transport ever starts sending them, the real ones
   win and this does nothing. Which happened is reported, because "the bridge
   was used" and "the bridge was not needed" are different facts about the
   run and the report should not have to guess.
   --------------------------------------------------------------------------- */
const bridged = { started: 0, stopped: 0, real: 0 };
const speaking = new Set();

function forward(ev) {
  const type = ev && ev.type;

  if (type === 'output_audio_buffer.started'
      || type === 'output_audio_buffer.stopped') {
    bridged.real++;
    controller.handle(ev);
    return;
  }

  if (type === 'response.output_audio.delta') {
    if (!speaking.has(ev.response_id)) {
      speaking.add(ev.response_id);
      bridged.started++;
      controller.handle({ type: 'output_audio_buffer.started',
                          response_id: ev.response_id });
    }
    mouth.push && mouth.push(ev.delta);
    controller.handle(ev);
    return;
  }

  if (type === 'response.done') {
    const id = (ev.response && ev.response.id) || ev.response_id;
    /* BEFORE `response.done`, not after. The controller treats
       `output_audio_buffer.stopped` and `response.done` as the same ending and
       finishes the response on whichever arrives first, so a stop sent
       afterwards would land on a response that is already closed and be
       dropped -- and a dictation would be finished by the wrong branch. */
    if (speaking.delete(id)) {
      bridged.stopped++;
      controller.handle({ type: 'output_audio_buffer.stopped',
                          response_id: id });
    }
    controller.handle(ev);
    return;
  }

  controller.handle(ev);
}

/* ---------------------------------------------------------------------------
   stdin
   --------------------------------------------------------------------------- */
let buf = '';
process.stdin.on('data', (d) => {
  buf += d;
  let i;
  while ((i = buf.indexOf('\n')) >= 0) {
    const line = buf.slice(0, i); buf = buf.slice(i + 1);
    if (!line.trim()) continue;
    let m;
    try { m = JSON.parse(line); } catch (e) { continue; }
    if (m.k === 'ev') {
      try { forward(m.ev); }
      catch (e) { out({ k: 'note', note: 'handler_threw',
                        error: String(e && e.message), type: m.ev && m.ev.type }); }
    } else if (m.k === 'wire') {
      if (transport.onmessage) transport.onmessage({ data: JSON.stringify(m.m) });
    } else if (m.k === 'gps') {
      gps.at = { lat: m.lat, lng: m.lng };
      if (gps.sink) gps.sink({ coords: { latitude: m.lat, longitude: m.lng,
                                         accuracy: 8 } });
    } else if (m.k === 'heard') {
      out({ k: 'note', note: 'audio_so_far',
            seconds: heard.reduce((a, h) => a + h.buf.length / RATE, 0),
            first_at: heard.length ? heard[0].at : null,
            counters: controller.state().counters });
    } else if (m.k === 'mark') {
      out({ k: 'note', note: 'mark', label: m.label,
            at: (Date.now() - t0) / 1000,
            audio_chunks: heard.length,
            counters: controller.state().counters });
    } else if (m.k === 'wav') {
      out({ k: 'note', note: 'wav', path: m.path, seconds: writeWav(m.path),
            mouth: textMode ? 'elevenlabs_sink' : 'live_session',
            bridged: bridged,
            counters: controller.state().counters });
    } else if (m.k === 'sim') {
      const nav = RIO.nav;
      if (m.on) {
        // A near-real-time tick on purpose. The simulator's clock is what
        // every TTL and cooldown downstream is measured against, so running
        // it faster than the wall clock would expire each line while its
        // audio was still being synthesised -- a recording of a drive with
        // no turns called in it, produced by the harness rather than by RIO.
        nav.simulate({ tickMs: m.tickMs || 1000, mph: m.mph || 25 });
      } else if (nav.stopSimulation) {
        nav.stopSimulation();
      }
      out({ k: 'note', note: 'sim', on: !!m.on });
    } else if (m.k === 'nav_state') {
      const nav = RIO.nav;
      const st = nav.state ? nav.state() : null;
      out({ k: 'note', note: 'nav_state',
            routing: !!nav.route,
            destination: nav.route ? (nav.route.destination.display_name || '') : null,
            generation_id: nav.route ? nav.route.generation_id : null,
            route_id: nav.route ? nav.route.route_id : null,
            remaining_m: st ? Math.round(st.remaining_m || 0) : null,
            tools: controller.state().tools || null,
            speak_stats: RIO.speak.stats ? RIO.speak.stats() : null });
    } else if (m.k === 'warm') {
      /* THE WARM-UP, and it waits for the socket for the same reason the page
         waits for the data channel. Called at module load -- which is what the
         first version did -- the create is written to stdout before python has
         connected the session socket, pump_node drops it on `and self.oai`,
         and the warm-up silently never happens. A warm that did not happen is
         worse than none: it reads in the numbers as one that did. */
      controller.warmSession();
    } else if (m.k === 'bye') {
      process.exit(0);
    }
  }
});
