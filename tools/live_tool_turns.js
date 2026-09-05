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
global.Audio = function () {
  return { preload: '', muted: false, currentTime: 0,
           play: () => Promise.resolve(), pause: () => {} };
};
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
RIO.speak = { provider: () => ({ play: () => Promise.resolve(), stop: () => {} }) };

require(path.join(REPO, 'static', 'rio_speech.js'));
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

/* The relay socket, as the sink's `transport`: send() goes to python, and
   python pushes what came back into onmessage. */
const transport = {
  readyState: 1,
  send(text) { out({ k: 'wire', obj: JSON.parse(text) }); },
  close() {},
};
const sink = eleven.createSink({ transport, context: ctx, sampleRate: RATE,
                                 onEvent: (ev) => out({ k: 'note',
                                                        note: 'voice', ev }) });
sink.open().catch(() => {});

/* ---------------------------------------------------------------------------
   The controller, wired exactly as static/index.html wires it.
   --------------------------------------------------------------------------- */
const session = JSON.parse(process.argv[4] || '{}');
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
  audio: { mute: () => sink.mute(), unmute: () => sink.unmute() },
  voice: sink,
  cedarVoice: session.cedar_voice || 'cedar',
  verbatimInstruction: session.verbatim_instruction,
  resumeInstruction: session.resume_instruction,
  speakTimeoutMs: session.speak_timeout_ms,
  lookAnswerMaxTokens: session.look_answer_max_tokens,
  bargeSustainMs: session.barge_sustain_ms,
  bargeConfirmMs: session.barge_confirm_ms,
  maxResumes: session.max_resumes,
  onEvent: (ev) => out({ k: 'note', note: 'live', ev }),
});

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
      try { controller.handle(m.ev); }
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
            counters: controller.state().counters });
    } else if (m.k === 'bye') {
      process.exit(0);
    }
  }
});
