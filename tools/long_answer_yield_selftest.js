/* long_answer_yield_selftest.js — a long answer still gives way, cleanly.
 *
 *   node tools/long_answer_yield_selftest.js
 *
 * WHY THIS EXISTS. On 2026-09-28 her answers stopped being short by
 * construction. The character now governs length ("tell me everything" earns a
 * real answer), and the xAI voice session was measured to ignore
 * max_output_tokens entirely: asked for 50 tokens, she spoke for 267 s. xAI also
 * generates about five times faster than real time, so a three-minute answer is
 * two and a half minutes of audio QUEUED IN THE PAGE by the time the server says
 * response.done.
 *
 * Every yield path was written when an answer was a sentence or two. This drives
 * a three-minute answer through the REAL stack -- rio_xai_session's transport and
 * playout, the real controller, the real arbiter, rio_speak's nav lines -- with a
 * fake socket and a fake AudioContext whose clock the test owns, and asks:
 *
 *   1. the driver barges in 20 s into it          does the sound stop, all of it?
 *   2. a junction call lands 20 s into it         does it cut straight through?
 *   3. an earlier turn call (patient) lands       how long does it wait, and does
 *                                                 it take the mouth or expire?
 *   4. generation ends long before the sound      is the mouth held until the
 *                                                 SOUND ends, or given back while
 *                                                 two minutes of her are queued?
 *   5. the answer simply runs past 90 s           is it cut off by a watchdog?
 */
'use strict';

const path = require('path');
const STATIC = path.join(__dirname, '..', 'static');

let checks = 0, failures = 0;
function ok(cond, what) {
  checks++;
  if (!cond) failures++;
  console.log((cond ? '  ok    ' : '  FAIL  ') + what);
}
function section(n) { console.log('\n=== ' + n + ' ==='); }
const tick = (ms) => new Promise(r => setTimeout(r, ms || 0));

global.window = global;
global.document = {
  addEventListener() {}, getElementById() { return null; },
  querySelector() { return null; }, querySelectorAll() { return []; },
  createElement() {
    const el = {
      pause() {}, addEventListener() {}, removeEventListener() {}, style: {},
      muted: false, onended: null, onerror: null,
      set src(v) { this._src = v; setTimeout(() => { if (el.onended) el.onended(); }, 0); },
      get src() { return this._src; },
      play() { return Promise.resolve(); },
    };
    return el;
  },
};
if (typeof global.atob !== 'function') {
  global.atob = (b) => Buffer.from(b, 'base64').toString('binary');
}
if (typeof global.btoa !== 'function') {
  global.btoa = (s) => Buffer.from(s, 'binary').toString('base64');
}
if (typeof global.AbortController !== 'function') {
  global.AbortController = class { constructor() { this.signal = {}; } abort() {} };
}

const playoutMod = require(path.join(STATIC, 'rio_playout.js'));
const provider = require(path.join(STATIC, 'rio_provider.js'));
const speech = require(path.join(STATIC, 'rio_speech.js'));
const rt = require(path.join(STATIC, 'rio_realtime.js'));
const session = require(path.join(STATIC, 'rio_xai_session.js'));
require(path.join(STATIC, 'rio_speak.js'));

const RATE = session.RATE;
const ONE_SECOND = RATE * 2;                     // PCM16 mono
const ANSWER_S = 180;                            // three minutes of her
const CHUNK_S = 4;

function fakeWS() {
  const sent = [];
  function WS() {
    WS.last = this;
    setTimeout(() => { if (this.onopen) this.onopen(); }, 0);
  }
  WS.prototype.send = function (s) { sent.push(JSON.parse(s)); };
  WS.prototype.close = function () { if (this.onclose) this.onclose(); };
  WS.sent = sent;
  return WS;
}

/* An AudioContext whose clock the test owns, and which remembers every source
   it scheduled and whether it was stopped -- the only honest reading of "is
   she still making a sound" a node test can have. */
function fakeCtx() {
  let t = 100;
  const sources = [];
  return {
    get currentTime() { return t; },
    _advance: (s) => { t += s; },
    _sources: sources,
    destination: {},
    createGain: () => ({ gain: { value: 1, cancelScheduledValues() {},
                                 setValueAtTime() {}, linearRampToValueAtTime() {} },
                         connect() {} }),
    createBuffer: (ch, len) => ({
      duration: len / RATE, length: len,
      copyToChannel: () => {}, getChannelData: () => new Float32Array(len),
    }),
    createBufferSource: () => {
      const s = { buffer: null, connect() {}, onended: null,
                  started: null, stopped: false,
                  start(at) { this.started = at === undefined ? t : at; },
                  stop() { this.stopped = true; } };
      sources.push(s);
      return s;
    },
  };
}

/* The real stack, wired the way attach() wires it, minus the microphone. */
function stack(opts) {
  opts = opts || {};
  const WS = fakeWS();
  const ctx = fakeCtx();
  const events = [];
  const bus = [];
  const arbiter = speech.makeArbiter();
  const prov = provider.create('xai_voice');
  const mute = { muted: false };
  let s = null;
  const cfg = {
    arbiter,
    send: (o) => s.send(o),
    tool: () => Promise.resolve({ ok: true }),
    audio: { mute: () => { mute.muted = true; }, unmute: () => { mute.muted = false; } },
    onEvent: (e) => bus.push(e),
    bargeSustainMs: 4, bargeConfirmMs: 8,
    provider: prov,
    tailFallbackMs: opts.tailFallbackMs || 15000,
    // What the xAI wire hands the controller: how much of her is still queued.
    audioRemainingMs: () => {
      const left = s && s.playout ? s.playout.untilIdle() : null;
      return (left === null || !isFinite(left)) ? null : left * 1000;
    },
  };
  if (opts.answerWatchdogMs) cfg.answerWatchdogMs = opts.answerWatchdogMs;
  const controller = rt.createController(cfg);
  const armed = [];
  s = session.open({
    mint: {
      ws_url: 'wss://example.test/v1/realtime?model=m',
      ws_subprotocol: 'xai-client-secret.tok',
      session: { type: 'realtime', output_modalities: ['audio'],
                 audio: { output: { voice: 'Eve' } } },
      silence_tail_ms: 400,
    },
    controller, provider: prov, ctx, tailGraceS: 0,
    playoutLib: playoutMod, WebSocketImpl: WS,
    setIntervalImpl: (fn, ms) => { armed.push({ fn, ms }); return armed.length; },
    clearIntervalImpl: () => { armed.length = 0; },
    onEvent: (e) => events.push(e),
  });
  global.RIO.realtime = { active: () => ({
    speak: (t, o) => controller.speak(t, o),
    cancelSpeak: (tok, why) => controller.cancelSpeak(tok, why),
    speechEnabled: () => true,
    speakTimeout: () => 2500,
  }) };
  RIO.speak.reset();
  const h = {
    s, WS, ctx, events, bus, arbiter, controller, mute,
    msg: (o) => s._onMessage(JSON.stringify(o)),
    advance: (sec) => { ctx._advance(sec); armed.forEach(a => a.fn()); },
    audible: () => ctx._sources.filter(x => !x.stopped
                                       && x.started + x.buffer.duration > ctx.currentTime),
    creates: () => WS.sent.filter(e => e.type === 'response.create'),
    cancels: () => WS.sent.filter(e => e.type === 'response.cancel'),
  };
  return h;
}

/* The driver asks; she answers for three minutes. Generation finishes after all
   of it has been sent -- about 36 s of wall time at xAI's measured 5x, which on
   this clock is nothing at all: the whole answer is queued at once. */
async function longAnswer(h, o) {
  o = o || {};
  await h.s.connect();
  h.controller.handle({ type: 'input_audio_buffer.speech_started' });
  h.controller.handle({ type: 'input_audio_buffer.speech_stopped' });
  await tick(5);
  h.msg({ type: 'response.created', response: { id: 'r1' } });
  for (let i = 0; i < ANSWER_S / CHUNK_S; i++) {
    h.msg({ type: 'response.output_audio_transcript.delta', response_id: 'r1',
            delta: 'The Roman Republic became an empire when ' });
    h.msg({ type: 'response.output_audio.delta', response_id: 'r1',
            delta: Buffer.alloc(ONE_SECOND * CHUNK_S).toString('base64') });
  }
  if (o.done !== false) {
    h.msg({ type: 'response.done', response: { id: 'r1', status: 'completed' } });
  }
  await tick(5);
}

function navLine(h, o) {
  const src = RIO.speak.provider({
    text: o.text, channel: 'nav', callType: o.callType,
    element: document.createElement('audio'),
  });
  const rec = { reason: null, at: Date.now(), startedAt: null };
  h.arbiter.say({
    priority: o.callType === 'junction' ? h.arbiter.P.TURN_NEAR : h.arbiter.P.NAV,
    patient: o.callType !== 'junction',
    group: 'nav:m1', id: 'nav:m1:' + o.callType, text: o.text,
    ttlMs: o.ttlMs || 6000,
    play: () => { rec.startedAt = Date.now(); return src.play(); },
    stop: src.stop,
    onDone: (r) => { rec.reason = r; },
  });
  return rec;
}

(async function main() {
  const liveFor = (h) => h.arbiter.state().speaking;

  section('1. the driver barges in, 20 s into three minutes');
  {
    const h = stack();
    await longAnswer(h);
    h.advance(20);
    ok(h.audible().length > 0, `she is audible 20 s in, with ${Math.round(
       h.s.playout.untilIdle())} s of her still queued`);
    h.controller.handle({ type: 'input_audio_buffer.speech_started' });
    await tick(40);                 // past sustain (4) and confirm (8)
    ok(h.cancels().length > 0, 'the answer is cancelled at the server');
    ok(h.audible().length === 0,
       'and EVERY queued second of her stops, not just the one playing — '
       + `${h.ctx._sources.filter(x => x.stopped).length} sources stopped`);
    ok(h.s.playout.idle(), 'the playout queue is empty, so nothing waits behind it');
    // xAI keeps streaming until the cancel lands.
    const before = h.ctx._sources.length;
    h.msg({ type: 'response.output_audio.delta', response_id: 'r1',
            delta: Buffer.alloc(ONE_SECOND * CHUNK_S).toString('base64') });
    const late = h.ctx._sources.slice(before).filter(x => !x.stopped);
    ok(late.length === 0 || h.mute.muted,
       `audio that arrives after the barge is not heard (${late.length} new `
       + `source(s), mouth ${h.mute.muted ? 'muted' : 'open'})`);
    ok(h.s.playout.idle(), '...and does not refill the queue');
  }

  section('2. a junction call lands 20 s into it');
  {
    const h = stack();
    await longAnswer(h);
    h.advance(20);
    const createsBefore = h.creates().length;
    const t0 = Date.now();
    const nav = navLine(h, { text: 'Turn left onto Sunset.', callType: 'junction' });
    await tick(5);
    ok(liveFor(h) && /junction$/.test(liveFor(h).id),
       `the junction call takes the mouth at once (${Date.now() - t0} ms)`);
    ok(h.audible().length === 0 || h.mute.muted,
       'and her queued minutes are silenced under it');
    ok(h.s.playout.idle(), 'the queue is flushed, so the dictation is not gated');
    const dict = h.creates().slice(createsBefore).filter(e =>
      /Turn left onto Sunset/.test((e.response || {}).instructions || ''));
    ok(dict.length === 1,
       'the turn call goes out on the wire immediately, not held behind her audio');
    // Late audio for the cancelled answer must not play under the turn call.
    const before = h.ctx._sources.length;
    h.msg({ type: 'response.output_audio.delta', response_id: 'r1',
            delta: Buffer.alloc(ONE_SECOND).toString('base64') });
    ok(h.ctx._sources.slice(before).every(x => x.stopped) || h.mute.muted,
       'stragglers from the cancelled answer are not heard under the call');
    ok(nav.reason === null, 'the call has not been dropped');
  }

  section('3. an earlier turn call (patient) lands 20 s into it');
  {
    const h = stack();
    await longAnswer(h);
    h.advance(20);
    const t0 = Date.now();
    const nav = navLine(h, { text: 'In half a mile, turn left onto Sunset.',
                             callType: 'prepare', ttlMs: 6000 });
    await tick(20);
    ok(liveFor(h) && /^live:/.test(liveFor(h).id),
       'it waits: an early call does not cut her off mid-sentence');
    await tick(speech.PATIENT_MAX_WAIT_MS + 200);
    const waited = nav.startedAt ? nav.startedAt - t0 : null;
    ok(nav.startedAt !== null,
       `...but not for three minutes: it takes the mouth after ${waited} ms `
       + `(the patience bound is ${speech.PATIENT_MAX_WAIT_MS} ms)`);
    ok(nav.reason !== 'expired',
       'and it is not dropped as expired for having waited — its 6 s TTL is '
       + 'credited for the time it was told to wait');
    ok(h.audible().length === 0 || h.mute.muted,
       'her answer is silenced when it does');
  }

  section('4. generation ends long before the sound does');
  {
    // tailFallbackMs shortened so the test does not sit for 15 real seconds;
    // what is asserted is the RELATION between it and the queued audio.
    const h = stack({ tailFallbackMs: 60 });
    await longAnswer(h);             // response.done arrives with 180 s queued
    h.advance(20);
    await tick(200);                 // well past the tail fallback
    ok(liveFor(h) && /^live:/.test(liveFor(h).id),
       'the mouth is still hers while 160 s of her answer are queued — the tail '
       + 'fallback counts from the end of the SOUND, not from response.done');
    ok(!h.bus.some(e => e.type === 'LIVE_TAIL_TIMEOUT'),
       'and no tail timeout is reported for audio that is simply still playing');
    const nav = navLine(h, { text: 'In half a mile, turn left.', callType: 'prepare' });
    await tick(20);
    ok(!(liveFor(h) && /prepare$/.test(liveFor(h).id)),
       'so an early turn call waits for her instead of starting on top of her');
    ok(nav.reason === null, '...and is still queued, not dropped');
  }

  section('5. an answer that simply runs long is not cut off by a watchdog');
  {
    const h = stack();
    await longAnswer(h, { done: false });
    const wd = h.controller.state().answer_watchdog_ms;
    ok(wd >= 5 * 60 * 1000,
       `the conversational watchdog is ${wd === undefined ? 'not exposed' : wd / 1000 + ' s'}`
       + ' — above the longest single response measured (323 s); it was 90 s, '
       + 'which would have cut this three-minute answer at the half');
  }

  console.log(failures ? `\nFAILED ${failures}/${checks} checks`
                       : `\nPASSED ${checks} checks`);
  process.exit(failures ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
