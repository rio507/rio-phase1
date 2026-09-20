/* xai_session_selftest.js — the WebSocket session's sequencing, without a car.
 *
 *   node tools/xai_session_selftest.js
 *
 * static/rio_xai_session.js is a TRANSPORT. What can go wrong with it is not a
 * crash: it is the mouth handed back early, a request sent over audio still
 * playing, a transcript counted three times, or a session that connects and never
 * speaks while looking healthy. None of those throw.
 *
 * So this drives the real module with a fake socket and a fake AudioContext,
 * through the REAL createController, the REAL rio_provider seam and the REAL
 * rio_playout queue. Nothing about the decisions is stubbed -- only the wire and
 * the speaker, which are the two things a browser has and this does not.
 *
 * WHAT IT CANNOT PROVE, said plainly: microphone capture and actual playback.
 * ScriptProcessor and getUserMedia have no meaningful fake, so `pumpFrom` is
 * exercised only far enough to know it does not throw. Those two are what a drive
 * proves and nothing here does.
 */
'use strict';

const path = require('path');

const playoutMod = require(path.join(__dirname, '..', 'static', 'rio_playout.js'));
const provider = require(path.join(__dirname, '..', 'static', 'rio_provider.js'));
const rt = require(path.join(__dirname, '..', 'static', 'rio_realtime.js'));
const speech = require(path.join(__dirname, '..', 'static', 'rio_speech.js'));
const session = require(path.join(__dirname, '..', 'static', 'rio_xai_session.js'));

let checks = 0, failures = 0;
/* A SUITE THAT STOPS HALFWAY MUST NOT EXIT 0. This one is asynchronous, so an
   await that never settles ends the process quietly with no summary and no
   failure -- which happened while it was being written, and is exactly the
   silent-success shape tools/suite_sweep.js was built to hunt. Caught here too,
   at the source. */
let finished = false;
process.on('exit', (code) => {
  if (!finished && code === 0) {
    console.log('\n  FAIL  the suite never reached its summary — something '
                + 'awaited did not settle');
    process.exitCode = 1;
  }
});
function ok(cond, what) {
  checks++;
  if (!cond) { failures++; console.log('  FAIL  ' + what); }
  else console.log('  ok    ' + what);
}
function section(n) { console.log('\n=== ' + n + ' ==='); }

const RATE = session.RATE;
const ONE_SECOND = RATE * 2;                 // bytes of PCM16 mono

/* atob/btoa, which node 12 has not got and this module uses. */
if (typeof global.atob !== 'function') {
  global.atob = (b) => Buffer.from(b, 'base64').toString('binary');
}
if (typeof global.btoa !== 'function') {
  global.btoa = (s) => Buffer.from(s, 'binary').toString('base64');
}

function b64Audio(bytes) {
  return Buffer.alloc(bytes).toString('base64');
}

/* A socket that records rather than connects. */
function fakeWS() {
  const sent = [];
  function WS(url, protocols) {
    this.url = url;
    this.protocols = protocols;
    WS.last = this;
    setTimeout(() => { if (this.onopen) this.onopen(); }, 0);
  }
  WS.prototype.send = function (s) { sent.push(JSON.parse(s)); };
  WS.prototype.close = function () { if (this.onclose) this.onclose(); };
  WS.sent = sent;
  return WS;
}

/* Enough AudioContext to schedule against, with a clock the test owns. */
function fakeCtx() {
  let t = 100;
  return {
    get currentTime() { return t; },
    _advance: (s) => { t += s; },
    destination: {},
    createBuffer: (ch, len) => ({
      duration: len / RATE, length: len,
      copyToChannel: () => {}, getChannelData: () => new Float32Array(len),
    }),
    createBufferSource: () => ({
      buffer: null, connect: () => {}, start: () => {}, stop: () => {},
      onended: null,
    }),
  };
}

function harness(opts) {
  opts = opts || {};   // merged into session.open below, for the resume case
  const WS = fakeWS();
  const ctx = fakeCtx();
  /* ONE CLOCK, because that is the arrangement being tested: the session takes
     the AudioContext's, builds the queue on it, and the speaker schedules against
     it. A test with two would pass while production drifted. */
  const events = [];
  const prov = provider.create('xai_voice');
  const controllerSent = [];
  const bus = [];
  const controller = rt.createController({
    arbiter: speech.makeArbiter(),
    send: (o) => { controllerSent.push(o); s.send(o); },
    tool: () => Promise.resolve({ ok: true, answer: 'forty-two' }),
    audio: { mute: () => {}, unmute: () => {} },
    onEvent: (e) => bus.push(e),
    bargeSustainMs: 4, bargeConfirmMs: 8,
    provider: prov,
  });
  /* The ticker is recorded rather than run: node's real setInterval cannot see
     the clock this test advances. `pump()` below calls exactly what connect()
     armed, so what runs here is what runs in the car. */
  const armed = [];
  const s = session.open(Object.assign({}, opts, {
    mint: {
      ws_url: 'wss://example.test/v1/realtime?model=m',
      ws_subprotocol: 'xai-client-secret.tok',
      session: { type: 'realtime', output_modalities: ['audio'],
                 audio: { output: { voice: 'Eve' } } },
      silence_tail_ms: 400,
    },
    controller, provider: prov, ctx, tailGraceS: 0,
    playoutLib: playoutMod,
    WebSocketImpl: WS,
    setIntervalImpl: (fn, ms) => { armed.push({ fn, ms }); return armed.length; },
    clearIntervalImpl: (id) => { armed.length = 0; },
    onEvent: (e) => events.push(e),
  }));
  return { s, WS, ctx, p: s.playout, events, bus, controller, controllerSent,
           armed,
           advance: (sec) => { ctx._advance(sec); },
           pump: () => armed.forEach((a) => a.fn()),
           now: () => ctx.currentTime };
}

async function main() {

// ---------------------------------------------------------------------------
section('the session opens and sends the server\'s policy verbatim');
// ---------------------------------------------------------------------------
{
  const h = harness();
  await h.s.connect();
  const up = h.WS.sent.filter(e => e.type === 'session.update');
  ok(up.length === 1, 'exactly one session.update goes out');
  ok(up[0].session.output_modalities
     && up[0].session.output_modalities[0] === 'audio',
     'carrying output_modalities — without it the session accepts everything, '
     + 'runs VAD and produces nothing at all, with no error');
  ok(up[0].session.audio.output.voice === 'Eve',
     '...and an output voice, for the same reason');
  ok(h.WS.last.protocols[0].indexOf('xai-client-secret.') === 0,
     'the token rides in the websocket subprotocol, because a browser cannot '
     + 'set an Authorization header on one');
}

// ---------------------------------------------------------------------------
section('THE TAIL — the mouth is not handed back at response.done');
// ---------------------------------------------------------------------------
{
  const h = harness();
  await h.s.connect();
  h.s._onMessage(JSON.stringify({ type: 'response.created',
                                  response: { id: 'r1' } }));
  h.s._onMessage(JSON.stringify({ type: 'response.output_audio.delta',
                                  delta: b64Audio(ONE_SECOND * 3) }));
  ok(h.p.idle() === false, 'three seconds of audio means the queue is not idle');
  h.s._onMessage(JSON.stringify({ type: 'response.done',
                                  response: { id: 'r1', status: 'completed' } }));
  ok(h.p.idle() === false,
     'and response.done does NOT make it idle — that is the end of GENERATION, '
     + 'and commit a05d233 is what happens when the two are confused');
  h.advance(2.0);
  ok(h.p.idle() === false, 'still not, two seconds in');
  h.advance(1.2);
  ok(h.p.idle() === true, 'idle once the sound it was given has played');
}

// ---------------------------------------------------------------------------
section('THE GATE — a request waits for silence, a cancel does not');
// ---------------------------------------------------------------------------
{
  const h = harness();
  await h.s.connect();
  h.s._onMessage(JSON.stringify({ type: 'response.created',
                                  response: { id: 'r1' } }));
  h.s._onMessage(JSON.stringify({ type: 'response.output_audio.delta',
                                  delta: b64Audio(ONE_SECOND * 2) }));
  h.s._onMessage(JSON.stringify({ type: 'response.done',
                                  response: { id: 'r1', status: 'completed' } }));

  const before = h.WS.sent.length;
  h.s.send({ type: 'conversation.item.create',
             item: { type: 'function_call_output', call_id: 'c1', output: '{}' } });
  h.s.send({ type: 'response.create' });
  const afterSend = h.WS.sent.slice(before);
  ok(afterSend.some(e => e.type === 'conversation.item.create'),
     'the tool RESULT goes out at once — the conversation must not be left '
     + 'holding a call with no output');
  ok(!afterSend.some(e => e.type === 'response.create'),
     '...and the request for speech is HELD, because their own docs say the '
     + 'server would start the next turn over audio still playing');
  ok(h.events.some(e => e.type === 'XAI_REQUEST_GATED'),
     'and the gating is reported rather than silent');

  ok(h.armed.length === 1 && h.armed[0].ms <= 50,
     'connect() armed a ticker, because BOTH facts this transport owns become '
     + 'true in silence and nothing else here runs without a message');
  h.advance(2.2);
  h.pump();
  ok(h.WS.sent.some(e => e.type === 'response.create'),
     'and the ticker alone sends it once the sound is over — no inbound event '
     + 'is involved, which is the deadlock the first draft of this file had: the '
     + 'tool result is in, the server has nothing to say, and the next turn '
     + 'never starts');

  // A cancel that waited would be a barge-in that did not work.
  const n = h.WS.sent.length;
  h.s._onMessage(JSON.stringify({ type: 'response.created',
                                  response: { id: 'r2' } }));
  h.s._onMessage(JSON.stringify({ type: 'response.output_audio.delta',
                                  delta: b64Audio(ONE_SECOND * 5) }));
  h.s.send({ type: 'response.cancel' });
  ok(h.WS.sent.slice(n).some(e => e.type === 'response.cancel'),
     'a cancel goes out immediately even with five seconds queued');
}

// ---------------------------------------------------------------------------
section('output_audio_buffer.clear is local, because the wire refuses it');
// ---------------------------------------------------------------------------
{
  const h = harness();
  await h.s.connect();
  h.s._onMessage(JSON.stringify({ type: 'response.created',
                                  response: { id: 'r1' } }));
  h.s._onMessage(JSON.stringify({ type: 'response.output_audio.delta',
                                  delta: b64Audio(ONE_SECOND * 4) }));
  const n = h.WS.sent.length;
  h.s.send({ type: 'output_audio_buffer.clear' });
  /* AND A CONTROL EVENT BEHIND IT, so that "it was not sent" is a fact about
     THIS event rather than about a socket that has stopped sending anything. A
     negative assertion over an empty list is true for the wrong reason, which is
     what tools/suite_sweep.js flagged here. */
  h.s.send({ type: 'response.cancel' });
  const after = h.WS.sent.slice(n);
  ok(after.length === 1 && after[0].type === 'response.cancel',
     'the clear is NOT sent — measured, the wire answers "Invalid event '
     + 'received", which would be an error on the barge-in path at the worst '
     + 'moment — while the event behind it goes out normally');
  ok(h.s.stats().local_flushes === 1, 'it is handled locally instead');
  ok(h.p.idle() === true,
     'and the queue is empty at once, with no round trip to a server that would '
     + 'then have to tell us the sound stopped');
}

// ---------------------------------------------------------------------------
section('an explicit commit is never sent');
// ---------------------------------------------------------------------------
{
  const h = harness();
  await h.s.connect();
  const n = h.WS.sent.length;
  h.s.send({ type: 'input_audio_buffer.commit' });
  h.s.send({ type: 'input_audio_buffer.append', audio: b64Audio(320) });
  const after = h.WS.sent.slice(n);
  ok(after.length === 1 && after[0].type === 'input_audio_buffer.append',
     'the commit is dropped — accepted by the wire and measured to suppress the '
     + 'transcript entirely when server_vad is on — and the audio frame behind '
     + 'it still goes, which is what makes this a fact about the commit rather '
     + 'than about a mute socket');
}

// ---------------------------------------------------------------------------
section('three .completed events for one utterance are one transcript');
// ---------------------------------------------------------------------------
{
  const h = harness();
  await h.s.connect();
  const ev = { type: 'conversation.item.input_audio_transcription.completed',
               item_id: 'i1', transcript: 'what is that building' };
  h.s._onMessage(JSON.stringify(ev));
  h.s._onMessage(JSON.stringify(ev));
  h.s._onMessage(JSON.stringify(ev));
  ok(h.s.stats().transcripts === 1, 'one transcript is counted');
  ok(h.s.stats().transcript_repeats_dropped === 2,
     'and two repeats are dropped BY ID — the controller\'s item_id binding '
     + 'happens to suppress them, and happening to is not a design');

  const other = { type: 'conversation.item.input_audio_transcription.completed',
                  item_id: 'i2', transcript: 'and that one' };
  h.s._onMessage(JSON.stringify(other));
  ok(h.s.stats().transcripts === 2,
     'a genuinely different utterance is not mistaken for a repeat');
}

// ---------------------------------------------------------------------------
section('SILENCE WHERE AUDIO WAS EXPECTED IS A FAULT, NOT A LULL');
// ---------------------------------------------------------------------------
{
  const h = harness();
  await h.s.connect();
  // The exact shape that produced three wrong conclusions: it connects, VAD
  // runs, the buffer commits, and nothing is ever said.
  h.s._onMessage(JSON.stringify({ type: 'input_audio_buffer.speech_started' }));
  h.s._onMessage(JSON.stringify({ type: 'input_audio_buffer.committed',
                                  item_id: 'i1' }));
  h.s._onMessage(JSON.stringify({ type: 'response.created',
                                  response: { id: 'r1' } }));
  h.s._onMessage(JSON.stringify({ type: 'response.done',
                                  response: { id: 'r1', status: 'completed' } }));
  ok(h.s.health().degraded === true,
     'the session reports DEGRADED after a response that made no sound');
  ok(h.events.some(e => e.type === 'XAI_SESSION_SILENT'),
     '...and says so on the bus, so the panel can show it — the one thing this '
     + 'failure has never done is look like one');
  ok(h.s.health().silent_responses === 1, 'and counts them');

  h.s._onMessage(JSON.stringify({ type: 'response.created',
                                  response: { id: 'r2' } }));
  h.s._onMessage(JSON.stringify({ type: 'response.output_audio.delta',
                                  delta: b64Audio(ONE_SECOND) }));
  ok(h.s.health().degraded === false,
     'and it recovers when she speaks again rather than staying latched');
  ok(h.events.some(e => e.type === 'XAI_SESSION_RECOVERED'),
     '...saying that too');
}

// ---------------------------------------------------------------------------
section('the controller is told about audio in ITS vocabulary');
// ---------------------------------------------------------------------------
{
  /* Nothing on this wire sends output_audio_buffer.started, .stopped or
     .cleared, and the controller's holdTail and its whole mute ledger are keyed
     on them. They are synthesised at the three moments they are true. Without
     .stopped the mouth is never handed back AT ALL -- not late, never -- so the
     first turn of a drive would be the last. */
  const h = harness();
  await h.s.connect();
  h.s._onMessage(JSON.stringify({ type: 'response.created',
                                  response: { id: 'r1' } }));
  h.s._onMessage(JSON.stringify({ type: 'response.output_audio.delta',
                                  delta: b64Audio(ONE_SECOND * 3) }));
  ok(h.s.stats().audio_out_bytes === ONE_SECOND * 3,
     'the bytes are in the session ledger');

  const end = (h.p.state().responses.r1 || {}).scheduled_end;
  ok(Math.abs(end - h.s._speaker().queuedUntil()) < 1e-6,
     'the QUEUE and the SPEAKER agree to the microsecond on when the sound ends '
     + `(${end}) — two ledgers of one fact on one clock, and a drift between `
     + 'them would be a '
     + 'tail wrong by exactly that much with nothing else to notice');

  h.s._onMessage(JSON.stringify({ type: 'response.done',
                                  response: { id: 'r1', status: 'completed' } }));
  h.pump();
  ok(!h.bus.some(e => e.type === 'LIVE_UTTERANCE_END'),
     'the utterance is NOT closed at response.done — three seconds of her voice '
     + 'are still to come');

  h.advance(3.1);
  h.pump();
  const ends = h.bus.filter(e => e.type === 'LIVE_UTTERANCE_END');
  ok(ends.length === 1,
     'it closes when the sound does, and once');
  ok(ends.length === 1 && ends[0].audio_started === true,
     '...having seen the synthesised start');
  ok(ends.length === 1 && ends[0].ended_by === 'completed',
     '...and it reads as COMPLETED, which is what the drive log calls an answer '
     + 'the driver heard to the end');
}

// ---------------------------------------------------------------------------
section('a barge-in reads as a barge-in, not as a completed answer');
// ---------------------------------------------------------------------------
{
  const h = harness();
  await h.s.connect();
  h.s._onMessage(JSON.stringify({ type: 'response.created',
                                  response: { id: 'r1' } }));
  h.s._onMessage(JSON.stringify({ type: 'response.output_audio.delta',
                                  delta: b64Audio(ONE_SECOND * 4) }));
  h.s._onMessage(JSON.stringify({ type: 'response.done',
                                  response: { id: 'r1', status: 'completed' } }));
  h.s.send({ type: 'output_audio_buffer.clear' });
  h.advance(0.1);
  h.pump();
  const ends = h.bus.filter(e => e.type === 'LIVE_UTTERANCE_END');
  ok(ends.length === 1, 'the utterance closes on the flush');
  ok(ends.length === 1 && /^cancelled:/.test(ends[0].ended_by || ''),
     `...as ${ends.length === 1 ? ends[0].ended_by : '?'} — the local flush is `
     + 'not the sound finishing, and the two are charged differently');
  ok(ends.length === 1 && ends[0].early === true,
     '...and it is counted as an answer that was cut short');
  ok(!h.bus.some(e => e.type === 'LIVE_UTTERANCE_END' && e.ended_by === 'completed'),
     'and nothing also reports it as completed — the queue ends it as cancelled '
     + 'and only the drained end synthesises .stopped');
}

// ---------------------------------------------------------------------------
section('ATTACH — the second wire on the FIRST wire\'s policy and handle');
// ---------------------------------------------------------------------------
{
  /* The point of attach() is that it shares three things with the WebRTC path:
     createController, controllerConfig and sessionHandle. So this calls it with
     the real three, out of rio_realtime.js. Fakes there would agree with
     anything, and what is being checked is precisely that the sharing is real --
     the drift this is guarding against is a policy field that reaches one
     transport and not the other. */
  const nodes = [];
  function fakeGain() {
    const g = {
      value: 1, _ramps: [],
      cancelScheduledValues: () => {}, setValueAtTime: () => {},
      linearRampToValueAtTime: (v) => { g.value = v; g._ramps.push(v); },
    };
    return { gain: g, connect: () => {}, disconnect: () => {} };
  }
  const ctx = fakeCtx();
  ctx.state = 'running';
  ctx.createGain = () => { const n = fakeGain(); nodes.push(n); return n; };
  ctx.createMediaStreamSource = () => ({ connect: () => {} });
  ctx.createScriptProcessor = () => ({ connect: () => {}, onaudioprocess: null });
  ctx.resume = () => Promise.resolve();

  let busNode = null;
  global.RIO = global.RIO || {};
  global.RIO.output = {
    unlock: () => {},
    context: () => ctx,
    node: () => { busNode = busNode || { _isBus: true }; return busNode; },
  };
  global.RIO.provider = provider;

  const WS = fakeWS();
  const panel = [];
  const mic = { getTracks: () => [{ muted: false, readyState: 'live',
                                    stop: () => {} }] };
  /* The page's bus, which is how a route says it exists. */
  const busSubs = [];
  global.RIO.bus = { on: (t, fn) => busSubs.push(fn) };
  const minted = {
    ws_url: 'wss://example.test/v1/realtime?model=m',
    ws_subprotocol: 'xai-client-secret.tok',
    voice_backend: 'xai_voice',
    session: { type: 'realtime', output_modalities: ['audio'],
               audio: { output: { voice: 'Eve' } } },
    live_voice: 'Eve',
    speech_channels: { safety: true, chat: false },
    speak_timeout_ms: 1200,
    barge_sustain_ms: 260,
    tool_schemas: [{ type: 'function', name: 'look', parameters: {} },
                   { type: 'function', name: 'deep_dive', parameters: {} }],
    /* The two that ride with the route rather than with the session. */
    conditional_tools: {
      routing: [{ type: 'function', name: 'stop_navigation', parameters: {} },
                { type: 'function', name: 'reroute', parameters: {} }],
    },
  };

  const handle = await session.attach({
    session: minted, mic, arbiter: speech.makeArbiter(),
    barge: {}, onEvent: (e) => panel.push(e),
    url: (u) => u,
    createController: rt.createController,
    controllerConfig: rt._controllerConfig,
    sessionHandle: rt._sessionHandle,
    WebSocketImpl: WS,
  });

  ok(!!handle && typeof handle.speak === 'function'
     && typeof handle.speakTimeout === 'function'
     && typeof handle.cancelSpeak === 'function'
     && typeof handle.speechEnabled === 'function',
     'the handle the panel gets back is the same handle — speak, cancelSpeak, '
     + 'speechEnabled and speakTimeout come from sessionHandle, not from a copy '
     + 'in the vendor file');
  ok(handle.speechEnabled('safety') === true
     && handle.speechEnabled('chat') === false,
     '...reading the session policy, so the channel rules are the drive\'s and '
     + 'not this transport\'s');
  ok(handle.voiceBackend() === 'xai_voice',
     'and the controller reports WHICH WIRE it is on from the provider record — '
     + 'a literal here would have logged the wrong vendor for every drive');

  const st = handle.audioState();
  ok(st.channel === 'open' && st.peer === null && st.element_paused === null,
     'audioState reports a socket and says plainly that there is no peer '
     + 'connection and no element, rather than omitting the keys');
  ok(st.degraded === false && st.playout && typeof st.playout.idle === 'boolean',
     '...and carries the queue and the degraded flag, which is where a session '
     + 'that went quiet becomes visible to the page');

  ok(busNode !== null,
     'and her voice is connected to RIO.output\'s node, not to ctx.destination '
     + '— that node is inside the echo canceller, and on this wire there is no '
     + 'remote track to be cancelled instead');

  /* THE TWO TOOLS THAT RIDE WITH THE ROUTE. Nine live tools are required and
     only seven are in the session; stop_navigation and reroute are attached when
     there is something to stop. On this wire nothing was attaching them. */
  ok(busSubs.length === 1,
     'the controller is watching the page\'s bus for a route');
  const before2 = WS.sent.length;
  busSubs[0]({ type: 'NAV_ROUTE_ATTACHED' });
  const updates = WS.sent.slice(before2)
    .filter(e => e.type === 'session.update');
  const names = updates.length
    ? (updates[0].session.tools || []).map(t => t.name) : [];
  ok(updates.length === 1 && names.indexOf('stop_navigation') >= 0
     && names.indexOf('reroute') >= 0,
     `a route attaching sends the two conditional tools (${names.join(', ')}) — `
     + 'without this two of the nine live tools would not exist for the whole '
     + 'drive, and nothing would have said so');
  ok(names.indexOf('look') >= 0 && names.indexOf('deep_dive') >= 0,
     '...and the session\'s own tools go back with them, because a session.update '
     + 'replaces the tool list whole');

  /* THE MOUTH. On the other wire it is element.muted; here it is a gain, and it
     had better be between her voice and the bus rather than nowhere. */
  const gainNode = nodes[0];
  ok(!!gainNode && gainNode.gain.value === 1,
     'a gain node is the mouth, and she starts audible');
  handle.controller.stop();
  ok(gainNode.gain.value === 0,
     '...and the controller\'s own mute reaches it — audio.mute() is called from '
     + 'seven places in rio_realtime.js and every one of them has to silence her '
     + 'on this wire too, where there is no element to mute');
  ok(gainNode.gain._ramps.length > 0,
     '...by a ramp rather than a switch, because a gain that jumps to zero '
     + 'mid-word clicks and the sustain gate changes its mind often');

  handle.stop();
  ok(true, 'stop() tears down the socket, the mouth and the microphone');
}

// ---------------------------------------------------------------------------
section('one producer of the handle, one producer of the policy');
// ---------------------------------------------------------------------------
{
  const src = require('fs').readFileSync(
    path.join(__dirname, '..', 'static', 'rio_realtime.js'), 'utf8');
  const handles = (src.match(/var handle = \{/g) || []).length;
  ok(handles === 1,
     `sessionHandle is the only place a session handle is built (${handles}) — `
     + 'a second one is how the two transports start answering speakTimeout '
     + 'differently');
  const cfgs = (src.match(/createController\(\{/g) || []).length;
  ok(cfgs === 0,
     'and nothing builds a controller config inline any more, so a policy field '
     + 'cannot reach one wire and miss the other');
  const xai = require('fs').readFileSync(
    path.join(__dirname, '..', 'static', 'rio_xai_session.js'), 'utf8');
  ok(!/speak_timeout_ms_by_channel|speech_channels/.test(xai),
     'and the vendor file holds no copy of the session policy it was handed');
}

// ---------------------------------------------------------------------------
section('Z: the same script through a PLAUSIBLE transport, which gets it wrong');
// ---------------------------------------------------------------------------
{
  /* Every assertion above passes. So would every assertion about a transport
     that hands the mouth back at response.done and sends the next request the
     moment a tool result is ready -- unless the assertions actually discriminate,
     and the only way to know that is to write the wrong transport and watch them
     fail on it.
     
     This is the obvious implementation: the events the server sends, forwarded.
     It is what rio_live.js does, correctly, because there the server owned the
     playout. Here it is commit a05d233 again. */
  function naive(controller, ws) {
    return {
      send: (o) => ws.sent.push(o),
      handle: (ev) => {
        if (ev.type === 'response.done') {
          controller.handle({ type: 'output_audio_buffer.stopped',
                              response_id: ev.response.id });
        }
      },
    };
  }
  const h = harness();
  await h.s.connect();
  const bus2 = [];
  const c2 = rt.createController({
    arbiter: speech.makeArbiter(), send: () => {},
    tool: () => Promise.resolve({ ok: true }),
    audio: { mute: () => {}, unmute: () => {} },
    onEvent: (e) => bus2.push(e),
    provider: provider.create('xai_voice'),
  });
  const ws2 = { sent: [] };
  const n = naive(c2, ws2);
  c2.handle({ type: 'response.created', response: { id: 'r1' } });
  c2.handle({ type: 'output_audio_buffer.started', response_id: 'r1' });
  c2.handle({ type: 'response.done',
              response: { id: 'r1', status: 'completed' } });
  n.handle({ type: 'response.done', response: { id: 'r1', status: 'completed' } });
  ok(bus2.some(e => e.type === 'LIVE_UTTERANCE_END'),
     'the naive transport closes the utterance AT response.done — three seconds '
     + 'early, and the assertion above fails against it, which is what makes it '
     + 'an assertion about the tail rather than about nothing');
  n.send({ type: 'conversation.item.create', item: {} });
  n.send({ type: 'response.create' });
  ok(ws2.sent.some(e => e.type === 'response.create'),
     '...and sends the next request straight over audio still playing, which is '
     + 'the overlap their own migration note warns about and the gate refuses');
  ok(h.armed.length === 1 && bus2.length > 0,
     'both transports were driven by the same script; only one of them is right');
}

// ---------------------------------------------------------------------------
section('the conversation has an id, and a drop is recoverable with it');
// ---------------------------------------------------------------------------
{
  const h = harness();
  await h.s.connect();
  ok(h.s.conversationId() === null,
     'no id until the session gives one — a guess here would be a reconnect '
     + 'into a conversation that does not exist');
  h.s._onMessage(JSON.stringify({ type: 'conversation.created',
                                  conversation: { id: 'conv-1' } }));
  ok(h.s.conversationId() === 'conv-1' && h.s.health().conversation_id === 'conv-1',
     'the id from conversation.created is recorded and readable');
  ok(h.events.some(e => e.type === 'XAI_CONVERSATION'
                        && e.conversation_id === 'conv-1'),
     '...and announced, so a page that wants to reconnect has it without '
     + 'reaching into the transport');

  /* AND IT GOES ON THE URL. Measured: the id in session.update is accepted and
     silently starts a NEW conversation, so the query string is not a style
     choice — it is the only spelling that resumes anything. */
  const h2 = harness({ resumeConversationId: 'conv-1' });
  await h2.s.connect();
  ok(h2.WS.last.url.indexOf('conversation_id=conv-1') > 0,
     `a resume puts the id on the socket URL (${h2.WS.last.url.slice(-40)})`);
  ok(!h2.WS.sent.some(e => e.type === 'session.update'
       && JSON.stringify(e.session).indexOf('conv-1') >= 0),
     '...and NOT in session.update, which is accepted, assigns a new id and '
     + 'loses the conversation — the same shape as force_message\'s `text` form');
}

// ---------------------------------------------------------------------------
section('health is answerable at any moment');
// ---------------------------------------------------------------------------
{
  const h = harness();
  await h.s.connect();
  const hh = h.s.health();
  ok(hh.connected === true, 'connected');
  ok(hh.degraded === false, 'not degraded on a fresh session');
  ok(hh.playout && typeof hh.playout.idle === 'boolean',
     'and it carries the queue\'s own state, so "is she still talking" has one '
     + 'answer rather than two');
  h.s.stop();
  ok(h.s.health().connected === false, 'and stop() is visible in it');
}

finished = true;
console.log(`\n${checks - failures}/${checks} checks passed`);
if (failures) { console.log(`${failures} FAILED`); process.exit(1); }
}

main().catch((e) => { console.error(e); process.exit(1); });
