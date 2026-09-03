/* live_voice_probe.js — ten turns of a live conversation, and why each one ended.
 *
 *   node tools/live_voice_probe.js
 *   node tools/live_voice_probe.js --post http://127.0.0.1:8888
 *
 * The complaint this exists for is "her voice keeps cutting out and I have to
 * ask again", and the reason it survived so long is that the four things that
 * cut an answer off are indistinguishable from the passenger seat. A driver
 * cannot tell a false barge-in from a token cap; both are RIO stopping in the
 * middle of a sentence.
 *
 * So this drives the REAL controller (static/rio_realtime.js) against the REAL
 * arbiter (static/rio_speech.js) through a scripted drive containing all four,
 * in the proportions a car actually produces them, and counts the outcome by
 * cause. No microphone, no key, no network -- every one of these is an event on
 * the data channel, and the controller is a pure handler over that stream
 * precisely so this is possible.
 *
 * WHAT IT IS NOT: a substitute for driving the car. It cannot tell you how
 * often the detector fires on a real cabin, because that is a property of the
 * cabin. It tells you what happens to an answer WHEN it does, which is the half
 * that lives in this repository and the half that was wrong.
 *
 * --post sends each event to a running server exactly as the panel does, so the
 * reporting path (/realtime/cutoff -> /realtime/cutoffs) is exercised over HTTP
 * rather than assumed.
 */
'use strict';

const path = require('path');
/* --controller lets the same script run against an older copy of the file,
   which is the only honest way to state a before-and-after: the same ten turns,
   the same arbiter, the same events, one implementation swapped. */
const ctlArg = (() => {
  const i = process.argv.indexOf('--controller');
  return i >= 0 ? process.argv[i + 1] : null;
})();
const rt = require(ctlArg
  ? path.resolve(ctlArg)
  : path.join(__dirname, '..', 'static', 'rio_realtime.js'));
const speech = require(path.join(__dirname, '..', 'static', 'rio_speech.js'));

const args = process.argv.slice(2);
const postTo = (() => {
  const i = args.indexOf('--post');
  return i >= 0 ? args[i + 1] : null;
})();

const sleep = (ms) => new Promise(r => setTimeout(r, ms));
let RECOVERABLE = 0;

/* The script. Ten turns, each a plausible thing that happens in a car.
 *
 * The mix is deliberately unkind and deliberately not uniform: echo and cabin
 * noise dominate because that is what a car produces, real interruptions are
 * common because drivers interrupt, and the rare causes appear once each so
 * they cannot hide. */
/* `recoverable` says whether the driver should still have heard the end of
   that answer. A real interruption and the token ceiling are deliberate stops
   and belong in neither column; everything else is an answer the car threw
   away for no reason, which is the thing being counted. */
const TURNS = [
  { name: 'clean answer', kind: 'clean', recoverable: false },
  { name: 'echo of her own voice, brief', kind: 'blip', recoverable: true },
  { name: 'driver really interrupts', kind: 'real_barge', recoverable: false },
  { name: 'door thunk, sustained past the gate, no words', kind: 'false_barge', recoverable: true },
  { name: 'clean answer', kind: 'clean', recoverable: false },
  { name: 'gap warning cuts in', kind: 'preempt', recoverable: true },
  { name: 'wipers, brief', kind: 'blip', recoverable: true },
  { name: 'detector fires on nothing, transcriber returns empty', kind: 'false_barge_empty', recoverable: true },
  { name: 'a long answer hits the ceiling', kind: 'token_cap', recoverable: false },
  { name: 'driver really interrupts', kind: 'real_barge', recoverable: false },
];

function makeSession(opts) {
  const arbiter = speech.makeArbiter();
  const sent = [];
  const events = [];
  const audio = { muted: false };
  const controller = rt.createController({
    arbiter,
    send: (o) => sent.push(o),
    tool: () => Promise.resolve({ ok: true }),
    audio: { mute: () => { audio.muted = true; }, unmute: () => { audio.muted = false; } },
    onEvent: (ev) => events.push(ev),
    bargeSustainMs: opts.sustain,
    bargeConfirmMs: opts.confirm,
    resumeInstruction: 'RESUME>>',
  });
  return { arbiter, sent, events, audio, controller };
}

async function runTurn(s, turn, n) {
  const rid = 'r' + n;
  const said = 'This is answer number ' + n + ' and it was going somewhere';
  s.controller.handle({ type: 'response.created', response: { id: rid } });
  s.controller.handle({ type: 'response.output_audio_transcript.delta',
                        response_id: rid, delta: said });
  await sleep(2);

  switch (turn.kind) {
    case 'clean':
      s.controller.handle({ type: 'response.done',
                            response: { id: rid, status: 'completed' } });
      break;

    case 'blip':
      // Shorter than the sustain gate: she should never stop.
      s.controller.handle({ type: 'input_audio_buffer.speech_started' });
      await sleep(1);
      s.controller.handle({ type: 'input_audio_buffer.speech_stopped' });
      await sleep(2);
      s.controller.handle({ type: 'response.done',
                            response: { id: rid, status: 'completed' } });
      break;

    case 'real_barge':
      // A real sentence: it runs well past the gate, THEN stops, and the
      // transcript follows a beat later. The long middle is the part that
      // matters -- an implementation that starts its confirmation clock at the
      // cancel rather than at the end of the speech declares this one a false
      // alarm and resumes over the driver.
      s.controller.handle({ type: 'input_audio_buffer.speech_started' });
      await sleep(40);
      s.controller.handle({ type: 'input_audio_buffer.speech_stopped' });
      await sleep(4);
      s.controller.handle({
        type: 'conversation.item.input_audio_transcription.completed',
        transcript: 'no, the one behind it' });
      break;

    case 'false_barge':
      // Sustained noise -- a truck going past -- and then nothing. It stops,
      // and no words ever arrive.
      s.controller.handle({ type: 'input_audio_buffer.speech_started' });
      await sleep(20);
      s.controller.handle({ type: 'input_audio_buffer.speech_stopped' });
      break;

    case 'false_barge_empty':
      s.controller.handle({ type: 'input_audio_buffer.speech_started' });
      await sleep(20);
      s.controller.handle({ type: 'input_audio_buffer.speech_stopped' });
      await sleep(4);
      s.controller.handle({
        type: 'conversation.item.input_audio_transcription.completed',
        transcript: '' });
      break;

    case 'preempt': {
      let done = null;
      const warning = {
        priority: s.arbiter.P.SAFETY, group: 'headway', id: 'headway:too_close',
        text: 'Too close', ttlMs: 2500,
        play: () => new Promise(r => { done = r; }),
        stop: () => { if (done) done(); },
      };
      s.arbiter.say(warning);
      await sleep(5);
      if (done) done();                       // the warning finishes speaking
      break;
    }

    case 'token_cap':
      s.controller.handle({
        type: 'response.done',
        response: { id: rid, status: 'incomplete',
                    status_details: { reason: 'max_output_tokens' } } });
      break;
  }
  // Long enough for both gates plus the arbiter's next-tick handoff.
  await sleep(140);
}

async function drive(opts) {
  const s = makeSession(opts);
  for (let i = 0; i < TURNS.length; i++) await runTurn(s, TURNS[i], i + 1);
  s.controller.stop();
  const st = s.controller.state();
  return {
    cutoffs: st.cutoffs,
    counters: st.counters,
    // For a build with no cause tracking: everything that ended without being
    // spoken and was not a barge-in (the pre-emption and the token cap).
    lostOther: s.events.filter(e => e.type === 'LIVE_RESPONSE_END' &&
                               e.reason && e.reason !== 'spoken').length,
    events: s.events,
    resumes: s.sent.filter(e => e.type === 'response.create' && e.response &&
                           /^RESUME>>/.test(e.response.instructions || '')).length,
  };
}

function line(label, v) { console.log('  %s %s', label.padEnd(34), v); }

async function report(r) {
  if (!r.cutoffs) {
    /* An implementation with no instrumentation in it. That is not a gap in
       this script -- it is the finding: every one of these ended the same way
       and there was nothing anywhere that could say why. */
    console.log('\n  cut-offs by cause, over %d turns', TURNS.length);
    console.log('    NOT INSTRUMENTED — this build cannot attribute a cause');
    console.log('');
    line('responses started', r.counters.responses);
    line('barge-ins seen', r.counters.barge_ins);
    line('answers interrupted', r.counters.interrupted);
    line('blips absorbed before any cost', 0);
    line('answers resumed', 0);
    console.log('');
    line('recoverable answers in this script', RECOVERABLE);
    line('...that the driver actually heard the end of', 0);
    line('ANSWERS LOST THAT SHOULD NOT HAVE BEEN', RECOVERABLE);
    return null;
  }
  const c = r.cutoffs;
  const total = Object.keys(c).reduce((a, k) => a + c[k], 0);
  console.log('\n  cut-offs by cause, over %d turns', TURNS.length);
  line('false barge-in (noise, no words)', c.false_barge_in);
  line('real barge-in (driver spoke)', c.barge_in);
  line('arbiter pre-emption', c.preempted);
  line('token cap', c.token_cap);
  line('transport drop', c.transport);
  line('other', c.other);
  line('TOTAL', total);
  console.log('');
  line('blips absorbed before any cost', r.counters.blips_absorbed);
  line('answers resumed', r.resumes);
  line('resumes declined (budget)', r.counters.resume_skipped);
  console.log('');
  line('recoverable answers in this script', RECOVERABLE);
  line('...that the driver actually heard the end of',
       r.counters.blips_absorbed + r.resumes);
  line('ANSWERS LOST THAT SHOULD NOT HAVE BEEN',
       RECOVERABLE - r.counters.blips_absorbed - r.resumes);
  return { total, cutoffs: c, resumes: r.resumes,
           absorbed: r.counters.blips_absorbed };
}

/* Node 12 on this pod has no global fetch, and pulling a dependency in for
   three POSTs would be worse than fifteen lines. */
function httpJson(url, method, body) {
  const u = new (require('url').URL)(url);
  const lib = u.protocol === 'https:' ? require('https') : require('http');
  const payload = body === undefined ? null : Buffer.from(JSON.stringify(body));
  return new Promise((resolve, reject) => {
    const req = lib.request({
      hostname: u.hostname, port: u.port, path: u.pathname + u.search,
      method: method,
      headers: payload
        ? { 'Content-Type': 'application/json', 'Content-Length': payload.length }
        : {},
    }, (res) => {
      let out = '';
      res.on('data', d => { out += d; });
      res.on('end', () => {
        try { resolve({ status: res.statusCode, json: JSON.parse(out || '{}') }); }
        catch (e) { resolve({ status: res.statusCode, json: {} }); }
      });
    });
    req.on('error', reject);
    if (payload) req.write(payload);
    req.end();
  });
}

async function postAll(base, events) {
  const MAP = { LIVE_CUTOFF: 'cutoff', LIVE_RESUME: 'resumed',
                LIVE_RESUME_SKIPPED: 'resume_skipped',
                LIVE_BARGE_ABSORBED: 'blips_absorbed' };
  let n = 0;
  for (const ev of events) {
    const kind = MAP[ev.type];
    if (!kind) continue;
    const body = JSON.stringify({
      kind, cause: ev.cause || kind,
      detail: { response_id: ev.response_id || null, reason: ev.reason || null,
                detail: ev.detail || null,
                said_chars: ev.said_chars === undefined ? null : ev.said_chars,
                by: ev.by || null },
    });
    const r = await httpJson(base + '/realtime/cutoff', 'POST', JSON.parse(body));
    if (r.status === 200 && r.json.ok) n++;
  }
  return n;
}

async function main() {
  console.log('=========================================================');
  console.log(' live voice probe — %d scripted turns through the real', TURNS.length);
  console.log(' controller and the real arbiter');
  console.log('=========================================================');
  RECOVERABLE = TURNS.filter(t => t.recoverable).length;
  TURNS.forEach((t, i) => console.log('  %d. %-52s %s', i + 1, t.name,
                                      t.recoverable ? '(should survive)' : ''));

  const r = await drive({ sustain: 4, confirm: 40 });
  const summary = await report(r);

  if (postTo && summary) {
    await httpJson(postTo + '/realtime/cutoffs/reset', 'POST', {});
    const n = await postAll(postTo, r.events);
    const tally = (await httpJson(postTo + '/realtime/cutoffs', 'GET')).json;
    console.log('\n  reported %d events to %s', n, postTo);
    console.log('  server tally: %s', JSON.stringify(tally.cutoffs));
    console.log('  server resumed=%d absorbed=%d skipped=%d',
                tally.resumed, tally.blips_absorbed, tally.resume_skipped);
    const agrees = JSON.stringify(tally.cutoffs) === JSON.stringify(summary.cutoffs)
      && tally.resumed === summary.resumes
      && tally.blips_absorbed === summary.absorbed;
    console.log('  %s', agrees
      ? 'server agrees with the browser-side tally'
      : 'MISMATCH between the page and the server');
    if (!agrees) process.exitCode = 1;
  }
  console.log('');
}

main();
