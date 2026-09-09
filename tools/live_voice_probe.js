/* live_voice_probe.js — ten turns of a live conversation, and why each one ended.
 *
 *   node tools/live_voice_probe.js
 *   node tools/live_voice_probe.js --voice elevenlabs
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
 *
 * --voice elevenlabs runs the SAME ten turns with RIO's voice coming out of the
 * dialogue sink instead of out of the session. That is the version of this
 * question that matters now, and it is a genuinely different one: in text mode
 * the model finishes an answer seconds before the driver hears the end of it,
 * so "how far did she get" comes from a speaker's clock rather than from a
 * transcript, the mouth is held past `response.done`, and a cancel has audio to
 * throw away rather than only a generation to stop. Every one of those is a new
 * way for a recoverable answer to be lost, and the number at the bottom of this
 * report is the one that says whether any of them is.
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
/* Which mouth. `elevenlabs` runs the same ten turns with the real dialogue
   sink between the controller and the speaker; anything else is the session
   speaking for itself, which is what this script measured before. */
const voiceArg = (() => {
  const i = args.indexOf('--voice');
  return i >= 0 ? (args[i + 1] || '') : null;
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
/* --phone: the drive that broke on 2026-09-09, reproduced.
 *
 * An iPhone in a cradle. Her voice comes out of the loudspeaker a few
 * centimetres from the microphone, goes back in, and the input transcriber
 * writes it down. Her opening line is "Hey. What's up." and it came back as
 * "Hello." and "What's up?" -- which the newest-wins supersede took for new
 * driver questions and cancelled the response that was producing the audio.
 * Ten times across two sessions. She never finished a sentence.
 *
 * So this script is five long answers with NOBODY TALKING, and the room full
 * of her own voice: echoes inside the onset guard, echoes the level test
 * catches, echoes that arrive as a transcript with no sustained speech at all.
 * The number that matters is zero.
 *
 * Then the two things that must still work: a driver who really interrupts,
 * and a driver who really asks something else.
 */
const PHONE_TURNS = [
  { name: 'long answer, her voice echoes back inside the onset guard',
    kind: 'echo_onset', recoverable: true },
  { name: 'long answer, echo arrives as a transcript, no sustained speech',
    kind: 'echo_transcript', recoverable: true },
  { name: 'long answer, her exact words come back verbatim',
    kind: 'echo_verbatim', recoverable: true },
  { name: 'long answer, echo louder than nothing but never sustained',
    kind: 'echo_blip_transcript', recoverable: true },
  { name: 'long answer, nobody talking at all',
    kind: 'clean', recoverable: false },
  { name: 'driver really interrupts', kind: 'real_barge', recoverable: false },
  { name: 'driver really asks something else, over the top of her',
    kind: 'real_second_question', recoverable: false },
  { name: 'driver asks again while a tool is still out',
    kind: 'real_second_question_tool', recoverable: false },
];

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
  /* The ElevenLabs mouth, over a fake wire and a clock this script turns by
     hand. Real sink, real bookkeeping — the only stubs are the socket and the
     speaker, because those are the two things a node process does not have. */
  let rig = null;
  if (opts.eleven) {
    const eleven = require(path.join(__dirname, '..', 'static', 'rio_voice_eleven.js'));
    const H = require(path.join(__dirname, 'voice_sink_harness.js'));
    rig = H.openSink(eleven, {});
  }
  const controller = rt.createController({
    arbiter,
    send: (o) => sent.push(o),
    tool: opts.slowTool
      ? ((name, a, controller) => new Promise((res) => {
          if (controller) {
            controller.signal.addEventListener('abort', () => {
              const e = new Error('aborted'); e.name = 'AbortError'; res(Promise.reject(e));
            });
          }
        }))
      : (() => Promise.resolve({ ok: true })),
    audio: rig
      ? { mute: () => { audio.muted = true; rig.sink.mute(); },
          unmute: () => { audio.muted = false; rig.sink.unmute(); } }
      : { mute: () => { audio.muted = true; }, unmute: () => { audio.muted = false; } },
    voice: rig ? rig.sink : null,
    liveVoice: opts.liveVoice || 'marin',
    onEvent: (ev) => events.push(ev),
    bargeSustainMs: opts.sustain,
    bargeConfirmMs: opts.confirm,
    resumeInstruction: 'RESUME>>',
    /* THE PHONE COLUMN. The onset guard and the level test only exist on a
       touch device, and they are two of the three gates the supersede was
       bypassing -- so a probe that runs the desktop numbers cannot see this
       bug at all. `levels` is the meter the echo test reads: microphone
       quieter than loudspeaker is her, which is the whole point. */
    ...(opts.phone ? {
      /* The onset guard is scaled with the other two timers, and it has to be:
         this script's turns are milliseconds apart, and a real 400 ms guard
         would hold EVERY barge in the whole run -- including the driver's --
         which reads as the gate working and is actually the clock being wrong.
         6 ms sits in the same proportion to these sleeps as 400 ms does to a
         real sentence. The echo margin is a dB and does not scale. */
      bargeOnsetGuardMs: opts.onsetGuard === undefined ? 6 : opts.onsetGuard,
      bargeEchoMarginDb: 6,
      bargeEchoFloorDb: -50,
      levels: () => phoneLevels,
      // Scaled with everything else. 600 ms is to a real sentence what 10 ms
      // is to one of these turns.
      echoTailMs: opts.echoTail === undefined ? 10 : opts.echoTail,
    } : {}),
  });
  return { arbiter, sent, events, audio, controller, rig };
}

/* What the meter says right now. Echo is the microphone hearing the speaker,
   so it is QUIETER than the speaker -- the margin is negative and the level
   test refuses to cancel on it. A real driver is louder. */
let phoneLevels = { mic: -46, out: -18 };
function asEcho() { phoneLevels = { mic: -46, out: -18 }; }
function asDriver() { phoneLevels = { mic: -12, out: -18 }; }

async function runTurn(s, turn, n) {
  const rid = 'r' + n;
  const said = 'This is answer number ' + n + ' and it was going somewhere';
  s.controller.handle({ type: 'response.created', response: { id: rid } });
  if (s.rig) {
    /* Text mode. The model writes the whole answer at once, the synthesiser
       is part-way through it, and the gap between those two is the thing
       every check below is really about. */
    s.controller.handle({ type: 'response.output_text.delta',
                          response_id: rid, delta: said });
    s.rig.say(rid, said, 2.0);
    s.rig.audio.advance(1.0);            // she is half-way through saying it
  } else {
    s.controller.handle({ type: 'response.output_audio_transcript.delta',
                          response_id: rid, delta: said });
  }
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
      asDriver();
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

    /* --- the phone's own failures -------------------------------------- */

    case 'echo_onset':
      /* Her first syllable comes straight back. The onset guard HOLDS the
         barge decision rather than taking it -- so there is no pendingBarge --
         and then the transcript lands. This is the exact path that produced
         `said_chars: 4` on the real device: four characters in, cancelled. */
      asEcho();
      s.controller.handle({ type: 'input_audio_buffer.speech_started' });
      await sleep(2);
      s.controller.handle({
        type: 'conversation.item.input_audio_transcription.completed',
        transcript: 'Hello.' });
      await sleep(4);
      s.controller.handle({ type: 'input_audio_buffer.speech_stopped' });
      await sleep(4);
      s.controller.handle({ type: 'response.done',
                            response: { id: rid, status: 'completed' } });
      break;

    case 'echo_transcript':
      /* No speech_started at all -- the detector never fired, and a transcript
         turned up anyway. Nothing in the barge machinery has an opinion about
         this one; only the gate does. */
      asEcho();
      s.controller.handle({
        type: 'conversation.item.input_audio_transcription.completed',
        transcript: 'What is up?' });
      await sleep(4);
      s.controller.handle({ type: 'response.done',
                            response: { id: rid, status: 'completed' } });
      break;

    case 'echo_verbatim':
      /* Her own words, exactly, after the guard has expired and sustained
         past the gate -- a phone at high volume in a hard-surfaced cabin. The
         structural gate would let this through; the text test is what catches
         it. */
      asEcho();
      s.controller.handle({ type: 'input_audio_buffer.speech_started' });
      await sleep(40);
      s.controller.handle({ type: 'input_audio_buffer.speech_stopped' });
      await sleep(4);
      s.controller.handle({
        type: 'conversation.item.input_audio_transcription.completed',
        transcript: said });
      await sleep(4);
      s.controller.handle({ type: 'response.done',
                            response: { id: rid, status: 'completed' } });
      break;

    case 'echo_blip_transcript':
      // Fires the detector, stops well inside the sustain gate, and a
      // transcript follows anyway.
      asEcho();
      s.controller.handle({ type: 'input_audio_buffer.speech_started' });
      await sleep(1);
      s.controller.handle({ type: 'input_audio_buffer.speech_stopped' });
      await sleep(2);
      s.controller.handle({
        type: 'conversation.item.input_audio_transcription.completed',
        transcript: 'Hello.' });
      await sleep(4);
      s.controller.handle({ type: 'response.done',
                            response: { id: rid, status: 'completed' } });
      break;

    case 'real_second_question':
      /* A DRIVER, not a loudspeaker: louder than the output, sustained well
         past the gate, and saying something she has not said. This must still
         supersede -- it is the whole point of newest-wins. */
      asDriver();
      s.controller.handle({ type: 'input_audio_buffer.speech_started' });
      await sleep(40);
      s.controller.handle({ type: 'input_audio_buffer.speech_stopped' });
      await sleep(4);
      s.controller.handle({
        type: 'conversation.item.input_audio_transcription.completed',
        transcript: 'Actually, where is the nearest petrol station?' });
      break;

    case 'real_second_question_tool':
      /* THE CASE NEWEST-WINS EXISTS FOR, and the one the phone gate must not
         break. She is not speaking -- she is waiting on a `look` that has been
         out for a while, which on the first real drive was up to 48 seconds --
         and the driver asks something else. Nothing of hers is in the room, so
         the gate lets it through, the tool is aborted and the turn supersedes. */
      asDriver();
      // She has finished saying "let me look" and is now SILENT, waiting on
      // the camera. That is the real shape of it: on the first drive these
      // calls ran up to 48 seconds with nothing coming out of the speaker.
      s.controller.handle({ type: 'response.done',
                            response: { id: rid, status: 'completed' } });
      s.controller.handle({ type: 'response.function_call_arguments.done',
                            name: 'look', call_id: 'probe_look',
                            arguments: JSON.stringify({ question: 'what is that' }) });
      // ...and her voice has been out of the room for longer than the echo
      // tail, which is what makes a transcript a question rather than an echo.
      await sleep(16);
      s.controller.handle({
        type: 'conversation.item.input_audio_transcription.completed',
        transcript: 'Never mind that, how far to the next junction?' });
      await sleep(4);
      break;

    case 'token_cap':
      s.controller.handle({
        type: 'response.done',
        response: { id: rid, status: 'incomplete',
                    status_details: { reason: 'max_output_tokens' } } });
      break;
  }
  if (s.rig) {
    // The rest of the audio plays out. Without this the mouth is never handed
    // back on a clean turn -- which is correct behaviour and would deadlock a
    // script that never lets the speaker finish.
    s.rig.done(rid);
    s.rig.audio.advance(5.0);
  }
  // Long enough for both gates plus the arbiter's next-tick handoff, and for
  // the drain poll behind a text-mode response.
  await sleep(240);
}

async function drive(opts) {
  const s = makeSession(opts);
  if (s.rig) await s.rig.opened;
  const script = opts.script || TURNS;
  for (let i = 0; i < script.length; i++) await runTurn(s, script[i], i + 1);
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

/* --phone: five long answers with nobody talking and the room full of her own
   voice, then the two things that must still work. The acceptance is stated in
   the output rather than left for a reader to total up. */
async function phoneRun() {
  console.log('=========================================================');
  console.log(' live voice probe — PHONE. An iPhone in a cradle: her voice');
  console.log(' out of the loudspeaker, back into the microphone, and');
  console.log(' through the input transcriber as a "new question".');
  console.log(' Onset guard 400 ms, sustain 600 ms, echo margin 6 dB.');
  console.log('=========================================================');
  PHONE_TURNS.forEach((t, i) => console.log(
    '  ' + String(i + 1).padStart(2) + '. ' + t.name.padEnd(58)
    + (t.recoverable ? '(must survive)' : '')));

  const r = await drive({ sustain: 4, confirm: 40, phone: true,
                          slowTool: true, script: PHONE_TURNS });
  const c = r.counters;
  const cut = r.cutoffs || {};
  const quiet = PHONE_TURNS.filter(t => t.recoverable).length;

  const phantom = r.events.filter(e => e.type === 'LIVE_TURN_PHANTOM');
  const superseded = r.events.filter(e => e.type === 'LIVE_TURN_SUPERSEDED');

  console.log('\n  the five answers nobody interrupted');
  line('cut-offs during them',
       Object.keys(cut).reduce((a, k) => a + cut[k], 0) - (cut.barge_in || 0));
  line('phantom transcripts refused', c.turns_phantom);
  // padEnd, not '%-24s': node's console.log understands %s and nothing about
  // widths — the same trap this file already documents further down.
  phantom.forEach(e => console.log('      refused ' + String(e.why).padEnd(24)
                                   + JSON.stringify(e.text)));
  line('supersedes', c.turns_superseded);
  superseded.forEach(e => console.log('      superseded by %s', JSON.stringify(e.by)));

  console.log('\n  the three that must still work');
  line('genuine barge-ins', cut.barge_in || 0);
  line('genuine supersedes', c.turns_superseded);
  line('tool calls aborted by one', c.tools_aborted);

  const fails = [];
  const noise = (cut.other || 0) + (cut.false_barge_in || 0) + (cut.transport || 0)
              + (cut.preempted || 0);
  if (noise) fails.push(noise + ' cut-off(s) with nobody talking');
  if (c.turns_phantom !== quiet) {
    fails.push('expected ' + quiet + ' phantom transcripts refused, got '
               + c.turns_phantom);
  }
  /* TWO barge-ins, not one: a driver talking over her is a barge-in whether
     they are interrupting or asking something new, and both turns 6 and 7 do
     exactly that. The supersede that matters is turn 8's -- the driver asking
     again while a tool is still out, which is the case newest-wins was built
     for and the one a phone gate could most easily have broken. */
  if ((cut.barge_in || 0) !== 2) {
    fails.push('expected 2 genuine barge-ins (turns 6 and 7), got '
               + (cut.barge_in || 0));
  }
  if (c.turns_superseded !== 1) {
    fails.push('expected exactly 1 supersede (turn 8, over a live tool call), '
               + 'got ' + c.turns_superseded);
  }
  if (c.tools_aborted !== 1) {
    fails.push('expected the superseded turn to abort its tool call, got '
               + c.tools_aborted + ' aborts');
  }

  console.log('');
  if (fails.length) {
    fails.forEach(f => console.log('  FAIL  ' + f));
    console.log('\n  PHONE: FAILED');
    return 1;
  }
  console.log('  PASS  five long answers, nobody talking, ZERO cut-offs');
  console.log('  PASS  ' + quiet + ' phantom transcripts refused, none of them '
              + 'cancelled anything');
  console.log('  PASS  a real interruption still stops her (2 barge-ins)');
  console.log('  PASS  a real second question over a live tool call still '
              + 'supersedes, and aborts it');
  console.log('\n  PHONE: PASSED');
  return 0;
}

async function main() {
  if (args.indexOf('--phone') >= 0) {
    process.exit(await phoneRun());
  }
  console.log('=========================================================');
  console.log(' live voice probe — %d scripted turns through the real', TURNS.length);
  console.log(' controller and the real arbiter');
  console.log(' voice: %s', voiceArg === 'elevenlabs'
    ? 'elevenlabs (text mode, through the dialogue sink)'
    : 'openai_realtime (the session speaks for itself)');
  console.log('=========================================================');
  RECOVERABLE = TURNS.filter(t => t.recoverable).length;
  // padEnd rather than '%-52s': node's console.log understands %d and %s and
  // nothing about widths, so the width spec was printed literally and the
  // column it was there to make never existed.
  TURNS.forEach((t, i) => console.log(
    '  ' + String(i + 1).padStart(2) + '. ' + t.name.padEnd(52)
    + (t.recoverable ? '(should survive)' : '')));

  const r = await drive({ sustain: 4, confirm: 40,
                          eleven: voiceArg === 'elevenlabs' });
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
