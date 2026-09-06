/* realtime_selftest.js — RIO's live session, driven without a microphone.
 *
 *   node tools/realtime_selftest.js
 *
 * The interesting failures in a speech-to-speech assistant are not the ones a
 * demo shows you. They are:
 *
 *   * a gap warning arriving while she is mid-sentence — does she actually
 *     stop, and does she stop GENERATING as well as playing, or does she carry
 *     on underneath the warning and reappear halfway through a word;
 *   * the driver starting to talk over her — does she yield instantly, or a
 *     beat later, which is the difference between a conversation and an
 *     argument with a machine;
 *   * the reasoning tool timing out — does she carry on, or does the turn die.
 *
 * None of those need audio, and all of them are reachable through the event
 * stream. static/rio_realtime.js keeps every decision in createController(),
 * a pure handler over an injected transport, precisely so this file can drive
 * it against the REAL arbiter from static/rio_speech.js.
 */
'use strict';

const path = require('path');

/* THE BROWSER GLOBAL THE SINK DECODES WITH, which node 12 has not got. It is
   installed before anything is required, because rio_voice_eleven reaches for
   it the moment a chunk of speech arrives and the failure is a thrown
   ReferenceError in the middle of a test rather than a red check. */
if (typeof global.atob !== 'function') {
  global.atob = (b64) => Buffer.from(b64, 'base64').toString('binary');
}

const rt = require(path.join(__dirname, '..', 'static', 'rio_realtime.js'));
const speech = require(path.join(__dirname, '..', 'static', 'rio_speech.js'));
// The real tracker, so the directions a test reads are computed the way the
// car computes them rather than written out by the test.
const navcore = require(path.join(__dirname, '..', 'static', 'rio_navcore.js'));

let checks = 0, failures = 0;
function ok(cond, what) {
  checks++;
  if (!cond) { failures++; console.log('  FAIL  ' + what); }
  else console.log('  ok    ' + what);
}
function section(name) { console.log('\n=== ' + name + ' ==='); }

/* ONE PAGE'S DOMContentLoaded HANDLERS, SHARED BY EVERY HARNESS IN THIS FILE.
 *
 * The same cached-require problem installBrowser explains at length, one step
 * further on. A panel module registers its init exactly once, against
 * whichever `document` existed when it was first required — so the SECOND
 * harness to build a page gets an empty handler list of its own and a
 * rio_nav.js still wired to the first harness's stubs. That reads as "the
 * panel is not subscribed to the position watch" in a section that has
 * nothing to do with whatever ran before it.
 *
 * So the registry is the file's, not the harness's: a new page re-runs the
 * handlers the modules actually registered, against the globals that are
 * current now. */
const pageInit = {};
const tick = () => new Promise(r => setTimeout(r, 0));
// Long enough for both barge-in timers in the harness (4 ms + 8 ms) to run.
// 40 ms unless a check needs to outlast a timer it set itself.
const settle = (ms) => new Promise(r => setTimeout(r, ms || 40));

/* One live session on a fake wire. `sent` is what went to the model, `muted`
   is what the driver can actually hear. */
function harness(opts) {
  opts = opts || {};
  const arbiter = speech.makeArbiter();
  const sent = [];
  const events = [];
  const audio = { muted: false };
  const controller = rt.createController({
    arbiter: arbiter,
    send: (obj) => sent.push(obj),
    tool: opts.tool || (() => Promise.resolve({ ok: true, answer: 'forty-two' })),
    audio: {
      mute: () => { audio.muted = true; },
      unmute: () => { audio.muted = false; },
    },
    onEvent: (ev) => events.push(ev),
    // Milliseconds instead of hundreds of them. The real values live in
    // config.py and travel with the session; what these tests check is the
    // SHAPE of the decision -- what happens before the gate, after it, and
    // after the transcript does or does not arrive -- and that shape is the
    // same at 4 ms as at 300.
    bargeSustainMs: opts.bargeSustainMs === undefined ? 4 : opts.bargeSustainMs,
    bargeConfirmMs: opts.bargeConfirmMs === undefined ? 8 : opts.bargeConfirmMs,
    maxResumes: opts.maxResumes,
    resumeInstruction: 'RESUME>>',
    toolSchemas: opts.toolSchemas,
    conditionalTools: opts.conditionalTools,
  });
  return { arbiter, sent, events, audio, controller,
           types: () => sent.map(e => e.type),
           evTypes: () => events.map(e => e.type),
           cutoffs: () => controller.state().cutoffs,
           // The resume, if one was asked for: a response.create carrying the
           // resume instruction rather than any other kind.
           resumeSent: () => sent.filter(
             e => e.type === 'response.create' && e.response &&
                  /^RESUME>>/.test(e.response.instructions || '')) };
}

function item(props) {
  let resolve;
  const it = Object.assign({
    play: () => new Promise(r => { resolve = r; }),
    stop: () => { if (resolve) resolve(); },
  }, props);
  it.finish = () => { if (resolve) resolve(); };
  return it;
}

async function main() {

// ---------------------------------------------------------------------------
section('speaking — RIO claims the mouth like everything else');
// ---------------------------------------------------------------------------
{
  const h = harness();
  h.controller.handle({ type: 'response.created', response: { id: 'r1' } });
  await tick();
  const speaking = h.arbiter.state().speaking;
  ok(!!speaking, 'starting a response takes the mouth');
  ok(speaking && speaking.priority === speech.P.CONVO,
     'at conversation priority — the tier that yields (P' +
     (speaking ? speaking.priority : '?') + ')');
  ok(speaking && speaking.group === 'convo',
     'in the same group as every other reply, so a newer one replaces it');
  ok(h.audio.muted === false, 'and the audio is audible');

  h.controller.handle({ type: 'response.done', response: { id: 'r1' } });
  await tick();
  ok(h.arbiter.state().speaking === null,
     'finishing releases it — the mouth is not held for the whole drive');
  ok(h.controller.state().counters.responses === 1, 'one response, counted');
}

// ---------------------------------------------------------------------------
section('a warning cuts her off — mid-word, and for real');
// ---------------------------------------------------------------------------
{
  const h = harness();
  h.controller.handle({ type: 'response.created', response: { id: 'r1' } });
  await tick();
  const warning = item({ priority: speech.P.SAFETY, group: 'headway',
                         id: 'too_close', text: 'Too close' });
  h.arbiter.say(warning);
  await tick();

  ok(h.arbiter.state().speaking.id === 'too_close',
     'a gap warning takes the mouth from her');
  ok(h.audio.muted === true,
     'her audio is muted instantly — the part already in flight goes quiet');
  ok(h.types().indexOf('response.cancel') >= 0,
     'and the model is told to STOP GENERATING, not just to be unheard');
  ok(h.types().indexOf('output_audio_buffer.clear') >= 0,
     'with the audio already queued discarded, so she cannot resume into the gap');
  ok(h.controller.state().counters.interrupted === 1, 'the interruption is counted');
  ok(h.controller.state().speaking === false,
     'and she is no longer holding a response open');

  warning.finish();
  await tick();
  ok(h.arbiter.state().speaking === null && h.arbiter.state().queued.length === 0,
     'nothing is resumed afterwards — the mouth moves forward');
}

// ---------------------------------------------------------------------------
section('...and so does a turn instruction, and a tire fault');
// ---------------------------------------------------------------------------
{
  for (const [tier, id, what] of [
    [speech.P.TURN_NEAR, 'nav:imminent', 'the imminent turn'],
    [speech.P.VEHICLE_HEALTH, 'health:blowout', 'a critical tire fault'],
    [speech.P.NAV, 'nav:primary', 'an ordinary navigation line'],
  ]) {
    const h = harness();
    h.controller.handle({ type: 'response.created', response: { id: 'r1' } });
    await tick();
    h.arbiter.say(item({ priority: tier, group: 'x:' + id, id: id }));
    await tick();
    ok(h.arbiter.state().speaking.id === id && h.audio.muted === true,
       what + ' interrupts her too');
  }
}

// ---------------------------------------------------------------------------
section('barge-in — the driver starts talking');
// ---------------------------------------------------------------------------
{
  const h = harness();
  h.controller.handle({ type: 'response.created', response: { id: 'r1' } });
  h.controller.handle({ type: 'response.output_audio_transcript.delta',
                        response_id: 'r1', delta: 'The car ahead is about' });
  await tick();
  ok(h.audio.muted === false, 'she is talking');
  h.controller.handle({ type: 'input_audio_buffer.speech_started' });
  await tick();
  ok(h.audio.muted === true,
     'the moment the driver speaks, she goes quiet — locally, without waiting '
     + 'for the server to agree');
  ok(h.arbiter.state().speaking !== null,
     'but the answer is NOT thrown away yet — muting is instant and undoable, '
     + 'cancelling is not');
  ok(h.types().indexOf('response.cancel') < 0,
     'and nothing has been cancelled inside the sustain window');
  ok(h.controller.state().counters.barge_ins === 1, 'counted as a barge-in');
  ok(h.events.some(e => e.type === 'LIVE_BARGE_IN'), 'and reported as one');

  await settle();
  ok(h.types().indexOf('response.cancel') >= 0,
     'speech that outlasts the gate does cancel her');
  ok(h.arbiter.state().speaking === null, 'and the mouth is released');
}

// ---------------------------------------------------------------------------
section('a noise that is not a driver — the answer survives it');
// ---------------------------------------------------------------------------
{
  // A blip inside the sustain window: a cough, a door, RIO's own voice coming
  // back through the cabin. The old behaviour threw the whole answer away for
  // one of these, which is the bug this section exists for.
  const h = harness();
  h.controller.handle({ type: 'response.created', response: { id: 'r1' } });
  h.controller.handle({ type: 'response.output_audio_transcript.delta',
                        response_id: 'r1', delta: 'The car ahead is about' });
  await tick();
  h.controller.handle({ type: 'input_audio_buffer.speech_started' });
  await tick();
  ok(h.audio.muted === true, 'she still goes quiet instantly — that part never changes');
  h.controller.handle({ type: 'input_audio_buffer.speech_stopped' });
  await tick();

  ok(h.audio.muted === false, 'the noise stops inside the gate and she carries on');
  ok(h.arbiter.state().speaking !== null, 'the answer was never given up');
  ok(h.types().indexOf('response.cancel') < 0, 'nothing was ever cancelled');
  ok(h.types().indexOf('input_audio_buffer.clear') >= 0,
     'and the blip is cleared out of the input buffer, best effort, so it does '
     + 'not commit as a turn of its own');
  ok(h.controller.state().counters.blips_absorbed === 1, 'counted as absorbed');
  ok(h.resumeSent().length === 0, 'nothing to resume — she never stopped');

  await settle();
  ok(h.arbiter.state().speaking !== null, 'and it stays that way past the gate');
}

// ---------------------------------------------------------------------------
section('FALSE barge-in — the detector fired and nobody spoke');
// ---------------------------------------------------------------------------
{
  const h = harness();
  h.controller.handle({ type: 'response.created', response: { id: 'r1' } });
  h.controller.handle({ type: 'response.output_audio_transcript.delta',
                        response_id: 'r1', delta: 'The car ahead is about thirty' });
  await tick();
  h.controller.handle({ type: 'input_audio_buffer.speech_started' });
  await new Promise(r => setTimeout(r, 6));   // outlasts the gate...
  ok(h.types().indexOf('response.cancel') >= 0, 'she stops generating');
  // ...the noise stops, and then nothing arrives. No transcript, because there
  // was nobody. The wait for one is timed from HERE, not from the cancel.
  h.controller.handle({ type: 'input_audio_buffer.speech_stopped' });
  await settle();

  ok(h.cutoffs().false_barge_in === 1,
     'with no transcript behind it, the cut-off is classified as FALSE');
  ok(h.cutoffs().barge_in === 0, 'and not as a real one');

  const r = h.resumeSent();
  ok(r.length === 1, 'the unfinished answer is resumed, once');
  ok(/thirty/.test(r[0].response.instructions),
     'carrying what she had already said, so she continues instead of '
     + 'starting the answer again');
  ok(h.events.some(e => e.type === 'LIVE_RESUME' && e.cause === 'false_barge_in'),
     'and it says why it resumed');
}

// ---------------------------------------------------------------------------
section('REAL interruption — a transcript followed, so it stays interrupted');
// ---------------------------------------------------------------------------
{
  const h = harness();
  h.controller.handle({ type: 'response.created', response: { id: 'r1' } });
  h.controller.handle({ type: 'response.output_audio_transcript.delta',
                        response_id: 'r1', delta: 'The car ahead is about thirty' });
  await tick();
  h.controller.handle({ type: 'input_audio_buffer.speech_started' });
  await tick(); await new Promise(r => setTimeout(r, 6));   // past the gate
  h.controller.handle({ type: 'input_audio_buffer.speech_stopped' });
  h.controller.handle({
    type: 'conversation.item.input_audio_transcription.completed',
    transcript: 'no, the other one' });
  await settle();

  ok(h.cutoffs().barge_in === 1, 'classified as a real interruption');
  ok(h.cutoffs().false_barge_in === 0, 'and not a false one');
  ok(h.resumeSent().length === 0,
     'and NOT resumed — finishing an answer somebody deliberately cut off is '
     + 'the rude version of the bug this fixes');
}

// ---------------------------------------------------------------------------
section('...and an empty transcript is evidence, not a failure');
// ---------------------------------------------------------------------------
{
  const h = harness();
  h.controller.handle({ type: 'response.created', response: { id: 'r1' } });
  h.controller.handle({ type: 'response.output_audio_transcript.delta',
                        response_id: 'r1', delta: 'It is a silver estate' });
  await tick();
  h.controller.handle({ type: 'input_audio_buffer.speech_started' });
  await tick(); await new Promise(r => setTimeout(r, 6));
  h.controller.handle({ type: 'input_audio_buffer.speech_stopped' });
  // The transcriber ran and found no words. That is the transcriber telling us
  // there was nobody there, and it arrives sooner than the timeout does.
  h.controller.handle({
    type: 'conversation.item.input_audio_transcription.completed',
    transcript: '   ' });
  await settle();
  ok(h.cutoffs().false_barge_in === 1, 'read as a false barge-in');
  ok(h.resumeSent().length === 1, 'and resumed without waiting out the timer');
}

// ---------------------------------------------------------------------------
section('ARBITER pre-emption — a warning takes the mouth, and gives it back');
// ---------------------------------------------------------------------------
{
  const h = harness();
  h.controller.handle({ type: 'response.created', response: { id: 'r1' } });
  h.controller.handle({ type: 'response.output_audio_transcript.delta',
                        response_id: 'r1', delta: 'That building on the left is' });
  await tick();
  ok(h.arbiter.state().speaking !== null, 'she has the mouth');

  const warning = item({ priority: h.arbiter.P.SAFETY, group: 'headway',
                         id: 'hw1', text: 'Too close' });
  h.arbiter.say(warning);
  await tick();

  ok(h.types().indexOf('response.cancel') >= 0,
     'the warning cuts her off mid-sentence — unchanged, and it has to be');
  ok(h.cutoffs().preempted === 1, 'the cut-off is recorded as a pre-emption');
  const cut = h.events.filter(e => e.type === 'LIVE_CUTOFF').pop();
  ok(cut && cut.by && cut.by.priority === h.arbiter.P.SAFETY,
     'naming what pre-empted it, and at what priority');
  ok(h.resumeSent().length === 0, 'nothing is resumed while the warning is speaking');

  warning.finish();
  await settle();
  ok(h.resumeSent().length === 1,
     'and once the warning is done the interrupted answer finishes itself, '
     + 'instead of the driver having to ask again');
  ok(/building on the left/.test(h.resumeSent()[0].response.instructions),
     'from where it left off');
}

// ---------------------------------------------------------------------------
section('a LONG real interruption is still a real interruption');
// ---------------------------------------------------------------------------
{
  /* The failure this exists for was in the fix, not in the original bug, and a
     ten-turn probe found it: the wait for a transcript was timed from the
     CANCEL. A driver saying two seconds of sentence produces a transcript two
     and a half seconds after the detector fired, the window expired long
     before, and RIO resumed her old answer over the top of somebody who was
     still finishing theirs -- worse than the cut-out it replaced.

     The clock starts when the SPEECH stops, which is the only moment after
     which a transcript can be expected at all. */
  const h = harness({ bargeConfirmMs: 8 });
  h.controller.handle({ type: 'response.created', response: { id: 'r1' } });
  h.controller.handle({ type: 'response.output_audio_transcript.delta',
                        response_id: 'r1', delta: 'The building on the left' });
  await tick();
  h.controller.handle({ type: 'input_audio_buffer.speech_started' });
  // A long sentence: far longer than the confirmation window would allow if
  // that window were being measured from the cancel.
  await new Promise(r => setTimeout(r, 60));
  ok(h.cutoffs().false_barge_in === 0,
     'while the driver is still talking, nothing has been declared false');
  ok(h.resumeSent().length === 0, '...and nothing has resumed over them');
  h.controller.handle({ type: 'input_audio_buffer.speech_stopped' });
  h.controller.handle({
    type: 'conversation.item.input_audio_transcription.completed',
    transcript: 'what is that building on the left' });
  await settle();
  ok(h.cutoffs().barge_in === 1, 'and when the words land it is a real interruption');
  ok(h.resumeSent().length === 0, 'still not resumed');
}

// ---------------------------------------------------------------------------
section('DESK TEST — a clip\'s warnings still fire, and stop costing the answer');
// ---------------------------------------------------------------------------
{
  /* Running headway on an uploaded clip is a test OF the warning path, so the
     warnings have to fire exactly as they do on the road -- same priority,
     same interruption, same voice. What they must not do any more is take the
     conversation down with them silently, which on a desk is how a warning
     working correctly looked identical to the live session having died.

     This is the real shape of it: the warning claims the mouth at P1 and its
     audio IS a dictation through the same session. */
  const h = harness();
  h.controller.handle({ type: 'response.created', response: { id: 'r1' } });
  h.controller.handle({ type: 'response.output_audio_transcript.delta',
                        response_id: 'r1', delta: 'The white van has been' });
  await tick();

  let released;
  const warning = {
    priority: h.arbiter.P.SAFETY, group: 'headway', id: 'headway:too_close',
    text: 'Too close', ttlMs: 2500,
    play: () => h.controller.speak('Too close.').then(() => {}),
    stop: () => {},
  };
  h.arbiter.say(warning);
  await tick();

  ok(h.controller.state().dictating === true,
     'the warning speaks in her own voice, dictated through the live session');
  ok(h.cutoffs().preempted === 1, 'the answer underneath it is recorded as pre-empted');
  ok(h.resumeSent().length === 0, 'and nothing resumes while the warning is talking');

  // The dictation plays and finishes.
  const dictated = h.sent.filter(e => e.type === 'response.create' &&
                                 e.response && e.response.conversation === 'none');
  ok(dictated.length === 1, 'the line went out as a verbatim, out-of-band response');
  h.controller.handle({ type: 'output_audio_buffer.started', response_id: 'd1' });
  await tick();
  h.controller.handle({ type: 'response.done', response: { id: 'd1' } });
  await settle();

  ok(h.resumeSent().length === 1,
     'the warning done, the interrupted answer finishes itself — on a desk and '
     + 'on the road, by the same rule');
  ok(/white van/.test(h.resumeSent()[0].response.instructions),
     'from where the warning cut it off');
}

// ---------------------------------------------------------------------------
section('a warning does not yield to a cough');
// ---------------------------------------------------------------------------
{
  const h = harness();
  const p = h.controller.speak('Too close.');
  p.catch(() => {});
  h.controller.handle({ type: 'output_audio_buffer.started', response_id: 'd1' });
  await tick();
  ok(h.audio.muted === false, 'the warning is speaking');

  h.controller.handle({ type: 'input_audio_buffer.speech_started' });
  await settle();
  ok(h.audio.muted === false,
     'and it keeps speaking through the driver — the one thing on the ladder '
     + 'that outranks them is the one that must not be silenced by a cough');
  ok(h.types().indexOf('response.cancel') < 0, 'nothing cancels it');
  ok(h.events.some(e => e.type === 'LIVE_BARGE_IN' && e.yielded === false),
     'recorded as a barge-in that was refused, not one that never happened');
}

// ---------------------------------------------------------------------------
section('resuming is bounded, and real speech ends it');
// ---------------------------------------------------------------------------
{
  const h = harness();
  // Cut off, resumed, cut off again: the second one is not resumed. An answer
  // in an argument with the cabin should stop, not keep saying "as I was
  // saying".
  h.controller.handle({ type: 'response.created', response: { id: 'r1' } });
  h.controller.handle({ type: 'response.output_audio_transcript.delta',
                        response_id: 'r1', delta: 'first half' });
  await tick();
  h.controller.handle({ type: 'input_audio_buffer.speech_started' });
  await new Promise(r => setTimeout(r, 6));
  h.controller.handle({ type: 'input_audio_buffer.speech_stopped' });
  await settle();
  ok(h.resumeSent().length === 1, 'first cut-off resumes');

  h.controller.handle({ type: 'response.created', response: { id: 'r2' } });
  h.controller.handle({ type: 'response.output_audio_transcript.delta',
                        response_id: 'r2', delta: 'second half' });
  await tick();
  h.controller.handle({ type: 'input_audio_buffer.speech_started' });
  await new Promise(r => setTimeout(r, 6));
  h.controller.handle({ type: 'input_audio_buffer.speech_stopped' });
  await settle();
  ok(h.resumeSent().length === 1, 'the second does not — one resume per answer');
  ok(h.controller.state().counters.resume_skipped === 1, 'and says it declined');
}
{
  const h = harness();
  h.controller.handle({ type: 'response.created', response: { id: 'r1' } });
  h.controller.handle({ type: 'response.output_audio_transcript.delta',
                        response_id: 'r1', delta: 'half an answer' });
  await tick();
  h.controller.handle({ type: 'response.done',
                        response: { id: 'r1', status: 'completed' } });
  await tick();
  ok(h.cutoffs().false_barge_in + h.cutoffs().barge_in + h.cutoffs().preempted
     + h.cutoffs().other === 0,
     'an answer that simply finished is not a cut-off of any kind');
}

// ---------------------------------------------------------------------------
section('the token cap is a limit, not a fault — counted, never resumed');
// ---------------------------------------------------------------------------
{
  const h = harness();
  h.controller.handle({ type: 'response.created', response: { id: 'r1' } });
  h.controller.handle({ type: 'response.output_audio_transcript.delta',
                        response_id: 'r1', delta: 'a very long answer indeed' });
  await tick();
  h.controller.handle({
    type: 'response.done',
    response: { id: 'r1', status: 'incomplete',
                status_details: { reason: 'max_output_tokens' } } });
  await settle();
  ok(h.cutoffs().token_cap === 1,
     'the cap is recorded, so an answer that stops at the same length every '
     + 'time is a fault with a name rather than an impression');
  ok(h.resumeSent().length === 0,
     'and not resumed — resuming it would be arguing with the limit');
}

// ---------------------------------------------------------------------------
section('the wire going away is its own cause');
// ---------------------------------------------------------------------------
{
  const h = harness();
  h.controller.handle({ type: 'response.created', response: { id: 'r1' } });
  h.controller.handle({ type: 'response.output_audio_transcript.delta',
                        response_id: 'r1', delta: 'mid sentence' });
  await tick();
  h.controller.transportLost('datachannel_closed');
  await settle();
  ok(h.cutoffs().transport === 1,
     'a dead channel is counted as a dead channel, not as her stopping');
  ok(h.resumeSent().length === 0, 'and nothing is resumed into a session that is gone');
  ok(h.audio.muted === true, 'the element is muted rather than left live');
}

// ---------------------------------------------------------------------------
section('the microphone asks for the constraints that make this work at all');
// ---------------------------------------------------------------------------
{
  const c = rt.micConstraints || {};
  ok(c.echoCancellation === true,
     'echoCancellation — without it the loudest thing the detector hears while '
     + 'she talks is her, and she interrupts herself');
  ok(c.noiseSuppression === true, 'noiseSuppression — road and wind');
  ok(c.autoGainControl === true, 'autoGainControl');
  const src = require('fs').readFileSync(
    path.join(__dirname, '..', 'static', 'rio_realtime.js'), 'utf8');
  const call = (src.match(/getUserMedia\(([^)]*)\)/) || [])[1] || '';
  ok(/MIC_CONSTRAINTS/.test(call),
     'and the real getUserMedia call uses them rather than a second copy that '
     + 'can drift');
}

// ---------------------------------------------------------------------------
section('escalation — she asks the reasoning model and speaks the answer');
// ---------------------------------------------------------------------------
{
  const asked = [];
  const h = harness({
    tool: (name, args) => { asked.push({ name, args });
                            return Promise.resolve({ ok: true, answer: 'It means the engine is running lean.',
                                                     took_ms: 5187 }); },
  });
  h.controller.handle({
    type: 'response.function_call_arguments.done',
    name: 'deep_dive', call_id: 'call_1',
    arguments: JSON.stringify({ question: 'What does P0171 mean?' }),
  });
  await tick(); await tick();

  ok(asked.length === 1 && asked[0].name === 'deep_dive',
     'the tool call reaches the server');
  ok(asked[0].args.question === 'What does P0171 mean?',
     'with the question parsed out of the arguments');
  const out = h.sent.find(e => e.type === 'conversation.item.create');
  ok(!!out && out.item.call_id === 'call_1' && out.item.type === 'function_call_output',
     'the result goes back into the session against the same call id');
  ok(out && JSON.parse(out.item.output).answer.indexOf('lean') >= 0,
     'carrying the answer');
  ok(h.types().indexOf('response.create') >= 0,
     'and she is asked to speak — without this she has the answer and no reason '
     + 'to say it');
  ok(h.controller.state().counters.tool_calls === 1 &&
     h.controller.state().counters.tool_failures === 0, 'counted as one clean call');
}

// ---------------------------------------------------------------------------
section('escalation fails — and RIO carries on');
// ---------------------------------------------------------------------------
{
  for (const [name, tool] of [
    ['the reasoning model times out',
     () => Promise.resolve({ ok: false, note: 'APITimeoutError' })],
    ['the tool endpoint is unreachable',
     () => Promise.reject(new Error('network'))],
    ['the server returns nothing at all', () => Promise.resolve(null)],
    ['the tool throws synchronously', () => { throw new Error('boom'); }],
  ]) {
    const h = harness({ tool });
    h.controller.handle({
      type: 'response.function_call_arguments.done',
      name: 'deep_dive', call_id: 'c', arguments: '{"question":"x"}',
    });
    await tick(); await tick();
    const out = h.sent.find(e => e.type === 'conversation.item.create');
    const body = out ? JSON.parse(out.item.output) : null;
    ok(body && body.ok === false, name + ' -> she is told ok:false');
    ok(h.types().indexOf('response.create') >= 0,
       '   ...and still asked to answer, so the turn does not die');
  }

  const h = harness({ tool: () => Promise.resolve({ ok: true, answer: 'x' }) });
  h.controller.handle({
    type: 'response.function_call_arguments.done',
    name: 'deep_dive', call_id: 'c', arguments: 'not json at all',
  });
  await tick(); await tick();
  ok(h.sent.some(e => e.type === 'conversation.item.create'),
     'unparseable arguments still produce a result rather than a dead turn');
}

// ---------------------------------------------------------------------------
section('the firewall — she only ever speaks as conversation');
// ---------------------------------------------------------------------------
{
  const h = harness();
  const priorities = [];
  const realSay = h.arbiter.say;
  h.arbiter.say = function (it) { priorities.push(it.priority); return realSay.call(h.arbiter, it); };

  h.controller.handle({ type: 'response.created', response: { id: 'r1' } });
  h.controller.handle({ type: 'output_audio_buffer.started', response_id: 'r1' });
  h.controller.handle({ type: 'response.done', response: { id: 'r1' } });
  h.controller.handle({ type: 'response.created', response: { id: 'r2' } });
  await tick();
  ok(priorities.length > 0 && priorities.every(p => p === speech.P.CONVO),
     'every item the live session creates is conversation priority (' +
     priorities.join(',') + ')');
  ok(priorities.length === 2,
     'and a duplicate audio-start does not open a second claim on the mouth');

  const src = require('fs').readFileSync(
    path.join(__dirname, '..', 'static', 'rio_realtime.js'), 'utf8');
  for (const forbidden of ['P.SAFETY', 'P.TURN_NEAR', 'P.VEHICLE_HEALTH', 'P.NAV']) {
    ok(src.indexOf(forbidden) < 0,
       'the live client cannot even name ' + forbidden + ' — no path to a safety tier');
  }
}

// ---------------------------------------------------------------------------
section('ending the session');
// ---------------------------------------------------------------------------
{
  const h = harness();
  h.controller.handle({ type: 'response.created', response: { id: 'r1' } });
  await tick();
  h.controller.stop();
  await tick();
  ok(h.arbiter.state().speaking === null,
     'stopping releases the mouth — a claim left open would mute every later reply');
  ok(h.audio.muted === true, 'and the audio is silenced');
  h.controller.handle({ type: 'response.created', response: { id: 'r2' } });
  await tick();
  ok(h.arbiter.state().speaking === null, 'a stopped session ignores further events');
}

// ---------------------------------------------------------------------------
section('transcripts');
// ---------------------------------------------------------------------------
{
  const h = harness();
  h.controller.handle({
    type: 'conversation.item.input_audio_transcription.completed',
    transcript: 'what does that light mean',
  });
  await tick();
  ok(h.controller.state().last_transcript === 'what does that light mean',
     'the driver-side transcript is available for the panel and the log');
  ok(h.events.some(e => e.type === 'LIVE_TRANSCRIPT'), 'and emitted as an event');
}

// ---------------------------------------------------------------------------
section('dictation — a deterministic line, in RIO\'s voice, word for word');
// ---------------------------------------------------------------------------
{
  const h = harness();
  const p = h.controller.speak('Back off — now.');
  await tick();
  const req = h.sent.find(e => e.type === 'response.create');
  ok(!!req, 'a dictated line goes out as a response.create');
  ok(req && req.response.conversation === 'none',
     'OUT OF BAND — a warning is a fact about the car, not something RIO said '
     + 'and can be asked about later');
  ok(req && req.response.output_modalities.join() === 'audio',
     'audio only: nothing is written back into the transcript');
  ok(req && req.response.instructions.indexOf('Back off — now.') >= 0,
     'carrying the exact words the policy wrote');
  ok(req && /word for word/i.test(req.response.instructions),
     'under an instruction to read them verbatim');

  ok(h.arbiter.state().speaking === null,
     'and it does NOT claim the mouth as conversation — the caller already '
     + 'holds it, at the priority the line deserves');

  h.controller.handle({ type: 'response.created', response: { id: 'd1' } });
  h.controller.handle({ type: 'output_audio_buffer.started', response_id: 'd1' });
  await tick();
  ok(h.audio.muted === false, 'the audio is audible while it speaks');
  ok(h.arbiter.state().speaking === null,
     'still no conversation item — a dictated response is not a reply');

  h.controller.handle({ type: 'response.output_audio_transcript.done',
                        response_id: 'd1', transcript: 'Back off — now.' });
  h.controller.handle({ type: 'response.done', response: { id: 'd1' } });
  const done = await p;
  ok(done.transcript === 'Back off — now.',
     'and it reports back what the model says it said, for the verbatim check');
  ok(h.controller.state().counters.dictated === 1, 'counted as one dictated line');
}

{
  const h = harness();
  let failed = null;
  h.controller.speak('Watch your distance.', { timeoutMs: 20 })
    .catch(e => { failed = e.message; });
  await new Promise(r => setTimeout(r, 40));
  ok(failed === 'timeout',
     'a dictation that never starts is abandoned, not waited on');
  ok(h.types().indexOf('response.cancel') >= 0,
     'and the model is told to drop it, so it cannot speak over the fallback');
  ok(h.controller.state().counters.dictation_failures === 1, 'counted as a failure');
}

{
  const h = harness();
  h.controller.handle({ type: 'response.created', response: { id: 'r1' } });
  await tick();
  ok(h.arbiter.state().speaking !== null, 'RIO is mid-sentence in conversation');
  const warning = item({ priority: speech.P.SAFETY, group: 'headway', id: 'warn' });
  h.arbiter.say(warning);
  await tick();
  const cancelAt = h.types().indexOf('response.cancel');
  // Nothing ever answers this dictation, so it times out ~700 ms later. Only
  // the ORDER of the two events matters here; the rejection is caught so a
  // longer-lived run (--server) does not print it as an unhandled one.
  h.controller.speak('You\'re too close.').catch(() => {});
  await tick();
  const speakAt = h.types().lastIndexOf('response.create');
  ok(cancelAt >= 0 && cancelAt < speakAt,
     'a warning cancels the conversation BEFORE it dictates over it — the two '
     + 'cannot overlap in one audio stream');
}

// ---------------------------------------------------------------------------
section('the fallback chain — a warning never waits on a cloud call');
// ---------------------------------------------------------------------------
{
  // rio_speak.js is the piece that chooses: dictate, else synthesise, else
  // play a clip. It runs in the browser, so the browser's three moving parts
  // are stubbed and nothing else is.
  const speak = require(path.join(__dirname, '..', 'static', 'rio_speak.js'));

  function stubBrowser(opts) {
    opts = opts || {};
    const played = [];
    global.URL = { createObjectURL: () => 'blob:x', revokeObjectURL: () => {} };
    global.fetch = (url) => {
      played.push('fetch:' + url);
      if (opts.ttsFails) return Promise.reject(new Error('offline'));
      return Promise.resolve({
        ok: true, headers: { get: () => null },
        blob: () => Promise.resolve({}),
      });
    };
    const element = {
      play: function () {
        played.push('element:' + (this.src || 'preloaded'));
        setTimeout(() => { if (this.onended) this.onended(); }, 0);
        return Promise.resolve();
      },
      pause: () => {},
    };
    global.RIO.realtime = { active: () => opts.session || null };
    speak.reset();
    return { played, element };
  }

  {
    const b = stubBrowser({
      session: { speak: () => Promise.resolve({ transcript: 'x' }),
                 speechEnabled: () => true },
    });
    const src = speak.provider({ text: 'You\'re too close.', channel: 'headway',
                                 ttsUrl: '/headway_voice?line=too_close',
                                 element: b.element });
    await src.play();
    ok(speak.stats().last === 'dictated',
       'with a session open, the line is dictated in RIO\'s voice');
    ok(b.played.length === 0, 'and nothing is fetched or synthesised');
  }

  {
    const b = stubBrowser({ session: null });
    const src = speak.provider({ text: 'You\'re too close.', channel: 'headway',
                                 ttsUrl: '/headway_voice?line=too_close',
                                 element: b.element });
    await src.play();
    ok(speak.stats().last === 'tts',
       'with NO session, it falls straight through to the synthesiser — a '
       + 'warning does not depend on a conversation being open');
    ok(b.played.some(p => p.indexOf('/headway_voice') >= 0),
       'through the endpoint that already existed');
  }

  {
    const b = stubBrowser({
      session: { speak: () => Promise.resolve(), speechEnabled: () => false },
    });
    const src = speak.provider({ text: 'Turn left by the Shell station.',
                                 channel: 'nav', ttsUrl: '/nav/voice?x=1',
                                 element: b.element });
    await src.play();
    ok(speak.stats().last === 'tts',
       'a channel switched off falls back too, session or no session');
  }

  {
    const b = stubBrowser({
      session: { speak: () => Promise.reject(new Error('timeout')),
                 speechEnabled: () => true },
    });
    const src = speak.provider({ text: 'You\'re too close.', channel: 'headway',
                                 ttsUrl: '/headway_voice?line=too_close',
                                 element: b.element });
    await src.play();
    ok(speak.stats().last === 'tts',
       'dictation that times out falls back rather than going silent');
  }

  {
    const b = stubBrowser({
      ttsFails: true,
      session: { speak: () => Promise.reject(new Error('timeout')),
                 speechEnabled: () => true },
    });
    const src = speak.provider({ text: 'You\'re too close.', channel: 'headway',
                                 ttsUrl: '/headway_voice?line=too_close',
                                 clipUrl: '/static/audio/too_close.mp3',
                                 element: b.element });
    await src.play();
    ok(speak.stats().last === 'clip',
       'and when the synthesiser is unreachable too, the pre-rendered clip '
       + 'plays — no network left in the path');
    ok(b.played.some(p => p.indexOf('too_close.mp3') >= 0), 'from the local file');
  }

  {
    const b = stubBrowser({ session: null, ttsFails: true });
    const src = speak.provider({ text: 'x', channel: 'nav',
                                 ttsUrl: '/nav/voice?x=1', element: b.element });
    let threw = false;
    await src.play().catch(() => { threw = true; });
    ok(threw && speak.stats().last === 'silent',
       'with nothing left to try it reports silence rather than pretending');
  }

  delete global.fetch;
  delete global.URL;
}

// ---------------------------------------------------------------------------
section('the clip bypass is untouched');
// ---------------------------------------------------------------------------
{
  const fs = require('fs');
  const html = fs.readFileSync(path.join(__dirname, '..', 'static', 'index.html'), 'utf8');
  ok(html.indexOf("playElement(el, null)") >= 0,
     'the red tier still plays its preloaded element directly, with no provider '
     + 'and no network in the path');
  const health = fs.readFileSync(path.join(__dirname, '..', 'static', 'rio_health.js'), 'utf8');
  ok(health.indexOf("ttsUrl: clip ? null :") >= 0,
     'and a health clip line is given no synthesiser url at all, so it cannot '
     + 'wait on one');
}

// ---------------------------------------------------------------------------
section('awareness — the three tools a live session needs');
// ---------------------------------------------------------------------------
{
  /* 1. A visual question reaches the camera, and what comes back is phrased —
   *    by HER, or by her observer, but never spoken as a raw caption.
   *
   *    The rule used to be "she phrases it, the pipeline does not", and it was
   *    enforced by there being exactly one way for a camera answer to be
   *    spoken: hand it to the model and ask. That cost ~450 ms of remote
   *    composition on an answer the camera had ready in four milliseconds, and
   *    it was most of the wait on the commonest question there is.
   *
   *    So there are two ways now, and the rule became "she or her observer
   *    phrases it — never a raw caption". The observer writes in her register
   *    and the SERVER checks each line against persona.lint() before it may be
   *    offered for direct speech; anything that fails comes back without
   *    `speak_directly` and takes the composing path exactly as before. Both
   *    halves are checked here. */
  const OBSERVATION = 'A white sedan two car lengths ahead in the same lane.';
  const h = harness({
    tool: (name, args) => {
      if (name !== 'look') return Promise.resolve({ ok: false, note: 'wrong tool' });
      return Promise.resolve({ ok: true, answer: OBSERVATION, took_ms: 1400 });
    },
  });
  h.controller.handle({
    type: 'response.function_call_arguments.done',
    name: 'look', call_id: 'v1',
    arguments: JSON.stringify({ question: "what's that car ahead" }),
  });
  await tick(); await tick();
  const out = h.sent.find(e => e.type === 'conversation.item.create');
  ok(out && JSON.parse(out.item.output).answer === OBSERVATION,
     'a visual question puts the camera observation back into the session');
  ok(h.types().indexOf('response.create') >= 0,
     'and an OBJECT question asks her to compose from it — that path is '
     + 'unchanged, because a crop needs a sentence written about it');
}

{
  // ...and a scene answer the observer already phrased is SPOKEN, not
  // rewritten. This is the pass that was removed, and this is the check that
  // it was removed only where the line is hers to begin with.
  const eleven = require(path.join(__dirname, '..', 'static', 'rio_voice_eleven.js'));
  const H = require(path.join(__dirname, 'voice_sink_harness.js'));
  const SPOKEN = 'Open freeway, light traffic — dry hills both sides';
  const arbiter = speech.makeArbiter();
  const sent = [];
  const events = [];
  const rig = H.openSink(eleven, {});
  const controller = rt.createController({
    arbiter: arbiter,
    send: (o) => sent.push(o),
    tool: () => Promise.resolve({
      ok: true, answer: SPOKEN, speech: SPOKEN, speak_directly: true,
      path: 'observer_direct', took_ms: 4, seen_s_ago: 0.6 }),
    audio: { mute: () => rig.sink.mute(), unmute: () => rig.sink.unmute() },
    voice: rig.sink,
    onEvent: (ev) => events.push(ev),
  });
  await rig.opened;
  controller.handle({
    type: 'response.function_call_arguments.done',
    name: 'look', call_id: 'v2',
    arguments: JSON.stringify({ question: 'what do you see' }),
  });
  await settle();

  ok(sent.filter(e => e.type === 'response.create').length === 0,
     'a scene answer asks NO model to compose it — that pass is the thing '
     + 'being removed');
  const said = rig.wire.ops('delta').map(d => d.text).join('');
  ok(said.indexOf('Open freeway') >= 0,
     'the observer\'s own sentence goes straight to the synthesiser (' +
     JSON.stringify(said) + ')');
  ok(rig.wire.ops('begin').length === 1,
     'as one utterance, with the mouth claimed for it');
  ok(arbiter.state().speaking && /^live:direct/.test(arbiter.state().speaking.id),
     'through the arbiter at conversation priority, so a warning still cuts '
     + 'through it exactly as it cuts through anything she says');

  const assistant = sent.find(e => e.type === 'conversation.item.create'
                              && e.item && e.item.role === 'assistant');
  ok(!!assistant, 'and the session is told what she said');
  ok(assistant && assistant.item.content[0].type === 'output_text'
     && assistant.item.content[0].text === SPOKEN,
     'as an assistant message carrying the exact words — so "tell me more '
     + 'about that" lands against a conversation that happened. output_text '
     + 'because the API refuses `text` by name');
  ok(controller.state().counters.spoken_directly === 1,
     'and the drive counts how many answers skipped the model');
}

{
  /* THE SAME ANSWER, ON THE BACKEND THAT HAS NO SYNTHESISER.
   *
   * Speech to speech: RIO's voice IS the session, so there is no sink to push
   * the observer's sentence into and it has to be READ by her -- the verbatim
   * injection the deterministic channels already use.
   *
   * This is a regression test for a SILENT drive, not for a nicety. Without
   * it the browser skipped speaking the line and asked for an ordinary
   * response anyway, while the tool result told the model in as many words
   * that the line "has ALREADY BEEN SAID to the driver". So she was told not
   * to repeat something nobody had said, and said nothing: measured on a real
   * drive as path=observer_direct, 44 ms to the camera, and one audio event
   * of silence.
   */
  const SPOKEN = 'Open freeway, light traffic — dry hills both sides';
  const arbiter = speech.makeArbiter();
  const sent = [];
  const events = [];
  const muted = [];
  const controller = rt.createController({
    arbiter: arbiter,
    send: (o) => sent.push(o),
    tool: () => Promise.resolve({
      ok: true, answer: SPOKEN, speech: SPOKEN, speak_directly: true,
      path: 'observer_direct', took_ms: 4, seen_s_ago: 0.6 }),
    audio: { mute: () => muted.push('mute'), unmute: () => muted.push('unmute') },
    voice: null,                       // <-- the whole difference
    verbatimInstruction: 'READ THIS VERBATIM>>',
    directSpeechTimeoutMs: 5000,
    onEvent: (ev) => events.push(ev),
  });
  controller.handle({
    type: 'response.function_call_arguments.done',
    name: 'look', call_id: 'v3',
    arguments: JSON.stringify({ question: 'what do you see' }),
  });
  await settle();

  const creates = sent.filter(e => e.type === 'response.create');
  ok(creates.length === 1,
     'exactly one response is asked for, and it is the line being read — not '
     + 'a model being asked to compose an answer that was already written');
  const req = creates[0].response || {};
  ok(req.conversation === 'none',
     'out of band, like every other verbatim line: the words are already going '
     + 'into the history as an assistant message, and twice would be twice');
  ok((req.output_modalities || [])[0] === 'audio', 'as audio');
  ok(req.instructions === 'READ THIS VERBATIM>>' + SPOKEN,
     'with the observer\'s exact sentence at the end of the verbatim '
     + 'instruction, which is the same injection a warning uses');
  ok(arbiter.state().speaking && /^live:direct/.test(arbiter.state().speaking.id),
     'through the arbiter at conversation priority, so a warning still cuts '
     + 'through it exactly as it cuts through anything she says');
  const assistantYet = () => sent.find(e => e.type === 'conversation.item.create'
                                    && e.item && e.item.role === 'assistant');
  /* NOT YET. Through the session the words are a REQUEST, and an assistant
     message is a claim about what the driver HEARD. Writing it here would be
     writing it before the line has made a sound -- and a line that then fails
     to start leaves a history saying she answered a question she was silent
     on. Through the sink it is written at this point, because handing the
     words to the sink IS the line being spoken. */
  ok(!assistantYet(),
     'the session is NOT told she said it yet — the words have only been '
     + 'asked for, and nothing has come out of the speaker');
  ok(controller.state().counters.spoken_directly === 1,
     'counted as an answer that skipped the model');

  /* ...AND THE MOUTH IS GIVEN BACK WHEN THE LINE IS DONE, not before and not
     never. A direct entry that is never released holds CONVO priority for the
     rest of the drive, and the next question is answered into a mouth that is
     still busy. */
  controller.handle({ type: 'response.created', response: { id: 'rd1' } });
  controller.handle({ type: 'output_audio_buffer.started', response_id: 'rd1' });
  ok(arbiter.state().speaking && /^live:direct/.test(arbiter.state().speaking.id),
     'the injected response does not claim the mouth for itself — the direct '
     + 'entry already holds it');
  const assistant = assistantYet();
  ok(!!assistant && assistant.item.content[0].text === SPOKEN,
     'and NOW the session is told what she said, in the same words — written '
     + 'from the audio starting, which is the moment it became true');
  controller.handle({ type: 'response.done',
                      response: { id: 'rd1', status: 'completed' } });
  await settle();
  ok(!arbiter.state().speaking,
     'and when it finishes, the mouth is free again');
  ok(controller.state().counters.direct_speech_failures === 0,
     'with nothing counted as a failure');
}

{
  /* A LINE THAT NEVER STARTS IS NOT A LINE THAT WAS SAID.
   *
   * The injection is given the same budget a dictated warning gets, and the
   * same treatment when it runs out: cancel, and say so. Reporting it as
   * spoken is how a turn ends silent with the history claiming an answer,
   * which is the failure this whole path was rebuilt around. */
  const arbiter = speech.makeArbiter();
  const sent = [];
  const events = [];
  const controller = rt.createController({
    arbiter: arbiter,
    send: (o) => sent.push(o),
    tool: () => Promise.resolve({
      ok: true, answer: 'x', speech: 'Dry hills both sides',
      speak_directly: true, path: 'observer_direct' }),
    audio: { mute: () => {}, unmute: () => {} },
    voice: null,
    verbatimInstruction: 'V>>',
    // The DIRECT budget, not the warning one: they are separate numbers
    // because the argument for 900 ms is about warnings and does not apply
    // to an answer. See config.REALTIME_DIRECT_SPEECH_TIMEOUT_MS.
    directSpeechTimeoutMs: 20,
    onEvent: (ev) => events.push(ev),
  });
  controller.handle({
    type: 'response.function_call_arguments.done',
    name: 'look', call_id: 'v4',
    arguments: JSON.stringify({ question: 'what do you see' }),
  });
  await settle(80);
  ok(sent.some(e => e.type === 'response.cancel'),
     'a line that has not started speaking inside the budget is cancelled');
  ok(events.some(e => e.type === 'LIVE_DIRECT_SPEECH_FAILED'),
     '...and reported, rather than counted as spoken');
  ok(controller.state().counters.direct_speech_failures === 1,
     '...and counted, so a drive can say how often it happened');
  ok(!arbiter.state().speaking,
     '...and the mouth is handed back either way — a failed line must not '
     + 'hold conversation priority for the rest of the drive');
  ok(!sent.some(e => e.type === 'conversation.item.create'
                && e.item && e.item.role === 'assistant'),
     '...and the session is NOT told she said a line that never made a sound');
  ok(sent.filter(e => e.type === 'response.create').length === 2,
     '...and the question is asked of the model instead, so it is answered '
     + 'late rather than never');

  /* AND A RESPONSE THAT ENDS WITHOUT EVER MAKING A SOUND IS THE SAME THING.
     `response.done` with status completed is the model's account of itself;
     whether the driver heard anything is a question about audio. */
  const c2 = rt.createController({
    arbiter: speech.makeArbiter(),
    send: () => {},
    tool: () => Promise.resolve({ ok: true, speech: 'Quiet road',
                                  speak_directly: true, path: 'observer_direct' }),
    audio: { mute: () => {}, unmute: () => {} },
    voice: null, verbatimInstruction: 'V>>', directSpeechTimeoutMs: 5000,
    onEvent: () => {},
  });
  c2.handle({ type: 'response.function_call_arguments.done', name: 'look',
              call_id: 'v5', arguments: '{}' });
  await settle();
  c2.handle({ type: 'response.created', response: { id: 'rd2' } });
  c2.handle({ type: 'response.done',
              response: { id: 'rd2', status: 'completed' } });
  await settle();
  ok(c2.state().counters.direct_speech_failures === 1,
     'a response that completed without ever starting audio is a line the '
     + 'driver did not hear, whatever its status says');
}

{
  /* A LINE GIVEN UP ON IS NOT A CUT-OFF ANSWER.
   *
   * Both out-of-band paths -- a dictated warning and an injected line -- are
   * recognised by their own state still being set when `response.done`
   * arrives. Give up on one early and that state is gone, so the late done
   * fell through to the CONVERSATION's branch and was filed as a truncated
   * answer.
   *
   * Measured on a real drive: a scene answer whose audio started a little
   * after its budget spoke perfectly well -- the full sentence, in her voice
   * -- and appeared in the tally as `token_cap`. A cut-off count that includes
   * lines nobody lost is worse than no count, because the number is the entire
   * reason it exists. */
  const cuts = [];
  const arbiter = speech.makeArbiter();
  const controller = rt.createController({
    arbiter: arbiter,
    send: () => {},
    tool: () => Promise.resolve({ ok: true, speech: 'Dry hills both sides',
                                  speak_directly: true, path: 'observer_direct' }),
    audio: { mute: () => {}, unmute: () => {} },
    voice: null, verbatimInstruction: 'V>>', directSpeechTimeoutMs: 20,
    onEvent: (ev) => { if (ev.type === 'LIVE_CUTOFF') cuts.push(ev); },
  });
  controller.handle({ type: 'response.function_call_arguments.done',
                      name: 'look', call_id: 'v6', arguments: '{}' });
  await settle();
  controller.handle({ type: 'response.created', response: { id: 'rd3' } });
  await settle(80);                       // the budget expires here
  // ...and the response the model was already generating lands afterwards.
  controller.handle({ type: 'response.done', response: { id: 'rd3',
    status: 'incomplete', status_details: { reason: 'max_output_tokens' } } });
  await settle();
  ok(cuts.length === 0,
     'a line the injection gave up on is not counted as a cut-off answer, '
     + 'even when the response that follows it says max_output_tokens');
  ok(controller.state().counters.cutoffs === undefined
     || !controller.state().counters.cutoffs,
     '...and nothing about the conversation was disturbed by it');
}

{
  // A line that is NOT in her voice never reaches the speaker unrewritten.
  // This is the whole of "never a raw caption": the server decides, and the
  // panel does what it is told.
  const eleven = require(path.join(__dirname, '..', 'static', 'rio_voice_eleven.js'));
  const H = require(path.join(__dirname, 'voice_sink_harness.js'));
  const CAPTION = 'The image shows a road with several cars on it.';
  const sent = [];
  const rig = H.openSink(eleven, {});
  const controller = rt.createController({
    arbiter: speech.makeArbiter(),
    send: (o) => sent.push(o),
    // No speak_directly: the server linted it and would not offer it.
    tool: () => Promise.resolve({ ok: true, answer: CAPTION,
                                  path: 'observer_composed', took_ms: 5 }),
    audio: { mute: () => {}, unmute: () => {} },
    voice: rig.sink,
  });
  await rig.opened;
  controller.handle({
    type: 'response.function_call_arguments.done',
    name: 'look', call_id: 'v3',
    arguments: JSON.stringify({ question: 'what do you see' }),
  });
  await settle();
  ok(sent.filter(e => e.type === 'response.create').length === 1,
     'a caption the observer could not phrase goes back to HER to compose');
  ok(rig.wire.ops('delta').length === 0,
     'and not one word of it is spoken as it stands');
}

{
  // 2. A route question is answered by the PANEL, from the same state the nav
  //    card paints, because progress lives in the browser and nowhere else.
  const rt = require(path.join(__dirname, '..', 'static', 'rio_realtime.js'));
  global.RIO = global.RIO || {};
  global.RIO.nav = { state: () => ({
    destination: { display_name: 'Griffith Observatory' },
    remaining_m: 4200, eta_epoch: 1787790000, speed_ms: 14,
    maneuver: { instruction: 'Turn left onto Lincoln Boulevard',
                direction: 'LEFT', road_name: 'Lincoln Boulevard' },
    to_maneuver_m: 180, tta_s: 12.4, maneuver_state: 'APPROACHING',
    maneuvers_left: 6, route_state: 'ON_ROUTE', gps_state: 'GPS_OK',
    arrived: false, context: { anchor: { label: 'Shell' } },
  }) };
  const st = rt.navStatus();
  ok(st.ok && st.destination === 'Griffith Observatory',
     'nav_status answers with the real destination');
  ok(st.next_maneuver.distance_m === 180 && st.next_maneuver.seconds_away === 12,
     'the real distance and time to the next maneuver (' +
     st.next_maneuver.distance_m + ' m, ' + st.next_maneuver.seconds_away + ' s)');
  ok(st.route_state === 'ON_ROUTE' && st.gps_state === 'GPS_OK',
     'off-route state and GPS health, which are answers in their own right');
  ok(/do not announce/i.test(st.rules || ''),
     'and the result itself carries the boundary: answer, do not announce');

  global.RIO.nav = { state: () => null };
  const none = rt.navStatus();
  ok(none.ok && none.routing === false && /no route/.test(none.note),
     'with no route set she is told to say so rather than guess a destination');
  delete global.RIO.nav;
}

{
  // 4. ANTI-DOUBLE-SPEAK. The turn the arbiter is about to call must not also
  //    be announced by the model. The tool tells her about it; the instruction
  //    tells her not to say it; and the deterministic call is a separate,
  //    higher-priority item she cannot pre-empt.
  const fs = require('fs');
  const rtSrc = fs.readFileSync(
    path.join(__dirname, '..', 'static', 'rio_realtime.js'), 'utf8');
  ok(/do not announce this maneuver/i.test(rtSrc),
     'every nav_status result repeats the boundary to the model');

  const h2 = harness();
  h2.controller.handle({ type: 'response.created', response: { id: 'chat' } });
  await tick();
  ok(h2.arbiter.state().speaking.priority === speech.P.CONVO,
     'anything the model says is conversation priority, whatever it is about');
  const turn = item({ priority: speech.P.TURN_NEAR, group: 'nav:m3', id: 'nav:imminent' });
  h2.arbiter.say(turn);
  await tick();
  ok(h2.arbiter.state().speaking.id === 'nav:imminent',
     'so if she did start talking over a turn call, the turn wins — the '
     + 'boundary is enforced by the ladder as well as by the instructions');
  ok(h2.types().indexOf('response.cancel') >= 0,
     'and she is cancelled at the model, not merely muted');
}

{
  // 5. FRAME SUPPLY. The link that was actually missing: the ring is fed by
  //    the drive loop, and a live session did not start one.
  const fs = require('fs');
  const html = fs.readFileSync(
    path.join(__dirname, '..', 'static', 'index.html'), 'utf8');
  ok(/if \(!RIO\.driving\) return;/.test(html),
     'the drive frame loop still stops itself when no drive is running');
  ok(/async function startLiveFrames/.test(html),
     'and a live session now starts its own feed');
  const liveBody = (html.match(/async function startLiveFrames[\s\S]*?\n}/) || [''])[0];
  ok(/RIO\.headway\.captureFromVideo\(/.test(liveBody),
     'through the SAME capture path the drive uses — no second pipeline');
  // ...from whatever the SOURCE is. It used to capture `video`, the live
  // camera element, having just opened a camera to fill it — which is how an
  // uploaded clip got replaced by the driver's face the moment a conversation
  // started. See static/rio_source.js and tools/source_selftest.js.
  ok(/captureFromVideo\(feed\.element\)/.test(liveBody),
     'and from the source the driver chose, not a camera of its own');
  ok(!/getUserMedia/.test(liveBody),
     'which it no longer opens');
  ok(/if \(RIO\.driving \|\| liveFrames\.timer/.test(html),
     'and not at all when a drive is already feeding the ring faster');
  ok(/stopLiveFrames\(\)/.test(html) &&
     (html.match(/stopLiveFrames\(\)/g) || []).length >= 3,
     'stopped when the conversation ends, and when it fails to start');
}

// ---------------------------------------------------------------------------
section('routing — asked to go somewhere, she takes them there');
// ---------------------------------------------------------------------------
/* The bug this covers: RIO could describe a route in detail and then tell the
   driver to set it themselves, because every navigation tool she had was a
   read. She now has one that acts, and the things worth asserting about it are
   that it goes through the panel's OWN routing entry point, that the route is
   really live afterwards, and that ambiguity still stops her — a tool that
   silently picked one of two Gettys would be worse than the bug. */

/* A stand-in for the navigation panel, with the same surface rio_nav.js
   exposes: routeToQuery resolves an outcome, state() answers nav_status from
   whatever is actually routed. `behaviour` decides what the provider says. */
/* A route fixture the real tracker will accept: a straight line long enough to
   hold the fixture's maneuvers, with each one pinned to the vertex its distance
   along the route puts it at. Ten-metre spacing, so a maneuver at 400 m is
   vertex 40 and no nearest-point search is involved. */
function trackable(route) {
  const lat0 = 34.0, lng0 = -118.4;
  const mLat = 111320, mLng = 111320 * Math.cos(lat0 * Math.PI / 180);
  const pts = [];
  const total = route.total_distance_m || 1000;
  for (let d = 0; d <= total + 10; d += 10) pts.push([lat0, lng0 + d / mLng]);
  return {
    route_id: 'r1', journey_id: 'j1', generation_id: 1, provider: 'fixture',
    geometry: pts, eta_epoch: route.eta_epoch || 0,
    destination: { display_name: 'fixture', lat: lat0, lng: lng0 },
    total_distance_m: total, arrival: { side: 'RIGHT' },
    maneuvers: (route.maneuvers || []).map(m => Object.assign({}, m, {
      lat: lat0, lng: lng0 + (m.route_distance_position || 0) / mLng,
      polyline_index: Math.round((m.route_distance_position || 0) / 10),
    })),
  };
}

function fakePanel(behaviour) {
  const panel = { asked: [], unlocked: false, routed: null, logged: [] };
  panel.nav = {
    unlock: () => { panel.unlocked = true; },
    routeToQuery: (text) => {
      panel.asked.push(text);
      const out = behaviour(text, panel.asked.length);
      if (out.status === 'routed') panel.routed = out;
      return Promise.resolve(out);
    },
    state: () => (panel.routed ? {
      destination: panel.routed.destination,
      remaining_m: panel.routed.route.total_distance_m,
      eta_epoch: panel.routed.route.eta_epoch, speed_ms: 14,
      maneuver: null, to_maneuver_m: null, tta_s: null,
      maneuver_state: 'NONE', maneuvers_left: 8,
      route_state: 'ON_ROUTE', gps_state: 'GPS_OK', arrived: false,
    } : null),
    /* The same shape rio_nav.js's directions() returns, and the maneuver list
       comes from the REAL tracker rather than from a hand-written array: what
       is under test is the whole path from "a route is loaded" to "she can
       read it", and a stubbed list would skip the half of it that computes
       how far away each turn is. */
    directions: (count) => {
      if (!panel.routed) return null;
      const r = panel.routed.route;
      const tracker = navcore.create(trackable(r));
      return {
        destination: panel.routed.destination,
        route_id: 'r1', generation_id: 1, landmarks_state: 'ready',
        total_maneuvers: (r.maneuvers || []).length,
        remaining_m: r.total_distance_m, eta_epoch: r.eta_epoch,
        route_state: 'ON_ROUTE', gps_state: 'GPS_OK', arrived: false,
        maneuvers: tracker.upcoming(count),
      };
    },
  };
  global.RIO = global.RIO || {};
  global.RIO.nav = panel.nav;
  global.RIO.bus = { emit: (type, payload) => panel.logged.push([type, payload]) };
  return panel;
}

/* A landmark candidate as the map hands one over: a place the lookup found
   near a turn, with the relation that makes a sentence about it true. NOT a
   sighting — nothing has looked at it yet. */
const SHELL = {
  anchor_id: 'm0a0', place_id: 'p_shell', label: 'Shell',
  spoken_label: 'the Shell station', type: 'gas_station', salience: 1.0,
  relation: 'JUST_AFTER', relation_confidence: 0.81,
  distance_to_maneuver_m: 24.0, speech: 'Turn right just after the Shell station.',
};

const LAX_MANEUVERS = [
  { id: 'm0', sequence: 0, type: 'TURN', direction: 'RIGHT',
    road_name: 'Lincoln Boulevard', instruction: 'Turn right onto Lincoln Boulevard',
    route_distance_position: 400, polyline_index: 4, anchors: [SHELL], speech: {} },
  { id: 'm1', sequence: 1, type: 'TURN', direction: 'LEFT',
    road_name: 'Sunset Boulevard', instruction: 'Turn left onto Sunset Boulevard',
    route_distance_position: 3200, polyline_index: 32, anchors: [], speech: {} },
  { id: 'm2', sequence: 2, type: 'ARRIVE', direction: 'RIGHT',
    road_name: '', instruction: 'Arrive at Los Angeles International Airport',
    route_distance_position: 21400, polyline_index: 214, anchors: [], speech: {} },
];

const LAX = {
  status: 'routed',
  destination: { display_name: 'Los Angeles International Airport',
                 formatted_address: '1 World Way, Los Angeles, CA 90045' },
  route: { total_distance_m: 21400, duration_s: 1080, eta_epoch: 1787790000,
           maneuvers: LAX_MANEUVERS },
};

/* The tool bridge exactly as connect() wires it: names in LOCAL_TOOLS are
   answered in the page, everything else goes to the server. */
function panelTools() {
  return (name, args) => (rt.localTools[name]
    ? Promise.resolve(rt.localTools[name](args))
    : Promise.resolve({ ok: false, note: 'not local' }));
}

{
  // 1. "TAKE ME TO LAX." One tool call, one route, one confirmation.
  const panel = fakePanel(() => LAX);
  const h = harness({ tool: panelTools() });
  h.controller.handle({
    type: 'response.function_call_arguments.done',
    name: 'start_navigation', call_id: 'n1',
    arguments: JSON.stringify({ destination: 'take me to LAX' }),
  });
  await tick(); await tick(); await tick();

  ok(panel.asked.length === 1 && panel.asked[0] === 'take me to LAX',
     'the spoken destination reaches routeToQuery — the same function the '
     + 'destination box calls on Enter, not a second router');
  ok(panel.unlocked, 'and announcement audio is unlocked, so the first turn '
     + 'call is not the thing that discovers it was not');

  const out = JSON.parse(h.sent.find(e => e.type === 'conversation.item.create')
                              .item.output);
  ok(out.ok === true && out.routing === true && out.status === 'routed',
     'the tool comes back saying the route is live');
  ok(out.destination === 'Los Angeles International Airport',
     "spelled the PROVIDER's way, not the driver's — LAX and LAS are one "
     + 'letter apart and she repeats back the one that was routed to');
  ok(out.minutes === 18, 'with the number she confirms with (' + out.minutes
     + ' minutes)');
  ok(/do NOT tell the driver to set it themselves/i.test(out.rules || ''),
     'and the result itself closes the door on the old behaviour');
  ok(/If they ASK for the directions, call nav_directions/i.test(out.rules || ''),
     'and points her at the tool for when they ask for the turns — reading '
     + 'them is answering');
  ok(/never do is call a turn as it arrives/i.test(out.rules || ''),
     'while leaving the CALLS exactly where they were: not hers');
  ok(out.total_maneuvers === 3 && out.first_steps.length === 3,
     'the confirmation carries the route summary — ' + out.total_maneuvers
     + ' maneuvers, first ' + out.first_steps.length
     + ' spelled out, so she needs no second call to confirm from');
  ok(out.first_steps[0].road_name === 'Lincoln Boulevard'
     && out.first_steps[0].distance_from_start_m === 400,
     'with real road names and real distances (' + out.first_steps[0].road_name
     + ' at ' + out.first_steps[0].distance_from_start_m + ' m)');
  ok(out.eta_epoch === 1787790000,
     'and the ETA, which is the other half of a confirmation');
  ok(h.types().indexOf('response.create') >= 0,
     'she is then asked to say so out loud — the driver hears a confirmation, '
     + 'not silence');

  // ROUTE ACTIVE, checked the way the driver would check it: by asking.
  const st = rt.navStatus();
  ok(st.ok && st.routing === true &&
     st.destination === 'Los Angeles International Airport',
     'and the route is genuinely active afterwards — nav_status answers from '
     + 'it with no dashboard step in between');
  ok(panel.logged.some(e => e[0] === 'NAV_VOICE_DESTINATION' &&
                            e[1].status === 'routed'),
     'the drive log records that this one was set by voice');
}

{
  // 1b. "WHAT ARE THE DIRECTIONS?" The bug this section exists for: RIO could
  //     start a route and then say she could not read it. nav_status answers
  //     "what is the next turn", which is a different question, and there was
  //     nothing else to ask. The list was in the tracker the whole time.
  const panel = fakePanel(() => LAX);
  const h = harness({ tool: panelTools() });
  h.controller.handle({
    type: 'response.function_call_arguments.done',
    name: 'start_navigation', call_id: 'd0',
    arguments: JSON.stringify({ destination: 'take me to LAX' }),
  });
  await tick(); await tick(); await tick();

  const dir = rt.navDirections({});
  ok(dir.ok === true && dir.routing === true,
     'with a route live, nav_directions answers from it');
  ok(dir.destination === 'Los Angeles International Airport',
     'naming the destination the route was actually built to');
  ok(dir.steps.length === 3 && dir.steps[0].road_name === 'Lincoln Boulevard'
     && dir.steps[1].road_name === 'Sunset Boulevard',
     'with the REAL steps of that route, in order (' +
     dir.steps.map(s2 => s2.road_name || s2.maneuver_type).join(' -> ') + ')');
  ok(dir.steps[0].direction === 'RIGHT' && dir.steps[1].direction === 'LEFT',
     'each carrying which way to turn');
  ok(dir.steps[0].distance_m === 400 && dir.steps[1].distance_m === 3200,
     'and how far to it (' + dir.steps[0].distance_m + ' m, ' +
     dir.steps[1].distance_m + ' m)');
  ok(dir.steps[1].leg_m === 2800,
     'plus the gap from the previous turn, which is what makes it read as ' +
     'directions rather than as a table (' + dir.steps[1].leg_m + ' m)');
  ok(dir.total_maneuvers === 3 && dir.truncated === false,
     'and says whether there are more than it listed');

  // THE LANDMARK, AS AN EXPECTATION. The map found a Shell near the first
  // turn; nothing has looked at it. That difference has to survive into what
  // she is allowed to say.
  const lm = dir.steps[0].landmark;
  ok(lm && lm.label === 'the Shell station',
     'a maneuver with a landmark candidate carries it (' +
     (lm ? lm.label : 'none') + ')');
  ok(lm && lm.phrase === 'just after the Shell station',
     'phrased by its RELATION to the turn, not as "near" for everything (' +
     (lm ? lm.phrase : 'none') + ')');
  ok(lm && lm.verified === false,
     'and marked unverified — this is a map lookup, not a sighting');
  ok(dir.steps[1].landmark === undefined,
     'a turn with no candidate gets no landmark rather than an invented one');
  ok(/there should be a Shell/i.test(dir.rules || ''),
     'the rules make her phrase it as an expectation, in those words');
  ok(/that is answering, not announcing/i.test(dir.rules || ''),
     'they say reading these IS answering...');
  ok(/Do NOT call any of these turns as instructions now/i.test(dir.rules || ''),
     '...and that calling them is still not hers');
  ok(!/^\s*1[.)]/m.test(dir.rules || '') && /not as a numbered list/i.test(dir.rules || ''),
     'and ask for a spoken answer rather than a list read aloud');

  // COUNT. The default is a few; "all" is the whole route.
  const five = rt.navDirections({ count: 2 });
  ok(five.steps.length === 2 && five.truncated === true,
     'a count truncates, and says so');
  const all = rt.navDirections({ count: 'all' });
  ok(all.steps.length === 3 && all.truncated === false,
     '"all" reads the whole route');
  const junk = rt.navDirections({ count: 'some' });
  ok(junk.steps.length === 3,
     'and anything unreadable as a number is treated as "all" rather than ' +
     'as an error the driver has to hear about');

  // The tool bridge answers it in the PANEL, like the other two.
  ok(typeof rt.localTools.nav_directions === 'function',
     'nav_directions is answered in the browser, where the route lives');
}

{
  // 1b-ii. "TAKE ME TO THE SECOND ONE." A place RIO just read out is already
  //        resolved: find_places returned its place_id, so start_navigation
  //        routes to THAT place rather than re-resolving a name. Re-resolving
  //        "Blue Bottle" as text can land on a different branch three miles
  //        away, with the right name and the wrong coffee.
  const panel = fakePanel(() => LAX);
  let setRouteArgs = null;
  panel.nav.setRoute = (opts) => {
    setRouteArgs = opts;
    return Promise.resolve({ ok: true, route: {
      total_distance_m: 1200, duration_s: 240, eta_epoch: 1787790000,
      destination: { display_name: 'Blue Bottle Coffee' }, maneuvers: [] } });
  };
  const h = harness({ tool: panelTools() });
  h.controller.handle({
    type: 'response.function_call_arguments.done',
    name: 'start_navigation', call_id: 'p1',
    arguments: JSON.stringify({ destination: 'Blue Bottle Coffee',
                                place_id: 'p_bluebottle' }),
  });
  await tick(); await tick(); await tick();

  ok(panel.asked.length === 0,
     'a place_id skips resolution entirely — routeToQuery is never called');
  ok(setRouteArgs && setRouteArgs.place_id === 'p_bluebottle',
     'the id goes straight to setRoute, the way tapping a suggestion does');
  ok(setRouteArgs && setRouteArgs.label === 'Blue Bottle Coffee',
     'with the name for the panel to show');
  const out = JSON.parse(h.sent.find(e => e.type === 'conversation.item.create')
                              .item.output);
  ok(out.ok === true && out.status === 'routed',
     'and the route comes back live, like any other');
  ok(out.destination === 'Blue Bottle Coffee',
     'named by the route it actually built (' + out.destination + ')');
  ok(panel.logged.some(e => e[0] === 'NAV_VOICE_DESTINATION'
                            && e[1].from_places === true
                            && e[1].place_id === 'p_bluebottle'),
     'and the drive log records that this one came from a place she read out');
}

{
  // 1c. NO ROUTE. The honest answer, not an invented one.
  fakePanel(() => LAX);
  global.RIO.nav.directions = () => null;
  const none = rt.navDirections({});
  ok(none.ok === true && none.routing === false && /no route/.test(none.note),
     'with no route set she is told to say so, not to invent turns');
  ok(!none.steps, 'and is handed no steps at all to be tempted by');
}

{
  // 2. AMBIGUITY SURVIVES. Two Gettys eight miles apart: she asks, and only
  //    then routes. This is the rule the tool is most able to break, because
  //    it is now the thing holding the steering wheel.
  const GETTY = [
    { display_name: 'The Getty Center', formatted_address: '1200 Getty Center Dr' },
    { display_name: 'Getty Villa', formatted_address: '17985 Pacific Coast Hwy' },
  ];
  const ROUTED = {
    status: 'routed',
    destination: { display_name: 'Getty Villa',
                   formatted_address: '17985 Pacific Coast Hwy' },
    route: { total_distance_m: 12800, duration_s: 900, eta_epoch: 1787790000 },
  };
  const panel = fakePanel((text, n) => (n === 1
    ? { status: 'ambiguous', query: 'the Getty', candidates: GETTY }
    : ROUTED));
  const h = harness({ tool: panelTools() });

  h.controller.handle({
    type: 'response.function_call_arguments.done',
    name: 'start_navigation', call_id: 'a1',
    arguments: JSON.stringify({ destination: "let's go to the Getty" }),
  });
  await tick(); await tick(); await tick();

  const first = JSON.parse(h.sent.filter(e => e.type === 'conversation.item.create')
                                 .pop().item.output);
  ok(first.ok === true && first.routing === false && first.status === 'ambiguous',
     'an ambiguous destination is a real answer, not a failure — but nothing '
     + 'is routed');
  ok(first.candidates.length === 2 &&
     first.candidates[0].name === 'The Getty Center' &&
     first.candidates[1].name === 'Getty Villa',
     'she is handed both places, by name, to put to the driver');
  ok(/do NOT pick one/i.test(first.rules || ''),
     'and told in the result itself not to choose');
  ok(/call start_navigation again/i.test(first.rules || ''),
     'and what to do with the answer when it comes');
  ok(rt.navStatus().routing === false,
     'meanwhile no route exists — the question was asked before anything was '
     + 'set, which is the whole point');
  ok(h.types().indexOf('response.create') >= 0,
     'she is asked to speak, and what she has to say is a question');

  // The driver answers. NOW she routes.
  h.controller.handle({
    type: 'response.function_call_arguments.done',
    name: 'start_navigation', call_id: 'a2',
    arguments: JSON.stringify({ destination: 'Getty Villa' }),
  });
  await tick(); await tick(); await tick();

  const second = JSON.parse(h.sent.filter(e => e.type === 'conversation.item.create')
                                  .pop().item.output);
  ok(second.ok && second.routing === true && second.destination === 'Getty Villa',
     'the chosen one routes on the second call');
  ok(panel.asked.length === 2 && panel.asked[1] === 'Getty Villa',
     'through the same entry point, with the driver’s choice as the query');
  ok(rt.navStatus().destination === 'Getty Villa',
     'and the route is now active to the place the driver actually named');
}

{
  // 3. THE UNHAPPY ONES. Neither is an excuse to hand the job back.
  const nf = fakePanel(() => ({ status: 'not_found', query: 'the blue one' }));
  const h1 = harness({ tool: panelTools() });
  h1.controller.handle({
    type: 'response.function_call_arguments.done',
    name: 'start_navigation', call_id: 'x1',
    arguments: JSON.stringify({ destination: 'the blue one' }),
  });
  await tick(); await tick(); await tick();
  const miss = JSON.parse(h1.sent.find(e => e.type === 'conversation.item.create')
                                 .item.output);
  ok(miss.ok === true && miss.status === 'not_found' && miss.routing === false,
     '"I could not find it" is an answer she gives, not an error she hides');
  ok(/not tell them to type it in/i.test(miss.rules || ''),
     'and still not a reason to send the driver to the keyboard');
  ok(nf.asked.length === 1, 'one attempt, no retry with a guess');

  const fail = fakePanel(() => ({ status: 'failed',
                                  error: 'no position — allow location' }));
  const h2 = harness({ tool: panelTools() });
  h2.controller.handle({
    type: 'response.function_call_arguments.done',
    name: 'start_navigation', call_id: 'x2',
    arguments: JSON.stringify({ destination: 'LAX' }),
  });
  await tick(); await tick(); await tick();
  const broke = JSON.parse(h2.sent.find(e => e.type === 'conversation.item.create')
                                  .item.output);
  ok(broke.ok === false && broke.status === 'failed',
     'a route that would not build comes back as a failure');
  ok(/allow location/.test(broke.note || ''),
     'carrying the reason, because this one the driver can actually fix');
  ok(/not.*screen/i.test(broke.rules || ''),
     'and even here she does not defer to the dashboard');
  ok(fail.asked.length === 1, 'and does not retry into a second routing call');

  const blank = await rt.localTools.start_navigation({ destination: '  ' });
  ok(blank.ok === false && /no destination/.test(blank.note),
     'an empty destination is answered rather than routed on — a model can '
     + 'emit anything');

  // No panel at all — the tool is called on a page that has no navigation.
  delete global.RIO.nav;
  const h3 = harness({ tool: panelTools() });
  h3.controller.handle({
    type: 'response.function_call_arguments.done',
    name: 'start_navigation', call_id: 'x3',
    arguments: JSON.stringify({ destination: 'LAX' }),
  });
  await tick(); await tick(); await tick();
  const none = JSON.parse(h3.sent.find(e => e.type === 'conversation.item.create')
                                 .item.output);
  ok(none.ok === false, 'with no navigation on the page it answers, rather '
     + 'than throwing into the session');
  delete global.RIO.bus;
}

{
  // 4. ONE ROUTER, NOT TWO. The reuse is the point of the fix: if this file
  //    ever grows its own /nav/route call, a spoken destination and a typed
  //    one stop being the same event and start being two things that drift.
  const fs = require('fs');
  const rtSrc = fs.readFileSync(
    path.join(__dirname, '..', 'static', 'rio_realtime.js'), 'utf8');
  ok(/nav\.routeToQuery\(/.test(rtSrc),
     'the tool calls the panel’s own routing entry point');
  ok(!/fetch\([^)]*nav\/(route|destination)/.test(rtSrc),
     'and asks the routing endpoints for nothing itself');

  const navSrc = fs.readFileSync(
    path.join(__dirname, '..', 'static', 'rio_nav.js'), 'utf8');
  ok((navSrc.match(/fetch\(RIO\.url\('\/nav\/route'\)/g) || []).length === 1,
     'there is still exactly one place in the panel that loads a route');
  ok(/routeToQuery: routeToQuery/.test(navSrc),
     'and it is reachable by name, which is how the voice path reaches it');
}

// ---------------------------------------------------------------------------
section('tools that wait for a reason to exist');
// ---------------------------------------------------------------------------
/* A tool schema is input on EVERY response for the whole drive, and two of
   them cannot be called until a route exists. The session therefore ships the
   seven that are always true and attaches the other two when they become
   callable — which is worth about 350 tokens a response out of a minute that
   holds 40,000, and matters most on exactly the turns that spend two of them.

   What is checked here is the SENDING, because the cost of getting it wrong is
   not a wasted token: a session.update per route event would replace the tool
   list in the middle of a turn that is using it. */
{
  const BASE = [{ type: 'function', name: 'look' },
                { type: 'function', name: 'start_navigation' }];
  const EXTRA = { routing: [{ type: 'function', name: 'stop_navigation' },
                            { type: 'function', name: 'reroute' }] };
  const h = harness({ toolSchemas: BASE, conditionalTools: EXTRA });

  const updates = () => h.sent.filter((e) => e.type === 'session.update');
  ok(updates().length === 0,
     'a session that has not been told anything sends no update — the seven '
     + 'it opened with are the seven it has');
  ok(h.controller.state().tools.join(',') === 'look,start_navigation',
     'and it says so: ' + h.controller.state().tools.join(','));

  ok(h.controller.setToolCondition('routing', true) === true,
     'a route attaching attaches the tools that need one');
  ok(updates().length === 1, 'in exactly one session.update');
  const namesSent = updates()[0].session.tools.map((t) => t.name);
  ok(namesSent.join(',') === 'look,start_navigation,stop_navigation,reroute',
     'carrying the whole list, base first — the API replaces `tools`, it does '
     + 'not add to it (' + namesSent.join(',') + ')');
  ok(Object.keys(updates()[0].session).join(',') === 'type,tools',
     'and nothing else: an update that re-sent turn detection would reset the '
     + 'detector mid-drive (' + Object.keys(updates()[0].session).join(',') + ')');
  ok(h.events.some((e) => e.type === 'LIVE_TOOLS_CHANGED' && e.on === true),
     'the drive log says when the toolset changed and why');

  ok(h.controller.setToolCondition('routing', true) === false,
     'a second route attaching changes nothing, so nothing is sent — every '
     + 'automatic reroute attaches a route, and each one would otherwise cost '
     + 'an update');
  ok(updates().length === 1, 'still one update, not two');

  ok(h.controller.setToolCondition('routing', false) === true &&
     updates().length === 2,
     'the route ending takes them away again');
  ok(updates()[1].session.tools.map((t) => t.name).join(',')
     === 'look,start_navigation',
     'leaving the session carrying only what is always true');

  ok(h.controller.setToolCondition('nonsense', true) === false &&
     updates().length === 2,
     'a condition the server never declared is not a tool list the browser '
     + 'may invent');

  // A session minted before this existed, or by a server that declares none:
  // nothing to attach, and nothing sent trying.
  const bare = harness({});
  ok(bare.controller.setToolCondition('routing', true) === false,
     'a session with no conditional tools declared sends no update at all');

  /* WATCHING THE ROUTE EVENTS IS THE CONTROLLER'S JOB, not the caller's.
   *
   * This is a method because leaving it to each embedder cost a live drive:
   * the recording harness builds a controller directly, nobody had wired the
   * route events into it, and asked to avoid the freeway with no `reroute`
   * attached the model reached for the research tool and narrated a reroute
   * it had not performed. */
  const w = harness({ toolSchemas: BASE, conditionalTools: EXTRA });
  const bus = (function () {
    const subs = [];
    return { on: (t, fn) => subs.push(fn),
             emit: (type, p) => subs.forEach(
               (fn) => fn(Object.assign({ type: type }, p || {}))) };
  })();
  ok(w.controller.watchToolConditions(bus) === true,
     'the controller subscribes to the drive\'s own route events');
  bus.emit('NAV_ROUTE_ATTACHED', { route_id: 'r1' });
  const wUpdates = () => w.sent.filter((e) => e.type === 'session.update');
  ok(wUpdates().length === 1 &&
     wUpdates()[0].session.tools.map((t) => t.name).indexOf('reroute') >= 0,
     'a route attaching brings the route tools with it');
  bus.emit('NAV_ROUTE_ATTACHED', { route_id: 'r2' });
  ok(wUpdates().length === 1,
     'and the automatic reroute that follows changes nothing');
  bus.emit('NAV_PROGRESS', {});
  ok(wUpdates().length === 1, 'nor does any other navigation event');
  bus.emit('NAV_STOPPED', { route_id: 'r2' });
  ok(wUpdates().length === 2 &&
     wUpdates()[1].session.tools.map((t) => t.name).indexOf('reroute') < 0,
     'stopping takes them away again');
  ok(w.controller.watchToolConditions(null) === false,
     'and a page with no bus at all is not an error — it is a controller with '
     + 'nothing to watch');
}

// ---------------------------------------------------------------------------
section('navigation by voice — stopping, rerouting, and who the tracker '
        + 'thinks started the route');
// ---------------------------------------------------------------------------
/* The real static/rio_nav.js, against a route server made of a function.
 *
 * A fake route rather than a fake nav: everything under test here — the
 * teardown, the generation, the queue, the tracker — is the panel's own, and
 * the only thing replaced is the HTTP call that hands it a route. That is what
 * lets a whole simulated drive run in a few milliseconds and still measure the
 * timing a car would have seen: the simulator advances its clock one second
 * per tick whatever the wall clock does.
 */
{
  const ROUTE_LAT = 34.0430, ROUTE_LNG = -118.2673;

  function fakeRoute(gen, avoid) {
    const mLat = 111320, mLng = 111320 * Math.cos(ROUTE_LAT * Math.PI / 180);
    const pts = [];
    const push = (x, y) => pts.push([ROUTE_LAT + y / mLat, ROUTE_LNG + x / mLng]);
    for (let d = 0; d <= 1200; d += 10) push(d, 0);
    for (let d = 10; d <= 900; d += 10) push(1200, d);
    const iTurn = 120, iEnd = pts.length - 1;
    return {
      route_id: 'r' + gen, journey_id: 'j1', generation_id: gen,
      provider: 'fake', created_at: 0, eta_epoch: 0,
      origin: { lat: pts[0][0], lng: pts[0][1] },
      destination: { display_name: 'Test Destination',
                     formatted_address: 'Test Destination, Los Angeles, CA',
                     lat: pts[iEnd][0], lng: pts[iEnd][1] },
      total_distance_m: 2100, duration_s: 190, route_length_m: 2100,
      arrival: { side: 'RIGHT' }, landmarks_state: 'not_requested',
      geometry: pts,
      timing: {
        gps_stale_timeout_s: 5, gps_accuracy_limit_m: 30, gps_degraded_bias_s: 2,
        off_route_distance_m: 45, off_route_persistence: 3, reroute_debounce_s: 12,
        progress_rewind_tolerance_m: 30, maneuver_passed_eps_m: 8,
        arrive_radius_m: 25, projection_back_m: 80, projection_fwd_m: 400,
        heading_min_displacement_m: 8, heading_max_sample_age_s: 3,
        heading_min_speed_ms: 1.5, stationary_speed_ms: 0.7,
        early_guidance_s: 25, anchor_acquisition_s: 11, context_call_s: 6,
        near_turn_s: 2.5, min_call_distance_m: 20, max_call_distance_m: 400,
        early_max_distance_m: 900, speed_floor_ms: 3, speed_nominal_ms: 11,
        duplicate_instruction_cooldown_s: 8, anchor_valid_for_s: 6,
        speech_ttl_s: { early: 8, primary: 5, imminent: 2.5, arrival: 8 },
        vision_enabled: false,
      },
      avoid_applied: avoid || [],
      avoid_unsupported: [],
      maneuvers: [
        { id: 'm0', sequence: 0, type: 'TURN', direction: 'LEFT',
          road_name: 'Lincoln Boulevard',
          instruction: 'Turn left onto Lincoln Boulevard',
          lat: pts[iTurn][0], lng: pts[iTurn][1],
          route_distance_position: 1200, polyline_index: iTurn, anchors: [],
          speech: { early: 'Left turn coming up onto Lincoln Boulevard.',
                    primary: 'Take the next left onto Lincoln Boulevard.',
                    imminent: 'Left here.' } },
        { id: 'm1', sequence: 1, type: 'ARRIVE', direction: 'RIGHT',
          road_name: '', instruction: 'Arrive at Test Destination',
          lat: pts[iEnd][0], lng: pts[iEnd][1],
          route_distance_position: 2100, polyline_index: iEnd, anchors: [],
          speech: { early: 'Almost there.',
                    primary: 'Your destination is on the right.',
                    arrival: 'Your destination is on the right.' } },
      ],
    };
  }

  // The route server: one generation per call, and it remembers what it was
  // asked for so the preference and the heading can be asserted rather than
  // assumed.
  const served = { calls: [], gen: 0 };
  global.fetch = function (url, opts) {
    const body = JSON.parse((opts && opts.body) || '{}');
    if (String(url).indexOf('/nav/route') >= 0) {
      served.calls.push(body);
      served.gen += 1;
      const r = fakeRoute(served.gen, body.avoid || []);
      r.avoid_unsupported = [];
      return Promise.resolve({ ok: true, status: 200,
                               json: () => Promise.resolve(r) });
    }
    if (String(url).indexOf('/nav/destination') >= 0) {
      // Resolution is not what this section tests; it is the step before it.
      // One place, unambiguously, so the interesting half of routeToQuery is
      // the half that runs.
      return Promise.resolve({ ok: true, status: 200, json: () =>
        Promise.resolve({ status: 'resolved', destination: {
          display_name: 'Test Destination',
          formatted_address: 'Test Destination, Los Angeles, CA',
          provider_place_id: 'p_test' } }) });
    }
    // Everything else the panel might reach for (the drive log) is accepted
    // and ignored: none of it is what this section is about.
    return Promise.resolve({ ok: true, status: 200,
                             json: () => Promise.resolve({}) });
  };

  global.window = global;
  const stubNode = () => ({
    style: {}, textContent: '', innerHTML: '', appendChild: () => {},
    querySelector: () => stubNode(), addEventListener: () => {},
    setAttribute: () => {},
  });
  global.document = {
    addEventListener: (t, f) => { (pageInit[t] = pageInit[t] || []).push(f); },
    getElementById: () => null, createElement: stubNode,
    createTextNode: () => ({}), head: { appendChild: () => {} },
  };
  global.Audio = function () {
    return { preload: '', muted: false, currentTime: 0,
             play: () => Promise.resolve(), pause: () => {} };
  };
  global.navigator = { geolocation: { getCurrentPosition: (okc) => okc({
    coords: { latitude: ROUTE_LAT, longitude: ROUTE_LNG, accuracy: 8 } }) } };

  global.RIO = global.RIO || {};
  global.RIO.sessionId = 'sel';
  global.RIO.url = (p2) => 'http://127.0.0.1:0' + p2;
  const watch = { sink: null };
  global.RIO.headway = { startWatch: () => {},
                         onPosition: (fn) => { watch.sink = fn; } };

  // Every line RIO would have said about the route, in the order she would
  // have said it. `hold` makes one line refuse to finish, which is the only
  // way to have something QUEUED rather than already spoken.
  const heard = [];
  const holding = { on: false, resolve: null };
  global.RIO.speak = {
    provider: function (o) {
      return {
        play: function () {
          heard.push(o.text);
          if (holding.on) {
            return new Promise((res) => { holding.resolve = res; });
          }
          return Promise.resolve();
        },
        stop: function () {},
      };
    },
  };

  require(path.join(__dirname, '..', 'static', 'rio_speech.js'));
  require(path.join(__dirname, '..', 'static', 'rio_navcore.js'));
  require(path.join(__dirname, '..', 'static', 'rio_navplan.js'));
  require(path.join(__dirname, '..', 'static', 'rio_nav.js'));
  (pageInit.DOMContentLoaded || []).forEach((f) => f());
  const nav = global.RIO.nav;

  const calls = [];
  global.RIO.bus.on('*', (ev) => {
    if (/NAV_(EARLY_GUIDANCE|CONTEXTUAL_CALL|NEAR_TURN|PRIMARY|ARRIVAL|SPEECH_SPOKEN)/
        .test(ev.type)) calls.push(ev);
  });

  // Where the car is, said out loud rather than inherited. attach() seeds a
  // new tracker from the last fix it has, which after a simulated drive is
  // the far end of the last route; a second route started from there is
  // arrived at before it starts.
  function parkAtStart() {
    if (!watch.sink) return;
    watch.sink({ coords: { latitude: ROUTE_LAT, longitude: ROUTE_LNG,
                           speed: 0, heading: null, accuracy: 6 } });
  }

  ok(!!watch.sink,
     'the panel subscribed to the position watch, the way it does in a car');
  ok(!!nav && typeof nav.stopRoute === 'function' &&
     typeof nav.reroute === 'function',
     'the panel exposes stopping and rerouting by name, which is how the '
     + 'voice path reaches the same code the buttons do');

  /* One simulated drive, run to the end. The clock inside it is the
     simulator's, so this measures the timing of a real drive at the speed of
     a test. */
  function driveToEnd(tag) {
    const before = calls.length;
    heard.length = 0;
    nav.simulate({ tickMs: 1, mph: 30 });
    return new Promise((resolve) => {
      const started = Date.now();
      const poll = setInterval(() => {
        const st = nav.state();
        if (!st || st.arrived || Date.now() - started > 8000) {
          clearInterval(poll);
          nav.stopSimulation();
          resolve({ calls: calls.slice(before), spoken: heard.slice() });
        }
      }, 5);
    });
  }

  const sequence = (r) => r.calls
    .filter((e) => e.call_type)
    .map((e) => e.maneuver_id + ':' + e.call_type);

  // --- STARTED BY VOICE, then STARTED BY THE BOX, and compared -------------
  await (async () => {
    parkAtStart();
    const byVoice = await rt.localTools.start_navigation(
      { destination: 'Test Destination' });
    ok(byVoice.ok === true && byVoice.routing === true,
       'a spoken destination routes (' + byVoice.destination + ')');
    const voiceDrive = await driveToEnd('voice');
    const voiceSeq = sequence(voiceDrive);
    ok(voiceSeq.length > 0,
       'and driving it calls the turns: ' + voiceSeq.join(' → '));
    ok(voiceDrive.spoken.some((t) => /Lincoln Boulevard/.test(t)),
       'in her own words, with the road in them — '
       + JSON.stringify(voiceDrive.spoken));

    nav.stopRoute('test');
    parkAtStart();
    const byBox = await nav.routeToQuery('Test Destination');
    ok(byBox.status === 'routed', 'the destination box routes the same place');
    const boxDrive = await driveToEnd('box');
    const boxSeq = sequence(boxDrive);

    // THE CLAIM, checked rather than assumed: the tracker does not know or
    // care who asked for the route. Both drives are the same geometry at the
    // same speed, so any difference here is the entry point leaking into the
    // guidance — which is exactly the bug this section exists to catch.
    ok(voiceSeq.join(' → ') === boxSeq.join(' → '),
       'a route started by voice calls its turns in the same sequence as one '
       + 'started on the dashboard\n        voice: ' + voiceSeq.join(' → ')
       + '\n        box:   ' + boxSeq.join(' → '));
    ok(voiceSeq.some((s) => /:early$/.test(s))
       && voiceSeq.some((s) => /:primary$/.test(s)),
       'and the sequence is the real one — an early call and an instruction, '
       + 'not one lonely event');
  })();

  // --- STOP: the route ends, and nothing queued survives it ----------------
  await (async () => {
    parkAtStart();
    const started = await rt.localTools.start_navigation(
      { destination: 'Test Destination' });
    ok(started.ok === true, 'a route to stop');

    /* Something is being said, and something else is waiting behind it. The
       first line is held open on purpose: an arbiter with nothing queued
       cannot demonstrate that a stop empties the queue, because there is
       nothing in it to lose. */
    holding.on = true;
    const dropped = [];
    global.RIO.speech.say({
      priority: 30, group: 'nav:m0', id: 'nav:m0:early', text: 'held line',
      play: () => new Promise(() => {}), stop: () => {},
      onDone: (why) => dropped.push(['held', why]),
    });
    global.RIO.speech.say({
      priority: 30, group: 'nav:m1', id: 'nav:m1:primary', text: 'queued line',
      play: () => Promise.resolve(), stop: () => {},
      onDone: (why) => dropped.push(['queued', why]),
    });
    ok(global.RIO.speech.state().queued.length >= 1,
       'one nav line is playing and another is queued behind it');

    const out = rt.localTools.stop_navigation();
    ok(out.ok === true && out.was_navigating === true,
       'stop_navigation reports that it stopped something real');
    ok(/navigation off/i.test(out.rules),
       'and tells her to confirm it in one line');
    ok(nav.route === null && rt.navStatus().routing === false,
       'the route is gone and nav_status says so — nothing to answer from');
    ok(dropped.some((d) => d[0] === 'queued' && d[1] === 'cleared'),
       'the queued line was discarded rather than left to play over a driver '
       + 'who has just said they know the way (' + JSON.stringify(dropped) + ')');
    ok(dropped.some((d) => d[0] === 'held'),
       'and the one already in flight was stopped too');
    holding.on = false;

    const again = rt.localTools.stop_navigation();
    ok(again.ok === true && again.was_navigating === false,
       'stopping when nothing is running is not an error — it is a sentence');
  })();

  // --- REROUTE: a new generation, and the old one goes quiet ---------------
  await (async () => {
    parkAtStart();
    const started = await rt.localTools.start_navigation(
      { destination: 'Test Destination' });
    const genBefore = nav.route.generation_id;
    const routeIdBefore = nav.route.route_id;
    ok(started.ok === true && genBefore >= 1, 'a route to reroute');

    /* A line about the plan that is about to be replaced, still waiting its
       turn behind one that is playing. Held open on purpose: a line the
       arbiter has already spoken proves nothing about what a reroute does to
       the queue. */
    holding.on = true;
    const fate = [];
    global.RIO.speech.say({
      priority: 30, group: 'nav:m0', id: 'nav:m0:early', text: 'held line',
      play: () => new Promise(() => {}), stop: () => {},
      onDone: (why) => fate.push(['playing', why]),
    });
    // A DIFFERENT MANEUVER, deliberately: same group means the newer line
    // replaces the older one, which is ordinary turn-taking and would leave
    // nothing in the queue to lose.
    global.RIO.speech.say({
      priority: 30, group: 'nav:m1', id: 'nav:m1:primary', text: 'old plan',
      play: () => Promise.resolve(), stop: () => {},
      valid: () => !!nav.route && nav.route.generation_id === genBefore,
      onDone: (why) => fate.push(['queued', why]),
    });
    ok(global.RIO.speech.state().queued.length >= 1,
       'a line about the old plan is queued when the reroute is asked for');

    const out = await rt.localTools.reroute(
      { avoid: ['highways'], other_preference: 'the scenic way' });
    ok(out.ok === true && out.status === 'rerouted',
       'reroute comes back with a live route');
    ok(nav.route.route_id !== routeIdBefore &&
       nav.route.generation_id === genBefore + 1,
       'a different route, one generation on — replaced, never patched ('
       + routeIdBefore + ' gen ' + genBefore + ' → ' + nav.route.route_id
       + ' gen ' + nav.route.generation_id + ')');
    ok(rt.navStatus().routing === true,
       'and the car is being navigated the whole time — a reroute is a '
       + 'replacement, not a gap');

    const asked = served.calls[served.calls.length - 1];
    ok(asked.reroute_of === routeIdBefore && asked.reason === 'driver_request',
       'the server was told which route this replaces and that a person '
       + 'asked for it, not the tracker');
    ok(JSON.stringify(asked.avoid) === JSON.stringify(['highways']),
       'with the preference the driver actually named');
    ok(out.avoid_unsupported.indexOf('the scenic way') >= 0,
       'and what the map cannot do comes back to be said out loud rather '
       + 'than dropped: ' + JSON.stringify(out.avoid_unsupported));
    ok(/never say you avoided something that is not in avoid_applied/
       .test(out.rules), 'the result forbids claiming a preference it did '
       + 'not get');

    // NOTHING ABOUT THE OLD PLAN SURVIVES THE REPLACEMENT. Two mechanisms
    // agree here and both are meant to: the queue is emptied of 'nav:' when
    // the new route attaches, and any line that got past the queue answers
    // its own validity question at dequeue, where the generation it was
    // written for no longer exists. (tools/nav_selftest.js drives that second
    // one on its own, against the planner.)
    await new Promise((r) => setTimeout(r, 5));
    ok(fate.some((f) => f[0] === 'queued' && f[1] !== 'spoken'),
       'the queued line about the old plan never reached the speaker ('
       + JSON.stringify(fate) + ')');
    ok(fate.some((f) => f[0] === 'playing'),
       'and the one mid-sentence about it was stopped, not talked over');
    holding.on = false;

    const off = rt.localTools.stop_navigation();
    ok(off.was_navigating === true, 'and the drive can still be stopped after '
       + 'a reroute — the teardown follows the new route, not the old one');
  })();

  delete global.fetch;
}

// ---------------------------------------------------------------------------
// LIVE — the real panel, the real server, over HTTP.
//
//   node tools/realtime_selftest.js --server [http://127.0.0.1:8888]
//
// Everything above runs against fakes, which is what makes it fast and what
// makes it prove decisions rather than plumbing. This part proves the
// plumbing, and it is the half that was actually broken: a driver asking to be
// taken somewhere ends with a route loaded and a tracker running, and nobody
// touched the dashboard.
//
// Nothing is stubbed here except the browser itself. static/rio_nav.js is
// loaded as written — the same routing, the same tracker, the same bus — with
// a DOM that answers "no such element" to everything, a fetch that speaks
// HTTP, and a Geolocation that reports one fixed position. The destination is
// resolved by the running server, against whichever provider it is configured
// with.
// ---------------------------------------------------------------------------
// node 12 has no fetch, and this needs about a tenth of one.
function installFetch() {
  if (global.fetch) return;
  const http = require('http');
  const { URL } = require('url');
  global.fetch = function (url, opts) {
    opts = opts || {};
    return new Promise((resolve, reject) => {
      const u = new URL(url);
      const req = http.request({
        hostname: u.hostname, port: u.port, path: u.pathname + u.search,
        method: opts.method || 'GET', headers: opts.headers || {},
      }, (res) => {
        let body = '';
        res.on('data', (d) => { body += d; });
        res.on('end', () => resolve({
          ok: res.statusCode < 400, status: res.statusCode,
          text: () => Promise.resolve(body),
          json: () => Promise.resolve(JSON.parse(body)),
        }));
      });
      req.on('error', reject);
      if (opts.body) req.write(opts.body);
      req.end();
    });
  };
}

function installBrowser(base, sessionId, origin) {
  installFetch();
  global.window = global;                       // so the panel's `root` is here
  const noElement = () => null;
  const stubNode = () => ({
    style: {}, textContent: '', innerHTML: '', appendChild: () => {},
    querySelector: () => stubNode(), addEventListener: () => {},
    setAttribute: () => {},
  });
  global.document = {
    addEventListener: (t, f) => { (pageInit[t] = pageInit[t] || []).push(f); },
    getElementById: noElement,
    createElement: stubNode,
    createTextNode: () => ({}),
    head: { appendChild: () => {} },
  };
  global.Audio = function () {
    return { preload: '', muted: false, currentTime: 0,
             play: () => Promise.resolve(), pause: () => {} };
  };
  // The one GPS watch the page owns, reporting a fixed position. `sink` is
  // where the panel's subscription lands, so this harness can drive the car
  // along the route afterwards exactly as a real fix would.
  const gps = { sink: null };
  global.navigator = {
    geolocation: {
      getCurrentPosition: (okc) => okc({ coords: { latitude: origin.lat,
                                                   longitude: origin.lng,
                                                   accuracy: 8 } }),
    },
  };
  /* EXTEND the page's namespace, never REPLACE it.
   *
   * Every panel file registers itself on `window.RIO` from an IIFE that runs
   * once, at load. In node that load is a `require`, and `require` is CACHED —
   * so a module this file already pulled in at the top (rio_speech,
   * rio_navcore, rio_realtime) will NOT run its registration again no matter
   * how many times it is required here.
   *
   * Assigning a fresh object to global.RIO therefore does not reset the page,
   * it DELETES those modules for the rest of the process. `attach()` then
   * calls RIO.navcore.create on undefined, setRoute catches it like any other
   * routing failure, and eight checks report that a route would not build —
   * naming the provider, the server and the tool, none of which were involved.
   * The server was answering with sixteen maneuvers throughout.
   *
   * So the harness adds its four stubs to whatever is already there. */
  const RIO = global.RIO || (global.RIO = {});
  RIO.sessionId = sessionId;
  RIO.url = (p) => base + p + (p.indexOf('?') >= 0 ? '&' : '?') +
                   'session_id=' + encodeURIComponent(sessionId);
  RIO.headway = { startWatch: () => {}, onPosition: (fn) => { gps.sink = fn; } };
  RIO.speak = { provider: () => ({ play: () => Promise.resolve(), stop: () => {} }) };

  require(path.join(__dirname, '..', 'static', 'rio_speech.js'));
  require(path.join(__dirname, '..', 'static', 'rio_navcore.js'));
  require(path.join(__dirname, '..', 'static', 'rio_navplan.js'));
  require(path.join(__dirname, '..', 'static', 'rio_nav.js'));
  (pageInit.DOMContentLoaded || []).forEach((f) => f());

  /* ...and say so HERE if one of them is missing anyway.
   *
   * This is the check the afternoon above was spent not having. A panel with
   * no tracker module fails at the first route it is asked to build, several
   * layers down, wearing the costume of a provider error — and the one thing
   * that would have named it is the question this asks. */
  const missing = ['speech', 'navcore', 'navplan', 'nav']
    .filter((k) => !RIO[k]);
  ok(missing.length === 0,
     'the panel has every module it needs before anything asks it to route'
     + (missing.length ? ' (missing: ' + missing.join(', ') + ')' : ''));
  return gps;
}

async function liveRouting(base) {
  section('LIVE — the real panel against ' + base);
  installFetch();

  const post = (p, body) => global.fetch(base + p, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body || {}),
  }).then((r) => r.json());

  // A real drive session, so what RIO does by voice lands in a real log.
  const started = await post('/session/start', {});
  const sessionId = started.session_id;
  ok(!!sessionId, 'the server opened a drive session (' + sessionId + ')');

  const ORIGIN = { lat: 34.0219, lng: -118.4814 };     // Santa Monica
  const gps = installBrowser(base, sessionId, ORIGIN);
  ok(!!(global.RIO.nav && global.RIO.nav.routeToQuery),
     'the real navigation panel loaded outside a browser');
  ok(rt.navStatus().routing === false,
     'and starts with no route — nothing is set until she sets it');

  const attached = [];
  global.RIO.bus.on('NAV_ROUTE_ATTACHED', (ev) => attached.push(ev));

  // THE WHOLE POINT: one tool call, spoken destination in, live route out.
  const result = await rt.localTools.start_navigation(
    { destination: 'take me to Griffith Observatory' });
  ok(result.ok === true && result.routing === true,
     'a spoken destination came back routed: ' + JSON.stringify({
       destination: result.destination, minutes: result.minutes,
       km: result.distance_km }));
  ok(/griffith/i.test(result.destination || ''),
     'named by the provider that resolved it (' + result.destination + ')');
  ok(result.minutes > 0 && result.distance_km > 0,
     'with a real ETA and distance to confirm out loud');
  ok(attached.length === 1 && attached[0].n_maneuvers > 1,
     'the route attached to the panel with ' +
     (attached[0] ? attached[0].n_maneuvers : 0) + ' maneuvers — the same '
     + 'event the destination box produces');

  // ...and the tracker is RUNNING, which is the difference between a route
  // that exists and a car that is being navigated. Fed through the panel's own
  // GPS subscription, from the route's own geometry.
  const st0 = rt.navStatus();
  ok(st0.routing === true && /griffith/i.test(st0.destination || ''),
     'nav_status now answers from the live route');
  ok(!!gps.sink, 'the panel is subscribed to the position watch');

  const route = global.RIO.nav.route;
  ok(!!(route && route.geometry && route.geometry.length > 2),
     'the route carries geometry to drive along');
  const before = rt.navStatus().distance_remaining_m;
  for (const idx of [0, 12, 30]) {
    const pt = route.geometry[Math.min(idx, route.geometry.length - 1)];
    gps.sink({ coords: { latitude: pt[0], longitude: pt[1], speed: 13,
                         heading: null, accuracy: 6 } });
    await new Promise((r) => setTimeout(r, 1100));
  }
  const st1 = rt.navStatus();
  ok(st1.route_state === 'ON_ROUTE',
     'driving the first few hundred metres of it tracks ON_ROUTE');
  ok(st1.distance_remaining_m < before,
     'and the distance remaining actually came down (' + before + ' m -> ' +
     st1.distance_remaining_m + ' m) — the tracker is running, not just loaded');
  ok(!!st1.next_maneuver && st1.next_maneuver.distance_m >= 0,
     'with a next maneuver selected: ' +
     (st1.next_maneuver ? st1.next_maneuver.instruction : 'none'));

  // Ambiguity, against the real provider. Whatever it returns, the rule is the
  // same: nothing new gets routed on a question.
  const amb = await rt.localTools.start_navigation({ destination: 'take me to LAX' });
  if (amb.status === 'ambiguous') {
    ok(amb.routing === false && amb.candidates.length > 1,
       'an ambiguous destination asks which one — ' +
       amb.candidates.map((c) => c.name).join(' / '));
    ok(rt.navStatus().destination === st1.destination,
       'and changes nothing while it waits for the answer');
    const chosen = amb.candidates[0].name;
    const after = await rt.localTools.start_navigation({ destination: chosen });
    ok(after.routing === true,
       'the driver’s choice then routes on its own tool call (' +
       after.destination + ')');
  } else {
    ok(amb.status === 'routed' || amb.status === 'not_found',
       'the provider read "LAX" as ' + amb.status + ' — no ambiguity to '
       + 'exercise against this provider today');
  }

  // --- THE SESSION'S OWN ACCOUNT OF WHAT SHE CAN REACH FOR ----------------
  const minted = await post('/realtime/session?session_id='
                            + encodeURIComponent(sessionId), {});
  ok((minted.tools || []).indexOf('stop_navigation') < 0,
     'the session the browser is handed carries no route tools: ' +
     (minted.tools || []).join(', '));
  ok(((minted.conditional_tools || {}).routing || [])
     .map((t) => t.name).join(',') === 'stop_navigation,reroute',
     'they arrive as a condition the browser turns on when a route exists');
  ok((minted.tool_schemas || []).length === (minted.tools || []).length,
     'with the base schemas alongside, because a session.update replaces the '
     + 'whole tool list and the browser must not write one of its own');

  // --- REROUTING, AGAINST THE REAL MAP ------------------------------------
  const beforeReroute = global.RIO.nav.route;
  const rr = await rt.localTools.reroute({ avoid: ['highways'],
                                           other_preference: 'the scenic way' });
  ok(rr.ok === true && rr.status === 'rerouted',
     'asked for another way, the provider computed one: ' +
     JSON.stringify({ minutes: rr.minutes, km: rr.distance_km }));
  ok(global.RIO.nav.route.route_id !== beforeReroute.route_id &&
     global.RIO.nav.route.generation_id === beforeReroute.generation_id + 1,
     'it replaced the route and moved the journey on one generation ('
     + beforeReroute.generation_id + ' -> '
     + global.RIO.nav.route.generation_id + ')');
  ok(global.RIO.nav.route.destination.display_name ===
     beforeReroute.destination.display_name,
     'to the same place, resolved once and never looked up again ('
     + global.RIO.nav.route.destination.display_name + ')');
  ok((rr.avoid_applied || []).indexOf('highways') >= 0,
     'with the preference the map actually honoured: '
     + JSON.stringify(rr.avoid_applied));
  ok((rr.avoid_unsupported || []).indexOf('the scenic way') >= 0,
     'and the one it cannot, named so she can say so rather than pretend');
  ok(rt.navStatus().routing === true,
     'and the car is still being navigated — a reroute is a replacement, not '
     + 'a gap');

  // --- STOPPING -----------------------------------------------------------
  const stopped = rt.localTools.stop_navigation();
  ok(stopped.ok === true && stopped.was_navigating === true,
     'and "stop navigation" stops it: ' + stopped.destination);
  ok(global.RIO.nav.route === null && rt.navStatus().routing === false,
     'the route is gone from the panel, and nav_status answers from nothing');
  ok(global.RIO.speech.state().queued.length === 0 &&
     !global.RIO.speech.state().speaking,
     'with nothing left queued to be said about it');

  // The drive log should say a human voice set this, not the box.
  await new Promise((r) => setTimeout(r, 400));
  const log = await global.fetch(
    base + '/session/' + encodeURIComponent(sessionId))
    .then((r) => (r.ok ? r.json() : null)).catch(() => null);
  const events = (log && log.events) || [];
  const logged = events.map((e) => (e.payload || {}).event);
  ok(logged.indexOf('NAV_VOICE_DESTINATION') >= 0,
     'the drive log records that this destination came from her voice');
  ok(logged.indexOf('NAV_ROUTE_STARTED') >= 0,
     'alongside the ordinary route record — one drive, one story, whoever '
     + 'asked for it');
  await post('/session/end?session_id=' + encodeURIComponent(sessionId), {});
}

// ---------------------------------------------------------------------------
section('text mode — she writes, and something else speaks');
// ---------------------------------------------------------------------------
/* Under VOICE_BACKEND=elevenlabs the session produces WORDS and a sink turns
   them into sound. Everything that was hard to get right about interruption is
   supposed to be unchanged; what these check is that "unchanged" is true of
   the parts that now have a different source of truth.

   Three of them, and each one is a different bug if it is wrong:

     the mouth is held until the LISTENER is done, not until the model is.
     Handing it back at response.done lets the next queued thing talk over the
     second half of her sentence.

     a cancel STOPS audio, rather than muting it. Muting is what a barge-in
     does before anyone knows whether a person spoke, and it has to stay
     undoable; a cancel is a fade and a flush and must not be.

     a resume carries what was HEARD. The model finishes an answer seconds
     before the synthesiser does, so resuming from the model's text makes RIO
     skip every word the interruption actually cost. */
{
  const eleven = require(path.join(__dirname, '..', 'static', 'rio_voice_eleven.js'));
  const H = require(path.join(__dirname, 'voice_sink_harness.js'));

  function textHarness(opts) {
    opts = opts || {};
    const arbiter = speech.makeArbiter();
    const sent = [];
    const events = [];
    const rig = H.openSink(eleven, {});
    const controller = rt.createController({
      arbiter: arbiter,
      send: (o) => sent.push(o),
      tool: opts.tool || (() => Promise.resolve({ ok: true })),
      audio: { mute: () => rig.sink.mute(), unmute: () => rig.sink.unmute() },
      voice: rig.sink,
      liveVoice: opts.noVoice ? undefined : 'marin',
      onEvent: (ev) => events.push(ev),
      bargeSustainMs: opts.bargeSustainMs === undefined ? 4 : opts.bargeSustainMs,
      bargeConfirmMs: opts.bargeConfirmMs === undefined ? 8 : opts.bargeConfirmMs,
      resumeInstruction: 'RESUME>>',
    });
    return { arbiter, sent, events, controller, rig,
             types: () => sent.map((e) => e.type),
             resumeSent: () => sent.filter(
               (e) => e.type === 'response.create' && e.response &&
                      /^RESUME>>/.test(e.response.instructions || '')) };
  }

  // --- the one piece of signal processing in the file ----------------------
  {
    /* Everything else the sink does is bookkeeping; this is the part that
       turns bytes into sound, and it is wrong in a way that is hard to hear
       and easy to write: a byte-order slip makes speech into noise, and a
       scale slip makes it quiet or clipped. Both are checked against samples
       whose values are known. */
    const audio = H.fakeAudio();
    const wire = H.fakeTransport();
    const sink = eleven.createSink({ transport: wire, context: audio.ctx,
                                     sampleRate: 24000 });
    const opened = sink.open();
    wire.deliver({ op: 'ready', open: true, sample_rate: 24000 });
    // Four 16-bit little-endian samples: 0, +full scale, -full scale, +half.
    const raw = Buffer.from([0x00, 0x00, 0xff, 0x7f, 0x00, 0x80, 0x00, 0x40]);
    wire.deliver({ op: 'audio', rid: 'x1', text: 'hi',
                   pcm: raw.toString('base64') });
    // begin() first, or the chunk belongs to no utterance and is dropped.
    sink.begin('x1');
    wire.deliver({ op: 'audio', rid: 'x1', text: 'hi',
                   pcm: raw.toString('base64') });
    const src = audio.state.sources.slice(-1)[0];
    ok(!!src && src.buffer && src.buffer.length === 4,
       'four bytes-pairs of PCM become four samples, not eight or two — the ' +
       'byte order is read as little-endian, which is what the socket sends');
    const ch = src.buffer ? src.buffer.getChannelData(0) : [];
    ok(Math.abs(ch[1] - 1.0) < 0.001 && Math.abs(ch[2] + 1.0) < 0.001,
       'full scale in both directions maps to ±1.0, so nothing clips and ' +
       'nothing is quiet');
    ok(Math.abs(ch[3] - 0.5) < 0.001, 'and half scale to 0.5');
    ok(Math.abs(src.buffer.duration - 4 / 24000) < 1e-9,
       'the buffer is as long as the samples say at the rate the server named');
    sink.close();
    await opened.catch(() => {});
  }

  // --- the words go to the sink, not into the void ------------------------
  {
    const h = textHarness();
    h.controller.handle({ type: 'response.created', response: { id: 'r1' } });
    h.controller.handle({ type: 'response.output_text.delta',
                          response_id: 'r1', delta: 'Traffic is thinning ' });
    h.controller.handle({ type: 'response.output_text.delta',
                          response_id: 'r1', delta: 'out ahead.' });
    const deltas = h.rig.wire.ops('delta');
    ok(h.rig.wire.ops('begin').length === 1,
       'a new response opens one utterance on the relay');
    ok(deltas.length === 2 &&
       deltas.map((d) => d.text).join('') === 'Traffic is thinning out ahead.',
       'and every text delta is forwarded verbatim — no phrasing decided in ' +
       'the browser, because phrasing is testable on a server and is not ' +
       'testable in a car');
    ok(h.types().indexOf('response.cancel') < 0,
       'and nothing is cancelled just because the words arrived as text');
  }

  // --- the mouth is held until the LISTENER is done ------------------------
  {
    const h = textHarness();
    let released = false;
    h.arbiter.onEvent((ev) => {
      if (ev.type === 'end' && ev.item && /^live:/.test(ev.item.id)) released = true;
    });
    h.controller.handle({ type: 'response.created', response: { id: 'r2' } });
    h.controller.handle({ type: 'response.output_text.delta',
                          response_id: 'r2', delta: 'Two seconds of speech.' });
    h.rig.say('r2', 'Two seconds of speech.', 2.0);
    h.controller.handle({ type: 'response.done',
                          response: { id: 'r2', status: 'completed' } });
    await settle();
    ok(!released,
       'the model finishing is NOT the mouth coming free — two seconds of ' +
       'her sentence are still unplayed');
    h.rig.done('r2');
    h.rig.audio.advance(2.5);
    await new Promise((r) => setTimeout(r, 160));
    ok(released,
       'it comes free when the audio actually runs out, which is the moment ' +
       'the next thing may start talking');
  }

  // --- how far she got is what came out of the speaker ---------------------
  {
    const h = textHarness();
    h.controller.handle({ type: 'response.created', response: { id: 'r3' } });
    // The model writes the whole answer in one breath...
    h.controller.handle({
      type: 'response.output_text.delta', response_id: 'r3',
      delta: 'The exit is about two miles out and the traffic clears after it.' });
    // ...and the synthesiser is four words into it.
    h.rig.say('r3', 'The exit is about two miles out ', 2.0);
    h.rig.audio.advance(1.0);
    const heard = h.controller.state().said_so_far;
    ok(heard.length > 0 && heard.length < 20,
       'half-way through the first phrase, "how far did she get" is about ' +
       'half of that phrase (' + JSON.stringify(heard) + ')');
    ok(h.controller.state().generated.length > heard.length + 20,
       'and it is much shorter than what the MODEL wrote — resuming from the ' +
       'model would skip everything the driver never heard');
  }

  // --- a cancel stops audio; a mute does not -------------------------------
  {
    const h = textHarness();
    h.controller.handle({ type: 'response.created', response: { id: 'r4' } });
    h.controller.handle({ type: 'response.output_text.delta',
                          response_id: 'r4', delta: 'A long answer, going on.' });
    h.rig.say('r4', 'A long answer, going on.', 3.0);
    h.rig.audio.advance(0.5);

    h.controller.handle({ type: 'input_audio_buffer.speech_started' });
    ok(h.rig.audio.state.stops === 0,
       'the instant a barge-in fires, nothing is stopped — she is muted, and ' +
       'a mute can be taken back when the noise turns out to be a cough');
    const ramp = h.rig.audio.state.ramps.slice(-1)[0];
    ok(ramp && ramp.to === 0 &&
       Math.abs(ramp.overS - eleven.FADE_MS / 1000) < 1e-6,
       'she is faded out over ' + eleven.FADE_MS + ' ms rather than cut, ' +
       'because a hard stop on a voice is a click and a click reads as a fault');

    await settle();                      // past the sustain gate
    ok(h.types().indexOf('response.cancel') >= 0,
       'past the gate the model is told to stop writing');
    ok(h.rig.wire.ops('cancel').length === 1,
       '...and the sink is told to throw away what it had queued — cancelling ' +
       'generation alone leaves the audio already made to play out from ' +
       'under whatever interrupted her');
    await new Promise((r) => setTimeout(r, 60));
    ok(h.rig.audio.state.stops > 0,
       'and the scheduled buffers really are stopped, after the fade');
  }

  // --- a tool follow-up lands while the holding line is still playing ------
  {
    /* The routine collision in text mode, and the one that cost an answer:
       RIO says "let me check", the model finishes WRITING that in a few
       hundred milliseconds, the tool comes back, and the follow-up response is
       created while the holding line is still coming out of the speaker. */
    const h = textHarness();
    h.controller.handle({ type: 'response.created', response: { id: 'r7' } });
    h.controller.handle({ type: 'response.output_text.delta',
                          response_id: 'r7', delta: 'Let me check.' });
    h.rig.say('r7', 'Let me check.', 2.0);
    h.controller.handle({ type: 'response.done',
                          response: { id: 'r7', status: 'completed' } });
    h.rig.done('r7');
    h.rig.audio.advance(0.3);            // 1.7 s of "let me check" still to go

    h.controller.handle({ type: 'response.created', response: { id: 'r8' } });
    ok(h.rig.wire.ops('begin').length === 2,
       'the follow-up opens its own utterance rather than being swallowed by ' +
       'a response that is only waiting on its tail');
    h.controller.handle({ type: 'response.output_text.delta',
                          response_id: 'r8', delta: 'It opens at nine.' });
    const forwarded = h.rig.wire.ops('delta').slice(-1)[0];
    ok(forwarded && forwarded.rid === 'r8',
       'and the answer\'s words reach the synthesiser under the new response');
    ok(h.controller.state().response_id === 'r8',
       'the mouth is held by the new answer');

    const before = h.rig.sink.state().next_at;
    h.rig.say('r8', 'It opens at nine.', 1.0);
    ok(h.rig.sink.state().next_at > before + 0.9,
       'and its audio is queued BEHIND the holding line rather than on top of ' +
       'it — "let me check" then the answer, which is the order anyway');
  }

  // --- A WHOLE TOOL TURN, END TO END --------------------------------------
  {
    /* The turn the driver actually loses when this is wrong, driven as one
       piece: the model calls a tool, the panel answers it, a second response
       is created for the answer, and the words of that second response have
       to reach the speaker.

       Written as a turn rather than as its parts because the parts all passed
       while the turn did not. What was missing was an ordering: the tool
       result comes back on the panel's own clock, and `response.done` for the
       response that called it comes back on the wire's, and nothing makes
       those two agree. */
    const turn = async (opts) => {
      const h = textHarness({ tool: opts.tool });
      h.controller.handle({ type: 'response.created', response: { id: 'q1' } });
      if (opts.holding) {
        h.controller.handle({ type: 'response.output_text.delta',
                              response_id: 'q1', delta: opts.holding });
        h.rig.say('q1', opts.holding, 1.2);
      }
      h.controller.handle({ type: 'response.function_call_arguments.done',
                            name: opts.name, call_id: 'k1', arguments: '{}' });
      const finish = () => {
        h.controller.handle({ type: 'response.done',
                              response: { id: 'q1', status: 'completed' } });
        h.rig.done('q1');
      };
      if (opts.doneFirst) finish();
      await settle();
      if (!opts.doneFirst) { finish(); await settle(); }
      const asked = h.sent.filter((e) => e.type === 'response.create').length;
      if (asked) {
        h.controller.handle({ type: 'response.created', response: { id: 'q2' } });
        h.controller.handle({ type: 'response.output_text.delta',
                              response_id: 'q2', delta: opts.answer });
        await settle();
      }
      const opened = h.rig.wire.ops('begin').map((o) => o.rid);
      const deltas = h.rig.wire.ops('delta');
      return {
        h, asked, opened, deltas,
        // What the relay would actually SPEAK. Text sent under a rid it never
        // opened an utterance for is dropped on the server (voice_dialogue.py
        // delta(): `if utt.rid != rid: return`), so forwarding it is not the
        // same as saying it, and only this second reading catches that.
        heard: deltas.filter((d) => opened.indexOf(d.rid) >= 0)
                     .map((d) => d.text).join(''),
        orphans: deltas.filter((d) => opened.indexOf(d.rid) < 0).length,
      };
    };

    // 1. The ordinary shape: a server tool, and the model answers from it.
    {
      const r = await turn({
        name: 'nav_directions', doneFirst: true,
        tool: () => Promise.resolve({ ok: true, steps: [] }),
        answer: 'Right onto Lincoln, then left on Sunset.' });
      ok(r.asked === 1,
         'a tool result is followed by a request for the spoken answer — ' +
         'without it the model has the result and no reason to say anything');
      ok(r.heard === 'Right onto Lincoln, then left on Sunset.',
         'and the words of that SECOND response reach the speaker, which is ' +
         'the half of a tool turn the driver actually hears');
      ok(r.orphans === 0,
         'with nothing sent under a turn the relay had not opened');
    }

    // 2. ...and the same when the tool answers before the wire says done.
    {
      const r = await turn({
        name: 'nav_directions', doneFirst: false,
        tool: () => Promise.resolve({ ok: true, steps: [] }),
        answer: 'Right onto Lincoln, then left on Sunset.' });
      ok(r.heard === 'Right onto Lincoln, then left on Sunset.',
         'a tool that answers in a microtask -- every tool the panel owns ' +
         'does -- still gets its answer spoken');
    }

    // 3. THE CAMERA'S FAST PATH, WHICH IS THE ONE THAT WAS SILENT.
    {
      /* "What do you see outside" is answered from the running observation
         with no model in the loop, and that is the whole point of it: the
         sentence is already written and already checked against her register,
         so it is spoken directly.

         Directly means IMMEDIATELY -- under a millisecond of server work --
         and that beats `response.done` for the function call coming back down
         the data channel. The controller used to refuse the mouth to anything
         while a response was still open, so the line never opened an
         utterance, its text was dropped by the relay as belonging to a turn
         that was over, and the driver got silence and no second chance.

         Both orders are checked. Only one of them was ever broken, and it is
         the fast one, which is to say the common one. */
      const direct = () => Promise.resolve({
        ok: true, speak_directly: true, path: 'observer_direct',
        speech: 'Clear road, silver sedan a few lengths ahead.' });

      for (const doneFirst of [true, false]) {
        const r = await turn({ name: 'look', doneFirst: doneFirst,
                               tool: direct, answer: '(never asked for)' });
        const when = doneFirst ? 'after response.done' : 'BEFORE response.done';
        ok(r.heard === 'Clear road, silver sedan a few lengths ahead.',
           'the camera answering ' + when + ' is spoken either way — the ' +
           'observation reaches the speaker instead of being dropped');
        ok(r.opened.indexOf('direct:1') >= 0,
           '...under an utterance of its own on the relay (' + when + '), so ' +
           'the context is flushed and the line can report itself finished');
        ok(r.orphans === 0,
           '...with nothing left orphaned under the response it superseded');
        ok(r.h.events.filter((e) => e.type === 'LIVE_DIRECT_ANSWER').length === 1,
           '...and the drive records it as the fast path it was');
        ok(r.asked === 0,
           '...and no model is asked to rewrite a sentence that was already ' +
           'hers');
      }
    }

    // 4. ...and if the line cannot be spoken, the turn is still answered.
    {
      /* THE INVARIANT, checked where it is reachable: a tool result always
         ends as either a spoken line or a request for one. Never as neither,
         which is what the driver was getting.

         A warning arriving first does NOT reach this -- the arbiter queues a
         conversational line behind a safety one rather than refusing it, so
         the fast path still happens, just after the warning. What does reach
         it is the observation coming back with nothing sayable in it, and the
         two things that must then be true are worth stating: the model is
         asked for an answer, and the session is NOT told she said a line she
         never said. That second one is why the assistant item is written
         after the line is spoken rather than before -- a history claiming she
         answered is how a driver asking again gets told she already did. */
      const h = textHarness({ tool: () => Promise.resolve({
        ok: true, speak_directly: true, path: 'observer_direct',
        speech: '   ' }) });
      h.controller.handle({ type: 'response.created', response: { id: 'w1' } });
      h.controller.handle({ type: 'response.function_call_arguments.done',
                            name: 'look', call_id: 'k9', arguments: '{}' });
      await settle();
      ok(h.controller.state().counters.direct_deferred === 1,
         'a direct line that cannot be spoken is counted rather than ' +
         'silently dropped');
      ok(h.sent.filter((e) => e.type === 'response.create').length === 1,
         '...and becomes an ordinary request for an answer, so the question ' +
         'is answered late instead of never');
      ok(h.sent.filter((e) => e.type === 'conversation.item.create' &&
                       e.item && e.item.role === 'assistant').length === 0,
         '...and the session is NOT told she said a line she never said');
      ok(h.rig.wire.ops('delta').length === 0,
         '...and nothing blank was sent to the synthesiser');
    }
  }

  // --- the finished response does not cancel the one that replaced it -----
  {
    /* THE RACE THAT ATE THE DIRECTIONS.
     *
     * A response that only calls a tool finishes the instant the call is
     * made: no text, so the sink's utterance completes with zero chunks and
     * there is no audio to wait for. The mouth is handed back, and the
     * server's follow-up response arrives in the same tick -- before the
     * arbiter has pumped the finished item off its queue.
     *
     * The new answer claims the mouth, the arbiter sees a second `convo` item
     * and supersedes the first, and the first one's `stop` callback runs.
     * `response.cancel` names no response and cancels whatever is generating,
     * which by then is the answer. Live, that came back
     * `cancelled / client_cancelled` with nothing said: RIO called
     * nav_directions, got the route, and read none of it. */
    const h = textHarness({ tool: () => Promise.resolve({ ok: true, steps: [] }) });
    h.controller.handle({ type: 'response.created', response: { id: 'd1' } });
    h.controller.handle({ type: 'response.function_call_arguments.done',
                          name: 'nav_directions', call_id: 'n1', arguments: '{}' });
    // No text in it at all -- the whole response was the call.
    h.controller.handle({ type: 'response.done',
                          response: { id: 'd1', status: 'completed' } });
    h.rig.done('d1');
    await settle();
    // ...and the answer, arriving before the arbiter has finished tidying up.
    h.controller.handle({ type: 'response.created', response: { id: 'd2' } });
    h.controller.handle({ type: 'response.output_text.delta', response_id: 'd2',
                          delta: 'First a right onto Lincoln, then left on Sunset.' });
    await settle();

    ok(h.types().indexOf('response.cancel') < 0,
       'the response that called the tool does not cancel the response that ' +
       'answers it on its way out');
    const opened = h.rig.wire.ops('begin').map((o) => o.rid);
    const spoken = h.rig.wire.ops('delta')
                    .filter((d) => opened.indexOf(d.rid) >= 0)
                    .map((d) => d.text).join('');
    ok(spoken === 'First a right onto Lincoln, then left on Sunset.',
       '...so the directions reach the speaker');
    ok(h.controller.state().response_id === 'd2',
       '...and the mouth belongs to the answer, not to the call');
  }

  // --- the words a tool is judged on are THIS turn's ----------------------
  {
    /* The camera's fast path is a judgement about what the DRIVER asked, so
       the panel sends the driver's transcript along with every tool call. The
       transcript for a turn arrives on its own schedule and is regularly
       later than the tool call it belongs to -- and what used to be sent then
       was the PREVIOUS turn's.

       Measured in a live drive: "what do you see outside" answered from the
       running observation in 42 ms, then "what kind of car is in front of us"
       answered in 5 ms with the same sentence about the road, because the
       transcript the panel held was still the first question. The second
       question never reached a camera. */
    const h = textHarness();
    h.controller.handle({
      type: 'conversation.item.input_audio_transcription.completed',
      transcript: 'what do you see outside' });
    ok(h.controller.state().spoken_this_turn === 'what do you see outside',
       'a tool called in the turn the driver just spoke is given their words');

    h.controller.handle({ type: 'input_audio_buffer.speech_started' });
    ok(h.controller.state().spoken_this_turn === '',
       '...and the moment they start a NEW sentence those words go stale, so ' +
       'the previous question cannot judge this one');
    ok(h.controller.state().last_transcript === 'what do you see outside',
       '...while the transcript itself is kept, because classifying a ' +
       'barge-in still needs whatever was said last');

    h.controller.handle({
      type: 'conversation.item.input_audio_transcription.completed',
      transcript: 'what kind of car is in front of us' });
    ok(h.controller.state().spoken_this_turn === 'what kind of car is in front of us',
       '...and the new turn\'s words are sent as soon as they land');
  }

  // --- the API refuses a response outright ---------------------------------
  {
    /* Not a cut-off: nothing was said, so there is nothing to resume from and
       `said` would be empty. It is almost always the per-minute token
       ceiling, and it lands on tool turns first because a tool turn spends
       two responses where an ordinary answer spends one -- which is exactly
       the shape the driver reported: "hello" works, anything that needs a
       tool does not. tools/realtime_selftest.py run_session_cost has the
       arithmetic.

       Silence used to be the whole behaviour. One retry, at the delay the API
       itself names, is the difference between a late answer and none. */
    const h = textHarness();
    h.controller.handle({
      type: 'conversation.item.input_audio_transcription.completed',
      transcript: 'what are the directions' });
    h.controller.handle({ type: 'response.created', response: { id: 'f1' } });
    h.controller.handle({
      type: 'response.done',
      response: { id: 'f1', status: 'failed', status_details: { error: {
        code: 'rate_limit_exceeded',
        message: 'Rate limit reached ... Please try again in 0.05s.' } } } });
    h.rig.done('f1');
    const failed = h.events.filter((e) => e.type === 'LIVE_RESPONSE_FAILED')[0];
    ok(failed && failed.code === 'rate_limit_exceeded',
       'a refused response is reported with the reason the API gave, not as ' +
       'the anonymous "other" every unnamed fault used to share');
    ok(h.controller.state().counters.responses_failed === 1,
       '...and counted, so a drive can say how often it happened');
    // Past the floor the retry waits: the API's own "try again in 0.05s" is
    // clamped up, because a retry that beats the ceiling it was refused by is
    // a second refusal.
    await new Promise((r) => setTimeout(r, 600));
    ok(h.sent.filter((e) => e.type === 'response.create').length === 1,
       '...and asked for again once, so the driver gets a late answer ' +
       'rather than silence');

    // ...once. A second refusal means the budget is genuinely gone.
    h.controller.handle({ type: 'response.created', response: { id: 'f2' } });
    h.controller.handle({
      type: 'response.done',
      response: { id: 'f2', status: 'failed', status_details: { error: {
        code: 'rate_limit_exceeded',
        message: 'Please try again in 0.05s.' } } } });
    h.rig.done('f2');
    await new Promise((r) => setTimeout(r, 600));
    ok(h.sent.filter((e) => e.type === 'response.create').length === 1,
       'and not again after that — asking a spent per-minute budget twice ' +
       'only spends the next minute');
    ok(h.controller.state().counters.responses_retried === 1,
       'one retry, counted');
  }

  // --- a false barge-in resumes from what was HEARD ------------------------
  {
    const h = textHarness();
    h.controller.handle({ type: 'response.created', response: { id: 'r5' } });
    h.controller.handle({
      type: 'response.output_text.delta', response_id: 'r5',
      delta: 'The exit is about two miles out and the traffic clears after it.' });
    h.rig.say('r5', 'The exit is about two miles out ', 2.0);
    h.rig.audio.advance(1.0);
    h.controller.handle({ type: 'input_audio_buffer.speech_started' });
    await settle();
    h.controller.handle({ type: 'input_audio_buffer.speech_stopped' });
    h.controller.handle({
      type: 'conversation.item.input_audio_transcription.completed',
      transcript: '' });
    await settle();
    const resume = h.resumeSent()[0];
    ok(!!resume, 'a door thunk with no words behind it still resumes the answer');
    const carried = resume ? resume.response.instructions.replace('RESUME>>', '') : '';
    ok(carried.length > 0 && carried.length < 25,
       'carrying the part she had SPOKEN (' + JSON.stringify(carried.trim()) + ')');
    ok(resume && resume.response.output_modalities[0] === 'text',
       'and the continuation is asked for as text, so the same mouth finishes ' +
       'the sentence it started');
  }

  // --- a warning does not wait on a dictation that cannot happen -----------
  {
    const h = textHarness();
    let rejected = null;
    await h.controller.speak('Watch your distance.').catch((e) => { rejected = e; });
    ok(rejected && /text_mode/.test(rejected.message),
       'dictation is refused at once in text mode, so rio_speak falls through ' +
       'to the synthesiser instead of spending a warning\'s budget waiting');
    ok(h.types().indexOf('response.create') < 0,
       'and nothing was asked of the session at all');
  }

  // --- ElevenLabs goes away entirely, and RIO takes her voice back ---------
  {
    const h = textHarness();
    ok(h.controller.state().voice_backend === 'elevenlabs',
       'the drive starts on ElevenLabs');
    ok(h.controller.useLiveVoice('service_down') === true,
       'and can be handed back to the session mid-drive');
    const update = h.sent.filter((e) => e.type === 'session.update').pop();
    ok(update && update.session.output_modalities[0] === 'audio',
       'the session is asked for audio from here on');
    ok(update && update.session.audio.output.voice === 'marin',
       'in the voice the SESSION carried, named now because a session that ' +
       'has produced no audio can still be told whose voice to use');
    ok(h.controller.state().voice_backend === 'openai_realtime',
       'and the drive reports the voice it is actually using');
    ok(h.controller.useLiveVoice('again') === false,
       'once, and only once — a voice that flickers because the network is ' +
       'flickering is worse than either voice');

    /* AND IT REFUSES RATHER THAN INVENTING A VOICE.
     *
     * This used to read `cfg.cedarVoice || 'cedar'`, which looks like a
     * harmless default and stopped being one the moment RIO's voice became
     * marin: a session payload that failed to carry the voice would have
     * handed the driver a different person mid-drive and logged that the
     * fallback succeeded. The voice is named in config.py and carried by the
     * mint; there is no third place for it to come from. */
    const bare = textHarness({ noVoice: true });
    ok(bare.controller.useLiveVoice('service_down') === false,
       'a session that carried no voice cannot be switched to one — the ' +
       'fallback refuses instead of picking a name out of the source');
    ok(!bare.sent.some((e) => e.type === 'session.update'),
       '...and asks the session for nothing, so the drive stays on the ' +
       'mouth it has rather than changing speaker on a guess');

    // ...and after the switch it behaves exactly like the live-voice path did.
    h.controller.handle({ type: 'response.created', response: { id: 'r6' } });
    h.controller.handle({ type: 'response.output_audio_transcript.delta',
                          response_id: 'r6', delta: 'Back in my own voice.' });
    ok(h.controller.state().said_so_far === 'Back in my own voice.',
       'reading how far she got from her own transcript again');
  }
}

const serverArg = process.argv.indexOf('--server');
if (serverArg >= 0) {
  const base = (process.argv[serverArg + 1] || '').indexOf('http') === 0
    ? process.argv[serverArg + 1] : 'http://127.0.0.1:8888';
  await liveRouting(base);
}

console.log('\n' + (failures ? 'FAILED ' + failures + '/' : 'PASSED ') + checks + ' checks');
process.exit(failures ? 1 : 0);
}

main().catch(e => { console.error(e); process.exit(2); });
