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
const tick = () => new Promise(r => setTimeout(r, 0));
// Long enough for both barge-in timers in the harness (4 ms + 8 ms) to run.
const settle = () => new Promise(r => setTimeout(r, 40));

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
  const docHandlers = {};
  global.document = {
    addEventListener: (t, f) => { (docHandlers[t] = docHandlers[t] || []).push(f); },
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
  (docHandlers.DOMContentLoaded || []).forEach((f) => f());

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
      tool: () => Promise.resolve({ ok: true }),
      audio: { mute: () => rig.sink.mute(), unmute: () => rig.sink.unmute() },
      voice: rig.sink,
      cedarVoice: 'cedar',
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
    ok(h.controller.useCedar('service_down') === true,
       'and can be handed back to the session mid-drive');
    const update = h.sent.filter((e) => e.type === 'session.update').pop();
    ok(update && update.session.output_modalities[0] === 'audio',
       'the session is asked for audio from here on');
    ok(update && update.session.audio.output.voice === 'cedar',
       'in the voice it was configured with, named now because a session that ' +
       'has produced no audio can still be told whose voice to use');
    ok(h.controller.state().voice_backend === 'openai_realtime',
       'and the drive reports the voice it is actually using');
    ok(h.controller.useCedar('again') === false,
       'once, and only once — a voice that flickers because the network is ' +
       'flickering is worse than either voice');

    // ...and after the switch it behaves exactly like the cedar path did.
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
