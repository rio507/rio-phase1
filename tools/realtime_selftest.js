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

let checks = 0, failures = 0;
function ok(cond, what) {
  checks++;
  if (!cond) { failures++; console.log('  FAIL  ' + what); }
  else console.log('  ok    ' + what);
}
function section(name) { console.log('\n=== ' + name + ' ==='); }
const tick = () => new Promise(r => setTimeout(r, 0));

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
  });
  return { arbiter, sent, events, audio, controller,
           types: () => sent.map(e => e.type) };
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
  await tick();
  ok(h.audio.muted === false, 'she is talking');
  h.controller.handle({ type: 'input_audio_buffer.speech_started' });
  await tick();
  ok(h.audio.muted === true,
     'the moment the driver speaks, she goes quiet — locally, without waiting '
     + 'for the server to agree');
  ok(h.arbiter.state().speaking === null, 'and the mouth is released');
  ok(h.controller.state().counters.barge_ins === 1, 'counted as a barge-in');
  ok(h.events.some(e => e.type === 'LIVE_BARGE_IN'), 'and reported as one');
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

console.log('\n' + (failures ? 'FAILED ' + failures + '/' : 'PASSED ') + checks + ' checks');
process.exit(failures ? 1 : 0);
}

main().catch(e => { console.error(e); process.exit(2); });
