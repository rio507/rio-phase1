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
  h.controller.speak('You\'re too close.');
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
  // 1. A visual question reaches the camera, and the observation comes back
  //    into the session for RIO to phrase.
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
     'and asks her to answer from it — she phrases it, the pipeline does not');
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
  ok(/RIO\.headway\.captureFromVideo\(video\)/.test(
       (html.match(/async function startLiveFrames[\s\S]*?\n}/) || [''])[0]),
     'through the SAME capture path the drive uses — no second pipeline');
  ok(/if \(RIO\.driving \|\| liveFrames\.timer/.test(html),
     'and not at all when a drive is already feeding the ring faster');
  ok(/stopLiveFrames\(\)/.test(html) &&
     (html.match(/stopLiveFrames\(\)/g) || []).length >= 3,
     'stopped when the conversation ends, and when it fails to start');
}

console.log('\n' + (failures ? 'FAILED ' + failures + '/' : 'PASSED ') + checks + ' checks');
process.exit(failures ? 1 : 0);
}

main().catch(e => { console.error(e); process.exit(2); });
