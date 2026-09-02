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
  global.RIO = {
    sessionId: sessionId,
    url: (p) => base + p + (p.indexOf('?') >= 0 ? '&' : '?') +
                'session_id=' + encodeURIComponent(sessionId),
    headway: { startWatch: () => {}, onPosition: (fn) => { gps.sink = fn; } },
    speak: { provider: () => ({ play: () => Promise.resolve(), stop: () => {} }) },
  };

  require(path.join(__dirname, '..', 'static', 'rio_speech.js'));
  require(path.join(__dirname, '..', 'static', 'rio_navcore.js'));
  require(path.join(__dirname, '..', 'static', 'rio_navplan.js'));
  require(path.join(__dirname, '..', 'static', 'rio_nav.js'));
  (docHandlers.DOMContentLoaded || []).forEach((f) => f());
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
