/* live_controller_selftest.js — the gpt-live-1 controller, without a browser.
 *
 *   node tools/live_controller_selftest.js
 *
 * The controller is a pure handler over the data-channel event stream, which
 * is the property that makes this possible: every interesting thing that can
 * happen in a car is a sequence of those events, and none of them needs a
 * microphone to produce.
 *
 * WHAT IS CHECKED, and each of these is a failure that has actually happened
 * on one backend or another:
 *
 *   the echo gate refuses her own voice          (measured 8/8 on the real API)
 *   ...and does NOT refuse a real question       (the gate that eats the drive)
 *   a tool call returns TWO events               (one and she goes silent)
 *   a deterministic line uses instructions       (commentary paraphrases it)
 *   delegation_id is present and null            (absent is a mute car)
 *   a refused line resolves as a failure         (so the synthesiser takes over)
 *   guards off means guards off                  (the flag has to actually work)
 */
'use strict';
const path = require('path');
const live = require(path.join(__dirname, '..', 'static', 'rio_live.js'));

let pass = 0, fail = 0;
function ok(name, cond, extra) {
  if (cond) { pass++; console.log('  ok   ' + name); }
  else { fail++; console.log('  FAIL ' + name + (extra ? '  ' + extra : '')); }
}

function harness(opts) {
  const sent = [];
  const seen = { driver: [], echo: [], said: [], errors: [] };
  const c = live.createController(Object.assign({
    send: function (o) { sent.push(o); },
    tool: function (n) { return Promise.resolve({ ok: true, tool: n }); },
    onDriver: function (t) { seen.driver.push(t); },
    onEcho: function (t) { seen.echo.push(t); },
    onSaid: function (t) { seen.said.push(t); },
    onError: function (e) { seen.errors.push(e); },
  }, opts || {}));
  return { c, sent, seen };
}

/* --- the echo gate ----------------------------------------------------- */
console.log('\nthe echo gate');
{
  const { c, seen } = harness({ guards: { echo_gate: true },
                                echoTailMs: 600, echoTextWindowS: 15,
                                echoTextOverlap: 0.8, echoTextMinWords: 2 });
  /* She speaks; her own words come straight back, which is what a phone
     speaker eight inches from a phone microphone does. */
  c.onEvent({ type: 'session.output_transcript.done',
              transcript: "Back off — now." });
  c.onEvent({ type: 'session.input_transcript.done',
              transcript: 'Back off now' });
  ok('her own line is refused as an echo', seen.echo.length === 1
     && seen.driver.length === 0, JSON.stringify(seen));
}
{
  const { c, seen } = harness({ guards: { echo_gate: true },
                                echoTailMs: 0, echoTextWindowS: 15,
                                echoTextOverlap: 0.8, echoTextMinWords: 2 });
  c.onEvent({ type: 'session.output_transcript.done',
              transcript: "Back off — now." });
  c.onEvent({ type: 'session.input_transcript.done',
              transcript: 'Is there a petrol station near the next exit' });
  ok('a real question is NOT refused', seen.driver.length === 1
     && seen.echo.length === 0, JSON.stringify(seen));
}
{
  const { c, seen } = harness({ guards: { echo_gate: false },
                                echoTailMs: 600 });
  c.onEvent({ type: 'session.output_transcript.done',
              transcript: "Back off — now." });
  c.onEvent({ type: 'session.input_transcript.done',
              transcript: 'Back off now' });
  ok('guards.echo_gate:false really disables it',
     seen.driver.length === 1 && seen.echo.length === 0);
}

/* --- the tool loop ------------------------------------------------------ */
console.log('\nthe tool loop');
{
  const { c, sent } = harness({});
  c.onEvent({ type: 'response.event', delegation_id: 'd1',
              event: { type: 'response.output_item.done',
                       item: { type: 'function', call_id: 'call_9',
                               name: 'nav_status', arguments: '{"x":1}' } } });
  setTimeout(function () {
    const types = sent.map(function (s) { return s.type; });
    ok('a tool call returns item.create AND response.create',
       types.indexOf('response.item.create') !== -1
       && types.indexOf('response.create') !== -1, types.join(','));
    const res = sent.filter(function (s) {
      return s.type === 'response.item.create'; })[0];
    ok('the result names the call it answers',
       res && res.item && res.item.call_id === 'call_9');
    runSpeech();
  }, 20);
}

/* --- deterministic speech ----------------------------------------------- */
function runSpeech() {
  console.log('\ndeterministic speech');
  {
    const { c, sent } = harness({});
    c.speak('In 300 feet, turn right onto Ocean Avenue.');
    const e = sent[0];
    ok('a line goes through instructions.append, not commentary',
       e && e.type === 'session.instructions.append', e && e.type);
    ok('the exact words are in it',
       e && e.content.indexOf('In 300 feet, turn right onto Ocean Avenue.')
       !== -1);
    ok('delegation_id is present and null',
       e && Object.prototype.hasOwnProperty.call(e, 'delegation_id')
       && e.delegation_id === null);
  }
  {
    /* A refused line must not leave the deterministic path waiting for a mouth
       that will never open -- rio_speak falls back to the synthesiser on a
       failed promise, and hangs forever on one that never settles. */
    const { c, sent } = harness({});
    let out = null;
    c.speak('Back off — now.').then(function (r) { out = r; });
    const id = sent[0].event_id;
    c.onEvent({ type: 'error',
                error: { client_event_id: id, message: 'nope' } });
    setTimeout(function () {
      ok('a refused line resolves as a failure', out && out.ok === false,
         JSON.stringify(out));
      runStart();
    }, 20);
  }
}

function runStart() {
  {
    const { c, sent } = harness({});
    let started = false, settled = null;
    c.speak('Watch your distance.', { onStart: function () { started = true; } })
      .then(function (r) { settled = r; });
    c.onEvent({ type: 'session.output_transcript.delta', delta: 'Watch' });
    ok('onStart fires when her mouth opens', started === true);
    c.onEvent({ type: 'session.output_transcript.done',
                transcript: 'Watch your distance.' });
    setTimeout(function () {
      ok('the line resolves when she finishes',
         settled && settled.ok === true, JSON.stringify(settled));
      done();
    }, 20);
  }
}

function done() {
  console.log('\n' + '-'.repeat(50));
  console.log('  ' + pass + ' passed, ' + fail + ' failed');
  process.exit(fail === 0 ? 0 : 1);
}
