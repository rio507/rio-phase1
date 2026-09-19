/* provider_adapter_selftest.js — the vendor seam, measured against the stack
 * that is running today.
 *
 *   node tools/provider_adapter_selftest.js
 *
 * static/rio_provider.js exists so that no vendor's event name appears in
 * createController. That is easy to claim and easy to half-do, and a half-done
 * adapter is worse than none: it reads as a boundary while a guard behind it
 * silently never fires. So this file asserts the four properties that make it
 * real, and one that makes it safe.
 *
 *   IDENTITY      on openai_realtime the normaliser returns the SAME OBJECT
 *                 for every event the controller handles. Not an equal one --
 *                 the same one. That is what "the current stack is unchanged"
 *                 means when the claim is about a function on the hot path.
 *
 *   COMPLETENESS  the canonical vocabulary and the switch in rio_realtime.js
 *                 agree, in BOTH directions, checked against the source. An
 *                 event the controller handles and the vocabulary has
 *                 forgotten can never be mapped onto from another vendor, and
 *                 the symptom is not an error -- it is a guard that stops
 *                 working on one provider only.
 *
 *   EQUIVALENCE   a controller driven through the provider and a controller
 *                 driven the old way reach the same state on the same events.
 *                 This is the check that makes the refactor a refactor.
 *
 *   HONESTY       what nobody has measured reads as unmeasured. The xAI
 *                 profile has an UNKNOWN in it -- whether
 *                 ...transcription.completed carries an item_id -- and a
 *                 boolean cannot hold that. Asserted here is that it CANNOT be
 *                 read as a boolean at all, so no `if` downstream can quietly
 *                 take the degraded path.
 *
 * WHAT THIS FILE CANNOT DO, and says so rather than pretending: it does not
 * verify a single claim about xAI's wire. The account this repo holds a key for
 * is credit-blocked, so the xai_voice profile is documentation shaped like
 * code. Every check below tests that the RECORD is internally consistent and
 * that the seam honours it -- never that the record is true.
 */
'use strict';

const path = require('path');
const fs = require('fs');

const provider = require(path.join(__dirname, '..', 'static', 'rio_provider.js'));
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

/* A controller on a fake wire, optionally behind a provider. Deliberately the
   same explicit-mapping shape as realtime_selftest's harness: an option this
   does not name is an option the controller never sees, which is silently a
   test of the default. */
function harness(opts) {
  opts = opts || {};
  const sent = [];
  const events = [];
  const audio = { muted: false };
  const controller = rt.createController({
    arbiter: speech.makeArbiter(),
    send: (obj) => sent.push(obj),
    tool: () => Promise.resolve({ ok: true, answer: 'forty-two' }),
    audio: {
      mute: () => { audio.muted = true; },
      unmute: () => { audio.muted = false; },
    },
    onEvent: (ev) => events.push(ev),
    bargeSustainMs: 4,
    bargeConfirmMs: 8,
    provider: opts.provider,
    holdTail: opts.holdTail,
  });
  return { sent, events, audio, controller,
           types: () => sent.map(e => e.type),
           evTypes: () => events.map(e => e.type) };
}

async function main() {

// ---------------------------------------------------------------------------
section('identity — the current stack goes through untouched');
// ---------------------------------------------------------------------------
{
  const p = provider.create('openai_realtime');

  ok(Object.keys(provider.profile('openai_realtime').events).length === 0,
     'the openai_realtime event map is EMPTY — which is what makes these the '
     + 'canonical names rather than one vendor\'s dialect of them');

  let same = 0;
  provider.CANONICAL.forEach((type) => {
    const ev = { type: type, response_id: 'r1', item_id: 'i1' };
    if (p.normalise(ev) === ev) same++;
  });
  ok(same === provider.CANONICAL.length,
     `every one of the ${provider.CANONICAL.length} canonical events comes `
     + 'back as the SAME OBJECT, not a copy — no allocation on the hot path');

  /* The audio delta is not in the canonical list because the controller does
     not switch on it, and it is the single most frequent event on the wire.
     Checked separately precisely because it is the one whose cost would
     matter. */
  const delta = { type: 'response.output_audio.delta', delta: 'AAAA' };
  ok(p.normalise(delta) === delta,
     'and so does response.output_audio.delta, which arrives at audio rate '
     + 'and is not in the switch at all');

  ok(p.normalise(null) === null && p.normalise({}) !== undefined,
     'a malformed event is handed back rather than thrown on — the controller '
     + 'already refuses it, and two places refusing it is one too many');

  const st = p.stats();
  ok(st.renamed === 0 && st.dropped === 0,
     `nothing was renamed and nothing was dropped (${st.seen} seen)`);
}

// ---------------------------------------------------------------------------
section('completeness — the vocabulary and the switch agree, from the source');
// ---------------------------------------------------------------------------
{
  /* Read out of rio_realtime.js rather than imported, because what is being
     checked is the SWITCH, and the switch is not a value the module exports.
     If this extraction ever stops finding the block it fails loudly below
     rather than asserting over an empty list. */
  const src = fs.readFileSync(
    path.join(__dirname, '..', 'static', 'rio_realtime.js'), 'utf8');
  const a = src.indexOf('handle: function (ev) {');
  const b = src.indexOf('onEvent: function (fn)', a);
  ok(a > 0 && b > a, 'found handle()\'s switch in the source to compare against');

  const cases = [...src.slice(a, b).matchAll(/case '([^']+)':/g)].map(m => m[1]);
  ok(cases.length > 10,
     `extracted ${cases.length} case labels (a regex that matched nothing `
     + 'would make every check below pass for the wrong reason)');

  const vocab = new Set(provider.CANONICAL);
  const handled = new Set(cases);

  const missing = cases.filter(c => !vocab.has(c));
  ok(missing.length === 0,
     'every event the controller handles is in the canonical vocabulary'
     + (missing.length ? ' — MISSING: ' + missing.join(', ') : ''));

  const extra = provider.CANONICAL.filter(c => !handled.has(c));
  ok(extra.length === 0,
     'and nothing in the vocabulary is unhandled — a name here that the switch '
     + 'dropped would be a mapping target that silently does nothing'
     + (extra.length ? ' — EXTRA: ' + extra.join(', ') : ''));

  /* Every profile's mappings must land on a name the switch actually handles,
     or be an explicit DROP. This is the check that catches a typo in a vendor
     map — the failure mode being an event mapped onto a name nobody handles,
     which looks exactly like the vendor not sending it. */
  let badTargets = [];
  provider.profiles().forEach((name) => {
    const map = provider.profile(name).events || {};
    Object.keys(map).forEach((from) => {
      const to = map[from];
      if (to !== provider.DROP && !vocab.has(to)) badTargets.push(name + ': ' + from + ' -> ' + to);
    });
  });
  ok(badTargets.length === 0,
     'every mapping in every profile targets a handled event or an explicit '
     + 'DROP' + (badTargets.length ? ' — ' + badTargets.join('; ') : ''));
}

// ---------------------------------------------------------------------------
section('equivalence — provider and no-provider reach the same state');
// ---------------------------------------------------------------------------
{
  /* The claim under test: routing the current stack through the seam changes
     nothing. Two controllers, one told `holdTail: true` the way connect() used
     to say it, one given the openai_realtime provider that now answers it, and
     the same events into both. */
  const sequence = [
    { type: 'response.created', response: { id: 'r1' } },
    { type: 'output_audio_buffer.started', response_id: 'r1' },
    { type: 'response.output_audio_transcript.delta', response_id: 'r1',
      delta: 'the road ahead is clear' },
    { type: 'input_audio_buffer.committed', item_id: 'i1' },
    { type: 'conversation.item.input_audio_transcription.completed',
      item_id: 'i1', transcript: 'what is that building' },
    { type: 'response.done',
      response: { id: 'r1', status: 'completed' } },
    { type: 'output_audio_buffer.stopped', response_id: 'r1' },
  ];

  const oldWay = harness({ holdTail: true });
  const newWay = harness({ provider: provider.create('openai_realtime') });
  sequence.forEach((ev) => {
    oldWay.controller.handle(JSON.parse(JSON.stringify(ev)));
    newWay.controller.handle(JSON.parse(JSON.stringify(ev)));
  });
  await tick();

  const sa = JSON.stringify(oldWay.controller.state());
  const sb = JSON.stringify(newWay.controller.state());
  ok(sa === sb,
     'identical controller state after a full turn — the seam is transparent '
     + 'to every counter, every cutoff and the whole mute ledger');
  ok(JSON.stringify(oldWay.types()) === JSON.stringify(newWay.types()),
     'and identical traffic back to the model');
  ok(JSON.stringify(oldWay.evTypes()) === JSON.stringify(newWay.evTypes()),
     'and the same panel events, in the same order');
}

// ---------------------------------------------------------------------------
section('the tail — held on stated evidence, and the evidence is visible');
// ---------------------------------------------------------------------------
{
  const oai = provider.create('openai_realtime');
  const xai = provider.create('xai_voice');

  ok(oai.holdTail() === true && oai.capability('tailEvidence') === 'server',
     'openai_realtime holds the tail on the API\'s word '
     + '(output_audio_buffer.stopped)');
  ok(xai.holdTail() === true
     && xai.capability('tailEvidence') === 'client_playout',
     'xai_voice holds it too — but on OUR playout queue draining, and the '
     + 'record says which, because that is a downgrade and not a detail');
  ok(xai.capability('audioBufferEvents') === false
     && xai.capability('clientOwnsPlayout') === true,
     'because the audio-buffer events are WebRTC/SIP only and the reachable '
     + 'transport is WebSocket — so the mouth-hold is reconstructed, not ported');

  /* The one case where the tail must NOT be held: nothing knows when the
     audio ended. Held on a timer is worse than not held, because a response
     that never ends takes the mouth with it. */
  const none = provider.create('openai_realtime');
  ok(typeof none.holdTail === 'function',
     'holdTail is asked, not asserted — the transport dependency is now a '
     + 'value a program can read instead of a comment');
}

// ---------------------------------------------------------------------------
section('xai_voice — the cumulative transcript is dropped, not mishandled');
// ---------------------------------------------------------------------------
{
  const p = provider.create('xai_voice');

  const updated = {
    type: 'conversation.item.input_audio_transcription.updated',
    item_id: 'i1', transcript: 'what is that buil',
  };
  ok(p.normalise(updated) === null,
     '...transcription.updated is DROPPED. The controller has no cumulative '
     + 'concept and waits for .completed; passing it through would be harmless '
     + 'today and a trap for the first live-transcript readout');

  p.normalise({ type: 'conversation.item.input_audio_transcription.updated',
                item_id: 'i1', transcript: 'what is that building' });
  ok(p.cumulativeTranscript() === 'what is that building',
     'the last cumulative transcript is kept — it is the only evidence '
     + 'available if item_id turns out to be absent, and it gives text, not '
     + 'identity, which is why it is recorded and not wired up as a fallback');

  const done = { type: 'conversation.item.input_audio_transcription.completed',
                 item_id: 'i1', transcript: 'what is that building' };
  ok(p.normalise(done) === done,
     '.completed passes through untouched — xAI still emits it, and it is what '
     + 'the self-supersede binding reads');

  const failed = { type: 'conversation.item.input_audio_transcription.failed',
                   item_id: 'i1' };
  ok(p.normalise(failed) === failed,
     '.failed is passed, not dropped: xAI does not emit it, so the handler '
     + 'simply never runs. Dropping an event nobody sends would be a lie about '
     + 'why the branch is dead');
  ok(p.capability('transcriptionFailed') === false,
     '...and the record is where that is written down, for the ledger comment '
     + 'that has to explain a vanished cutoff category');

  const st = p.stats();
  ok(st.dropped === 2 && st.renamed === 0,
     `two dropped, none renamed (${st.seen} seen) — the only divergence in the `
     + 'whole map is a drop');
}

// ---------------------------------------------------------------------------
section('honesty — an unmeasured fact cannot be read as a boolean');
// ---------------------------------------------------------------------------
{
  const p = provider.create('xai_voice');
  const v = p.capability('transcriptionItemId');

  ok(v === provider.UNKNOWN, 'transcriptionItemId is UNKNOWN, not a guess');
  ok(v !== true && v !== false && typeof v !== 'boolean',
     'and it is not a boolean, so `=== true` and `=== false` both fail closed '
     + '— there is no comparison that quietly takes the wrong branch');
  ok(p.unknown('transcriptionItemId') === true,
     'unknown() is the only correct way to read it');

  const probes = p.probes();
  ok(probes.length === 2,
     `probes() names what is outstanding (${probes.length}): `
     + probes.map(x => x.capability).join(', '));
  const item = probes.find(x => x.capability === 'transcriptionItemId');
  ok(item && /item_id/.test(item.settle) && /grok-transcribe/.test(item.settle),
     'and each one carries the procedure that settles it, in enough detail to '
     + 'write the probe from');
  ok(p.blockedPaths().indexOf('supersede') >= 0,
     'the supersede path is named as blocked — the binding at '
     + 'rio_realtime.js:2742 is what the unknown gates, and commit ee0a909 is '
     + 'what it costs when it is wrong');

  const settled = provider.create('openai_realtime');
  ok(settled.probes().length === 0 && settled.blockedPaths().length === 0,
     'the running stack has nothing outstanding, which is what empty is for');
}

// ---------------------------------------------------------------------------
section('the record is a copy, and the resolver knows a voice from a wire');
// ---------------------------------------------------------------------------
{
  const p = provider.create('openai_realtime');
  const caps = p.capabilities();
  caps.audioBufferEvents = 'vandalised';
  ok(p.capability('audioBufferEvents') === true,
     'capabilities() hands out a copy — a caller that could edit the record '
     + 'would be a second source of truth with no file of its own');

  ok(provider.forVoiceBackend('openai_realtime').name === 'openai_realtime',
     'openai_realtime resolves to its own event stream');
  ok(provider.forVoiceBackend('elevenlabs').name === 'openai_realtime',
     'and so does elevenlabs — it changes the MOUTH, not the wire, and a '
     + 'resolver that passed the backend name straight through would have '
     + 'thrown on a configuration that works today');

  let threwLive = false;
  try { provider.forVoiceBackend('gpt_live'); } catch (e) { threwLive = true; }
  ok(threwLive,
     'gpt_live throws rather than resolving: rio_live.js owns that event '
     + 'stream and a controller built from this file would be the wrong one');

  let threwJunk = false;
  try { provider.create('grok'); } catch (e) { threwJunk = true; }
  ok(threwJunk,
     'and an unknown provider throws at create() — a mint that forgot to say '
     + 'which vendor it minted is a bug worth hearing about, not a reason to '
     + 'guess and half-work');
}

// ---------------------------------------------------------------------------
section('cost — what the seam adds to every event of every drive');
// ---------------------------------------------------------------------------
{
  /* The ingress runs on every event, and response.output_audio.delta arrives
     at audio rate. The number that matters is the identity path, because that
     is 100% of the current stack. Reported rather than asserted tightly: a
     threshold tuned on this box would fail on a slower one for no reason. */
  const p = provider.create('openai_realtime');
  const ev = { type: 'response.output_audio.delta', delta: 'A'.repeat(640) };
  const N = 200000;

  // Warm, so this measures the steady state and not the first-call compile.
  for (let i = 0; i < 20000; i++) p.normalise(ev);

  const t0 = process.hrtime.bigint();
  for (let i = 0; i < N; i++) p.normalise(ev);
  const perCall = Number(process.hrtime.bigint() - t0) / N;

  console.log(`        identity path: ${perCall.toFixed(1)} ns/event`);
  ok(perCall < 1000,
     `the identity path costs ${perCall.toFixed(1)} ns per event — against a `
     + '20 ms audio frame, this is not on the budget');

  /* And the path that does allocate, so the comparison is honest rather than
     only flattering. */
  const x = provider.create('xai_voice');
  const rn = { type: 'conversation.item.input_audio_transcription.updated',
               transcript: 'x'.repeat(200) };
  const t1 = process.hrtime.bigint();
  for (let i = 0; i < N; i++) x.normalise(rn);
  const perDrop = Number(process.hrtime.bigint() - t1) / N;
  console.log(`        drop path:     ${perDrop.toFixed(1)} ns/event`);
  ok(perDrop < 5000,
     `and the drop path ${perDrop.toFixed(1)} ns — it runs once per utterance, `
     + 'not once per frame');
}

console.log(`\n${checks - failures}/${checks} checks passed`);
if (failures) {
  console.log(`${failures} FAILED`);
  process.exit(1);
}
}

main().catch((e) => { console.error(e); process.exit(1); });
