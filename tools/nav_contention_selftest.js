/* nav_contention_selftest.js — two nav lines, one mouth.
 *
 *   node tools/nav_contention_selftest.js
 *
 * THE DRIVE. 2026-09-17, session 0233da0d, an iPhone on a mount. A route to
 * the Grand Canyon is started at 18:35:34 and the log says this, in 36 ms:
 *
 *   18:35:35.197  NAV_ROUTE_START_CALL   depart  "Head east, then turn left
 *                                                 onto Palisades Dr."
 *   18:35:35.202  NAV_SPEECH_SPOKEN      depart  reason=superseded
 *   18:35:35.223  NAV_CONTEXTUAL_CALL    near    "Turn left onto Palisades Dr."
 *                                                 at_m=150  tta_s=4.8
 *   18:35:35.225  VOICE_SILENT   {channel:nav, callType:near,
 *                                 reason:"busy", detail:"no_clip"}
 *   18:35:35.233  NAV_SPEECH_SPOKEN      near    reason=error
 *
 * The near call for a junction 4.8 seconds away was never said. Not dropped by
 * the arbiter, not expired, not pre-empted by anything more urgent: refused by
 * the line it had itself just superseded.
 *
 * THE MECHANISM, which is four files agreeing that somebody else is doing it:
 *
 *   rio_navplan   both calls are arbiter items in group `nav:m0`, so the near
 *                 call supersedes the depart call. Correct, and the whole
 *                 point of grouping by maneuver.
 *   rio_speech    supersede => stopCurrent() => item.stop(), then start() on
 *                 the replacement, in the same tick.
 *   rio_speak     stop() cancelled the CLIP and left the dictation alone, on
 *                 the stated grounds that "a dictation in flight is cancelled
 *                 by the arbiter's own pre-emption of the item that owns it".
 *                 The arbiter pre-empts by calling that very function. Nobody
 *                 was cancelling it.
 *   rio_realtime  one dictation slot, and `speak()` refuses `busy` while it is
 *                 held. The depart call's budget is 2500 ms, so the mouth was
 *                 held for up to two and a half seconds by a line that had
 *                 already been superseded.
 *
 * A near call has no pre-rendered clip -- it names a road, and only the four
 * roadless junction sentences are rendered to disk -- so `busy` with nothing
 * underneath it is silence.
 *
 * WHAT THIS ASSERTS
 * -----------------
 *   1. The replacement line is SPOKEN. Same group, same tick, real arbiter,
 *      real rio_speak, real controller.
 *   2. The superseded line's dictation is cancelled BY ID, so the cancel
 *      cannot land on the line that replaced it.
 *   3. The mouth still works AFTERWARDS -- the junction call two lines later
 *      is dictated, not refused.
 *   4. A turn call pre-empting a spoken conversational answer cancels that
 *      answer's out-of-band response and does NOT fire a recovery
 *      `response.create` underneath the warning.
 *   5. A refusal that does happen is REPORTED, naming the line that held the
 *      mouth -- the drive log's version of this failure was a silence.
 */
'use strict';

const path = require('path');
const REPO = path.join(__dirname, '..');
const STATIC = path.join(REPO, 'static');

let checks = 0, failures = 0;
function ok(cond, what) {
  checks++;
  if (!cond) failures++;
  console.log((cond ? '  ok    ' : '  FAIL  ') + what);
}
function section(n) { console.log('\n=== ' + n + ' ==='); }
const tick = (ms) => new Promise(r => setTimeout(r, ms || 0));

/* A DOM that plays nothing. Every assertion here is about which path was
   taken, not about sound. */
global.window = global;
global.document = {
  addEventListener() {}, getElementById() { return null; },
  querySelector() { return null; }, querySelectorAll() { return []; },
  createElement() {
    const el = {
      pause() {}, addEventListener() {}, removeEventListener() {}, style: {},
      muted: false, onended: null, onerror: null,
      set src(v) { this._src = v; setTimeout(() => { if (el.onended) el.onended(); }, 0); },
      get src() { return this._src; },
      play() { return Promise.resolve(); },
    };
    return el;
  },
};
if (typeof global.atob !== 'function') {
  global.atob = (b64) => Buffer.from(b64, 'base64').toString('binary');
}
if (typeof global.AbortController !== 'function') {
  global.AbortController = class { constructor() { this.signal = {}; } abort() {} };
}

const speech = require(path.join(STATIC, 'rio_speech.js'));
const rt = require(path.join(STATIC, 'rio_realtime.js'));
require(path.join(STATIC, 'rio_speak.js'));

/* THE SESSION, DRIVEN WITHOUT A MICROPHONE.
 *
 * The real controller over a transport that records what was sent and can be
 * fed events back. `respond()` is the server: it answers a `response.create`
 * with a `response.created`, which is the moment a dictation stops being
 * unbound -- and the moment the whole orphan/by-id distinction turns on. */
function session(opts) {
  opts = opts || {};
  const arbiter = speech.makeArbiter();
  const sent = [];
  const events = [];
  const audio = { muted: false };
  const controller = rt.createController({
    arbiter: arbiter,
    send: (obj) => sent.push(obj),
    tool: () => Promise.resolve({ ok: true }),
    audio: { mute: () => { audio.muted = true; },
             unmute: () => { audio.muted = false; } },
    onEvent: (ev) => events.push(ev),
    bargeSustainMs: 4, bargeConfirmMs: 8,
    /* THE TAIL, WHEN A TEST ASKS FOR IT. With this off, `response.done` and
       the end of the sound are the same event and a test cannot tell which
       one a gate is watching -- which is precisely the distinction the
       route-start section below exists to pin down. */
    holdTail: !!opts.holdTail,
    tailFallbackMs: opts.tailFallbackMs || 15000,
  });
  let n = 0;
  const h = {
    arbiter, sent, events, audio, controller,
    creates: () => sent.filter(e => e.type === 'response.create'),
    cancels: () => sent.filter(e => e.type === 'response.cancel'),
    evTypes: () => events.map(e => e.type),
    ev: (t) => events.filter(e => e.type === t),
    /* The server answering the most recent create that has not been answered.
       Returns the id it assigned. */
    respond() {
      const id = 'resp_' + (++n);
      controller.handle({ type: 'response.created', response: { id: id } });
      return id;
    },
    /* ...and the audio for it, which is what makes a dictation "started". */
    speaks(id, words) {
      controller.handle({ type: 'response.output_audio_transcript.delta',
                          response_id: id, delta: words || 'x' });
    },
    done(id) {
      controller.handle({ type: 'response.done',
                          response: { id: id, status: 'completed' } });
    },
    // rio_speak looks the session up through this.
    install() {
      global.RIO.realtime = { active: () => ({
        speak: (t, o) => controller.speak(t, o),
        cancelSpeak: (tok, why) => controller.cancelSpeak(tok, why),
        speechEnabled: () => true,
        speakTimeout: (ch, ct) => (opts.budget || 2500),
      }) };
    },
  };
  h.install();
  return h;
}

/* One nav line, exactly as rio_navplan builds it: an arbiter item in the
   maneuver's group whose play/stop are rio_speak's. */
function navLine(h, o) {
  const src = RIO.speak.provider({
    text: o.text, channel: 'nav', callType: o.callType,
    clipUrl: o.clipUrl || null, clipFirst: !!o.clipUrl,
    element: document.createElement('audio'),
  });
  /* THE SAME TWO LINES rio_navplan.js USES, and they have to stay the same
     two lines: the junction call is the one tier that may cut through, and
     everything else is patient. A copy of that rule that drifts would let this
     file pass while the car interrupts her. */
  h.arbiter.say({
    priority: o.callType === 'junction' ? h.arbiter.P.TURN_NEAR : h.arbiter.P.NAV,
    patient: o.callType !== 'junction',
    group: 'nav:' + (o.maneuver || 'm0'),
    id: 'nav:' + (o.maneuver || 'm0') + ':' + o.callType,
    text: o.text, ttlMs: o.ttlMs || 12000,
    play: src.play, stop: src.stop,
    onDone: (reason) => { (o.done || (() => {}))(reason); },
  });
  return src;
}

(async function main() {

  section('the drive of 2026-09-17 — a depart call superseded by its own near call');
  {
    const h = session();
    RIO.speak.reset();
    const reasons = {};
    const silences = [];
    RIO.noteSilence = (info) => silences.push(info);

    // 18:35:35.197 — the depart call takes the mouth and is dictated.
    navLine(h, { text: 'Head east, then turn left onto Palisades Dr.',
                 callType: 'depart',
                 done: (r) => { reasons.depart = r; } });
    const departCreate = h.creates().length;
    ok(departCreate === 1, 'the depart call asks the session to dictate it');
    const departId = h.respond();          // the server binds it to a response

    // 18:35:35.223 — 26 ms later, the near call for the SAME maneuver.
    await tick(26);
    navLine(h, { text: 'Turn left onto Palisades Dr.', callType: 'near',
                 done: (r) => { reasons.near = r; } });

    ok(reasons.depart === 'superseded',
       'the near call supersedes the depart call — same maneuver, same group');

    /* (2) BY ID. A bare cancel would cancel whatever is active a moment later,
       which is the line that replaced this one. */
    const byId = h.cancels().filter(e => e.response_id === departId);
    ok(byId.length === 1,
       'the superseded dictation is cancelled BY ID, not by "whatever is active"');
    ok(h.sent.some(e => e.type === 'output_audio_buffer.clear'),
       'and the audio already queued for it is thrown away');

    /* (1) THE POINT. The replacement line reaches the mouth. */
    ok(h.creates().length === 2,
       'the near call is dictated — the mouth was given back in the same tick '
       + `(${h.creates().length} dictations asked for)`);
    const nearId = h.respond();
    h.speaks(nearId, 'Turn left onto Palisades Dr.');
    h.done(nearId);
    await tick(5);

    ok(silences.length === 0,
       'nothing is recorded as silent — on the drive this was '
       + 'VOICE_SILENT {reason:"busy", detail:"no_clip"}');
    ok(RIO.speak.stats().silent === 0,
       `and the silent tier stays at zero (${RIO.speak.stats().silent})`);
    ok(RIO.speak.stats().dictated >= 1,
       `the line came out of her own mouth (dictated=${RIO.speak.stats().dictated})`);

    /* (3) AND THE MOUTH STILL WORKS. On the drive the junction call a minute
       later did get said -- but only because it has a clip. A line with no
       clip had to survive too. */
    const before = h.creates().length;
    navLine(h, { text: 'Turn left onto Bowers Avenue.', callType: 'near',
                 maneuver: 'm1', done: (r) => { reasons.next = r; } });
    ok(h.creates().length === before + 1,
       'a later line on a different maneuver is dictated too — the session was '
       + 'not left mute behind a slot nobody released');
    ok(RIO.speak.stats().silent === 0,
       'and it is not silent either');
  }

  section('the refusal that remains — reported, with what was holding the mouth');
  {
    const h = session();
    RIO.speak.reset();
    const silences = [];
    RIO.noteSilence = (info) => silences.push(info);

    // Two lines in DIFFERENT groups: nothing supersedes, so the second one
    // genuinely has to wait, and the arbiter queues it. What must never happen
    // again is that refusal being invisible.
    h.controller.speak('You are too close.', { channel: 'headway' });
    h.respond();
    const p = h.controller.speak('Turn left onto Palisades Dr.',
                                 { channel: 'nav', callType: 'near' });
    let why = null;
    p.catch(e => { why = e && e.message; });
    await tick(5);
    ok(why === 'busy', 'a second deterministic line is still refused — one mouth');
    const ref = h.ev('LIVE_DICTATION_REFUSED');
    ok(ref.length === 1,
       'and the refusal is now an event rather than a silence in the log');
    ok(ref[0] && ref[0].holder === 'You are too close.',
       `naming the line that held the mouth (${ref[0] && JSON.stringify(ref[0].holder)})`);
    ok(ref[0] && typeof ref[0].holder_age_ms === 'number',
       'and how long it had held it');
  }

  section('a turn call over a conversational answer — interrupts, does not orphan');
  {
    const h = session();
    // She answers a question about the road directly: a real out-of-band
    // response under speech-to-speech, claiming the mouth at CONVO.
    const spoke = h.controller.speakDirect('A red pickup, two cars ahead.',
                                           { path: 'observer_direct' });
    ok(spoke && typeof spoke.then === 'function',
       'the vetted answer takes the mouth and is injected');
    const directId = h.respond();
    h.speaks(directId, 'A red pickup,');
    const cancelsBefore = h.cancels().length;
    const createsBefore = h.creates().length;

    // ...and the junction arrives.
    navLine(h, { text: 'Turn left.', callType: 'junction', maneuver: 'm2' });

    ok(h.cancels().some(e => e.response_id === directId),
       'the turn call cancels the answer it pre-empted, BY ID — under '
       + 'speech-to-speech that answer is a real response, and skipping the '
       + 'cancel left the model talking underneath the warning');
    ok(h.cancels().length > cancelsBefore, 'so a cancel was actually sent');
    ok(h.audio.muted === true || h.sent.some(e => e.type === 'output_audio_buffer.clear'),
       'and the audio already on its way is silenced');

    // The recovery create belongs to a line that never got out, not to one
    // that was outranked.
    await tick(30);
    const recovery = h.creates().slice(createsBefore)
      .filter(e => !(e.response && e.response.conversation === 'none'));
    ok(recovery.length === 0,
       'and NO recovery response is asked for — a pre-empted answer must not '
       + 'reappear underneath the warning that displaced it');

    ok(h.creates().some(e => e.response &&
                        /Turn left\./.test(e.response.instructions || '')),
       'the turn call itself is dictated');
  }

  section('a route start mid-sentence — the callout waits for the audio to end');
  {
    /* THE DRIVE, 2026-09-24, session 4989d12e. A route is started while RIO is
       finishing a sentence. 325 ms later the route-start call takes the mouth;
       her answer is logged `orphan_silenced` and the driver hears half of it.
       The first maneuver was minutes of driving away.

       THE GATE IS THE SOUND, NOT THE GENERATION, and that is why this runs
       with holdTail on: `response.done` and the end of the audio are seconds
       apart under a speaking backend, and a callout released at the first of
       them still lands on top of her. */
    const h = session({ holdTail: true });
    RIO.speak.reset();

    // An ordinary answer to the driver -- the closing sentence of a turn, which
    // is what a route start lands on top of.
    h.controller.handle({ type: 'input_audio_buffer.speech_started' });
    h.controller.handle({ type: 'input_audio_buffer.speech_stopped' });
    await tick(5);
    const answerId = h.respond();
    h.controller.handle({ type: 'output_audio_buffer.started',
                          response_id: answerId });
    h.speaks(answerId, 'A red pickup, two cars ahead.');
    ok(h.arbiter.state().speaking
       && h.arbiter.state().speaking.priority === speech.P.CONVO,
       'she has the mouth, out loud, at CONVO');

    let departReason = null;
    navLine(h, { text: 'Head east, then turn left onto Palisades Dr.',
                 callType: 'depart',
                 done: (r) => { departReason = r; } });

    const speakingNow = h.arbiter.state().speaking;
    ok(speakingNow && speakingNow.priority === speech.P.CONVO,
       'the route-start call does NOT take the mouth — this is the assertion '
       + 'that fails if a callout begins while audio is still playing');
    ok(h.arbiter.state().queued.some(i => /:depart$/.test(i.id)),
       'it is queued behind her instead');
    ok(departReason === null, 'and it has not been dropped');
    ok(!h.cancels().some(e => e.response_id === answerId),
       'her answer is not cancelled — nothing pre-empted it');

    // GENERATION ENDS. The sound has not.
    h.done(answerId);
    await tick(5);
    const afterDone = h.arbiter.state().speaking;
    ok(afterDone && afterDone.priority === speech.P.CONVO,
       'response.done does not release the mouth — generation is over, the '
       + 'sound is not');
    ok(h.arbiter.state().queued.some(i => /:depart$/.test(i.id)),
       '...so the callout is still waiting');

    // ...and now it has.
    h.controller.handle({ type: 'output_audio_buffer.stopped',
                          response_id: answerId });
    await tick(10);
    const afterAudio = h.arbiter.state().speaking;
    ok(afterAudio && /:depart$/.test(afterAudio.id),
       'the route-start call begins when the AUDIO ends, on the same event '
       + 'the mouth already waits for');
  }

  section('...and the junction call still cuts straight through her');
  {
    /* THE OTHER SIDE OF THE LINE, asserted in the same file so the two cannot
       drift apart. A junction call is 35 m out with a 2.5 s TTL: made patient
       it would wait behind a sentence, outlive its window and be dropped,
       which is the one nav line that must never be lost. */
    const h = session({ holdTail: true });
    RIO.speak.reset();
    h.controller.handle({ type: 'input_audio_buffer.speech_started' });
    h.controller.handle({ type: 'input_audio_buffer.speech_stopped' });
    await tick(5);
    const answerId = h.respond();
    h.controller.handle({ type: 'output_audio_buffer.started',
                          response_id: answerId });
    h.speaks(answerId, 'A red pickup, two cars ahead.');

    navLine(h, { text: 'Turn left.', callType: 'junction', maneuver: 'm2' });
    const nowSpeaking = h.arbiter.state().speaking;
    ok(nowSpeaking && /:junction$/.test(nowSpeaking.id),
       'the junction call takes the mouth immediately, mid-sentence');
    /* BARE, NOT BY ID, and that is right here: this answer IS the response the
       session is generating, so `response.cancel` naming nothing cancels
       exactly it. The by-id form belongs to the out-of-band direct line, which
       the section above this one covers -- there the entry can already be over
       and an unnamed cancel would land on its successor. */
    ok(h.cancels().length > 0,
       'and cancels the answer it pre-empted');
    ok(h.sent.some(e => e.type === 'output_audio_buffer.clear'),
       'and clears the audio already on its way, so she does not carry on '
       + 'underneath it');
  }

  section('turn end to first audio — the number the drive log did not have');
  {
    const h = session();
    h.controller.handle({ type: 'input_audio_buffer.speech_started' });
    h.controller.handle({ type: 'input_audio_buffer.speech_stopped' });
    await tick(20);
    const id = h.respond();
    h.speaks(id, 'It is');
    h.speaks(id, ' a torii gate.');
    const spoke = h.ev('LIVE_SPOKE');
    ok(spoke.length === 1,
       `one event per response, on the first audio (${spoke.length})`);
    ok(spoke[0] && spoke[0].turn_kind === 'conversation',
       'classified as a conversational answer');
    /* A floor well under the 20 ms slept, not the 20 itself: setTimeout is
       allowed to fire a millisecond early and Date.now() is coarse on some
       hosts, and a test that fails once in fifty for that reason teaches
       people to re-run it rather than to read it. What is being asserted is
       that the clock runs from the END OF THE TURN and not from the response
       opening -- at zero it would be measuring nothing. */
    ok(spoke[0] && spoke[0].wait_ms >= 10,
       `carrying the wait from turn end to first audio (${spoke[0] && spoke[0].wait_ms} ms)`);
    ok(spoke[0] && spoke[0].create_ms !== null
       && spoke[0].wait_ms > spoke[0].create_ms,
       'and the driver waited longer than the session did — the two clocks are '
       + 'measured from different moments');

    // A dictated line answers nobody, so it is not judged on that clock.
    const h2 = session();
    h2.controller.speak('Turn left.', { channel: 'nav', callType: 'junction' });
    const nid = h2.respond();
    h2.speaks(nid, 'Turn left.');
    const s2 = h2.ev('LIVE_SPOKE');
    ok(s2.length === 1 && s2[0].turn_kind === 'dictation',
       'a dictated line is reported as a dictation');
    ok(s2[0] && s2[0].wait_ms === null,
       'with no driver wait — a turn call is not late because nobody asked');
    ok(s2[0] && typeof s2[0].asked_ms === 'number',
       `and its own budget clock instead (asked_ms=${s2[0] && s2[0].asked_ms})`);
    ok(s2[0] && s2[0].channel === 'nav',
       'named by the channel it belongs to');
  }

  console.log(failures ? `\nFAILED ${failures}/${checks} checks`
                       : `\nPASSED ${checks} checks`);
  process.exit(failures ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
