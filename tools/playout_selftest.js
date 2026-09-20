/* playout_selftest.js — the tail, and a test that can prove it would notice.
 *
 *   node tools/playout_selftest.js
 *
 * WHY THIS FILE IS SHAPED UNUSUALLY.
 *
 * Handing the mouth back too early does not throw, does not log, and does not
 * fail a test written optimistically. It sounds like RIO cutting herself off
 * mid-sentence, and it cost two drives to find the first time (commit a05d233:
 * "the mouth was handed back seconds before the sound stopped, and every mute in
 * that window had no owner"). On WebRTC the API told us when the sound stopped.
 * On a WebSocket it does not, so static/rio_playout.js computes it -- and a
 * computed answer needs a test that would go red if the computation were wrong,
 * not one that goes green because it is right.
 *
 * So section Z runs a DELIBERATELY WRONG playout -- one that ends the audio at
 * generation-done, which is exactly the bug -- through the same assertions, and
 * FAILS THE SUITE IF THEY ALL PASS. A test that has never been seen to fail is a
 * test nobody has any evidence about.
 *
 * The contract, stated as an inequality because an inequality can be violated on
 * purpose:
 *
 *      audio-end for a response must not fire before its last sample is
 *      scheduled to finish.
 *
 * The grace in rio_playout is added AFTER that instant and never subtracted, so
 * the queue may be late and may not be early. Both directions are asserted.
 */
'use strict';

const path = require('path');
const playout = require(path.join(__dirname, '..', 'static', 'rio_playout.js'));

let checks = 0, failures = 0;
function ok(cond, what) {
  checks++;
  if (!cond) { failures++; console.log('  FAIL  ' + what); }
  else console.log('  ok    ' + what);
}
function section(name) { console.log('\n=== ' + name + ' ==='); }

const RATE = 24000;
/* One second of PCM16 at 24 kHz, mono. Named because every duration below is
   expressed in these rather than in bytes, and bytes are where an off-by-two
   hides. */
const ONE_SECOND = RATE * 2;

/* A clock a test owns. Seconds, monotonic, moves only when told. */
function clock(start) {
  let t = start === undefined ? 1000 : start;
  return { now: () => t, advance: (s) => { t += s; }, set: (s) => { t = s; } };
}

/* The harness: a playout plus the ledger of what it announced. */
function harness(make, opts) {
  opts = opts || {};
  const c = clock();
  const ends = [];
  const starts = [];
  const p = (make || playout.createPlayout)({
    sampleRate: RATE,
    now: c.now,
    tailGraceS: opts.tailGraceS === undefined ? 0 : opts.tailGraceS,
    onAudioEnd: (rid, d) => ends.push(Object.assign({ rid }, d)),
    onAudioStart: (rid) => starts.push(rid),
  });
  return { c, p, ends, starts };
}

/* ------------------------------------------------------------------------
   THE ASSERTIONS, AS A FUNCTION, so the same ones can be pointed at a correct
   implementation and at a broken one. This is the whole trick of the file: if
   they only ever run against the good version, nobody knows what they catch.
   ------------------------------------------------------------------------ */
function tailAssertions(make, label, report) {
  const say = report || ok;
  const h = harness(make);

  // Three seconds of audio, handed over in one go.
  h.p.push('r1', ONE_SECOND * 3);
  // ...and generation ends IMMEDIATELY, which is the real shape: response.done
  // arrives while seconds of sound are still queued.
  h.p.generationDone('r1');
  h.p.tick();
  say(h.ends.length === 0,
      `${label}: at response.done with 3 s still queued, the sound is NOT over`);

  h.c.advance(1.5);
  h.p.tick();
  say(h.ends.length === 0,
      `${label}: halfway through, still not over`);

  h.c.advance(1.4);
  h.p.tick();
  say(h.ends.length === 0,
      `${label}: 0.1 s before the end, still not over`);

  h.c.advance(0.2);
  h.p.tick();
  say(h.ends.length === 1,
      `${label}: past the end, over exactly once`);
  say(h.ends.length === 1 && h.ends[0].reason === 'drained',
      `${label}: ...and it ended by draining, not by being cancelled`);

  h.p.tick(); h.p.tick();
  say(h.ends.length === 1,
      `${label}: ticking again does not announce it twice`);
}

async function main() {

// ---------------------------------------------------------------------------
section('the arithmetic, before anything depends on it');
// ---------------------------------------------------------------------------
{
  ok(playout.pcm16Seconds(ONE_SECOND, RATE, 1) === 1,
     'one second of mono PCM16 at 24 kHz is one second');
  ok(playout.pcm16Seconds(ONE_SECOND, RATE, 2) === 0.5,
     'the same bytes in stereo are half as long — channels are not assumed');
  ok(playout.pcm16Seconds(0, RATE, 1) === 0, 'no bytes is no time');
}

// ---------------------------------------------------------------------------
section('the tail is held until the sound is actually over');
// ---------------------------------------------------------------------------
tailAssertions(null, 'correct');

// ---------------------------------------------------------------------------
section('audio arriving slower than real time does not run the clock ahead');
// ---------------------------------------------------------------------------
{
  /* THE UNDERRUN, which is the case a naive "end = start + total duration" gets
     wrong in the dangerous direction. Deltas arriving slower than they play mean
     the queue drains between chunks; the next chunk must start at NOW, not
     extend an end that is already in the past. Getting this wrong computes an
     end EARLIER than the truth, which is the bug this file exists for. */
  const h = harness();
  h.p.push('r1', ONE_SECOND);          // 1 s of audio at t=0
  h.c.advance(3);                      // ...and nothing for 3 s: it has drained
  h.p.push('r1', ONE_SECOND);          // another second, starting NOW
  h.p.generationDone('r1');
  h.p.tick();
  ok(h.ends.length === 0, 'the second chunk has not finished yet');
  h.c.advance(0.9);
  h.p.tick();
  ok(h.ends.length === 0, '...still playing 0.9 s in');
  h.c.advance(0.2);
  h.p.tick();
  ok(h.ends.length === 1, '...and over at 1.1 s after the last chunk, not at 2 s');
  ok(h.ends[0].bytes === ONE_SECOND * 2, 'both chunks are in the ledger');
}

// ---------------------------------------------------------------------------
section('audio arriving FASTER than real time queues rather than overlaps');
// ---------------------------------------------------------------------------
{
  const h = harness();
  for (let i = 0; i < 5; i++) h.p.push('r1', ONE_SECOND);   // 5 s, instantly
  h.p.generationDone('r1');
  h.p.tick();
  ok(h.ends.length === 0, 'five seconds handed over at once is five seconds of sound');
  h.c.advance(4.9);
  h.p.tick();
  ok(h.ends.length === 0, '...not over at 4.9 s');
  h.c.advance(0.2);
  h.p.tick();
  ok(h.ends.length === 1, '...over at 5.1 s');
}

// ---------------------------------------------------------------------------
section('the grace may make it late and must never make it early');
// ---------------------------------------------------------------------------
{
  const h = harness(null, { tailGraceS: 0.25 });
  h.p.push('r1', ONE_SECOND);
  h.p.generationDone('r1');
  h.c.advance(1.0);
  h.p.tick();
  ok(h.ends.length === 0,
     'at exactly the computed end, with a grace, it is not yet called over');
  h.c.advance(0.2);
  h.p.tick();
  ok(h.ends.length === 0, '...nor part-way through the grace');
  h.c.advance(0.1);
  h.p.tick();
  ok(h.ends.length === 1, '...and over once the grace has passed');
  ok(h.ends[0].at >= h.ends[0].scheduled_end,
     'the announced end is never earlier than the computed one — the asymmetry '
     + 'is the whole design');
}

// ---------------------------------------------------------------------------
section('still generating is never over, however long the silence');
// ---------------------------------------------------------------------------
{
  const h = harness();
  h.p.push('r1', ONE_SECOND);
  h.c.advance(60);
  h.p.tick();
  ok(h.ends.length === 0,
     'a minute past the audio, with no response.done, the response is NOT over '
     + '— more audio may still come, and calling it over would hand the mouth '
     + 'back mid-answer');
  h.p.generationDone('r1');
  h.p.tick();
  ok(h.ends.length === 1, '...and over the instant generation finishes');
}

// ---------------------------------------------------------------------------
section('a barge-in ends it now, and says it was cancelled');
// ---------------------------------------------------------------------------
{
  const h = harness();
  h.p.push('r1', ONE_SECOND * 5);
  h.c.advance(1);
  const stopped = h.p.flush('r1', 'barge_in');
  ok(stopped.length === 1 && h.ends.length === 1, 'the sound stops at once');
  ok(h.ends[0].reason === 'cancelled:barge_in',
     'and it is recorded as cancelled, not as drained — the mute ledger tells '
     + 'those apart and so must this');
  h.c.advance(10);
  h.p.tick();
  ok(h.ends.length === 1, 'and the audio it never played does not end it again');
  ok(h.p.idle() === true, 'the mouth is free immediately, with no round trip');
}

// ---------------------------------------------------------------------------
section('the gate on response.create — their docs describe our code');
// ---------------------------------------------------------------------------
{
  /* "If your client immediately sends conversation.item.create (with the
     function result) followed by response.create, the server starts generating
     the next response right away — even if the client is still playing audio
     from the previous turn. This causes overlapping audio." */
  const h = harness();
  h.p.push('r1', ONE_SECOND * 2);
  h.p.generationDone('r1');
  ok(h.p.idle() === false,
     'with two seconds still to play, the next request must NOT go out');
  ok(h.p.untilIdle() > 1.9 && h.p.untilIdle() <= 2.0,
     `and the caller is told how long to wait (${h.p.untilIdle().toFixed(2)} s)`);
  h.c.advance(2.1);
  h.p.tick();
  ok(h.p.idle() === true, 'once the sound is over, it may');
  ok(h.p.untilIdle() === 0, 'and the wait is zero');
}
{
  const h = harness();
  h.p.push('r1', ONE_SECOND);
  ok(h.p.untilIdle() === Infinity,
     'a response still GENERATING reports an unknown wait rather than a short '
     + 'one — a caller must read that as "ask again", never as "never"');
}

// ---------------------------------------------------------------------------
section('two responses do not end each other');
// ---------------------------------------------------------------------------
{
  const h = harness();
  h.p.push('r1', ONE_SECOND);
  h.p.generationDone('r1');
  h.p.push('r2', ONE_SECOND * 4);
  h.p.generationDone('r2');
  h.c.advance(1.1);
  h.p.tick();
  ok(h.ends.length === 1 && h.ends[0].rid === 'r1',
     'the short one ends on its own schedule');
  ok(h.p.idle() === false, 'and the long one still holds the mouth');
  h.c.advance(4);
  h.p.tick();
  ok(h.ends.length === 2, 'then it ends too');
}

// ---------------------------------------------------------------------------
section('Z. THE MUTATION — the same assertions, against the actual bug');
// ---------------------------------------------------------------------------
{
  /* A playout that ends the audio when GENERATION ends. That is not a strawman:
     it is precisely what rio_realtime.js did before commit a05d233, and
     precisely what a port to WebSocket would do by accident if it treated
     response.done as the end of the sound.
     If the assertions above cannot tell this apart from the real thing, they are
     decoration. */
  function brokenPlayout(o) {
    const real = playout.createPlayout(o);
    return Object.assign({}, real, {
      generationDone: function (rid) {
        const r = real.generationDone(rid);
        real.flush(rid, 'generation_done');   // <-- the bug, in one line
        return r;
      },
    });
  }

  let caught = 0, ran = 0;
  tailAssertions(brokenPlayout, 'BROKEN', (cond, what) => {
    ran++;
    if (!cond) caught++;
  });

  ok(ran > 0, `the mutation ran the same ${ran} assertions`);
  ok(caught > 0,
     `and ${caught} of ${ran} FAILED against it — the assertions have teeth`);
  /* The specific one that matters most: the first, which is the moment the bug
     happens. If every later assertion caught it but that one did not, the suite
     would be finding the damage rather than the cause. */
  const h = harness(brokenPlayout);
  const t0 = h.c.now();
  h.p.push('r1', ONE_SECOND * 3);
  h.p.generationDone('r1');
  h.p.tick();
  ok(h.ends.length > 0,
     'the broken one really does end the sound at response.done — which is what '
     + 'makes it the right mutation to test against');
  /* AGAINST THE AUDIO, NOT AGAINST THE FIELD. The first version of this compared
     the announced end with `scheduled_end`, which flush() clamps down to now —
     so it was asking the bug whether it had happened, using a number the bug had
     already moved. Three seconds of PCM went in; the only honest question is
     whether the end was announced before three seconds of it could have been
     heard. */
  ok(h.ends.length > 0 && (h.ends[0].at - t0) < 3,
     `and it announces the end ${(3 - (h.ends[0].at - t0)).toFixed(1)} s before `
     + 'three seconds of audio could have been heard — the inequality this whole '
     + 'file exists to defend');
}

console.log(`\n${checks - failures}/${checks} checks passed`);
if (failures) { console.log(`${failures} FAILED`); process.exit(1); }
}

main().catch((e) => { console.error(e); process.exit(1); });
