/* page_background_selftest.js — the page goes away, and the session survives it.
 *
 *   node tools/page_background_selftest.js
 *
 * THE DRIVE. 2026-09-17, session 0233da0d. At 18:36:37.7 a bus_health report
 * arrived twenty seconds after the previous one, on a cadence of thirty: the
 * forced closing report stopBusWatch() posts. At 18:36:39.8, FRAMES_HIDDEN.
 * The page then spent 118 s hidden and RUNNING -- twelve heartbeats at the
 * ten-second cadence, 1319 frames, GPS re-arming every two seconds -- with
 * the camera going black at 25 s and the voice session already gone.
 *
 * Not one of those sentences could be read off the log directly. The session
 * end was timing on an unrelated report; its cause was nowhere; the audio
 * session's state at the moment the page went away was nowhere; whether the
 * wake lock had ever been granted was nowhere. This file pins the record
 * that replaces that, and the one behaviour change that came with it.
 *
 * WHAT THIS ASSERTS
 * -----------------
 *   1. A peer connection that says `disconnected` and then `connected` inside
 *      the grace is NOT a lost session. That drive could not say whether this
 *      ever fired; the spec says the state is transient; the page treated it
 *      as death.
 *   2. One that stays `disconnected` past the grace IS lost, with the reason
 *      it always had -- and `failed` / `closed` are lost at once.
 *   3. Every transition is an event, so the next drive can say.
 *   4. The page reports every way a session ends, with the talk button
 *      recorded BEFORE stop() runs, and tears a dead session down instead of
 *      listening on it.
 *   5. The server keeps the fields all of this is made of.
 */
'use strict';

const fs = require('fs');
const path = require('path');
const STATIC = path.join(__dirname, '..', 'static');

let checks = 0, failures = 0;
function ok(cond, what) {
  checks++;
  if (!cond) failures++;
  console.log((cond ? '  ok    ' : '  FAIL  ') + what);
}
function section(n) { console.log('\n=== ' + n + ' ==='); }
const tick = (ms) => new Promise(r => setTimeout(r, ms || 0));

if (typeof global.atob !== 'function') {
  global.atob = (b64) => Buffer.from(b64, 'base64').toString('binary');
}
if (typeof global.AbortController !== 'function') {
  global.AbortController = class { constructor() { this.signal = {}; } abort() {} };
}
const rt = require(path.join(STATIC, 'rio_realtime.js'));

function watch(graceMs) {
  const events = [];
  let lost = null;
  const w = rt.peerWatch({
    graceMs, ice: () => 'checking',
    emit: (ev) => events.push(ev),
    onLost: (why) => { lost = why; },
  });
  return { w, events, lost: () => lost,
           states: () => events.map(e => e.state + (e.recovered ? '+' : e.expired ? '!' : '')) };
}

(async function main() {

  section('a tunnel — disconnected, then connected inside the grace');
  {
    const h = watch(40);
    h.w.state('connected');
    h.w.state('disconnected');
    await tick(15);
    h.w.state('connected');
    await tick(60);
    ok(h.lost() === null,
       'the session is NOT declared lost -- `disconnected` is a state a '
       + 'connection passes through, and this page used to tear the drive\'s '
       + 'voice down on the first one');
    ok(h.states().join(',') === 'connected,disconnected,connected+',
       `every transition is an event, and the recovery says so (${h.states().join(',')})`);
    ok(h.events[1].grace_ms === 40,
       'the disconnected event carries the grace it is being given');
  }

  section('a real loss — disconnected past the grace');
  {
    const h = watch(30);
    h.w.state('connected');
    h.w.state('disconnected');
    await tick(60);
    ok(h.lost() === 'peer_disconnected',
       `declared lost with the reason it always had (${h.lost()})`);
    ok(h.states().slice(-1)[0] === 'disconnected!',
       'and the expiry is its own event, so the log says the grace ran out '
       + 'rather than that the state was seen');
    h.w.state('connected');
    ok(h.events.length === 3, 'nothing after a loss is reported -- the session is gone');
  }

  section('failed and closed are final, at once');
  {
    const a = watch(1000);
    a.w.state('failed');
    ok(a.lost() === 'peer_failed', `failed: lost immediately (${a.lost()})`);
    const b = watch(1000);
    b.w.state('disconnected');
    b.w.state('closed');
    ok(b.lost() === 'peer_closed', `closed during a grace: lost immediately (${b.lost()})`);
    await tick(5);
    ok(b.events.filter(e => e.expired).length === 0,
       'and the grace clock is cancelled rather than firing a second loss later');
  }

  section('the page: every session end says why, before it is gone');
  {
    const html = fs.readFileSync(path.join(STATIC, 'index.html'), 'utf8');
    const stopBranch = html.indexOf("reportSessionEnded('talk_button')");
    const stopCall = html.indexOf('live.stop();', stopBranch);
    ok(stopBranch > 0 && stopCall > stopBranch && stopCall - stopBranch < 600,
       'the talk button reports the end BEFORE live.stop() -- the snapshot is '
       + 'of the session as it was, not as stop() leaves it');
    ok(html.indexOf("reportSessionEnded('transport:' + why)") > 0,
       'a lost transport is a session end with the transport\'s own reason');
    const lostHandler = html.indexOf("ev.type === 'LIVE_TRANSPORT_LOST'");
    const tail = html.slice(lostHandler, lostHandler + 900);
    ok(lostHandler > 0 && /stopBusWatch\(\)/.test(tail) && /setMicState\('failed'\)/.test(tail),
       'and the page tears the dead session down -- no more "listening" over '
       + 'a closed channel, no more bus_health for a bus carrying nothing');
    ok(html.indexOf("LIVE_SESSION_ENDED: 'session_ended'") > 0
       && html.indexOf("LIVE_PEER_STATE: 'peer_state'") > 0
       && html.indexOf("LIVE_MIC_STATE: 'mic_state'") > 0
       && html.indexOf("LIVE_AUDIO_INTERRUPTED: 'audio_interrupted'") > 0,
       'all of it is REPORTED, not merely emitted -- the gap every one of '
       + 'today\'s fixes has been about');
    /* EITHER NAME, AND THE REASON THERE ARE TWO. These three checks looked for
     * `pageMark(` and `live.resumeAudio(`, and had been failing since 16f5e95
     * -- the commit whose whole subject is that `pageMark` was not defined where
     * the drive block called it. The fix there was a local `mark()` wrapping
     * `RIO.pageMark` (index.html ~4542), because the two live in different
     * <script> blocks and the outer function is not in scope in the inner one.
     * The behaviour these assert is present and correct; only the call site's
     * spelling moved, and nobody updated the tests.
     *
     * So this suite spent months reporting FAILED 3/32 over working code, which
     * is its own kind of silence: a suite that is always a bit red is one whose
     * red stops being read. tools/suite_sweep.js calls it FAILS HONESTLY and
     * that is exactly what it was doing -- the failure was nobody looking.
     *
     * Matching either helper rather than pinning the new one: both are real, a
     * call site may legitimately use whichever is in scope, and a test that
     * breaks when a wrapper is renamed is the thing being fixed here. */
    const marks = /\b(?:RIO\.)?(?:page)?[Mm]ark\(/.source;
    ok(new RegExp(marks + "'PAGE_VISIBILITY'").test(html)
       && /live_age_s/.test(html) && /wake_lock: !!wakeLock/.test(html),
       'a visibility change carries the live session, its age, the audio '
       + 'state and the wake lock in one line');
    ok(new RegExp(marks + "'WAKE_LOCK', \\{ state: 'denied'").test(html)
       && html.indexOf("state: 'released'") > 0,
       'the wake lock reports denial and release -- a page that goes hidden a '
       + 'minute into a route with no lock is the auto-lock, not a mystery');
    ok(/\.resumeAudio\('visible'\)/.test(html),
       'and coming back to the front replays her element -- Safari does not '
       + 'resume a paused element after an interruption by itself');
  }

  section('the controller: the mouth and the microphone are watched');
  {
    const src = fs.readFileSync(path.join(STATIC, 'rio_realtime.js'), 'utf8');
    ok(/addEventListener\('mute'/.test(src) && /addEventListener\('unmute'/.test(src)
       && /addEventListener\('ended'/.test(src),
       'the capture track\'s mute / unmute / ended are events');
    ok(/element\.addEventListener\('pause'/.test(src)
       && /element\.addEventListener\('playing'/.test(src),
       'and so are her element\'s pause and playing');
    ok(/audioState: function/.test(src) && /resumeAudio: function/.test(src),
       'the handle answers "what is the audio session doing" and "play again"');
    ok(/closed = true;/.test(src) && /if \(closed \|\| !element\.srcObject\) return;/.test(src),
       'a pause caused by stop() is not an interruption');
  }

  section('the server keeps the fields');
  {
    const py = fs.readFileSync(path.join(__dirname, '..', 'realtime.py'), 'utf8');
    for (const k of ['"session_ended"', '"peer_state"', '"mic_state"',
                     '"audio_interrupted"', '"audio_resumed"']) {
      ok(py.indexOf(k) > 0, `kind ${k} is accepted`);
    }
    for (const f of ['"grace_ms"', '"recovered"', '"hidden"', '"audio_state"',
                     '"age_s"', '"ready_state"']) {
      ok(py.indexOf(f) > 0, `field ${f} survives the whitelist`);
    }
    const cfg = fs.readFileSync(path.join(__dirname, '..', 'config.py'), 'utf8');
    ok(/REALTIME_PEER_DISCONNECT_GRACE_MS = \d+/.test(cfg)
       && py.indexOf('peer_disconnect_grace_ms') > 0,
       'the grace is decided in config.py and travels with the session');
  }

  console.log(failures ? `\nFAILED ${failures}/${checks} checks`
                       : `\nPASSED ${checks} checks`);
  process.exit(failures ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
