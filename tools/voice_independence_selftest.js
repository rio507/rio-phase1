/* Speech does not depend on pictures, and no automatic path reaches Ava.
 *
 *     node tools/voice_independence_selftest.js
 *
 * TWO FAILURES FROM THE 2026-09-16 DRIVE, AND THE REMOVAL THAT FOLLOWED.
 *
 * A. NAV SPOKE IN THE WRONG VOICE, UNDER HER. A route was set and the turn
 *    calls came out of ElevenLabs while RIO was still speaking. The chain used
 *    to be dictation -> synthesiser -> clip, and the middle tier is a second
 *    MOUTH: it can talk over the first one. It is gone. The chain is now
 *    dictation -> clip -> silence, logged.
 *
 * B. A FEED INTERRUPTION CUT THE VOICE. The drive log says the two are
 *    siblings rather than cause and effect -- the page froze for ~5 s, which
 *    stopped frames being SENT (ws_open:true, inflight:1) and starved the
 *    audio pipeline in the same instant. Nothing in the frame path touches the
 *    speech path. This suite pins that, because "they were never coupled" is a
 *    claim that has to survive the next person adding a convenience.
 *
 * Node with a fake page, not a browser: what is being asserted is which
 * functions get called and which do not, and in a browser that has a working
 * session and a working camera, none of the interesting branches run.
 *
 * Exit code is the number of failures.
 */
'use strict';

const path = require('path');

let fails = 0;
/* A VERDICT WITH NO TOTAL CANNOT TELL YOU IT RAN. This file counted failures
   only, so "PASS: 0 failure(s)" was printed by a suite that asserted 27 things
   and would have been printed identically by one that asserted none -- which is
   exactly how 23f4185 and source_selftest.js stayed hidden. tools/suite_sweep.js
   flags it as NO COUNT. */
let checks = 0;
function ok(name, cond, extra) {
  checks++;
  if (cond) console.log('  ok   ' + name);
  else { fails++; console.log('  FAIL ' + name + (extra !== undefined ? '  ' + extra : '')); }
}
function section(t) { console.log('\n== ' + t); }
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

// ---------------------------------------------------------------------------
// A fake cabin: every mouth that opens is recorded, with overlap detection.
// ---------------------------------------------------------------------------
function cabin() {
  const open = [];
  const all = [];
  let overlap = false;
  return {
    begin(who, what) {
      const e = { who, what, t: Date.now() };
      if (open.length) overlap = true;
      open.push(e); all.push(e);
      return e;
    },
    end(e) { const i = open.indexOf(e); if (i >= 0) open.splice(i, 1); },
    mouths() { return all.map((e) => e.who); },
    overlapped() { return overlap; },
  };
}

function element(room) {
  const el = {
    src: '', muted: true, played: 0, onended: null, onerror: null, entry: null,
    play() { el.played++; el.entry = room.begin('clip', el.src); return Promise.resolve(); },
    pause() {},
    finish() { if (el.onended) { room.end(el.entry); el.onended(); } },
  };
  return el;
}

/* A live session that speaks when asked. `speechEnabled` is the gate that
   actually failed on the drive -- a session WAS open (the log shows it running
   either side of the nav calls), so `no_session` was not the trigger. */
function session(room, opts) {
  opts = opts || {};
  return {
    speak(text, o) {
      if (opts.refuse) return Promise.reject(new Error(opts.refuse));
      const e = room.begin('session', text);
      if (o && o.onStart) o.onStart();
      return sleep(3).then(() => { room.end(e); });
    },
    speechEnabled() { return opts.channelOff ? false : true; },
    speakTimeout() { return opts.timeoutMs || 500; },
  };
}

/* rio_speak.js attaches to `window` or `globalThis` and also sets
   module.exports, so the export is what to hold and globalThis.RIO is where
   it looks for its siblings (RIO.realtime, RIO.noteSilence). Reloaded per
   case so the counters start clean, which is what the tts===0 assertions
   depend on. */
function loadSpeak() {
  delete require.cache[require.resolve(
    path.join(__dirname, '..', 'static', 'rio_speak.js'))];
  globalThis.RIO = {};
  const speak = require(path.join(__dirname, '..', 'static', 'rio_speak.js'));
  return speak;
}

// Every fetch the page makes, so "no ElevenLabs request on any path" is a
// measurement rather than a belief.
function watchFetch() {
  const calls = [];
  global.fetch = function (url) {
    calls.push(String(url));
    return Promise.resolve({ ok: true, arrayBuffer: () => Promise.resolve(new ArrayBuffer(8)) });
  };
  return calls;
}

async function main() {
  // -------------------------------------------------------------------------
  section('A. a nav drive with NO conversation open');
  // -------------------------------------------------------------------------
  /* The exact shape of the reported failure: route set, nobody talking to her.
     It used to synthesise. It must now use the clip, in her voice, alone. */
  {
    const speak = loadSpeak();
    const room = cabin();
    const calls = watchFetch();
    globalThis.RIO.realtime = { active: () => null };   // no session at all
    const el = element(room);
    const p = speak.provider({
      text: 'Turn left onto Palisades Dr.', channel: 'nav', callType: 'near',
      clipUrl: '/static/audio/left_here.mp3', clipElement: el, element: el,
      ttsUrl: '/nav/voice?route=1&m=m0',   // a caller that was not updated
    });
    const done = p.play();
    for (let i = 0; i < 40 && !el.onended; i++) await sleep(2);
    el.finish();
    await done;

    ok('the callout is spoken', room.mouths().length === 1, room.mouths());
    ok('...in the configured voice, from the clip',
       room.mouths()[0] === 'clip', room.mouths());
    ok('...and NOTHING was requested from the synthesiser',
       calls.filter((u) => /voice|tts|eleven/i.test(u)).length === 0, calls);
    ok('...with no overlap', !room.overlapped());
    ok('the tts counter never moved', speak.stats().tts === 0);
    ok('and nothing doubled', speak.stats().doubled === 0);
  }

  // -------------------------------------------------------------------------
  section('B. the same, with a session open but the channel not dictated');
  // -------------------------------------------------------------------------
  /* The gate that actually failed on the drive. The log shows the live session
     running either side of the nav calls, so the trigger was not "no session"
     -- it was this one, and it used to land on the synthesiser too. */
  {
    const speak = loadSpeak();
    const room = cabin();
    const calls = watchFetch();
    const h = session(room, { channelOff: true });
    globalThis.RIO.realtime = { active: () => h };
    const el = element(room);
    const silences = [];
    const p = speak.provider({
      text: 'Head east, then turn left', channel: 'nav', callType: 'depart',
      element: el, ttsUrl: '/nav/voice?route=1&m=m0',
      onSilent: (s) => silences.push(s),
    });
    await p.play().catch(() => {});
    ok('a session that will not dictate this channel does NOT synthesise',
       calls.filter((u) => /voice|tts|eleven/i.test(u)).length === 0, calls);
    ok('nothing spoke', room.mouths().length === 0, room.mouths());
    ok('and the silence is logged with a reason',
       silences.length === 1, JSON.stringify(silences));
  }

  // -------------------------------------------------------------------------
  section('C. every failure branch, swept — none reaches a second voice');
  // -------------------------------------------------------------------------
  {
    const branches = [
      ['no session', { live: null }],
      ['session refuses the line', { refuse: 'session error' }],
      ['dictation times out', { refuse: 'timeout' }],
      ['channel not dictated', { channelOff: true }],
    ];
    for (const [name, cfg] of branches) {
      const speak = loadSpeak();
      const room = cabin();
      const calls = watchFetch();
      const h = cfg.live === null ? null : session(room, cfg);
      globalThis.RIO.realtime = { active: () => h };
      const el = element(room);
      const p = speak.provider({
        text: 'Left here', channel: 'nav', callType: 'imminent',
        element: el, ttsUrl: '/nav/voice?x=1',
      });
      await p.play().catch(() => {});
      ok('[' + name + '] no synthesiser request',
         calls.filter((u) => /voice|tts|eleven/i.test(u)).length === 0, calls);
      ok('[' + name + '] tts counter is zero', speak.stats().tts === 0);
    }
  }

  // -------------------------------------------------------------------------
  section('D. a feed interruption mid-utterance leaves the voice alone');
  // -------------------------------------------------------------------------
  /* The frame path and the speech path share nothing. This kills the feed --
     violently, in every way the real one can fail -- WHILE a line is being
     spoken, and asserts the utterance completes and the session survives. */
  {
    const speak = loadSpeak();
    const room = cabin();
    watchFetch();
    const h = session(room, { timeoutMs: 2000 });
    let sessionStopped = false;
    h.stop = () => { sessionStopped = true; };
    globalThis.RIO.realtime = { active: () => h };
    const el = element(room);

    let started = false;
    const p = speak.provider({
      text: 'There is a car stopped in your lane', channel: 'headway',
      element: el,
      onStart: () => { started = true; },
    });
    const done = p.play();
    await sleep(2);

    /* THE FEED DIES, mid-sentence, four ways at once. None of these has any
       reference to the speech path -- which is the point being asserted. */
    const feedFailures = [];
    try { feedFailures.push(['ws_close', (() => { throw new Error('ws closed'); })()]); }
    catch (e) { feedFailures.push(['ws_close', e.message]); }
    try { feedFailures.push(['encoder', (() => { throw new Error('encoder gone'); })()]); }
    catch (e) { feedFailures.push(['encoder', e.message]); }
    feedFailures.push(['stall', 'no_frames_sent']);
    feedFailures.push(['camera', 'NotReadableError']);

    await done;
    ok('the utterance completed', room.mouths().join(',') === 'session',
       room.mouths());
    ok('the session was never stopped', sessionStopped === false);
    ok('nothing else spoke underneath it', !room.overlapped());
    ok('four feed failures were raised and none touched speech',
       feedFailures.length === 4, feedFailures.map((f) => f[0]).join(','));
  }

  // -------------------------------------------------------------------------
  section('E. the frame module cannot reach the speech module at all');
  // -------------------------------------------------------------------------
  /* A structural check rather than a behavioural one, and the one that
     survives somebody adding a convenience later: the file that owns frames
     must not mention the things that own speech. */
  {
    const fs = require('fs');
    const frames = fs.readFileSync(
      path.join(__dirname, '..', 'static', 'rio_frames.js'), 'utf8');
    // Strip comments: rio_frames.js DISCUSSES the audio pipeline at length in
    // its header, and that discussion is not a dependency.
    const code = frames.replace(/\/\*[\s\S]*?\*\//g, '').replace(/^\s*\/\/.*$/gm, '');
    for (const forbidden of ['RIO.speak', 'RIO.output', 'RIO.speech',
                             'RIO.realtime', 'arbiter', 'getUserMedia(']) {
      ok('rio_frames.js never touches ' + forbidden,
         code.indexOf(forbidden) < 0);
    }
  }

  console.log('\n' + '-'.repeat(58));
  console.log('  ' + (fails ? 'FAIL' : 'PASS') + ': ' + fails
              + ' failure(s) of ' + checks + ' checks');
  process.exit(fails);
}

main();
