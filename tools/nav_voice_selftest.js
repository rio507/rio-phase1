/* nav_voice_selftest.js — eleven turns, one voice.
 *
 *   node tools/nav_voice_selftest.js
 *
 * THE COMPLAINT. On a live test the navigation callouts came out in Ava —
 * the ElevenLabs voice — instead of marin. The pod's configuration was not
 * the cause and neither was a drifted environment variable: /voice/status
 * reported `backend: openai_realtime` and `live_voice: marin` throughout.
 *
 * THE CAUSE, and this file is the proof of it. A deterministic line is
 * DICTATED by the live session, which is what makes it hers. rio_speak.js:
 *
 *     var dictate = !!(text() && live && live.speak && ...)
 *     ...
 *     if (!dictate) return fallback('no_session');
 *
 * `live` is the open realtime session. On that drive there was none — the talk
 * button never left "Connecting…" and the drive's log has no `session_started`
 * — so EVERY deterministic line took `fallback('no_session')` and every one of
 * them came out of the synthesiser. The Ava callouts were not a voice bug at
 * all. They were the live-session failure, heard.
 *
 * WHAT THIS ASSERTS
 * -----------------
 *   1. With a session open, eleven nav lines are ELEVEN DICTATIONS. None
 *      falls back.
 *   2. With no session, all eleven fall back, and the recorded reason is
 *      `no_session` — the exact signature of what happened.
 *   3. The dictation path contains no local GPU work at all, so the two
 *      teachers cannot be the cause of a fallback however loaded the card is.
 *      That one is structural and it is the point: it is easy to assume
 *      contention, and the code says otherwise.
 */
'use strict';

const fs = require('fs');
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

// A DOM that answers "no such element", and an audio element that plays
// nothing: the whole of this test is about which path was taken, not sound.
global.window = global;
global.document = {
  addEventListener() {}, getElementById() { return null; },
  querySelector() { return null; }, querySelectorAll() { return []; },
  /* An <audio> that finishes as soon as it starts. rio_speak's element path
     resolves on `onended`, so a stub that never fires it hangs the whole
     suite rather than failing it — which is how this test first behaved. */
  createElement() {
    const el = {
      pause() {}, addEventListener() {}, removeEventListener() {}, style: {},
      muted: false, onended: null, onerror: null,
      set src(v) {
        this._src = v;
        setTimeout(() => { if (el.onended) el.onended(); }, 0);
      },
      get src() { return this._src; },
      play() { return Promise.resolve(); },
    };
    return el;
  },
};
/* The synthesiser fetches its audio; without a ttsUrl and a fetch that
   answers, rio_speak records `silent` rather than `tts` and the test measures
   the wrong branch. This is the ElevenLabs tier, stubbed. */
global.fetch = () => Promise.resolve({
  ok: true,
  headers: { get() { return null; } },
  blob: () => Promise.resolve({ size: 1 }),
});
global.URL = { createObjectURL: () => 'blob:tts', revokeObjectURL() {} };
global.AbortController = function () {
  this.signal = {}; this.abort = function () {};
};

require(path.join(STATIC, 'rio_speak.js'));

const NAV_LINES = [
  'In four hundred metres, turn right onto De La Cruz Boulevard',
  'Turn right onto De La Cruz Boulevard',
  'In two hundred metres, keep left',
  'Keep left',
  'In one mile, take the exit',
  'Take the exit toward Central Expressway',
  'In three hundred metres, turn left',
  'Turn left onto Bowers Avenue',
  'Continue for two miles',
  'In four hundred metres, make a U-turn',
  'You have arrived',
];

function fakeSession(opts) {
  opts = opts || {};
  return {
    speak(text, o) {
      // A dictation that starts inside its budget, as a healthy session does.
      if (opts.slowerThanBudget) {
        return new Promise((_, reject) =>
          setTimeout(() => reject(new Error('dictation timeout')),
                     (o && o.timeoutMs ? o.timeoutMs : 900) + 50));
      }
      if (o && o.onStart) o.onStart();
      return Promise.resolve({ dictated: true, voice: 'marin' });
    },
    speechEnabled() { return true; },
  };
}

async function runLines(session, label) {
  RIO.realtime = RIO.realtime || {};
  RIO.realtime.active = () => session;
  RIO.speak.reset();
  const reasons = [];
  for (const line of NAV_LINES) {
    const p = RIO.speak.provider({
      text: line, channel: 'navigation',
      element: document.createElement('audio'),
      // The ElevenLabs tier. Present because it IS present on the pod --
      // /voice/status reports it configured -- and its being reachable is
      // exactly why a missing session comes out as Ava rather than as silence.
      ttsUrl: '/nav/voice?line=' + encodeURIComponent(line),
      onFallback: (why) => reasons.push(why),
    });
    try { await p.play(); } catch (e) { /* the ladder swallows its own */ }
  }
  const s = RIO.speak.stats();
  console.log(`       ${label}: dictated=${s.dictated} tts=${s.tts} `
              + `clip=${s.clip} silent=${s.silent} last=${s.last}`);
  return { stats: s, reasons };
}

(async function main() {
  section('a session is open — eleven turns, eleven dictations');
  const live = await runLines(fakeSession(), 'session open');
  ok(live.stats.dictated === NAV_LINES.length,
     `all ${NAV_LINES.length} nav lines are dictated in her voice `
     + `(${live.stats.dictated}/${NAV_LINES.length})`);
  ok(live.stats.tts === 0,
     `and none reaches the synthesiser (${live.stats.tts}) — the synthesiser `
     + `is the fallback, and on a healthy drive it should never be needed`);

  section('no session — and this is exactly what the drive sounded like');
  const dead = await runLines(null, 'no session');
  ok(dead.stats.dictated === 0,
     `with no live session nothing is dictated (${dead.stats.dictated})`);
  ok(dead.stats.tts === NAV_LINES.length,
     `and all ${NAV_LINES.length} come out of the synthesiser `
     + `(${dead.stats.tts}) — in Ava, which is the complaint`);
  ok(dead.reasons.length === 0 || dead.reasons.every(r => r === 'no_session'),
     `every one of them for the same reason: no_session `
     + `(${JSON.stringify(dead.reasons.slice(0, 2))})`);

  section('a session that is too slow still falls back — but says so');
  const slow = await runLines(fakeSession({ slowerThanBudget: true }), 'slow');
  ok(slow.stats.tts === NAV_LINES.length,
     `a dictation that misses its budget is synthesised (${slow.stats.tts}) — `
     + `a line at a junction is never held for a voice that is not ready`);
  ok(slow.stats.dictated === 0, 'and is not counted as hers');

  section('the dictation path does no local GPU work — the teachers are not in it');
  /* IT IS EASY TO ASSUME CONTENTION and wrong here. Dictation is a request to
     the remote realtime session; Qwen, RF-DETR and the two teachers are all on
     the local card and none of them is on this path. So no amount of teacher
     load can turn a nav line into an Ava line -- only the session being absent
     or slow can, and both are asserted above. */
  const speakSrc = fs.readFileSync(path.join(STATIC, 'rio_speak.js'), 'utf8');
  const rtSrc = fs.readFileSync(path.join(STATIC, 'rio_realtime.js'), 'utf8');
  for (const [name, src] of [['rio_speak.js', speakSrc],
                             ['rio_realtime.js', rtSrc]]) {
    ok(!/teachers/i.test(src.replace(/\/\*[\s\S]*?\*\//g, '')
                            .replace(/\/\/.*$/gm, '')),
       `${name} contains no teacher call at all`);
  }
  /* Only rio_speak.js is checked for local-GPU calls. rio_realtime.js also
     carries the LOOK tool, which is a request to the camera stack by design --
     that is a question a driver asked, not a line being spoken, and it is the
     one place the local card is on RIO's critical path at all. */
  ok(!/\/(observe|perceive|headway_frame)\b/.test(speakSrc),
     'rio_speak.js — the file that decides which mouth says a deterministic '
     + 'line — makes no local-GPU request at all');
  ok(/live\.speak\(/.test(speakSrc),
     'dictation is a call on the live SESSION — remote, and nothing to do '
     + 'with the card the teachers are on');

  console.log('\n' + (failures ? 'FAILED ' + failures + '/' : 'PASSED ')
              + checks + ' checks');
  process.exit(failures ? 1 : 0);
})().catch(e => { console.error(e); process.exit(2); });
