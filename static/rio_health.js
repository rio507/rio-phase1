/* rio_health.js — the mouth for Vehicle Health. It decides nothing.
 *
 * The whole of this file is: ask the server whether RIO has something to say
 * about the car, and if she does, hand it to the arbiter. Every judgement —
 * whether a fault is critical, whether it has already been announced, whether
 * enough time has passed, and the words themselves — is made server-side in
 * vehicle_health_policy.py, which imports nothing and cannot reach a model.
 *
 * That split is deliberate and it is the same one navigation uses. The reason
 * the decision is not here is not that JavaScript could not do it: it is that a
 * decision made in the browser is a decision that cannot be replayed from a
 * log, cannot be tested without a browser, and exists in as many versions as
 * there are open tabs.
 *
 * WHY THERE IS NO SPEECH SYSTEM IN THIS FILE
 * ------------------------------------------
 * There is exactly one, in rio_speech.js, and this submits to it like everything
 * else with a mouth. Priority VEHICLE_HEALTH (2): below a headway warning, which
 * is measured in seconds, and above navigation, which is measured in minutes.
 * Group "health", so a newer critical replaces an older one mid-sentence rather
 * than queueing behind it — a blowout announced after the "your tire is low"
 * line has finished is the wrong order to hear them in.
 *
 * The text comes back from /vehicle/health/voice, addressed by the id the policy
 * issued. This file never sends a sentence to the server, and could not: it does
 * not have one until the audio arrives.
 */
(function (root) {
  'use strict';

  /* Only used when the fetch itself failed — the real cadence always comes from
     the payload's poll_ms (config.HEALTH_POLL_MS), which we do not have when the
     request never landed. Same shape as rio_vehicle.js's RETRY_MS. */
  var RETRY_MS = 8000;

  /* A critical announcement is about a fault that is still there. Unlike a nav
     line it does not stop being true in three seconds — but it must not sit in a
     queue behind a long conversational answer either, because by the time it
     played the driver would have been told about a tire they have been driving
     on for a minute. Long enough to wait out a gap warning, short enough that a
     stale one is dropped rather than delayed. */
  var TTL_MS = 20000;

  var pollMs = null;
  var timer = null;
  var wired = false;
  var lastId = null;

  /* Its own audio element, unlocked in a user gesture like every other audio
     path on this page — iOS will not play from a timer otherwise. Fetched as a
     blob rather than streamed so the exact sentence comes back with it on
     X-Health-Text and the Voice Layer column shows the words being spoken
     instead of its own guess at them. */
  var audio = new Audio();
  audio.preload = 'auto';
  var unlocked = false;

  function unlock() {
    if (unlocked) return;
    unlocked = true;
    try {
      audio.muted = true;
      var p = audio.play();
      if (p && p.then) {
        p.then(function () { audio.pause(); audio.currentTime = 0; audio.muted = false; })
         .catch(function () { audio.muted = false; });
      } else { audio.muted = false; }
    } catch (e) { audio.muted = false; }
  }

  // ---- speaking ----------------------------------------------------------

  function announce(ann) {
    if (!ann || !ann.id) return;
    /* The same announcement offered twice is a poll that overlapped, not a
       second event — the policy issues a new id for a genuine repeat. */
    if (ann.id === lastId) return;
    lastId = ann.id;

    var ctl = (typeof AbortController !== 'undefined') ? new AbortController() : null;
    var stopped = false;
    var cleanup = null;      // set once the audio blob exists; see stop()

    RIO.speech.say({
      priority: RIO.speech.P.VEHICLE_HEALTH,
      group: 'health',
      id: 'health:' + ann.id,
      text: ann.text || '',        // the policy's own words, until the audio confirms them
      ttlMs: TTL_MS,
      meta: { key: ann.key, type: ann.type, severity: ann.severity,
              reason: ann.reason },
      play: function () {
        return fetch('/vehicle/health/voice?id=' + encodeURIComponent(ann.id),
                     ctl ? { signal: ctl.signal } : undefined)
          .then(function (r) {
            if (!r.ok) throw new Error('health voice ' + r.status);
            var said = r.headers.get('X-Health-Text');
            if (said) {
              try { RIO.bus.emit('speaking', { text: decodeURIComponent(said) }); }
              catch (e) {}
            }
            return r.blob();
          })
          .then(function (blob) {
            if (stopped) return;
            return new Promise(function (resolve, reject) {
              var url = URL.createObjectURL(blob);
              var settled = false;
              var done = function (fn) {
                if (settled) return;
                settled = true;
                audio.onended = audio.onerror = null;
                URL.revokeObjectURL(url);
                fn();
              };
              // A pre-empted line never reaches 'ended', so stop() has to be
              // the one that releases the blob.
              cleanup = function () { done(resolve); };
              audio.onended = function () { done(resolve); };
              audio.onerror = function () { done(reject); };
              audio.src = url;
              var p = audio.play();
              if (p && p.catch) p.catch(function (e) { done(function () { reject(e); }); });
            });
          });
      },
      stop: function () {
        stopped = true;
        if (ctl) { try { ctl.abort(); } catch (e) {} }
        try { audio.pause(); } catch (e) {}
        if (cleanup) { cleanup(); cleanup = null; }
      },
    });
  }

  // ---- polling -----------------------------------------------------------

  /* Chained timeouts, not setInterval — the same discipline as the perception,
     headway, tire and telemetry loops: a slow response must not let calls stack
     up. Here it matters more than usual, because each response IS a tick of the
     announcement policy and two in flight at once would tick it twice. */
  function schedule(ms) {
    clearTimeout(timer);
    timer = setTimeout(tick, ms);
  }

  async function tick() {
    /* A backgrounded tab stops asking. Not an optimisation: the poll ticks the
       server's policy, and a tab nobody is in front of consuming an announcement
       the driver never hears is the one failure this loop can cause. It picks
       straight back up on visibilitychange below. */
    if (document.hidden) { schedule(pollMs || RETRY_MS); return; }
    try {
      var r = await fetch('/vehicle/health/announcement', { cache: 'no-store' });
      var j = await r.json();
      if (j && j.poll_ms) pollMs = j.poll_ms;
      if (j && j.announce) announce(j.announce);
      schedule(pollMs || RETRY_MS);
    } catch (e) {
      schedule(RETRY_MS);
    }
  }

  function init() {
    if (wired) return;
    wired = true;

    /* Any first interaction unlocks playback. The dashboard's own unlock runs on
       the mic button, and a driver may never touch it — a critical announcement
       that cannot play because nobody pressed the right control is worse than
       one that is a beat late. */
    ['pointerdown', 'keydown', 'touchstart'].forEach(function (ev) {
      document.addEventListener(ev, unlock, { once: true, passive: true });
    });

    document.addEventListener('visibilitychange', function () {
      if (!document.hidden) tick();
    });

    tick();
  }

  root.RIO = root.RIO || {};
  root.RIO.health = { unlock: unlock, announce: announce };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})(window);
