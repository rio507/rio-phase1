/* rio_speak.js — one voice for everything RIO says deterministically.
 *
 * A gap warning, a turn instruction and a tire announcement are written by
 * policy code, word for word, and until now they were spoken by a different
 * synthesiser from the one RIO answers questions with. Two voices in one car,
 * and the second one — the one that only ever says something important — was
 * the one that sounded like a different product.
 *
 * So deterministic lines are DICTATED to the live session: same voice, same
 * mouth, word for word. What this file owns is the part that has to be true
 * even when that is not possible:
 *
 *   1. the live session, if one is open and the channel is dictated;
 *   2. the synthesiser, if it is not — ElevenLabs, complete and dormant behind
 *      VOICE_BACKEND, reached through the server's TTS endpoints, which are
 *      the fallback path by definition;
 *   3. a pre-rendered clip, if the line has one.
 *
 * A warning NEVER waits on a cloud call it is not getting. Dictation is given
 * a bounded time to START and then abandoned — the synthesiser is about 180 ms
 * away and a late warning has stopped being a warning.
 *
 * HOW LONG THAT IS DEPENDS ON THE LINE, and getting this wrong is not a matter
 * of taste. With a budget B, a line that starts in time is heard at up to B in
 * her voice, and a line that misses is heard at about B + 180 ms in the other
 * one. So a looser budget buys vocal consistency and lengthens the WORST CASE.
 * "Left here." at the junction wants the worst case bounded; "Right turn coming
 * up onto Cloverfield Blvd.", issued seconds out, wants her voice. The table
 * lives in config.py (REALTIME_SPEAK_TIMEOUT_MS_BY_CHANNEL), travels with the
 * session, and is read here per channel and per call type.
 *
 * What this file does NOT do is decide anything about priority, ordering or
 * interruption. Every caller keeps the arbiter item it already had; this
 * replaces only the audio behind it. A gap warning is still P1 and still cuts
 * RIO off mid-sentence — and now it cuts her off in her own voice.
 */
(function (root) {
  'use strict';

  var stats = { dictated: 0, tts: 0, clip: 0, silent: 0, last: null };

  function fetchBlobAudio(element, url) {
    var ctl = (typeof AbortController !== 'undefined') ? new AbortController() : null;
    var stopped = false;
    var release = null;
    return {
      abort: function () {
        stopped = true;
        if (ctl) { try { ctl.abort(); } catch (e) {} }
        try { element.pause(); } catch (e) {}
        if (release) { release(); release = null; }
      },
      play: function () {
        return fetch(url, ctl ? { signal: ctl.signal } : undefined)
          .then(function (r) {
            if (!r.ok) throw new Error('tts ' + r.status);
            var said = r.headers.get('X-Nav-Text');
            if (said) {
              try { root.RIO.bus && root.RIO.bus.emit('speaking',
                    { text: decodeURIComponent(said) }); } catch (e) {}
            }
            return r.blob();
          })
          .then(function (blob) {
            if (stopped) return;
            return new Promise(function (resolve, reject) {
              var objUrl = URL.createObjectURL(blob);
              var settled = false;
              var done = function (fn, arg) {
                if (settled) return;
                settled = true;
                element.onended = element.onerror = null;
                URL.revokeObjectURL(objUrl);
                fn(arg);
              };
              release = function () { done(resolve); };
              element.onended = function () { done(resolve); };
              element.onerror = function () { done(reject, new Error('audio error')); };
              element.muted = false;
              element.src = objUrl;
              var p = element.play();
              if (p && p.catch) p.catch(function (e) { done(reject, e); });
            });
          });
      },
    };
  }

  function playClip(element, url) {
    var stopped = false;
    var release = null;
    return {
      abort: function () {
        stopped = true;
        try { element.pause(); } catch (e) {}
        if (release) { release(); release = null; }
      },
      play: function () {
        if (stopped) return Promise.resolve();
        return new Promise(function (resolve, reject) {
          var settled = false;
          var done = function (fn, arg) {
            if (settled) return;
            settled = true;
            element.onended = element.onerror = null;
            fn(arg);
          };
          release = function () { done(resolve); };
          element.onended = function () { done(resolve); };
          element.onerror = function () { done(reject, new Error('clip error')); };
          element.muted = false;
          element.src = url;
          var p = element.play();
          if (p && p.catch) p.catch(function (e) { done(reject, e); });
        });
      },
    };
  }

  /* An audio provider for an arbiter item: {play, stop}.
   *
   *   text      the exact words, as written by the policy that decided them
   *   channel   'headway' | 'health' | 'nav' — which dictation switch applies
   *   callType  for nav, which of the four calls this is: a backup call at the
   *             junction and one issued seconds out have different deadlines
   *   ttsUrl    the fallback synthesiser endpoint for this line
   *   clipUrl   a pre-rendered file, if this line has one
   *   element   the audio element to play fallbacks through
   *   timeoutMs an explicit override; otherwise the budget is resolved from
   *             the channel and call type, per config.speak_timeout_ms
   */
  function provider(opts) {
    opts = opts || {};
    var element = opts.element;
    var current = null;
    var stopped = false;
    var live = (root.RIO && root.RIO.realtime && root.RIO.realtime.active) ?
               root.RIO.realtime.active() : null;
    var dictate = !!(text() && live && live.speak &&
                     (!live.speechEnabled || live.speechEnabled(opts.channel)));

    function text() { return (opts.text || '').trim(); }

    /* HOW LONG THIS LINE WAITS, and it is not one number for all of them.
     *
     * The budget is the point at which a line stops waiting for her voice and
     * is synthesised instead, so it is a question about THIS sentence's
     * deadline: "Left here." at the junction cannot be late, and "Right turn
     * coming up onto Cloverfield Blvd." can. Asked of the session, which
     * carries the table config.py decided, so there is one copy of it and it
     * is not this one. */
    function budget() {
      if (opts.timeoutMs) return opts.timeoutMs;
      if (live && live.speakTimeout) {
        var ms = live.speakTimeout(opts.channel, opts.callType);
        if (ms) return ms;
      }
      return undefined;          // the controller's own default
    }

    function record(path) { stats[path]++; stats.last = path; }

    function fallback(reason) {
      if (stopped) return Promise.resolve();
      if (opts.ttsUrl) {
        record('tts');
        current = fetchBlobAudio(element, opts.ttsUrl);
        return current.play().catch(function (e) {
          // The synthesiser failed too. A pre-rendered clip has no network in
          // its path at all, which is exactly the situation this is now in.
          if (opts.clipUrl && !stopped) {
            record('clip');
            current = playClip(element, opts.clipUrl);
            return current.play();
          }
          record('silent');
          throw e;
        });
      }
      if (opts.clipUrl) {
        record('clip');
        current = playClip(element, opts.clipUrl);
        return current.play();
      }
      record('silent');
      return Promise.reject(new Error(reason || 'no audio path'));
    }

    return {
      play: function () {
        if (stopped) return Promise.resolve();
        if (!dictate) return fallback('no_session');
        return live.speak(text(), { timeoutMs: budget() })
          .then(function (r) { record('dictated'); return r; })
          .catch(function (e) {
            // Dictation did not start in time, the session went away, or it was
            // already busy saying something else. Say it the other way.
            return fallback(e && e.message);
          });
      },
      stop: function () {
        stopped = true;
        if (current && current.abort) current.abort();
        // A dictation in flight is cancelled by the arbiter's own pre-emption
        // of the item that owns it; nothing to do here but stop the fallback.
      },
    };
  }

  root.RIO = root.RIO || {};
  root.RIO.speak = {
    provider: provider,
    stats: function () { return stats; },
    reset: function () {
      stats.dictated = stats.tts = stats.clip = stats.silent = 0;
      stats.last = null;
    },
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = root.RIO.speak;
  }
})(typeof window !== 'undefined' ? window : globalThis);
