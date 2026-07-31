/* rio_speech.js — Speech arbiter v1. RIO has ONE mouth.
 *
 * Before this file the headway warnings owned their own <audio> elements and
 * played whenever the server said so. That works while there is exactly one
 * thing that can talk. The moment navigation can also talk, "two systems each
 * certain they should be heard" becomes two voices over each other, and the one
 * that loses is always the safety line, because it is the shortest.
 *
 * So everything that speaks goes through say(). The arbiter owns the decision
 * of what is audible right now; callers own only the audio.
 *
 *   P1  safety — headway warnings. Pre-empts anything.
 *   P2  the near-tier turn. The one nav line that is genuinely time-critical:
 *       four seconds out, saying it late is the same as not saying it.
 *   P3  every other nav announcement.
 *
 * Two rules beyond the priority ladder:
 *
 *   Supersede by group. A newer item in the same group replaces an older one,
 *   playing or queued. Group is the maneuver for nav ("nav:m3"), so the near
 *   tier cuts off its own far tier mid-sentence rather than queueing behind it —
 *   "in 300 meters, turn right" finishing after you should already be turning
 *   is worse than silence. For headway the group is the whole channel, which
 *   preserves the behaviour the red tier already had: it interrupts coaching.
 *
 *   Expire, never catch up. A queued announcement that outlived its moment is
 *   dropped, not delayed. Nav lines are true only in a window; a backlog of
 *   turn-by-turn played late is actively dangerous.
 *
 * Pre-empted and superseded items are never resumed. The mouth moves forward.
 *
 * Pure queue logic — no DOM. Items carry their own play()/stop(), which is what
 * lets tools/nav_selftest.js drive this file under node with fake items.
 */
(function (root) {
  'use strict';

  var P = { SAFETY: 1, TURN_NEAR: 2, NAV: 3 };

  // A play() that never settles would leave RIO mute for the rest of the drive,
  // so every item is on a watchdog. Longer than any line RIO says.
  var MAX_ITEM_MS = 15000;

  function now() { return Date.now(); }

  function makeArbiter() {
    var current = null;     // {item, stopped}
    var queue = [];
    var seq = 0;
    var listeners = [];

    function emit(ev) {
      for (var i = 0; i < listeners.length; i++) {
        try { listeners[i](ev); } catch (e) { /* a bad listener must not mute RIO */ }
      }
    }

    function finish(entry, reason) {
      if (entry.done) return;
      entry.done = true;
      if (entry.watchdog) { clearTimeout(entry.watchdog); entry.watchdog = null; }
      var item = entry.item;
      emit({ type: 'end', reason: reason, item: publicItem(item) });
      if (typeof item.onDone === 'function') {
        try { item.onDone(reason); } catch (e) {}
      }
      if (current === entry) {
        current = null;
        pump();
      }
    }

    function publicItem(item) {
      return {
        id: item.id, group: item.group, priority: item.priority,
        text: item.text, meta: item.meta || null,
      };
    }

    function drop(item, reason) {
      emit({ type: 'drop', reason: reason, item: publicItem(item) });
      if (typeof item.onDone === 'function') {
        try { item.onDone(reason); } catch (e) {}
      }
    }

    function stopCurrent(reason) {
      if (!current) return;
      var entry = current;
      try { if (typeof entry.item.stop === 'function') entry.item.stop(); } catch (e) {}
      finish(entry, reason);
    }

    function start(item) {
      var entry = { item: item, done: false, watchdog: null };
      current = entry;
      emit({ type: 'start', item: publicItem(item) });
      entry.watchdog = setTimeout(function () {
        try { if (typeof item.stop === 'function') item.stop(); } catch (e) {}
        finish(entry, 'timeout');
      }, item.maxMs || MAX_ITEM_MS);
      var p;
      try {
        p = item.play();
      } catch (e) {
        finish(entry, 'error');
        return;
      }
      if (p && typeof p.then === 'function') {
        p.then(function () { finish(entry, 'spoken'); },
               function () { finish(entry, 'error'); });
      } else {
        finish(entry, 'spoken');
      }
    }

    // Best first: priority, then arrival order. A stable sort would do it, but
    // being explicit costs nothing and the queue is never more than a few deep.
    function insert(item) {
      var i = 0;
      while (i < queue.length &&
             (queue[i].priority < item.priority ||
              (queue[i].priority === item.priority && queue[i].seq < item.seq))) i++;
      queue.splice(i, 0, item);
    }

    function pump() {
      while (!current && queue.length) {
        var next = queue.shift();
        if (next.expiresAt && now() > next.expiresAt) {
          drop(next, 'expired');
          continue;
        }
        start(next);
      }
    }

    return {
      P: P,

      /* item: {priority, group, id, text, play():Promise, stop(), ttlMs, maxMs,
                meta, onDone(reason)}
         reasons a caller can see: spoken | superseded | preempted | expired |
         timeout | error | cleared */
      say: function (item) {
        if (!item || typeof item.play !== 'function') return false;
        item.seq = ++seq;
        item.priority = item.priority || P.NAV;
        item.group = item.group || ('anon:' + item.seq);
        item.at = now();
        item.expiresAt = item.ttlMs ? item.at + item.ttlMs : 0;

        // Supersede within the group, queue first so a newer item cannot be
        // dropped by its own predecessor's cleanup.
        for (var i = queue.length - 1; i >= 0; i--) {
          if (queue[i].group === item.group) {
            drop(queue.splice(i, 1)[0], 'superseded');
          }
        }
        if (current && current.item.group === item.group) {
          stopCurrent('superseded');
        }

        if (!current) { start(item); return true; }

        if (item.priority < current.item.priority) {
          // Strictly more urgent than what is speaking: cut in. The interrupted
          // line is gone for good — see the header.
          stopCurrent('preempted');
          start(item);
          return true;
        }
        insert(item);
        return true;
      },

      /* Drop everything, optionally only the groups starting with `prefix`
         (nav uses "nav:" when a route is cleared or a drive ends). */
      clear: function (prefix) {
        for (var i = queue.length - 1; i >= 0; i--) {
          if (!prefix || queue[i].group.indexOf(prefix) === 0) {
            drop(queue.splice(i, 1)[0], 'cleared');
          }
        }
        if (current && (!prefix || current.item.group.indexOf(prefix) === 0)) {
          stopCurrent('cleared');
        }
      },

      /* Observability for the panel and the session log. */
      onEvent: function (fn) { if (typeof fn === 'function') listeners.push(fn); },

      state: function () {
        return {
          speaking: current ? publicItem(current.item) : null,
          queued: queue.map(publicItem),
        };
      },
    };
  }

  root.RIO = root.RIO || {};
  root.RIO.speech = root.RIO.speech || makeArbiter();
  root.RIO.speech.makeArbiter = makeArbiter;   // tests build their own

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { makeArbiter: makeArbiter, P: P };
  }
})(typeof window !== 'undefined' ? window : globalThis);
