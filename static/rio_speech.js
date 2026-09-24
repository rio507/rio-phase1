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
 *   P2  vehicle health, critical only. A possible blowout, a tire losing air
 *       fast, oil pressure gone: things that end the drive badly if the next
 *       four seconds are spent hearing about a turn instead. It sits BELOW the
 *       gap warning because a collision is measured in seconds and a failing
 *       tire in minutes, and ABOVE navigation because missing a junction costs
 *       three minutes and missing this costs the car. The deterministic policy
 *       that decides whether anything at this tier ever fires is server-side,
 *       in vehicle_health_policy.py — this file only decides who gets the
 *       mouth, exactly as it does for headway.
 *   P3  the near-tier turn. The one nav line that is genuinely time-critical:
 *       four seconds out, saying it late is the same as not saying it.
 *   P4  every other nav announcement.
 *   P5  conversation — RIO answering the driver, including a visual answer
 *       about something out of the window. Lowest on purpose: it is the only
 *       tier the driver can simply ask for again, and it is the longest, so a
 *       turn or a gap warning arriving mid-sentence must cut straight through
 *       it. Everything above this stays true only inside a window; a reply to
 *       "what kind of car is that" does not.
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
 * TWO LISTS, NOT ONE: WHO GOES FIRST, AND WHO MAY INTERRUPT
 * ---------------------------------------------------------
 * Priority used to answer both, and they are not the same question. A
 * route-start line outranks a conversational answer for ORDER -- if both are
 * waiting, the turn goes first -- and it has no business cutting her off,
 * because the first maneuver is minutes away and her sentence is seconds from
 * ending.
 *
 * THE DRIVE, 2026-09-24, session 4989d12e. A route starts while RIO is
 * finishing a sentence. Within 350 ms the nav tier takes the mouth and her
 * answer is logged `orphan_silenced`; the driver hears half a sentence and
 * then a turn instruction over the top of it.
 *
 * So an item may set `patient: true`: it is inserted by priority and it never
 * pre-empts. Safety and the junction call do not set it and still cut through
 * anything. Nothing else about the ladder changes -- supersede-by-group,
 * expiry and validity are all unchanged, so a patient line that waited past
 * its window is dropped rather than said late.
 *
 * Pure queue logic — no DOM. Items carry their own play()/stop(), which is what
 * lets tools/nav_selftest.js drive this file under node with fake items.
 */
(function (root) {
  'use strict';

  /* Lower is more urgent. Callers name these rather than passing integers, so
     inserting a tier is an edit here and nowhere else — which is what this
     change was: VEHICLE_HEALTH went in at 2 and everything below it moved down
     one, with every relative order preserved. */
  var P = { SAFETY: 1, VEHICLE_HEALTH: 2, TURN_NEAR: 3, NAV: 4, CONVO: 5 };

  // A play() that never settles would leave RIO mute for the rest of the drive,
  // so every item is on a watchdog. Longer than any line RIO says.
  var MAX_ITEM_MS = 15000;

  /* HOW LONG PATIENCE LASTS, AND WHY IT HAS TO END.
   *
   * A patient item waits for the current item's play() promise to settle. That
   * is the right gate and it has one failure mode: a `current` whose promise
   * never settles. Conversation passes maxMs 90000, so a conversational entry
   * that is never resolved holds the mouth for a minute and a half -- and
   * before patience existed nothing noticed, because a nav line simply
   * pre-empted it. Patience turns an invisible stuck entry into a silence.
   *
   * So waiting is bounded. Past this, a patient item stops being patient and
   * takes the mouth if it outranks what is holding it. Eight seconds is longer
   * than any single answer in the drive logs and far short of the watchdog it
   * is protecting against.
   *
   * IT IS ALSO WHAT MAKES "EXPIRED WHILE WAITING" IMPOSSIBLE. The shortest TTL
   * on any patient tier is the near call's 6 s... which is under this, so the
   * bound alone is not enough -- see `blockedMs` in admit(), which stops the
   * expiry clock for exactly the time an item spent waiting for a mouth it was
   * told not to take. A line that waited is never dropped FOR having waited;
   * it speaks, or valid() says the road moved on, and those are the only two
   * honest outcomes. */
  var PATIENT_MAX_WAIT_MS = 8000;

  function now() { return Date.now(); }

  /* `opts.patientMaxWaitMs` overrides the bound. Only the selftests pass it --
     the production page builds the arbiter with no arguments -- and it exists
     because the alternative is a test that sleeps for eight seconds to prove a
     timer fires. */
  function makeArbiter(opts) {
    var patientMaxWaitMs = (opts && opts.patientMaxWaitMs) || PATIENT_MAX_WAIT_MS;
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

    function unblock(item) {
      if (item && item.patienceTimer) {
        clearTimeout(item.patienceTimer);
        item.patienceTimer = null;
      }
      if (item && item.blockedAt) {
        item.blockedMs = (item.blockedMs || 0) + (now() - item.blockedAt);
        item.blockedAt = 0;
      }
    }

    function drop(item, reason) {
      unblock(item);
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
      unblock(item);
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

    /* Admission, checked at DEQUEUE and never only at creation.
     *
     * An announcement is true inside a window and false outside it, and the
     * window can close while the line waits behind a safety warning. Two
     * separate reasons it can close:
     *
     *   expired   its own TTL ran out;
     *   invalid   the world moved on — the maneuver was passed, or the route
     *             it belongs to was replaced by a reroute. Callers supply
     *             valid() and the arbiter asks it the moment before speaking,
     *             which is the only moment the answer is worth anything.
     *
     * Both drop the item silently. Nothing is ever spoken late and nothing
     * catches up.
     */
    function admit(item) {
      /* THE EXPIRY CLOCK DOES NOT RUN WHILE AN ITEM IS BLOCKED.
         A patient item was told to wait; dropping it afterwards for having
         waited would punish it for obeying, and the silence reads to a driver
         exactly like the announcement never existing. Time spent queued behind
         a mouth is credited back, so `expired` keeps meaning what it says --
         the line outlived its own window while it COULD have been spoken --
         and a line that could never have spoken is not expired, it is stuck,
         which the patience bound above is what prevents. */
      unblock(item);
      if (item.expiresAt && now() > item.expiresAt + (item.blockedMs || 0)) {
        drop(item, 'expired');
        return false;
      }
      if (typeof item.valid === 'function') {
        var ok = true;
        try { ok = !!item.valid(); } catch (e) { ok = false; }
        if (!ok) {
          drop(item, 'invalid');
          return false;
        }
      }
      return true;
    }

    function pump() {
      while (!current && queue.length) {
        var next = queue.shift();
        if (!admit(next)) continue;
        start(next);
      }
    }

    return {
      P: P,

      /* item: {priority, group, id, text, play():Promise, stop(), ttlMs, maxMs,
                meta, valid(), onDone(reason), patient}
         `patient` means "queue, never cut in" -- see the header. Absent is
         falsy, so every existing caller keeps the behaviour it had.
         `valid` is optional and is asked at dequeue, not at creation — see
         admit(). Navigation supplies one; headway and conversation do not,
         because a warning about the road ahead is either current or expired
         and has no third state.
         reasons a caller can see: spoken | superseded | preempted | expired |
         invalid | timeout | error | cleared */
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

        if (!current) {
          if (admit(item)) start(item);
          return true;
        }

        if (item.priority < current.item.priority && !item.patient) {
          // Strictly more urgent than what is speaking: cut in. The interrupted
          // line is gone for good — see the header.
          if (!admit(item)) return true;
          stopCurrent('preempted');
          start(item);
          return true;
        }
        /* PATIENT: outranks what is speaking for ORDER, but will not cut it
           off. It goes to the front of the queue and waits for the mouth.

           See the header's two-lists note. The wait ends when the current
           item's play() promise settles, which for a conversational answer is
           the end of the SOUND and not the end of generation — rio_realtime
           holds that promise open across the audio tail. So "wait for her to
           finish" needs no new clock here and cannot drift from the one the
           mouth already uses.

           ...AND A BOUND ON IT, because that promise is somebody else's. See
           PATIENT_MAX_WAIT_MS: a current item that never settles would
           otherwise hold this one until its own watchdog, which for
           conversation is ninety seconds. The bound is not a second gate on
           the normal path — it never fires while anything is actually
           speaking — it is the floor under the abnormal one. */
        insert(item);
        /* ONLY AN ITEM THAT WAS TOLD TO WAIT GETS CREDITED FOR WAITING.
           Everything else in this queue is here because something more urgent
           is speaking, and "expire, never catch up" is the rule for those --
           a far-guidance line stuck behind a collision warning has genuinely
           outlived its window and must not be played late. Patience is the
           one case where the delay is the arbiter's own instruction, so it is
           the one case that cannot count against the line. */
        if (!item.patient) return true;
        item.blockedAt = now();
        emit({ type: 'wait', item: publicItem(item),
               behind: publicItem(current.item) });
        item.patienceTimer = setTimeout(function () {
          item.patienceTimer = null;
          if (queue.indexOf(item) < 0) return;      // already spoken or dropped
          if (!current) { pump(); return; }
          if (item.priority >= current.item.priority) return;   // not ours to take
          unblock(item);
          if (!admit(item)) {
            queue.splice(queue.indexOf(item), 1);
            return;
          }
          queue.splice(queue.indexOf(item), 1);
          emit({ type: 'impatient', item: publicItem(item),
                 behind: publicItem(current.item), waited_ms: patientMaxWaitMs });
          stopCurrent('preempted');
          start(item);
        }, patientMaxWaitMs);
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
  root.RIO.speech.PATIENT_MAX_WAIT_MS = PATIENT_MAX_WAIT_MS;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { makeArbiter: makeArbiter, P: P,
                       PATIENT_MAX_WAIT_MS: PATIENT_MAX_WAIT_MS };
  }
})(typeof window !== 'undefined' ? window : globalThis);
