/* rio_navplan.js — the navigation speech planner and the context lifecycle.
 *
 * The tracker says where the car is. This file decides what, if anything, is
 * worth saying about it, and hands the result to the one arbiter. It holds no
 * geometry and it cannot alter navigation truth: every sentence it can produce
 * was written by the server at route load and arrives in the route payload.
 *
 * THREE OPPORTUNITIES, NOT THREE ANNOUNCEMENTS
 * --------------------------------------------
 *   EARLY     ~25 s out   "Right turn coming up."      optional, no camera
 *   PRIMARY   ~6 s out    "Turn right by the Shell station."
 *                         ...or "Take the next right." when there is no anchor
 *   IMMINENT  ~2.5 s out  "Right here."                only when it adds something
 *
 * The primary call REPLACES distance narration. RIO does not say "Turn right in
 * 200 feet" and then "Turn right by the Shell" — the second sentence is the
 * whole instruction, and the first is what a GPS says, not what a passenger
 * says.
 *
 * THE CONTEXT LIFECYCLE IS ITS OWN MACHINE
 * ----------------------------------------
 *   INACTIVE -> ACQUIRING -> VERIFIED -> CALLED
 *                    \           \
 *                     -> EXPIRED  -> EXPIRED
 *
 * It runs alongside the maneuver state machine and never inside it. A maneuver
 * sits in APPROACHING while its context sits in ACQUIRING; if acquisition
 * fails, times out, or the camera is not there at all, the maneuver's state is
 * unaffected and the primary call goes out with the canonical sentence. That
 * separation is what makes "vision is optional" structural rather than
 * aspirational: there is no path by which a perception failure delays or
 * suppresses a navigation instruction.
 *
 * VALIDITY IS CHECKED WHEN THE LINE IS ABOUT TO BE SPOKEN
 * ------------------------------------------------------
 * Not when it is queued. A "Right here." created 2 s before a junction and
 * dequeued 3 s later, after a safety warning finished, is a lie about a turn
 * the car has already taken. Every candidate carries a `valid()` the arbiter
 * calls at dequeue: right maneuver, right route generation, not passed, not
 * expired. Invalid lines are dropped silently — never spoken late, never
 * "caught up".
 */
(function (root) {
  'use strict';

  var CALL = { EARLY: 'early', PRIMARY: 'primary', IMMINENT: 'imminent', ARRIVAL: 'arrival' };
  var CTX = {
    INACTIVE: 'INACTIVE', ACQUIRING: 'ACQUIRING', VERIFIED: 'VERIFIED',
    CALLED: 'CALLED', EXPIRED: 'EXPIRED'
  };
  var EV = {
    EARLY_GUIDANCE: 'NAV_EARLY_GUIDANCE',
    CONTEXT_ACQUISITION_STARTED: 'NAV_CONTEXT_ACQUISITION_STARTED',
    ANCHOR_CANDIDATE: 'NAV_ANCHOR_CANDIDATE',
    ANCHOR_VERIFIED: 'NAV_ANCHOR_VERIFIED',
    ANCHOR_REJECTED: 'NAV_ANCHOR_REJECTED',
    CONTEXTUAL_CALL: 'NAV_CONTEXTUAL_CALL',
    NEAR_TURN: 'NAV_NEAR_TURN',
    SPEECH_EXPIRED: 'NAV_SPEECH_EXPIRED',
    SPEECH_INVALIDATED: 'NAV_SPEECH_INVALIDATED',
    SPEECH_SPOKEN: 'NAV_SPEECH_SPOKEN'
  };

  var DEFAULTS = {
    early_guidance_s: 25.0,
    anchor_acquisition_s: 11.0,
    context_call_s: 6.0,
    near_turn_s: 2.5,
    min_call_distance_m: 20.0,
    max_call_distance_m: 400.0,
    early_max_distance_m: 900.0,
    gps_degraded_bias_s: 2.0,
    stationary_speed_ms: 0.7,
    duplicate_instruction_cooldown_s: 8.0,
    // Closest two navigation lines for one maneuver may land together. Below
    // this the imminent backup is dropped rather than stacked on the primary.
    imminent_min_gap_s: 2.0,
    // A full instruction takes about two seconds to say, and a sentence still
    // playing when the driver has to act is worse than a shorter one that
    // finished. Inside this lead the primary call is skipped and the junction
    // gets "Left here." instead — never a longer line begun too late.
    primary_min_lead_s: 4.0,
    anchor_valid_for_s: 6.0,
    vision_enabled: true,
    // How many times acquisition may be attempted for one maneuver before the
    // context is given up on. Two: one early look, one second chance if the
    // landmark had not come into view yet. A third would be spending the
    // driver's approach on a landmark that is not there.
    acquisition_attempts: 2
  };

  function create(cfg) {
    cfg = cfg || {};
    var tracker = cfg.tracker;
    var arbiter = cfg.arbiter;
    var route = cfg.route || (tracker && tracker.route);
    // The planner's clock is the TRACKER's clock, in seconds, not wall time.
    // A simulated drive and a real one then time identically, and every
    // expiry in here is expressed in the same units the tracker reports.
    var verify = cfg.verify || null;      // fn(request) -> Promise<anchor|null>
    var audio = cfg.audio || null;        // fn(candidate) -> {play, stop}
    /* Which route generation is CURRENTLY live, asked freshly each time.
     *
     * Not `route.generation_id`: a reroute does not mutate the route object
     * this planner was built with, it replaces it — so a planner holding
     * generation 1 would go on believing generation 1 is current forever, and
     * a line queued against it would still look valid at dequeue. The glue
     * passes a function reading the active route; the fallback is only right
     * for a planner that outlives nothing. */
    var activeGeneration = cfg.activeGeneration ||
        function () { return route.generation_id; };

    var opt = {}, k;
    for (k in DEFAULTS) opt[k] = DEFAULTS[k];
    for (k in ((route && route.timing) || {})) if (opt[k] !== undefined) opt[k] = route.timing[k];
    for (k in (cfg.options || {})) opt[k] = cfg.options[k];

    var P = (arbiter && arbiter.P) || { TURN_NEAR: 3, NAV: 4 };
    var listeners = [];
    var perMan = {};              // maneuver_id -> bookkeeping
    var lastSpoken = {};          // text -> seconds, for duplicate suppression
    var clock = 0;                // tracker clock, seconds
    var stopped = false;
    var counters = { candidates: 0, spoken: 0, invalidated: 0, expired: 0,
                     anchors_verified: 0, anchors_rejected: 0 };

    function emit(type, payload) {
      var ev = { t: clock, route_id: route.route_id,
                 generation_id: route.generation_id };
      for (var kk in (payload || {})) ev[kk] = payload[kk];
      ev.type = type;
      for (var i = 0; i < listeners.length; i++) {
        try { listeners[i](ev); } catch (e) { /* never let a listener mute RIO */ }
      }
    }

    function book(id) {
      if (!perMan[id]) {
        perMan[id] = {
          called: {}, context: CTX.INACTIVE, anchor: null, attempts: 0,
          acquiring: false, contextReason: null, primaryLate: false
        };
      }
      return perMan[id];
    }

    function manById(id) {
      var list = route.maneuvers || [];
      for (var i = 0; i < list.length; i++) if (list[i].id === id) return list[i];
      return null;
    }

    function bias() {
      var st = tracker && tracker.state ? tracker.state() : null;
      // GPS degraded near a maneuver: speak EARLIER, never later. A fix we do
      // not trust is a reason to give the driver more room, not less.
      return (st && st.gps_state === 'GPS_DEGRADED') ? opt.gps_degraded_bias_s : 0;
    }

    /* --- the anchor ------------------------------------------------------ */
    function anchorUsable(b) {
      if (!b.anchor) return false;
      if (b.anchor.valid_until && clock > b.anchor.valid_until) {
        b.context = CTX.EXPIRED;
        b.contextReason = 'anchor_expired';
        emit(EV.ANCHOR_REJECTED, { maneuver_id: b.id, reason: 'anchor_expired',
                                   label: b.anchor.label });
        counters.anchors_rejected++;
        b.anchor = null;
        return false;
      }
      return true;
    }

    function startAcquisition(man, b, snapshot) {
      if (!opt.vision_enabled || !verify) return;
      if (!man.anchors || !man.anchors.length) return;
      if (b.acquiring || b.context === CTX.CALLED) return;
      if (b.attempts >= opt.acquisition_attempts) return;
      b.attempts++;
      b.acquiring = true;
      if (b.context === CTX.INACTIVE || b.context === CTX.EXPIRED) b.context = CTX.ACQUIRING;
      emit(EV.CONTEXT_ACQUISITION_STARTED, {
        maneuver_id: man.id, attempt: b.attempts,
        tta_s: snapshot.tta_s, to_maneuver_m: snapshot.to_maneuver_m,
        candidates: man.anchors.map(function (a) { return a.anchor_id; })
      });
      man.anchors.forEach(function (a) {
        emit(EV.ANCHOR_CANDIDATE, {
          maneuver_id: man.id, anchor_id: a.anchor_id, label: a.label,
          type: a.type, relation: a.relation,
          relation_confidence: a.relation_confidence,
          distance_to_maneuver_m: a.distance_to_maneuver_m
        });
      });

      var gen = route.generation_id;
      var request = {
        route_id: route.route_id, generation_id: gen, maneuver_id: man.id,
        candidates: man.anchors, tta_s: snapshot.tta_s
      };
      var settle = function (anchor, reason, rejections) {
        b.acquiring = false;
        // A verification that lands after the route was replaced, or after the
        // maneuver was passed, describes a world that no longer exists.
        if (stopped || gen !== activeGeneration()) return;
        if (tracker && tracker.isPassed && tracker.isPassed(man.id)) return;
        if (anchor && anchor.label) {
          anchor.valid_until = clock + (anchor.valid_for_s || opt.anchor_valid_for_s);
          b.anchor = anchor;
          b.context = CTX.VERIFIED;
          counters.anchors_verified++;
          emit(EV.ANCHOR_VERIFIED, {
            maneuver_id: man.id, anchor_id: anchor.anchor_id, label: anchor.label,
            type: anchor.type, relation: anchor.turn_relation_to_anchor,
            identity_confidence: anchor.identity_confidence,
            visibility_confidence: anchor.visibility_confidence,
            relation_confidence: anchor.relation_confidence,
            valid_until: anchor.valid_until
          });
        } else {
          b.context = (b.attempts >= opt.acquisition_attempts) ? CTX.EXPIRED : CTX.ACQUIRING;
          b.contextReason = reason || (anchor && anchor.reason) || 'not_verified';
          counters.anchors_rejected++;
          emit(EV.ANCHOR_REJECTED, {
            maneuver_id: man.id, reason: b.contextReason, attempt: b.attempts,
            rejections: rejections || null
          });
        }
      };
      /* The verifier may answer synchronously (a test, a cached result) or
         with a promise (the real one, which is an HTTP call), and it may
         answer with either a bare anchor or the server's
         {anchor, reason, rejections} envelope. All four shapes unwrap the
         same way, here, once — an unwrapping that differs between the sync
         and async paths is a bug that only ever shows up in production. */
      var unwrap = function (r) {
        if (!r) return { anchor: null, reason: null };
        if (r.anchor !== undefined || r.reason !== undefined) {
          return { anchor: r.anchor || null, reason: r.reason || null,
                   rejections: r.rejections || null };
        }
        return { anchor: r, reason: null };
      };
      var p;
      try {
        p = verify(request);
      } catch (e) {
        settle(null, 'verifier_error');
        return;
      }
      if (p && typeof p.then === 'function') {
        p.then(function (r) { var u = unwrap(r); settle(u.anchor, u.reason, u.rejections); },
               function () { settle(null, 'verifier_error'); });
      } else {
        var u = unwrap(p);
        settle(u.anchor, u.reason, u.rejections);
      }
    }

    /* --- speech candidates ----------------------------------------------- */
    function ttlMs(callType) {
      var ttl = (route.timing && route.timing.speech_ttl_s) || {};
      return Math.round((ttl[callType] || 5.0) * 1000);
    }

    function speak(man, callType, text, anchor, snapshot) {
      if (!text) return false;
      // Duplicate suppression: the same sentence twice inside the cooldown is
      // RIO repeating itself, which reads as a fault even when it is not.
      var key = callType + '|' + text;
      if (lastSpoken[key] !== undefined &&
          (clock - lastSpoken[key]) < opt.duplicate_instruction_cooldown_s) {
        return false;
      }
      lastSpoken[key] = clock;

      var candidate = {
        text: text,
        maneuver_id: man.id,
        route_generation: route.generation_id,
        route_id: route.route_id,
        call_type: callType,
        anchor_id: anchor ? anchor.anchor_id : null,
        priority: (callType === CALL.IMMINENT) ? P.TURN_NEAR : P.NAV,
        created_at: clock,
        expires_at: clock + (ttlMs(callType) / 1000)
      };
      counters.candidates++;

      /* Deterministic validity, evaluated by the arbiter at DEQUEUE (§25).
         Four questions, all cheap, all answered from state that cannot lie:
         the same maneuver, the same route generation, not yet passed, not yet
         expired. */
      function valid() {
        if (stopped) return false;
        if (candidate.route_generation !== activeGeneration()) return false;
        if (clock > candidate.expires_at) return false;
        if (!tracker) return true;
        var active = tracker.maneuver ? tracker.maneuver() : null;
        if (!active || active.id !== candidate.maneuver_id) return false;
        if (tracker.isPassed && tracker.isPassed(candidate.maneuver_id)) return false;
        return true;
      }

      var item = audio ? audio(candidate) : { play: function () { return Promise.resolve(); } };
      var said = false;
      arbiter.say({
        priority: candidate.priority,
        group: 'nav:' + man.id,
        id: 'nav:' + man.id + ':' + callType,
        text: text,
        ttlMs: ttlMs(callType),
        meta: { maneuver_id: man.id, call_type: callType,
                anchor_id: candidate.anchor_id,
                route_generation: candidate.route_generation },
        valid: valid,
        play: item.play,
        stop: item.stop,
        onDone: function (reason) {
          if (said) return;
          said = true;
          if (reason === 'spoken') counters.spoken++;
          else if (reason === 'invalid') counters.invalidated++;
          else if (reason === 'expired') counters.expired++;
          emit(reason === 'invalid' ? EV.SPEECH_INVALIDATED
               : (reason === 'expired' ? EV.SPEECH_EXPIRED : EV.SPEECH_SPOKEN), {
            maneuver_id: man.id, call_type: callType, reason: reason, text: text,
            anchor_id: candidate.anchor_id
          });
        }
      });

      var payload = {
        maneuver_id: man.id, call_type: callType, text: text,
        tta_s: snapshot.tta_s, to_maneuver_m: snapshot.to_maneuver_m,
        gps_state: snapshot.gps_state, speed_ms: snapshot.speed_ms,
        anchor_id: candidate.anchor_id,
        anchor_label: anchor ? anchor.label : null,
        relation: anchor ? anchor.turn_relation_to_anchor : null
      };
      if (callType === CALL.EARLY) emit(EV.EARLY_GUIDANCE, payload);
      else if (callType === CALL.IMMINENT) emit(EV.NEAR_TURN, payload);
      else if (anchor) emit(EV.CONTEXTUAL_CALL, payload);
      else emit(EV.CONTEXTUAL_CALL, payload);   // canonical primary: same slot,
                                                // anchor_id null. One event, so
                                                // "how often was there context"
                                                // is a filter, not a join.
      return true;
    }

    /* --- the tick --------------------------------------------------------- */
    function onProgress(ev) {
      if (stopped) return;
      clock = ev.t;
      var man = manById(ev.maneuver_id);
      if (!man) return;
      var b = book(man.id);
      b.id = man.id;
      var snapshot = {
        tta_s: ev.tta_s, to_maneuver_m: ev.to_maneuver_m,
        gps_state: ev.gps_state, speed_ms: ev.speed_ms
      };
      var tta = ev.tta_s, dist = ev.to_maneuver_m;
      var bs = bias();
      var stationary = (ev.speed_ms || 0) < opt.stationary_speed_ms;

      if (man.type === 'ARRIVE') {
        // Arrival gets one call, and the side comes from the provider or is
        // simply not said. There is no camera path to this sentence.
        if (!b.called[CALL.ARRIVAL] &&
            (tta <= opt.context_call_s + bs || dist <= opt.min_call_distance_m)) {
          b.called[CALL.ARRIVAL] = true;
          speak(man, CALL.ARRIVAL,
                (man.speech && (man.speech.arrival || man.speech.primary)) || null,
                null, snapshot);
        }
        return;
      }

      // EARLY — optional, and skipped entirely if the drive started inside the
      // window. A preparation line for a turn that is already imminent is
      // noise on top of the instruction that matters.
      if (!b.called[CALL.EARLY] && man.speech && man.speech.early &&
          tta <= opt.early_guidance_s + bs && tta > opt.context_call_s + bs &&
          dist <= opt.early_max_distance_m && !stationary) {
        b.called[CALL.EARLY] = true;
        speak(man, CALL.EARLY, man.speech.early, null, snapshot);
      }

      // ACQUISITION — the camera is asked about an expected landmark, once the
      // maneuver is close enough that the landmark should be in view. This is
      // the only thing perception is ever asked, and nothing below waits on it.
      if (tta <= opt.anchor_acquisition_s + bs && b.context !== CTX.CALLED &&
          !b.anchor && man.anchors && man.anchors.length) {
        startAcquisition(man, b, snapshot);
      }

      // PRIMARY — the instruction. With an anchor if one is verified and still
      // valid; with the canonical sentence otherwise, which is not a fallback
      // in any apologetic sense: it is a complete instruction.
      if (!b.called[CALL.PRIMARY] &&
          (tta <= opt.context_call_s + bs || dist <= opt.min_call_distance_m) &&
          dist <= opt.max_call_distance_m) {
        if (tta <= opt.primary_min_lead_s + bs) {
          // Too late to begin a full instruction: say the short line at the
          // junction instead of one that would still be playing through it.
          b.called[CALL.PRIMARY] = true;
          b.primaryLate = true;
        } else {
          b.called[CALL.PRIMARY] = true;
          var anchor = anchorUsable(b) ? b.anchor : null;
          var text = null;
          if (anchor) {
            for (var i = 0; i < man.anchors.length; i++) {
              if (man.anchors[i].anchor_id === anchor.anchor_id) {
                text = man.anchors[i].speech;
                break;
              }
            }
            // The sentence must come from the route's own table. An anchor
            // that does not resolve to one is not spoken about.
            if (!text) anchor = null;
          }
          if (!text) text = man.speech && man.speech.primary;
          if (speak(man, CALL.PRIMARY, text, anchor, snapshot)) {
            b.primarySpokenAt = clock;
            if (anchor) b.context = CTX.CALLED;
          }
        }
      }

      // IMMINENT — the backup at the junction. It stays armed even when a
      // contextual call has already been spoken (§11C): "Turn right by the
      // Shell" explains the turn, "Right here" confirms it is this one. The
      // one case it is skipped is when the primary line was spoken moments
      // ago — at a crawl the distance clamp can bring both calls due within a
      // second of each other, and two instructions stacked back to back is
      // RIO talking over itself.
      if (!b.called[CALL.IMMINENT] && man.speech && man.speech.imminent &&
          (tta <= opt.near_turn_s + bs || dist <= opt.min_call_distance_m)) {
        b.called[CALL.IMMINENT] = true;
        var stacked = (b.primarySpokenAt !== undefined) &&
                      (clock - b.primarySpokenAt) < opt.imminent_min_gap_s;
        if (stacked) {
          emit(EV.NEAR_TURN, { maneuver_id: man.id, call_type: CALL.IMMINENT,
                               text: null, skipped: 'primary_just_spoken',
                               tta_s: snapshot.tta_s,
                               to_maneuver_m: snapshot.to_maneuver_m });
        } else {
          speak(man, CALL.IMMINENT, man.speech.imminent, null, snapshot);
        }
      }
    }

    function attach() {
      if (!tracker || !tracker.onEvent) return;
      tracker.onEvent(function (ev) {
        if (ev.type === 'NAV_PROGRESS') onProgress(ev);
        else if (ev.type === 'NAV_MANEUVER_PASSED') {
          var b = perMan[ev.maneuver_id];
          if (b && b.context !== CTX.CALLED) { b.context = CTX.EXPIRED; b.anchor = null; }
        } else if (ev.type === 'NAV_OFF_ROUTE_CONFIRMED') {
          // The route is wrong. Everything queued about it is wrong too.
          stopped = true;
          if (arbiter && arbiter.clear) arbiter.clear('nav:');
        }
      });
    }
    attach();

    return {
      CALL: CALL, CTX: CTX, EVENTS: EV,
      onEvent: function (fn) { if (typeof fn === 'function') listeners.push(fn); },
      /* Exposed for the dashboard and for tests; the planner drives itself
         from the tracker's events. */
      onProgress: onProgress,
      stop: function () { stopped = true; },
      contextState: function (maneuverId) {
        var b = perMan[maneuverId];
        return b ? b.context : CTX.INACTIVE;
      },
      anchorFor: function (maneuverId) {
        var b = perMan[maneuverId];
        return (b && b.anchor) || null;
      },
      state: function () {
        var man = tracker && tracker.maneuver ? tracker.maneuver() : null;
        var b = man ? perMan[man.id] : null;
        return {
          maneuver_id: man ? man.id : null,
          context_state: b ? b.context : CTX.INACTIVE,
          context_reason: b ? b.contextReason : null,
          anchor: b && b.anchor ? {
            label: b.anchor.label, type: b.anchor.type,
            relation: b.anchor.turn_relation_to_anchor,
            identity_confidence: b.anchor.identity_confidence,
            visibility_confidence: b.anchor.visibility_confidence,
            relation_confidence: b.anchor.relation_confidence,
            valid_until: b.anchor.valid_until
          } : null,
          calls: b ? Object.keys(b.called) : [],
          counters: counters
        };
      }
    };
  }

  root.RIO = root.RIO || {};
  root.RIO.navplan = { create: create, CALL: CALL, CTX: CTX, EVENTS: EV, DEFAULTS: DEFAULTS };

  if (typeof module !== 'undefined' && module.exports) module.exports = root.RIO.navplan;
})(typeof window !== 'undefined' ? window : globalThis);
