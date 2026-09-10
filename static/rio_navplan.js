/* rio_navplan.js — the navigation speech planner and the context lifecycle.
 *
 * The tracker says where the car is. This file decides what, if anything, is
 * worth saying about it, and hands the result to the one arbiter. It holds no
 * geometry and it cannot alter navigation truth: every sentence it can produce
 * was written by the server at route load and arrives in the route payload.
 *
 * THE CADENCE IS GOOGLE MAPS', AND IT IS A LADDER OF DISTANCES
 * ------------------------------------------------------------
 *   ROUTE START  immediately   "Head north on Lincoln Blvd, then turn right
 *                               onto Ocean Ave."
 *   FAR          0.5 mi surface / 2 mi highway
 *                              "In half a mile, turn right onto Ocean Ave."
 *   FAR_MID      1 mi, highway only
 *                              "In one mile, take exit 43 toward Sunset."
 *   NEAR         150 m surface / 400 m highway
 *                              "Turn right onto Ocean Ave."
 *   JUNCTION     35 m surface / 150 m highway, and ONLY if NEAR was >10 s ago
 *                              "Turn right."
 *   ARRIVAL      150 m         "Your destination is on the right."
 *   ARRIVED      at the kerb   "You have arrived."
 *
 * WHAT CHANGED, AND WHY IT HAD TO
 * -------------------------------
 * The tiers used to be timed in SECONDS TO THE TURN with metre floors under
 * them. On session 738fbb82 that produced, for one maneuver, in this order:
 *
 *     t=76.6   "Take the next right onto 14th St."   31.1 m    the instruction
 *     t=102.3  "Coming up on a right onto 14th St."  28.2 m    the PREPARATION
 *
 * The preparation line after the instruction and closer to the junction,
 * because at 0.12 m/s both floors were crossed before the route finished
 * loading. A speed-scaled floor is the wrong shape for this problem: what a
 * driver needs is not "n seconds of warning", it is the beat they already know
 * — half a mile, then the street name, then the confirmation.
 *
 * So every threshold here is now a DISTANCE, and every distance arrives with
 * the route (`maneuver.speech.tiers`), per maneuver, chosen by road class.
 * Nothing in this file holds a number that decides when a turn is called.
 *
 * THE >10 s RULE IS THE ONLY TIME TERM LEFT, and it is not a threshold — it is
 * a suppression. "Turn right onto Ocean Ave." at 150 m and "Turn right." at
 * 35 m are two instructions about the same turn; whether the second is useful
 * or is RIO talking over herself depends entirely on how long ago the first
 * was. In town that gap is ten seconds and the confirmation is wanted. On a
 * freeway the same two distances are four seconds apart and it is noise.
 *
 * "THEN" CHAINING. Two junctions inside NAV_CHAIN_WINDOW_M are one move to a
 * driver, so the server writes the near call as "Turn right onto Ocean, then
 * turn left onto 2nd." and marks the second maneuver `chained_to`. This file
 * then gives that second maneuver NO far and NO near call of its own — it has
 * already been announced — and keeps only its junction confirmation.
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
 * unaffected and the near call goes out with the canonical sentence. That
 * separation is what makes "vision is optional" structural rather than
 * aspirational: there is no path by which a perception failure delays or
 * suppresses a navigation instruction.
 *
 * VALIDITY IS CHECKED WHEN THE LINE IS ABOUT TO BE SPOKEN
 * ------------------------------------------------------
 * Not when it is queued. A "Turn right." created 2 s before a junction and
 * dequeued 3 s later, after a safety warning finished, is a lie about a turn
 * the car has already taken. Every candidate carries a `valid()` the arbiter
 * calls at dequeue: right maneuver, right route generation, not passed, not
 * expired. Invalid lines are dropped silently — never spoken late, never
 * "caught up".
 *
 * SHE IS THE NAVIGATION, AND THESE ARE HER WORDS. Nothing in this file, and
 * nothing in the table it reads from, may describe navigation in the third
 * person — no "the car will call it out", no "the system". The turn calls ARE
 * RIO; the fact that a deterministic planner fires them is architecture, not
 * something the driver is told about. Enforced by the nav-voice lint in
 * tools/nav_server_selftest.py and tools/nav_selftest.js.
 */
(function (root) {
  'use strict';

  var CALL = { DEPART: 'depart', FAR: 'far', FAR_MID: 'far_mid', NEAR: 'near',
               JUNCTION: 'junction', ARRIVAL: 'arrival', ARRIVED: 'arrived' };
  // The tiers that are fired by crossing a distance, outermost first. ARRIVAL
  // is on this list too -- it is the ARRIVE maneuver's only distance tier.
  var DISTANCE_TIERS = ['far', 'far_mid', 'near', 'junction', 'arrival'];
  var CTX = {
    INACTIVE: 'INACTIVE', ACQUIRING: 'ACQUIRING', VERIFIED: 'VERIFIED',
    CALLED: 'CALLED', EXPIRED: 'EXPIRED'
  };
  var EV = {
    ROUTE_START_CALL: 'NAV_ROUTE_START_CALL',
    FAR_GUIDANCE: 'NAV_FAR_GUIDANCE',
    NEAR_CALL: 'NAV_NEAR_CALL',
    JUNCTION_CALL: 'NAV_JUNCTION_CALL',
    ARRIVAL_CALL: 'NAV_ARRIVAL_CALL',
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
    /* How far BEFORE the near call the camera is asked about the landmark.
       A distance, like every other threshold here: the anchor has to be in
       hand before the sentence it belongs to is spoken, and a window measured
       in seconds is inside the near call at a crawl. */
    anchor_acquisition_lead_m: 150.0,
    /* THE TIER DISTANCES DO NOT LIVE HERE. Every one of them arrives per
       maneuver on `speech.tiers`, chosen server-side by road class -- see
       config.NAV_TIER_DISTANCES_M. What is left in this table is the fallback
       ladder for a route built before tiers existed, and the rules that are
       about arbitration rather than about distance. */
    fallback_tiers: [{ call: 'far', at_m: 804.7 },
                     { call: 'near', at_m: 150.0 },
                     { call: 'junction', at_m: 35.0 }],
    /* Clamps. A call is not made from further out than its ladder allows even
       if a tier says so, and never from inside the junction. */
    min_call_distance_m: 20.0,
    far_max_distance_m: 4000.0,

    /* THE ONLY TIME TERM LEFT, and it suppresses rather than fires.
       "Turn right onto Ocean Ave." then "Turn right." is a confirmation in
       town and an interruption on a freeway, and the difference is entirely
       how many seconds apart the two landed. */
    junction_min_gap_s: 10.0,
    /* A FULL INSTRUCTION TAKES ABOUT TWO SECONDS TO SAY, and a sentence still
       playing when the driver has to act is worse than a shorter one that
       finished. Inside this lead the near call is skipped and the junction
       gets the two-word line instead -- never a longer line begun too late.
       It is also what makes the junction call fire at all in that case: the
       >10 s rule asks how long ago the near call was, and the honest answer
       here is that there was no room for one. */
    near_min_lead_s: 4.0,

    /* THE LEAD, so the junction call fires on the tick BEFORE the crossing.
       Fallbacks only; the real values arrive with the route. See
       config.NAV_PROGRESS_TICK_S and NAV_CLIP_START_LATENCY_S. */
    progress_tick_s: 0.5,
    clip_start_latency_s: 0.05,
    junction_lead_margin_m: 2.0,
    /* GPS we do not trust is a reason to give the driver MORE room, so a
       degraded or stale fix widens every tier by what the car covers in this
       many seconds. Distance, now, rather than a second added to a second. */
    gps_degraded_bias_s: 2.0,
    stationary_speed_ms: 0.7,
    duplicate_instruction_cooldown_s: 8.0,
    arrival_call_m: 150.0,
    anchor_valid_for_m: 400.0,
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
    var started = false;          // has the route-start line gone out?
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
      if (!st) return 0;
      /* GPS degraded near a maneuver: speak EARLIER, never later. A fix we do
         not trust is a reason to give the driver more room, not less.

         GPS_STALE gets the same widening, and that is a change the first real
         drive forced. A stale fix used to mean the planner was not ticked at
         all -- the tracker only emitted progress when a fix arrived -- so
         "what should we say while stale" had no answer because nothing was
         being said. The tracker dead-reckons now (rio_navcore.tick), so the
         question is live, and the answer is the same one degraded gets: room,
         not silence. */
      return (st.gps_state === 'GPS_DEGRADED' || st.gps_state === 'GPS_STALE')
        ? opt.gps_degraded_bias_s : 0;
    }

    /* THE SAME WIDENING, IN METRES. Every threshold in this file is a distance
       now, so a bias expressed in seconds has to be turned into road at the
       speed the car is doing. At a standstill it collapses to nothing, which
       is right: a stale fix on a stationary car is not a reason to call a turn
       early, it is a reason to call nothing. */
    function biasM(speedMs) {
      var b = bias();
      return b > 0 ? Math.max(0, speedMs || 0) * b : 0;
    }

    /* The tier ladder for one maneuver: the server's, or the fallback. */
    function tiersOf(man) {
      var t = man && man.speech && man.speech.tiers;
      return (t && t.length) ? t : opt.fallback_tiers;
    }

    /* Was this maneuver already announced as the tail of the previous one's
       near call -- "turn right onto Ocean, THEN turn left onto 2nd"? If so it
       has had its instruction and gets no far or near call of its own. */
    function announcedByChain(man) {
      var list = route.maneuvers || [];
      for (var i = 0; i < list.length; i++) {
        if (list[i].speech && list[i].speech.chained_to === man.id) {
          var b = perMan[list[i].id];
          return !!(b && b.called[CALL.NEAR]);
        }
      }
      return false;
    }

    /* --- the anchor ------------------------------------------------------ */
    /* HOW FAR THE CAR HAS COME SINCE THE ANCHOR WAS VERIFIED, which is what
       actually makes a visual observation stale. `valid_from_m` is the
       distance to the maneuver at the moment the camera answered; the car has
       travelled the difference. */
    function anchorUsable(b, toManeuverM) {
      if (!b.anchor) return false;
      var travelled = (b.anchor.valid_from_m === undefined
                       || toManeuverM === null || toManeuverM === undefined)
        ? 0 : (b.anchor.valid_from_m - toManeuverM);
      if (travelled > (b.anchor.valid_for_m || opt.anchor_valid_for_m)) {
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
          anchor.valid_from_m = snapshot.to_maneuver_m;
          anchor.valid_for_m = anchor.valid_for_m || opt.anchor_valid_for_m;
          b.anchor = anchor;
          b.context = CTX.VERIFIED;
          counters.anchors_verified++;
          emit(EV.ANCHOR_VERIFIED, {
            maneuver_id: man.id, anchor_id: anchor.anchor_id, label: anchor.label,
            type: anchor.type, relation: anchor.turn_relation_to_anchor,
            identity_confidence: anchor.identity_confidence,
            visibility_confidence: anchor.visibility_confidence,
            relation_confidence: anchor.relation_confidence,
            valid_from_m: anchor.valid_from_m,
            valid_for_m: anchor.valid_for_m
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

    function speak(man, callType, text, anchor, snapshot, o) {
      o = o || {};
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
        // THE PRE-RENDERED FILE FOR THIS EXACT SENTENCE, if the server named
        // one. Only the imminent call has one, and only because it is the one
        // call that names no road: a closed set of four sentences is a set
        // that can be rendered once and played off disk at the junction. The
        // browser is TOLD which file, next to the words, so the two cannot
        // come to disagree about what the audio says.
        clip: (man.speech && man.speech.clips
               && man.speech.clips[callType]) || null,
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
        // Arrival is exempt from the rest, and only arrival. See arrivalCall:
        // once the tracker has arrived there is no active maneuver for the
        // other three questions to be about, and asking them would drop the
        // one line the whole route was for.
        if (o.arrival) return true;
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
        /* WHERE THE TIER SAID TO CALL IT, next to where the car actually was.
           The whole of the timing review is `to_maneuver_m` against `at_m`,
           and a log that carries only the first cannot say whether the ladder
           was followed or the car simply happened to be there. */
        at_m: (o.at_m === undefined ? null : o.at_m),
        road_class: man.road_class || null,
        tta_s: snapshot.tta_s, to_maneuver_m: snapshot.to_maneuver_m,
        gps_state: snapshot.gps_state, speed_ms: snapshot.speed_ms,
        anchor_id: candidate.anchor_id,
        anchor_label: anchor ? anchor.label : null,
        relation: anchor ? anchor.turn_relation_to_anchor : null
      };
      /* ONE EVENT PER TIER, named for the tier. NAV_CONTEXTUAL_CALL stays the
         name of the near call whether or not it carried an anchor -- "how
         often was there context" is then a filter on anchor_id rather than a
         join across two event types. */
      if (callType === CALL.DEPART) emit(EV.ROUTE_START_CALL, payload);
      else if (callType === CALL.FAR || callType === CALL.FAR_MID) emit(EV.FAR_GUIDANCE, payload);
      else if (callType === CALL.JUNCTION) emit(EV.JUNCTION_CALL, payload);
      else if (callType === CALL.ARRIVAL || callType === CALL.ARRIVED) emit(EV.ARRIVAL_CALL, payload);
      else emit(EV.CONTEXTUAL_CALL, payload);
      return true;
    }

    /* THE ARRIVAL SENTENCES, and the one thing about them that is different.
     *
     * Their validity cannot ask "is this still the active maneuver", because
     * by the time the tracker says ARRIVED there is no active maneuver --
     * manIdx has run off the end, tracker.maneuver() is null, and the ordinary
     * valid() would drop the line at dequeue every single time. So they are
     * validated on the two things that still mean something: the drive has not
     * been stopped, and this is still the route we are on.
     *
     * TWO OF THEM, because Google says two and they say different things.
     * "Your destination is on the right." at 150 m is a lane instruction --
     * it is the last thing that changes what the driver does. "You have
     * arrived." at the kerb changes nothing and confirms everything.
     */
    function arrivalCall(man, callType, snapshot, atM) {
      var b = book(man.id);
      if (b.called[callType]) return false;
      var text = (man.speech && man.speech[callType]) || null;
      if (!text) return false;
      b.called[callType] = true;
      // The tier the route carries, not this file's fallback: on a route whose
      // last leg is 50 m the arrival call is due at 50 m, and logging it
      // against a 150 m default would read as a call 118 m late.
      return speak(man, callType, text, null, snapshot || {
        tta_s: 0, to_maneuver_m: 0, gps_state: null, speed_ms: null
      }, { arrival: true, at_m: (atM === undefined ? null : atM) });
    }

    /* The tracker has decided the drive is over. It emits NAV_ARRIVED and
     * RETURNS, before any progress event for the ARRIVE maneuver -- so until
     * this existed, the arrival sentence was written by the server, carried to
     * the browser in the route, and unreachable by any path. Measured: session
     * 06af3214, NAV_ARRIVED at t=600.2, nothing spoken. */
    function onArrived() {
      var list = route.maneuvers || [];
      var man = null;
      for (var i = list.length - 1; i >= 0; i--) {
        if (list[i].type === 'ARRIVE') { man = list[i]; break; }
      }
      // A route with no ARRIVE maneuver still arrives; the last maneuver's
      // arrival line is the sentence, if the server wrote one.
      if (!man && list.length) man = list[list.length - 1];
      if (!man) return;
      arrivalCall(man, CALL.ARRIVED, { tta_s: 0, to_maneuver_m: 0,
                                       gps_state: null, speed_ms: null }, 0);
    }

    /* THE ROUTE-START LINE, said once and said IMMEDIATELY.
     *
     * The one tier the old cadence had no slot for at all. Session 738fbb82
     * started a route at t=76.1 and the first thing the driver heard, half a
     * second later, was a turn call at 31 m; nothing ever told them what road
     * they were on or which way they were pointing. Google says the whole
     * first move before the car has left the kerb, and so does this.
     *
     * Not gated on distance, speed or GPS state: at route start there is no
     * approach yet, and the sentence is about the plan rather than about a
     * junction. The only gate is "once".
     */
    function onRouteStart() {
      if (started || stopped) return false;
      started = true;
      var text = route.depart_speech || '';
      if (!text) return false;
      var list = route.maneuvers || [];
      var man = null;
      for (var i = 0; i < list.length; i++) {
        if (list[i].type !== 'ARRIVE' && list[i].type !== 'DEPART') { man = list[i]; break; }
      }
      if (!man) man = list[0];
      if (!man) return false;
      return speak(man, CALL.DEPART, text, null,
                   { tta_s: null, to_maneuver_m: null, gps_state: null, speed_ms: null },
                   { arrival: true, at_m: null });
    }

    /* --- the tick ---------------------------------------------------------
     *
     * ONE LOOP OVER THE LADDER, and the whole of the timing policy is in it.
     * A tier fires when the car is inside its distance and has not fired it
     * before; the widening for an untrusted fix is metres of road rather than
     * seconds; and the two rules that are not about distance -- the junction
     * gap and the "then" chain -- are stated where they apply.
     */
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
      var dist = ev.to_maneuver_m;
      var speedMs = Math.max(0, ev.speed_ms || 0);
      var extra = biasM(speedMs);
      var stationary = speedMs < opt.stationary_speed_ms;

      // ACQUISITION — the camera is asked about an expected landmark, once the
      // maneuver is close enough that the landmark should be in view. This is
      // the only thing perception is ever asked, and nothing below waits on it.
      var nearAt = 0;
      var tiersNow = tiersOf(man);
      for (var q = 0; q < tiersNow.length; q++) {
        if (tiersNow[q].call === CALL.NEAR) nearAt = tiersNow[q].at_m;
      }
      if (man.type !== 'ARRIVE' && b.context !== CTX.CALLED && !b.anchor &&
          man.anchors && man.anchors.length && !b.called[CALL.NEAR] &&
          dist <= nearAt + opt.anchor_acquisition_lead_m + extra) {
        startAcquisition(man, b, snapshot);
      }

      var chainSuppressed = (man.type !== 'ARRIVE') && announcedByChain(man);
      var tiers = tiersOf(man);

      for (var i = 0; i < tiers.length; i++) {
        var call = tiers[i].call;
        var atM = tiers[i].at_m;
        if (b.called[call]) continue;
        if (DISTANCE_TIERS.indexOf(call) < 0) continue;

        /* THE THRESHOLD, widened two ways and clamped one.
           - `extra` is the untrusted-fix widening, in road.
           - the junction call alone is LED by one tick of travel plus the clip
             start, because a threshold is only ever CHECKED on a tick and
             without the lead "Turn right." arrives 6-11 m late -- measured, on
             a live route at 25 mph, against a 35 m floor. Late is the one
             direction this call must not be wrong in. */
        var threshold = atM + extra;
        if (call === CALL.JUNCTION) {
          var leadS = (opt.progress_tick_s || 0) + (opt.clip_start_latency_s || 0);
          threshold += speedMs * leadS + (opt.junction_lead_margin_m || 0);
        }
        if (call === CALL.FAR || call === CALL.FAR_MID) {
          if (threshold > opt.far_max_distance_m) threshold = opt.far_max_distance_m;
        }
        if (dist > threshold) continue;

        /* A FAR CALL'S TEXT CONTAINS A DISTANCE, and it was written at route
           load from the tier rather than from the car. Said at 40 m, "In half
           a mile, turn left onto Lincoln Boulevard." is not a rounding, it is
           false. So a far tier the car is already well past is marked done and
           never spoken -- which is what happens whenever a route is set from
           inside its own first approach. */
        if ((call === CALL.FAR || call === CALL.FAR_MID) && dist <= nearAt) {
          b.called[call] = true;
          continue;
        }

        /* A tier crossed while the car is stopped is a tier that will be
           crossed again when it moves. The exception is the junction call and
           the arrival call: a car stopped 30 m from its turn is at a light,
           and it still wants to be told which way. */
        if (stationary && (call === CALL.FAR || call === CALL.FAR_MID)) continue;

        if (call === CALL.ARRIVAL) {
          // arrivalCall() owns the `called` flag: setting it here first would
          // make its own idempotence guard reject the only call it ever gets.
          arrivalCall(man, CALL.ARRIVAL, snapshot, atM);
          continue;
        }

        /* A CHAINED MANEUVER HAS ALREADY HAD ITS INSTRUCTION. "Turn right onto
           Ocean, then turn left onto 2nd" is one sentence covering two
           junctions; saying "turn left onto 2nd" again forty metres later is
           the car repeating itself at the worst possible moment. It keeps its
           junction confirmation, which is the half that is still news. */
        if (chainSuppressed && (call === CALL.FAR || call === CALL.FAR_MID
                                || call === CALL.NEAR)) {
          b.called[call] = true;
          continue;
        }

        if (call === CALL.NEAR) {
          b.called[call] = true;
          /* TOO LATE TO BEGIN A FULL INSTRUCTION. Say the short line at the
             junction instead of one that would still be playing through it. */
          if (speedMs > 0.5 && (dist / speedMs) < opt.near_min_lead_s) {
            b.nearTooLate = true;
            emit(EV.CONTEXTUAL_CALL, {
              maneuver_id: man.id, call_type: CALL.NEAR, text: null,
              skipped: 'no_room_to_finish', at_m: atM,
              road_class: man.road_class || null,
              tta_s: snapshot.tta_s, to_maneuver_m: snapshot.to_maneuver_m
            });
            continue;
          }
          // With an anchor if one is verified and still valid; with the
          // canonical sentence otherwise, which is not a fallback in any
          // apologetic sense: it is a complete instruction.
          var anchor = anchorUsable(b, dist) ? b.anchor : null;
          var text = null;
          if (anchor) {
            for (var k = 0; k < man.anchors.length; k++) {
              if (man.anchors[k].anchor_id === anchor.anchor_id) {
                text = man.anchors[k].speech;
                break;
              }
            }
            // The sentence must come from the route's own table. An anchor
            // that does not resolve to one is not spoken about.
            if (!text) anchor = null;
          }
          if (!text) text = man.speech && man.speech[CALL.NEAR];
          if (speak(man, CALL.NEAR, text, anchor, snapshot, { at_m: atM })) {
            b.nearSpokenAt = clock;
            /* A CHAINED SENTENCE IS THE NEXT MANEUVER'S NEAR CALL TOO.
               "...then turn right onto Fell Street" is when the driver was
               told about m1, so it is the instant m1's junction call measures
               its ten seconds from. Without this the chained maneuver looks
               like one that was never instructed, and its confirmation is
               dropped as `no_near_call` -- the one turn on the route that most
               needs confirming, because its instruction arrived early and
               attached to something else. */
            var chainId = man.speech && man.speech.chained_to;
            if (chainId) book(chainId).nearSpokenAt = clock;
            if (anchor) b.context = CTX.CALLED;
          }
          continue;
        }

        if (call === CALL.JUNCTION) {
          b.called[call] = true;
          /* ...AND ONLY IF THE NEAR CALL WAS LONG ENOUGH AGO.
             The near call already named the road. This one exists to catch the
             driver who is looking for a street sign they cannot read; inside
             ten seconds of the instruction it is not a catch, it is RIO
             talking over herself. Never spoken at all when the near call has
             not happened yet -- a bare "Turn right." with no road named is not
             an instruction. */
          var gap = (b.nearSpokenAt === undefined)
            ? null : (clock - b.nearSpokenAt);
          // ...unless there was no room for a near call at all, in which case
          // this two-word line is the whole instruction and must go out.
          if (b.nearTooLate) gap = Infinity;
          if (gap === null || gap < opt.junction_min_gap_s) {
            emit(EV.JUNCTION_CALL, {
              maneuver_id: man.id, call_type: CALL.JUNCTION, text: null,
              skipped: gap === null ? 'no_near_call' : 'near_call_too_recent',
              since_near_s: gap === null ? null : Math.round(gap * 10) / 10,
              at_m: atM, road_class: man.road_class || null,
              tta_s: snapshot.tta_s, to_maneuver_m: snapshot.to_maneuver_m
            });
            continue;
          }
          var jt = man.speech && man.speech[CALL.JUNCTION];
          if (jt) speak(man, CALL.JUNCTION, jt, null, snapshot, { at_m: atM });
          continue;
        }

        // FAR and FAR_MID: the distance-phrased preparation lines. Their text
        // already contains the rounded distance, written at route load.
        b.called[call] = true;
        var ft = man.speech && man.speech[call];
        if (ft) speak(man, call, ft, null, snapshot, { at_m: atM });
      }
    }

    function attach() {
      if (!tracker || !tracker.onEvent) return;
      tracker.onEvent(function (ev) {
        if (ev.type === 'NAV_PROGRESS') onProgress(ev);
        else if (ev.type === 'NAV_ARRIVED') onArrived();
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
      onArrived: onArrived,
      /* Called by the glue the moment a route becomes the live one. Not driven
         off a progress event: at route start there has not been one yet, and
         waiting for one is how the driver ends up hearing a turn call before
         they have been told what road they are on. */
      onRouteStart: onRouteStart,
      started: function () { return started; },
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
            valid_from_m: b.anchor.valid_from_m,
            valid_for_m: b.anchor.valid_for_m
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
