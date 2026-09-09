/* rio_realtime.js — RIO live. One session: her ears, her brain, her voice.
 *
 * The browser holds the audio connection to the model directly, over WebRTC.
 * That is the whole reason this is a conversation rather than a sequence of
 * turns: the driver's microphone is open, RIO's audio comes back as a live
 * track, and either of them can cut the other off mid-word. Routing that audio
 * through the server would put a round trip on both halves of every sentence.
 *
 * WHAT THIS FILE IS NOT ALLOWED TO DO
 * ----------------------------------
 * Speak over a warning. RIO's live audio is CONVERSATION, which is the lowest
 * tier on the arbiter's ladder and the tier that yields — a closing gap, a
 * tire losing air and a turn four seconds out all cut straight through her,
 * exactly as they cut through the old recorded replies. So every spoken
 * response claims the mouth through the arbiter like everything else, and when
 * the arbiter takes it away this file does two things at once: mutes the
 * element (instant, for the audio already in flight) and tells the model to
 * stop generating (so she does not resume into the gap).
 *
 * THE SPLIT, AND WHY
 * ------------------
 *   createController()  every decision — arbitration, barge-in, the tool
 *                       bridge — as a pure event handler over an injected
 *                       transport. No DOM, no WebRTC, node-testable.
 *   connect()           the WebRTC and getUserMedia wiring that builds a real
 *                       transport and hands it to the controller.
 *
 * That split is not architecture for its own sake: the interesting failures
 * here are "a warning arrived while she was mid-sentence" and "the tool timed
 * out", and neither of those is reachable in a test that needs a microphone.
 */
(function (root) {
  'use strict';

  var CALLS_URL = 'https://api.openai.com/v1/realtime/calls';

  /* Echo cancellation is not optional in a car. RIO's voice comes out of the
     same box the microphone is in, so without it the loudest thing the
     detector hears while she is talking is her -- she interrupts herself, and
     the driver watches her abandon an answer nobody asked her to stop. AGC and
     noise suppression are the same argument for road noise and wind.
     
     Named rather than inline so the selftest can assert on them: this is a
     three-word constraint object whose absence is invisible until you are in a
     moving car, which is exactly the kind of thing that gets dropped in a
     refactor and not noticed for a month. */
  var MIC_CONSTRAINTS = {
    echoCancellation: true,
    noiseSuppression: true,
    autoGainControl: true,
  };

  /* ---------------------------------------------------------------------
     Tools answered HERE, in the panel, rather than on the server.

     nav_status is one of them, and it has to be: the route comes from the
     server but PROGRESS along it does not. The tracker runs in this page, at
     1 Hz, with no network — that was a deliberate choice about announcement
     timing and it has the side effect that the server does not know how far
     the next turn is. Answering from a second source on the server would
     produce two answers to one question, and the wrong one would be the one
     that sounded authoritative.

     So this reads exactly what the dashboard's own nav card reads.
     --------------------------------------------------------------------- */
  function navStatus() {
    var nav = root.RIO && root.RIO.nav;
    var st = (nav && nav.state) ? nav.state() : null;
    if (!st) {
      return { ok: true, routing: false,
               note: 'no route is set — say that, do not guess a destination' };
    }
    var man = st.maneuver || null;
    var out = {
      ok: true,
      routing: true,
      destination: (st.destination && (st.destination.display_name ||
                                       st.destination.formatted_address)) || null,
      distance_remaining_m: Math.round(st.remaining_m || 0),
      eta_epoch: st.eta_epoch || null,
      minutes_remaining: (st.remaining_m && st.speed_ms)
        ? Math.round((st.remaining_m / Math.max(3, st.speed_ms)) / 60) : null,
      next_maneuver: man ? {
        instruction: man.instruction,
        direction: man.direction,
        road_name: man.road_name,
        distance_m: st.to_maneuver_m === null ? null : Math.round(st.to_maneuver_m),
        seconds_away: st.tta_s === null ? null : Math.round(st.tta_s),
        state: st.maneuver_state,
      } : null,
      maneuvers_left: st.maneuvers_left,
      route_state: st.route_state,        // ON_ROUTE / OFF_ROUTE_* — a real answer
      gps_state: st.gps_state,            // ...and whether we can trust any of it
      arrived: !!st.arrived,
      // The boundary, restated where it is about to be tempting: a tool result
      // showing a turn four seconds out is context for answering, never a cue.
      rules: 'Answer the question. Do NOT announce this maneuver — the ' +
             'navigation system calls it out loud itself.',
    };
    if (st.context && st.context.anchor) {
      out.next_maneuver_landmark = st.context.anchor.label;
    }
    return out;
  }

  /* nav_directions — the turn-by-turn, because she was asked for it.
   *
   * nav_status answers "what is the next turn", which is the question an
   * announcement needs. "What are the directions?" is a different question and
   * RIO could not answer it: the maneuver list was in the tracker the whole
   * time and nothing exposed it, so she said she could not read them. She
   * could. Nothing was stopping her but the shape of the tools.
   *
   * READING IS ANSWERING. The boundary this sits next to is about ANNOUNCING —
   * calling a turn at the moment it needs calling, over the driver, from a
   * tool result that happened to mention it. That is the arbiter's job and
   * still is. Reading the route out when the driver asks for it is the
   * opposite: it is a question with an answer, and refusing to answer it was
   * never the point of the rule.
   *
   * Landmarks ride along where the map found one, and they are marked as
   * EXPECTATIONS rather than sightings. A landmark in the route is a candidate
   * from a places lookup; the thing that turns one into "there's the Shell,
   * go left after it" is the visual verifier at the junction, and that has not
   * run yet when the directions are being read. "There should be a Shell" is
   * true at this point. "There's a Shell" is not.
   */
  var REL_PHRASE = {
    NEAR: 'by', JUST_AFTER: 'just after', JUST_BEFORE: 'just before',
  };

  function landmarkOf(man) {
    // The best candidate is the first: landmarks.py sorts by salience, then
    // relation confidence, then distance, and truncates. Taking anything else
    // would be second-guessing a ranking made with more information.
    var a = (man && man.anchors && man.anchors.length) ? man.anchors[0] : null;
    if (!a || !a.label) return null;
    var rel = REL_PHRASE[a.relation] || 'near';
    return {
      label: a.spoken_label || a.label,
      relation: a.relation,
      // The fragment, not a sentence: RIO writes the sentence. Handing her the
      // prepared line (a.speech, "Turn left just after the Shell.") would be
      // handing her the ANNOUNCEMENT, which is exactly the thing she must not
      // say on her own initiative.
      phrase: rel + ' ' + (a.spoken_label || a.label),
      confidence: a.relation_confidence,
      verified: false,
    };
  }

  function navDirections(args) {
    var nav = root.RIO && root.RIO.nav;
    if (!nav || typeof nav.directions !== 'function') {
      return { ok: true, routing: false,
               note: 'no route is set — say that, do not invent turns' };
    }
    // "all" (or 0, or anything unreadable as a positive number) means the whole
    // route. A model asked for "the directions" says "all" as often as it says
    // nothing, and both mean the same thing.
    var raw = args && (args.count !== undefined ? args.count : args.n);
    var count = 5;
    if (typeof raw === 'string' && /^all$/i.test(raw.trim())) count = null;
    else if (raw !== undefined && raw !== null && raw !== '') {
      var n = parseInt(raw, 10);
      count = (isFinite(n) && n > 0) ? Math.min(n, 40) : null;
    }

    var d = nav.directions(count);
    if (!d) {
      return { ok: true, routing: false,
               note: 'no route is set — say that, do not invent turns' };
    }
    var mans = d.maneuvers || [];
    var steps = mans.map(function (m, i) {
      var step = {
        step: i + 1,
        instruction: m.instruction,
        road_name: m.road_name,
        maneuver_type: m.maneuver_type,
        direction: m.direction,
        // From where the car is now, which is what "how far to it" means when
        // the question is asked mid-drive.
        distance_m: m.distance_m,
        // ...and from the previous step, which is what makes a list of turns
        // read as directions rather than as a table.
        leg_m: m.leg_m,
      };
      var lm = landmarkOf(m);
      if (lm) step.landmark = lm;
      return step;
    });

    return {
      ok: true,
      routing: true,
      destination: (d.destination && (d.destination.display_name ||
                                      d.destination.formatted_address)) || null,
      distance_remaining_m: Math.round(d.remaining_m || 0),
      eta_epoch: d.eta_epoch || null,
      total_maneuvers: d.total_maneuvers,
      maneuvers_left: mans.length,
      truncated: count !== null && d.total_maneuvers > steps.length,
      route_state: d.route_state,
      gps_state: d.gps_state,
      arrived: !!d.arrived,
      steps: steps,
      rules: 'The driver asked for these, so read them — that is answering, ' +
             'not announcing. In your own voice and in one flowing sentence ' +
             'or two, not as a numbered list: name the roads, round the ' +
             'distances the way a person would, and stop after the first few ' +
             'unless they asked for all of it. A landmark here is what the ' +
             'MAP expects, not something anyone has seen: say "there should ' +
             'be a Shell", never "there\'s a Shell". Do NOT call any of ' +
             'these turns as instructions now — when each one arrives the ' +
             'navigation system calls it itself.',
    };
  }

  /* start_navigation — the driver said "take me there", so she takes them.
   *
   * This is answered in the panel for the same reason nav_status is, plus a
   * stronger one. The route is loaded by RIO.nav and tracked by RIO.nav; a
   * server-side version of this could resolve a destination but could not
   * make the car navigate to it, so it would have to hand the browser a
   * result and hope — which is exactly the arrangement that ended with RIO
   * describing a route and then asking the driver to set it themselves.
   *
   * Nothing about routing is written here. `RIO.nav.routeToQuery` is the
   * function the destination box calls when the driver presses Enter: the
   * same resolution, the same "which Getty?" question, the same /nav/route
   * call, the same tracker and planner attached to the same bus. A spoken
   * destination and a typed one are the same event by the time they get past
   * this line, and the only thing added here is the shape RIO needs to be
   * able to speak about what happened.
   */
  function startNavigation(args) {
    var nav = root.RIO && root.RIO.nav;
    if (!nav || typeof nav.routeToQuery !== 'function') {
      return Promise.resolve({ ok: false, note: 'no navigation on this page' });
    }
    var text = String((args && (args.destination || args.query)) || '').trim();
    var placeId = String((args && args.place_id) || '').trim();
    if (!text && !placeId) {
      return Promise.resolve({ ok: false, note: 'no destination' });
    }
    // Announcement audio is unlocked inside a user gesture everywhere else on
    // this page. A route RIO starts has no gesture behind it — the gesture was
    // starting the conversation — so it is unlocked here, before the first
    // turn call needs it rather than at it.
    if (typeof nav.unlock === 'function') { try { nav.unlock(); } catch (e) {} }

    /* A place_id means find_places already resolved this exact place, so the
     * resolution step is skipped entirely: setRoute takes the id the way the
     * autocomplete list does when the driver taps a suggestion.
     *
     * This is not only a saved call. Re-resolving "Blue Bottle" as text can
     * land on a different branch three miles the other way, and the driver
     * would have no way to tell -- they asked for the one RIO just read out,
     * with the rating and the four-minute drive, and would be taken somewhere
     * else with the right name.
     */
    var started = (placeId && typeof nav.setRoute === 'function')
      ? Promise.resolve(nav.setRoute({ place_id: placeId, label: text }))
          .then(function (res) {
            if (res && res.ok) {
              return { status: 'routed',
                       destination: (res.route && res.route.destination) || null,
                       route: res.route };
            }
            return { status: 'failed',
                     error: (res && res.error) || 'could not build a route' };
          })
      : Promise.resolve(nav.routeToQuery(text));

    return started.then(function (out) {
      out = out || {};
      // Into the drive's log, through the bus every other nav event uses: a
      // review of this drive should be able to see that the route was set by
      // voice, what was asked for, and what came of it.
      var bus = root.RIO && root.RIO.bus;
      if (bus && bus.emit) {
        try {
          bus.emit('NAV_VOICE_DESTINATION',
                   { query: text, status: out.status || 'unknown',
                     // Which path resolved it, because "she routed to the
                     // place she had just read out" and "she re-resolved a
                     // name" are different events in a review.
                     place_id: placeId || null,
                     from_places: !!placeId });
        } catch (e) { /* logging must never break a route */ }
      }

      if (out.status === 'routed') {
        var route = out.route || {};
        var dest = out.destination || route.destination || {};
        // The first turns, taken from the ROUTE THIS CALL JUST LOADED rather
        // than from the tracker.
        //
        // The tracker is already attached by the time this resolves — setRoute
        // calls attach() before it returns — so a nav_status or a
        // nav_directions in the same turn does see the route. But "already
        // attached" and "has had a GPS fix" are different things, and until
        // the first fix lands the tracker's distances are measured from the
        // start of the route rather than from the car. Carrying the summary in
        // this result means the confirmation RIO speaks needs no second call
        // and cannot race anything: it is the route she just started, as the
        // provider described it.
        var mans = route.maneuvers || [];
        var first = mans.slice(0, 3).map(function (m, i) {
          var step = {
            step: i + 1, instruction: m.instruction, road_name: m.road_name,
            maneuver_type: m.type, direction: m.direction,
            // From the START of the route here, not from the car: this is the
            // route as loaded, before anyone has driven any of it.
            distance_from_start_m: Math.round(m.route_distance_position || 0),
          };
          var lm = landmarkOf(m);
          if (lm) step.landmark = lm;
          return step;
        });
        return {
          ok: true, routing: true, status: 'routed',
          // The provider's own spelling of the place, not the driver's and
          // not the transcriber's. This is the word she repeats back.
          destination: dest.display_name || dest.formatted_address || text,
          minutes: route.duration_s
            ? Math.max(1, Math.round(route.duration_s / 60)) : null,
          distance_km: route.total_distance_m
            ? Math.round(route.total_distance_m / 100) / 10 : null,
          eta_epoch: route.eta_epoch || null,
          total_maneuvers: mans.length,
          first_steps: first,
          rules: 'The route is live and the car is navigating already. ' +
                 'Confirm it once, briefly, in your own words, using this ' +
                 'destination name exactly as spelled here. Do NOT tell the ' +
                 'driver to set it themselves — it is set. Do not read the ' +
                 'turns out now: confirming is one line. If they ASK for the ' +
                 'directions, call nav_directions and read them — that is ' +
                 'answering. What you never do is call a turn as it arrives; ' +
                 'the navigation system does that itself, at the moment it ' +
                 'matters.',
        };
      }
      if (out.status === 'ambiguous') {
        return {
          ok: true, routing: false, status: 'ambiguous',
          query: out.query || text,
          candidates: (out.candidates || []).map(function (c) {
            return { name: c.display_name || c.formatted_address || '',
                     address: c.formatted_address || '' };
          }),
          rules: 'More than one place answers to that. Do NOT pick one. Ask ' +
                 'the driver which of these they meant, naming them, and when ' +
                 'they answer call start_navigation again with their choice.',
        };
      }
      if (out.status === 'not_found') {
        return {
          ok: true, routing: false, status: 'not_found',
          query: out.query || text,
          rules: 'Say you could not find that place and ask them to say it ' +
                 'another way. Do not route to something else instead, and do ' +
                 'not tell them to type it in.',
        };
      }
      return {
        ok: false, routing: false, status: 'failed',
        note: out.error || 'route failed',
        destination: (out.destination && (out.destination.display_name ||
                                          out.destination.formatted_address)) || null,
        rules: 'The route did not start. Say so plainly, with the reason if ' +
               'it is something the driver can do anything about. Do not ' +
               'send them to the screen to do it themselves.',
      };
    });
  }

  /* stop_navigation — "I know the way from here."
   *
   * The panel again, and for the reason start_navigation is: a server-side
   * version could decide a route should end but could not end it, because the
   * route, the tracker and the queue of things about to be said all live here.
   *
   * Nothing about stopping is written here either. RIO.nav.stopRoute is what
   * the Clear button on the dashboard calls — the same teardown, the same
   * emptied queue, the same generation going dead so that an announcement
   * already in flight fails its own validity check instead of being spoken
   * over a driver who has just said they do not need it.
   */
  function stopNavigation() {
    var nav = root.RIO && root.RIO.nav;
    if (!nav || typeof nav.stopRoute !== 'function') {
      return { ok: false, note: 'no navigation on this page' };
    }
    var out = nav.stopRoute('voice') || {};
    if (!out.was_navigating) {
      return {
        ok: true, was_navigating: false,
        rules: 'There was no route running, so nothing was stopped. Say ' +
               'that in one short line. Do not apologise for it and do not ' +
               'offer to start one.',
      };
    }
    return {
      ok: true, was_navigating: true, destination: out.destination || null,
      rules: 'The route is off and the turn-by-turn is silent. Confirm it ' +
             'once, briefly — "Okay, navigation off." Do not ask whether ' +
             'they are sure, do not offer to start it again, and do not ' +
             'read out where they were going.',
    };
  }

  /* reroute — "find another way", "avoid the freeway".
   *
   * A COMMAND, and kept separate from the reroute the tracker does on its own
   * when the car leaves the route. That one is a correction and announces
   * nothing; this one was asked for and is answered.
   *
   * The destination is not re-resolved: RIO.nav.reroute goes back to the same
   * destination OBJECT, so "find another way" cannot quietly land on a
   * different branch of the same chain. What can change is how the route is
   * allowed to get there, and only in the ways the map itself supports —
   * anything else comes back named in `avoid_unsupported`, to be said out
   * loud rather than silently ignored.
   */
  function rerouteNavigation(args) {
    var nav = root.RIO && root.RIO.nav;
    if (!nav || typeof nav.reroute !== 'function') {
      return Promise.resolve({ ok: false, note: 'no navigation on this page' });
    }
    var wanted = [];
    var list = (args && args.avoid) || [];
    for (var i = 0; i < list.length; i++) {
      var a = String(list[i] || '').trim().toLowerCase();
      if (a) wanted.push(a);
    }
    // Anything the driver asked for that the schema has no word for. It is
    // carried as far as the answer and no further: it changes no route, and
    // exists so RIO can say "I can't pick the scenic one" instead of
    // rerouting and letting the driver assume she did.
    var other = String((args && args.other_preference) || '').trim();

    return Promise.resolve(nav.reroute({ avoid: wanted })).then(function (out) {
      out = out || {};
      if (out.status === 'no_route') {
        return {
          ok: true, status: 'no_route',
          rules: 'Nothing is being navigated, so there is no route to ' +
                 'change. Say that in one line and offer to set one.',
        };
      }
      if (out.status === 'busy') {
        return {
          ok: true, status: 'busy',
          rules: 'A route is already being worked out right now. Say you are ' +
                 'on it, in one line, and say nothing else about it.',
        };
      }
      if (out.status !== 'rerouted') {
        return {
          ok: false, status: 'failed', note: out.error || 'reroute failed',
          rules: 'The reroute did not happen and the route they were already ' +
                 'on is still running — say both, in one line. Do not tell ' +
                 'them to do it themselves on the screen.',
        };
      }
      var route = out.route || {};
      var dest = route.destination || {};
      var unsupported = (route.avoid_unsupported || []).slice();
      if (other) unsupported.push(other);
      return {
        ok: true, status: 'rerouted',
        destination: dest.display_name || dest.formatted_address || null,
        minutes: route.duration_s
          ? Math.max(1, Math.round(route.duration_s / 60)) : null,
        distance_km: route.total_distance_m
          ? Math.round(route.total_distance_m / 100) / 10 : null,
        eta_epoch: route.eta_epoch || null,
        generation_id: route.generation_id || null,
        avoid_applied: route.avoid_applied || [],
        avoid_unsupported: unsupported,
        rules: 'The new route is live and the car is already on it. One ' +
               'line: what changed and roughly how long it is now — ' +
               '"Rerouting — about twenty minutes." If avoid_unsupported ' +
               'has anything in it, say plainly that you cannot do that one ' +
               'and that you have rerouted anyway; never say you avoided ' +
               'something that is not in avoid_applied. Do not read the ' +
               'turns out, and do not call them — the navigation system ' +
               'still does that itself.',
      };
    });
  }

  var LOCAL_TOOLS = { nav_status: navStatus, nav_directions: navDirections,
                      start_navigation: startNavigation,
                      stop_navigation: stopNavigation,
                      reroute: rerouteNavigation };

  /* ---------------------------------------------------------------------
     Where the car is, for the tools that run on the SERVER.

     find_places needs a position to make "near me" mean anything, and the
     server does not have one: the GPS watch lives in this page, which is the
     same reason nav_status is answered here. So the fix rides along with every
     server tool call, and the server decides whether it is fresh enough to
     use.

     Age is computed HERE, from one clock. Sending a timestamp and letting the
     server subtract its own would put two clocks in an argument about whether
     a fix is stale, and a phone's clock is not the server's.
     --------------------------------------------------------------------- */
  var lastFix = null;

  function noteFix(pos) {
    var c = pos && (pos.coords || pos);
    if (!c) return;
    var lat = (typeof c.latitude === 'number') ? c.latitude : c.lat;
    var lng = (typeof c.longitude === 'number') ? c.longitude : c.lng;
    if (typeof lat !== 'number' || typeof lng !== 'number') return;
    if (!isFinite(lat) || !isFinite(lng)) return;
    lastFix = { lat: lat, lng: lng,
                accuracy_m: (typeof c.accuracy === 'number') ? c.accuracy : null,
                at: Date.now() };
  }

  function currentFix() {
    if (!lastFix) return null;
    return { lat: lastFix.lat, lng: lastFix.lng,
             accuracy_m: lastFix.accuracy_m,
             age_s: Math.max(0, (Date.now() - lastFix.at) / 1000) };
  }

  /* ---------------------------------------------------------------------
     The controller: events in, decisions out.
     cfg = {
       arbiter,                      RIO.speech
       send(obj),                    put an event on the data channel
       tool(name, args) -> Promise,  run one tool call server-side
       audio: { mute(), unmute() },  whatever RIO's voice comes out of
       voice,                        the ElevenLabs sink, or absent when the
                                     live session speaks for herself
       onEvent(ev)                   observability
     }

     TWO BACKENDS, ONE HANDLER
     -------------------------
     Which voice RIO has changes what the session PRODUCES — audio, or text
     that something else speaks — and almost nothing else. Interruption,
     resumption, arbitration, the tool bridge and every counter below are the
     same code either way, and deliberately: they are the parts that were hard
     to get right, and a second copy of them for a second voice is a second set
     of bugs.

     `voice` is what makes the difference. Absent, the session speaks for
     itself and this file mutes an element. Present, the session writes and the
     sink speaks — so the text deltas are forwarded to it, the mouth is held
     until the SINK has finished rather than until the model has, and "how far
     did she get" is answered by what came out of a speaker instead of by what
     the model said it was saying.
     --------------------------------------------------------------------- */
  function createController(cfg) {
    cfg = cfg || {};
    var arbiter = cfg.arbiter;
    var send = cfg.send || function () {};
    var runTool = cfg.tool || function () { return Promise.resolve({ ok: false }); };
    var audio = cfg.audio || { mute: function () {}, unmute: function () {} };
    /* RIO's mouth, when it is not the model's own. Nullable, and null is the
       live-voice path — every `if (sink)` below reads as "unless she is speaking
       for herself", which is what it means. */
    var sink = cfg.voice || null;
    var listeners = [];
    if (typeof cfg.onEvent === 'function') listeners.push(cfg.onEvent);

    /* TOOLS THAT WAIT FOR A REASON TO EXIST.
     *
     * Every tool schema is input on every response for the whole drive. Two
     * of them -- stop_navigation and reroute -- cannot be called at all until
     * a route exists, and a route exists for a minority of most drives; the
     * session was paying about 350 tokens a response, twice that on a tool
     * turn, to offer the model two things it could only fail at.
     *
     * So a condition is named, the server says which tools ride on it, and
     * this sends ONE session.update when the answer changes. Not on every
     * route event: a session.update per GPS-triggered reroute would spend
     * more than it saves and would reset the tool list mid-turn.
     *
     * The list is replaced whole because that is what the API does with
     * `tools` -- which is why the base schemas ride along in the session
     * payload rather than being reconstructed here. Nothing about a tool is
     * decided in this file; it holds the server's words and picks which of
     * them are true right now. */
    var baseTools = cfg.toolSchemas || [];
    var conditionalTools = cfg.conditionalTools || {};
    var conditionsOn = {};

    function toolsForNow() {
      var out = baseTools.slice();
      for (var name in conditionalTools) {
        if (conditionsOn[name]) out = out.concat(conditionalTools[name] || []);
      }
      return out;
    }

    var speaking = null;      // { responseId, resolve, cancelled }
    var stopped = false;
    var counters = { responses: 0, interrupted: 0, barge_ins: 0,
                     tool_calls: 0, tool_failures: 0,
                     dictated: 0, dictation_failures: 0,
                     // Barge-ins absorbed before they cost anything: the
                     // detector fired, RIO went quiet, the noise stopped
                     // inside the sustain window and she carried straight on.
                     // The single most useful number here -- every one of
                     // these used to be a lost answer.
                     blips_absorbed: 0,
                     /* Detector firings the echo gate refused outright: the
                        microphone never beat the loudspeaker, so nothing was
                        muted, nothing was cancelled, and the answer carried on.
                        On a desk this is always zero -- the gate is off. */
                     echo_suppressed: 0,
                     /* Dictations abandoned on their budget whose response
                        turned up anyway and was killed before it could read
                        the line a second voice was already reading. The number
                        that must be equal to `dictation_failures` from the
                        timeout branch: every disowned line accounted for. */
                     orphans_silenced: 0,
                     /* ...and the ones held over her opening syllable and then
                        allowed through, because the speech was still going. A
                        real interruption, a fifth of a second late. */
                     onset_deferred: 0,
                     resumed: 0, resume_skipped: 0, resume_failures: 0,
                     // Utterances the sink could not speak in the voice it was
                     // meant to, and the one time a drive gave up on it.
                     voice_fallbacks: 0, voice_backend_changed: 0,
                     direct_speech_failures: 0,
                     // Answers spoken straight from the running observation,
                     // with no model between the sentence and the speaker.
                     spoken_directly: 0,
                     // ...and the ones that could not be, because the mouth
                     // was still held when the camera answered. Not a lost
                     // answer any more -- it becomes an ordinary request to
                     // the model -- but it IS the fast path not being taken,
                     // which is worth a number rather than a shrug.
                     direct_deferred: 0,
                     // Responses the API refused outright. Almost always the
                     // token-per-minute ceiling, and a tool turn spends two
                     // responses where a plain answer spends one, which is
                     // why "she only fails when she uses a tool" is the shape
                     // the driver sees. Retried once; counted either way.
                     responses_failed: 0, responses_retried: 0,
                     /* NEWEST WINS -- the three numbers item 4 exists to make
                        non-zero, and the one it exists to make small.

                        turns_superseded  a driver asked again before the last
                                          answer arrived, and the old one was
                                          dropped rather than spoken.
                        tools_aborted     tool calls belonging to those turns.
                                          On the first real drive these would
                                          have been four `look` calls of 40.1,
                                          12.9, 48.5 and 27.1 seconds.
                        turns_coalesced   fragments semantic VAD split that
                                          were one question. Superseding these
                                          would be the same bug backwards.
                        commands_preempted  "stop navigation" acted on at once
                                          instead of behind a 40 s tool call. */
                     turns_superseded: 0, tools_aborted: 0,
                     turns_coalesced: 0, commands_preempted: 0,
                     /* Transcripts that were NOT a driver turn: her own voice
                        back through the speaker, or speech the barge gate
                        never confirmed. Every one of these used to cancel the
                        answer it was an echo of. On the iPhone test that
                        produced this counter, ten of fifteen supersedes were
                        one of these. */
                     turns_phantom: 0,
                     // Tool results that came back for a turn nobody was
                     // waiting for. Never spoken; the number that says how
                     // often a stale answer WOULD have been.
                     tool_results_discarded: 0 };
    var directs = 0;

    /* WHY EVERY ANSWER STOPPED. One counter per cause, because "she cuts out"
       is four different faults wearing one coat and they have four different
       fixes:

         false_barge_in  the detector called it speech and no words followed.
                         Echo, a cough, a door. Resumable, and the reason this
                         whole mechanism exists.
         barge_in        the driver really did talk over her. Working as
                         intended, and never resumed -- finishing an answer
                         somebody deliberately cut off is the rude version of
                         this bug.
         preempted       a warning, a turn or a health line took the mouth.
                         Resumed once the mouth comes back.
         token_cap       she hit REALTIME_MAX_RESPONSE_TOKENS. A deliberate
                         ceiling, not a fault, and NOT resumed: resuming it
                         would be arguing with the limit.
         transport       the data channel or the peer connection went away.
                         Nothing to resume into.
         other           the arbiter's watchdog, an error, a session ending. */
    var cutoffs = { false_barge_in: 0, barge_in: 0, preempted: 0,
                    token_cap: 0, transport: 0, other: 0 };
    var now = (root.performance && root.performance.now)
      ? function () { return root.performance.now(); }
      : function () { return Date.now(); };

    var lastTranscript = '';
    /* What she has said so far in the response now playing, accumulated from
       the audio transcript deltas. This is what a resume carries: without it
       the continuation is a guess, and a model asked to continue from a guess
       starts the answer again -- which is the thing the driver was already
       doing by hand. */
    var partial = '';
    /* ...and what the MODEL has written, which in text mode is a different and
       much longer string. The model finishes an answer in a few hundred
       milliseconds; the driver is four seconds behind it. Resuming from what
       was written rather than from what was heard would have RIO skip
       everything the synthesiser had not reached, which is most of the answer
       and exactly the words the interruption cost. */
    var generated = '';
    var pendingBarge = null;    // { responseId, cancelled, timer, confirm, said }
    var pendingResume = null;   // { cause, said }
    var resumeChain = 0;        // resumes spent on THIS answer
    /* One retry of a refused response per driver turn. See responseFailed:
       the ceiling this exists for is per-minute, so a second attempt inside
       one turn is asking the same question of the same empty budget. */
    var retryArmed = true;
    var turnSeq = 0;            // which driver turn is outstanding

    /* ---- NEWEST WINS ------------------------------------------------------
     *
     * WHY THIS IS NOT TIDINESS. On the first real drive the four `look` calls
     * took 40.1 s, 12.9 s, 48.5 s and 27.1 s -- the visual path went out to
     * Qwen and one /perceive on that drive took 52.6 seconds. A driver does
     * not wait forty seconds; they ask again. And until this existed the first
     * question's tool call went on running, came back, and asked the model to
     * speak about it -- an answer to a question two questions ago, arriving
     * while the driver was waiting for the one they had just asked.
     *
     * Three things a new driver utterance now does, in this order:
     *
     *   1. decide whether it is a NEW turn at all. A fragment that continues
     *      the last one inside the coalesce window is the same question, and
     *      cancelling its own tool call would be this bug in the other
     *      direction.
     *   2. if it is new: cancel the response being generated, abort every tool
     *      call belonging to an older turn, and drop anything conversational
     *      still waiting for the mouth.
     *   3. log it. Every supersede, every abort, every coalesce.
     *
     * A tool result that lands for a superseded turn is never spoken and never
     * turned into a request for a response. The output is still handed back to
     * the session -- a function call left with no output is a malformed
     * conversation, and the next turn pays for it -- but it says what happened
     * and asks for nothing.
     */
    var inflightTools = {};     // call_id -> {name, turn, controller, at}
    var lastTurnAt = 0;         // when the last driver fragment landed
    var lastTurnText = '';

    /* WHAT SHE HAS RECENTLY SAID, and when she last made a sound.
     *
     * Both exist for one question: is this transcript a new question, or is it
     * her own voice coming back through the phone's speaker? On a live iPhone
     * test her opening line -- "Hey. What's up." -- came back through the
     * microphone as "Hello." and "What's up?", was taken for a new driver
     * turn, and cancelled the response that was producing the audio. Ten times
     * across two sessions. She never finished a sentence.
     *
     * `recentSaid` is her output transcript over a short window; `lastAudioAt`
     * is when the last of it was produced. Between them and the barge gate,
     * `supersedeGate` decides whether a cancel is a supersede. */
    var recentSaid = [];        // [{t, text}] of her own transcript deltas
    var lastAudioAt = 0;
    /* Fallbacks only, and the same arrangement rio_navcore uses for its
       timing: the real values live in config.py (REALTIME_DRIVER_COMMANDS,
       REALTIME_COALESCE_*) and arrive with the session. These exist so a
       controller built before the session payload lands, or by a test, still
       has a policy rather than none -- and tools/realtime_selftest.py asserts
       the shipped copy against config so the two cannot drift. */
    var TURN_DEFAULTS = {
      commands: {
        stop_navigation: ['stop( the)? nav(igation)?', 'cancel( the)? nav(igation)?',
                          'end( the)? nav(igation)?', 'stop( the)? route',
                          'cancel( the)? route', 'stop navigating', 'stop guiding me'],
        reroute: ['re-?route', 're-?calculate', 'find (me )?another way',
                  'different route', 'new route'],
        silence: ['stop', 'stop talking', 'be quiet', 'quiet', 'shut up',
                  'never ?mind', 'forget it', 'cancel that'],
      },
      command_max_words: 5,
      coalesce_gap_ms: 1500,
      coalesce_openers: ['and', 'or', 'but', 'then', 'also', 'plus', 'so',
                         'um', 'uh', 'er', 'like', 'actually', 'i mean', 'near',
                         'next to', 'on', 'in', 'at', 'to', 'for', 'with', 'by',
                         'from', 'about', 'around', 'over', 'under', 'after',
                         'before', 'because', 'which', 'that', 'who', 'just'],
      coalesce_trailers: ['and', 'or', 'but', 'the', 'a', 'an', 'of', 'to',
                          'for', 'with', 'at', 'on', 'in', 'is', 'are', 'was',
                          'were', 'near', 'by', 'from', 'that', 'like', 'about',
                          'some', 'any', 'my', 'your', "it's", "there's"],
    };
    var turnPolicy = {};
    for (var tk in TURN_DEFAULTS) turnPolicy[tk] = TURN_DEFAULTS[tk];
    for (var tk2 in (cfg.turnPolicy || {})) turnPolicy[tk2] = cfg.turnPolicy[tk2];
    var commandRes = null;      // lazily compiled from turnPolicy.commands

    function policyList(name, fallback) {
      var v = turnPolicy[name];
      return (v && v.length) ? v : fallback;
    }

    /* A command is an utterance that is essentially NOTHING BUT the command.
       Anchored, and length-capped, because "stop" inside "don't stop at the
       next light" is not an instruction to end the route. */
    function driverCommand(text) {
      var t = (text || '').toLowerCase().trim()
        .replace(/[.!?,;:]+$/g, '')
        .replace(/^(hey |ok |okay |rio[,! ]*)+/g, '')
        .trim();
      if (!t) return null;
      var maxWords = turnPolicy.command_max_words || 5;
      if (t.split(/\s+/).length > maxWords) return null;
      if (!commandRes) {
        commandRes = [];
        var src = turnPolicy.commands || {};
        for (var kind in src) {
          for (var i = 0; i < src[kind].length; i++) {
            try {
              commandRes.push({ kind: kind,
                                re: new RegExp('^(please\\s+)?' + src[kind][i]
                                               + '(\\s+please)?$') });
            } catch (e) { /* a bad pattern must not break the session */ }
          }
        }
      }
      for (var j = 0; j < commandRes.length; j++) {
        if (commandRes[j].re.test(t)) return commandRes[j].kind;
      }
      return null;
    }

    /* Is this fragment the back half of the last one?
     *
     * Semantic VAD splits an utterance the moment the driver breathes, and
     * "is there a petrol station" / "near the next exit" is one question. Two
     * signals, either of which is enough, both inside a short window: the new
     * fragment opens like a continuation, or the old one ended unfinished. */
    function isContinuation(prev, next, gapMs) {
      if (!prev || !next) return false;
      if (gapMs > (turnPolicy.coalesce_gap_ms || 1500)) return false;
      var a = prev.toLowerCase().trim();
      var b = next.toLowerCase().trim().replace(/^[,;\s]+/, '');
      if (!a || !b) return false;
      // A finished sentence followed by a question is two questions.
      if (/[.!?]$/.test(a) && /^(what|where|when|who|why|how|is|are|can|could|do|does|did|will|would|should|tell|show|find|take|go|play|call)\b/.test(b)) {
        return false;
      }
      var openers = policyList('coalesce_openers', ['and', 'or', 'but', 'then']);
      for (var i = 0; i < openers.length; i++) {
        if (b === openers[i] || b.indexOf(openers[i] + ' ') === 0) return true;
      }
      var tail = a.replace(/[.!?,;:]+$/, '').split(/\s+/).pop();
      var trailers = policyList('coalesce_trailers', ['and', 'or', 'the', 'a']);
      for (var j = 0; j < trailers.length; j++) {
        if (tail === trailers[j]) return true;
      }
      return false;
    }

    /* --- IS THIS A REAL NEW QUESTION? --------------------------------------
     *
     * The whole of the fix, in one function. A supersede CANCELS: it stops the
     * response she is producing, aborts the tool calls behind it and clears
     * the mouth. That is a barge-in by another name, and it has to clear the
     * same bar barge-in does.
     *
     * Two gates, and the first one is the load-bearing one.
     */
    function noteSaid(text) {
        if (!text) return;
        var at = now();
        lastAudioAt = at;
        recentSaid.push({ t: at, text: String(text) });
        var cutoff = at - echoTextWindowMs;
        while (recentSaid.length && recentSaid[0].t < cutoff) recentSaid.shift();
    }

    function normWords(text) {
        return String(text || '').toLowerCase()
            .replace(/[^a-z0-9\s]/g, ' ').split(/\s+/).filter(Boolean);
    }

    /* Does this transcript look like something she just said?
     *
     * THE SECOND NET, and it is second for a measured reason: her audio is
     * transcribed twice by two different models -- once as her output, once as
     * microphone input -- and they disagree. She said "Hey"; the input
     * transcriber wrote "Hello". A text test alone would have missed the exact
     * case that motivated it. It catches the verbatim ones, which on that test
     * was "What's up?" against her own "What's up.", and the structural gate
     * catches the rest. */
    function looksLikeEcho(text) {
        var words = normWords(text);
        if (!words.length) return false;
        var mine = [];
        var at = now();
        for (var i = 0; i < recentSaid.length; i++) {
            if (at - recentSaid[i].t <= echoTextWindowMs) {
                mine = mine.concat(normWords(recentSaid[i].text));
            }
        }
        if (!mine.length) return false;
        var hay = ' ' + mine.join(' ') + ' ';
        if (hay.indexOf(' ' + words.join(' ') + ' ') >= 0) return true;
        // Below a couple of words, only exact containment counts: "yes" and
        // "no" share every word with almost anything she has ever said.
        if (words.length < echoTextMinWords) return false;
        var hit = 0;
        for (var j = 0; j < words.length; j++) {
            if (hay.indexOf(' ' + words[j] + ' ') >= 0) hit++;
        }
        return (hit / words.length) >= echoTextOverlap;
    }

    /* -> {allow, why}. `why` is recorded on every refusal, because "she
     * stopped finishing sentences" needs a number over a session rather than a
     * memory of the last one. */
    function supersedeGate(text) {
        if (looksLikeEcho(text)) {
            return { allow: false, why: 'echo_of_her_own_words' };
        }
        var speakingNow = !!(speaking && !speaking.cancelled);
        var tail = now() - lastAudioAt;
        if (!speakingNow && tail > echoTailMs) {
            // Nothing of hers is in the room. A transcript is a question, and
            // this is the path a genuine second question takes while she is
            // waiting on a tool call.
            return { allow: true, why: null };
        }
        /* She is speaking, or has only just stopped. The ONLY evidence that
           this is a driver and not a loudspeaker is the barge gate -- which
           has already refused it in three of its paths, all of them meaning
           "this is her": a dictation in progress, the onset guard over her
           opening syllable, and the level test that says the microphone never
           got louder than the speaker. None of those creates a pendingBarge,
           and the supersede used to fire anyway. */
        if (!pendingBarge) {
            return { allow: false, why: onsetHold ? 'inside_onset_guard'
                                                  : 'no_confirmed_barge' };
        }
        if (!pendingBarge.cancelled) {
            // The detector fired and the sustain gate has not agreed with it.
            return { allow: false, why: 'barge_not_sustained' };
        }
        return { allow: true, why: null };
    }

    /* Abort every tool call that belongs to a turn nobody is waiting for. */
    function abortStaleTools(reason) {
      var n = 0;
      for (var id in inflightTools) {
        var e = inflightTools[id];
        if (e.turn >= turnSeq) continue;
        n++;
        e.aborted = true;
        try { if (e.controller) e.controller.abort(); } catch (x) {}
        counters.tools_aborted++;
        emit('LIVE_TOOL_ABORTED', {
          tool: e.name, call_id: id, turn: e.turn, now_turn: turnSeq,
          reason: reason, age_ms: Math.round(now() - e.at),
        });
      }
      return n;
    }

    /* Everything a superseded turn leaves behind.
     *
     * Returns false when there was nothing to supersede, which is the ordinary
     * case for the first question of a conversation and for every question
     * asked after RIO has finished answering the last one. Logging one there
     * would make "she answered the wrong question" impossible to count,
     * because the count would be "every turn". */
    function supersedeTurn(reason, detail) {
      var outstanding = false;
      for (var k in inflightTools) {
        if (inflightTools[k].turn < turnSeq) { outstanding = true; break; }
      }
      if (!outstanding && !(speaking && !speaking.cancelled)) return false;
      var rid = speaking ? speaking.responseId : null;
      var said = saidSoFar();
      var aborted = abortStaleTools(reason);
      // The response being generated is about the old question.
      if (speaking && !speaking.cancelled) {
        cancelGeneration();
        if (rid) endResponse(rid);
      } else {
        // Nothing is playing, but a response may still be on its way from a
        // create we have not seen `response.created` for yet.
        try { send({ type: 'response.cancel' }); } catch (e) {}
      }
      // ...and anything conversational still queued for the mouth. A safety
      // warning or a turn call is NOT dropped: those are about the road, not
      // about the question.
      try {
        if (root.RIO && RIO.speech && RIO.speech.clear) RIO.speech.clear('convo');
      } catch (e) {}
      counters.turns_superseded++;
      emit('LIVE_TURN_SUPERSEDED', {
        turn: turnSeq, response_id: rid, said_chars: (said || '').length,
        tools_aborted: aborted, reason: reason,
        superseded: (detail && detail.previous || '').slice(0, 160),
        by: (detail && detail.next || '').slice(0, 160),
      });
      return true;
    }
    /* IS THE LAST TRANSCRIPT ABOUT THE QUESTION BEING ASKED NOW?
     *
     * Only until the driver opens their mouth again. `lastTranscript` is kept
     * for the whole drive because a barge-in has to be classified against
     * whatever was said last, but a TOOL is a different question: the camera's
     * fast path is a judgement about what the driver asked, and the transcript
     * for the turn being asked about arrives on its own schedule -- often
     * after the model has already called the tool.
     *
     * Sent stale, the previous question judges this one. Measured, in a real
     * drive: "what do you see outside" (a scene question, answered from the
     * running observation in 42 ms) followed by "what kind of car is in front
     * of us" -- and the second one came back in 5 ms with the sentence about
     * the road, because the transcript the panel had was still the first
     * question. A wrong answer, delivered fast, to a question the driver had
     * asked perfectly clearly.
     *
     * So it goes stale the moment new speech starts, and the tool falls back
     * to the model's paraphrase, which is what it did before any of this
     * existed and is honest about being second-best. */
    var transcriptFresh = false;
    var lastArbiterStart = null;

    /* If the detector never says the speech ended, nothing will ever classify
       the cut-off and the state would leak for the rest of the drive. This is
       not a decision about anyone -- it is a cleanup, counted as `other` so it
       cannot masquerade as a diagnosis. Twenty seconds is longer than any
       utterance and shorter than a drive. */
    var BARGE_BACKSTOP_MS = 20000;

    var bargeSustainMs = cfg.bargeSustainMs || 300;
    var bargeConfirmMs = cfg.bargeConfirmMs || 1500;
    /* Fallbacks only; the real values arrive with the session, exactly as the
       barge policy does. See config.REALTIME_ECHO_*. */
    var echoTailMs = cfg.echoTailMs || 600;
    var echoTextWindowMs = (cfg.echoTextWindowS || 15) * 1000;
    var echoTextOverlap = cfg.echoTextOverlap || 0.8;
    var echoTextMinWords = cfg.echoTextMinWords || 2;

    /* ---- THE ECHO GATE, and why the desk does not have one -----------------
     *
     * On a phone, RIO comes out of a loudspeaker eight inches from the
     * microphone, and only part of what she says goes out through a renderer
     * the platform's echo canceller has a reference for (see
     * tools/echo_barge_probe.js, which reads the paths out of the source).
     * Everything else returns to the microphone at full level, the turn
     * detector upstream calls it speech, and she interrupts herself. Measured
     * before this existed: five long answers into an empty car, five answers
     * cut off.
     *
     * Three tests, and each one separates echo from a person differently:
     *
     *   onsetGuardMs   the opening of her utterance is when the canceller has
     *                  least to work with and the level jumps hardest, so the
     *                  detector is not believed alone there. DEFERRED, never
     *                  discarded: if the speech is still going when the guard
     *                  expires it becomes a barge-in from that moment, so a
     *                  driver who cuts in on her first word still stops her.
     *   sustain        echo stops when she pauses. A person does not pause on
     *                  her schedule.
     *   echoMarginDb   physics, not timing: echo cannot be louder than what
     *                  produced it. The microphone has to beat the audio being
     *                  rendered by this margin before a cut-off is allowed to
     *                  cost an answer.
     *
     * All three are ZERO on a desk, where the numbers are the ones they always
     * were and the behaviour is byte for byte what it was. `levels` is the
     * page's meter -- {mic, out} in dBFS -- and its absence disables the level
     * test rather than blocking anything: no meter is not evidence of echo. */
    var bargeOnsetGuardMs = cfg.bargeOnsetGuardMs || 0;
    var bargeEchoMarginDb = cfg.bargeEchoMarginDb || 0;
    /* Below this the output is not loud enough to be echoing anything and the
       level test is skipped entirely. */
    var bargeEchoFloorDb = (cfg.bargeEchoFloorDb === undefined ||
                            cfg.bargeEchoFloorDb === null)
      ? -50 : cfg.bargeEchoFloorDb;
    var levels = (typeof cfg.levels === 'function') ? cfg.levels : null;
    /* How often the margin is re-read while a barge-in is pending. Fast enough
       that a 300 ms window is sampled several times, cheap enough to leave
       running -- one analyser read per tick. */
    var LEVEL_POLL_MS = 40;
    var levelTimer = null;
    /* The loudest the microphone got over the audio, across the pending
       window. A peak rather than an average: a driver's first syllable is the
       evidence, and averaging it against the silence around it is how a real
       interruption gets called echo. */
    var peakMargin = null;
    var lastLevels = null;
    /* The decision held over her opening syllable, waiting to see whether the
       thing that fired the detector is still there when the guard expires. */
    var onsetHold = null;
    /* What the detector currently says: between speech_started and
       speech_stopped. The onset hold and the echo watch both need to know
       whether there is still something going on to decide about. */
    var speechActive = false;
    var maxResumes = (cfg.maxResumes === undefined || cfg.maxResumes === null)
      ? 1 : cfg.maxResumes;
    var resumeInstruction = cfg.resumeInstruction ||
      'You were cut off part-way through an answer by noise, not by the ' +
      'driver. Finish it in one or two short sentences, beginning with ' +
      '"As I was saying". Do not start over.\n\nWHAT YOU HAD SAID SO FAR:\n';
    /* A dictated line — a warning, a turn, a health announcement — in flight.
       It is NOT a conversation response and must never be treated as one: it
       does not claim the mouth (its caller already holds it, at its own
       priority) and it does not enter the conversation history. */
    var dictation = null;
    /* A vetted line being spoken by the SESSION rather than by a synthesiser.
       Only ever set under the speech-to-speech backend; see speakDirect. */
    var directSpeech = null;
    /* THE RESPONSES THAT ARE NOT THE CONVERSATION, remembered past the point
       where the thing tracking them let go.
     *
     * A dictated warning and an injected line are both out of band, and both
     * are normally recognised by `dictation`/`directSpeech` still being set
     * when their `response.done` arrives. When one of them is given up on
     * early -- a budget expiring -- that state is cleared, and the late
     * `response.done` then fell through to the CONVERSATION's branch and was
     * filed as a cut-off answer.
     *
     * Which is how a scene answer that spoke perfectly well appeared in the
     * tally as `token_cap`: the line was slow to start, the budget fired, and
     * the response that arrived afterwards looked like an answer that had been
     * truncated. A tally that counts a warning as a lost answer is worse than
     * no tally, because the number is the whole reason it exists. */
    var outOfBand = [];
    function markOutOfBand(responseId) {
      if (!responseId) return;
      outOfBand.push(responseId);
      if (outOfBand.length > 8) outOfBand.shift();
    }
    function isOutOfBand(responseId) {
      return !!responseId && outOfBand.indexOf(responseId) >= 0;
    }
    /* ...AND THE ONE THAT HAS NO ID YET WHEN IT IS GIVEN UP ON.
     *
     * The binding above needs a `response.created` to attach to. Give up on a
     * line before that arrives and the response is still coming: it gets
     * created, it belongs to nobody, and without this it would claim the mouth
     * as an ordinary answer and be counted as one. Exactly one is expected --
     * the abandoned line -- and the recovery response asked for immediately
     * after it is the conversation and must still claim the mouth normally. */
    var orphanOutOfBand = 0;
    /* ...and whether that orphan must also be SILENCED rather than merely not
       counted. An abandoned dictation has already been replaced by another
       voice saying the same words; an abandoned direct line has not, and its
       audio is the answer. Same mechanism, opposite conclusion about the
       speaker, so they are two flags and not one. */
    var silenceOrphan = false;
    var verbatimInstruction = cfg.verbatimInstruction ||
      'Read the text below out loud, exactly as written, word for word. ' +
      'Add nothing. Remove nothing. Do not rephrase.\n\nTEXT:\n';
    var speakTimeoutMs = cfg.speakTimeoutMs || 700;
    /* A vetted answer read into the session is injected the same way a warning
       is, and waits a different length of time for it. See injectDirect. */
    var directSpeechTimeoutMs = cfg.directSpeechTimeoutMs || 2500;
    // The ceiling on a camera answer, from config.py by way of the session
    // payload. Null leaves the session's own limit in charge.
    var lookAnswerMaxTokens = cfg.lookAnswerMaxTokens || null;

    /* How far the DRIVER got to hear. The only version of this question worth
       asking, and the two backends answer it from different places: the model's
       own transcript when she speaks for herself, the speaker's clock when
       something else is speaking for her. */
    function saidSoFar() {
      if (sink && speaking) {
        try { return sink.spokenPrefix(speaking.responseId) || ''; }
        catch (e) { return partial; }
      }
      return partial;
    }

    function emit(type, payload) {
      var ev = payload || {};
      ev.type = type;
      for (var i = 0; i < listeners.length; i++) {
        try { listeners[i](ev); } catch (e) { /* never let a listener mute RIO */ }
      }
    }

    /* One answer stopped early, and why. Recorded once per cut-off, at the
       point the cause is actually known rather than at the point the audio
       stopped -- those are different moments for a barge-in, which is not
       classifiable until the transcript either arrives or does not. */
    function noteCutoff(cause, detail) {
      if (cutoffs[cause] === undefined) cause = 'other';
      cutoffs[cause]++;
      var ev = detail || {};
      ev.cause = cause;
      ev.said_chars = (ev.said || '').length;
      delete ev.said;
      emit('LIVE_CUTOFF', ev);
    }

    /* Stop the model generating, and clear the audio it has already queued.
       Both, always: cancelling generation alone leaves whatever is in the
       output buffer to play out from under a warning. */
    function cancelGeneration() {
      /* A directly-spoken line has no response behind it to cancel. Sending
         `response.cancel` with nothing generating is answered with an error
         event, which is a real error in the log for a thing that worked. */
      if (!(speaking && speaking.direct)) {
        try { send({ type: 'response.cancel' }); } catch (e) {}
        try { send({ type: 'output_audio_buffer.clear' }); } catch (e) {}
      }
      /* The same two things, on the other side of the mouth: stop the words
         being produced, and throw away the sound already made from them. In
         text mode the second one is the sink's — the model's output buffer is
         empty because the model was never making audio, and everything queued
         is queued here. */
      if (sink && speaking) {
        try { sink.cancel(speaking.responseId); } catch (e) {}
      }
    }

    function clearBarge() {
      stopEchoWatch();
      if (!pendingBarge) return;
      if (pendingBarge.timer) clearTimeout(pendingBarge.timer);
      if (pendingBarge.confirm) clearTimeout(pendingBarge.confirm);
      if (pendingBarge.backstop) clearTimeout(pendingBarge.backstop);
      pendingBarge = null;
    }

    /* Is the mouth free? A resume is a conversation-priority thing and must
       wait behind whatever took it away -- resuming into the middle of the
       warning that pre-empted you is the same fault in the other direction. */
    function mouthFree() {
      if (!arbiter || typeof arbiter.state !== 'function') return true;
      try { return !arbiter.state().speaking; } catch (e) { return true; }
    }

    /* Remember an answer worth finishing. Not every cut-off is: an answer
       nobody had started hearing has nothing to carry on from, and one that
       has already been resumed once is in an argument with the cabin. */
    function armResume(cause, said) {
      if (stopped) return;
      said = (said || '').trim();
      if (!said) return;
      if (resumeChain >= maxResumes) {
        counters.resume_skipped++;
        emit('LIVE_RESUME_SKIPPED', { cause: cause, reason: 'budget' });
        return;
      }
      pendingResume = { cause: cause, said: said };
    }

    /* ...and finish it, once there is a mouth to finish it with. Called on
       every event that could free one: the arbiter releasing, a dictation
       ending, the cut-off being classified. */
    function tryResume() {
      if (!pendingResume || stopped || speaking || dictation) return;
      if (!mouthFree()) return;
      var r = pendingResume;
      pendingResume = null;
      resumeChain++;
      counters.resumed++;
      emit('LIVE_RESUME', { cause: r.cause, said: r.said });
      try {
        send({
          type: 'response.create',
          response: {
            // What the session is asked to PRODUCE is the one thing the two
            // backends disagree about. Everything else on this payload — that
            // it is in the conversation rather than out of band, and what she
            // is told — is the same sentence either way.
            output_modalities: sink ? ['text'] : ['audio'],
            // In the conversation, not out of band: the truncated half is
            // already in the history and leaving the other half out would make
            // the next question land against an answer that stops mid-sentence.
            instructions: resumeInstruction + r.said,
          },
        });
      } catch (e) {
        counters.resume_failures++;
        emit('LIVE_RESUME_FAILED', { cause: r.cause });
      }
    }

    /* The arbiter tells us two things worth knowing: what took the mouth (so a
       pre-emption can say what pre-empted it) and when it is free again (so
       the answer it interrupted can be finished). */
    if (arbiter && typeof arbiter.onEvent === 'function') {
      arbiter.onEvent(function (ev) {
        if (!ev || stopped) return;
        if (ev.type === 'start') {
          lastArbiterStart = ev.item || null;
          /* THE ONE PLACE THAT SEES EVERYTHING RIO SAYS. A gap warning, a turn
             call, a health line and a conversational answer all claim the
             mouth here and all come back through the microphone the same way.
             Without this the echo test would only know about the words the
             model composed, and "Left here." would still be able to cancel the
             answer that followed it. */
          if (ev.item && ev.item.text) noteSaid(ev.item.text);
        }
        if (ev.type === 'end' || ev.type === 'drop') {
          // Next tick: the arbiter is mid-pump and `current` is not settled
          // until it returns.
          setTimeout(tryResume, 0);
        }
      });
    }

    /* RIO has started saying something. Claim the mouth for it.
     *
     * One arbiter item per RESPONSE, not per session: a session lasts a drive,
     * and an item that lasts a drive would either block every warning or be
     * pre-empted once and never recover. */
    function beginResponse(responseId, opts) {
      if (stopped) return;
      opts = opts || {};
      /* The first response after a dictation was sent IS that dictation. It
         gets bound here rather than claiming the mouth: a warning arriving as
         a conversation-priority item would be a warning that yields to
         navigation, which is upside down. */
      if (dictation && !dictation.responseId) {
        dictation.responseId = responseId;
        markOutOfBand(responseId);
        return;
      }
      /* ...and so does the response a DIRECT LINE was injected as. Same
         correlation and for the same reason -- the first response created
         after it was sent is it -- but the opposite conclusion about the
         mouth: a dictated warning claims the mouth at its caller's priority,
         while a direct line has ALREADY claimed it, at CONVO, in speakDirect.
         Either way this response must not claim it a second time. */
      if (directSpeech && !directSpeech.realId
          && String(responseId || '').indexOf('direct:') !== 0) {
        directSpeech.realId = responseId;
        markOutOfBand(responseId);
        return;
      }
      if (orphanOutOfBand > 0
          && String(responseId || '').indexOf('direct:') !== 0) {
        orphanOutOfBand--;
        markOutOfBand(responseId);
        if (silenceOrphan) {
          /* The line this response was going to read is already being read by
             something else. Cancel it BY ID -- the bare cancel that ran when
             it was abandoned had no response to name -- and mute what is
             already in flight, which is the same pair of moves a barge-in
             makes and for the same reason: what has left the speaker cannot be
             recalled, but the next 20 ms can. */
          silenceOrphan = false;
          counters.orphans_silenced++;
          try { send({ type: 'response.cancel', response_id: responseId }); }
          catch (e) {}
          try { audio.mute(); } catch (e) {}
          emit('LIVE_ORPHAN_SILENCED', { response_id: responseId });
        }
        return;
      }
      /* A NEW ANSWER WHILE THE LAST ONE IS STILL BEING HEARD.
       *
       * Only reachable in text mode, and reachable there routinely: RIO says
       * "let me check", the model finishes writing that in a few hundred
       * milliseconds, the tool comes back, and the follow-up response is
       * created while the holding line is still coming out of the speaker.
       *
       * `speaking` is still set at that instant -- deliberately, because the
       * mouth belongs to the listener and the listener is not finished -- and
       * the old early return meant the new response never claimed the mouth,
       * never opened an utterance, and was silent. The answer to the question
       * simply never got said.
       *
       * So a response that is only waiting on its tail yields to a new one.
       * The tail is NOT cut off: the sink queues the new utterance behind
       * whatever is still scheduled, which is the right order anyway -- "let
       * me check" and then the answer. What is still refused is a second
       * response over a live one, which is a real overlap.
       *
       * ...AND SO DOES THE RESPONSE A DIRECT LINE IS THE ANSWER TO, whether
       * or not it has finished writing. That is a second case and it was the
       * one that lost whole answers.
       *
       * Every other caller here is a `response.created` off the wire, and the
       * server orders those: it does not open a second response until the
       * first has reported done, so by the time one arrives the response
       * before it is always `finishing` and the rule above is enough.
       * speakDirect is the one caller with no server between it and the mouth
       * -- it fires the instant the camera answers, which for a question the
       * running observation already covers is under a millisecond. That beats
       * `response.done` for the function call down the data channel, and the
       * early return then meant the line never opened an utterance, its text
       * was dropped by the relay as belonging to a turn that was over, and
       * "what do you see outside" got silence.
       *
       * There is nothing to protect in the response being superseded: it
       * produced a function call, not speech, and the line taking the mouth
       * from it IS the answer to that call. */
      if (speaking) {
        if (speaking.responseId === responseId) return;
        if (!speaking.finishing && !opts.direct) return;
        endResponse(speaking.responseId);
      }
      var entry = { responseId: responseId, resolve: null, cancelled: false,
                    direct: !!opts.direct,
                    // For the onset guard: when the response opened, and (set
                    // on the first transcript delta) when she began speaking.
                    startedAt: Date.now(), audioAt: 0 };
      speaking = entry;
      counters.responses++;
      partial = '';
      generated = '';
      if (sink) { try { sink.begin(responseId); } catch (e) {} }
      audio.unmute();
      arbiter.say({
        priority: arbiter.P.CONVO,
        group: 'convo',
        id: 'live:' + (responseId || String(counters.responses)),
        text: '',
        meta: { source: 'realtime', response_id: responseId },
        // No TTL: an answer does not expire on a clock the way a turn does.
        // The watchdog is long because a considered answer can run to several
        // sentences, and longer still while a tool is running.
        maxMs: 90000,
        play: function () {
          return new Promise(function (resolve) { entry.resolve = resolve; });
        },
        /* The arbiter has given the mouth to something that matters more.
           Mute first — that is instant, and covers the audio already on its way
           — then tell the model to stop, so she does not carry on underneath a
           warning and reappear halfway through a sentence.

           BOTH OF THOSE ONLY IF THIS IS STILL THE ANSWER BEING SPOKEN, and
           that guard is the whole of a bug that ate tool answers.

           `response.cancel` and a mute name nothing: they stop whatever is
           generating and silence whatever is playing, which is right when this
           entry IS that. It is exactly wrong when the entry is already over.
           And it routinely is: the arbiter finishes an item on its own pump, a
           tick after endResponse resolves it, while a response that called a
           tool is a response the server finishes and replaces IMMEDIATELY --
           there is no audio to wait for, so the utterance completes with zero
           chunks and the follow-up arrives inside that tick.

           The new answer then claims the mouth, the arbiter sees a second item
           in the `convo` group and supersedes the first, the first one's
           `stop` runs -- and cancels the answer that replaced it. Measured in
           a live drive: "what are the directions", nav_directions called and
           answered, and the response carrying the directions came back
           `cancelled / client_cancelled` having said nothing. */
        stop: function () {
          entry.cancelled = true;
          entry.said = saidSoFar();     // captured before the deltas stop
          if (speaking !== entry) return;
          audio.mute();
          cancelGeneration();
        },
        onDone: function (reason) {
          var wasCurrent = (speaking === entry);
          if (wasCurrent) speaking = null;
          if (reason === 'spoken') {
            /* NOT a signal that the answer finished. `spoken` is what the
               arbiter is told whenever the mouth is handed back cleanly, and a
               barge-in hands it back cleanly -- endResponse resolves the item
               so the queue moves on. The resume budget is reset where the
               model says the answer actually completed (response.done, status
               completed) and where the driver starts a turn of their own. */
          } else {
            counters.interrupted++;
            // Same reason as `stop` above: an entry that is no longer the one
            // speaking must not silence the one that is.
            if (wasCurrent) audio.mute();
            var said = entry.said || saidSoFar();
            if (reason === 'preempted') {
              /* Something that matters more took the mouth mid-sentence. That
                 is correct and stays correct -- but the answer underneath it
                 was not wrong, it was just outranked, and dropping it made the
                 driver ask again for something RIO had already worked out.
                 It waits, and finishes when the mouth comes back.

                 Recorded a tick later, because at this instant the arbiter has
                 stopped us and has NOT yet started whatever stopped us -- both
                 happen inside one say() and this is the middle of it. Asking
                 now for the name of the pre-empting item gets the name of the
                 item being pre-empted. */
              /* A DIRECT LINE IS NOT RESUMED, and that is the right
                 behaviour rather than a gap. It describes what the road looked
                 like a second ago; by the time the warning that pre-empted it
                 has finished, the second half of it is a description of
                 somewhere the car has left. The observer will have written a
                 newer one before she is asked again. */
              if (!entry.direct) armResume('preempted', said);
              setTimeout(function () {
                noteCutoff('preempted', {
                  response_id: responseId, said: said,
                  by: lastArbiterStart ? {
                    id: lastArbiterStart.id, group: lastArbiterStart.group,
                    priority: lastArbiterStart.priority,
                  } : null,
                });
                tryResume();
              }, 0);
            } else if (reason !== 'superseded') {
              /* superseded is ordinary turn-taking -- a newer answer replacing
                 an older one in the same group -- and is nobody's fault and
                 nothing to resume. Everything else (the watchdog, an error, a
                 clear) is counted so it cannot hide inside "she cut out". */
              noteCutoff('other', { response_id: responseId, reason: reason });
            }
          }
          emit('LIVE_RESPONSE_END', { response_id: responseId, reason: reason });
        },
      });
      emit('LIVE_RESPONSE_START', { response_id: responseId });
    }

    /* THE API REFUSED THE RESPONSE OUTRIGHT.
     *
     * Not a cut-off: nothing was said and there is nothing to resume from. It
     * used to be filed under `other` with everything else that has no name,
     * and it is worth its own, because in a live drive it has one cause and a
     * shape the driver can describe.
     *
     * The cause is the token-per-minute ceiling, and the shape is "she only
     * fails when she has to use a tool". A tool turn spends TWO responses --
     * the one that calls the tool and the one that answers from the result --
     * where an ordinary reply spends one, and each carries the whole
     * instruction set and tool list as input. So the questions that need a
     * camera, a route or the reasoning model are the ones that run the budget
     * out, and "hello" keeps working right through it. The measurement is in
     * tools/realtime_selftest.py (run_session_cost).
     *
     * ONE RETRY, at the delay the API itself names. Not more: a second refusal
     * means the budget is genuinely gone and asking again only spends the next
     * minute's. Not sooner either -- an immediate retry is refused by the same
     * ceiling and buys nothing.
     *
     * A retry is armed once per driver turn. The one thing worse than an
     * answer arriving late is two of them arriving. */
    function retryAfterS(error) {
      var m = /try again in ([0-9.]+)\s*s/i.exec((error && error.message) || '');
      var s = m ? parseFloat(m[1]) : 1.0;
      if (!isFinite(s) || s < 0.4) s = 0.4;
      return Math.min(s, 10);
    }

    function responseFailed(responseId, error) {
      counters.responses_failed++;
      var code = (error && error.code) || 'unknown';
      emit('LIVE_RESPONSE_FAILED', { response_id: responseId, code: code,
                                     message: (error && error.message) || null,
                                     retrying: retryArmed });
      if (!retryArmed || stopped || dictation) {
        // Nothing more to try. Recorded as a cut-off so a drive's tally still
        // accounts for the answer the driver never got.
        noteCutoff('other', { response_id: responseId,
                              reason: 'response_failed:' + code });
        return;
      }
      retryArmed = false;
      counters.responses_retried++;
      var waitMs = Math.round(retryAfterS(error) * 1000);
      /* WHICH TURN THIS BELONGS TO. The wait is seconds long and a driver
         does not stop talking for it -- if they have asked something else by
         the time it comes round, the answer to the old question is no longer
         wanted and saying it would be RIO answering a question nobody is
         still asking. */
      var seq = turnSeq;
      setTimeout(function () {
        if (stopped || dictation || turnSeq !== seq) return;
        // A response that is only waiting on its tail is not in the way; one
        // still being written is.
        if (speaking && !speaking.finishing) return;
        try { send({ type: 'response.create' }); } catch (e) {}
        emit('LIVE_RESPONSE_RETRY', { after_ms: waitMs, code: code });
      }, waitMs);
    }

    /* The model has stopped writing. That is not the same moment as the driver
       having heard the end of it, and the mouth belongs to the second one.
     *
     * When RIO speaks for herself the two are the same event and this is a
     * pass-through. When something else speaks for her they are seconds apart:
     * the model finishes an answer in a few hundred milliseconds and the
     * synthesiser is still four seconds from the end of it. Handing the mouth
     * back at `response.done` would let the next queued thing start talking
     * over the second half of her sentence -- which is not a warning
     * pre-empting her, it is two voices at once. */
    function finishResponse(responseId) {
      if (!sink || !speaking) { endResponse(responseId); return; }
      if (responseId && speaking.responseId && speaking.responseId !== responseId) {
        endResponse(responseId);
        return;
      }
      var entry = speaking;
      // Written down rather than inferred: from here the answer is finished
      // being WRITTEN and is only waiting to finish being HEARD, and a new
      // response arriving in that window is allowed to take the mouth.
      entry.finishing = true;
      var release = function () {
        if (speaking === entry) endResponse(responseId);
      };
      try { sink.end(responseId).then(release, release); }
      catch (e) { release(); }
    }

    function endResponse(responseId) {
      if (!speaking) return;
      if (responseId && speaking.responseId && speaking.responseId !== responseId) return;
      var entry = speaking;
      speaking = null;
      /* Nothing held over an utterance outlives the utterance. A guard still
         counting down against a response that is over would fire bargeIn()
         into the next one, at whatever point of ITS opening it happened to
         land. */
      clearOnsetHold();
      stopEchoWatch();
      if (entry.resolve) entry.resolve();          // the arbiter marks it spoken
    }

    /* THE DRIVER MIGHT HAVE STARTED TALKING.
     *
     * "Might" is the whole change. This used to be certain: the detector fired,
     * RIO went quiet and her answer was thrown away, and the server had already
     * cancelled her generation before this code ran. In a car the thing that
     * fires the detector is very often not the driver -- it is RIO's own voice
     * returning through the cabin, a cough, an indicator, a door -- and every
     * one of those cost a complete answer that the driver then had to ask for
     * again. That is the bug.
     *
     * Two clocks now, and they do different jobs:
     *
     *   MUTE IS INSTANT and unconditional. Talking over the driver is the one
     *   thing that must never happen, it costs nothing to be wrong about, and
     *   it is undoable -- if the noise stops, she is unmuted mid-sentence and
     *   carries on. The driver loses a few hundred milliseconds of audio, not
     *   an answer.
     *
     *   CANCELLING IS DELAYED by bargeSustainMs. Below that it was a noise and
     *   nothing is cancelled at all: the input buffer is cleared so the blip
     *   cannot become a turn of its own, and she is unmuted. Above it, she
     *   stops generating -- and even then the cut-off is not yet classified,
     *   because whether a person was there is a question only the transcript
     *   can answer. See transcriptArrived().
     *
     * The server no longer does any of this (interrupt_response is off, see
     * realtime.session_config): deciding it there would be deciding it a
     * network hop away from the microphone, on the first sample that crossed a
     * threshold, with nothing available to check the guess against.
     */
    /* ---- the meter, and what it is allowed to conclude --------------------
     *
     * `levels()` is the page's own measurement of two things it is the only
     * place that can measure: what the microphone is hearing, and what this
     * page is rendering to the speaker, both in dBFS. The controller does no
     * audio work of its own -- it asks, and it is prepared to be told nothing.
     *
     * ABSENCE IS NOT EVIDENCE. No meter, a meter that throws, an output below
     * the floor: all of them mean the level test has nothing to say, and the
     * gate behaves exactly as it did before any of this. The test can only
     * ever SUPPRESS a barge-in it is confident about; it can never cause one.
     */
    function readLevels() {
      if (!levels) return null;
      var v;
      try { v = levels(); } catch (e) { return null; }
      if (!v || typeof v.mic !== 'number' || typeof v.out !== 'number') return null;
      if (!isFinite(v.mic) || !isFinite(v.out)) return null;
      lastLevels = v;
      return v;
    }

    /* Is the microphone hearing the room, or the loudspeaker?
     *
     * PEAK, NOT AVERAGE, and the choice matters. A driver's interruption is a
     * syllable followed by more of them, and the evidence is the loudest
     * instant; averaging it against the quiet either side is exactly how a
     * real interruption gets mistaken for echo. So the margin is tracked as a
     * running maximum over the firing, and once it has been beaten it stays
     * beaten. */
    function echoShaped() {
      if (!bargeEchoMarginDb) return false;         // desk: the test is off
      var v = readLevels();
      if (!v) return false;                         // no meter: say nothing
      if (v.out < bargeEchoFloorDb) return false;   // too quiet to be echoing
      var margin = v.mic - v.out;
      if (peakMargin === null || margin > peakMargin) peakMargin = margin;
      return peakMargin < bargeEchoMarginDb;
    }

    /* Suppressed, but still listening. The detector fires once per utterance,
       so a firing dismissed as echo must not close the door on the driver who
       starts talking 200 ms into it -- there will be no second speech_started
       to reconsider. This keeps reading the meter for as long as the detector
       says something is going on, and hands control back to bargeIn() the
       moment the microphone beats the loudspeaker. */
    function startEchoWatch(rid) {
      if (levelTimer || !levels || !bargeEchoMarginDb) return;
      levelTimer = setInterval(function () {
        if (stopped || !speechActive || !speaking || speaking.responseId !== rid) {
          stopEchoWatch();
          return;
        }
        if (!echoShaped()) {              // it is a person after all
          stopEchoWatch();
          bargeIn();
        }
      }, LEVEL_POLL_MS);
    }

    /* The same meter, read for a different reason: while the sustain gate is
       open, nothing is being decided tick by tick -- the readings are being
       ACCUMULATED, so that when the gate closes there is a whole window of
       evidence rather than the one sample that happened to coincide with the
       detector firing. */
    function startPendingWatch() {
      if (levelTimer || !levels || !bargeEchoMarginDb) return;
      levelTimer = setInterval(function () {
        if (stopped || !pendingBarge) { stopEchoWatch(); return; }
        echoShaped();
      }, LEVEL_POLL_MS);
    }

    function stopEchoWatch() {
      if (levelTimer) { clearInterval(levelTimer); levelTimer = null; }
    }

    /* How much of her opening is left. Anchored on the first evidence that she
       is actually producing speech for this response rather than on the moment
       the response opened: between those two is a tool call, a reasoning pass
       and a network hop, and a guard measured from the wrong one of them is
       either useless or a gag. */
    function onsetRemaining() {
      if (!bargeOnsetGuardMs || !speaking) return 0;
      var began = speaking.audioAt || speaking.startedAt || 0;
      if (!began) return 0;
      var left = bargeOnsetGuardMs - (Date.now() - began);
      return left > 0 ? left : 0;
    }

    function clearOnsetHold() {
      if (onsetHold) { clearTimeout(onsetHold); onsetHold = null; }
    }

    function bargeIn() {
      counters.barge_ins++;
      /* A DICTATION IS NOT YIELDED. A gap warning, a turn or a tire fault is
         the one thing on this ladder that outranks the driver -- that is what
         the priority tiers are for, and cutting through the person in the seat
         is precisely its job. Muting it because somebody coughed hands the
         cabin a veto over the safety channel, which is the same fault as the
         one being fixed here and with a much worse ending. Conversation
         yields; warnings do not. */
      if (dictation) {
        emit('LIVE_BARGE_IN', { during: 'dictation', yielded: false });
        return;
      }

      /* HER OPENING SYLLABLE IS THE WORST EVIDENCE THERE IS. The canceller has
         not converged, the level steps from nothing to driving volume, and on
         a phone that step is the single most reliable way to make the detector
         fire on her. So the decision is HELD for the rest of the guard rather
         than taken -- and then taken anyway, from that moment, if whatever
         fired the detector is still going. A driver who talks over her first
         word still stops her; they stop her a fifth of a second later. */
      var onsetLeft = onsetRemaining();
      if (onsetLeft > 0 && speaking && !pendingBarge) {
        if (!onsetHold) {
          var heldFor = speaking.responseId;
          emit('LIVE_BARGE_DEFERRED', { response_id: heldFor,
                                        guard_ms: bargeOnsetGuardMs,
                                        holding_ms: Math.round(onsetLeft) });
          onsetHold = setTimeout(function () {
            onsetHold = null;
            if (stopped || !speechActive) return;
            if (!speaking || speaking.responseId !== heldFor) return;
            counters.onset_deferred++;
            bargeIn();                   // past the guard: decide it properly
          }, onsetLeft);
        }
        return;
      }

      /* ...AND IS THE MICROPHONE EVEN LOUDER THAN THE SPEAKER? Echo cannot be
         louder than what produced it. This is the one test in the whole gate
         that does not depend on timing, and it is the reason a phone can now
         tell "she is talking" from "somebody is talking over her" at all. */
      peakMargin = null;
      if (echoShaped()) {
        counters.echo_suppressed++;
        var rid0 = speaking ? speaking.responseId : null;
        emit('LIVE_ECHO_SUPPRESSED', {
          response_id: rid0,
          mic_db: lastLevels ? Math.round(lastLevels.mic * 10) / 10 : null,
          out_db: lastLevels ? Math.round(lastLevels.out * 10) / 10 : null,
          margin_db: peakMargin === null ? null : Math.round(peakMargin * 10) / 10,
          required_db: bargeEchoMarginDb,
          reason: 'level_margin',
        });
        // Nothing muted, nothing cancelled, and the meter stays on it.
        startEchoWatch(rid0);
        return;
      }
      stopEchoWatch();

      audio.mute();                      // instant, always, undoable
      if (!speaking || pendingBarge) {
        emit('LIVE_BARGE_IN', { response_id: speaking ? speaking.responseId : null });
        return;
      }
      var rid = speaking.responseId;
      emit('LIVE_BARGE_IN', { response_id: rid });
      pendingBarge = { responseId: rid, cancelled: false, timer: null,
                       confirm: null, said: '', direct: !!speaking.direct };
      startPendingWatch();
      pendingBarge.timer = setTimeout(function () {
        if (!pendingBarge) return;
        pendingBarge.timer = null;
        /* THE WHOLE WINDOW, NOT THE FIRST SAMPLE. The gate has been open for
           bargeSustainMs and the meter has been read all through it. If the
           microphone never once beat the loudspeaker in that time, the thing
           that has been going on for 600 ms is her: unmute her and let the
           answer finish. This is the case the first read cannot decide -- an
           utterance that had not reached the speaker yet when the detector
           fired. */
        if (bargeEchoMarginDb && peakMargin !== null &&
            peakMargin < bargeEchoMarginDb) {
          counters.echo_suppressed++;
          emit('LIVE_ECHO_SUPPRESSED', {
            response_id: rid,
            margin_db: Math.round(peakMargin * 10) / 10,
            required_db: bargeEchoMarginDb,
            reason: 'level_margin_sustained',
          });
          clearBarge();
          try { send({ type: 'input_audio_buffer.clear' }); } catch (e) {}
          if (speaking && !speaking.cancelled && !stopped) audio.unmute();
          startEchoWatch(rid);
          return;
        }
        pendingBarge.cancelled = true;
        pendingBarge.said = saidSoFar();
        // Sustained past the gate: stop generating and hand back the mouth.
        cancelGeneration();
        endResponse(rid);
        /* Cancelled, and NOT yet blamed on anyone. The clock that decides
           whether there was a person there does not start here -- it starts
           when the speech stops, in speechStopped(). See the note there; this
           is only the backstop for a detector that never says it stopped. */
        pendingBarge.backstop = setTimeout(function () {
          if (!pendingBarge) return;
          var said = pendingBarge.said;
          pendingBarge = null;
          noteCutoff('other', { response_id: rid, said: said,
                                reason: 'speech_never_ended' });
        }, BARGE_BACKSTOP_MS);
      }, bargeSustainMs);
    }

    /* The speech stopped. Two very different situations, decided by whether
       the gate had already closed.
     *
     * Before the gate: nothing was ever cancelled and she simply carries on.
     * The input buffer is cleared on the way past, best effort -- the server
     * may already have committed the blip by the time this lands, and what
     * actually stops it becoming a second answer over the top of the first is
     * the API's own rule that a conversation has one active response at a
     * time. The clear is worth sending anyway for the case where it arrives in
     * time, and costs nothing when it does not.
     *
     * After it: she has already been stopped, and NOW the wait for a transcript
     * begins. Starting that clock at the cancel instead -- which is what this
     * did first, and what the ten-turn probe caught -- times the wait against
     * the wrong event entirely. A driver saying two seconds' worth of sentence
     * produces a transcript two and a half seconds after the detector fired,
     * and a confirmation window measured from the cancel would have expired
     * long before, declared that nobody had spoken, and had RIO resume her old
     * answer over the top of a driver who was still finishing theirs. That is
     * the one failure worse than the bug this whole change is fixing.
     *
     * Measured from here, the window means what it is supposed to mean: the
     * words are already in the buffer, transcription follows within a beat, and
     * silence past it really is silence.
     */
    function speechStopped() {
      speechActive = false;
      stopEchoWatch();
      /* A firing that never got past her opening syllable, and has now stopped
         on its own. That is the shape of echo and it cost nothing: no mute, no
         cancel, no gap in the answer. Counted where the other suppressions are
         counted, because the question it answers is the same one -- how often
         is the detector firing on her. */
      if (onsetHold) {
        clearOnsetHold();
        counters.echo_suppressed++;
        emit('LIVE_ECHO_SUPPRESSED', {
          response_id: speaking ? speaking.responseId : null,
          reason: 'onset_guard',
          guard_ms: bargeOnsetGuardMs,
        });
        try { send({ type: 'input_audio_buffer.clear' }); } catch (e) {}
        return;
      }
      if (!pendingBarge) return;
      if (!pendingBarge.cancelled) {
        clearBarge();
        try { send({ type: 'input_audio_buffer.clear' }); } catch (e) {}
        counters.blips_absorbed++;
        emit('LIVE_BARGE_ABSORBED', {
          response_id: speaking ? speaking.responseId : null });
        if (speaking && !speaking.cancelled && !stopped) audio.unmute();
        return;
      }
      if (pendingBarge.confirm) return;          // already waiting
      var wasDirect = pendingBarge.direct;
      if (pendingBarge.backstop) {
        clearTimeout(pendingBarge.backstop);
        pendingBarge.backstop = null;
      }
      var rid = pendingBarge.responseId;
      pendingBarge.confirm = setTimeout(function () {
        if (!pendingBarge) return;
        var said = pendingBarge.said;
        pendingBarge = null;
        noteCutoff('false_barge_in', { response_id: rid, said: said,
                                      detail: 'no transcript followed' });
        // ...unless she was reading the road out. See beginResponse.
        if (!wasDirect) { armResume('false_barge_in', said); tryResume(); }
      }, bargeConfirmMs);
    }

    /* The words, or the absence of them. This is what classifies a barge-in,
       and it is the only thing that can: a detector says "energy", a
       transcript says "someone spoke". An empty transcript is a real answer
       here and not a failure -- it is the transcriber saying there was nothing
       to write down. */
    function transcriptArrived(text) {
      var real = !!(text && text.trim());

      /* IS IT HERS? Asked before anything is done about it.
       *
       * A transcript that does not clear this bar is not a driver turn at all:
       * it does not supersede, it does not advance the turn counter, it does
       * not become the question the camera answers, and it does not reset the
       * resume budget. It is her own voice, and the only correct response to
       * hearing yourself is to carry on.
       *
       * TWO DIFFERENT BARS, and the difference is what each path DOES.
       *
       * A supersede cancels: it stops the response, aborts the tool calls
       * behind it, and clears the mouth. That is a barge-in by another name
       * and it clears the barge-in bar -- sustained speech the level test did
       * not call an echo.
       *
       * A continuation cancels far less: the driver is still mid-sentence and
       * semantic VAD split them, so the second half is the SAME speech and may
       * well be absorbed as a blip by a gate that already confirmed the first
       * half. Requiring a fresh confirmed barge for it would answer half of
       * every question that has a breath in it. It gets the echo TEXT test
       * alone -- which is the right test anyway, because her own words are
       * never a continuation of the driver's.
       */
      var at = now();
      var gap = lastTurnAt ? (at - lastTurnAt) : Infinity;
      var cmd = real ? driverCommand(text) : null;
      var cont = real && !cmd && isContinuation(lastTurnText, text, gap);
      var gate = !real ? { allow: false, why: 'empty' }
               : cont ? (looksLikeEcho(text)
                          ? { allow: false, why: 'echo_of_her_own_words' }
                          : { allow: true, why: null })
               : supersedeGate(text);
      if (real && !gate.allow) {
        counters.turns_phantom++;
        emit('LIVE_TURN_PHANTOM', {
          why: gate.why, text: text.slice(0, 160), turn: turnSeq,
          speaking: !!(speaking && !speaking.cancelled),
          said_chars: (saidSoFar() || '').length,
          since_audio_ms: Math.round(now() - lastAudioAt),
        });
        real = false;
      }
      if (real) {
        transcriptFresh = true;
        // A new turn from the driver. Whatever was outstanding is theirs to
        // have interrupted, and the resume budget starts again -- and so does
        // the one retry a refused response gets.
        pendingResume = null;
        resumeChain = 0;
        retryArmed = true;

        if (cont) {
          /* ONE QUESTION IN TWO BREATHS. The turn number does NOT advance, so
             nothing belonging to it is aborted -- the tool call already
             running is running for the question being asked. What IS cancelled
             is the response the server created for half a question; both
             fragments are already in the conversation, so the response that
             follows this one answers the whole thing. */
          counters.turns_coalesced++;
          if (speaking && !speaking.cancelled) {
            var rid0 = speaking.responseId;
            cancelGeneration();
            endResponse(rid0);
          }
          emit('LIVE_TURN_COALESCED', {
            turn: turnSeq, gap_ms: Math.round(gap),
            first: lastTurnText.slice(0, 160), second: text.slice(0, 160),
          });
          lastTurnText = (lastTurnText + ' ' + text).trim();
          lastTurnAt = at;
        } else {
          var previous = lastTurnText;
          turnSeq++;
          lastTurnText = text;
          lastTurnAt = at;
          /* Anything outstanding belongs to the question before this one.
             Superseding is unconditional: it costs a response that nobody is
             waiting for, and NOT superseding costs an answer to the wrong
             question, out loud, in a car. */
          supersedeTurn(cmd ? 'driver_command' : 'new_utterance',
                        { previous: previous, next: text });
          if (cmd) driverCommandTurn(cmd, text);
        }
      }
      if (!pendingBarge) return;
      var rid = pendingBarge.responseId;
      var said = pendingBarge.said || saidSoFar();
      var wasCancelled = pendingBarge.cancelled;
      var wasDirect = pendingBarge.direct;
      clearBarge();
      if (!wasCancelled) {
        // Words arrived before the gate even closed. Real, and early: cancel
        // now rather than waiting out a timer that is about to agree.
        if (real && speaking && speaking.responseId === rid) {
          var heard = saidSoFar();
          cancelGeneration();
          endResponse(rid);
          noteCutoff('barge_in', { response_id: rid, said: heard });
        }
        return;
      }
      if (real) {
        noteCutoff('barge_in', { response_id: rid, said: said });
        return;                          // deliberate. Never resumed.
      }
      noteCutoff('false_barge_in', { response_id: rid, said: said,
                                    detail: 'empty transcript' });
      if (!wasDirect) { armResume('false_barge_in', said); tryResume(); }
    }

    /* A COMMAND IS ACTED ON, NOT QUEUED.
     *
     * supersedeTurn has already run by the time this is called, so the mouth
     * is free and no tool call is still out. What is left is the difference
     * between the two kinds:
     *
     *   silence   the driver asked for quiet. Answered by being quiet -- and
     *             deliberately NOT by asking for a response, because replying
     *             to "be quiet" with speech is the wrong shape of obedience.
     *             The model is told, out of band, so the next thing said is
     *             not a continuation of what was just cut off.
     *
     *   nav       stopping or replacing a route is teardown of objects that
     *             only exist in this page, and the panel already owns both
     *             (LOCAL_TOOLS). Doing it here rather than waiting for the
     *             model to call the same function saves the round trip that
     *             this whole item is about -- on the first real drive that
     *             round trip queued behind tool calls of forty seconds.
     *             The phrases are anchored whole-utterance matches with no
     *             other meaning in a car; see config.REALTIME_DRIVER_COMMANDS.
     */
    function driverCommandTurn(kind, text) {
      counters.commands_preempted++;
      emit('LIVE_DRIVER_COMMAND', { command: kind, text: text.slice(0, 160),
                                    turn: turnSeq });
      if (kind === 'silence') {
        try { audio.mute(); } catch (e) {}
        try {
          if (root.RIO && RIO.speech && RIO.speech.clear) RIO.speech.clear('convo');
        } catch (e) {}
        // Told, not asked. `conversation: none` would hide it from the
        // history; this belongs IN the history, because "she stopped because I
        // told her to" is context for the next thing she says.
        try {
          send({ type: 'conversation.item.create',
                 item: { type: 'message', role: 'assistant',
                         content: [{ type: 'output_text',
                                     text: '(stopped at the driver\'s request)' }] } });
        } catch (e) {}
        return;
      }
      var toolName = (kind === 'reroute') ? 'reroute' : 'stop_navigation';
      var fn = LOCAL_TOOLS[toolName];
      if (!fn) return;
      var result;
      try { result = fn({}); } catch (e) { result = { ok: false, note: 'panel error' }; }
      Promise.resolve(result).then(function (r) {
        emit('LIVE_COMMAND_DONE', { command: kind, ok: !!(r && r.ok),
                                    note: (r && r.note) || null });
        // The session is told what the panel did, so the model does not call
        // the same tool a second time and does not answer as though the route
        // were still running.
        try {
          send({ type: 'conversation.item.create',
                 item: { type: 'message', role: 'assistant',
                         content: [{ type: 'output_text',
                                     text: JSON.stringify({ command: kind,
                                                            done: true,
                                                            result: r || null }) }] } });
        } catch (e) {}
      });
    }

    function dictationStarted() {
      if (!dictation || dictation.started) return;
      dictation.started = true;
      if (dictation.timer) { clearTimeout(dictation.timer); dictation.timer = null; }
      audio.unmute();
      emit('LIVE_DICTATION_START', { text: dictation.text });
      if (typeof dictation.onStart === 'function') {
        try { dictation.onStart(); } catch (e) {}
      }
    }

    function finishDictation(err) {
      if (!dictation) return;
      var d = dictation;
      dictation = null;
      if (d.timer) { clearTimeout(d.timer); d.timer = null; }
      if (err) {
        counters.dictation_failures++;
        emit('LIVE_DICTATION_FAILED', { text: d.text, reason: err });
        d.reject(new Error(err));
      } else {
        counters.dictated++;
        emit('LIVE_DICTATION_END', { text: d.text, transcript: d.transcript || '' });
        d.resolve({ transcript: d.transcript || '' });
      }
      /* A dictated line is what usually pre-empts a conversation -- a gap
         warning, a turn, a tire fault. It does not go through the arbiter, so
         the arbiter's own "mouth is free" event is not enough on its own:
         the answer it interrupted waits here too. This is also the whole of
         what desk-testing a clip needs. The warnings from an uploaded video
         still fire, exactly as they do on the road, and they stop costing the
         conversation that was underneath them. */
      setTimeout(tryResume, 0);
    }

    /* SPEAK A LINE THAT NEEDS NO COMPOSING.
     *
     * The running observation is already one short sentence in RIO's register
     * — the observer is prompted for her voice and the server checks the
     * result against persona.lint() before ever offering it here — so for a
     * general question about the road there is nothing for a model to add.
     *
     * Measured, that model pass cost ~450 ms of remote composition plus the
     * round trip either side, on an answer the camera had ready in four
     * milliseconds. This is the same sentence, spoken.
     *
     * It is a full conversational utterance in every other respect: it claims
     * the mouth at CONVO priority through the arbiter, a warning cuts through
     * it, a barge-in stops it, and the session is told what she said so the
     * next question lands against a conversation that happened. What it does
     * not do is ask a model to say it. */
    /* THE SAME LINE, SPOKEN BY THE SESSION ITSELF.
     *
     * Under the speech-to-speech backend there is no sink to push words into:
     * RIO's voice IS the session, so a line that is not to be composed has to
     * be READ by her -- which is the verbatim injection the deterministic
     * channels already use, out of band, with the words at the end of the
     * instruction.
     *
     * WITHOUT THIS THE QUESTION WENT SILENT, and silently. The server offers
     * the vetted sentence and, in the same tool result, tells the model:
     * "This has ALREADY BEEN SAID to the driver, out loud, in your voice. Do
     * not say it again." That is true when something speaks it. When the
     * browser had no sink it skipped speaking it and asked for an ordinary
     * response anyway -- so the model was told not to repeat a line nobody had
     * said, and said nothing at all. Measured on a real drive: "What do you
     * see outside?" answered by the camera in 44 ms, path=observer_direct, and
     * one audio event of silence.
     *
     * Not `speak()`, though it is the same wire message: `speak()` is
     * dictation, which is out of band AND at the caller's own warning
     * priority. This is a conversational answer -- CONVO priority, cut off by
     * a warning, stopped by a barge-in -- and speakDirect has already claimed
     * the mouth on those terms before this is reached. */
    function injectDirect(line, release, onSpoken) {
      return new Promise(function (resolve) {
        var d = { line: line, realId: null, started: false, timer: null,
                  onSpoken: onSpoken };
        d.finish = function (okay, why) {
          if (directSpeech !== d) return;
          directSpeech = null;
          if (d.timer) { clearTimeout(d.timer); d.timer = null; }
          if (!okay) {
            // Given up on before it was ever bound to a response? Then the
            // response is still on its way and is nobody's. See
            // orphanOutOfBand.
            if (!d.realId) orphanOutOfBand++;
            counters.direct_speech_failures++;
            emit('LIVE_DIRECT_SPEECH_FAILED', { text: line, reason: why });
            /* A LAST TRY, rather than a silent turn. The tool result is
               already in the conversation, so asking for a response gets the
               question answered late instead of not at all -- the same
               recovery the mouth-was-busy branch makes. Nothing was written
               into the history claiming she spoke, because that is written
               from `started` and this line never started. */
            try { send({ type: 'response.create' }); } catch (e) {}
          }
          release();
          resolve(okay);
        };
        directSpeech = d;
        /* The same budget a dictated warning gets, and for the same reason:
           NOT the warning budget, and the difference is the argument rather
           than the mechanism: 900 ms exists because a late warning has stopped
           being a warning, and a scene answer arriving a second later has not
           stopped being an answer. On a real drive the warning budget fired on
           a line that then spoke perfectly well. See
           config.REALTIME_DIRECT_SPEECH_TIMEOUT_MS. */
        d.timer = setTimeout(function () {
          try { send({ type: 'response.cancel' }); } catch (e) {}
          d.finish(false, 'timeout');
        }, directSpeechTimeoutMs);
        try {
          audio.unmute();
          send({
            type: 'response.create',
            response: {
              conversation: 'none',
              output_modalities: ['audio'],
              instructions: verbatimInstruction + line,
            },
          });
        } catch (e) {
          d.finish(false, 'send_failed');
        }
      });
    }

    function speakDirect(text, meta) {
      var line = (text || '').trim();
      if (stopped || !line) return false;
      var id = 'direct:' + (++directs);
      beginResponse(id, { direct: true });
      var entry = speaking;
      /* COULD NOT TAKE THE MOUTH, AND SAYS SO. `false` rather than a promise
         of `false`, because the caller has to know NOW: a line that is not
         going to be spoken has to become a request to the model in the same
         turn, and a driver's question is not something to discover was
         dropped one microtask later. */
      if (!entry || entry.responseId !== id) return false;
      counters.spoken_directly++;
      emit('LIVE_DIRECT_ANSWER', {
        text: line, path: (meta && meta.path) || 'observer_direct',
        seen_s_ago: meta && meta.seen_s_ago });
      var release = function () {
        if (speaking === entry) endResponse(id);
        return true;
      };
      if (!sink) return injectDirect(line, release, meta && meta.onSpoken);
      try {
        sink.delta(id, line);
        return sink.end(id).then(release, release);
      } catch (e) {
        release();
        return Promise.resolve(false);
      }
    }

    function toolCall(name, callId, argsJson) {
      counters.tool_calls++;
      emit('LIVE_TOOL_CALL', { tool: name, call_id: callId, turn: turnSeq });
      var args = argsJson;
      if (typeof argsJson === 'string') {
        try { args = JSON.parse(argsJson || '{}'); } catch (e) { args = {}; }
      }
      /* WHICH QUESTION THIS CALL IS FOR, and a handle to stop it.
       *
       * The turn is captured now, at the moment the model asks for the tool,
       * and checked again when the result lands. Everything between those two
       * moments is where the first real drive spent up to 48 seconds. */
      /* `aborted` is the flag that actually protects the driver, and it works
         with or without an AbortController: a result marked aborted is never
         spoken and never asks for a response. The controller is what
         additionally stops the HTTP request and tells the server, and every
         browser this runs in has had one for years. Where it is missing the
         answer is still discarded; only the network call carries on. */
      var entry = { name: name, turn: turnSeq, at: now(), aborted: false,
                    controller: (typeof AbortController === 'function')
                      ? new AbortController() : null };
      inflightTools[callId] = entry;

      return Promise.resolve()
        .then(function () { return runTool(name, args, entry.controller); })
        .catch(function (e) {
          // The server is unreachable, refused, or this call was aborted
          // because the driver asked something else. Not an error the driver
          // hears about either way.
          if (entry.aborted || (e && e.name === 'AbortError')) {
            return { ok: false, note: 'superseded' };
          }
          return { ok: false, note: 'unreachable' };
        })
        .then(function (result) {
          delete inflightTools[callId];
          result = result || { ok: false, note: 'no result' };
          if (!result.ok && !entry.aborted) counters.tool_failures++;
          if (stopped) return result;

          /* THE ANSWER TO A QUESTION NOBODY IS WAITING FOR.
           *
           * The output still goes back -- a function call left with no output
           * is a malformed conversation and the NEXT turn pays for it -- but
           * it says what happened, nothing is spoken, and no response is
           * requested. This is the line that stops a forty-second-old answer
           * arriving on top of the question the driver actually asked. */
          if (entry.turn !== turnSeq) {
            counters.tool_results_discarded++;
            emit('LIVE_TOOL_RESULT_DISCARDED', {
              tool: name, call_id: callId, turn: entry.turn, now_turn: turnSeq,
              aborted: entry.aborted, took_ms: Math.round(now() - entry.at),
            });
            try {
              send({ type: 'conversation.item.create',
                     item: { type: 'function_call_output', call_id: callId,
                             output: JSON.stringify({
                               ok: false,
                               note: 'superseded — the driver asked something '
                                     + 'else before this came back' }) } });
            } catch (e) {}
            return result;
          }

          send({
            type: 'conversation.item.create',
            item: {
              type: 'function_call_output',
              call_id: callId,
              output: JSON.stringify(result),
            },
          });
          /* Ask for the spoken answer. Without this the model has the result
             and no reason to say anything about it.
           *
           * A CAMERA ANSWER GETS A CEILING. The observation it is composing
           * from is one sentence about the road, and the instructions ask for
           * one sentence back — but instructions are guidance, and measured on
           * this path the answers came back at 23 words median and 44 at p95.
           * Every one of those words is synthesised and played, so on a voice
           * that speaks the whole answer it is time the driver spends being
           * read a caption twice.
           *
           * Only for `look`, and only as a cap: an ordinary answer is a
           * fraction of it. Everything else keeps the session's own limit. */
          /* ...unless the answer is already said. A scene question comes back
             with the sentence itself, in her voice, and asking a model to
             rewrite it is the slowest part of the turn.

             The session is still told what came out, as an assistant message,
             so "tell me more about that" lands against a conversation that
             happened rather than a gap. `output_text` is the only content type
             the API accepts for an assistant item — `text` is refused by
             name. */
          /* SPOKEN FIRST, AND TOLD TO THE SESSION ONLY IF IT WAS SPOKEN.
             The assistant item is a claim about what the driver heard, and
             writing it before knowing whether the line reached a speaker is
             how a turn ends with RIO silent and the history saying she
             answered. If the mouth is not available the line is not lost --
             the tool result is already in the conversation, so asking for a
             response gets the same sentence composed by the model, a few
             hundred milliseconds later instead of none at all. */
          /* NO `&& sink` HERE. That was the whole of the silent scene
             answer: the condition read "...and there is a synthesiser", so on
             the speech-to-speech backend the line was never spoken and the
             ordinary request went out underneath a tool result that said it
             already had been. Whether she CAN say it directly is speakDirect's
             question, and it answers `false` when the mouth is not available
             — which is the branch below. */
          if (name === 'look' && result.speak_directly && result.speech) {
            /* WHEN TO TELL THE SESSION SHE SAID IT, and it is not the same
               moment under both backends.
               Through the sink, handing the words over IS the line being
               spoken -- the sink owns delivery from there and reports its own
               failures. Through the session, the words are a REQUEST, and the
               moment that becomes speech is the audio starting. So the write
               is handed to speakDirect as a callback and fired at whichever
               of those two moments this backend has. */
            var tellSession = function () {
              send({
                type: 'conversation.item.create',
                item: { type: 'message', role: 'assistant',
                        content: [{ type: 'output_text', text: result.speech }] },
              });
            };
            var spoken = speakDirect(result.speech, {
              path: result.path, seen_s_ago: result.seen_s_ago,
              onSpoken: tellSession });
            if (spoken !== false) {
              if (sink) tellSession();
              emit('LIVE_TOOL_RESULT', { tool: name, call_id: callId,
                                         ok: true, path: result.path,
                                         took_ms: result.took_ms || null,
                                         spoke_directly: true });
              return result;
            }
            counters.direct_deferred++;
            emit('LIVE_DIRECT_DEFERRED', { path: result.path,
                                           response_id: speaking
                                             ? speaking.responseId : null });
          }
          var ask = { type: 'response.create' };
          if (name === 'look' && lookAnswerMaxTokens) {
            ask.response = { max_output_tokens: lookAnswerMaxTokens };
          }
          send(ask);
          emit('LIVE_TOOL_RESULT', { tool: name, call_id: callId,
                                     ok: !!result.ok, path: result.path || null,
                                     took_ms: result.took_ms || null,
                                     spoke_directly: false,
                                     note: result.note || null });
          return result;
        });
    }

    return {
      /* One event off the data channel. Everything this file decides is
         decided here, which is what makes it testable without a microphone. */
      handle: function (ev) {
        if (!ev || !ev.type || stopped) return;
        switch (ev.type) {
          case 'response.created':
            beginResponse((ev.response && ev.response.id) || ev.response_id);
            break;
          case 'output_audio_buffer.started':
            if (dictation && dictation.responseId === ev.response_id) {
              // The line is being spoken. Whatever fallback was armed against
              // this taking too long can stand down.
              dictationStarted();
              break;
            }
            if (directSpeech && directSpeech.realId === ev.response_id) {
              // Ditto for a vetted line the session is reading: it has
              // started, so it will not be given up on -- and NOW the session
              // is told she said it. Not before: an assistant message is a
              // claim about what the driver heard, and writing it while the
              // line might still fail to start is how a turn ends silent with
              // the history saying she answered.
              directSpeech.started = true;
              if (directSpeech.timer) {
                clearTimeout(directSpeech.timer);
                directSpeech.timer = null;
              }
              if (directSpeech.onSpoken) {
                var tell = directSpeech.onSpoken;
                directSpeech.onSpoken = null;
                try { tell(); } catch (e) {}
              }
              break;
            }
            // Belt and braces: on some paths audio starts without a
            // response.created having been seen by this client.
            beginResponse(ev.response_id);
            break;
          case 'response.done':
          case 'output_audio_buffer.stopped':
            if (dictation && dictation.responseId ===
                ((ev.response && ev.response.id) || ev.response_id)) {
              finishDictation(null);
              break;
            }
            if (directSpeech && directSpeech.realId ===
                ((ev.response && ev.response.id) || ev.response_id)) {
              /* SPOKEN, OR NOT SPOKEN, AND THE DIFFERENCE IS THE AUDIO.
                 A response that ended without ever starting audio is a line
                 the driver did not hear, whatever its status says, and
                 reporting it as spoken is how a turn ends silent with the
                 history claiming an answer. */
              directSpeech.finish(directSpeech.started, 'ended_unspoken');
              break;
            }
            /* A response can be "done" because it finished or because it ran
               out of room, and the difference is invisible from the audio. The
               cap is deliberate (REALTIME_MAX_RESPONSE_TOKENS) so this is
               counted and never resumed -- resuming it would be arguing with
               the limit -- but it is counted, because a driver hearing an
               answer stop at the same length every time is hearing a fault
               with a name. */
            if (ev.type === 'response.done' && ev.response
                && isOutOfBand(ev.response.id)) {
              // A warning or a vetted line, arriving after whatever was
              // tracking it gave up. Not the conversation, so not a cut-off
              // answer and not a failed one.
              //
              // ...and if it was a DISOWNED one, the element was muted while
              // it died. Give the mouth back here: every other path unmutes on
              // its way in, but a session left muted by a response nobody owns
              // is a drive with no RIO, and the cost of being sure is a line.
              if (!dictation && !speaking && !stopped) {
                try { audio.unmute(); } catch (e) {}
              }
              finishResponse(ev.response.id);
              break;
            }
            if (ev.type === 'response.done' && ev.response) {
              var det = ev.response.status_details || {};
              if (ev.response.status === 'completed') {
                // A whole answer got out. Whatever chain of interruptions led
                // here is over, and the next one starts with a full budget.
                resumeChain = 0;
              }
              if (ev.response.status === 'incomplete'
                  && det.reason === 'max_output_tokens') {
                noteCutoff('token_cap', { response_id: ev.response.id,
                                         said: partial });
              } else if (ev.response.status === 'failed') {
                responseFailed(ev.response.id, det.error || {});
              }
            }
            finishResponse((ev.response && ev.response.id) || ev.response_id);
            break;
          /* TEXT MODE. The words, as they are written, forwarded to whatever
             is speaking them. Nothing is buffered here and no phrasing is
             decided here: the relay chunks at clause boundaries because that
             decision is testable on a server and is not testable in a car. */
          case 'response.output_text.delta':
            if (sink && (!dictation || dictation.responseId !== ev.response_id)) {
              generated += (ev.delta || '');
              // In text mode the sink is the mouth, so this is the moment
              // these words become sound in the room.
              noteSaid(ev.delta || '');
              try { sink.delta(ev.response_id, ev.delta || ''); } catch (e) {}
            }
            break;
          case 'response.output_text.done':
            // What the model says it wrote. Kept for the log and for the
            // panel; the resume still carries what was HEARD, not this.
            if (sink && ev.text) generated = ev.text;
            break;
          case 'response.output_audio_transcript.delta':
            // What she is saying, as she says it. The only record of how far
            // an answer got, and therefore the only thing a resume can carry.
            //
            // A dictated warning and a vetted direct line are both excluded,
            // and for one reason: neither is resumable. Their words were
            // chosen by policy, not composed, so "carry on from where you
            // stopped" is not a thing the model can do with them -- it would
            // rewrite them. A direct line cut off is dropped, which is the
            // same decision transcriptArrived already makes via `wasDirect`.
            /* EVERYTHING THAT REACHES THE SPEAKER, including the dictated
               and direct lines excluded from `partial` below. `partial` is for
               a resume and only a resumable answer belongs in it; this is for
               recognising her own voice on the way back in, and a nav line
               echoes exactly as readily as an answer does. */
            noteSaid(ev.delta || '');
            if ((!dictation || dictation.responseId !== ev.response_id)
                && (!directSpeech || directSpeech.realId !== ev.response_id)) {
              partial += (ev.delta || '');
              /* WHEN SHE ACTUALLY STARTED TALKING, which is not when the
                 response opened: a tool call and a reasoning pass sit between
                 those two and the onset guard measured from the wrong one is
                 either spent before she makes a sound or still running a
                 second into her answer. The first transcript delta is the
                 earliest evidence that audio is on its way out. */
              if (speaking && speaking.responseId === ev.response_id
                  && !speaking.audioAt) {
                speaking.audioAt = Date.now();
              }
            }
            break;
          case 'response.output_audio_transcript.done':
            // What the model says it said. The tests compare it with what it
            // was asked to say; in the car it is what the log records.
            if (dictation && dictation.responseId === ev.response_id) {
              dictation.transcript = ev.transcript || '';
            } else if (directSpeech && directSpeech.realId === ev.response_id) {
              directSpeech.transcript = ev.transcript || '';
            } else if (ev.transcript) {
              partial = ev.transcript;
            }
            break;
          case 'input_audio_buffer.speech_started':
            // Whatever was last transcribed is about the turn before this one.
            transcriptFresh = false;
            speechActive = true;
            bargeIn();
            break;
          case 'input_audio_buffer.speech_stopped':
            speechStopped();
            break;
          case 'response.function_call_arguments.done':
            toolCall(ev.name, ev.call_id, ev.arguments);
            break;
          case 'conversation.item.input_audio_transcription.completed':
            lastTranscript = ev.transcript || '';
            emit('LIVE_TRANSCRIPT', { transcript: lastTranscript, role: 'driver' });
            transcriptArrived(lastTranscript);
            break;
          case 'conversation.item.input_audio_transcription.failed':
            // The transcriber could not make words out of it. That is not
            // proof nobody spoke, but it is the same evidence an empty
            // transcript gives, and the safe reading of "we cannot tell" is
            // the one that does not talk over a driver who might be mid-word:
            // classified, not resumed.
            if (pendingBarge) {
              noteCutoff(pendingBarge.cancelled ? 'barge_in' : 'other',
                         { response_id: pendingBarge.responseId,
                           reason: 'transcription_failed' });
              clearBarge();
            }
            break;
          case 'error':
            emit('LIVE_ERROR', { error: (ev.error && ev.error.message) || 'unknown' });
            break;
          default:
            break;
        }
      },

      onEvent: function (fn) { if (typeof fn === 'function') listeners.push(fn); },

      /* Say one line as her, with no model between it and the speaker. Only
         the visual fast path uses this, and only for a line the server has
         already checked against her register.

         Returns a promise while the line is being spoken, or `false` -- NOT a
         promise of false -- when the mouth could not be had at all, so a
         caller can turn the line into an ordinary request in the same turn. */
      speakDirect: speakDirect,

      /* Dictate one deterministic line — a warning, a turn, a health
         announcement — in RIO's voice, word for word.
       *
       * Out of band (`conversation: "none"`), so the line never enters the
       * conversation history: a warning is a fact about the car, not something
       * RIO said and can later be asked about.
       *
       * The caller already holds the mouth at its OWN priority, which is the
       * whole point — a gap warning dictated here is still a gap warning, and
       * it pre-empted the conversation before it got this far. That is also
       * why this does not go anywhere near the arbiter itself.
       *
       * Rejects if audio has not STARTED within the timeout, so the caller can
       * fall back to a synthesiser that is 200 ms away rather than wait on a
       * session that has gone quiet. */
      speak: function (text, opts) {
        opts = opts || {};
        var line = (text || '').trim();
        if (stopped || !line) return Promise.reject(new Error('no session'));
        /* Not in text mode, and not as a limitation. Dictation exists so a
           warning comes out of the same mouth as the conversation; when that
           mouth is ElevenLabs, /nav/voice and /headway_voice ARE that mouth —
           same voice id, on the model that is fastest to first byte — so the
           line goes there directly. Rejecting immediately is what makes it go:
           rio_speak's fallback is one `catch` away and costs nothing, where
           waiting out the dictation budget would cost a warning most of a
           second for no reason at all. */
        if (sink) return Promise.reject(new Error('text_mode'));
        if (dictation) return Promise.reject(new Error('busy'));
        return new Promise(function (resolve, reject) {
          dictation = {
            text: line, resolve: resolve, reject: reject, started: false,
            responseId: null, transcript: '', onStart: opts.onStart, timer: null,
          };
          dictation.timer = setTimeout(function () {
            /* NEVER HEARD IT START -- AND IT IS STILL COMING.
             *
             * This is the two-voices bug. The budget expires, this gives up,
             * rio_speak.js catches the rejection and synthesises the line
             * instead; and then the response we asked for arrives anyway,
             * because a bare `response.cancel` cancels the ACTIVE response and
             * at this moment there is not one yet -- the create is still in
             * flight. It lands, belongs to nobody, claims the mouth as an
             * ordinary answer, and reads the turn out in her voice over the
             * top of the synthesiser reading the same turn. Measured on a
             * phone: both voices, same sentence, on the same junction.
             *
             * So the line is not merely abandoned, it is DISOWNED: the next
             * unclaimed response is recorded as this one's, cancelled by id
             * the moment it exists, and muted until it is gone. Exactly one
             * mouth per utterance, and it is the one that got there. */
            if (dictation && !dictation.responseId) {
              orphanOutOfBand++;
              silenceOrphan = true;
            }
            try { send({ type: 'response.cancel' }); } catch (e) {}
            finishDictation('timeout');
          }, opts.timeoutMs || speakTimeoutMs);
          try {
            audio.unmute();
            send({
              type: 'response.create',
              response: {
                conversation: 'none',
                output_modalities: ['audio'],
                instructions: verbatimInstruction + line,
              },
            });
          } catch (e) {
            finishDictation('send_failed');
          }
        });
      },

      /* TIER 2: ElevenLabs is not answering at all, and RIO takes her own
         voice back mid-drive.
       *
       * One-way and sticky. A voice that alternates between two people because
       * the network is alternating is worse than either of them, and the
       * driver has no way to interpret it -- so this happens once, is logged
       * once, and the drive finishes in the live session's own voice.
       *
       * The session is asked for audio from here on. It has produced none so
       * far, which is the only reason the voice can be named this late: a
       * realtime session's voice is fixed once it has spoken.
       *
       * NO LITERAL VOICE NAME IN HERE, and that is the point of the throw. The
       * old code said `cfg.cedarVoice || 'cedar'`, which reads as a harmless
       * default and is not one: the day the voice became marin, a session
       * payload that failed to carry it would have handed the driver a
       * different person mid-drive and logged that everything was fine.
       * config.py names the voice, the mint carries it, and if it did not
       * arrive this refuses rather than inventing one. */
      useLiveVoice: function (why) {
        if (!sink || stopped) return false;
        var voice = cfg.liveVoice;
        if (!voice) {
          emit('LIVE_ERROR', { where: 'useLiveVoice', why: 'no_voice_in_session' });
          return false;
        }
        var gone = sink;
        sink = null;
        counters.voice_backend_changed++;
        try { gone.close(); } catch (e) {}
        try {
          send({
            type: 'session.update',
            session: {
              type: 'realtime',
              output_modalities: ['audio'],
              audio: { output: { voice: voice } },
            },
          });
        } catch (e) {}
        emit('LIVE_VOICE_BACKEND', {
          backend: 'openai_realtime', voice: voice,
          why: why || 'elevenlabs_unavailable' });
        return true;
      },

      /* THE PRECONDITIONS, WATCHED, so that attaching them is not something
       * every embedder has to remember.
       *
       * The panel wires this from connect(); the recording harness builds a
       * controller directly and needs the same behaviour, and the drive where
       * it did not have it is the reason this is a method rather than six
       * lines in one caller: asked to avoid the freeway with no `reroute`
       * attached, the model reached for the research tool and narrated a
       * reroute it had not done.
       *
       * `defer` exists for the browser, where a session.update sent before
       * the data channel finishes opening is silently nothing. */
      watchToolConditions: function (bus, defer) {
        if (!bus || !bus.on) return false;
        var self = this;
        var when = defer || function (fn) { fn(); };
        bus.on('*', function (ev) {
          var on = ev.type === 'NAV_ROUTE_ATTACHED' ? true
                 : (ev.type === 'NAV_STOPPED' ? false : null);
          if (on === null) return;
          when(function () { self.setToolCondition('routing', on); });
        });
        // A route that was already live when the session opened — she can be
        // started mid-drive, and the tools have to arrive with the session
        // rather than at the next route event, which may never come.
        if (root.RIO && root.RIO.nav && root.RIO.nav.route) {
          when(function () { self.setToolCondition('routing', true); });
        }
        return true;
      },

      /* A PRECONDITION CHANGED, so the tool list does.
       *
       * Returns whether anything was actually sent, and sends nothing when
       * the answer has not changed: route events arrive several times a drive
       * — every automatic reroute attaches a route — and a session.update per
       * event would spend more than the tools cost and would replace the tool
       * list in the middle of a turn that is using it.
       *
       * `tools` is sent alone. A session.update carrying the whole session
       * would re-send the instructions and the turn detection with it, and
       * setting turn detection again mid-drive resets the detector. */
      setToolCondition: function (name, on) {
        if (!conditionalTools[name]) return false;
        if (!baseTools.length) return false;   // nothing to replace it with
        on = !!on;
        if (!!conditionsOn[name] === on) return false;
        conditionsOn[name] = on;
        var tools = toolsForNow();
        try {
          send({ type: 'session.update',
                 session: { type: 'realtime', tools: tools } });
        } catch (e) { return false; }
        emit('LIVE_TOOLS_CHANGED', {
          condition: name, on: on,
          tools: tools.map(function (t) { return t.name; }),
        });
        return true;
      },

      /* One utterance did not come out the way it was meant to. Counted here
         rather than only in the relay, so a drive's own tally says how often
         RIO's voice was not quite her voice. */
      noteVoiceFallback: function (detail) {
        counters.voice_fallbacks++;
        emit('LIVE_VOICE_FALLBACK', detail || {});
      },

      /* The wire went away mid-sentence -- the data channel closed, or ICE
         gave up. Called by connect(), which is the only thing that can see it:
         nothing arrives on a dead channel, so this failure is invisible from
         the event stream and used to be indistinguishable from RIO simply
         stopping. Counted, and never resumed: there is nothing left to resume
         into. */
      transportLost: function (reason) {
        if (stopped) return;
        if (speaking || pendingBarge) {
          noteCutoff('transport', {
            response_id: speaking ? speaking.responseId : null,
            reason: reason || 'closed', said: partial });
        }
        clearBarge();
        pendingResume = null;
        emit('LIVE_TRANSPORT_LOST', { reason: reason || 'closed' });
        this.stop();
      },

      /* Ending the session releases the mouth: an item left claimed would
         block every conversational reply for the rest of the drive. */
      stop: function () {
        stopped = true;
        clearOnsetHold();
        stopEchoWatch();
        audio.mute();
        clearBarge();
        pendingResume = null;
        if (dictation) finishDictation('session_stopped');
        if (speaking) endResponse(speaking.responseId);
        // The dialogue socket is one per live session and dies with it. Left
        // open it would hold a seat in a pool that is counted separately from
        // ordinary synthesis, for a drive that is over.
        if (sink) { try { sink.close(); } catch (e) {} sink = null; }
      },

      state: function () {
        return {
          speaking: !!speaking,
          dictating: !!dictation,
          // Which mouth this session is using, and what the sink is doing.
          voice_backend: sink ? 'elevenlabs' : 'openai_realtime',
          speaking_directly: !!(speaking && speaking.direct),
          voice: sink && sink.state ? sink.state() : null,
          generated: generated,
          response_id: speaking ? speaking.responseId : null,
          stopped: stopped,
          last_transcript: lastTranscript,
          // ...and the same words only while they are still THIS turn's.
          // This is what a tool call is given; see transcriptFresh.
          spoken_this_turn: transcriptFresh ? lastTranscript : '',
          counters: counters,
          // The answer to "why does she keep cutting out", as a tally rather
          // than as an impression. Read it from the console after a drive, or
          // let the panel post it -- see /realtime/cutoffs.
          cutoffs: cutoffs,
          pending_barge: !!pendingBarge,
          pending_resume: pendingResume ? pendingResume.cause : null,
          said_so_far: saidSoFar(),
          policy: { barge_sustain_ms: bargeSustainMs,
                    barge_confirm_ms: bargeConfirmMs,
                    // The phone half of the gate. All three are 0 on a desk,
                    // which is the fastest way to answer "is this machine
                    // running the touch numbers or the desktop ones".
                    barge_onset_guard_ms: bargeOnsetGuardMs,
                    barge_echo_margin_db: bargeEchoMarginDb,
                    barge_echo_floor_db: bargeEchoFloorDb,
                    echo_meter: !!levels,
                    max_resumes: maxResumes },
          // What the meter last read, for a panel that wants to show it.
          levels: lastLevels,
          // Which tools the session is carrying right now, and why. `null`
          // rather than `[]` when the session did not hand the schemas over:
          // "this controller was never told" and "this session has no tools"
          // are different answers and an empty list reads as the second.
          tools: baseTools.length
            ? toolsForNow().map(function (t) { return t.name; }) : null,
          tool_conditions: conditionsOn,
        };
      },
    };
  }

  /* ---------------------------------------------------------------------
     The wiring. Everything below needs a browser.
     --------------------------------------------------------------------- */
  /* WHICH COLUMN OF THE INTERRUPTION POLICY THIS MACHINE IS IN.
   *
   * Decided here and not on the server, and that is not a style preference: a
   * User-Agent is a guess, and a guess that puts a driver's phone in the
   * desktop column is the exact bug the touch column exists to fix. The
   * browser knows. A coarse pointer or a real multi-touch digitiser is a
   * device whose speaker and microphone are two inches apart.
   *
   * Wrong in the safe direction on a touchscreen laptop: it would get the
   * phone numbers, which cost a fifth of a second on an interruption nobody
   * would notice, and it does not cost an answer. */
  function isTouchDevice() {
    try {
      var nav = root.navigator || {};
      if ((nav.maxTouchPoints || 0) > 1) return true;
      if (root.matchMedia && root.matchMedia('(pointer: coarse)').matches) return true;
      if (root.matchMedia && root.matchMedia('(hover: none)').matches) return true;
    } catch (e) { /* a browser that cannot answer is a desk */ }
    return false;
  }

  /* THE METER THE ECHO GATE READS.
   *
   * Two numbers in dBFS: what the microphone is hearing, and what this page is
   * rendering. Both are measured here because both are facts about audio the
   * controller has no way to reach -- it gets a function and a promise not to
   * be given nonsense.
   *
   * The output figure is the LOUDER of two things, because RIO speaks through
   * two of them: the live session's own voice, which arrives as a remote
   * WebRTC track, and everything else, which goes through the shared output
   * bus (clips, TTS, the synthesiser). Taking the max is the right reading of
   * "how loud is this page right now" -- whichever mouth is open is the one
   * the microphone is hearing.
   *
   * Returns null for anything it cannot measure, and null means the gate has
   * no level evidence rather than that there is no echo. */
  function makeMeter(ctxIn, micStream, remoteStream) {
    var C = root.AudioContext || root.webkitAudioContext;
    if (!C) return null;
    var ctx = ctxIn;
    if (!ctx) { try { ctx = new C(); } catch (e) { return null; } }
    var buf = null;
    function analyserFor(stream) {
      if (!stream) return null;
      try {
        var src = ctx.createMediaStreamSource(stream);
        var an = ctx.createAnalyser();
        an.fftSize = 1024;
        an.smoothingTimeConstant = 0.2;
        src.connect(an);
        // Deliberately NOT connected onward. An analyser is a tap, and a tap
        // that reaches the speaker is a feedback loop.
        return an;
      } catch (e) { return null; }
    }
    var micAn = analyserFor(micStream);
    var outAn = analyserFor(remoteStream);
    if (!micAn) return null;
    function db(an) {
      if (!an) return -100;
      var n = an.fftSize;
      if (!buf || buf.length !== n) buf = new Float32Array(n);
      try {
        if (!an.getFloatTimeDomainData) return -100;
        an.getFloatTimeDomainData(buf);
      } catch (e) { return -100; }
      var sum = 0;
      for (var i = 0; i < n; i++) sum += buf[i] * buf[i];
      var rms = Math.sqrt(sum / n);
      if (!(rms > 0)) return -100;
      var v = 20 * Math.log10(rms);
      return v < -100 ? -100 : v;
    }
    return function () {
      var busDb = -100;
      try {
        var o = root.RIO && root.RIO.output;
        if (o && o.level) busDb = o.level();
      } catch (e) { busDb = -100; }
      var trackDb = db(outAn);
      return { mic: db(micAn), out: Math.max(busDb, trackDb) };
    };
  }

  function connect(opts) {
    opts = opts || {};
    // The shared watch, subscribed to rather than started: rio_nav.js reads the
    // same one, and two Geolocation watches on one page is two batteries'
    // worth of GPS for one car.
    if (root.RIO && root.RIO.headway && root.RIO.headway.onPosition) {
      try { root.RIO.headway.onPosition(noteFix); } catch (e) {}
      if (root.RIO.headway.startWatch) {
        try { root.RIO.headway.startWatch(); } catch (e) {}
      }
    }
    var arbiter = opts.arbiter || (root.RIO && root.RIO.speech);
    var url = opts.url || function (p) { return p; };
    var element = opts.element;
    var session = null;

    return fetch(url('/realtime/session'), { method: 'POST' })
      .then(function (r) { return r.json(); })
      .then(function (j) {
        if (!j || j.error || !j.client_secret) {
          throw new Error((j && j.error) || 'no session');
        }
        session = j;
        return navigator.mediaDevices.getUserMedia({ audio: MIC_CONSTRAINTS });
      })
      .then(function (mic) {
        var touch = isTouchDevice();
        var barge = (touch ? session.barge_touch : session.barge_desktop) || {};
        var meter = null;
        /* Built once, from the microphone that was just opened and the remote
           track when it arrives. Failing is allowed and is not reported as an
           error: a session with no meter is a session with the desk gate,
           which is what every session had until now. */
        function armMeter() {
          if (meter) return;
          if (!barge.echo_margin_db) return;      // desk: nothing to measure
          var shared = null;
          try {
            var o = root.RIO && root.RIO.output;
            if (o && o.context) shared = o.context();
          } catch (e) { shared = null; }
          meter = makeMeter(shared, mic, remoteStream);
          if (opts.onEvent) {
            try {
              opts.onEvent({ type: 'LIVE_ECHO_METER', ok: !!meter,
                             device: touch ? 'touch' : 'desktop',
                             margin_db: barge.echo_margin_db,
                             onset_guard_ms: barge.onset_guard_ms,
                             sustain_ms: barge.sustain_ms });
            } catch (e) {}
          }
        }
        var pc = new RTCPeerConnection();
        var channel = pc.createDataChannel('oai-events');
        mic.getTracks().forEach(function (t) { pc.addTrack(t, mic); });

        /* THE ONE PATH iOS COULD ALREADY CANCEL. A remote MediaStream on an
           element is rendered by the same WebRTC stack that owns the capture,
           so it IS the reference the canceller subtracts. Everything else RIO
           says goes through RIO.output for the same reason. */
        var remoteStream = null;
        pc.ontrack = function (e) {
          remoteStream = e.streams[0];
          element.srcObject = remoteStream;
          var p = element.play();
          if (p && p.catch) p.catch(function () {});
          // The meter cannot be built until there is something to measure.
          armMeter();
        };

        /* WHICH MOUTH, AND HOW IT CAN CHANGE MID-DRIVE
         *
         * The element is where the session's own voice comes out; the sink is
         * where ElevenLabs comes out. The controller talks to neither
         * directly, because it has to be possible to swap them at the moment
         * ElevenLabs stops answering -- and the controller must not have to
         * know that happened in order to keep muting the right thing.
         *
         * The element stays muted for the whole drive under the ElevenLabs
         * backend. It is carrying no audio (a text-mode session produces
         * none), and a muted element is the honest expression of that rather
         * than a track everybody assumes is silent. */
        var elementMouth = {
          mute: function () { element.muted = true; },
          unmute: function () { element.muted = false; },
        };
        var sink = null;
        if (session.voice_backend === 'elevenlabs' && root.RIO
            && root.RIO.voiceEleven) {
          var proto = (root.location && root.location.protocol === 'https:')
            ? 'wss://' : 'ws://';
          var host = (root.location && root.location.host) || '';
          sink = root.RIO.voiceEleven.createSink({
            wsUrl: proto + host + url('/voice/dialogue'),
            sampleRate: session.voice_sample_rate,
            onEvent: opts.onEvent,
          });
          element.muted = true;
        }
        var mouth = { at: sink || elementMouth };
        var audioFacade = {
          mute: function () { mouth.at.mute(); },
          unmute: function () { mouth.at.unmute(); },
        };

        /* The driver's last words, read back off the controller.
         *
         * Safe to reference before `controller` is assigned: this is only ever
         * CALLED from a tool call, which cannot happen until the session is up
         * and the controller has been built. */
        function controllerTranscript() {
          try { return controller.state().spoken_this_turn || ''; }
          catch (e) { return ''; }
        }

        var controller = createController({
          arbiter: arbiter,
          // Dictation policy comes from the server with the session, so the
          // browser holds no second copy of the verbatim instruction to drift
          // from the one the tests check.
          verbatimInstruction: session.verbatim_instruction,
          speakTimeoutMs: session.speak_timeout_ms,
          directSpeechTimeoutMs: session.direct_speech_timeout_ms,
          lookAnswerMaxTokens: session.look_answer_max_tokens,
          // Interruption policy, decided in config.py and carried here with
          // the session exactly as the dictation policy is. The browser holds
          // no numbers of its own to drift from the ones the tests check.
          resumeInstruction: session.resume_instruction,
          bargeConfirmMs: session.barge_confirm_ms,
          /* THE PHONE COLUMN OR THE DESK COLUMN. Both travel with the session
             (config.py decides them, realtime.mint_client_secret sends them)
             and the machine picks its own — see isTouchDevice. The fallback to
             the flat barge_sustain_ms is what an older server sends, and it is
             the desktop behaviour, unchanged. */
          bargeSustainMs: barge.sustain_ms || session.barge_sustain_ms,
          bargeOnsetGuardMs: barge.onset_guard_ms || 0,
          bargeEchoMarginDb: barge.echo_margin_db || 0,
          bargeEchoFloorDb: session.barge_echo_floor_db,
          /* The meter is installed later, when there is a remote track to
             measure; until then the level test has no evidence and the gate
             behaves as it does on a desk. */
          levels: function () { return meter ? meter() : null; },
          maxResumes: session.max_resumes,
          /* How long her own voice may still be in the room after she stops,
             and what counts as hearing herself. config.py decides both; see
             REALTIME_ECHO_* and the iPhone test that produced them. */
          echoTailMs: session.echo_tail_ms,
          echoTextWindowS: session.echo_text_window_s,
          echoTextOverlap: session.echo_text_overlap,
          echoTextMinWords: session.echo_text_min_words,
          /* NEWEST WINS. The commands that preempt, how long two fragments may
             be apart and still be one question, and what a continuation looks
             like — all decided in config.py and carried here with the session,
             exactly as the barge and dictation policies are. */
          turnPolicy: session.turn_policy,
          send: function (obj) {
            if (channel.readyState === 'open') channel.send(JSON.stringify(obj));
          },
          tool: function (name, args, controller) {
            // Answered in the page when the page is the source of truth;
            // everything else goes to the server, which holds the camera, the
            // vehicle context and the reasoning model.
            if (LOCAL_TOOLS[name]) {
              try { return Promise.resolve(LOCAL_TOOLS[name](args)); }
              catch (e) { return Promise.resolve({ ok: false, note: 'panel error' }); }
            }
            return fetch(url('/realtime/tool'), {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              /* THE HANDLE THAT STOPS IT. One controller per tool call, held by
                 the turn that asked for it; a new driver utterance aborts every
                 controller belonging to an older turn. Aborting the fetch also
                 drops the TCP connection, which is how the server finds out --
                 /realtime/tool watches for it and stops waiting on work nobody
                 is going to hear.

                 On the first real drive these calls ran 40.1, 12.9, 48.5 and
                 27.1 seconds. Nothing could stop one. */
              signal: controller ? controller.signal : undefined,
              // `where` is the car's own fix. Only find_places reads it, but it
              // is attached to every call rather than to one, so a tool that
              // needs it later does not have to re-plumb this.
              //
              // `spoken` is the DRIVER'S OWN LAST WORDS, and it is here for a
              // measured reason. The model paraphrases what it was asked --
              // "what's around us right now" reaches the tool as "describe the
              // current scene" -- and the camera's fast path is a judgement
              // about the question, so it has to be able to see the question
              // rather than the relay. This page is where Whisper's transcript
              // lands, so this is the only place that can send it.
              body: JSON.stringify({ name: name, arguments: args,
                                     where: currentFix(),
                                     spoken: controllerTranscript() }),
            }).then(function (r) { return r.json(); });
          },
          // Muting rather than pausing: a track is live and a paused element
          // resumes into stale audio, and the sink's fade is undoable for the
          // same reason -- the sustain gate has to be able to change its mind.
          audio: audioFacade,
          voice: sink,
          liveVoice: session.live_voice || session.voice,
          // The tool list, and the part of it that waits for a precondition.
          // Both come from the server; see mint_client_secret.
          toolSchemas: session.tool_schemas,
          conditionalTools: session.conditional_tools,
          onEvent: opts.onEvent,
        });

        channel.onmessage = function (e) {
          var ev;
          try { ev = JSON.parse(e.data); } catch (err) { return; }
          controller.handle(ev);
        };

        /* Anything that has to be SAID to the session has to wait for a
           channel to say it on. The one that matters is the tier-2 fallback:
           the relay can refuse before the data channel has finished opening,
           and a session.update sent into a channel that is not open yet is not
           a fallback, it is a drive with no voice at all. */
        var channelOpen = false;
        var whenOpen = [];
        function onChannelOpen(fn) {
          if (channelOpen) { try { fn(); } catch (e) {} return; }
          whenOpen.push(fn);
        }
        channel.onopen = function () {
          channelOpen = true;
          var pending = whenOpen.splice(0, whenOpen.length);
          for (var i = 0; i < pending.length; i++) {
            try { pending[i](); } catch (e) {}
          }
        };

        /* WHEN THE PRECONDITION BECOMES TRUE, AND WHEN IT STOPS.
         *
         * On the bus rather than by calling into rio_nav.js, because the
         * condition is a fact about the drive and not a favour between two
         * files: anything that attaches a route says so there, including the
         * automatic reroute, and this asks one question of each event rather
         * than knowing who sent it.
         *
         * Sent through the same wait-for-the-channel gate the tier-2 fallback
         * uses. A route set before the data channel finishes opening is
         * ordinary — the driver can ask for one in the first second — and a
         * session.update into a channel that is not open yet is silently
         * nothing. */
        controller.watchToolConditions(root.RIO && root.RIO.bus, onChannelOpen);

        /* The sink's own bad news. A per-utterance fallback is counted and the
           drive carries on; a tier-2 fallback changes what the session is asked
           to produce, and the mouth moves back to the element in the same
           breath so the very next response is audible. */
        if (sink) {
          sink.onEvent(function (ev) {
            if (!ev) return;
            if (ev.type === 'VOICE_FALLBACK') {
              if (ev.tier === 'live_voice') {
                mouth.at = elementMouth;
                onChannelOpen(function () {
                  controller.useLiveVoice(ev.cause || 'elevenlabs_unavailable');
                });
              } else {
                controller.noteVoiceFallback(ev);
              }
            } else if (ev.type === 'VOICE_TRANSPORT_LOST') {
              mouth.at = elementMouth;
              onChannelOpen(function () { controller.useLiveVoice('voice_relay_lost'); });
            }
          });
        }

        /* A session that dies mid-answer looks exactly like an answer that
           stopped, and the driver reports the same symptom for both. These are
           the only places the difference is visible -- nothing arrives on a
           dead channel to tell the controller about it. */
        channel.onclose = function () { controller.transportLost('datachannel_closed'); };
        channel.onerror = function () { controller.transportLost('datachannel_error'); };
        pc.onconnectionstatechange = function () {
          if (pc.connectionState === 'failed' || pc.connectionState === 'closed'
              || pc.connectionState === 'disconnected') {
            controller.transportLost('peer_' + pc.connectionState);
          }
        };

        /* The relay and the model are opened together rather than in a line.
           Both are a round trip the driver is waiting through, and they have
           nothing to say to each other until the first word is spoken. */
        var voiceReady = Promise.resolve(null);
        if (sink) {
          voiceReady = sink.open().catch(function () {
            mouth.at = elementMouth;
            onChannelOpen(function () {
              controller.useLiveVoice('voice_relay_unavailable');
            });
            return null;
          });
        }

        return voiceReady
          .then(function () { return pc.createOffer(); })
          .then(function (offer) { return pc.setLocalDescription(offer).then(function () { return offer; }); })
          .then(function (offer) {
            return fetch(CALLS_URL + '?model=' + encodeURIComponent(session.model), {
              method: 'POST',
              headers: {
                'Authorization': 'Bearer ' + session.client_secret,
                'Content-Type': 'application/sdp',
              },
              body: offer.sdp,
            });
          })
          .then(function (r) {
            if (!r.ok) return r.text().then(function (t) {
              throw new Error('realtime call ' + r.status + ': ' + t.slice(0, 160));
            });
            return r.text();
          })
          .then(function (answer) {
            return pc.setRemoteDescription({ type: 'answer', sdp: answer });
          })
          .then(function () {
            var handle = {
              session: session,
              controller: controller,
              /* Dictate a deterministic line in RIO's voice. Warnings,
                 turns and health announcements come through here; each of
                 them already holds the mouth at its own priority. */
              speak: function (text, o) { return controller.speak(text, o); },
              speechEnabled: function (channel) {
                if (session.speech_enabled === false) return false;
                var chans = session.speech_channels || {};
                return chans[channel] !== false;
              },
              /* How long THIS line may take to start speaking before it is
                 synthesised instead. Per channel, and for navigation per call
                 type, because a backup call at the junction and one issued
                 seconds out have different deadlines. The table is decided in
                 config.py and travels with the session; this only reads it. */
              speakTimeout: function (channel, callType) {
                var table = session.speak_timeout_ms_by_channel || {};
                var byCall = table[channel];
                if (!byCall) return session.speak_timeout_ms;
                return byCall[callType] || byCall._default
                       || session.speak_timeout_ms;
              },
              /* Which voice this drive is actually using, for the panel and
                 for the tests. Read from the controller rather than from the
                 session payload: the payload says what was INTENDED, and after
                 a tier-2 fallback those are two different answers. */
              voiceBackend: function () {
                return controller.state().voice_backend;
              },
              stop: function () {
                controller.stop();
                if (sink) { try { sink.close(); } catch (e) {} }
                try { channel.close(); } catch (e) {}
                try { pc.close(); } catch (e) {}
                mic.getTracks().forEach(function (t) { t.stop(); });
                element.srcObject = null;
                if (active === handle) active = null;
              },
            };
            active = handle;
            return handle;
          });
      });
  }

  /* The one live session, if there is one. Deterministic speech asks for it
     by name rather than being handed it: a warning fires from a code path that
     has never heard of the conversation panel and must not have to. */
  var active = null;

  root.RIO = root.RIO || {};
  root.RIO.realtime = {
    createController: createController,
    connect: connect,
    navStatus: navStatus,
    navDirections: navDirections,
    noteFix: noteFix,
    currentFix: currentFix,
    startNavigation: startNavigation,
    localTools: LOCAL_TOOLS,
    micConstraints: MIC_CONSTRAINTS,
    active: function () { return active; },
    /* Tests and the panel: pretend a session is open, or that none is. */
    _setActive: function (h) { active = h; },
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { createController: createController, navStatus: navStatus,
                       navDirections: navDirections,
                       noteFix: noteFix, currentFix: currentFix,
                       startNavigation: startNavigation,
                       localTools: LOCAL_TOOLS,
                       micConstraints: MIC_CONSTRAINTS };
  }
})(typeof window !== 'undefined' ? window : globalThis);
