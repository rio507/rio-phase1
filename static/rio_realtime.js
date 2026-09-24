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

  /* HOW LONG EACH STEP OF A CONNECT MAY TAKE, AND WHY THERE ARE NUMBERS HERE
   * AT ALL.
   *
   * On 2026-09-10 the talk button sat on "Connecting…" for a whole drive and
   * never moved. The session log for that drive has no `session_started` and
   * no `session_failed` -- so the mint never completed -- and the button had
   * no way to say so, because NOT ONE STEP OF THIS PATH HAD A DEADLINE.
   *
   * `fetch` has no default timeout worth the name (Chrome will wait minutes),
   * getUserMedia waits on a human, and the SDP exchange with api.openai.com
   * waits on a network this pod does not control. Any one of them hanging
   * produced an indefinite "Connecting…" with no reason, no failure, and no
   * fallback to hold-to-talk -- and nothing in any log to say which.
   *
   * So every step is bounded and every step is NAMED. A connect that fails now
   * fails with "mint: timed out after 10s", which is a sentence a driver can
   * read and an engineer can act on.
   *
   * The microphone is the exception and deliberately so: it is waiting on a
   * person to answer a permission prompt, and a person is allowed to take
   * longer than a network. It gets a minute and its own progress state, so
   * the screen says "allow the microphone" rather than "stand by". */
  var CONNECT_BUDGET_MS = {
    mint: 10000,        // POST /realtime/session — our own server, loopback
    mic: 60000,         // a human answering a permission prompt
    voice: 8000,        // the ElevenLabs relay, where that backend is in use
    offer: 8000,        // createOffer + setLocalDescription, all local
    negotiate: 20000,   // the SDP exchange with api.openai.com
    answer: 8000,       // setRemoteDescription, local again
  };

  /* Race a step against its budget, and name it in the failure.
   *
   * There is no way to ABORT most of these -- getUserMedia has no signal in
   * older Safari, and a peer connection's negotiation is not cancellable --
   * so the loser is abandoned rather than stopped, exactly as the connect
   * canceller in index.html abandons a connect it no longer wants. What
   * matters is that the CALLER stops waiting and the driver is told. */
  function step(name, promise, onProgress) {
    var ms = CONNECT_BUDGET_MS[name] || 10000;
    if (onProgress) { try { onProgress(name); } catch (e) {} }
    return new Promise(function (resolve, reject) {
      var done = false;
      var timer = setTimeout(function () {
        if (done) return;
        done = true;
        reject(new Error(name + ': timed out after ' + Math.round(ms / 1000) + 's'));
      }, ms);
      Promise.resolve(promise).then(function (v) {
        if (done) return;
        done = true; clearTimeout(timer); resolve(v);
      }, function (e) {
        if (done) return;
        done = true; clearTimeout(timer);
        // Every failure carries its step, so "connecting" can never again be
        // the last thing anybody knows.
        var msg = (e && e.message) ? e.message : String(e);
        reject(new Error(msg.indexOf(name + ':') === 0 ? msg : name + ': ' + msg));
      });
    });
  }

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
      rules: 'Answer the question. Do NOT call this maneuver now — you ' +
             'call it yourself when the car gets to it, and that is not this ' +
             'moment. Speak as the one who is driving them there: "I\'ve got ' +
             'the route", "I\'ll take you onto Colorado" — never in the ' +
             'third person about a car or a navigation.',
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
             'these turns as instructions now — you call each one yourself ' +
             'when the car gets to it. Say so in the first person if it ' +
             'comes up: "I\'ll call each turn as we get there." Never that a ' +
             'car or a navigation does it.',
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
        /* ONE TURN, AND IT IS NOT A LIST.
         *
         * This handed over the first THREE maneuvers as `first_steps`, with a
         * rule underneath saying not to read them out. On 2026-09-24 (session
         * 1fd4de92) she read them out: the route locked and she recited the
         * directions for twenty-one seconds.
         *
         * A rule against using what you were given loses to the thing you were
         * given. Three numbered steps with instructions, road names and
         * landmarks IS a list of directions, and an instruction not to read a
         * list of directions is an instruction to ignore most of the payload.
         * So the payload changes: the confirmation needs where they are going
         * and what the first turn is, and that is all it now contains.
         *
         * A SINGLE OBJECT RATHER THAN A ONE-ELEMENT ARRAY, deliberately. An
         * array of one is still an array and reads as the start of an
         * enumeration; `first_turn` has nothing after it to enumerate.
         *
         * Every subsequent turn goes out on the route engine's own cadence --
         * far, far_mid, near, junction, arrival -- in her voice, at the moment
         * it matters. That is rio_navplan's job and it has never needed the
         * model's help with it. */
        var mans = route.maneuvers || [];
        var m0 = null;
        for (var mi = 0; mi < mans.length; mi++) {
          if (mans[mi].type !== 'ARRIVE' && mans[mi].type !== 'DEPART') {
            m0 = mans[mi]; break;
          }
        }
        if (!m0) m0 = mans[0] || null;
        var firstTurn = null;
        if (m0) {
          firstTurn = {
            instruction: m0.instruction, road_name: m0.road_name,
            maneuver_type: m0.type, direction: m0.direction,
            // From the START of the route, not from the car: this is the route
            // as loaded, before anyone has driven any of it.
            distance_from_start_m: Math.round(m0.route_distance_position || 0),
          };
          var lm0 = landmarkOf(m0);
          if (lm0) firstTurn.landmark = lm0;
        }
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
          first_turn: firstTurn,
          rules: 'The route is live and you are taking them there now. ' +
                 'Confirm it in ONE SENTENCE, in your own words and in the ' +
                 'FIRST PERSON — "I\'ve got it, about eighteen minutes" — ' +
                 'using this destination name exactly as spelled here. You ' +
                 'may name the first turn and nothing after it. There is only ' +
                 'one turn here because one is all a confirmation carries; ' +
                 'the rest are not withheld, they go out in your voice as the ' +
                 'car reaches them. Do NOT tell the driver to set it ' +
                 'themselves; it is set. If they ASK for the directions, call ' +
                 'nav_directions and read them — that is answering. What you ' +
                 'never do is call a turn early; if the driver asks who is ' +
                 'calling them the answer is you: "I\'ll call each turn as we ' +
                 'get there."',
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
      rules: 'The route is off and you will not be calling any more turns. ' +
             'Confirm it once, briefly and in the first person — "Okay, I\'ll ' +
             'stop guiding you." Do not ask whether they are sure, do not ' +
             'offer to start it again, and do not read out where they were ' +
             'going.',
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
          rules: 'You are not guiding them anywhere, so there is no route ' +
                 'to ' +
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
        rules: 'The new route is live and you are taking them that way ' +
               'now. One line, first person: what changed and roughly how ' +
               'long it is now — "Got you a different way, about twenty ' +
               'minutes." If avoid_unsupported has anything in it, say ' +
               'plainly that you cannot do that one and that you have ' +
               'rerouted anyway; never say you avoided something that is not ' +
               'in avoid_applied. Do not read the turns out and do not call ' +
               'them early — you still call each one yourself when the car ' +
               'gets to it.',
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
    /* WHICH VENDOR IS ON THE OTHER END, AND THE ONE THING THIS FILE IS ALLOWED
     * TO KNOW ABOUT IT: nothing.
     *
     * The provider (static/rio_provider.js) does two jobs at this boundary. It
     * maps the vendor's event names onto the ones the switch below speaks, so
     * that no xAI event name — or any future vendor's — ever appears in this
     * file. And it answers capability questions as FACTS, so that the code
     * downstream asks "does anything know when the audio ended?" instead of
     * "is this WebRTC?". The second half is the one that matters: a vendor
     * check spreads, a fact does not.
     *
     * OPTIONAL, and that is deliberate. ~4,000 lines of
     * tools/realtime_selftest.js build a controller with no provider and feed
     * it canonical events directly. Without one, `normalise` is identity and
     * every cfg flag behaves exactly as it did — which is what makes this a
     * refactor rather than a rewrite, and is asserted in
     * tools/provider_adapter_selftest.js. */
    var provider = cfg.provider || null;
    var normalise = (provider && provider.normalise)
      ? provider.normalise
      : function (ev) { return ev; };
    var arbiter = cfg.arbiter;
    var send = cfg.send || function () {};
    var runTool = cfg.tool || function () { return Promise.resolve({ ok: false }); };
    var rawAudio = cfg.audio || { mute: function () {}, unmute: function () {} };
    /* THE MOUTH, WITH A LEDGER.
     *
     * Every mute on this page is a moment the driver stops hearing her, and
     * until now none of them was written down: `audio.mute()` was called from
     * seven places for seven reasons and the drive log could say only that an
     * answer had been cut off -- and only for the two of the seven that
     * classified themselves. The other five muted the speaker and moved on.
     *
     * So the facade keeps the ledger. A mute records its reason and its start;
     * an unmute closes the span and charges it to whichever utterance was
     * STREAMING at the time (see `streamingId`) -- which is the only honest
     * assignment, because under speech-to-speech the element carries one
     * stream and a mute made "for" one response lands on whatever is coming
     * out of the speaker, usually the tail of the answer before it. That is
     * the fault this ledger was built to make visible. */
    var muted = false;
    var muteSince = 0;
    var muteReason = null;
    var audio = {
      mute: function (why) {
        try { rawAudio.mute(); } catch (e) {}
        if (muted) return;
        muted = true;
        muteSince = Date.now();
        muteReason = why || 'unspecified';
        if (streamingId && utter[streamingId]) {
          var u = utter[streamingId];
          u.mute_reasons[muteReason] = (u.mute_reasons[muteReason] || 0) + 1;
        }
      },
      unmute: function (why) {
        try { rawAudio.unmute(); } catch (e) {}
        if (!muted) return;
        muted = false;
        chargeMute(Date.now());
        muteReason = null;
      },
      isMuted: function () { return muted; },
    };
    /* Close the open mute span against the streaming utterance, if any. */
    function chargeMute(at) {
      if (!muteSince) return;
      var span = Math.max(0, at - muteSince);
      muteSince = 0;
      if (streamingId && utter[streamingId]) utter[streamingId].muted_ms += span;
    }

    /* ---- THE UTTERANCE LEDGER: how much was written, how much was heard ----
     *
     * One record per response, kept from `response.created` to the end of
     * its AUDIO -- not the end of its generation, which under speech-to-
     * speech is seconds earlier and was the only end this file used to know.
     * Reported once, as LIVE_UTTERANCE_END, with:
     *
     *   generated_chars  the transcript the model wrote
     *   audio_ms         from output_audio_buffer.started to .stopped/.cleared
     *   muted_ms         how much of that the element was muted for, and why
     *   heard_ms/frac    audio_ms less muted_ms; the driver's share
     *   ended_by         completed | cancelled:<who> | cleared | muted
     *
     * "She stopped mid-sentence" is heard_frac < 1 or ended_by != completed,
     * and the drive of 2026-09-17 could not produce that sentence for a single
     * response: there was no record of what was written, none of when the
     * audio started or stopped, and none of the mutes. */
    var utter = {};
    var utterOrder = [];
    var streamingId = null;
    /* Responses cancelled on sight (noise replies, orphans). They are NOT
       muted when cancelled -- the mute would land on the tail of the answer
       before them -- but if their audio does start before the cancel lands,
       that start is the moment to mute, and only that. */
    var silencedIds = {};
    function utterFor(rid, kind) {
      if (!rid) return null;
      if (!utter[rid]) {
        utter[rid] = { response_id: rid, turn_kind: kind || 'conversation',
                       created_at: Date.now(), audio_started_at: 0,
                       audio_ended_at: 0, generated_chars: 0, transcript_done: false,
                       status: null, status_reason: null, cancel_reason: null,
                       muted_ms: 0, mute_reasons: {}, muted_at_end: false,
                       ended_by: null, reported: false, done: false };
        utterOrder.push(rid);
        while (utterOrder.length > 24) delete utter[utterOrder.shift()];
      } else if (kind && utter[rid].turn_kind === 'conversation') {
        utter[rid].turn_kind = kind;
      }
      return utter[rid];
    }
    /* Our own cancels name themselves; the server's response.done then
       confirms. A cancel the server reports that nobody here asked for is
       recorded as the server's. */
    function noteCancel(rid, why) {
      var u = rid ? utter[rid] : null;
      if (u && !u.cancel_reason) u.cancel_reason = why || 'unspecified';
    }
    function utteranceEnded(rid, how) {
      var u = utter[rid];
      if (!u || u.reported) return;
      /* Not until BOTH halves are in: the server's verdict on the response
         and the end of its audio. Whichever comes second reports. A response
         that never started audio has only the first half to wait for. */
      var audioOver = !u.audio_started_at || u.audio_ended_at > 0;
      if (!(u.done && audioOver)) return;
      u.reported = true;
      var at = u.audio_ended_at || Date.now();
      var audioMs = u.audio_started_at ? Math.max(0, at - u.audio_started_at) : 0;
      var mutedMs = Math.min(audioMs, u.muted_ms);
      var heardMs = Math.max(0, audioMs - mutedMs);
      var ended = u.ended_by || how || 'completed';
      if (ended === 'completed' && u.muted_at_end) ended = 'muted';
      var frac = audioMs ? Math.round((heardMs / audioMs) * 1000) / 1000 : (u.audio_started_at ? 1 : 0);
      var reasons = [];
      for (var k in u.mute_reasons) reasons.push(k + 'x' + u.mute_reasons[k]);
      counters.utterances++;
      var early = ended !== 'completed' || frac < 0.98;
      if (early) counters.utterances_early++;
      emit('LIVE_UTTERANCE_END', {
        response_id: rid, turn_kind: u.turn_kind,
        generated_chars: u.generated_chars, status: u.status,
        status_reason: u.status_reason, ended_by: ended,
        cancel_reason: u.cancel_reason,
        audio_started: !!u.audio_started_at, audio_ms: Math.round(audioMs),
        muted_ms: Math.round(mutedMs), heard_ms: Math.round(heardMs),
        heard_frac: frac, muted_at_end: !!u.muted_at_end,
        mute_reasons: reasons.join(','), early: early,
        // The driver's share of the words, by time. An ESTIMATE: audio is
        // not evenly worded, and the log says so by the name.
        heard_chars_est: Math.round(u.generated_chars * frac),
      });
    }
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
                     /* A deterministic line the mouth could not take, and one
                        the caller took back. `dictation_refused` is contention
                        -- a second line arriving while the first still holds
                        the single dictation slot -- and on the drive of
                        2026-09-17 exactly one of these was the junction call
                        that never got said. `dictation_cancelled` is the other
                        half of the same fault, now that superseding a line
                        actually releases the mouth it was holding. */
                     dictation_refused: 0, dictation_cancelled: 0,
                     /* Suppressions a real transcript arrived behind: the echo
                        gate's FALSE NEGATIVES, which nothing counted. Read
                        against false_barge_in -- the same threshold being
                        wrong in the other direction -- this is the pair that
                        says whether the phone column is set right. */
                     barges_missed: 0,
                     /* Every response's audio, accounted for: how many, and
                        how many the driver did not hear the whole of. */
                     utterances: 0, utterances_early: 0,
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
                     /* HOW OFTEN THE MARGIN WAS MEASURED AT ALL — the number
                        whose absence made every threshold a guess. See
                        startCensus(); it runs on a desk and in a car, and it
                        decides nothing. */
                     echo_census_responses: 0, echo_census_samples: 0,
                     /* Dictations abandoned on their budget whose response
                        turned up anyway and was killed before it could read
                        the line a second voice was already reading. The number
                        that must be equal to `dictation_failures` from the
                        timeout branch: every disowned line accounted for. */
                     orphans_silenced: 0,
                     orphan_claims_expired: 0,
                     /* ROAD NOISE, and the four numbers that say what it cost.
                        Session 738fbb82 had twenty of these transcripts and
                        four of them inside ten milliseconds; each committed
                        utterance had a response created for it server-side,
                        so each was an answer to nothing.

                        fragments_coalesced   transcripts buffered rather than
                                              answered one at a time
                        fragments_recovered   ...that turned out to add up to a
                                              real question after all
                        noise_replies         "Didn't catch that." — at most
                                              one per coalesce window, and one
                                              per cooldown
                        noise_replies_suppressed  the ones that would have been
                                              a queue of apologies
                        noise_responses_silenced  server-created answers to
                                              fragments, cancelled by id */
                     fragments_coalesced: 0, fragments_recovered: 0,
                     noise_replies: 0, noise_replies_suppressed: 0,
                     noise_responses_silenced: 0,
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
                     /* Turns the client had to end itself because the server's
                        detector had not, after the microphone had been quiet
                        for REALTIME_TURN_BACKSTOP_MS. Zero is the number to
                        want: it means semantic_vad never needed catching. */
                     turns_backstopped: 0,
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
    /* The instant the detector said the driver had finished. Paired with the
       first audio of the answer in `noteFirstAudio`. */
    var turnEndAt = 0;
    /* Responses already reported as having made a sound. One event per
       response, not one per delta. */
    var firstAudioSeen = {};
    var firstAudioIds = [];
    // Set by tryResume, claimed by the response it asked for.
    var resumeExpected = false;
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
    /* WHICH UTTERANCE THE ANSWER IN FLIGHT IS FOR, and the bug it closes.
     *
     * The input transcription is ASYNCHRONOUS and races the model. On a turn
     * whose answer is a tool call, the model can call the tool before the
     * transcriber has finished writing down the question that caused it --
     * measured at 86 ms apart on a live session. The late transcript then
     * arrived at transcriptArrived() looking exactly like a brand new
     * question, superseded the turn it belonged to, aborted the look() that
     * was running for it, discarded the result and left the driver in
     * silence. The question superseded itself.
     *
     * It reproduced on every visual turn -- 3 of 3 runs -- and never on a
     * fast tool (25 ms: the answer is already out before the transcript
     * lands) or on deep_dive (a holding line is spoken first, so the barge
     * gate protects it). A long tool call with nothing said over it was the
     * exact shape that lost.
     *
     * The session gives the identity needed to tell the two apart for free:
     * speech_started, speech_stopped, committed and the transcription of one
     * utterance all carry the SAME item_id, and a genuinely new question is a
     * different one. So the rule is not a timer or a text comparison -- it is
     * identity. A transcript for the utterance the in-flight response is
     * already answering is not news. */
    var committedItemId = null;   // the input item the server last closed
    var answeringItemId = null;   // ...and the one the live response is for

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
      /* ROAD NOISE, AND THE ONE ANSWER IT GETS. See the block above
         noteFragment() for what these are and why each one is here. */
      noise_coalesce_ms: 2000,
      noise_max_words: 4,
      noise_reply_cooldown_ms: 12000,
      noise_reply: "Didn't catch that.",
      noise_tokens: ['uh', 'um', 'er', 'erm', 'mm', 'mmm', 'hmm', 'huh', 'ah',
                     'oh', 'eh', 'hey', 'hello', 'hi', 'yeah', 'yep', 'yes',
                     'no', 'nope', 'ok', 'okay', 'right', 'sure', 'thanks',
                     'thank', 'you', 'got', 'it', 'wow', 'well', 'so', 'like',
                     'the', 'a', 'and', 'for', 'your', 'help', 'good', 'nice',
                     'cool', 'alright', 'sorry', 'please', 'there'],
      /* The words above that are a WHOLE TURN when she is not the one saying
         them. See config.REALTIME_SOCIAL_TOKENS and the desk session of
         2026-09-20, where "hello" into a working microphone was answered
         "Didn't catch that." twice. */
      social_tokens: ['hey', 'hello', 'hi', 'yeah', 'yep', 'yes', 'no',
                      'nope', 'ok', 'okay', 'right', 'sure', 'thanks',
                      'thank', 'wow', 'good', 'nice', 'cool', 'alright',
                      'sorry', 'please'],
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

    /* --- ROAD NOISE, AND THE FIVE ANSWERS IT USED TO GET --------------------
     *
     * WHAT THE DRIVE OF 2026-09-09 DID. Session 738fbb82, t=180.97 to 180.98
     * -- ten milliseconds:
     *
     *     "Hello."               said_chars 0
     *     "Hello."               said_chars 4
     *     "Thanks for your help." said_chars 0
     *     "Got it."              said_chars 85
     *
     * Four transcripts, none of them anything the driver said: a phone on a
     * mount at 11 m/s with the windows down, and a transcriber doing its job
     * on a second of road roar. At t=271.77, two more ("Hey.", "Hey."). Twenty
     * over the drive.
     *
     * The barge gate refused every one of them as a phantom, which is what the
     * `turn_phantom` events in that log are -- so not one of them superseded,
     * and the counter was right. What the gate does NOT touch is the response
     * the SERVER already created for each committed utterance
     * (turn_detection.create_response is on, realtime.py:1030). Four
     * commits, four responses, four answers to nothing, stacked behind
     * whatever she was already saying.
     *
     * THE POLICY, and it is three rules:
     *
     *   COALESCE      Fragments inside noise_coalesce_ms are ONE utterance,
     *                 whatever the words are. isContinuation() coalesces on
     *                 LEXICAL cues -- an opener, a dangling conjunction -- and
     *                 that is right for a driver taking a breath mid-question
     *                 and useless for road noise, which produces four
     *                 grammatical fragments with nothing joining them.
     *
     *   ANSWER ONCE   If the coalesced text is still unintelligible, one
     *                 "Didn't catch that." and the rest go silently. Never a
     *                 queue of them: a driver who said nothing and hears the
     *                 car apologise five times has a car that is broken in a
     *                 new way.
     *
     *   AND ONLY ONCE Even that one is on a cooldown, because a rough road is
     *                 a rough road for minutes at a time.
     *
     * NEWEST-WINS IS UNTOUCHED. A real question is not a fragment, does not
     * enter this buffer, and supersedes exactly as it did.
     */
    var noise = null;             // { text, first, last, n, timer }
    var lastNoiseReplyAt = 0;
    var silenceResponses = 0;     // responses to cancel on sight, by id

    /* Is this transcript words, or is it the road?
     *
     * Deliberately NOT a confidence threshold: the transcriber does not send
     * one on this path, and a length test alone would swallow "What's that?"
     * -- three words, twelve characters, and the most common real question in
     * the car. So the test is short AND made only of tokens that carry no
     * request in them. "Turn left" is two words and survives; "Yeah, ok" is
     * two words and does not.
     */
    /* Is there a REQUEST anywhere in these words? Length is not part of it:
       five fragments of road noise joined together are nine words long and
       still contain nothing to answer, which is exactly what the drive
       produced ("Hello. Hello. Thanks for your help. Got it. Hey."). */
    function noContentWords(text) {
      var words = normWords(text);
      if (!words.length) return true;
      var noiseWords = policyList('noise_tokens', []);
      for (var i = 0; i < words.length; i++) {
        if (noiseWords.indexOf(words[i]) < 0) return false;
      }
      return true;
    }

    /* SHE IS NOT IN THE ROOM. Nothing she has said could be coming back.
     *
     * The one fact that separates "hello" the greeting from "hello" the echo
     * of her own opening line, and the only reason the social words below can
     * be treated differently from the rest of the noise list at all. It is
     * deliberately stricter than looksLikeEcho(): not "these words are not
     * hers" but "there is nothing of hers in the air", so a short reply is
     * answered only when the room is genuinely quiet. */
    function nothingToEcho() {
      if (speaking && !speaking.cancelled) return false;
      if (!lastAudioAt) return true;
      return (now() - lastAudioAt) > echoTailMs;
    }

    /* Is this the shortest complete turn there is — a greeting, a yes, a no?
     *
     * THE FAULT THIS CLOSES, measured on the desk on 2026-09-20: the driver
     * said "hello" into a working microphone and was told "Didn't catch that."
     * twice, and answered "No." to a question twice more. The transcripts were
     * perfect. grok-transcribe returned the word, server_vad closed the turn,
     * and the gate threw it away because it was one word long and that word is
     * in noise_tokens.
     *
     * Those words are in the list because they are what HER voice comes back
     * as -- and that is a statement about a moment, not about a vocabulary.
     * When she has said nothing to echo, a greeting is a greeting. */
    function shortSocialTurn(text) {
      var words = normWords(text);
      if (!words.length || words.length > 3) return false;
      var social = policyList('social_tokens', []);
      for (var i = 0; i < words.length; i++) {
        if (social.indexOf(words[i]) < 0) return false;
      }
      // Her own voice, or a moment where it could be: the old behaviour
      // exactly, which is what keeps the echo loop broken.
      if (!nothingToEcho() || looksLikeEcho(text)) return false;
      return true;
    }

    function unintelligible(text) {
      var t = String(text == null ? '' : text).trim();
      if (!t) return true;
      // A question mark is a question, at any length.
      if (t.indexOf('?') >= 0) return false;
      var words = normWords(t);
      if (!words.length) return true;
      /* THE LENGTH TEST AND THE CONTENT TEST, and one alone is not enough.
         Length alone swallows "What's that?", three words and the most common
         real question in the car. Content alone would call a whole spoken
         sentence noise the moment it happened to be made of common words. */
      if (words.length > (turnPolicy.noise_max_words || 4)) return false;
      return noContentWords(t);
    }

    /* One fragment into the buffer, and the window pushed out.
     *
     * The response the server created for it is cancelled on sight: she must
     * not answer road noise even once, let alone five times. */
    function noteFragment(text) {
      var at = now();
      if (!noise) noise = { text: '', first: at, last: at, n: 0 };
      noise.text = (noise.text + ' ' + String(text || '')).trim();
      noise.last = at;
      noise.n++;
      counters.fragments_coalesced++;
      /* THE ANSWER IN FLIGHT IS LEFT ALONE, and that is deliberate.
       *
       * A fragment is by definition something nobody could confirm was a
       * person, and half of them are her own voice off the windscreen. Ending
       * her sentence on that evidence is the exact failure the echo gate was
       * built to remove -- five long answers into an empty car, five cut off.
       * If it really was a barge-in, bargeIn() has already muted and the barge
       * path owns the cancel.
       *
       * What IS stopped is the response the SERVER creates for the committed
       * utterance, cancelled by id the moment it is announced. That is the
       * stacking this whole mechanism exists for, and it costs her nothing. */
      silenceResponses++;
      emit('LIVE_TURN_FRAGMENT', {
        turn: turnSeq, n: noise.n, text: String(text || '').slice(0, 160),
        span_ms: Math.round(noise.last - noise.first),
      });
      if (noise.timer) clearTimeout(noise.timer);
      noise.timer = setTimeout(flushNoise, turnPolicy.noise_coalesce_ms || 2000);
    }

    /* The window closed. Either the fragments add up to something, or they
     * were the road. */
    function flushNoise() {
      if (!noise) return;
      var buf = noise;
      noise = null;
      if (buf.timer) clearTimeout(buf.timer);
      silenceResponses = 0;
      var joined = buf.text;

      /* THREE FRAGMENTS THAT ADD UP TO A QUESTION are a question. Semantic VAD
         cuts a driver into pieces on a rough road too, and the pieces are then
         individually short and individually meaningless. */
      /* NO LENGTH CEILING HERE, deliberately: a buffer is as long as the road
         was rough, and what decides it is whether anything in it is a
         request. */
      var recovered = (!noContentWords(joined) && normWords(joined).length > 1)
                      ? 'coalesced'
      /* ...OR IT WAS SOMEBODY SAYING HELLO.
       *
       * ONE fragment in the window, made of nothing but social words, with
       * nothing of hers in the air to be an echo of. The burst is what makes
       * road noise road noise -- "Hello. Hello. Thanks for your help. Got it."
       * inside ten milliseconds, five pieces with nothing joining them -- and
       * a person who says hello says it once and then waits.
       *
       * DECIDED HERE, AT THE CLOSE OF THE WINDOW, and not when the fragment
       * arrived, because when it arrived there was no evidence either way:
       * the first fragment of a burst and a greeting are the same transcript.
       * Two seconds later they are not, and the fragment path in the meantime
       * is unchanged -- the server's stacked response is still cancelled on
       * sight, which is what that path is for.
       *
       * MEASURED, the desk session of 2026-09-20: "Hello." at t=517.4 and
       * again at t=538.2, twenty-one seconds apart, one fragment in each
       * window, both answered "Didn't catch that." */
                      : (buf.n === 1 && shortSocialTurn(joined)) ? 'social'
                      : null;
      if (recovered) {
        counters.fragments_recovered++;
        emit('LIVE_TURN_RECOVERED', {
          turn: turnSeq, n: buf.n, text: joined.slice(0, 200),
          why: recovered,
          span_ms: Math.round(buf.last - buf.first),
        });
        turnSeq++;
        lastTurnText = joined;
        lastTurnAt = now();
        transcriptFresh = true;
        // Every fragment is already an item in the conversation; one response
        // now answers all of them together.
        try { send({ type: 'response.create' }); } catch (e) {}
        return;
      }

      var since = now() - lastNoiseReplyAt;
      if (lastNoiseReplyAt && since < (turnPolicy.noise_reply_cooldown_ms || 12000)) {
        counters.noise_replies_suppressed++;
        emit('LIVE_NOISE_DROPPED', {
          turn: turnSeq, n: buf.n, text: joined.slice(0, 200),
          since_reply_ms: Math.round(since),
        });
        return;
      }
      lastNoiseReplyAt = now();
      counters.noise_replies++;
      emit('LIVE_NOISE_REPLY', {
        turn: turnSeq, n: buf.n, text: joined.slice(0, 200),
        span_ms: Math.round(buf.last - buf.first),
      });
      /* ONE LINE, SPOKEN DIRECTLY. Not a response.create: asking the model
         what to say about noise is asking it to invent a reason it could not
         hear, and it is a round trip for four words that are always the same
         four words. */
      try { speakDirect(turnPolicy.noise_reply || "Didn't catch that."); }
      catch (e) {}
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
    /* SHE MADE A SOUND — and how long the driver waited for it.
     *
     * Emitted once per response, on the first transcript delta, which is the
     * earliest evidence that audio is on its way out of the speaker. Three
     * clocks, because "slow" means a different thing for each kind of line:
     *
     *   wait_ms    turn end to first audio. THE number: what the driver
     *              actually experienced, across the tool call, the model and
     *              the network. Only meaningful for a line that answers a
     *              question, so it is null for anything the driver did not ask
     *              for -- a turn call is not late because nobody spoke first.
     *   create_ms  the response opening to first audio: the session's own
     *              share of that wait, with the driver's timing taken out.
     *   asked_ms   for a dictated or direct line, how long from asking for it
     *              to hearing it. This is the number the speak budgets are
     *              set against, and until now nothing recorded whether they
     *              were right.
     *
     * The kind matters as much as the number: an answer, a warning read
     * verbatim, a vetted line spoken straight, and a resumed half-answer have
     * four different acceptable latencies and used to be one undifferentiated
     * silence in the log. */
    function noteFirstAudio(responseId) {
      if (!responseId || firstAudioSeen[responseId]) return;
      firstAudioSeen[responseId] = 1;
      firstAudioIds.push(responseId);
      /* Bounded, and small: this is a dedupe set for responses that are alive
         now, not a record of the drive. */
      while (firstAudioIds.length > 32) delete firstAudioSeen[firstAudioIds.shift()];
      var at = Date.now();
      var kind = 'conversation';
      var asked = 0;
      var channel = null;
      if (dictation && dictation.responseId === responseId) {
        kind = 'dictation'; asked = dictation.at || 0;
        channel = dictation.channel || null;
      } else if (directSpeech && directSpeech.realId === responseId) {
        kind = 'direct'; asked = directSpeech.at || 0;
      } else if (speaking && speaking.responseId === responseId
                 && speaking.resumed) {
        kind = 'resume';
      }
      var opened = (speaking && speaking.responseId === responseId)
        ? speaking.startedAt : 0;
      emit('LIVE_SPOKE', {
        response_id: responseId,
        turn_kind: kind,
        channel: channel,
        turn: turnSeq,
        /* A turn call answers nobody, so "how long since the driver stopped
           talking" is not a wait it can be judged on. Null rather than a
           large number, so an average over a drive stays an average of
           answers. */
        wait_ms: (kind === 'conversation' || kind === 'direct' || kind === 'resume')
                 && turnEndAt ? Math.round(at - turnEndAt) : null,
        create_ms: opened ? Math.round(at - opened) : null,
        asked_ms: asked ? Math.round(at - asked) : null,
      });
    }

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
        /* CONTAINMENT, WITH A CLOCK ON IT FOR THE SHORT ONES.
         *
         * Fifteen seconds is the right memory for a verbatim sentence of hers
         * coming back. It is far too long for one word: "yeah" appears
         * somewhere in almost anything she says, so any driver who agrees with
         * her inside a quarter of a minute is refused as an echo.
         *
         * MEASURED on the desk, 2026-09-20: turn_phantom "Yeah." refused as
         * echo_of_her_own_words with since_audio_ms = 10932. A loudspeaker in
         * the same room is not eleven seconds late; that was a driver
         * answering, and the answer was dropped.
         *
         * So a short utterance is only hers if her voice was in the room just
         * now -- see config.REALTIME_ECHO_TEXT_SHORT_MS. */
        var shortOne = words.length < echoTextMinWords;
        if (shortOne && lastAudioAt && (at - lastAudioAt) > echoTextShortMs
            && !(speaking && !speaking.cancelled)) {
          return false;
        }
        if (hay.indexOf(' ' + words.join(' ') + ' ') >= 0) return true;
        // Below a couple of words, only exact containment counts: "yes" and
        // "no" share every word with almost anything she has ever said.
        if (shortOne) return false;
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
        /* A RESPONSE THAT HAS NOT MADE A SOUND IS NOT A VOICE IN THE ROOM.
         *
         * The branch below says it is the path a genuine second question takes
         * while she is waiting on a tool call. It was not: `speaking` is set
         * from `response.created`, and a response can sit open for seconds
         * before it produces anything -- the model deliberating, or writing a
         * function call. The drive of 2026-09-17 has two of them open with no
         * audio for 12.7 s and 37.3 s. For every one of those seconds
         * `speakingNow` was true, so the branch could not be reached, and a
         * question asked into the silence was refused on the grounds that it
         * might be an echo of a voice that had not spoken.
         *
         * `audioAt` is set on the first audio of a response and is the honest
         * test: no audio yet, and nothing of hers within the echo tail, means
         * there is nothing for this transcript to be an echo OF. A direct line
         * is excluded because its audio does not come through the delta path
         * that sets `audioAt` -- it is audible while looking silent here, and
         * `lastAudioAt` is what catches it. */
        var silentSoFar = !!(speaking && !speaking.direct && !speaking.audioAt);
        var tail = now() - lastAudioAt;
        if ((!speakingNow || silentSoFar) && tail > echoTailMs) {
            // Nothing of hers is in the room. A transcript is a question, and
            // this is the path a genuine second question takes while she is
            // waiting on a tool call.
            return { allow: true, why: null };
        }
        /* A LOUDSPEAKER IS NOT SIX SECONDS LATE.
         *
         * The test above needs BOTH "she is not speaking" and "her tail has
         * passed", and `speaking` stays true for the whole of a response --
         * including the gap between its last audio chunk and response.done,
         * which on a tool turn is however long the tool takes. So a driver who
         * asks a second question into that gap is refused as her own echo.
         *
         * MEASURED, 2026-09-21: turn_phantom "What are they playing?" refused
         * as barge_not_sustained with since_audio_ms = 6483, speaking true,
         * self_answered false. Her voice had not been in the room for six and
         * a half seconds. The question was a driver's and it was dropped; the
         * showtimes it was asking about were asked for again four seconds
         * later.
         *
         * This is the same fault this file already fixed once, one path over:
         * the short-utterance echo test carries a note reading "a loudspeaker
         * in the same room is not eleven seconds late; that was a driver
         * answering, and the answer was dropped." The reasoning was never
         * applied here.
         *
         * So: past ECHO_IMPOSSIBLE_MS of actual silence, an echo is not
         * physically available as an explanation and `speaking` does not get a
         * vote. An acoustic path across a cabin is tens of milliseconds and
         * the playout tail is a fraction of a second; two seconds is far past
         * both, and still well inside the gap a tool call opens. */
        if (tail > ECHO_IMPOSSIBLE_MS) {
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
        cancelGeneration('superseded');
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
    /* THE BACKSTOP under semantic_vad's tail. 0 is off. Fallbacks only; the
       real values arrive with the session. See config.REALTIME_TURN_BACKSTOP_MS. */
    var turnBackstopMs = cfg.turnBackstopMs || 0;
    var turnBackstopMicDb = (cfg.turnBackstopMicDb === undefined
                             || cfg.turnBackstopMicDb === null)
                            ? -45 : cfg.turnBackstopMicDb;
    /* How often the margin is sampled while she is audible. 100 ms is the
       same cadence startEchoWatch already uses to make a live decision, so the
       census and the gate see the same signal at the same rate. */
    var CENSUS_SAMPLE_MS = 100;
    var echoTailMs = cfg.echoTailMs || 600;
    /* How long her voice must have been out of the room before an echo stops
       being a possible explanation for a transcript. Not a tuning knob: it is
       a claim about acoustics, and it lives in config.py beside the tail it is
       not the same as. See the note in the supersede gate. */
    var ECHO_IMPOSSIBLE_MS = cfg.echoImpossibleMs || 2000;
    var echoTextWindowMs = (cfg.echoTextWindowS || 15) * 1000;
    var echoTextOverlap = cfg.echoTextOverlap || 0.8;
    var echoTextMinWords = cfg.echoTextMinWords || 2;
    /* How recent her voice has to be for a ONE-WORD transcript to be counted
       as hers. Not the same number as the window above and not the same test:
       that one is about her words, this one is about the room. */
    var echoTextShortMs = cfg.echoTextShortMs || 2000;

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
    /* WHICH RESPONSE HAS ALREADY SPENT ITS GUARD, and the race it closes.
     *
     * The guard holds a barge decision over her opening syllable and then takes
     * it "from that moment" when the hold expires. Expiring re-enters bargeIn(),
     * which re-asks onsetRemaining() -- and setTimeout and Date.now() are not
     * the same clock. Measured under CPU load (tools/echo_barge_selftest.js
     * section C, about 1 run in 15): a 38 ms hold came back with 1 ms still on
     * the clock, so the guard armed a SECOND hold for that millisecond and the
     * decision was deferred twice.
     *
     * The millisecond is harmless -- 39 ms of hold instead of 38, and at
     * production scale 1 ms out of 400. What is not harmless is a guard that can
     * re-arm itself from its own expiry: the count of holds is then a function
     * of scheduler jitter rather than of anything the driver did, and the test
     * that pinned it at one was right to. Spent once per response, so a NEW
     * response still gets its own guard, which is the whole point of the thing. */
    var onsetSpentFor = null;
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
    /* One per dictated line, so a caller that started one can cancel THAT one
       and nothing else. See speak()/cancelSpeak. */
    var dictationSeq = 0;
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
    /* ...AND THE CLAIM EXPIRES, which it did not, and that is a fault measured
     * on 2026-09-24 (session 1fd4de92).
     *
     * This is a COUNTER WITH NO IDENTITY. It says "the next unclaimed response
     * is the one I gave up on" -- true when exactly one create is in flight,
     * and false the moment a second one is. At a route start there are three
     * within a quarter of a second: the depart dictation, the near dictation
     * that supersedes it, and the CONVERSATIONAL ANSWER to the
     * start_navigation tool result. The depart line is abandoned before it has
     * a response id, so the claim is armed -- and then it takes whichever
     * response the server happens to create next.
     *
     * In that session the driver got the turn call and then silence until they
     * said "hello", which started a new turn and produced a response nothing
     * was waiting to eat. Reproduced in tools/nav_contention_selftest.js: three
     * creates, and the one silenced by id is the second, not the abandoned
     * one.
     *
     * A deadline is the cheap half of the fix and the one that does not
     * require guessing which response is which. The create being given up on
     * was ALREADY IN FLIGHT when it was abandoned, so its response arrives in
     * the time a round trip takes. In the drive the orphan fired 3.1 s after
     * the dictation was abandoned, which is not a response that was already on
     * its way; it is a different response entirely. Past this, the claim
     * lapses and a response claims the mouth the ordinary way -- which costs,
     * at worst, a turn call said twice (the arbiter supersedes by group) and
     * saves, at best, every conversational answer that lands in the window. */
    var ORPHAN_CLAIM_MS = cfg.orphanClaimMs || 1500;
    var orphanClaims = [];        // arm times, oldest first
    /* ...and whether that orphan must also be SILENCED rather than merely not
       counted. An abandoned dictation has already been replaced by another
       voice saying the same words; an abandoned direct line has not, and its
       audio is the answer. Same mechanism, opposite conclusion about the
       speaker, so they are two flags and not one. */
    var silenceOrphan = false;

    /* Arm a claim, and drop any that have gone stale. */
    function armOrphan(silence) {
      orphanClaims.push(now());
      orphanOutOfBand++;
      if (silence) silenceOrphan = true;
    }

    /* Is there still a live claim? Expires the stale ones on the way past, so
       a claim armed and never used cannot sit there waiting to eat an answer
       ten seconds later. */
    function orphanClaimLive() {
      var t = now();
      while (orphanClaims.length && (t - orphanClaims[0]) > ORPHAN_CLAIM_MS) {
        orphanClaims.shift();
        orphanOutOfBand--;
        counters.orphan_claims_expired++;
        emit('LIVE_ORPHAN_CLAIM_EXPIRED', { after_ms: ORPHAN_CLAIM_MS });
      }
      if (orphanOutOfBand <= 0) { silenceOrphan = false; return false; }
      return true;
    }
    var verbatimInstruction = cfg.verbatimInstruction ||
      'Read the text below out loud, exactly as written, word for word. ' +
      'Add nothing. Remove nothing. Do not rephrase.\n\nTEXT:\n';
    var speakTimeoutMs = cfg.speakTimeoutMs || 700;
    /* THE MOUTH IS HERS UNTIL THE SOUND STOPS, on a transport that can say
       when that is. `response.done` is the end of GENERATION; under speech-
       to-speech the audio streams on for seconds after it, and this file used
       to hand the mouth back at the first of the two -- so for every one of
       those seconds a detector firing found `speaking` null, muted the tail
       without a pendingBarge to classify or absorb it, and never unmuted;
       and a noise reply created in that window muted the tail on its way to
       being cancelled. On WebRTC the API sends output_audio_buffer.stopped
       (or .cleared) at the real end, so connect() sets this and the mouth
       waits for it. The node harness sends neither, and keeps the old end.

       NOW ASKED RATHER THAN ASSERTED. This used to be `connect()` passing a
       hardcoded `true` under a comment naming WebRTC, which meant the one
       dependency on the transport was written down in a comment and nowhere a
       program could read. The provider answers it from `tailEvidence`, which
       distinguishes the API having said the sound stopped from OUR OWN playout
       queue having drained — both hold the tail, on very different evidence,
       and the second is a downgrade worth being able to see in a log rather
       than inferring from a version number. `cfg.holdTail` still wins when
       there is no provider, which is every existing test. */
    var holdTail = provider ? provider.holdTail() : !!cfg.holdTail;
    var tailFallbackMs = cfg.tailFallbackMs || 15000;
    var tailTimer = null;
    function armTail(rid) {
      if (tailTimer) clearTimeout(tailTimer);
      tailTimer = setTimeout(function () {
        tailTimer = null;
        // The end of the audio never arrived. The mouth is not held forever
        // for it: give it back and say so.
        if (speaking && speaking.responseId === rid) {
          emit('LIVE_TAIL_TIMEOUT', { response_id: rid, after_ms: tailFallbackMs });
          endResponse(rid);
        }
      }, tailFallbackMs);
    }
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
    function cancelGeneration(why) {
      if (speaking) noteCancel(speaking.responseId, why || 'unspecified');
      if (speaking && speaking.direct && directSpeech && directSpeech.realId) {
        noteCancel(directSpeech.realId, why || 'unspecified');
      }
      /* A directly-spoken line has no response behind it to cancel -- IN TEXT
         MODE. Sending `response.cancel` with nothing generating is answered
         with an error event, which is a real error in the log for a thing that
         worked.
       *
       * UNDER SPEECH-TO-SPEECH THE SAME LINE IS A RESPONSE, and that is the
       * half this used to get wrong. There is no sink to push words into, so
       * speakDirect reaches injectDirect, which creates a real out-of-band
       * response and reads the line through it. Skipping the cancel there does
       * not leave "nothing generating" -- it leaves the model producing a line
       * whose mouth has just been taken away, audible underneath the turn call
       * that pre-empted it, with its own audio still queued in the output
       * buffer. And when the direct budget then expires, its recovery
       * `response.create` fires a fresh conversational answer roughly two and
       * a half seconds behind the warning.
       *
       * So the test is not "is this direct" but "is there a response behind
       * it": cancel by id where injectDirect has one, disown it where the id
       * has not arrived yet (the orphan path, exactly as the budget's own
       * timeout does), and stay quiet only where there is genuinely nothing to
       * cancel. */
      var oob = (speaking && speaking.direct) ? directSpeech : null;
      if (!(speaking && speaking.direct) || oob) {
        var oobId = oob ? oob.realId : null;
        /* Not yet bound to a response: the create is still in flight and the
           response that lands belongs to nobody. Marked so it is silenced on
           sight rather than claiming the mouth. `orphanOutOfBand` is
           incremented by finish() below, which owns that bookkeeping. */
        if (oob && !oobId) silenceOrphan = true;
        try {
          send(oobId ? { type: 'response.cancel', response_id: oobId }
                     : { type: 'response.cancel' });
        } catch (e) {}
        try { send({ type: 'output_audio_buffer.clear' }); } catch (e) {}
        /* Told it was pre-empted rather than merely failed, because the two
           want opposite things: a line that never started is worth asking for
           again, and one that was outranked by a turn call is not. */
        if (oob) { try { oob.finish(false, 'preempted'); } catch (e) {} }
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
      /* The next response to open is this resume. Consumed in beginResponse
         so the latency of a resumed half-answer is not read as the latency of
         an answer to a question nobody asked twice. */
      resumeExpected = true;
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
        utterFor(responseId, 'dictation');
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
        utterFor(responseId, 'direct');
        return;
      }
      /* A RESPONSE THE SERVER CREATED FOR A FRAGMENT. It is an answer to road
         noise; it never claims the mouth and it is cancelled by id, which is
         the same pair of moves the orphan path below makes. */
      if (silenceResponses > 0
          && String(responseId || '').indexOf('direct:') !== 0) {
        silenceResponses--;
        markOutOfBand(responseId);
        counters.noise_responses_silenced++;
        try { send({ type: 'response.cancel', response_id: responseId }); }
        catch (e) {}
        /* NOT MUTED HERE. This used to mute the element on the way to
           cancelling -- and under speech-to-speech the element is one stream,
           so the mute landed on whatever was coming out of it, which is the
           tail of the answer before this one. The drive of 2026-09-17 did it
           twice in five seconds, once across "Didn't catch that." itself.
           The response is cancelled before it has made a sound; if a sound
           does arrive, output_audio_buffer.started names it and it is muted
           THEN. See silencedIds. */
        silencedIds[responseId] = 1;
        noteCancel(responseId, 'noise_silenced');
        utterFor(responseId, 'noise');
        emit('LIVE_NOISE_SILENCED', { response_id: responseId });
        return;
      }
      if (orphanClaimLive()
          && String(responseId || '').indexOf('direct:') !== 0) {
        orphanOutOfBand--;
        orphanClaims.shift();
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
          // Same rule as a noise reply: muted if and when its audio starts.
          silencedIds[responseId] = 1;
          noteCancel(responseId, 'orphan_silenced');
          utterFor(responseId, 'orphan');
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
                    direct: !!opts.direct, resumed: resumeExpected,
                    // For the onset guard: when the response opened, and (set
                    // on the first transcript delta) when she began speaking.
                    startedAt: Date.now(), audioAt: 0 };
      utterFor(responseId, opts.direct ? 'direct' : (resumeExpected ? 'resume' : 'conversation'));
      resumeExpected = false;
      speaking = entry;
      counters.responses++;
      partial = '';
      generated = '';
      if (sink) { try { sink.begin(responseId); } catch (e) {} }
      audio.unmute('claim');
      arbiter.say({
        priority: arbiter.P.CONVO,
        group: 'convo',
        id: 'live:' + (responseId || String(counters.responses)),
        /* USUALLY EMPTY, and for a directly-spoken line it must not be.
           The model composes an ordinary answer as it goes, so there is no
           text to hand over here and the echo gate learns the words from the
           transcript deltas instead. A direct line has none of those -- it is
           already written -- so without this the one place that sees
           everything RIO says (the arbiter's `start`, which calls noteSaid)
           never sees it, and "Didn't catch that." coming back through the
           microphone on a rough road is not recognised as her own voice. */
        text: opts.text || '',
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
          audio.mute('preempted');
          cancelGeneration('preempted');
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
            if (wasCurrent) audio.mute('interrupted');
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
      stopCensus('response_end');
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
    /* ---- THE TURN BACKSTOP ------------------------------------------------
     *
     * A timer UNDER semantic_vad, and only under it: it fires when the
     * driver's microphone has been quiet for turnBackstopMs and the server has
     * still not said the turn is over. Then the browser commits the buffer
     * itself.
     *
     * It exists because semantic_vad's tail is a judgement rather than a
     * timer, so it has no worst case -- median 705 ms, p95 1308 ms, and 829 to
     * 2482 ms on the first turn of a live session. A driver feels the worst
     * case, not the median.
     *
     * WHY THE METER AND NOT THE SERVER'S OWN EVENTS. The thing being
     * backstopped IS the server's detector, so its silence cannot be the
     * evidence that there is silence. `levels()` is the page's own microphone
     * measurement and the only independent signal there is. No meter, no
     * backstop -- the same rule the level test follows, and for the same
     * reason: absence is not evidence.
     *
     * It is OFF unless config says otherwise, and it must stay far enough out
     * to be invisible on the turns semantic_vad already handles well. A
     * backstop that fires on an ordinary turn is server_vad with extra steps. */
    var backstopTimer = null;
    var backstopQuietSince = 0;

    function stopBackstop() {
      if (backstopTimer) { clearInterval(backstopTimer); backstopTimer = null; }
      backstopQuietSince = 0;
    }

    function startBackstop() {
      stopBackstop();
      if (!turnBackstopMs || !levels) return;
      backstopQuietSince = 0;
      backstopTimer = setInterval(function () {
        if (stopped || !speechActive) { stopBackstop(); return; }
        var v = readLevels();
        if (!v) return;                       // no meter: say nothing
        if (v.mic > turnBackstopMicDb) {      // still talking
          backstopQuietSince = 0;
          return;
        }
        var at = now();
        if (!backstopQuietSince) { backstopQuietSince = at; return; }
        if (at - backstopQuietSince < turnBackstopMs) return;
        // Quiet this long with the detector still holding the turn open.
        stopBackstop();
        counters.turns_backstopped++;
        emit('LIVE_TURN_BACKSTOP', {
          turn: turnSeq, quiet_ms: Math.round(at - backstopQuietSince),
          mic_db: v.mic,
        });
        try { send({ type: 'input_audio_buffer.commit' }); } catch (e) {}
      }, 100);
    }

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
    /* THE GATE'S OTHER ERROR, WHICH NOTHING HAS EVER COUNTED.
     *
     * `false_barge_in` counts the times the gate stopped her for something
     * that turned out not to be a person. There is no number anywhere for the
     * opposite: the times it refused a barge-in that WAS a person. Both are
     * the same threshold being wrong, in opposite directions, and a drive that
     * reports only one of them can only ever argue for tightening.
     *
     * The evidence is free and nobody was collecting it. A suppression says
     * "the microphone never beat the loudspeaker, so that was her". If a real
     * transcript -- not her own words back, not a question already being
     * answered -- arrives in the moment after one, the gate was wrong: there
     * was somebody there and she did not stop for them.
     *
     * Recorded with the margin that refused it, which is the only number that
     * can move REALTIME_BARGE_ECHO_MARGIN_DB_TOUCH honestly. The drive of
     * 2026-09-17 suppressed at -17.3, -1.3 and -6.2 dB against a required
     * +6 dB and the log could not say whether any of the three was a person.
     * Until it can, that 6 is not a number anyone should be editing. */
    var lastSuppress = null;   // { at, responseId, margin, mic, out, reason }

    function noteSuppressed(reason, rid, v) {
      lastSuppress = {
        at: now(), responseId: rid || null, reason: reason,
        margin: (peakMargin === null || peakMargin === undefined)
                ? null : Math.round(peakMargin * 10) / 10,
        mic: v && v.mic !== undefined ? Math.round(v.mic * 10) / 10 : null,
        out: v && v.out !== undefined ? Math.round(v.out * 10) / 10 : null,
      };
    }

    /* ...and the other half: a transcript that proves the last suppression
       wrong. Called from transcriptArrived once the words have been judged
       real, driver's, and not an answer already in flight. */
    function noteMissedBarge(text) {
      if (!lastSuppress) return;
      var age = now() - lastSuppress.at;
      var s = lastSuppress;
      lastSuppress = null;
      /* Only the moment after. Further out than the window a barge-in would
         have been confirmed in, the transcript belongs to a later turn and
         blaming the suppression for it would invent the number this exists to
         measure honestly. */
      if (age > bargeConfirmMs) return;
      counters.barges_missed++;
      emit('LIVE_BARGE_MISSED', {
        response_id: s.responseId, reason: s.reason,
        margin_db: s.margin, mic_db: s.mic, out_db: s.out,
        required_db: bargeEchoMarginDb,
        since_suppress_ms: Math.round(age),
        text: String(text || '').slice(0, 160),
      });
    }

    /* ---- THE MARGIN, MEASURED, WHETHER OR NOT THE GATE IS ARMED ----------
     *
     * "That margin is a number to measure per environment, not guess — and
     * desk and car need different ones." Both halves of that were impossible
     * before this. config.py has carried two columns since the iPhone fix
     * (REALTIME_BARGE_ECHO_MARGIN_DB = 0 on a desk, 6 on touch) and NEITHER
     * number came from a measurement of a room, because the only code that
     * ever read the meter was code gated on the number already being set:
     *
     *     function echoShaped() {
     *       if (!bargeEchoMarginDb) return false;   // desk: the test is off
     *       var v = readLevels();                   // <- never reached
     *
     * So on a desk the meter was built, attached, and never read; the only
     * mic-vs-output numbers that ever reached the log came from sessions where
     * the gate fired, which is exactly the sample you cannot set a threshold
     * from. echo_suppressed: 0 meant "no evidence", and it was being read as
     * "no echo".
     *
     * This samples the same meter while she is SPEAKING and the output is
     * above the floor -- the only moments when echo is possible at all -- and
     * reports the distribution. A desk and a car then produce two histograms
     * of the same quantity, and the margin for each is a reading off them
     * rather than an opinion.
     *
     * It decides nothing. No gate consults it; it cannot suppress or cause a
     * barge-in. It is instrumentation, and it runs on both columns. */
    var census = null;          // { n, sum, min, max, sorted[] } per response
    var censusTimer = null;

    function censusSample() {
      var v = readLevels();
      if (!v) return;
      if (v.out < bargeEchoFloorDb) return;   // she is not making a sound
      var margin = v.mic - v.out;
      if (!census) return;
      census.n++;
      census.sum += margin;
      census.samples.push(margin);
      if (census.min === null || margin < census.min) census.min = margin;
      if (census.max === null || margin > census.max) census.max = margin;
      if (v.mic > (census.mic_max === null ? -Infinity : census.mic_max)) census.mic_max = v.mic;
      if (v.out > (census.out_max === null ? -Infinity : census.out_max)) census.out_max = v.out;
    }

    function startCensus(rid) {
      if (!levels || censusTimer) return;
      census = { rid: rid, n: 0, sum: 0, min: null, max: null,
                 mic_max: null, out_max: null, samples: [] };
      censusTimer = setInterval(censusSample, CENSUS_SAMPLE_MS);
    }

    function stopCensus(why) {
      if (censusTimer) { clearInterval(censusTimer); censusTimer = null; }
      var c = census;
      census = null;
      if (!c || !c.n) return;
      var sorted = c.samples.slice().sort(function (a, b) { return a - b; });
      var q = function (f) {
        return Math.round(sorted[Math.min(sorted.length - 1,
                          Math.floor(f * sorted.length))] * 10) / 10;
      };
      counters.echo_census_responses++;
      counters.echo_census_samples += c.n;
      emit('LIVE_ECHO_CENSUS', {
        response_id: c.rid,
        why: why || null,
        n: c.n,
        sample_ms: CENSUS_SAMPLE_MS,
        /* mic minus output, in dB, while she was audible. NEGATIVE is the
           normal state -- the microphone hears less than the speaker renders.
           A margin at or above the configured requirement is what the gate
           would have called a driver.

           Flat, not a nested object: these rows are read one per line out of
           /realtime/cutoffs next to mic_db and out_db from the gate's own
           events, and one of the two shapes has to give. */
        margin_p50_db: q(0.5),
        margin_p90_db: q(0.9),
        margin_max_db: Math.round(c.max * 10) / 10,
        margin_min_db: Math.round(c.min * 10) / 10,
        margin_mean_db: Math.round((c.sum / c.n) * 10) / 10,
        mic_peak_db: c.mic_max === null ? null : Math.round(c.mic_max * 10) / 10,
        out_peak_db: c.out_max === null ? null : Math.round(c.out_max * 10) / 10,
        /* WHAT THE SHIPPED NUMBER WOULD HAVE DONE with this room in it, so the
           two columns can be compared against one recording rather than
           against each other's reputation. */
        required_db: bargeEchoMarginDb,
        floor_db: bargeEchoFloorDb,
        device: cfg.device || 'unknown',
      });
    }

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
      if (onsetLeft > 0 && speaking && !pendingBarge
          && onsetSpentFor !== speaking.responseId) {
        if (!onsetHold) {
          var heldFor = speaking.responseId;
          emit('LIVE_BARGE_DEFERRED', { response_id: heldFor,
                                        guard_ms: bargeOnsetGuardMs,
                                        holding_ms: Math.round(onsetLeft) });
          onsetHold = setTimeout(function () {
            onsetHold = null;
            if (stopped || !speechActive) return;
            if (!speaking || speaking.responseId !== heldFor) return;
            /* SPENT BEFORE THE RE-ENTRY, not after. bargeIn() below re-asks
               onsetRemaining(), which can still report a millisecond left --
               see onsetSpentFor. Marking it here is what makes the hold happen
               once per response instead of once per scheduler hiccup. */
            onsetSpentFor = heldFor;
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
        noteSuppressed('level_margin', rid0, lastLevels);
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

      audio.mute('barge');               // instant, always, undoable
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
          noteSuppressed('level_margin_sustained', rid, lastLevels);
          emit('LIVE_ECHO_SUPPRESSED', {
            response_id: rid,
            margin_db: Math.round(peakMargin * 10) / 10,
            required_db: bargeEchoMarginDb,
            reason: 'level_margin_sustained',
          });
          clearBarge();
          try { send({ type: 'input_audio_buffer.clear' }); } catch (e) {}
          if (speaking && !speaking.cancelled && !stopped) audio.unmute('barge_absorbed');
          startEchoWatch(rid);
          return;
        }
        pendingBarge.cancelled = true;
        pendingBarge.said = saidSoFar();
        // Sustained past the gate: stop generating and hand back the mouth.
        cancelGeneration('barge');
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
      /* WHEN THE DRIVER STOPPED TALKING -- the start of the only latency
         measurement that describes what the drive felt like.
       *
       * "She takes ages to answer" is a stopwatch from the end of the
       * question to the first sound of the answer, and until this clock
       * existed the drive log had neither end of it. It had tool durations,
       * which are a component and not the number: a `look` that took 4.3 s is
       * not the same fact as a driver waiting 6 s in silence, and on the drive
       * of 2026-09-17 the log could state the first and not the second. */
      turnEndAt = Date.now();
      // The detector got there on its own, which is the ordinary case and the
      // one the backstop must never pre-empt.
      stopBackstop();
      stopEchoWatch();
      /* A firing that never got past her opening syllable, and has now stopped
         on its own. That is the shape of echo and it cost nothing: no mute, no
         cancel, no gap in the answer. Counted where the other suppressions are
         counted, because the question it answers is the same one -- how often
         is the detector firing on her. */
      if (onsetHold) {
        clearOnsetHold();
        counters.echo_suppressed++;
        noteSuppressed('onset_guard', speaking ? speaking.responseId : null, null);
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
        if (speaking && !speaking.cancelled && !stopped) audio.unmute('barge_absorbed');
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
    function transcriptArrived(text, itemId) {
      var real = !!(text && text.trim());

      /* THE ANSWER IN FLIGHT IS ALREADY THIS QUESTION'S.
       *
       * The transcription races the model, and on a tool turn the model wins:
       * look() is already running for this utterance by the time the words
       * for it arrive. Superseding here cancels the answer to the question
       * that is arriving -- measured, it aborted the tool, discarded the
       * result and left the turn silent, every time, on a live session.
       *
       * Identity rather than heuristics: the transcription carries the SAME
       * item_id the utterance was committed under, and the response created
       * after that commit was bound to it. Same id means the response being
       * cancelled IS the answer to these words. A real second question
       * arrives under a different id and still supersedes -- which is the
       * behaviour the phone probe pins down and which stays pinned. */
      var selfAnswered = !!(real && itemId && itemId === answeringItemId);
      if (selfAnswered) {
        emit('LIVE_TURN_SELF', {
          turn: turnSeq, item_id: itemId, text: String(text).slice(0, 160),
        });
        // It IS this turn's text -- it just is not a new turn. Recorded so a
        // genuine follow-up is still measured for continuation against the
        // right sentence, and so the camera resolves "the black one" against
        // the question actually asked.
        lastTurnText = text;
        lastTurnAt = now();
        transcriptFresh = true;
      }
      /* NOT `real = false`, deliberately. The driver DID speak -- these are
         their words -- so the barge block below must still classify the
         speech as real. Setting it false there would file a genuine question
         as a false_barge_in with 'empty transcript' for a reason and arm a
         resume against it. What this utterance is not is a NEW turn, and that
         is the only thing selfAnswered suppresses. */

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
      /* A TRANSCRIPT ALREADY BEING ANSWERED IS NOT A PHANTOM, and this is the
       * `!selfAnswered` that was missing from the drive of 2026-09-17.
       *
       * Transcription loses the race with the model, routinely. The server
       * commits the utterance, creates a response for it, and the words for
       * that same utterance arrive afterwards -- by which time she has already
       * started answering them. `speaking` is true, the gate looks for a
       * confirmed barge-in behind a transcript that never was one, finds
       * nothing, and files an ordinary question as a refused barge-in.
       *
       * ELEVEN OF ELEVEN on that drive. Every phantom in the log was the
       * driver asking something perfectly normally; not one was an echo, an
       * interruption, or a gate refusal of anything real. The tally exists to
       * answer "is her voice coming back through the phone and being taken
       * for a driver", and it answered with false positives only -- which is
       * worse than not answering, because the number reads as a diagnosis.
       *
       * It also overrode a decision made deliberately thirty lines above: the
       * `real = false` below cancels the comment that says NOT to do that, so
       * a genuine barge-in whose transcript happened to be late was then
       * classified by the block at the bottom of this function as a
       * false_barge_in with no transcript behind it.
       *
       * Nothing is lost by skipping it. selfAnswered already suppresses the
       * new turn, the supersede and the noise buffer; LIVE_TURN_SELF already
       * records that this happened. */
      /* THE SUPPRESSION BEFORE THIS ONE WAS WRONG, if these are a driver's
         words. Asked before the gate's verdict is acted on, and only for
         words that are real, not hers, and not an answer already in flight --
         which is exactly the set a suppression claimed did not exist. */
      if (real && !selfAnswered && !looksLikeEcho(text)) noteMissedBarge(text);
      if (real && !selfAnswered && !gate.allow) {
        counters.turns_phantom++;
        emit('LIVE_TURN_PHANTOM', {
          why: gate.why, text: text.slice(0, 160), turn: turnSeq,
          speaking: !!(speaking && !speaking.cancelled),
          said_chars: (saidSoFar() || '').length,
          since_audio_ms: Math.round(now() - lastAudioAt),
          /* WAS IT ANSWERED ANYWAY? The difference between a lost question and
             a harmless one, and the log could not tell them apart.
           *
           * A refused transcript whose response already exists -- same
           * item_id, created by the server when the utterance was committed --
           * is being answered as these words arrive. It is not a new turn and
           * does not supersede, which is all the gate refused; nothing was
           * lost. A refused transcript WITHOUT one goes to the noise buffer
           * and is answered two seconds later, apologised for, or dropped.
           *
           * The drive of 2026-09-17 logged eleven phantoms, every one of them
           * a real question the driver asked out loud, and reading that tally
           * honestly meant cross-checking each against the fragment events
           * that did not follow it. Written down instead. */
          self_answered: selfAnswered,
        });
        real = false;
      }

      /* ROAD NOISE, COALESCED RATHER THAN ANSWERED ONE FRAGMENT AT A TIME.
       *
       * Reached two ways, and both are the same event from the driver's seat:
       * a transcript the phantom gate refused (`!real` after the block above,
       * which is what the drive of 2026-09-09 produced twenty of), and one
       * that is real enough but carries no request in it. Either way the
       * SERVER has already created a response for the committed utterance and
       * she is about to answer nothing.
       *
       * A driver command is exempt at any length: "stop" is two letters and is
       * the most important thing anyone says in this car.
       */
      /* HER OWN VOICE IS NOT A FRAGMENT, and this test comes first.
       *
       * A transcript refused because it was RIO -- inside the onset guard,
       * over the level margin, or her words verbatim -- is already handled:
       * nothing is superseded and she carries on talking. Feeding it to the
       * noise buffer would cancel the very answer it is an echo of, which is
       * the bug the echo gate was built to remove. Only a transcript that
       * failed the barge test for want of CONFIRMATION (nobody could tell
       * whether anyone spoke) is a candidate for the buffer.
       */
      var herOwnVoice = !real && gate.why && gate.why !== 'no_confirmed_barge'
                        && gate.why !== 'empty';
      if (!cmd && !selfAnswered && !herOwnVoice
          && (unintelligible(text)
              || (!real && text && !cont && gate.why === 'no_confirmed_barge'))) {
        noteFragment(text);
        return;
      }

      /* A REAL TURN ENDS THE WINDOW, and takes what is in it.
       *
       * "Hey. ... so what is that building?" is one question with a rough road
       * in the middle of it. The fragments are already items in the
       * conversation and the model will read them either way; what must not
       * happen is the buffer flushing afterwards and apologising for words the
       * driver has just been answered about. */
      if (noise) {
        var buf0 = noise;
        noise = null;
        if (buf0.timer) clearTimeout(buf0.timer);
        silenceResponses = 0;
        counters.fragments_recovered++;
        emit('LIVE_TURN_RECOVERED', {
          turn: turnSeq, n: buf0.n,
          text: (buf0.text + ' ' + String(text || '')).trim().slice(0, 200),
          span_ms: Math.round(now() - buf0.first),
        });
      }
      if (real && !selfAnswered) {
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
            cancelGeneration('coalesced');
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
          cancelGeneration('barge');
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
        try { audio.mute('driver_command'); } catch (e) {}
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
                  onSpoken: onSpoken, at: Date.now() };
        d.finish = function (okay, why) {
          if (directSpeech !== d) return;
          directSpeech = null;
          if (d.timer) { clearTimeout(d.timer); d.timer = null; }
          if (!okay) {
            // Given up on before it was ever bound to a response? Then the
            // response is still on its way and is nobody's. See
            // orphanOutOfBand.
            if (!d.realId) armOrphan(false);
            counters.direct_speech_failures++;
            emit('LIVE_DIRECT_SPEECH_FAILED', { text: line, reason: why });
            /* A LAST TRY, rather than a silent turn. The tool result is
               already in the conversation, so asking for a response gets the
               question answered late instead of not at all -- the same
               recovery the mouth-was-busy branch makes. Nothing was written
               into the history claiming she spoke, because that is written
               from `started` and this line never started.
             *
             * ...EXCEPT WHEN SOMETHING MORE URGENT TOOK THE MOUTH. The
             * recovery exists for a line that never got out at all. A line
             * pre-empted by a turn call was not lost, it was OUTRANKED, and
             * the arbiter has already given the mouth to the warning. Asking
             * for a response here puts a conversational answer underneath
             * that warning and -- on the direct budget -- roughly two and a
             * half seconds after the driver stopped hearing it. */
            if (why !== 'preempted') {
              try { send({ type: 'response.create' }); } catch (e) {}
            }
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
      beginResponse(id, { direct: true, text: line });
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

    /* WHAT THE SESSION WAS DOING WHEN A TOOL RESULT WENT BACK.
     *
     * A tool answered in the PAGE leaves no server row at all -- no call, no
     * result, no response.create -- so on 2026-09-24 the question "was a
     * response created after start_navigation returned" could not be answered
     * from the drive log, which is where a route that locks in silence hides.
     * This is the answer, attached to LIVE_TOOL_RESULT: whether the create was
     * sent, and the four pieces of state that decide whether anything comes of
     * it. */
    function sessionState() {
      var st = null;
      try { st = arbiter && arbiter.state ? arbiter.state() : null; }
      catch (e) { st = null; }
      return {
        // The realtime entry holding the mouth, if any. A response.create sent
        // while one of these is open is a response that may never claim.
        speaking_response_id: speaking ? speaking.responseId : null,
        speaking_finishing: speaking ? !!speaking.finishing : null,
        // A deterministic line mid-flight. The one that abandons unbound is
        // what arms an orphan claim.
        dictation: dictation ? (dictation.text || '').slice(0, 40) : null,
        dictation_bound: dictation ? !!dictation.responseId : null,
        // Outstanding claims on the next unclaimed response. Non-zero here
        // means a response created next may be cancelled on sight.
        orphan_claims: orphanOutOfBand,
        // What the mouth is actually doing, from the arbiter rather than from
        // this file's opinion of it.
        arbiter_speaking: st && st.speaking ? st.speaking.id : null,
        arbiter_queued: st ? st.queued.map(function (q) { return q.id; }) : null,
      };
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
                                         spoke_directly: true,
                                         // No create, and that is correct: the
                                         // line has already been spoken.
                                         response_requested: false,
                                         session_state: sessionState() });
              return result;
            }
            counters.direct_deferred++;
            emit('LIVE_DIRECT_DEFERRED', { path: result.path,
                                           response_id: speaking
                                             ? speaking.responseId : null });
          }
          /* CITATIONS, on their own event and before the answer is asked for.
             OpenAI's web search terms require that when web results, or
             information drawn from them, are shown to a person, inline
             citations are clearly visible and clickable. RIO's answer is
             SPOKEN, so the dashboard is the only surface that can carry them,
             and it has to have them before she starts talking rather than
             after she stops.

             A separate event rather than more fields on LIVE_TOOL_RESULT:
             that event is consumed by the drive log and by the counters, and
             a licensing obligation should not ride inside something whose
             shape is tuned for telemetry. See LICENSING.md section 4. */
          if (result && result.citations && result.citations.length) {
            emit('LIVE_SOURCES', {
              tool: name, call_id: callId,
              question: (args && args.question) || null,
              citations: result.citations,
            });
          }
          var ask = { type: 'response.create' };
          if (name === 'look' && lookAnswerMaxTokens) {
            ask.response = { max_output_tokens: lookAnswerMaxTokens };
          }
          var asked = true;
          try { send(ask); } catch (e) { asked = false; }
          emit('LIVE_TOOL_RESULT', { tool: name, call_id: callId,
                                     ok: !!result.ok, path: result.path || null,
                                     took_ms: result.took_ms || null,
                                     spoke_directly: false,
                                     /* THE QUESTION THE DRIVE LOG COULD NOT
                                        ANSWER. True means the words went back
                                        AND a spoken answer was asked for;
                                        anything the driver then fails to hear
                                        happened after this line, and
                                        session_state says what was in the way. */
                                     response_requested: asked,
                                     session_state: sessionState(),
                                     note: result.note || null });
          return result;
        });
    }

    return {
      /* One event off the data channel. Everything this file decides is
         decided here, which is what makes it testable without a microphone. */
      handle: function (ev) {
        if (!ev || !ev.type || stopped) return;
        /* THE DOOR. Past this line every name is one this file chose.
         *
         * Before the `stopped` check would be wrong — a session being torn
         * down should not be renaming events — and after the switch would be
         * too late. Identity and allocation-free when there is nothing to map,
         * which is every event on the current stack; null when the provider
         * says the controller must not see this one at all. */
        ev = normalise(ev);
        if (!ev || !ev.type) return;
        switch (ev.type) {
          case 'response.created':
            /* The response being created now is the model's answer to the
               last utterance the server closed. Binding the two here is what
               lets a late transcript for that same utterance be recognised as
               the question already being answered rather than a new one. */
            answeringItemId = committedItemId;
            beginResponse((ev.response && ev.response.id) || ev.response_id);
            break;
          case 'input_audio_buffer.committed':
            // The turn's audio is closed and handed to the model. Its item_id
            // is the identity the transcription will arrive under, whenever
            // the transcriber gets round to it.
            committedItemId = ev.item_id || null;
            // Committed by either side: nothing left for the backstop to do.
            stopBackstop();
            break;
          case 'output_audio_buffer.started':
            (function () {
              var now0 = Date.now();
              /* A mute open across the change of speaker is charged to the
                 utterance it was actually on, then restarted for this one. */
              if (muted) { chargeMute(now0); muteSince = now0; }
              var u0 = utterFor(ev.response_id);
              if (u0 && !u0.audio_started_at) u0.audio_started_at = now0;
              streamingId = ev.response_id;
              if (u0 && muted && muteReason) {
                u0.mute_reasons[muteReason] = (u0.mute_reasons[muteReason] || 0) + 1;
              }
            })();
            if (silencedIds[ev.response_id]) {
              /* A response cancelled on sight has made a sound anyway: the
                 cancel was in flight. NOW it is muted -- and the buffer is
                 cleared -- because now the mute lands on it and not on the
                 answer before it. */
              audio.mute('silenced_started');
              try { send({ type: 'output_audio_buffer.clear' }); } catch (e) {}
              break;
            }
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
          case 'output_audio_buffer.cleared':
            (function () {
              var rid1 = (ev.response && ev.response.id) || ev.response_id;
              var u1 = utterFor(rid1);
              var now1 = Date.now();
              if (!u1) return;
              if (ev.type === 'response.done') {
                var det1 = (ev.response && ev.response.status_details) || {};
                u1.done = true;
                u1.status = (ev.response && ev.response.status) || null;
                u1.status_reason = det1.reason || det1.type || null;
                if (u1.status === 'cancelled') {
                  u1.ended_by = 'cancelled:' + (u1.cancel_reason || 'server');
                } else if (u1.status === 'incomplete') {
                  u1.ended_by = 'incomplete:' + (det1.reason || 'unknown');
                } else if (u1.status === 'failed') {
                  u1.ended_by = 'failed';
                }
              } else {
                /* The sound stopped. If the element was muted at that
                   instant, the driver did not hear the end of it -- and that
                   is the fact this whole ledger exists to record. */
                if (muted && streamingId === rid1) {
                  chargeMute(now1); muteSince = now1;
                  u1.muted_at_end = true;
                }
                if (!u1.audio_ended_at) u1.audio_ended_at = now1;
                if (ev.type === 'output_audio_buffer.cleared' && !u1.ended_by) {
                  u1.ended_by = 'cancelled:' + (u1.cancel_reason || 'cleared');
                }
                if (streamingId === rid1) streamingId = null;
                if (silencedIds[rid1]) {
                  delete silencedIds[rid1];
                  if (!stopped && !speaking) audio.unmute('silenced_ended');
                }
                if (tailTimer && speaking && speaking.responseId === rid1) {
                  clearTimeout(tailTimer); tailTimer = null;
                }
              }
              utteranceEnded(rid1);
            })();
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
            /* GENERATION IS OVER; THE SOUND MAY NOT BE. On WebRTC the mouth
               is held until output_audio_buffer.stopped (or .cleared) says
               the audio is, so a detector firing into the tail meets a
               response to classify and absorb against, instead of a mute with
               no owner. `finishing` still lets the next response take over.
               A response that never started audio has no tail to wait for. */
            (function () {
              var rid2 = (ev.response && ev.response.id) || ev.response_id;
              var u2 = utter[rid2];
              if (holdTail && !sink && ev.type === 'response.done'
                  && speaking && speaking.responseId === rid2
                  && u2 && u2.audio_started_at && !u2.audio_ended_at) {
                speaking.finishing = true;
                armTail(rid2);
                return;
              }
              finishResponse(rid2);
            })();
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
            noteFirstAudio(ev.response_id);
            (function () {
              var u3 = utter[ev.response_id];
              if (u3 && !u3.transcript_done) u3.generated_chars += (ev.delta || '').length;
              // The earliest evidence audio is on its way; on a transport
              // that never sends output_audio_buffer.started, the only one.
              if (u3 && !u3.audio_started_at) {
                u3.audio_started_at = Date.now();
                if (!streamingId) streamingId = ev.response_id;
              }
              /* A response cancelled on sight, speaking anyway. Muted NOW,
                 on its own first words, and not a moment earlier -- see the
                 noise-silence branch of beginResponse for why earlier was
                 the tail of somebody else's answer. */
              if (silencedIds[ev.response_id] && !muted) {
                audio.mute('silenced_started');
                try { send({ type: 'output_audio_buffer.clear' }); } catch (e) {}
              }
            })();
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
                // Her voice is on its way out of the speaker: the only window
                // in which a microphone reading says anything about echo.
                startCensus(ev.response_id);
              }
            }
            break;
          case 'response.output_audio_transcript.done':
            // What the model says it said. The tests compare it with what it
            // was asked to say; in the car it is what the log records.
            (function () {
              var u4 = utter[ev.response_id];
              if (u4 && ev.transcript) {
                u4.generated_chars = ev.transcript.length;
                u4.transcript_done = true;
              }
            })();
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
            // The turn is open. Arm the floor under the detector's tail.
            startBackstop();
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
            transcriptArrived(lastTranscript, ev.item_id || null);
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
        /* THE MOUTH IS ALREADY DICTATING SOMETHING ELSE, and this is the
         * refusal that cost a junction on the drive of 2026-09-17.
         *
         * There is one dictation slot, and until cancelSpeak existed nothing
         * outside this file could give it back. So when the arbiter superseded
         * the depart call with the near call for the same maneuver -- 26 ms
         * apart, the ordinary shape of a route starting -- the depart line's
         * `stop()` did not release the slot, the near line arrived here, was
         * refused `busy`, had no clip of its own to fall back to, and went
         * silent. "Turn left onto Palisades Dr.", 150 m and 4.8 s from the
         * junction, never said.
         *
         * The refusal itself stays: two lines cannot share one mouth. What is
         * new is that it is REPORTED, with the line that is holding the slot
         * named, so a silence on the road is readable afterwards as contention
         * rather than as the mouth having stopped working. */
        if (dictation) {
          counters.dictation_refused++;
          emit('LIVE_DICTATION_REFUSED', {
            text: line, reason: 'busy',
            holder: dictation.text, holder_started: !!dictation.started,
            holder_age_ms: Math.round(Date.now() - dictation.at),
          });
          return Promise.reject(new Error('busy'));
        }
        /* WHICH DICTATION THIS IS, so the caller can take it back.
           The slot is single and global; a token is what lets `stop()` on the
           item that started THIS line cancel it without any chance of
           cancelling the line that replaced it. */
        var token = ++dictationSeq;
        return new Promise(function (resolve, reject) {
          dictation = {
            text: line, resolve: resolve, reject: reject, started: false,
            responseId: null, transcript: '', onStart: opts.onStart, timer: null,
            token: token, at: Date.now(), channel: opts.channel || null,
          };
          /* Handed over synchronously, inside the executor, because the
             arbiter can supersede this line in the same tick it started it --
             which is exactly what a route beginning does. */
          if (typeof opts.onToken === 'function') {
            try { opts.onToken(token); } catch (e) {}
          }
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
            if (dictation && !dictation.responseId) armOrphan(true);
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

      /* GIVE THE MOUTH BACK, NOW — the other half of speak().
       *
       * THE FAULT THIS CLOSES. rio_speak's `stop()` carried a comment saying a
       * dictation in flight is cancelled by the arbiter's own pre-emption of
       * the item that owns it. It is not: the arbiter's `stopCurrent` calls
       * exactly that `stop()`, and `stop()` cancelled only the clip. So a
       * superseded or pre-empted line went on holding the single dictation
       * slot for the rest of its budget -- up to 2500 ms for a depart call --
       * and every line that landed in that window was refused `busy`. One
       * mouth, held by a line nobody was listening for any more.
       *
       * Cancelling is the same pair of moves the budget's own timeout makes,
       * and they split on whether the response exists yet:
       *
       *   BOUND    cancel it BY ID and clear the output buffer. Naming it
       *            matters -- a bare `response.cancel` cancels whatever is
       *            active, which a moment later is the line that replaced
       *            this one.
       *   UNBOUND  the create is still in flight and the response that lands
       *            belongs to nobody. Disowned through the orphan path, so it
       *            is silenced on sight instead of claiming the mouth and
       *            reading out a turn the car has already taken.
       *
       * Either way `finishDictation` frees the slot SYNCHRONOUSLY, which is
       * what lets the arbiter start the replacement line in the same tick --
       * the tick it already starts it in.
       *
       * The token is checked rather than trusted: by the time a slow caller
       * gets round to stopping, the slot may hold a different line, and
       * cancelling that one would turn one missed call into two. */
      cancelSpeak: function (token, reason) {
        if (!dictation) return false;
        if (token && dictation.token !== token) return false;
        var rid = dictation.responseId;
        if (!rid) armOrphan(true);
        try {
          send(rid ? { type: 'response.cancel', response_id: rid }
                   : { type: 'response.cancel' });
        } catch (e) {}
        try { send({ type: 'output_audio_buffer.clear' }); } catch (e) {}
        counters.dictation_cancelled++;
        finishDictation(reason || 'cancelled');
        return true;
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
        audio.mute('session_stopped');
        clearBarge();
        pendingResume = null;
        if (tailTimer) { clearTimeout(tailTimer); tailTimer = null; }
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
          /* WHICH MOUTH AND WHICH WIRE. The sink answers first because it IS
             the mouth when it exists; otherwise the provider record says whose
             wire this is, and it says so because the server told it (see
             forVoiceBackend). A literal here was right while there was one
             wire, and would have quietly reported the wrong vendor in every
             drive log the moment there were two. */
          voice_backend: sink ? 'elevenlabs'
                              : ((provider && provider.name) || 'openai_realtime'),
          speaking_directly: !!(speaking && speaking.direct),
          voice: sink && sink.state ? sink.state() : null,
          generated: generated,
          response_id: speaking ? speaking.responseId : null,
          stopped: stopped,
          last_transcript: lastTranscript,
          // The words that became the CURRENT turn -- which is not the same as
          // the last transcript, because a fragment is a transcript and is not
          // a turn. "the noise did not become the question" is a sentence
          // about these two being different.
          last_turn: lastTurnText,
          noise_pending: noise ? noise.n : 0,
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
  /* IS THE PEER CONNECTION GONE, OR JUST GOING THROUGH A TUNNEL?
   *
   * `connectionState === 'disconnected'` used to be treated as death: one
   * event, and transportLost() tore the whole session down for the rest of
   * the drive. The WebRTC spec says the opposite -- `disconnected` is a state
   * a connection passes THROUGH, "may trigger intermittently and resolve just
   * as spontaneously", and on a phone it does exactly that: a cell handover,
   * a page going to the background and its socket being paused, a few lost
   * consent checks. Only `failed` and `closed` mean it is over.
   *
   * So `disconnected` starts a clock instead of a teardown. Come back to
   * `connected` inside the grace and nothing happened but a line in the log;
   * stay gone past it and the loss is declared with the same reason it
   * always had. The drive of 2026-09-17 cannot say whether this ever fired --
   * peer state was not reported -- which is why every transition is now an
   * event, and why the number that decides it lives in config.py rather than
   * here. */
  function peerWatch(o) {
    o = o || {};
    var graceMs = o.graceMs || 8000;
    var timer = null;
    var last = null;
    var lost = false;
    var setT = o.setTimeout || root.setTimeout;
    var clearT = o.clearTimeout || root.clearTimeout;
    function say(state, extra) {
      if (typeof o.emit !== 'function') return;
      var ev = { type: 'LIVE_PEER_STATE', state: state,
                 ice: o.ice ? o.ice() : null, was: last };
      if (extra) for (var k in extra) ev[k] = extra[k];
      try { o.emit(ev); } catch (e) {}
    }
    function declare(reason) {
      if (lost) return;
      lost = true;
      if (typeof o.onLost === 'function') { try { o.onLost(reason); } catch (e) {} }
    }
    return {
      state: function (state) {
        if (lost) return;
        if (state === 'disconnected') {
          say(state, { grace_ms: graceMs });
          if (!timer) {
            timer = setT(function () {
              timer = null;
              say('disconnected', { grace_ms: graceMs, expired: true });
              declare('peer_disconnected');
            }, graceMs);
          }
        } else if (state === 'failed' || state === 'closed') {
          if (timer) { clearT(timer); timer = null; }
          say(state);
          declare('peer_' + state);
        } else {
          var recovered = !!timer && state === 'connected';
          if (timer) { clearT(timer); timer = null; }
          say(state, recovered ? { recovered: true } : null);
        }
        last = state;
      },
      stop: function () { if (timer) { clearT(timer); timer = null; } lost = true; },
    };
  }

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

  /* ------------------------------------------------------------------------
   * THE CONTROLLER'S POLICY, IN ONE PLACE, BECAUSE THERE ARE NOW TWO WIRES.
   *
   * Every field below is a decision made in config.py and carried here with the
   * session. It used to sit inside connect(), which was fine while connect() was
   * the only way a session came into being -- and stopped being fine the moment
   * the xAI backend arrived with a WebSocket instead of a peer connection.
   *
   * The alternative was a second call site with its own copy of this list, and
   * that is the failure this project has already had: a policy field added to one
   * path and not the other drifts silently, because a missing field is a default,
   * not an error. Two transports, one policy.
   *
   * `w` is the WIRE: the handful of things that genuinely differ between them --
   * how an event is sent, what the mouth is, whether there is a level meter, and
   * which provider record describes the far end.
   * --------------------------------------------------------------------- */
  /* ------------------------------------------------------------------------
   * THE HANDLE THE PANEL HOLDS, and the second thing that had to stop living
   * inside connect() when a second transport arrived.
   *
   * Everything here except the four members in `parts` is a READ of the session
   * payload -- which channels may speak, how long a line has to start, which
   * voice ended up being used. None of it depends on the wire, and a copy of it
   * per transport is the same drift controllerConfig() exists to prevent.
   *
   * parts: { controller, audioState(), resumeAudio(why), stop() }
   * --------------------------------------------------------------------- */
  function sessionHandle(session, parts) {
    var controller = parts.controller;
    var handle = {
      controller: controller,
      session: session,
      /* Dictate a deterministic line in RIO's voice. Warnings,
         turns and health announcements come through here; each of
         them already holds the mouth at its own priority. */
      speak: function (text, o) { return controller.speak(text, o); },
      /* ...and give it back. The arbiter takes the mouth away from a
         line by calling `stop()` on the item that owns it, and until
         this was reachable from there the line went on holding the
         session's single dictation slot to the end of its own budget
         -- so the line that replaced it was refused `busy` and, with
         no clip of its own, went silent. */
      cancelSpeak: function (token, reason) {
        return controller.cancelSpeak(token, reason);
      },
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
      audioState: parts.audioState,
      resumeAudio: parts.resumeAudio,
      stop: function () {
        try { parts.stop(); } catch (e) {}
        if (active === handle) active = null;
      },
    };
    active = handle;
    return handle;
  }

  function controllerConfig(session, w) {
    return {
          arbiter: w.arbiter,
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
          /* THE VENDOR, AS A RECORD RATHER THAN AS KNOWLEDGE.
             This used to be `holdTail: true` under a comment reading "WebRTC
             sends output_audio_buffer.stopped: the mouth waits for it" — true,
             and the only written record of the dependency. The provider now
             answers it, along with every other capability question, so the
             controller holds no vendor names and this call site holds no
             transport assumptions. See static/rio_provider.js.

             The backend travels with the session (config.VOICE_BACKEND, sent
             by mint_client_secret), and `elevenlabs` resolves to the same
             OpenAI event stream it has always been — it changes the mouth, not
             the wire. */
          provider: w.provider,
          /* THE PHONE COLUMN OR THE DESK COLUMN. Both travel with the session
             (config.py decides them, realtime.mint_client_secret sends them)
             and the machine picks its own — see isTouchDevice. The fallback to
             the flat barge_sustain_ms is what an older server sends, and it is
             the desktop behaviour, unchanged. */
          bargeSustainMs: w.barge.sustain_ms || session.barge_sustain_ms,
          bargeOnsetGuardMs: w.barge.onset_guard_ms || 0,
          bargeEchoMarginDb: w.barge.echo_margin_db || 0,
          bargeEchoFloorDb: session.barge_echo_floor_db,
          /* WHICH COLUMN THIS SESSION IS ON, carried so the margin census is
             labelled with the environment it was measured in. A desk reading
             and a car reading are the same quantity from two rooms, and a
             number without its room is not a measurement. */
          device: isTouchDevice() ? 'touch' : 'desktop',
          /* The meter is installed later, when there is a remote track to
             measure; until then the level test has no evidence and the gate
             behaves as it does on a desk. */
          levels: w.levels || function () { return null; },
          maxResumes: session.max_resumes,
          /* How long her own voice may still be in the room after she stops,
             and what counts as hearing herself. config.py decides both; see
             REALTIME_ECHO_* and the iPhone test that produced them. */
          /* The floor under semantic_vad's tail. 0 is off. */
          turnBackstopMs: session.turn_backstop_ms,
          turnBackstopMicDb: session.turn_backstop_mic_db,
          echoTailMs: session.echo_tail_ms,
          echoImpossibleMs: session.echo_impossible_ms,
          echoTextWindowS: session.echo_text_window_s,
          echoTextOverlap: session.echo_text_overlap,
          echoTextMinWords: session.echo_text_min_words,
          echoTextShortMs: session.echo_text_short_ms,
          /* NEWEST WINS. The commands that preempt, how long two fragments may
             be apart and still be one question, and what a continuation looks
             like — all decided in config.py and carried here with the session,
             exactly as the barge and dictation policies are. */
          turnPolicy: session.turn_policy,
          send: w.send,
          tool: function (name, args, controller) {
            // Answered in the page when the page is the source of truth;
            // everything else goes to the server, which holds the camera, the
            // vehicle context and the reasoning model.
            if (LOCAL_TOOLS[name]) {
              try { return Promise.resolve(LOCAL_TOOLS[name](args)); }
              catch (e) { return Promise.resolve({ ok: false, note: 'panel error' }); }
            }
            return fetch(w.url('/realtime/tool'), {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              /* THE HANDLE THAT STOPS IT. One controller per tool call, held by
                 the turn that asked for it; a new driver utterance aborts every
                 controller belonging to an older turn. Aborting the fetch also
                 drops the TCP connection, which is how the server finds out --
                 /realtime/tool watches for it and stops waiting on work nobody
                 is going to hear.

                 On the first real drive these calls ran 40.1, 12.9, 48.5 and
                 27.1 seconds. Nothing could stop one. */
              signal: w.controller() ? w.controller().signal : undefined,
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
                                     spoken: w.transcript() }),
            }).then(function (r) { return r.json(); });
          },
          // Muting rather than pausing: a track is live and a paused element
          // resumes into stale audio, and the sink's fade is undoable for the
          // same reason -- the sustain gate has to be able to change its mind.
          audio: w.audio,
          voice: w.voice,
          liveVoice: session.live_voice || session.voice,
          // The tool list, and the part of it that waits for a precondition.
          // Both come from the server; see mint_client_secret.
          toolSchemas: session.tool_schemas,
          conditionalTools: session.conditional_tools,
          onEvent: w.onEvent,
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

    /* Told which step is running, so the button can say "allow the
       microphone" instead of "stand by" for the one step that is waiting on a
       person. Optional: a caller that does not care passes nothing. */
    var progress = opts.onProgress || null;

    return step('mint', fetch(url('/realtime/session'), { method: 'POST' })
        .then(function (r) { return r.json(); }), progress)
      .then(function (j) {
        if (!j || j.error || !j.client_secret) {
          throw new Error('mint: ' + ((j && j.error) || 'no session'));
        }
        session = j;
        return step('mic',
                    navigator.mediaDevices.getUserMedia({ audio: MIC_CONSTRAINTS }),
                    progress);
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
          /* BUILT ON BOTH COLUMNS. This used to read
           *
           *     if (!barge.echo_margin_db) return;   // desk: nothing to measure
           *
           * and that comment is the reason the desk number could only ever be
           * guessed: the meter was the only thing that could measure a room,
           * and it was not built in the room whose margin is zero. "Nothing to
           * measure" was a statement about the GATE, which is off on a desk
           * and should stay off; it was not true of the measurement.
           *
           * An analyser on the microphone costs a few floats per frame. The
           * gate is still governed by echo_margin_db alone -- see echoShaped()
           * -- so a desk session behaves exactly as it did and now produces
           * the readings the next threshold has to come from. */
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
        /* ---- A DIFFERENT WIRE ENTIRELY -------------------------------------
         *
         * The xAI backend is a WebSocket: no peer connection, no SDP, no remote
         * track, and the audio is rendered in the page instead of by the WebRTC
         * stack. Everything below this point is about the connection that
         * backend does not have, so the branch is here -- before any of it --
         * rather than threaded through it.
         *
         * WHAT IS SHARED IS THE PART THAT MATTERS: the same createController,
         * the same controllerConfig, the same sessionHandle. This file hands
         * those three to the transport and the transport hands back the four
         * members that genuinely differ. Nothing about a vendor's events or its
         * audio format is decided here, and nothing about the drive's policy is
         * decided there.
         *
         * The mic is already open and is passed on: getUserMedia waits on a
         * human and there is no reason for a second prompt. */
        if (session.voice_backend === 'xai_voice') {
          var xs = root.RIO && root.RIO.xaiSession;
          if (!xs || !xs.attach) {
            throw new Error('mint: session asks for xai_voice and '
                            + 'rio_xai_session.js is not loaded');
          }
          /* THE ECHO METER, WHICH THIS WIRE CAN HAVE AFTER ALL.
           *
           * attach() was written with `levels: null` on the reasoning that
           * makeMeter needs a remote MediaStream and this transport has none.
           * Half right: it needs a MIC stream, which we have, and its `out`
           * reading falls back to RIO.output.level() -- which measures the BUS,
           * and her voice on this wire is connected into that bus. So the
           * measurement was available the whole time.
           *
           * The cost of having got that wrong is in the drive of 2026-09-20:
           * eleven barge detections in ninety seconds, three answers cut off as
           * false_barge_in with "no transcript followed", and echo_suppressed at
           * ZERO -- because with no meter the level test has no evidence and
           * every one of them reached the gate. Her own voice out of the phone
           * speaker was being heard as the driver. */
          var xmeter = null;
          try {
            var xo = root.RIO && root.RIO.output;
            var xctx = (xo && xo.context) ? xo.context() : null;
            xmeter = makeMeter(xctx, mic, null);
          } catch (e) { xmeter = null; }
          if (opts.onEvent) {
            try {
              opts.onEvent({ type: 'LIVE_ECHO_METER', ok: !!xmeter,
                             device: touch ? 'touch' : 'desktop',
                             margin_db: barge.echo_margin_db,
                             onset_guard_ms: barge.onset_guard_ms,
                             sustain_ms: barge.sustain_ms,
                             source: 'output_bus' });
            } catch (e) {}
          }
          return xs.attach({
            session: session, mic: mic, url: url, progress: progress,
            onEvent: opts.onEvent, arbiter: arbiter, touch: touch, barge: barge,
            step: step, levels: xmeter,
            createController: createController,
            controllerConfig: controllerConfig,
            sessionHandle: sessionHandle,
          });
        }

        var pc = new RTCPeerConnection();
        var channel = pc.createDataChannel('oai-events');
        mic.getTracks().forEach(function (t) { pc.addTrack(t, mic); });
        /* Set by stop(), read by every listener below: a track that ends and
           an element that pauses because the session was CLOSED are not
           interruptions, and reporting them as such would put a fault in the
           log at the end of every healthy drive. */
        var closed = false;

        /* THE MICROPHONE, WATCHED. iOS mutes a page's capture track when it
           goes to the background, when a call comes in, when Siri takes the
           audio session -- and un-mutes it, usually, on the way back. While it
           is muted she hears nothing, and a driver who talks into a muted
           microphone and gets no answer reports that she stopped talking.
           None of that left a mark before. `ended` is worse: the capture was
           revoked and cannot be re-opened without a tap, so it is reported and
           NOT silently retried. */
        mic.getTracks().forEach(function (t) {
          function micState(state) {
            if (closed || !opts.onEvent) return;
            try {
              opts.onEvent({ type: 'LIVE_MIC_STATE', state: state,
                             muted: !!t.muted, ready_state: t.readyState,
                             hidden: !!(root.document && root.document.hidden) });
            } catch (e) {}
          }
          try {
            t.addEventListener('mute', function () { micState('muted'); });
            t.addEventListener('unmute', function () { micState('unmuted'); });
            t.addEventListener('ended', function () { micState('ended'); });
          } catch (e) {}
        });

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

        /* HER MOUTH, WATCHED, AND RE-OPENED.
         *
         * Under speech-to-speech her voice is this element. iOS pauses a
         * playing media element when the audio session is interrupted -- a
         * call, Siri, another app taking the output -- and Safari does NOT
         * resume it when the interruption ends: the page has to call play()
         * again, and this page never did. So an interruption that lasted
         * three seconds silenced her for the rest of the drive while the
         * session underneath kept generating answers into a paused element.
         *
         * A pause that was not stop() is therefore reported AND answered: one
         * play() a moment later, in case the interruption is already over,
         * and again from resumeAudio() when the page comes back to the front.
         * `playing` says whether either worked. */
        var replayTimer = null;
        function replay(why) {
          if (closed || !element.srcObject) return;
          try {
            var pp = element.play();
            if (pp && pp.catch) {
              pp.catch(function (err) {
                if (closed || !opts.onEvent) return;
                try {
                  opts.onEvent({ type: 'LIVE_AUDIO_INTERRUPTED', state: 'play_refused',
                                 why: why, error: (err && err.name) || String(err) });
                } catch (e) {}
              });
            }
          } catch (e) {}
        }
        function elementEvent(state, why) {
          if (closed || !opts.onEvent) return;
          try {
            opts.onEvent({ type: state === 'playing' ? 'LIVE_AUDIO_RESUMED'
                                                     : 'LIVE_AUDIO_INTERRUPTED',
                           state: state, why: why || null,
                           paused: !!element.paused, muted: !!element.muted,
                           hidden: !!(root.document && root.document.hidden) });
          } catch (e) {}
        }
        try {
          element.addEventListener('pause', function () {
            if (closed || !element.srcObject) return;
            elementEvent('paused', 'element_pause');
            if (replayTimer) root.clearTimeout(replayTimer);
            replayTimer = root.setTimeout(function () {
              replayTimer = null;
              if (element.paused) replay('after_pause');
            }, 250);
          });
          element.addEventListener('playing', function () {
            if (closed || !element.srcObject) return;
            elementEvent('playing');
          });
        } catch (e) {}

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

        var controller = createController(controllerConfig(session, {
          arbiter: arbiter,
          barge: barge,
          levels: function () { return meter ? meter() : null; },
          /* THE VENDOR, AS A RECORD RATHER THAN AS KNOWLEDGE.
             This used to be `holdTail: true` under a comment reading "WebRTC
             sends output_audio_buffer.stopped: the mouth waits for it" — true,
             and the only written record of the dependency. The provider now
             answers it, along with every other capability question, so the
             controller holds no vendor names and this call site holds no
             transport assumptions. See static/rio_provider.js.

             The backend travels with the session (config.VOICE_BACKEND, sent
             by mint_client_secret), and `elevenlabs` resolves to the same
             OpenAI event stream it has always been — it changes the mouth, not
             the wire. */
          provider: (root.RIO && root.RIO.provider)
            ? root.RIO.provider.forVoiceBackend(session.voice_backend
                                                || 'openai_realtime')
            : null,
          send: function (obj) {
            if (channel.readyState === 'open') channel.send(JSON.stringify(obj));
          },
          url: url,
          transcript: controllerTranscript,
          controller: function () { return controller; },
          audio: audioFacade,
          voice: sink,
          onEvent: opts.onEvent,
        }));

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
        /* `disconnected` is a state a connection passes through, not the
           end of one. See peerWatch. */
        var peer = peerWatch({
          graceMs: session.peer_disconnect_grace_ms || 8000,
          ice: function () { try { return pc.iceConnectionState; } catch (e) { return null; } },
          emit: function (ev) { if (opts.onEvent) opts.onEvent(ev); },
          onLost: function (reason) { controller.transportLost(reason); },
        });
        pc.onconnectionstatechange = function () { peer.state(pc.connectionState); };

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

        /* The clip prefix for this voice. The realtime backend's clips are
           the shipped set in /static/audio/, so this changes nothing today --
           it is here so that the page reads the prefix from the session under
           BOTH backends rather than from a literal under one of them. */
        if (session.clip_base && root.RIO && root.RIO.setClipBase) {
          root.RIO.setClipBase(session.clip_base);
        }

        return step('voice', voiceReady, progress)
          .then(function () {
            return step('offer', pc.createOffer().then(function (offer) {
              return pc.setLocalDescription(offer).then(function () { return offer; });
            }), progress);
          })
          .then(function (offer) {
            return step('negotiate',
              fetch(CALLS_URL + '?model=' + encodeURIComponent(session.model), {
                method: 'POST',
                headers: {
                  'Authorization': 'Bearer ' + session.client_secret,
                  'Content-Type': 'application/sdp',
                },
                body: offer.sdp,
              }).then(function (r) {
                if (!r.ok) return r.text().then(function (t) {
                  throw new Error('realtime call ' + r.status + ': '
                                  + t.slice(0, 160));
                });
                return r.text();
              }), progress);
          })
          .then(function (answer) {
            return step('answer',
                        pc.setRemoteDescription({ type: 'answer', sdp: answer }),
                        progress);
          })
          .then(function () {
            return sessionHandle(session, {
              controller: controller,
              /* WHAT THE AUDIO SESSION LOOKS LIKE RIGHT NOW. Read by the
                 page on every visibility change, so the drive log carries
                 the state of the mouth and the microphone at the moment the
                 page went away and the moment it came back -- which is the
                 whole of what "she went quiet when I locked the phone" needs
                 in order to be answered. */
              audioState: function () {
                var t = mic.getTracks()[0] || null;
                return {
                  element_paused: !!element.paused,
                  element_muted: !!element.muted,
                  element_ready: element.readyState,
                  has_remote: !!element.srcObject,
                  mic_muted: t ? !!t.muted : null,
                  mic_state: t ? t.readyState : null,
                  peer: pc.connectionState || null,
                  ice: pc.iceConnectionState || null,
                  channel: channel.readyState || null,
                };
              },
              /* The page is back, or the interruption is over: give the
                 element another play() and wake the shared bus. Idempotent
                 and cheap, so it is called on every return to the front. */
              resumeAudio: function (why) {
                replay(why || 'resume');
                try {
                  var o = root.RIO && root.RIO.output;
                  if (o && o.context) {
                    var c = o.context();
                    if (c && c.state === 'suspended' && c.resume) c.resume().catch(function () {});
                  }
                } catch (e) {}
              },
              stop: function () {
                closed = true;
                if (replayTimer) { root.clearTimeout(replayTimer); replayTimer = null; }
                peer.stop();
                controller.stop();
                if (sink) { try { sink.close(); } catch (e) {} }
                try { channel.close(); } catch (e) {}
                try { pc.close(); } catch (e) {}
                mic.getTracks().forEach(function (t) { t.stop(); });
                element.srcObject = null;
              },
            });
          });
      });
  }

  /* The one live session, if there is one. Deterministic speech asks for it
     by name rather than being handed it: a warning fires from a code path that
     has never heard of the conversation panel and must not have to. */
  var active = null;

  /* Test seams. The connect path cannot be driven end to end without a
     browser, a microphone and a network -- but the thing that actually failed
     is the ABSENCE of deadlines, and that is testable on its own. */
  function _connectBudgets() {
    var out = {};
    for (var k in CONNECT_BUDGET_MS) out[k] = CONNECT_BUDGET_MS[k];
    return out;
  }

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
    /* THE CONNECT DEADLINES, exposed so the node suite can drive them.
       The connect path itself needs a browser, a microphone and a network --
       but what actually failed on 2026-09-10 was the ABSENCE of these, and
       that is testable without any of the three. */
    _connectBudgets: _connectBudgets,
    _step: step,
    // The peer-state clock, for tools/page_background_selftest.js.
    _peerWatch: peerWatch,
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { createController: createController, navStatus: navStatus,
                       peerWatch: peerWatch,
                       // The connect deadlines, for tools/realtime_selftest.js.
                       // See the export block above for why these are seams.
                       _connectBudgets: _connectBudgets, _step: step,
                       /* The two pieces both transports share. Exported so
                          tools/xai_session_selftest.js can attach the second
                          wire to the REAL policy and the REAL handle rather than
                          to a pair of fakes that would agree with anything. */
                       _controllerConfig: controllerConfig,
                       _sessionHandle: sessionHandle,
                       navDirections: navDirections,
                       noteFix: noteFix, currentFix: currentFix,
                       startNavigation: startNavigation,
                       localTools: LOCAL_TOOLS,
                       micConstraints: MIC_CONSTRAINTS };
  }
})(typeof window !== 'undefined' ? window : globalThis);
