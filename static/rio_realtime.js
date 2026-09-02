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

  var LOCAL_TOOLS = { nav_status: navStatus, nav_directions: navDirections,
                      start_navigation: startNavigation };

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
       audio: { mute(), unmute() },  the element RIO's voice comes out of
       onEvent(ev)                   observability
     }
     --------------------------------------------------------------------- */
  function createController(cfg) {
    cfg = cfg || {};
    var arbiter = cfg.arbiter;
    var send = cfg.send || function () {};
    var runTool = cfg.tool || function () { return Promise.resolve({ ok: false }); };
    var audio = cfg.audio || { mute: function () {}, unmute: function () {} };
    var listeners = [];
    if (typeof cfg.onEvent === 'function') listeners.push(cfg.onEvent);

    var speaking = null;      // { responseId, resolve, cancelled }
    var stopped = false;
    var counters = { responses: 0, interrupted: 0, barge_ins: 0,
                     tool_calls: 0, tool_failures: 0,
                     dictated: 0, dictation_failures: 0 };
    var lastTranscript = '';
    /* A dictated line — a warning, a turn, a health announcement — in flight.
       It is NOT a conversation response and must never be treated as one: it
       does not claim the mouth (its caller already holds it, at its own
       priority) and it does not enter the conversation history. */
    var dictation = null;
    var verbatimInstruction = cfg.verbatimInstruction ||
      'Read the text below out loud, exactly as written, word for word. ' +
      'Add nothing. Remove nothing. Do not rephrase.\n\nTEXT:\n';
    var speakTimeoutMs = cfg.speakTimeoutMs || 700;

    function emit(type, payload) {
      var ev = payload || {};
      ev.type = type;
      for (var i = 0; i < listeners.length; i++) {
        try { listeners[i](ev); } catch (e) { /* never let a listener mute RIO */ }
      }
    }

    /* RIO has started saying something. Claim the mouth for it.
     *
     * One arbiter item per RESPONSE, not per session: a session lasts a drive,
     * and an item that lasts a drive would either block every warning or be
     * pre-empted once and never recover. */
    function beginResponse(responseId) {
      if (stopped) return;
      /* The first response after a dictation was sent IS that dictation. It
         gets bound here rather than claiming the mouth: a warning arriving as
         a conversation-priority item would be a warning that yields to
         navigation, which is upside down. */
      if (dictation && !dictation.responseId) {
        dictation.responseId = responseId;
        return;
      }
      if (speaking) return;
      var entry = { responseId: responseId, resolve: null, cancelled: false };
      speaking = entry;
      counters.responses++;
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
           warning and reappear halfway through a sentence. */
        stop: function () {
          entry.cancelled = true;
          audio.mute();
          try { send({ type: 'response.cancel' }); } catch (e) {}
          try { send({ type: 'output_audio_buffer.clear' }); } catch (e) {}
        },
        onDone: function (reason) {
          if (speaking === entry) speaking = null;
          if (reason !== 'spoken') {
            counters.interrupted++;
            audio.mute();
          }
          emit('LIVE_RESPONSE_END', { response_id: responseId, reason: reason });
        },
      });
      emit('LIVE_RESPONSE_START', { response_id: responseId });
    }

    function endResponse(responseId) {
      if (!speaking) return;
      if (responseId && speaking.responseId && speaking.responseId !== responseId) return;
      var entry = speaking;
      speaking = null;
      if (entry.resolve) entry.resolve();          // the arbiter marks it spoken
    }

    /* The driver started talking. The server interrupts the response itself —
       that is what interrupt_response is for — but the audio already in the
       element keeps playing for a moment, and a machine that talks over you
       after you have started is the single most irritating thing an assistant
       can do. So mute locally on the first sign of speech, immediately. */
    function bargeIn() {
      counters.barge_ins++;
      audio.mute();
      if (speaking) {
        emit('LIVE_BARGE_IN', { response_id: speaking.responseId });
        endResponse(speaking.responseId);
      } else {
        emit('LIVE_BARGE_IN', {});
      }
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
    }

    function toolCall(name, callId, argsJson) {
      counters.tool_calls++;
      emit('LIVE_TOOL_CALL', { tool: name, call_id: callId });
      var args = argsJson;
      if (typeof argsJson === 'string') {
        try { args = JSON.parse(argsJson || '{}'); } catch (e) { args = {}; }
      }
      return Promise.resolve()
        .then(function () { return runTool(name, args); })
        .catch(function (e) {
          // The server is unreachable, or refused. Not an error the driver
          // hears about: RIO is told the tool did not work and carries on.
          return { ok: false, note: 'unreachable' };
        })
        .then(function (result) {
          result = result || { ok: false, note: 'no result' };
          if (!result.ok) counters.tool_failures++;
          if (stopped) return result;
          send({
            type: 'conversation.item.create',
            item: {
              type: 'function_call_output',
              call_id: callId,
              output: JSON.stringify(result),
            },
          });
          // Ask for the spoken answer. Without this the model has the result
          // and no reason to say anything about it.
          send({ type: 'response.create' });
          emit('LIVE_TOOL_RESULT', { tool: name, call_id: callId,
                                     ok: !!result.ok, took_ms: result.took_ms || null,
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
            endResponse((ev.response && ev.response.id) || ev.response_id);
            break;
          case 'response.output_audio_transcript.done':
            // What the model says it said. The tests compare it with what it
            // was asked to say; in the car it is what the log records.
            if (dictation && dictation.responseId === ev.response_id) {
              dictation.transcript = ev.transcript || '';
            }
            break;
          case 'input_audio_buffer.speech_started':
            bargeIn();
            break;
          case 'response.function_call_arguments.done':
            toolCall(ev.name, ev.call_id, ev.arguments);
            break;
          case 'conversation.item.input_audio_transcription.completed':
            lastTranscript = ev.transcript || '';
            emit('LIVE_TRANSCRIPT', { transcript: lastTranscript, role: 'driver' });
            break;
          case 'error':
            emit('LIVE_ERROR', { error: (ev.error && ev.error.message) || 'unknown' });
            break;
          default:
            break;
        }
      },

      onEvent: function (fn) { if (typeof fn === 'function') listeners.push(fn); },

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
        if (dictation) return Promise.reject(new Error('busy'));
        return new Promise(function (resolve, reject) {
          dictation = {
            text: line, resolve: resolve, reject: reject, started: false,
            responseId: null, transcript: '', onStart: opts.onStart, timer: null,
          };
          dictation.timer = setTimeout(function () {
            // Never heard it start. Give up on the live voice for this line;
            // a warning that arrives late has stopped being a warning.
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

      /* Ending the session releases the mouth: an item left claimed would
         block every conversational reply for the rest of the drive. */
      stop: function () {
        stopped = true;
        audio.mute();
        if (dictation) finishDictation('session_stopped');
        if (speaking) endResponse(speaking.responseId);
      },

      state: function () {
        return {
          speaking: !!speaking,
          dictating: !!dictation,
          response_id: speaking ? speaking.responseId : null,
          stopped: stopped,
          last_transcript: lastTranscript,
          counters: counters,
        };
      },
    };
  }

  /* ---------------------------------------------------------------------
     The wiring. Everything below needs a browser.
     --------------------------------------------------------------------- */
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
        // Echo cancellation is not optional in a car: RIO's own voice comes
        // out of the same box the microphone is in, and without it she hears
        // herself and interrupts herself.
        return navigator.mediaDevices.getUserMedia({
          audio: { echoCancellation: true, noiseSuppression: true,
                   autoGainControl: true },
        });
      })
      .then(function (mic) {
        var pc = new RTCPeerConnection();
        var channel = pc.createDataChannel('oai-events');
        mic.getTracks().forEach(function (t) { pc.addTrack(t, mic); });

        pc.ontrack = function (e) {
          element.srcObject = e.streams[0];
          var p = element.play();
          if (p && p.catch) p.catch(function () {});
        };

        var controller = createController({
          arbiter: arbiter,
          // Dictation policy comes from the server with the session, so the
          // browser holds no second copy of the verbatim instruction to drift
          // from the one the tests check.
          verbatimInstruction: session.verbatim_instruction,
          speakTimeoutMs: session.speak_timeout_ms,
          send: function (obj) {
            if (channel.readyState === 'open') channel.send(JSON.stringify(obj));
          },
          tool: function (name, args) {
            // Answered in the page when the page is the source of truth;
            // everything else goes to the server, which holds the camera, the
            // vehicle context and the reasoning model.
            if (LOCAL_TOOLS[name]) {
              try { return Promise.resolve(LOCAL_TOOLS[name](args)); }
              catch (e) { return Promise.resolve({ ok: false, note: 'panel error' }); }
            }
            return fetch(url('/realtime/tool'), {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              // `where` is the car's own fix. Only find_places reads it, but it
              // is attached to every call rather than to one, so a tool that
              // needs it later does not have to re-plumb this.
              body: JSON.stringify({ name: name, arguments: args,
                                     where: currentFix() }),
            }).then(function (r) { return r.json(); });
          },
          // Muting the ELEMENT rather than pausing it: the track is live, and a
          // paused element resumes into stale audio.
          audio: {
            mute: function () { element.muted = true; },
            unmute: function () { element.muted = false; },
          },
          onEvent: opts.onEvent,
        });

        channel.onmessage = function (e) {
          var ev;
          try { ev = JSON.parse(e.data); } catch (err) { return; }
          controller.handle(ev);
        };

        return pc.createOffer()
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
              stop: function () {
                controller.stop();
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
    active: function () { return active; },
    /* Tests and the panel: pretend a session is open, or that none is. */
    _setActive: function (h) { active = h; },
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { createController: createController, navStatus: navStatus,
                       navDirections: navDirections,
                       noteFix: noteFix, currentFix: currentFix,
                       startNavigation: startNavigation,
                       localTools: LOCAL_TOOLS };
  }
})(typeof window !== 'undefined' ? window : globalThis);
