/* rio_xai_session.js — the live conversation over a WebSocket, in the browser.
 *
 * WHAT THIS IS NOT: a second controller. Every decision about a live turn stays
 * in createController (rio_realtime.js) -- the arbiter, the supersede, the barge
 * gate, the echo gate, the resume budget, the mute ledger. This file is the
 * TRANSPORT: it opens a socket, moves audio in both directions, and hands events
 * through the seam in rio_provider so the controller never sees a vendor's name.
 *
 * rio_live.js took the other road for gpt-live-1 -- a smaller controller that
 * re-implements the echo gate and drops the rest -- and that was right there,
 * because the events those guards hang off genuinely do not exist on that model.
 * Here they mostly DO, under the same names, so forking the controller would fork
 * every guard to gain nothing.
 *
 * WHAT THIS TRANSPORT OWNS THAT WEBRTC DID NOT
 * --------------------------------------------
 * On WebRTC the browser is handed a remote track and the API says when her audio
 * stops. Here we render it, so:
 *
 *   THE TAIL IS OURS. response.done is the end of GENERATION; seconds of sound
 *   can still be queued. rio_playout computes the end from the bytes and the
 *   controller's holdTail waits for THAT. Handing the mouth back at response.done
 *   is commit a05d233 -- "every mute in that window had no owner" -- and it is
 *   what section Z of tools/playout_selftest.js breaks on purpose.
 *
 *   THE GATE IS OURS. Their own docs: a tool result followed immediately by
 *   response.create makes the server start the next turn "even if the client is
 *   still playing audio from the previous turn". That warning describes
 *   rio_realtime.js:3265->3276 exactly, and it was harmless only while the server
 *   owned playout. Same queue answers both, because both are one question: how
 *   much sound is still to come.
 *
 *   output_audio_buffer.clear IS OURS. Measured, the wire answers "Invalid event
 *   received" -- an error, on the barge-in path, at the moment the driver is
 *   trying to interrupt. The provider's outbound map turns it into a local flush,
 *   which is better than the event it replaces: no round trip to a server that
 *   then has to tell us the sound stopped.
 *
 * FOUR THINGS MEASURED BEFORE THIS WAS WRITTEN, honoured rather than rediscovered
 * ------------------------------------------------------------------------------
 *   1. session.update MUST carry output_modalities and an output voice. Without
 *      either, the session accepts every field, echoes the config back, runs VAD,
 *      commits audio -- and produces NOTHING, with no error. It is assembled
 *      server-side (xai_voice.session_policy) and sent verbatim; this file does
 *      not build one.
 *   2. server_vad has to HEAR the end of an utterance. Audio that stops dead on
 *      the last syllable leaves the turn open forever. A live microphone sends
 *      silence naturally, which is why the stream is never paused between turns --
 *      see the note on `pumping` below. It matters for injected audio, which is
 *      what the silence_tail_ms policy is for.
 *   3. .completed arrives THREE TIMES per utterance, same id, same text. The
 *      controller's item_id binding happens to suppress the repeats; happening to
 *      is not a design, so they are dropped here by id before it ever sees them.
 *   4. An explicit input_audio_buffer.commit suppresses the transcript entirely
 *      when server_vad is on. The outbound map drops it; this file never sends one.
 *
 * AND A SESSION THAT GOES QUIET IS A FAULT, NOT A LULL. Three wrong conclusions
 * came from a socket that connected, ran VAD, fired speech_started and committed,
 * and then never spoke. `silence` below is that condition, named, surfaced and
 * reported -- because the one thing this failure has never done is look like one.
 */
'use strict';
(function (root) {

  var RATE = 24000;
  var FRAME_MS = 40;

  function b64ToBytes(b64) {
    var bin = atob(b64);
    var out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  /* THE ANTI-ALIAS FILTER, and why it is a real one rather than an average.
   *
   * Taking every Nth sample folds everything above the far end's Nyquist back
   * down INTO the speech band: at 48 kHz in and 24 kHz out, a 15 kHz sound
   * arrives as a 9 kHz one, sitting on top of the consonants a transcriber
   * works from. Two cheaper filters were tried against that exact tone in
   * tools/mic_rate_selftest.js and both are written down here because "an
   * average is surely enough" is the reasoning that has to lose an argument to
   * a number rather than to taste:
   *
   *     every Nth sample        the fault, for reference
   *     box over one period     67% of the tone came back at 9 kHz
   *     triangle over two       30%
   *     this                    see the suite; it asserts under 1%
   *
   * A windowed sinc is the textbook answer and costs ~16 multiplies per output
   * sample, which at 24 kHz is a rounding error next to the FFT the echo meter
   * already runs on the same thread.
   *
   * `TAPS_PER_SIDE` is in output samples: the kernel is that many zero
   * crossings either side of centre, scaled up by `ratio` to reach the same
   * distance in input samples. Four is the usual place to stop for speech --
   * the stopband is already deep and every extra tap is latency.
   */
  var TAPS_PER_SIDE = 4;

  function sinc(x) {
    if (x === 0) return 1;
    var pix = Math.PI * x;
    return Math.sin(pix) / pix;
  }

  /* One output frame at RATE, read out of a buffer captured at some other
     rate. -> {out, pos} or null when there is not enough audio yet.

     `pos` is FRACTIONAL and is handed back, because the ratio is not generally
     an integer -- 44100 Hz gives 1.8375 -- and a remainder dropped at the end
     of each callback is a sample of drift per frame.

     The kernel is centred `half` samples INTO the buffer rather than on `pos`
     itself, so it only ever reads forward and the caller needs no history
     margin. That is a fixed delay of TAPS_PER_SIDE output samples -- 167 µs --
     which is not a number anyone can hear or name.

     Module level, and on the export seam below, so the node suite can run the
     resampler the browser runs rather than a second copy of the arithmetic. */
  function frameAtRate(buf, pos, frames, ratio) {
    if (!(ratio > 0) || !isFinite(ratio)) ratio = 1;
    // Cut off just below the far end's Nyquist, expressed against the INPUT
    // rate. Never above 0.45, so a context already at RATE is not filtered
    // against a frequency higher than it has.
    var fc = Math.min(0.45, 0.45 / ratio);
    var half = TAPS_PER_SIDE / (2 * fc);          // kernel half-width, input samples
    if (Math.ceil(pos + (frames - 1) * ratio + 2 * half) + 1 > buf.length) return null;
    var out = new Float32Array(frames);
    var p = pos;
    for (var i = 0; i < frames; i++) {
      var c = p + half;
      var a = Math.floor(c - half);
      var b = Math.ceil(c + half);
      if (a < 0) a = 0;
      if (b > buf.length) b = buf.length;
      var sum = 0, wsum = 0;
      for (var j = a; j < b; j++) {
        var d = (j + 0.5) - c;
        if (d <= -half || d >= half) continue;
        // Hamming over the kernel's own width, then the band-limiting sinc.
        var w = (0.54 + 0.46 * Math.cos(Math.PI * d / half)) * sinc(2 * fc * d);
        sum += buf[j] * w;
        wsum += w;
      }
      out[i] = wsum > 0 ? sum / wsum : 0;
      p += ratio;
    }
    return { out: out, pos: p };
  }

  function bytesToB64(bytes) {
    var s = '';
    for (var i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
    return btoa(s);
  }

  /* PCM16 little-endian -> Float32 in [-1, 1], which is what Web Audio wants. */
  function pcm16ToFloat(bytes) {
    var n = Math.floor(bytes.length / 2);
    var out = new Float32Array(n);
    var dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (var i = 0; i < n; i++) out[i] = dv.getInt16(i * 2, true) / 32768;
    return out;
  }

  function floatToPcm16(f32) {
    var out = new Uint8Array(f32.length * 2);
    var dv = new DataView(out.buffer);
    for (var i = 0; i < f32.length; i++) {
      var v = Math.max(-1, Math.min(1, f32[i]));
      dv.setInt16(i * 2, v < 0 ? v * 32768 : v * 32767, true);
    }
    return out;
  }

  /* ---------------------------------------------------------------------
     THE SOUND, OUT.

     Scheduled against the AudioContext's own clock rather than played as it
     arrives: deltas do not arrive at the rate they play, and "start it now"
     produces either a gap or an overlap on every chunk.

     This is a SECOND ledger of the same fact -- rio_playout computes when the
     sound ends from the byte counts, and `nextAt` here is where the next buffer
     is actually placed. They are the same arithmetic on the same clock and they
     must agree, so the selftest compares them rather than trusting that they do:
     if they ever drift, the tail the controller waits on is wrong by the drift
     and nothing else in the system would notice.
     --------------------------------------------------------------------- */
  function createSpeaker(ctx, destination) {
    var nextAt = 0;
    var live = [];
    return {
      /* -> the time this chunk will finish, in ctx clock seconds. */
      play: function (bytes) {
        var f32 = pcm16ToFloat(bytes);
        if (!f32.length) return nextAt;
        var buf = ctx.createBuffer(1, f32.length, RATE);
        buf.copyToChannel ? buf.copyToChannel(f32, 0)
                          : buf.getChannelData(0).set(f32);
        var src = ctx.createBufferSource();
        src.buffer = buf;
        src.connect(destination);
        var now = ctx.currentTime;
        var at = nextAt > now ? nextAt : now;
        src.start(at);
        nextAt = at + buf.duration;
        live.push(src);
        src.onended = function () {
          var i = live.indexOf(src);
          if (i >= 0) live.splice(i, 1);
        };
        return nextAt;
      },
      /* THE BARGE-IN. Stops what is already scheduled, which is the half a
         server-side buffer clear could never do for us as quickly. */
      stop: function () {
        live.slice().forEach(function (s) {
          try { s.stop(); } catch (e) {}
        });
        live.length = 0;
        nextAt = 0;
      },
      queuedUntil: function () { return nextAt; },
    };
  }

  /* ---------------------------------------------------------------------
     THE SESSION
     --------------------------------------------------------------------- */

  /* opts:
       mint          what POST /realtime/session returned (ws_url, ws_subprotocol,
                     session, silence_tail_ms, tool_schemas, ...). Assembled on the
                     server; this file adds nothing to it.
       controller    a createController, already built with provider + holdTail
       ctx           AudioContext
       destination   where her voice goes (RIO.output's node, or ctx.destination)
       mic           MediaStream from getUserMedia, or null for a silent session
       onEvent       panel events
       WebSocketImpl / now — seams for the node suite
  */
  function open(opts) {
    opts = opts || {};
    var mint = opts.mint || {};
    var controller = opts.controller;
    var provider = opts.provider
      || (root.RIO && root.RIO.provider
          && root.RIO.provider.create('xai_voice'));
    /* ONE CLOCK FOR THE LEDGER AND THE SCHEDULE. The speaker starts buffers
       against ctx.currentTime; the queue computes the end of the sound from the
       bytes. If those two read different clocks they drift apart and the tail is
       wrong by the drift -- so when there is a context, its clock is the clock,
       and the selftest asserts the two agree on when the sound ends. */
    var now = opts.now
      || (opts.ctx ? function () { return opts.ctx.currentTime; }
                   : function () { return Date.now() / 1000; });
    var WS = opts.WebSocketImpl || root.WebSocket;
    var emit = opts.onEvent || function () {};

    /* THE QUEUE IS BUILT HERE, ON THAT CLOCK, and that is the whole reason it is
       not passed in. A caller that constructed it with Date.now() while the
       speaker schedules against ctx.currentTime would produce a tail wrong by the
       offset between two clocks -- a mistake with no symptom until a drive, and
       one this file cannot check for. It is reachable as `api.playout` for the
       panel and the suite; an injected one is still honoured, because the drive
       harness has its own. */
    var playoutLib = opts.playoutLib
      || (root.RIO && root.RIO.playout) || null;
    var playout = opts.playout
      || (playoutLib ? playoutLib.createPlayout({
            sampleRate: RATE, now: now, channels: 1,
            tailGraceS: opts.tailGraceS,
          }) : null);

    var ws = null;
    var speaker = null;
    var closed = false;
    /* THE CONVERSATION'S OWN ID, so a drop is recoverable rather than merely
       reported. Measured (tools/xai_resumption_probe.py): the session assigns
       this in a conversation.created event, and a new socket opened with
       `?conversation_id=<id>` comes back with the conversation intact -- the same
       id, and she can answer a question about something said before the drop.
       Only the query string works; the id in session.update is accepted and
       silently starts a new conversation.

       RECORDED, NOT YET USED. Reconnecting with it overlaps the controller's own
       resume, which carries what the driver HEARD rather than what was generated,
       and two resume mechanisms disagreeing about what she said is worse than
       one. Keeping the id here means that decision is a policy change and not
       another round of probing. */
    var conversationId = null;
    /* WHICH UTTERANCES HAVE BEEN SEEN, so the three .completed events for one
       utterance become one. By id, not by count: counting would break the day the
       repeat count changes, and the id is what makes them provably the same
       utterance rather than three that happen to match. */
    var transcriptSeen = Object.create(null);
    /* PENDING TOOL RESULTS, held until the sound is over. See flushToolResults. */
    var pending = [];
    var stats = {
      audio_in_frames: 0, audio_out_bytes: 0, transcripts: 0,
      /* WHAT THE MICROPHONE IS ACTUALLY AT, and what had to be done about it.
         Declared here so the shape is stable and so a drive can be asked the
         question afterwards -- the 2026-09-21 fault was a mic at 48000 feeding
         a session opened at 24000, and nothing anywhere recorded either number.
         A ratio of 1 means the context happened to match. */
      audio_in_rate: null, audio_in_ratio: null,
      transcript_repeats_dropped: 0, local_flushes: 0, gated_requests: 0,
      responses: 0, spoke: 0, silent_responses: 0, gate_timeouts: 0,
      /* Cancelled, failed or incomplete responses that made no sound. Counted
         apart from silent_responses because they are not a fault -- a noise reply
         cancelled on sight is the noise gate working -- and because a drive log
         that cannot tell the two apart is what produced a false "degraded". */
      silent_by_design: 0,
      /* ...of which THIS many were a tool call, counted apart so "silent by
         design" can be read rather than trusted. A drive whose every response
         is a silent tool call and whose driver heard nothing is a real fault,
         and it would hide inside the aggregate. */
      silent_tool_calls: 0,
      // Set while a response is emitting a function call; cleared at
      // response.done beside spoke_this_response.
      tool_this_response: false,
    };
    /* SILENT-SESSION DETECTION. `expecting` is set when a response opens and
       cleared by its first audio; a response that closes with it still set is one
       the driver heard nothing from. */
    var expecting = null;
    var silence = { since: null, responses: 0, degraded: false };

    function send(obj) {
      if (closed || !ws) return;
      /* OUTBOUND THROUGH THE SEAM. One of these is an error on this wire and one
         must never be sent at all; neither fact belongs in the controller. */
      var out = provider ? provider.normaliseOut(obj) : obj;
      if (out === null) return;                 // dropped, deliberately
      if (out && out.local) {
        handleLocal(out.local);
        return;
      }
      /* A REQUEST FOR SPEECH WAITS FOR SILENCE. Everything else goes at once:
         a cancel that waited would be a barge-in that did not work. */
      if (out.type === 'response.create' && playout && !playout.idle()) {
        stats.gated_requests++;
        var wait = playout.untilIdle();
        emit({ type: 'XAI_REQUEST_GATED',
               wait_s: wait === Infinity ? null : wait });
        pending.push({ ev: out, at: now() });
        return;
      }
      try { ws.send(JSON.stringify(out)); } catch (e) {}
    }

    function handleLocal(what) {
      if (what === 'output_audio_buffer.clear') {
        stats.local_flushes++;
        if (speaker) speaker.stop();
        if (playout) playout.flush(null, 'barge_in');
        /* AND THE CONTROLLER IS TOLD THE SOUND STOPPED, in its own vocabulary.
           On WebRTC the server sent output_audio_buffer.cleared back and the mute
           ledger closed the span on it; nothing will send it here, so the event is
           synthesised at the only moment it is true. */
        deliver({ type: 'output_audio_buffer.cleared',
                  response_id: expecting });
      }
    }

    /* Anything held back by the gate, once there is silence to send it into --
       AND A CEILING ON THE HOLD, because the gate must not be able to mute her
       for the rest of the drive.
       
       The queue's own arithmetic bounds the wait in every ordinary case: once
       generation is done it knows to the millisecond when the sound ends. What it
       cannot bound is a response that never closes -- a response.done that does
       not arrive, a cancel the server never confirms -- and then `idle()` is false
       forever and every request for speech queues behind it. That is a mute car,
       which is worse than the overlap the gate exists to prevent: overlapping
       audio is two seconds of mess, and silence is the whole drive. So the hold
       has a ceiling, and passing it is reported rather than quietly survived. */
    function flushPending() {
      if (!pending.length) return;
      var overdue = pending[0] && (now() - pending[0].at) * 1000 > gateMaxHoldMs;
      if (playout && !playout.idle() && !overdue) return;
      if (overdue) {
        stats.gate_timeouts++;
        emit({ type: 'XAI_GATE_TIMEOUT',
               held_ms: Math.round((now() - pending[0].at) * 1000),
               playout: playout ? playout.state() : null });
      }
      var go = pending.slice();
      pending.length = 0;
      go.forEach(function (o) {
        try { ws.send(JSON.stringify(o.ev)); } catch (e) {}
      });
    }

    function deliver(ev) {
      if (!controller) return;
      try { controller.handle(ev); } catch (e) {}
    }

    /* ---- THE END OF THE SOUND, TOLD TO THE CONTROLLER ---------------------
     *
     * Nothing on this wire will ever say her audio finished: on WebRTC the API
     * sent output_audio_buffer.stopped and the controller's holdTail waits for
     * exactly that before handing the mouth back. Here the queue is the only
     * thing that knows, so it is subscribed to and the event is synthesised at
     * the one moment it is true.
     *
     * WITHOUT THIS the mouth is never handed back at all -- not late, never --
     * and the first turn of the drive is the last one. It is the reason the queue
     * takes a list of listeners rather than one callback: the drive harness and
     * the selftest were already using the constructor slot.
     *
     * A CANCELLED end is not reported here. handleLocal has already sent
     * output_audio_buffer.cleared for it, which is what WebRTC sent for a clear
     * and is a different fact from the sound finishing: the mute ledger charges
     * them differently (`ended_by`), and sending both would make every barge-in
     * look like a completed utterance as well. */
    if (playout && playout.onAudioEnd) {
      playout.onAudioEnd(function (rid, detail) {
        if (!detail || detail.reason !== 'drained') return;
        deliver({ type: 'output_audio_buffer.stopped', response_id: rid });
      });
    }

    /* ---- THE ONE THING THAT HAS TO HAPPEN WITH NO EVENT ARRIVING ----------
     *
     * Both of the facts this transport owns become true in SILENCE. The sound
     * finishes when the clock passes the end of the last chunk; a held
     * response.create becomes sendable at the same instant. Neither is a
     * message, and everything else in this file is driven by one.
     *
     * The first version of this file flushed the gate only at the end of
     * onMessage, which deadlocks in the exact case the gate exists for: the tool
     * result is in, the server has nothing more to say, and the request waiting
     * for silence is never sent because silence does not arrive as an event. The
     * node suite had to call _flushPending() by hand to see a turn complete --
     * which was the tell, and is now asserted without it. */
    var ticker = null;
    var tickMs = opts.tickMs || 25;
    /* How long a request for speech may wait for silence before it goes anyway.
       Four seconds is longer than any answer's tail (a 30-second answer is 30
       seconds of queue, but its END is known) and short enough that a driver
       notices one pause rather than a drive of them. */
    var gateMaxHoldMs = opts.gateMaxHoldMs === undefined ? 4000
                                                         : opts.gateMaxHoldMs;

    function tick() {
      if (closed) return;
      if (playout) playout.tick();
      flushPending();
    }

    function startTicker() {
      if (ticker) return;
      var set = opts.setIntervalImpl || root.setInterval;
      if (set) ticker = set(tick, tickMs);
    }

    function stopTicker() {
      var clr = opts.clearIntervalImpl || root.clearInterval;
      if (ticker && clr) clr(ticker);
      ticker = null;
    }

    /* ---- inbound ---------------------------------------------------------- */
    function onMessage(raw) {
      var ev;
      try { ev = JSON.parse(raw); } catch (e) { return; }
      var t = ev && ev.type;

      if (t === 'ping') return;

      if (t === 'conversation.created') {
        var conv = ev.conversation || {};
        conversationId = conv.id || ev.conversation_id || conversationId;
        emit({ type: 'XAI_CONVERSATION', conversation_id: conversationId,
               resumed: !!opts.resumeConversationId });
      }

      if (t === 'session.updated') {
        emit({ type: 'XAI_SESSION_UP',
               model: (ev.session || {}).model || null });
      }

      if (t === 'response.created') {
        stats.responses++;
        expecting = (ev.response && ev.response.id) || ev.response_id || 'r';
        silence.since = now();
      }

      if (t === 'response.output_audio.delta' && ev.delta) {
        var bytes = b64ToBytes(ev.delta);
        stats.audio_out_bytes += bytes.length;
        if (!stats.spoke_this_response) {
          stats.spoke++;
          stats.spoke_this_response = true;
        }
        expectingHeard();
        /* THE LEDGER FIRST, THEN THE SOUND. The queue is what the controller's
           tail waits on, so it must know about this chunk before anything can
           ask whether the audio is over. */
        if (playout) playout.push(expecting, bytes.length);
        if (speaker) speaker.play(bytes);
        /* AND THE CONTROLLER IS TOLD AUDIO STARTED, in its own vocabulary: on
           WebRTC output_audio_buffer.started said so, and nothing here will. */
        if (!startedSent) {
          startedSent = true;
          deliver({ type: 'output_audio_buffer.started',
                    response_id: expecting });
        }
        return;                       // not an event the controller switches on
      }

      if (t === 'conversation.item.input_audio_transcription.completed') {
        var iid = ev.item_id;
        if (iid && transcriptSeen[iid]) {
          stats.transcript_repeats_dropped++;
          return;                     // the second and third of three
        }
        if (iid) transcriptSeen[iid] = true;
        stats.transcripts++;
      }

      /* A RESPONSE THAT IS A TOOL CALL IS SUPPOSED TO MAKE NO SOUND.
         Tracked here rather than read off response.done, because the shape of
         `response.output` is a vendor's business and the seam exists so this
         file does not have to know it. The arguments-done event is canonical
         across both backends (rio_provider.CANONICAL) and it fires exactly
         once per tool call, before the response completes. */
      if (t === 'response.function_call_arguments.done') {
        stats.tool_this_response = true;
      }

      if (t === 'response.done') {
        /* GENERATION IS OVER; THE SOUND MAY NOT BE. The queue is told, and the
           controller's holdTail is what waits. */
        if (playout) playout.generationDone(expecting);
        /* A RESPONSE THAT WAS CANCELLED IS *SUPPOSED* TO BE SILENT, and the
           first version of this counted those too. On the drive of 2026-09-20 it
           reported four degraded sessions and put "she is listening but not
           answering" on the screen while she was answering normally: three of the
           four were noise replies cancelled on sight (turn_kind "noise",
           generated_chars 0 -- the controller's own noise gate working exactly as
           designed) and the fourth was a response cancelled by a barge-in. She
           spoke three times in the same ninety seconds.
           
           So the shape this detector exists for is narrower than "no audio": it
           is a response the server said it COMPLETED that nonetheless made no
           sound. A cancelled one is not evidence of anything. */
        var status = (ev.response && ev.response.status) || null;
        /* ...AND THAT IS THE FOURTH KIND OF SILENCE THIS HAS HAD TO LEARN.
           The note above lists three -- cancelled, noise-gated, barged -- and
           a TOOL CALL is the fourth: the model emits a function call, the
           response completes having said nothing, and the words arrive in the
           NEXT response once the result is submitted. That is the design.

           On the drive of 2026-09-21 all five `session_silent` marks were
           this. Every one landed within 70 ms of a tool call starting:
           vehicle_status at 36.0 s, look at 47.6 s, find_places at 65.1 s,
           deep_dive at 81.1 s and again at 128.7 s -- five of roughly a dozen
           responses, reported as "she is listening but not answering" while
           she was doing exactly what the tool path asks of her. A detector
           that fires on healthy behaviour is worse than no detector: it is the
           one that teaches everyone to ignore the alarm. */
        if (stats.spoke_this_response !== true && status === 'completed'
            && stats.tool_this_response !== true) {
          stats.silent_responses++;
          silence.responses++;
          noteSilence('a response the server called completed made no sound');
        } else if (stats.spoke_this_response !== true) {
          stats.silent_by_design++;
          if (stats.tool_this_response === true) stats.silent_tool_calls++;
        }
        stats.spoke_this_response = false;
        stats.tool_this_response = false;
        startedSent = false;
      }

      /* INBOUND THROUGH THE SEAM, then into the one controller. */
      var mapped = provider ? provider.normalise(ev) : ev;
      if (mapped && mapped.type) deliver(mapped);

      /* ...and anything the gate is holding may now be sendable. */
      flushPending();
    }

    var startedSent = false;

    function expectingHeard() {
      silence.since = null;
      if (silence.degraded) {
        silence.degraded = false;
        emit({ type: 'XAI_SESSION_RECOVERED' });
      }
    }

    function noteSilence(why) {
      if (silence.degraded) return;
      silence.degraded = true;
      /* NOT A LOG LINE. Three wrong conclusions came from this exact shape -- a
         session that connects, runs VAD, commits audio and never speaks -- and
         every one of them was reached because nothing said so. It is a health
         state, it reaches the panel, and it says what was expected. */
      emit({ type: 'XAI_SESSION_SILENT', why: why,
             responses: silence.responses });
    }

    /* ---- outbound audio --------------------------------------------------- */
    /* THE STREAM IS NEVER PAUSED BETWEEN TURNS, and that is the silence tail.
       server_vad decides where an utterance ENDS and has to hear the end; a live
       microphone supplies that for free by continuing to send the room. Pausing
       the pump when nobody is talking would be the measured failure -- the turn
       stays open and no transcript ever arrives -- so `pumping` stops only when
       the session does. */
    var pumping = false;

    /* THE MICROPHONE IS NOT AT 24 kHz AND THIS IS WHERE THAT GETS FIXED.
     *
     * RATE is what the session was OPENED with -- xai_voice.mint_client_secret
     * sends audio.input.format = {type: 'audio/pcm', rate: 24000} -- so it is
     * what the far end will decode these bytes as. The AudioContext these
     * samples arrive on is the shared output bus, built at 48000 (see
     * rio_output.js), because her voice has to go through the echo canceller
     * with everything else RIO says.
     *
     * Those two numbers were never reconciled. This function took 48 kHz
     * samples, cut them into 960-sample pieces because 960 is 40 ms AT 24 kHz,
     * and shipped them raw. Every frame the driver spoke arrived at the
     * transcriber as twice its length an octave down -- a real voice saying
     * real words, slowed to half speed. grok-transcribe transcribed it the way
     * anything transcribes a tape at half speed, and the drive of 2026-09-21
     * has the shape of that in its own numbers: four barge detections, three
     * responses, and those three answers 15, 24 and 15 characters long. That
     * is the length of "Didn't catch that." She was not silent and she was not
     * cut off -- cutoffs_total was 0 and all three played to completion. She
     * was answering a garbled sentence with the only honest reply to one.
     *
     * So the stream is resampled to RATE before it is sent. A BOX AVERAGE over
     * each output sample's input span, not a nearest-sample pick: dropping
     * every other sample folds everything above 12 kHz back into the speech
     * band as aliasing, and a band-limit is the entire difference between
     * decimation and damage. For a context that already runs at RATE the span
     * is one sample and this is a copy.
     *
     * `ratio` is read per call rather than cached: the sample rate belongs to
     * the context, and a context can be rebuilt under us by an unlock. */
    function pumpFrom(micStream) {
      if (!micStream || !opts.ctx) return;
      var ctx = opts.ctx;
      var src = ctx.createMediaStreamSource(micStream);
      var frames = Math.round(RATE * FRAME_MS / 1000);
      var node = ctx.createScriptProcessor
        ? ctx.createScriptProcessor(4096, 1, 1) : null;
      if (!node) return;              // a browser without it gets no capture
      pumping = true;
      /* One flat buffer with a FRACTIONAL read position, because the ratio is
         not generally an integer (44100 Hz gives 1.8375) and the leftover has
         to survive into the next callback or the stream drifts a sample every
         frame. */
      var buf = new Float32Array(0);
      var pos = 0;
      node.onaudioprocess = function (e) {
        if (!pumping) return;
        var ch = e.inputBuffer.getChannelData(0);
        var grown = new Float32Array(buf.length + ch.length);
        grown.set(buf, 0);
        grown.set(ch, buf.length);
        buf = grown;
        var ratio = (ctx.sampleRate || RATE) / RATE;
        if (!(ratio > 0) || !isFinite(ratio)) ratio = 1;
        stats.audio_in_rate = ctx.sampleRate || null;
        stats.audio_in_ratio = ratio;
        for (;;) {
          var got = frameAtRate(buf, pos, frames, ratio);
          if (!got) break;            // not enough audio yet; wait for more
          pos = got.pos;
          var drop = Math.floor(pos);
          if (drop > 0) {
            buf = buf.slice(drop);     // slice, not subarray: a view would
            pos -= drop;               // keep the whole backing store alive
          }
          stats.audio_in_frames++;
          send({ type: 'input_audio_buffer.append',
                 audio: bytesToB64(floatToPcm16(got.out)) });
        }
      };
      src.connect(node);
      node.connect(ctx.destination);   // required for onaudioprocess to fire
    }

    /* ---- the socket ------------------------------------------------------- */
    function connect() {
      return new Promise(function (resolve, reject) {
        var sock;
        /* RESUMING ONE. The id goes on the URL because that is the only place it
           works; see the note on conversationId above. */
        var url = mint.ws_url;
        if (opts.resumeConversationId) {
          url += (url.indexOf('?') >= 0 ? '&' : '?')
            + 'conversation_id=' + encodeURIComponent(opts.resumeConversationId);
        }
        try {
          sock = new WS(url, [mint.ws_subprotocol]);
        } catch (e) { reject(e); return; }
        ws = sock;
        sock.onmessage = function (m) { onMessage(m.data); };
        sock.onerror = function () {
          emit({ type: 'XAI_SESSION_ERROR' });
        };
        sock.onclose = function () {
          closed = true;
          stopTicker();
          emit({ type: 'XAI_SESSION_CLOSED', stats: stats });
        };
        sock.onopen = function () {
          if (opts.ctx) {
            speaker = createSpeaker(opts.ctx,
                                   opts.destination || opts.ctx.destination);
          }
          /* THE POLICY IS THE SERVER'S. Sent verbatim, because three of its
             fields are the difference between a working session and a silent
             one and none of them should be assembled twice. */
          try {
            sock.send(JSON.stringify({ type: 'session.update',
                                       session: mint.session }));
          } catch (e) {}
          if (opts.mic) pumpFrom(opts.mic);
          startTicker();
          resolve(api);
        };
      });
    }

    var api = {
      connect: connect,
      send: send,
      playout: playout,
      /* For the controller's cfg.send: everything it emits comes through here. */
      stats: function () {
        return JSON.parse(JSON.stringify(stats));
      },
      /* For a reconnect that wants the conversation back. Null until the
         session says so. */
      conversationId: function () { return conversationId; },
      health: function () {
        return {
          connected: !!ws && !closed,
          conversation_id: conversationId,
          degraded: silence.degraded,
          silent_responses: silence.responses,
          silent_by_design: stats.silent_by_design,
          silent_tool_calls: stats.silent_tool_calls,
          pending_requests: pending.length,
          gate_timeouts: stats.gate_timeouts,
          queued_until: speaker ? speaker.queuedUntil() : 0,
          playout: playout ? playout.state() : null,
        };
      },
      stop: function () {
        pumping = false;
        closed = true;
        stopTicker();
        if (speaker) speaker.stop();
        try { if (ws) ws.close(); } catch (e) {}
      },
      /* Exposed for the node suite, which drives onMessage directly. */
      _onMessage: onMessage,
      _flushPending: flushPending,
      _tick: tick,
      _speaker: function () { return speaker; },
    };
    return api;
  }

  /* ---------------------------------------------------------------------
     ATTACH — one live conversation on this wire, from a minted session.

     Called by rio_realtime.connect() when the mint says this backend, and handed
     three things from there: createController, controllerConfig and
     sessionHandle. That is the whole arrangement in one sentence -- THE POLICY
     AND THE HANDLE ARE SHARED, THE TRANSPORT IS NOT. Nothing here decides how
     long a line may wait for the mouth or which channels may speak; nothing
     there knows an audio format or an event name.

     WHAT IS DIFFERENT ABOUT THIS WIRE, in the four members it hands back:

       audioState   there is no media element, no peer connection and no data
                    channel to report. There is a socket, a queue and a mouth,
                    and the degraded flag from a session that has gone quiet.
       resumeAudio  nothing to re-play: her voice is Web Audio, so what can be
                    suspended is the context, and that is what is resumed.
       stop         a socket and a microphone, not a peer connection.
       controller   the same one, over a different send.
     --------------------------------------------------------------------- */
  function attach(o) {
    o = o || {};
    var session = o.session || {};
    var out = (root.RIO && root.RIO.output) || null;

    /* HER VOICE GOES THROUGH THE SAME BUS AS EVERYTHING ELSE RIO SAYS, and on a
       phone that is not a tidiness argument. RIO.output.node() is inside the
       echo canceller (see the loopback in rio_output.js); ctx.destination is not.
       On WebRTC the remote track was the one path iOS could already cancel and
       this backend does not have it -- so if this connected to the destination
       instead, every word she said would come back into the microphone at full
       level and the barge gate would hear her as the driver. */
    var ctx = null, bus = null;
    try { if (out && out.unlock) out.unlock(); } catch (e) {}
    try { if (out && out.context) ctx = out.context(); } catch (e) {}
    try { if (out && out.node) bus = out.node(); } catch (e) {}
    if (!ctx) ctx = root.AudioContext ? new root.AudioContext() : null;
    if (!ctx) return Promise.reject(new Error('xai_voice: no AudioContext'));

    /* THE MOUTH. On the other backend it is a media element and muting it is
       one property; here her audio is a graph, so the mute is a gain of our own
       between the speaker and the bus. Ramped rather than switched: a gain that
       jumps to zero mid-word clicks, and the sustain gate changes its mind often
       enough for that to be audible. */
    var gain = ctx.createGain();
    gain.gain.value = 1;
    gain.connect(bus || ctx.destination);
    function ramp(to) {
      var t = ctx.currentTime;
      try {
        gain.gain.cancelScheduledValues(t);
        gain.gain.setValueAtTime(gain.gain.value, t);
        gain.gain.linearRampToValueAtTime(to, t + 0.015);
      } catch (e) { gain.gain.value = to; }
    }
    var mouth = {
      mute: function () { ramp(0); },
      unmute: function () { ramp(1); },
    };

    /* ONE PROVIDER RECORD, read by both halves: the transport asks it what to
       rename and what never to send, the controller asks it what the far end can
       do. Two instances would be two opinions. */
    var prov = (root.RIO && root.RIO.provider)
      ? root.RIO.provider.forVoiceBackend(session.voice_backend || 'xai_voice')
      : null;

    /* The controller does not exist yet -- its config needs a send, which needs
       the socket, which needs somewhere to deliver events. The indirection is
       one line and the alternative is two objects that each half-own a turn. */
    var controller = null;
    var proxy = { handle: function (ev) { if (controller) controller.handle(ev); } };
    var stopping = false;

    var sess = open({
      mint: session, controller: proxy, provider: prov,
      ctx: ctx, destination: gain, mic: o.mic,
      /* The node suite's seams, passed straight through: the socket, the clock,
         the ticker and the queue library. All undefined in a browser, which is
         what makes them seams rather than configuration. */
      WebSocketImpl: o.WebSocketImpl, now: o.now,
      setIntervalImpl: o.setIntervalImpl,
      clearIntervalImpl: o.clearIntervalImpl,
      playoutLib: o.playoutLib, tailGraceS: o.tailGraceS,
      onEvent: function (ev) {
        if (o.onEvent) { try { o.onEvent(ev); } catch (e) {} }
        if (!ev) return;
        /* THE SOCKET WENT AWAY MID-DRIVE. There is no equivalent of WebRTC's
           `disconnected` here -- a WebSocket close is final, so there is nothing
           for peerWatch's grace to wait out and the controller is told at once.
           A close we asked for is not news. */
        if ((ev.type === 'XAI_SESSION_CLOSED' || ev.type === 'XAI_SESSION_ERROR')
            && !stopping && controller) {
          controller.transportLost(ev.type === 'XAI_SESSION_ERROR'
                                   ? 'websocket_error' : 'websocket_closed');
        }
      },
    });

    controller = o.createController(o.controllerConfig(session, {
      arbiter: o.arbiter,
      barge: o.barge || {},
      /* THE ECHO METER, PASSED IN. There is no remote MediaStream on this wire,
         and for a while this said `null` for that reason -- wrongly. The meter
         needs the MICROPHONE, which exists, and its output reading falls back to
         RIO.output.level(), which measures the bus her voice is connected into.
         connect() builds it and hands it here; without it the level test has no
         evidence and every detector firing reaches the barge gate, which is
         measured: eleven barge detections and three cut-off answers in the first
         ninety seconds of the first drive. */
      levels: o.levels || function () { return null; },
      provider: prov,
      send: function (obj) { sess.send(obj); },
      url: o.url || function (p) { return p; },
      transcript: function () {
        try { return controller.state().spoken_this_turn || ''; }
        catch (e) { return ''; }
      },
      controller: function () { return controller; },
      audio: mouth,
      /* No relay sink: this backend speaks for itself, which is what makes the
         voice Eve rather than a second vendor's rendering of her. */
      voice: null,
      onEvent: o.onEvent,
    }));

    if (session.clip_base && root.RIO && root.RIO.setClipBase) {
      root.RIO.setClipBase(session.clip_base);
    }

    var opening = sess.connect();
    if (o.step) opening = o.step('socket', opening, o.progress);

    return opening.then(function () {
      /* THE TWO TOOLS THAT ARRIVE WITH THEIR PRECONDITION. stop_navigation and
         reroute are unusable until a route exists, so they ride with the route
         rather than with the session -- attached and removed by a session.update
         the controller sends. Without this call they would never be attached at
         all on this wire, and two of the nine live tools would simply not exist
         for the whole drive. Subscribed after the socket is open because that
         update has to have somewhere to go; a route that was ALREADY live when
         she was started mid-drive is picked up by the same call. */
      controller.watchToolConditions(root.RIO && root.RIO.bus,
                                     function (fn) { fn(); });
      return o.sessionHandle(session, {
        controller: controller,
        audioState: function () {
          var t = o.mic && o.mic.getTracks ? (o.mic.getTracks()[0] || null) : null;
          var h = sess.health();
          return {
            /* There is no element and no peer connection. Reported as null
               rather than omitted, so a drive log from either backend has the
               same shape and a missing key means a missing report. */
            element_paused: null, element_muted: null, element_ready: null,
            has_remote: null, peer: null, ice: null,
            channel: h.connected ? 'open' : 'closed',
            mic_muted: t ? !!t.muted : null,
            mic_state: t ? t.readyState : null,
            context: ctx.state || null,
            /* ...and what this wire has instead. `degraded` is the session
               that connected and never spoke. */
            degraded: h.degraded,
            silent_responses: h.silent_responses,
            pending_requests: h.pending_requests,
            queued_until: h.queued_until,
            playout: h.playout,
            stats: sess.stats(),
          };
        },
        resumeAudio: function () {
          /* Nothing to re-play -- there is no element to have been paused. What
             can be asleep is the context, and on iOS it will be. */
          try {
            if (ctx.state === 'suspended' && ctx.resume) {
              ctx.resume().catch(function () {});
            }
          } catch (e) {}
        },
        stop: function () {
          stopping = true;
          try { controller.stop(); } catch (e) {}
          try { sess.stop(); } catch (e) {}
          try { gain.disconnect(); } catch (e) {}
          if (o.mic && o.mic.getTracks) {
            o.mic.getTracks().forEach(function (t) {
              try { t.stop(); } catch (e) {}
            });
          }
        },
      });
    });
  }

  var apiOut = { open: open, attach: attach, createSpeaker: createSpeaker,
                 pcm16ToFloat: pcm16ToFloat, floatToPcm16: floatToPcm16,
                 frameAtRate: frameAtRate,
                 RATE: RATE, FRAME_MS: FRAME_MS };

  root.RIO = root.RIO || {};
  root.RIO.xaiSession = apiOut;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = apiOut;
  }
})(typeof window !== 'undefined' ? window : globalThis);
