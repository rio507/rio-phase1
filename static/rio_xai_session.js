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
    /* WHICH UTTERANCES HAVE BEEN SEEN, so the three .completed events for one
       utterance become one. By id, not by count: counting would break the day the
       repeat count changes, and the id is what makes them provably the same
       utterance rather than three that happen to match. */
    var transcriptSeen = Object.create(null);
    /* PENDING TOOL RESULTS, held until the sound is over. See flushToolResults. */
    var pending = [];
    var stats = {
      audio_in_frames: 0, audio_out_bytes: 0, transcripts: 0,
      transcript_repeats_dropped: 0, local_flushes: 0, gated_requests: 0,
      responses: 0, spoke: 0, silent_responses: 0,
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
        pending.push(out);
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

    /* Anything held back by the gate, once there is silence to send it into. */
    function flushPending() {
      if (!pending.length) return;
      if (playout && !playout.idle()) return;
      var go = pending.slice();
      pending.length = 0;
      go.forEach(function (o) {
        try { ws.send(JSON.stringify(o)); } catch (e) {}
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

      if (t === 'response.done') {
        /* GENERATION IS OVER; THE SOUND MAY NOT BE. The queue is told, and the
           controller's holdTail is what waits. */
        if (playout) playout.generationDone(expecting);
        if (stats.spoke_this_response !== true) {
          stats.silent_responses++;
          silence.responses++;
          noteSilence('a response closed without making a sound');
        }
        stats.spoke_this_response = false;
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

    function pumpFrom(micStream) {
      if (!micStream || !opts.ctx) return;
      var ctx = opts.ctx;
      var src = ctx.createMediaStreamSource(micStream);
      var frames = Math.round(RATE * FRAME_MS / 1000);
      var node = ctx.createScriptProcessor
        ? ctx.createScriptProcessor(4096, 1, 1) : null;
      if (!node) return;              // a browser without it gets no capture
      pumping = true;
      var acc = [];
      var accLen = 0;
      node.onaudioprocess = function (e) {
        if (!pumping) return;
        var ch = e.inputBuffer.getChannelData(0);
        acc.push(new Float32Array(ch));
        accLen += ch.length;
        while (accLen >= frames) {
          var out = new Float32Array(frames);
          var got = 0;
          while (got < frames) {
            var head = acc[0];
            var take = Math.min(frames - got, head.length);
            out.set(head.subarray(0, take), got);
            got += take;
            if (take === head.length) acc.shift();
            else acc[0] = head.subarray(take);
          }
          accLen -= frames;
          stats.audio_in_frames++;
          send({ type: 'input_audio_buffer.append',
                 audio: bytesToB64(floatToPcm16(out)) });
        }
      };
      src.connect(node);
      node.connect(ctx.destination);   // required for onaudioprocess to fire
    }

    /* ---- the socket ------------------------------------------------------- */
    function connect() {
      return new Promise(function (resolve, reject) {
        var sock;
        try {
          sock = new WS(mint.ws_url, [mint.ws_subprotocol]);
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
      health: function () {
        return {
          connected: !!ws && !closed,
          degraded: silence.degraded,
          silent_responses: silence.responses,
          pending_requests: pending.length,
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
      /* NO ECHO METER ON THIS WIRE YET, and that is a real difference rather
         than an oversight: makeMeter measures the microphone against a remote
         MediaStream, and there is no remote stream here -- her voice is a graph.
         Without one the barge gate behaves as it does on a desk, which is the
         behaviour every session had before the meter existed. It is the one
         thing this backend gives up on a phone, and it is written down here
         because a silent `null` would read as "no echo problem". */
      levels: function () { return null; },
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
                 RATE: RATE, FRAME_MS: FRAME_MS };

  root.RIO = root.RIO || {};
  root.RIO.xaiSession = apiOut;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = apiOut;
  }
})(typeof window !== 'undefined' ? window : globalThis);
