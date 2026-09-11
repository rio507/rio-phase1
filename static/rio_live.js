/* rio_live.js — the gpt-live-1 conversation, in the browser.
 *
 * THE SHAPE OF THE DIFFERENCE, because this file is not a port of
 * rio_realtime.js and trying to read it as one will mislead. Under
 * gpt-realtime the page is a participant in the turn: it watches for
 * `input_audio_buffer.speech_started`, decides on its own evidence whether
 * that was a driver or a door, and sends `response.cancel` if it was a
 * driver. Under gpt-live-1 none of those three things exists. The model is
 * full duplex -- it listens while it speaks -- and it owns its own turn.
 *
 * So this controller is SMALLER on purpose, and the things missing from it
 * are missing because the events they hung off are gone:
 *
 *   no manual commit         audio streams continuously; the migration
 *                            guide's first instruction is to stop committing.
 *   no barge-in gate         nothing to cancel. See config.GUARD_BARGE_SUSTAIN.
 *   no fragment coalescing   no committed fragments to join. Transcripts
 *                            arrive as revisable overlapping deltas.
 *   no supersede             ditto; there is no response to supersede.
 *
 * WHAT IS NOT MISSING, and must never be: the echo gate. Measured on the real
 * API (tools/live_echo_probe.py), RIO's own voice played back into the session
 * the way a phone speaker plays it back into a phone microphone was
 * transcribed as the driver and ANSWERED, eight times out of eight, from full
 * level down to -20 dB. Full duplex does not mean deaf to itself; it means the
 * opposite. The gate is what stops a pre-rendered clip -- which the browser's
 * echo canceller has no reference for -- from starting a conversation.
 *
 * WHICH GUARDS RUN IS NOT DECIDED HERE. It arrives with the session, from
 * config.guards(), so re-enabling one is an env var on the server rather than
 * a new build of this file.
 */
'use strict';
(function (root) {
  var RT = function () { return (root.RIO && root.RIO.realtime) || {}; };

  function url(p) { return p; }

  /* ---------------------------------------------------------------------
     THE ECHO GATE, which is the one guard this backend still needs.

     It answers one question: is this transcript the driver, or is it RIO's
     own voice coming back through the cabin? Two tests, and neither is
     sufficient alone -- the same two the realtime controller uses, for the
     same reasons, kept deliberately identical so a drive can be compared
     across backends.

       structural   did her audio stop less than echoTailMs ago? Her voice
                    reaches the microphone through a speaker a few centimetres
                    away, so anything arriving in the tail of her own sentence
                    is her.
       textual      does the transcript look like something she just said?
                    Unreliable on its own -- her audio is transcribed twice by
                    two different models and they disagree ("Hey" came back as
                    "Hello") -- which is exactly why it is the second net.
     --------------------------------------------------------------------- */
  function createEchoGate(cfg) {
    cfg = cfg || {};
    /* `a || b` IS THE WRONG DEFAULT FOR A THRESHOLD, and the selftest caught
       it: echoTailMs of 0 means "do not use the structural test", and `0 ||
       600` turns that into 600 ms of every real question being refused as an
       echo. A threshold that was explicitly set to zero is a decision, not an
       absence. */
    function num(v, dflt) {
      return (typeof v === 'number' && !isNaN(v)) ? v : dflt;
    }
    var tailMs = num(cfg.echoTailMs, 600);
    var windowS = num(cfg.echoTextWindowS, 15);
    var overlapFrac = num(cfg.echoTextOverlap, 0.8);
    var minWords = num(cfg.echoTextMinWords, 2);
    var spokenRecently = [];       /* {t, words} of what SHE said */
    var mouthClosedAt = 0;         /* when her audio last stopped */

    function norm(s) {
      return String(s || '').toLowerCase().replace(/[^a-z0-9 ]/g, ' ')
        .split(/\s+/).filter(Boolean);
    }

    return {
      /* Called as her transcript arrives, so the text test has something to
         compare against. */
      noteSpoke: function (text) {
        var w = norm(text);
        if (!w.length) return;
        spokenRecently.push({ t: Date.now(), words: w });
        mouthClosedAt = Date.now();
        var cut = Date.now() - windowS * 1000;
        while (spokenRecently.length && spokenRecently[0].t < cut) {
          spokenRecently.shift();
        }
      },
      mouthClosed: function () { mouthClosedAt = Date.now(); },
      /* True when this incoming transcript should be treated as her own voice
         rather than as a question. */
      isEcho: function (text) {
        if (!cfg.enabled) return false;
        var w = norm(text);
        if (!w.length) return false;
        var sinceMouth = Date.now() - mouthClosedAt;
        var inTail = mouthClosedAt > 0
                     && sinceMouth < tailMs;
        var recent = [];
        var cut = Date.now() - windowS * 1000;
        for (var i = 0; i < spokenRecently.length; i++) {
          if (spokenRecently[i].t >= cut) {
            recent = recent.concat(spokenRecently[i].words);
          }
        }
        var overlap = 0;
        for (var j = 0; j < w.length; j++) {
          if (recent.indexOf(w[j]) !== -1) overlap++;
        }
        var frac = w.length ? overlap / w.length : 0;
        /* Below the word floor only an exact containment counts: "yes" and
           "no" share every word with almost anything. */
        var textSays = (w.length >= minWords)
          ? frac >= overlapFrac
          : (recent.join(' ').indexOf(w.join(' ')) !== -1 && recent.length > 0);
        return inTail || textSays;
      },
      _state: function () {
        return { mouthClosedAt: mouthClosedAt,
                 remembered: spokenRecently.length };
      },
    };
  }

  /* ---------------------------------------------------------------------
     The controller: a pure handler over the data-channel event stream.

     Pure on purpose, exactly as rio_realtime.js is: everything interesting
     that can go wrong in a car is a sequence of these events, and a handler
     that needs a microphone and a network to be exercised is a handler nobody
     tests. tools/live_voice_probe.js drives this with no browser at all.
     --------------------------------------------------------------------- */
  function createController(opts) {
    opts = opts || {};
    var send = opts.send || function () {};
    var tool = opts.tool || function () { return Promise.resolve({}); };
    var guards = opts.guards || {};
    var gate = createEchoGate({
      enabled: guards.echo_gate !== false,
      echoTailMs: opts.echoTailMs,
      echoTextWindowS: opts.echoTextWindowS,
      echoTextOverlap: opts.echoTextOverlap,
      echoTextMinWords: opts.echoTextMinWords,
    });

    var heard = '';               /* the driver, as it arrives */
    var said = '';                /* her, as it arrives */
    var pending = {};             /* event_id -> {resolve, started} for speak */
    var stats = { delegations: 0, tools: 0, echoes: 0, errors: 0, lines: 0 };
    var lastDriver = '';

    function onEvent(e) {
      if (!e || !e.type) return;
      switch (e.type) {
        case 'session.started':
          if (opts.onSession) opts.onSession(e.session || {});
          break;

        case 'session.input_transcript.delta':
          heard += e.delta || '';
          break;

        /* A finished stretch of the driver. This is where the echo gate acts,
           and it acts by DISCARDING rather than by cancelling: there is
           nothing to cancel, so the only thing the page can do with her own
           voice coming back is refuse to treat it as a question. */
        case 'session.input_transcript.done':
          var t = (e.transcript || heard || '').trim();
          heard = '';
          if (!t) break;
          if (gate.isEcho(t)) {
            stats.echoes++;
            if (opts.onEcho) opts.onEcho(t);
            break;
          }
          lastDriver = t;
          if (opts.onDriver) opts.onDriver(t);
          break;

        case 'session.output_transcript.delta':
          if (!said && opts.onSpeaking) opts.onSpeaking();
          said += e.delta || '';
          /* The first delta after a line was asked for is the closest thing
             this API has to "her mouth opened". rio_speak needs that moment to
             decide whether to fall back to the synthesiser, and the honest
             alternative -- waiting for audio on the media track -- needs an
             analyser node per line. */
          for (var id in pending) {
            if (!pending[id].started) {
              pending[id].started = true;
              if (pending[id].onStart) pending[id].onStart();
            }
          }
          break;

        case 'session.output_transcript.done':
          var full = (e.transcript || said || '').trim();
          said = '';
          if (full) gate.noteSpoke(full);
          gate.mouthClosed();
          for (var k in pending) {
            var p = pending[k];
            delete pending[k];
            if (p.resolve) p.resolve({ ok: true, said: full });
          }
          if (opts.onSaid) opts.onSaid(full);
          break;

        /* She has decided a question needs the backend. Nothing to do -- the
           Responses loop runs on OpenAI's side -- but it is the event that
           says a silence is work rather than a failure, and the panel says so. */
        case 'session.delegation.created':
          stats.delegations++;
          if (opts.onDelegation) opts.onDelegation(e.delegation || {});
          break;

        /* The backend's own stream, wrapped. A tool call is two levels down. */
        case 'response.event':
          handleResponseEvent(e);
          break;

        case 'session.commentary.appended':
          break;

        case 'error':
          stats.errors++;
          if (opts.onError) opts.onError(e.error || e);
          /* A line that was refused must not leave rio_speak waiting for a
             mouth that will never open: resolve it as a failure so the
             synthesiser tier takes over. */
          var cid = (e.error && e.error.client_event_id) || e.client_event_id;
          if (cid && pending[cid]) {
            var pe = pending[cid];
            delete pending[cid];
            if (pe.resolve) pe.resolve({ ok: false, note: 'refused' });
          }
          break;

        default:
          break;
      }
    }

    function handleResponseEvent(e) {
      var inner = e.event || {};
      if (inner.type !== 'response.output_item.done') return;
      var item = inner.item || {};
      if (item.type !== 'function' && item.type !== 'function_call') return;
      var args = {};
      try { args = JSON.parse(item.arguments || '{}'); } catch (err) { args = {}; }
      stats.tools++;
      if (opts.onTool) opts.onTool(item.name, args);
      Promise.resolve(tool(item.name, args))
        .then(function (out) { returnTool(item.call_id, out); })
        .catch(function () {
          returnTool(item.call_id, { ok: false, note: 'tool failed' });
        });
    }

    /* TWO EVENTS AND NOT ONE. The result is added to the backend's input, and
       then the backend is asked to carry on. Sending only the first leaves a
       model holding an answer nobody told it to use, which presents as RIO
       going quiet after a tool call. */
    function returnTool(callId, out) {
      send({ type: 'response.item.create',
             event_id: 'tool_result_' + callId,
             item: { type: 'function_call_output', call_id: callId,
                     output: typeof out === 'string' ? out
                                                     : JSON.stringify(out) } });
      send({ type: 'response.create', event_id: 'continue_' + callId });
    }

    /* ONE DETERMINISTIC LINE, through the event that is actually verbatim.
       session.commentary.append is the obvious candidate and is the wrong one:
       the docs describe it as material the model MAY PARAPHRASE, and measured
       on RIO's own lines it does -- "In 300 feet" came back as "About 300
       feet", a hedge added to a distance the policy stated exactly.
       session.instructions.append is a directive, is what the guide gives for
       exact wording, and holds. See live.verbatim_event for the numbers. */
    var seq = 0;
    function speak(text, o) {
      o = o || {};
      var line = String(text || '').trim();
      if (!line) return Promise.resolve({ ok: false, note: 'no text' });
      var id = 'rio_line_' + (++seq);
      stats.lines++;
      return new Promise(function (resolve) {
        pending[id] = { resolve: resolve, started: false,
                        onStart: o.onStart };
        send({ type: 'session.instructions.append', event_id: id,
               /* Null, not absent. Omitting the key is a
                  missing_required_parameter error and a car that says nothing;
                  null is the documented way to say "this is not an answer to a
                  delegation", which is what every warning RIO issues is. */
               delegation_id: null,
               content: 'Say this out loud right now, reproducing it EXACTLY '
                        + 'as written, with no additions, no omissions, no '
                        + 'rewording and no introduction: "' + line + '"' });
        var budget = o.timeoutMs || 0;
        if (budget > 0) {
          setTimeout(function () {
            if (pending[id]) {
              delete pending[id];
              resolve({ ok: false, note: 'timeout' });
            }
          }, budget);
        }
      });
    }

    return {
      onEvent: onEvent,
      speak: speak,
      stats: function () { return JSON.parse(JSON.stringify(stats)); },
      lastDriver: function () { return lastDriver; },
      state: function () {
        return { voice_backend: 'gpt_live', echo: gate._state(),
                 guards: guards };
      },
      stop: function () {
        for (var k in pending) {
          var p = pending[k];
          delete pending[k];
          if (p.resolve) p.resolve({ ok: false, note: 'stopped' });
        }
      },
      _gate: gate,
    };
  }

  /* ---------------------------------------------------------------------
     Connecting. One extra hop compared with the realtime path, and the
     comment on /live/session in app.py says why: the live endpoint takes the
     session config and the SDP offer in one request, so the only way to keep
     the account key out of this page is to let the server make it.
     --------------------------------------------------------------------- */
  function connect(o) {
    o = o || {};
    var element = o.element;
    function emit(ev) { if (o.onEvent) { try { o.onEvent(ev); } catch (e) {} } }
    function step(name) { if (o.onProgress) { try { o.onProgress(name); } catch (e) {} } }
    var pc = new RTCPeerConnection();
    var channel = pc.createDataChannel('oai-events');
    var controller = null;
    var session = null;
    var mic = null;

    step('mic');
    return navigator.mediaDevices
      .getUserMedia({ audio: RT().micConstraints || true })
      .then(function (stream) {
        mic = stream;
        stream.getTracks().forEach(function (t) { pc.addTrack(t, stream); });
        pc.ontrack = function (ev) {
          if (element) element.srcObject = ev.streams[0];
        };
        return pc.createOffer();
      })
      .then(function (offer) { return pc.setLocalDescription(offer).then(function () { return offer; }); })
      .then(function (offer) {
        step('negotiate');
        return fetch(url('/live/session'), {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ sdp: offer.sdp }),
        }).then(function (r) { return r.json(); });
      })
      .then(function (body) {
        if (!body || !body.sdp) {
          throw new Error((body && body.error) || 'no answer from /live/session');
        }
        session = body;
        /* The clip prefix for this voice, applied before anything can play a
           clip. See config.CLIP_DIRS: the files are kept per voice, and the
           three sites that build a clip URL all read it from here. */
        if (root.RIO && root.RIO.setClipBase && body.clip_base) {
          root.RIO.setClipBase(body.clip_base);
        }
        controller = createController({
          guards: body.guards || {},
          echoTailMs: body.echo_tail_ms,
          echoTextWindowS: body.echo_text_window_s,
          echoTextOverlap: body.echo_text_overlap,
          echoTextMinWords: body.echo_text_min_words,
          /* THE PAGE'S OWN EVENT VOCABULARY, emitted from here so index.html
             does not need to know which backend it is talking to. The names
             are the realtime controller's: a panel that had to learn a second
             set would be a second panel. */
          onSession: o.onSession,
          onDriver: function (t) {
            emit({ type: 'LIVE_TRANSCRIPT', transcript: t });
            if (o.onDriver) o.onDriver(t);
          },
          onSaid: function (t) {
            emit({ type: 'LIVE_RESPONSE_END', text: t });
            if (o.onSaid) o.onSaid(t);
          },
          onSpeaking: function () { emit({ type: 'LIVE_RESPONSE_START' }); },
          onTool: function (name, args) {
            emit({ type: 'LIVE_TOOL_CALL', name: name, arguments: args });
            if (o.onTool) o.onTool(name, args);
          },
          onDelegation: function (d) {
            emit({ type: 'LIVE_DELEGATION', delegation: d });
            if (o.onDelegation) o.onDelegation(d);
          },
          onEcho: function (t) {
            /* Reported, not hidden. An echo the gate caught is the gate
               working, and the cut-off tally is where that gets counted. */
            emit({ type: 'LIVE_ECHO_SUPPRESSED', transcript: t });
            if (o.onEcho) o.onEcho(t);
          },
          onError: function (e) {
            emit({ type: 'LIVE_ERROR', error: e });
            if (o.onError) o.onError(e);
          },
          send: function (obj) {
            if (channel.readyState === 'open') channel.send(JSON.stringify(obj));
          },
          /* The SAME tool implementations the realtime path uses. Imported
             rather than reimplemented: a second copy of nav_status is a second
             answer to "where are we", and the two would disagree on the drive
             nobody could reproduce. */
          tool: function (name, args) {
            var localTools = RT().localTools || {};
            if (localTools[name]) {
              try { return Promise.resolve(localTools[name](args)); }
              catch (e) { return Promise.resolve({ ok: false, note: 'panel error' }); }
            }
            return fetch(url('/realtime/tool'), {
              method: 'POST', headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                name: name, arguments: args,
                where: RT().currentFix ? RT().currentFix() : null,
                spoken: controller ? controller.lastDriver() : '',
              }),
            }).then(function (r) { return r.json(); });
          },
        });
        channel.onmessage = function (m) {
          var e; try { e = JSON.parse(m.data); } catch (err) { return; }
          controller.onEvent(e);
        };
        return pc.setRemoteDescription({ type: 'answer', sdp: body.sdp });
      })
      .then(function () {
        /* The floor for one channel: its own if the table names it, the
           session-wide one otherwise. A channel the table has not been taught
           about gets the general floor rather than none, so a new deterministic
           channel is patient by default rather than accidentally impatient. */
        function floorFor(channelName) {
          var t = session.speak_timeout_floor_ms_by_channel || {};
          var v = channelName ? t[channelName] : undefined;
          return (typeof v === 'number') ? v
                 : (session.speak_timeout_floor_ms || 0);
        }

        var handle = {
          session: session,
          controller: controller,
          speak: function (text, opt) {
            opt = opt || {};
            /* The floor this backend's slower deterministic path needs. The
               per-channel budgets in config.py were measured against
               gpt-realtime's 390-585 ms; the verbatim directive that makes
               this backend exact runs p50 1324 / p95 2317, so a 900 ms budget
               in front of it does not mean "fall back if something goes
               wrong", it means "fall back every time".

               PER CHANNEL, because headway must NOT be raised: its arbiter
               item carries a 2500 ms TTL and a late gap warning is worse than
               one in a slightly different voice. config.py holds the argument;
               this only reads the table. */
            var floor = floorFor(opt.channel);
            if (!opt.timeoutMs || opt.timeoutMs < floor) {
              opt = Object.assign({}, opt, { timeoutMs: floor || opt.timeoutMs });
            }
            return controller.speak(text, opt);
          },
          speechEnabled: function (channelName) {
            if (session.speech_enabled === false) return false;
            var chans = session.speech_channels || {};
            return chans[channelName] !== false;
          },
          speakTimeout: function (channelName, callType) {
            var table = session.speak_timeout_ms_by_channel || {};
            var byCall = table[channelName];
            var ms = !byCall ? session.speak_timeout_ms
                   : (byCall[callType] || byCall._default
                      || session.speak_timeout_ms);
            return Math.max(ms || 0, floorFor(channelName));
          },
          voiceBackend: function () { return 'gpt_live'; },
          stop: function () {
            if (controller) controller.stop();
            try { channel.close(); } catch (e) {}
            try { pc.close(); } catch (e) {}
            if (mic) mic.getTracks().forEach(function (t) { t.stop(); });
            if (element) element.srcObject = null;
            if (active === handle) active = null;
          },
        };
        active = handle;
        return handle;
      });
  }

  var active = null;

  root.RIO = root.RIO || {};
  root.RIO.live = {
    createController: createController,
    createEchoGate: createEchoGate,
    connect: connect,
    active: function () { return active; },
    _setActive: function (h) { active = h; },
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { createController: createController,
                       createEchoGate: createEchoGate };
  }
})(typeof window !== 'undefined' ? window : globalThis);
