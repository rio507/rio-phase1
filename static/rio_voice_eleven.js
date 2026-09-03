/* rio_voice_eleven.js — RIO's voice when it comes from ElevenLabs.
 *
 * The live session is in text mode: it hears the driver and does the thinking,
 * and the words it produces are forwarded to the server, synthesised there,
 * and played back HERE. This file is the playback half, and it owns three
 * things the server cannot know because they are facts about a speaker:
 *
 *   WHEN a sample is audible.   Audio arrives faster than it plays. The server
 *                               knows what it generated; only the page knows
 *                               what has actually come out.
 *   HOW FAST she stops.         A warning takes the mouth and RIO has to be
 *                               gone in 30 ms — a ramp, not a cut, because a
 *                               hard stop on a voice is a click and a click in
 *                               a car reads as a fault.
 *   HOW FAR she got.            The spoken prefix. A false barge-in resumes
 *                               from it, and it is the difference between RIO
 *                               finishing a sentence and starting it again.
 *
 * WHY WEB AUDIO AND NOT AN <audio> ELEMENT
 * ----------------------------------------
 * An element can be muted, and mute is not the same as stop: the stream plays
 * on inaudibly and there is nothing to fade. Both are wanted here and they are
 * different operations — mute is instant and UNDOABLE (the sustain gate exists
 * precisely to undo it when the noise turns out to have been a cough), stop is
 * a fade and a flush and is not undoable. A gain node does the first and a
 * scheduled buffer queue does the second.
 *
 * That is also why the audio arrives as raw PCM. An MP3 chunk out of the
 * middle of a stream cannot be decoded on its own; a 24 kHz frame can be
 * decoded and scheduled the instant it lands, which is the whole latency
 * argument for streaming at all.
 *
 * NO DECISIONS ABOUT SPEECH ARE MADE HERE. Phrasing, audio tags, flushing and
 * both fallback tiers are the server's, in voice_dialogue.py, where they can
 * be tested without a browser. What arrives here is samples and the words they
 * are for.
 */
(function (root) {
  'use strict';

  /* Long enough not to click, short enough to be gone before a warning's first
     syllable. 30 ms is about one pitch period of a low voice; below it the
     ramp is audible as an edge, above it RIO is still fading under the thing
     that interrupted her. */
  var FADE_MS = 30;

  /* How far ahead of the clock a chunk is scheduled. Enough to absorb a
     scheduling hiccup, small enough that a cancel does not have half a second
     of audio already committed underneath it. */
  var LEAD_S = 0.06;

  function b64ToBytes(b64) {
    var bin = root.atob(b64);
    var out = new Uint8Array(bin.length);
    for (var i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
    return out;
  }

  /* 16-bit little-endian PCM to a mono AudioBuffer at the source rate. The
     context resamples on playback; asking it to do that is cheaper and more
     correct than doing it here badly. */
  function pcmToBuffer(ctx, bytes, rate) {
    var n = bytes.length >> 1;
    var buf = ctx.createBuffer(1, Math.max(1, n), rate);
    var ch = buf.getChannelData(0);
    var dv = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    for (var i = 0; i < n; i++) ch[i] = dv.getInt16(i * 2, true) / 32768;
    return buf;
  }

  /* -------------------------------------------------------------------
     The sink. cfg = {
       wsUrl,            where the relay is
       transport,        an open socket, for tests — anything with
                         send()/close() and onmessage/onopen/onclose
       context,          an AudioContext, or a factory, or neither
       sampleRate,       what the server is sending (it says so on `ready`)
       onEvent(ev)       observability, and the cedar fallback
     }
     ------------------------------------------------------------------- */
  function createSink(cfg) {
    cfg = cfg || {};
    var listeners = [];
    if (typeof cfg.onEvent === 'function') listeners.push(cfg.onEvent);

    var ws = null;
    var ctx = null;
    var gain = null;
    var ready = false;
    var closed = false;
    var sampleRate = cfg.sampleRate || 24000;
    var fadeMs = cfg.fadeMs === undefined ? FADE_MS : cfg.fadeMs;

    /* Everything scheduled, in the order it will be heard:
       { rid, text, at, until, source }. `at`/`until` are context-clock
       seconds, which is the only clock that says what a listener has heard. */
    var queue = [];
    var nextAt = 0;
    var muted = true;              // silent until there is something to say
    var current = null;            // { rid, ended, done }
    var stats = { chunks: 0, bytes: 0, utterances: 0, cancels: 0,
                  fallbacks: 0, underruns: 0 };

    function emit(type, payload) {
      var ev = payload || {};
      ev.type = type;
      for (var i = 0; i < listeners.length; i++) {
        try { listeners[i](ev); } catch (e) { /* never let a listener mute RIO */ }
      }
    }

    function now() { return ctx ? ctx.currentTime : 0; }

    function makeContext() {
      if (ctx) return ctx;
      if (cfg.context) {
        ctx = (typeof cfg.context === 'function') ? cfg.context() : cfg.context;
      } else {
        var C = root.AudioContext || root.webkitAudioContext;
        if (!C) return null;
        ctx = new C();
      }
      gain = ctx.createGain();
      gain.gain.value = 0;
      gain.connect(ctx.destination);
      return ctx;
    }

    /* -- what a listener has actually heard ---------------------------- */

    /* The spoken prefix, at this instant, for one response.
     *
     * Chunks entirely in the past count in full. The one straddling the clock
     * counts proportionally — by characters, which is a rough model of speech
     * and a much better one than counting it as nothing or as all of it. This
     * is what a resume carries, and a resume that overstates makes RIO skip a
     * clause the driver never heard. */
    function spokenPrefix(rid) {
      var t = now();
      var out = '';
      for (var i = 0; i < queue.length; i++) {
        var c = queue[i];
        if (rid && c.rid !== rid) continue;
        if (!c.text) continue;
        if (c.until <= t) { out += c.text; continue; }
        if (c.at >= t) break;
        var frac = (t - c.at) / Math.max(0.001, c.until - c.at);
        out += c.text.slice(0, Math.round(c.text.length * frac));
        break;
      }
      return out;
    }

    /* Drop what nobody will ask about again: chunks from an earlier response
       that have already been heard. Anything still in the FUTURE stays,
       whichever response it belongs to — a holding line that is still coming
       out of the speaker is not rubbish to be swept up, it is the reason the
       next utterance has to queue behind it rather than start on top of it. */
    function prune(rid) {
      var t = now();
      queue = queue.filter(function (c) { return c.rid === rid || c.until > t; });
    }

    /* -- scheduling ---------------------------------------------------- */
    function schedule(rid, bytes, text) {
      if (!makeContext()) return;
      if (current && current.rid !== rid) return;   // audio for a dead turn
      var buf = pcmToBuffer(ctx, bytes, sampleRate);
      var t = now();
      // An underrun — the queue drained while the model was still writing —
      // is not an error, but it IS a seam in the middle of a sentence, and it
      // is the number that says whether the chunker is pacing well.
      if (nextAt < t + LEAD_S) {
        if (nextAt > 0) stats.underruns++;
        nextAt = t + LEAD_S;
      }
      var src = ctx.createBufferSource();
      src.buffer = buf;
      src.connect(gain);
      src.start(nextAt);
      queue.push({ rid: rid, text: text || '', at: nextAt,
                   until: nextAt + buf.duration, source: src });
      nextAt += buf.duration;
      stats.chunks++;
      stats.bytes += bytes.length;
    }

    function stopAll() {
      for (var i = 0; i < queue.length; i++) {
        try { queue[i].source.stop(); } catch (e) {}
      }
      nextAt = 0;
    }

    /* -- the wire ------------------------------------------------------ */
    function send(obj) {
      if (!ws) return false;
      try { ws.send(JSON.stringify(obj)); return true; } catch (e) { return false; }
    }

    function onMessage(data) {
      var m;
      try { m = JSON.parse(data); } catch (e) { return; }
      if (m.op === 'ready') {
        ready = true;
        sampleRate = m.sample_rate || sampleRate;
        emit('VOICE_READY', { open: !!m.open, model: m.model,
                              voice_id: m.voice_id, degraded: !!m.degraded });
        if (m.open === false) {
          // Nothing to fall back FROM. The controller is told now so the drive
          // starts in the voice it is going to keep.
          emit('VOICE_FALLBACK', { tier: 'cedar', cause: 'socket_unavailable' });
        }
        return;
      }
      if (m.op === 'audio') {
        schedule(m.rid, b64ToBytes(m.pcm), m.text);
        return;
      }
      if (m.op === 'event') {
        if (m.kind === 'fallback') {
          stats.fallbacks++;
          emit('VOICE_FALLBACK', m.detail || {});
        } else if (m.kind === 'utterance_done') {
          if (current && current.rid === (m.detail || {}).rid) current.done = true;
          emit('VOICE_UTTERANCE_DONE', m.detail || {});
        } else if (m.kind === 'reconnected') {
          emit('VOICE_RECONNECTED', m.detail || {});
        }
        return;
      }
    }

    /* -- the interface the controller uses ----------------------------- */
    var sink = {
      open: function () {
        if (closed) return Promise.reject(new Error('closed'));
        /* The context is built HERE, not at the first thing she says.
           connect() is called from the mic button's own tap handler, and a
           browser will only start an AudioContext inside a user gesture — one
           created later comes up suspended, and on iOS stays that way. The
           gain starts at zero, so building it early is silent. */
        makeContext();
        return new Promise(function (resolve, reject) {
          if (cfg.transport) {
            ws = cfg.transport;
          } else {
            var W = root.WebSocket;
            if (!W) { reject(new Error('no websocket')); return; }
            ws = new W(cfg.wsUrl);
          }
          var settled = false;
          ws.onmessage = function (e) {
            onMessage(e.data);
            if (!settled && ready) { settled = true; resolve(sink); }
          };
          ws.onclose = function () {
            if (closed) return;
            emit('VOICE_TRANSPORT_LOST', {});
            if (!settled) { settled = true; reject(new Error('voice socket closed')); }
          };
          ws.onerror = function () {
            if (!settled) { settled = true; reject(new Error('voice socket error')); }
          };
          if (ws.readyState === 1 && ws.onopen) { /* already open */ }
          // A relay that never answers is a drive that never starts. Six
          // seconds is far past a loopback round trip and far short of a
          // driver deciding the button is broken.
          setTimeout(function () {
            if (!settled) { settled = true; reject(new Error('voice socket timeout')); }
          }, 6000);
        });
      },

      /* A new answer. One utterance, and the queue belongs to it now. */
      /* A new answer. One utterance, and the queue belongs to it now.
       *
       * `nextAt` is deliberately NOT reset. It is where the audio already
       * scheduled runs out, and the new utterance starts there — so a tool
       * follow-up arriving while "let me check" is still playing is heard
       * after it rather than over it. When nothing is pending, `nextAt` is
       * already in the past and schedule() clamps it to now. */
      begin: function (rid) {
        prune(rid);
        current = { rid: rid, ended: false, done: false };
        stats.utterances++;
        send({ op: 'begin', rid: rid });
      },

      delta: function (rid, text) {
        if (!text) return;
        send({ op: 'delta', rid: rid, text: text });
      },

      /* The model has stopped writing. Resolves when the LISTENER is done,
         which is later and is the moment the mouth can be handed back. */
      end: function (rid) {
        if (current && current.rid === rid) current.ended = true;
        send({ op: 'end', rid: rid });
        return new Promise(function (resolve) {
          var giveUp = Date.now() + 60000;
          (function wait() {
            if (closed || !current || current.rid !== rid) return resolve();
            var drained = current.done && now() >= nextAt;
            if (drained || Date.now() > giveUp) return resolve();
            setTimeout(wait, 60);
          })();
        });
      },

      /* Stop. Returns what the driver actually heard, which the caller needs
         before the queue is thrown away. */
      cancel: function (rid) {
        var said = spokenPrefix(rid);
        stats.cancels++;
        sink.mute();
        // Stop the sources AFTER the ramp has run, or the fade is a cut with
        // extra steps.
        setTimeout(stopAll, fadeMs + 5);
        if (current && current.rid === rid) current.done = true;
        send({ op: 'cancel', rid: rid });
        return said;
      },

      /* Instant and undoable — this is what a barge-in does before anyone
         knows whether a person spoke. The ramp is the only thing that makes it
         inaudible; the audio underneath keeps playing, exactly as a muted
         element would, so a false alarm can be undone mid-word. */
      mute: function () {
        if (!gain || muted) { muted = true; return; }
        muted = true;
        try {
          var t = now();
          gain.gain.cancelScheduledValues(t);
          gain.gain.setValueAtTime(gain.gain.value, t);
          gain.gain.linearRampToValueAtTime(0, t + fadeMs / 1000);
        } catch (e) { try { gain.gain.value = 0; } catch (e2) {} }
      },

      unmute: function () {
        if (!makeContext()) return;
        muted = false;
        try {
          if (ctx.state === 'suspended' && ctx.resume) ctx.resume();
          var t = now();
          gain.gain.cancelScheduledValues(t);
          gain.gain.setValueAtTime(gain.gain.value, t);
          gain.gain.linearRampToValueAtTime(1, t + fadeMs / 1000);
        } catch (e) { try { gain.gain.value = 1; } catch (e2) {} }
      },

      spokenPrefix: spokenPrefix,
      onEvent: function (fn) { if (typeof fn === 'function') listeners.push(fn); },

      state: function () {
        return { ready: ready, muted: muted, rid: current ? current.rid : null,
                 queued: queue.length, next_at: nextAt, now: now(),
                 stats: stats };
      },

      close: function () {
        closed = true;
        stopAll();
        try { if (ws) ws.close(); } catch (e) {}
        try { if (ctx && ctx.close && !cfg.context) ctx.close(); } catch (e) {}
      },
    };
    return sink;
  }

  root.RIO = root.RIO || {};
  root.RIO.voiceEleven = { createSink: createSink, FADE_MS: FADE_MS };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = root.RIO.voiceEleven;
  }
})(typeof window !== 'undefined' ? window : globalThis);
