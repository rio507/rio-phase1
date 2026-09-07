/* rio_output.js — one speaker for everything RIO says, so a phone can cancel it.
 *
 * THE PROBLEM THIS SOLVES IS NOT A MIXING PROBLEM. Every line RIO speaks was
 * already arbitrated — rio_speech.js decides what is audible and there has
 * never been anything to mix. What was wrong was WHERE the audio came out.
 *
 * Safari's echo canceller belongs to the WebRTC capture path. It subtracts
 * what that path renders and it has no reference at all for anything else, so
 * an ordinary <audio> element playing a clip, and an AudioContext rendering to
 * ctx.destination, both come back into the microphone at full level. On a
 * phone in a car that is RIO's own voice arriving at the turn detector, which
 * calls it speech, and she interrupts herself. `node tools/echo_barge_probe.js`
 * counts what it costs: five long answers into an empty car, five cut off.
 *
 * So this file gives the page ONE output, and routes it back out through a
 * peer connection to itself:
 *
 *      clips, TTS blobs, the ElevenLabs sink
 *            |
 *            v
 *      bus (GainNode) -> analyser -> MediaStreamDestination
 *                                          |
 *                                    RTCPeerConnection loopback
 *                                          |
 *                                    <audio> sink element   <-- WebRTC renders
 *                                                               this, so the
 *                                                               canceller has
 *                                                               a reference
 *
 * A local loopback is a real peer connection with a real encode/decode on it,
 * and it costs about 20 ms of latency and a few percent of one core. That buys
 * every deterministic line the same echo cancellation the live session's own
 * voice already had.
 *
 * AND IT IS ALWAYS ALLOWED TO FAIL. The bus starts wired straight to
 * ctx.destination — audible, uncancelled, exactly what the page did before —
 * and only moves to the loopback once the loopback is up AND the sink element
 * is genuinely playing. Every caller keeps its old element path as a fallback
 * and takes it if anything here throws or has not started. A gap warning that
 * does not play is worse than a gap warning that echoes, and no rearrangement
 * of the audio graph is worth that trade.
 *
 * The other half of what this file is for: the ANALYSER. The echo gate in
 * rio_realtime.js needs to know how loud what we are rendering is, in order to
 * ask whether the microphone is beating it. That measurement only exists if
 * everything goes through one node, which is this one.
 */
(function (root) {
  'use strict';

  var ctx = null;
  var bus = null;             // everything connects into this
  var analyser = null;        // ...and it is measured here
  var dest = null;            // MediaStreamDestination feeding the loopback
  var sinkEl = null;          // where the loopback comes back out
  var pcA = null, pcB = null;
  var covered = false;        // is the loopback carrying the audio?
  var toDestination = false;  // ...or is the bus wired straight out?
  var building = null;
  var buffers = {};           // url -> decoded AudioBuffer
  var stats = { plays: 0, decoded: 0, fallbacks: 0, bus_failures: 0 };
  var scratch = null;         // reused Float32Array for the meter

  function AC() { return root.AudioContext || root.webkitAudioContext; }

  function haveWebAudio() { return !!AC(); }

  /* The context is shared with anything else on the page that needs one --
     rio_voice_eleven.js takes this one rather than opening a second. Two
     contexts on iOS is two audio units, and the second one is not covered by
     anything. */
  function context() {
    if (ctx) return ctx;
    var C = AC();
    if (!C) return null;
    try { ctx = new C(); } catch (e) { ctx = null; }
    return ctx;
  }

  /* The graph, built once. Wired to ctx.destination to begin with: that is
     what the page has always done and it is audible with no further steps. */
  function ensure() {
    if (bus) return bus;
    var c = context();
    if (!c) return null;
    try {
      bus = c.createGain();
      bus.gain.value = 1;
      analyser = c.createAnalyser();
      analyser.fftSize = 1024;
      analyser.smoothingTimeConstant = 0.2;
      bus.connect(analyser);
      bus.connect(c.destination);
      toDestination = true;
    } catch (e) {
      bus = null; analyser = null;
      stats.bus_failures++;
    }
    return bus;
  }

  /* ---- the loopback ------------------------------------------------------
     Two peer connections, offered and answered against each other in the same
     page. Nothing leaves the device. */
  function startLoopback() {
    if (covered || building) return building || Promise.resolve(covered);
    if (!root.RTCPeerConnection || !ensure()) return Promise.resolve(false);
    var c = context();
    building = new Promise(function (resolve) {
      try {
        dest = c.createMediaStreamDestination();
        pcA = new root.RTCPeerConnection();
        pcB = new root.RTCPeerConnection();
        pcA.onicecandidate = function (e) {
          if (e.candidate) { try { pcB.addIceCandidate(e.candidate); } catch (x) {} }
        };
        pcB.onicecandidate = function (e) {
          if (e.candidate) { try { pcA.addIceCandidate(e.candidate); } catch (x) {} }
        };
        pcB.ontrack = function (e) {
          if (!sinkEl) return;
          sinkEl.srcObject = e.streams[0];
          var p = sinkEl.play();
          if (p && p.catch) p.catch(function () {});
        };
        dest.stream.getAudioTracks().forEach(function (t) {
          pcA.addTrack(t, dest.stream);
        });
        pcA.createOffer()
          .then(function (offer) {
            return pcA.setLocalDescription(offer)
              .then(function () { return pcB.setRemoteDescription(offer); });
          })
          .then(function () { return pcB.createAnswer(); })
          .then(function (answer) {
            return pcB.setLocalDescription(answer)
              .then(function () { return pcA.setRemoteDescription(answer); });
          })
          .then(function () {
            /* THE SWITCH, and it is deliberately the last thing that happens.
               Until the sink element is actually playing, moving the bus off
               ctx.destination would make RIO silent rather than cancelled. */
            return settled(sinkEl);
          })
          .then(function (playing) {
            if (!playing) { resolve(false); return; }
            try {
              if (toDestination) { bus.disconnect(c.destination); toDestination = false; }
              bus.connect(dest);
              covered = true;
            } catch (e) { covered = false; }
            resolve(covered);
          })
          .catch(function () { resolve(false); });
      } catch (e) { resolve(false); }
    }).then(function (ok) { building = null; return ok; });
    return building;
  }

  /* Is the sink element really carrying audio? `play()` resolving is not on
     its own enough on iOS -- an element can resolve and then be paused by the
     audio session. One frame later, and then check. */
  function settled(el) {
    if (!el) return Promise.resolve(false);
    return new Promise(function (resolve) {
      var tries = 0;
      var check = function () {
        tries++;
        if (el.srcObject && !el.paused) { resolve(true); return; }
        if (tries > 20) { resolve(false); return; }
        root.setTimeout(check, 25);
      };
      check();
    });
  }

  /* ---- the gesture -------------------------------------------------------
     iOS starts an AudioContext suspended and will not resume it outside a user
     gesture, and will not play the sink element outside one either. Every
     audio path on this page already has an unlock() called from Start Drive
     and from the mic tap; this is the same thing for the bus. */
  function unlock(el) {
    var c = context();
    if (!c) return Promise.resolve(false);
    if (el && !sinkEl) sinkEl = el;
    if (!sinkEl) {
      // No element was given. Make one: it is never seen and never controlled.
      try {
        sinkEl = root.document.createElement('audio');
        sinkEl.autoplay = true;
        sinkEl.setAttribute('playsinline', '');
        sinkEl.style.display = 'none';
        root.document.body.appendChild(sinkEl);
      } catch (e) { sinkEl = null; }
    }
    ensure();
    var resumed = (c.state === 'running')
      ? Promise.resolve()
      : (c.resume ? c.resume().catch(function () {}) : Promise.resolve());
    return resumed.then(function () { return startLoopback(); });
  }

  /* Anything that suspends the context — a phone call, a route change, the app
     going to the background — leaves every attached source silent, which for
     this page means a warning that does not play. So it is resumed on the way
     back, and before every play. */
  function wake() {
    var c = ctx;
    if (c && c.state === 'suspended' && c.resume) {
      try { c.resume(); } catch (e) {}
    }
  }

  if (root.document && root.document.addEventListener) {
    root.document.addEventListener('visibilitychange', function () {
      if (root.document.visibilityState === 'visible') wake();
    });
  }

  /* ---- playing a file through the bus ------------------------------------
     Decoded once and kept: the clips are a handful of short files and the
     whole point of them is that firing one costs nothing. Returns the same
     {play, abort} shape rio_speak.js already uses for elements, so a caller
     falls back by using the element path it already has. */
  function playUrl(url, fetchImpl) {
    var stopped = false;
    var node = null;
    var settle = null;
    return {
      abort: function () {
        stopped = true;
        /* Order matters: stopping the source fires its own onended, which runs
           after this returns and must find something to call. So the settle is
           taken, called, and only then dropped -- and `finish` below is
           idempotent, because both paths reach it. */
        if (node) { try { node.stop(); } catch (e) {} node = null; }
        var f = settle; settle = null;
        if (f) f();
      },
      play: function () {
        if (stopped) return Promise.resolve();
        if (!ready()) return Promise.reject(new Error('bus not ready'));
        wake();
        return decode(url, fetchImpl).then(function (buf) {
          if (stopped || !buf) return;
          return new Promise(function (resolve, reject) {
            try {
              var c = context();
              node = c.createBufferSource();
              node.buffer = buf;
              node.connect(bus);
              var done = false;
              var finish = function () { if (!done) { done = true; resolve(); } };
              settle = finish;
              node.onended = function () { node = null; finish(); };
              node.start();
              stats.plays++;
            } catch (e) { reject(e); }
          });
        });
      },
    };
  }

  /* One decode per URL, cached. Safari's decodeAudioData still wants the
     callback form; the promise form returns undefined there. */
  function decode(url, fetchImpl) {
    if (buffers[url]) return Promise.resolve(buffers[url]);
    var f = fetchImpl || root.fetch;
    if (!f) return Promise.reject(new Error('no fetch'));
    return f(url).then(function (r) {
      if (!r.ok) throw new Error('clip ' + r.status);
      return r.arrayBuffer();
    }).then(function (bytes) {
      var c = context();
      return new Promise(function (resolve, reject) {
        var ok = function (buf) { buffers[url] = buf; stats.decoded++; resolve(buf); };
        var bad = function (e) { reject(e || new Error('decode failed')); };
        var p;
        try { p = c.decodeAudioData(bytes, ok, bad); } catch (e) { bad(e); return; }
        if (p && p.then) p.then(ok, bad);
      });
    });
  }

  /* Play an already-decoded blob (a TTS response) the same way. Kept separate
     from playUrl because these are one-shot and must not be cached: every
     sentence is different. */
  function playBlob(blob) {
    var stopped = false, node = null, settle = null;
    return {
      abort: function () {
        stopped = true;
        if (node) { try { node.stop(); } catch (e) {} node = null; }
        var f = settle; settle = null;
        if (f) f();
      },
      play: function () {
        if (stopped) return Promise.resolve();
        if (!ready()) return Promise.reject(new Error('bus not ready'));
        wake();
        return blob.arrayBuffer().then(function (bytes) {
          var c = context();
          return new Promise(function (resolve, reject) {
            var start = function (buf) {
              if (stopped) { resolve(); return; }
              try {
                node = c.createBufferSource();
                node.buffer = buf;
                node.connect(bus);
                var done = false;
                var finish = function () { if (!done) { done = true; resolve(); } };
                settle = finish;
                node.onended = function () { node = null; finish(); };
                node.start();
                stats.plays++;
              } catch (e) { reject(e); }
            };
            var p;
            try { p = c.decodeAudioData(bytes, start, reject); }
            catch (e) { reject(e); return; }
            if (p && p.then) p.then(start, reject);
          });
        });
      },
    };
  }

  /* Usable at all? The bus exists, the context is running, and the loopback
     either took or fell back to a destination that is at least audible. A
     suspended context is NOT ready: a source attached to it makes no sound,
     and a caller told "ready" would have played a warning into silence. */
  function ready() {
    if (!bus) return false;
    var c = ctx;
    return !!(c && c.state === 'running');
  }

  /* ---- the meter ---------------------------------------------------------
     RMS of everything the bus is rendering, in dBFS. This is the reference the
     echo gate compares the microphone against; see the level test in
     rio_realtime.js. Silence reads -100 rather than -Infinity so arithmetic on
     it stays arithmetic. */
  function level() {
    if (!analyser) return -100;
    var n = analyser.fftSize;
    if (!scratch || scratch.length !== n) scratch = new Float32Array(n);
    try {
      if (analyser.getFloatTimeDomainData) analyser.getFloatTimeDomainData(scratch);
      else return -100;
    } catch (e) { return -100; }
    var sum = 0;
    for (var i = 0; i < n; i++) sum += scratch[i] * scratch[i];
    var rms = Math.sqrt(sum / n);
    if (!(rms > 0)) return -100;
    var db = 20 * Math.log10(rms);
    return db < -100 ? -100 : db;
  }

  root.RIO = root.RIO || {};
  root.RIO.output = {
    unlock: unlock,
    ready: ready,
    covered: function () { return covered; },
    context: context,
    /* The node an external source connects into — the ElevenLabs sink takes
       this instead of ctx.destination, which is what puts RIO's synthesised
       voice inside the canceller too. */
    node: function () { return ensure(); },
    playUrl: playUrl,
    playBlob: playBlob,
    level: level,
    stats: function () { return stats; },
    noteFallback: function () { stats.fallbacks++; },
    state: function () {
      return { covered: covered, ready: ready(),
               context: ctx ? ctx.state : 'none',
               to_destination: toDestination,
               decoded: Object.keys(buffers).length,
               stats: stats };
    },
    /* Tests only: forget everything and start again. */
    _reset: function () {
      try { if (pcA) pcA.close(); if (pcB) pcB.close(); } catch (e) {}
      ctx = bus = analyser = dest = sinkEl = pcA = pcB = null;
      covered = false; toDestination = false; building = null; buffers = {};
      stats = { plays: 0, decoded: 0, fallbacks: 0, bus_failures: 0 };
    },
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = root.RIO.output;
  }
})(typeof window !== 'undefined' ? window : globalThis);
