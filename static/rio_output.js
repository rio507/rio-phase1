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
 *
 * ---------------------------------------------------------------------------
 * TWO CLOCKS, AND THE CRACKLE THAT LIVES BETWEEN THEM
 * ---------------------------------------------------------------------------
 * The complaint after the drive of 2026-09-09 was that the iPhone speaker
 * starts crackling after a few minutes. LIVE_BUS_HEALTH was added for exactly
 * that question and reported ONCE in 656 seconds:
 *
 *     t=15.6   covered: true, context: "running", to_destination: false,
 *              bus_failures: 0, fallbacks: 0
 *
 * It is reported on change, so one event over a whole drive is a clean answer
 * to all four of the questions it asks: the loopback held, the context never
 * suspended, the bus never failed and never fell back. Which rules out
 * everything this file could previously see, and leaves the thing it could
 * not.
 *
 * THE LOOPBACK HAS TWO INDEPENDENT CLOCKS IN IT. The AudioContext renders on
 * its own clock; the <audio> element plays out on the audio DEVICE's clock.
 * They are never the same clock and they are never quite the same rate. A
 * receiver absorbs that in its jitter buffer, and a jitter buffer with a
 * steady bias in one direction either grows without bound or runs dry — and a
 * jitter buffer running dry is concealment, and concealment on speech is
 * exactly the sound being described.
 *
 * It takes MINUTES, which is the shape of the complaint: a few parts per
 * million of drift is inaudible for the first sixty seconds and is a hundred
 * milliseconds of accumulated error by the fifth.
 *
 * So: MEASURE IT, then FIX IT WITHOUT EVER GOING SILENT.
 *
 *   measure   pcB.getStats() carries the answer directly, and nothing was
 *             reading it: concealedSamples (minus silentConcealedSamples,
 *             which is the part nobody can hear), concealmentEvents,
 *             insertedSamplesForDeceleration / removedSamplesForAcceleration
 *             (the receiver stretching audio to keep up, which IS the
 *             artefact), and jitterBufferDelay / jitterBufferEmittedCount,
 *             whose ratio is the buffer's depth and whose GROWTH is the drift.
 *             Plus ctx.currentTime against sink.currentTime, which is the two
 *             clocks read side by side.
 *
 *   fix       rebuild the loopback when concealment starts or the buffer has
 *             drifted past a bound. A rebuild resets the jitter buffer, which
 *             is the whole of the cure.
 *
 *   silently  MAKE BEFORE BREAK, and this is why there are two sink elements
 *             rather than one. Both are primed inside the Start Drive gesture,
 *             because iOS will not start an element from a timer that has
 *             never been played inside a tap — so a resync that created a new
 *             element would be a resync that left her mute. The new loopback
 *             is built and PROVED playing on the spare element before the bus
 *             moves onto it, and only then is the old one torn down.
 *
 * ...and the sample rate is asked for rather than accepted. WebRTC's opus path
 * is 48 kHz; a context at 44.1 puts a resampler in front of the encoder and
 * another behind the decoder. Requesting 48000 removes the first one, and on a
 * device that cannot give it the constructor simply returns what it has, which
 * is what the old code got anyway.
 */
(function (root) {
  'use strict';

  var ctx = null;
  var bus = null;             // everything connects into this
  var analyser = null;        // ...and it is measured here
  var link = null;            // the live loopback: {dest, pcA, pcB, el, since}
  var spare = null;           // ...and the one being built to replace it
  /* TWO SINK ELEMENTS, BOTH PRIMED IN THE GESTURE. iOS will not start an
     element from a timer unless that element has been played once inside a
     tap, so a resync that MADE a new element would be a resync that left her
     mute. These are made and primed together in unlock(), and the loopback
     alternates between them. */
  var sinks = [null, null];
  var activeSink = 0;
  var covered = false;        // is the loopback carrying the audio?
  var toDestination = false;  // ...or is the bus wired straight out?
  var building = null;
  var buffers = {};           // url -> decoded AudioBuffer
  var stats = { plays: 0, decoded: 0, fallbacks: 0, bus_failures: 0,
                resyncs: 0, resync_failures: 0 };
  var scratch = null;         // reused Float32Array for the meter
  /* HOW MANY SOURCES ARE ATTACHED TO THE BUS RIGHT NOW, and it exists for one
     reason: a resync while she is mid-sentence would put a seam in the middle
     of the sentence, which is the fault being fixed arriving from the other
     direction. Balanced through `held()` rather than by hand, because a leak
     here does not crash anything -- it silently disables the resync for the
     rest of the drive, which is the worst way for a fix to fail. */
  var playing = 0;

  function held() {
    var done = false;
    playing++;
    return function () {
      if (done) return;
      done = true;
      playing = Math.max(0, playing - 1);
    };
  }

  /* --- what the loopback is doing to her voice ---------------------------
     Sampled from pcB.getStats() and from the two clocks, and CUMULATIVE
     across resyncs so a drive's total is a drive's total. See health(). */
  var health = {
    sample_rate: null, base_latency_ms: null, output_latency_ms: null,
    // The receiver stretching or dropping audio to keep up. This IS the
    // artefact: `concealed` is samples invented to cover a buffer that ran
    // dry, minus the silent ones nobody can hear.
    /* PER-LINK, because getStats() belongs to the peer connection and a
       resync makes a new one. `_total` is the drive's number; the bare one is
       "since this loopback came up", which is what the resync trigger has to
       read or it would be triggering on history. The 10-minute soak of
       2026-09-10 reported 12,010 concealed samples where the true total was
       an order of magnitude more, because nothing carried the count across
       the twenty-five rebuilds it had done. */
    concealed: 0, concealed_total: 0,
    concealment_events: 0, concealment_events_total: 0,
    inserted: 0, removed: 0, inserted_total: 0, removed_total: 0,
    total_samples: 0, packets_lost: 0,
    // Depth of the jitter buffer now, and how far it has moved since the
    // loopback came up. Growth in either direction is the drift.
    jitter_ms: null, jitter_ms_at_start: null, jitter_growth_ms: 0,
    /* DRIFT, MEASURED BY THE THING THAT EXPERIENCES IT.
     *
     * The obvious measurement — ctx.currentTime against sinkEl.currentTime —
     * is a reading of the OUTPUT DEVICE's clock, and it is only a measurement
     * where there is an output device. In headless Chromium the element clock
     * does not advance against the context's at all and the figure comes out
     * as -1,000,000 ppm, which is not drift, it is the absence of a speaker.
     * It is kept as `clock_delta_ms` because on a phone it is the real thing,
     * and it is not what anything asserts.
     *
     * What IS asserted is the receiver's own accounting: every sample it had
     * to invent to slow down (insertedSamplesForDeceleration) or discard to
     * speed up (removedSamplesForAcceleration) is a sample of accumulated
     * clock difference it had to absorb. The net, in milliseconds, is the
     * drift — measured by the jitter buffer, which is the component the drift
     * actually happens to, and available wherever getStats is.
     */
    drift_ms: 0, drift_ppm: 0,
    clock_delta_ms: null, clock_delta_ppm: null,
    since_s: 0, samples: 0
  };
  var baseline = null;        // per-loopback zero for the two measurements above
  /* What every loopback BEFORE this one contributed, so the drive's totals
     survive a rebuild. */
  var carried = { concealed: 0, concealment_events: 0, inserted: 0, removed: 0 };

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
    /* ASKED FOR, NOT ASSUMED. The loopback encodes opus at 48 kHz whatever
       this context runs at, so a context at 44.1 puts a resampler in front of
       the encoder and another behind the decoder — two conversions on every
       sample of her voice, for nothing. A device that cannot give 48 kHz
       throws or ignores it, and the fallback is exactly what this line used to
       do unconditionally. */
    try { ctx = new C({ sampleRate: 48000 }); } catch (e) { ctx = null; }
    if (!ctx) { try { ctx = new C(); } catch (e2) { ctx = null; } }
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
     page. Nothing leaves the device.

     BUILT AS A UNIT, ONTO A NAMED SINK, so that a second one can be built
     alongside the first and the bus moved across without a gap. `startLoopback`
     is the first build; `resync` is every one after it. */
  function buildLink(el) {
    var c = context();
    if (!root.RTCPeerConnection || !c || !el) return Promise.resolve(null);
    var l = { dest: null, pcA: null, pcB: null, el: el, since: 0 };
    return new Promise(function (resolve) {
      try {
        l.dest = c.createMediaStreamDestination();
        l.pcA = new root.RTCPeerConnection();
        l.pcB = new root.RTCPeerConnection();
        l.pcA.onicecandidate = function (e) {
          if (e.candidate) { try { l.pcB.addIceCandidate(e.candidate); } catch (x) {} }
        };
        l.pcB.onicecandidate = function (e) {
          if (e.candidate) { try { l.pcA.addIceCandidate(e.candidate); } catch (x) {} }
        };
        l.pcB.ontrack = function (e) {
          l.el.srcObject = e.streams[0];
          var p = l.el.play();
          if (p && p.catch) p.catch(function () {});
        };
        l.dest.stream.getAudioTracks().forEach(function (t) {
          l.pcA.addTrack(t, l.dest.stream);
        });
        l.pcA.createOffer()
          .then(function (offer) {
            return l.pcA.setLocalDescription(offer)
              .then(function () { return l.pcB.setRemoteDescription(offer); });
          })
          .then(function () { return l.pcB.createAnswer(); })
          .then(function (answer) {
            return l.pcB.setLocalDescription(answer)
              .then(function () { return l.pcA.setRemoteDescription(answer); });
          })
          .then(function () {
            /* PROVED PLAYING BEFORE ANYTHING MOVES. Until the sink element is
               actually playing, moving the bus onto it would make RIO silent
               rather than cancelled. */
            return settled(l.el);
          })
          .then(function (ok) {
            if (!ok) { fail(l, null, 'sink_never_played'); resolve(null); return; }
            l.since = nowMs();
            resolve(l);
          })
          .catch(function (err) { fail(l, err, 'negotiate'); resolve(null); });
      } catch (e) { fail(l, e, 'build'); resolve(null); }
    });
  }

  /* WHY A LOOPBACK DID NOT COME UP, kept rather than swallowed.
     Every failure here used to resolve `null` and leave the page on
     ctx.destination -- audible, uncancelled, and with nothing anywhere saying
     which of five steps had failed. `bus_failures` counted only the graph. */
  var lastFailure = null;

  function fail(l, err, at) {
    stats.bus_failures++;
    lastFailure = { at: at, error: err ? String(err.message || err) : null,
                    when: nowMs() };
    emit('loopback_failed', lastFailure);
    dropLink(l);
  }

  /* Is the sink element really carrying audio? `play()` resolving is not on
     its own enough on iOS -- an element can resolve and then be paused by the
     audio session. One frame later, and then check.

     A SECOND AND A HALF, not half a second. `ontrack` fires once the answer
     has been applied AND the receiver has produced a track, which on this
     stack is around 600 ms from the offer -- past the old 500 ms budget. A
     loopback that reports failure on a budget leaves the bus on
     ctx.destination: audible, and uncancelled, which is the one thing this
     file exists to stop. Nothing waits on this -- the bus is already wired to
     a working output while it runs -- so the budget costs only patience. */
  function settled(el) {
    if (!el) return Promise.resolve(false);
    return new Promise(function (resolve) {
      var tries = 0;
      var check = function () {
        tries++;
        if (el.srcObject && !el.paused) { resolve(true); return; }
        if (tries > 60) { resolve(false); return; }
        root.setTimeout(check, 25);
      };
      check();
    });
  }

  function dropLink(l) {
    if (!l) return;
    try { if (l.pcA) l.pcA.close(); } catch (e) {}
    try { if (l.pcB) l.pcB.close(); } catch (e) {}
    try { if (l.el) { l.el.pause(); l.el.srcObject = null; } } catch (e) {}
    try { if (l.dest) l.dest.disconnect(); } catch (e) {}
  }

  function nowMs() {
    return (root.performance && root.performance.now)
      ? root.performance.now() : Date.now();
  }

  function startLoopback() {
    if (covered || building) return building || Promise.resolve(covered);
    if (!ensure()) return Promise.resolve(false);
    var c = context();
    building = buildLink(sinks[activeSink]).then(function (l) {
      if (!l) return false;
      try {
        if (toDestination) { bus.disconnect(c.destination); toDestination = false; }
        bus.connect(l.dest);
        link = l;
        covered = true;
        baseline = null;                 // a new loopback is a new zero
        health.jitter_ms_at_start = null;
        startWatch();
      } catch (e) { dropLink(l); covered = false; }
      return covered;
    }).then(function (ok) { building = null; return ok; });
    return building;
  }

  /* ---- resync: the crackle, cured --------------------------------------
   *
   * MAKE BEFORE BREAK. A second loopback is built on the spare sink and proved
   * playing; only then does the bus move onto it, and only then is the old one
   * closed. At no point is the bus connected to nothing.
   *
   * NEVER MID-SENTENCE. `playing` counts what is attached to the bus, and a
   * resync while a source is running would put a seam in the middle of a
   * sentence — which is the thing being fixed, arriving from the other
   * direction. The watch simply tries again on the next tick; a car is not
   * speaking most of the time.
   */
  function resync(why) {
    if (!covered || !link || spare || !bus) return Promise.resolve(false);
    if (playing > 0) return Promise.resolve(false);
    var c = context();
    if (!c || c.state !== 'running') return Promise.resolve(false);
    var el = sinks[1 - activeSink];
    if (!el) return Promise.resolve(false);
    var old = link;
    spare = buildLink(el).then(function (l) {
      spare = null;
      if (!l) {
        stats.resync_failures++;
        emit('resync_failed', { why: why });
        return false;
      }
      try {
        bus.connect(l.dest);             // both live for an instant
        bus.disconnect(old.dest);
        link = l;
        activeSink = 1 - activeSink;
        stats.resyncs++;
        // The old connection's counters go into the running totals before its
        // stats object stops existing.
        carried.concealed += health.concealed;
        carried.concealment_events += health.concealment_events;
        carried.inserted += health.inserted;
        carried.removed += health.removed;
        health.concealed = 0; health.concealment_events = 0;
        health.inserted = 0; health.removed = 0;
        baseline = null;
        health.jitter_ms_at_start = null;
        emit('resync', { why: why, resyncs: stats.resyncs,
                         backoff_ms: Math.round(backoffMs) });
      } catch (e) {
        dropLink(l);
        stats.resync_failures++;
        return false;
      }
      dropLink(old);
      return true;
    });
    return spare;
  }

  /* ---- the health watch -------------------------------------------------
   *
   * ONE SAMPLE A SECOND, of the two things that were invisible: what the
   * receiver is doing to keep up, and how far the two clocks have moved apart.
   * Both come out of interfaces that have always been there and that nothing
   * was reading.
   */
  var watch = null;
  var listeners = [];

  function emit(kind, detail) {
    for (var i = 0; i < listeners.length; i++) {
      try { listeners[i](kind, detail || {}); } catch (e) {}
    }
  }

  function startWatch() {
    if (watch !== null || !root.setInterval) return;
    watch = root.setInterval(function () { sample(); }, 1000);
  }

  function stopWatch() {
    if (watch === null) return;
    root.clearInterval(watch);
    watch = null;
  }

  /* THE TWO CLOCKS, READ SIDE BY SIDE. ctx.currentTime advances on the audio
     context's render clock; el.currentTime advances on the output device's.
     Their difference is constant only if the two rates are identical, which
     they never are. */
  function readClocks() {
    var c = ctx;
    var el = link && link.el;
    if (!c || !el) return null;
    var elT = 0;
    try { elT = el.currentTime || 0; } catch (e) { return null; }
    return { ctx: c.currentTime, el: elT, at: nowMs() };
  }

  function sample() {
    var c = ctx;
    if (!c || !link) return Promise.resolve(health);
    health.samples++;
    health.sample_rate = c.sampleRate || null;
    health.base_latency_ms = (typeof c.baseLatency === 'number')
      ? Math.round(c.baseLatency * 10000) / 10 : null;
    health.output_latency_ms = (typeof c.outputLatency === 'number')
      ? Math.round(c.outputLatency * 10000) / 10 : null;
    health.since_s = Math.round((nowMs() - link.since) / 100) / 10;

    var clocks = readClocks();
    if (clocks) {
      if (!baseline) {
        baseline = clocks;
      } else {
        var elapsed = clocks.at - baseline.at;
        var d = ((clocks.el - baseline.el) - (clocks.ctx - baseline.ctx)) * 1000;
        health.clock_delta_ms = Math.round(d * 10) / 10;
        health.clock_delta_ppm = elapsed > 0
          ? Math.round((d / elapsed) * 1e6) : 0;
      }
    }

    if (!link.pcB || !link.pcB.getStats) return Promise.resolve(health);
    return link.pcB.getStats().then(function (report) {
      report.forEach(function (r) {
        if (r.type !== 'inbound-rtp' || r.kind !== 'audio') return;
        /* CONCEALMENT IS THE CRACKLE, and `silentConcealedSamples` is the part
           of it nobody can hear: a buffer running dry during silence is
           covered by more silence. Subtracting it is the difference between
           "the receiver ran out" and "the receiver ran out DURING SPEECH". */
        var concealed = (r.concealedSamples || 0) - (r.silentConcealedSamples || 0);
        health.concealed = Math.max(0, concealed);
        health.concealment_events = r.concealmentEvents || 0;
        health.inserted = r.insertedSamplesForDeceleration || 0;
        health.removed = r.removedSamplesForAcceleration || 0;
        /* The drive's totals, carried over every rebuild. `carried` is what
           the links before this one contributed. */
        health.concealed_total = carried.concealed + health.concealed;
        health.concealment_events_total =
          carried.concealment_events + health.concealment_events;
        health.inserted_total = carried.inserted + health.inserted;
        health.removed_total = carried.removed + health.removed;
        health.total_samples = r.totalSamplesReceived || 0;
        health.packets_lost = r.packetsLost || 0;
        /* THE DRIFT, in milliseconds of audio the receiver had to invent or
           throw away to keep the two clocks together. Net, because inserting
           and then removing is a buffer that wandered and came back. */
        var rate = c.sampleRate || 48000;
        var net = (health.inserted_total - health.removed_total) / rate * 1000;
        health.drift_ms = Math.round(net * 10) / 10;
        health.drift_ppm = health.since_s > 0
          ? Math.round(net / (health.since_s * 1000) * 1e6) : 0;
        if (r.jitterBufferEmittedCount) {
          var depth = (r.jitterBufferDelay || 0) / r.jitterBufferEmittedCount;
          health.jitter_ms = Math.round(depth * 10000) / 10;
          if (health.jitter_ms_at_start === null) {
            health.jitter_ms_at_start = health.jitter_ms;
          }
          health.jitter_growth_ms =
            Math.round((health.jitter_ms - health.jitter_ms_at_start) * 10) / 10;
        }
      });
      maybeResync();
      return health;
    }, function () { return health; });
  }

  /* WHEN A REBUILD IS WORTH ITS SEAM. Two triggers, and both are cumulative
     rather than instantaneous — a single concealment event is a hiccup and
     five in a minute is the fault being described.

     `_thresholds` so a test can drive them; the defaults are what a drive
     uses. */
  var limits = { concealed_samples: 4800,      // 100 ms of invented audio
                 jitter_growth_ms: 120,        // buffer moved this far
                 drift_ms: 150,                // clocks this far apart
                 min_gap_ms: 30000,            // never more often than this
                 max_gap_ms: 600000 };         // ...and eventually, not at all
  var lastResyncAt = 0;
  var backoffMs = 0;
  var lastConcealRate = null;

  /* A CURE THAT IS NOT CURING STOPS BEING APPLIED.
   *
   * The 10-minute soak of 2026-09-10 did twenty-five resyncs — one every
   * thirty seconds, for the whole run — because this container conceals audio
   * at a steady environmental rate whatever the loopback does, and a trigger
   * on a total against a fixed floor will therefore fire forever. Each one is
   * a peer connection torn down and rebuilt; twenty-five of those is not a fix
   * for a crackle, it is a second source of them.
   *
   * So the gap DOUBLES whenever a resync fails to reduce the concealment rate,
   * up to ten minutes, and resets the moment one works. A drifting clock is
   * cured by a rebuild and the backoff never engages; a noisy output device is
   * not, and the mechanism gets out of its way.
   */
  function maybeResync() {
    if (!covered || playing > 0) return;
    var gap = limits.min_gap_ms + backoffMs;
    if (gap > limits.max_gap_ms) return;
    var since = nowMs() - (lastResyncAt || (link ? link.since : 0));
    if (since < gap) return;

    var rate = health.since_s > 0 ? (health.concealed / health.since_s) : 0;
    var why = null;
    if (health.concealed >= limits.concealed_samples) why = 'concealment';
    else if (Math.abs(health.jitter_growth_ms) >= limits.jitter_growth_ms) why = 'jitter_growth';
    else if (Math.abs(health.drift_ms) >= limits.drift_ms) why = 'clock_drift';
    if (!why) { lastConcealRate = rate; return; }

    if (why === 'concealment' && lastConcealRate !== null) {
      // Did the LAST rebuild help? If the rate is no better, this one will not
      // either, and the cost is a peer connection.
      if (rate >= lastConcealRate * 0.8) {
        backoffMs = backoffMs ? Math.min(backoffMs * 2, limits.max_gap_ms)
                              : limits.min_gap_ms;
        emit('resync_backoff', { rate: Math.round(rate),
                                 was: Math.round(lastConcealRate),
                                 backoff_ms: Math.round(backoffMs) });
      } else {
        backoffMs = 0;
      }
    }
    lastConcealRate = rate;
    lastResyncAt = nowMs();
    resync(why);
  }

  /* ---- the gesture -------------------------------------------------------
     iOS starts an AudioContext suspended and will not resume it outside a user
     gesture, and will not play the sink element outside one either. Every
     audio path on this page already has an unlock() called from Start Drive
     and from the mic tap; this is the same thing for the bus. */
  function makeSink() {
    try {
      var a = root.document.createElement('audio');
      a.autoplay = true;
      a.setAttribute('playsinline', '');
      a.style.display = 'none';
      root.document.body.appendChild(a);
      return a;
    } catch (e) { return null; }
  }

  function unlock(el) {
    var c = context();
    if (!c) return Promise.resolve(false);
    if (el && !sinks[0]) sinks[0] = el;
    if (!sinks[0]) sinks[0] = makeSink();
    /* THE SPARE, MADE HERE AND NOWHERE ELSE. It exists so a resync has an
       element that has already been played inside this gesture; one created
       later, from a timer, is one iOS will not start. */
    if (!sinks[1]) sinks[1] = makeSink();
    ensure();
    var resumed = (c.state === 'running')
      ? Promise.resolve()
      : (c.resume ? c.resume().catch(function () {}) : Promise.resolve());
    return resumed.then(function () { return startLoopback(); });
  }

  /* Both sinks, played once, from inside the tap. index.html's unlock pass
     calls this next to every other element it primes; without it the spare is
     an element iOS refuses to start when a resync needs it three minutes
     later, and the resync silently fails every time. */
  function primeSinks(silentUrl) {
    var n = 0;
    for (var i = 0; i < sinks.length; i++) {
      var a = sinks[i];
      if (!a) continue;
      /* A SINK ALREADY CARRYING THE LOOPBACK IS ALREADY UNLOCKED: it is
         playing, which is how it got there. Touching its source would take the
         audio away from it. */
      if (a.srcObject) continue;
      try {
        /* ON SILENCE, like every other element this page primes. An element
           played with no source at all is not reliably unlocked on iOS, and
           one played on its real contents is a warning that escapes during the
           tap — which this page has been caught doing once already. */
        if (silentUrl) {
          a.muted = true;
          try { a.removeAttribute('src'); a.load(); } catch (x) {}
          a.src = silentUrl;
        }
        var p = a.play();
        if (p && p.catch) p.catch(function () {});
        n++;
        /* ...AND THEN LET IT GO. An element left holding a decoded buffer is
           an element that will play it the next time anything calls play() on
           it, and the next thing to call play() on this one is a resync
           attaching a MediaStream. Released on the next task, after the play
           that does the unlocking has been accepted. */
        (function (el) {
          root.setTimeout(function () {
            try {
              if (el.srcObject) return;         // a loopback got there first
              el.pause();
              el.removeAttribute('src');
              el.load();
              el.muted = false;
            } catch (x) {}
          }, 0);
        }(a));
      } catch (e) {}
    }
    return n;
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

  /* COMING BACK FROM THE BACKGROUND is where the two clocks are furthest
     apart: iOS suspends the context, the element's own clock does whatever the
     audio session did, and the jitter buffer has whatever is left in it. The
     baseline is meaningless across that gap, so it is dropped rather than
     turned into a drift figure that measures a suspension. */
  function reclock(why) {
    baseline = null;
    health.jitter_ms_at_start = null;
    lastResyncAt = 0;                 // a resync is allowed immediately
    backoffMs = 0;                    // ...and the backoff starts again: this
    lastConcealRate = null;           // is a different audio route

    emit('reclock', { why: why || 'wake' });
  }

  if (root.document && root.document.addEventListener) {
    root.document.addEventListener('visibilitychange', function () {
      if (root.document.visibilityState === 'visible') { wake(); reclock('visible'); }
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
    var release = null;
    return {
      abort: function () {
        stopped = true;
        if (release) { release(); release = null; }
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
              var finish = function () {
                if (release) { release(); release = null; }
                if (!done) { done = true; resolve(); }
              };
              settle = finish;
              node.onended = function () { node = null; finish(); };
              release = held();
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
    var stopped = false, node = null, settle = null, release = null;
    return {
      abort: function () {
        stopped = true;
        if (release) { release(); release = null; }
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
                var finish = function () {
                  if (release) { release(); release = null; }
                  if (!done) { done = true; resolve(); }
                };
                settle = finish;
                node.onended = function () { node = null; finish(); };
                release = held();
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
               last_failure: lastFailure,
               decoded: Object.keys(buffers).length,
               stats: stats,
               /* WHAT THE LOOPBACK IS DOING TO HER VOICE. Sampled once a
                  second while covered; see the health block above. Reported by
                  index.html next to the four booleans that were all this
                  could say on 2026-09-09. */
               health: {
                 sample_rate: health.sample_rate,
                 base_latency_ms: health.base_latency_ms,
                 output_latency_ms: health.output_latency_ms,
                 concealed: health.concealed,
                 concealment_events: health.concealment_events,
                 inserted: health.inserted,
                 removed: health.removed,
                 total_samples: health.total_samples,
                 packets_lost: health.packets_lost,
                 jitter_ms: health.jitter_ms,
                 jitter_growth_ms: health.jitter_growth_ms,
                 drift_ms: health.drift_ms,
                 drift_ppm: health.drift_ppm,
                 clock_delta_ms: health.clock_delta_ms,
                 clock_delta_ppm: health.clock_delta_ppm,
                 since_s: health.since_s,
                 samples: health.samples,
                 concealed_total: health.concealed_total,
                 concealment_events_total: health.concealment_events_total,
                 resyncs: stats.resyncs,
                 resync_failures: stats.resync_failures,
                 resync_backoff_ms: Math.round(backoffMs),
                 playing: playing
               } };
    },
    /* Both sinks played from inside the tap. See primeSinks. */
    prime: primeSinks,
    /* One health sample, now, rather than at the next tick. The 10-minute
       headless run drives this; a car uses the interval. */
    sample: function () { return Promise.resolve(sample()); },
    health: function () { return health; },
    /* Rebuild the loopback deliberately. `why` goes into the event. */
    resync: resync,
    onEvent: function (fn) { if (typeof fn === 'function') listeners.push(fn); },
    /* Tests only: drive the resync triggers at a scale a suite can reach. */
    _limits: function (o) {
      for (var k in (o || {})) if (limits[k] !== undefined) limits[k] = o[k];
      return limits;
    },
    /* Tests only: forget everything and start again. */
    _reset: function () {
      stopWatch();
      dropLink(link);
      link = null; spare = null;
      ctx = bus = analyser = null;
      sinks = [null, null]; activeSink = 0;
      covered = false; toDestination = false; building = null; buffers = {};
      playing = 0; baseline = null; lastResyncAt = 0; listeners = [];
      backoffMs = 0; lastConcealRate = null;
      carried = { concealed: 0, concealment_events: 0, inserted: 0, removed: 0 };
      stats = { plays: 0, decoded: 0, fallbacks: 0, bus_failures: 0,
                resyncs: 0, resync_failures: 0 };
      health = { sample_rate: null, base_latency_ms: null,
                 output_latency_ms: null, concealed: 0, concealment_events: 0,
                 inserted: 0, removed: 0, total_samples: 0, packets_lost: 0,
                 concealed_total: 0, concealment_events_total: 0,
                 inserted_total: 0, removed_total: 0,
                 jitter_ms: null, jitter_ms_at_start: null,
                 jitter_growth_ms: 0, drift_ms: 0, drift_ppm: 0,
                 clock_delta_ms: null, clock_delta_ppm: null,
                 since_s: 0, samples: 0 };
    },
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = root.RIO.output;
  }
})(typeof window !== 'undefined' ? window : globalThis);
