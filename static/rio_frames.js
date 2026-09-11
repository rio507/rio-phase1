/* rio_frames.js — the live frame transport. One socket, a pipe kept exactly as
 * full as frame age allows, and a frame that knows how old it is.
 *
 * WHAT THE FIRST REAL DRIVE MEASURED, and why this file exists.
 *
 * Session 06af3214: an iPhone, 607.8 s, 1949 frames through /headway_frame.
 *
 *   frames arrived            258 ms apart at p50, 406 at p90, 1503 at p99
 *   the server spent          23.6 ms on each of them (p90 33.5)
 *   frames carrying a
 *     capture timestamp       0 of 1949
 *
 * The middle line says the server was never the problem. The first says the
 * page's own 250 ms floor was almost the whole of the frame rate. The third is
 * the one that mattered most: nothing in the system knew how old a picture was
 * when the detector ran on it, so "is this warning about the road we are on"
 * had no answer at all — only "how long ago did this frame turn up".
 *
 * So, three changes, and no change to what a frame means:
 *
 *   ONE CONNECTION.   A persistent binary WebSocket instead of one HTTP POST
 *                     per picture. The JPEG goes on the wire as bytes, behind
 *                     a small JSON header, with no multipart envelope wrapped
 *                     around a container that already is one.
 *
 *   EVERY FRAME IS STAMPED.  With the instant it was captured, converted into
 *                     the SERVER's clock first (see clock sync below), so the
 *                     server can subtract and report a real frame age rather
 *                     than the browser guessing from a round trip that also
 *                     contains the answer coming back.
 *
 *   DROP, NEVER QUEUE.  The rule the whole file is built around. A tick that
 *                     finds the pipe full, or the socket's send buffer backed
 *                     up, takes NO PICTURE AT ALL and comes back at the next
 *                     tick. A queued frame is a frame that will be measured
 *                     against a dt that has already passed, and the moment
 *                     frames start queueing, frame age grows without bound —
 *                     which is the one failure this transport is not allowed
 *                     to have.
 *
 * "The pipe" is a DEPTH, and it is frame age that sets it. With strictly one
 * frame on the wire the achievable rate is 1/(serialise + round trip), which
 * on the link the first drive was measured over is about 4 fps however fast
 * the loop ticks — measured: 161 ms p50 frame age at 3.85 fps, with 182
 * captures skipped. Perfect freshness at the old frame rate. So the depth is
 * allowed to grow to at most three, and ONLY while measured age is sitting
 * comfortably under target; one frame over the ceiling collapses it back to
 * one on the spot. Nothing accumulates on the far end either — the server
 * holds exactly one waiting frame and a newer one evicts it.
 *
 * The rate is adaptive between the bounds the server sends (8–15 fps) and so
 * is JPEG quality, and all three knobs answer to the SERVER-REPORTED frame
 * age, not to a round-trip time: the number being controlled is the number
 * the driver's safety depends on. When it climbs, depth goes first, then
 * rate, then quality — because a shallower pipe costs nothing a Kalman filter
 * can feel and a blurrier car ahead does.
 *
 * No policy lives here. This file moves pictures and reports numbers; what to
 * say about them is decided server-side, exactly as before.
 *
 * ---------------------------------------------------------------------------
 * AND THEN, ON 2026-09-09, IT STOPPED AND NEVER STARTED AGAIN
 * ---------------------------------------------------------------------------
 * Session 738fbb82. The last frame the socket carried was idx 232 at t=40.4 s.
 * The next result in the whole drive was idx 233 at t=482.7 -- a 442-SECOND
 * GAP, in the middle of a drive that ran to 656 s.
 *
 * What the log says about it is mostly what it does NOT say:
 *
 *   headway_ws_close     t=400.4, received 233, processed 232. The socket
 *                        stayed OPEN for six minutes after the last frame.
 *   FRAMES_WS_LOST       never emitted. onclose did not fire.
 *   FRAMES_STALE         never emitted.
 *   frame_age_ms         220 -> 469 -> 506 over the last three frames, with
 *                        transport_ms going 148 -> 401.
 *   the first frame back  frame_age_ms 1888.8, described as current.
 *
 * So this was not a network drop and not a server close. THE CAPTURE LOOP
 * DIED, with the socket open and the page alive, and nothing anywhere was
 * watching for that.
 *
 * It could die because of exactly one line: `schedule()` was called at the END
 * of `tick()`, after `await once()`. A promise inside once() that never
 * settles -- and canvas.toBlob on iOS does not fire its callback while the
 * page is backgrounded, which is what a phone does at a red light when the
 * driver glances at Maps -- means the next tick is never scheduled. Not late:
 * never. The loop is one awaited promise away from being over for the drive,
 * and there was no watchdog, no timeout, no reconnect and nothing on the HUD.
 *
 * Four things, and the first is the load-bearing one:
 *
 *   THE LOOP CANNOT DIE.     schedule() is in a `finally`, and the work inside
 *                            a tick races a deadline. A tick that overruns is
 *                            abandoned and counted; the loop goes on.
 *   A WATCHDOG WATCHES IT.   Nothing sent for stall_ms while running is a
 *                            stall, and a stall REBUILDS the pipeline --
 *                            socket, canvas and all -- rather than hoping.
 *   THE SOCKET IS PROVED.    Pings were already going out and nothing read the
 *                            pongs. Three intervals with no pong is a dead
 *                            socket however open it looks, and reconnects
 *                            reset once one has carried frames for a while, so
 *                            three drops over ten minutes no longer retire the
 *                            transport for the rest of the drive.
 *   THE DRIVER IS TOLD.      onState fires on every change between live, lost
 *                            and reconnecting, and the HUD says so.
 */
(function (root) {
  'use strict';

  /* Only used before the server's own tuning arrives — one tick's worth. */
  var FALLBACK_TUNING = {
    min_fps: 8, max_fps: 15, start_fps: 10,
    target_age_ms: 220, max_age_ms: 400,
    max_side_px: 640, quality: 0.62, quality_min: 0.40, quality_max: 0.78,
    target_bytes: 24000, buffer_limit_bytes: 120000, max_inflight: 3
  };

  /* How long the socket gets to open before the POST path takes over. A drive
     that starts with no picture because a WebSocket was still handshaking is
     worse than a drive at 4 fps. */
  var OPEN_BUDGET_MS = 4000;
  var PING_INTERVAL_MS = 5000;
  /* No pong for this many ping intervals and the socket is dead however open
     it looks. A TCP connection through a carrier NAT that has silently dropped
     the flow stays `readyState === OPEN` forever. */
  var PONG_MISSES = 3;
  /* Nothing SENT for this long, while running, is a stall. Generous next to a
     15 fps cadence and tight next to the 442 s the drive lost. */
  var STALL_MS = 4000;
  /* How long one tick's capture-and-encode may take before it is abandoned.
     A tick that has not produced bytes in this long is not slow, it is stuck:
     the drive's stall was canvas.toBlob never calling back at all. */
  var TICK_BUDGET_MS = 2500;
  /* A socket that has carried frames for this long has proved itself, and the
     reconnect budget starts again. Without it three drops spread over a long
     drive retire the transport permanently -- MAX_RECONNECTS was a lifetime
     count with nothing ever resetting it. */
  var HEALTHY_MS = 20000;
  var SUPERVISOR_MS = 1000;
  /* Consecutive socket failures after which this page stops trying and stays
     on POST for the rest of the drive. Reconnecting forever against a proxy
     that strips upgrades is a battery leak with no upside. */
  var MAX_RECONNECTS = 3;

  function now() { return (root.performance && root.performance.now)
    ? root.performance.now() : Date.now(); }

  /* performance.now() -> epoch seconds, in THIS machine's clock. timeOrigin is
     missing on older Safari; Date.now() at load is the same idea and is only
     ever used as an origin, so a millisecond of skew in it is invisible. */
  var T_ORIGIN = (root.performance && typeof root.performance.timeOrigin === 'number')
    ? root.performance.timeOrigin : (Date.now() - now());
  function epochSec(pnow) { return (T_ORIGIN + pnow) / 1000; }

  function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }

  function create(cfg) {
    cfg = cfg || {};
    var wsUrl = cfg.wsUrl;
    var postUrl = cfg.postUrl;
    var element = cfg.element || function () { return null; };
    var speed = cfg.speed || function () { return { v: null, age: 0 }; };
    var sourceKind = cfg.source || function () { return 'camera'; };
    var onResult = cfg.onResult || function () {};
    var onEvent = cfg.onEvent || function () {};

    var tuning = {};
    for (var k in FALLBACK_TUNING) tuning[k] = FALLBACK_TUNING[k];
    for (var k2 in (cfg.tuning || {})) tuning[k2] = cfg.tuning[k2];

    /* THE LIVENESS CONSTANTS, OVERRIDABLE. A stall watchdog whose period
       cannot be shortened is a stall watchdog no test can drive: the shipped
       values are seconds, and a suite that waited them out would take minutes
       to assert a millisecond of logic. Same reason the tuning above arrives
       from outside — the SHAPE is what is under test, not the number. */
    var stallMs = cfg.stallMs || STALL_MS;
    var tickBudgetMs = cfg.tickBudgetMs || TICK_BUDGET_MS;
    var pingMs = cfg.pingIntervalMs || PING_INTERVAL_MS;
    var pongMisses = cfg.pongMisses || PONG_MISSES;
    var healthyMs = cfg.healthyMs === undefined ? HEALTHY_MS : cfg.healthyMs;
    var superviseMs = cfg.superviseMs || SUPERVISOR_MS;
    var maxReconnects = cfg.maxReconnects === undefined ? MAX_RECONNECTS
                                                        : cfg.maxReconnects;
    var reconnectBaseMs = cfg.reconnectBaseMs || 500;
    var openDelayMs = cfg.openDelayMs === undefined ? 200 : cfg.openDelayMs;

    var ws = null;
    var wsOpen = false;
    var mode = 'post';            // 'ws' once the socket is carrying frames
    var running = false;
    var timer = null;
    var pingTimer = null;
    var openTimer = null;
    var reconnects = 0;
    var giveUpOnSocket = false;

    var seq = 0;
    var inflight = 0;
    var quality = tuning.quality;
    var fps = tuning.start_fps;
    /* PIPELINE DEPTH, NOT QUEUE LENGTH. Starts at one and is raised only
       while measured frame age sits well under target; any age above the
       ceiling collapses it to one on the same result that reported it. */
    var maxInflight = 1;
    var ageEma = null;
    var starvedByInflight = 0;

    /* Clock offset: add this to a local epoch time to get the server's. Null
       until the first pong, and a frame sent before then carries no cap_t —
       which is honest, and reads as `frame_age_ms: null` rather than as a
       fabricated age. */
    var offsetSec = null;
    var bestRtt = Infinity;

    /* --- liveness ---------------------------------------------------------
       Everything the supervisor needs to answer one question: is this
       transport moving pictures RIGHT NOW? Nothing here existed on the drive
       that lost 442 seconds, which is why nothing noticed. */
    var lastSentAt = 0;           // when a frame last went on the wire
    var lastResultAt = 0;         // ...and when one last came back
    var openedAt = 0;             // when this socket opened
    var pingsOut = 0;             // pings since the last pong
    var supervisor = null;
    var watchedEl = null;
    var visibilityBound = false;
    var feedState = 'idle';       // idle | live | reconnecting | lost
    var stallRecoveries = 0;
    var tickOverruns = 0;

    var stats = {
      sent: 0, results: 0, skipped_inflight: 0, skipped_buffer: 0,
      skipped_capture: 0, bytes: 0, dropped_server: 0,
      // Which thread did the encode. A drive that is entirely on_thread is a
      // drive where the worker never came up, which is worth knowing before
      // the crackle is blamed on something else.
      encoded_off_thread: 0, encoded_on_thread: 0,
      ages: [], rtts: [],
      /* THE STAGES THE SERVER CANNOT SEE.
         frame_age_ms already covers capture -> detection, and the server
         breaks its own share down. What nothing measured was the part that
         happens before the frame leaves the page, which is where a slow clip
         mode would hide: how often a picture is actually taken, how long the
         encode costs, and how long the overlay takes to draw the result. */
      capture_gaps: [], encodes: [], pushes: [], draws: [],
      server_stage: {}      // stage -> samples, straight from timing_ms
    };
    var lastCaptureAt = null;

    function note(name, detail) { try { onEvent(name, detail || {}); } catch (e) {} }

    /* WHAT THE DRIVER IS LOOKING AT, and it is a state rather than an event.
       A HUD that freezes and says nothing is worse than a HUD that says the
       feed is gone: the first one is indistinguishable from an empty road. */
    function setState(next, why) {
      if (feedState === next) return;
      var prev = feedState;
      feedState = next;
      note('FRAMES_STATE', { state: next, was: prev, why: why || null });
      if (cfg.onState) { try { cfg.onState(next, { was: prev, why: why || null }); } catch (e) {} }
    }

    /* p50/p90/p99 and n, for one list of samples. */
    function band(list) {
      return { p50: pct(list, 0.5), p90: pct(list, 0.9),
               p99: pct(list, 0.99), n: list.length };
    }

    function record(list, v) {
      list.push(v);
      // A drive is ten minutes and this is a diagnostic, not a time series.
      if (list.length > 4000) list.splice(0, list.length - 4000);
    }

    /* ---------------- the socket ----------------------------------------- */

    function openSocket() {
      if (giveUpOnSocket || !wsUrl || typeof root.WebSocket !== 'function') return;
      try { ws = new root.WebSocket(wsUrl); }
      catch (e) { giveUpOnSocket = true; return; }
      ws.binaryType = 'arraybuffer';

      openTimer = setTimeout(function () {
        if (!wsOpen) {
          note('FRAMES_WS_SLOW', { budget_ms: OPEN_BUDGET_MS });
          try { ws.close(); } catch (e) {}
        }
      }, OPEN_BUDGET_MS);

      ws.onopen = function () {
        wsOpen = true;
        openedAt = now();
        pingsOut = 0;
        if (openTimer) { clearTimeout(openTimer); openTimer = null; }
        ping();
        pingTimer = setInterval(ping, pingMs);
      };

      ws.onmessage = function (e) {
        var msg;
        try { msg = JSON.parse(typeof e.data === 'string' ? e.data : ''); }
        catch (err) { return; }
        if (!msg) return;
        if (msg.op === 'ready') {
          for (var kk in (msg.tuning || {})) tuning[kk] = msg.tuning[kk];
          quality = clamp(quality, tuning.quality_min, tuning.quality_max);
          fps = clamp(tuning.start_fps, tuning.min_fps, tuning.max_fps);
          // Frames only start flowing down here: until the tuning has landed
          // this page does not know how big a picture is allowed to be.
          mode = 'ws';
          note('FRAMES_WS_READY', { tuning: tuning });
          return;
        }
        if (msg.op === 'pong') {
          pingsOut = 0;             // the far end is answering
          var c2 = epochSec(now());
          var rtt = c2 - msg.c;
          if (rtt >= 0 && rtt < bestRtt) {
            bestRtt = rtt;
            // The reply was composed at some instant between sending and
            // receiving; the midpoint is the best guess a symmetric link
            // allows, and keeping only the LOWEST round trip seen is what
            // stops one congested exchange from poisoning the offset.
            offsetSec = msg.s - (msg.c + c2) / 2;
          }
          record(stats.rtts, rtt * 1000);
          return;
        }
        if (msg.op === 'stale') {
          note('FRAMES_STALE', { reason: msg.reason });
          stop();
          if (cfg.onStale) { try { cfg.onStale(msg); } catch (e2) {} }
          return;
        }
        if (msg.op === 'skip' || msg.op === 'error' || msg.op === 'warming'
            || msg.op === 'stats') {
          if (msg.op === 'skip' || msg.op === 'error') inflight = Math.max(0, inflight - 1);
          return;
        }
        // Anything else is a headway result.
        inflight = Math.max(0, inflight - 1);
        lastResultAt = now();
        setState('live', 'result');
        handleResult(msg);
      };

      ws.onclose = function () {
        wsOpen = false;
        if (openTimer) { clearTimeout(openTimer); openTimer = null; }
        if (pingTimer) { clearInterval(pingTimer); pingTimer = null; }
        /* A SOCKET THAT PROVED ITSELF EARNS THE BUDGET BACK.
           MAX_RECONNECTS was a LIFETIME count with nothing resetting it, so a
           ten-minute drive that dropped three times an hour apart retired the
           transport for good and finished on the POST path at 1 fps. A socket
           that carried frames for HEALTHY_MS did not fail because the network
           refuses upgrades; it failed because a phone moved between cells. */
        var lived = openedAt ? (now() - openedAt) : 0;
        if (lived >= healthyMs && reconnects > 0) {
          note('FRAMES_WS_HEALTHY_RESET', { lived_ms: Math.round(lived),
                                            reconnects: reconnects });
          reconnects = 0;
        }
        ws = null;
        inflight = 0;
        openedAt = 0;
        if (mode === 'ws') note('FRAMES_WS_LOST', { reconnects: reconnects });
        mode = 'post';
        if (!running) return;
        if (++reconnects > maxReconnects) {
          giveUpOnSocket = true;
          setState('live', 'post_fallback');
          note('FRAMES_WS_GIVEN_UP', { reconnects: reconnects });
          return;
        }
        setState('reconnecting', 'ws_closed');
        /* Backoff, capped. 500 x n was unbounded in principle and is bounded
           in practice only because the budget above used to be permanent. */
        var delay = Math.min(8000, reconnectBaseMs * Math.pow(2, reconnects - 1));
        setTimeout(function () { if (running) openSocket(); }, delay);
      };

      ws.onerror = function () { /* onclose does the work; this only silences it */ };
    }

    function ping() {
      if (!wsOpen) return;
      /* THE PONGS WERE ALWAYS COMING BACK AND NOTHING READ THEM. A flow
         dropped by a carrier NAT leaves readyState OPEN indefinitely: the
         page goes on calling send() into a socket that will never deliver
         anything again, which looks exactly like a camera that stopped. */
      if (pingsOut >= pongMisses) {
        note('FRAMES_WS_UNANSWERED', { pings: pingsOut });
        try { ws.close(); } catch (e) {}     // onclose does the reconnect
        return;
      }
      pingsOut++;
      try { ws.send(JSON.stringify({ op: 'ping', c: epochSec(now()) })); } catch (e) {}
    }

    /* ---------------- the encoder worker ---------------------------------
     *
     * THE ENCODE IS NOT ON THIS THREAD ANY MORE, WHERE IT CAN BE.
     *
     * drawImage + toBlob + arrayBuffer, ten to fifteen times a second, on the
     * same thread as the media pipeline that feeds the <audio> element the
     * output bus plays out of. None of it is slow in isolation and all of it
     * lands in the same place: a main thread that stalls for 20 ms at the
     * wrong moment is a jitter buffer that runs dry, which is a crackle. See
     * the note at the top of rio_output.js.
     *
     * `createImageBitmap(videoEl)` is a GPU-side copy that transfers to a
     * worker with no serialisation; everything after it happens over there.
     *
     * AND IT IS ALWAYS ALLOWED NOT TO EXIST. OffscreenCanvas, transferable
     * ImageBitmaps and Workers are not everywhere, and a browser without them
     * takes exactly the path this file has always taken. The worker is PROVED
     * with a ping before a single frame is trusted to it, because a worker
     * that failed to load looks identical to one that is merely slow. */
    var worker = null;
    var workerReady = false;
    var workerJobs = {};
    var workerJobSeq = 0;
    var workerMisses = 0;

    function workerUsable() {
      return !!(cfg.encoderUrl !== null && root.Worker && root.OffscreenCanvas
                && root.createImageBitmap);
    }

    function startWorker() {
      if (worker || !workerUsable()) return;
      try {
        worker = new root.Worker(cfg.encoderUrl || '/static/rio_frame_encoder.js');
      } catch (e) { worker = null; return; }
      worker.onmessage = function (e) {
        var m = e.data || {};
        if (m.op === 'pong') {
          workerReady = true;
          note('FRAMES_ENCODER_READY', {});
          return;
        }
        var job = workerJobs[m.seq];
        if (!job) return;
        delete workerJobs[m.seq];
        if (m.op === 'frame' && m.bytes) job.resolve(new Uint8Array(m.bytes));
        else job.resolve(null);
      };
      worker.onerror = function () {
        /* A worker that cannot run is not a failure to report loudly: the
           encode simply happens here instead, exactly as it always did. */
        note('FRAMES_ENCODER_LOST', {});
        workerReady = false;
        try { worker.terminate(); } catch (e) {}
        worker = null;
        for (var k in workerJobs) { workerJobs[k].resolve(null); }
        workerJobs = {};
      };
      try { worker.postMessage({ op: 'ping' }); } catch (e) {}
    }

    /* One frame, encoded over there. Resolves null on any failure at all, and
       the caller then takes the main-thread path for that frame. */
    function encodeOffThread(el, w, h) {
      if (!worker || !workerReady) return Promise.resolve(null);
      var seq = ++workerJobSeq;
      return root.createImageBitmap(el).then(function (bitmap) {
        return new Promise(function (resolve) {
          workerJobs[seq] = { resolve: resolve };
          try {
            worker.postMessage({ op: 'encode', seq: seq, bitmap: bitmap,
                                 w: w, h: h, quality: quality }, [bitmap]);
          } catch (e) {
            delete workerJobs[seq];
            try { bitmap.close(); } catch (x) {}
            resolve(null);
          }
        });
      }, function () { return null; });
    }

    /* ---------------- capture -------------------------------------------- */

    /* Downscale on the way out. The camera hands over 1280x720 or better; the
       detector letterboxes to 560, depth to 518, lanes to 800x320. Sending the
       full frame was paying mobile uplink for pixels no model ever sees — and
       it is why the old path cost 71–90 KB a frame. */
    var canvas = null;
    function grab(el) {
      if (!el || !el.videoWidth || !el.videoHeight) return null;
      var w = el.videoWidth, h = el.videoHeight;
      var scale = Math.min(1, tuning.max_side_px / Math.max(w, h));
      var cw = Math.max(2, Math.round(w * scale));
      var ch = Math.max(2, Math.round(h * scale));
      if (!canvas) canvas = document.createElement('canvas');
      if (canvas.width !== cw || canvas.height !== ch) {
        canvas.width = cw; canvas.height = ch;
      }
      canvas.getContext('2d').drawImage(el, 0, 0, cw, ch);
      return canvas;
    }

    function toBlob(cv) {
      return new Promise(function (resolve) {
        try {
          cv.toBlob(function (b) { resolve(b || null); }, 'image/jpeg', quality);
        } catch (e) { resolve(null); }
      });
    }

    /* ---------------- the loop ------------------------------------------- */

    function intervalMs() { return Math.max(20, Math.round(1000 / fps)); }

    function schedule(delay) {
      if (!running) return;
      timer = setTimeout(tick, delay === undefined ? intervalMs() : delay);
    }

    /* A PROMISE THAT MAY NEVER SETTLE, GIVEN A DEADLINE.
       canvas.toBlob does not call back while an iOS page is backgrounded, and
       blob.arrayBuffer() can sit on the same stall. Neither rejects. Racing
       them against a timer turns "the drive is over" into "one frame was
       skipped". */
    function withBudget(promise, ms, label) {
      return new Promise(function (resolve) {
        var done = false;
        var t = setTimeout(function () {
          if (done) return;
          done = true;
          tickOverruns++;
          note('FRAMES_TICK_OVERRUN', { budget_ms: ms, at: label });
          resolve(null);
        }, ms);
        Promise.resolve(promise).then(function (v) {
          if (done) return;
          done = true; clearTimeout(t); resolve(v);
        }, function () {
          if (done) return;
          done = true; clearTimeout(t); resolve(null);
        });
      });
    }

    async function tick() {
      if (!running) return;
      var t0 = now();
      try {
        /* THE WHOLE TICK, NOT JUST THE ENCODE. Whatever inside once() hangs --
           the encode, the blob read, a getter on a dead video element -- the
           loop is out of it inside TICK_BUDGET_MS. */
        await withBudget(once(), tickBudgetMs, 'tick');
      } catch (e) { /* one bad frame never stops the loop */ }
      finally {
        /* IN A `finally`, AND THAT IS THE FIX.
           This line used to be the last statement of the function, so an
           awaited promise that never settled meant it never ran and the drive
           had no more pictures -- 442 s of exactly that on session 738fbb82.
           The next tick is the cadence MINUS what this one cost, so encoding
           does not silently halve the frame rate. */
        schedule(Math.max(0, intervalMs() - (now() - t0)));
      }
    }

    /* ONE TICK. Returns without taking a picture whenever taking one would
       mean queueing it — which is most of what this function is. */
    /* The output size, which both encode paths need and only one of them used
       to compute. */
    function outSize(el) {
      if (!el || !el.videoWidth || !el.videoHeight) return null;
      var w = el.videoWidth, h = el.videoHeight;
      var scale = Math.min(1, tuning.max_side_px / Math.max(w, h));
      return { w: Math.max(2, Math.round(w * scale)),
               h: Math.max(2, Math.round(h * scale)) };
    }

    async function once() {
      if (mode === 'ws') {
        if (!wsOpen) { stats.skipped_buffer++; return; }
        if (inflight >= maxInflight) {
          stats.skipped_inflight++;
          starvedByInflight++;
          return;
        }
        if (ws.bufferedAmount > tuning.buffer_limit_bytes) {
          // The radio is behind. Another picture on the back of that queue
          // only makes the next one older.
          stats.skipped_buffer++;
          return;
        }
      } else if (inflight >= 1) {
        // The POST path is strictly one at a time: it has no eviction on the
        // far end, so a second request would be a genuine queue.
        stats.skipped_inflight++;
        return;
      }

      var el = element();
      var size = outSize(el);
      if (!size) { stats.skipped_capture++; return; }

      /* THE WORKER PATH FIRST, when there is one. The capture instant is taken
         after createImageBitmap for the same reason it used to be taken after
         drawImage: that is when the pixels left the element. */
      var body = null;
      var capturedAt = null;
      var encodeT0 = now();
      if (worker && workerReady) {
        var jobId = workerJobSeq + 1;
        var offP = encodeOffThread(el, size.w, size.h);
        capturedAt = now();
        body = await withBudget(offP, tickBudgetMs, 'worker_encode');
        if (body) {
          stats.encoded_off_thread++;
          workerMisses = 0;
        } else {
          /* THE JOB IS ABANDONED, AND SO IS ITS SLOT. A timed-out encode
             leaves an entry in workerJobs holding a resolver nobody will call,
             and the ImageBitmap it was transferred is held with it — a
             megabyte of GPU memory per lost frame. */
          delete workerJobs[jobId];
          /* ...AND A WORKER THAT KEEPS MISSING IS STOOD DOWN. Falling back
             costs a whole tick budget before the main-thread path even
             starts, so paying that on every frame is worse than never having
             had a worker. */
          if (++workerMisses >= 3) {
            note('FRAMES_ENCODER_LOST', { misses: workerMisses });
            workerReady = false;
            try { worker.terminate(); } catch (e) {}
            worker = null;
            workerJobs = {};
          }
        }
      }

      if (!body) {
        var cv = grab(el);
        if (!cv) { stats.skipped_capture++; return; }
        // THE CAPTURE INSTANT. Taken after drawImage, which is when the pixels
        // were actually read out of the element, and before the encode, which
        // is part of the age and not outside it.
        capturedAt = now();
        var blob = await withBudget(toBlob(cv), tickBudgetMs, 'encode');
        if (!blob) { stats.skipped_capture++; return; }
        var bytes0 = await withBudget(blob.arrayBuffer(), tickBudgetMs, 'arraybuffer');
        if (!bytes0) { stats.skipped_capture++; return; }
        body = new Uint8Array(bytes0);
        stats.encoded_on_thread++;
      }

      /* HOW OFTEN A PICTURE IS ACTUALLY TAKEN, which is not the same as the
         rate the loop asks for: a tick that skips for want of a slot, a
         buffer or a frame does not get here. The gap between successive
         CAPTURES is the effective capture interval, and its p90 is where a
         mode that is quietly starving shows up. */
      if (lastCaptureAt !== null) record(stats.capture_gaps, capturedAt - lastCaptureAt);
      lastCaptureAt = capturedAt;
      record(stats.encodes, now() - encodeT0);

      var s = speed();
      var capServer = (offsetSec === null) ? null : (epochSec(capturedAt) + offsetSec);
      seq++;
      stats.sent++;
      stats.bytes += body.length;
      adaptQuality(body.length);

      if (mode === 'ws' && wsOpen) {
        var header = JSON.stringify({
          seq: seq, cap_t: capServer,
          v: (s.v === null ? null : s.v), va: s.age,
          src: safeSource()
        });
        var hbytes = new TextEncoder().encode(header);
        var out = new Uint8Array(4 + hbytes.length + body.length);
        new DataView(out.buffer).setUint32(0, hbytes.length, false);
        out.set(hbytes, 4);
        out.set(body, 4 + hbytes.length);
        inflight++;
        try { ws.send(out); lastSentAt = now(); }
        catch (e) { inflight = Math.max(0, inflight - 1); }
        return;
      }

      // --- the POST path, unchanged in shape and now stamped the same way ---
      inflight++;
      lastSentAt = now();
      try {
        var fd = new FormData();
        fd.append('image', new Blob([body], { type: 'image/jpeg' }), 'frame.jpg');
        fd.append('v_host', s.v === null ? '' : String(s.v));
        fd.append('v_host_age_s', String(s.age));
        if (capServer !== null) fd.append('cap_t', String(capServer));
        fd.append('source', safeSource());
        var r = await fetch(postUrl, { method: 'POST', body: fd });
        var j = await r.json();
        if (j && j.ok !== false) { j.seq = seq; handleResult(j); }
      } finally {
        inflight = Math.max(0, inflight - 1);
      }
    }

    function safeSource() {
      try { return sourceKind(); } catch (e) { return 'camera'; }
    }

    /* ---------------- the controller ------------------------------------- */

    /* Quality tracks a BYTE BUDGET, not an age: bytes are what the uplink
       charges for, and holding a frame near the budget is what makes a higher
       frame rate affordable at all. */
    function adaptQuality(bytes) {
      var over = bytes / tuning.target_bytes;
      if (over > 1.25) quality = clamp(quality - 0.04, tuning.quality_min, tuning.quality_max);
      else if (over < 0.75) quality = clamp(quality + 0.02, tuning.quality_min, tuning.quality_max);
    }

    /* Rate tracks FRAME AGE, and only frame age. That is the number the
       warning's truthfulness rests on: a 15 fps stream whose pictures are
       half a second old is worse for a driver than a 9 fps stream whose
       pictures are current, so when age climbs the rate comes down — and
       comes down before quality does, because dropping to 10 fps costs
       nothing a Kalman filter can feel and blurring the car ahead does. */
    function adaptRate(ageMs) {
      if (ageMs === null || ageMs === undefined) return;
      // Smoothed, because one slow frame is a radio hiccup and the controller
      // must not shed a third of the frame rate over one of those.
      ageEma = (ageEma === null) ? ageMs : (ageEma * 0.8 + ageMs * 0.2);

      if (ageMs > tuning.max_age_ms || ageEma > tuning.max_age_ms) {
        // Age over the ceiling: the pipe comes back to one frame IMMEDIATELY
        // and the rate comes down. Depth first, because depth is the thing
        // that can put a picture behind another picture.
        maxInflight = 1;
        fps = clamp(fps * 0.8, tuning.min_fps, tuning.max_fps);
        if (fps <= tuning.min_fps + 0.01) {
          quality = clamp(quality - 0.06, tuning.quality_min, tuning.quality_max);
        }
        starvedByInflight = 0;
        return;
      }

      if (ageEma < tuning.target_age_ms) {
        fps = clamp(fps + 0.5, tuning.min_fps, tuning.max_fps);
        /* THE LINK, NOT THE LOOP, IS THE LIMIT -- and this is how that is
           told apart. Captures are being skipped because a frame is still in
           flight, while the frames that DO come back are comfortably fresh.
           That is a pipe with idle capacity, not a system falling behind, and
           the answer is one more frame on the wire rather than a slower loop.
           The margin is deliberate: depth only grows while age has real room
           under the target, so the first sign of trouble collapses it. */
        if (starvedByInflight >= 2 && ageEma < tuning.target_age_ms * 0.75) {
          maxInflight = Math.min(tuning.max_inflight || 1, maxInflight + 1);
          starvedByInflight = 0;
        }
      }
    }

    function handleResult(j) {
      stats.results++;
      if (typeof j.dropped === 'number') stats.dropped_server = j.dropped;
      if (typeof j.frame_age_ms === 'number') {
        record(stats.ages, j.frame_age_ms);
        adaptRate(j.frame_age_ms);
      }
      /* PUSH: send -> result, measured on THIS clock, so it is the whole
         round trip including everything the server did. frame_age_ms is the
         other half of the same question (capture -> detection) and the two
         differ by the encode plus whatever the server queued. */
      if (lastSentAt) record(stats.pushes, now() - lastSentAt);
      /* ...AND THE SERVER'S OWN BREAKDOWN, carried up so one report can show
         the whole pipeline instead of two halves that have to be joined by
         hand. These are the server's numbers verbatim; nothing is recomputed
         here. */
      var tm = j.timing_ms;
      if (tm && typeof tm === 'object') {
        for (var k in tm) {
          if (typeof tm[k] !== 'number') continue;
          if (!stats.server_stage[k]) stats.server_stage[k] = [];
          record(stats.server_stage[k], tm[k]);
        }
      }
      for (var f = 0; f < 3; f++) {
        var name = ['transport_ms', 'queue_ms', 'server_ms'][f];
        if (typeof j[name] !== 'number') continue;
        if (!stats.server_stage[name]) stats.server_stage[name] = [];
        record(stats.server_stage[name], j[name]);
      }
      try { onResult(j); } catch (e) {}
    }

    /* THE OVERLAY'S SHARE. Called by whatever draws the boxes, because this
       module does not own the canvas and should not reach for it. A draw that
       is slow is a draw that steals from the next capture on the same thread,
       which is exactly the failure mode a clip mode with a second video
       element is prone to. */
    function noteDraw(ms) {
      if (typeof ms === 'number' && isFinite(ms)) record(stats.draws, ms);
    }

    /* ---------------- the supervisor -------------------------------------
     *
     * ONE THING IT ASKS, ONCE A SECOND: has a picture gone on the wire
     * recently? Everything that can go wrong with this transport ends up
     * answering that question the same way, which is why it is the only
     * question worth asking:
     *
     *   the loop died                   nothing sent
     *   the camera track ended          grab() returns null, nothing sent
     *   the element lost its stream     same
     *   the socket died with no close   sends succeed into nothing, no
     *                                   results, and inflight pins the pipe
     *
     * A stall REBUILDS rather than waits. The socket is closed (its onclose
     * reconnects), the canvas is dropped so a fresh one is made against
     * whatever the element is now, and inflight is released -- because a pipe
     * held full by frames that will never be answered is a loop that will
     * never take another picture, whatever else is fixed.
     */
    function stalled() {
      if (!running) return false;
      var since = now() - (lastSentAt || openedAt || 0);
      return since > stallMs;
    }

    function recover(why) {
      stallRecoveries++;
      note('FRAMES_STALLED', {
        why: why,
        since_sent_ms: lastSentAt ? Math.round(now() - lastSentAt) : null,
        since_result_ms: lastResultAt ? Math.round(now() - lastResultAt) : null,
        inflight: inflight, mode: mode, ws_open: wsOpen,
        recoveries: stallRecoveries,
      });
      setState('reconnecting', why);
      inflight = 0;
      maxInflight = 1;
      canvas = null;                 // rebuilt against the element as it is now
      if (ws) { try { ws.close(); } catch (e) {} }   // onclose reopens
      // ...and kick the loop, in case the timer itself is what was lost.
      if (timer) { clearTimeout(timer); timer = null; }
      schedule(0);
    }

    function supervise() {
      if (!running) return;
      if (stalled()) { recover('no_frames_sent'); return; }
      if (mode === 'ws' && wsOpen && lastResultAt
          && (now() - lastResultAt) > stallMs * 2) {
        // Frames are going out and nothing is coming back. Same symptom from
        // the driver's seat, different half of the pipe.
        recover('no_results');
        return;
      }
      if (lastSentAt && (now() - lastSentAt) < stallMs) setState('live', 'sending');
    }

    /* THE CAMERA ITSELF CAN END, and on iOS it does: another app takes the
       camera, the OS revokes it under a phone call, the track goes `ended` and
       the element keeps its srcObject and reports videoWidth 0 forever. grab()
       then returns null on every tick, silently, and stats.skipped_capture is
       the only trace. */
    function watchElement() {
      var el = null;
      try { el = element(); } catch (e) { el = null; }
      if (!el || el === watchedEl) return;
      watchedEl = el;
      var stream = el.srcObject;
      if (!stream || !stream.getVideoTracks) return;
      stream.getVideoTracks().forEach(function (t) {
        if (t.__rioWatched) return;
        t.__rioWatched = true;
        t.addEventListener('ended', function () {
          note('FRAMES_TRACK_ENDED', { label: t.label || null });
          setState('lost', 'track_ended');
          if (cfg.onTrackEnded) { try { cfg.onTrackEnded(); } catch (e) {} }
        });
      });
    }

    /* ---------------- lifecycle ------------------------------------------ */

    function start() {
      if (running) return;
      running = true;
      seq = 0; inflight = 0;
      lastSentAt = now(); lastResultAt = 0; openedAt = 0; pingsOut = 0;
      stallRecoveries = 0; tickOverruns = 0; watchedEl = null;
      setState('reconnecting', 'starting');
      lastCaptureAt = null;
      stats = { capture_gaps: [], encodes: [], pushes: [], draws: [],
                server_stage: {},
                sent: 0, results: 0, skipped_inflight: 0, skipped_buffer: 0,
                skipped_capture: 0, bytes: 0, dropped_server: 0,
                encoded_off_thread: 0, encoded_on_thread: 0, ages: [], rtts: [] };
      workerMisses = 0;
      startWorker();
      maxInflight = 1; ageEma = null; starvedByInflight = 0;
      openSocket();
      // The first tick waits for the socket's budget only if there is a socket
      // to wait for; with none, the POST path starts immediately and the drive
      // is never worse off than it was.
      schedule(wsUrl && !giveUpOnSocket ? openDelayMs : 0);
      if (supervisor === null) {
        supervisor = setInterval(function () {
          try { watchElement(); } catch (e) {}
          try { supervise(); } catch (e) {}
        }, superviseMs);
      }
      bindVisibility();
    }

    function stop() {
      running = false;
      setState('idle', 'stopped');
      if (supervisor !== null) { clearInterval(supervisor); supervisor = null; }
      if (timer) { clearTimeout(timer); timer = null; }
      if (pingTimer) { clearInterval(pingTimer); pingTimer = null; }
      if (openTimer) { clearTimeout(openTimer); openTimer = null; }
      if (ws) { try { ws.close(); } catch (e) {} ws = null; }
      wsOpen = false; mode = 'post'; inflight = 0;
    }

    /* COMING BACK FROM THE BACKGROUND is the moment this transport is most
       likely to be broken and least likely to notice: iOS throttles the
       timers, may have ended the camera track, and canvas.toBlob was not
       calling back for the whole time the page was hidden. So the return is
       treated as a stall whether or not the clock says so. */
    function bindVisibility() {
      if (visibilityBound || !root.document || !root.document.addEventListener) return;
      visibilityBound = true;
      root.document.addEventListener('visibilitychange', function () {
        if (!running) return;
        if (root.document.visibilityState !== 'visible') {
          note('FRAMES_HIDDEN', {});
          return;
        }
        note('FRAMES_VISIBLE', {
          since_sent_ms: lastSentAt ? Math.round(now() - lastSentAt) : null,
        });
        watchedEl = null;            // re-bind to whatever the element is now
        recover('became_visible');
      });
    }

    function pct(list, p) {
      if (!list.length) return null;
      var v = list.slice().sort(function (a, b) { return a - b; });
      return Math.round(v[Math.min(v.length - 1, Math.round((v.length - 1) * p))] * 10) / 10;
    }

    return {
      start: start,
      stop: stop,
      running: function () { return running; },
      mode: function () { return mode; },
      /* Everything a drive review needs about the transport, in one object. */
      stats: function () {
        return {
          mode: mode, fps: Math.round(fps * 10) / 10,
          quality: Math.round(quality * 100) / 100,
          max_inflight: maxInflight,
          frame_age_ema_ms: ageEma === null ? null : Math.round(ageEma),
          sent: stats.sent, results: stats.results,
          skipped_inflight: stats.skipped_inflight,
          skipped_buffer: stats.skipped_buffer,
          skipped_capture: stats.skipped_capture,
          dropped_server: stats.dropped_server,
          encoded_off_thread: stats.encoded_off_thread,
          encoded_on_thread: stats.encoded_on_thread,
          encoder: worker ? (workerReady ? 'worker' : 'starting') : 'main',
          mean_bytes: stats.sent ? Math.round(stats.bytes / stats.sent) : null,
          frame_age_ms: { p50: pct(stats.ages, 0.5), p90: pct(stats.ages, 0.9),
                          p99: pct(stats.ages, 0.99), n: stats.ages.length },
          rtt_ms: { p50: pct(stats.rtts, 0.5), best: bestRtt === Infinity
                    ? null : Math.round(bestRtt * 1000) },
          clock_offset_s: offsetSec,
          /* WHETHER PICTURES ARE MOVING, which is the question the drive of
             2026-09-09 could not answer from anything in its log. */
          state: feedState,
          since_sent_ms: lastSentAt ? Math.round(now() - lastSentAt) : null,
          since_result_ms: lastResultAt ? Math.round(now() - lastResultAt) : null,
          stall_recoveries: stallRecoveries,
          tick_overruns: tickOverruns,
          reconnects: reconnects,
          /* THE WHOLE PIPELINE, one object, p50/p90/p99 each. The point of
             gathering it here rather than in the probe is that these are the
             page's own numbers: a bench that re-derives them is measuring the
             bench. */
          stages: (function () {
            var out = {
              capture_interval_ms: band(stats.capture_gaps),
              encode_ms: band(stats.encodes),
              push_ms: band(stats.pushes),
              overlay_draw_ms: band(stats.draws),
              frame_age_ms: band(stats.ages)
            };
            for (var k in stats.server_stage) {
              out['srv_' + k] = band(stats.server_stage[k]);
            }
            return out;
          })()
        };
      },
      noteDraw: noteDraw,
      state: function () { return feedState; },
      /* For the page to call when it knows something this file cannot see --
         a camera re-acquired, a source changed. */
      kick: function (why) { if (running) recover(why || 'kicked'); },
      /* Test seam: the pacing controller with no camera and no socket. */
      _adaptRate: function (ageMs) { adaptRate(ageMs); return fps; },
      _adaptQuality: function (bytes) { adaptQuality(bytes); return quality; },
      _set: function (o) { if (o.fps !== undefined) fps = o.fps;
                           if (o.quality !== undefined) quality = o.quality;
                           for (var kk in (o.tuning || {})) tuning[kk] = o.tuning[kk]; },
      _state: function () { return { fps: fps, quality: quality, tuning: tuning }; }
    };
  }

  root.RIO = root.RIO || {};
  root.RIO.frames = { create: create, FALLBACK_TUNING: FALLBACK_TUNING };

  if (typeof module !== 'undefined' && module.exports) module.exports = root.RIO.frames;
})(typeof window !== 'undefined' ? window : globalThis);
