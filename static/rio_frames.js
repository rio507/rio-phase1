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

    var stats = {
      sent: 0, results: 0, skipped_inflight: 0, skipped_buffer: 0,
      skipped_capture: 0, bytes: 0, dropped_server: 0,
      ages: [], rtts: []
    };

    function note(name, detail) { try { onEvent(name, detail || {}); } catch (e) {} }

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
        if (openTimer) { clearTimeout(openTimer); openTimer = null; }
        ping();
        pingTimer = setInterval(ping, PING_INTERVAL_MS);
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
        handleResult(msg);
      };

      ws.onclose = function () {
        wsOpen = false;
        if (openTimer) { clearTimeout(openTimer); openTimer = null; }
        if (pingTimer) { clearInterval(pingTimer); pingTimer = null; }
        ws = null;
        inflight = 0;
        if (mode === 'ws') note('FRAMES_WS_LOST', { reconnects: reconnects });
        mode = 'post';
        if (!running) return;
        if (++reconnects > MAX_RECONNECTS) {
          giveUpOnSocket = true;
          note('FRAMES_WS_GIVEN_UP', { reconnects: reconnects });
          return;
        }
        setTimeout(function () { if (running) openSocket(); }, 500 * reconnects);
      };

      ws.onerror = function () { /* onclose does the work; this only silences it */ };
    }

    function ping() {
      if (!wsOpen) return;
      try { ws.send(JSON.stringify({ op: 'ping', c: epochSec(now()) })); } catch (e) {}
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

    async function tick() {
      if (!running) return;
      var t0 = now();
      try {
        await once();
      } catch (e) { /* one bad frame never stops the loop */ }
      // The next tick is the cadence MINUS what this one cost, so encoding
      // does not silently halve the frame rate.
      schedule(Math.max(0, intervalMs() - (now() - t0)));
    }

    /* ONE TICK. Returns without taking a picture whenever taking one would
       mean queueing it — which is most of what this function is. */
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
      var cv = grab(el);
      if (!cv) { stats.skipped_capture++; return; }
      // THE CAPTURE INSTANT. Taken after drawImage, which is when the pixels
      // were actually read out of the element, and before the encode, which
      // is part of the age and not outside it.
      var capturedAt = now();
      var blob = await toBlob(cv);
      if (!blob) { stats.skipped_capture++; return; }

      var s = speed();
      var capServer = (offsetSec === null) ? null : (epochSec(capturedAt) + offsetSec);
      seq++;
      stats.sent++;
      stats.bytes += blob.size;
      adaptQuality(blob.size);

      if (mode === 'ws' && wsOpen) {
        var header = JSON.stringify({
          seq: seq, cap_t: capServer,
          v: (s.v === null ? null : s.v), va: s.age,
          src: safeSource()
        });
        var hbytes = new TextEncoder().encode(header);
        var body = new Uint8Array(await blob.arrayBuffer());
        var out = new Uint8Array(4 + hbytes.length + body.length);
        new DataView(out.buffer).setUint32(0, hbytes.length, false);
        out.set(hbytes, 4);
        out.set(body, 4 + hbytes.length);
        inflight++;
        try { ws.send(out); }
        catch (e) { inflight = Math.max(0, inflight - 1); }
        return;
      }

      // --- the POST path, unchanged in shape and now stamped the same way ---
      inflight++;
      try {
        var fd = new FormData();
        fd.append('image', blob, 'frame.jpg');
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
      try { onResult(j); } catch (e) {}
    }

    /* ---------------- lifecycle ------------------------------------------ */

    function start() {
      if (running) return;
      running = true;
      seq = 0; inflight = 0;
      stats = { sent: 0, results: 0, skipped_inflight: 0, skipped_buffer: 0,
                skipped_capture: 0, bytes: 0, dropped_server: 0, ages: [], rtts: [] };
      maxInflight = 1; ageEma = null; starvedByInflight = 0;
      openSocket();
      // The first tick waits for the socket's budget only if there is a socket
      // to wait for; with none, the POST path starts immediately and the drive
      // is never worse off than it was.
      schedule(wsUrl && !giveUpOnSocket ? 200 : 0);
    }

    function stop() {
      running = false;
      if (timer) { clearTimeout(timer); timer = null; }
      if (pingTimer) { clearInterval(pingTimer); pingTimer = null; }
      if (openTimer) { clearTimeout(openTimer); openTimer = null; }
      if (ws) { try { ws.close(); } catch (e) {} ws = null; }
      wsOpen = false; mode = 'post'; inflight = 0;
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
          mean_bytes: stats.sent ? Math.round(stats.bytes / stats.sent) : null,
          frame_age_ms: { p50: pct(stats.ages, 0.5), p90: pct(stats.ages, 0.9),
                          p99: pct(stats.ages, 0.99), n: stats.ages.length },
          rtt_ms: { p50: pct(stats.rtts, 0.5), best: bestRtt === Infinity
                    ? null : Math.round(bestRtt * 1000) },
          clock_offset_s: offsetSec
        };
      },
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
