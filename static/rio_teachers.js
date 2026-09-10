/* rio_teachers.js — the Teachers card, the IMU stream, and nothing else.
 *
 * THE ONE RULE THIS FILE OBEYS
 * ----------------------------
 * It reads /teachers/state and it writes to the DOM and the overlay. It does
 * not call RIO.speech, RIO.output, RIO.speak, look(), or anything that can
 * make RIO say or decide something. Two models are being watched here, not
 * consulted -- and the moment a shadow reading can reach a mouth it stops
 * being a shadow. tools/teacher_firewall_selftest.py checks this file for the
 * names above, so the rule is enforced rather than remembered.
 *
 * The traffic goes the other way once, deliberately: whatever the arbiter
 * starts playing is reported to /teachers/spoken, so a corpus row carries
 * RIO's own line beside the two readings of the same instant. That is the
 * single most valuable column in the record and it costs one POST per spoken
 * line.
 *
 * DEVICEMOTION, AND THE PERMISSION PROMPT
 * ---------------------------------------
 * Alpamayo conditions on ego motion, and a phone can supply the yaw rate that
 * dead reckoning needs. On iOS 13+ that requires an explicit permission, and
 * the request only works from a real user gesture -- so it is asked for on the
 * Start Drive tap and never on page load, where it would be silently refused
 * and never asked again.
 *
 * Declining is a supported outcome. The history is then built from speed
 * alone, the record says `yaw_rate: "none"`, and nothing else changes.
 *
 * WHAT IS SENT, AND HOW OFTEN
 * ---------------------------
 * Samples are taken at up to ~20 Hz and posted in one batch a second. Not on
 * the frame socket: frames are the one message whose size and cadence are
 * being controlled to hold frame age down, and an IMU rider on them would
 * couple the two rates and put a hole in the motion history every time a
 * frame is dropped. One small POST a second, off to the side, costs nothing.
 *
 * Every sample carries the RAW rotation rate and the RAW gravity vector, and
 * the server resolves the yaw rate from them -- see teachers/egomotion.py.
 * That is on purpose: a phone's own axes are useless without knowing how it is
 * mounted, the resolution is three lines of arithmetic, and it belongs
 * somewhere it can be tested rather than in a browser.
 */
(function (root) {
  'use strict';

  /* THE CARD FOLLOWS THE FRAMES, NOT THE DRIVE.
   *
   * This used to start only from startDrive(), so replaying a clip -- which
   * pushes frames, runs the whole headway pipeline and DOES raise keyframes on
   * the server -- left the card showing whatever it had painted at page load.
   * Observed: "fresh 154 s" while a clip was playing and the teachers were
   * answering every two seconds behind it. The readings were being taken; the
   * card was not asking for them.
   *
   * So: fast while frames are arriving, from any source, and slow when the
   * page is idle. `noteFrame()` is called from the headway result handler --
   * one assignment per frame -- so "frames are flowing" is measured rather
   * than inferred from which button was last pressed.
   */
  var POLL_MS = 1000;          /* while frames are arriving */
  var IDLE_POLL_MS = 10000;    /* when nothing is */
  /* No frame for this long and the page is idle. Generous next to the 4-15 fps
     the transport runs at, tight next to a 2 s keyframe floor. */
  var FRAME_IDLE_MS = 4000;
  var EGO_POST_MS = 1000;      /* one batch a second */
  var EGO_MIN_DT_MS = 45;      /* ~20 Hz; DeviceMotion fires at up to 60 */
  var EGO_MAX_BATCH = 60;      /* a backgrounded tab must not post a minute */

  /* THE SHARED ROWS, trimmed to one sentence each.
     Two models answering the same question is the comparison; two paragraphs
     of it is a wall. The full text is one tap away and all of it is in the
     corpus, so nothing is lost by showing the first sentence here -- and what
     is gained is that the row BELOW, the one only this model can fill, is on
     the screen without scrolling. */
  var FIELDS = [
    { key: 'scene', label: 'Scene', clamp: 1 },
    { key: 'attention', label: 'Attention · next 2 s', clamp: 1 }
  ];
  var TRACE_LABEL = {
    'alpamayo1.5': 'Chain-of-Causation',
    'cosmos-reason2': 'Physical reasoning'
  };

  /* First sentence, or the whole thing if it is already short. Used for the
     shared rows; the expander shows the rest. */
  function firstSentence(text) {
    var t = String(text || '').trim();
    if (t.length < 140) return t;
    var m = t.match(/^[\s\S]*?[.!?](\s|$)/);
    return m ? m[0].trim() : t;
  }
  var DISPLAY_NAME = {
    'alpamayo1.5': 'Alpamayo 1.5',
    'cosmos-reason2': 'Cosmos-Reason2'
  };
  /* Left to right, always. Two columns that swap places between polls would be
     unreadable, and Object.keys order over a JSON payload is not a promise. */
  var ORDER = ['alpamayo1.5', 'cosmos-reason2'];

  function $(id) { return document.getElementById(id); }

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text !== undefined && text !== null) n.textContent = text;
    return n;
  }

  function fmtS(v, digits) {
    if (v === null || v === undefined || !isFinite(v)) return '--';
    return Number(v).toFixed(digits === undefined ? 1 : digits) + 's';
  }

  /* ---------------------------------------------------------------------
     the card
     --------------------------------------------------------------------- */
  function renderColumn(model, entry) {
    var r = (entry && entry.reading) || {};
    var a = (entry && entry.association) || {};

    var col = el('div', 'teach-col');
    col.setAttribute('data-model', model);
    if (r.fresh) col.classList.add('is-fresh');

    var name = el('div', 'teach-name');
    name.appendChild(el('span', 'teach-dot'));
    name.appendChild(el('span', null, DISPLAY_NAME[model] || model));
    col.appendChild(name);

    /* THE THREE NUMBERS EVERY READING CARRIES, and they are not decoration.
       Freshness is how old the ROAD in this reading is -- the only one that
       decides whether it may be believed. Latency is what the model cost.
       Precision is which weights answered, because a bf16 reading and an fp8
       reading of the same road are not the same evidence. */
    var meta = el('div', 'teach-meta');
    var fresh = el('span', r.fresh ? null : 'stale');
    fresh.appendChild(el('i', null, 'fresh '));
    fresh.appendChild(el('b', null, fmtS(r.freshness_s)));
    meta.appendChild(fresh);
    var lat = el('span');
    lat.appendChild(el('i', null, 'lat '));
    lat.appendChild(el('b', null, r.latency_ms
      ? (r.latency_ms / 1000).toFixed(2) + 's' : '--'));
    meta.appendChild(lat);
    var prec = el('span');
    prec.appendChild(el('i', null, 'prec '));
    prec.appendChild(el('b', null, r.precision || '--'));
    meta.appendChild(prec);
    if (r.queue_ms) {
      var q = el('span');
      q.appendChild(el('i', null, 'queue '));
      q.appendChild(el('b', null, (r.queue_ms / 1000).toFixed(2) + 's'));
      meta.appendChild(q);
    }
    col.appendChild(meta);

    if (r.error) {
      var err = el('div', 'teach-field');
      err.appendChild(el('div', 'teach-label', 'Service'));
      err.appendChild(el('div', 'teach-text empty', r.error));
      col.appendChild(err);
    }

    var flags = r.flags || {};
    FIELDS.forEach(function (f) {
      var label = f.label || TRACE_LABEL[model] || 'Reasoning';
      var full = r[f.key];
      var text = f.clamp ? firstSentence(full) : full;
      var field = el('div', 'teach-field');
      var head = el('div', 'teach-label', label);
      /* THIS ANSWER LOOKS RECITED. Structural only -- see teachers/canned.py.
         Marked rather than hidden: it is a real thing the model said and it
         belongs in the record and on the card. What it must not do is sit
         here looking like perception. */
      if (flags[f.key]) {
        var strong = flags[f.key].strength !== 'weak';
        /* "recited?" is a claim about the model; "repeating" is a claim about
           the text. A repeated answer on an unchanging road may simply be
           right again, and saying "recited" about it would teach a driver to
           ignore the marker -- which is the only way a marker can fail. */
        var warn = el('span', 'teach-recited' + (strong ? '' : ' weak'),
                      strong ? ' recited?' : ' repeating');
        warn.title = flags[f.key].why || 'looks recited rather than observed';
        head.appendChild(warn);
      }
      field.appendChild(head);
      var body = el('div', 'teach-text', text || 'No reading yet.');
      if (!text) body.classList.add('empty');
      field.appendChild(body);
      /* Show all reveals whatever was trimmed -- the rest of the sentence
         for a shared row, the whole trace for a model's own row. The corpus
         keeps every word regardless of what this shows. */
      if (full && full !== text) {
        var more = el('button', 'teach-more', 'Show all');
        more.type = 'button';
        more.addEventListener('click', function () {
          var open = field.classList.toggle('open');
          body.textContent = open ? full : text;
          more.textContent = open ? 'Show less' : 'Show all';
        });
        field.appendChild(more);
      } else if (text && text.length > 240) {
        var more2 = el('button', 'teach-more', 'Show all');
        more2.type = 'button';
        more2.addEventListener('click', function () {
          var open = field.classList.toggle('open');
          more2.textContent = open ? 'Show less' : 'Show all';
        });
        field.appendChild(more2);
      }
      col.appendChild(field);
    });

    /* ---- THE ROW ONLY THIS COLUMN CAN FILL -------------------------------
       Two models answering the same three questions is a comparison, and a
       comparison of two things that can do the same thing tells you least
       about what each is FOR. Alpamayo predicts a path and nothing else here
       does; Cosmos reasons about other road users' motion and nothing else
       here does. Those get a row apiece, and they are the reason each column
       is worth its VRAM. */
    if (model === 'alpamayo1.5') col.appendChild(decisionRow(r));
    if (model === 'cosmos-reason2') col.appendChild(physicsRow(r));

    /* THE REFERENCED ACTOR. Either a track id the overlay is ringing right
       now, or the reason there is not one -- and the reason is the useful
       half: "names_undetectable_class:traffic light" says the association
       failed because RF-DETR has no such class, which is a different fact
       from the model having named nothing. */
    var actorField = el('div', 'teach-field');
    actorField.appendChild(el('div', 'teach-label', 'Referenced actor'));
    var chip;
    if (a.matched) {
      chip = el('span', 'teach-actor matched');
      chip.textContent = '#' + a.track_id + ' ' + (a.label || 'object')
        + (typeof a.range_m === 'number' ? ' · ' + a.range_m.toFixed(1) + ' m' : '');
      chip.title = 'matched on ' + (a.reason || '') + ' (score '
        + (a.score || 0) + ') — ringed on the picture';
    } else {
      chip = el('span', 'teach-actor unmatched');
      chip.textContent = a.reason ? ('unmatched · ' + a.reason) : 'no actor named';
      chip.title = a.phrase || '';
    }
    actorField.appendChild(chip);
    col.appendChild(actorField);
    return col;
  }

  /* ALPAMAYO: what its predicted path means, in two words.
     Derived server-side by arithmetic on the 64 waypoints
     (teachers/decision.py) -- not asked of the model, so it is the same answer
     every time and can be re-derived from the corpus row. Display only. */
  function decisionRow(r) {
    var field = el('div', 'teach-field');
    field.appendChild(el('div', 'teach-label', 'Driving decision'));
    var d = r.decision;
    if (!d || !d.text) {
      var none = el('div', 'teach-text empty',
                    r.trajectory ? 'Path too short to read.'
                                 : 'No predicted path.');
      field.appendChild(none);
      return field;
    }
    var line = el('div', 'teach-decision');
    line.appendChild(el('b', 'dec-long dec-' + d.longitudinal, d.longitudinal));
    line.appendChild(el('span', 'dec-sep', '·'));
    line.appendChild(el('b', 'dec-lat', d.lateral));
    field.appendChild(line);
    /* The numbers it decided on, so the two words can be argued with rather
       than trusted. */
    var nums = el('div', 'teach-meta');
    function num(label, value) {
      var sp = el('span');
      sp.appendChild(el('i', null, label + ' '));
      sp.appendChild(el('b', null, value));
      nums.appendChild(sp);
    }
    num('v', d.v_start_ms.toFixed(1) + '\u2192' + d.v_end_ms.toFixed(1) + ' m/s');
    num('a', (d.accel_ms2 >= 0 ? '+' : '') + d.accel_ms2.toFixed(2) + ' m/s\u00b2');
    num('lat', (d.lateral_end_m >= 0 ? '+' : '') + d.lateral_end_m.toFixed(1) + ' m');
    num('reach', d.reach_m.toFixed(0) + ' m / ' + (d.horizon_s || 6.4) + ' s');
    field.appendChild(nums);
    var hint = el('div', 'teach-hint',
      'derived from the predicted path — display only, nothing reads it');
    field.appendChild(hint);
    /* THE CHAIN-OF-CAUSATION, behind Show all — symmetric with Cosmos's row.
       It is no longer a shared row (the shared rows are the two questions both
       models answer), but it is Alpamayo's own reasoning about the path these
       two words describe, and it belongs next to them rather than nowhere. */
    if (r.reasoning) {
      var coc = el('div', 'teach-text');
      coc.style.display = 'none';
      coc.textContent = r.reasoning;
      field.appendChild(coc);
      var more = el('button', 'teach-more', 'Chain-of-Causation');
      more.type = 'button';
      more.addEventListener('click', function () {
        var open = coc.style.display === 'none';
        coc.style.display = open ? '' : 'none';
        more.textContent = open ? 'Hide' : 'Chain-of-Causation';
      });
      field.appendChild(more);
    }
    return field;
  }

  /* COSMOS: per-actor motion, and its own plausibility verdict.
     The full chain of thought is behind Show all -- Cosmos does not emit a
     separate <think> block under this chat template, so the reasoning IS the
     answer and "show all" means the whole of it rather than a hidden trace. */
  function physicsRow(r) {
    var field = el('div', 'teach-field');
    var head = el('div', 'teach-label', 'Physics');
    var ph = r.physics || {};
    if (ph.implausible === true) {
      var bad = el('span', 'teach-implausible', ' implausible');
      bad.title = ph.plausibility || '';
      head.appendChild(bad);
    } else if (ph.implausible === false) {
      var okm = el('span', 'teach-plausible', ' plausible');
      okm.title = ph.plausibility || '';
      head.appendChild(okm);
    }
    field.appendChild(head);
    var actors = ph.actors || r.reasoning || '';
    var body = el('div', 'teach-text', actors || 'No reading yet.');
    if (!actors) body.classList.add('empty');
    field.appendChild(body);
    if (ph.plausibility) {
      field.appendChild(el('div', 'teach-hint', ph.plausibility));
    } else if (ph.has_verdict === false) {
      field.appendChild(el('div', 'teach-hint',
        'no plausibility verdict in this answer'));
    }
    if (r.thinking || (actors && actors.length > 240)) {
      var more = el('button', 'teach-more', 'Show all');
      more.type = 'button';
      more.addEventListener('click', function () {
        var open = field.classList.toggle('open');
        body.textContent = open ? (r.thinking
          ? r.thinking + '\n\n' + (r.reasoning || '') : (r.reasoning || actors))
          : actors;
        more.textContent = open ? 'Show less' : 'Show all';
      });
      field.appendChild(more);
    }
    return field;
  }

  function renderTally(state) {
    var host = $('teachtally');
    if (!host) return;
    var t = state.tally || {};
    host.textContent = '';
    function add(label, value, title) {
      var span = el('span', null, label);
      span.appendChild(el('b', null, value));
      if (title) span.title = title;
      host.appendChild(span);
    }
    add('Keyframes', String(t.keyframes || 0),
        'Instants sent to both teachers this drive');
    add('Agreement', t.agreement_pct === null || t.agreement_pct === undefined
      ? '--' : t.agreement_pct + '%',
        'How often both models named the SAME RF-DETR track as the critical '
        + 'actor, out of the keyframes both answered. Two abstentions are not '
        + 'agreement.');
    add('Matched tracks', t.matched_pct === null || t.matched_pct === undefined
      ? '--' : t.matched_pct + '%',
        'How often a named road user could be pinned to a detector track');
    if (t.dropped_build) {
      add('Dropped', String(t.dropped_build),
          'Keyframes refused before they were sent — usually a stale frame: '
          + (state.last_build_refusal || ''));
    }
  }

  function renderServices(state) {
    var host = $('teachsvc');
    if (!host) return;
    var svc = state.services || {};
    var bits = [];
    ORDER.forEach(function (name) {
      var s = svc[name];
      if (!s) return;
      if (!s.loaded) {
        bits.push('<span class="down">' + name + ': '
          + (s.last_error ? 'no answer' : 'not loaded') + '</span>');
        return;
      }
      bits.push(name + ': ' + (s.model || '?') + ' @ ' + (s.precision || '?')
        + (s.evicted ? ' · ' + s.evicted + ' evicted' : '')
        + (s.stale_dropped ? ' · ' + s.stale_dropped + ' stale' : ''));
    });
    host.innerHTML = bits.join(' &nbsp;·&nbsp; ');
  }

  function render(state) {
    var cols = $('teachcols');
    if (!cols) return;
    cols.textContent = '';
    var models = state.models || {};
    ORDER.forEach(function (name) {
      cols.appendChild(renderColumn(name, models[name]));
    });
    renderTally(state);
    renderServices(state);
    /* Hand the state to the overlay, which is the only other consumer. */
    if (root.RIO && root.RIO.overlay && root.RIO.overlay.teachers) {
      root.RIO.overlay.teachers(state);
    }
  }

  /* ---------------------------------------------------------------------
     the layer switches
     --------------------------------------------------------------------- */
  function bindLayers() {
    var host = $('teachlayers');
    if (!host) return;
    var state = {};
    Array.prototype.forEach.call(host.querySelectorAll('.teach-layer'),
      function (btn) {
        var key = btn.getAttribute('data-layer');
        state[key] = btn.getAttribute('aria-pressed') === 'true';
        btn.addEventListener('click', function () {
          var on = btn.getAttribute('aria-pressed') !== 'true';
          btn.setAttribute('aria-pressed', on ? 'true' : 'false');
          var patch = {};
          patch[key] = on;
          if (root.RIO && root.RIO.overlay && root.RIO.overlay.teacherLayers) {
            root.RIO.overlay.teacherLayers(patch);
          }
        });
      });
    if (root.RIO && root.RIO.overlay && root.RIO.overlay.teacherLayers) {
      root.RIO.overlay.teacherLayers(state);
    }
  }

  /* ---------------------------------------------------------------------
     the IMU stream
     --------------------------------------------------------------------- */
  function createEgo(cfg) {
    var buf = [];
    var lastAt = 0;
    var timer = null;
    var handler = null;
    var running = false;
    var stats = { sampled: 0, posted: 0, dropped: 0, permission: 'unknown' };

    function sample(ev) {
      var now = (root.performance && root.performance.now)
        ? root.performance.now() : Date.now();
      if (now - lastAt < EGO_MIN_DT_MS) return;
      lastAt = now;
      var rr = ev.rotationRate;
      var ag = ev.accelerationIncludingGravity;
      var ac = ev.acceleration;
      if (!rr || !ag) return;
      /* Gravity as the difference between the two accelerations the API
         reports. Some browsers give `acceleration` as null rather than zeros,
         in which case accelerationIncludingGravity IS gravity plus whatever
         the car is doing -- which over a 100 ms sample is dominated by gravity
         and is good enough to find "up". The server refuses anything whose
         magnitude is not gravity-like, so a bad sample costs a sample. */
      var g = [
        ag.x - ((ac && ac.x) || 0),
        ag.y - ((ac && ac.y) || 0),
        ag.z - ((ac && ac.z) || 0)
      ];
      buf.push({
        /* Epoch seconds in the CLIENT's clock. The offset that brings it into
           the server's is measured by the frame transport and sent with the
           batch, so the IMU and the frames end up on one ruler. */
        t: Date.now() / 1000,
        /* DeviceMotion names its rotation rates after the axes they turn
           about: alpha about z, beta about x, gamma about y. Sent as a plain
           (x, y, z) vector so the server does not have to know that. */
        rr: [rr.beta || 0, rr.gamma || 0, rr.alpha || 0],
        g: g
      });
      stats.sampled++;
      if (buf.length > EGO_MAX_BATCH) {
        buf.splice(0, buf.length - EGO_MAX_BATCH);
        stats.dropped++;
      }
    }

    function flush() {
      if (!buf.length) return;
      var batch = buf;
      buf = [];
      var offset = 0;
      try {
        var st = cfg.frameStats && cfg.frameStats();
        if (st && typeof st.clock_offset_s === 'number') offset = st.clock_offset_s;
      } catch (e) { /* no transport yet */ }
      var url = '/teachers/ego' + (cfg.sessionId && cfg.sessionId()
        ? '?session_id=' + encodeURIComponent(cfg.sessionId()) : '');
      fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ samples: batch, offset_s: offset })
      }).then(function () { stats.posted += batch.length; })
        .catch(function () { /* a lost batch is a gap in the history, not an error */ });
    }

    /* iOS 13+ gates DeviceMotion behind a permission that can ONLY be
       requested from a user gesture. Called from the Start Drive handler for
       exactly that reason; called anywhere else it resolves to 'denied' and
       can never be asked again for the life of the page. */
    function request() {
      var DM = root.DeviceMotionEvent;
      if (!DM) {
        stats.permission = 'unsupported';
        return Promise.resolve(false);
      }
      if (typeof DM.requestPermission !== 'function') {
        stats.permission = 'granted';   /* Android / desktop: no gate */
        return Promise.resolve(true);
      }
      return DM.requestPermission().then(function (res) {
        stats.permission = res;
        return res === 'granted';
      }).catch(function () {
        stats.permission = 'denied';
        return false;
      });
    }

    function start() {
      if (running) return Promise.resolve(true);
      return request().then(function (ok) {
        if (!ok) return false;
        handler = sample;
        root.addEventListener('devicemotion', handler);
        timer = root.setInterval(flush, EGO_POST_MS);
        running = true;
        return true;
      });
    }

    function stop() {
      if (handler) root.removeEventListener('devicemotion', handler);
      handler = null;
      if (timer) { root.clearInterval(timer); timer = null; }
      flush();
      running = false;
    }

    return { start: start, stop: stop, running: function () { return running; },
             stats: function () { return JSON.parse(JSON.stringify(stats)); },
             _sample: sample, _buffer: function () { return buf; } };
  }

  /* ---------------------------------------------------------------------
     wiring
     --------------------------------------------------------------------- */
  function create(cfg) {
    cfg = cfg || {};
    var sessionId = cfg.sessionId || function () { return null; };
    var ego = createEgo({ sessionId: sessionId, frameStats: cfg.frameStats });
    var poll = null;
    var pollMs = 0;
    var lastFrameAt = 0;
    var stopped = false;

    function url(path) {
      var sid = sessionId();
      return path + (sid ? '?session_id=' + encodeURIComponent(sid) : '');
    }

    function tick() {
      fetch(url('/teachers/state'))
        .then(function (r) { return r.json(); })
        .then(function (j) { if (j) render(j); })
        .catch(function () { /* the card keeps its last paint */ });
    }

    /* A CHAINED TIMEOUT, NOT AN INTERVAL, for the two reasons this codebase
       already uses one for the perception loop:

       A slow fetch must not let calls stack up. setInterval fires on a
       schedule whatever the last call is doing; a chain arms the next one only
       when the last has finished, so a card polling a busy server falls behind
       gracefully instead of queueing.

       And the period changes. An interval whose rate depends on whether
       frames are flowing has to be torn down and rebuilt on every change; a
       chain just picks its next delay.

       (It also stopped tools/output_bus_selftest.py counting this as the
       realtime bus watch, which it identifies by a 1000 ms period. That is a
       fragile identification and it is not this file's to fix -- but a poll
       that collides with it is this file's to avoid.) */
    function pace() {
      if (stopped) return;
      var flowing = (Date.now() - lastFrameAt) < FRAME_IDLE_MS;
      pollMs = flowing ? POLL_MS : IDLE_POLL_MS;
      if (poll) root.clearTimeout(poll);
      poll = root.setTimeout(function () { poll = null; tick(); pace(); },
                             pollMs);
    }

    return {
      /* Called by the headway result handler for every frame that comes back,
         from a camera or a clip. One assignment. */
      noteFrame: function () {
        var was = (Date.now() - lastFrameAt) < FRAME_IDLE_MS;
        lastFrameAt = Date.now();
        // Frames just started after a quiet spell: paint now and re-pace to
        // the fast rate rather than waiting out the idle delay.
        if (!was) { tick(); pace(); }
      },
      /* PACING ONLY, AND NO PERMISSION PROMPT. Safe to call at page load,
         which is the point: the card should be current whenever frames are
         flowing, and that is not only during a drive.

         It must NOT touch the IMU. On iOS, DeviceMotionEvent.requestPermission
         called outside a user gesture resolves to 'denied' and can never be
         asked again for the life of the page -- so arming the card at load
         would quietly cost every drive after it its yaw rate. */
      arm: function () {
        stopped = false;
        tick();
        pace();
      },
      /* The drive: pacing AND the IMU stream. Called from the Start Drive
         tap, because that tap is the gesture the permission needs. */
      start: function () {
        stopped = false;
        tick();
        pace();
        return ego.start();
      },
      /* Ends the drive's IMU stream. The card keeps polling -- a clip can be
         replayed after a drive ends, and the readings from it are just as
         real. */
      stop: function () {
        ego.stop();
      },
      /* Stops everything. Only the page unloading wants this. */
      halt: function () {
        stopped = true;
        if (poll) { root.clearTimeout(poll); poll = null; pollMs = 0; }
        ego.stop();
      },
      refresh: tick,
      ego: ego,
      /* RIO said something. One POST, one direction. */
      spoken: function (text, kind) {
        if (!text) return;
        fetch(url('/teachers/spoken'), {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ text: text, kind: kind || 'speech' })
        }).catch(function () {});
      },
      /* Something happened that the next frame should be a keyframe about. */
      event: function (kind) {
        fetch(url('/teachers/event'), {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ kind: kind || 'manual' })
        }).catch(function () {});
      },
      _render: render,
      _renderColumn: renderColumn
    };
  }

  root.RIO = root.RIO || {};
  root.RIO.teachers = { create: create, bindLayers: bindLayers,
                        ORDER: ORDER, DISPLAY_NAME: DISPLAY_NAME };

  if (typeof document !== 'undefined') {
    document.addEventListener('DOMContentLoaded', bindLayers);
  }
  if (typeof module !== 'undefined' && module.exports) module.exports = root.RIO.teachers;
})(typeof window !== 'undefined' ? window : globalThis);
