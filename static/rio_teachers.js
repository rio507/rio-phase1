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

  var POLL_MS = 1000;          /* the card. A reading changes every ~2 s. */
  var EGO_POST_MS = 1000;      /* one batch a second */
  var EGO_MIN_DT_MS = 45;      /* ~20 Hz; DeviceMotion fires at up to 60 */
  var EGO_MAX_BATCH = 60;      /* a backgrounded tab must not post a minute */

  /* The three fields, in the order a reading is read. Scene first because it
     is the one both models answer the same way; the trace second because it
     is the interesting one; attention last because it is the forward-looking
     one and belongs next to the actor chip. */
  var FIELDS = [
    { key: 'scene', label: 'Scene' },
    { key: 'reasoning', label: null },   /* per-model — see TRACE_LABEL */
    { key: 'attention', label: 'Attention · next 2 s' }
  ];
  var TRACE_LABEL = {
    'alpamayo1.5': 'Chain-of-Causation',
    'cosmos-reason2': 'Physical reasoning'
  };
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

    FIELDS.forEach(function (f) {
      var label = f.label || TRACE_LABEL[model] || 'Reasoning';
      var text = r[f.key];
      var field = el('div', 'teach-field');
      field.appendChild(el('div', 'teach-label', label));
      var body = el('div', 'teach-text', text || 'No reading yet.');
      if (!text) body.classList.add('empty');
      field.appendChild(body);
      /* The trace is the field that runs long, so it is the one that gets an
         expander. Clamped to four lines by the stylesheet; the corpus keeps
         every word regardless of what this shows. */
      if (text && text.length > 240) {
        var more = el('button', 'teach-more', 'Show all');
        more.type = 'button';
        more.addEventListener('click', function () {
          var open = field.classList.toggle('open');
          more.textContent = open ? 'Show less' : 'Show all';
        });
        field.appendChild(more);
      }
      col.appendChild(field);
    });

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

    return {
      start: function () {
        if (!poll) poll = root.setInterval(tick, POLL_MS);
        tick();
        return ego.start();
      },
      stop: function () {
        if (poll) { root.clearInterval(poll); poll = null; }
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
