/* rio_vehicle.js — the Vehicle Health column.
 *
 * This file paints. It does not decide.
 *
 * There is no threshold in here, no band, no unit conversion and no arithmetic
 * on a sensor value. GET /vehicle/telemetry returns a status name, a trend
 * glyph and a set of already-formatted strings per channel; GET
 * /vehicle/insights returns sentences that were written server-side. Every line
 * below either writes one of those strings into an element or turns a status
 * name into a CSS class. That is the whole contract: when the mock ECU is
 * replaced by a real Holley serial reader, a CAN bus or RIO Connect, this file
 * does not change and does not need to know it happened. See telemetry.py.
 *
 * The render functions
 * --------------------
 * The spec asks for VehicleStatusBanner / TelemetryList / TelemetryRow /
 * VehicleInsights / VehicleInsightCard as components. This is a vanilla
 * dashboard with no build step, so they are pure functions from data to DOM:
 *
 *     renderStatusBanner(state)   -> Element
 *     renderTelemetryList(rows)   -> DocumentFragment
 *     renderTelemetryRow(row)     -> Element
 *     renderInsights(entries)     -> DocumentFragment
 *     renderInsightCard(entry)    -> Element
 *
 * Same separation and the same file-level testability: each takes data and
 * returns nodes, reads no globals, fetches nothing, and holds no state. They
 * are exported on RIO.vehicle so they can be exercised without a server. The
 * polling below is the only stateful thing in the file and it does no
 * rendering of its own.
 *
 * Every row is generated from the telemetry array. There is no per-sensor code
 * anywhere in this file — adding boost, transmission temperature or suspension
 * travel is a row in telemetry.py's SENSORS table and nothing here.
 *
 * RIO does not speak about telemetry in this phase. Nothing here touches
 * RIO.speech — the arbiter is not involved, deliberately, and an insight is not
 * an alert. See the headers of telemetry.py and insights.py.
 */
(function (root) {
  'use strict';

  /* The ONLY intervals in this file, and neither is a telemetry setting: they
     are how long to wait before trying the server again after the fetch itself
     failed. The real cadences always come from each payload's poll_ms
     (config.TELEMETRY_POLL_MS / config.INSIGHTS_POLL_MS), which we do not have
     when the request never landed. */
  var RETRY_MS = 4000;
  var INSIGHT_RETRY_MS = 20000;

  // ---- tiny DOM helpers --------------------------------------------------

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  function $(id) { return document.getElementById(id); }

  // ---- VehicleStatusBanner -----------------------------------------------

  /* The banner the spec puts at the top of the column: one headline, one dot,
     and the three facts underneath it. The status name drives the colour of the
     dot, the rail and the headline together, because at a glance the colour IS
     the message — the same one-class-per-band shape #headwayhud uses. */
  function renderStatusBanner(state) {
    state = state || {};
    var box = el('div', 'veh-banner veh-' + (state.status || 'UNAVAILABLE'));

    var head = el('div', 'veh-banner-head');
    head.appendChild(el('span', 'veh-dot'));
    head.appendChild(el('span', 'veh-banner-text',
                        state.status_text || 'Awaiting Telemetry'));
    box.appendChild(head);

    var meta = el('div', 'veh-banner-meta');
    [['Link', state.connection || '--', 'veh-link-' + (state.connection_state || 'lost')],
     ['Updated', state.updated_display || '--:--:--', ''],
     ['Runtime', state.runtime_display || '--:--', '']
    ].forEach(function (pair) {
      var cell = el('div', 'veh-meta-cell');
      cell.appendChild(el('span', 'veh-meta-key', pair[0]));
      cell.appendChild(el('span', 'veh-meta-val' + (pair[2] ? ' ' + pair[2] : ''), pair[1]));
      meta.appendChild(cell);
    });
    box.appendChild(meta);
    return box;
  }

  // ---- TelemetryRow ------------------------------------------------------

  /* One channel. Four fields, on one line, in the order the spec lists them:
     name, value, status, trend. Nothing in here is per-sensor — a row knows
     only what the server put in it, which is what lets the same function draw
     a coolant temperature and a suspension travel sensor that does not exist
     yet. */
  function renderTelemetryRow(row) {
    row = row || {};
    var node = el('div', 'tm-row st-' + (row.status_class || 'no-data')
                         + ' tr-' + (row.trend || 'none'));
    node.setAttribute('data-id', row.id || '');
    /* The long-form reason, for anyone who hovers. The row itself stays at four
       fields — that is the point of it. */
    if (row.detail) node.title = row.detail;

    node.appendChild(el('span', 'tm-label', row.label || ''));

    var val = el('span', 'tm-value');
    val.appendChild(el('b', null, row.value_text != null ? row.value_text : '--'));
    if (row.units) val.appendChild(el('i', null, row.units));
    node.appendChild(val);

    node.appendChild(el('span', 'tm-status', row.status || ''));
    node.appendChild(el('span', 'tm-trend', row.trend_glyph || '—'));
    return node;
  }

  // ---- TelemetryList -----------------------------------------------------

  /* The rows, under their group headings. Groups come from the data and are
     emitted in first-seen order, so the server owns the ordering exactly as it
     owns everything else on this side of the wire. */
  function renderTelemetryList(rows) {
    var frag = document.createDocumentFragment();
    var seen = null;
    (rows || []).forEach(function (row) {
      if (row.group && row.group !== seen) {
        seen = row.group;
        frag.appendChild(el('div', 'tm-group', row.group));
      }
      frag.appendChild(renderTelemetryRow(row));
    });
    return frag;
  }

  // ---- VehicleInsightCard ------------------------------------------------

  /* One line of the log: when, an icon, and a sentence. Closer to Apple Health
     than to a fault log, per the spec — which mostly means the severity is a
     4px rail and a glyph rather than a shouting badge, and that a healthy car
     still produces entries. */
  function renderInsightCard(entry) {
    entry = entry || {};
    var card = el('div', 'vi-card vi-' + (entry.severity || 'info'));

    var head = el('div', 'vi-head');
    head.appendChild(el('span', 'vi-icon', entry.icon || '◈'));
    head.appendChild(el('span', 'vi-when', entry.when || ''));
    /* Seeded demo history is labelled on screen, always. An entry the car did
       not measure must never be indistinguishable from one it did. */
    if (entry.seeded) head.appendChild(el('span', 'vi-seed', 'Seed'));
    card.appendChild(head);

    card.appendChild(el('div', 'vi-text', entry.text || ''));
    return card;
  }

  // ---- VehicleInsights ---------------------------------------------------

  function renderInsights(entries) {
    var frag = document.createDocumentFragment();
    if (!entries || !entries.length) {
      frag.appendChild(el('div', 'vi-empty', 'No observations yet'));
      return frag;
    }
    entries.forEach(function (entry) { frag.appendChild(renderInsightCard(entry)); });
    return frag;
  }

  // ---- the dev scenario selectors ----------------------------------------

  /* Same pattern the tire panel used: a select that builds itself from the
     payload and removes itself entirely when the live provider has no
     scenarios, which is every real one. */
  function renderScenarioSelect(sel, list, current) {
    if (!sel) return;
    if (!list || !list.length) { sel.style.display = 'none'; return; }
    sel.style.display = '';
    var want = list.map(function (s) { return s.name; }).join('|');
    if (sel.dataset.built !== want) {
      sel.textContent = '';
      list.forEach(function (s) {
        var opt = document.createElement('option');
        opt.value = s.name;
        opt.textContent = s.label;
        sel.appendChild(opt);
      });
      sel.dataset.built = want;
    }
    if (current && sel.value !== current) sel.value = current;
  }

  // ---- mounting ----------------------------------------------------------

  var pollMs = null;
  var insightPollMs = null;
  var timer = null;
  var insightTimer = null;
  var wired = false;
  var lastInsightSig = null;

  function mountTelemetry(snap) {
    var banner = $('vehbanner');
    if (banner) banner.replaceChildren(renderStatusBanner(snap));

    var list = $('vehtelemetry');
    if (list) list.replaceChildren(renderTelemetryList(snap.rows));

    renderScenarioSelect($('vehscenario'), snap.scenarios, snap.scenario);
    renderScenarioSelect($('tirescenario'), snap.tire_scenarios, snap.tire_scenario);

    if (snap.poll_ms) pollMs = snap.poll_ms;
  }

  function mountInsights(snap) {
    var host = $('vehinsights');
    if (host) {
      /* Rebuild only when the content actually changed. The log moves on the
         order of minutes while this polls on the order of seconds, and
         replacing the nodes underneath a driver who is mid-scroll would throw
         them back to the top of a list they were reading. */
      var sig = (snap.entries || []).map(function (e) {
        return e.at + '|' + e.text;
      }).join('\n');
      if (sig !== lastInsightSig) {
        host.replaceChildren(renderInsights(snap.entries));
        lastInsightSig = sig;
      }
    }
    if (snap.poll_ms) insightPollMs = snap.poll_ms;
  }

  /* The server could not be reached. This is the one message in the column the
     browser writes for itself, and it is deliberately about the link to the
     server rather than about the car — we know nothing about the car. The rows
     keep whatever they last showed; blanking them would claim the sensors had
     gone quiet, which is a different failure. */
  function mountOffline() {
    var banner = $('vehbanner');
    if (!banner) return;
    banner.replaceChildren(renderStatusBanner({
      status: 'UNAVAILABLE',
      status_text: 'No Response From Server',
      connection: 'No Link',
      connection_state: 'lost'
    }));
  }

  // ---- polling -----------------------------------------------------------

  /* Chained timeouts, not setInterval — the same discipline as the perception,
     headway and tire loops: a slow response must not let calls stack up. */
  function schedule(ms) {
    clearTimeout(timer);
    timer = setTimeout(tick, ms);
  }

  function scheduleInsights(ms) {
    clearTimeout(insightTimer);
    insightTimer = setTimeout(tickInsights, ms);
  }

  async function tick() {
    /* A backgrounded tab does not need telemetry. It picks straight back up on
       visibilitychange below, so nothing is lost by not asking. */
    if (document.hidden) { schedule(pollMs || RETRY_MS); return; }
    try {
      var r = await fetch('/vehicle/telemetry', { cache: 'no-store' });
      mountTelemetry(await r.json());
      schedule(pollMs || RETRY_MS);
    } catch (e) {
      mountOffline();
      schedule(RETRY_MS);
    }
  }

  async function tickInsights() {
    if (document.hidden) { scheduleInsights(insightPollMs || INSIGHT_RETRY_MS); return; }
    try {
      var r = await fetch('/vehicle/insights', { cache: 'no-store' });
      mountInsights(await r.json());
      scheduleInsights(insightPollMs || INSIGHT_RETRY_MS);
    } catch (e) {
      scheduleInsights(INSIGHT_RETRY_MS);
    }
  }

  async function setScenario(url, name) {
    try {
      var r = await fetch(url + '?name=' + encodeURIComponent(name), { method: 'POST' });
      var snap = await r.json();
      /* Both POSTs answer with a full snapshot, so the panel repaints on this
         round trip instead of showing the old state until the next poll. The
         tire endpoint answers in the tire payload's shape, which has no rows —
         fall through to a normal telemetry read in that case. */
      if (snap && snap.rows) { mountTelemetry(snap); schedule(pollMs || RETRY_MS); }
      else tick();
    } catch (e) {
      tick();
    }
  }

  function init() {
    if (wired) return;
    wired = true;

    var ecu = $('vehscenario');
    if (ecu) {
      ecu.addEventListener('change', function () {
        setScenario('/vehicle/telemetry/scenario', ecu.value);
      });
    }
    var tire = $('tirescenario');
    if (tire) {
      tire.addEventListener('change', function () {
        setScenario('/vehicle/tires/scenario', tire.value);
      });
    }

    document.addEventListener('visibilitychange', function () {
      if (!document.hidden) { tick(); tickInsights(); }
    });

    tick();
    tickInsights();
  }

  // ---- exports -----------------------------------------------------------

  root.RIO = root.RIO || {};
  root.RIO.vehicle = {
    renderStatusBanner: renderStatusBanner,
    renderTelemetryList: renderTelemetryList,
    renderTelemetryRow: renderTelemetryRow,
    renderInsights: renderInsights,
    renderInsightCard: renderInsightCard
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})(window);
