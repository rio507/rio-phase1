/* rio_tires.js — the Vehicle Health column.
 *
 * This file paints. It does not decide.
 *
 * There is no threshold in here, no target pressure, no temperature band and no
 * arithmetic on a tire value. GET /vehicle/tires returns a state name and a set
 * of already-formatted strings per corner, and every line below either writes
 * one of those strings into an element or turns a state name into a CSS class.
 * That is the whole contract: when the mock provider is replaced by a Bluetooth
 * receiver, an RF TPMS dongle, an ESP32 or RIO Connect, this file does not
 * change and does not need to know it happened. See tires.py.
 *
 * RIO does not speak about tires in this phase. Nothing here touches
 * RIO.speech — the arbiter is not involved, deliberately. Tire-aware speech is a
 * later phase and goes out at coaching priority when it arrives.
 */
(function () {
  'use strict';

  var CORNERS = ['FL', 'FR', 'RL', 'RR'];

  /* The ONLY interval in this file, and it is not a tire setting: it is how long
   * to wait before trying the server again after the fetch itself failed. The
   * real cadence always comes from the payload's poll_ms (config.TIRE_POLL_MS),
   * which we do not have when the request never landed. */
  var RETRY_MS = 4000;

  var pollMs = null;
  var timer = null;
  var wired = false;

  function $(id) { return document.getElementById(id); }

  /* Swap the st-* class without disturbing anything else on the element. */
  function setState(el, state) {
    if (!el) return;
    var keep = [];
    (el.getAttribute('class') || '').split(/\s+/).forEach(function (c) {
      if (c && c.indexOf('st-') !== 0) keep.push(c);
    });
    keep.push('st-' + state);
    el.setAttribute('class', keep.join(' '));
  }

  function field(block, name, value) {
    var el = block.querySelector('[data-f="' + name + '"]');
    if (el) el.textContent = value == null ? '' : value;
    return el;
  }

  // ---- rendering ---------------------------------------------------------

  function renderTire(t) {
    var block = document.querySelector('.tire[data-corner="' + t.corner + '"]');
    if (!block) return;

    field(block, 'pressure', t.pressure_text);
    field(block, 'unit', t.pressure_unit);
    field(block, 'temp', t.temp_text);
    field(block, 'note', t.note);

    var link = field(block, 'link', t.link_text);
    if (link) link.className = 'tire-link lk-' + t.link_state;
    var batt = field(block, 'battery', t.battery_text);
    if (batt) batt.className = 'tire-batt bt-' + t.battery_state;

    /* The long-form reason, for anyone who hovers. The panel itself stays at
       four words per corner — that is the point of it. */
    block.title = t.detail || '';

    setState(block, t.state);
    setState(document.querySelector('.tmap-wheel[data-wheel="' + t.corner + '"]'), t.state);
  }

  function renderBanners(banners) {
    var host = $('tirealerts');
    if (!host) return;
    host.textContent = '';
    (banners || []).forEach(function (b) {
      var box = document.createElement('div');
      box.className = 'tire-alert lv-' + b.level;
      var title = document.createElement('div');
      title.className = 'tire-alert-title';
      title.textContent = b.title;
      box.appendChild(title);
      if (b.detail) {
        var detail = document.createElement('div');
        detail.className = 'tire-alert-detail';
        detail.textContent = b.detail;
        box.appendChild(detail);
      }
      host.appendChild(box);
    });
  }

  function renderScenarios(snap) {
    var sel = $('tirescenario');
    if (!sel) return;
    /* A provider with no scenarios is a provider reading real tires. The dev
       control removes itself rather than sitting there doing nothing. */
    if (!snap.scenarios || !snap.scenarios.length) {
      sel.style.display = 'none';
      return;
    }
    sel.style.display = '';
    var want = snap.scenarios.map(function (s) { return s.name; }).join('|');
    if (sel.dataset.built !== want) {
      sel.textContent = '';
      snap.scenarios.forEach(function (s) {
        var opt = document.createElement('option');
        opt.value = s.name;
        opt.textContent = s.label;
        sel.appendChild(opt);
      });
      sel.dataset.built = want;
    }
    if (snap.scenario && sel.value !== snap.scenario) sel.value = snap.scenario;
  }

  function render(snap) {
    var status = $('tirestatus');
    if (status) {
      status.className = 'veh-status veh-' + snap.status;
      $('tirestatustext').textContent = snap.status_text;
      $('tireupdated').textContent = snap.updated_display;
    }
    (snap.tires || []).forEach(renderTire);
    renderBanners(snap.banners);
    renderScenarios(snap);
    if (snap.poll_ms) pollMs = snap.poll_ms;
  }

  /* The server could not be reached. This is the one message in the column the
     browser writes for itself, and it is deliberately about the link to the
     server rather than about the tires — we know nothing about the tires. The
     corners keep whatever they last showed; blanking them would claim the
     sensors had gone quiet, which is a different failure. */
  function renderOffline() {
    var status = $('tirestatus');
    if (!status) return;
    status.className = 'veh-status veh-UNAVAILABLE';
    $('tirestatustext').textContent = 'No Response From Server';
  }

  // ---- polling -----------------------------------------------------------

  /* Chained timeout, not setInterval — the same discipline as the perception
     and headway loops: a slow response must not let calls stack up. */
  function schedule(ms) {
    clearTimeout(timer);
    timer = setTimeout(tick, ms);
  }

  async function tick() {
    /* A backgrounded tab does not need tire readings. It picks straight back up
       on visibilitychange below, so nothing is lost by not asking. */
    if (document.hidden) { schedule(pollMs || RETRY_MS); return; }
    try {
      var r = await fetch('/vehicle/tires', { cache: 'no-store' });
      render(await r.json());
      schedule(pollMs || RETRY_MS);
    } catch (e) {
      renderOffline();
      schedule(RETRY_MS);
    }
  }

  async function setScenario(name) {
    try {
      var r = await fetch('/vehicle/tires/scenario?name=' + encodeURIComponent(name),
                          { method: 'POST' });
      var snap = await r.json();
      /* The POST answers with a full snapshot, so the panel repaints on this
         round trip instead of showing the old state until the next poll. */
      if (snap && snap.status) { render(snap); schedule(pollMs || RETRY_MS); }
      else tick();
    } catch (e) {
      tick();
    }
  }

  function init() {
    if (wired) return;
    wired = true;

    var sel = $('tirescenario');
    if (sel) sel.addEventListener('change', function () { setScenario(sel.value); });

    document.addEventListener('visibilitychange', function () {
      if (!document.hidden) tick();
    });

    tick();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
