/* rio_dtc.js — Flagged Error Codes, RIO-Observed Conditions, and the report button.
 *
 * This file paints. It does not decide.
 *
 * There is no threshold in here, no severity ordering, no status label and no
 * grouping rule. GET /vehicle/dtc returns groups that are already in §17.3's
 * order, each card already carrying its §17.6 status label, its severity label,
 * its provenance label and its wording. GET /vehicle/conditions returns RIO's
 * observations already separated from the codes and already sorted. Every line
 * below either writes one of those strings into an element or turns a name into
 * a CSS class — the same contract rio_vehicle.js holds, and for the same reason:
 * a status label computed in JavaScript is a label that will one day disagree
 * with the lifecycle that produced it.
 *
 * WHY CODES AND CONDITIONS ARE TWO SECTIONS AND NOT ONE LIST
 * ----------------------------------------------------------
 * Because they are different claims. A code is the VEHICLE saying something; a
 * condition is RIO saying something. §17.8 asks for them under different
 * headings and the reason is not tidiness — a driver who cannot tell them apart
 * has been told something false by a system that only said true things. The two
 * sections have different headings, different provenance lines, and no shared
 * container.
 *
 * WHY THIS FILE DOES NOT SPEAK
 * ----------------------------
 * Nothing here touches RIO.speech. A pending fault code is not an alert: it is
 * something the ECU has noticed and not confirmed, and interrupting a drive for
 * one would be exactly the behaviour §19.5 forbids. What speaks about the car is
 * rio_health.js, driven by a server-side policy that only ever fires on a
 * CRITICAL fault. The same firewall rio_vehicle.js has, unchanged.
 *
 * THE REPORT BUTTON IS THE ONE THING HERE THAT ASKS THE CAR
 * ---------------------------------------------------------
 * And it is a READ. It runs Modes 03, 07 and 0A and retrieves freeze frames;
 * none of them changes anything in the vehicle. There is no control anywhere in
 * this file that could clear a code, because no such endpoint exists to call.
 */
(function (root) {
  'use strict';

  /* Only used when the fetch itself failed — the real cadence comes from each
     payload's poll_ms, which we do not have when the request never landed. */
  var RETRY_MS = 8000;
  var DEFAULT_MS = 6000;

  var pollMs = null;
  var timer = null;
  var wired = false;
  var lastSig = null;

  // ---- tiny DOM helpers --------------------------------------------------

  function el(tag, cls, text) {
    var node = document.createElement(tag);
    if (cls) node.className = cls;
    if (text != null) node.textContent = text;
    return node;
  }

  function $(id) { return document.getElementById(id); }

  function cls(name) {
    return String(name || '').toLowerCase().replace(/[^a-z0-9]+/g, '-');
  }

  function field(parent, key, value) {
    if (value == null || value === '') return;
    var row = el('div', 'dtc-field');
    row.appendChild(el('span', 'dtc-key', key));
    row.appendChild(el('span', 'dtc-val', String(value)));
    parent.appendChild(row);
  }

  /* The server sends epoch seconds; a date is the one thing the browser is
     better placed to format, because it is the only party that knows the
     driver's timezone. */
  function when(ts) {
    if (!ts) return null;
    try {
      return new Date(ts * 1000).toLocaleString(undefined, {
        month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit'
      });
    } catch (e) { return null; }
  }

  // ---- CodeCard ----------------------------------------------------------

  /* One code. Every field §17.5 requires, in the order a person reads them:
     what it is, how sure the vehicle is, who said so, when, and only then what
     it might mean. Possible causes come LAST and are labelled as possibilities,
     because a card that led with them would read as a diagnosis. */
  function renderCodeCard(card) {
    card = card || {};
    var box = el('div', 'dtc-card sev-' + cls(card.severity)
                        + (card.early_detection ? ' dtc-early' : ''));

    var head = el('div', 'dtc-head');
    head.appendChild(el('span', 'dtc-code', card.code || ''));
    head.appendChild(el('span', 'dtc-status st-' + cls(card.status_label),
                        card.status_label || ''));
    box.appendChild(head);

    box.appendChild(el('div', 'dtc-desc', card.description || ''));

    var meta = el('div', 'dtc-meta');
    field(meta, 'Source', card.source_ecu
          ? 'Vehicle ECU (' + card.source_ecu + ')' : 'Vehicle ECU');
    field(meta, 'Detection Stage', card.dtc_category);
    field(meta, 'Affected System', card.system_label);
    field(meta, 'Severity', card.severity_label);
    field(meta, 'Check-Engine Light',
          card.mil_commanded_on == null ? null
            : (card.mil_commanded_on ? 'On' : 'Off'));
    field(meta, 'First Detected', when(card.first_seen_at));
    field(meta, 'Last Reported', when(card.last_seen_at));
    field(meta, 'Times Seen Pending', card.pending_scan_count);
    field(meta, 'Drives Observed', card.drive_cycle_count_observed);
    if (card.recurrence_count) field(meta, 'Returned', card.recurrence_count + '×');
    field(meta, 'Provenance', card.provenance_label);
    box.appendChild(meta);

    if (card.what_this_means) {
      var means = el('div', 'dtc-block');
      means.appendChild(el('div', 'dtc-block-head', 'What This Means'));
      means.appendChild(el('div', 'dtc-block-body', card.what_this_means));
      box.appendChild(means);
    }

    /* Possibilities, presented as possibilities. The cause status sits directly
       underneath so the two cannot be read apart. */
    var causes = el('div', 'dtc-block');
    causes.appendChild(el('div', 'dtc-block-head', 'Possible Causes'));
    if (card.possible_causes && card.possible_causes.length) {
      var list = el('ul', 'dtc-causes');
      card.possible_causes.forEach(function (c) {
        list.appendChild(el('li', null, c));
      });
      causes.appendChild(list);
    } else {
      causes.appendChild(el('div', 'dtc-block-body',
        card.manufacturer_specific
          ? 'This code is specific to the manufacturer. RIO does not have a '
            + 'definition for it and will not guess.'
          : 'RIO does not have a list of causes for this code.'));
    }
    causes.appendChild(el('div', 'dtc-cause-status',
                          'Cause: ' + (card.cause_status || 'Not Confirmed')));
    box.appendChild(causes);

    /* Freeze frame: the conditions the VEHICLE recorded. Labelled as historical
       every time it is shown, because a table of live-looking numbers next to a
       fault is read as live numbers. */
    if (card.freeze_frame_available && card.freeze_frame) {
      var ff = el('details', 'dtc-ff');
      ff.appendChild(el('summary', null,
                        'View Conditions Recorded When Fault Was Set'));
      var note = el('div', 'dtc-ff-note',
        'Recorded by the vehicle at the moment the fault was set. These are '
        + 'historical values, not live readings.');
      ff.appendChild(note);
      var table = el('div', 'dtc-ff-rows');
      Object.keys(card.freeze_frame).forEach(function (k) {
        if (k === 'source_ecu') return;
        field(table, k.split('.').pop().replace(/_/g, ' '),
              card.freeze_frame[k]);
      });
      ff.appendChild(table);
      box.appendChild(ff);
    }

    if (card.rio_snapshot_id) {
      box.appendChild(el('div', 'dtc-snap',
        'RIO also recorded the minute either side of this code appearing'
        + (card.rio_snapshot_complete ? '.' : ' (still being captured).')));
    }
    return box;
  }

  // ---- FlaggedErrorCodes -------------------------------------------------

  function renderCodes(section) {
    section = section || {};
    var frag = document.createDocumentFragment();

    if (!section.ecu_responding) {
      var lost = el('div', 'dtc-empty dtc-warn');
      lost.appendChild(el('div', null,
        'The vehicle’s computer is not answering diagnostic requests.'));
      lost.appendChild(el('div', 'dtc-caveat',
        'Nothing about fault codes can be judged while this is true. This is a '
        + 'connection problem, not a statement about the engine.'));
      frag.appendChild(lost);
      return frag;
    }

    if (!section.groups || !section.groups.length) {
      var empty = el('div', 'dtc-empty');
      empty.appendChild(el('div', null, section.empty_state || ''));
      /* The caveat comes from the server, deliberately. It is the one sentence
         in this section most likely to be paraphrased into something false. */
      if (section.empty_state_caveat) {
        empty.appendChild(el('div', 'dtc-caveat', section.empty_state_caveat));
      }
      frag.appendChild(empty);
      return frag;
    }

    section.groups.forEach(function (group) {
      frag.appendChild(el('div', 'dtc-group', group.title));
      group.cards.forEach(function (card) {
        frag.appendChild(renderCodeCard(card));
      });
    });
    return frag;
  }

  // ---- RIOObservedConditions ---------------------------------------------

  /* Deliberately a different shape from a code card. Same information density
     would invite the eye to read them as the same kind of thing, and they are
     not: one is the vehicle's verdict and one is RIO's inference. */
  function renderConditions(payload) {
    payload = payload || {};
    var frag = document.createDocumentFragment();
    var rows = payload.conditions || [];

    if (!rows.length) {
      var empty = el('div', 'dtc-empty');
      empty.appendChild(el('div', null, payload.empty_state || ''));
      if (payload.empty_state_caveat) {
        empty.appendChild(el('div', 'dtc-caveat', payload.empty_state_caveat));
      }
      frag.appendChild(empty);
    }

    rows.forEach(function (c) {
      var box = el('div', 'cond-card sev-' + cls(c.severity));
      var head = el('div', 'cond-head');
      head.appendChild(el('span', 'cond-prov', c.provenance_label || 'Observed by RIO'));
      head.appendChild(el('span', 'cond-sev st-' + cls(c.severity), c.severity || ''));
      box.appendChild(head);
      box.appendChild(el('div', 'cond-text', c.observation || ''));

      var meta = el('div', 'dtc-meta');
      field(meta, 'Based on', c.observation_window);
      if (c.confidence != null) field(meta, 'Confidence', c.confidence);
      field(meta, 'First observed', when(c.first_observed_at));
      if (c.supporting_signals && c.supporting_signals.length) {
        field(meta, 'Supporting signals', c.supporting_signals.join(', '));
      }
      field(meta, 'Possible explanation', c.possible_explanation);
      field(meta, 'Cause', c.cause_confirmed ? 'Confirmed' : 'Not Confirmed');
      box.appendChild(meta);
      frag.appendChild(box);
    });

    /* What RIO could not look at. Shown, and shown apart — a panel that hid its
       own blind spots would let an unmonitored subsystem read as a healthy one,
       and one that mixed them in would make an unmonitored car look troubled. */
    if (payload.gap_count) {
      frag.appendChild(el('div', 'dtc-group', 'Not Evaluated'));
      (payload.coverage_gaps || []).forEach(function (g) {
        frag.appendChild(el('div', 'cond-gap', g.observation || ''));
      });
    }
    return frag;
  }

  // ---- DiagnosticReportControl -------------------------------------------

  var reportBusy = false;

  function renderReportProgress(view) {
    var host = $('vehreport');
    if (!host) return;
    host.textContent = '';

    var btn = el('button', 'dtc-run', 'Run Diagnostic Report');
    btn.disabled = reportBusy;
    btn.addEventListener('click', runReport);
    host.appendChild(btn);

    if (!view) return;

    /* The stage list comes from the server and is walked in order. The driver
       is watching a list of questions being asked; a spinner would be less
       honest rather than more compact. */
    var stages = view.stages || [];
    var idx = view.stage_index == null ? -1 : view.stage_index;
    var list = el('div', 'dtc-stages');
    stages.forEach(function (name, i) {
      var state = i < idx ? 'done' : (i === idx ? 'now' : 'todo');
      var row = el('div', 'dtc-stage dtc-stage-' + state);
      row.appendChild(el('span', 'dtc-stage-dot', ''));
      row.appendChild(el('span', null, name));
      list.appendChild(row);
    });
    host.appendChild(list);

    if (view.status === 'failed') {
      host.appendChild(el('div', 'dtc-caveat',
                          'The report could not be completed: ' + (view.error || '')));
      return;
    }

    var body = view.report;
    if (!body) return;

    var summary = el('div', 'dtc-summary');
    summary.appendChild(el('div', 'dtc-block-head', 'Summary'));
    /* The server's own sentence, printed verbatim. It is assembled
       deterministically from the report's fields — see vehicle/report.py on why
       it must not be regenerated. */
    summary.appendChild(el('div', 'dtc-block-body', body.summary || ''));
    host.appendChild(summary);

    var counts = el('div', 'dtc-meta');
    field(counts, 'Detected early', body.early_detection.count);
    field(counts, 'Confirmed by the vehicle', body.confirmed_faults.count);
    field(counts, 'Observed by RIO', body.rio_observations.count);
    if (body.rio_observations.gap_count) {
      field(counts, 'Could not evaluate', body.rio_observations.gap_count);
    }
    field(counts, 'Generated', when(body.generated_at));
    host.appendChild(counts);
  }

  async function runReport() {
    if (reportBusy) return;
    reportBusy = true;
    renderReportProgress({ stages: null, stage_index: 0, status: 'running' });
    try {
      var vid = (RIO.vehicleId || 'vehicle_prototype_01');
      var sid = (typeof RIO !== 'undefined' && RIO.getSessionId)
        ? RIO.getSessionId() : null;
      var url = '/api/v1/vehicles/' + encodeURIComponent(vid)
        + '/diagnostic-reports' + (sid ? '?session_id=' + encodeURIComponent(sid) : '');
      var r = await fetch(url, { method: 'POST' });
      renderReportProgress(await r.json());
    } catch (e) {
      renderReportProgress({ status: 'failed', error: String(e), stages: [] });
    } finally {
      reportBusy = false;
      /* Re-enable the button without discarding what came back. */
      var btn = document.querySelector('#vehreport .dtc-run');
      if (btn) btn.disabled = false;
    }
  }

  // ---- mounting ----------------------------------------------------------

  function mount(section, conditions) {
    /* Rebuild only when the content actually changed. These sections move on
       the order of minutes while this polls on the order of seconds, and
       replacing the nodes underneath a driver who has a freeze frame open would
       close it in their face. */
    var sig = JSON.stringify([section && section.groups,
                              section && section.ecu_responding,
                              conditions && conditions.conditions,
                              conditions && conditions.coverage_gaps]);
    if (sig === lastSig) return;
    lastSig = sig;

    var codes = $('vehdtc');
    if (codes) codes.replaceChildren(renderCodes(section));
    var cond = $('vehconditions');
    if (cond) cond.replaceChildren(renderConditions(conditions));
  }

  // ---- polling -----------------------------------------------------------

  /* Chained timeouts, not setInterval — the same discipline as every other loop
     on this page: a slow response must not let calls stack up. */
  function schedule(ms) {
    clearTimeout(timer);
    timer = setTimeout(tick, ms);
  }

  async function tick() {
    if (document.hidden) { schedule(pollMs || DEFAULT_MS); return; }
    try {
      var results = await Promise.all([
        fetch('/vehicle/dtc', { cache: 'no-store' }).then(function (r) { return r.json(); }),
        fetch('/vehicle/conditions', { cache: 'no-store' }).then(function (r) { return r.json(); })
      ]);
      if (results[1] && results[1].poll_ms) pollMs = results[1].poll_ms;
      mount(results[0], results[1]);
      schedule(pollMs || DEFAULT_MS);
    } catch (e) {
      schedule(RETRY_MS);
    }
  }

  function init() {
    if (wired) return;
    wired = true;
    renderReportProgress(null);
    document.addEventListener('visibilitychange', function () {
      if (!document.hidden) tick();
    });
    tick();
  }

  // ---- exports -----------------------------------------------------------

  root.RIO = root.RIO || {};
  root.RIO.dtc = {
    renderCodeCard: renderCodeCard,
    renderCodes: renderCodes,
    renderConditions: renderConditions,
    runReport: runReport
  };

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})(window);
