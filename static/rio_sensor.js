/* rio_sensor.js — the Perception card, as the sensor column it now is.
 *
 * WHAT THIS REPLACED, AND WHY IT HAD TO BE REPLACED
 * ------------------------------------------------
 * One div, one string, `obs.textContent = j.observation`. That was adequate
 * while the resident model was asked for a sentence about the road. It is not
 * adequate now that it is asked for a READING -- three named fields on one
 * line, separated by bars -- because the box showed the separators and the
 * field names as prose and a driver had to parse the format by eye:
 *
 *     ROAD: four, two, asphalt, moderate | TRAFFIC: car 20 km/h, car 40 km/h |
 *     RISK: none seen
 *
 * and, on 2026-09-21, this, which read as a description of a road:
 *
 *     ROAD: single|single|curb|CURB|CURB|CURB|CURB|CURB|CURB|CURB|...
 *
 * So the reading gets the layout the Cosmos-Reason2 teacher column had before
 * the teachers came out of the live stack -- a named instrument, its own meta
 * row, and one field per row -- fed by the live observer instead of by a
 * teacher pass. The CSS is that column's, unchanged and shared: two cards that
 * show a model reading a road should look like the same kind of thing.
 *
 * THE THREE RULES THIS FILE KEEPS
 * -------------------------------
 * 1. IT PARSES NOTHING. The fields arrive already split, from
 *    rio_prompts.split_sensor_reading on the server -- the same call
 *    realtime.look() makes for the payload the live session is given. A second
 *    parser here would be a second opinion about what the model said, and the
 *    first time the two disagreed the glass would be lying about her input.
 *
 * 2. A MISSING FIELD IS SAID, NOT LEFT BLANK. The model was asked for three
 *    fields; two is a fact about the reading and an empty row is a fact about
 *    nothing. `present: false` gets words.
 *
 * 3. IT NAMES NO MODEL. The instrument's name comes from the reading, which
 *    got it from config.local_vision_label(). tools/local_vision_selftest.py
 *    section 6 checks this page for hard-coded checkpoints, and a card that
 *    said "Cosmos" would be out of date the next time the eye changed.
 */
(function (root) {
  'use strict';

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (text != null) n.textContent = text;
    return n;
  }

  function card() { return document.getElementById('sensorcard'); }

  /* A short, honest age. Under a second is "just now" rather than "0.4 s ago":
     the reading is about a frame, frames arrive several times a second, and a
     decimal there is precision about nothing. */
  function ageText(s) {
    if (typeof s !== 'number' || !isFinite(s)) return 'age unknown';
    if (s < 1) return 'just now';
    if (s < 10) return s.toFixed(1) + ' s ago';
    return Math.round(s) + ' s ago';
  }

  function meta(reading) {
    var m = el('div', 'teach-meta');
    function pair(label, value, stale) {
      var sp = el('span', stale ? 'stale' : null);
      sp.appendChild(el('i', null, label + ' '));
      sp.appendChild(el('b', null, value));
      m.appendChild(sp);
    }
    /* THE AGE, WHICH IS THE FIELD THAT MAKES THE REST OF THE CARD SAFE.
       A description of a road is true for a second or two at speed and false
       after ten, and `fresh_s` is the observer's OWN threshold -- the age past
       which it stops serving a reading as current -- shipped with the reading
       rather than guessed at here. Over it, the row goes warn-coloured: what
       is on the glass is then a description of a moment that has passed. */
    var age = reading.age_s;
    var fresh = (typeof reading.fresh_s === 'number') ? reading.fresh_s : 2;
    pair('read', ageText(age), typeof age === 'number' && age > fresh);
    if (reading.frame_id) pair('frame', reading.frame_id);
    /* An instrument's reading is never spoken in her voice. Said on the card
       so nobody reads these words as words RIO would use -- she is given them
       as evidence and composes her own sentence. See observer._record. */
    pair('role', reading.speakable ? 'her words' : 'sensor · she rephrases');
    return m;
  }

  function fieldRow(f) {
    var row = el('div', 'teach-field');
    // Title case for the row label; the wire keeps the model's own casing.
    var name = f.name || '';
    row.appendChild(el('div', 'teach-label',
                       name.charAt(0) + name.slice(1).toLowerCase()));
    if (f.present) {
      row.appendChild(el('div', 'teach-text', f.text));
    } else {
      /* RULE 2. Not blank, and not "--": the sentence says which of the two
         things happened, because "the model did not report this" and "the
         model reported nothing" are different faults. */
      row.appendChild(el('div', 'teach-text empty',
                         'Not in this reading — the model gave the other '
                         + 'fields and not this one.'));
    }
    return row;
  }

  /* THE WHOLE READING, UNEDITED, UNDER THE ROWS.
     The rows are a rendering; this is the string. It is the same string
     realtime.look() hands the live session as `answer`, off the same observer
     record, so the card can be checked against what she was told by reading
     both -- which is what tools/sensor_card_selftest.py does automatically. */
  function rawRow(reading) {
    var row = el('div', 'teach-field');
    row.appendChild(el('div', 'teach-label', 'As given to RIO'));
    var body = el('div', 'teach-text', reading.raw || '');
    row.appendChild(body);
    if ((reading.raw || '').length > 90) {
      var more = el('button', 'teach-more', 'Show all');
      more.type = 'button';
      more.addEventListener('click', function () {
        var open = row.classList.toggle('open');
        more.textContent = open ? 'Show less' : 'Show all';
      });
      row.appendChild(more);
    }
    return row;
  }

  /* A reading the parser could not find any field in. Shown whole rather than
     rendered as three empty rows -- three "not in this reading" rows would say
     the model answered badly, and what actually happened is that it answered
     in a different shape. */
  function unparsedRow(reading) {
    var row = el('div', 'teach-field');
    row.appendChild(el('div', 'teach-label', 'Reading'));
    row.appendChild(el('div', 'teach-text', reading.extra || reading.raw || ''));
    var hint = el('div', 'teach-hint',
                  'not in the three-field sensor format — shown as written');
    row.appendChild(hint);
    return row;
  }

  function shell(nameText, freshClass) {
    var cols = el('div', 'teach-cols');
    var col = el('div', 'teach-col sensor-col' + (freshClass ? ' is-fresh' : ''));
    var head = el('div', 'teach-name');
    head.appendChild(el('span', 'teach-dot'));
    head.appendChild(document.createTextNode(nameText));
    col.appendChild(head);
    cols.appendChild(col);
    return { cols: cols, col: col };
  }

  function paint(node) {
    var c = card();
    if (!c) return;
    c.textContent = '';
    c.appendChild(node);
  }

  /* One plain line, for the states that are not a reading: no reading yet, a
     frame being looked at, a failure. Same shell so the card does not change
     shape under the eye while a drive is running. */
  function note(text, nameText) {
    var s = shell(nameText || 'Perception', false);
    var row = el('div', 'teach-field');
    row.appendChild(el('div', 'teach-text empty', text));
    s.col.appendChild(row);
    paint(s.cols);
  }

  var lastName = 'Perception';

  var api = {
    /* The /perceive response, whole. Everything this needs is on it. */
    render: function (j) {
      j = j || {};
      var reading = j.reading;
      if (!reading || !reading.raw) {
        /* No observer record for this key yet -- or a caption from
           perceive's own prompt, which is prose and has no fields. Both are
           real states and both say which one they are. */
        if (j.caption_source === 'perceive' && j.observation) {
          var s2 = shell(lastName, true);
          s2.col.appendChild(meta({ age_s: j.caption_age_s, speakable: false,
                                    fresh_s: 2 }));
          var r2 = el('div', 'teach-field');
          r2.appendChild(el('div', 'teach-label', 'Caption'));
          r2.appendChild(el('div', 'teach-text', j.observation));
          r2.appendChild(el('div', 'teach-hint',
                            'the frame-by-frame caption, not the running '
                            + 'sensor reading'));
          s2.col.appendChild(r2);
          paint(s2.cols);
          return;
        }
        note(j.observation || 'No reading yet.', lastName);
        return;
      }
      if (reading.model) lastName = reading.model;
      var fresh = (typeof reading.fresh_s === 'number') ? reading.fresh_s : 2;
      var s = shell(reading.model || lastName,
                    typeof reading.age_s === 'number' && reading.age_s <= fresh);
      s.col.appendChild(meta(reading));
      if (reading.parsed) {
        (reading.fields || []).forEach(function (f) {
          s.col.appendChild(fieldRow(f));
        });
        /* Anything the model wrote outside the three fields. Kept, because a
           model that added a sentence to a sensor format has done something
           worth seeing rather than something worth hiding. */
        if (reading.extra) {
          var ex = el('div', 'teach-field');
          ex.appendChild(el('div', 'teach-label', 'Also written'));
          ex.appendChild(el('div', 'teach-text', reading.extra));
          s.col.appendChild(ex);
        }
      } else {
        s.col.appendChild(unparsedRow(reading));
      }
      s.col.appendChild(rawRow(reading));
      paint(s.cols);
    },
    busy: function () { note('Looking…', lastName); },
    fail: function (e) { note('Could not read the frame: ' + e, lastName); },
  };

  root.RIO = root.RIO || {};
  root.RIO.sensor = api;

  if (typeof module !== 'undefined' && module.exports) module.exports = api;
})(typeof window !== 'undefined' ? window : globalThis);
