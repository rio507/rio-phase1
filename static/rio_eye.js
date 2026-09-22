/* rio_eye.js — the eye's reading, laid out the way NVIDIA lays it out.
 *
 * WHY THIS IS NOT A SHAPE WE CHOSE
 * --------------------------------
 * cosmos-reason2/assets/outputs/*.log is what these models are meant to
 * produce, and it is four named blocks in a fixed order:
 *
 *     --------------------
 *     System:      You are a helpful assistant.
 *     --------------------
 *     User:        <the question, and the answer-format instruction>
 *     --------------------
 *     Reasoning:   <the <think> trace>
 *     --------------------
 *     Assistant:   <the final answer>
 *     --------------------
 *
 * Anyone who has run `cosmos-reason2-inference online --reasoning` has read
 * that exact layout. A card that contained Cosmos text in a shape we invented
 * would make them work out the mapping every time; this one they can read at a
 * glance, and a disagreement between our card and their CLI is then a bug
 * somebody can see rather than a translation nobody can check.
 *
 * THE BIGGEST DIFFERENCE FROM WHAT WE DID BEFORE IS `Reasoning`.
 * We used to strip the <think> trace and count it as a fault. NVIDIA neither
 * strips it nor hides it -- `--reasoning-parser qwen3` exists to hand it back
 * as its own field, and their sample logs print it above the answer. It is
 * also the single most useful thing on this card: a fabrication is usually
 * visible in the trace one step before it reaches the answer, and throwing it
 * away was throwing away the evidence.
 *
 * WHAT IS OURS, AND IT SAYS SO
 * ----------------------------
 * Three things on this card are not in NVIDIA's output and are marked as ours
 * with an explicit rule and label, because a reader who knows Cosmos should
 * never wonder whether the model produced them:
 *
 *   GIVEN      what the eye was handed -- tracks, gap, TTC, speed -- so a bad
 *              reading can be blamed on bad input or bad reasoning by looking,
 *              rather than by re-running it.
 *   CHECKED    which measurements in the answer came from our geometry
 *              (sourced), which were invented and cut (unmeasured), and
 *              whether the road users it named are ones we are tracking.
 *   WINDOW     the span and the end time. Not "age": a window has a beginning
 *              and an end and a single age number would describe neither.
 *
 * IT NAMES NO MODEL. The instrument's name comes from the record, which got it
 * from config.local_vision_label(). A card with a checkpoint in a literal is
 * out of date the next time the eye changes -- this page has shipped that bug
 * twice, and tools/local_vision_selftest.py section 6 greps for it.
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

  /* A window is a stretch, so it gets a stretch's description: how long it is
     and how long ago it ended. "4.2 s of road, ending 1.3 s ago" is the whole
     truth about when; "age 1.3 s" was true of one frame and is a lie about
     twenty-four. */
  function windowText(w) {
    if (!w) return 'no window';
    var span = (typeof w.span_s === 'number') ? w.span_s.toFixed(1) + ' s' : '?';
    var end = w.end_age_s;
    var endTxt = (typeof end !== 'number' || !isFinite(end)) ? 'unknown'
               : end < 1 ? 'just now'
               : end < 10 ? end.toFixed(1) + ' s ago'
               : Math.round(end) + ' s ago';
    return span + ' of road, ending ' + endTxt;
  }

  function metaRow(rec) {
    var m = el('div', 'teach-meta');
    function pair(label, value, warn) {
      var sp = el('span', warn ? 'stale' : null);
      sp.appendChild(el('i', null, label + ' '));
      sp.appendChild(el('b', null, value));
      m.appendChild(sp);
    }
    var w = rec.window || {};
    pair('window', windowText(w), (w.end_age_s || 0) > 6);
    pair('frames', (w.n_frames || 0) + ' @ ' + (w.fps || '?') + ' fps');
    /* HOW MUCH OF THE WINDOW THE TOWER ACTUALLY SAW. grid_thw[0] is the number
       of temporal groups the vision tower built, which is frames / 2. A window
       whose metadata was wrong silently collapses to 2 here while everything
       else on the card still says twenty-four, so this is the field that makes
       that failure visible instead of invisible. */
    if (w.grid_thw && w.grid_thw.length === 3) {
      pair('tower', w.grid_thw[0] + '×' + w.grid_thw[1] + '×'
                  + w.grid_thw[2]);
    }
    if (rec.timing && rec.timing.gen_ms) {
      pair('generate', (rec.timing.gen_ms / 1000).toFixed(1) + ' s');
    }
    if (w.dropped_unverified) {
      pair('dropped', w.dropped_unverified + ' unverified', true);
    }
    /* An instrument's reading is never spoken as hers, and the video path
       changes nothing about that. Said here so nobody reads these words as
       words RIO would use. */
    pair('role', 'sensor · she rephrases');
    return m;
  }

  /* One of NVIDIA's four blocks. The rule above the label is their separator
     line; the label is their field name, spelled their way. */
  function block(name, text, opts) {
    opts = opts || {};
    var row = el('div', 'eye-block' + (opts.cls ? ' ' + opts.cls : ''));
    row.appendChild(el('div', 'eye-rule'));
    var head = el('div', 'teach-label', name);
    if (opts.note) {
      var n = el('span', opts.warn ? 'teach-recited' : 'teach-recited weak',
                 ' ' + opts.note);
      if (opts.noteTitle) n.title = opts.noteTitle;
      head.appendChild(n);
    }
    row.appendChild(head);
    var body = el('div', 'teach-text' + (opts.textCls ? ' ' + opts.textCls : ''),
                  text || '');
    if (!text) {
      body.className += ' empty';
      body.textContent = opts.emptyText || '—';
    }
    row.appendChild(body);
    if ((text || '').length > 160) {
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

  /* OURS, AND LABELLED AS OURS. Everything below the "RIO" rule is measured by
     this car or computed by this server; none of it came from the model. The
     separation is the point -- see this file's header. */
  function oursHeader(text) {
    var d = el('div', 'eye-ours-head');
    d.appendChild(el('span', 'eye-ours-tag', 'RIO'));
    d.appendChild(el('span', null, text));
    return d;
  }

  function givenBlock(state) {
    var row = el('div', 'eye-block eye-ours');
    row.appendChild(oursHeader('what the eye was given — measured, not '
                             + 'from the model'));
    var hw = (state && state.headway) || {};
    var ego = (state && state.ego) || {};
    var lane = (state && state.lane) || {};
    var grid = el('div', 'eye-given');

    function cell(label, value, src) {
      var c = el('div', 'eye-cell');
      c.appendChild(el('div', 'eye-cell-k', label));
      c.appendChild(el('div', 'eye-cell-v', value));
      if (src) c.appendChild(el('div', 'eye-cell-s', src));
      grid.appendChild(c);
    }
    cell('speed',
         (ego.speed_ms == null) ? 'none'
           : ego.speed_ms.toFixed(1) + ' m/s ('
             + Math.round(ego.speed_ms * 3.6) + ' km/h)',
         ego.speed_source || 'no source');
    cell('lead gap',
         (hw.gap_m == null) ? 'none' : hw.gap_m.toFixed(1) + ' m',
         (hw.gap_m == null) ? (hw.gap_invalid_reason || 'no lead')
                            : 'depth + corridor');
    cell('time to contact',
         (hw.ttc_s == null) ? 'none' : hw.ttc_s.toFixed(1) + ' s',
         'headway filter');
    cell('band', hw.band || 'unknown', 'state machine');
    cell('range check', hw.plausibility || 'n/a', 'size plausibility');
    cell('lanes',
         (lane.n_detected == null) ? 'none' : String(lane.n_detected),
         'lane net' + (lane.source ? ' · ' + lane.source : ''));
    row.appendChild(grid);

    var tracks = (state && state.tracks) || [];
    var tr = el('div', 'eye-tracks');
    if (!tracks.length) {
      tr.appendChild(el('div', 'teach-text empty',
                        'no road users tracked in this window'));
    } else {
      tracks.slice(0, 8).forEach(function (t) {
        var line = el('div', 'eye-track');
        line.appendChild(el('span', 'eye-track-id', '#' + t.id));
        var bits = [t.label || 'object'];
        if (t.last_range_m != null) bits.push(t.last_range_m.toFixed(1) + ' m');
        else if (t.range_reject) bits.push('no range (' + t.range_reject + ')');
        if (t.side) bits.push(t.side);
        if (t.range_rate_ms != null) {
          bits.push((t.range_rate_ms > 0 ? 'opening ' : 'closing ')
                    + Math.abs(t.range_rate_ms).toFixed(1) + ' m/s');
        }
        if (t.is_lead) bits.push('lead');
        if (t.vulnerable) bits.push('vulnerable');
        line.appendChild(el('span', 'eye-track-t', bits.join(' · ')));
        tr.appendChild(line);
      });
      if (tracks.length > 8) {
        tr.appendChild(el('div', 'teach-text empty',
                          '… and ' + (tracks.length - 8) + ' more'));
      }
    }
    row.appendChild(tr);
    return row;
  }

  function checkedBlock(rec) {
    var row = el('div', 'eye-block eye-ours');
    row.appendChild(oursHeader('what was checked'));
    var list = el('div', 'eye-checks');

    function line(cls, label, text) {
      var d = el('div', 'eye-check ' + cls);
      d.appendChild(el('span', 'eye-check-k', label));
      d.appendChild(el('span', 'eye-check-v', text));
      list.appendChild(d);
    }

    var src = rec.sourced || [];
    var inv = rec.invented || [];
    var corr = rec.corroboration || {};

    if (src.length) {
      line('ok', 'sourced', src.map(function (s) {
        return s.text + ' ← ' + s.from;
      }).join('; '));
    }
    if (inv.length) {
      line('bad', 'unmeasured', inv.map(function (s) {
        return s.text;
      }).join('; ') + ' — cut from the answer: we did not measure it');
    }
    if (!src.length && !inv.length) {
      line('ok', 'numbers', 'the reading states no distance, speed or time');
    }
    if ((corr.cited || []).length) {
      line('ok', 'cites', corr.cited.map(function (i) { return '#' + i; })
                              .join(' '));
    }
    if ((corr.bad_cites || []).length) {
      line('bad', 'bad cites', corr.bad_cites.map(function (i) {
        return '#' + i;
      }).join(' ') + ' — no such track in this window');
    }
    /* THE ROW A CITATION COUNT CANNOT PRODUCE. A reading that cites a real
       track and then says the opposite of what was measured about it is using
       its grounding as decoration, and it looks identical to a good reading
       unless this line exists. */
    if ((corr.contradicts || []).length) {
      line('bad', 'contradicts', corr.contradicts.map(function (c) {
        return '#' + c.id + ' called ' + c.said + ', measured ' + c.measured
             + ' at ' + Math.abs(c.range_rate_ms).toFixed(1) + ' m/s';
      }).join('; '));
    }
    if ((corr.fabricated || []).length) {
      line('bad', 'not tracked', corr.fabricated.join(', ')
           + ' — named, and the detector held none in six seconds');
    }
    if ((corr.missed || []).length) {
      line('warn', 'not mentioned', corr.missed.map(function (m) {
        return '#' + m.id + ' ' + (m.label || '');
      }).join(', ') + ' — the loop held these and the reading did not '
        + 'name them');
    }
    if (rec.advisory && rec.advisory.length) {
      line('warn', 'advisory', rec.advisory.join(', ')
           + ' — an instruction to the driver, not an observation');
    }
    row.appendChild(list);

    /* THE LABEL ON THE WHOLE READING, which survives from the still card
       because the reason for it survives: Cosmos is right about the gist and
       fluent about specifics it cannot see. What changed is that the specifics
       it CAN now check -- every distance, every speed, every named track --
       are checked above, so this caveat is narrower than it was and says
       which part it still applies to. */
    var verdict = corr.verdict || 'unverified';
    row.appendChild(el('div', 'teach-unverified',
      verdict === 'corroborated'
        ? 'Every road user named is one the detector is tracking, and every '
        + 'measurement came from geometry. What is still the model’s own '
        + 'is the judgement about what matters — that is what it was asked '
        + 'for and it is not checkable here.'
        : verdict === 'uncited'
        ? 'The reading names road users without pointing at a track, so '
        + 'nothing in it can be matched to a measurement. Treat the specifics '
        + 'as an impression.'
        : 'Unverified — something in this reading does not match what was '
        + 'measured. The marked lines above say which.'));
    return row;
  }

  function paint(node) {
    var c = card();
    if (!c) return;
    c.textContent = '';
    c.appendChild(node);
  }

  function shell(nameText, fresh) {
    var cols = el('div', 'teach-cols');
    var col = el('div', 'teach-col sensor-col eye-col' + (fresh ? ' is-fresh' : ''));
    var head = el('div', 'teach-name');
    head.appendChild(el('span', 'teach-dot'));
    head.appendChild(document.createTextNode(nameText));
    col.appendChild(head);
    cols.appendChild(col);
    return { cols: cols, col: col };
  }

  var lastName = 'Perception';

  var api = {
    /* The /eye/window record, whole. */
    render: function (rec) {
      rec = rec || {};
      if (rec.ok === false || (!rec.answer && !rec.reasoning && !rec.refused)) {
        var s0 = shell(lastName, false);
        s0.col.appendChild(el('div', 'teach-text empty',
          rec.skipped === 'warming' ? 'Models still loading.'
          : rec.skipped === 'no_window'
            ? 'Not enough road yet — the ring is holding fewer than four '
              + 'verified frames.'
            : 'No reading yet.'));
        paint(s0.cols);
        return;
      }
      if (rec.model) lastName = rec.model;
      var w = rec.window || {};
      var s = shell(rec.model || lastName, (w.end_age_s || 99) <= 3);
      s.col.appendChild(metaRow(rec));

      /* NVIDIA'S FOUR BLOCKS, THEIR NAMES, THEIR ORDER. */
      s.col.appendChild(block('System', rec.system || ''));
      s.col.appendChild(block('User', rec.prompt || '', {
        note: rec.grounded ? 'includes measured state' : 'ungrounded',
        noteTitle: rec.grounded
          ? 'the measured block is part of the prompt — it is shown '
            + 'separately below as well, because it is ours and not the '
            + 'model’s'
          : 'this reading was taken without the measured block, for comparison'
      }));
      s.col.appendChild(block('Reasoning', rec.reasoning || '', {
        cls: 'eye-reasoning',
        emptyText: 'the model returned no trace',
        note: rec.refused === 'reasoning_unterminated' ? 'never closed' : null,
        warn: rec.refused === 'reasoning_unterminated',
        noteTitle: 'the trace filled the token budget, so there is no answer '
                 + 'after it'
      }));
      s.col.appendChild(block('Assistant', rec.answer || '', {
        textCls: 'prose',
        emptyText: rec.refused
          ? 'refused: ' + rec.refused
          : 'the model said nothing after its trace',
        note: rec.truncated ? 'cut off' : null,
        warn: !!rec.truncated,
        noteTitle: 'the reading hit its token budget mid-sentence — the '
                 + 'last clause is a fragment, not a finding'
      }));

      /* OURS. */
      if (rec.state && (rec.state.tracks || rec.state.headway)) {
        s.col.appendChild(givenBlock(rec.state));
      }
      s.col.appendChild(checkedBlock(rec));
      paint(s.cols);
    },

    windowText: windowText
  };

  root.RIOEye = api;
}(window));
