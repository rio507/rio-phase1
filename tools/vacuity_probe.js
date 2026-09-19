/* vacuity_probe.js — preloaded, records assertions that had nothing to assert.
 *
 *   node --require tools/vacuity_probe.js tools/<suite>.js
 *
 * tools/suite_sweep.js preloads this into every suite it runs. It is not a
 * suite, it asserts nothing, and it changes no behaviour.
 *
 * WHY EVIDENCE AND NOT A PATTERN MATCH. suite_sweep first looked for the vacuous
 * pass by grepping for the shapes that can be one -- every(), !some(),
 * filter().length === 0 -- and found 220 of them across sixteen suites. That
 * number is useless. Almost all of them are correct NEGATIVE assertions, where
 * an empty collection is the whole point: "nothing was synthesised behind her
 * back", "the session was asked for nothing", "no response was cancelled". A
 * list of 220 candidates nobody can triage is how a real finding gets lost.
 *
 * So this watches the collections themselves. It patches every, some and filter
 * to record when they are called on an EMPTY array, with the call site, which
 * turns "could be vacuous" into "was, on this run, over nothing". Across the
 * sixteen node suites that is eleven sites, all of them the negative kind, and
 * not one every() among them -- so today no node suite is vacuous, and the ones
 * that could become vacuous are the ones to guard (tools/assert_guard.js).
 *
 * WHAT IT STILL DOES NOT KNOW: whether an empty collection was INTENDED. That is
 * a question about what the assertion means, and only the call site can answer
 * it -- which is the argument for okAll/okNone, where the answer is written down
 * instead of inferred. This file finds the places worth asking about.
 */
'use strict';

const hits = new Map();
const CWD = process.cwd();

function note(kind) {
  if (this.length !== 0) return;
  /* The frame above the patched method. Stack shapes differ between node
     versions, so two frames are tried rather than one index being trusted. */
  const st = new Error().stack.split('\n');
  const frame = (st[3] || st[2] || '').trim();
  const m = frame.match(/\(?((?:\/|[A-Za-z]:)[^ ()]+:\d+:\d+)\)?$/);
  if (!m) return;
  /* Only the repo's own test code. A hit inside node internals or a dependency
     is not something anybody here can act on. */
  if (m[1].indexOf(CWD) !== 0) return;
  const key = kind + ' @ ' + m[1].slice(CWD.length + 1);
  hits.set(key, (hits.get(key) || 0) + 1);
}

['every', 'some', 'filter'].forEach((k) => {
  const orig = Array.prototype[k];
  Object.defineProperty(Array.prototype, k, {
    value: function () {
      try { note.call(this, k); } catch (e) {}
      return orig.apply(this, arguments);
    },
    writable: true, configurable: true,
  });
});

/* Printed on a marked line so suite_sweep can lift it out of stdout without
   parsing the suite's own output, and so a human running one suite by hand sees
   it too. `exit` rather than a summary call: the point is to report even when
   the suite dies part-way, which is when this is most interesting. */
process.on('exit', () => {
  if (!hits.size) return;
  const rows = [...hits.entries()].sort((a, b) => b[1] - a[1]);
  console.log('\n[vacuity] ' + rows.length + ' site(s) asserted over an empty '
              + 'collection on this run:');
  rows.forEach(([k, n]) => console.log(`[vacuity]   ${String(n).padStart(4)}x  ${k}`));
});
