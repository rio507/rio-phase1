/* suite_sweep.js — which suites actually assert, and which only say they did.
 *
 *   node tools/suite_sweep.js            every browser/node suite
 *   node tools/suite_sweep.js --json     machine-readable, for a guard
 *   node tools/suite_sweep.js nav        only suites whose name matches
 *
 * WHY THIS EXISTS, WHICH IS TWO SUITES FOUND BY ACCIDENT.
 *
 * 23f4185: the realtime suite stopped running the moment node moved to the
 * volume. 463 checks had quietly stopped executing, and nothing said so --
 * `node tools/realtime_selftest.js` failed at the shell, not at a check, and a
 * failure at the shell is a thing you scroll past.
 *
 * Then source_selftest.js, found while looking at something else entirely:
 * it throws in stubPage on node 22, where `navigator` became a getter that
 * cannot be assigned. 39 assertion sites, none of them reached, exit code 1,
 * and nobody had looked at the exit code.
 *
 * Both were suites REPORTING NOTHING rather than reporting failure, and both
 * were found by luck. A suite that reports success without asserting is worse
 * than one that fails: the failing one gets fixed.
 *
 * HOW IT DECIDES, and why it is not a count of checks.
 *
 * Counting reported checks cannot find this class of bug, because the number is
 * printed by the suite -- the thing under suspicion. A suite that exits early
 * prints a smaller number and looks fine; a suite that crashes prints nothing
 * and looks like a shell problem.
 *
 * So the evidence is what the suite PRINTED, not what it summarised. All
 * sixteen suites share one convention -- every check writes a line beginning
 * `  ok    ` or `  FAIL  ` as it runs -- so counting those lines counts
 * assertions that actually executed, and it counts them whether the suite then
 * crashed, returned early, or lied in its total. The static count of `ok(` call
 * sites is the ceiling. Printed against ceiling is the measurement.
 *
 * V8 coverage (NODE_V8_COVERAGE) is collected alongside, for one job only:
 * naming the LINES that never ran, which a count cannot do. It is corroboration
 * and not the measurement, because V8 reports BLOCK coverage -- a site after a
 * throw, inside a block that was entered, is still inside a covered range. The
 * first version of this file used coverage as the measurement and reported
 * source_selftest.js as running 10 assertions when it prints none at all. A
 * sweep that over-counts execution is the bug it was written to find, so the
 * over-count is described here rather than left for the next person.
 *
 * WHAT IT STILL CANNOT SEE, stated so nobody reads a green sweep as more than
 * it is. Coverage proves an assertion RAN. It cannot prove the assertion had
 * anything to assert about: `ok(results.every(hasDate))` executes, passes, and
 * means nothing when `results` is empty. That is the vacuous pass, it is a
 * different failure from this one, and tools/assert_guard.js is what refuses
 * it. This file flags the shapes that CAN be vacuous (every/some/filter/length
 * over data that might be empty) so they can be looked at, and says plainly
 * that flagging is not proving.
 */
'use strict';

const fs = require('fs');
const path = require('path');
const os = require('os');
const { spawnSync } = require('child_process');

const TOOLS = __dirname;
const REPO = path.join(__dirname, '..');

/* The suites, and the one deliberate exclusion. This file is not a suite and
   sweeping itself would be a recursion with nothing at the bottom of it. */
const SELF = path.basename(__filename);

function suites(filter) {
  return fs.readdirSync(TOOLS)
    .filter(f => f.endsWith('.js') && f !== SELF)
    .filter((f) => {
      /* A suite is a file with its own `ok()` reporter. That is the convention
         every one of them follows, and it is a better test of "is this a suite"
         than the filename: nav_drive_replay.js has 22 assertion sites and does
         not say selftest anywhere. */
      const src = fs.readFileSync(path.join(TOOLS, f), 'utf8');
      return /^function ok\s*\(/m.test(src);
    })
    .filter(f => !filter || f.indexOf(filter) >= 0)
    .sort();
}

/* ---------------------------------------------------------------------------
   Assertion sites, by byte offset.

   A regex and not a parser, deliberately: what is wanted is the offset of every
   `ok(` that is a CALL, and the suites are hand-written files with one
   convention. The two things that would fool a naive match are handled --
   `function ok(` is the definition rather than a site, and an `ok(` inside a
   string or comment is not code. Comment and string stripping is
   character-by-character rather than by regex, because a regex that strips
   strings is a regex that eats apostrophes in prose.
   --------------------------------------------------------------------------- */
function assertionSites(src) {
  const masked = maskNonCode(src);
  const sites = [];
  const re = /\bok\s*\(/g;
  let m;
  while ((m = re.exec(masked)) !== null) {
    // The definition is not a call site.
    const before = masked.slice(Math.max(0, m.index - 20), m.index);
    if (/\bfunction\s+$/.test(before)) continue;
    sites.push({ offset: m.index, line: lineOf(src, m.index) });
  }
  return sites;
}

/* Replace every comment and string body with spaces, preserving offsets so that
   a match in the masked text is at the same byte as in the original. */
function maskNonCode(src) {
  const out = Buffer.from(src, 'utf8').toString('utf8').split('');
  let i = 0, n = src.length;
  const blank = (from, to) => {
    for (let k = from; k < to && k < n; k++) if (out[k] !== '\n') out[k] = ' ';
  };
  while (i < n) {
    const c = src[i], d = src[i + 1];
    if (c === '/' && d === '/') {
      let j = src.indexOf('\n', i); if (j < 0) j = n;
      blank(i, j); i = j; continue;
    }
    if (c === '/' && d === '*') {
      let j = src.indexOf('*/', i + 2); j = j < 0 ? n : j + 2;
      blank(i, j); i = j; continue;
    }
    if (c === '"' || c === "'" || c === '`') {
      let j = i + 1;
      while (j < n) {
        if (src[j] === '\\') { j += 2; continue; }
        if (src[j] === c) { j++; break; }
        j++;
      }
      blank(i + 1, j - 1); i = j; continue;
    }
    i++;
  }
  return out.join('');
}

function lineOf(src, offset) {
  let line = 1;
  for (let i = 0; i < offset && i < src.length; i++) if (src[i] === '\n') line++;
  return line;
}

/* ---------------------------------------------------------------------------
   Shapes that pass on nothing.

   Each of these is TRUE for an empty collection, so an assertion built on one
   proves nothing until something is known to be in the collection:

     arr.every(p)              true when arr is empty
     !arr.some(p)              ditto
     arr.filter(p).length===0  ditto
     arr.length === 0          ditto, when the intent was "nothing bad"
     arr.indexOf(x) < 0        ditto

   Flagged, not failed. Plenty of them are correct -- "no error events were
   emitted" is a real assertion about an empty list. The point is that the list
   is the thing to check, and these are where to look.
   --------------------------------------------------------------------------- */
const VACUOUS_SHAPES = [
  { name: 'every', re: /\.every\s*\(/ },
  { name: '!some', re: /![\w.]*\.some\s*\(/ },
  { name: 'filter-empty', re: /\.filter\s*\([^;]*\)\s*\.length\s*===?\s*0/ },
  { name: 'length-zero', re: /\.length\s*===?\s*0/ },
  { name: 'indexOf-absent', re: /\.indexOf\s*\([^;]*\)\s*<\s*0/ },
];

function vacuousCandidates(src, sites) {
  const masked = maskNonCode(src);
  const out = [];
  sites.forEach((s) => {
    /* The assertion's condition, roughly: from `ok(` to the comma that starts
       the message, or 400 bytes, whichever comes first. Rough is fine -- this
       is a pointer to a line, not a proof about it. */
    const chunk = masked.slice(s.offset, s.offset + 400);
    VACUOUS_SHAPES.forEach((v) => {
      if (v.re.test(chunk)) out.push({ line: s.line, shape: v.name });
    });
  });
  return out;
}

/* ---------------------------------------------------------------------------
   Run one suite under coverage.
   --------------------------------------------------------------------------- */
function run(file) {
  const src = fs.readFileSync(path.join(TOOLS, file), 'utf8');
  const sites = assertionSites(src);
  const covDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sweep-'));

  const t0 = Date.now();
  const res = spawnSync(process.execPath, [
    '--require', path.join(TOOLS, 'vacuity_probe.js'),
    path.join(TOOLS, file),
  ], {
    cwd: REPO,
    env: Object.assign({}, process.env, { NODE_V8_COVERAGE: covDir }),
    encoding: 'utf8',
    timeout: 10 * 60 * 1000,
    maxBuffer: 64 * 1024 * 1024,
  });
  const ms = Date.now() - t0;

  const stdout = res.stdout || '';
  const stderr = res.stderr || '';

  /* What the suite SAYS it did. Parsed from several formats because the suites
     do not agree on one, and a suite that reports no number at all is itself a
     finding -- it cannot tell 0 checks from 100. */
  const reported = parseReported(stdout);

  const cov = coldSites(covDir, file, sites);
  rmrf(covDir);

  /* THE MEASUREMENT. One line per check, written as the check runs, in all
     sixteen suites. Independent of the suite's own summary, which is the number
     under suspicion.
     THE PATTERN IS DELIBERATELY LOOSE ABOUT THE GAP. A first attempt allowed at
     most six spaces after the verb and reported realtime_selftest.js as having
     summarised four checks it never printed. It had printed all of them; four
     messages simply begin with their own indent ("   ...and still asked to
     answer"), pushing the gap to seven. The suite was honest and the sweep was
     wrong, which — in a file whose whole subject is tests that misreport — is
     worth a comment rather than a quiet regex change. Two or more spaces is the
     real convention: every helper pads the verb to a column. */
  const printed = (stdout.match(/^\s*(ok|FAIL)\s{2,}\S/gm) || []).length;

  /* SKIPS THE SUITE DECLARED. An optional section that announces itself is
     information; one that goes quiet is the bug. Both leave the same cold sites
     behind, so the only way to tell them apart is whether the suite said so --
     which is what tools/assert_guard.js `skips()` prints. Without this, a suite
     that has correctly declared its skip stays red in every future sweep, and a
     permanently red sweep is one people stop reading. */
  const declaredSkips = (stdout.match(/^\s*SKIP\s{2,}(.+)$/gm) || [])
    .map(l => l.replace(/^\s*SKIP\s+/, '').trim());

  return {
    file, ms, code: res.status, signal: res.signal,
    crashed: res.status !== 0,
    threw: /^\s*(\w*Error|Uncaught)/m.test(stderr) || /\n\s+at .*:\d+:\d+/.test(stderr),
    stderrHead: stderr.split('\n').filter(Boolean).slice(0, 3).join(' | '),
    sites: sites.length,
    ran: printed,
    cold: cov.cold,
    coverage: cov.usable,
    reported,
    /* The suite's own total against what it actually printed. A gap means the
       summary is not counting what ran -- in either direction. */
    summaryHonest: reported.checks === null ? null : reported.checks === printed,
    declaredSkips,
    /* Shapes that COULD be vacuous, from the source. Kept for the sites that
       never ran -- a cold site cannot be observed, so static is all there is
       for it -- and otherwise superseded by the line below. */
    vacuousShapes: vacuousCandidates(src, sites).length,
    /* Shapes that WERE vacuous on this run, from tools/vacuity_probe.js. This is
       the one worth reading: evidence rather than candidates. */
    vacuous: (stdout.match(/^\[vacuity\] {3}\s*\d+x {2}.+$/gm) || [])
      .map(l => l.replace(/^\[vacuity\]\s*/, '')),
  };
}

function parseReported(stdout) {
  let m;
  if ((m = stdout.match(/PASSED\s+(\d+)\s+checks/))) return { checks: +m[1], fails: 0 };
  if ((m = stdout.match(/FAILED\s+(\d+)\/(\d+)\s+checks/))) return { checks: +m[2], fails: +m[1] };
  if ((m = stdout.match(/(\d+)\/(\d+)\s+checks passed/))) return { checks: +m[2], fails: +m[2] - +m[1] };
  if ((m = stdout.match(/(\d+)\s+passed,\s*(\d+)\s+failed/))) return { checks: +m[1] + +m[2], fails: +m[2] };
  if ((m = stdout.match(/(\d+)\s+failure\(s\)\s+of\s+(\d+)\s+checks/))) return { checks: +m[2], fails: +m[1] };
  /* The shape that cannot answer the question: a pass/fail verdict with no
     count behind it. Recorded as such rather than as zero. */
  if (/PASS:\s*0\s*failure/.test(stdout)) return { checks: null, fails: 0 };
  if (/PASS|PASSED/.test(stdout)) return { checks: null, fails: 0 };
  return { checks: null, fails: null };
}

/* WHICH SITES V8 NEVER REACHED. Corroboration, not the measurement — see the
   header for why block coverage cannot be the measurement. A site named cold
   here really is cold; a site NOT named cold may still not have run. */
function coldSites(covDir, file, sites) {
  let zero = null;
  try {
    for (const f of fs.readdirSync(covDir)) {
      const data = JSON.parse(fs.readFileSync(path.join(covDir, f), 'utf8'));
      for (const script of data.result || []) {
        if (!script.url || !script.url.endsWith('/' + file)) continue;
        /* Every range V8 recorded with count 0 — the bytes that never ran.
           Collected across all functions; a site inside any of them is cold. */
        const z = [];
        for (const fn of script.functions || []) {
          for (const r of fn.ranges || []) {
            if (r.count === 0) z.push([r.startOffset, r.endOffset]);
          }
        }
        zero = zero ? zero.concat(z) : z;
      }
    }
  } catch (e) { zero = null; }

  if (zero === null) return { usable: false, cold: [] };
  const cold = sites.filter(s => zero.some(
    ([a, b]) => s.offset >= a && s.offset < b));
  return { usable: true, cold: cold };
}

function rmrf(p) { try { fs.rmSync(p, { recursive: true, force: true }); } catch (e) {} }

/* ---------------------------------------------------------------------------
   A verdict per suite. The categories are the ones that matter when the
   question is "is this suite doing anything".
   --------------------------------------------------------------------------- */
function verdict(r) {
  if (r.sites === 0) return { tag: 'NOT A SUITE', why: 'no assertion sites' };
  /* A POSITIVE CLAIM THAT RAN OVER NOTHING. `every()` on an empty array is true,
     so an assertion built on one passed without testing anything -- which is the
     vacuous pass, and it is a failure of the suite and not of the code.
     some() and filter() on empty are NOT failed here: those are how a correct
     NEGATIVE assertion is written ("nothing was synthesised", "no update was
     sent"), and measured across these sixteen suites every empty-collection call
     is that kind. Failing them would make the sweep cry wolf about eleven
     correct checks, and a sweep that cries wolf is one nobody reads.
     Where empty must be a pass or a failure DELIBERATELY, tools/assert_guard.js
     has okNone and okAll so the call site says which. */
  const vacuousPositives = r.vacuous.filter(v => /\bevery\b/.test(v));
  if (vacuousPositives.length) {
    return { tag: 'VACUOUS PASS',
             why: `${vacuousPositives.length} positive assertion(s) ran over an `
                + 'empty collection and passed without testing anything — '
                + vacuousPositives.join('; ') };
  }
  if (r.ran === 0) {
    return { tag: 'ASSERTS NOTHING',
             why: `printed no checks at all against ${r.sites} assertion sites`
                + (r.threw ? ' — threw: ' + r.stderrHead : '') };
  }
  /* Printed checks against static sites. It can legitimately EXCEED 1 -- an
     `ok(` inside a loop is one site and many checks -- so this is a floor on
     coverage and not a percentage of it. Which specific sites never ran is a
     question only V8's cold ranges can answer, and that is the next branch. */
  const frac = r.ran / r.sites;
  const coldLines = () => r.cold.slice(0, 8).map(c => c.line).join(', ')
                        + (r.cold.length > 8 ? ', …' : '');

  /* EXIT 1 IS WHAT A WORKING SUITE DOES WHEN A CHECK FAILS, and the first
     version of this file called that a crash -- which would have reported a
     suite doing its job exactly as it should as one of the broken ones, in a
     sweep whose whole purpose is telling those two apart. A non-zero exit is
     only a crash when the suite did NOT report failures of its own. */
  if (r.crashed && r.reported.fails > 0 && frac >= 0.75) {
    return { tag: 'FAILS HONESTLY',
             why: `${r.reported.fails} of ${r.ran} checks failed, exit ${r.code} `
                + '— the suite works; the code under it does not' };
  }
  if (r.crashed) {
    return { tag: 'CRASHED PART-WAY',
             why: `${r.ran} checks printed against ${r.sites} sites, exit ${r.code}`
                + (r.stderrHead ? ' — ' + r.stderrHead : '') };
  }
  if (frac < 0.75) {
    return { tag: 'EXITS EARLY',
             why: `only ${r.ran} checks printed against ${r.sites} assertion `
                + `sites (${Math.round(frac * 100)}%)` };
  }
  /* Ran plenty, and still has assertions that never execute. This is the
     quiet one: the suite looks healthy, the summary is a big number, and a
     section of it has been dead for however long. nav_selftest is here. */
  if (r.cold.length && r.declaredSkips.length) {
    return { tag: 'DECLARED SKIP',
             why: `${r.ran} checks ran; ${r.cold.length} site(s) dark in a `
                + `section the suite declared — ${r.declaredSkips.join('; ')}` };
  }
  if (r.cold.length) {
    return { tag: 'COLD SITES',
             why: `${r.ran} checks printed, but ${r.cold.length} assertion `
                + `site(s) never ran — lines ${coldLines()}` };
  }
  if (r.reported.checks === null) {
    return { tag: 'NO COUNT',
             why: `${r.ran} checks ran, but the suite reports no count — its `
                + 'verdict cannot tell 0 checks from all of them' };
  }
  if (r.summaryHonest === false) {
    return { tag: 'SUMMARY WRONG',
             why: `printed ${r.ran} checks but summarised ${r.reported.checks}` };
  }
  return { tag: 'RUNS', why: `${r.ran} checks ran, no cold sites` };
}

function main() {
  const args = process.argv.slice(2);
  const asJson = args.indexOf('--json') >= 0;
  const filter = args.filter(a => !a.startsWith('--'))[0];

  const list = suites(filter);
  if (!asJson) {
    console.log(`sweeping ${list.length} suites under NODE_V8_COVERAGE `
                + `(node ${process.version})\n`);
  }

  const results = [];
  for (const file of list) {
    const r = run(file);
    r.verdict = verdict(r);
    results.push(r);
    if (asJson) continue;
    const name = file.replace(/\.js$/, '');
    console.log(`${pad(name, 30)} ${pad(r.verdict.tag, 18)} ${r.verdict.why}`);
    if (r.vacuous.length) {
      console.log(`${' '.repeat(31)}asserted over an empty collection: `
                  + `${r.vacuous.length} site(s) `
                  + `(of ${r.vacuousShapes} that could)`);
      r.vacuous.forEach(v => console.log(`${' '.repeat(33)}${v}`));
    }
  }

  if (asJson) { console.log(JSON.stringify(results, null, 1)); return 0; }

  /* 'FAILS HONESTLY' is deliberately not in this list. A suite reporting a real
     failure is the system working; the sweep is looking for suites that report
     nothing. Those two have to stay separable or the sweep's own output becomes
     a thing people learn to ignore. */
  const bad = results.filter(r => ['ASSERTS NOTHING', 'CRASHED PART-WAY',
                                   'EXITS EARLY', 'COLD SITES', 'NO COUNT',
                                   'SUMMARY WRONG', 'VACUOUS PASS']
                                  .indexOf(r.verdict.tag) >= 0);
  const totalSites = results.reduce((a, r) => a + r.sites, 0);
  const totalRan = results.reduce((a, r) => a + (r.ran || 0), 0);
  const totalCold = results.reduce((a, r) => a + r.cold.length, 0);

  console.log(`\n${totalRan} checks printed across ${results.length} suites, `
              + `from ${totalSites} assertion sites; ${totalCold} site(s) never ran`);
  if (bad.length) {
    console.log(`\n${bad.length} SUITE(S) NOT DOING THEIR JOB:`);
    bad.forEach(r => console.log(`  ${r.file} — ${r.verdict.tag}: ${r.verdict.why}`));
    return 1;
  }
  console.log('every suite executes its assertions');
  return 0;
}

function pad(s, n) { s = String(s); return s + ' '.repeat(Math.max(0, n - s.length)); }

if (require.main === module) process.exit(main());
module.exports = { suites, assertionSites, vacuousCandidates, maskNonCode };
