/* assert_guard.js — assertions that cannot pass on nothing, and skips that
 * cannot hide.
 *
 * THE FAILURE THIS REFUSES.
 *
 *     ok(results.every(hasDate), 'every spoken result is dated')
 *
 * That line is correct, it is important, and when `results` is empty it passes
 * and means nothing. `[].every(anything)` is true. It happened for real, in
 * tools/news_selftest.py: a live news question spent 23 cents, `audit()` kept
 * nothing, and three checks -- dated, sourced, geographically corroborated --
 * all reported ok over an empty list. The run that went wrong is precisely the
 * run those checks reported clean.
 *
 * A suite that reports success without asserting is worse than one that fails,
 * because the failing one gets fixed. tools/suite_sweep.js catches the version
 * of that where the assertion never runs. This catches the version where it runs
 * and has nothing to run on.
 *
 * THE DISTINCTION THAT MATTERS, because half of these are fine.
 *
 *   POSITIVE over a collection   "every result is dated". Empty is VACUOUS: the
 *                               claim is about the members, and there are none.
 *                               -> okAll, which fails on empty.
 *   NEGATIVE over a collection   "no synthesiser request was made". Empty is the
 *                               WHOLE POINT and passing on it is correct.
 *                               -> okNone, which is happy with empty and says so.
 *
 * Both exist here so that the intent is written down at the call site. The
 * status quo infers it from the shape of the predicate, which is how a genuine
 * negative assertion and a vacuous positive one come to look identical in a
 * diff. Measured across the sixteen node suites (a patched every/some/filter
 * recording every call on an empty array): every empty-collection assertion
 * currently in them is the negative kind and correct, and not one `every()`
 * ever ran on an empty array. So this module is not fixing thirteen bugs -- it
 * is making the thirteen places that are one data change away from being bugs
 * say which kind they are.
 */
'use strict';

/* Wire in the suite's own reporter, so a guarded assertion is counted and
   printed exactly like every other check rather than becoming a second channel
   with its own totals. `ok` is the suite's function; nothing here prints. */
function create(ok) {
  if (typeof ok !== 'function') {
    throw new Error('assert_guard: needs the suite\'s ok(cond, what)');
  }

  /* A POSITIVE CLAIM ABOUT EVERY MEMBER. Empty fails, and the message says why
     it failed rather than restating what was being asserted -- "0 of 0" reads
     like a pass, and the whole point is that it is not one. */
  function okAll(arr, pred, what) {
    const list = Array.isArray(arr) ? arr : [];
    if (list.length === 0) {
      return ok(false, what + ' — NOTHING TO ASSERT ON: the collection was '
                     + 'empty, so this check proved nothing. Empty is not a '
                     + 'pass for a claim about every member.');
    }
    const bad = list.filter((x, i) => !pred(x, i));
    return ok(bad.length === 0,
              what + ` (${list.length - bad.length}/${list.length})`
              + (bad.length ? ' — first failing: '
                 + JSON.stringify(bad[0]).slice(0, 160) : ''));
  }

  /* A NEGATIVE CLAIM: no member may match. Empty PASSES, deliberately, and the
     call site saying `okNone` is what records that emptiness was expected
     rather than unnoticed. */
  function okNone(arr, pred, what) {
    const list = Array.isArray(arr) ? arr : [];
    const hits = list.filter(pred);
    return ok(hits.length === 0,
              what + (list.length === 0 ? ' (nothing to match, as expected)'
                                        : ` (${list.length} candidate(s), 0 matched)`)
              + (hits.length ? ' — found: '
                 + JSON.stringify(hits[0]).slice(0, 160) : ''));
  }

  /* The precondition on its own, for a section whose later checks are not
     collection-shaped but still depend on something having been produced. */
  function nonEmpty(arr, what) {
    const n = Array.isArray(arr) ? arr.length : (arr && arr.length) || 0;
    return ok(n > 0, what + (n ? ` (${n})` : ' — EMPTY, so everything asserted '
                                           + 'about it below is vacuous'));
  }

  /* Exactly n, where n may be 0 — but saying 0 has to be deliberate, because
     `.length === 0` is the shape that hides a collection that failed to fill. */
  function okCount(arr, n, what) {
    const list = Array.isArray(arr) ? arr : [];
    return ok(list.length === n, what + ` (${list.length}, wanted ${n})`);
  }

  return { okAll, okNone, nonEmpty, okCount };
}

/* ---------------------------------------------------------------------------
   SKIPS THAT CANNOT HIDE.

   nav_selftest.js has an optional section gated on a route fixture passed as
   argv[2]. Without one it silently does not run, and the suite prints
   "PASSED 111 checks" with four assertions dark and no mention of it. That is
   the same class of quiet as a crashed suite -- a clean-looking verdict over
   work that did not happen -- and the sweep found it as four cold sites.

   So a skip is DECLARED. It prints, it is counted, and the summary carries it,
   which means "passed" can never again mean "passed, and also a section you
   were not told about did not run".
   --------------------------------------------------------------------------- */
function skips() {
  const declared = [];
  return {
    /* reason: what did not run. how: what would make it run. */
    skip(reason, how) {
      declared.push({ reason, how: how || null });
      console.log(`  SKIP  ${reason}` + (how ? ` — ${how}` : ''));
      return false;
    },
    declared: () => declared.slice(),
    /* The fragment a summary line appends. Empty string when nothing was
       skipped, so the ordinary case reads exactly as it always has. */
    summary() {
      if (!declared.length) return '';
      return `, ${declared.length} section(s) SKIPPED (`
           + declared.map(d => d.reason).join('; ') + ')';
    },
  };
}

module.exports = { create, skips };
