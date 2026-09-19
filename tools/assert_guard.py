"""assert_guard.py — assertions that cannot pass on nothing.

The Python half of tools/assert_guard.js, and the half where the failure was
real rather than hypothetical.

    ok("every spoken result is dated", all(x.get("published_ts") for x in results))

`all([])` is True. On the live news run of 2026-09-19 that line, and two beside
it, printed green over an empty list: the question spent 23 cents, `audit()` kept
nothing, and the three checks that exist to prove RIO may speak a result all
reported clean about no results at all. The run that went wrong is exactly the
run they called clean.

THE DISTINCTION, because half of these are correct as they stand.

    POSITIVE over a collection   "every result is dated" -- the claim is about
                                 the members, so empty proves nothing.
                                 -> ok_all, which FAILS on empty.
    NEGATIVE over a collection   "no issue is announceable" -- empty is the
                                 point, and passing on it is right.
                                 -> ok_none, which is happy with empty and says so.

Both exist so the intent is written at the call site instead of inferred from the
shape of a predicate. That inference is how a genuine negative assertion and a
vacuous positive one come to look identical in a diff.

WHAT EMPTY MEANS WHEN ok_all REFUSES IT. Not "something is broken" -- "nothing
was checked". Those are different findings and the message says which, because an
empty result set is frequently a legitimate ANSWER for RIO (localnews' own
instruction says an empty list is correct and better than a padded one) and is
never a legitimate TEST.

WIRING. Each suite has its own reporter with its own argument order -- `ok(name,
cond)` in some, `ok(cond, name)` in others, `check(cond, name, extra)` in the
vehicle ones. `bind` takes the suite's function and how it is shaped, so nothing
here prints and every guarded check is counted and formatted exactly like its
neighbours.
"""


class Skips:
    """Skips that cannot hide.

    An optional section that announces itself is information; one that goes quiet
    is the bug. Both leave the same assertions dark, so the only way to tell them
    apart is whether the suite said so.

    Found twice by sweeping: nav_selftest's real-geometry section, gated on a
    route fixture in argv[2], and output_bus_selftest's loopback soak, gated on
    --soak-s which defaults to 0. The second printed "49/49 checks passed" with
    three assertions about the loopback never executed and no mention of it.

    The point is that "passed" must never be able to mean "passed, and also a
    section you were not told about did not run".
    """

    def __init__(self):
        self.declared = []

    def skip(self, reason, how=""):
        self.declared.append((reason, how))
        print(f"  SKIP {reason}" + (f" — {how}" if how else ""))
        return False

    def summary(self):
        """The fragment a summary line appends. Empty when nothing was skipped,
        so the ordinary case reads exactly as it always has."""
        if not self.declared:
            return ""
        return (f", {len(self.declared)} section(s) SKIPPED ("
                + "; ".join(r for r, _ in self.declared) + ")")


def bind(report, order="cond_first"):
    """Wrap a suite's reporter.

    report: the suite's own ok()/check().
    order:  "cond_first" for ok(cond, name, extra), "name_first" for
            ok(name, cond, extra).
    """
    if order not in ("cond_first", "name_first"):
        raise ValueError("order must be cond_first or name_first")

    def say(cond, name, extra=""):
        if order == "cond_first":
            return report(cond, name, extra) if extra != "" else report(cond, name)
        return report(name, cond, extra) if extra != "" else report(name, cond)

    def ok_all(name, items, pred, extra=""):
        """A claim about EVERY member. Empty fails, saying nothing was checked."""
        items = list(items or [])
        if not items:
            return say(False, name + "  — NOTHING TO ASSERT ON: the collection "
                              "was empty, so this check proved nothing. Empty is "
                              "not a pass for a claim about every member.")
        bad = [x for x in items if not pred(x)]
        if bad:
            return say(False, f"{name} ({len(items) - len(bad)}/{len(items)})",
                       extra if extra != "" else repr(bad[0])[:160])
        return say(True, f"{name} ({len(items)}/{len(items)})")

    def ok_none(name, items, pred, extra=""):
        """A claim that NO member matches. Empty passes, and says it was empty."""
        items = list(items or [])
        hits = [x for x in items if pred(x)]
        if hits:
            return say(False, name, extra if extra != "" else repr(hits[0])[:160])
        tail = ("nothing to match, as expected" if not items
                else f"{len(items)} candidate(s), 0 matched")
        return say(True, f"{name} ({tail})")

    def non_empty(name, items, extra=""):
        """The precondition alone, for a section whose later checks are not
        collection-shaped but still depend on something having been produced."""
        n = len(list(items or []))
        if n:
            return say(True, f"{name} ({n})")
        return say(False, name + "  — EMPTY, so everything asserted about it "
                          "below is vacuous", extra)

    return ok_all, ok_none, non_empty
