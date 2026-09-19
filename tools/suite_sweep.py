"""suite_sweep.py — which Python suites actually assert, and which only say they did.

    python -m tools.suite_sweep                 every suite, Python AND node
    python -m tools.suite_sweep news weather    only suites whose name matches
    python -m tools.suite_sweep --json          machine-readable
    python -m tools.suite_sweep --cold          also name the lines that never ran
    python -m tools.suite_sweep --no-node       skip the node half
    python -m tools.suite_sweep --live         include the paid/live sections

THE COUNTERPART TO tools/suite_sweep.js, AND WHY IT EXISTS SEPARATELY.

Five instances of one failure have now surfaced, every one of them by accident
while looking at something else:

  23f4185                   the realtime node suite stopped running when node
                            moved to the volume; 463 checks silently stopped.
  source_selftest.js        threw on node 22's navigator getter; 38 assertion
                            sites, zero checks, exit 1, nobody read the code.
  news_selftest.py          three checks green over an empty result list.
  https_selftest.py         section E unreachable whenever the TLS proxy is
                            down, which it was; 15 checks never ran and the
                            suite still exited 0.
  output_bus_selftest.py    the whole loopback soak skipped by default, under
                            "49/49 checks passed".

The node sweep catches this class for the node suites. Three of those five are
Python, and nothing was looking. So this is the same method, in the same order of
trust, for the other half of the repo.

HOW IT DECIDES, WHICH IS NOT BY ASKING THE SUITE.

A suite's own total is printed by the thing under suspicion: one that exits early
prints a smaller number and looks fine, one that crashes prints nothing and looks
like a shell problem. So the measurement is the per-check lines the suite prints
AS IT RUNS, counted against the `ok(`/`check(` call sites in its source. Printed
against ceiling is the measurement; the suite's summary is then compared to the
printed count as a check on the SUITE, not as evidence about it.

The Python suites do not agree on a marker the way the node ones do -- "ok  ",
"FAIL", "[PASS]", "[FAIL]" and an indented "[PASS]" all occur -- so MARKERS below
is the union, and every pattern in it was confirmed against a real run rather
than guessed. The self-check for that is built in: where a suite reports its own
total AND the count here disagrees, one of the two is wrong and the sweep says so
instead of quietly trusting itself. That is how the node sweep's first regex was
caught accusing a suite of over-reporting four checks it had in fact printed.

COLD LINES ARE OPTIONAL HERE, and the reason is cost. The node sweep gets block
coverage free from V8. Python's stdlib `trace` costs a callback per line, which
on suites that already take minutes is not free, so --cold is a flag rather than
the default. Without it a suite that exits early is still caught -- by the count
-- it just cannot be told WHICH assertions went dark.

--offline IS THE DEFAULT, AND IT COST MONEY TO LEARN THAT. The first run of this
file swept every suite with no arguments, which is the DEFAULT path -- and for
news_selftest, weather_selftest and safety_speech_selftest the default path is
the live one. It spent about 25 cents of real API budget on a run whose only
purpose was to count printed lines, and the tell was a count that disagreed with
the suite's own: 93 against 84 for news, because nine of those checks were the
live section.

A sweep meant to be part of the standard run must be free to run. So --offline is
passed to every suite that accepts it (asked via --help, not from a list here
that would go stale), and --live is the opt-in. The node sweep never needed this;
nothing in the browser suites bills anybody.

WHAT IT STILL CANNOT SEE: whether an assertion that ran had anything to assert
about. `all([])` is True. That is the vacuous pass, it is a different failure, and
tools/assert_guard.py is what refuses it.
"""
import argparse
import os
import re
import subprocess
import sys
import time

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)
SELF = os.path.basename(__file__)

# Every per-check line format in use, confirmed against real output rather than
# guessed -- and the guess was wrong twice, which is why this comment names them.
#
#   "  ok    what"      most suites
#   "  FAIL  what"
#   "  [PASS] what"     vehicle_health, vehicle_acceptance, visual
#   "  [ok  ] what"     preflight, and the one the first version missed: the
#   "  [MISS] what"     brackets defeated `ok\s` and the lowercase defeated
#                       `\[(?:PASS|FAIL)\]`, so a suite with 32 real checks was
#                       reported as asserting nothing. A sweep that cries wolf
#                       about a working suite is one nobody reads, so the union
#                       is explicit and each entry names who uses it.
MARKERS = re.compile(
    r"^\s{0,6}(?:\[(?:PASS|FAIL|ok\s*|MISS)\]|ok\s{1,4}|FAIL\s{1,4}|ok$|FAIL$)",
    re.M)

# What a suite says about itself, in the several shapes it gets said in.
SUMMARIES = [
    re.compile(r"(\d+)\s*/\s*(\d+)\s+checks passed"),        # n/total
    re.compile(r"PASSED\s+(\d+)\s+checks"),                  # total only
    re.compile(r"FAILED\s+(\d+)\s*/\s*(\d+)\s+checks"),      # fails/total
    re.compile(r"(\d+)\s+failure\(s\)\s+of\s+(\d+)\s+checks"),
]
# A verdict with no number behind it: it cannot tell 0 checks from all of them.
BARE_VERDICT = re.compile(r"^\s*(?:PASS|FAIL):\s*(\d+)\s+failure", re.M)

SKIPS = re.compile(r"^\s*SKIP\s{1,4}(.+)$", re.M)


def suites(filters):
    out = []
    for f in sorted(os.listdir(TOOLS)):
        if not f.endswith(".py") or f == SELF:
            continue
        src = open(os.path.join(TOOLS, f), encoding="utf-8").read()
        # A suite is a file with its own reporter. Better than the filename:
        # it is the convention every one of them actually follows.
        if not re.search(r"^def (?:ok|check)\s*\(", src, re.M):
            continue
        if filters and not any(x in f for x in filters):
            continue
        out.append(f)
    return out


def mask_non_code(src):
    """Blank every string and comment, preserving offsets and newlines, so a
    match in the result is at the same place in the original. Character by
    character rather than by regex, because a regex that strips strings is a
    regex that eats apostrophes in prose."""
    out = list(src)
    i, n = 0, len(src)

    def blank(a, b):
        for k in range(a, min(b, n)):
            if out[k] != "\n":
                out[k] = " "

    while i < n:
        c = src[i]
        if c == "#":
            j = src.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
            continue
        if c in "\"'":
            triple = src[i:i + 3]
            if triple in ('"""', "'''"):
                j = src.find(triple, i + 3)
                j = n if j < 0 else j + 3
                blank(i, j)
                i = j
                continue
            j = i + 1
            while j < n:
                if src[j] == "\\":
                    j += 2
                    continue
                if src[j] == c or src[j] == "\n":
                    j += 1
                    break
                j += 1
            blank(i, j)
            i = j
            continue
        i += 1
    return "".join(out)


def assertion_sites(src):
    """Offsets and lines of every ok()/check() CALL. The definition is not a
    site, and a name inside a string or comment is not code."""
    masked = mask_non_code(src)
    sites = []
    for m in re.finditer(r"\b(?:ok|ok_all|ok_none|non_empty|check)\s*\(", masked):
        before = masked[max(0, m.start() - 24):m.start()]
        if re.search(r"\bdef\s+$", before):
            continue
        sites.append(masked[:m.start()].count("\n") + 1)
    return sites


def parse_summary(stdout):
    """(total, fails) as the suite reports them, or (None, None)."""
    for i, rx in enumerate(SUMMARIES):
        m = rx.search(stdout)
        if not m:
            continue
        if i == 0:
            return int(m.group(2)), int(m.group(2)) - int(m.group(1))
        if i == 1:
            return int(m.group(1)), 0
        if i == 2:
            return int(m.group(2)), int(m.group(1))
        return int(m.group(2)), int(m.group(1))
    m = BARE_VERDICT.search(stdout)
    if m:
        return None, int(m.group(1))        # a verdict with no total
    return None, None


def cold_lines(suite, sites, extra_args):
    """Which assertion sites never executed, via stdlib trace. Slow, so only on
    --cold. Returns None when it could not be determined -- never an empty list,
    because "no cold lines" and "could not tell" are different answers and a
    sweep that conflated them would be the bug it exists to find."""
    target = os.path.join(TOOLS, suite)
    prog = (
        "import sys, trace, runpy, json, os\n"
        f"t = trace.Trace(count=1, trace=0, ignoredirs=[sys.prefix, sys.exec_prefix])\n"
        f"sys.argv = [{target!r}] + {list(extra_args)!r}\n"
        "try:\n"
        f"    t.runfunc(runpy.run_path, {target!r}, run_name='__main__')\n"
        "except SystemExit:\n"
        "    pass\n"
        "except Exception:\n"
        "    pass\n"
        "hit = sorted(l for (f, l) in t.results().counts if os.path.abspath(f) == "
        f"os.path.abspath({target!r}))\n"
        "sys.stderr.write('__COLD__' + json.dumps(hit))\n"
    )
    try:
        r = subprocess.run([sys.executable, "-c", prog], cwd=REPO,
                           capture_output=True, text=True, timeout=900)
    except Exception:
        return None
    m = re.search(r"__COLD__(\[.*\])", r.stderr or "", re.S)
    if not m:
        return None
    import json as _json
    hit = set(_json.loads(m.group(1)))
    if not hit:
        return None
    return [l for l in sites if l not in hit]


# Suites whose default run spends money or reaches the network. Detected rather
# than listed, by asking each suite what flags it takes.
_OFFLINE_CACHE = {}


def takes_flag(suite, flag):
    """Does this suite accept `flag`? Asked via --help, not from a list here that
    would go stale the first time a suite grew or lost one."""
    key = (suite, flag)
    if key in _OFFLINE_CACHE:
        return _OFFLINE_CACHE[key]
    try:
        r = subprocess.run([sys.executable, os.path.join(TOOLS, suite), "--help"],
                           cwd=REPO, capture_output=True, text=True, timeout=120)
        # A WORD, NOT A SUBSTRING, and getting this wrong broke a suite. Asking
        # `"-v" in help` matched visual_selftest's own `[--video VIDEO]`, so this
        # file passed -v to a suite that does not take it, argparse exited 2, and
        # 34 checks that had been running stopped -- caused by the tool written to
        # find exactly that. Bounded on both sides so -v does not match --video
        # and --live does not match --live-only.
        help_text = (r.stdout or "") + (r.stderr or "")
        _OFFLINE_CACHE[key] = bool(
            re.search(r"(?<![-\w])" + re.escape(flag) + r"(?![-\w])", help_text))
    except Exception:
        _OFFLINE_CACHE[key] = False
    return _OFFLINE_CACHE[key]


def takes_offline(suite):
    """Does this suite accept --offline? Asked, not assumed: a list here would
    go stale the first time a suite grew the flag."""
    if suite in _OFFLINE_CACHE:
        return _OFFLINE_CACHE[suite]
    try:
        r = subprocess.run([sys.executable, os.path.join(TOOLS, suite), "--help"],
                           cwd=REPO, capture_output=True, text=True, timeout=120)
        _OFFLINE_CACHE[suite] = "--offline" in (r.stdout or "") + (r.stderr or "")
    except Exception:
        _OFFLINE_CACHE[suite] = False
    return _OFFLINE_CACHE[suite]


def run(suite, extra_args, want_cold):
    src = open(os.path.join(TOOLS, suite), encoding="utf-8").read()
    sites = assertion_sites(src)
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, os.path.join(TOOLS, suite)]
                           + list(extra_args),
                           cwd=REPO, capture_output=True, text=True, timeout=900)
        out, err, code = r.stdout or "", r.stderr or "", r.returncode
        timed_out = False
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        code, timed_out = None, True
    ms = int((time.time() - t0) * 1000)

    printed = len(MARKERS.findall(out))
    total, fails = parse_summary(out)
    skipped = [s.strip() for s in SKIPS.findall(out)]
    threw = bool(re.search(r"^Traceback \(most recent call last\)", err, re.M))
    usage = bool(re.search(r"^usage:", (out + err), re.M))
    tail = [l for l in err.strip().splitlines() if l.strip()][-1:] if err else []

    cold = cold_lines(suite, sites, extra_args) if want_cold else None

    return {
        "suite": suite, "ms": ms, "code": code, "timed_out": timed_out,
        "sites": len(sites), "printed": printed, "reported": total,
        "fails": fails, "skipped": skipped, "threw": threw, "usage": usage,
        "stderr_tail": tail[0][:120] if tail else "",
        "cold": cold,
    }


def verdict(r):
    if r["sites"] == 0:
        return "NOT A SUITE", "no assertion sites"
    if r["timed_out"]:
        return "TIMED OUT", f"{r['printed']} checks printed before the 900 s cap"
    if r["printed"] == 0:
        # FOUR DIFFERENT THINGS LOOK LIKE THIS, and only one of them is the
        # finding. Conflating them is how a sweep becomes noise: the first run of
        # this file reported a server, a tool needing a filename, and a suite with
        # an unmatched marker format all as "ASSERTS NOTHING", alongside the one
        # real case -- which is precisely the signal it exists to carry.
        if r["threw"]:
            return ("ASSERTS NOTHING",
                    f"printed no checks at all against {r['sites']} assertion "
                    f"sites — threw: {r['stderr_tail']}")
        if r["code"] == 2 or r["usage"]:
            return ("NEEDS ARGS",
                    "not runnable bare — it wants arguments, so nothing ran "
                    "(not a suite failure)")
        if r["timed_out"]:
            return ("NOT A SUITE",
                    "ran until the cap without asserting or summarising — a "
                    "server or a daemon, not a suite")
        if r["reported"] is None:
            return ("NOT A SUITE",
                    "no checks, no summary, clean exit — nothing here reports")
        # It has a total and printed no lines: quiet on success by design.
        return ("SILENT ON SUCCESS",
                f"prints only failures; its own summary says {r['reported']} "
                f"checks, which is REPORTED rather than observed")
        

    frac = r["printed"] / r["sites"]

    # Exit non-zero WITH failures of its own is a working suite. The node sweep's
    # first version called that a crash and would have filed a suite doing its
    # job as one of the broken ones.
    # A TRACEBACK IS WHAT MAKES IT A CRASH, NOT A RATIO. This used to require 75%
    # of sites to have run, which filed teacher_timing_selftest -- 7/9 checks, two
    # real failures, a clean exit 1 and no traceback -- as CRASHED PART-WAY. Its
    # later sections do not run BECAUSE the early ones failed, which is a suite
    # working exactly as intended. The shortfall is still reported, as a note
    # rather than as an accusation.
    if r["code"] not in (0, None) and (r["fails"] or 0) > 0 and not r["threw"]:
        note = ""
        if frac < 1:
            note = (f"; {r['sites'] - r['printed']} site(s) did not run, which is "
                    "usually the sections after the failure")
        return ("FAILS HONESTLY",
                f"{r['fails']} of {r['printed']} checks failed, exit {r['code']} "
                f"— the suite works; the code under it does not{note}")
    if r["code"] not in (0, None):
        why = (f"{r['printed']} checks printed against {r['sites']} sites, "
               f"exit {r['code']}")
        if r["stderr_tail"]:
            why += f" — {r['stderr_tail']}"
        return "CRASHED PART-WAY", why
    # A DECLARED SKIP EXPLAINS A SHORTFALL; AN UNDECLARED ONE IS THE FINDING.
    # This test used to come first, so voice_selftest -- which had just been
    # taught to declare all three of its opt-in sections -- was still reported as
    # EXITS EARLY at 74%, because 97/131 is under the threshold either way. A
    # suite that says what it did not run has done its job.
    if r["skipped"]:
        return ("DECLARED SKIP",
                f"{r['printed']} of {r['sites']} sites ran; "
                + "; ".join(r["skipped"]))
    if frac < 0.75:
        return ("EXITS EARLY",
                f"only {r['printed']} checks printed against {r['sites']} "
                f"assertion sites ({frac * 100:.0f}%)")
    if r["cold"]:
        tag = "DECLARED SKIP" if r["skipped"] else "COLD SITES"
        why = (f"{r['printed']} checks printed, but {len(r['cold'])} assertion "
               f"site(s) never ran — lines "
               + ", ".join(str(x) for x in r["cold"][:8])
               + (", …" if len(r["cold"]) > 8 else ""))
        if r["skipped"]:
            why += " — declared: " + "; ".join(r["skipped"])
        return tag, why
    if r["reported"] is None:
        return ("NO COUNT",
                f"{r['printed']} checks ran, but the suite reports no total — "
                "its verdict cannot tell 0 checks from all of them")
    if r["reported"] != r["printed"]:
        return ("SUMMARY WRONG",
                f"printed {r['printed']} checks but summarised {r['reported']}")
    return "RUNS", f"{r['printed']} checks ran"


# What counts as "not doing its job". NEEDS ARGS and NOT A SUITE are facts about
# the FILE rather than faults in it; SILENT ON SUCCESS is a measurement this tool
# cannot make, labelled as such, and not an accusation.
BAD = {"ASSERTS NOTHING", "CRASHED PART-WAY", "EXITS EARLY", "COLD SITES",
       "NO COUNT", "SUMMARY WRONG", "TIMED OUT"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("filters", nargs="*")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--cold", action="store_true",
                    help="also name the assertion sites that never ran "
                         "(stdlib trace; slow)")
    ap.add_argument("--args", default="",
                    help="flags to pass through to every suite")
    ap.add_argument("--live", action="store_true",
                    help="do NOT pass --offline — runs the paid/live sections")
    ap.add_argument("--no-node", action="store_true",
                    help="skip the node half (tools/suite_sweep.js)")
    a = ap.parse_args()
    extra = a.args.split() if a.args else []

    names = suites(a.filters)
    if not a.json:
        print(f"sweeping {len(names)} Python suites"
              + ("" if a.live else " (--offline where accepted)")
              + (f" with {' '.join(extra)}" if extra else "")
              + (" (+cold lines)" if a.cold else "")
              + f" — python {sys.version.split()[0]}\n", flush=True)

    results = []
    for s in names:
        args_for = list(extra)
        if not a.live and "--offline" not in args_for and takes_offline(s):
            args_for.append("--offline")
        # A SUITE THAT IS QUIET ON SUCCESS CANNOT BE COUNTED BY ITS OUTPUT.
        # vehicle_acceptance prints a line only when a check FAILS, or when -v is
        # given -- so a clean run printed nothing and looked identical to a suite
        # that asserted nothing. Asking for verbosity turns a measurement this
        # file cannot make into one it can: 245 real lines instead of zero.
        if "-v" not in args_for and takes_flag(s, "-v"):
            args_for.append("-v")
        r = run(s, args_for, a.cold)
        r["offline"] = "--offline" in args_for
        r["verdict"], r["why"] = verdict(r)
        results.append(r)
        if a.json:
            continue
        # flush: this runs for minutes and Python block-buffers a redirected
        # stdout, so without it the whole sweep appears at the end and a hung
        # suite is indistinguishable from a slow one.
        print(f"{s[:-3]:<34} {r['verdict']:<18} {r['why']}", flush=True)

    if a.json:
        import json
        print(json.dumps(results, indent=1))
        return 0

    # BOTH HALVES, ONE COMMAND. The repo's suites are half node and half Python and
    # the disease is in both -- three of the five known instances are Python, two
    # are node. A sweep that covered one half would leave exactly the gap that let
    # these sit for months, so this runs the node sweep too unless told not to.
    node_code = 0
    if not a.no_node and not a.filters:
        js = os.path.join(TOOLS, "suite_sweep.js")
        node = None
        for cand in ("node", "/workspace/node/bin/node"):
            try:
                subprocess.run([cand, "--version"], capture_output=True, timeout=30)
                node = cand
                break
            except Exception:
                continue
        if node is None:
            print("\n!! node not found — the node half of the sweep did NOT run, "
                  "which is itself the kind of silence this tool exists to find",
                  flush=True)
            node_code = 1
        else:
            print("\n" + "-" * 72 + "\nthe node half:\n", flush=True)
            r = subprocess.run([node, js], cwd=REPO)
            node_code = r.returncode or 0

    bad = [r for r in results if r["verdict"] in BAD]
    tsites = sum(r["sites"] for r in results)
    tprint = sum(r["printed"] for r in results)
    print(f"\n{tprint} checks printed across {len(results)} suites, "
          f"from {tsites} assertion sites")
    if bad:
        print(f"\n{len(bad)} PYTHON SUITE(S) NOT DOING THEIR JOB:")
        for r in bad:
            print(f"  {r['suite']} — {r['verdict']}: {r['why']}")
    elif not names:
        print("no suites matched")
    else:
        print("every Python suite executes its assertions")
    return 1 if (bad or node_code) else 0


if __name__ == "__main__":
    sys.exit(main())
