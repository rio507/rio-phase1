"""look() never waits on a teacher, and never waits long on anything local.

    python -m tools.look_yield_selftest

THE RULE THIS GUARDS, from the brief that produced it: the teachers' opinion is
worth exactly as much as it costs to have ALREADY. look() takes the latest
cached reading if one is fresh; if there is none it answers without one. It
never triggers an inference, never waits for one, and never delays the answer.

THE CHECK THAT MATTERS is the last one: a look() with NO cached reading must
cost the same as a look() with the teachers switched off entirely. If those two
differ, something on the answer path is waiting for a teacher.

...and the same discipline applied to the local passes, because the
measurement that started this found the teachers innocent and the local Qwen
calls guilty: an attribute read at 6.4 s, a reference tie-break at 5-6 s and a
clarification at 11.7 s, all inside a turn whose budget is three seconds.
Every one of them is now bounded, and the bounds are asserted here.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                               # noqa: E402

_fails = []


def ok(name, cond, extra=""):
    if cond:
        print(f"  ok   {name}")
    else:
        _fails.append(name)
        print(f"  FAIL {name}{('  ' + str(extra)) if extra else ''}")


def main() -> int:
    from teachers import panel

    print("\n== the teacher fetch is bounded and never infers")
    t = time.perf_counter()
    got = panel.context_now("no-such-session")
    miss_ms = (time.perf_counter() - t) * 1000
    ok("a session with no reading returns {} at once",
       got == {} and miss_ms < 20.0, f"{miss_ms:.2f} ms")

    # ...and it cannot be made to wait, even by a lock somebody else is holding.
    import threading
    held = threading.Event()
    release = threading.Event()

    def hog():
        with panel._lock:
            held.set()
            release.wait(timeout=5)

    threading.Thread(target=hog, daemon=True).start()
    held.wait(timeout=2)
    t = time.perf_counter()
    got = panel.context_now("no-such-session")
    blocked_ms = (time.perf_counter() - t) * 1000
    release.set()
    cap = float(config.TEACHER_CONTEXT_TIMEOUT_MS)
    ok("a contended lock costs the bound and no more",
       got == {} and blocked_ms <= cap * 1.6,
       f"{blocked_ms:.0f} ms against a {cap:.0f} ms bound")
    ok("...and it gave up rather than answering late",
       panel.context_tally()["timed_out"] >= 1, panel.context_tally())

    print("\n== a look with no reading costs what teachers-off costs")
    # THE COMPARISON THE BRIEF ASKS FOR, made on the fetch itself because that
    # is the only place a teacher can enter the answer path -- realtime.look
    # receives the block as an argument and does no I/O for it at all.
    t = time.perf_counter()
    for _ in range(200):
        panel.context_now("no-such-session")
    with_teachers = (time.perf_counter() - t) * 1000 / 200
    was = config.TEACHERS_ENABLED
    config.TEACHERS_ENABLED = False
    try:
        t = time.perf_counter()
        for _ in range(200):
            panel.context_now("no-such-session")
        without = (time.perf_counter() - t) * 1000 / 200
    finally:
        config.TEACHERS_ENABLED = was
    ok("no cached reading == teachers disabled, within 1 ms",
       abs(with_teachers - without) < 1.0,
       f"{with_teachers:.3f} ms vs {without:.3f} ms")

    print("\n== look() imports nothing that could start an inference")
    import realtime
    src = Path(realtime.__file__).read_text()
    ok("realtime.py does not import teachers",
       "import teachers" not in src and "from teachers" not in src)
    ok("...and takes the block as an argument instead",
       "def look(" in src and "teachers: dict = None" in src)

    print("\n== every local pass on the answer path has a deadline")
    for name, attr, ceiling in (
            ("on-demand observation", "OBSERVER_ON_DEMAND_DEADLINE_MS", 1500),
            ("attribute read", "ENRICH_ANSWER_DEADLINE_MS", 1500),
            ("reference tie-break", "RESOLVE_VLM_DEADLINE_MS", 1500),
            ("clarification reads", "CLARIFY_ENRICH_DEADLINE_MS", 1500)):
        v = float(getattr(config, attr, 0) or 0)
        ok(f"{name} is bounded ({attr}={v:.0f} ms)", 0 < v <= ceiling)

    import observer
    ok("observer.observe_soon exists and is what look() calls",
       hasattr(observer, "observe_soon")
       and "observe_soon" in Path(realtime.__file__).read_text())
    ok("observer.recent serves a labelled older description",
       hasattr(observer, "recent")
       and float(config.OBSERVER_ANSWER_MAX_AGE_S) > config.OBSERVER_FRESH_S)

    print("\n" + "-" * 56)
    print(f'  {"PASS" if not _fails else "FAIL"}: {len(_fails)} failure(s)')
    for f in _fails:
        print(f"    - {f}")
    return len(_fails)


if __name__ == "__main__":
    raise SystemExit(main())
