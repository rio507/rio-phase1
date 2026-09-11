"""The teachers yield to the live pipeline, and can be proved to.

    python -m tools.teacher_yield_selftest

WHAT THIS GUARDS. The teacher panel is shadow: it may cost a corpus row, it
may not cost a frame. Measured before this existed (tools/pipeline_probe.py,
same clip, same server):

    teachers unloaded      detector  4.2 ms   server total 15.7 ms
    loaded but IDLE        detector  4.5 ms   server total 16.7 ms
    loaded and INFERRING   detector 26.2 ms   server total 51.5 ms

Loaded is free. Inferring is 5.8x on the detector. So the rules are about WHEN
they infer, and each one is asserted here rather than described:

  cadence     while frames are flowing the keyframe floor is the long one
  one at a time   both clients share a semaphore of exactly one
  session     no new work while a live conversation is open
  answering   no new work while RIO is answering (the original hold)
  self-healing    a session whose close never arrives does not pause them
                  for the life of the process
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config                                               # noqa: E402
import teachers.client as tc                                # noqa: E402
import teachers.panel as panel                              # noqa: E402
from teachers.client import TeacherClient                   # noqa: E402

_fails = []


def ok(name, cond, extra=""):
    if cond:
        print(f"  ok   {name}")
    else:
        _fails.append(name)
        print(f"  FAIL {name}{('  ' + str(extra)) if extra else ''}")


def main() -> int:
    t = TeacherClient("selftest", "http://127.0.0.1:9")

    print("\n== cadence follows the frames")
    panel._LAST_FRAME_AT["t"] = 0.0
    idle_floor = panel._floor_s()
    panel._LAST_FRAME_AT["t"] = time.time()
    live_floor = panel._floor_s()
    ok("idle uses the short floor", idle_floor == config.TEACHER_KEYFRAME_FLOOR_S,
       idle_floor)
    ok("frames flowing uses the long floor",
       live_floor == config.TEACHER_KEYFRAME_FLOOR_S_LIVE, live_floor)
    ok("the long floor really is longer", live_floor > idle_floor,
       f"{live_floor} vs {idle_floor}")
    panel._LAST_FRAME_AT["t"] = 0.0

    print("\n== one teacher at a time")
    ok("the gate admits exactly one", tc._GATE._value == 1, tc._GATE._value)
    got = tc._GATE.acquire(blocking=False)
    ok("...and the second caller is refused",
       got and not tc._GATE.acquire(blocking=False))
    if got:
        tc._GATE.release()
    ok("the gate is shared by both clients, not per client",
       "_GATE" in dir(tc) and "_GATE" not in dir(t))

    print("\n== no new work while a driver is waiting")
    ok("idle: not paused", not t.paused())
    tc.session_note(True)
    ok("a live session pauses them", t.paused())
    tc.session_note(False)
    ok("closing it resumes them", not t.paused())
    with tc.hold_gpu():
        ok("answering pauses them", t.paused())
    ok("finishing the answer resumes them", not t.paused())

    print("\n== a lost close does not pause them forever")
    tc.session_note(True, ttl_s=0.4)
    ok("open is live", tc.session_live())
    time.sleep(0.6)
    ok("the pause expires without a close", not tc.session_live())
    ok("...and the client is running again", not t.paused())
    tc.session_note(False)

    print("\n== and none of it can be switched off by accident")
    ok("the concurrency cap is at least 1",
       int(config.TEACHER_MAX_CONCURRENT) >= 1)
    ok("the pause is on by default", config.TEACHER_PAUSE_DURING_SESSION)
    ok("the TTL is long next to a conversational gap",
       float(config.TEACHER_SESSION_TTL_S) >= 30.0)

    print("\n" + "-" * 52)
    print(f'  {"PASS" if not _fails else "FAIL"}: {len(_fails)} failure(s)')
    for f in _fails:
        print(f"    - {f}")
    return len(_fails)


if __name__ == "__main__":
    raise SystemExit(main())
