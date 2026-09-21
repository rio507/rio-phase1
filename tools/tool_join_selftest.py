"""tool_join_selftest.py — the same research question is not researched twice.

    python tools/tool_join_selftest.py        (needs the server running)

WHAT IT DEFENDS
---------------
On the drive of 2026-09-21 the live session called deep_dive with

    "What movies are currently playing at Regal Sherman Oaks Galleria?"

at t=81.1 s, and called it again with the byte-identical string at t=92.1 s --
while the first was still running and 2.3 s from returning. The second took
16.70 s. Sixteen seconds of a 167-second drive, and a second billed reasoning
call, spent re-answering a question that was about to be answered.

THE RULE IS NARROWER THAN "DO NOT REPEAT WORK", and the narrowness is the
point. A call may be joined only if its answer depends on its arguments and not
on the moment it was asked. `look` must never be joined: the road moves, and
two identical "what do you see" a second apart are two questions about two
different pictures. Neither may vehicle_status, nav_status or find_places,
whose answers turn on where the car is right now.

Exit 0 on success, 1 on failure, 2 if the server is not reachable -- which is
not a failure of the code under test and should not read as one.
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

BASE = os.getenv("RIO_BASE", "http://127.0.0.1:8888")
checks = 0
failures = 0


def ok(name, cond, extra=""):
    global checks, failures
    checks += 1
    if not cond:
        failures += 1
    print(("  ok   " if cond else "  FAIL ") + name + (f" — {extra}" if extra else ""))


def call(tool, args, key, out):
    t0 = time.time()
    r = subprocess.run(
        ["curl", "-s", "--max-time", "120", "-X", "POST",
         f"{BASE}/realtime/tool?client_id={key}",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"name": tool, "arguments": args})],
        capture_output=True, text=True)
    try:
        out["body"] = json.loads(r.stdout)
    except Exception:
        out["body"] = {"_raw": r.stdout[:200]}
    out["wall_s"] = time.time() - t0


print("\n== 1. the rule is a named set, not 'any tool' ==\n")
src = (REPO / "app.py").read_text()
ok("there is an explicit joinable set", "JOINABLE_TOOLS = frozenset(" in src)
m = __import__("re").search(r"JOINABLE_TOOLS = frozenset\((\{[^}]*\})\)", src)
joinable = eval(m.group(1)) if m else set()
ok("deep_dive is in it", "deep_dive" in joinable, str(sorted(joinable)))
ok("look is NOT -- the road moves between two identical questions",
   "look" not in joinable)
for t in ("vehicle_status", "nav_status", "find_places", "nav_directions"):
    ok(f"...nor {t}, whose answer turns on where the car is now",
       t not in joinable)

print("\n== 2. and on a live server, an identical call in flight is joined ==\n")
probe = subprocess.run(["curl", "-s", "--max-time", "8",
                        f"{BASE}/realtime/status"], capture_output=True, text=True)
if "observer" not in probe.stdout:
    print(f"  server not reachable at {BASE}")
    sys.exit(2)

Q = {"question": "What films are showing tonight at the Vista Theater in Los Angeles?"}
a, b = {}, {}
t_first = threading.Thread(target=call, args=("deep_dive", Q, "tooljoin", a))
t_first.start()
time.sleep(4)                       # well inside the first call's runtime
t0 = time.time()
call("deep_dive", Q, "tooljoin", b)
t_first.join()

print(f"  first : {a['wall_s']:.2f} s")
print(f"  second: {b['wall_s']:.2f} s (started 4 s later)")
ok("both calls succeed", a["body"].get("ok") and b["body"].get("ok"))
ok("...with the same answer", a["body"].get("answer") == b["body"].get("answer"))
ok("...and the same timing, because it is the same piece of work",
   a["body"].get("took_ms") == b["body"].get("took_ms"),
   f"{a['body'].get('took_ms')} vs {b['body'].get('took_ms')}")
# The decisive one: the second finished WITH the first, not 4 s after it
# started plus its own full runtime.
ok("the second call finishes with the first rather than running again",
   b["wall_s"] < a["wall_s"] - 3,
   f"second {b['wall_s']:.1f} s < first {a['wall_s']:.1f} s - 3")
ok("...and it is marked as joined in the result the drive log records",
   b["body"].get("joined") is True or a["body"].get("joined") is True,
   f"first joined={a['body'].get('joined')} second joined={b['body'].get('joined')}")

print("\n== 3. a DIFFERENT question is not joined to it ==\n")
c, d = {}, {}
t_c = threading.Thread(target=call, args=(
    "deep_dive", {"question": "How tall is the Hollywood sign?"}, "tooljoin", c))
t_c.start()
time.sleep(2)
call("deep_dive", {"question": "How old is the Griffith Observatory?"}, "tooljoin", d)
t_c.join()
ok("two different questions get two different answers",
   c["body"].get("answer") != d["body"].get("answer"))
ok("...and neither is marked joined",
   not c["body"].get("joined") and not d["body"].get("joined"))

print(f"\n{checks - failures}/{checks} checks passed")
sys.exit(1 if failures else 0)
