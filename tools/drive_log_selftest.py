"""drive_log_selftest.py — the drive log can answer the questions asked of it.

    python tools/drive_log_selftest.py

A drive log is only worth writing if a question about the drive can be put to
it afterwards. Three could not be, and each cost an afternoon of archaeology on
2026-09-21:

  "HOW OFTEN WAS THE READING TRUNCATED?"  The perceive row carried the caption
  and nothing about it -- not the truncation, not the measurements stripped out
  of it, not whether the detector contradicted it, not which picture it was of.
  All four were on the Perception card at the time. Answering meant replaying
  150 captions through the guards by hand.

  "HOW MANY READINGS DID THE GUARDS REFUSE?"  vision.model_info() has carried
  every counter since the model swap and NOTHING SERVED IT. The only available
  answer was to grep stdout for lines that print at counts 1, 10 and 100, which
  bounds the number and does not give it.

  "WHAT DID THE FRAME TRANSPORT DO?"  It ended up in stray-<id>.jsonl. The page
  posts that summary and /session/end as two fire-and-forget fetches and the
  server handled the end first, closing the file a millisecond early.
"""
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import sessions                                            # noqa: E402

checks = 0
failures = 0


def ok(name, cond, extra=""):
    global checks, failures
    checks += 1
    if not cond:
        failures += 1
    print(("  ok   " if cond else "  FAIL ") + name + (f" — {extra}" if extra else ""))


print("\n== 1. a perceive row carries what is wrong with the reading ==\n")
READING = {
    "raw": "ROAD: four | TRAFFIC: none | RISK: van st",
    "model": "Cosmos-Reason2-2B", "age_s": 0.4, "frame_id": "f000123",
    "source": "clip", "truncated": True, "stripped": ["20 km/h"],
    "contested": ["TRAFFIC"], "detector": {"account": "one pedestrian"},
    "unknown_fields": ["LIGHT"], "parsed": True,
    "fields": [{"name": "ROAD", "text": "four", "present": True}],
}
facts = sessions._reading_facts(READING)
for k in ("model", "age_s", "frame_id", "source", "truncated", "stripped",
          "contested", "detector", "unknown_fields", "parsed"):
    ok(f"...{k} is on the row", k in facts, repr(facts.get(k)))
ok("...and the fields are NOT repeated: the caption is the whole reading and "
   "the split is deterministic from it", "fields" not in facts)
ok("a row with no reading block says so rather than inventing one",
   sessions._reading_facts(None) is None)

print("\n== 2. the guards' counts are served, not only printed ==\n")
import realtime                                            # noqa: E402
eye = realtime._eye_status()
ok("the eye's status is reachable without a model loaded", isinstance(eye, dict))
flags = (eye.get("flags") or {})
for k in ("prompt_example", "prompt_echo", "canned", "repeated", "truncated",
          "invented_units", "advisory", "blank_frame", "think_trace"):
    ok(f"...{k} is served with a rate", k in flags and k in (flags.get("rate") or {}))
ok("...beside the model that produced them", bool(eye.get("label")))
ok("...and the cap they were measured against",
   isinstance(eye.get("max_new_tokens"), int))
src = (REPO / "realtime.py").read_text()
ok("and it is on the status the page already fetches",
   '"eye": _eye_status()' in src)

print("\n== 3. a row that arrives late belongs to the drive it is about ==\n")
sid = sessions.start_session({"test": "drive_log_selftest"})
path = sessions.SESSIONS_DIR / f"{sid}.jsonl"
stray = sessions.SESSIONS_DIR / f"stray-{sid}.jsonl"
try:
    sessions.mark(sid, "before_end", {"n": 1})
    sessions.end_session(sid)
    # The transport summary and a slow tool result, both arriving after the
    # file was closed -- which is exactly what happened.
    sessions.mark(sid, "frame_transport", {"note": "late"})
    sessions.flush()
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    tags = [r["payload"].get("tag") for r in rows if r["kind"] == "mark"]
    ok("a mark posted after session_end lands in the DRIVE's log",
       "frame_transport" in tags, str(tags))
    ok("...and not in a stray file", not stray.exists(),
       "stray file was written")
    ok("...after the session_end row, so the order still reads true",
       [r["kind"] for r in rows].index("session_end")
       < max(i for i, r in enumerate(rows)
             if r["kind"] == "mark" and r["payload"].get("tag") == "frame_transport"))

    # ...and the window closes. A row for a drive nobody was on is still a
    # stray, which is what the mechanism was written for.
    sessions._recently_ended[sid] = time.time() - (sessions.LATE_WRITE_S + 1)
    sessions.mark(sid, "much_later", {"n": 2})
    sessions.flush()
    ok(f"a row arriving more than {sessions.LATE_WRITE_S:.0f} s late is a stray "
       "again", stray.exists())
    ok("...and the drive's own log did not grow",
       "much_later" not in path.read_text())
finally:
    for f in (path, stray):
        try: f.unlink()
        except Exception: pass

print("\n== 4. and the page does not race its own session end ==\n")
page = (REPO / "static" / "index.html").read_text()
ok("the transport mark is awaited before the session is closed",
   "await fetch('/session/mark?session_id=' + encodeURIComponent(sid)" in page)

print(f"\n{checks - failures}/{checks} checks passed")
sys.exit(1 if failures else 0)
