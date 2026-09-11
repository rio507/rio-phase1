"""Which model phrases a safety line, and how long the driver waits for it.

    python -m tools.safety_phrase_bench
    python -m tools.safety_phrase_bench --n 5
    python -m tools.safety_phrase_bench --models gpt-5.6-luna,gpt-5.6-terra

TWO THINGS AT ONCE, because for this job they are the same question:

  latency   the WHOLE line, not the first token. Nothing can be spoken until
            the sentence is finished, so time-to-first-token is not a number
            anyone waits on here -- unlike a conversational answer, which
            starts coming out of the speaker while the rest is still being
            written.
  honesty   how often the line passes safety_speech.check(). A model that is
            fast and invents a pressure is not a candidate; the fallback
            catches it, but a fallback that fires half the time is the old
            behaviour with a delay in front of it.

...and a third, which is the point of the exercise: VARIATION. The same event
is put to the model repeatedly with everything it has already said fed back,
and the bench counts how many distinct sentences come out. One is a recording.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv                              # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                               # noqa: E402
import safety_speech                                        # noqa: E402

# The events, in the shape the announcement path will hand over. Every one is a
# real finding this system can produce; the numbers are the ones its evidence
# would carry.
EVENTS = [
    {"key": "tire.slow_leak.FL",
     "what": "a possible slow leak in the front left tire",
     "severity": "warning", "location": "front left",
     "provenance": "RIO worked this out from the pressure trend, the car has "
                   "not reported a fault",
     "observation_window": "the last 40 minutes",
     "evidence": "down 4 PSI over the window, now 28",
     "action": "worth a look when you stop",
     "numbers": [4, 28]},
    {"key": "tire.sensor_loss.RR",
     "what": "the pressure sensor on the rear right has stopped reporting, "
             "and that corner was already losing air",
     "severity": "warning", "location": "rear right",
     "provenance": "the car's own computer stopped reporting it",
     "observation_window": "the last 3 minutes",
     "evidence": "no reading since; last good reading 26 PSI and falling",
     "action": "check it by hand at the next stop",
     "numbers": [26]},
    {"key": "headway.calm",
     "what": "the gap to the car in front is getting short",
     "severity": "coaching",
     "provenance": "RIO is watching it through the forward camera",
     "observation_window": "right now",
     "evidence": "about two and a half seconds of following distance",
     "action": "ease off a little",
     "numbers": []},
    {"key": "engine.coolant_warm",
     "what": "coolant temperature is running higher than usual",
     "severity": "warning",
     "provenance": "the car's own computer reported it",
     "observation_window": "the last 12 minutes",
     "evidence": "228 degrees, about 15 above the usual cruise figure",
     "action": "keep an eye on it",
     "numbers": [228, 15]},
    {"key": "dtc.pending.P0171",
     "what": "the car has picked up a lean-mixture condition but has not "
             "confirmed it",
     "severity": "info", "unconfirmed": True,
     "provenance": "the car's own computer, as a pending code",
     "observation_window": "this drive",
     "evidence": "P0171, pending, not confirmed",
     "action": "nothing yet; it may clear on its own",
     "numbers": []},
    {"key": "link.telemetry_lost",
     "what": "the link to the car's data has dropped",
     "severity": "info",
     "provenance": "RIO noticed the feed stop",
     "observation_window": "the last 30 seconds",
     "evidence": "no telemetry for 30 seconds",
     "action": "nothing to do; RIO will pick it back up",
     "numbers": [30]},
]

for _e in EVENTS:
    _e.setdefault("fallback", _e["what"].capitalize() + ".")


def run(model: str, n: int) -> dict:
    old = config.SAFETY_PHRASE_MODEL
    config.SAFETY_PHRASE_MODEL = model
    lat, passed, total, distinct_counts = [], 0, 0, []
    rejects = []
    try:
        for ev in EVENTS:
            safety_speech.forget("bench")
            seen = set()
            for _ in range(n):
                got = safety_speech.phrase(ev, session_key="bench",
                                           issue_key=ev["key"],
                                           timeout_s=20.0)
                total += 1
                lat.append(got["ms"])
                if got["source"] == "generated":
                    passed += 1
                    seen.add(safety_speech._norm(got["text"]))
                else:
                    rejects.append((ev["key"], got.get("note", "")))
            distinct_counts.append((ev["key"], len(seen), n))
    finally:
        config.SAFETY_PHRASE_MODEL = old
    return {"model": model, "lat": lat, "passed": passed, "total": total,
            "distinct": distinct_counts, "rejects": rejects}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--models", default="gpt-5.6-luna,gpt-5.6-terra,gpt-5.5")
    ap.add_argument("--show", action="store_true", help="print every line")
    a = ap.parse_args()
    rows = []
    for m in [x.strip() for x in a.models.split(",") if x.strip()]:
        print(f"-- {m}", flush=True)
        r = run(m, a.n)
        rows.append(r)
        p50 = statistics.median(r["lat"])
        p95 = (statistics.quantiles(r["lat"], n=20)[18]
               if len(r["lat"]) >= 20 else max(r["lat"]))
        print(f'   honest {r["passed"]}/{r["total"]}   '
              f'whole line p50 {p50:.0f} ms  p95 {p95:.0f} ms', flush=True)
        for key, d, n in r["distinct"]:
            flag = "  <- REPEATED" if d < n else ""
            print(f'     {key:<24} {d}/{n} distinct{flag}', flush=True)
        for key, why in r["rejects"][:5]:
            print(f'     rejected {key}: {why}', flush=True)
    print()
    print(f'{"model":<18}{"honest":>10}{"p50 ms":>10}{"p95 ms":>10}')
    for r in rows:
        p50 = statistics.median(r["lat"])
        p95 = (statistics.quantiles(r["lat"], n=20)[18]
               if len(r["lat"]) >= 20 else max(r["lat"]))
        print(f'{r["model"]:<18}{r["passed"]}/{r["total"]:<7}'
              f'{p50:>10.0f}{p95:>10.0f}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
