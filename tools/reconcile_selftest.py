"""reconcile_selftest.py — the detector wins on existence, and only on that.

    python tools/reconcile_selftest.py

WHAT IS BEING DEFENDED
----------------------
Cosmos read a frame as "TRAFFIC: none" while RF-DETR was tracking a pedestrian
and a motorcycle on it. Both went to the driver -- one as boxes, one as the
sentence RIO composed her answer from -- and nothing compared them.

The rule is asymmetric and the asymmetry is the whole design, so it is asserted
in both directions here:

  a reading that claims the road is EMPTY, contradicted by confirmed tracks,
  is marked;

  a reading that REPORTS something the detector has no track for is never
  touched, because RF-DETR has seven classes and a tractor is not one of them.

And the things that must NOT be contested: RISK, which is a judgement rather
than an existence claim; a field the model never wrote; a single-frame
detection that has not been held; and a census from a feed that has stopped.
"""
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import reconcile as rc                                    # noqa: E402
import rio_prompts as rp                                  # noqa: E402
from headway import census as census_mod                  # noqa: E402

checks = 0
failures = 0


def ok(name, cond, extra=""):
    global checks, failures
    checks += 1
    if not cond:
        failures += 1
    print(("  ok   " if cond else "  FAIL ") + name + (f" — {extra}" if extra else ""))


def obj(label, confirmed=True, age_s=2.0, vulnerable=False):
    return {"label": label, "confirmed": confirmed, "vulnerable": vulnerable,
            "is_lead": False, "age_s": age_s, "range_m": 20.0}


def contest(text, objects, field="TRAFFIC"):
    reading = rp.split_sensor_reading(
        f"ROAD: two lanes, asphalt | TRAFFIC: {text} | RISK: none seen"
        if field == "TRAFFIC" else
        f"ROAD: two lanes, asphalt | TRAFFIC: two cars | RISK: {text}")
    census_mod.clear()
    census_mod.note("k", objects)
    return rc.check(reading["fields"], census_mod.current("k"))


print("\n== 1. what the field is claiming ==\n")
for t in ("none", "none seen", "no traffic", "the road is clear", "clear ahead",
          "nothing ahead", "no road users", "no cars in the lane ahead", "--"):
    ok(f"a claim of an empty road: {t!r}", rc.claims_empty(t))
for t in ("two cars ahead", "queue ahead, none braking", "car 20 km/h",
          "one pedestrian on the right", "none of the cars are braking",
          "unreadable", "no view of the road", "roadworks, no through traffic lane"):
    ok(f"...and not one: {t!r}", not rc.claims_empty(t))

print("\n== 2. the detector wins on existence ==\n")
r = contest("none", [obj("pedestrian", vulnerable=True), obj("motorcycle", vulnerable=True)])
ok("'TRAFFIC: none' against a tracked pedestrian and motorcycle is contested",
   r["contested"] == ["TRAFFIC"], str(r["contested"]))
traffic = [f for f in r["fields"] if f["name"] == "TRAFFIC"][0]
ok("...and the tracker's own account rides with it",
   traffic.get("detector") == "one motorcycle and one pedestrian",
   str(traffic.get("detector")))
ok("...the model's words are KEPT, not rewritten", traffic["text"] == "none")
ok("...and RIO is told which instrument to believe",
   "TRACKER IS RIGHT ABOUT WHAT IS THERE" in rc.rule_for(r)
   and "one motorcycle and one pedestrian" in rc.rule_for(r))

print("\n== 3. ...and loses on absence, which is the asymmetry ==\n")
r = contest("two cars ahead", [])
ok("a reading that REPORTS traffic with no tracks at all is left alone",
   r["contested"] == [] and r["detector"] is None)
r = contest("a tractor pulling out", [])
ok("...including a class RF-DETR does not have", r["contested"] == [])
r = contest("two cars ahead", [obj("car")])
ok("...and agreement is not a contest either", r["contested"] == [])

print("\n== 4. what must never be contested ==\n")
r = contest("none seen", [obj("pedestrian", vulnerable=True)], field="RISK")
ok("RISK is a judgement, not an existence claim, so it is never contested",
   r["contested"] == [], str(r["contested"]))
reading = rp.split_sensor_reading("ROAD: two lanes | RISK: none seen")
census_mod.clear(); census_mod.note("k", [obj("pedestrian")])
r = rc.check(reading["fields"], census_mod.current("k"))
ok("a field the model never wrote is missing, not a false claim",
   r["contested"] == [], str(r["contested"]))
r = contest("none", [obj("car", confirmed=False)])
ok("an unconfirmed box -- below the size floor -- does not overrule a reading",
   r["contested"] == [])
r = contest("none", [obj("car", age_s=0.1)])
ok(f"...nor one held for less than {census_mod.MIN_HELD_S} s", r["contested"] == [])

print("\n== 5. a census goes stale with the feed ==\n")
census_mod.clear()
census_mod.note("k", [obj("pedestrian")], t=time.time() - (census_mod.MAX_AGE_S + 1))
ok("a census older than the window is not served at all",
   census_mod.current("k") == {})
reading = rp.split_sensor_reading("ROAD: two lanes | TRAFFIC: none | RISK: none")
ok("...so a clip that ended cannot go on contradicting readings forever",
   rc.check(reading["fields"], census_mod.current("k"))["contested"] == [])

print("\n== 6. the census is a by-product, not a second opinion ==\n")
src = (REPO / "headway" / "live.py").read_text()
ok("headway/live.py never imports or reads the census",
   "census" not in src)
pol = (REPO / "headway" / "live_policy.py").read_text()
ok("...and neither does the policy that raises bands", "census" not in pol)
rec_src = (REPO / "reconcile.py").read_text()
ok("reconcile.py cannot speak, warn or decide",
   not any(w in rec_src for w in ("speak", "warn(", "band", "urgency")))

print(f"\n{checks - failures}/{checks} checks passed")
sys.exit(1 if failures else 0)
