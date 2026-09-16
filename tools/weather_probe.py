"""What Google's Weather API actually returns on THIS key, asked rather than assumed.

    python -m tools.weather_probe
    python -m tools.weather_probe --lat 51.5 --lng -0.12

Amendment A: confirm the API is enabled on the existing key and report what it
really does — fields, forecast horizon, units, limits — before any of it is
believed. This is the script that produced the findings written into
LICENSING.md §3 and weather.py's header, kept in the tree so those findings can
be re-checked rather than trusted because they were once true.

It is a READ of the live API and it spends billed requests: about a dozen,
which at the Weather Usage rate is a fraction of a cent and is inside the free
monthly allowance many times over. It is not run by boot.sh for that reason —
a probe that costs money should be a thing somebody decides to do.

Why this exists as a file at all, rather than as a paragraph somebody wrote
once: "the API returns X" is a claim with a date on it. Endpoints gain fields,
horizons change, an alerts endpoint may one day exist — and the day it does,
the honest thing is for this to say so rather than for weather.py's `alerts: []`
to keep quietly meaning "not known" forever.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv                              # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import httpx                                                # noqa: E402

import config                                               # noqa: E402
import weather                                              # noqa: E402

BASE = "https://weather.googleapis.com/v1"


def _key():
    k = os.getenv("GOOGLE_MAPS_API_KEY", "").strip()
    if not k:
        print("GOOGLE_MAPS_API_KEY is not set — nothing to probe.")
        raise SystemExit(2)
    return k


def _call(path, lat, lng, **extra):
    params = {"key": _key(), "location.latitude": lat,
              "location.longitude": lng}
    params.update(extra)
    t0 = time.time()
    r = httpx.get(f"{BASE}/{path}", params=params, timeout=30)
    ms = round((time.time() - t0) * 1000)
    return r, ms


def _leaves(obj, prefix=""):
    """Every field the response actually carried, flattened.

    The point is the SET of fields, not their values: a field that exists today
    and is absent tomorrow is the kind of change that turns a normaliser into a
    silent producer of Nones.
    """
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _leaves(v, f"{prefix}.{k}" if prefix else k)
    elif isinstance(obj, list):
        if obj:
            out += _leaves(obj[0], f"{prefix}[]")
    else:
        out.append(prefix)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=34.0522)
    ap.add_argument("--lng", type=float, default=-118.2437)
    a = ap.parse_args()
    lat, lng = a.lat, a.lng
    spent = 0

    print(f"Probing Google Weather API at {lat}, {lng}\n" + "=" * 62)

    # 1. Is it enabled at all?
    print("\n-- 1. is the Weather API enabled on this key")
    r, ms = _call("currentConditions:lookup", lat, lng)
    spent += 1
    print(f"   currentConditions:lookup -> HTTP {r.status_code}  ({ms} ms)")
    if r.status_code != 200:
        print("   " + r.text[:400])
        print("\n   NOT ENABLED. Enable the Weather API on this project's key "
              "before anything else here means anything.")
        return 1
    cur = r.json()

    print("\n-- 2. what a current-conditions response contains")
    for f in sorted(set(_leaves(cur))):
        print(f"   {f}")

    # 3. Units are a REQUEST parameter, not a response property.
    print("\n-- 3. units follow the request, not the location")
    for system in ("METRIC", "IMPERIAL"):
        r, _ms = _call("currentConditions:lookup", lat, lng,
                       unitsSystem=system)
        spent += 1
        d = r.json()
        print(f"   {system:9s} temperature={d.get('temperature')}  "
              f"wind={(d.get('wind') or {}).get('speed')}  "
              f"visibility={d.get('visibility')}")

    # 4. Forecast horizons, found by asking rather than by reading.
    print("\n-- 4. forecast horizon, by bisection against the real limit")
    for path, param, lo, hi in (("forecast/hours:lookup", "hours", 1, 300),
                                ("forecast/days:lookup", "days", 1, 20)):
        good, bad = lo, hi
        while bad - good > 1:
            mid = (good + bad) // 2
            r, _ms = _call(path, lat, lng, **{param: mid})
            spent += 1
            if r.status_code == 200:
                good = mid
            else:
                bad = mid
        print(f"   {path:26s} max {param} = {good}")

    print("\n-- 5. page size, and what one forecast hour contains")
    r, ms = _call("forecast/hours:lookup", lat, lng)
    spent += 1
    d = r.json()
    hrs = d.get("forecastHours") or []
    print(f"   default page: {len(hrs)} hours, "
          f"nextPageToken={'yes' if d.get('nextPageToken') else 'no'}  ({ms} ms)")
    if hrs:
        for f in sorted(set(_leaves(hrs[0]))):
            print(f"   {f}")

    print("\n-- 6. the daily call, for sunrise/sunset and the day's high/low")
    r, ms = _call("forecast/days:lookup", lat, lng, days=1)
    spent += 1
    days = (r.json().get("forecastDays") or [])
    if days:
        print(f"   day-level fields: {', '.join(sorted(days[0].keys()))}")
        print(f"   sunEvents: {json.dumps(days[0].get('sunEvents'))}")

    # 7. The finding that shapes the whole context: there is no alerts endpoint.
    print("\n-- 7. severe weather alerts")
    found_alerts = False
    for path in ("alerts:lookup", "weatherAlerts:lookup",
                 "severeWeather:lookup", "alerts"):
        try:
            r, _ms = _call(path, lat, lng)
            spent += 1
            mark = "FOUND" if r.status_code == 200 else f"{r.status_code}"
            print(f"   /{path:24s} -> {mark}")
            if r.status_code == 200:
                found_alerts = True
        except Exception as e:
            print(f"   /{path:24s} -> {type(e).__name__}")
    if found_alerts:
        print("\n   !! AN ALERTS ENDPOINT ANSWERED. weather.py documents that "
              "none exists and hard-codes `alerts: []` on that basis — that "
              "comment, and the field, now need revisiting.")
    else:
        print("\n   None. Confirms weather.py's `alerts` is always empty and "
              "means NOT KNOWN — never 'no severe weather'.")

    # 8. Latency, which is what decides whether this can sit in a voice turn.
    print("\n-- 8. what a full refresh costs, end to end")
    weather.forget("_probe")
    t0 = time.time()
    ctx = weather.get_weather_context(lat, lng, session_key="_probe")
    ms = round((time.time() - t0) * 1000)
    spent += int(ctx.get("billed_requests") or 0)
    print(f"   cold: {ms} ms, {ctx.get('billed_requests')} billed requests")
    t0 = time.time()
    ctx2 = weather.get_weather_context(lat, lng, session_key="_probe")
    print(f"   warm: {round((time.time() - t0) * 1000)} ms, "
          f"{ctx2.get('billed_requests')} billed requests "
          f"(cached={ctx2.get('cached')})")

    print("\n-- 9. the normalized context RIO actually reasons over")
    print(json.dumps({k: v for k, v in ctx.items() if k != "rules"},
                     indent=1, default=str))

    print("\n" + "=" * 62)
    print(f"  ~{spent} billed requests spent by this probe.")
    print("  Attribution required on display: "
          f"{ctx.get('attribution')!r} — see LICENSING.md §3.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
