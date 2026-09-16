"""Every shape of weather question, answered out of live data, end to end.

    python -m tools.weather_shapes
    python -m tools.weather_shapes --lat 29.76 --lng -95.37

The selftest asserts that the honesty rules HOLD. This shows what they sound
like. It runs the real service against real coordinates, hands the real
normalized context and the real rule text to the real conversational model, and
prints what RIO would actually say for each of the shapes the spec names:

    1  "What's the weather?"
    2  "Is it going to rain?"
    3  "What do you see?"          -- weather visibly relevant
    4  proactive                   -- gated by weather_policy, usually silent
    5  vision vs data conflict     -- wet road, no reported rain
    6  the forecast is unavailable -- network, or a region Google does not cover

Why it is a tool and not a paragraph in a commit message: these six sentences
are the feature. Everything else in weather.py exists to make them true, and
the only way to see whether they read like a passenger rather than a weather
report is to read them. A regression here would pass every assertion in the
selftest and still be wrong.

It spends a handful of billed weather requests and a few model calls.
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv                              # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                               # noqa: E402
import weather                                              # noqa: E402
import weather_policy as wp                                 # noqa: E402

VOICE = (
    "You are RIO, riding shotgun — she/her. Sharp, easygoing, genuinely into "
    "cars. Contractions always, fragments fine. Two or three sentences at "
    "most. Straight into it: no preamble, no 'great question'. Never call "
    "yourself an AI. Never address the driver by any name."
)


def ask(rules: str, user: str) -> str:
    from openai import OpenAI

    cl = OpenAI(timeout=40)
    r = cl.responses.create(
        model=config.OPENAI_CHAT_MODEL,
        instructions=VOICE + "\n\n" + rules,
        input=[{"role": "user", "content": user}],
        max_output_tokens=400,
        reasoning={"effort": "none"},
    )
    return (getattr(r, "output_text", "") or "").strip()


def block(ctx: dict) -> str:
    """The context as the model receives it — facts only, no prose."""
    return (f"current: {ctx['current']}\n"
            f"units: {ctx['units']}\n"
            f"forecast: {ctx['forecast']}\n"
            f"age_s: {ctx['age_s']}  fetched_for: {ctx['fetched_for']}")


def shape(n, title, driver, ctx, seen=None, rules=None):
    print(f"\n{'-' * 70}\n{n}. {title}")
    print(f"   DRIVER: {driver}")
    if seen:
        print(f"   CAMERA: {seen}")
    parts = []
    if ctx is None:
        parts.append("WEATHER DATA: unavailable — could not be reached.")
    else:
        parts.append("WEATHER DATA (only source for numbers and for later):\n"
                     + block(ctx))
    if seen:
        parts.append("WHAT THE CAMERA SEES (only source for what is "
                     f"visible):\n{seen}")
    parts.append(f'DRIVER: "{driver}"')
    out = ask(rules or (weather.RULES_ABSENT if ctx is None
                        else weather.RULES_OK), "\n\n".join(parts))
    print(f"   RIO:    {out}")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, default=34.0522)
    ap.add_argument("--lng", type=float, default=-118.2437)
    a = ap.parse_args()

    print(f"Live weather shapes at {a.lat}, {a.lng}")
    weather.forget("_shapes")
    ctx = weather.get_weather_context(a.lat, a.lng, session_key="_shapes")
    if not ctx.get("ok"):
        print(f"  no context here ({ctx.get('note')}) — the failure shape is "
              f"all that can be shown.")
        shape(6, "the forecast is unavailable", "Is it going to rain?", None,
              seen="Heavy dark cloud building ahead over the freeway.")
        return 0

    cur, fc = ctx["current"], ctx["forecast"]
    print(f"  {cur['condition']}, {cur['temperature']}°, wind "
          f"{cur['wind_speed']}, visibility {cur['visibility']}, "
          f"next-hour precip {fc['next_hour_precipitation_probability']}%, "
          f"{ctx['billed_requests']} billed requests, {ctx['took_ms']} ms")

    shape(1, '"What\'s the weather?"', "What's the weather?", ctx)

    shape(2, '"Is it going to rain?"', "Is it going to rain?", ctx)

    shape(3, '"What do you see?" — weather visibly relevant',
          "What do you see out there?", ctx,
          seen="Traffic moving normally, three lanes, moderate density. Cloud "
               "building on the horizon ahead, darker than the sky overhead.")

    # 4. Proactive. The policy decides, not the model, and it is usually silent
    #    — which is the feature. A clear sky prints the refusal and its reason.
    print(f"\n{'-' * 70}\n4. proactive — the policy decides whether to speak")
    pol = wp.WeatherPolicy()
    good, why = weather.usable(ctx, a.lat, a.lng)
    d = pol.decide(ctx, time.time(), usable=good)
    print(f"   findings: {[f['kind'] for f in wp.findings(ctx)] or 'none'}")
    print(f"   decision: speak={d['speak']} reason={d['reason']} "
          f"priority={d['priority']} severity={d['severity']}")
    if d["speak"]:
        f = d["finding"]
        out = ask(weather.RULES_OK + "\n\nYou are VOLUNTEERING this — the "
                  "driver did not ask. One sentence, worth the interruption, "
                  "no preamble. Say what the numbers are ABOUT, not just the "
                  "numbers.",
                  f"An advisory the car raised.\n"
                  f"WHAT IT IS ABOUT: {f['what']}\n"
                  f"FACTS: {f['facts']}\n\nSay it in one line.")
        print(f"   RIO:      {out}")
    else:
        print("   RIO:      (silence — correct; there is no reason to speak)")

    shape(5, "vision vs data conflict — wet road, no reported rain",
          "Is it raining?", ctx,
          seen="Road surface wet and dark, visible spray off the tyres of the "
               "truck ahead. Overcast. Nothing hitting the windscreen.")

    shape(6, "the forecast is unavailable", "Is it going to rain?", None,
          seen="Heavy dark cloud building directly ahead over the freeway, "
               "much darker than the sky behind. Light has gone flat and grey.")

    print(f"\n{'-' * 70}")
    print(f"  total billed weather requests this run: "
          f"{ctx['billed_requests']} (one refresh; every shape after the "
          f"first was served from cache)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
