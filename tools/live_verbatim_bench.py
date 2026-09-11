"""Does gpt-live-1 say a deterministic line the way it was written?

    python -m tools.live_verbatim_bench
    python -m tools.live_verbatim_bench --n 4

THE QUESTION THIS ANSWERS IS THE ONE THE MIGRATION TURNS ON. Under
gpt-realtime a deterministic line is exact BY CONSTRUCTION: response.create
carries the words and the model reads them. gpt-live-1 has no such event. The
nearest thing is session.commentary.append, and the documentation is explicit
that the model "is trained to paraphrase the appended text" and may decline to
say it at all.

That is not a defect -- it is what makes the thing a conversation rather than
a tape player -- but RIO dictates turn calls, health announcements and headway
coaching into that same mouth, and "she usually says the right sentence" is not
the standard those were written to.

TWO SCORES, because two different things can go wrong and only one of them
matters:

  spoken-identical   the sentence a driver HEARS. Punctuation is ignored, and
                     so is the difference between "24" and "twenty-four" --
                     those are the same sound coming out of a speaker, and
                     counting them as failures measures the transcriber rather
                     than the model.
  word-identical     the strict comparison, kept because it is the one that
                     catches an ADDED word. "Heads up, your front left is down
                     to twenty-four PSI" for "Front left tire is down to 24
                     PSI" is a real change: she introduced a line that was
                     written not to be introduced.

THE LINES ARE THE CAR'S. navigation.speech and vehicle_health_policy own them,
and they are imported rather than retyped so this bench cannot pass on
sentences the car does not actually say.
"""
import argparse
import asyncio
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv                              # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                               # noqa: E402
from live_harness import LiveSession, ScriptedMic, silence  # noqa: E402

_NUM = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
        "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
        "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
        "14": "fourteen", "15": "fifteen", "16": "sixteen",
        "17": "seventeen", "18": "eighteen", "19": "nineteen",
        "20": "twenty", "24": "twenty four", "30": "thirty",
        "31": "thirty one", "300": "three hundred", "400": "four hundred"}


def _words(s: str) -> list:
    s = (s or "").lower().replace("-", " ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return s.split()


def _spoken(s: str) -> list:
    """The words as a SPEAKER produces them: digits spelled, hyphens opened."""
    out = []
    for w in _words(s):
        out.extend(_NUM.get(w, w).split())
    return out


def lines() -> list:
    """The sentences RIO actually dictates, from the modules that own them."""
    from headway import live_policy
    from navigation import speech as nav_speech
    import vehicle_health_policy as vhp

    out = []
    # Headway: only the coaching tiers are dictated. The red tier plays a clip
    # and never comes through here, which is the point.
    for k in ("calm", "escalate"):
        if k in live_policy.LINE_TEXT:
            out.append(live_policy.LINE_TEXT[k])
    # Navigation: the calls with room in front of them.
    try:
        out.append(nav_speech.far_text(
            {"modifier": "right", "name": "Ocean Avenue"}, 800))
    except Exception:
        out.append("In half a mile, turn right onto Ocean Avenue.")
    out.append("In 300 feet, turn right onto Ocean Avenue.")
    # Health: real templates with real values in them.
    out.append(vhp.LINE["critical_low_pressure"].format(
        loc="front left", value="24 PSI"))
    out.append(vhp.LINE["tire_sensor_lost_driving"].format(loc="rear right"))
    out.append(vhp.LINE["coolant_temp_critical"].format(value="248 degrees"))
    return [x for x in out if x]


async def bench(n: int) -> int:
    ls = lines()
    print(f"{len(ls)} lines x {n} trials, voice={config.GPT_LIVE_VOICE}\n")
    spoken_ok = word_ok = silent = total = 0
    lat = []
    for line in ls:
        for _ in range(n):
            mic = ScriptedMic(silence(30000))
            s = LiveSession(mic)
            try:
                await s.open()
                await asyncio.sleep(0.8)
                t0 = time.time()
                s.say_line(line)
                await s.wait_quiet(max_s=16.0, quiet_s=0.9)
            finally:
                await s.close()
            said = s.said.strip()
            total += 1
            if not said:
                silent += 1
            sp = _spoken(said) == _spoken(line)
            wd = _words(said) == _words(line)
            spoken_ok += sp
            word_ok += wd
            if s.first_audio_t:
                lat.append((s.first_audio_t - t0) * 1000)
            flag = "same " if sp else "DIFF "
            print(f"  {flag} {said[:66]!r}")
            if not sp:
                print(f"        wanted {line[:66]!r}")
    import statistics
    print("\n" + "-" * 62)
    print(f"  spoken-identical  {spoken_ok}/{total}")
    print(f"  word-identical    {word_ok}/{total}")
    print(f"  said nothing      {silent}/{total}")
    if lat:
        p95 = (statistics.quantiles(lat, n=20)[18] if len(lat) >= 20
               else max(lat))
        print(f"  append -> audio   p50 {statistics.median(lat):.0f} ms   "
              f"p95 {p95:.0f} ms")
    return 0 if spoken_ok == total else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3)
    return asyncio.run(bench(ap.parse_args().n))


if __name__ == "__main__":
    raise SystemExit(main())
