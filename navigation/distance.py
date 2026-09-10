"""How far, said the way a driver hears it.

WHY THIS IS ITS OWN FILE, AND WHY IT ROUNDS SO HARD
---------------------------------------------------
Every navigation system a driver has ever used says "in half a mile", and none
of them say "in 780 metres" or "in 0.48 miles". The rounding is not sloppiness
about the measurement — it is the measurement being converted into the unit the
listener actually reasons in. A driver hearing "in half a mile" is not being
told a distance, they are being told WHICH BEAT of the approach this is: there
is one more call before the turn.

So the set of phrases this file can produce is CLOSED and small, and the
mapping into it is a bucket lookup rather than arithmetic on the live distance.
That has three consequences worth stating:

  the phrase is stable        A call fired at 790 m and one fired at 810 m both
                              say "half a mile". Two drives down the same road
                              say the same words at the same corner.

  the phrase is precomputed   The planner fires a tier, and each tier's sentence
                              — distance phrase included — was written at route
                              load. Nothing formats language while the car is
                              moving (§23), which is the rule the whole speech
                              path is built on and which a live "in %d feet"
                              would have quietly broken.

  the phrase is checkable     A closed set is a set a test can enumerate. See
                              run_distance_phrasing in tools/nav_server_selftest.

FEET, BECAUSE UNDER A TENTH OF A MILE "MILES" STOPS MEANING ANYTHING
--------------------------------------------------------------------
"In 0.1 miles" is a number a driver has to convert. Under 528 ft the unit
changes to feet, rounded to 50 — because that is the range where a driver is
picking one gap out of a row of driveways, and "in 250 feet" and "in 350 feet"
mean different things to someone doing it.

METRIC IS A CONFIGURATION, NOT A TRANSLATION
--------------------------------------------
The metric ladder is its own set of round numbers (50, 100, 200, 300, 500, 800
m, then kilometres), not the imperial one converted — because 805 m is not a
round number in the unit a metric driver thinks in, and "in 800 metres" is.
"""
from typing import Optional

M_PER_MILE = 1609.344
M_PER_FOOT = 0.3048

IMPERIAL = "imperial"
METRIC = "metric"

# Spoken, not printed. "In 2 miles" read by a TTS engine is a coin flip between
# "two" and "2"; the word removes the question. Ten is the ceiling because past
# it a navigation call is not being made anyway.
_NUMBER_WORD = {
    1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
    6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
}


def _number(n: int) -> str:
    return _NUMBER_WORD.get(n, str(n))


def _feet_phrase(meters: float) -> str:
    feet = meters / M_PER_FOOT
    if feet < 500:
        # 50-ft granularity below 500 ft, which is where a driver is picking a
        # specific gap out of a row of driveways. "In 250 feet" and "in 350
        # feet" are both things Google says and both things that mean
        # something; rounding them together to 300 does not.
        step = 50
    else:
        step = 100
    rounded = int(round(feet / step) * step)
    if rounded < 100:
        rounded = 100
    return f"{_number(rounded) if rounded <= 10 else rounded} feet"


def _mile_phrase(meters: float) -> str:
    miles = meters / M_PER_MILE
    # The three fractions everyone says, and nothing between them. A call at
    # 0.4 miles is "a quarter mile" or "half a mile"; it is never "0.4 miles".
    if miles < 0.375:
        return "a quarter mile"
    if miles < 0.625:
        return "half a mile"
    if miles < 0.875:
        return "three quarters of a mile"
    whole = int(round(miles))
    if whole <= 1:
        return "one mile"
    return f"{_number(whole)} miles"


def _metric_phrase(meters: float) -> str:
    if meters < 1000:
        # The metric ladder, as round as the imperial one and in its own
        # numbers: not 805 m, which is half a mile wearing metric clothes.
        for step in (50, 100, 200, 300, 400, 500, 600, 800):
            if meters <= step * 1.25:
                return f"{step} meters"
        return "800 meters"
    km = meters / 1000.0
    whole = int(round(km))
    if whole <= 1:
        return "one kilometer"
    return f"{_number(whole)} kilometers"


def phrase(meters: float, units: str = IMPERIAL) -> str:
    """"half a mile", "500 feet", "two miles" — no leading "in".

    The caller composes the sentence, because "In half a mile, turn right" and
    "Turn right in half a mile" are both wanted and the preposition belongs to
    whichever one is being built.
    """
    meters = max(0.0, float(meters or 0.0))
    if units == METRIC:
        return _metric_phrase(meters)
    if meters < M_PER_MILE * 0.1:
        return _feet_phrase(meters)
    return _mile_phrase(meters)


def in_phrase(meters: float, units: str = IMPERIAL) -> str:
    """"In half a mile" — capitalised, ready to begin a sentence."""
    p = phrase(meters, units)
    return "In " + p


def units_from_config(cfg=None) -> str:
    """Which ladder this deployment speaks in.

    Read once, at route build, like every other language decision on this path.
    """
    if cfg is None:
        import config as cfg  # local, so the module imports standalone in tests
    u = str(getattr(cfg, "NAV_UNITS", IMPERIAL) or IMPERIAL).strip().lower()
    return METRIC if u.startswith("m") else IMPERIAL


def announce_ladder(units: str = IMPERIAL) -> tuple:
    """Every phrase this file can produce for the tier distances, in order.

    Exists so a test can enumerate the closed set rather than sampling it, and
    so a change to the buckets shows up as a diff in one place.
    """
    probes = (30, 60, 100, 150, 200, 300, 400, 500, 800, 1200, 1609, 2400,
              3218, 4800)
    seen, out = set(), []
    for m in probes:
        p = phrase(m, units)
        if p not in seen:
            seen.add(p)
            out.append(p)
    return tuple(out)


def meters_for(miles: Optional[float] = None,
               kilometers: Optional[float] = None) -> float:
    """Tier distances are written in the unit they were designed in."""
    if miles is not None:
        return miles * M_PER_MILE
    if kilometers is not None:
        return kilometers * 1000.0
    return 0.0
