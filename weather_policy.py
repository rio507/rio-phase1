"""weather_policy.py — whether RIO says something about the weather unasked.

Amendment B, and nothing else. Answering "is it going to rain" is the
conversation path and never comes through here: the driver asked, so the only
question is whether the answer is honest, which weather.py settles. This file
is for the other case — RIO noticing something and opening her mouth about it
with nobody having said a word — and that case needs a gate, because the
failure mode is not a wrong number. It is a passenger who comments on the
weather.

    "It's sunny outside."
    "Still sunny."
    "Nice day."

Each of those is true, harmless, and the reason somebody turns the assistant
off. RIO is silence-first; proactive weather needs a REASON, and this module is
the only thing that decides there is one.

WHY IT IS A SEPARATE MODULE, LIKE THE HEALTH POLICY
---------------------------------------------------
Same shape as vehicle_health_policy.py and for a related reason, weakened only
where the stakes genuinely are lower. There, a model must never be able to
raise or suppress an alarm. Here, a model must never be able to decide it is
worth interrupting a quiet drive — which is the same structural rule about a
lower-stakes decision, and it gets the same structure:

    THE LLM MAY ANSWER WEATHER QUESTIONS, AND PHRASE AN ADVISORY IT IS GIVEN.
    THE LLM NEVER DECIDES WHETHER AN UNPROMPTED WEATHER LINE HAPPENS.

So this file imports nothing but the standard library, keeps its tunables as
module constants rather than config entries, and takes the clock as an argument
on every call. The first two make the boundary countable; the third makes a
drive replayable from a log, which is the only way "was RIO too chatty" is ever
answerable with something other than an opinion.

WHERE IT SITS IN THE MOUTH
--------------------------
A finding from here is submitted to the speech arbiter at CONVERSATION
priority — P.CONVO, the lowest tier in static/rio_speech.js. That is not a
convention, it is the mechanism: CONVO is below SAFETY, VEHICLE_HEALTH,
TURN_NEAR and NAV, so a weather line structurally cannot preempt a collision
warning, a failing tire or a turn. It is cut off by all four, and being cut off
is correct — the driver can ask again, and a turn cannot.

On top of the tier, three gates that the arbiter cannot provide because it does
not know what weather is:

  worth saying    a threshold per kind. Rain far enough out to act on, fog
                  that is actually closing in, wind that moves a car. Not "it
                  is raining", which the driver can see through the windscreen.
  cooldown        per kind, so the same fog is mentioned once and not every
                  tick until it clears.
  minimum gap     nothing at all inside a window of the last proactive line,
                  whatever it was about, so two conditions crossing together
                  are two sentences a beat apart rather than one on top of the
                  other.

Every decision returns a reason, spoken or not. A silence with a reason is as
useful for tuning as an utterance, and both belong in the same log.
"""

# ===========================================================================
# PROVISIONAL BLOCK -- every tunable value in this module lives here.
#
# Same inherited convention as headway/live_policy.py and
# vehicle_health_policy.py, and it still applies:
#   "PROVISIONAL prototype values. These are engineering starting points for
#    shadow-mode tuning -- NOT validated safety thresholds. Do not represent
#    them as such anywhere, ever."
#
# Nothing in this file is a safety threshold in any case. Fog is mentioned here
# as context; if visibility ever becomes a reason to SLOW DOWN, that decision
# belongs in the deterministic safety layer and not in a conversational
# advisory.
# ===========================================================================

# --- rain that has not started yet -----------------------------------------
# The only genuinely useful proactive weather line: precipitation is coming,
# far enough out that knowing changes something (a stop, a route, a lane) and
# close enough that it is about this drive.
#
# 60% rather than the 30% weather.py uses to find the first rainy hour. That
# lower number answers "when does it start" for a driver who ASKED; this one
# decides whether to interrupt somebody who did not, and interrupting on a coin
# flip is how a feature becomes noise.
PROACTIVE_PRECIP_PROB = 60.0

# The window it has to fall in. Beyond 45 minutes it is not about this stretch
# of road and it can wait to be asked about; under 5 it is either already
# visible through the glass or too late to be worth the attention.
PRECIP_LEAD_MAX_MIN = 45.0
PRECIP_LEAD_MIN_MIN = 5.0

# --- visibility ------------------------------------------------------------
# Fog worth naming. The number is the one Google reports, in whatever unit the
# context carries, so the comparison is done in miles or kilometres to match --
# see `_vis_threshold`. Two miles is the point at which a driver has already
# noticed and a figure is genuinely useful; below a mile it is the thing
# dominating the drive.
VIS_LOW_MI = 2.0
VIS_LOW_KM = 3.2

# --- wind ------------------------------------------------------------------
# Gusts that move a car rather than wind that exists. A high-sided vehicle on a
# bridge is the case; 30 mph sustained or 45 gusting is where a driver feels it
# through the wheel.
WIND_HIGH_MPH = 30.0
WIND_GUST_HIGH_MPH = 45.0
WIND_HIGH_KPH = 48.0
WIND_GUST_HIGH_KPH = 72.0

# --- thunderstorms ---------------------------------------------------------
THUNDER_PROB = 50.0

# --- the quiet rules -------------------------------------------------------
# How long the same KIND of finding stays quiet once said. Fifteen minutes: a
# driver told rain is coming does not need telling again while it is still
# coming, and a two-hour drive through changing weather still gets more than
# one mention.
COOLDOWN_S = 900.0

# Nothing at all inside this window of the previous proactive weather line,
# whatever it was about. The arbiter would order them anyway; this stops the
# second being submitted, so the reason is recorded rather than lost inside a
# supersede.
MIN_GAP_S = 120.0

# After the driver asks about the weather, RIO does not volunteer more of it
# for this long. Without it, the most likely moment for an unprompted weather
# line is ninety seconds after answering a weather question — which reads as
# not having listened.
POST_ANSWER_QUIET_S = 300.0

# A finding whose facts have not changed is not a new finding. This is how much
# a probability has to move to count as news inside the cooldown.
WORSEN_PROB_PTS = 25.0

# ===========================================================================
# END PROVISIONAL BLOCK
# ===========================================================================

# --- the kinds, which are also the cooldown keys ---------------------------
K_RAIN_SOON = "rain_soon"
K_VISIBILITY = "visibility_low"
K_WIND = "wind_high"
K_THUNDER = "thunderstorm"

# --- reasons recorded on every decision ------------------------------------
R_SPEAK = "speak"
R_NOTHING = "nothing_worth_saying"
R_COOLDOWN = "suppressed_by_cooldown"
R_MIN_GAP = "suppressed_by_min_gap"
R_POST_ANSWER = "suppressed_after_weather_answer"
R_NO_CONTEXT = "suppressed_no_usable_context"
R_BUSY = "suppressed_mouth_busy_above_convo"
R_DISABLED = "suppressed_disabled"

SPEAK_REASONS = (R_SPEAK,)


def _f(v):
    """A number or None. Never a default: absent and zero are different."""
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return float(v)


def _is_imperial(ctx: dict) -> bool:
    u = ((ctx.get("units") or {}).get("temperature") or "").upper()
    return u.startswith("F")


def _vis_threshold(ctx: dict) -> float:
    return VIS_LOW_MI if _is_imperial(ctx) else VIS_LOW_KM


def _wind_thresholds(ctx: dict) -> tuple:
    return ((WIND_HIGH_MPH, WIND_GUST_HIGH_MPH) if _is_imperial(ctx)
            else (WIND_HIGH_KPH, WIND_GUST_HIGH_KPH))


# ---------------------------------------------------------------------------
# What is worth saying, before anything about whether it may be said
# ---------------------------------------------------------------------------

def findings(ctx: dict) -> list:
    """The weather context -> advisory findings, highest first. No timing.

    Deliberately separate from the timing gates so that "was there anything to
    say" and "was RIO allowed to say it" are two answers in the log rather than
    one. A drive that suppressed forty findings and a drive that had none are
    very different drives and they should not look the same.

    Every finding carries FACTS ONLY — numbers straight out of the context,
    with their units, plus a `what` naming what those numbers are ABOUT. No
    sentence is composed here. The phrasing happens where everything RIO says
    is phrased, and it is checked against these facts.

    The `what` is not decoration and it was not in the first version. Without
    it a thunderstorm finding reached the phrasing step as
    {"probability": 60, "condition": "Partly sunny"} and came back as "partly
    sunny ahead, with about a 60 percent chance showing" — every number real,
    every honesty check passed, and the sentence says nothing. A fact that does
    not name its own subject is not a fact yet, and neither is a key called
    `condition` on an advisory about something the current condition is not.
    """
    if not isinstance(ctx, dict) or not ctx.get("ok"):
        return []
    cur = ctx.get("current") or {}
    fc = ctx.get("forecast") or {}
    units = ctx.get("units") or {}
    out = []

    # Rain that has not started. The one line a driver actually uses.
    nxt = fc.get("next_precipitation") or None
    if isinstance(nxt, dict):
        p = _f(nxt.get("probability"))
        lead = _f(nxt.get("in_minutes"))
        if (p is not None and p >= PROACTIVE_PRECIP_PROB
                and lead is not None
                and PRECIP_LEAD_MIN_MIN <= lead <= PRECIP_LEAD_MAX_MIN):
            out.append({
                "kind": K_RAIN_SOON,
                "rank": 3,
                # `what` names the subject of every number beside it. Without
                # it the facts are a probability with nothing attached, and a
                # model handed {"probability": 60} will write "about a 60
                # percent chance showing" -- a sentence that is not wrong so
                # much as meaningless, and which passed every honesty check
                # here because the number in it was real.
                "what": "rain starting",
                "facts": {
                    "precipitation_probability": p,
                    "starts_in_minutes": lead,
                    "starts_around": nxt.get("at"),
                    "precipitation_type": nxt.get("type") or "RAIN",
                    "forecast_condition": nxt.get("condition"),
                },
                "magnitude": p,
            })

    # Fog. Reported because the figure is the useful part — the driver can see
    # that it is foggy and cannot see how far it extends.
    vis = _f(cur.get("visibility"))
    if vis is not None and vis <= _vis_threshold(ctx):
        out.append({
            "kind": K_VISIBILITY,
            "rank": 3,
            "what": "visibility dropping",
            "facts": {
                "visibility": vis,
                "visibility_unit": units.get("visibility") or "",
                "current_condition": cur.get("condition"),
            },
            # Lower visibility is worse, so magnitude inverts to keep "higher
            # is worse" true across every kind.
            "magnitude": max(0.0, _vis_threshold(ctx) - vis),
        })

    # Thunderstorms.
    th = _f(cur.get("thunderstorm_probability"))
    if th is not None and th >= THUNDER_PROB:
        out.append({
            "kind": K_THUNDER, "rank": 3,
            "what": "thunderstorms in the area",
            "facts": {"thunderstorm_probability": th,
                      "current_condition": cur.get("condition")},
            "magnitude": th,
        })

    # Wind that moves the car.
    hi, gust_hi = _wind_thresholds(ctx)
    ws = _f(cur.get("wind_speed"))
    if ws is not None and ws >= hi:
        out.append({
            "kind": K_WIND, "rank": 2,
            "what": "strong wind",
            "facts": {"wind_speed": ws,
                      "wind_speed_unit": units.get("wind_speed") or "",
                      "wind_direction": cur.get("wind_direction") or ""},
            "magnitude": ws,
        })

    out.sort(key=lambda f: (-f["rank"], -float(f.get("magnitude") or 0.0)))
    return out


# ---------------------------------------------------------------------------
# Whether it may be said, which is the part with a memory
# ---------------------------------------------------------------------------

class WeatherPolicy:
    """One drive's memory of what it has volunteered about the sky.

    The clock is passed in on every call rather than read. That is the same
    choice live_policy.py and vehicle_health_policy.py make, for the same
    reason: a policy that reads the clock itself cannot be replayed from a log,
    and a proactive-speech policy that cannot be replayed cannot be tuned.
    """

    def __init__(self):
        self._last_spoke_t = None        # any proactive weather line
        self._said = {}                  # kind -> {"t", "magnitude"}
        self._answered_t = None          # last time a weather QUESTION was answered

    # -- the two things the rest of the system tells it ---------------------

    def answered(self, t: float) -> None:
        """The driver asked about the weather and RIO answered.

        Not a suppression in itself — it starts the quiet window. RIO having
        just discussed the forecast is the worst possible moment to volunteer
        more of it.
        """
        self._answered_t = float(t)

    def reset(self) -> None:
        """A new drive volunteers nothing on the strength of the last one."""
        self._last_spoke_t = None
        self._said.clear()
        self._answered_t = None

    # -- the decision -------------------------------------------------------

    def decide(self, ctx: dict, t: float, usable: bool = True,
               mouth_busy_above_convo: bool = False,
               enabled: bool = True) -> dict:
        """-> {"speak": bool, "reason": str, "finding": dict|None}

        `usable` is weather.usable() — amendment C's staleness and distance
        test — passed IN rather than computed, because this module does not
        import weather and must not. A context that may not be spoken when
        asked for certainly may not be volunteered.

        `mouth_busy_above_convo` is the arbiter's own state. The arbiter would
        handle this by ordering, and the tier already guarantees a weather line
        cannot preempt anything above it. Refusing to submit at all while
        something more important is being said is the stronger version: a
        weather advisory that queues behind a turn and plays after it has
        outlived the moment it was true for, which is the failure `expire,
        never catch up` exists to prevent one tier up.
        """
        t = float(t)
        if not enabled:
            return self._quiet(R_DISABLED)
        if not usable or not isinstance(ctx, dict) or not ctx.get("ok"):
            return self._quiet(R_NO_CONTEXT)
        if mouth_busy_above_convo:
            return self._quiet(R_BUSY)

        found = findings(ctx)
        if not found:
            return self._quiet(R_NOTHING)

        # Timing gates last, so that a suppressed line is one that was actually
        # worth saying. Recording a "nothing to say" as a cooldown suppression
        # would make the tuning data say the opposite of what happened.
        if (self._answered_t is not None
                and (t - self._answered_t) < POST_ANSWER_QUIET_S):
            return self._quiet(R_POST_ANSWER, found[0])
        if (self._last_spoke_t is not None
                and (t - self._last_spoke_t) < MIN_GAP_S):
            return self._quiet(R_MIN_GAP, found[0])

        for f in found:
            prev = self._said.get(f["kind"])
            if prev is not None and (t - prev["t"]) < COOLDOWN_S:
                # Inside the cooldown, but the fact may have genuinely moved.
                # A 40% chance that became 85% is news; 61% becoming 64% is the
                # same sentence again.
                before = float(prev.get("magnitude") or 0.0)
                now_mag = float(f.get("magnitude") or 0.0)
                if (now_mag - before) < WORSEN_PROB_PTS:
                    continue
            self._said[f["kind"]] = {"t": t,
                                     "magnitude": float(f.get("magnitude") or 0.0)}
            self._last_spoke_t = t
            return {"speak": True, "reason": R_SPEAK, "finding": f,
                    # The tier, named here so that a caller cannot pick a
                    # different one. It is the whole of "may never preempt
                    # safety, health or navigation".
                    "priority": "CONVO",
                    "severity": "advisory"}
        return self._quiet(R_COOLDOWN, found[0])

    def _quiet(self, reason: str, finding: dict = None) -> dict:
        return {"speak": False, "reason": reason, "finding": finding,
                "priority": "CONVO", "severity": "advisory"}

    def state(self) -> dict:
        return {
            "last_spoke_t": self._last_spoke_t,
            "answered_t": self._answered_t,
            "said": {k: dict(v) for k, v in self._said.items()},
        }
