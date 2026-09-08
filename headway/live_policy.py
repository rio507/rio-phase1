"""Live headway policy v3 — three bands, voice fires automatically on band entry.

Design ref: docs/live_headway_v3.md, which **overrides §1 (bands) and §4/§6
(voice policy) of warning_logic_v2.md** for the live drive loop. Everything else
v2 specifies -- hysteresis, temporal confirmation, cooldowns, the coaching
warm-up, the confidence floor -- is preserved here in mechanism, retuned to the
three-band ladder.

Why this is a separate module rather than an edit to state.py
-------------------------------------------------------------
state.py implements v2 exactly and is covered by headway/selftest.py (109
checks) and the Stage 0 clip harness. v3 is not a retune of v2, it is a
different policy: v2's coaching is *trend-gated* (GETTING_UNSAFE is silent
unless the gap is shrinking) and v3's is *entry-gated* (crossing into the band
speaks, whatever the trend is doing). Folding both into one class would mean a
mode flag through every branch of the voice table. Two modules, one live and one
offline, each readable on its own.

The two share the pure maths in state.py (compute_tau, compute_ttc,
classify_trend, compute_confidence) so there is exactly one implementation of
each.

LLM firewall (v2 §9), preserved verbatim in force: this module imports only the
stdlib and headway.state. Nothing in tick() can read a model output because
there is nothing here to read one with -- Qwen supplies the ROI upstream and
never reaches this file. headway/live_selftest.py asserts that.

BAND LADDER (v3, overriding v2 §1's five bands)
-----------------------------------------------
Every band is a statement about TIME HEADWAY -- gap divided by own speed, the
two-second rule -- and never about metres. That is the point: 20 m behind a car
is a comfortable gap in town and a second and a bit on a freeway, and only the
time version means the same thing at both ends.

    NORMAL           tau >= 3.0      silent, unobtrusive display
    GETTING_UNSAFE   2.0 <= tau < 3.0  amber; calm line on confirmed entry
    UNSAFE           tau < 2.0       red; pre-rendered alert on confirmed entry

    SUPPRESSED       v_host < 5 m/s  parking-lot creep: tau is meaningless
    IMPLAUSIBLE      tau < 0.35 at speed -- not a following distance
    UNKNOWN          no speed at all  never classify on a speed we do not have

TWO THINGS THE FIRST REAL DRIVE ADDED
-------------------------------------
TTC IS A TRIGGER AGAIN. v2 §1/§9 makes time-to-contact under 2.5 s with the gap
collapsing urgent from any band, cooldown or not. It was computed on every
frame of session 06af3214 and logged on every frame, and tick() was never given
it -- so it triggered nothing at all. 41 frames of that drive had TTC under
2.5 s. It is a parameter now.

DEGRADED IS WIDER, NOT QUIETER. A speed that is coasted rather than fresh, or
one whose two sources disagree, moves both band edges OUT by
DEGRADED_TAU_BIAS_S and relaxes the confidence floor. The old code had one
state for "less sure" and it was silence. A system that stops warning because
the sky went quiet under a bridge teaches the driver that its silence means
nothing.
"""
import math

import config

from . import state as v2

# ===========================================================================
# THE TUNABLES ARE IN config.py NOW.
#
# They lived here, in a PROVISIONAL block, until the first real drive. The
# punch list moved them, and the reason is the one it gives: a threshold that
# decides whether a car speaks has to be readable, diffable and settable in one
# place, next to the navigation timing and the frame transport tuning that were
# already there. This module holds NO numbers of its own -- every name below is
# a binding, and headway/live_selftest.py asserts that each one still points at
# the config value it names.
#
# The v2 header's warning travels with them and is repeated in config.py:
# PROVISIONAL prototype values, engineering starting points for shadow-mode
# tuning, NOT validated safety thresholds.
# ===========================================================================

# --- band entry thresholds (v3), in TIME HEADWAY: gap / own speed.
TAU_ENTER_GETTING_UNSAFE = config.HEADWAY_TAU_GETTING_UNSAFE_S
TAU_ENTER_UNSAFE = config.HEADWAY_TAU_UNSAFE_S

# --- exit hysteresis: leaving a worse band needs tau > entry + HYST.
HYST_S = config.HEADWAY_TAU_HYSTERESIS_S

# --- temporal confirmation.
#
# TIME-based, not frame-based, because the live loop's rate is now a control
# variable between 8 and 15 fps (see config.HEADWAY_WS_*) and a frame count
# would mean a different latency at every one of them. v2 §3's "3 frames @
# 12 Hz" IS 0.25 s; this is the same idea expressed in the unit that actually
# matters. The floor of 2 frames is v2 §3's "never fire from one frame".
CONFIRM_S = config.HEADWAY_CONFIRM_S
# Tolerance on the time term only: frame arrival jitters either side of the
# nominal cadence, and without this a loop whose second frame lands a hair
# early would be pushed to a third and double the red tier's latency for no
# gain in certainty.
CONFIRM_TOL_S = config.HEADWAY_CONFIRM_TOLERANCE_S
MIN_CONFIRM_FRAMES = config.HEADWAY_MIN_CONFIRM_FRAMES
# De-escalating is deliberately slower than escalating (v2 §3: 12 frames down
# vs 3 up). Dropping a warning band early is how flicker gets back in.
CONFIRM_DOWN_S = config.HEADWAY_CONFIRM_DOWN_S

# --- voice cooldowns. Gate VOICE ONLY -- band, display and log always update.
COOLDOWN_CALM_S = config.HEADWAY_COOLDOWN_CALM_S
COOLDOWN_UNSAFE_S = config.HEADWAY_COOLDOWN_UNSAFE_S
# Continuous NORMAL for this long is a *genuine clear*: the hazard resolved
# rather than hovering at a boundary. It re-arms every cooldown, because danger
# that clears and returns deserves a fresh warning rather than a silent one.
GENUINE_CLEAR_S = config.HEADWAY_GENUINE_CLEAR_S
# Second, firmer line while still in GETTING_UNSAFE with the gap closing.
ESCALATE_AFTER_S = config.HEADWAY_ESCALATE_AFTER_S

# --- speed validity
V_MIN_COACH = config.HEADWAY_MIN_COACH_SPEED_MS
V_HOST_STALE_S = config.HEADWAY_V_HOST_STALE_S

# --- the physical floor on a time headway. See config for the drive that
# produced it: 706 frames of a lead under 6 m at over 10 m/s, and fifteen of
# the twenty-two warnings that were actually spoken.
TAU_IMPLAUSIBLE_S = config.HEADWAY_TAU_IMPLAUSIBLE_S

# --- TTC: the urgent trigger, from any band (v2 §1/§9).
TTC_URGENT_S = config.HEADWAY_TTC_URGENT_S

# --- degraded speed: wider, never quieter.
DEGRADED_TAU_BIAS_S = config.HEADWAY_DEGRADED_TAU_BIAS_S
DEGRADED_CONF_RELIEF = config.HEADWAY_DEGRADED_CONF_RELIEF

# --- how long a band entry stays actionable after the gates that suppressed it
# clear. A band entry is one-shot, so a transient gate landing on exactly that
# frame must defer it rather than delete it -- otherwise the whole occupancy is
# silent no matter how long it lasts or how bad it gets. Bounded because an
# entry the system could not judge for seconds is no longer news; the band is
# still displayed throughout either way.
PENDING_ENTRY_MAX_S = config.HEADWAY_PENDING_ENTRY_MAX_S

# --- coaching warm-up after any anchor reset / filter re-init (v2's
# COACH_WARMUP_S, unchanged). The Kalman seeds d_dot at 0 with sd 5 m/s and
# takes ~0.57 s to converge; coaching off an unconverged velocity is coaching
# off noise.
COACH_WARMUP_S = v2.COACH_WARMUP_S       # 0.6

# --- confidence floor (v2 §8). Below this: suppress voice. Never warn louder
# on worse data.
CONF_FLOOR = v2.CONF_FLOOR               # 0.4


def coast_budget_s(v_host) -> float:
    """How long a lost lead may be coasted, at THIS speed.

    The old rule was a flat 1.0 s, which is 30 m of road at 30 m/s and 5 m at
    5 m/s -- one number meaning two entirely different things. What actually
    stays true across the range is a DISTANCE: about eighteen metres of road is
    as far as a vehicle may be extrapolated before the extrapolation stops
    being about anything. So the budget is that distance divided by the speed
    of the moment, clamped at both ends so it never becomes absurd.

        30 m/s   0.6 s   (the floor: at freeway speed there is no slack)
        18 m/s   1.0 s   (the old constant, which was tuned around here)
         7 m/s   2.5 s   (the ceiling)

    No speed at all falls back to the v2 constant, which is the behaviour that
    existed before this function.
    """
    try:
        v = float(v_host)
    except (TypeError, ValueError):
        return v2.MAX_COAST_S
    if not math.isfinite(v) or v <= 0.1:
        return config.HEADWAY_COAST_MAX_S
    return max(config.HEADWAY_COAST_MIN_S,
               min(config.HEADWAY_COAST_MAX_S,
                   config.HEADWAY_COAST_DISTANCE_M / v))


# --- bands ---
NORMAL = "NORMAL"
GETTING_UNSAFE = "GETTING_UNSAFE"
UNSAFE = "UNSAFE"
SUPPRESSED = "SUPPRESSED"
UNKNOWN = "UNKNOWN"
# A range the geometry says cannot be a following distance. Its own band rather
# than a flavour of UNKNOWN, because the two mean opposite things to whoever is
# reading the HUD: UNKNOWN is "no speed, so no judgement", and this is "a speed
# and a range that do not go together, so the range is wrong".
IMPLAUSIBLE = "IMPLAUSIBLE"

TAU_BANDS = (NORMAL, GETTING_UNSAFE, UNSAFE)
ALL_BANDS = TAU_BANDS + (SUPPRESSED, UNKNOWN, IMPLAUSIBLE)

# Severity ordering for "is this an escalation?". SUPPRESSED/UNKNOWN sit below
# NORMAL: leaving either one into a warning band is an escalation and takes the
# fast confirmation.
_SEV = {UNKNOWN: -1, SUPPRESSED: -1, IMPLAUSIBLE: -1,
        NORMAL: 0, GETTING_UNSAFE: 1, UNSAFE: 2}

# --- trend, reported to the client in the spec's three-value vocabulary ---
OPENING = "opening"
STABLE = "stable"
CLOSING = "closing"

# --- voice lines ---
LINE_CALM = "calm"
LINE_ESCALATE = "escalate"
LINE_TOO_CLOSE = "too_close"
LINE_WATCH_DISTANCE = "watch_distance"
LINE_BACK_OFF = "back_off"

# The words are fixed here, in the deterministic layer. The bible governs how
# they are *delivered*; it never gets to choose whether or what to warn.
LINE_TEXT = {
    LINE_CALM: "Beep beep — you're getting a little close there.",
    LINE_ESCALATE: "Still closing — ease off a touch.",
    LINE_TOO_CLOSE: "You're too close.",
    LINE_WATCH_DISTANCE: "Watch your distance.",
    LINE_BACK_OFF: "Back off — now.",
}

# Red tier plays pre-rendered clips: an ElevenLabs round-trip is 300-800 ms and
# the whole point of the tier is that it is already too late to wait. The calm
# tier may use live TTS -- it is coaching, not an alert.
LINE_AUDIO = {
    LINE_CALM: "tts",
    LINE_ESCALATE: "tts",
    LINE_TOO_CLOSE: LINE_TOO_CLOSE,
    LINE_WATCH_DISTANCE: LINE_WATCH_DISTANCE,
    LINE_BACK_OFF: LINE_BACK_OFF,
}

# Cooldowns are per TIER, not per line. The spec says "same line not repeated
# within 30 s / 15 s"; applying that per line would let "You're too close." and
# "Watch your distance." fire back to back and satisfy it on a technicality,
# which is exactly the nagging the rule exists to prevent. Per tier is strictly
# stronger and implies the stated rule.
TIER_CALM = "calm"
TIER_UNSAFE = "unsafe"
TIER_COOLDOWN = {TIER_CALM: COOLDOWN_CALM_S, TIER_UNSAFE: COOLDOWN_UNSAFE_S}
LINE_TIER = {
    LINE_CALM: TIER_CALM,
    LINE_ESCALATE: TIER_CALM,
    LINE_TOO_CLOSE: TIER_UNSAFE,
    LINE_WATCH_DISTANCE: TIER_UNSAFE,
    LINE_BACK_OFF: TIER_UNSAFE,
}

# Reasons recorded on every voice decision, spoken or not. These are the tuning
# data: a silence with a reason is as informative as an utterance.
R_BAND_ENTRY = "band_entry"
R_WORSENING = "worsening"
R_ESCALATION = "escalation"
R_COOLDOWN = "suppressed_by_cooldown"
R_CONFIDENCE = "suppressed_by_confidence"
R_WARMUP = "suppressed_by_warmup"
R_LOW_SPEED = "suppressed_by_low_speed"
R_NO_SPEED = "suppressed_by_unknown_speed"
R_NEW_LEAD = "suppressed_by_new_lead"
R_SILENT = "silent"
# A tau below the physical floor. Recorded separately from every other
# suppression because it is the one that says the MEASUREMENT is wrong rather
# than that the situation is fine -- and because counting it is how the
# upstream fix (a portrait phone's hood filling the lower half of the frame)
# gets tracked from one drive to the next.
R_IMPLAUSIBLE = "suppressed_by_implausible_headway"
# TTC under the urgent threshold with the gap collapsing. v2 §1/§9.
R_TTC_URGENT = "ttc_urgent"


def classify_tau(tau, prev_band=None, bias_s: float = 0.0) -> str:
    """v3 band from tau, with v2 §2's exit hysteresis.

    Entry uses the bare threshold; exit requires tau > threshold + HYST when the
    previous band was this bad or worse.

    `bias_s` pushes both edges OUT -- a wider band, entered earlier -- and is
    the only thing a degraded speed does to the ladder. It never pushes them
    in, and there is no path by which uncertainty makes RIO quieter.
    """
    if tau is None or not math.isfinite(tau):
        # No lead, or no usable range: nothing to coach about. NORMAL is the
        # silent posture, which is the right default for "no target".
        return NORMAL

    prev = _SEV.get(prev_band, -1)
    bias = max(0.0, float(bias_s or 0.0))
    edge = TAU_ENTER_UNSAFE + bias + (HYST_S if prev >= _SEV[UNSAFE] else 0.0)
    if tau < edge:
        return UNSAFE
    edge = (TAU_ENTER_GETTING_UNSAFE + bias
            + (HYST_S if prev >= _SEV[GETTING_UNSAFE] else 0.0))
    if tau < edge:
        return GETTING_UNSAFE
    return NORMAL


def trend_word(v2_trend: str) -> str:
    """v2's four trend classes -> the client's three-value vocabulary."""
    if v2_trend == v2.INCREASING:
        return OPENING
    if v2_trend in v2.SHRINKING:
        return CLOSING
    return STABLE


def urgency_of(band: str, v2_trend: str) -> int:
    """0-3, for the display. 3 is reserved for UNSAFE with the gap collapsing."""
    if band == UNSAFE:
        return 3 if v2_trend == v2.RAPIDLY_SHRINKING else 2
    if band == GETTING_UNSAFE:
        return 1
    return 0


class LivePolicy:
    """Per-session band + voice decision. Pure: no I/O, no model, no clock.

    `t` is supplied by the caller on every tick (seconds, monotonic within a
    session) so the whole policy is replayable from a log -- which is what
    live_selftest.py does.
    """

    def __init__(self):
        self.band = NORMAL
        self.prev_band = NORMAL
        self.band_entered_t = 0.0

        self._cand = None
        self._cand_t0 = 0.0
        self._cand_n = 0

        self._tier_last = {}          # tier -> t of last utterance
        self._clear_t = -1e9          # t of the most recent genuine clear
        self._normal_since = None
        self._escalated = False       # calm-tier second line, once per occupancy
        self._back_off_fired = False  # sharper red line, once per occupancy
        # The TTC trigger, once per band occupancy. Reset in _on_band_change
        # alongside the other one-shots, and on a genuine clear -- a collision
        # course that resolved and came back is a new event, not a repeat.
        self._ttc_fired = False
        self._unsafe_variant = 0      # alternates too_close / watch_distance
        self._pending = None          # band entry awaiting a gate to clear

        self.voice_log = []
        self.transitions = []

    # -- main entry ---------------------------------------------------------
    def tick(self, tau, v2_trend, confidence, v_host, v_host_stale,
             t, since_reset_s, new_lead=False, track_lost=False,
             ttc=None, speed_degraded=False, speed_source=None) -> dict:
        """One frame. Returns the band/voice decision as a plain dict.

        `ttc` and `speed_degraded` are the two parameters the first real drive
        showed were missing. TTC was computed and logged on all 1949 of its
        frames and reached no decision; `speed_degraded` did not exist, because
        the only thing an uncertain speed could do was silence everything.
        """
        entered = False
        bias = DEGRADED_TAU_BIAS_S if speed_degraded else 0.0
        candidate = self._raw_band(tau, v_host, v_host_stale, bias)

        if candidate in (SUPPRESSED, UNKNOWN, IMPLAUSIBLE):
            # Entered immediately, never confirmed. Delaying an "I cannot
            # classify this" verdict would mean coaching from data already
            # judged unusable -- v2 §3 treats DEGRADED/LOST the same way.
            if candidate != self.band:
                entered = True
                self.prev_band = self.band
                self.band = candidate
                self.band_entered_t = t
                self._on_band_change()
            self._cand, self._cand_n = None, 0
        else:
            entered = self._confirm(candidate, t)

        self._track_clear(t)

        voice, reason, suppressed_line = self._decide_voice(
            entered, v2_trend, confidence, t, since_reset_s, new_lead,
            track_lost, ttc, speed_degraded)

        if entered:
            self.transitions.append({
                "t": round(float(t), 4), "from": self.prev_band, "to": self.band,
            })

        word = trend_word(v2_trend)
        return {
            "t": round(float(t), 4),
            "band": self.band,
            "band_entered": bool(entered),
            "prev_band": self.prev_band,
            "trend": word,
            "trend_detail": v2_trend,
            "urgency": urgency_of(self.band, v2_trend),
            "speak": voice,
            "voice_reason": reason,
            "voice_line": (voice or {}).get("line") or suppressed_line,
            "confirm": {"candidate": self._cand, "frames": self._cand_n,
                        "elapsed_s": (None if self._cand is None
                                      else round(t - self._cand_t0, 3))},
            # Which speed decided this frame, and whether the bands were
            # widened for it. Every band above rests on the denominator of
            # tau, so the log has to say where the denominator came from.
            "speed_source": speed_source,
            "speed_degraded": bool(speed_degraded),
            "tau_bias_s": bias,
            "cooldowns": {k: round(t - v, 2) for k, v in self._tier_last.items()},
        }

    # -- band ---------------------------------------------------------------
    def _raw_band(self, tau, v_host, v_host_stale, bias_s: float = 0.0) -> str:
        """Ordering matters: speed validity gates everything above it.

        Never classify on a speed we do not have (spec §4) and never coach at
        creep speed (v2 §7), because tau = d/v explodes as v -> 0 and a 4 m gap
        in a parking queue reads as "getting unsafe".

        And then, above the speed floor, never classify on a time headway that
        is not one. See TAU_IMPLAUSIBLE_S and config.py for the drive that
        produced it: a full-frame-width box across the bottom of a portrait
        phone's picture, ranged at 3.7 m, held as the lead at 20 m/s for 706
        frames. The check is deliberately BELOW the speed floor in this gate
        order -- at creep speed a genuinely tiny gap is ordinary, and a car
        park is exactly where 3 m behind something is the truth.
        """
        if v_host is None or v_host_stale or not math.isfinite(float(v_host)):
            return UNKNOWN
        if float(v_host) < V_MIN_COACH:
            return SUPPRESSED
        if (tau is not None and math.isfinite(tau)
                and tau < TAU_IMPLAUSIBLE_S):
            return IMPLAUSIBLE
        return classify_tau(tau, self.band, bias_s)

    def _confirm(self, candidate: str, t: float) -> bool:
        """Time-based confirmation with a 2-frame floor. True on a confirmed entry."""
        if candidate == self.band:
            self._cand, self._cand_n = None, 0
            return False

        if candidate != self._cand:
            self._cand = candidate
            self._cand_t0 = t
            self._cand_n = 1
        else:
            self._cand_n += 1

        escalating = _SEV[candidate] > _SEV.get(self.band, -1)
        required = CONFIRM_S if escalating else CONFIRM_DOWN_S
        elapsed = t - self._cand_t0

        if self._cand_n >= MIN_CONFIRM_FRAMES and elapsed >= required - CONFIRM_TOL_S:
            self.prev_band = self.band
            self.band = candidate
            self.band_entered_t = t
            self._cand, self._cand_n = None, 0
            self._on_band_change()
            return True
        return False

    def _on_band_change(self) -> None:
        """One-shot flags are per band *occupancy*, so they reset on every entry."""
        self._escalated = False
        self._back_off_fired = False
        self._ttc_fired = False
        # Any entry still waiting on a gate describes the band we just left.
        self._pending = None

    def _track_clear(self, t: float) -> None:
        """Record the moment a stretch of NORMAL becomes a genuine clear."""
        if self.band == NORMAL:
            if self._normal_since is None:
                self._normal_since = t
            elif (t - self._normal_since) >= GENUINE_CLEAR_S:
                self._clear_t = t
                # A genuine clear re-arms the urgent trigger for the same
                # reason it re-arms every cooldown: the hazard resolved, and
                # the next one is a new hazard.
                self._ttc_fired = False
        else:
            self._normal_since = None

    # -- voice --------------------------------------------------------------
    def _decide_voice(self, entered, v2_trend, confidence, t, since_reset_s,
                      new_lead, track_lost, ttc=None, speed_degraded=False):
        """Returns (speak|None, reason, suppressed_line|None).

        Gate order is deliberate: the states where we cannot judge come first,
        so a suppression is always attributed to the *first* reason that applies
        rather than whichever branch happens to run last.
        """
        # A band entry is a ONE-SHOT event, so a gate that suppresses it must
        # defer it, not delete it. Latching first is what makes that possible:
        # without it a transient gate on exactly the entry frame silences the
        # whole band occupancy, however long it lasts and however bad it gets.
        if entered:
            self._pending = {"band": self.band, "prev": self.prev_band, "t": t}

        if self.band == UNKNOWN:
            self._pending = None
            return None, R_NO_SPEED, None
        if self.band == SUPPRESSED:
            self._pending = None
            return None, R_LOW_SPEED, None
        if self.band == IMPLAUSIBLE:
            # The range is not believable at this speed, and TTC is derived
            # from the same range -- so the urgent branch below is not reached
            # either. Refusing both is the point: one bad number must not be
            # laundered into a warning by a second number computed from it.
            self._pending = None
            return None, R_IMPLAUSIBLE, None
        if new_lead or track_lost:
            return None, R_NEW_LEAD, None
        # DEGRADED RELAXES THIS FLOOR, it does not raise it. The alternative to
        # a slightly-less-certain warning is no warning at all, and that is the
        # trade the old code made every time a fix went quiet.
        floor = CONF_FLOOR - (DEGRADED_CONF_RELIEF if speed_degraded else 0.0)
        if confidence is None or confidence < floor:
            return None, R_CONFIDENCE, None

        # Warm-up neutralises the TREND, it does not silence the tier.
        #
        # This is v2's own device (state.py's `coach_trend = STABLE`), applied at
        # the granularity v3 actually needs. The warm-up exists because the
        # Kalman seeds d_dot at 0 with sd 5 m/s and reads its own settling
        # residual as a ~-0.4 m/s closure for the first ~0.57 s -- it is a
        # VELOCITY artefact. v2 could suppress the whole decision because v2's
        # coaching was trend-gated, so neutralising the trend and silencing the
        # tier were the same thing.
        #
        # v3's coaching is ENTRY-gated: it rests on tau, which is a position
        # measurement the reset does not corrupt (the filter re-seeds d to the
        # measured range directly), and which the 2-frame confirmation already
        # protects against a single bad sample. Blanket-suppressing here
        # silenced every cut-in outright: the reset lands ~0.5 s before the
        # confirmation completes, so at 2 fps the entry ALWAYS fell inside the
        # window. Neutralising the trend keeps the real protection -- no line
        # may be *chosen* from an unconverged d_dot, so a warm-up entry gets
        # "You're too close." rather than "Back off — now.", and neither the
        # escalation nor the in-band collapse branch can fire.
        warming = since_reset_s is not None and since_reset_s < COACH_WARMUP_S
        if warming:
            v2_trend = v2.STABLE

        rapid = (v2_trend == v2.RAPIDLY_SHRINKING)
        closing = (v2_trend in v2.SHRINKING)
        quiet = R_WARMUP if warming else R_SILENT

        # --- TTC: urgent from ANY band (v2 §1/§9) --------------------------
        #
        # Not a band, not confirmed, not cooled down. Time to contact under the
        # threshold WITH the gap collapsing is the one situation where the
        # ladder is beside the point: a 4 s headway closing at 12 m/s is
        # NORMAL by tau and two seconds from a collision, and the band machine
        # would have said nothing about it.
        #
        # Once per occupancy, cleared by a genuine clear like everything else,
        # so it is a trigger and not a siren.
        if (ttc is not None and math.isfinite(ttc) and ttc < TTC_URGENT_S
                and rapid and not self._ttc_fired):
            spoke = self._fire(LINE_BACK_OFF, t, force=True, reason=R_TTC_URGENT)
            if spoke[0] is not None:
                self._ttc_fired = True
                self._back_off_fired = True
                self._pending = None
            return spoke

        # Act on a fresh entry, or on a latched one whose band still stands.
        # `entry_prev` comes from the latch so the worsening test is applied to
        # the transition that actually happened, not to wherever we are now.
        entry_prev = self.prev_band
        if (self._pending and self._pending["band"] == self.band
                and (t - self._pending["t"]) <= PENDING_ENTRY_MAX_S):
            entered = True
            entry_prev = self._pending["prev"]
        elif self._pending and self._pending["band"] != self.band:
            self._pending = None

        # --- red tier ------------------------------------------------------
        if self.band == UNSAFE:
            if entered:
                self._pending = None
                # Any confirmed entry into UNSAFE from a real tau band speaks,
                # cooldown or not.
                #
                #   GETTING_UNSAFE -> UNSAFE   the driver was already told once
                #                              and it got worse anyway
                #   NORMAL         -> UNSAFE   a two-band jump: a cut-in
                #                              collapsing tau from 4 s to 1.5 s
                #                              is the clearest case there is for
                #                              speaking, and it is exactly the
                #                              one a cooldown would swallow
                #
                # SUPPRESSED and UNKNOWN are deliberately NOT worsening origins.
                # Neither carries any tau information, and both are entered
                # immediately with no confirmation, so a flaky GPS fix bouncing
                # UNKNOWN <-> UNSAFE could fire the red tier every second or so.
                # Coming out of them is a fresh classification, not a
                # deterioration, and it stays cooldown-gated.
                #
                # What remains protected against oscillation is the single-band
                # re-entry NORMAL <-> GETTING_UNSAFE, which is the flicker-prone
                # boundary and is gated by the 30 s calm cooldown below.
                worsening = _SEV.get(entry_prev, -1) >= _SEV[NORMAL]
                line = LINE_BACK_OFF if rapid else self._next_unsafe_line()
                if line == LINE_BACK_OFF:
                    # Set whether or not the cooldown lets this through. The
                    # flag means "back_off has been DECIDED for this occupancy",
                    # not "back_off was heard": the in-band branch below exists
                    # to catch a LATER deterioration, and must not become a
                    # one-frame retry that walks straight around a cooldown
                    # that just suppressed this very entry.
                    self._back_off_fired = True
                return self._fire(line, t, force=worsening,
                                  reason=R_WORSENING if worsening else R_BAND_ENTRY)

            # Already settled in UNSAFE and the gap NOW starts collapsing --
            # a deterioration inside the highest tier, which is the same
            # argument as the worsening rule, so it bypasses the cooldown. Once
            # per occupancy, and only when the entry itself was not already
            # rapid (that case is handled above).
            if rapid and not self._back_off_fired:
                spoke = self._fire(LINE_BACK_OFF, t, force=True, reason=R_WORSENING)
                if spoke[0] is not None:
                    self._back_off_fired = True
                return spoke
            return None, quiet, None

        # --- amber tier ----------------------------------------------------
        if self.band == GETTING_UNSAFE:
            # Entry from a *better* band only. Arriving here by de-escalating
            # out of UNSAFE is the situation improving, and congratulating the
            # driver with a warning is how the feature gets muted.
            if entered:
                self._pending = None
                if _SEV.get(entry_prev, -1) < _SEV[GETTING_UNSAFE]:
                    return self._fire(LINE_CALM, t, reason=R_BAND_ENTRY)

            if (closing and not self._escalated
                    and (t - self.band_entered_t) >= ESCALATE_AFTER_S):
                # force=True is load-bearing. The escalation shares the calm
                # tier's cooldown with the entry line that fired 5 s ago, so
                # without the bypass the 30 s cooldown would swallow the second
                # line every single time and the escalation could never happen.
                # It cannot nag: it is one-shot per band occupancy, and an
                # occupancy costs 5 s of continuous amber with the gap closing.
                spoke = self._fire(LINE_ESCALATE, t, force=True, reason=R_ESCALATION)
                if spoke[0] is not None:
                    self._escalated = True
                return spoke
            return None, quiet, None

        return None, quiet, None

    def _next_unsafe_line(self) -> str:
        """Alternate the two red-tier phrasings so a repeat is not word-for-word."""
        line = (LINE_TOO_CLOSE, LINE_WATCH_DISTANCE)[self._unsafe_variant % 2]
        self._unsafe_variant += 1
        return line

    def _fire(self, line: str, t: float, force: bool = False, reason: str = R_BAND_ENTRY):
        tier = LINE_TIER[line]
        last = self._tier_last.get(tier)
        cooldown = TIER_COOLDOWN[tier]

        if not force and last is not None and (t - last) < cooldown:
            # A genuine clear after the last utterance re-arms the tier: the
            # hazard resolved and came back, which is a new event, not a repeat.
            if last >= self._clear_t:
                self.voice_log.append({
                    "t": round(float(t), 4), "line": line, "band": self.band,
                    "spoken": False, "reason": R_COOLDOWN,
                    "remaining_s": round(cooldown - (t - last), 2),
                })
                return None, R_COOLDOWN, line

        self._tier_last[tier] = t
        speak = {
            "line": line,
            "text": LINE_TEXT[line],
            "audio": LINE_AUDIO[line],
            "tier": tier,
            "reason": reason,
        }
        self.voice_log.append({
            "t": round(float(t), 4), "line": line, "band": self.band,
            "spoken": True, "reason": reason, "audio": LINE_AUDIO[line],
        })
        return speak, reason, line
