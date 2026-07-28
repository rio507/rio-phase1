"""τ / TTC / trend maths and the warning-state machine.

Design ref: headway_design.md §3 (maths), §4 (confirmation + confidence),
§6 (state machine pseudocode -- implemented literally).

This module is the safety core and is deliberately pure: no I/O, no model calls,
no Qwen. `tick()` is a function of filtered numbers only, which is what §1's LLM
firewall requires -- Qwen can influence *which* object is measured, never
*whether* a warning fires.
"""
import math
from dataclasses import dataclass, field, replace

# ---------------------------------------------------------------------------
# States (§6)
# ---------------------------------------------------------------------------
COMFORTABLE = "COMFORTABLE"
MONITOR = "MONITOR"
CLOSE = "CLOSE"
WARNING = "WARNING"
URGENT = "URGENT"
LOST = "LOST"
DEGRADED = "DEGRADED"

STATES = (COMFORTABLE, MONITOR, CLOSE, WARNING, URGENT, LOST, DEGRADED)

# Severity ladder for the escalate/de-escalate asymmetry. LOST and DEGRADED are
# not on the ladder -- they are sensor verdicts, not risk levels.
_SEVERITY = {COMFORTABLE: 0, MONITOR: 1, CLOSE: 2, WARNING: 3, URGENT: 4}

# ---------------------------------------------------------------------------
# Trend classes (§3)
# ---------------------------------------------------------------------------
INCREASING = "INCREASING"
STABLE = "STABLE"
SLOWLY_SHRINKING = "SLOWLY_SHRINKING"
RAPIDLY_SHRINKING = "RAPIDLY_SHRINKING"

TREND_ORDER = (RAPIDLY_SHRINKING, SLOWLY_SHRINKING, STABLE, INCREASING)

RAPID_SHRINK_MS = 1.5        # §3: ḋ ≤ −1.5 m/s is rapid
DEAD_BAND_LOW_MS = 0.3       # §3: |ḋ| ≤ 0.3 is STABLE at low speed
DEAD_BAND_HIGH_MS = 0.5      # §3: widen to ±0.5 at highway range
DEAD_BAND_V_LOW = 15.0       # m/s -- below this use the low dead-band
DEAD_BAND_V_HIGH = 25.0      # m/s -- at/above this use the high one
# §3 asks for hysteresis so classes don't flicker. Sized against the measured
# ḋ jitter (~0.29 m/s 1σ, see filter.py): a third of that stops boundary
# chatter without making a genuine trend change feel sluggish.
TREND_HYSTERESIS_MS = 0.10

# ---------------------------------------------------------------------------
# Base thresholds (§6) -- soft bands, modulated by conditions
# ---------------------------------------------------------------------------
TAU_COMFORT = 3.0     # s
TAU_MONITOR = 2.0
TAU_CLOSE = 1.5
TTC_WARNING = 4.0
TTC_URGENT = 2.5

# ---------------------------------------------------------------------------
# Confirmation frame counts (§4 + §6)
# ---------------------------------------------------------------------------
ESCALATE_FRAMES = 3          # §4: ≈0.25 s at 12 Hz. Fast.
DEESCALATE_FRAMES = 12       # §4: ≈1 s. Slow.
URGENT_BYPASS_FRAMES = 2     # §4: only under high confidence. Never 1.
URGENT_STRICT_FRAMES = 3     # when the §4 bypass preconditions are not met

# §4 bypass preconditions
URGENT_BYPASS_TTC = 2.0
URGENT_BYPASS_DEPTH_CONF = 0.7
URGENT_BYPASS_TRACK_QUALITY = 0.7

# ---------------------------------------------------------------------------
# Confidence + gating (§4, §5)
# ---------------------------------------------------------------------------
CONFIDENCE_FLOOR = 0.4       # §4: below this -> DEGRADED, and say nothing
MAX_COAST_S = 1.0            # §5: never warn from coasted data older than this
ANCHOR_FRESH_S = 5.0         # §7.4: re-anchor at least every 5 s
ANCHOR_STALE_S = 10.0        # fully stale -> heavy confidence penalty
TAU_MIN_V_HOST = 0.5         # §3: guard div-by-zero
TAU_SUPPRESS_V_HOST = 2.0    # §3: below ~2 m/s headway is meaningless
TTC_MIN_CLOSING_MS = 0.2     # §3: below this closing speed, TTC is ∞

# §5: after a NEW_LEAD reset, warnings must re-confirm before they may fire.
NEW_LEAD_HOLD_S = 0.3

# Speed-source trust (§2: GPS lags ~1 s and dies in tunnels)
V_HOST_SOURCE_CONF = {"CAN": 1.0, "OBD": 0.95, "GPS": 0.85, "FIXED": 0.9}

# §6 on_transition: no repeat non-urgent line within 30 s
NON_URGENT_RATE_LIMIT_S = 30.0


@dataclass
class Thresholds:
    tau_comfort: float = TAU_COMFORT
    tau_monitor: float = TAU_MONITOR
    tau_close: float = TAU_CLOSE
    ttc_warning: float = TTC_WARNING
    ttc_urgent: float = TTC_URGENT

    def scale(self, k: float) -> "Thresholds":
        return Thresholds(
            tau_comfort=self.tau_comfort * k,
            tau_monitor=self.tau_monitor * k,
            tau_close=self.tau_close * k,
            ttc_warning=self.ttc_warning * k,
            ttc_urgent=self.ttc_urgent * k,
        )


@dataclass
class Context:
    """Condition flags feeding modulate().

    STAGE 0 STUBS: `rain` and `night` have no detector yet -- §2 lists wiper and
    light state as optional inputs, and §7.5 wants them driving DEGRADED. They
    stay False here and are wired up in a later stage; the modulation path around
    them is fully implemented and tested so that wiring is a one-line change.
    """
    v_host: float = 0.0
    rain: bool = False           # STUB
    night: bool = False          # STUB
    v_host_source: str = "FIXED"
    yaw_rate: float = None       # STUB -- no IMU in Stage 0 (§5.2, §7.3)


@dataclass
class Measurement:
    """Everything tick() is allowed to look at."""
    d: float = None
    d_dot: float = None
    tau: float = float("inf")
    ttc: float = float("inf")
    trend: str = STABLE
    confidence: float = 0.0
    depth_conf: float = 0.0
    track_quality: float = 0.0
    track_lost: bool = False
    coast_age: float = 0.0
    new_lead: bool = False
    v_host_stale: bool = False
    anchor_age_s: float = 0.0
    ctx: Context = field(default_factory=Context)


def modulate(base: Thresholds, ctx: Context) -> Thresholds:
    """§6 modulate(), implemented in the spec's order.

    The 1.25 scale applies to everything first; the highway TAU_CLOSE override is
    then absolute, not scaled -- that ordering is the spec's and it matters, since
    in rain at 26 m/s the override *tightens* tau_close from 1.875 back to 1.8.
    """
    t = base
    if ctx.rain or ctx.night:
        t = t.scale(1.25)
    if ctx.v_host > 25:
        t = replace(t, tau_close=1.8)
    return t


def dead_band(v_host: float) -> float:
    """§3: ±0.3 m/s dead-band, widening to ±0.5 at highway range.

    Ramped rather than stepped -- a step would make the trend class jump as the
    host crosses one speed, which is exactly the flicker §3 is trying to avoid.
    """
    if v_host <= DEAD_BAND_V_LOW:
        return DEAD_BAND_LOW_MS
    if v_host >= DEAD_BAND_V_HIGH:
        return DEAD_BAND_HIGH_MS
    frac = (v_host - DEAD_BAND_V_LOW) / (DEAD_BAND_V_HIGH - DEAD_BAND_V_LOW)
    return DEAD_BAND_LOW_MS + frac * (DEAD_BAND_HIGH_MS - DEAD_BAND_LOW_MS)


def classify_trend(d_dot, v_host: float, current: str = None) -> str:
    """§3 gap-trend classes with hysteresis on every boundary."""
    if d_dot is None or not math.isfinite(d_dot):
        return current or STABLE

    band = dead_band(v_host)
    # Ascending in ḋ: RAPIDLY | SLOWLY | STABLE | INCREASING
    edges = [-RAPID_SHRINK_MS, -band, band]

    def index_with(shift: float) -> int:
        return sum(1 for e in edges if d_dot > e + shift)

    idx = index_with(0.0)
    cand = TREND_ORDER[idx]
    if current is None or cand == current or current not in TREND_ORDER:
        return cand

    # Require an overshoot past the boundary in whichever direction we're moving,
    # so a value sitting exactly on an edge stays where it is.
    cur_idx = TREND_ORDER.index(current)
    if idx > cur_idx:
        idx = index_with(+TREND_HYSTERESIS_MS)
    elif idx < cur_idx:
        idx = index_with(-TREND_HYSTERESIS_MS)
    return TREND_ORDER[idx]


def compute_tau(d, v_host):
    """§3: τ = d / max(v_host, 0.5); meaningless below ~2 m/s -> suppress."""
    if d is None or not math.isfinite(d):
        return float("inf")
    if v_host < TAU_SUPPRESS_V_HOST:
        # Not a warning-worthy regime: at walking pace a 3 m gap is normal.
        return float("inf")
    return d / max(v_host, TAU_MIN_V_HOST)


def compute_ttc(d, d_dot):
    """§3: TTC = d / max(−ḋ, ε), ∞ unless genuinely closing.

    This is the number §0 Challenge 2 calls the most robust: a constant depth
    scale bias multiplies d and ḋ alike and cancels in the ratio.
    """
    if d is None or d_dot is None or not math.isfinite(d) or not math.isfinite(d_dot):
        return float("inf")
    v_close = -d_dot
    if v_close <= TTC_MIN_CLOSING_MS:
        return float("inf")
    return d / v_close


def compute_confidence(depth_conf, track_quality, anchor_age_s, coast_age,
                       ctx: Context) -> float:
    """§4 confidence, 0-1. Multiplicative: any one bad input suppresses.

    Product, not average, on purpose -- §4's rule is that low confidence must
    *suppress* warnings rather than fire cautious ones, and an average would let
    a strong depth score paper over a tracker that has clearly lost the vehicle.
    """
    depth_term = max(0.0, min(1.0, float(depth_conf)))
    track_term = max(0.0, min(1.0, float(track_quality)))

    # §7.4 wants a re-anchor at least every 5 s; trust ramps down after that
    # rather than falling off a cliff.
    if anchor_age_s <= ANCHOR_FRESH_S:
        anchor_term = 1.0
    elif anchor_age_s >= ANCHOR_STALE_S:
        anchor_term = 0.3
    else:
        span = ANCHOR_STALE_S - ANCHOR_FRESH_S
        anchor_term = 1.0 - 0.7 * (anchor_age_s - ANCHOR_FRESH_S) / span

    # §5: coasting through an occlusion decays confidence to zero by MAX_COAST_S.
    coast_term = max(0.0, 1.0 - float(coast_age) / MAX_COAST_S)

    cond_term = 1.0
    if ctx.rain:
        cond_term *= 0.75
    if ctx.night:
        cond_term *= 0.8

    source_term = V_HOST_SOURCE_CONF.get(ctx.v_host_source, 0.8)

    return float(max(0.0, min(1.0, depth_term * track_term * anchor_term
                              * coast_term * cond_term * source_term)))


class WarningStateMachine:
    """§6 tick() + asymmetric confirmation + transition side effects.

    Confirmation model: tick() names a *candidate* state and how many consecutive
    frames it needs. A candidate only becomes committed once it has held for that
    many frames; changing candidate restarts the count. DEGRADED and LOST bypass
    confirmation entirely -- they are the spec's early returns, and delaying a
    "I can't see" verdict would mean warning from data we've already judged bad.
    """

    def __init__(self, base: Thresholds = None, rate_hz: float = 15.0,
                 audio_sink=None, shadow_mode: bool = True):
        self.base = base or Thresholds()
        self.rate_hz = float(rate_hz)
        self.state = COMFORTABLE
        self.shadow_mode = bool(shadow_mode)
        self.audio_sink = audio_sink

        self._candidate = None
        self._candidate_count = 0
        self._required = 0
        self._new_lead_hold_s = 0.0
        self._last_non_urgent_audio_t = {}   # line key -> last play time
        self.transitions = []

    # -- main entry ----------------------------------------------------------
    def tick(self, m: Measurement, t: float = 0.0, dt: float = None) -> dict:
        """Advance one frame. Returns a dict describing this tick."""
        dt = (1.0 / self.rate_hz) if dt is None else float(dt)

        if m.new_lead:
            # §5: a cut-in / new lead resets the filter, so every confirmation
            # counter it fed is stale too. Warnings must earn their way back.
            self._candidate, self._candidate_count, self._required = None, 0, 0
            self._new_lead_hold_s = NEW_LEAD_HOLD_S
        elif self._new_lead_hold_s > 0:
            self._new_lead_hold_s = max(0.0, self._new_lead_hold_s - dt)

        thresholds = modulate(self.base, m.ctx)
        candidate, frames, reason = self._evaluate(m, thresholds)

        if candidate in (DEGRADED, LOST):
            committed = self._commit(candidate, m, t, reason, immediate=True)
        else:
            committed = self._confirm(candidate, frames, m, t, reason)

        return {
            "t": round(float(t), 4),
            "state": self.state,
            "candidate": candidate,
            "candidate_count": self._candidate_count,
            "required_frames": self._required,
            "committed": committed,
            "reason": reason,
            "thresholds": {
                "tau_comfort": round(thresholds.tau_comfort, 3),
                "tau_monitor": round(thresholds.tau_monitor, 3),
                "tau_close": round(thresholds.tau_close, 3),
                "ttc_urgent": round(thresholds.ttc_urgent, 3),
                "ttc_warning": round(thresholds.ttc_warning, 3),
            },
            "new_lead_hold_s": round(self._new_lead_hold_s, 3),
        }

    # -- §6 tick() body ------------------------------------------------------
    def _evaluate(self, m: Measurement, t: Thresholds):
        """Literal transcription of the spec's tick(), in the spec's order.

        Order is load-bearing: DEGRADED before LOST before any risk state, so a
        sensor we don't trust can never produce a warning; and URGENT (TTC) is
        tested before WARNING (τ) because §0 Challenge 2 makes TTC the number to
        act on and τ the number to coach on.
        """
        if m.confidence < CONFIDENCE_FLOOR or m.v_host_stale:
            return DEGRADED, 0, "confidence_below_floor" if not m.v_host_stale else "v_host_stale"

        if m.track_lost and m.coast_age > MAX_COAST_S:
            return LOST, 0, "track_lost_coast_expired"

        warnings_held = self._new_lead_hold_s > 0

        if m.ttc < t.ttc_urgent and m.trend == RAPIDLY_SHRINKING:
            if warnings_held:
                return CLOSE, ESCALATE_FRAMES, "urgent_held_new_lead"
            return URGENT, self._urgent_frames(m), "ttc_below_urgent"

        if m.tau < t.tau_close and m.trend in (SLOWLY_SHRINKING, RAPIDLY_SHRINKING):
            if warnings_held:
                return CLOSE, ESCALATE_FRAMES, "warning_held_new_lead"
            return WARNING, ESCALATE_FRAMES, "tau_below_close_and_shrinking"

        if m.tau < t.tau_monitor:
            # 6 frames when the gap is opening: no need to react quickly to a
            # situation that is already improving.
            frames = ESCALATE_FRAMES if m.trend != INCREASING else 6
            return CLOSE, frames, "tau_below_monitor"

        if m.tau < t.tau_comfort:
            return MONITOR, 6, "tau_below_comfort"

        return COMFORTABLE, DEESCALATE_FRAMES, "tau_comfortable"

    def _urgent_frames(self, m: Measurement) -> int:
        """§4 URGENT bypass: 2 frames only under high confidence, never 1.

        §6 writes confirm(URGENT, frames=2) unconditionally while §4 attaches
        preconditions to that 2. Implemented as §4's stricter reading -- the
        bypass is a concession to urgency and should not apply when the inputs
        justifying it are weak. Falls back to the normal 3-frame escalation.
        """
        if (m.ttc < URGENT_BYPASS_TTC
                and m.depth_conf >= URGENT_BYPASS_DEPTH_CONF
                and m.track_quality >= URGENT_BYPASS_TRACK_QUALITY):
            return URGENT_BYPASS_FRAMES
        return URGENT_STRICT_FRAMES

    # -- confirmation --------------------------------------------------------
    def _confirm(self, candidate, frames, m, t, reason) -> bool:
        if candidate == self.state:
            self._candidate, self._candidate_count, self._required = None, 0, 0
            return False

        # §4: de-escalation is slow. Whatever tick() asked for, stepping *down*
        # the ladder needs 12 frames -- warnings should outlast the moment.
        required = int(frames)
        cur_sev = _SEVERITY.get(self.state)
        new_sev = _SEVERITY.get(candidate)
        if cur_sev is not None and new_sev is not None and new_sev < cur_sev:
            required = max(required, DEESCALATE_FRAMES)

        if candidate != self._candidate:
            self._candidate = candidate
            self._candidate_count = 1
            self._required = required
        else:
            self._candidate_count += 1
            self._required = required

        if self._candidate_count >= self._required:
            return self._commit(candidate, m, t, reason)
        return False

    def _commit(self, new_state, m, t, reason, immediate=False) -> bool:
        if new_state == self.state:
            self._candidate, self._candidate_count, self._required = None, 0, 0
            return False

        old = self.state
        self.state = new_state
        self._candidate, self._candidate_count, self._required = None, 0, 0
        self.on_transition(old, new_state, m, t, reason, immediate)
        return True

    # -- §6 on_transition ----------------------------------------------------
    def on_transition(self, old, new, m, t, reason, immediate=False) -> None:
        action = None
        if new == URGENT:
            # Pre-rendered and local (§0 Challenge 3): a TTS round trip here
            # would be a crash narration, not a warning.
            action = {"kind": "play_local", "clip": "urgent.wav", "interrupts": True}
        elif new == WARNING and old not in (WARNING, URGENT):
            action = {"kind": "play_local", "clip": "warning.wav", "interrupts": False}
        elif new == CLOSE and old in (COMFORTABLE, MONITOR):
            action = {"kind": "rio_speak",
                      "text": "Beep beep — you're getting a little close there."}

        # §6 rate-limit: "no repeat non-urgent line within 30 s". URGENT is exempt.
        #
        # Keyed PER LINE, not globally. A global timer looked right until the
        # synthetic clip ran it: a CLOSE line at t=5.6 s muted the WARNING line
        # at t=7.0 s, i.e. the rate limiter silenced an *escalation*. Since the
        # spec says "no repeat ... line", keying on the line itself both reads
        # more literally and keeps every escalation audible while still stopping
        # the same line from nagging.
        if action is not None and new != URGENT:
            key = action.get("clip") or action.get("text") or action["kind"]
            last = self._last_non_urgent_audio_t.get(key, -1e9)
            if (t - last) < NON_URGENT_RATE_LIMIT_S:
                action = {"kind": "suppressed_rate_limit", "would_be": action, "key": key}
            else:
                self._last_non_urgent_audio_t[key] = t

        record = {
            "t": round(float(t), 4),
            "from": old,
            "to": new,
            "reason": reason,
            "immediate": bool(immediate),
            "action": action,
            "shadow_mode": self.shadow_mode,
            "snapshot": snapshot(m),
        }
        self.transitions.append(record)

        # §0/§8: Stage 0 is shadow mode -- decisions are logged, audio is muted.
        if self.audio_sink is not None and not self.shadow_mode:
            self.audio_sink(record)


def snapshot(m: Measurement) -> dict:
    """Full measurement snapshot for the permanent audit trail (§0, §6)."""
    def num(v):
        if v is None:
            return None
        v = float(v)
        return None if math.isinf(v) else round(v, 4)

    return {
        "d": num(m.d),
        "d_dot": num(m.d_dot),
        "tau": num(m.tau),
        "ttc": num(m.ttc),
        "tau_inf": bool(m.tau == float("inf")),
        "ttc_inf": bool(m.ttc == float("inf")),
        "trend": m.trend,
        "confidence": num(m.confidence),
        "depth_conf": num(m.depth_conf),
        "track_quality": num(m.track_quality),
        "track_lost": bool(m.track_lost),
        "coast_age": num(m.coast_age),
        "new_lead": bool(m.new_lead),
        "v_host": num(m.ctx.v_host),
        "v_host_source": m.ctx.v_host_source,
        "v_host_stale": bool(m.v_host_stale),
        "anchor_age_s": num(m.anchor_age_s),
        "rain": bool(m.ctx.rain),
        "night": bool(m.ctx.night),
    }
