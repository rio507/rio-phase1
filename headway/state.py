"""Warning logic v2 — five-band tau posture + orthogonal TTC urgent path.

Design ref: docs/warning_logic_v2.md, which **supersedes §6 of
headway_design.md**. Sections of headway_design.md that v2 does not replace
(§3 maths, §4 filtering, §5 lead selection/coasting) remain in force.

Architecture change from v1: URGENT is no longer a rung on the tau ladder. tick()
returns a tau band *and* an independent urgent flag, because the two answer
different questions (v2 §0 Challenge 1):

  - tau  describes *following posture*. It coaches. tau < 1.0 s with a stable gap
         is ordinary dense-traffic driving, and barking at it teaches the driver
         to mute RIO.
  - TTC  detects the *actual emergency*, and cancels monocular scale bias. A
         tau of 2.5 s with TTC of 1.8 s is a real emergency the bands miss.

tau coaches; TTC saves.

This module stays pure -- no model imports, no I/O -- which is what makes the
LLM firewall (v2 §9) auditable: nothing in tick() or voice() can read an LLM
output, because there is nothing here to read one with.
"""
import math
from dataclasses import dataclass, field

# ===========================================================================
# PROVISIONAL BLOCK -- every tunable value in this module lives here.
#
# From warning_logic_v2.md's header, verbatim in intent:
#   "PROVISIONAL prototype values. These are engineering starting points for
#    shadow-mode tuning -- NOT validated safety thresholds. Do not represent
#    them as such anywhere, ever."
#
# Tuning sheet is v2 §10. Every one of these is falsifiable from the shadow-mode
# JSONL that run_clip.py writes -- that is the point of Stage 0/3.
# ===========================================================================

# --- tau band entry thresholds (v2 §1). A band is entered when tau < its value.
TAU_ENTER_NORMAL = 2.5           # above this -> COMFORTABLE
TAU_ENTER_GETTING_UNSAFE = 2.0
TAU_ENTER_UNSAFE = 1.5
TAU_ENTER_CRITICAL = 1.0

# --- exit hysteresis (v2 §2): leaving a worse band needs tau > entry + HYST.
HYST_S = 0.2

# --- TTC (v2 §1, §9). Orthogonal to the bands.
TTC_URGENT = 2.5                 # + RAPIDLY_SHRINKING -> URGENT from any band
TTC_UNSAFE_ASSIST = 4.0          # in UNSAFE + RAPIDLY_SHRINKING -> urgent clip

# --- low-speed handling (v2 §7)
V_MIN_COACH = 5.0                # below: SUPPRESSED_LOW_SPEED, TTC stays armed
V_SOFT = 8.0                     # below: relax thresholds
V_SOFT_SCALE = 1.2
V_HOST_STALE_S = 2.0             # staler than this -> DEGRADED
TAU_MIN_V_HOST = 0.5             # div-by-zero guard only

# --- temporal confirmation, frames (v2 §3)
CONFIRM_UP = 3                   # escalate one band     ~0.25 s @ 12 Hz
CONFIRM_URGENT = 2               # escalate to URGENT    ~0.17 s
CONFIRM_DOWN = 12                # de-escalate any band  ~1.0 s
CONFIRM_RECOVER = 6              # leave LOST/DEGRADED   ~0.5 s

# --- URGENT 2-frame bypass preconditions.
# v2 §3 requires confidence >= 0.7 AND track_quality >= 0.7. The TTC and
# depth_conf conditions below are carried over from headway_design.md §4 and
# retained deliberately (confirmed intended): the bypass is a concession to
# urgency and must not apply when the inputs justifying it are weak. Net effect
# is a conjunction stricter than v2 §3 alone -- URGENT takes 3 frames instead of
# 2 when 2.0 <= TTC < 2.5, costing ~0.08 s at 12 Hz.
URGENT_BYPASS_TTC = 2.0
URGENT_BYPASS_CONFIDENCE = 0.7
URGENT_BYPASS_TRACK_QUALITY = 0.7
URGENT_BYPASS_DEPTH_CONF = 0.7

# --- trend classes (v2 §4)
TREND_DEAD_BAND_MS = 0.3
TREND_RAPID_MS = 1.5
TREND_HIGHWAY_V = 25.0           # above this host speed, bands x1.5
TREND_HIGHWAY_SCALE = 1.5
TREND_HYSTERESIS_MS = 0.10       # anti-flicker; ~1/3 of measured d_dot jitter
# A trend class is only believed once the estimate is this many sigma clear of
# zero, using the Kalman's own sd(d_dot). Not in v2 -- added because the filter's
# velocity uncertainty starts at 5 m/s and takes ~0.3 s to converge, so the first
# frames after any init or NEW_LEAD reset produce a d_dot that is pure noise. On
# the stable tau=0.9 clip that fired a spurious "too_close" at t=0.13 s from a
# -0.53 m/s reading whose own sd was 1.56 m/s (0.34 sigma). NEW_LEAD's 3-frame
# hold does not cover this: confirming a *band* is fast, converging a *velocity*
# is not. 1.0 sigma is the bar; converged sd(d_dot) is ~0.28 m/s, so genuine
# closures clear it easily.
TREND_SIGNIFICANCE_SIGMA = 1.0
# Coaching voice stays silent for this long after any filter init or NEW_LEAD
# reset. Also not in v2, and it is the gate that actually fixes the spurious
# "too_close": the significance and 3-frame trend gates do not, because the
# artefact is not flicker. Seeded at d_dot=0 with P_vv=25, the filter settles
# ~0.1 m below a steady truth and the constant-velocity model reads that residual
# as a real -0.4 m/s closure held for ~5 consecutive frames at 1.0-1.5 sigma --
# indistinguishable from a genuine slow closure by any per-frame test.
#
# Sized from the measured convergence of sd(d_dot) under the configured Q/R:
# 5.0 m/s at init, falling inside ~5% of its 0.284 m/s floor by ~0.57 s. 0.6 s
# covers it. This is the velocity-domain analogue of v2 §5's NEW_LEAD confirm --
# 3 frames is right for confirming a *band*, far too short for converging a
# *velocity*. URGENT/TTC is deliberately NOT gated by this.
COACH_WARMUP_S = 0.6

# --- confidence (v2 §8). Weighted SUM (v1 used a product); weights provisional.
CONF_W_VALID = 0.30              # depth valid-pixel ratio
CONF_W_VARIANCE = 0.30           # 1 - normalised ROI variance
CONF_W_TRACK = 0.25              # track_quality
CONF_W_ANCHOR = 0.15             # anchor freshness
CONF_FLOOR = 0.4                 # below -> DEGRADED, no voice
CONF_FULL = 0.7                  # at/above -> full operation incl. urgent bypass
ANCHOR_FRESH_S = 5.0             # freshness 1.0 up to here
ANCHOR_STALE_S = 10.0            # freshness 0.0 from here
MAX_COAST_S = 1.0                # headway_design §5: coast <=1 s, decaying conf

# --- voice cooldowns, seconds (v2 §6). Gate VOICE ONLY -- overlay/log always update.
COOLDOWN_CALM_S = 30.0
COOLDOWN_STRONG_S = 15.0
URGENT_MIN_GAP_S = 2.0           # URGENT has no cooldown, only a min gap
COOLDOWN_TAILGATE_CONV_S = 600.0  # persistent_tailgate conversational surfacing

# ===========================================================================
# END PROVISIONAL BLOCK
# ===========================================================================

# --- tau bands (v2 §1) ---
COMFORTABLE = "COMFORTABLE"
NORMAL = "NORMAL"
GETTING_UNSAFE = "GETTING_UNSAFE"
UNSAFE = "UNSAFE"
CRITICAL = "CRITICAL"

TAU_BANDS = (COMFORTABLE, NORMAL, GETTING_UNSAFE, UNSAFE, CRITICAL)
# Ascending severity, so "worse_or_equal" in v2 §9's classify_tau is a >= here.
_BAND_SEVERITY = {b: i for i, b in enumerate(TAU_BANDS)}
_BAND_ENTER = {
    NORMAL: TAU_ENTER_NORMAL,
    GETTING_UNSAFE: TAU_ENTER_GETTING_UNSAFE,
    UNSAFE: TAU_ENTER_UNSAFE,
    CRITICAL: TAU_ENTER_CRITICAL,
}

# --- system states (v2 §1) ---
LOST = "LOST"
DEGRADED = "DEGRADED"
SUPPRESSED_LOW_SPEED = "SUPPRESSED_LOW_SPEED"
SYSTEM_STATES = (LOST, DEGRADED, SUPPRESSED_LOW_SPEED)

# --- urgent overlay (display only; not a band) ---
URGENT = "URGENT"

ALL_STATES = TAU_BANDS + SYSTEM_STATES + (URGENT,)

# --- trend classes (v2 §4) ---
INCREASING = "INCREASING"
STABLE = "STABLE"
SLOWLY_SHRINKING = "SLOWLY_SHRINKING"
RAPIDLY_SHRINKING = "RAPIDLY_SHRINKING"
TREND_ORDER = (RAPIDLY_SHRINKING, SLOWLY_SHRINKING, STABLE, INCREASING)
SHRINKING = (SLOWLY_SHRINKING, RAPIDLY_SHRINKING)

# --- voice line keys, and which bands re-arm them (v2 §6) ---
LINE_CALM = "calm"
LINE_CLOSING_FAST = "closing_fast"
LINE_TOO_CLOSE = "too_close"
LINE_BRAKE = "brake"

# "must leave <band> and re-enter" -- a line re-arms once the band it belongs to
# is no longer current.
_LINE_HOME_BANDS = {
    LINE_CALM: (GETTING_UNSAFE,),
    LINE_CLOSING_FAST: (GETTING_UNSAFE,),
    LINE_TOO_CLOSE: (UNSAFE, CRITICAL),
}


@dataclass
class Context:
    """Host/condition inputs.

    NOTE -- `rain` and `night` no longer modulate anything. v1 (headway_design
    §6) scaled thresholds by 1.25 for them; v2 replaces that section and its only
    threshold modulation is the low-speed relaxation in §7, with no rain/night
    term in the §8 confidence formula either. The fields are kept as declared
    stubs so the plumbing exists when detectors land, but under v2 they are inert.
    """
    v_host: float = 0.0
    v_host_source: str = "FIXED"
    rain: bool = False           # STUB -- inert under v2
    night: bool = False          # STUB -- inert under v2
    yaw_rate: float = None       # STUB -- no IMU in Stage 0


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


# ---------------------------------------------------------------------------
# Maths (headway_design §3, still in force)
# ---------------------------------------------------------------------------
def compute_tau(d, v_host):
    """tau = d / max(v_host, 0.5).

    No low-speed suppression here -- v2 §7 handles that as the
    SUPPRESSED_LOW_SPEED system state, so tau stays a pure measurement and the
    policy decision lives in one place.
    """
    if d is None or not math.isfinite(d):
        return float("inf")
    return d / max(float(v_host), TAU_MIN_V_HOST)


def compute_ttc(d, d_dot):
    """TTC = d / max(-d_dot, eps); infinite unless genuinely closing.

    The most robust number available: a constant depth scale bias multiplies d
    and d_dot alike and cancels in the ratio (v2 §0 Challenge 1).
    """
    if d is None or d_dot is None or not math.isfinite(d) or not math.isfinite(d_dot):
        return float("inf")
    v_close = -float(d_dot)
    if v_close <= 0.2:
        return float("inf")
    return d / v_close


def trend_bands(v_host: float):
    """(dead_band, rapid_threshold) -- both x1.5 above 25 m/s host speed (v2 §4)."""
    scale = TREND_HIGHWAY_SCALE if v_host > TREND_HIGHWAY_V else 1.0
    return TREND_DEAD_BAND_MS * scale, TREND_RAPID_MS * scale


def classify_trend(d_dot, v_host: float, current: str = None,
                   d_dot_sigma: float = None) -> str:
    """v2 §4 trend classes, with hysteresis on every boundary.

    `d_dot_sigma` is the Kalman's own sd(d_dot) (sqrt of P[1,1]). When supplied,
    an estimate closer to zero than TREND_SIGNIFICANCE_SIGMA sigma is reported as
    STABLE regardless of its value: the filter is saying it does not yet know the
    closing speed, and STABLE is the class that neither warns nor escalates.
    Omit it to get the raw v2 behaviour.
    """
    if d_dot is None or not math.isfinite(d_dot):
        return current or STABLE

    if (d_dot_sigma is not None and math.isfinite(d_dot_sigma)
            and abs(d_dot) < TREND_SIGNIFICANCE_SIGMA * d_dot_sigma):
        return STABLE

    dead, rapid = trend_bands(v_host)
    edges = [-rapid, -dead, dead]      # ascending in d_dot

    def index_with(shift: float) -> int:
        return sum(1 for e in edges if d_dot > e + shift)

    idx = index_with(0.0)
    cand = TREND_ORDER[idx]
    if current is None or current not in TREND_ORDER or cand == current:
        return cand

    # Require overshoot in whichever direction we're moving, so a value sitting
    # exactly on an edge stays put.
    cur_idx = TREND_ORDER.index(current)
    if idx > cur_idx:
        idx = index_with(+TREND_HYSTERESIS_MS)
    elif idx < cur_idx:
        idx = index_with(-TREND_HYSTERESIS_MS)
    return TREND_ORDER[idx]


def classify_tau(tau, prev_band: str = None) -> str:
    """v2 §2/§9 classify_tau: descending band check with exit hysteresis.

    Entry uses the bare threshold; exit requires tau > threshold + HYST, applied
    when the previous band was this bad or worse. That yields exactly v2 §2:

        enter GETTING_UNSAFE at tau < 2.0, exit to NORMAL at tau > 2.2
        enter UNSAFE         at tau < 1.5, exit at tau > 1.7
        enter CRITICAL       at tau < 1.0, exit at tau > 1.2

    The same rule is applied uniformly at the NORMAL/COMFORTABLE boundary too
    (exit NORMAL at tau > 2.7), following v2 §9's loop which includes NORMAL.
    That boundary is not a warning boundary, so the extra stickiness is harmless.
    """
    if tau is None or not math.isfinite(tau):
        return COMFORTABLE

    prev_sev = _BAND_SEVERITY.get(prev_band, -1)
    for band in (CRITICAL, UNSAFE, GETTING_UNSAFE, NORMAL):
        enter = _BAND_ENTER[band]
        edge = enter + (HYST_S if prev_sev >= _BAND_SEVERITY[band] else 0.0)
        if tau < edge:
            return band
    return COMFORTABLE


def anchor_freshness(anchor_age_s: float) -> float:
    """1.0 while fresh, ramping to 0.0 by ANCHOR_STALE_S."""
    age = float(anchor_age_s)
    if age <= ANCHOR_FRESH_S:
        return 1.0
    if age >= ANCHOR_STALE_S:
        return 0.0
    return 1.0 - (age - ANCHOR_FRESH_S) / (ANCHOR_STALE_S - ANCHOR_FRESH_S)


def compute_confidence(depth_valid_ratio, roi_variance_norm, track_quality,
                       anchor_age_s, coast_age=0.0) -> float:
    """v2 §8 confidence: a weighted SUM.

        .30*depth_valid_ratio + .30*(1 - roi_variance_norm)
      + .25*track_quality     + .15*anchor_freshness

    This is a deliberate change from v1, which multiplied the terms. A product
    lets any single weak input veto everything; v2 chose a sum, so a strong depth
    reading can partly carry a mediocre track. The floor/full tiers (0.4/0.7) are
    what enforce suppression now, not the combination rule.

    The coast multiplier is *not* from v2 §8 -- it carries over from
    headway_design §5 ("coast Kalman prediction max 1.0 s with decaying
    confidence"), which v2 does not supersede. Without it, coasted data would
    keep full confidence right up to the LOST cutoff.
    """
    valid = _clip01(depth_valid_ratio)
    var = _clip01(roi_variance_norm)
    track = _clip01(track_quality)
    fresh = anchor_freshness(anchor_age_s)

    base = (CONF_W_VALID * valid
            + CONF_W_VARIANCE * (1.0 - var)
            + CONF_W_TRACK * track
            + CONF_W_ANCHOR * fresh)

    coast = max(0.0, 1.0 - float(coast_age) / MAX_COAST_S)
    return _clip01(base * coast)


def _clip01(v) -> float:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return max(0.0, min(1.0, v))


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------
class WarningStateMachine:
    """v2 §9 tick() + voice(), with confirmation and cooldowns.

    Confirmation model: each tick names a *candidate* band and how many
    consecutive frames it needs (CONFIRM_UP escalating, CONFIRM_DOWN
    de-escalating). DEGRADED/LOST are entered immediately -- delaying an "I can't
    see" verdict would mean warning from data already judged bad -- but leaving
    them takes CONFIRM_RECOVER frames.
    """

    def __init__(self, rate_hz: float = 12.0, audio_sink=None, shadow_mode: bool = True):
        self.rate_hz = float(rate_hz)
        self.shadow_mode = bool(shadow_mode)
        self.audio_sink = audio_sink

        self.tau_band = COMFORTABLE
        self.system_state = None
        self.urgent = False
        self.state = COMFORTABLE          # display state, for overlay/timeline

        self._cand_band = None
        self._cand_count = 0
        self._cand_required = 0
        self._urgent_count = 0
        self._urgent_required = 0
        self._recover_count = 0
        self._new_lead_hold_frames = 0
        self._last_trend = None
        self._trend_run = 0
        self._since_reset_s = 0.0

        self._line_last_fired = {}
        self._line_armed = {k: True for k in _LINE_HOME_BANDS}
        self._last_tailgate_event_t = -1e9
        self.urgent_buffer_armed = False

        self.transitions = []
        self.events = []
        self.voice_log = []

    # -- main entry ----------------------------------------------------------
    def tick(self, m: Measurement, t: float = 0.0, dt: float = None) -> dict:
        dt = (1.0 / self.rate_hz) if dt is None else float(dt)

        # Trend run-length. v2 §3's "never fire from one frame" is applied to the
        # trend as well as the band: the band is confirmed before it can coach,
        # but a single-frame trend flip could still fire a coaching line off one
        # noisy d_dot. Only the *coaching* tiers wait for this -- the URGENT/TTC
        # path keeps its own 2-3 frame confirmation so its latency is unaffected.
        if m.trend == self._last_trend:
            self._trend_run += 1
        else:
            self._last_trend = m.trend
            self._trend_run = 1

        if m.new_lead:
            self._since_reset_s = 0.0
        else:
            self._since_reset_s += dt

        if m.new_lead:
            # v2 §5: a cut-in resets the filter, so every confirmation counter it
            # fed is stale. Full 3-frame confirm before any voice.
            self._cand_band, self._cand_count, self._cand_required = None, 0, 0
            self._urgent_count = 0
            self._new_lead_hold_frames = CONFIRM_UP
        elif self._new_lead_hold_frames > 0:
            self._new_lead_hold_frames -= 1

        system = self._system_state(m)
        logs = []

        if system in (DEGRADED, LOST):
            # v2 §9: these return before the urgent path is even evaluated.
            self.urgent = False
            self._urgent_count = 0
            self._commit_system(system, m, t)
            return self._result(m, t, dt, logs, voice=None, tau_candidate=None)

        # Urgent is evaluated for every non-DEGRADED/LOST tick, including while
        # SUPPRESSED_LOW_SPEED -- v2 §7 keeps the TTC path armed at low speed.
        urgent = self._evaluate_urgent(m)

        if system == SUPPRESSED_LOW_SPEED:
            # No tau coaching at all, but the band is still tracked internally so
            # that re-entering coaching speed doesn't start from a stale band.
            self.system_state = SUPPRESSED_LOW_SPEED
            self._recover_count = 0
            voice = self._voice(None, urgent, m, t, logs)
            self._update_display(m, t, urgent, SUPPRESSED_LOW_SPEED)
            return self._result(m, t, dt, logs, voice, tau_candidate=None)

        # Clearing a system state takes CONFIRM_RECOVER frames (v2 §3).
        if self.system_state in (DEGRADED, LOST):
            self._recover_count += 1
            if self._recover_count < CONFIRM_RECOVER:
                voice = self._voice(None, urgent, m, t, logs)
                self._update_display(m, t, urgent, self.system_state)
                return self._result(m, t, dt, logs, voice, tau_candidate=None)
        self.system_state = None
        self._recover_count = 0

        # --- tau band, with low-speed threshold relaxation (v2 §7/§9) --------
        scale = V_SOFT_SCALE if m.ctx.v_host < V_SOFT else 1.0
        tau_scaled = m.tau / scale if math.isfinite(m.tau) else m.tau
        candidate = classify_tau(tau_scaled, self.tau_band)
        self._confirm_band(candidate)

        voice = self._voice(self.tau_band, urgent, m, t, logs)
        self._update_display(m, t, urgent, None)
        return self._result(m, t, dt, logs, voice, tau_candidate=candidate,
                            scale=scale, tau_scaled=tau_scaled)

    # -- system states -------------------------------------------------------
    def _system_state(self, m: Measurement):
        """v2 §9 ordering: DEGRADED, then LOST, then SUPPRESSED_LOW_SPEED."""
        if m.v_host_stale or m.confidence < CONF_FLOOR:
            return DEGRADED
        if m.track_lost and m.coast_age > MAX_COAST_S:
            return LOST
        if m.ctx.v_host < V_MIN_COACH:
            return SUPPRESSED_LOW_SPEED
        return None

    # -- urgent path ---------------------------------------------------------
    def _urgent_frames(self, m: Measurement) -> int:
        """2 frames only under the full conjunction; otherwise the standard 3."""
        if (m.ttc < URGENT_BYPASS_TTC
                and m.confidence >= URGENT_BYPASS_CONFIDENCE
                and m.track_quality >= URGENT_BYPASS_TRACK_QUALITY
                and m.depth_conf >= URGENT_BYPASS_DEPTH_CONF):
            return CONFIRM_URGENT
        return CONFIRM_UP

    def _evaluate_urgent(self, m: Measurement) -> bool:
        """v2 §1/§9: TTC < TTC_URGENT and trend RAPIDLY_SHRINKING, from any band."""
        raw = (math.isfinite(m.ttc) and m.ttc < TTC_URGENT
               and m.trend == RAPIDLY_SHRINKING)

        if not raw:
            self._urgent_count = 0
            self._urgent_required = 0
            self.urgent = False
            return False

        if self._new_lead_hold_frames > 0:
            # v2 §5: no voice at all until the new lead has been confirmed.
            self._urgent_count = 0
            self.urgent = False
            return False

        self._urgent_required = self._urgent_frames(m)
        self._urgent_count += 1
        # "Never fire from one frame" (v2 §3) -- guaranteed since both
        # CONFIRM_URGENT and CONFIRM_UP are >= 2.
        self.urgent = self._urgent_count >= self._urgent_required
        return self.urgent

    # -- band confirmation ---------------------------------------------------
    def _confirm_band(self, candidate: str) -> bool:
        if candidate == self.tau_band:
            self._cand_band, self._cand_count, self._cand_required = None, 0, 0
            return False

        escalating = _BAND_SEVERITY[candidate] > _BAND_SEVERITY[self.tau_band]
        required = CONFIRM_UP if escalating else CONFIRM_DOWN

        if candidate != self._cand_band:
            self._cand_band = candidate
            self._cand_count = 1
        else:
            self._cand_count += 1
        self._cand_required = required

        if self._cand_count >= required:
            self.tau_band = candidate
            self._cand_band, self._cand_count, self._cand_required = None, 0, 0
            return True
        return False

    # -- voice policy (v2 §4 table + §9 voice()) -----------------------------
    def _voice(self, band, urgent: bool, m: Measurement, t: float, logs: list):
        """Returns the voice action taken, or None for silence.

        Implements v2 §4's table. Where §9's pseudocode and §4's table disagree,
        the table wins -- see the CRITICAL branch below.
        """
        if self._new_lead_hold_frames > 0:
            return self._suppressed("new_lead_confirming", logs)

        # URGENT overrides everything and has no cooldown, only a min gap.
        if urgent:
            return self._fire(LINE_BRAKE, "play_local", "brake.wav", t,
                              min_gap=URGENT_MIN_GAP_S, interrupts=True, band=band)

        if band is None:
            # SUPPRESSED_LOW_SPEED or an unconfirmed system state: no tau coaching.
            return None

        # Coaching tiers act on the *confirmed* trend, and only once the filter's
        # velocity estimate has had time to converge. An unconfirmed or
        # not-yet-trustworthy trend reads as STABLE, the silent class -- fail
        # quiet, per v2 §8's rule that low certainty suppresses rather than
        # warning cautiously.
        if self._since_reset_s < COACH_WARMUP_S or self._trend_run < CONFIRM_UP:
            coach_trend = STABLE
        else:
            coach_trend = m.trend

        if band == GETTING_UNSAFE:
            if coach_trend == SLOWLY_SHRINKING:
                return self._fire(LINE_CALM, "rio_speak",
                                  "Beep beep — you're getting a little close there.",
                                  t, cooldown=COOLDOWN_CALM_S, band=band)
            if coach_trend == RAPIDLY_SHRINKING:
                return self._fire(LINE_CLOSING_FAST, "play_local", "closing_fast.wav",
                                  t, cooldown=COOLDOWN_STRONG_S, band=band)
            return None                                     # silent when stable

        if band == UNSAFE:
            if coach_trend in SHRINKING:
                if coach_trend == RAPIDLY_SHRINKING and m.ttc < TTC_UNSAFE_ASSIST:
                    # The real branch for TTC_UNSAFE_ASSIST: inside UNSAFE with a
                    # rapidly shrinking gap, TTC < 4 s escalates the voice from
                    # the stronger line straight to the urgent clip.
                    return self._fire(LINE_BRAKE, "play_local", "brake.wav", t,
                                      min_gap=URGENT_MIN_GAP_S, interrupts=True,
                                      band=band, reason="unsafe_ttc_assist")
                return self._fire(LINE_TOO_CLOSE, "play_local", "too_close.wav", t,
                                  cooldown=COOLDOWN_STRONG_S, band=band)
            # Stable tailgating: silent as a warning, but logged (v2 §4 note 1).
            self._log_tailgate(t, logs, coach_trend)
            return None

        if band == CRITICAL:
            # v2 §4 note 2: arm the pre-cached clip for zero-latency fire.
            self.urgent_buffer_armed = True
            if m.trend != INCREASING:
                logs.append("persistent_critical")

            # DEVIATION FROM §9, FOLLOWING §4's TABLE.
            # §9's voice() matches `tau_state == UNSAFE` exactly, so CRITICAL
            # falls through to silence -- making CRITICAL *quieter* than UNSAFE,
            # which cannot be intended. §4's table says CRITICAL is
            # "URGENT: Brake." when RAPIDLY_SHRINKING and "urgent-adjacent" when
            # SLOWLY_SHRINKING. Implemented per the table, which also restores
            # monotonicity: CRITICAL is never quieter than UNSAFE.
            if coach_trend == RAPIDLY_SHRINKING:
                return self._fire(LINE_BRAKE, "play_local", "brake.wav", t,
                                  min_gap=URGENT_MIN_GAP_S, interrupts=True,
                                  band=band, reason="critical_rapid")
            if coach_trend == SLOWLY_SHRINKING:
                return self._fire(LINE_TOO_CLOSE, "play_local", "too_close.wav", t,
                                  cooldown=COOLDOWN_STRONG_S, band=band,
                                  reason="critical_urgent_adjacent")
            self._log_tailgate(t, logs, coach_trend)
            return None

        return None      # COMFORTABLE / NORMAL: silence is the default posture

    def _log_tailgate(self, t: float, logs: list, trend: str) -> None:
        """v2 §4 note 1 / §6: stable tailgating is never a warning, only a log.

        Restricted to STABLE, not the whole INCREASING/STABLE column. Note 1's
        wording is "*Stable* tailgating stays silent as warning -- but logs
        persistent_tailgate", and a driver whose gap is actively opening is
        recovering, not tailgating. Logging those frames inflated the metric by
        ~3 s on the shrinking clip, which would mislead exactly the shadow-mode
        review the tag exists to feed.

        The condition is recorded on every qualifying tick (`persistent_tailgate`
        in the per-frame record); the *conversational* surfacing RIO may act on is
        rate-limited to 10 minutes and governed by bible pacing.
        """
        if trend != STABLE:
            return
        logs.append("persistent_tailgate")
        if (t - self._last_tailgate_event_t) >= COOLDOWN_TAILGATE_CONV_S:
            self._last_tailgate_event_t = t
            self.events.append({"t": round(float(t), 4), "event": "persistent_tailgate",
                                "conversational": True})

    def _fire(self, key, kind, payload, t, cooldown=None, min_gap=None,
              interrupts=False, band=None, reason=None):
        """Apply v2 §6 cooldown/re-arm, then emit or suppress. Voice only."""
        last = self._line_last_fired.get(key, -1e9)
        gap = min_gap if min_gap is not None else cooldown
        if gap is not None and (t - last) < gap:
            return self._suppressed(f"cooldown:{key}", None, key=key,
                                    remaining=round(gap - (t - last), 3))

        # Re-arm requirement: "must leave the band and re-enter" (v2 §6).
        if key in _LINE_HOME_BANDS and not self._line_armed.get(key, True):
            return self._suppressed(f"needs_reentry:{key}", None, key=key)

        self._line_last_fired[key] = t
        if key in self._line_armed:
            self._line_armed[key] = False

        action = {"kind": kind, "line": key, "interrupts": bool(interrupts)}
        if kind == "play_local":
            action["clip"] = payload
        else:
            action["text"] = payload
        if reason:
            action["reason"] = reason

        record = {"t": round(float(t), 4), "band": band, "action": action,
                  "shadow_mode": self.shadow_mode}
        self.voice_log.append(record)
        if self.audio_sink is not None and not self.shadow_mode:
            self.audio_sink(record)
        return action

    def _suppressed(self, why, logs, key=None, remaining=None):
        out = {"kind": "suppressed", "why": why}
        if key:
            out["line"] = key
        if remaining is not None:
            out["remaining_s"] = remaining
        return out

    # -- display / bookkeeping ----------------------------------------------
    def _update_display(self, m, t, urgent, system):
        """Re-arm any line whose home band is no longer current, then set state."""
        for key, bands in _LINE_HOME_BANDS.items():
            if self.tau_band not in bands:
                self._line_armed[key] = True

        if urgent:
            new_state = URGENT
        elif system is not None:
            new_state = system
        else:
            new_state = self.tau_band

        if new_state != self.state:
            old = self.state
            self.state = new_state
            self.transitions.append({
                "t": round(float(t), 4), "from": old, "to": new_state,
                "tau_band": self.tau_band, "urgent": bool(urgent),
                "system_state": system, "snapshot": snapshot(m),
            })

    def _commit_system(self, system, m, t):
        self.system_state = system
        self._recover_count = 0
        self._cand_band, self._cand_count, self._cand_required = None, 0, 0
        self._update_display(m, t, False, system)

    def _result(self, m, t, dt, logs, voice, tau_candidate,
                scale=1.0, tau_scaled=None):
        return {
            "t": round(float(t), 4),
            "state": self.state,
            "tau_band": self.tau_band,
            "system_state": self.system_state,
            "urgent": bool(self.urgent),
            "tau_candidate": tau_candidate,
            "band_confirm": {"candidate": self._cand_band,
                             "count": self._cand_count,
                             "required": self._cand_required},
            "urgent_confirm": {"count": self._urgent_count,
                               "required": self._urgent_required},
            "voice": voice,
            "voice_fired": bool(voice and voice.get("kind") in ("play_local", "rio_speak")),
            "logs": logs,
            "persistent_tailgate": "persistent_tailgate" in logs,
            "urgent_buffer_armed": bool(self.urgent_buffer_armed),
            "new_lead_hold_frames": self._new_lead_hold_frames,
            "low_speed_scale": scale,
            "tau_scaled": (None if tau_scaled is None or not math.isfinite(tau_scaled)
                           else round(tau_scaled, 4)),
            "confidence_tier": ("full" if m.confidence >= CONF_FULL
                                else "reduced" if m.confidence >= CONF_FLOOR
                                else "degraded"),
        }


def snapshot(m: Measurement) -> dict:
    """Full measurement snapshot for the permanent audit trail."""
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
    }
