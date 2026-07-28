# RIO Warning Logic v2 — Deterministic State Machine (supersedes §6 of headway_design.md)

**Status: PROVISIONAL prototype values. These are engineering starting points for shadow-mode tuning — NOT validated safety thresholds. Do not represent them as such anywhere, ever.**

---

## 0. Two challenges to the requested thresholds (accepted, with corrections)

**Challenge 1 — "Critical" must key on TTC, not time headway.**
τ < 1.0 s with a STABLE gap is ordinary dense-traffic driving (most commuters sit at 0.7–1.0 s, unwisely but stably). Barking "Brake" at every stable tailgater teaches the driver to mute RIO. Meanwhile a τ of 2.5 s with TTC of 1.8 s (lead braked hard) is a genuine emergency the τ bands would miss entirely.
**Resolution:** the five τ bands are kept exactly as named — they describe *following posture*. The URGENT/"Brake." decision is driven by **TTC** (= d / closing speed), which cancels monocular scale bias and detects the actual emergency. τ coaches; TTC saves.

**Challenge 2 — low-speed traffic breaks τ entirely.**
τ = d/v → ∞ as v → 0. At 5 mph in a parking queue, a 4 m gap reads τ = 1.8 s "getting unsafe" — nonsense, and stop-and-go beeping is how ADAS features get disabled by their owners.
**Resolution:** below `V_MIN_COACH = 5 m/s (~11 mph)`, all τ-based coaching is **suppressed**. Only the TTC urgent path stays live. Between 5–8 m/s, thresholds relax 20%.

Everything else preserved: five named bands, trend gating, no single-frame warnings, silence above 2.0 s.

---

## 1. States and provisional thresholds

**Measurement states** (from filtered τ):

| State | τ band (s) | Overlay |
|---|---|---|
| COMFORTABLE | > 2.5 | none/dim |
| NORMAL | 2.0 – 2.5 | dim cyan |
| GETTING_UNSAFE | 1.5 – 2.0 | amber |
| UNSAFE | 1.0 – 1.5 | orange |
| CRITICAL | < 1.0 | red |

**System states:** LOST (track gone > 1.0 s), DEGRADED (confidence < floor), SUPPRESSED_LOW_SPEED (v_host < 5 m/s).

**Urgent overlay (orthogonal to τ bands):** `TTC_URGENT = 2.5 s` **and** trend = RAPIDLY_SHRINKING → URGENT, from any τ state.

## 2. Hysteresis (anti-flicker)

Enter each worse state at its boundary; exit only 0.2 s of τ beyond it:

```
enter GETTING_UNSAFE at τ < 2.0   exit back to NORMAL at τ > 2.2
enter UNSAFE         at τ < 1.5   exit back at τ > 1.7
enter CRITICAL       at τ < 1.0   exit back at τ > 1.2
```

H = 0.2 s is a tuning knob; widen if shadow logs show flicker.

## 3. Temporal confirmation (12 Hz loop)

| Transition | Frames | ≈ time |
|---|---|---|
| Escalate one band | 3 | 0.25 s |
| Escalate to URGENT | 2 (needs confidence ≥ 0.7 AND track_quality ≥ 0.7) | 0.17 s |
| De-escalate any band | 12 | 1.0 s |
| Recover from LOST/DEGRADED | 6 | 0.5 s |

Never fire from one frame. Never delay URGENT beyond 2 frames at high confidence.

## 4. Stable vs shrinking gaps (trend gate on voice)

Trend from filtered ḋ:

```
INCREASING          ḋ > +0.3 m/s
STABLE              |ḋ| ≤ 0.3
SLOWLY_SHRINKING    −1.5 < ḋ ≤ −0.3
RAPIDLY_SHRINKING   ḋ ≤ −1.5        (bands ×1.5 above 25 m/s host speed)
```

Voice policy:

| τ band | INCREASING / STABLE | SLOWLY_SHRINKING | RAPIDLY_SHRINKING |
|---|---|---|---|
| > 2.0 | silent | silent | silent (URGENT path watches TTC) |
| GETTING_UNSAFE | silent | **calm line** | stronger line |
| UNSAFE | silent¹ | **stronger line** | **urgent if TTC < 4 s** |
| CRITICAL | silent¹ ² | urgent-adjacent | **URGENT: "Brake."** |

¹ Stable tailgating stays silent as warning — but logs `persistent_tailgate`; RIO may raise it conversationally later via bible pacing, NOT as a warning.
² CRITICAL+stable arms the pre-cached urgent clip for zero-latency fire on trend flip.

## 5. Cut-ins and sudden braking

- **Cut-in:** innovation gate rejects 3 consecutive → reset filter to new d, tag NEW_LEAD, full 3-frame confirm before any voice.
- **Sudden braking (same lead):** continuous d, ḋ spikes negative within gate — no reset; trend flips in 2–3 frames; TTC drops; URGENT path evaluates. This is what the 2-frame bypass exists for.
- Distinction: cut-in = discontinuous d (reset+confirm); braking = continuous d, discontinuous ḋ (no reset, fast escalate).

## 6. Voice cooldowns (anti-nag)

| Tier | Cooldown | Re-arm |
|---|---|---|
| Calm line | 30 s | must leave GETTING_UNSAFE and re-enter |
| Stronger line | 15 s | re-entry required |
| URGENT "Brake." | none | repeats only while TTC < threshold persists; min 2.0 s between utterances |
| persistent_tailgate conversational | 10 min | bible pacing governs |

Cooldowns gate **voice only** — overlay and log always update.

## 7. Low-speed handling

```
v_host < 5 m/s          → SUPPRESSED_LOW_SPEED: no τ coaching, overlay dims, TTC urgent stays armed
5 ≤ v_host < 8 m/s      → thresholds × 1.2
v_host stale > 2 s      → DEGRADED
```

## 8. Confidence handling

```
confidence = .3·depth_valid_ratio + .3·(1 − roi_variance_norm)
           + .25·track_quality + .15·anchor_freshness      (weights provisional)

< 0.4        → DEGRADED: no voice, overlay "monitoring limited"
0.4 – 0.7    → coaching allowed, URGENT bypass disabled (urgent needs standard 3-frame confirm)
≥ 0.7        → full operation
```

Low confidence suppresses; it never "warns cautiously."

## 9. Full pseudocode

```python
# ---- constants (ALL PROVISIONAL — shadow-mode tuning required) ----
TAU = {COMFORTABLE: 2.5, NORMAL: 2.0, GETTING_UNSAFE: 1.5, UNSAFE: 1.0}   # band floors
HYST = 0.2
TTC_URGENT = 2.5;  TTC_UNSAFE_ASSIST = 4.0
V_MIN_COACH = 5.0; V_SOFT = 8.0
CONFIRM_UP = 3; CONFIRM_URGENT = 2; CONFIRM_DOWN = 12
COOLDOWN = {calm: 30, strong: 15, urgent_gap: 2.0}

def classify_tau(tau, prev_state):
    # descending check with hysteresis: exiting a worse state needs tau > floor + HYST
    for s in [CRITICAL, UNSAFE, GETTING_UNSAFE, NORMAL]:
        floor = band_floor(s)
        edge  = floor + (HYST if prev_state worse_or_equal s else 0)
        if tau < edge: return s
    return COMFORTABLE

def tick(m):                                   # m: filtered bundle @ 12 Hz
    if m.v_host_stale or m.confidence < 0.4:   return system_state(DEGRADED)
    if m.track_lost and m.coast_age > 1.0:     return system_state(LOST)
    if m.v_host < V_MIN_COACH:                 return system_state(SUPPRESSED_LOW_SPEED)  # TTC path armed

    scale = 1.2 if m.v_host < V_SOFT else 1.0
    tau_state = classify_tau(m.tau / scale, prev_tau_state)
    tau_state = confirmed(tau_state, up=CONFIRM_UP, down=CONFIRM_DOWN)

    urgent = (m.TTC < TTC_URGENT and m.trend == RAPIDLY_SHRINKING
              and confirmed_urgent(frames=CONFIRM_URGENT if m.confidence >= 0.7 else CONFIRM_UP))

    return tau_state, urgent

def voice(tau_state, urgent, m):
    if urgent:
        return play_local("brake.wav", min_gap=COOLDOWN.urgent_gap)      # pre-rendered
    if tau_state == GETTING_UNSAFE and m.trend == SLOWLY_SHRINKING:
        return say_calm("Beep beep — you're getting a little close there.",
                        cooldown=COOLDOWN.calm, rearm=LEFT_AND_REENTERED)
    if tau_state == GETTING_UNSAFE and m.trend == RAPIDLY_SHRINKING:
        return play_local("closing_fast.wav", cooldown=COOLDOWN.strong)
    if tau_state == UNSAFE and m.trend in (SLOWLY_SHRINKING, RAPIDLY_SHRINKING):
        if m.trend == RAPIDLY_SHRINKING and m.TTC < TTC_UNSAFE_ASSIST:
            return play_local("brake.wav", min_gap=COOLDOWN.urgent_gap)
        return play_local("too_close.wav", cooldown=COOLDOWN.strong)     # pre-rendered, direct
    if tau_state == CRITICAL:
        arm_urgent_buffer()                                              # zero-latency standby
        if m.trend != INCREASING: log("persistent_critical")
    return silent()                                                      # default posture

# every tick, regardless of voice:
log_jsonl(ts, d, d_dot, tau, TTC, trend, tau_state, urgent, confidence, voice_fired)
```

**LLM firewall (restated):** Qwen supplies the ROI and scene flags upstream. Nothing in tick() or voice() reads any LLM output. The bible's TTS renders the calm line's delivery; the words and the trigger are deterministic.

## 10. Provisional-values summary (the tuning sheet)

| Knob | Value | Tune against |
|---|---|---|
| τ band floors | 2.5 / 2.0 / 1.5 / 1.0 | shadow logs vs "felt close" judgment |
| Hysteresis H | 0.2 s | state-flicker count per hour |
| TTC_URGENT | 2.5 s | zero missed hard-brakes; ~zero false urgents |
| Confirm up/urgent/down | 3 / 2 / 12 | warning latency vs flicker |
| Trend dead-band | ±0.3 m/s (×1.5 hwy) | trend-class churn |
| V_MIN_COACH | 5 m/s | stop-and-go annoyance |
| Confidence floor/full | 0.4 / 0.7 | DEGRADED % in night/rain clips |
| Cooldowns | 30 / 15 s | subjective nag factor |

Every knob is falsifiable from the shadow-mode JSONL — that's the point of Stage 0/3.
