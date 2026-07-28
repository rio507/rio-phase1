# RIO Headway Monitoring — System Design v1

Camera-first time-headway and gap-trend monitoring. Qwen-VL for semantics, deterministic code for safety.

---

## 0. Three design challenges before the architecture (read first)

**Challenge 1 — Qwen cannot be in the distance loop, even indirectly.**
Your plan has Qwen "identify the lead vehicle" feeding Depth Anything. On Jetson, Qwen runs at 0.2–0.5 Hz. A car closing at 20 mph relative covers ~36 m between two Qwen inferences — your entire headway. The fix is not a detector; it's a **classical CV tracker** (OpenCV CSRT/KCF — not a neural model) that holds Qwen's lead-vehicle box between Qwen updates, so Depth Anything can sample distance at 10–15 Hz. Qwen re-anchors the tracker every few seconds. This is the one component your spec was missing, and without it the design does not work.

**Challenge 2 — monocular metric depth has a scale problem; exploit the math that cancels it.**
Depth Anything V2 Metric is typically ±10–20% absolute at automotive ranges, and the error is mostly a consistent *scale bias* per installation. Two consequences:
- **Time headway (τ = d/v_host) inherits the bias** → calibrate once per install against 2–3 known distances (parking-lot cones), and treat τ thresholds as soft bands, not hard lines.
- **TTC = d/ḋ cancels constant scale bias entirely** (numerator and derivative share the bias). Your most safety-critical number is your most robust one. Lean on TTC for urgent decisions, τ for comfort coaching.

**Challenge 3 — urgent audio cannot go through the TTS pipeline.**
ElevenLabs round trip is 1–2 s. An urgent warning that arrives 2 seconds late is a crash narration. **Pre-render the urgent clips** (RIO's voice, recorded once, stored on-device) and play them locally in <100 ms. Only non-critical lines ("you're getting a little close there") may use live TTS. This also means urgent warnings still work with zero connectivity.

One more honesty note: this is a driver-assist advisory feature, not certified ADAS. Run it in **shadow mode** (log decisions, mute audio) for your first drives, compare its warnings against what actually happened, then enable audio. Log every state transition permanently — that audit trail is both your tuning data and your liability posture.

---

## 1. Architecture (text diagram)

```
                 ┌────────── SLOW SEMANTIC LOOP · 0.2–0.5 Hz ──────────┐
                 │  Qwen-VL (Jetson INT4, or cloud during prototype)    │
Camera ──┬──────►│  • find lead vehicle in ego lane → bounding box      │
         │       │  • lane/scene context: "cut-in from right",          │
         │       │    "construction", "motorcycle", "curve ahead"       │
         │       └───────────────┬─────────────────────────────────────┘
         │                       │ lead ROI + semantic flags (advisory only)
         │                       ▼
         │       ┌────────── FAST DETERMINISTIC LOOP · 10–15 Hz ────────┐
         ├──────►│ 1. ROI tracker (OpenCV CSRT) — holds lead box        │
         │       │    between Qwen re-anchors                           │
         │       │ 2. Depth Anything V2 Metric-S (TensorRT FP16)        │
         │       │    → median depth inside ROI = d_raw(t)              │
         │       │ 3. Kalman filter, state [d, ḋ] + innovation gate     │
         │       │ 4. τ = d / v_host   ·   TTC = d / max(−ḋ, ε)         │
         │       │ 5. Gap-trend classifier (on filtered ḋ)              │
         │       │ 6. Warning-state machine (pure deterministic code)   │
         │       └──────┬──────────────────────────┬───────────────────┘
         │              │ state, τ, TTC, trend     │ URGENT / WARNING
GPS/OBD ─┴─ v_host ────►│                          ▼
IMU (opt) ─ yaw,pitch ─►│               pre-rendered local audio clips
                        ▼               (<100 ms, works offline)
              RIO voice layer (bible tone, live TTS)
              — non-critical lines only —
              + optional overlay (OpenCV/Supervision)
```

**LLM firewall:** Qwen's outputs (ROI, flags) may *inform* which object is measured. They may never *fire* a warning. Warning states are a pure function of filtered numbers.

---

## 2. Data contracts between components

| From → To | Payload | Rate |
|---|---|---|
| Camera → both loops | 1080p frame, timestamped (monotonic clock) | 15–30 fps |
| Qwen → tracker | `{lead_box: [x1,y1,x2,y2], lead_class, confidence, flags: [cut_in, curve, motorcycle, none_found], ts}` | 0.2–0.5 Hz |
| Tracker → depth | `{roi, track_quality: 0–1, ts}` | 10–15 Hz |
| Depth → filter | `{d_raw (m), depth_conf: ROI variance + valid-pixel %, ts}` | 10–15 Hz |
| GPS/OBD → filter | `{v_host (m/s), source: CAN\|OBD\|GPS, staleness_ms}` | ≥1 Hz (CAN: 10+ Hz) |
| IMU (optional) → selector | `{yaw_rate, pitch}` | 10+ Hz |
| Filter → state machine | `{d, ḋ, τ, TTC, trend, confidence, lead_id, age_since_anchor}` | 10–15 Hz |
| State machine → audio | `{state, transition, urgency}` | on transition only |

**Required inputs:** camera, host speed (GPS floor; OBD/CAN strongly preferred — GPS lags ~1 s and dies in tunnels), rigid camera mount with known height + pitch.
**Optional inputs:** OBD/CAN speed (upgrade), IMU yaw rate (curve handling), steering angle, wiper/light state (confidence modifiers).

---

## 3. Mathematical logic

**Time headway**

```
τ = d / max(v_host, 0.5 m/s)        # guard div-by-zero; below ~2 m/s headway is meaningless → suppress
```

**Closing speed and TTC**

```
v_close = −ḋ                        # from Kalman state, positive = gap shrinking
TTC     = d / v_close   if v_close > 0.2 m/s, else ∞
```

**Gap trend** (on filtered ḋ, with hysteresis so classes don't flicker):

```
ḋ > +0.3 m/s            → INCREASING
|ḋ| ≤ 0.3               → STABLE
−1.5 < ḋ ≤ −0.3         → SLOWLY_SHRINKING
ḋ ≤ −1.5                → RAPIDLY_SHRINKING
```

Scale the ±0.3 dead-band up with speed (noise grows with range; at highway range use ±0.5).

**Lead-vehicle speed is never needed explicitly.** ḋ *is* the relative speed — the Kalman derivative of your own distance series gives you closing risk without knowing the lead's absolute speed.

---

## 4. Smoothing and confidence logic

**Kalman filter** — constant-velocity model on the gap:

```
state x = [d, ḋ]ᵀ
F = [[1, Δt], [0, 1]]        # Δt from frame timestamps, not assumed
z = median depth in ROI       # median, not mean — rejects windshield/sky pixels
R = R₀ / depth_conf           # inflate measurement noise when ROI variance is high,
                              # valid-pixel ratio low, night/rain flagged
Q tuned so filter tracks a 3 m/s² braking lead within ~0.3 s
```

**Outlier gating with cut-in escape hatch:**

```
if |z − H·x̂| > 3σ:  reject sample, increment consec_rejects
if consec_rejects ≥ 3:  # not noise — the world changed (cut-in, new lead, occlusion end)
    reset filter with z as new d, flag NEW_LEAD, require confirmation before any warning
```

**Asymmetric confirmation:**

- **Escalate** (toward warning): 3 consecutive agreeing frames ≈ 0.25 s at 12 Hz. Fast.
- **De-escalate** (toward comfortable): 12 consecutive frames ≈ 1 s. Slow.
- **URGENT bypass:** if TTC < 2.0 s AND depth_conf high AND track_quality high → 2 frames. Never single-frame.

**Confidence score** (0–1, multiplied into all thresholds):

```
confidence = f(depth ROI variance, valid-pixel %, track_quality,
               time since last Qwen anchor, night/rain flags, v_host source)
if confidence < 0.4 → state = DEGRADED (RIO says nothing, overlay shows "monitoring limited")
```

Low confidence must **suppress** warnings, not fire cautious ones.

---

## 5. Lead-vehicle selection

**Ego corridor** — deterministic geometric gate:

1. Project a trapezoidal corridor from the camera: width = lane width (default 3.5 m) at each depth, using camera height/pitch calibration.
2. Bend the corridor by yaw rate when IMU/steering available: lateral offset at distance L ≈ `(yaw_rate / v_host) · L² / 2`. Without IMU, straight corridor — known curve weakness.
3. Candidate = vehicles whose box-bottom-center falls inside the corridor.
4. Lead = nearest candidate by depth.

**Qwen's role:** Qwen proposes the lead box; deterministic code validates against the corridor. If they disagree, trust the corridor and re-query.

**Event handling:**

- **Cut-in:** depth drops discontinuously → 3-consecutive-rejects catches it → reset filter → 0.3 s confirmation → warnings live again.
- **Own lane change:** treat like cut-in (reset + confirm).
- **Curves:** yaw-rate bend first; Qwen semantic tiebreak; lane-line detection only if data proves need.
- **Hills:** IMU pitch shifts horizon, else confidence-penalize.
- **Temporary occlusion:** coast Kalman prediction max 1.0 s with decaying confidence, then LOST. Never warn from coasted data older than 1 s.

---

## 6. Warning-state machine (pseudocode)

```python
STATES = COMFORTABLE, MONITOR, CLOSE, WARNING, URGENT, LOST, DEGRADED

TAU_COMFORT   = 3.0     # s — soft bands, modulated by conditions
TAU_MONITOR   = 2.0
TAU_CLOSE     = 1.5
TTC_WARNING   = 4.0
TTC_URGENT    = 2.5

def modulate(thresholds, ctx):
    if ctx.rain or ctx.night:       thresholds.scale(1.25)
    if ctx.v_host > 25:             thresholds.TAU_CLOSE = 1.8
    return thresholds

def tick(m):   # 10–15 Hz
    if m.confidence < 0.4 or m.v_host_stale:   return DEGRADED
    if m.track_lost and m.coast_age > 1.0:     return LOST
    t = modulate(BASE, m.ctx)
    if m.TTC < t.TTC_URGENT and m.trend == RAPIDLY_SHRINKING:
        return confirm(URGENT, frames=2)
    if m.tau < t.TAU_CLOSE and m.trend in (SLOWLY_SHRINKING, RAPIDLY_SHRINKING):
        return confirm(WARNING, frames=3)
    if m.tau < t.TAU_MONITOR:
        return confirm(CLOSE, frames=3 if m.trend != INCREASING else 6)
    if m.tau < t.TAU_COMFORT:  return confirm(MONITOR, frames=6)
    return confirm(COMFORTABLE, frames=12)

def on_transition(old, new):
    if new == URGENT:   play_local("urgent.wav")        # pre-rendered, interrupts all
    elif new == WARNING and old not in (WARNING, URGENT):
        play_local("warning.wav")
    elif new == CLOSE and old in (COMFORTABLE, MONITOR):
        rio_speak("Beep beep — you're getting a little close there.")   # live TTS OK
    # rate-limit: no repeat non-urgent line within 30 s
    log_transition(old, new, snapshot())                # permanent audit trail
```

---

## 7. Likely failure modes

| # | Failure | Mitigation |
|---|---|---|
| 1 | Monocular scale bias | Per-install calibration; TTC unaffected |
| 2 | Depth flattens beyond ~50 m | Confidence-gate by range |
| 3 | Straight corridor on curves | Yaw-rate bend; Qwen tiebreak; log it |
| 4 | Tracker drift onto adjacent car | track_quality monitor; re-anchor ≤5 s; corridor re-validation |
| 5 | Night/rain/glare | DEGRADED suppression, never cautious-fire |
| 6 | Motorcycles (few ROI pixels) | Widen margin, raise confirmation frames |
| 7 | GPS speed lag/tunnels | Staleness gate → DEGRADED; OBD dongle fix |
| 8 | Qwen misidentifies lead | Geometry validation rejects; re-query |
| 9 | DA temporal flicker | Kalman + R inflation |
| 10 | Latency stacking | Timestamp at capture; budget <150 ms end-to-end |
| 11 | Jetson thermal throttle | Rate monitor; DEGRADED below 8 Hz |
| 12 | Driver tunes out warnings | Bible pacing; rate-limit |

---

## 8. Prototype plan (phased)

- **Stage 0 — recorded video, cloud pod.** Full fast loop on L40S vs dashcam clips. Tune Kalman Q/R, thresholds, confirmations. Deliverable: warning timeline overlaid on video.
- **Stage 1 — Jetson bench.** Same clips on Orin, TensorRT DA-V2-S. ≥10 Hz sustained, <150 ms.
- **Stage 2 — parked + calibration.** Camera mount, cone test 10/20/40 m, scale factor.
- **Stage 3 — shadow-mode drives.** Audio muted, everything logged, review vs reality.
- **Stage 4 — audio on.** CLOSE tier first; WARNING/URGENT only after shadow data proves them.

Pre-render urgent/warning clips in RIO's ElevenLabs voice once; ship on device.

---

## 9. When to add RF-DETR + ByteTrack

Add when shadow data shows: Qwen anchor wrong >5% of cycles; CSRT drift >2×/hour; cut-in confirmation >0.5 s; multi-object (pedestrian/cyclist) need; or Qwen starves the fast loop below 10 Hz. The deterministic core carries over unchanged.

---

## 10. MVP vs production

**MVP:** Camera → Qwen-VL anchor (0.2–0.5 Hz, cloud OK) → CSRT tracker → DA-V2 **Metric-Small** (Apache 2.0 ONLY — larger metric variants are CC-BY-NC) → Kalman → τ/TTC/trend → state machine → pre-rendered urgent audio + live TTS casual lines. GPS speed. Shadow mode first.

**Production:** Camera → RF-DETR (TensorRT 30 fps) → ByteTrack → per-track DA-V2-S depth → same Kalman/state machine → CAN speed + IMU corridor → Qwen slow loop for semantics/personality → local alert audio. All Apache-2.0/MIT.
