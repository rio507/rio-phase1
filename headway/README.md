# headway — Stage 0 deterministic core

Camera-first time-headway and gap-trend monitoring for RIO.
Implements **Stage 0** of [`docs/headway_design.md`](../docs/headway_design.md):
the full fast loop running against recorded clips on a cloud pod, with the
warning timeline overlaid on video.

Self-contained by design — nothing here imports `app.py`, `vision.py` or
`llm_interface.py`, so the headway loop can be run, tuned and broken without
touching the live RIO voice service.

---

## The one rule

**Qwen never fires a warning.** It proposes *which pixels* to measure; a
geometric corridor check vetoes it, and every warning state is a pure function of
filtered numbers (design §1, "LLM firewall"). `state.py` has no model imports and
no I/O, which is what makes that rule auditable rather than aspirational.

---

## Pipeline

```
frame ──► anchor.py   Qwen3-VL proposes a lead box  (slow loop, ~1 Hz)
          │           ego-corridor geometry vetoes it
          ▼
          tracker.py  CSRT holds the box between anchors  (fast loop)
          │
          ▼
          depth.py    DA-V2 Metric-Small → median depth in ROI + depth_conf
          │
          ▼
          filter.py   Kalman [d, ḋ] + 3σ gate + cut-in reset
          │
          ▼
          state.py    τ / TTC / trend → warning state machine
```

| File | Role |
|---|---|
| `depth.py` | Depth Anything V2 Metric-**Small** loader (Apache-2.0 only), `depth_map()`, `roi_depth()` |
| `tracker.py` | CSRT wrapper, `init()` / `update()` → box + quality |
| `anchor.py` | Qwen3-VL grounding + `EgoCorridor` geometric validation |
| `filter.py` | Kalman on the gap, innovation gating, `--tune` harness for Q |
| `state.py` | τ/TTC/trend maths + the §6 state machine (pure, no I/O) |
| `run_clip.py` | Stage 0 harness + synthetic clip generator |
| `selftest.py` | 50 spec-compliance checks, no GPU, ~1 s |

---

## Install

Already installed on this pod. From scratch:

```bash
pip install 'opencv-contrib-python-headless<5'   # <5: stable CSRT API
# transformers >= 4.57 already provides Depth Anything V2 and Qwen3-VL natively
```

Models are pulled from the HF cache on first use:
`Depth-Anything-V2-Metric-Outdoor-Small-hf` (~100 MB) and, for real-clip
anchoring, `Qwen3-VL-8B-Instruct` (~17 GB).

---

## Quick start

```bash
# 1. Generate a synthetic clip with embedded ground truth
python -m headway.run_clip --make-synthetic /tmp/synth.mp4

# 2. Validate filter + state mechanics against that ground truth
python -m headway.run_clip /tmp/synth.mp4 --v-host 25 --depth gt --anchor gt

# 3. A real dashcam clip: real depth, real Qwen anchoring
python -m headway.run_clip dashcam.mp4 --v-host 25

# 4. Deterministic core checks (no GPU)
python -m headway.selftest

# 5. Kalman Q trade curve
python -m headway.filter --sweep
```

`--v-host` is the **fixed** host speed in m/s for Stage 0 — there is no GPS or
OBD feed yet, so every τ depends on this being roughly right for the clip.
25 m/s ≈ 56 mph.

### Outputs (`runs/<clip-stem>/`)

| File | Contents |
|---|---|
| `annotated.mp4` | Lead box, d, τ, TTC, trend, state, plus the full-clip timeline bar |
| `transitions.jsonl` | **The audit trail.** Every state transition + full measurement snapshot |
| `frames.jsonl` | Per-frame record — tuning data, not the audit |
| `summary.json` | State timeline, occupancy, transition count, achieved fps |

### Useful flags

| Flag | Why |
|---|---|
| `--depth gt` | Use ground-truth depth (synthetic clips) to isolate filter/state logic |
| `--anchor gt\|manual\|qwen` | Anchor source; `manual` needs `--init-box x1,y1,x2,y2` |
| `--anchor-interval N` | Frames between Qwen re-anchors (default 60) |
| `--no-video` | Skip encoding — roughly 2× faster when you only want logs |
| `--max-frames N` | Truncate for quick iteration |

---

## Measured on this pod (L40S, 720p)

| Metric | Value |
|---|---|
| Full pipeline, real depth + annotated video | **28.1 fps** (target ≥10 Hz ✓) |
| DA-V2 Metric-Small inference | ~12 ms/frame (24.8 M params, fp16) |
| Qwen3-VL anchor, warm | **1.09 s** (§1 budgets 0.2–0.5 Hz ✓) |
| Kalman lag under 3 m/s² braking | **0.286 s** (§4 target ~0.3 s ✓) |
| ḋ noise on a steady gap | 0.294 m/s (1σ) |
| Self-test | 50/50 |

> Preprocessing, not inference, was the original bottleneck: CPU-side
> resize/normalise cost ~70 ms/frame against ~12 ms for the model, which alone
> held the loop at 9.5 fps. `depth.py` hands the processor a CUDA tensor
> instead, which is where the 2.95× came from.

---

## Tuning knobs

### `filter.py` — smoothing

| Constant | Default | Effect |
|---|---|---|
| `SIGMA_A` | `3.0` m/s² | Process noise. **Derived from the spec's own "3 m/s² braking lead" target.** ↑ tracks harder braking sooner, ↓ smooths ḋ but lags |
| `R_REL_FRAC` | `0.005` | Depth noise as a fraction of range. **The main thing to re-fit from real clips** |
| `R_ABS_M` | `0.10` m | Noise floor for near targets |
| `GATE_SIGMA` | `3.0` | Innovation gate width. ↓ rejects more, risks gating real cut-ins |
| `MAX_CONSEC_REJECTS` | `3` | Rejects before declaring NEW_LEAD |

**Re-fit `R_REL_FRAC` first on real footage.** `summary.json` reports
`measured_innovation_frac_1sigma` on every run for exactly this. On the
synthetic clip it read `0.009` against the configured `0.005` — expect real
dashcam depth to be noisier still, and note the 0.286 s lag figure only holds
for the configured R.

### `state.py` — thresholds and confirmation

| Constant | Default | Effect |
|---|---|---|
| `TAU_COMFORT` / `TAU_MONITOR` / `TAU_CLOSE` | `3.0` / `2.0` / `1.5` s | The τ bands (§6) |
| `TTC_URGENT` | `2.5` s | URGENT trigger — the number to trust (scale bias cancels in TTC) |
| `TTC_WARNING` | `4.0` s | Defined by §6 but **not read by `tick()`** — see below |
| `ESCALATE_FRAMES` / `DEESCALATE_FRAMES` | `3` / `12` | Warn fast, calm down slowly |
| `URGENT_BYPASS_FRAMES` | `2` | Only under high confidence; never 1 frame |
| `CONFIDENCE_FLOOR` | `0.4` | Below → DEGRADED and silence |
| `TREND_HYSTERESIS_MS` | `0.10` m/s | Stops trend classes flickering on a boundary |
| `NON_URGENT_RATE_LIMIT_S` | `30.0` | Per-line, so escalations stay audible |

### `anchor.py` — geometry (calibrate in Stage 2)

| Constant | Default | Effect |
|---|---|---|
| `HFOV_DEG` | `60.0` | **Biggest single source of corridor error.** Sets focal length |
| `CAMERA_HEIGHT_M` | `1.3` | Scales every geometric range estimate |
| `CAMERA_PITCH_RAD` | `0.0` | Non-zero pitch shifts the horizon and all ranges |
| `LANE_WIDTH_M` / `CORRIDOR_MARGIN_M` | `3.5` / `0.6` | Corridor width; ↑ margin accepts more candidates |
| `DEFAULT_ANCHOR_INTERVAL` | `60` frames | Re-anchor cadence (§7.4 wants ≤5 s) |

### `tracker.py` — drift detection

`max_area_ratio_per_s` (8.0), `max_shift_frac_per_s` (3.0),
`max_aspect_ratio_change` (0.35). Deliberately generous: these catch the tracker
*jumping to another vehicle*, not fast approach. A lead closing at 9 m/s from
19 m legitimately changes box area ~2.7×/s.

---

## Two places the spec is ambiguous

**1. `TTC_WARNING = 4.0` is defined but never used.** §6 declares it alongside
the other thresholds, but `tick()` only ever tests `TTC` against `TTC_URGENT`.
Implemented as written — τ drives WARNING, TTC drives URGENT, which matches §0
Challenge 2 ("lean on TTC for urgent decisions, τ for comfort coaching"). If a
TTC-based WARNING tier was intended, `tick()` needs a new branch.

**2. URGENT confirmation frames.** §6 says `confirm(URGENT, frames=2)`
unconditionally; §4 attaches preconditions to that 2 (TTC < 2.0 **and** high
depth_conf **and** high track_quality) and insists on "never single-frame".
Implemented as §4's stricter reading: 2 frames when those hold, 3 otherwise. The
bypass is a concession to urgency and shouldn't apply when the inputs justifying
it are weak.

---

## Stage 0 limitations

- **Fixed host speed.** No GPS/OBD, so τ is only as good as `--v-host`. TTC is
  unaffected (it never uses host speed).
- **No calibration.** Depth carries an uncalibrated scale bias (§0 Challenge 2).
  τ inherits it; TTC cancels it. Cone calibration is Stage 2.
- **Straight corridor.** No IMU, so no yaw-rate bend — the known curve weakness
  (§7.3). The `yaw_rate` path is implemented and unused.
- **`rain` / `night` are stubs.** The modulation logic around them is complete
  and tested; only the detectors are missing.
- **Shadow mode always.** Audio actions are recorded as intent, never played
  (§0 Challenge 3, §8 Stage 3). Wire `audio_sink` and set `shadow_mode=False`
  only after shadow data justifies it.
- **Synthetic clip needs `--depth gt`.** A flat-shaded rectangle gives DA-V2 no
  real geometry to infer, so ground truth is substituted to isolate the
  filter/state mechanics. Real clips use real depth.

---

## Ready for a real dashcam clip

```bash
python -m headway.run_clip your_dashcam.mp4 --v-host <actual m/s>
```

Then check, in order:

1. `annotated.mp4` — is the box on the correct vehicle, and does it stay there?
2. `summary.json` → `measured_innovation_frac_1sigma` — re-fit `R_REL_FRAC`.
3. `summary.json` → `qwen_anchor_calls` and DEGRADED occupancy — heavy DEGRADED
   means depth_conf or track_quality is too low for real footage.
4. `transitions.jsonl` — did warnings fire where you'd have wanted them?

Expect the first real clip to need `HFOV_DEG` and `CAMERA_HEIGHT_M` set to the
actual camera before the corridor accepts the right vehicle.
