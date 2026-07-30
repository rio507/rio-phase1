# headway — Stage 0 deterministic core

Camera-first time-headway and gap-trend monitoring for RIO.

- Pipeline and filtering: [`docs/headway_design.md`](../docs/headway_design.md)
- Warning logic: [`docs/warning_logic_v2.md`](../docs/warning_logic_v2.md), which
  **supersedes §6** of the design doc. Everything else in the design doc (§3
  maths, §4 filtering, §5 lead selection and coasting) remains in force.

> **All thresholds are PROVISIONAL prototype values** — engineering starting
> points for shadow-mode tuning, **not validated safety thresholds**, and must
> never be represented as such. They live in one block at the top of `state.py`.

Self-contained by design — nothing here imports `app.py`, `vision.py` or
`llm_interface.py`, so the headway loop can be run, tuned and broken without
touching the live RIO voice service.

---

## The two rules

**1. Qwen never fires a warning.** It proposes *which pixels* to measure; a
geometric corridor check vetoes it, and every warning is a pure function of
filtered numbers. `state.py` imports only `math` and `dataclasses` — the
self-test asserts that, which is what makes the firewall auditable rather than
aspirational.

**2. τ coaches; TTC saves.** τ describes *following posture* and drives the five
bands. The urgent decision is driven by TTC, which cancels monocular scale bias
and catches the actual emergency. They are **orthogonal**: `tick()` returns a
band *and* an independent urgent flag, so τ = 2.6 s with TTC = 1.8 s still fires
(v2 §0 Challenge 1), and a stable τ = 0.9 s tailgate does not.

---

## Pipeline

```
frame ──► lanes.py    UFLDv2 finds the ego lane        (~2 ms, every frame)
          │           → LaneCorridor, or the static trapezoid if unsure
          ▼
          detect.py   RF-DETR: every road user in view (~5 ms, every frame)
          │           Apache-2.0. No LLM anywhere on this path.
          ▼
          membership  overlap ≥40% of box-bottom edge → in lane   (~0.5 ms)
          │           held 0.5 s → lead-eligible; nearest wins
          │           rising overlap + <40 m → early merge promotion
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
          state.py    τ bands ⟂ TTC urgent → voice policy
```

| File | Role |
|---|---|
| `depth.py` | Depth Anything V2 Metric-**Small** loader (Apache-2.0 only), `depth_map()`, `roi_depth()` |
| `tracker.py` | CSRT wrapper, `init()` / `update()` → box + quality |
| `lanes.py` | **UFLDv2 lane geometry** — `detect_lanes()`, ego-pair selection, `LaneDriftMonitor` (logs only) |
| `membership.py` | **In-lane membership** — bottom-edge overlap, hysteresis + dwell, merge promotion, lead selection |
| `detect.py` | **RF-DETR** candidate source (Apache-2.0), ego-bonnet filter |
| `anchor.py` | `EgoCorridor` / `LaneCorridor` + `build_corridor()` switch. Its Qwen anchor is now used only by `run_clip.py` |
| `filter.py` | Kalman on the gap, innovation gating, `--tune` harness for Q |
| `state.py` | **Warning logic v2** — five τ bands, TTC urgent, voice policy (pure) |
| `run_clip.py` | Stage 0 harness + synthetic clip generator |
| `selftest.py` | 284 spec-compliance checks, no GPU, ~1 s |

---

## Warning logic v2 at a glance

**Five τ bands** (v2 §1), entered at the threshold, exited only 0.2 s of τ beyond
it (v2 §2):

| Band | τ | Exit back above |
|---|---|---|
| COMFORTABLE | > 2.5 | — |
| NORMAL | 2.0 – 2.5 | 2.7 |
| GETTING_UNSAFE | 1.5 – 2.0 | 2.2 |
| UNSAFE | 1.0 – 1.5 | 1.7 |
| CRITICAL | < 1.0 | 1.2 |

**System states:** `DEGRADED` (confidence < 0.4, or host speed stale),
`LOST` (track gone > 1.0 s), `SUPPRESSED_LOW_SPEED` (< 5 m/s — τ coaching off,
**TTC urgent stays armed**).

**Urgent overlay:** `TTC < 2.5 s` **and** trend `RAPIDLY_SHRINKING`, from *any*
band.

**Voice policy** (v2 §4). Cooldowns gate voice only — overlay and logs always update.

| Band | INCREASING / STABLE | SLOWLY_SHRINKING | RAPIDLY_SHRINKING |
|---|---|---|---|
| > 2.0 | silent | silent | silent (TTC path watches) |
| GETTING_UNSAFE | silent | calm line (TTS, 30 s) | `closing_fast.wav` (15 s) |
| UNSAFE | silent + log | `too_close.wav` (15 s) | `brake.wav` if TTC < 4 s, else `too_close.wav` |
| CRITICAL | silent + log, arms buffer | `too_close.wav` | `brake.wav` |

**Confirmation** (v2 §3): escalate 3 frames · de-escalate 12 · urgent 2 (high
confidence) or 3 · recover from LOST/DEGRADED 6. Never one frame.

**Confidence** (v2 §8) is a weighted **sum** — `.30` valid-pixel ratio, `.30`
(1 − ROI variance), `.25` track quality, `.15` anchor freshness — times a coast
decay from design §5. Tiers: `< 0.4` DEGRADED · `0.4–0.7` coaching but no urgent
bypass · `≥ 0.7` full.

---

## Install

Already installed on this pod. From scratch:

```bash
pip install 'opencv-contrib-python-headless<5'   # <5: stable CSRT API
# transformers >= 4.57 already provides Depth Anything V2 and Qwen3-VL natively
```

Models come from the HF cache on first use:
`Depth-Anything-V2-Metric-Outdoor-Small-hf` (~100 MB) and, for real-clip
anchoring, `Qwen3-VL-8B-Instruct` (~17 GB).

---

## Quick start

```bash
# Deterministic core checks (no GPU) -- run this first
python -m headway.selftest

# Synthetic clips with embedded ground truth
python -m headway.run_clip --make-synthetic /tmp/shrink.mp4   --scenario shrinking
python -m headway.run_clip --make-synthetic /tmp/tailgate.mp4 --scenario tailgate

# Validate filter + state mechanics against that ground truth
python -m headway.run_clip /tmp/shrink.mp4   --v-host 25 --depth gt --anchor gt
python -m headway.run_clip /tmp/tailgate.mp4 --v-host 25 --depth gt --anchor gt

# A real dashcam clip: real depth, real Qwen anchoring
python -m headway.run_clip dashcam.mp4 --v-host 25

# Kalman Q trade curve
python -m headway.filter --sweep
```

`--v-host` is the **fixed** host speed in m/s for Stage 0 — no GPS or OBD feed
yet, so every τ depends on this being right for the clip. 25 m/s ≈ 56 mph.
TTC does not use it.

### Scenarios

| `--scenario` | What it proves |
|---|---|
| `shrinking` | Walks NORMAL → GETTING_UNSAFE → UNSAFE → CRITICAL → URGENT and back down. The braking phase is exactly the 3 m/s² manoeuvre `filter.py`'s Q is derived against |
| `tailgate` | Stable τ = 0.9 s. Must stay **voice-silent** and log `persistent_tailgate` — the case v2 §0 Challenge 1 exists for |

### Outputs (`runs/<clip-stem>/`)

| File | Contents |
|---|---|
| `annotated.mp4` | Lead box, d, τ, TTC, trend, band, urgent, plus the full-clip timeline bar |
| `transitions.jsonl` | **The audit trail** — every display-state transition + full snapshot |
| `voice.jsonl` | Every voice action, with the band and trigger reason |
| `frames.jsonl` | Per-frame v2 §9 log line — tuning data, not the audit |
| `summary.json` | Band timeline, occupancy, urgent/tailgate totals, voice events, achieved fps |

### Useful flags

| Flag | Why |
|---|---|
| `--depth gt` | Ground-truth depth (synthetic clips) to isolate filter/state logic |
| `--anchor gt\|manual\|qwen` | Anchor source; `manual` needs `--init-box x1,y1,x2,y2` |
| `--anchor-interval N` | Frames between Qwen re-anchors (default 60) |
| `--no-video` | Skip encoding — roughly 2× faster when you only want logs |
| `--max-frames N` | Truncate for quick iteration |

---

## Verified on this pod (L40S, 720p)

| Metric | Value |
|---|---|
| Self-test | **109/109** |
| Full pipeline, real depth + annotated video | **23.5 fps** (target ≥10 Hz ✓) |
| DA-V2 Metric-Small inference | ~12 ms/frame (24.8 M params, fp16) |
| Qwen3-VL anchor, warm | **1.09 s** (design §1 budgets 0.2–0.5 Hz ✓) |
| Kalman lag under 3 m/s² braking | **0.286 s** (design §4 target ~0.3 s ✓) |
| ḋ noise on a steady gap | 0.294 m/s (1σ), converged sd(ḋ) 0.284 m/s |

Shrinking clip, ground truth vs. run (v_host = 25):

```
GT band change 5.5s  -> GETTING_UNSAFE confirmed 5.63s   (3 frames)
GT band change 7.0s  -> UNSAFE         confirmed 7.03s   (3 frames)
GT band change 8.4s  -> CRITICAL       confirmed 8.43s   (3 frames)
GT TTC crosses 2.5s  -> URGENT         confirmed 8.73s
voice: 5.63 closing_fast · 7.03 too_close · 7.20 brake (TTC<4) · 9.23 brake
```

Tailgate clip: **0 voice events**, 417/420 frames logging `persistent_tailgate`,
0 urgent frames.

> Preprocessing, not inference, was the original bottleneck: CPU-side
> resize/normalise cost ~70 ms/frame against ~12 ms for the model, holding the
> loop at 9.5 fps. `depth.py` hands the processor a CUDA tensor instead.

---

## Tuning knobs

### `state.py` — the PROVISIONAL block

Everything tunable is in one block at the top of the file. v2 §10 is the tuning
sheet; every knob is falsifiable from the shadow-mode JSONL.

| Constant | Value | Tune against |
|---|---|---|
| `TAU_ENTER_*` | 2.5 / 2.0 / 1.5 / 1.0 | shadow logs vs "felt close" judgment |
| `HYST_S` | 0.2 | state-flicker count per hour |
| `TTC_URGENT` | 2.5 | zero missed hard-brakes, ~zero false urgents |
| `TTC_UNSAFE_ASSIST` | 4.0 | how eagerly UNSAFE escalates to the urgent clip |
| `CONFIRM_UP/URGENT/DOWN/RECOVER` | 3 / 2 / 12 / 6 | warning latency vs flicker |
| `TREND_DEAD_BAND_MS` | 0.3 (×1.5 above 25 m/s) | trend-class churn |
| `V_MIN_COACH` / `V_SOFT` | 5.0 / 8.0 | stop-and-go annoyance |
| `CONF_FLOOR` / `CONF_FULL` | 0.4 / 0.7 | DEGRADED % on night/rain clips |
| `CONF_W_*` | .30/.30/.25/.15 | which input actually predicts bad measurements |
| `COOLDOWN_CALM_S` / `_STRONG_S` / `URGENT_MIN_GAP_S` | 30 / 15 / 2.0 | subjective nag factor |
| `TREND_SIGNIFICANCE_SIGMA` | 1.0 | **not in v2** — see below |
| `COACH_WARMUP_S` | 0.6 | **not in v2** — see below |

### `filter.py` — smoothing

| Constant | Default | Effect |
|---|---|---|
| `SIGMA_A` | `3.0` m/s² | Process noise. **Derived from the spec's own "3 m/s² braking lead" target** |
| `R_REL_FRAC` | `0.005` | Depth noise as a fraction of range. **The main thing to re-fit from real clips** |
| `GATE_SIGMA` | `3.0` | Innovation gate width |
| `MAX_CONSEC_REJECTS` | `3` | Rejects before declaring NEW_LEAD |

**Re-fit `R_REL_FRAC` first on real footage.** `summary.json` reports
`measured_innovation_frac_1sigma` for exactly this. On the synthetic clip it read
`0.009` against the configured `0.005`; expect real dashcam depth to be noisier,
and note the 0.286 s lag figure only holds for the configured R.

### `anchor.py` — geometry (calibrate in Stage 2)

`HFOV_DEG` (60.0) is the **biggest single source of corridor error** — it sets
focal length. Then `CAMERA_HEIGHT_M` (1.3), `CAMERA_PITCH_RAD` (0.0),
`LANE_WIDTH_M` (3.5), `CORRIDOR_MARGIN_M` (0.6).

### `lanes.py` — lane geometry (UFLDv2)

`LANE_CONF_MIN` (0.55) is the switch: at or above it the corridor is built from
detected paint, below it the static trapezoid takes over. `CORRIDOR_MARGIN_M /
LANE_WIDTH_M` becomes `LANE_MARGIN_FRAC` in image space, so the lane corridor
needs no focal-length calibration at all — only the range gate still does.

Drift (advisory log only): `DRIFT_RATIO` (0.70), `DRIFT_HOLD_S` (1.0),
`DRIFT_REARM_RATIO` (0.50), `center_bias` (0.0 — set this once real-drive logs
show the camera's fixed lean; the raw `lane_offset` in the JSONL is what you
read it off).

### `membership.py` — in-lane membership

`MEMBER_ENTER_FRAC` (0.40) / `MEMBER_EXIT_FRAC` (0.25) are the hysteresis band:
a candidate must be substantially in the lane to be adopted, only marginally in
it to be kept. `MEMBER_HOLD_S` (0.5) is the dwell before it can hold the lead
lock. Merge promotion: `MERGE_MIN_SLOPE` (0.25 /s) over `MERGE_WINDOW_S` (0.75),
`MERGE_MIN_OVERLAP` (0.15), `MERGE_MAX_RANGE_M` (40) — and never off the static
trapezoid.

`RANGE_MIN_SAMPLES` (2) is the corroboration guard on lead selection — a rate
bound was tried first and cannot work, because the bad 30 m → 5 m jump (107 m/s)
sits inside the legitimate distribution, whose max is 95–107 m/s.

`CANDIDATE_MAX_UNDETECTED_S` (1.0) is measured from the last *detection*. With
per-frame detection, four missed frames means gone. Association is by **centre
displacement in box-diagonals**, not IoU: at 4 fps a correctly-tracked vehicle
scores IoU 0.10–0.30 against its own previous box, so IoU matching (what
SORT/ByteTrack use at 25–30 fps) mints a new candidate almost every frame.

### `detect.py` — the candidate source

`VARIANT` (`nano`) picks the RF-DETR size; `small` is already on disk and costs
+0.4 ms if nano's recall on distant vehicles ever proves too thin. `SCORE_MIN`
(0.35) is deliberately low — membership decides what matters, not a class
score. The `BONNET_*` constants reject the host car's own bonnet, which the
detector otherwise calls a `car` at zero range dead centre in the lane.

There is no anchor schedule any more. `ANCHOR_MAX_AGE_S`,
`TRACK_QUALITY_REANCHOR`, `CANDIDATE_REFRESH_S` and `ANCHOR_MIN_INTERVAL_S`
were all rationing for Qwen and have been deleted rather than zeroed.

### `tracker.py` — drift detection

`max_area_ratio_per_s` (8.0), `max_shift_frac_per_s` (3.0),
`max_aspect_ratio_change` (0.35). Deliberately generous: these catch the tracker
*jumping to another vehicle*, not fast approach. A lead closing at 9 m/s from
19 m legitimately changes box area ~2.7×/s.

---

## Where this deviates from v2, and why

Four documented departures. Each is marked in the code.

**1. `CRITICAL` is never quieter than `UNSAFE`.** v2 §9's `voice()` matches
`tau_state == UNSAFE` exactly, so CRITICAL falls through to silence — making the
worst band the quietest. §4's table says CRITICAL is "URGENT: Brake." when
RAPIDLY_SHRINKING and "urgent-adjacent" when SLOWLY_SHRINKING. Implemented per
the table, which restores monotonicity.

**2. `persistent_tailgate` logs only on STABLE**, not the whole
INCREASING/STABLE column. §4 note 1 says "*Stable* tailgating", and a driver
whose gap is opening is recovering, not tailgating. Logging those frames
overstated the metric by ~3 s on the shrinking clip.

**3. `TREND_SIGNIFICANCE_SIGMA` (new).** A trend class is believed only once the
estimate is 1σ clear of zero, using the Kalman's own sd(ḋ). The filter's velocity
uncertainty starts at 5 m/s, so early frames carry a ḋ that is pure noise.

**4. `COACH_WARMUP_S` (new).** Coaching voice stays silent for 0.6 s after any
filter init or NEW_LEAD reset. Seeded at ḋ = 0 with P_vv = 25, the filter settles
~0.1 m below a steady truth and the constant-velocity model reads that residual
as a real −0.4 m/s closure held for ~5 consecutive frames at 1.0–1.5σ — which
fired a spurious `too_close` on the stable τ = 0.9 clip. No per-frame test can
separate that from a genuine slow closure, which is why the fix is temporal.
v2 §5's 3-frame NEW_LEAD hold does not cover it: confirming a *band* is fast,
converging a *velocity* is not. **URGENT is deliberately not warm-up gated** —
an emergency during warm-up is still an emergency.

Also note: **`rain` and `night` are now inert.** v1 scaled thresholds ×1.25 for
them; v2 replaces that section, and its only threshold modulation is the
low-speed relaxation in §7, with no rain/night term in the §8 confidence formula.
The `Context` fields survive as declared stubs so the plumbing is ready when
detectors land — but under v2 they change nothing. Flag if that was not intended.

---

## Stage 0 limitations

- **Fixed host speed.** τ is only as good as `--v-host`. TTC is unaffected.
- **No calibration.** Depth carries an uncalibrated scale bias (design §0
  Challenge 2). τ inherits it; TTC cancels it. Cone calibration is Stage 2.
- **Straight corridor — only when the paint is unreadable.** UFLDv2 now bounds
  the corridor with the real ego-lane lines, which fixes the curve weakness
  (design §7.3) wherever lane confidence clears `LANE_CONF_MIN`. Below it the
  straight trapezoid is back, and there is still no IMU, so the `yaw_rate` bend
  path remains implemented and unused. Measured on a winding, glare-heavy
  mountain clip: 71.5% of frames on lane geometry, 28.5% fallen back.
- **The trapezoid may not veto an established track.** It filters fresh Qwen
  proposals only. On a bend it reads a correctly-tracked lead as out-of-lane for
  many consecutive frames, so `lead_corridor_check()` gives that vote to the
  lane corridor alone.
- **~~Enumeration is decode-bound~~ — FIXED by RF-DETR (§9 graduation).** Qwen
  enumeration cost 0.6–1.5 s per call and forced a token cap, a 1 s anchor
  floor and a 5 s candidate refresh. RF-DETR Nano runs in ~5 ms, so all of that
  is gone and detection is per-frame. Whole loop: **p95 20 ms at 4 fps, 12×
  headroom, ~50 fps sustainable.**
- **Depth is now the most expensive stage** (7 ms of a 22 ms frame), followed
  by RF-DETR (6 ms) and UFLDv2 (2.5 ms). Nothing on the path is near the
  budget; Stage 1's TensorRT work should start with DA-V2.
- **DA-V2 collapses under windscreen veiling glare, confidently.** Measured on
  the winding clip: low sun flaring the screen made the model read the whole
  scene as one near surface — the entire depth map maxing at 10–12.6 m — and it
  reported a lead at **5.19 m for a real ~31 m gap with depth_conf 0.98**. No
  statistic taken from the depth map can catch that, because the model and its
  own confidence are wrong together. `depth.frame_trust()` gates on the IMAGE
  instead (black level: veiling glare leaves nothing dark), and a refused frame
  is treated as *no* depth so the Kalman coasts. Those three frames now read
  30.6/30.5/30.3 m with confidence decaying to 0.006, i.e. DEGRADED.
  - **The gate is not complete, so there is a second line of defence.**
    `GLARE_BLACK_LEVEL = 90` was chosen for perfect specificity (0 false
    positives in 410 frames; the clean clip peaks at 30), which means a
    *partially* glared frame with a dark object in shot slips through.
    `membership.RANGE_MIN_SAMPLES` catches what the gate misses: a candidate
    needs ≥2 corroborated range samples to take the lead lock, so the first
    reading after a blind spell cannot. That removed all three false leads
    (10.0 m, 5.12 m, 10.67 m) without delaying a single real acquisition or the
    merge promotion. `glare_p01` is logged on every frame so the threshold can
    be re-derived from real drives, the same way `DRIFT_RATIO` is.
  - **What it looks like when both fire:** the Kalman coasts and confidence
    decays to 0, so the band goes stale and the loop is silent. The *displayed*
    distance drifts with the coast and can read low — it is flagged by
    `confidence 0.0`, but the overlay does not currently grey it out.
  - A gate on the depth *range* was tried first and rejected: 32% of the winding
    clip has a 95th-percentile depth under 20 m and most of it is correct — a
    redwood corner genuinely has nothing 50 m away. That gate would blind the
    system in the tightest terrain.
- **No LLM anywhere in the headway path, live or offline.** `run_clip.py`
  defaults to `--anchor detr` too: RF-DETR + the corridor, through the same
  `LeadAnchor.select_from()` the live loop uses. `--anchor qwen` is kept for
  comparison, but the default no longer needs 17 GB of VLM resident to
  annotate a clip, and it is deterministic — which for an offline harness
  matters as much as the speed.
- **Lane departure is logged, never spoken.** `lane_drift` events go to the
  session JSONL and stop there, by design, until real-drive logs justify a
  threshold. It is not steering guidance and must not become any.
- **Shadow mode always.** Voice actions are recorded as intent, never played
  (design §0 Challenge 3). Wire `audio_sink` and set `shadow_mode=False` only
  after shadow data justifies it. The urgent clips must be **pre-rendered** —
  a TTS round trip is a crash narration, not a warning.
- **Synthetic clips need `--depth gt`.** A flat-shaded rectangle gives DA-V2 no
  real geometry to infer, so ground truth is substituted to isolate the
  filter/state mechanics. Real clips use real depth.

---

## Ready for a real dashcam clip

```bash
python -m headway.run_clip your_dashcam.mp4 --v-host <actual m/s>   # RF-DETR by default
```

Then check, in order:

1. `annotated.mp4` — is the box on the correct vehicle, and does it stay there?
2. `summary.json` → `measured_innovation_frac_1sigma` — re-fit `R_REL_FRAC`.
3. `summary.json` → DEGRADED occupancy — heavy DEGRADED means the confidence
   inputs are weaker on real footage than on synthetic.
4. `voice.jsonl` — did lines fire where you'd have wanted them, and did any nag?
5. `transitions.jsonl` — band flicker per hour, for tuning `HYST_S`.

Set `HFOV_DEG` and `CAMERA_HEIGHT_M` to the actual camera first, or the corridor
will reject the right vehicle.
