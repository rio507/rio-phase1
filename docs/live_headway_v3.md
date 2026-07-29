# RIO Live Headway v3 — three bands, automatic voice

**Status: PROVISIONAL prototype values. Engineering starting points for
shadow-mode tuning — NOT validated safety thresholds. Do not represent them as
such anywhere, ever.**

This document **overrides §1 (bands) and §4/§6 (voice policy) of
`warning_logic_v2.md`** for the live drive loop. Everything else v2 specifies —
hysteresis, temporal confirmation, cooldowns, the coaching warm-up, the
confidence floor, the LLM firewall — is preserved in mechanism and retuned to
the three-band ladder.

`warning_logic_v2.md` is **not** superseded as a whole. It still governs
`headway/state.py` and the Stage 0 clip harness (`headway/run_clip.py`), which
are unchanged and still pass their 109-check suite. v3 lives in its own module,
`headway/live_policy.py`.

## Why a separate module rather than a retune of state.py

v3 is not a different set of numbers for v2's policy, it is a different policy.

- v2's coaching is **trend-gated**: GETTING_UNSAFE is *silent* unless the gap is
  shrinking, and a stable tailgate is logged rather than spoken.
- v3's coaching is **entry-gated**: crossing into the band speaks, whatever the
  trend is doing.

Folding both into one class would mean a mode flag threaded through every branch
of the voice table. Two modules — one live, one offline — each readable on its
own. They share the pure maths in `state.py` (`compute_tau`, `compute_ttc`,
`classify_trend`, `compute_confidence`) so there is exactly one implementation of
each.

## 1. Bands

τ = distance / v_host.

| Band | τ (s) | Display | Voice |
|---|---|---|---|
| NORMAL | ≥ 3.0 | dim cyan, unobtrusive | silent |
| GETTING_UNSAFE | 2.0 – 3.0 | amber | **calm line on confirmed entry**, + escalation after 5 s if closing |
| UNSAFE | < 2.0 | red | **pre-rendered alert on confirmed entry** |
| SUPPRESSED | v_host < 5 m/s | dim | silent |
| UNKNOWN | no / stale speed | dim | silent |

Exit hysteresis 0.2 s τ:

```
enter GETTING_UNSAFE at τ < 3.0   exit to NORMAL at τ > 3.2
enter UNSAFE         at τ < 2.0   exit back    at τ > 2.2
```

## 2. Voice

| Line | Trigger | Delivery |
|---|---|---|
| "Beep beep — you're getting a little close there." | confirmed entry into GETTING_UNSAFE **from a better band** | live TTS (`/headway_voice?line=calm`) |
| "Still closing — ease off a touch." | 5 s in GETTING_UNSAFE with the gap still closing | live TTS |
| "You're too close." / "Watch your distance." | confirmed entry into UNSAFE | **pre-rendered clip** |
| "Back off — now." | entry into UNSAFE while RAPIDLY closing, or the gap starting to collapse while already in UNSAFE | **pre-rendered clip** |

The red tier is pre-rendered because an ElevenLabs round-trip measures **267 ms**
on this stack, and the tier exists precisely for the case where that is already
too late. The amber tier is coaching, not an alert, so TTS is fine there.

Entry into GETTING_UNSAFE **from a better band only**. Arriving there by
de-escalating out of UNSAFE is the situation *improving*; congratulating the
driver with a warning is how the feature gets muted.

The two red phrasings alternate, so a repeat is not word-for-word.

## 3. Anti-nag rules

- **Confirmation** — a band entry needs ≥ 2 frames *and* ≥ 0.5 s of consistent
  readings. De-escalation needs 1.0 s. Never single-frame.
- **Hysteresis** — 0.2 s τ on exit, so boundary flicker cannot re-trigger entry.
- **Cooldowns** — 30 s calm tier, 15 s unsafe tier, applied **per tier, not per
  line**. The spec says "same line not repeated within 30 s / 15 s"; per-line
  would let "You're too close." and "Watch your distance." fire back to back and
  satisfy that on a technicality. Per tier is strictly stronger and implies it.
- **Genuine clear re-arms** — continuous NORMAL for > 10 s clears every cooldown.
  Danger that resolves and returns deserves a fresh warning.
- **Worsening always speaks** — **any** confirmed entry into UNSAFE from a real
  τ band bypasses the cooldown: GETTING_UNSAFE → UNSAFE (already warned, got
  worse anyway) *and* the two-band jump NORMAL → UNSAFE (a cut-in collapsing τ
  from 4 s to 1.5 s, which is the clearest case there is for speaking and
  exactly the one a cooldown would swallow).
  SUPPRESSED and UNKNOWN are **not** worsening origins: neither carries any τ
  information and both are entered with no confirmation, so a flaky GPS fix
  bouncing UNKNOWN ↔ UNSAFE could otherwise fire the red tier every second.
  What stays oscillation-protected is the single-band NORMAL ↔ GETTING_UNSAFE
  re-entry, gated by the 30 s calm cooldown.
- **Warm-up neutralises the trend, not the tier** — for 0.6 s after any anchor
  reset or filter re-init, the trend is forced to STABLE. The warm-up exists
  because the Kalman seeds ḋ at 0 with sd 5 m/s and reads its own settling
  residual as a ~−0.4 m/s closure for ~0.57 s: it is a **velocity** artefact.
  v2 could silence the whole decision because v2's coaching was trend-gated, so
  the two were the same thing. v3's coaching is entry-gated and rests on τ — a
  *position* measurement the reset does not corrupt, and one the 2-frame
  confirmation already protects. Silencing the tier here made **every cut-in
  silent**: the reset lands ~0.5 s before the confirmation completes, so at
  2 fps the entry always fell inside the window. Neutralising the trend keeps
  the real protection — no line may be *chosen* from an unconverged ḋ, so a
  warm-up entry gets "You're too close." rather than "Back off — now.", and
  neither the escalation nor the in-band collapse branch can fire.
- **A transient gate defers an entry, it does not delete one** — a band entry is
  a one-shot event, so a confidence dip (or a NEW_LEAD frame) landing on exactly
  the entry frame would otherwise silence the entire occupancy however long it
  lasted. The entry is latched and fires when the gate clears, judged against
  the transition that actually happened. Bounded at `PENDING_ENTRY_MAX_S` = 3 s:
  an entry the system could not judge for that long is no longer news. The band
  displays throughout either way.
- **Low speed** — below 5 m/s the band is SUPPRESSED. τ = d/v explodes as v → 0,
  and a 4 m gap in a parking queue reads as "getting unsafe".
- **Confidence floor** — below 0.4 the band still displays but the voice is
  suppressed. Low certainty suppresses; it never warns cautiously.
- **No speed, no classification** — a missing or > 2 s stale fix yields UNKNOWN,
  never a guess. `v_host` is passed as an empty string rather than 0 when there
  is no fix, because 0 m/s is a real and different state (stopped).

## 4. Cadence — why 0.5 s confirmation works at 2 fps

Confirmation is **time-based with a 2-frame floor**, not frame-count-based. v2
used "3 frames @ 12 Hz", which *is* 0.25 s; the live loop runs at ~2 fps, where a
frame count would mean a completely different latency.

The spec's two constraints — "never single-frame" and "UNSAFE confirmation may
not exceed 0.5 s" — meet exactly at 2 fps: two consecutive frames span 0.5 s, so
a band entry confirms on the second frame and UNSAFE lands *on* its cap rather
than over it. At any higher capture rate the time term dominates and more frames
are required, which is the correct direction.

`CONFIRM_TOL_S = 0.1` absorbs frame-arrival jitter. Without it a second frame
landing at 0.48 s would be pushed to a third and the red tier's latency would
double for no gain in certainty.

## 5. Architecture

```
POST /headway_frame   image + v_host + v_host_age_s + frame_t  (+ ?session_id)
```

Per frame, in `headway/live.py`:

1. `depth_map()` over the frame — DA-V2 Metric-Small, **~10 ms**
2. re-anchor **only** if: first frame, track lost, tracker quality < 0.35, or
   anchor older than 20 s — Qwen3-VL, ~1.2 s, off the steady-state path
3. `tracker.update()` — CSRT carries the box, ~2 ms
4. `roi_depth()` — median depth in the ROI + depth confidence
5. `kf.step(dt)` — Kalman on [d, ḋ] with the **real** dt between frame timestamps
6. τ / TTC / trend / confidence
7. `LivePolicy.tick()` — band + voice decision

**No Qwen call per frame.** That is the whole design, and
`tools/headway_bench.py` is where the claim is checked: p50 anchor cost across a
32-frame run is 0.0 ms.

Response: `lead_box, distance_m, tau_s, ttc_s, band, trend, urgency, speak,
confidence, anchor_age_s, corridor, timing_ms` and the filter/track diagnostics.

`speak` is `null` or `{line, text, audio, tier, reason}` where `audio` is either
`"tts"` or a clip id the browser has preloaded.

State is per session, keyed by `session_id`, evicted after 10 minutes idle and
dropped by `/session/end`. A frame arriving while the session is mid-frame is
**dropped, not queued**: at 2 fps a queued frame would be measured against a dt
that has already passed, corrupting the velocity estimate the warning rests on.

## 6. Logging

Every frame appends one `"headway"` event to the session JSONL, including the
voice decision **and its reason** — `band_entry`, `worsening`, `escalation`,
`suppressed_by_cooldown`, `suppressed_by_confidence`, `suppressed_by_warmup`,
`suppressed_by_low_speed`, `suppressed_by_unknown_speed`, `silent`. A silence
with a reason is as informative as an utterance, and this is the data every knob
below is falsifiable from.

## 7. LLM firewall

Preserved verbatim from v2 §9. `live_policy.py` imports `math` and
`headway.state` and nothing else — `headway/live_selftest.py` asserts it by AST.
Qwen supplies the ROI upstream and never reaches the policy. `/headway_voice`
looks its text up in the policy's own table and refuses anything else, so the
warning channel cannot be turned into a general text-to-speech endpoint.

## 8. Verification

- `python -m headway.live_selftest` — 107 checks. Full pipeline over the
  shrinking-gap synthetic clip at 2 fps (ground-truth depth, for the reason
  below), plus scripted policy scenarios and the firewall audit.
- `python -m tools.headway_bench --clip <mp4> --v-host <m/s>` — the real stack
  over HTTP: real DA-V2, real Qwen anchoring, real session logging, per-frame
  latency.

**Ground-truth depth on synthetic clips is not a shortcut.** The synthetic clip
is a flat-shaded rectangle on a flat road; DA-V2 has no real geometry to infer
and returns 14 m for a true 60 m — at 0.98 confidence, because a uniform ROI is
perfectly *coherent* while being completely wrong. `run_clip.py` offers
`--depth gt` for exactly this. The synthetic clip validates the mechanism; only
real video validates the measurement.

## 9. Tuning sheet

| Knob | Value | Tune against |
|---|---|---|
| τ band floors | 3.0 / 2.0 | shadow logs vs "felt close" judgement |
| Hysteresis | 0.2 s | band-flicker count per hour |
| Confirm up / down | 0.5 s / 1.0 s (≥ 2 frames) | warning latency vs flicker |
| Cooldowns | 30 s calm / 15 s unsafe | subjective nag factor |
| Deferred-entry window | 3.0 s | how stale an entry may be and still be news |
| Genuine clear | 10 s | how often a re-warning feels earned |
| Escalation delay | 5 s | whether the second line lands as helpful or nagging |
| Warm-up | 0.6 s | spurious lines right after a re-anchor |
| Confidence floor | 0.4 | DEGRADED % in night/rain clips |
| V_MIN_COACH | 5 m/s | stop-and-go annoyance |
| Anchor max age | 20 s | drift vs GPU budget |
| Capture cadence | 2 fps | measured p95 is 44 ms, so ~23 fps is available |

## 10. Open questions for road testing

1. **Cut-in latency is dominated by the Kalman gate, not by the policy.**
   Measured end to end through the real filter at 2 fps (`B10`), a τ 4.0 s →
   1.5 s cut-in produces a warning **1.5 s** after it happens: the innovation
   gate rejects the discontinuity for `MAX_CONSEC_REJECTS = 3` frames before
   re-seeding (**1.0 s** of that total), then the band confirms 0.5 s later.
   That reject count is **frame-based**, so it costs 0.25 s at the Stage 0
   12 Hz and 1.0 s at the live 2 fps — the same frames-vs-time mismatch already
   fixed in the policy's confirmation. Making it time-based (or raising the
   capture cadence, which the 44 ms p95 easily allows) would cut cut-in latency
   by roughly two thirds. Not changed here: `filter.py` is v2 code shared with
   the Stage 0 harness, and this is a tuning decision to take on real logs.
2. **The escalation bypasses the calm cooldown by necessity.** It shares the calm
   tier with the entry line that fired 5 s earlier, so without the bypass the
   30 s cooldown would swallow it every time. It is one-shot per band occupancy,
   and an occupancy costs 5 s of continuous amber with the gap closing.
3. **`τ ≥ 3.0` is a wide NORMAL.** At 30 m/s that is a 90 m gap. Whether NORMAL
   should have a COMFORTABLE tier above it is a display question, not a safety
   one, but it will show up in the occupancy stats immediately.
4. **Anchor staleness at 20 s is long.** The tracker-quality gate at 0.35 is what
   actually catches drift; staleness is only the backstop. If real logs show
   drift surviving the quality gate, the fix is a better quality signal, not a
   shorter timer — a Qwen call is 1.2 s of a 500 ms budget.
