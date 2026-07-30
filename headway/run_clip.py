"""Stage 0 harness — run the full fast loop against a recorded clip.

Design ref: headway_design.md §8 Stage 0 ("recorded video, cloud pod ...
Deliverable: warning timeline overlaid on video").

    python -m headway.run_clip clip.mp4 --v-host 25
    python -m headway.run_clip --make-synthetic /tmp/synth.mp4

Outputs (into --out-dir, default ./runs/<clip-stem>/):
  annotated.mp4     per-frame lead box, d, τ, TTC, trend, state + timeline bar
  transitions.jsonl shadow log: every state transition + full snapshot (§0, §6)
  frames.jsonl      per-frame record (tuning data; transitions.jsonl is the audit)
  summary.json      state timeline, transition count, achieved fps

Stage 0 runs in shadow mode (§0): audio actions are recorded as intent, never played.
"""
import argparse
import json
import math
import os
import time

import cv2
import numpy as np

from . import anchor as anchor_mod
from . import detect as detect_mod
from . import state as state_mod
from .filter import HeadwayFilter
from .state import Context, Measurement, WarningStateMachine
from .tracker import LeadTracker

# BGR overlay colours per state (v2 §1 overlay column).
STATE_COLOR = {
    state_mod.COMFORTABLE: (90, 200, 90),        # none/dim
    state_mod.NORMAL: (190, 190, 120),           # dim cyan
    state_mod.GETTING_UNSAFE: (60, 200, 255),    # amber
    state_mod.UNSAFE: (40, 130, 255),            # orange
    state_mod.CRITICAL: (50, 50, 220),           # red
    state_mod.URGENT: (60, 60, 255),             # urgent overlay
    state_mod.LOST: (150, 150, 150),
    state_mod.DEGRADED: (190, 160, 110),
    state_mod.SUPPRESSED_LOW_SPEED: (140, 120, 100),
}

# Re-anchor early if the tracker stops looking believable (§7.4).
TRACK_QUALITY_REANCHOR = 0.35


# ---------------------------------------------------------------------------
# Depth providers
# ---------------------------------------------------------------------------
class Dav2Depth:
    """Real Depth Anything V2 Metric-Small."""

    name = "dav2"

    def __init__(self):
        from . import depth as depth_mod
        self._depth = depth_mod
        depth_mod.warm()

    def frame_depth(self, frame):
        return self._depth.depth_map(frame)

    def measure(self, frame, box, depth_map, frame_idx):
        return self._depth.roi_depth(depth_map, box)


class GroundTruthDepth:
    """Synthetic depth from the clip's ground truth, for mechanics validation.

    The synthetic clip is a flat-shaded rectangle on a flat road -- DA-V2 has no
    real geometry to infer there, so running it would measure nothing meaningful.
    Substituting ground truth plus representative noise isolates exactly what the
    synthetic test is for: that the filter and state machine behave correctly on
    a known trajectory.
    """

    name = "gt"

    def __init__(self, gt, noise_frac=0.005, noise_abs=0.10, seed=0,
                 depth_conf=0.85, valid_frac=0.95, rel_spread=0.05):
        self.gt = {int(g["frame"]): g for g in gt}
        self.noise_frac = float(noise_frac)
        self.noise_abs = float(noise_abs)
        self.depth_conf = float(depth_conf)
        # Representative ROI statistics so the v2 §8 weighted confidence has real
        # components to work with rather than a single lumped depth_conf.
        self.valid_frac = float(valid_frac)
        self.rel_spread = float(rel_spread)
        self.rng = np.random.default_rng(seed)

    def frame_depth(self, frame):
        return None

    def measure(self, frame, box, depth_map, frame_idx):
        g = self.gt.get(int(frame_idx))
        if g is None or g.get("d_true") is None:
            return float("nan"), 0.0, {"reason": "no_gt", "valid_frac": 0.0,
                                       "rel_spread": 1.0}
        d = float(g["d_true"])
        sigma = max(self.noise_abs, self.noise_frac * d)
        z = d + float(self.rng.normal(0.0, sigma))
        return z, self.depth_conf, {"gt_d": round(d, 3), "sigma": round(sigma, 3),
                                    "valid_frac": self.valid_frac,
                                    "rel_spread": self.rel_spread}


# ---------------------------------------------------------------------------
# Synthetic clip generation
# ---------------------------------------------------------------------------
def _band_from_tau(tau):
    """Nominal v2 §1 band for a tau, ignoring hysteresis.

    Ground-truth labelling only. The live classifier is hysteretic, so a run will
    legitimately lag these labels by the exit margin near a boundary.
    """
    if tau is None or not math.isfinite(tau):
        return state_mod.COMFORTABLE
    if tau < state_mod.TAU_ENTER_CRITICAL:
        return state_mod.CRITICAL
    if tau < state_mod.TAU_ENTER_UNSAFE:
        return state_mod.UNSAFE
    if tau < state_mod.TAU_ENTER_GETTING_UNSAFE:
        return state_mod.GETTING_UNSAFE
    if tau < state_mod.TAU_ENTER_NORMAL:
        return state_mod.NORMAL
    return state_mod.COMFORTABLE


def make_synthetic(path, fps=30.0, width=1280, height=720, v_host=25.0,
                   d0=None, hfov_deg=anchor_mod.HFOV_DEG,
                   cam_h=anchor_mod.CAMERA_HEIGHT_M, scenario="shrinking"):
    """Render a lead vehicle on a scripted, physically consistent trajectory and
    write per-frame ground truth (including the nominal v2 band) alongside.

    scenario="shrinking" -- walks the whole v2 ladder and back down:
      0-3 s    matched speed at 60 m (tau 2.4)   -> NORMAL
      3-6 s    lead brakes at 3 m/s²             -> GETTING_UNSAFE, then UNSAFE
      6-9 s    lead holds -9 m/s relative        -> CRITICAL + URGENT (TTC < 2.5)
      9-12.4 s lead accelerates away at 5 m/s²   -> de-escalation (12 frames)
      12.4-16s gap opens at +8 m/s relative      -> back up the bands

    The 3 m/s² braking phase is deliberately the exact manoeuvre filter.py's Q is
    derived against, so the clip doubles as an end-to-end check of that tuning.

    scenario="tailgate" -- the case v2 §0 Challenge 1 exists for: a *stable* gap
    held at tau = 0.9 s, i.e. deep in CRITICAL with a STABLE trend. Ordinary
    (unwise) dense-traffic driving. Must stay voice-silent and log
    persistent_tailgate instead; barking at it is how ADAS gets muted.
    """
    f_px = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    cx, cy = width / 2.0, height / 2.0
    dt = 1.0 / fps

    if scenario == "tailgate":
        # tau = d / v_host = 0.9 s, held flat.
        d0 = 0.9 * v_host if d0 is None else d0
        duration = 14.0

        def a_rel(t):
            return 0.0
    elif scenario == "shrinking":
        d0 = 60.0 if d0 is None else d0
        duration = 16.0

        def a_rel(t):
            if t < 3.0:
                return 0.0
            if t < 6.0:
                return -3.0      # the spec's braking manoeuvre
            if t < 9.0:
                return 0.0
            if t < 12.4:
                return 5.0       # recovery
            return 0.0
    else:
        raise ValueError(f"unknown scenario {scenario!r}")

    n = int(duration * fps)

    car_w_m, car_h_m = 1.8, 1.5
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"could not open VideoWriter for {path}")

    gt = []
    d, v_rel = float(d0), 0.0
    for i in range(n):
        t = i * dt
        v_rel = float(np.clip(v_rel + a_rel(t) * dt, -20.0, 8.0))
        d = max(d + v_rel * dt, 4.0)

        frame = _render_road(width, height, cy, f_px, i, fps)

        box_h = f_px * car_h_m / d
        box_w = f_px * car_w_m / d
        v_bottom = cy + f_px * cam_h / d
        x1, x2 = cx - box_w / 2.0, cx + box_w / 2.0
        y1, y2 = v_bottom - box_h, v_bottom
        _render_car(frame, x1, y1, x2, y2, braking=(3.0 <= t < 6.0))

        writer.write(frame)
        tau_true = d / max(v_host, 0.5)
        gt.append({
            "frame": i, "t": round(t, 4),
            "d_true": round(d, 4), "v_rel_true": round(v_rel, 4),
            "box": [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)],
            "tau_true": round(tau_true, 4),
            "ttc_true": round(d / max(-v_rel, 1e-6), 4) if v_rel < -0.2 else None,
            "band_true": _band_from_tau(tau_true),
            "trend_true": state_mod.classify_trend(v_rel, v_host),
        })

    writer.release()
    gt_path = os.path.splitext(path)[0] + ".gt.json"
    with open(gt_path, "w") as fh:
        json.dump({"fps": fps, "width": width, "height": height,
                   "v_host": v_host, "hfov_deg": hfov_deg, "scenario": scenario,
                   "camera_height_m": cam_h,
                   "bands": list(state_mod.TAU_BANDS), "frames": gt}, fh, indent=1)
    return path, gt_path, n


def _render_road(width, height, cy, f_px, frame_idx, fps):
    """Sky, road, and perspective lane lines with scrolling dashes."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    horizon = int(cy)
    frame[:horizon] = (168, 148, 126)          # hazy sky
    frame[horizon:] = (68, 68, 72)             # asphalt

    cx = width / 2.0
    cam_h = anchor_mod.CAMERA_HEIGHT_M

    def project(lat_m, fwd_m):
        v = cy + f_px * cam_h / fwd_m
        u = cx + f_px * lat_m / fwd_m
        return int(round(u)), int(round(v))

    # Solid lane edges at ±1.75 m out to 120 m.
    for lat in (-1.75, 1.75):
        pts = [project(lat, fwd) for fwd in np.linspace(4.0, 120.0, 60)]
        cv2.polylines(frame, [np.array(pts, np.int32)], False, (215, 215, 215), 2, cv2.LINE_AA)

    # Centre dashes, scrolling to sell motion (cosmetic only -- nothing measures them).
    phase = (frame_idx / fps) * 25.0
    for k in range(40):
        seg = k * 9.0 - (phase % 9.0)
        if seg < 4.0:
            continue
        a, b = project(0.0, seg), project(0.0, seg + 3.5)
        cv2.line(frame, a, b, (225, 225, 225), 2, cv2.LINE_AA)

    # Shoulder posts give the tracker some non-uniform texture to reject.
    for k in range(24):
        seg = k * 20.0 - (phase % 20.0)
        if seg < 6.0:
            continue
        for lat in (-6.0, 6.0):
            u, v = project(lat, seg)
            top = int(v - f_px * 1.0 / seg)
            if 0 <= u < width and 0 <= top < height:
                cv2.line(frame, (u, v), (u, top), (200, 200, 205), 2, cv2.LINE_AA)
    return frame


def _render_car(frame, x1, y1, x2, y2, braking=False):
    h, w = frame.shape[:2]
    xi1, yi1 = int(round(max(0, x1))), int(round(max(0, y1)))
    xi2, yi2 = int(round(min(w, x2))), int(round(min(h, y2)))
    if xi2 <= xi1 or yi2 <= yi1:
        return

    cv2.rectangle(frame, (xi1, yi1), (xi2, yi2), (48, 42, 120), -1)
    bh = yi2 - yi1
    # Cabin band + shadow: gives CSRT internal structure to lock onto, and makes
    # a uniform box less likely to be tracked by its edges alone.
    cv2.rectangle(frame, (xi1, yi1), (xi2, yi1 + max(1, int(bh * 0.42))), (34, 30, 88), -1)
    cv2.rectangle(frame, (xi1, yi2 - max(1, int(bh * 0.10))), (xi2, yi2), (25, 25, 25), -1)

    lamp = (60, 60, 255) if braking else (40, 40, 150)
    lw = max(2, int((xi2 - xi1) * 0.16))
    lh = max(2, int(bh * 0.14))
    ly = yi1 + int(bh * 0.55)
    cv2.rectangle(frame, (xi1 + 2, ly), (xi1 + 2 + lw, ly + lh), lamp, -1)
    cv2.rectangle(frame, (xi2 - 2 - lw, ly), (xi2 - 2, ly + lh), lamp, -1)
    cv2.rectangle(frame, (xi1, yi1), (xi2, yi2), (15, 15, 15), 1)


# ---------------------------------------------------------------------------
# Confidence inputs
# ---------------------------------------------------------------------------
ROI_VAR_NORM_SCALE = 0.25   # rel_spread at which normalised ROI variance hits 1.0


def _conf_components(dstats):
    """Split ROI stats into v2 §8's two depth terms.

    v2 §8 weights depth_valid_ratio and (1 - roi_variance_norm) separately, so
    the single lumped depth_conf that v1 used is not enough on its own.
    """
    valid_ratio = float(dstats.get("valid_frac", 0.0) or 0.0)
    if "spread_score" in dstats:
        var_norm = 1.0 - float(dstats["spread_score"])
    else:
        var_norm = min(1.0, float(dstats.get("rel_spread", 1.0)) / ROI_VAR_NORM_SCALE)
    return valid_ratio, max(0.0, min(1.0, var_norm))


# ---------------------------------------------------------------------------
# Overlay
# ---------------------------------------------------------------------------
def _fmt(v, unit="", nd=1):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "--"
    return f"{v:.{nd}f}{unit}"


def draw_overlay(frame, rec, corridor, timeline, total_frames, show_corridor=True):
    h, w = frame.shape[:2]
    st = rec["state"]
    color = STATE_COLOR.get(st, (200, 200, 200))

    if show_corridor:
        poly = np.array(corridor.polygon(near_m=5.0, far_m=70.0), np.int32)
        overlay = frame.copy()
        cv2.polylines(overlay, [poly], True, (90, 90, 90), 1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    box = rec.get("box")
    if box:
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        if rec.get("d") is not None:
            cv2.putText(frame, f"{rec['d']:.1f} m", (x1, max(16, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

    panel_h = 132
    cv2.rectangle(frame, (0, 0), (int(w * 0.44), panel_h), (18, 18, 18), -1)
    cv2.rectangle(frame, (0, 0), (int(w * 0.44), panel_h), color, 2)

    cv2.putText(frame, st, (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)
    # tau band shown alongside the display state, because URGENT is an orthogonal
    # overlay in v2 -- the band underneath it still matters for interpretation.
    if rec.get("tau_band") and rec["tau_band"] != st:
        cv2.putText(frame, f"[{rec['tau_band']}]", (14, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    STATE_COLOR.get(rec["tau_band"], (200, 200, 200)), 1, cv2.LINE_AA)
    flags = []
    if rec.get("new_lead"):
        flags.append(("NEW_LEAD", (60, 200, 255)))
    if rec.get("persistent_tailgate"):
        flags.append(("TAILGATE(log)", (140, 200, 255)))
    if rec.get("voice_fired"):
        flags.append(("VOICE", (80, 255, 120)))
    for i, (txt, col) in enumerate(flags):
        cv2.putText(frame, txt, (int(w * 0.44) - 150, 22 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

    line1 = (f"d {_fmt(rec.get('d'), ' m')}   "
             f"ddot {_fmt(rec.get('d_dot'), ' m/s', 2)}   "
             f"tau {_fmt(rec.get('tau'), ' s', 2)}   "
             f"TTC {_fmt(rec.get('ttc'), ' s', 2)}")
    line2 = f"trend {rec.get('trend', '--')}"
    line3 = (f"conf {_fmt(rec.get('confidence'), '', 2)}   "
             f"depth {_fmt(rec.get('depth_conf'), '', 2)}   "
             f"track {_fmt(rec.get('track_quality'), '', 2)}")
    for i, line in enumerate((line1, line2, line3)):
        cv2.putText(frame, line, (12, 62 + i * 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, (225, 225, 225), 1, cv2.LINE_AA)

    _draw_timeline(frame, timeline, total_frames, rec["frame"])
    return frame


def _draw_timeline(frame, timeline, total_frames, cur_frame):
    """The §8 deliverable: the whole clip's warning timeline, always visible."""
    h, w = frame.shape[:2]
    bar_h, pad = 18, 10
    y1 = h - bar_h - pad
    y2 = h - pad
    cv2.rectangle(frame, (pad, y1), (w - pad, y2), (30, 30, 30), -1)

    span = max(1, total_frames)
    inner_w = (w - 2 * pad)
    for i, st in enumerate(timeline):
        x = pad + int(inner_w * i / span)
        xe = pad + int(inner_w * (i + 1) / span)
        cv2.rectangle(frame, (x, y1), (max(xe, x + 1), y2), STATE_COLOR.get(st, (80, 80, 80)), -1)

    cur_x = pad + int(inner_w * min(cur_frame, span) / span)
    cv2.line(frame, (cur_x, y1 - 3), (cur_x, y2 + 3), (255, 255, 255), 1)
    cv2.rectangle(frame, (pad, y1), (w - pad, y2), (90, 90, 90), 1)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def run(video_path, v_host, out_dir=None, depth_source="dav2", anchor_source="detr",
        anchor_interval=anchor_mod.DEFAULT_ANCHOR_INTERVAL, init_box=None,
        max_frames=None, no_video=False, show_corridor=True, gt_path=None):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames:
        total = min(total, max_frames) if total else max_frames
    dt = 1.0 / fps

    stem = os.path.splitext(os.path.basename(video_path))[0]
    out_dir = out_dir or os.path.join("runs", stem)
    os.makedirs(out_dir, exist_ok=True)

    # Ground truth (synthetic clips only).
    gt_frames = None
    if gt_path is None:
        cand = os.path.splitext(video_path)[0] + ".gt.json"
        gt_path = cand if os.path.exists(cand) else None
    if gt_path:
        with open(gt_path) as fh:
            gt_frames = json.load(fh)["frames"]

    if depth_source == "gt":
        if not gt_frames:
            raise SystemExit("--depth gt requires a <clip>.gt.json next to the clip")
        provider = GroundTruthDepth(gt_frames)
    else:
        provider = Dav2Depth()

    corridor = anchor_mod.EgoCorridor(width, height)
    lead_anchor = anchor_mod.LeadAnchor(
        width, height, interval=anchor_interval,
        use_qwen=(anchor_source == "qwen"), corridor=corridor,
    )
    gt_by_frame = {int(g["frame"]): g for g in (gt_frames or [])}

    tracker = LeadTracker()
    kf = HeadwayFilter()
    sm = WarningStateMachine(rate_hz=fps, shadow_mode=True)

    writer = None
    if not no_video:
        writer = cv2.VideoWriter(os.path.join(out_dir, "annotated.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    frames_log = open(os.path.join(out_dir, "frames.jsonl"), "w")
    timeline = []
    band_timeline = []
    tailgate_frames = 0
    urgent_frames = 0
    trend = state_mod.STABLE
    innovations = []
    n_anchor_calls = 0
    last_anchor_t = 0.0
    t_start = time.time()
    idx = 0

    while True:
        ok, frame = cap.read()
        if not ok or (max_frames and idx >= max_frames):
            break
        t = idx * dt

        # --- anchor / track -------------------------------------------------
        need_anchor = (not tracker.active or tracker.box is None
                       or lead_anchor.due(idx) or tracker.quality < TRACK_QUALITY_REANCHOR)
        anchor_info = None
        if need_anchor:
            box = None
            if anchor_source == "gt":
                g = gt_by_frame.get(idx)
                box = tuple(g["box"]) if g else None
                anchor_info = {"reason": "gt"}
                lead_anchor.last_anchor_frame = idx
            elif anchor_source == "detr":
                # RF-DETR supplies the candidates; the corridor picks the lead
                # through the SAME select_from() the Qwen path uses. Default
                # because it is ~5 ms instead of ~1 s, needs no 17 GB of
                # weights resident, and is deterministic -- for an offline
                # annotation harness that reproducibility matters as much as
                # the speed does.
                depth_full = provider.frame_depth(frame)
                dets = detect_mod.detect(frame)["detections"]
                box, anchor_info = lead_anchor.select_from(dets, depth=depth_full)
                lead_anchor.last_anchor_frame = idx
                n_anchor_calls += 1
            elif anchor_source == "manual":
                # Anchor once from --init-box, then let CSRT carry it.
                box = tuple(init_box) if (init_box and not tracker.active) else None
                anchor_info = {"reason": "manual"}
                lead_anchor.last_anchor_frame = idx
            else:
                depth_full = provider.frame_depth(frame)
                box, anchor_info = lead_anchor.anchor(frame, idx, depth=depth_full)
                n_anchor_calls += 1

            if box is not None:
                tracker.init(frame, box)
                last_anchor_t = t

        box, quality = (tracker.update(frame, dt) if tracker.active else (None, 0.0))

        # --- depth ----------------------------------------------------------
        z, depth_conf, dstats = (float("nan"), 0.0, {})
        depth_full = None
        if box is not None:
            depth_full = provider.frame_depth(frame)
            z, depth_conf, dstats = provider.measure(frame, box, depth_full, idx)

        # --- filter ---------------------------------------------------------
        snap = kf.step(None if (z is None or not np.isfinite(z)) else z, depth_conf, dt)
        if snap["accepted"] and snap.get("innovation") is not None and snap["d"]:
            innovations.append(snap["innovation"] / max(snap["d"], 1e-6))

        d, d_dot = snap["d"], snap["d_dot"]
        tau = state_mod.compute_tau(d, v_host)
        ttc = state_mod.compute_ttc(d, d_dot)
        # Hand the classifier the filter's own velocity uncertainty so an
        # unconverged d_dot cannot be read as a trend (see TREND_SIGNIFICANCE_SIGMA).
        p_vv = snap.get("P_vv")
        d_dot_sigma = math.sqrt(p_vv) if p_vv and p_vv > 0 else None
        trend = state_mod.classify_trend(d_dot, v_host, trend, d_dot_sigma=d_dot_sigma)

        track_lost = box is None
        anchor_age = t - last_anchor_t
        ctx = Context(v_host=v_host, v_host_source="FIXED")
        valid_ratio, var_norm = _conf_components(dstats)
        confidence = state_mod.compute_confidence(
            valid_ratio, var_norm, quality, anchor_age, snap["coast_age"])

        m = Measurement(
            d=d, d_dot=d_dot, tau=tau, ttc=ttc, trend=trend,
            confidence=confidence, depth_conf=depth_conf, track_quality=quality,
            track_lost=track_lost, coast_age=snap["coast_age"],
            new_lead=snap["new_lead"], v_host_stale=False,
            anchor_age_s=anchor_age, ctx=ctx,
        )
        tick = sm.tick(m, t=t, dt=dt)

        timeline.append(sm.state)
        band_timeline.append(tick["tau_band"])
        if tick["persistent_tailgate"]:
            tailgate_frames += 1
        if tick["urgent"]:
            urgent_frames += 1
        # v2 §9's per-tick log line: ts, d, d_dot, tau, TTC, trend, tau_state,
        # urgent, confidence, voice_fired -- all present here.
        rec = {
            "frame": idx, "t": round(t, 4), "state": sm.state,
            "tau_band": tick["tau_band"], "urgent": tick["urgent"],
            "system_state": tick["system_state"],
            "box": [round(v, 2) for v in box] if box else None,
            "d": d, "d_dot": d_dot,
            "tau": None if math.isinf(tau) else round(tau, 4),
            "ttc": None if math.isinf(ttc) else round(ttc, 4),
            "trend": trend, "confidence": round(confidence, 4),
            "confidence_tier": tick["confidence_tier"],
            "depth_conf": round(float(depth_conf), 4),
            "track_quality": round(float(quality), 4),
            "voice": tick["voice"], "voice_fired": tick["voice_fired"],
            "logs": tick["logs"],
            "persistent_tailgate": tick["persistent_tailgate"],
            "new_lead": snap["new_lead"], "filter": snap,
            "tick": tick, "depth_stats": dstats,
        }
        if anchor_info:
            rec["anchor"] = anchor_info
        if idx in gt_by_frame:
            g = gt_by_frame[idx]
            rec["gt"] = {"d_true": g["d_true"], "v_rel_true": g["v_rel_true"],
                         "band_true": g.get("band_true"),
                         "tau_true": g.get("tau_true")}
        frames_log.write(json.dumps(rec) + "\n")

        if writer is not None:
            draw_overlay(frame, rec, corridor, timeline, total or (idx + 1), show_corridor)
            writer.write(frame)
        idx += 1

    elapsed = time.time() - t_start
    cap.release()
    frames_log.close()
    if writer is not None:
        writer.release()

    with open(os.path.join(out_dir, "transitions.jsonl"), "w") as fh:
        for tr in sm.transitions:
            fh.write(json.dumps(tr) + "\n")

    with open(os.path.join(out_dir, "voice.jsonl"), "w") as fh:
        for v in sm.voice_log:
            fh.write(json.dumps(v) + "\n")

    summary = _summarise(timeline, sm.transitions, idx, elapsed, fps, dt,
                         innovations, n_anchor_calls, out_dir, video_path,
                         depth_source, anchor_source, v_host, sm, band_timeline,
                         tailgate_frames, urgent_frames)
    with open(os.path.join(out_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def _runs_of(timeline, dt, key="state"):
    runs = []
    for i, st in enumerate(timeline):
        if not runs or runs[-1][key] != st:
            runs.append({key: st, "start_frame": i, "end_frame": i})
        else:
            runs[-1]["end_frame"] = i
    for r in runs:
        r["start_s"] = round(r["start_frame"] * dt, 3)
        r["end_s"] = round((r["end_frame"] + 1) * dt, 3)
        r["duration_s"] = round(r["end_s"] - r["start_s"], 3)
    return runs


def _occupancy_of(timeline, dt):
    occ = {}
    for st in timeline:
        occ[st] = occ.get(st, 0) + 1
    return {k: {"frames": v, "seconds": round(v * dt, 3),
                "pct": round(100.0 * v / max(len(timeline), 1), 2)}
            for k, v in sorted(occ.items(), key=lambda kv: -kv[1])}


def _summarise(timeline, transitions, n_frames, elapsed, fps, dt, innovations,
               n_anchor_calls, out_dir, video_path, depth_source, anchor_source,
               v_host, sm, band_timeline, tailgate_frames, urgent_frames):
    runs = _runs_of(timeline, dt)
    occupancy = _occupancy_of(timeline, dt)

    # Empirical measurement-noise estimate, for re-fitting filter.R_REL_FRAC
    # against real footage (see filter.py's Q/R caveat).
    innov_frac = None
    if len(innovations) > 10:
        innov_frac = round(float(np.std(innovations)), 5)

    return {
        "clip": os.path.abspath(video_path),
        "out_dir": os.path.abspath(out_dir),
        "depth_source": depth_source,
        "anchor_source": anchor_source,
        "v_host_m_s": v_host,
        "frames": n_frames,
        "clip_fps": round(fps, 3),
        "wall_seconds": round(elapsed, 3),
        "achieved_fps": round(n_frames / elapsed, 2) if elapsed > 0 else None,
        "realtime_factor": round((n_frames / elapsed) / fps, 3) if elapsed > 0 and fps else None,
        "meets_10hz_target": bool(elapsed > 0 and (n_frames / elapsed) >= 10.0),
        "transition_count": len(transitions),
        "anchor_calls": n_anchor_calls,
        "state_occupancy": occupancy,
        "state_timeline": runs,
        "tau_band_occupancy": _occupancy_of(band_timeline, dt),
        "tau_band_timeline": _runs_of(band_timeline, dt, key="tau_band"),
        "urgent_frames": urgent_frames,
        "urgent_seconds": round(urgent_frames * dt, 3),
        "persistent_tailgate_frames": tailgate_frames,
        "persistent_tailgate_seconds": round(tailgate_frames * dt, 3),
        "voice_event_count": len(sm.voice_log),
        "voice_events": [
            {"t": v["t"], "band": v["band"], "kind": v["action"]["kind"],
             "line": v["action"].get("line"),
             "clip": v["action"].get("clip"), "text": v["action"].get("text"),
             "reason": v["action"].get("reason")}
            for v in sm.voice_log
        ],
        "log_events": sm.events,
        "transitions": [
            {"t": tr["t"], "from": tr["from"], "to": tr["to"],
             "tau_band": tr["tau_band"], "urgent": tr["urgent"]}
            for tr in transitions
        ],
        "measured_innovation_frac_1sigma": innov_frac,
    }


def _print_summary(s):
    print("\n" + "=" * 72)
    print(f"clip            {s['clip']}")
    print(f"frames          {s['frames']}  @ {s['clip_fps']} fps")
    print(f"depth / anchor  {s['depth_source']} / {s['anchor_source']}")
    print(f"achieved fps    {s['achieved_fps']}  "
          f"({s['realtime_factor']}x realtime, >=10Hz: {s['meets_10hz_target']})")
    print(f"transitions     {s['transition_count']}")
    if s.get("measured_innovation_frac_1sigma") is not None:
        print(f"measured depth noise (1σ, frac of range): "
              f"{s['measured_innovation_frac_1sigma']}  "
              f"[filter.R_REL_FRAC is set to 0.005]")
    print("-" * 72)
    print("state occupancy:")
    for st, v in s["state_occupancy"].items():
        print(f"  {st:<12} {v['seconds']:>7.2f}s  {v['pct']:>6.2f}%")
    print("-" * 72)
    print("tau band timeline (posture):")
    for r in s["tau_band_timeline"]:
        print(f"  {r['start_s']:>7.2f}s -> {r['end_s']:>7.2f}s  "
              f"({r['duration_s']:>6.2f}s)  {r['tau_band']}")
    print("-" * 72)
    print(f"urgent (TTC path):        {s['urgent_seconds']:.2f}s "
          f"({s['urgent_frames']} frames)")
    print(f"persistent_tailgate:      {s['persistent_tailgate_seconds']:.2f}s "
          f"({s['persistent_tailgate_frames']} frames, log-only, never voice)")
    print("-" * 72)
    print(f"voice events: {s['voice_event_count']}")
    for v in s["voice_events"]:
        what = v["clip"] or (v["text"][:40] + "..." if v["text"] else "")
        extra = f"  ({v['reason']})" if v.get("reason") else ""
        print(f"  {v['t']:>7.2f}s  [{v['band']}] {v['kind']:<10} {what}{extra}")
    if s["log_events"]:
        print("-" * 72)
        print("log events (conversational, not warnings):")
        for e in s["log_events"]:
            print(f"  {e['t']:>7.2f}s  {e['event']}")
    print("-" * 72)
    print("display-state transitions:")
    for tr in s["transitions"]:
        u = "  URGENT" if tr["urgent"] else ""
        print(f"  {tr['t']:>7.2f}s  {tr['from']:<22} -> {tr['to']:<22} "
              f"[band {tr['tau_band']}]{u}")
    print("=" * 72)
    print(f"outputs in {s['out_dir']}")


def main():
    ap = argparse.ArgumentParser(description="Stage 0 headway harness (design §8)")
    ap.add_argument("video", nargs="?", help="input clip")
    ap.add_argument("--v-host", type=float, default=25.0,
                    help="fixed host speed in m/s (Stage 0 has no GPS/OBD)")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--depth", choices=("dav2", "gt"), default="dav2")
    ap.add_argument("--anchor", choices=("detr", "qwen", "gt", "manual"),
                    default="detr",
                    help="detr: RF-DETR + corridor (default). qwen: the Qwen3-VL "
                         "enumeration, kept for comparison and for frames where "
                         "you want to ask a VLM what it sees.")
    ap.add_argument("--anchor-interval", type=int, default=anchor_mod.DEFAULT_ANCHOR_INTERVAL)
    ap.add_argument("--init-box", default=None, help="x1,y1,x2,y2 for --anchor manual")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--no-video", action="store_true", help="skip annotated.mp4")
    ap.add_argument("--no-corridor", action="store_true")
    ap.add_argument("--gt", default=None, help="explicit ground-truth json")
    ap.add_argument("--make-synthetic", metavar="OUT_MP4", default=None)
    ap.add_argument("--scenario", choices=("shrinking", "tailgate"), default="shrinking",
                    help="synthetic scenario: closing gap, or stable tau=0.9 tailgate")
    ap.add_argument("--fps", type=float, default=30.0)
    args = ap.parse_args()

    if args.make_synthetic:
        path, gt_path, n = make_synthetic(args.make_synthetic, fps=args.fps,
                                          v_host=args.v_host, scenario=args.scenario)
        print(f"wrote {path} ({n} frames @ {args.fps} fps, scenario={args.scenario})")
        print(f"wrote {gt_path}")
        return

    if not args.video:
        ap.error("video is required unless --make-synthetic is given")

    init_box = None
    if args.init_box:
        init_box = [float(v) for v in args.init_box.split(",")]
        if len(init_box) != 4:
            ap.error("--init-box needs x1,y1,x2,y2")

    s = run(args.video, args.v_host, out_dir=args.out_dir, depth_source=args.depth,
            anchor_source=args.anchor, anchor_interval=args.anchor_interval,
            init_box=init_box, max_frames=args.max_frames, no_video=args.no_video,
            show_corridor=not args.no_corridor, gt_path=args.gt)
    _print_summary(s)


if __name__ == "__main__":
    main()
