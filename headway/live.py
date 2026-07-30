"""Live headway loop — per-session state for the /headway_frame endpoint.

Design ref: docs/live_headway_v3.md. This is the real-time counterpart to
run_clip.py: the same fast loop (track -> depth -> filter -> classify -> voice),
driven one frame at a time by HTTP instead of by a VideoCapture, and holding its
state per session between calls.

Per frame:
  1. UFLDv2 lane detection -> ego-lane boundaries + corridor      (~2 ms, GPU)
  2. CSRT tracker.update() carries the lead box forward           (~2 ms, CPU)
  3. Depth Anything V2 Metric-Small over the frame                (~15 ms, GPU)
  4. Median depth in the ROI -> range + depth confidence
  5. Kalman update with the REAL dt between frame timestamps
  6. tau / TTC / trend / confidence
  7. LivePolicy -> band + voice decision

Qwen runs on **none** of that. It is called only to (re-)anchor: on the first
frame of a session, when the tracker loses or stops believing the box, and when
the anchor goes stale past ANCHOR_MAX_AGE_S. That is the design's slow loop, and
it is what keeps the per-frame cost at tens of milliseconds instead of seconds.

Model sharing: the anchor path uses whatever Qwen3-VL the app already has
resident, via anchor.set_qwen_provider(). perceive.py installs that provider at
import time; _ensure_qwen_provider() below imports perceive if nothing has. A
second ~17 GB copy of the weights is never loaded.

Lane geometry (headway/lanes.py) runs on every frame, unlike Qwen. At ~2 ms it
is cheap enough to belong on the fast loop, and it has to be there: the corridor
it feeds is what validates the lead, and a corridor that only updated every 20 s
would be describing a bend the car left ten seconds ago. When lane confidence is
low the corridor falls back to the static trapezoid and the loop behaves exactly
as it did before -- see anchor.build_corridor().
"""
import io
import math
import threading
import time

import numpy as np
from PIL import Image

from . import anchor as anchor_mod
from . import depth as depth_mod
from . import lanes as lanes_mod
from . import live_policy as policy_mod
from . import state as v2
from .filter import HeadwayFilter
from .lanes import LaneDriftMonitor
from .live_policy import LivePolicy
from .tracker import LeadTracker

# --- anchor scheduling (v3 spec §1) ---------------------------------------
# Qwen re-anchor when the anchor is older than this. The spec's number. v2 §7.4
# wanted <= 5 s at 12 Hz; at 4 fps a Qwen call is ~1.3 s, i.e. five whole frame
# slots, so 20 s is the concession that keeps the fast path fast. The tracker
# quality gate below is what actually catches a bad box early -- staleness is
# only the backstop.
ANCHOR_MAX_AGE_S = 20.0
# Re-anchor early when CSRT stops looking believable (v2 §7.4). This fires long
# before the staleness timer on any real drift.
TRACK_QUALITY_REANCHOR = 0.35

# --- frame timing ---------------------------------------------------------
NOMINAL_DT_S = 0.25          # the client's ~4 fps cadence; used for frame 1 only
MIN_DT_S = 0.02
MAX_DT_S = 2.0
# A gap longer than this means the session stopped and restarted (tab
# backgrounded, video paused, driver parked). Carrying a Kalman state across it
# would coast a stale gap into the present, so the segment is restarted instead.
SESSION_GAP_RESET_S = 3.0

# Idle sessions are dropped so a long-running server does not accumulate
# trackers and filters for drives that ended without /session/end.
SESSION_IDLE_EVICT_S = 600.0

# A lead that has left the ego lane for this many consecutive frames is treated
# as gone rather than followed out of the corridor. Not one frame: a box jitters
# across a boundary on any lane-splitting truck, and the whole point of the
# tracker is that it survives a bad frame. At 4 fps this is ~0.75 s.
LEAD_OUT_OF_CORRIDOR_FRAMES = 3

# Serialises GPU work across sessions. One driver is the expected case, so this
# costs nothing in practice; it exists so two concurrent sessions cannot race
# depth's lazy module-level load or stack two depth passes on the GPU at once.
_gpu_lock = threading.Lock()

_sessions = {}
_sessions_lock = threading.Lock()

_provider_checked = False


def _ensure_qwen_provider() -> None:
    """Point headway.anchor at the app's resident Qwen3-VL, once.

    perceive.py already does this at import time, so under the live app this is
    a no-op. It matters when live.py is reached first (a self-test, a different
    entry point): without it anchor.py would fall back to loading its own copy
    of the weights, which is exactly what the design forbids.
    """
    global _provider_checked
    if _provider_checked:
        return
    _provider_checked = True
    if anchor_mod.has_qwen_provider():
        return
    try:
        import perceive  # noqa: F401  -- import side effect installs the provider
    except Exception as e:
        print(f"[headway.live] could not install Qwen provider: {e}", flush=True)


# ROI stats -> v2 §8's two separate depth terms. Mirrors run_clip._conf_components;
# kept local so the live path does not import the Stage 0 CLI harness.
ROI_VAR_NORM_SCALE = 0.25


def _conf_components(dstats: dict):
    valid_ratio = float(dstats.get("valid_frac", 0.0) or 0.0)
    if "spread_score" in dstats:
        var_norm = 1.0 - float(dstats["spread_score"])
    else:
        var_norm = min(1.0, float(dstats.get("rel_spread", 1.0)) / ROI_VAR_NORM_SCALE)
    return valid_ratio, max(0.0, min(1.0, var_norm))


def _round(v, nd=3):
    if v is None:
        return None
    v = float(v)
    if not math.isfinite(v):
        return None
    return round(v, nd)


def lead_corridor_check(corridor, box, prev_count):
    """Is the tracked lead still in our lane? -> (consecutive_misses, geo|None).

    ONLY THE LANE CORRIDOR IS ALLOWED TO INVALIDATE AN ESTABLISHED TRACK.
    The trapezoid's documented weakness (§7.3) is bends -- it projects straight
    ahead off a curving road -- so on exactly the geometry where it disagrees
    with the tracker most, it is the one that is wrong. It was only ever
    trusted to filter a *fresh* Qwen proposal, where a bad rejection costs one
    re-query. Letting it also discard a lead the tracker has been holding would
    turn that known weakness into dropped leads mid-corner, which is worse than
    the behaviour this replaced. Measured on the winding-road clip: the
    trapezoid reads the lead as outside for 14 consecutive frames through one
    bend that UFLD tracks cleanly. So when we have fallen back, the count is
    held at zero and the tracker is left alone.

    The measurement is not dropped on the frames that do read as outside,
    either: a corridor edge is not exact, and blanking range for a frame or two
    would inject a fake gap into the Kalman. The count runs up and a later
    frame re-anchors, which is the existing mechanism for "we are following the
    wrong object".
    """
    if box is None:
        return 0, None
    u, v = (box[0] + box[2]) / 2.0, box[3]
    inside, geo = corridor.contains(u, v)
    if getattr(corridor, "source", "static") != "ufld":
        return 0, geo
    return (0 if inside else int(prev_count) + 1), geo


class LiveSession:
    """One drive's worth of live headway state. Not thread-safe; hold `lock`."""

    def __init__(self, key: str, use_qwen: bool = True):
        self.key = key
        self.use_qwen = bool(use_qwen)
        self.lock = threading.Lock()

        self.tracker = LeadTracker()
        self.kf = HeadwayFilter()
        self.policy = LivePolicy()

        self.anchor = None          # LeadAnchor, built once the frame size is known
        self.corridor = None        # this frame's corridor: LaneCorridor or EgoCorridor
        self.base_corridor = None   # the static trapezoid, always kept as fallback
        self.size = None            # (w, h)

        self.drift = LaneDriftMonitor()
        self.lane_error = None      # first lane-detection failure, reported once
        self.n_lane_ufld = 0        # frames whose corridor came from paint
        self.n_lane_static = 0      # ...and frames that fell back
        self.out_of_corridor = 0    # consecutive frames the lead was outside

        self.t = 0.0                # session clock, seconds
        self.last_wall = None
        self.last_frame_t = None
        self.frame_idx = 0
        self.anchor_t = None        # session time of the last successful anchor
        self.reset_t = 0.0          # session time of the last filter/anchor reset
        self.trend = v2.STABLE
        self.created = time.time()
        self.last_used = time.time()

        self.n_anchor_calls = 0
        self.n_frames = 0

    # -- geometry -----------------------------------------------------------
    def _ensure_geometry(self, w: int, h: int) -> None:
        if self.size == (w, h):
            return
        # Frame size changed (rotation, a different upload): the corridor is a
        # projection and every pixel threshold in it is now wrong.
        self.size = (w, h)
        self.base_corridor = anchor_mod.EgoCorridor(w, h)
        self.corridor = self.base_corridor
        self.anchor = anchor_mod.LeadAnchor(
            w, h, use_qwen=self.use_qwen, corridor=self.corridor)
        self.tracker.reset()
        self.drift.reset()

    # -- clock --------------------------------------------------------------
    def _advance_clock(self, frame_t) -> float:
        """Advance the session clock and return dt.

        Prefers the client's frame timestamp when it supplies one: for the video
        test mode the media element's currentTime is the true spacing between
        the frames actually analysed, while wall-clock at the server also counts
        network and queueing. Falls back to arrival time for the live camera.
        """
        wall = time.time()
        if frame_t is not None and math.isfinite(float(frame_t)):
            frame_t = float(frame_t)
            raw = NOMINAL_DT_S if self.last_frame_t is None else (frame_t - self.last_frame_t)
            self.last_frame_t = frame_t
        else:
            raw = NOMINAL_DT_S if self.last_wall is None else (wall - self.last_wall)
            self.last_frame_t = None
        self.last_wall = wall

        gap = raw > SESSION_GAP_RESET_S or raw < 0
        dt = NOMINAL_DT_S if gap else max(MIN_DT_S, min(MAX_DT_S, raw))
        if gap and self.frame_idx > 0:
            self._restart_segment()
        self.t += dt
        return dt

    def _restart_segment(self) -> None:
        """Drop continuity-dependent state after a break in the frame stream."""
        self.tracker.reset()
        self.kf = HeadwayFilter()
        self.anchor_t = None
        self.reset_t = self.t
        self.trend = v2.STABLE
        # A drift excursion measured across a break in the stream was never
        # observed continuously, so its timer goes with everything else.
        self.drift.reset()
        self.out_of_corridor = 0

    # -- main entry ---------------------------------------------------------
    def process(self, image_bytes: bytes, v_host, v_host_age_s, frame_t=None,
                allow_anchor: bool = True) -> dict:
        t_start = time.perf_counter()
        self.last_used = time.time()

        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        w, h = pil.size
        self._ensure_geometry(w, h)
        frame = np.ascontiguousarray(np.asarray(pil)[:, :, ::-1])   # RGB -> BGR
        t_decode = time.perf_counter()

        dt = self._advance_clock(frame_t)
        t = self.t

        # --- speed validity (spec §4). Decided here, applied in the policy. ---
        stale = (v_host is None
                 or not math.isfinite(float(v_host))
                 or (v_host_age_s is not None
                     and float(v_host_age_s) > policy_mod.V_HOST_STALE_S))

        # --- lanes: ego-lane geometry for this frame's corridor -------------
        lane_result = None
        with _gpu_lock:
            try:
                lane_result = lanes_mod.detect_lanes(frame)
            except Exception as e:
                # Lane detection is an upgrade to the corridor, not a
                # dependency of it. If it fails the trapezoid is still there,
                # so the loop keeps running on exactly the geometry it used
                # before UFLDv2 existed. Recorded once, not once per frame.
                if self.lane_error is None:
                    self.lane_error = f"{type(e).__name__}: {e}"
                    print(f"[headway.live] lane detection off: {self.lane_error}",
                          flush=True)

        self.corridor, lane_info = anchor_mod.build_corridor(
            self.base_corridor, lane_result)
        # The anchor validates against whatever this frame's corridor is.
        self.anchor.corridor = self.corridor
        if self.corridor.source == "ufld":
            self.n_lane_ufld += 1
        else:
            self.n_lane_static += 1
        if self.lane_error:
            lane_info["lane_error"] = self.lane_error

        lane_conf = float(lane_info.get("lane_conf") or 0.0)

        # --- lane departure: ADVISORY LOG ONLY, never a voice line ----------
        drift = self.drift.update(lane_result, w, h, t, lane_conf)
        t_lanes = time.perf_counter()   # covers detect + corridor build + drift

        # --- depth: one pass, shared by the anchor and the ROI measurement ---
        depth_full = None
        depth_err = None
        with _gpu_lock:
            try:
                depth_full = depth_mod.depth_map(frame)
            except Exception as e:
                # A depth failure costs this frame's range, not the session.
                depth_err = str(e)
        t_depth = time.perf_counter()

        # --- anchor (slow loop): first frame, lost track, drift, or staleness --
        anchor_age = (ANCHOR_MAX_AGE_S + 1.0) if self.anchor_t is None else (t - self.anchor_t)
        need_anchor = (not self.tracker.active
                       or self.tracker.box is None
                       or self.tracker.quality < TRACK_QUALITY_REANCHOR
                       or anchor_age >= ANCHOR_MAX_AGE_S
                       # ...or the thing we are tracking has left our lane.
                       or self.out_of_corridor >= LEAD_OUT_OF_CORRIDOR_FRAMES)
        anchor_info = None
        anchored = False
        if need_anchor and allow_anchor and self.use_qwen:
            box, anchor_info = self.anchor.anchor(frame, self.frame_idx, depth=depth_full)
            self.n_anchor_calls += 1
            if box is not None:
                try:
                    self.tracker.init(frame, box)
                    self.anchor_t = t
                    self.reset_t = t          # starts the coaching warm-up
                    anchored = True
                    anchor_age = 0.0
                    self.out_of_corridor = 0
                except ValueError:
                    anchor_info = dict(anchor_info or {}, reason="degenerate_box")
        t_anchor = time.perf_counter()

        # --- track ---------------------------------------------------------
        box, quality = (self.tracker.update(frame, dt) if self.tracker.active
                        else (None, 0.0))

        # --- lead validation against this frame's corridor -------------------
        # Cheap, and the reason lane detection runs every frame rather than at
        # anchor time. A lead that changes lane, or one the tracker has slid off
        # onto roadside scenery, shows up here within a few frames.
        #
        # ONLY THE LANE CORRIDOR IS ALLOWED TO INVALIDATE AN ESTABLISHED TRACK.
        # The trapezoid's documented weakness (§7.3) is bends -- it projects
        # straight ahead off a curving road -- so on exactly the geometry where
        # it disagrees with the tracker most, it is the one that is wrong. It
        # was only ever trusted to filter a *fresh* Qwen proposal, where a bad
        # rejection costs one re-query. Letting it also discard a lead the
        # tracker has been holding would turn that known weakness into dropped
        # leads mid-corner, which is worse than the behaviour this replaced.
        # Measured on the winding-road clip: the trapezoid reads the lead as
        # outside for 14 consecutive frames through one bend that UFLD tracks
        # cleanly. So when we have fallen back, the tracker is left alone.
        #
        # The measurement is NOT dropped on the frames the lead reads as
        # outside either: a corridor edge is not exact, and blanking range for a
        # frame or two would inject a fake gap into the Kalman. The count runs
        # up and a later frame re-anchors, which is the existing mechanism for
        # "we are following the wrong object".
        self.out_of_corridor, lead_geo = lead_corridor_check(
            self.corridor, box, self.out_of_corridor)

        # --- measure -------------------------------------------------------
        z, depth_conf, dstats = float("nan"), 0.0, {}
        if box is not None and depth_full is not None:
            z, depth_conf, dstats = depth_mod.roi_depth(depth_full, box)

        # --- filter --------------------------------------------------------
        snap = self.kf.step(None if (z is None or not np.isfinite(z)) else z,
                            depth_conf, dt)
        if snap["new_lead"]:
            self.reset_t = t

        d, d_dot = snap["d"], snap["d_dot"]
        v_for_tau = 0.0 if stale else float(v_host)
        tau = v2.compute_tau(d, v_for_tau) if not stale else float("inf")
        ttc = v2.compute_ttc(d, d_dot)

        # Hand the classifier the filter's own sd(d_dot): an unconverged velocity
        # must read as STABLE, not as a trend (v2's TREND_SIGNIFICANCE_SIGMA).
        p_vv = snap.get("P_vv")
        d_dot_sigma = math.sqrt(p_vv) if p_vv and p_vv > 0 else None
        self.trend = v2.classify_trend(d_dot, v_for_tau, self.trend,
                                       d_dot_sigma=d_dot_sigma)

        valid_ratio, var_norm = _conf_components(dstats)
        confidence = v2.compute_confidence(valid_ratio, var_norm, quality,
                                           anchor_age, snap["coast_age"],
                                           lane_conf=lane_conf,
                                           corridor_source=self.corridor.source)

        # --- policy --------------------------------------------------------
        tick = self.policy.tick(
            tau=tau, v2_trend=self.trend, confidence=confidence,
            v_host=(None if stale else float(v_host)), v_host_stale=stale,
            t=t, since_reset_s=(t - self.reset_t),
            new_lead=snap["new_lead"], track_lost=(box is None),
        )

        self.frame_idx += 1
        self.n_frames += 1
        t_end = time.perf_counter()

        return {
            "ok": True,
            "session_key": self.key,
            "frame_idx": self.frame_idx - 1,
            "t": round(t, 4),
            "dt": round(dt, 4),

            "lead_box": ([round(float(v), 1) for v in box] if box else None),
            "distance_m": _round(d, 2),
            "d_dot_ms": _round(d_dot, 3),
            "tau_s": _round(tau, 3),
            "ttc_s": _round(ttc, 3),

            "band": tick["band"],
            "band_entered": tick["band_entered"],
            "prev_band": tick["prev_band"],
            "trend": tick["trend"],
            "trend_detail": tick["trend_detail"],
            "urgency": tick["urgency"],
            "speak": tick["speak"],
            "voice_reason": tick["voice_reason"],
            "voice_line": tick["voice_line"],

            "confidence": _round(confidence, 3),
            "depth_conf": _round(depth_conf, 3),
            "track_quality": _round(quality, 3),
            "anchor_age_s": _round(anchor_age, 2),
            "anchored": anchored,
            "anchor_info": anchor_info,
            "new_lead": bool(snap["new_lead"]),
            "coast_age_s": _round(snap["coast_age"], 3),
            "track_lost": box is None,

            "v_host": _round(v_host, 3) if v_host is not None else None,
            "v_host_stale": bool(stale),

            "corridor": [[round(float(x), 1), round(float(y), 1)]
                         for x, y in self.corridor.polygon()],
            "corridor_source": self.corridor.source,
            "lane_conf": round(lane_conf, 3),
            "lane_info": lane_info,
            # Every detected lane, not just the ego pair — the overlay draws
            # them all, and a review of a bad corridor needs to see which line
            # the ego pair was picked out of.
            "lanes": [[[round(float(x), 1), round(float(y), 1)] for x, y in pts]
                      for pts in ((lane_result or {}).get("lanes") or [])],
            "lead_geo": lead_geo,
            "lead_out_of_corridor": self.out_of_corridor,
            "lane_offset": drift.get("offset"),
            "lane_drift": drift,

            "image": {"w": w, "h": h},
            "depth_error": depth_err,
            "timing_ms": {
                "decode": round((t_decode - t_start) * 1000, 1),
                "lanes": round((t_lanes - t_decode) * 1000, 1),
                "depth": round((t_depth - t_lanes) * 1000, 1),
                "anchor": round((t_anchor - t_depth) * 1000, 1),
                "track_filter": round((t_end - t_anchor) * 1000, 1),
                "total": round((t_end - t_start) * 1000, 1),
            },
        }


# ---------------------------------------------------------------------------
# Session registry
# ---------------------------------------------------------------------------
def get_session(key: str, use_qwen: bool = True) -> LiveSession:
    _ensure_qwen_provider()
    with _sessions_lock:
        _evict_idle_locked()
        s = _sessions.get(key)
        if s is None:
            s = LiveSession(key, use_qwen=use_qwen)
            _sessions[key] = s
        return s


def reset_session(key: str) -> bool:
    with _sessions_lock:
        return _sessions.pop(key, None) is not None


def _evict_idle_locked() -> None:
    now = time.time()
    for k in [k for k, s in _sessions.items()
              if now - s.last_used > SESSION_IDLE_EVICT_S]:
        _sessions.pop(k, None)


def active_sessions() -> dict:
    with _sessions_lock:
        return {k: {"frames": s.n_frames, "anchor_calls": s.n_anchor_calls,
                    "band": s.policy.band, "age_s": round(time.time() - s.created, 1),
                    "lane_ufld_frames": s.n_lane_ufld,
                    "lane_static_frames": s.n_lane_static,
                    "lane_ufld_pct": (round(100.0 * s.n_lane_ufld
                                            / max(1, s.n_lane_ufld + s.n_lane_static), 1)),
                    "lane_drift_events": s.drift.n_events,
                    "lane_error": s.lane_error}
                for k, s in _sessions.items()}
