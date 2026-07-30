"""Session manager for RIO Phase 2.5 behavior lab.

Each live session (a drive, a walk, a passenger test) gets a session_id.
Every /observe and /talk call tagged with that session_id appends one event
to /workspace/rio-phase1/training_data/<session_id>.jsonl.

Markers (good / hazard / too_much / silent_correct / custom) are appended
the same way.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Optional

SESSIONS_DIR = Path("/workspace/rio-phase1/training_data")
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)

# Open writer handles keyed by session_id.
_open_handles: dict = {}


def start_session(metadata: Optional[dict] = None) -> str:
    sid = str(uuid.uuid4())
    path = SESSIONS_DIR / f"{sid}.jsonl"
    f = path.open("a", buffering=1)  # line-buffered
    _open_handles[sid] = f
    _write(sid, "session_start", {"metadata": metadata or {}})
    return sid


def end_session(session_id: str) -> bool:
    _write(session_id, "session_end", {})
    f = _open_handles.pop(session_id, None)
    if f:
        f.close()
        return True
    return False


def mark(session_id: str, tag: str, payload: Optional[dict] = None) -> None:
    body = {"tag": tag}
    if payload:
        body.update(payload)
    _write(session_id, "mark", body)


def log_observe(session_id: Optional[str], frame_bytes_len: int, observation: str, latency_ms: float) -> None:
    if not session_id:
        return
    _write(session_id, "observe", {
        "frame_bytes": frame_bytes_len,
        "observation": observation,
        "latency_ms": round(latency_ms, 1),
    })


def log_perceive(session_id: Optional[str], frame_bytes_len: int, result: dict, latency_ms: float) -> None:
    """Structured perception event — boxes, distances, corridor, caption.

    Kept as its own kind rather than folded into "observe": this is the labelled
    geometry that makes a drive usable as training data, and a consumer reading
    the JSONL should be able to select it without inspecting payload shape.
    """
    if not session_id:
        return
    _write(session_id, "perceive", {
        "frame_bytes": frame_bytes_len,
        "caption": result.get("caption", ""),
        "boxes": result.get("boxes", []),
        "corridor": result.get("corridor", []),
        "image": result.get("image", {}),
        "lead_range_m": result.get("lead_range_m"),
        "timing_ms": result.get("timing_ms", {}),
        "latency_ms": round(latency_ms, 1),
    })


def log_headway(session_id: Optional[str], result: dict, latency_ms: float) -> None:
    """One live headway frame — the Stage 0 shadow log, recorded in-session.

    This is the tuning data the whole v3 knob sheet is falsifiable from, so it
    keeps the *decision* as well as the measurement: `voice_reason` records why
    a frame did or did not speak (band entry / worsening / escalation /
    suppressed-by-cooldown / suppressed-by-confidence / warm-up / low speed).
    A silence with a reason is as informative as an utterance.

    The corridor polygon and the raw anchor reply are deliberately dropped: at
    ~2 Hz for a whole drive they would dominate the file, and neither is needed
    to replay a decision.
    """
    if not session_id:
        return
    _write(session_id, "headway", {
        "frame_idx": result.get("frame_idx"),
        "t": result.get("t"),
        "dt": result.get("dt"),
        "lead_box": result.get("lead_box"),
        "distance_m": result.get("distance_m"),
        "d_dot_ms": result.get("d_dot_ms"),
        "tau_s": result.get("tau_s"),
        "ttc_s": result.get("ttc_s"),
        "band": result.get("band"),
        "band_entered": result.get("band_entered"),
        "prev_band": result.get("prev_band"),
        "trend": result.get("trend"),
        "trend_detail": result.get("trend_detail"),
        "urgency": result.get("urgency"),
        "speak": result.get("speak"),
        "voice_reason": result.get("voice_reason"),
        "voice_line": result.get("voice_line"),
        "confidence": result.get("confidence"),
        "depth_conf": result.get("depth_conf"),
        "track_quality": result.get("track_quality"),
        "anchor_age_s": result.get("anchor_age_s"),
        "anchored": result.get("anchored"),
        "new_lead": result.get("new_lead"),
        "track_lost": result.get("track_lost"),
        "v_host": result.get("v_host"),
        "v_host_stale": result.get("v_host_stale"),
        # Lane geometry: which corridor decided this frame, how sure the paint
        # was, and where in the lane the car sat. The polylines themselves are
        # dropped for the same reason the corridor polygon is -- four lanes of
        # 72 points at 4 Hz would be most of the file -- but the three scalars
        # that explain a decision are kept.
        "corridor_source": result.get("corridor_source"),
        "lane_conf": result.get("lane_conf"),
        "lane_offset": result.get("lane_offset"),
        "lane_fallback_reason": (result.get("lane_info") or {}).get("fallback_reason"),
        "lead_out_of_corridor": result.get("lead_out_of_corridor"),
        "timing_ms": result.get("timing_ms", {}),
        "latency_ms": round(latency_ms, 1),
    })


def log_lane_drift(session_id: Optional[str], result: dict) -> None:
    """A confirmed lane-departure excursion. ADVISORY RECORD ONLY.

    Its own event kind, deliberately: this is the raw material for deciding
    whether lane departure should ever have a voice, and that review needs to
    select drift events on their own and score them against what the drive
    actually looked like. Folding them into the headway stream at 4 Hz would
    bury a handful of events in tens of thousands of frames.

    Nothing consumes this at runtime. No voice line is attached to it, and none
    should be until real-drive logs say the threshold is right.
    """
    if not session_id:
        return
    drift = result.get("lane_drift") or {}
    _write(session_id, "lane_drift", {
        "frame_idx": result.get("frame_idx"),
        "t": result.get("t"),
        "side": drift.get("side"),
        "offset": drift.get("offset"),
        "held_s": drift.get("held_s"),
        "threshold": drift.get("threshold"),
        "lane_conf": result.get("lane_conf"),
        "corridor_source": result.get("corridor_source"),
        "event_index": drift.get("event_index"),
        "v_host": result.get("v_host"),
        "band": result.get("band"),
    })


def log_talk(session_id: Optional[str], transcript: str, reply: str, audio_bytes_len: int, latency_ms: float) -> None:
    if not session_id:
        return
    _write(session_id, "talk", {
        "transcript": transcript,
        "reply": reply,
        "audio_bytes": audio_bytes_len,
        "latency_ms": round(latency_ms, 1),
    })


def is_active(session_id: str) -> bool:
    return session_id in _open_handles


def _write(session_id: str, kind: str, payload: dict) -> None:
    f = _open_handles.get(session_id)
    if not f:
        # Session not active. Drop the event but keep a breadcrumb in a stray file.
        stray = SESSIONS_DIR / f"stray-{session_id}.jsonl"
        with stray.open("a") as g:
            g.write(_format(kind, payload) + "\n")
        return
    f.write(_format(kind, payload) + "\n")


def _format(kind: str, payload: dict) -> str:
    return json.dumps({
        "t": time.time(),
        "kind": kind,
        "payload": payload,
    })
