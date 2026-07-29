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
