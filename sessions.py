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

# Last time each open session was heard from: a heartbeat, a frame, anything at
# all tagged with its id.
#
# WHY THIS EXISTS. A drive ends in one of two ways. The tidy way is
# /session/end, sent by the dashboard when the driver taps End Drive or the page
# is closed. The other way is that the client simply stops being there -- a
# phone that ran out of battery, a tab closed while the laptop was asleep, a
# browser that never got to fire pagehide -- and nothing arrives to say so.
#
# Before this, such a session stayed open forever: an fd on a JSONL nobody was
# writing to, a tracker and Kalman filter in headway.live, a frame ring holding
# pictures of a road the car left an hour ago. Worse, a stale tab that kept
# POSTing frames against a session the server had forgotten (after a restart)
# was served in full -- new state created on demand, GPU work done, events
# spilled into stray-<id>.jsonl -- for a drive nobody was on.
#
# 54 sessions on this pod, 6 of them never closed, 16 stray files.
_last_seen: dict = {}

# How long a session may go unheard-from before it is treated as abandoned.
#
# Deliberately generous, and it is the client's timer that is short (10 s), not
# this. A backgrounded tab is throttled hard by mobile browsers -- a driver
# taking a call can stop the heartbeat for a minute without having stopped
# driving -- and reaping a live drive would split its log in half and reset the
# headway state mid-road. Twelve missed beats is unambiguous; two is not.
SESSION_ABANDON_S = 120.0


def start_session(metadata: Optional[dict] = None) -> str:
    sid = str(uuid.uuid4())
    path = SESSIONS_DIR / f"{sid}.jsonl"
    f = path.open("a", buffering=1)  # line-buffered
    _open_handles[sid] = f
    _last_seen[sid] = time.time()
    _write(sid, "session_start", {"metadata": metadata or {}})
    return sid


def end_session(session_id: str, reason: str = "closed") -> bool:
    _write(session_id, "session_end", {"reason": reason})
    _last_seen.pop(session_id, None)
    f = _open_handles.pop(session_id, None)
    if f:
        f.close()
        return True
    return False


def touch(session_id: Optional[str]) -> bool:
    """Note that a session is still being driven. -> False if it is not ours.

    False is the answer that matters: it means the client is talking about a
    session this process has no record of, which after a restart is exactly
    what a tab left open from the previous run looks like. The caller is
    expected to refuse the work and tell the client to stop, rather than
    quietly minting fresh state for a drive nobody is on.
    """
    if not session_id or session_id not in _open_handles:
        return False
    _last_seen[session_id] = time.time()
    return True


def idle_for(session_id: str) -> Optional[float]:
    """Seconds since this session was last heard from, or None if unknown."""
    seen = _last_seen.get(session_id)
    return None if seen is None else (time.time() - seen)


def abandoned(timeout_s: float = None) -> list:
    """Open sessions whose client has stopped reporting in."""
    timeout_s = SESSION_ABANDON_S if timeout_s is None else timeout_s
    now = time.time()
    return [sid for sid in list(_open_handles)
            if (now - _last_seen.get(sid, now)) > timeout_s]


def active_sessions() -> dict:
    return {sid: {"idle_s": round(idle_for(sid) or 0.0, 1)}
            for sid in list(_open_handles)}


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
        # Per-candidate membership, every frame. This is the largest thing in
        # the log by some way -- six candidates of ~9 short keys at 4 Hz is
        # roughly 1.5 kB/s, ~5 MB an hour. It is kept in full anyway because
        # tuning MEMBER_ENTER_FRAC / MEMBER_HOLD_S / the merge slope is
        # impossible from summary statistics: you need the frame where a
        # candidate sat at 0.38 for four frames and never got adopted.
        "membership": result.get("membership"),
        "membership_info": result.get("membership_info"),
        "lead_id": result.get("lead_id"),
        "lead_switch": result.get("lead_switch"),
        # Whether this frame's depth was believed, and the black-level
        # statistic behind that call. Logged on EVERY frame, not just the
        # refused ones, because GLARE_BLACK_LEVEL is provisional and the
        # only way to re-derive it is the distribution from real drives.
        "depth_trusted": result.get("depth_trusted"),
        "glare_p01": result.get("glare_p01"),
        "n_detections": len((result.get("membership") or [])),
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


def log_merge_promotion(session_id: Optional[str], result: dict) -> None:
    """A neighbouring vehicle promoted to lead-eligible on a rising overlap.

    Its own event kind for the same reason lane_drift has one: these are rare,
    they are the thing most likely to be tuned wrong, and reviewing them means
    selecting them out of a whole drive. Each record carries the slope and the
    window that justified it, so a false promotion can be argued with rather
    than just noticed.

    Unlike lane_drift this one DOES affect behaviour -- a promoted candidate can
    take the lead lock, and the lead is what the voice path measures. It is
    still not a voice line of its own.
    """
    if not session_id:
        return
    for ev in (result.get("merge_promotions") or []):
        _write(session_id, "merge_promotion", {
            "frame_idx": result.get("frame_idx"),
            "t": result.get("t"),
            "candidate_id": ev.get("candidate_id"),
            "label": ev.get("label"),
            "overlap": ev.get("overlap"),
            "slope_per_s": ev.get("slope_per_s"),
            "window_s": ev.get("window_s"),
            "samples": ev.get("samples"),
            "range_m": ev.get("range_m"),
            "box": ev.get("box"),
            "corridor_source": ev.get("corridor_source"),
            "lane_conf": result.get("lane_conf"),
            "v_host": result.get("v_host"),
            "band": result.get("band"),
        })


def log_nav(session_id: Optional[str], event: str, payload: Optional[dict] = None) -> None:
    """One navigation event — route_set, maneuver_approach, maneuver_complete,
    reroute, arrived — under a single kind so a whole drive's navigation can be
    selected out of the stream in one pass.

    One kind rather than one per event, unlike lane_drift and merge_promotion:
    those are rare findings pulled out of a 4 Hz stream, whereas these ARE the
    stream — a handful of events per turn, and their order is the thing under
    review. Splitting them across kinds would make "did the far tier fire before
    the near tier" a join instead of a read.

    The payload is whatever the provider and the progression engine recorded for
    that event; it always carries the route_id, so events survive a reroute
    without becoming ambiguous.
    """
    if not session_id:
        return
    body = {"event": event}
    if payload:
        body.update(payload)
    _write(session_id, "nav", body)


def log_talk(session_id: Optional[str], transcript: str, reply: str, audio_bytes_len: int, latency_ms: float) -> None:
    if not session_id:
        return
    _write(session_id, "talk", {
        "transcript": transcript,
        "reply": reply,
        "audio_bytes": audio_bytes_len,
        "latency_ms": round(latency_ms, 1),
    })


def log_live(session_id: Optional[str], event: str, payload: Optional[dict] = None) -> None:
    """One event from the live speech-to-speech conversation.

    Its own kind, and a thin one on purpose. The live session's audio never
    reaches this process — the browser talks to the model directly — so what
    can be recorded here is the shape of the conversation rather than its
    content: when a session opened, when RIO reached for the reasoning model,
    what she asked it, how long it took and whether it worked.

    That is exactly the part worth reviewing. "RIO went quiet for eight seconds
    somewhere near the exit" is a question about escalation timing, and it is
    unanswerable from a transcript.
    """
    if not session_id:
        return
    body = {"event": event}
    if payload:
        body.update(payload)
    _write(session_id, "live", body)


def log_visual_qa(session_id: Optional[str], meta: dict) -> None:
    """One visual conversation turn, every stage of it.

    Its own kind because it is the only event in the file that records a
    DECISION CHAIN rather than a measurement: what the driver asked, how it was
    classified and by what (rules or model), which frame was chosen out of the
    ring and how much better it scored than simply taking the newest one, which
    track the reference resolved to and by which method, what the crop actually
    contained, what went to the model, and how long every stage took.

    That chain is the only way to argue with a bad answer. "RIO described the
    wrong car" has at least four distinct causes -- misrouted, wrong frame,
    wrong track, or the model misreading a good crop -- and they are
    indistinguishable from the reply alone.

    NO IMAGES. Not the frame, not the crop, not a thumbnail. Only their ids,
    sizes and scores. Images are written only when config.RING_PERSIST is on,
    by framebuf.persist, and then this record carries the PATH -- so a review
    can find them if they were kept and can see that they were not if they
    weren't.
    """
    if not session_id or not meta:
        return
    route = meta.get("route") or {}
    _write(session_id, "visual_qa", {
        "talk_id": meta.get("talk_id"),
        "question": meta.get("question"),
        # -- classification
        "request_type": route.get("request_type"),
        "route_method": route.get("method"),
        "route_confidence": route.get("confidence"),
        "object_reference": route.get("object_reference"),
        "phase_b_type": route.get("phase_b"),
        # -- frame selection
        "frame_selection": meta.get("frame_selection"),
        "object_frame_selection": meta.get("object_frame_selection"),
        "ring": meta.get("ring"),
        # -- grounding
        "resolution": meta.get("resolution"),
        "referent": meta.get("referent"),
        "referent_source": meta.get("referent_source"),
        "referent_visible": meta.get("referent_visible"),
        "enrichment": meta.get("enrichment"),
        "crop": meta.get("crop"),
        "crop_source": meta.get("crop_source"),
        # -- the request and what came back
        "request": meta.get("request"),
        "reply": meta.get("reply"),
        "error": meta.get("error"),
        "visual_unavailable": meta.get("visual_unavailable"),
        # -- retained images, if the operator turned that on
        "frame_path": meta.get("frame_path"),
        "crop_path": (meta.get("referent") or {}).get("last_crop_path"),
        "timing_ms": meta.get("timing_ms", {}),
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
