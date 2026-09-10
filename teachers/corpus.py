"""The record — one row per keyframe, written to be read in a year.

WHAT IS BEING WRITTEN, AND FOR WHOM
-----------------------------------
Not a debug log. A row here is a training example: four frames of road, the
motion that led into them, what two purpose-built driving models said about
them, which tracked object each was talking about, what RIO's deterministic
pipeline had computed at the same instant, and whether RIO said anything out
loud. That set is chosen so a later reader can ask the questions that matter --
did the models see the hazard RIO's geometry found, did they see one it missed,
and what does a good spoken line look like at an instant like this -- without
needing this code, this dashboard, or anyone's memory of the drive.

Which is why teachers/schema.py exists and why validate_row() is asserted in
the selftest against real rows rather than mocks.

LAYOUT
------
    training_data/teachers/<session_id>/
        keyframes.jsonl          one JSON object per line, seq order
        frames/<kf_id>/0.jpg     the four frames, slot order, oldest first
        ...

JSONL rather than one file per row because a drive produces hundreds and the
natural read is a scan. Frames beside it rather than inline because base64 in a
JSONL triples the file and makes it unreadable with `head`.

WHAT IS NOT WRITTEN
-------------------
Nothing that is not in the schema, and in particular no partial rows: a
keyframe only becomes a row once both teachers have answered it, so every line
in the file has both columns. A model that failed writes a row with ok:false
and its error, which is data; a model that has not answered yet writes nothing
at all, which is why the file can lag the drive by a few seconds at the end.
"""
import json
import os
import threading
import time
from pathlib import Path

import config

from . import schema

_lock = threading.Lock()
_handles = {}       # session -> open file handle
_stats = {"rows": 0, "frames": 0, "bytes": 0, "errors": 0,
          "rows_without_readings": 0, "last_error": None}


def _root() -> Path:
    return Path(getattr(config, "TEACHER_CORPUS_DIR",
                        "/workspace/rio-phase1/training_data/teachers"))


def session_dir(session_id: str) -> Path:
    return _root() / str(session_id or "default")


def _handle(session_id: str):
    """One line-buffered append handle per drive, opened lazily.

    Line-buffered on purpose, matching sessions.py: a drive that ends by the
    phone's battery dying should still leave every completed row on disk.
    """
    h = _handles.get(session_id)
    if h is None or h.closed:
        d = session_dir(session_id)
        d.mkdir(parents=True, exist_ok=True)
        h = (d / "keyframes.jsonl").open("a", buffering=1)
        _handles[session_id] = h
    return h


def write(kf: dict, readings: dict, associations: dict) -> dict:
    """One complete keyframe -> one row (and four JPEGs). -> {"ok", ...}

    NEVER RAISES. It is called from a teacher's worker thread; a full disk or a
    read-only volume costs the corpus, never the panel.
    """
    if not getattr(config, "TEACHER_CORPUS_ENABLED", True):
        return {"ok": False, "reason": "disabled"}
    try:
        return _write(kf, readings, associations)
    except Exception as e:
        with _lock:
            _stats["errors"] += 1
            _stats["last_error"] = f"{type(e).__name__}: {e}"
        print(f"[teachers.corpus] {type(e).__name__}: {e}", flush=True)
        return {"ok": False, "reason": str(e)}


def _write(kf, readings, associations):
    session_id = kf["session_key"]
    d = session_dir(session_id)
    window = json.loads(json.dumps(kf["window"]))     # deep copy; paths go in it

    # --- the pictures ------------------------------------------------------
    # KEPT ONLY WHEN SOMETHING SAW THEM. A row where both teachers failed is
    # still worth writing -- it records that the panel was down at this instant,
    # and with which error -- but its four JPEGs are pictures nothing read.
    #
    # This is not a micro-optimisation. With both services stopped, a drive
    # raises a keyframe every two seconds for its whole length, and keeping the
    # window each time is ~180 MB an hour of road photographs whose entire
    # accompanying content is "connection refused". On a volume with a quota,
    # that is the difference between a shadow feature and an outage.
    n_frames = 0
    any_reading = any(not (readings.get(m) or {}).get("not_run")
                      for m in schema.MODELS)
    if any_reading and getattr(config, "TEACHER_CORPUS_KEEP_FRAMES", True):
        fdir = d / "frames" / kf["kf_id"]
        fdir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(kf["frames"]):
            path = fdir / f"{i}.jpg"
            try:
                path.write_bytes(frame.jpeg)
            except Exception:
                # A frame that cannot be written leaves its slot's path null,
                # which is exactly what the schema means by null there. The row
                # is still worth having.
                continue
            n_frames += 1
            with _lock:
                _stats["bytes"] += len(frame.jpeg)
            # RELATIVE to the session directory, deliberately. An absolute path
            # in a corpus is a path that stops working the moment the corpus is
            # copied somewhere else, which is the only thing corpora ever have
            # done to them.
            window["frames"][i]["path"] = str(
                path.relative_to(d)).replace(os.sep, "/")

    row = {
        "schema": schema.SCHEMA_VERSION,
        "kf_id": kf["kf_id"],
        "seq": kf["seq"],
        "session_id": session_id,
        "t0_wall": round(float(kf["t0"]), 4),
        "t0_session": kf.get("t0_session"),
        "trigger": kf["trigger"],
        "window": window,
        "ego": kf.get("ego"),
        "state": kf["state"],
        "spoken": kf.get("spoken"),
        "readings": {m: _clean_reading(readings.get(m) or schema.blank_reading(m))
                     for m in schema.MODELS},
        "associations": {m: (associations.get(m) or {}) for m in schema.MODELS},
        "tracks": kf.get("tracks") or [],
        "models_ran": [m for m in schema.MODELS
                       if not (readings.get(m) or {}).get("not_run")],
    }
    # Said on the row rather than inferred from the null paths, because "the
    # pictures were not kept" and "the pictures could not be written" are
    # different facts and a corpus reader is entitled to both.
    row["frames_kept"] = bool(n_frames)
    if not any_reading:
        row["frames_skipped_reason"] = "no model produced a reading"

    with _lock:
        _handle(session_id).write(json.dumps(row, ensure_ascii=False,
                                             default=_jsonable) + "\n")
        _stats["rows"] += 1
        _stats["frames"] += n_frames
        if not any_reading:
            _stats["rows_without_readings"] = \
                _stats.get("rows_without_readings", 0) + 1
    return {"ok": True, "kf_id": kf["kf_id"], "frames": n_frames}


# Keys the PANEL puts on a reading for its own bookkeeping. Not the model's,
# not part of the row format, and a corpus reader should not have to wonder
# whether they are.
_WORKING_KEYS = frozenset({"at", "t0", "kf_id", "fresh", "age_s", "not_run"})


def _clean_reading(rec: dict) -> dict:
    """The schema's fields in the schema's order, and everything else in `extra`.

    "Every field the model offers, verbatim" has to survive the schema not
    having thought of it. A service reports things the named fields do not
    cover -- per-stage timings, `cameras: 1`, `frames_as: video`,
    `ego_synthetic` -- and every one of them is a fact about how the reading
    was produced that a training run may want to filter on. Dropping them for
    being unanticipated is exactly the failure a versioned schema is supposed
    to prevent, so they go in `extra` instead and the named fields stay stable.

    The MODEL's own output is never touched either way: `raw` is copied
    straight through.
    """
    out = {k: rec.get(k) for k in schema.READING_FIELDS}
    extra = {k: v for k, v in rec.items()
             if k not in schema.READING_FIELDS and k not in _WORKING_KEYS}
    out["extra"] = extra
    return out


def _jsonable(o):
    """Last resort for anything numpy handed us. Lists and floats, never repr."""
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    return str(o)


def close(session_id: str) -> bool:
    with _lock:
        h = _handles.pop(str(session_id or "default"), None)
    if h is None:
        return False
    try:
        h.close()
    except Exception:
        pass
    return True


def close_all() -> None:
    for sid in list(_handles):
        close(sid)


def stats() -> dict:
    with _lock:
        out = dict(_stats)
    out["open"] = len(_handles)
    out["dir"] = str(_root())
    return out


def read_rows(session_id: str, limit: int = None) -> list:
    """Rows back out again — for the selftest, the replay tool and any reader.

    A corpus nothing can read is a corpus that will turn out to be malformed at
    the worst possible moment, so the reader ships with the writer and the
    selftest uses it.
    """
    path = session_dir(session_id) / "keyframes.jsonl"
    if not path.exists():
        return []
    out = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
            if limit and len(out) >= limit:
                break
    return out
