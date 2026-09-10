"""The panel — when to ask, who to ask, and what to do with the answers.

This is the ONLY module the rest of RIO touches, and the surface is deliberately
tiny: frames go in, ego samples go in, spoken lines go in, and a JSON state
object comes out for the dashboard. Nothing comes out that anything else in RIO
consumes. See teachers/__init__.py for why that asymmetry is the whole design,
and tools/teacher_firewall_selftest.py for the test that holds it.

CADENCE, AND WHY IT IS NOT JUST A TIMER
---------------------------------------
A floor alone (a keyframe every 2 s) would spend most of its readings on empty
motorway and miss the thing worth having: the instant the band changed, the
instant the driver asked something, the moment the car was thrown by a pothole
or turned into a junction. So there are two sources.

  EVENTS are why this exists. A band change, a nav maneuver, an IMU jolt, a
  driver question -- these are the instants a review actually wants two second
  opinions on, and they are the instants a fixed timer is least likely to land
  on.

  THE FLOOR is what makes the corpus a corpus. Without it a quiet drive
  produces nothing, and "nothing happened" is a class the training set needs as
  much as it needs the near-misses.

  THE MINIMUM GAP is what stops a flapping band turning into a queue. Band
  transitions cluster; without a gap, six keyframes are built in a second, five
  are dropped stale, and the GPU is spent on the drop.

WHAT on_frame() IS ALLOWED TO COST
----------------------------------
It runs on the same path as a headway frame, after the result is computed and
outside the session lock, next to the framebuf push. It is allowed: a handful
of float comparisons, one deque append, and -- only when a keyframe is actually
due -- a list of four object references and a base64 pass on ~100 kB, handed to
another thread. It is NOT allowed to block on the network, take the GPU, or
raise. tools/teacher_timing_selftest.py measures the first of those claims and
fails the build if the fast loop moves.
"""
import threading
import time

import config

from . import associate as assoc_mod
from . import corpus as corpus_mod
from . import egomotion
from . import keyframe as kf_mod
from . import project
from . import schema
from .client import TeacherClient

MODEL_NAMES = ("alpamayo1.5", "cosmos-reason2")

_lock = threading.RLock()
_clients = {}
_sessions = {}      # key -> per-drive panel state
_started = False


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------
def start() -> bool:
    """Bring up the two clients. Idempotent; called from the app lifespan."""
    global _started
    if not getattr(config, "TEACHERS_ENABLED", False):
        return False
    with _lock:
        if _started:
            return False
        _clients["alpamayo1.5"] = TeacherClient(
            "alpamayo1.5", config.TEACHER_ALPAMAYO_URL, _on_reading)
        _clients["cosmos-reason2"] = TeacherClient(
            "cosmos-reason2", config.TEACHER_COSMOS_URL, _on_reading)
        for c in _clients.values():
            c.start()
        _started = True
    # One health probe each, off-thread: it is two loopback requests, and a
    # card that knows which precision each teacher loaded on the first paint is
    # worth more than two seconds of startup.
    threading.Thread(target=refresh_health, daemon=True,
                     name="teachers:health").start()
    return True


def stop() -> None:
    global _started
    with _lock:
        for c in _clients.values():
            c.stop()
        _clients.clear()
        _started = False


def refresh_health() -> dict:
    out = {}
    for name, c in list(_clients.items()):
        out[name] = c.health()
    return out


def _state(key):
    st = _sessions.get(key)
    if st is None:
        st = {
            "seq": 0,
            "last_kf_at": 0.0,
            "pending_trigger": None,
            "last_band": None,
            "readings": {},        # model -> latest reading dict
            "assoc": {},           # model -> latest association dict
            "kf": {},              # kf_id -> the keyframe awaiting readings
            "pending": {},         # kf_id -> {"readings", "assoc"} being gathered
            "spoken": None,        # {"text", "at", "kind"}
            "tally": {"keyframes": 0, "readings": 0, "agree": 0,
                      "agree_eligible": 0, "matched": 0, "actors": 0,
                      "dropped_build": 0},
            "last_build_refusal": None,
        }
        _sessions[key] = st
    return st


# ---------------------------------------------------------------------------
# the frame path
# ---------------------------------------------------------------------------
def on_frame(session_key: str, result: dict, ring) -> bool:
    """One headway result has landed. -> was a keyframe raised?

    NEVER RAISES. Wrapped whole, because the caller is the frame endpoint and a
    shadow feature may not cost a warning. A fault here costs a corpus row.
    """
    try:
        return _on_frame(session_key, result, ring)
    except Exception as e:
        print(f"[teachers] on_frame: {type(e).__name__}: {e}", flush=True)
        return False


def _on_frame(session_key, result, ring):
    if not (_started and getattr(config, "TEACHERS_ENABLED", False)):
        return False
    if not result or result.get("ok") is False:
        return False
    key = str(session_key or "default")
    now = time.time()

    with _lock:
        st = _state(key)

    # Feed the ego history from the speed the WARNING was computed on, not
    # from the page's own GPS -- see egomotion's header for why they must be
    # the same number.
    speed = result.get("speed") or {}
    egomotion.note_speed(key, speed.get("v_ms", result.get("v_host")),
                         speed.get("source") or "none", at=now)

    # --- is this instant worth a keyframe? ---------------------------------
    trigger = None
    band = result.get("band")
    with _lock:
        last_band, st["last_band"] = st["last_band"], band
        pending = st["pending_trigger"]
        st["pending_trigger"] = None
        last_at = st["last_kf_at"]

    if pending:
        trigger = pending
    elif last_band is not None and band != last_band:
        trigger = "band"
    elif (now - last_at) >= float(getattr(config, "TEACHER_KEYFRAME_FLOOR_S", 2.0)):
        trigger = "floor"
    if trigger is None:
        return False
    # The gap applies to EVERY trigger including events: a driver asking three
    # questions in three seconds is one road, and three keyframes of it would
    # be two drops and a reading.
    if (now - last_at) < float(getattr(config, "TEACHER_KEYFRAME_MIN_GAP_S", 1.0)):
        return False

    with _lock:
        seq = st["seq"]
        spoken = st["spoken"]

    ego = egomotion.history(key, float(getattr(ring.latest(), "wall_t", now)))
    kf = kf_mod.build(key, ring, result, trigger, ego=ego, spoken=spoken, seq=seq)
    if not kf.get("ok"):
        with _lock:
            st["tally"]["dropped_build"] += 1
            st["last_build_refusal"] = kf.get("reason")
        return False

    job = kf_mod.to_job(kf)
    with _lock:
        st["seq"] = seq + 1
        st["last_kf_at"] = now
        st["tally"]["keyframes"] += 1
        # Held so the readings can be filed against it when they come back --
        # and bounded, because two teachers can both fail to answer and a dict
        # keyed by keyframe id would otherwise be a slow leak across a drive.
        st["kf"][kf["kf_id"]] = kf
        # Deep enough to outlive the SLOWER teacher. Each keyframe here holds
        # four frame references and the scene at t0; 24 of them is a couple of
        # seconds of ring plus some dicts, and losing one to an eager trim
        # costs a corpus row that both models answered.
        if len(st["kf"]) > 24:
            for old in sorted(st["kf"])[:-24]:
                st["kf"].pop(old, None)
        clients = list(_clients.values())

    for c in clients:
        c.submit(dict(job))
    return True


def raise_event(session_key: str, kind: str) -> None:
    """Something happened that the next frame should be a keyframe about.

    Deliberately a FLAG rather than an immediate keyframe: the event knows the
    instant but not the pixels, and the next frame is at most ~100 ms away with
    a full window behind it. Setting a flag also means an event storm collapses
    into one keyframe instead of racing the frame path.
    """
    if kind not in schema.TRIGGERS:
        return
    with _lock:
        _state(str(session_key or "default"))["pending_trigger"] = kind


def note_spoken(session_key: str, text: str, kind: str = "speech") -> None:
    """RIO said something. One way, in.

    THE DIRECTION IS THE DESIGN. The speech path may tell the panel what was
    said, because a corpus row is worth much more with RIO's own line beside
    the two models' readings. The panel may never tell the speech path
    anything. tools/teacher_firewall_selftest.py asserts that asymmetry
    explicitly -- this function is on the allowed list and there is no
    counterpart pointing the other way.
    """
    text = (text or "").strip()
    if not text:
        return
    with _lock:
        _state(str(session_key or "default"))["spoken"] = {
            "text": text[:400], "at": time.time(), "kind": str(kind or "speech")}


def ingest_ego(session_key: str, samples: list, offset_s: float = 0.0) -> dict:
    """DeviceMotion batch from the client. Straight through to egomotion."""
    return egomotion.ingest(session_key, samples, offset_s)


# ---------------------------------------------------------------------------
# readings coming back
# ---------------------------------------------------------------------------
def _on_reading(model: str, job: dict, reading: dict) -> None:
    """A teacher answered (or failed to). Called on that teacher's thread."""
    key = None
    kf = None
    with _lock:
        for k, st in _sessions.items():
            if job["kf_id"] in st["kf"]:
                key, kf = k, st["kf"][job["kf_id"]]
                break
    if key is None:
        # The drive ended, or the keyframe aged out of the holding dict while
        # this model was still generating. The reading is still shown -- it is
        # a real reading -- but there is nothing to record it against.
        key = job["kf_id"].rsplit("-", 1)[0]

    rec = schema.blank_reading(model)
    # EVERYTHING THE SERVICE SENT, not a filtered subset. The named fields are
    # what the card draws; anything else the service reported -- per-stage
    # timings, `cameras: 1`, `frames_as: video`, `ego_synthetic` -- is a fact
    # about how the reading was produced and belongs in the corpus. It lands in
    # `extra` on the way to disk (teachers/corpus._clean_reading) rather than
    # being dropped for not having been thought of when the schema was written.
    rec.update(reading or {})
    rec["model"] = model
    # WHEN THIS READING CAME BACK, and WHICH INSTANT it is about. Two clocks,
    # two questions: `at` is how old the ANSWER is (the card's live dot),
    # `t0` is the instant of the road it describes, and the age a driver
    # actually cares about -- now minus t0 -- grows while the card sits there
    # and can only be computed if t0 is on the record.
    rec["at"] = time.time()
    rec["t0"] = float(job.get("t0") or rec["at"])
    rec["kf_id"] = job["kf_id"]
    # A job the client threw away before running it -- evicted by a newer
    # keyframe, or gone stale while it waited. It is not a reading and must not
    # replace one on the card; it exists so the accumulator below knows this
    # model is never going to answer this keyframe.
    not_run = bool(rec.get("not_run"))

    # Associate the named actor to a track. Done HERE, on the teacher's thread,
    # so the frame path never pays for it and so the association is stored with
    # the reading it belongs to rather than recomputed for every dashboard poll.
    tracks = (kf or {}).get("tracks") or []
    img = (kf or {}).get("image") or job.get("image") or {}
    a = assoc_mod.associate(rec.get("critical_actor") or "", tracks,
                            img.get("w") or 0, img.get("h") or 0)

    # Alpamayo's predicted path, projected onto the picture. Display only --
    # see teachers/project.py. Computed once here rather than in the browser so
    # the ribbon and the ego corridor come out of the same camera model.
    traj = rec.get("trajectory")
    if traj and traj.get("xyz") and img.get("w"):
        try:
            traj["pixels"] = project.trajectory_pixels(
                traj["xyz"], img["w"], img["h"])
        except Exception:
            traj["pixels"] = None

    kf_for_record = None
    readings_copy = assoc_copy = None
    with _lock:
        st = _state(key)
        if not not_run:
            st["readings"][model] = rec
            st["assoc"][model] = a
            if rec.get("ok"):
                st["tally"]["readings"] += 1
            if rec.get("critical_actor"):
                st["tally"]["actors"] += 1
                if a.get("matched"):
                    st["tally"]["matched"] += 1

        # --- ONE ROW PER KEYFRAME, ASSEMBLED BY KEYFRAME ID ------------------
        #
        # THE BUG THIS REPLACES. The row used to be written when both models'
        # LATEST readings named the same keyframe -- which is true only while
        # the two of them stay in step, and they do not. They have their own
        # queues and their own eviction, so a fast teacher runs keyframes 10,
        # 11, 12 while a slow one runs 10 and then 13, and from keyframe 11
        # onwards "both latest agree" is almost never true again. The corpus
        # would have quietly thinned to nothing a minute into a drive.
        #
        # So readings are accumulated against the keyframe they belong to, and
        # the row is written when every model has REPORTED on that keyframe --
        # answered it, failed it, or been told it was thrown away.
        slot = st["pending"].setdefault(job["kf_id"], {"readings": {}, "assoc": {}})
        slot["readings"][model] = rec
        slot["assoc"][model] = a
        if all(m in slot["readings"] for m in MODEL_NAMES):
            st["pending"].pop(job["kf_id"], None)
            # Agreement is asked ONCE per keyframe, here, where both answers
            # are about the same instant by construction. Only when both
            # actually ran: a model that never saw the keyframe has not
            # disagreed with anything.
            if all(not slot["readings"][m].get("not_run") for m in MODEL_NAMES):
                st["tally"]["agree_eligible"] += 1
                if assoc_mod.agree(slot["assoc"][MODEL_NAMES[0]],
                                   slot["assoc"][MODEL_NAMES[1]]):
                    st["tally"]["agree"] += 1
            # A keyframe NEITHER model ran is not a row. It is already counted
            # as an eviction on the service strip, and a corpus full of "the
            # queue was full" rows is a corpus nobody reads.
            if any(not slot["readings"][m].get("not_run") for m in MODEL_NAMES):
                kf_for_record = kf
                readings_copy = {m: dict(slot["readings"][m]) for m in MODEL_NAMES}
                assoc_copy = {m: dict(slot["assoc"][m]) for m in MODEL_NAMES}
            st["kf"].pop(job["kf_id"], None)
        # Bounded, because a model that dies mid-generation never reports and
        # its slot would otherwise sit here for the rest of the drive.
        if len(st["pending"]) > 16:
            for old_id in sorted(st["pending"])[:-16]:
                st["pending"].pop(old_id, None)

    # WRITTEN WHEN EVERY MODEL HAS REPORTED, not when the first one does. A
    # corpus row with one column filled is a row that has to be merged later by
    # whoever reads it, and merging by keyframe id after the fact is exactly
    # the sort of chore that makes a corpus go unused.
    if kf_for_record is not None:
        corpus_mod.write(kf_for_record, readings_copy, assoc_copy)


# ---------------------------------------------------------------------------
# what the dashboard reads
# ---------------------------------------------------------------------------
def state(session_key: str) -> dict:
    """Everything the Teachers card draws, in one object.

    Read-only by construction: it copies. A dashboard poll must not be able to
    change what the next corpus row says.
    """
    key = str(session_key or "default")
    fresh_s = float(getattr(config, "TEACHER_READING_FRESH_S", 4.0))
    now = time.time()
    with _lock:
        st = _sessions.get(key)
        if st is None:
            return {"enabled": bool(getattr(config, "TEACHERS_ENABLED", False)),
                    "session": key, "models": {}, "tally": {}, "services": _svc()}
        out = {
            "enabled": True,
            "session": key,
            "tally": dict(st["tally"]),
            "last_build_refusal": st["last_build_refusal"],
            "spoken": dict(st["spoken"]) if st["spoken"] else None,
            "models": {},
            "services": _svc(),
        }
        for m in MODEL_NAMES:
            r = dict(st["readings"].get(m) or schema.blank_reading(m))
            a = dict(st["assoc"].get(m) or {})
            # TWO AGES, AND THE CARD SHOWS THE ONE THAT MATTERS. `age_s` is
            # how long ago the ANSWER arrived. `freshness_s` is how old the
            # ROAD in it is RIGHT NOW -- recomputed on every poll from t0, so
            # it keeps growing while a card sits open, which is the whole
            # point of showing it. The stored value was the age at arrival and
            # would have frozen there.
            r["age_s"] = round(now - r["at"], 2) if r.get("at") else None
            if r.get("t0"):
                r["freshness_s"] = round(now - float(r["t0"]), 2)
            r["fresh"] = bool(r.get("at") and (now - r["at"]) <= fresh_s)
            out["models"][m] = {"reading": r, "association": a}
    t = out["tally"]
    t["agreement_pct"] = (round(100.0 * t.get("agree", 0)
                                / max(1, t.get("agree_eligible", 0)))
                          if t.get("agree_eligible") else None)
    t["matched_pct"] = (round(100.0 * t.get("matched", 0)
                              / max(1, t.get("actors", 0)))
                        if t.get("actors") else None)
    return out


def _svc():
    return {name: c.status() for name, c in _clients.items()}


def status() -> dict:
    """For /health and the selftests."""
    with _lock:
        return {
            "enabled": bool(getattr(config, "TEACHERS_ENABLED", False)),
            "started": _started,
            "services": _svc(),
            "sessions": {k: {"keyframes": st["tally"]["keyframes"],
                             "readings": st["tally"]["readings"],
                             "seq": st["seq"]}
                         for k, st in _sessions.items()},
            "ego": egomotion.status(),
        }


def drop(session_key: str) -> bool:
    """A drive ended. Let go of its keyframes and its ego ring."""
    key = str(session_key or "default")
    egomotion.drop(key)
    with _lock:
        return _sessions.pop(key, None) is not None


def reset_all() -> None:
    with _lock:
        _sessions.clear()
    egomotion.reset_all()
