"""observer.py — a running description of the road, so answering is not thinking.

"What do you see?" was slow, and the measurement said why: not the frame ring
(20 ms), not the detector, not Qwen — the multimodal answer is a remote call to
a reasoning model, and it costs ~1.1 s to the first word and ~2.0 s in full.
Over ten questions on this pod, 99% of the wait at p50 was that one call.

That call is worth its price for "what colour is the car on the left", which
needs a crop, a reference resolved against tracked objects, and a model careful
enough to say when it cannot tell. It is not worth its price for "what do you
see", which is the same question about the same road every time and whose
answer changes on its own every few seconds.

So during a live conversation the answer is prepared BEFORE it is asked for.
This runs Qwen — resident, local, ~0.4 s a frame — over the newest frame in the
session's ring about once a second, and keeps the sentence with the timestamp
of the frame it came from. `look()` serves a scene question straight out of it.

WHY IT IS SAFE TO SERVE A CACHED ANSWER, AND EXACTLY WHEN IT IS NOT
-------------------------------------------------------------------
Because the record carries WHEN. A description of the road is true for a second
or two at 60 km/h and false after ten, and the difference between a fast
assistant and a lying one is entirely whether it knows which it is holding.
`fresh()` refuses anything older than config.OBSERVER_FRESH_S, and a refusal is
not a failure -- it falls back to the full path, which looks at the road now.

A scene question only. Anything about a particular object -- "the black one",
"the building on the right", "what does that sign say" -- is not in a
one-sentence caption and must not be answered from one. The router already
draws that line for its own reasons; this reuses it rather than inventing a
second, differently-wrong one.

WHY IT IS NOT ALWAYS RUNNING
----------------------------
It costs ~0.4 s of the same GPU the 4 fps headway loop is using, which is real
money on a card that is also running detection, depth and lanes on every frame.
So it runs only while a live conversation is open, only while frames are
actually arriving, and it stops itself when neither is true.
"""
import contextlib
import threading
import time

import config

_lock = threading.Lock()
_sessions = {}          # key -> {"thread", "stop", "last_used", "record", "n", "errors"}


def _record(text, frame):
    return {
        "text": text,
        "at": time.time(),
        # The frame's OWN clock, not the observation's: the difference between
        # them is how long Qwen took, and a driver asking "what do you see"
        # cares about when the picture was taken.
        "frame_wall_t": getattr(frame, "wall_t", None),
        "frame_id": getattr(frame, "frame_id", None),
        "frame_age_s": round(getattr(frame, "age_s", 0.0) or 0.0, 2),
    }


@contextlib.contextmanager
def hold(session_key: str):
    """Stop observing while a real question is being answered.

    The observer and the visual turn want the same GPU, and the turn is the one
    somebody is waiting for. Measured: with the observer running free, the
    enrichment step of an object question went from ~500 ms to ~1000 ms --
    every object question paying for the scene questions to be fast. It yields
    instead.

    Re-entrant by count, because two questions can overlap and the first one
    finishing must not resume observing under the second.
    """
    key = str(session_key or "default")
    with _lock:
        st = _sessions.get(key)
        if st is not None:
            st["hold"] = st.get("hold", 0) + 1
    try:
        yield
    finally:
        with _lock:
            st = _sessions.get(key)
            if st is not None:
                st["hold"] = max(0, st.get("hold", 0) - 1)


def _tick(key, state):
    """One observation, or nothing. Never raises into the loop."""
    import framebuf
    import vision

    if state.get("hold", 0) > 0:
        return False
    ring = framebuf.peek_ring(key)
    if ring is None:
        return False
    frame = ring.latest()
    if frame is None:
        return False
    # The same frame twice is the same sentence twice, at the price of a
    # forward pass. Skipped -- which is also what makes a parked car cheap.
    last = state.get("record")
    if last and last.get("frame_id") == getattr(frame, "frame_id", None):
        return False
    jpeg = getattr(frame, "jpeg", None)
    if not jpeg:
        return False
    text = vision.observe(jpeg, frame_id=getattr(frame, "frame_id", None))
    if not text:
        return False
    with _lock:
        state["record"] = _record(text, frame)
        state["n"] += 1
    return True


def _loop(key, state):
    period = float(config.OBSERVER_PERIOD_S)
    while not state["stop"].is_set():
        t0 = time.time()
        try:
            _tick(key, state)
        except Exception as e:
            # A failed observation costs a sentence, never the conversation.
            # Counted rather than printed every second: on a GPU that is out of
            # memory this would otherwise be the loudest thing in the log.
            with _lock:
                state["errors"] += 1
                if state["errors"] in (1, 10, 100):
                    print(f"[observer] {key}: {type(e).__name__}: {e}", flush=True)
        # Idle: nobody has asked to see anything for a while. Stop rather than
        # hold the GPU for a conversation that has moved on to the route.
        if (time.time() - state["last_used"]) > config.OBSERVER_IDLE_S:
            break
        # Pace from the START of the tick, so a slow forward pass does not add
        # itself to the period and halve the rate.
        state["stop"].wait(max(0.05, period - (time.time() - t0)))
    with _lock:
        if _sessions.get(key) is state:
            _sessions.pop(key, None)


def start(session_key: str) -> bool:
    """Begin (or keep) observing for one session. -> did it start one?

    Idempotent: a second call on a running observer only pushes its idle
    deadline out, which is what every look() does.
    """
    if not (config.VISION_ENABLED and config.OBSERVER_ENABLED):
        return False
    key = str(session_key or "default")
    with _lock:
        st = _sessions.get(key)
        if st is not None:
            st["last_used"] = time.time()
            return False
        st = {"stop": threading.Event(), "last_used": time.time(),
              "record": None, "n": 0, "errors": 0, "started": time.time(),
              "hold": 0}
        _sessions[key] = st
    st["thread"] = threading.Thread(target=_loop, args=(key, st),
                                    name=f"observer:{key}", daemon=True)
    st["thread"].start()
    return True


def stop(session_key: str) -> bool:
    key = str(session_key or "default")
    with _lock:
        st = _sessions.get(key)
    if st is None:
        return False
    st["stop"].set()
    return True


def touch(session_key: str) -> None:
    """This session is still being asked about."""
    with _lock:
        st = _sessions.get(str(session_key or "default"))
        if st is not None:
            st["last_used"] = time.time()


def cached(session_key: str) -> dict:
    """The latest observation for this session, whatever its age. -> {} if none."""
    with _lock:
        st = _sessions.get(str(session_key or "default"))
        rec = dict(st["record"]) if st and st.get("record") else {}
    if rec:
        rec["age_s"] = round(time.time() - rec["at"], 2)
    return rec


def fresh(session_key: str, max_age_s: float = None) -> dict:
    """The observation IF it is still true. -> {} when it is not.

    The only question this module exists to answer, and the reason the record
    carries a timestamp at all. Age is measured from the FRAME, not from the
    observation: Qwen taking 400 ms to describe a picture does not make the
    picture newer, and it is the picture the driver is being told about.
    """
    if max_age_s is None:
        max_age_s = config.OBSERVER_FRESH_S
    rec = cached(session_key)
    if not rec or not rec.get("text"):
        return {}
    wall = rec.get("frame_wall_t")
    age = (time.time() - wall) if wall else rec.get("age_s")
    if age is None or age > float(max_age_s):
        return {}
    rec["age_s"] = round(age, 2)
    return rec


def observe_now(session_key: str, max_age_s: float = None) -> dict:
    """Describe the CURRENT frame, now, synchronously. -> record or {}

    The fallback between the two paths, and the reason the fast path degrades
    gently instead of falling off a cliff.

    The background loop describes a frame about once a second, so a question
    can arrive in the gap: frames are current, the observation is one tick
    behind them, and `fresh()` correctly refuses it. Before this, that miss cost
    the full remote turn -- ~2 s to answer a question whose answer is one local
    forward pass away.

    So the miss runs that forward pass instead: ~0.4 s on this GPU, on the frame
    that is in front of the car right now. Only worth doing when there IS such a
    frame -- with nothing recent in the ring there is nothing to describe, and
    the honest answer comes from the full path, which knows how to say so.
    """
    if not (config.VISION_ENABLED and config.OBSERVER_ENABLED):
        return {}
    if max_age_s is None:
        max_age_s = config.OBSERVER_FRESH_S
    key = str(session_key or "default")
    import framebuf

    ring = framebuf.peek_ring(key)
    frame = ring.latest() if ring is not None else None
    if frame is None or (frame.age_s or 0) > float(max_age_s):
        return {}
    with _lock:
        st = _sessions.get(key)
    if st is None:
        start(key)
        with _lock:
            st = _sessions.get(key)
        if st is None:
            return {}
    try:
        import vision

        text = vision.observe(frame.jpeg, frame_id=frame.frame_id)
    except Exception as e:
        print(f"[observer] {key}: on-demand observation failed: "
              f"{type(e).__name__}: {e}", flush=True)
        return {}
    if not text:
        return {}
    rec = _record(text, frame)
    with _lock:
        st["record"] = rec
        st["n"] += 1
    out = dict(rec)
    out["age_s"] = round(time.time() - (rec.get("frame_wall_t") or rec["at"]), 2)
    out["on_demand"] = True
    return out


def status() -> dict:
    """What is running, for /health and the selftests."""
    with _lock:
        return {key: {"observations": st["n"], "errors": st["errors"],
                      "held": st.get("hold", 0),
                      "idle_s": round(time.time() - st["last_used"], 1),
                      "has_record": bool(st.get("record"))}
                for key, st in _sessions.items()}


def stop_all() -> None:
    with _lock:
        states = list(_sessions.values())
    for st in states:
        st["stop"].set()
