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
It costs the same GPU the headway loop is using, which is real money on a card
that is also running detection, depth and lanes on every frame. So it runs only
while a live conversation is open, only while frames are actually arriving, and
it stops itself when neither is true.

AND IT IS NO LONGER A 1 Hz LOOP. READ THIS BEFORE TRUSTING ANYTHING ABOVE.
--------------------------------------------------------------------------
Everything above was written when the eye was Qwen answering a three-field
template in ~0.4 s. The eye is now Cosmos-Reason2 asked the way its own model
card says to ask a reasoning model -- a question, a <think> trace, 512 tokens
-- and that is a different instrument.

MEASURED on this pod, with the detector feeding at ~15 fps on the same card:

    one reading per ~10 seconds.

Not 1 Hz. A reading is typically 5 s old when a question arrives and can be 10
or more, which is past OBSERVER_ANSWER_MAX_AGE_S, so THE FAST PATH USUALLY
MISSES AND THE FULL VISUAL PATH ANSWERS. That is not a regression to be tuned
away by widening the window: a fifteen-second-old description served at 13 m/s
is a description of a road two hundred metres back, and the full path takes
about two seconds and looks at the road NOW.

So what this module is for has changed, and the honest statement of it is:

    the CARD is the consumer. It repaints every 15 s and a 10 s cadence
    feeds it perfectly well.

    RIO gets the reading when it happens to be recent, as labelled and
    timestamped evidence, and otherwise asks properly. Which is the right
    trade: on the measurement that produced this note, the cached reading
    invented "a white sedan accelerating rapidly toward another vehicle
    ahead" and the full path, on the same frame, said "three lanes ahead,
    empty mostly, white sedan cruising in the right lane".
"""
import contextlib
import threading
import time

import config
import gpu_health
import persona

_lock = threading.Lock()
_sessions = {}          # key -> {"thread", "stop", "last_used", "record", "n", "errors"}


def _clean(text: str) -> str:
    """Trim what a vision model adds around a sentence it was asked for.

    Quotes, a stray "Answer:", a trailing full stop on something meant to be
    spoken. None of this is the model failing — it is the shape of a chat
    completion — and none of it should reach a speaker.
    """
    t = (text or "").strip().strip('"').strip("'").strip()
    for prefix in ("answer:", "observation:", "sentence:", "rio:"):
        if t.lower().startswith(prefix):
            t = t[len(prefix):].strip()
    return t.rstrip(".").strip()


def _record(text, frame, key=None, meta=None):
    # WHAT ELSE THE READING CARRIED. vision.observe() returns a string -- every
    # caller expects one -- so the facts ABOUT it come across separately and
    # are pinned onto the record here, where they outlive the call. `truncated`
    # is the one that changes what a driver may conclude: a reading cut at the
    # token cap ends mid-clause, and its last field is a fragment wearing the
    # clothes of a finished answer.
    meta = meta if isinstance(meta, dict) else {}
    # IS THIS SPEAKABLE AS HER? Decided when the line is written rather than
    # when a driver is waiting for it, and stored — the answer is a property of
    # the sentence and does not change with time the way freshness does.
    #
    # A line that fails is still kept. It is a perfectly good description and
    # the composed path still works from it; what it may not do is go straight
    # to a speaker as something RIO said.
    #
    # AND UNDER A SENSOR MODEL THE ANSWER IS ALWAYS NO, whatever the words look
    # like. Cosmos-Reason2 is asked for a reading (rio_prompts.SENSOR_PROMPT),
    # not for her sentence: it is an instrument, its output is evidence, and a
    # reading that happened to pass a persona lint would still be an instrument
    # speaking in her voice. So the gate is the MODEL's role, asked first, and
    # the lint only decides anything for the model that was asked to write a
    # line for her. See config.local_vision_speaks_directly().
    #
    # This is what makes "the local model never speaks as her" a property of the
    # code rather than a property of how the prompt happens to read today.
    import config as _config
    if not _config.local_vision_speaks_directly():
        return {
            "text": text,
            "speakable": False,
            "faults": ["sensor_model"],
            # THE NAME OF THE INSTRUMENT, beside its reading. A reading shown
            # without the model that produced it is a reading nobody can weigh,
            # and this is the field the glass renders.
            "model": _config.local_vision_label(),
            "at": time.time(),
            "frame_wall_t": getattr(frame, "wall_t", None),
            "frame_id": getattr(frame, "frame_id", None),
            "frame_age_s": round(getattr(frame, "age_s", 0.0) or 0.0, 2),
            "origin": getattr(frame, "origin", None),
            "session_key": str(key) if key else None,
            "truncated": bool(meta.get("truncated")),
            "stripped": list(meta.get("stripped") or []),
        }
    faults = persona.lint(text)
    return {
        "text": text,
        "speakable": not faults,
        "faults": faults,
        "model": _config.local_vision_label(),
        "at": time.time(),
        # The frame's OWN clock, not the observation's: the difference between
        # them is how long Qwen took, and a driver asking "what do you see"
        # cares about when the picture was taken.
        "frame_wall_t": getattr(frame, "wall_t", None),
        "frame_id": getattr(frame, "frame_id", None),
        "frame_age_s": round(getattr(frame, "age_s", 0.0) or 0.0, 2),
        # WHOSE FRAME THIS DESCRIBES. Carried on the record because the record
        # outlives the frame: without it, "the observer has something to say"
        # and "the observer has something to say about YOUR road" are the same
        # question, and they are not. See serve_to() and look().
        "origin": getattr(frame, "origin", None),
        "session_key": str(key) if key else None,
        "truncated": bool(meta.get("truncated")),
        "stripped": list(meta.get("stripped") or []),
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
    # THE NEWEST FRAME IS NOT NECESSARILY A CURRENT ONE.
    #
    # `latest()` is the newest thing in the ring, which on a healthy feed is a
    # picture from 200 ms ago and on a stalled one is whatever arrived before
    # the transport died. On 2026-09-09 that was a frame from t=40 being held
    # while the drive ran to t=656.
    #
    # fresh() already refuses to SERVE a description that old, so the answer
    # was never wrong -- but the forward pass was still spent, once per new
    # frame id, on a picture of a road the car left minutes ago. Gate it here
    # too, and say so, because "the observer went quiet" and "the observer is
    # describing history" are different faults with the same symptom.
    stale_after = float(getattr(config, "OBSERVER_MAX_FRAME_AGE_S",
                                config.OBSERVER_FRESH_S))
    age = getattr(frame, "age_s", None) or 0.0
    if age > stale_after:
        with _lock:
            state["stale_frames"] = state.get("stale_frames", 0) + 1
            if state["stale_frames"] in (1, 10, 100):
                print(f"[observer] {key}: newest frame is {age:.1f}s old "
                      f"(> {stale_after:.1f}s) — not describing it", flush=True)
        return False
    # The same frame twice is the same sentence twice, at the price of a
    # forward pass. Skipped -- which is also what makes a parked car cheap.
    last = state.get("record")
    if last and last.get("frame_id") == getattr(frame, "frame_id", None):
        return False
    jpeg = getattr(frame, "jpeg", None)
    if not jpeg:
        return False
    _t_obs = time.time()
    text = _clean(vision.observe(jpeg, frame_id=getattr(frame, "frame_id", None)))
    # ONE ROW PER GENERATE, in the drive log. The observer is the only thing
    # that runs Qwen every second of a drive, and until now no log said when
    # it ran or how long -- so a frame stall could only be laid beside it by
    # guessing. `lock_wait_ms` says how much of the tick was queueing.
    try:
        import sessions as _sessions
        _sessions.log_live(_session_id_of(key), "observer_tick", {
            "ms": round((time.time() - _t_obs) * 1000.0, 1),
            "lock_wait_ms": round(vision.last_lock_wait_ms(), 1),
            "chars": len(text or ""),
        })
    except Exception:
        pass
    if not text:
        # Includes the case where the model returned one of its own prompt's
        # examples: vision refuses those outright. Nothing is recorded, so
        # look() finds nothing and the honest path answers instead.
        return False
    rec = _record(text, frame, key, vision.last_reading_meta())
    with _lock:
        state["record"] = rec
        state["n"] += 1
        if not rec["speakable"]:
            # Counted, and named. "She stopped sounding like herself" is a
            # complaint somebody will make after a prompt change, and the
            # answer wants to be a number with reasons attached rather than a
            # shrug. Printed rarely, for the same reason errors are.
            #
            # A SENSOR MODEL IS NOT A REGRESSION, AND MUST NOT READ AS ONE. Under
            # cosmos every record is unspeakable BY DESIGN -- it is an
            # instrument's reading, not her line -- so counting it in the same
            # tally as "Qwen wrote something that failed the persona lint" would
            # make a healthy drive report 100% unspeakable and train whoever
            # reads that number to ignore it. Two counters, because they are two
            # facts: one is the design, the other is a fault.
            if rec["faults"] == ["sensor_model"]:
                state["readings"] = state.get("readings", 0) + 1
            else:
                state["unspeakable"] = state.get("unspeakable", 0) + 1
                if state["unspeakable"] in (1, 10, 100):
                    print(f"[observer] {key}: not in her voice "
                          f"({rec['faults']}): {text!r}", flush=True)
    return True


def _session_id_of(key):
    """The drive's session id, if this key is one. app._visual_key hands a
    drive its own session id as the key and a tab a `client:` key; only the
    former has a session log to write to, and a wrong id would become a stray
    file rather than a row."""
    k = str(key or "")
    if len(k) == 36 and k.count("-") == 4:
        return k
    return None


def _tick_window(key, state):
    """One grounded video reading, or nothing. Never raises into the loop.

    THE SECOND CADENCE, RUNNING BESIDE THE FIRST AND NOT REPLACING IT.
    ------------------------------------------------------------------
    _tick above reads ONE FRAME and its record is what RIO is handed as
    evidence and what reconcile.py checks against the detector. That contract
    has a dozen callers and this does not touch it.

    This reads a WINDOW -- six seconds of video and every number measured over
    those same six seconds -- and its record goes to the card and nowhere else
    yet. The spec that asked for it is explicit that whether Cosmos may trigger
    speech is decided after the corroboration numbers are in, so the reading
    deliberately has no route into the warning path, the arbiter, or RIO's
    evidence. When those numbers justify it, the change is to give this record
    a consumer; until then the cost of being wrong is a card that says
    something odd.

    CADENCE. Measured on this pod with the detector feeding at 4 fps on the
    same card: a grounded read is 3.5-7 s of generate, p50 ~5.4 s, and one
    outlier at 15 s where the reasoning trace filled its budget. So the period
    is 8 s (config.EYE_WINDOW_PERIOD_S) rather than the still path's 10 s, and
    a reading describes a window that ENDED 4-9 s ago. That is fine for a card
    and is exactly why this is not wired to anything that has to be current.
    """
    import eyeread
    import eyewindow
    import grounding as _g

    if state.get("hold", 0) > 0:
        return False
    window = eyewindow.from_ring(key)
    if window is None:
        return False
    # THE SAME FRAME TWICE IS THE SAME READING TWICE, at the price of a
    # five-second forward pass. A window whose last frame is one we have
    # already read to the end is a window that has not moved.
    last = state.get("window_record") or {}
    ids = (window.to_meta() or {}).get("frame_ids") or []
    if ids and last.get("last_frame_id") == ids[-1]:
        return False
    st = _g.window_state(window)
    rec = eyeread.read_window(window, st, grounded=True)
    rec["last_frame_id"] = ids[-1] if ids else None
    with _lock:
        state["window_record"] = rec
        state["window_n"] = state.get("window_n", 0) + 1
        if rec.get("refused"):
            state["window_refused"] = state.get("window_refused", 0) + 1
    try:
        import sessions as _sessions
        _sessions.log_live(_session_id_of(key), "eye_window", {
            "ms": (rec.get("timing") or {}).get("gen_ms"),
            "span_s": (rec.get("window") or {}).get("span_s"),
            "n_frames": (rec.get("window") or {}).get("n_frames"),
            "refused": rec.get("refused"),
            "verdict": (rec.get("corroboration") or {}).get("verdict"),
            "sourced": len(rec.get("sourced") or []),
            "invented": len(rec.get("invented") or []),
        })
    except Exception:
        pass
    return True


def window_reading(session_key: str) -> dict:
    """The last grounded video reading for this session. -> {}|record."""
    with _lock:
        state = _sessions.get(session_key)
        rec = dict(state.get("window_record") or {}) if state else {}
    if rec:
        # Re-age on the way out. The window's own end_age_s was computed when
        # the record was built and a card polling every few seconds needs the
        # age NOW, not the age at filing.
        import time as _t
        w = dict(rec.get("window") or {})
        filed = rec.get("at")
        if filed and w.get("end_age_s") is not None:
            w["end_age_s"] = round(float(w["end_age_s"]) + (_t.time() - filed), 2)
            w["start_age_s"] = round(float(w["end_age_s"]) + float(w.get("span_s") or 0), 2)
        rec["window"] = w
    return rec


def _loop(key, state):
    period = float(config.OBSERVER_PERIOD_S)
    # The video read is slower and rarer than the still read, so it is paced
    # by its own clock rather than by a divisor of this loop's: a period that
    # was "every Nth tick" would drift the moment a generate ran long.
    win_period = float(getattr(config, "EYE_WINDOW_PERIOD_S", 8.0))
    win_next = 0.0
    while not state["stop"].is_set():
        t0 = time.time()
        try:
            _tick(key, state)
            if getattr(config, "EYE_WINDOW_ENABLED", True) and t0 >= win_next:
                # Charged from the END of the read, not the start: a five
                # second generate inside an eight second period would
                # otherwise leave three seconds of gap and then run again
                # immediately, which is not a cadence, it is a queue.
                try:
                    _tick_window(key, state)
                finally:
                    win_next = time.time() + win_period
        except Exception as e:
            # A failed observation costs a sentence, never the conversation.
            # Counted rather than printed every second: on a GPU that is out of
            # memory this would otherwise be the loudest thing in the log.
            #
            # ...and that is exactly the case worth telling somebody about, so
            # it is also counted where /health can see it. The quiet handling
            # here was right and it was the whole problem: a drive with no
            # observations looked identical to a drive with nothing to say.
            gpu_health.note("observer", e)
            with _lock:
                state["errors"] += 1
                if state["errors"] in (1, 10, 100):
                    print(f"[observer] {key}: {type(e).__name__}: {e}", flush=True)
                    # ...AND THE TRACEBACK, ON THE FIRST ONE.
                    #
                    # "AssertionError: " is a real line this printed, with an
                    # empty message, from inside a compiled generate several
                    # layers below anything in this file. One line of type and
                    # message is enough to know the observer is failing and
                    # useless for knowing why -- which is the same lesson
                    # app.py's warm handler already carries, for the same
                    # reason. Once, not every second: a GPU that is out of
                    # memory would otherwise fill the log with identical stacks.
                    if state["errors"] == 1:
                        import traceback
                        traceback.print_exc()
                        import sys as _sys
                        _sys.stdout.flush()
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
              "hold": 0, "unspeakable": 0, "stale_frames": 0}
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


def reading(session_key: str, rec: dict = None) -> dict:
    """The record, split into fields and checked against the detector. -> {}|dict

    ONE BUILDER, TWO CONSUMERS, AND THAT IS THE POINT. /perceive calls this for
    the Perception card and realtime.look() calls it for the live session. If
    they built their own the card would eventually show a reading RIO was not
    given, which is the one thing the card exists not to do --
    tools/sensor_card_selftest.py compares the two payloads byte for byte on a
    running server.

    -> {raw, fields, extra, parsed, model, age_s, frame_id, frame_age_s,
        speakable, fresh_s, contested, detector}

    `fields` are RECONCILED: a TRAFFIC field claiming the road is empty while
    the detector holds road users on it comes back marked, with the tracker's
    account beside it. Nothing is rewritten -- see reconcile.py for why the
    rule is asymmetric and why RISK is left alone.
    """
    import rio_prompts as _rp
    import reconcile as _rc
    from headway import census as _census

    rec = cached(session_key) if rec is None else rec
    if not isinstance(rec, dict) or not (rec.get("text") or "").strip():
        return {}
    out = _rp.split_sensor_reading(rec["text"])
    out["model"] = rec.get("model") or config.local_vision_label()
    # HOW OLD THE PICTURE IS, not how long ago the model stopped typing.
    #
    # cached() ages a record from `at`, the moment the reading was FILED. With
    # a reasoning model that is three to four seconds after the frame was
    # taken, so the card was under-reporting the age of what it was showing by
    # the whole length of the generate -- and recent(), which gates whether RIO
    # may be told at all, measures from the frame. Two different ages for one
    # reading, and the smaller one on the glass.
    #
    # A driver reading "2 s ago" wants to know when the PICTURE was, which is
    # the only question the number can usefully answer.
    _frame_t = rec.get("frame_wall_t") or rec.get("at")
    out["age_s"] = (round(time.time() - float(_frame_t), 2)
                    if _frame_t else rec.get("age_s"))
    # ...and the filing age beside it, because "the model took four seconds"
    # and "the picture is four seconds old" are different facts and a slow
    # model is a thing somebody will want to see.
    out["filed_age_s"] = rec.get("age_s")
    out["frame_id"] = rec.get("frame_id")
    out["frame_age_s"] = rec.get("frame_age_s")
    # An instrument's reading is never spoken as hers. See _record above.
    out["speakable"] = bool(rec.get("speakable"))
    # The observer's own threshold, shipped rather than guessed at by a card.
    out["fresh_s"] = float(getattr(config, "OBSERVER_FRESH_S", 2.0))
    # WHAT WAS WRONG WITH THIS READING THAT ITS WORDS DO NOT SHOW.
    #
    # Truncation is marked on the FIELD the cut landed in -- the last one with
    # any text -- because that is the row a driver has to distrust, and a card
    # that only said "truncated" somewhere near the top would leave them
    # reading a fragment as a finding. A reading cut just after "RISK:" is the
    # dangerous case: it looks like a completed all-clear.
    out["truncated"] = bool(rec.get("truncated"))
    out["stripped"] = list(rec.get("stripped") or [])
    # WHICH PICTURE THIS IS A READING OF. The record carries the frame's origin
    # as "<session key>:<kind>" (app._frame_origin); the kind is the half a
    # driver and RIO both need, because "the camera" and "the clip you loaded"
    # are different answers to "what are you looking at" and a drive is no
    # longer what decides which one is feeding. See the source audit in the
    # capture block of index.html.
    origin = str(rec.get("origin") or "")
    out["source"] = origin.rsplit(":", 1)[-1] if ":" in origin else None
    if out["truncated"]:
        present = [f for f in out["fields"] if f.get("present")]
        if present:
            present[-1]["truncated"] = True
        else:
            # Cut before any field was finished -- there is nothing to mark, so
            # the whole reading carries it.
            out["parsed"] = out["parsed"] and False
    # THE DETECTOR, WHICH WINS ON EXISTENCE. Keyed by the visual key, which is
    # the key this reading is filed under too -- see headway/census.py.
    try:
        checked = _rc.check(out["fields"], _census.current(session_key))
        out["fields"] = checked["fields"]
        out["contested"] = checked["contested"]
        out["detector"] = checked["detector"]
    except Exception as e:
        # A reconciliation that fails may not cost the reading.
        print(f"[observer] reconcile failed: {type(e).__name__}: {e}", flush=True)
        out["contested"] = []
        out["detector"] = None
    return out


def serve_to(rec: dict, session_key: str) -> bool:
    """May THIS session be told THIS observation? Two questions, both hard no.

    IS IT ABOUT THIS SESSION'S ROAD. The record names the frame's origin --
    "<session key>:<source>" -- so an observation may only be served to the
    session whose key it names. A drive is never told about the keyless ring,
    and a keyless page is never told about a drive.

    IS IT FROM A REAL SOURCE. `api:` frames are posted by something that is not
    a browser looking through a windscreen: a bench, a curl, an acceptance
    harness feeding a demo clip through the same endpoint. Those may satisfy a
    keyless API caller asking about the frames it just posted, and they may
    never satisfy a live session -- a driver asking what is out there gets an
    answer about their own camera or gets told there is nothing to see.

    An unstamped record (nothing has pushed a frame since this ran, or an older
    ring) fails both: unknown provenance is not provenance.
    """
    if not rec:
        return False
    # The rule itself lives in framebuf.owns, because this is not the only
    # reader that needs it -- and for a while it behaved as though it were,
    # which is how a phone got answered from a desktop's clip.
    import framebuf

    return framebuf.owns(rec.get("origin"), session_key)


def fresh(session_key: str, max_age_s: float = None) -> dict:
    """The observation IF it is still true, AND if it is this session's. -> {}

    Two refusals, and they fail the same way on purpose -- an empty dict, which
    every caller already reads as "no observation" and answers honestly from.

    WHEN. Age is measured from the FRAME, not from the observation: Qwen taking
    400 ms to describe a picture does not make the picture newer, and it is the
    picture the driver is being told about.

    WHOSE. See serve_to(). A record is a sentence about somebody's road, and
    which somebody is not something a timestamp can answer.
    """
    if max_age_s is None:
        max_age_s = config.OBSERVER_FRESH_S
    rec = cached(session_key)
    if not rec or not rec.get("text"):
        return {}
    if not serve_to(rec, session_key):
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
    # ...and it has to be this session's frame. Describing it now rather than a
    # second ago does not make somebody else's road this driver's.
    if not serve_to({"origin": getattr(frame, "origin", None)}, key):
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

        text = _clean(vision.observe(frame.jpeg, frame_id=frame.frame_id))
    except Exception as e:
        gpu_health.note("observer", e)
        print(f"[observer] {key}: on-demand observation failed: "
              f"{type(e).__name__}: {e}", flush=True)
        return {}
    if not text:
        return {}
    rec = _record(text, frame, key, vision.last_reading_meta())
    with _lock:
        st["record"] = rec
        st["n"] += 1
    if not serve_to(rec, key):
        return {}
    out = dict(rec)
    out["age_s"] = round(time.time() - (rec.get("frame_wall_t") or rec["at"]), 2)
    out["on_demand"] = True
    return out


def recent(session_key: str, max_age_s: float = None) -> dict:
    """The latest observation if it is recent ENOUGH TO ANSWER WITH. -> {} if not.

    A SECOND, WIDER WINDOW, and the two are for different jobs.

    `fresh()` is the one the fast path was built on: OBSERVER_FRESH_S, two
    seconds, the window inside which a description is about the road the car is
    on. That is the right bar for serving a sentence as though it were current.

    This one is the bar for serving it AT ALL, with its age attached. At 13 m/s
    a three-second-old description is about forty metres back -- still the road
    being driven, and the caller says how old it is, which is what makes the
    difference between a stale answer and a timestamped one.

    It exists because the alternative to a slightly old sentence was a
    synchronous Qwen pass measured at 1.0-5.8 s inside the turn. A driver would
    rather hear about three seconds ago than wait five for now.
    """
    s_max = float(max_age_s if max_age_s is not None
                  else getattr(config, "OBSERVER_ANSWER_MAX_AGE_S", 5.0))
    rec = cached(session_key)
    if not rec:
        return {}
    if not serve_to(rec, session_key):
        return {}
    frame_t = rec.get("frame_wall_t") or rec.get("at")
    age = time.time() - float(frame_t or 0)
    if age > s_max:
        return {}
    out = dict(rec)
    out["age_s"] = round(age, 2)
    out["stale_ok"] = True
    return out


def observe_soon(session_key: str, deadline_ms: float = None) -> dict:
    """observe_now, but it gives up on the CALLER's behalf. -> record or {}

    WHY THIS EXISTS. observe_now runs a Qwen forward pass synchronously inside
    look(), which is inside the turn a driver is waiting through. On an idle
    card that is ~0.4 s, which is the number its docstring quotes and the
    number it was written against. Measured on a working drive -- the detector
    at 15 fps, this loop at 1 Hz, and the shadow panel's models resident
    -- the same call ran 1.0 to 5.8 seconds. A scene question is supposed to
    take under a second in total.

    The pass cannot be interrupted: it is a local GPU call and Python has no
    way to stop a thread. So this does not try to cancel it. It runs it OFF the
    answer path, waits `deadline_ms`, and if it has not finished by then the
    answer goes on without it -- and the pass keeps running and FILES ITS
    RESULT, so the question after this one finds it in the cache.

    That is the same trade the tool endpoint already makes for an abandoned
    call: the work finishes into nothing rather than being waited on. Here it
    finishes into the cache, which is better than nothing.
    """
    ms = float(deadline_ms if deadline_ms is not None
               else getattr(config, "OBSERVER_ON_DEMAND_DEADLINE_MS", 600.0))
    key = str(session_key or "default")
    box = {}
    done = threading.Event()

    def work():
        try:
            box["rec"] = observe_now(key)
        except Exception:
            box["rec"] = {}
        finally:
            done.set()

    threading.Thread(target=work, daemon=True,
                     name=f"observe_soon:{key[:8]}").start()
    if done.wait(timeout=max(0.0, ms / 1000.0)):
        return box.get("rec") or {}
    with _lock:
        st = _sessions.get(key)
        if st is not None:
            st["on_demand_late"] = st.get("on_demand_late", 0) + 1
    return {}


def status() -> dict:
    """What is running, for /health and the selftests."""
    with _lock:
        return {key: {"observations": st["n"], "errors": st["errors"],
                      "held": st.get("hold", 0),
                      # Ticks that found the newest frame already too old to
                      # describe. Non-zero means the TRANSPORT is the problem
                      # and the observer is doing the right thing about it --
                      # a distinction the drive of 2026-09-09 had no way to
                      # make, because a stalled feed and a stalled observer
                      # look identical from the outside.
                      "stale_frames": st.get("stale_frames", 0),
                      "idle_s": round(time.time() - st["last_used"], 1),
                      "has_record": bool(st.get("record")),
                      # WHAT it is holding and WHICH FRAME it came from. Added
                      # while chasing an answer about a freeway served to a
                      # phone pointed at a desk: "the observer has a record"
                      # and "the record is about this session's road" are
                      # different facts, and only the first one was visible.
                      "text": (st.get("record") or {}).get("text", "")[:80],
                      "frame_id": (st.get("record") or {}).get("frame_id"),
                      "origin": (st.get("record") or {}).get("origin"),
                      # How often the running description came out in a voice
                      # that is not hers, and so could not be spoken without
                      # her composing over it.
                      "unspeakable": st.get("unspeakable", 0),
                      "speakable_now": bool((st.get("record") or {})
                                            .get("speakable"))}
                for key, st in _sessions.items()}


def stop_all() -> None:
    with _lock:
        states = list(_sessions.values())
    for st in states:
        st["stop"].set()
