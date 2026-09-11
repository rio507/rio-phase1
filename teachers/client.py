"""One teacher, one loopback socket, one job in flight — and stale work thrown away.

WHAT THIS FILE IS DEFENDING AGAINST
-----------------------------------
A 10-billion-parameter model takes seconds to answer, and a keyframe is worth
about one second. Those two numbers do not fit, and there are only two ways to
reconcile them: queue, or drop. Queueing is what makes a shadow feature into a
liability -- the queue grows, every reading on the dashboard describes a road
that is further and further behind, and the memory holding four JPEGs per
waiting job grows with it. The transport layer of this project learned this the
expensive way and wrote it down (rio_frames.js, headway_ws): DROP, NEVER QUEUE.

So: a bounded slot list, config.TEACHER_QUEUE_DEPTH deep. A new job evicts the
oldest waiting one. A job that has been waiting long enough that its t0 is past
config.TEACHER_FRAME_MAX_AGE_S is not run at all -- it is dropped where it sits,
counted, and the counter is on the dashboard, because "the panel is quiet" and
"the panel is dropping everything" look identical otherwise.

WHY EACH TEACHER GETS ITS OWN THREAD
------------------------------------
So the two of them run at once. They are separate processes on separate ports
and there is no reason for Cosmos to wait for Alpamayo -- and, more to the
point, no reason for a hung Alpamayo to silence Cosmos. Every failure here is
per-teacher: a service that is down backs off on its own, and the other column
of the card carries on.

WHAT IT NEVER DOES
------------------
Raise into a caller. `submit()` is called from the frame path and returns a
bool. Everything else happens on this file's own thread.
"""
import base64
import json
import threading
import time
import urllib.error
import urllib.request

import config

# HOW MANY THINGS RIO IS DOING THAT A DRIVER IS WAITING ON. Module level and
# shared by both clients, because the card is shared. Re-entrant by count: two
# overlapping questions must not have the first one's exit resume the teachers
# under the second.
_HOLD = {"n": 0, "held_s": 0.0, "holds": 0, "since": 0.0}

# ---------------------------------------------------------------------------
# A LIVE SESSION IS OPEN. Same shape as the hold and a different question.
#
# The hold means "RIO is answering RIGHT NOW". This means "the driver could
# speak at any moment", which is most of a drive, and it is the state in which
# starting a 700 ms teacher pass is a bet that the next question arrives after
# it finishes. Measured: with both teachers inferring, one observer forward
# pass goes from 368 ms to 704 ms at p50 and 1379 ms at worst.
#
# Counted rather than flagged, for the same reason the hold is: two overlapping
# sessions must not have the first one's close resume the teachers under the
# second.
_SESSION = {"n": 0, "until": 0.0}

# ...AND ONLY ONE TEACHER INFERS AT A TIME.
#
# Two 8-10B models decoding at once on the same card is the peak of the
# contention this whole surface exists to bound, and the corpus does not want
# simultaneity -- it wants both models' reading of the same keyframe, which
# serialising delivers just as well and later. Module level, shared by both
# clients, because the card is shared. See config.TEACHER_MAX_CONCURRENT.
_GATE = threading.Semaphore(max(1, int(
    getattr(config, "TEACHER_MAX_CONCURRENT", 1) or 1)))


def set_max_concurrent(n: int):
    """Rebuild the gate. For tests, and for nothing on the live path.

    A semaphore's size is fixed at construction, so a test that wants the old
    both-at-once behaviour -- and there is one, because corpus row accounting
    under a speed skew between the two models is a different question from how
    many may decode at once -- cannot get it by assigning to the config.
    """
    global _GATE
    _GATE = threading.Semaphore(max(1, int(n or 1)))


def hold_gpu():
    """Context manager: no teacher starts new work while this is open."""
    return _Hold()


class _Hold:
    def __enter__(self):
        if _HOLD["n"] == 0:
            _HOLD["since"] = time.time()
        _HOLD["n"] += 1
        _HOLD["holds"] += 1
        return self

    def __exit__(self, *exc):
        _HOLD["n"] = max(0, _HOLD["n"] - 1)
        if _HOLD["n"] == 0 and _HOLD["since"]:
            _HOLD["held_s"] += time.time() - _HOLD["since"]
            _HOLD["since"] = 0.0
        return False


def hold_status():
    return {"held": _HOLD["n"], "holds": _HOLD["holds"],
            "held_s": round(_HOLD["held_s"], 2)}


def session_note(open_: bool, ttl_s: float = None):
    """A live conversation session opened or closed.

    A DEADLINE AS WELL AS A COUNT, and the deadline is the important half.
    A counter that is incremented on open and decremented on close is correct
    exactly as long as every close arrives -- and the cases this system already
    has machinery for (the phone that went flat, the laptop that slept, the tab
    closed while hidden; see _reap_abandoned_sessions in app.py) are precisely
    the ones where it does not.

    A leaked increment would pause the teachers for the life of the process:
    the panel would go quiet, no corpus row would ever be written again, and
    nothing would look broken. So the pause EXPIRES. Anything that knows a
    session is still open refreshes it; if nothing does, the teachers resume
    on their own.
    """
    ttl = float(ttl_s if ttl_s is not None
                else getattr(config, "TEACHER_SESSION_TTL_S", 180.0))
    if open_:
        _SESSION["n"] += 1
        _SESSION["until"] = max(_SESSION["until"], time.time() + ttl)
    else:
        _SESSION["n"] = max(0, _SESSION["n"] - 1)
        if _SESSION["n"] == 0:
            _SESSION["until"] = 0.0


def session_ping(ttl_s: float = None):
    """This session is still open. Pushes the deadline out, nothing else."""
    if _SESSION["n"] <= 0:
        return
    ttl = float(ttl_s if ttl_s is not None
                else getattr(config, "TEACHER_SESSION_TTL_S", 180.0))
    _SESSION["until"] = max(_SESSION["until"], time.time() + ttl)


def session_live() -> bool:
    return _SESSION["n"] > 0 and time.time() < _SESSION["until"]


def live_status():
    return {"sessions": _SESSION["n"],
            "live": session_live(),
            "expires_in_s": (round(max(0.0, _SESSION["until"] - time.time()), 1)
                             if _SESSION["n"] > 0 else None),
            "max_concurrent": max(1, int(
                getattr(config, "TEACHER_MAX_CONCURRENT", 1) or 1))}


class TeacherClient:
    """The RIO-side half of one teacher service."""

    def __init__(self, name: str, url: str, on_reading=None):
        self.name = name
        self.url = str(url or "").rstrip("/")
        self._on_reading = on_reading or (lambda name, job, reading: None)

        self._lock = threading.Lock()
        self._slots = []            # pending jobs, oldest first
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread = None

        self.depth = max(1, int(getattr(config, "TEACHER_QUEUE_DEPTH", 2)))
        self.timeout_s = float(getattr(config, "TEACHER_HTTP_TIMEOUT_S", 90.0))

        # Health, all of it visible on the card. A teacher is allowed to be
        # down; it is not allowed to be down invisibly.
        self.stats = {
            "submitted": 0, "evicted": 0, "stale_dropped": 0, "sent": 0,
            "ok": 0, "failed": 0, "streak": 0, "backoff_until": 0.0,
            "last_error": None, "last_ok_at": 0.0, "busy": False,
            "yielded": 0,
        }
        self.info = {}              # whatever /health last said about the model

    # -- lifecycle ----------------------------------------------------------
    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return False
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                        name=f"teacher:{self.name}")
        self._thread.start()
        return True

    def stop(self):
        self._stop.set()
        self._wake.set()

    # -- the queue ----------------------------------------------------------
    def submit(self, job: dict) -> bool:
        """Offer a keyframe. -> was it accepted into a slot?

        Called on the frame path. Takes a lock held only for a list operation;
        the eviction it may cause is REPORTED off this thread, below.
        """
        evicted = None
        with self._lock:
            if len(self._slots) >= self.depth:
                # NEWEST WINS. The evicted job describes a road already driven.
                evicted = self._slots.pop(0)
                self.stats["evicted"] += 1
            self._slots.append(job)
            self.stats["submitted"] += 1
        if evicted is not None:
            # EVERY SUBMITTED JOB PRODUCES EXACTLY ONE OUTCOME PER MODEL, and
            # this is why. The panel assembles a corpus row when every model
            # has reported on a keyframe; a job dropped here would otherwise
            # leave that row waiting forever for an answer that is never
            # coming, and the row -- including the OTHER model's real reading
            # of the same instant -- would be lost.
            self._report(evicted, "evicted")
        self._wake.set()
        return True

    def _report(self, job, why):
        """Tell the panel this job will never run. Never on the frame path."""
        threading.Thread(
            target=self._deliver, name=f"teacher:{self.name}:drop",
            args=(job, {"ok": False, "error": why, "not_run": True,
                        "latency_ms": 0.0}), daemon=True).start()

    def _deliver(self, job, reading):
        try:
            self._on_reading(self.name, job, reading)
        except Exception as e:
            print(f"[teachers.{self.name}] delivery failed: "
                  f"{type(e).__name__}: {e}", flush=True)

    def paused(self) -> bool:
        """Is RIO using the GPU for something a driver is waiting on -- or
        about to be?

        Two conditions now. The hold is an answer in progress. The session is
        the state in which one may begin at any moment, and starting a teacher
        pass in it is a bet on the driver staying quiet for the length of the
        pass. See config.TEACHER_PAUSE_DURING_SESSION.
        """
        if _HOLD["n"] > 0:
            return True
        if (getattr(config, "TEACHER_PAUSE_DURING_SESSION", True)
                and session_live()):
            return True
        return False

    def _take(self):
        """The next job worth doing, dropping any that have gone stale."""
        # THE TEACHERS YIELD. Measured on this pod: one observer forward pass
        # is 368 ms with the teachers idle and 704 ms (p50, 1379 max) with both
        # of them inferring -- 1.9x, on the call a driver is waiting through
        # when they ask what RIO can see. The teachers are shadow and the
        # driver is not, so while RIO is answering they stop taking work.
        #
        # A job already in flight is NOT preempted: it is inside another
        # process and there is nothing to preempt it with. What this buys is
        # that no NEW teacher inference starts during an answer, which is the
        # difference between one overlapping pass and an unbounded queue of
        # them.
        if self.paused():
            return None
        max_age = float(getattr(config, "TEACHER_FRAME_MAX_AGE_S", 1.0))
        now = time.time()
        with self._lock:
            while self._slots:
                job = self._slots.pop(0)
                # THE SECOND FRESHNESS GATE, and the one that actually fires.
                # The first is at keyframe build time, when the frame is new by
                # construction. This one is at pick-up, after the job has spent
                # however long the previous inference took sitting in a slot --
                # which is where the seconds actually go.
                age = now - float(job.get("t0") or now)
                if age > max_age * 2.0:
                    self.stats["stale_dropped"] += 1
                    self._report(job, "stale_dropped")
                    continue
                return job
        return None

    # -- the worker ---------------------------------------------------------
    # How often an idle worker re-asks its service what it is. A service
    # started AFTER the panel -- which is the normal order, since boot.sh
    # brings uvicorn up first -- would otherwise show "not loaded" on the card
    # for the whole drive, because health was probed once at start-up.
    HEALTH_EVERY_S = 20.0

    def _loop(self):
        last_health = 0.0
        while not self._stop.is_set():
            self._wake.wait(timeout=1.0)
            self._wake.clear()
            if (not self.info.get("loaded")
                    and time.time() - last_health > self.HEALTH_EVERY_S):
                last_health = time.time()
                self.health(timeout_s=2.0)
            while not self._stop.is_set():
                if time.time() < self.stats["backoff_until"]:
                    break
                if self.paused():
                    with self._lock:
                        self.stats["yielded"] += 1
                    break
                job = self._take()
                if job is None:
                    break
                self._run(job)

    def _run(self, job):
        t_send = time.time()
        with self._lock:
            self.stats["busy"] = True
            self.stats["sent"] += 1
        # ONE AT A TIME, ACROSS BOTH CLIENTS. Held around the HTTP call, which
        # is where the GPU work actually happens -- the service on the far end
        # is synchronous per request, so a client waiting here is a model not
        # decoding. Acquired outside the try so a failure to get it cannot be
        # reported as an inference failure.
        _GATE.acquire()
        try:
            reading = self._post(job)
            with self._lock:
                self.stats["ok"] += 1
                self.stats["streak"] = 0
                self.stats["last_ok_at"] = time.time()
                self.stats["last_error"] = None
        except Exception as e:
            msg = f"{type(e).__name__}: {e}"
            with self._lock:
                self.stats["failed"] += 1
                self.stats["streak"] += 1
                self.stats["last_error"] = msg[:200]
                if self.stats["streak"] >= int(getattr(config, "TEACHER_FAIL_STREAK", 3)):
                    self.stats["backoff_until"] = time.time() + float(
                        getattr(config, "TEACHER_FAIL_BACKOFF_S", 15.0))
            reading = {"ok": False, "error": msg[:200]}
        finally:
            _GATE.release()
            with self._lock:
                self.stats["busy"] = False

        reading = dict(reading or {})
        reading["queue_ms"] = round((t_send - float(job.get("submitted_at")
                                                    or t_send)) * 1000.0, 1)
        reading["freshness_s"] = round(time.time() - float(job.get("t0") or t_send), 2)
        # A reading that cannot be filed is a reading lost, never a worker
        # thread lost. The next keyframe still gets one.
        self._deliver(job, reading)

    def _post(self, job):
        body = json.dumps(self._payload(job)).encode()
        req = urllib.request.Request(
            self.url + "/infer", data=body,
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as r:
            return json.loads(r.read().decode())

    @staticmethod
    def _payload(job):
        """The wire form. Both services take exactly this, which is the point.

        The frames go as base64 JPEG -- the same bytes the client sent and the
        detector measured, never re-encoded, so the teacher and RF-DETR are
        looking at identical pixels and a disagreement between them is about
        the models rather than about two JPEG passes.
        """
        return {
            "kf_id": job["kf_id"],
            "t0": job["t0"],
            "frames": [base64.b64encode(b).decode() for b in job["jpegs"]],
            "frame_offsets_s": job["frame_offsets_s"],
            "ego_history_xyz": job.get("ego_xyz"),
            "ego_history_yaw": job.get("ego_yaw"),
            "prompts": job["prompts"],
            # Alpamayo ignores this key; Cosmos asks it as its fourth question.
            # Sent to both anyway, so the two services take the SAME payload
            # and the selftest can assert that with one fixture.
            "physical_prompt": job.get("physical_prompt") or "",
            "image": job.get("image") or {},
        }

    # -- health -------------------------------------------------------------
    def health(self, timeout_s: float = 3.0) -> dict:
        """Ask the service what it is. Cheap, and safe to call from a request."""
        try:
            with urllib.request.urlopen(self.url + "/health", timeout=timeout_s) as r:
                self.info = json.loads(r.read().decode())
        except Exception as e:
            self.info = {"ok": False, "error": f"{type(e).__name__}: {e}"}
        return self.info

    def status(self) -> dict:
        with self._lock:
            st = dict(self.stats)
            st["queued"] = len(self._slots)
        st["name"] = self.name
        st["url"] = self.url
        st["backoff_s"] = max(0.0, round(st.pop("backoff_until") - time.time(), 1))
        st["model"] = self.info.get("model_id") or None
        st["precision"] = self.info.get("precision") or None
        st["loaded"] = bool(self.info.get("loaded"))
        return st
