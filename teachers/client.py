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

    def _take(self):
        """The next job worth doing, dropping any that have gone stale."""
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
                job = self._take()
                if job is None:
                    break
                self._run(job)

    def _run(self, job):
        t_send = time.time()
        with self._lock:
            self.stats["busy"] = True
            self.stats["sent"] += 1
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
