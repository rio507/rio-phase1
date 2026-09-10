"""The shell both teacher services live in: a socket, a bound, and a clock.

STDLIB ONLY, and that is a requirement rather than a preference. This file is
imported by two Python environments that share nothing with each other and
nothing with RIO -- different torch, different transformers, different Python
patch level. The one thing they can all be relied on to have is the standard
library, so http.server it is. There is no FastAPI here and there should never
be one: the entire API is two GETs and a POST.

THE BOUND
---------
A GPU does one thing at a time. Two concurrent /infer calls do not go twice as
fast; they go half as fast each, allocate twice the activations, and turn a
tight VRAM budget into an out-of-memory. So inference is serialised behind one
semaphore, and callers beyond `queue_depth` are refused IMMEDIATELY with a 503
rather than being queued behind a model that will not finish in time to matter.

That refusal is the honest answer. RIO's own client drops stale work for the
same reason (teachers/client.py), and a service that queued politely would just
move the staleness somewhere the dashboard cannot see it.

THE CLOCK
---------
`latency_ms` on every response is the model's time and only the model's time:
it starts after the request is parsed and the frames are decoded, and stops
when generation returns. Weight loading is excluded because it happens once,
and JPEG decode is excluded because it is a property of the transport rather
than of the model -- and the whole point of the number is to compare BF16
against FP8 and Alpamayo against Cosmos.
"""
import base64
import json
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class Bound:
    """`depth` callers may be inside at once; the rest are turned away now."""

    def __init__(self, depth=1):
        self._sem = threading.Semaphore(depth)
        self._lock = threading.Lock()
        self.depth = depth
        self.inflight = 0
        self.refused = 0

    def __enter__(self):
        if not self._sem.acquire(blocking=False):
            with self._lock:
                self.refused += 1
            raise Busy(f"busy: {self.inflight}/{self.depth} in flight")
        with self._lock:
            self.inflight += 1
        return self

    def __exit__(self, *exc):
        with self._lock:
            self.inflight -= 1
        self._sem.release()
        return False


class Busy(RuntimeError):
    pass


def decode_frames(payload):
    """base64 JPEGs -> PIL images, oldest first. Raises on a malformed batch.

    Imported lazily so this module stays importable without Pillow -- the
    selftest reads it, and the selftest has no business needing a venv.
    """
    from io import BytesIO

    from PIL import Image

    out = []
    for b64 in payload.get("frames") or []:
        img = Image.open(BytesIO(base64.b64decode(b64)))
        out.append(img.convert("RGB"))
    if not out:
        raise ValueError("no frames in payload")
    return out


def vram_mb():
    """Peak allocation this process has reached, in MB. 0 without CUDA."""
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1)
    except Exception:
        return 0.0


def vram_reserved_mb():
    """What the process is HOLDING, which is the number that fills a card.

    Different from max_memory_allocated on purpose: the allocator keeps freed
    blocks, so "how much does this model need" and "how much of the GPU is
    gone" are different questions and an L40S budget cares about the second.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return round(torch.cuda.max_memory_reserved() / (1024 ** 2), 1)
    except Exception:
        return 0.0


class Service:
    """Subclass this: implement `load()`, `infer(payload)` and `describe()`."""

    name = "teacher"

    def __init__(self, depth=1):
        self.bound = Bound(depth)
        self.loaded = False
        self.load_error = None
        self.started_at = time.time()
        self.stats = {"infer": 0, "ok": 0, "failed": 0, "busy": 0,
                      "last_error": None, "last_latency_ms": None}

    # -- to implement -------------------------------------------------------
    def load(self):
        raise NotImplementedError

    def infer(self, payload):
        raise NotImplementedError

    def describe(self):
        return {}

    # -- the shell ----------------------------------------------------------
    def ensure_loaded(self):
        if self.loaded:
            return True
        try:
            self.load()
            self.loaded = True
            self.load_error = None
        except Exception as e:
            self.load_error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        return self.loaded

    def health(self):
        d = {"ok": bool(self.loaded), "name": self.name,
             "loaded": self.loaded, "load_error": self.load_error,
             "uptime_s": round(time.time() - self.started_at, 1),
             "inflight": self.bound.inflight, "queue_depth": self.bound.depth,
             "refused": self.bound.refused,
             "vram_peak_mb": vram_mb(), "vram_reserved_mb": vram_reserved_mb(),
             "stats": dict(self.stats)}
        d.update(self.describe())
        return d

    def handle_infer(self, payload):
        with self.bound:
            if not self.ensure_loaded():
                return {"ok": False, "error": self.load_error or "not loaded"}
            self.stats["infer"] += 1
            t0 = time.perf_counter()
            try:
                out = self.infer(payload)
                out.setdefault("ok", True)
                self.stats["ok"] += 1
            except Exception as e:
                traceback.print_exc()
                self.stats["failed"] += 1
                self.stats["last_error"] = f"{type(e).__name__}: {e}"
                out = {"ok": False, "error": f"{type(e).__name__}: {e}"}
            ms = round((time.perf_counter() - t0) * 1000.0, 1)
            out.setdefault("latency_ms", ms)
            self.stats["last_latency_ms"] = out["latency_ms"]
            out.setdefault("kf_id", payload.get("kf_id"))
            out["gpu"] = {"vram_peak_mb": vram_mb(),
                          "vram_reserved_mb": vram_reserved_mb()}
            out.update(self.describe())
            return out


def serve(service, host="127.0.0.1", port=8801, warm=True):
    """Run `service` until killed. Loopback by default and by intention."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, code, obj):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path.startswith("/health"):
                self._send(200, service.health())
            elif self.path.startswith("/stats"):
                self._send(200, dict(service.stats))
            else:
                self._send(404, {"ok": False, "error": "not found"})

        def do_POST(self):
            if not self.path.startswith("/infer"):
                self._send(404, {"ok": False, "error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(n).decode())
            except Exception as e:
                self._send(400, {"ok": False, "error": f"bad payload: {e}"})
                return
            try:
                self._send(200, service.handle_infer(payload))
            except Busy as e:
                service.stats["busy"] += 1
                # 503, not 429: this is "the GPU is occupied", which is a
                # capacity statement about the server, and the client's own
                # bounded queue is what decides whether to try again.
                self._send(503, {"ok": False, "error": str(e), "busy": True})

        def log_message(self, fmt, *args):
            # The default handler logs every request to stderr, which at a
            # keyframe every two seconds for a whole drive is a lot of noise
            # around the two lines that matter. Failures are logged where they
            # happen instead.
            pass

    if warm:
        print(f"[{service.name}] loading...", flush=True)
        t0 = time.time()
        ok = service.ensure_loaded()
        print(f"[{service.name}] {'ready' if ok else 'FAILED'} in "
              f"{time.time() - t0:.1f}s  vram={vram_reserved_mb()} MB",
              flush=True)

    httpd = ThreadingHTTPServer((host, port), Handler)
    print(f"[{service.name}] listening on http://{host}:{port}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
