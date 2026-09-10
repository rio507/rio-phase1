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


def flatten_patch_embed(visual, verify=True, log=print):
    """Make Qwen3-VL's vision patch embedding fast, without changing a number.

    THE MEASUREMENT THAT MOTIVATES THIS. On this pod, one forward pass of
    Cosmos-Reason2's vision tower over a four-frame window took 15.0 s. Broken
    down: the 27 transformer blocks 16 ms, the positional embedding 2 ms, the
    rotary table 0.5 ms, the merger 0.1 ms -- and `patch_embed` 13.7 s. One
    module, 99.9% of the time, and four of them per keyframe made Cosmos a
    57-second teacher on a 2-second cadence. Every keyframe would have been
    evicted; the second column of the card would have been empty on the road.

    WHY IT IS SLOW, AND WHY THIS IS NOT A TRICK. The module is an nn.Conv3d
    whose kernel_size EQUALS its stride and equals the whole spatial extent of
    each input volume: 2400 separate (3, 2, 16, 16) blocks, each producing
    exactly one output voxel. cuDNN picks a pathological algorithm for that
    shape here -- `torch.backends.cudnn.benchmark = True` does not help, it was
    measured at 13.8 s -- but the operation itself is, exactly and by
    definition, a matrix multiply: with no overlap and no padding there is one
    dot product per block between the flattened block and the flattened kernel.

    So the weight is reshaped from (E, C, T, P, P) to (E, C*T*P*P), the input
    from (N, C, T, P, P) to (N, C*T*P*P), and the conv becomes one GEMM. Same
    weights, same bias, same arithmetic, same order of the flattened axes.
    0.0 ms.

    AND IT IS CHECKED, not asserted in a comment. `verify` runs both forms on
    random input at load and refuses the swap if they disagree beyond bf16
    rounding -- because a silent numerical change in the first layer of the
    vision tower would be invisible in every reading afterwards, and would look
    exactly like the model being worse at driving.

    -> a dict describing what happened, for the service's /health.
    """
    import torch
    from torch import nn

    pe = getattr(visual, "patch_embed", None)
    proj = getattr(pe, "proj", None)
    if pe is None or not isinstance(proj, nn.Conv3d):
        return {"applied": False, "reason": "no Conv3d patch_embed"}
    k, st, pad, dil = (tuple(proj.kernel_size), tuple(proj.stride),
                       tuple(proj.padding), tuple(proj.dilation))
    # The equivalence holds ONLY when the patches do not overlap, are not
    # padded and are not dilated. Anything else and this is a different
    # operation, so it is left alone.
    if k != st or any(pad) or any(d != 1 for d in dil) or proj.groups != 1:
        return {"applied": False,
                "reason": f"not a non-overlapping patch conv "
                          f"(k={k} stride={st} pad={pad} dil={dil})"}

    E = proj.out_channels
    fan = proj.in_channels * k[0] * k[1] * k[2]
    W = proj.weight.detach().reshape(E, fan).contiguous()
    b = None if proj.bias is None else proj.bias.detach().clone()

    if verify:
        torch.manual_seed(0)
        x = torch.randn(64 * fan, dtype=W.dtype, device=W.device)
        with torch.no_grad():
            ref = proj(x.view(-1, proj.in_channels, *k)).view(-1, E)
            got = torch.nn.functional.linear(x.view(-1, fan), W, b)
        # bf16 has ~3 decimal digits; a real difference is orders larger.
        tol = 5e-2 if W.dtype in (torch.bfloat16, torch.float16) else 1e-4
        err = (ref.float() - got.float()).abs().max().item()
        if err > tol:
            return {"applied": False,
                    "reason": f"forms disagree by {err:.4g} (> {tol})"}
    else:
        err = None

    class LinearPatchEmbed(nn.Module):
        def __init__(self, weight, bias, in_channels, kernel, embed_dim):
            super().__init__()
            self.register_buffer("weight", weight, persistent=False)
            if bias is None:
                self.bias = None
            else:
                self.register_buffer("bias", bias, persistent=False)
            self.in_channels = in_channels
            self.kernel = kernel
            self.embed_dim = embed_dim
            self.fan = in_channels * kernel[0] * kernel[1] * kernel[2]

        def forward(self, hidden_states):
            return torch.nn.functional.linear(
                hidden_states.view(-1, self.fan).to(self.weight.dtype),
                self.weight, self.bias)

    visual.patch_embed = LinearPatchEmbed(W, b, proj.in_channels, k, E)
    log(f"[patch_embed] Conv3d{k} -> linear ({fan} -> {E}); "
        f"max deviation {err:.3g}" if err is not None
        else f"[patch_embed] Conv3d{k} -> linear ({fan} -> {E})")
    return {"applied": True, "kernel": list(k), "fan_in": fan,
            "embed_dim": E, "max_deviation": err}


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
    """The PEAK the allocator has ever reserved, since the last reset.

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


def vram_now_mb():
    """What the process is holding RIGHT NOW.

    THE NUMBER AN L40S BUDGET ACTUALLY NEEDS, and it is not the same as the
    peak. `max_memory_reserved` is a high-water mark that is never reset, so it
    includes whatever the LOAD spiked to -- and loading an FP8 checkpoint
    spikes hard, because compressed-tensors materialises each layer before
    placing it. Reporting only the peak made FP8 look like it needed MORE card
    than BF16 while its weights are a third smaller, which is true of the load
    and false of the drive.

    So: three numbers, and they answer three questions. `weights_mb` is what
    the model costs to hold, `vram_now_mb` is what the process is sitting on
    between keyframes, and `vram_reserved_mb` is the worst it has ever been.
    """
    try:
        import torch

        if not torch.cuda.is_available():
            return 0.0
        return round(torch.cuda.memory_reserved() / (1024 ** 2), 1)
    except Exception:
        return 0.0


def reset_vram_peak():
    """Forget the load's high-water mark, so inference is measured on its own."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            return True
    except Exception:
        pass
    return False


class Service:
    """Subclass this: implement `load()`, `infer(payload)` and `describe()`."""

    name = "teacher"

    def __init__(self, depth=1):
        self.bound = Bound(depth)
        self.loaded = False
        self.load_error = None
        self.started_at = time.time()
        self.weights_mb = None
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
            # WHAT THE WEIGHTS COST, captured before a single inference has
            # allocated anything -- and then the peak is reset, so every
            # number after this describes the drive rather than the load.
            self.weights_mb = vram_now_mb()
            reset_vram_peak()
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
             "vram_now_mb": vram_now_mb(), "weights_mb": self.weights_mb,
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
                          "vram_reserved_mb": vram_reserved_mb(),
                          "vram_now_mb": vram_now_mb(),
                          "weights_mb": self.weights_mb}
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
