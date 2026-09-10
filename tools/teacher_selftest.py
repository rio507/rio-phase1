"""The teacher panel, without a GPU: shapes, refusals, isolation and the record.

    python -m tools.teacher_selftest
    python -m tools.teacher_selftest --live      also drive a running service

WHAT IS UNDER TEST AND WHAT IS STUBBED
--------------------------------------
Stubbed: the two models. What they say is not this suite's business -- it is
the models' -- and a test that needs 37 GB of weights is a test nobody runs.

Under test: everything between the frame arriving and the corpus row being
written, which is where all of the bugs actually live. The window that has to
be four frames at the right spacing. The ego history that has to end at the
origin. The freshness gate that has to drop rather than queue. The service that
has to fail without taking the panel with it. The row that has to still be
readable in a year.

THE ISOLATION TESTS ARE THE POINT
---------------------------------
Two of them, in opposite directions, because they fail in different ways:

  A DEAD TEACHER MUST COST NOTHING. Not the frame, not the drive, not the other
  teacher's column. Asserted by pointing a client at a closed port and checking
  that everything else still works and that the failure is visible on the card
  rather than only in a log.

  A LIVE TEACHER MUST NOT BE ABLE TO REACH ANYTHING. That is the firewall's
  job -- tools/teacher_firewall_selftest.py -- and it is a separate suite
  because it reads the syntax tree rather than running anything. Run both.
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                                     # noqa: E402

config.TEACHERS_ENABLED = True

import framebuf                                                   # noqa: E402
from teachers import associate as assoc_mod                       # noqa: E402
from teachers import client as client_mod                         # noqa: E402
from teachers import corpus as corpus_mod                         # noqa: E402
from teachers import egomotion                                    # noqa: E402
from teachers import keyframe as kf_mod                           # noqa: E402
from teachers import panel                                        # noqa: E402
from teachers import schema                                       # noqa: E402

PASS, FAIL = [], []


def ok(cond, what):
    (PASS if cond else FAIL).append(what)
    print(("  ok    " if cond else "  FAIL  ") + what)


def section(name):
    print(f"\n=== {name} ===")


# ---------------------------------------------------------------------------
# a fake teacher: the same wire protocol, none of the weights
# ---------------------------------------------------------------------------
class FakeTeacher:
    """One thread, one port, and a knob for every way a model can misbehave."""

    def __init__(self, name, port, delay=0.0):   # delay: how slow this one is
        self.name = name
        self.port = port
        self.delay = delay
        self.seen = []
        self.lock = threading.Lock()
        svc = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def _send(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                self._send(200, {"ok": True, "loaded": True, "name": svc.name,
                                 "model_id": "fake/" + svc.name,
                                 "precision": "bf16"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(n).decode())
                with svc.lock:
                    svc.seen.append(payload)
                if svc.delay:
                    time.sleep(svc.delay)
                self._send(200, svc.answer(payload))

            def log_message(self, *a):
                pass

        self.httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    def answer(self, payload):
        return {
            "ok": True, "kf_id": payload.get("kf_id"),
            "model": self.name, "model_id": "fake/" + self.name,
            "revision": "deadbeef", "precision": "bf16",
            "latency_ms": 1234.5,
            "raw": {"scene": {"question": "Describe the scene.",
                              "answer": "A two-lane road with a white van ahead."}},
            # VARIES PER KEYFRAME, because a real model's does. A fixture
            # that answers every keyframe with the same sentence is itself
            # reciting, and teachers/canned.py flags it -- correctly, which is
            # how this line came to be written.
            "scene": ("A two-lane road with a white van ahead ("
                      + str(payload.get("kf_id")) + ")."),
            "critical_actor": "The white van directly ahead, because it is braking.",
            "attention": "The van's brake lights and the gap closing.",
            "reasoning": "The van ahead is decelerating; the ego should ease off.",
            "thinking": "step one. step two.",
            "meta_action": "decelerate",
            # Only Alpamayo predicts a path. The fake models differ here for
            # the same reason the real ones do, so the row's `trajectory: null`
            # for Cosmos is exercised rather than assumed.
            "trajectory": ({"xyz": [[i * 1.4, 0.0, 0.0] for i in range(64)],
                            "hz": 10, "horizon_s": 6.4, "frame": "FLU_at_t0",
                            "pixels": None}
                           if self.name == "alpamayo1.5" else None),
            "gpu": {"vram_reserved_mb": 24000.0},
            # Fields the schema never named, which the REAL services do send.
            # Here so the corpus's `extra` passthrough is exercised rather
            # than assumed -- it is the thing that stops "every field the
            # model offers, verbatim" quietly meaning "every field somebody
            # thought of in advance".
            "timings_ms": {"rollout_ms": 900.0, "scene_ms": 110.0},
            "cameras": 1,
            "code_revision": "36aeb4c",
        }

    def stop(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def make_jpeg(w=640, h=480, seed=0):
    """A real JPEG, so nothing under test is exercising a placeholder."""
    try:
        import cv2
        import numpy as np

        a = np.zeros((h, w, 3), np.uint8)
        a[: h // 2] = (200 - seed, 180, 160)
        a[h // 2:] = (70, 70, 75 + seed)
        return cv2.imencode(".jpg", a)[1].tobytes()
    except Exception:
        # No cv2 in this interpreter: the bytes only have to be bytes for
        # everything except the service, which is not under test here.
        return b"\xff\xd8\xff\xe0" + bytes([seed % 251]) * 900 + b"\xff\xd9"


def push_window(key, n=8, spacing=0.1, base=None, **result_extra):
    """n frames into `key`'s ring, `spacing` apart, newest at `base`."""
    ring = framebuf.get_ring(key)
    base = time.time() if base is None else base
    result = {"ok": True, "scene_objects": [], "image": {"w": 640, "h": 480},
              "t": 0.0}
    result.update(result_extra)
    out = []
    for i in range(n):
        r = dict(result)
        r["t"] = round(i * spacing, 3)
        f = ring.push(make_jpeg(seed=i), r, origin=f"{key}:camera")
        if f is not None:
            f.wall_t = base - (n - 1 - i) * spacing
            out.append(f)
    return ring, out


TRACKS = [
    {"id": 7, "label": "truck", "box": [560, 260, 840, 430], "range_m": 21.4,
     "score": 0.9, "member": True, "is_lead": True, "confirmed": True},
    {"id": 8, "label": "pedestrian", "box": [90, 300, 150, 450], "range_m": 11.0,
     "score": 0.7, "member": False, "is_lead": False, "confirmed": True},
]

RESULT = {
    "ok": True, "t": 12.5, "band": "GETTING_UNSAFE", "distance_m": 21.4,
    "ttc_s": 4.1, "tau_s": 1.7, "trend": "closing", "lead_id": 7,
    "corridor_source": "ufld", "lane_conf": 0.72,
    "speed": {"v_ms": 12.5, "source": "obd", "degraded": False,
              "reason": "obd fresh"},
    "scene_objects": TRACKS, "image": {"w": 640, "h": 480},
}


# ---------------------------------------------------------------------------
def run_window():
    section("A. the shared input — one window, both teachers")
    key = "kf-window"
    framebuf.drop_ring(key)
    ring, frames = push_window(key, n=10, spacing=0.1)
    kf = kf_mod.build(key, ring, RESULT, "floor", ego=None, seq=0)
    ok(kf.get("ok"), f"a healthy 10 Hz ring yields a window ({kf.get('reason')})")
    ok(len(kf["frames"]) == config.TEACHER_WINDOW_FRAMES,
       f"four frames, not three and not five ({len(kf['frames'])})")
    ok(kf["frame_offsets_s"] == [-0.3, -0.2, -0.1, 0.0],
       f"at t0-0.3 .. t0, in that order ({kf['frame_offsets_s']})")
    ok(kf["window"]["exact"],
       f"and at 10 fps the window is exact "
       f"({kf['window']['spacing_error_ms']} ms worst slot error)")
    ids = [f.frame_id for f in kf["frames"]]
    ok(len(set(ids)) == 4, f"four distinct frames ({ids})")
    ok(kf["window"]["origin"] == f"{key}:camera",
       "the window names whose camera it came from")

    section("A2. the same window, at the POST path's 4 fps")
    key2 = "kf-slow"
    framebuf.drop_ring(key2)
    ring2, _ = push_window(key2, n=8, spacing=0.25)
    kf2 = kf_mod.build(key2, ring2, RESULT, "floor", seq=0)
    ok(kf2.get("ok"), "a 4 fps ring still yields a window")
    ok(not kf2["window"]["exact"],
       f"...and it is honestly marked inexact "
       f"({kf2['window']['spacing_error_ms']} ms worst slot error)")
    ok(len(kf2["frames"]) == 4,
       "four slots are still filled, by repeating the nearest frame — the "
       "model's prompt scaffolding is indexed by position and cannot take three")

    section("A3. both teachers are handed the SAME bytes")
    job = kf_mod.to_job(kf)
    p = client_mod.TeacherClient._payload(job)
    ok(len(p["frames"]) == 4, "four base64 frames on the wire")
    import base64

    ok([base64.b64decode(b) for b in p["frames"]] == [f.jpeg for f in kf["frames"]],
       "the JPEGs are the client's own bytes, not re-encoded — so the teachers "
       "and RF-DETR are looking at identical pixels")
    ok(p["t0"] == kf["t0"], "and at the same t0")
    for k in ("kf_id", "t0", "frames", "frame_offsets_s", "ego_history_xyz",
              "ego_history_yaw", "prompts", "physical_prompt", "image"):
        ok(k in p, f"the payload carries {k}")

    section("A4. the payload is what the services actually read")
    # Parsed out of the service sources rather than imported: those modules run
    # in a different interpreter and this one cannot load torch.
    import ast
    import re as _re

    for svc in ("alpamayo_service", "cosmos_service"):
        src = open(os.path.join(os.path.dirname(__file__), "..", "teachers",
                                "service", f"{svc}.py")).read()
        wanted = set(_re.findall(r'payload\.get\("([a-z_]+)"', src))
        missing = wanted - set(p)
        ok(not missing,
           f"{svc} reads only keys the payload has"
           + (f" — MISSING {sorted(missing)}" if missing else ""))
        ast.parse(src)


def run_ego():
    section("B. ego-motion history — 16 poses ending at the origin")
    egomotion.reset_all()
    key = "ego"
    t0 = time.time()
    for i in range(40):
        egomotion.note_speed(key, 14.0, "obd", at=t0 - 3.0 + i * 0.1)
    # A steady left turn: gravity down the phone's -z, rotation about +z.
    for i in range(60):
        egomotion.ingest(key, [{"t": t0 - 3.0 + i * 0.05,
                                "rr": [0.0, 0.0, 6.0], "g": [0.0, 0.0, -9.81]}])
    h = egomotion.history(key, t0)
    ok(h is not None, "a history is built")
    ok(h["steps"] == 16 and len(h["xyz"]) == 16 and len(h["yaw"]) == 16,
       f"16 poses at {h['step_s']} s ({len(h['xyz'])})")
    ok(h["xyz"][-1] == [0.0, 0.0, 0.0] and abs(h["yaw"][-1]) < 1e-9,
       f"the LAST pose is the origin — the frame is the ego frame AT t0 "
       f"({h['xyz'][-1]}, yaw {h['yaw'][-1]})")
    ok(h["xyz"][0][0] < -15.0,
       f"1.5 s at 14 m/s puts the oldest pose ~21 m behind ({h['xyz'][0][0]} m)")
    ok(h["source"]["speed"] == "obd" and h["source"]["yaw_rate"] == "imu",
       f"and it says where both ingredients came from ({h['source']})")
    ok(h["yaw"][0] < 0,
       f"a LEFT turn leaves the past heading clockwise of now ({h['yaw'][0]})")

    section("B2. the frame convention, and the matrices that carry it")
    rots = egomotion.rot_matrices(h["yaw"])
    ok(len(rots) == 16 and len(rots[0]) == 3 and len(rots[0][0]) == 3,
       "16 3x3 rotation matrices")
    import math

    det = sum(rots[0][0][i] * (rots[0][1][(i + 1) % 3] * rots[0][2][(i + 2) % 3]
                               - rots[0][1][(i + 2) % 3] * rots[0][2][(i + 1) % 3])
              for i in range(3))
    ok(abs(det - 1.0) < 1e-6, f"and they are rotations (det {det:.6f})")
    ok(h["frame"] == "FLU_at_t0",
       "the convention is on the record — x forward, y left, z up, at t0")

    section("B3. no IMU is a supported drive, not a broken one")
    egomotion.reset_all()
    for i in range(40):
        egomotion.note_speed("noimu", 9.0, "gps", at=t0 - 3.0 + i * 0.1)
    h2 = egomotion.history("noimu", t0)
    ok(h2 is not None, "a history is still built from speed alone")
    ok(h2["source"]["yaw_rate"] == "none" and not h2["quality"]["imu"],
       "...and it says so, so a corpus reader can filter it out")
    ok(all(abs(y) < 1e-9 for y in h2["yaw"]), "a straight line behind the car")

    section("B4. no speed at all is a refusal")
    egomotion.reset_all()
    ok(egomotion.history("nothing", t0) is None,
       "with no speed there is no history — a zero history would tell the "
       "model the car is stopped, which is a lie it would act on")

    section("B5. the phone's mounting does not matter")
    egomotion.reset_all()
    for mount, g, rr in (("flat on a seat", [0, 0, -9.81], [0, 0, 5.0]),
                         ("upright in a cradle", [0, 9.81, 0], [0, -5.0, 0]),
                         ("tilted", [0, 6.94, -6.94], [0, -3.54, 3.54])):
        y = egomotion.yaw_rate_from_sample(rr, g)
        ok(y is not None and abs(math.degrees(y) - 5.0) < 0.2,
           f"{mount}: {math.degrees(y):.2f} deg/s (want 5.00) — resolved "
           f"against gravity, not against the phone's own axes")
    ok(egomotion.yaw_rate_from_sample([0, 0, 5.0], [0, 0, -0.2]) is None,
       "a sample whose gravity is not gravity is refused rather than trusted")


def run_stale():
    section("C. stale is dropped, never queued")
    key = "stale"
    framebuf.drop_ring(key)
    old = time.time() - 5.0
    ring, _ = push_window(key, n=8, spacing=0.1, base=old)
    kf = kf_mod.build(key, ring, RESULT, "floor", seq=0)
    ok(not kf.get("ok") and kf.get("reason") == "stale_frame",
       f"a 5 s old ring produces no keyframe ({kf.get('reason')})")

    section("C2. ...and again at pick-up, which is where the seconds go")
    c = client_mod.TeacherClient("x", "http://127.0.0.1:9")
    fresh_job = {"kf_id": "a", "t0": time.time(), "submitted_at": time.time()}
    stale_job = {"kf_id": "b", "t0": time.time() - 30.0,
                 "submitted_at": time.time()}
    c.submit(stale_job)
    took = c._take()
    ok(took is None and c.stats["stale_dropped"] == 1,
       "a job whose t0 has aged out while it waited is dropped at pick-up, "
       "and counted")
    c.submit(fresh_job)
    ok((c._take() or {}).get("kf_id") == "a", "a fresh one is taken")

    section("C3. the queue is bounded and newest wins")
    c2 = client_mod.TeacherClient("y", "http://127.0.0.1:9")
    now = time.time()
    for i in range(6):
        c2.submit({"kf_id": f"k{i}", "t0": now, "submitted_at": now})
    ok(len(c2._slots) == config.TEACHER_QUEUE_DEPTH,
       f"at most {config.TEACHER_QUEUE_DEPTH} wait ({len(c2._slots)})")
    ok(c2.stats["evicted"] == 6 - config.TEACHER_QUEUE_DEPTH,
       f"and the evictions are counted ({c2.stats['evicted']})")
    ok([j["kf_id"] for j in c2._slots] == ["k4", "k5"],
       "the ones kept are the NEWEST — an evicted keyframe describes a road "
       "the car has already driven")


def run_service_isolation():
    section("D. a dead teacher costs nothing")
    panel.reset_all()
    panel.stop()
    config.TEACHER_ALPAMAYO_URL = "http://127.0.0.1:1"     # nothing listens
    config.TEACHER_COSMOS_URL = "http://127.0.0.1:1"
    panel.start()
    key = "dead"
    framebuf.drop_ring(key)
    ring, _ = push_window(key, n=8, spacing=0.1)
    for i in range(40):
        egomotion.note_speed(key, 12.0, "obd", at=time.time() - 3.0 + i * 0.1)

    t = time.perf_counter()
    raised = panel.on_frame(key, RESULT, ring)
    cost_ms = (time.perf_counter() - t) * 1000
    ok(raised, "a keyframe is still raised")
    ok(cost_ms < 60, f"and the frame path pays {cost_ms:.1f} ms for it, not a "
                     f"round trip to a closed port")

    deadline = time.time() + 8
    while time.time() < deadline:
        st = panel.status()["services"]
        if all(s["failed"] for s in st.values()):
            break
        time.sleep(0.2)
    st = panel.status()["services"]
    ok(all(s["failed"] >= 1 for s in st.values()),
       f"both clients record the failure ({[s['failed'] for s in st.values()]})")
    ok(all(s["last_error"] for s in st.values()),
       "with the error kept, so the card can say what happened")
    rows = corpus_mod.read_rows(key)
    ok(rows, f"a row is still written, recording that the panel was down "
             f"({len(rows)})")
    if rows:
        ok(not rows[0]["frames_kept"]
           and all(f["path"] is None for f in rows[0]["window"]["frames"]),
           "...but its four JPEGs are NOT kept — with both services stopped a "
           "drive would otherwise write 180 MB an hour of road photographs "
           "whose entire content is 'connection refused'")
        ok(rows[0]["readings"]["alpamayo1.5"]["error"],
           "and the error is on the row, which is the data worth having")
    state = panel.state(key)
    ok(state["enabled"] and "models" in state,
       "the panel still answers /teachers/state")
    ok(all(not state["models"][m]["reading"]["ok"] for m in panel.MODEL_NAMES),
       "both columns say they have nothing, rather than showing a stale one "
       "as if it were current")

    section("D2. one teacher down does not silence the other")
    panel.stop()
    panel.reset_all()
    up = FakeTeacher("cosmos-reason2", 18902)
    config.TEACHER_ALPAMAYO_URL = "http://127.0.0.1:1"
    config.TEACHER_COSMOS_URL = "http://127.0.0.1:18902"
    panel.start()
    key = "half"
    framebuf.drop_ring(key)
    ring, _ = push_window(key, n=8, spacing=0.1)
    for i in range(40):
        egomotion.note_speed(key, 12.0, "obd", at=time.time() - 3.0 + i * 0.1)
    panel.on_frame(key, RESULT, ring)
    deadline = time.time() + 10
    while time.time() < deadline:
        s = panel.state(key)
        if s["models"]["cosmos-reason2"]["reading"].get("ok"):
            break
        time.sleep(0.2)
    s = panel.state(key)
    ok(s["models"]["cosmos-reason2"]["reading"].get("ok"),
       "the live teacher answers")
    ok(not s["models"]["alpamayo1.5"]["reading"].get("ok"),
       "...while the dead one stays empty, in its own column")
    up.stop()
    panel.stop()

    section("D3. RIO's interpreter never loads a teacher's stack")
    heavy = [m for m in ("torch", "transformers", "alpamayo1_5", "accelerate")
             if m in sys.modules]
    ok(not heavy,
       f"importing the whole panel pulls in no model runtime ({heavy or 'none'})"
       " — the teachers live in their own environments and this one has not "
       "been made to carry them")


def run_flow_and_corpus():
    section("E. end to end, with both teachers answering")
    panel.reset_all()
    panel.stop()
    a = FakeTeacher("alpamayo1.5", 18901)
    c = FakeTeacher("cosmos-reason2", 18902)
    config.TEACHER_ALPAMAYO_URL = "http://127.0.0.1:18901"
    config.TEACHER_COSMOS_URL = "http://127.0.0.1:18902"
    config.TEACHER_CORPUS_DIR = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "training_data", "teachers")
    panel.start()

    key = "selftest-drive"
    framebuf.drop_ring(key)
    import shutil

    shutil.rmtree(corpus_mod.session_dir(key), ignore_errors=True)
    ring, _ = push_window(key, n=10, spacing=0.1)
    for i in range(40):
        egomotion.note_speed(key, 13.0, "obd", at=time.time() - 3.0 + i * 0.1)
        egomotion.ingest(key, [{"t": time.time() - 3.0 + i * 0.1,
                                "rr": [0, 0, 1.0], "g": [0, 0, -9.81]}])
    panel.note_spoken(key, "Ease off — that van is slowing.", "warning")
    ok(panel.on_frame(key, RESULT, ring), "a keyframe goes out")

    deadline = time.time() + 15
    while time.time() < deadline:
        if corpus_mod.read_rows(key):
            break
        time.sleep(0.2)
    rows = corpus_mod.read_rows(key)
    ok(len(rows) == 1, f"one complete corpus row ({len(rows)})")
    if not rows:
        panel.stop(); a.stop(); c.stop()
        return
    row = rows[0]
    bad = schema.validate_row(row)
    ok(not bad, "and it validates against the schema"
                + (f" — {bad[:4]}" if bad else ""))

    section("E2. what the row actually carries")
    ok(row["schema"] == schema.SCHEMA_VERSION, f"schema {row['schema']}")
    ok(row["trigger"] in schema.TRIGGERS, f"trigger {row['trigger']!r}")
    ok(row["state"]["band"] == "GETTING_UNSAFE"
       and row["state"]["gap_m"] == 21.4 and row["state"]["ttc_s"] == 4.1
       and row["state"]["speed_ms"] == 12.5
       and row["state"]["speed_source"] == "obd",
       "RIO's deterministic state at t0 — band, gap, TTC, speed and its source")
    ok(row["spoken"] and "Ease off" in row["spoken"]["text"],
       f"RIO's spoken line ({(row['spoken'] or {}).get('text')!r})")
    ok(row["ego"] and len(row["ego"]["xyz"]) == 16,
       "the ego history")
    ok(len(row["window"]["frames"]) == 4, "the four-frame window")
    kept = [f["path"] for f in row["window"]["frames"]]
    ok(all(kept), f"with the pictures on disk beside it ({kept[0]})")
    d = corpus_mod.session_dir(key)
    ok(all(os.path.exists(os.path.join(d, p)) for p in kept),
       "and the paths are relative and real")
    ok(len(row["tracks"]) == len(TRACKS),
       "the RF-DETR scene at t0, so an association can be re-checked without "
       "the pictures")
    for m in schema.MODELS:
        r = row["readings"][m]
        ok(r["ok"] and r["scene"] and r["reasoning"],
           f"{m}: a reading with a scene and a trace")
        ok(r["raw"], f"{m}: the model's own output verbatim, in `raw`")
        ok(r["latency_ms"] and r["freshness_s"] is not None and r["precision"],
           f"{m}: latency {r['latency_ms']}, freshness {r['freshness_s']}, "
           f"precision {r['precision']}")
        assoc = row["associations"][m]
        ok(assoc["matched"] and assoc["track_id"] == 7,
           f"{m}: the named actor is pinned to track 7 ({assoc['reason']})")

    section("E3. the trajectory is display-only, and reaches the display")
    traj = row["readings"]["alpamayo1.5"]["trajectory"]
    ok(traj and len(traj["xyz"]) == 64 and traj["hz"] == 10,
       "64 waypoints at 10 Hz — 6.4 seconds")
    ok(traj.get("pixels") and len(traj["pixels"]) > 4,
       f"projected onto the picture for the ribbon "
       f"({len((traj or {}).get('pixels') or [])} points)")
    ok(row["readings"]["cosmos-reason2"]["trajectory"] is None,
       "Cosmos predicts none, and the row says null rather than omitting it")

    section("E4. the tally strip counts what it claims to")
    t = panel.state(key)["tally"]
    ok(t["keyframes"] == 1 and t["readings"] == 2,
       f"one keyframe, two readings ({t['keyframes']}, {t['readings']})")
    ok(t["agree_eligible"] == 1 and t["agree"] == 1,
       "both named the same track, so that is one agreement out of one "
       "eligible")
    ok(t["matched_pct"] == 100, f"matched-track rate {t['matched_pct']}%")

    section("E5. the cadence")
    before = panel.state(key)["tally"]["keyframes"]
    fired = sum(1 for _ in range(20) if panel.on_frame(key, RESULT, ring))
    ok(fired == 0,
       f"twenty more frames in the same instant raise nothing ({fired}) — the "
       f"minimum gap is what stops a flapping band becoming a queue")
    r2 = dict(RESULT)
    r2["band"] = "UNSAFE"
    time.sleep(config.TEACHER_KEYFRAME_MIN_GAP_S + 0.05)
    ring2, _ = push_window(key, n=6, spacing=0.1)
    ok(panel.on_frame(key, r2, ring2), "a band change past the gap does raise one")
    ok(panel.state(key)["tally"]["keyframes"] == before + 1,
       "exactly one")

    panel.stop()
    a.stop()
    c.stop()
    corpus_mod.close(key)


def run_desync():
    section("G. two teachers at different speeds — the corpus must not thin out")
    # THE BUG THIS EXISTS FOR. Each teacher has its own queue and its own
    # eviction, so a fast one runs keyframes 10, 11, 12 while a slow one runs
    # 10 and then 13. The row used to be written when both models' LATEST
    # readings named the same keyframe -- true only while they stay in step,
    # which they stop doing within seconds. The corpus quietly thinned to
    # almost nothing a minute into any real drive.
    panel.reset_all()
    panel.stop()
    fast = FakeTeacher("alpamayo1.5", 18921, delay=0.02)
    slow = FakeTeacher("cosmos-reason2", 18922, delay=0.9)
    config.TEACHER_ALPAMAYO_URL = "http://127.0.0.1:18921"
    config.TEACHER_COSMOS_URL = "http://127.0.0.1:18922"
    floor = config.TEACHER_KEYFRAME_FLOOR_S
    gap = config.TEACHER_KEYFRAME_MIN_GAP_S
    config.TEACHER_KEYFRAME_FLOOR_S = 0.12
    config.TEACHER_KEYFRAME_MIN_GAP_S = 0.05
    panel.start()

    key = "desync"
    framebuf.drop_ring(key)
    import shutil

    shutil.rmtree(corpus_mod.session_dir(key), ignore_errors=True)
    for i in range(40):
        egomotion.note_speed(key, 12.0, "obd", at=time.time() - 3.0 + i * 0.1)

    raised = 0
    for _ in range(60):
        ring, _ = push_window(key, n=5, spacing=0.1)
        if panel.on_frame(key, RESULT, ring):
            raised += 1
        time.sleep(0.05)
    ok(raised >= 8, f"the drive raised {raised} keyframes while one teacher "
                    f"took 45x as long as the other")

    deadline = time.time() + 40
    while time.time() < deadline:
        if not panel.status()["services"]["cosmos-reason2"]["queued"] \
                and not panel.status()["services"]["cosmos-reason2"]["busy"]:
            break
        time.sleep(0.5)
    time.sleep(1.5)
    rows = corpus_mod.read_rows(key)
    svc = panel.status()["services"]
    evicted = svc["cosmos-reason2"]["evicted"] + svc["cosmos-reason2"]["stale_dropped"]
    ran = svc["cosmos-reason2"]["ok"]
    print(f"       raised {raised} · slow teacher ran {ran}, dropped {evicted} "
          f"· rows {len(rows)}")
    ok(len(rows) >= 2,
       f"rows are still written when the two go out of step ({len(rows)})")
    ok(len(rows) == raised,
       f"one row per keyframe, still ({len(rows)} rows, {raised} raised) — "
       f"under the old rule almost all of these were lost the moment the two "
       f"models stopped being on the same keyframe")
    both = [r for r in rows if len(r["models_ran"]) == 2]
    solo = [r for r in rows if len(r["models_ran"]) == 1]
    ok(len(both) == ran,
       f"{len(both)} rows are COMPARISONS — one per keyframe the slower "
       f"teacher actually ran ({ran})")
    ok(len(solo) == evicted,
       f"and {len(solo)} carry the faster teacher's reading alone, one per "
       f"keyframe the slower one dropped ({evicted}) — a real reading of a "
       f"real window, kept, and marked so an analysis can filter it out")
    for row in rows:
        bad = schema.validate_row(row)
        ok(not bad, f"row {row['seq']} validates" + (f" — {bad[:3]}" if bad else ""))
        for m in schema.MODELS:
            r = row["readings"][m]
            if m in row["models_ran"]:
                ok(r["ok"] and r["scene"],
                   f"row {row['seq']}: {m} ran and has a reading")
            else:
                ok(not r["ok"] and r["error"],
                   f"row {row['seq']}: {m} did not run, and the row says why "
                   f"({r['error']})")
    ok(len(rows) + 0 == raised and evicted > 0,
       f"every raised keyframe is accounted for: {len(rows)} rows, of which "
       f"{len(both)} comparisons, against {raised} raised and {evicted} "
       f"dropped by the slow teacher")

    section("G2. the extras the service reported survive to the corpus")
    if rows:
        extra = rows[0]["readings"]["alpamayo1.5"].get("extra") or {}
        ok("timings_ms" in extra or extra != {},
           f"a field the schema never named is kept rather than dropped "
           f"({sorted(extra)[:4]})")

    config.TEACHER_KEYFRAME_FLOOR_S = floor
    config.TEACHER_KEYFRAME_MIN_GAP_S = gap
    panel.stop()
    fast.stop()
    slow.stop()
    corpus_mod.close(key)


def run_canned():
    section("H. a recited answer is flagged, not filtered")
    from teachers import canned as canned_mod

    # THE REAL STRING, from the acceptance run. Alpamayo returned this verbatim
    # on eight of ten keyframes of a clip with no lane change, no stopped lead
    # and no distracted driver in it -- and the giveaway is in the characters:
    # non-breaking spaces, carried out of a label spreadsheet.
    RECITED = ("A vehicle controls\u00a0loss. The vehicle driver is in "
               "distracted driving.")
    REAL = ("The scene shows a clear day with good visibility. The road is a "
            "multi-lane highway with a concrete divider.")
    ok(canned_mod.looks_canned(RECITED),
       f"the non-breaking spaces in a memorised label are seen "
       f"({canned_mod.markers(RECITED)})")
    ok(not canned_mod.looks_canned(REAL),
       "a real description is not flagged")
    ok(not canned_mod.looks_canned(""), "and neither is nothing")

    t = canned_mod.RepeatTracker()
    counts = [t.note("m", "scene", REAL) for _ in range(4)]
    ok(counts == [1, 2, 3, 4],
       f"the same answer keyframe after keyframe is counted ({counts})")
    ok(not canned_mod.describe(REAL, 2),
       "twice is a coincidence on a motorway where nothing changes")
    ok(canned_mod.describe(REAL, 3).get("repeated") == 3,
       "three times is the model not looking, whatever the words are — which "
       "is the half of this that needs no list of known phrases")
    ok(t.note("m", "critical_actor", REAL) == 1,
       "counted per FIELD: Alpamayo's critical-actor answers were specific on "
       "the very keyframes where its scene answers were a fixed string")
    ok(t.note("m2", "scene", REAL) == 1, "...and per model")
    ok(canned_mod.describe(RECITED, 5).get("why"),
       "and a flag says in words why it fired, for whoever reads the corpus")
    ok(canned_mod.describe(RECITED, 5).get("strength") == "strong",
       "a non-breaking space between two words is near-proof — strong")
    LOOP = ("The ego vehicle is on the freeway. "
            + "The sedan is also further away from the ego vehicle. " * 5)
    ok(canned_mod.loop_run(LOOP) == 5,
       f"a sentence repeated inside ONE answer is counted "
       f"({canned_mod.loop_run(LOOP)}) — Cosmos filled its whole token budget "
       f"with one sentence on the acceptance clip")
    ok(canned_mod.describe(LOOP, 0).get("strength") == "strong",
       "a decoding loop is strong evidence: it is not analysis whatever it "
       "says, and a corpus row that looks like four hundred words of "
       "reasoning and is one sentence should say so")
    ok(canned_mod.loop_run(REAL) < canned_mod.LOOP_FLOOR,
       "a real answer that restates nothing is not a loop")
    ok(canned_mod.describe(REAL, 5).get("strength") == "weak",
       "repetition alone is a HINT — weak. On a straight empty motorway "
       "'keep lane, the lane is clear' three keyframes running is the model "
       "being right three times, and calling that recited would teach whoever "
       "reads the card to ignore the flag")

    section("H2. the shipped prompt set is the reworded one")
    ok("Describe the scene." not in config.TEACHER_PROMPTS.values(),
       "'Describe the scene.' is NOT asked — it returns a training-set label "
       "from Alpamayo on every sample, seeded or not")
    ok("weather" in config.TEACHER_PROMPTS["scene"],
       f"the scene question asks for road, traffic and weather instead "
       f"({config.TEACHER_PROMPTS['scene']!r})")
    ok("where is it" in config.TEACHER_PROMPTS["attention"],
       "and the attention question asks WHERE the hazard is, which gets a real "
       "answer from both models and gives the association something to match")
    ok(len(set(config.TEACHER_PROMPTS.values())) == 3,
       "three distinct questions, asked of both models word for word")

    section("H3. end to end, a recited reading reaches the record flagged")
    panel.reset_all()
    panel.stop()

    class Reciter(FakeTeacher):
        def answer(self, payload):
            out = super().answer(payload)
            out["scene"] = RECITED          # the same string every time
            return out

    a = Reciter("alpamayo1.5", 18931)
    c = FakeTeacher("cosmos-reason2", 18932)
    config.TEACHER_ALPAMAYO_URL = "http://127.0.0.1:18931"
    config.TEACHER_COSMOS_URL = "http://127.0.0.1:18932"
    panel.start()
    key = "canned"
    framebuf.drop_ring(key)
    import shutil

    shutil.rmtree(corpus_mod.session_dir(key), ignore_errors=True)
    for i in range(40):
        egomotion.note_speed(key, 12.0, "obd", at=time.time() - 3.0 + i * 0.1)
    floor, gap = config.TEACHER_KEYFRAME_FLOOR_S, config.TEACHER_KEYFRAME_MIN_GAP_S
    config.TEACHER_KEYFRAME_FLOOR_S = 0.15
    config.TEACHER_KEYFRAME_MIN_GAP_S = 0.1
    for _ in range(30):
        ring, _ = push_window(key, n=5, spacing=0.1)
        panel.on_frame(key, RESULT, ring)
        time.sleep(0.12)
    config.TEACHER_KEYFRAME_FLOOR_S, config.TEACHER_KEYFRAME_MIN_GAP_S = floor, gap
    deadline = time.time() + 25
    while time.time() < deadline and len(corpus_mod.read_rows(key)) < 4:
        time.sleep(0.5)
    rows = corpus_mod.read_rows(key)
    ok(len(rows) >= 3, f"the drive produced rows ({len(rows)})")
    flagged = [r for r in rows
               if (r["readings"]["alpamayo1.5"].get("flags") or {}).get("scene")]
    ok(len(flagged) >= 1,
       f"the reciting model's scene answers are flagged in the corpus "
       f"({len(flagged)} of {len(rows)})")
    if flagged:
        f = flagged[0]["readings"]["alpamayo1.5"]["flags"]["scene"]
        ok("markers" in f,
           f"with the structural evidence attached ({f.get('markers')})")
        ok(flagged[0]["readings"]["alpamayo1.5"]["scene"] == RECITED,
           "and the text is kept EXACTLY as the model said it — flagging is "
           "not filtering, and a recited answer is still evidence")
    clean = [r for r in rows
             if not (r["readings"]["cosmos-reason2"].get("flags") or {}).get("scene")]
    ok(len(clean) == len(rows),
       f"the other model's varying answers are not flagged ({len(clean)}/"
       f"{len(rows)})")
    panel.stop()
    a.stop()
    c.stop()
    corpus_mod.close(key)


def run_association():
    section("F. association — deterministic, and allowed to abstain")
    cases = [
        ("The white truck directly ahead, because it is braking hard.",
         True, 7, "a class the detector has, in the lane it is in"),
        ("A pedestrian on the left stepping off the kerb.",
         True, 8, "class and side together"),
        ("The traffic light ahead has turned amber.",
         False, None, "a class RF-DETR does not detect"),
        ("An oncoming van in the opposite lane.",
         False, None, "oncoming traffic, which nothing here tracks"),
        ("A cyclist filtering up the inside.",
         False, None, "a class with no track in this scene"),
        ("", False, None, "no answer at all"),
    ]
    for text, want_match, want_id, why in cases:
        a = assoc_mod.associate(text, TRACKS, 640, 480)
        ok(a["matched"] == want_match and a["track_id"] == want_id,
           f"{why}: matched={a['matched']} id={a['track_id']} "
           f"({a['reason']})")

    section("F2. the same input gives the same answer, always")
    runs = {json.dumps(assoc_mod.associate(
        "the truck ahead", TRACKS, 640, 480), sort_keys=True) for _ in range(20)}
    ok(len(runs) == 1, "twenty runs, one answer")
    shuffled = list(reversed(TRACKS))
    ok(assoc_mod.associate("the truck ahead", shuffled, 640, 480)["track_id"]
       == assoc_mod.associate("the truck ahead", TRACKS, 640, 480)["track_id"],
       "and the order the tracks arrive in does not change it")

    section("F3. two abstentions are not an agreement")
    miss = assoc_mod.associate("a traffic light", TRACKS, 640, 480)
    ok(not assoc_mod.agree(miss, miss),
       "counting them as one would make the tally read best when the "
       "association is working worst")
    hit = assoc_mod.associate("the truck ahead", TRACKS, 640, 480)
    ok(assoc_mod.agree(hit, hit), "two matches on the same track are")


def run_live(url_a, url_c):
    section("G. against the real services")
    import urllib.request

    key = "live-probe"
    framebuf.drop_ring(key)
    ring, _ = push_window(key, n=8, spacing=0.1)
    for i in range(40):
        egomotion.note_speed(key, 12.0, "obd", at=time.time() - 3.0 + i * 0.1)
    ego = egomotion.history(key, ring.latest().wall_t)
    kf = kf_mod.build(key, ring, RESULT, "manual", ego=ego, seq=0)
    if not kf.get("ok"):
        ok(False, f"could not build a keyframe: {kf.get('reason')}")
        return
    payload = client_mod.TeacherClient._payload(kf_mod.to_job(kf))
    for name, url in (("alpamayo1.5", url_a), ("cosmos-reason2", url_c)):
        try:
            req = urllib.request.Request(
                url.rstrip("/") + "/infer", data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"}, method="POST")
            with urllib.request.urlopen(req, timeout=300) as r:
                out = json.loads(r.read().decode())
        except Exception as e:
            ok(False, f"{name}: {type(e).__name__}: {e}")
            continue
        ok(out.get("ok"), f"{name} answered ({out.get('error')})")
        if not out.get("ok"):
            continue
        print(f"       scene    : {str(out.get('scene'))[:120]}")
        print(f"       actor    : {str(out.get('critical_actor'))[:120]}")
        print(f"       reasoning: {str(out.get('reasoning'))[:160]}")
        print(f"       latency  : {out.get('latency_ms')} ms  "
              f"vram {out.get('gpu', {}).get('vram_reserved_mb')} MB")
        for field in ("scene", "critical_actor", "attention", "reasoning"):
            ok(bool(str(out.get(field) or "").strip()),
               f"{name}: {field} came back non-empty")
        if name == "alpamayo1.5":
            t = out.get("trajectory") or {}
            ok(len(t.get("xyz") or []) == 64,
               f"{name}: 64 waypoints ({len(t.get('xyz') or [])})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="also POST one keyframe to the real services")
    ap.add_argument("--alpamayo", default="http://127.0.0.1:8801")
    ap.add_argument("--cosmos", default="http://127.0.0.1:8802")
    args = ap.parse_args()

    try:
        run_window()
        run_ego()
        run_stale()
        run_service_isolation()
        run_flow_and_corpus()
        run_desync()
        run_canned()
        run_association()
        if args.live:
            run_live(args.alpamayo, args.cosmos)
    finally:
        panel.stop()
        corpus_mod.close_all()

    print("\n" + "=" * 72)
    total = len(PASS) + len(FAIL)
    print(f"{len(PASS)}/{total} checks passed")
    if FAIL:
        print("\nFAILED:")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 72)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
