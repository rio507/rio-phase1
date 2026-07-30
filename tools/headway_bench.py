"""Drive a clip through the live /headway_frame endpoint and report latency.

    python -m tools.headway_bench --clip /tmp/synth.mp4 --v-host 18 --fps 2

Unlike headway/live_selftest.py this uses the REAL stack end to end: real HTTP,
real Depth Anything V2, real Qwen3-VL anchoring, the real session log. It is the
measurement the loop's cadence has to be justified against — if a frame costs
more than the 500 ms budget, 2 fps is not achievable and the confirmation
windows shift with it.

Prints a per-frame timeline plus a latency breakdown, separating the anchor
frames (which pay a Qwen generate) from the steady-state frames (which do not).
That split is the whole point: the design's claim is that Qwen is off the
per-frame path, and this is where that claim is either true or it is not.
"""
import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

import cv2
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def pct(values, p):
    if not values:
        return float("nan")
    s = sorted(values)
    k = min(len(s) - 1, int(round((p / 100.0) * (len(s) - 1))))
    return s[k]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8888")
    ap.add_argument("--v-host", type=float, default=18.0)
    ap.add_argument("--fps", type=float, default=2.0, help="sampling rate from the clip")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--no-session", action="store_true",
                    help="skip session start/end (no JSONL will be written)")
    ap.add_argument("--realtime", action="store_true",
                    help="pace requests to --fps instead of running flat out")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.clip}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    stride = max(1, int(round(src_fps / args.fps)))
    print(f"clip {args.clip}  src {src_fps:.0f} fps  sampling every {stride} "
          f"-> {src_fps / stride:.2f} fps  v_host {args.v_host} m/s")

    sid = None
    if not args.no_session:
        r = requests.post(f"{args.base}/session/start", json={"metadata": {
            "source": "headway_bench", "clip": os.path.basename(args.clip),
            "v_host": args.v_host}}, timeout=30)
        sid = r.json().get("session_id")
        print(f"session {sid}")

    qs = f"?session_id={sid}" if sid else ""
    url = f"{args.base}/headway_frame{qs}"

    rtt, server_total, depth_ms, anchor_ms, tail_ms = [], [], [], [], []
    lane_ms = []
    anchor_frames, steady_frames = [], []
    lane_src = {"ufld": 0, "static": 0}
    lane_fallback_why = {}
    lane_confs = []
    drifts = []
    rows, spoken = [], []
    idx, n = 0, 0
    t_run = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            if args.max_frames and n >= args.max_frames:
                break
            ok_enc, buf = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            files = {"image": ("frame.jpg", buf.tobytes(), "image/jpeg")}
            data = {"v_host": str(args.v_host), "v_host_age_s": "0",
                    "frame_t": str(idx / src_fps)}
            t0 = time.perf_counter()
            resp = requests.post(url, files=files, data=data, timeout=120)
            dt_ms = (time.perf_counter() - t0) * 1000
            j = resp.json()

            if j.get("ok"):
                tm = j.get("timing_ms", {})
                rtt.append(dt_ms)
                server_total.append(tm.get("total", 0.0))
                depth_ms.append(tm.get("depth", 0.0))
                anchor_ms.append(tm.get("anchor", 0.0))
                tail_ms.append(tm.get("track_filter", 0.0))
                lane_ms.append(tm.get("lanes", 0.0))

                src = j.get("corridor_source") or "static"
                lane_src[src] = lane_src.get(src, 0) + 1
                lane_confs.append(j.get("lane_conf") or 0.0)
                if src != "ufld":
                    why = (j.get("lane_info") or {}).get("fallback_reason", "?")
                    lane_fallback_why[why] = lane_fallback_why.get(why, 0) + 1
                if (j.get("lane_drift") or {}).get("drift"):
                    drifts.append((idx / src_fps, j["lane_drift"]))
                (anchor_frames if j.get("anchored") else steady_frames).append(
                    tm.get("total", 0.0))
                rows.append((idx / src_fps, j))
                if j.get("speak"):
                    spoken.append((idx / src_fps, j["speak"]))
            else:
                print(f"  frame {n}: NOT OK -> {j}")
            n += 1
            if args.realtime:
                time.sleep(max(0.0, (1.0 / args.fps) - (time.perf_counter() - t0)))
        idx += 1

    cap.release()
    elapsed = time.time() - t_run

    print(f"\n{'t':>6} {'tau':>7} {'d':>7} {'band':>15} {'trend':>8} {'urg':>4} "
          f"{'conf':>5} {'anch':>5} {'ms':>7}  voice")
    for t, j in rows:
        sp = j.get("speak")
        voice = f"{sp['line']} [{sp['audio']}]" if sp else (
            "" if j["voice_reason"] == "silent" else "- " + j["voice_reason"])
        print(f"{t:6.2f} "
              f"{(j['tau_s'] if j['tau_s'] is not None else float('nan')):7.2f} "
              f"{(j['distance_m'] if j['distance_m'] is not None else float('nan')):7.1f} "
              f"{j['band']:>15} {j['trend']:>8} {j['urgency']:>4} "
              f"{(j['confidence'] or 0):5.2f} {'Y' if j.get('anchored') else '.':>5} "
              f"{j['timing_ms']['total']:7.1f}  {voice}")

    if spoken:
        print("\nvoice timeline:")
        for t, sp in spoken:
            print(f"  {t:6.2f}s  {sp['line']:15} [{sp['audio']}] "
                  f"({sp['reason']})  {sp['text']!r}")

    def stats(label, vals):
        if not vals:
            print(f"  {label:22} --")
            return
        print(f"  {label:22} n={len(vals):3}  mean {statistics.mean(vals):7.1f}  "
              f"p50 {pct(vals, 50):7.1f}  p95 {pct(vals, 95):7.1f}  max {max(vals):7.1f}")

    print(f"\nlatency (ms), {n} frames in {elapsed:.1f}s")
    stats("client round-trip", rtt)
    stats("server total", server_total)
    stats("  lanes (UFLDv2)", lane_ms)
    stats("  depth (DA-V2)", depth_ms)
    stats("  anchor (Qwen)", anchor_ms)
    stats("  track+filter+policy", tail_ms)
    print()
    stats("ANCHOR frames", anchor_frames)
    stats("STEADY frames", steady_frames)
    if steady_frames:
        budget = 1000.0 / args.fps
        p95 = pct(steady_frames, 95)
        print(f"\n  cadence budget at {args.fps} fps: {budget:.0f} ms/frame")
        print(f"  steady-state p95:                {p95:.0f} ms  "
              f"({'OK — ' if p95 < budget else 'OVER — '}"
              f"{budget / max(p95, 1e-6):.1f}x headroom)")
        print(f"  max sustainable rate (steady):   {1000.0 / max(p95, 1e-6):.1f} fps")

    # --- lane geometry -----------------------------------------------------
    # The number that matters is the split, not the average: a corridor built
    # from paint and a corridor guessed from a trapezoid are different systems,
    # and "how often was each one running" is the claim this tool exists to
    # settle.
    total_lane = sum(lane_src.values())
    if total_lane:
        print(f"\nlane geometry, {total_lane} frames")
        for src in sorted(lane_src, key=lambda s: -lane_src[s]):
            n_src = lane_src[src]
            print(f"  corridor {src:8} {n_src:4} frames  "
                  f"{100.0 * n_src / total_lane:5.1f}%")
        if lane_fallback_why:
            print("  fallback reasons:")
            for why, c in sorted(lane_fallback_why.items(), key=lambda kv: -kv[1]):
                print(f"    {why:24} {c:4}")
        if lane_confs:
            print(f"  lane confidence        mean {statistics.mean(lane_confs):.3f}  "
                  f"p05 {pct(lane_confs, 5):.3f}  p50 {pct(lane_confs, 50):.3f}  "
                  f"p95 {pct(lane_confs, 95):.3f}")
        if lane_ms:
            print(f"  cost of lane detection mean {statistics.mean(lane_ms):.2f} ms  "
                  f"({statistics.mean(lane_ms) / max(statistics.mean(server_total), 1e-9) * 100:.0f}% "
                  f"of server frame time)")

    print(f"\nlane_drift events (advisory log only, no voice): {len(drifts)}")
    for t, d in drifts:
        print(f"  {t:6.2f}s  {d['side']:>5}  offset {d['offset']:+.2f}  "
              f"held {d['held_s']:.2f}s")

    if sid:
        requests.post(f"{args.base}/session/end?session_id={sid}", timeout=30)
        print(f"\nsession log: training_data/{sid}.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
