"""Render the headway overlay onto clip frames, offline.

    python -m tools.lane_frame --clip drive.mp4 --at 4.5 --out /tmp/frame.jpg

The browser draws this overlay on a canvas over the <video>. That is the right
place for it live, and the wrong place to get a still out of: it needs a real
tab, a real upload and a screenshot tool. This draws the same thing from the
same /headway_frame response, so a frame can be pulled out of any clip for a
review, a bug report, or a slide.

It is a MIRROR, not a second implementation of anything: every number and every
polyline comes from the endpoint's JSON, exactly as the canvas gets it. Colours
and weights follow static/index.html so the two read as the same system. If the
overlay and this file ever disagree, this file is the one that is stale.
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# static/index.html's palette, BGR for cv2.
CYAN = (232, 179, 95)
LEAD = (255, 214, 155)
DIM = (150, 140, 130)
WHITE = (245, 245, 245)
BAND_COLOR = {
    "NORMAL": (232, 179, 95),
    "GETTING_UNSAFE": (77, 184, 255),
    "UNSAFE": (107, 107, 255),
    "SUPPRESSED": (150, 140, 130),
    "UNKNOWN": (150, 140, 130),
}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def label(img, text, org, color, scale=0.5, thick=1):
    """Outlined text — legible over a bright road scene, like drawLabel()."""
    cv2.putText(img, text, org, FONT, scale, (8, 3, 2), thick + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def dashed(img, pts, color, dash=14, gap=10, thick=1):
    """The corridor outline is dashed on the canvas; match it."""
    pts = np.asarray(pts, dtype=np.float32)
    for i in range(len(pts)):
        a, b = pts[i], pts[(i + 1) % len(pts)]
        seg = np.linalg.norm(b - a)
        if seg < 1e-6:
            continue
        n, drawn = (b - a) / seg, 0.0
        while drawn < seg:
            p0 = a + n * drawn
            p1 = a + n * min(seg, drawn + dash)
            cv2.line(img, tuple(p0.astype(int)), tuple(p1.astype(int)),
                     color, thick, cv2.LINE_AA)
            drawn += dash + gap


def brackets(img, x1, y1, x2, y2, color, thick=2):
    ln = int(max(8, min(26, min(x2 - x1, y2 - y1) * 0.28)))
    for (cx, cy, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1),
                             (x2, y2, -1, -1), (x1, y2, 1, -1)):
        cv2.line(img, (cx, cy), (cx + dx * ln, cy), color, thick, cv2.LINE_AA)
        cv2.line(img, (cx, cy), (cx, cy + dy * ln), color, thick, cv2.LINE_AA)


def render(frame, j):
    """Draw one /headway_frame response over its frame."""
    vis = frame.copy()
    h, w = vis.shape[:2]

    # Corridor: translucent fill + dashed outline.
    corridor = j.get("corridor") or []
    if len(corridor) >= 3:
        poly = np.asarray(corridor, dtype=np.int32)
        shade = vis.copy()
        cv2.fillPoly(shade, [poly], CYAN)
        cv2.addWeighted(shade, 0.07, vis, 0.93, 0, vis)
        dashed(vis, corridor, CYAN, thick=1)

    # Detected lane lines, thin cyan.
    for pts in (j.get("lanes") or []):
        if len(pts) >= 2:
            cv2.polylines(vis, [np.asarray(pts, dtype=np.int32)], False,
                          CYAN, 1, cv2.LINE_AA)

    # Lead box, coloured by band.
    band = j.get("band") or "UNKNOWN"
    color = BAND_COLOR.get(band, BAND_COLOR["UNKNOWN"])
    if j.get("lead_box"):
        x1, y1, x2, y2 = [int(round(v)) for v in j["lead_box"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color,
                      3 if band == "UNSAFE" else 2, cv2.LINE_AA)
        brackets(vis, x1, y1, x2, y2, color)
        bits = []
        bits.append(f"TAU {j['tau_s']:.1f}s" if j.get("tau_s") is not None else "TAU --")
        if j.get("trend"):
            bits.append(str(j["trend"]))
        if j.get("distance_m") is not None:
            bits.append(f"{j['distance_m']:.1f}m")
        ty = y2 + 20 if y1 - 8 < 16 else y1 - 8
        label(vis, "  ".join(bits), (max(4, x1), ty), color, 0.55, 1)

    # HUD strip, mirroring #headwayhud including the LANES indicator.
    src = (j.get("corridor_source") or "static").upper()
    conf = j.get("lane_conf")
    lane_txt = f"LANES: {src}" + ("" if conf is None else f"  {conf:.2f}")
    lines = [
        (band, color, 0.62),
        (f"TAU {j['tau_s']:.2f}s" if j.get("tau_s") is not None else "TAU --", WHITE, 0.5),
        (f"gap {j['distance_m']:.1f} m" if j.get("distance_m") is not None
         else "gap --", WHITE, 0.5),
        (f"conf {j['confidence']:.2f}" if j.get("confidence") is not None
         else "conf --", WHITE, 0.5),
        (lane_txt, CYAN if src == "UFLD" else DIM, 0.5),
    ]
    off = j.get("lane_offset")
    if off is not None:
        lines.append((f"lane offset {off:+.2f}", WHITE, 0.5))
    if (j.get("lane_drift") or {}).get("drift"):
        lines.append((f"LANE DRIFT {j['lane_drift']['side'].upper()} (logged)",
                      (77, 184, 255), 0.5))

    pad, lh = 10, 24
    box_h = pad * 2 + lh * len(lines)
    panel = vis[8:8 + box_h, 8:8 + 300].copy()
    vis[8:8 + box_h, 8:8 + 300] = cv2.addWeighted(
        np.full_like(panel, (8, 3, 2), dtype=np.uint8), 0.72, panel, 0.28, 0)
    cv2.rectangle(vis, (8, 8), (308, 8 + box_h), (60, 50, 45), 1)
    cv2.line(vis, (8, 8), (8, 8 + box_h), color, 2)
    for i, (text, c, s) in enumerate(lines):
        label(vis, text, (8 + pad, 8 + pad + lh * (i + 1) - 7), c, s, 1)
    return vis


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--clip", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8888")
    ap.add_argument("--at", type=float, action="append", required=True,
                    help="clip timestamp(s) in seconds to render")
    ap.add_argument("--v-host", type=float, default=18.0)
    ap.add_argument("--out", default="/tmp/headway_frame.jpg")
    ap.add_argument("--warmup", type=float, default=3.0,
                    help="seconds of clip fed in before --at, so the tracker "
                         "and Kalman are in the state a real drive would be in")
    ap.add_argument("--fps", type=float, default=4.0)
    ap.add_argument("--session", default="lane_frame")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.clip)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.clip}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    url = f"{args.base}/headway_frame?session_id={args.session}"

    shots = []
    for target in sorted(args.at):
        # A cold session has no tracker and no filter history, so a frame
        # grabbed straight out of one shows an empty overlay. Feeding the run-up
        # first makes the still representative of the drive at that moment.
        requests.post(f"{args.base}/headway_reset?session_id={args.session}",
                      timeout=60)
        start = max(0.0, target - args.warmup)
        t = start
        last = None
        while t <= target + 1e-6:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                break
            _, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            r = requests.post(
                url, files={"image": ("f.jpg", buf.tobytes(), "image/jpeg")},
                data={"v_host": str(args.v_host), "v_host_age_s": "0",
                      "frame_t": str(t)}, timeout=180)
            j = r.json()
            if j.get("ok"):
                last = (frame, j)
            t += 1.0 / args.fps

        if last is None:
            print(f"  {target:.2f}s: no usable response")
            continue
        frame, j = last
        vis = render(frame, j)
        out = args.out if len(args.at) == 1 else (
            f"{os.path.splitext(args.out)[0]}_{target:.2f}"
            f"{os.path.splitext(args.out)[1]}")
        cv2.imwrite(out, vis)
        shots.append(out)
        print(f"  {target:6.2f}s  {j.get('corridor_source'):>6}  "
              f"conf {j.get('lane_conf')}  band {j.get('band')}  -> {out}")

    cap.release()
    return 0 if shots else 1


if __name__ == "__main__":
    raise SystemExit(main())
