"""geom_clip.py — a clip whose geometry is known exactly.

The overlay bug this exists for is not a detection bug: the boxes were the
right SHAPE and the wrong PLACE. Nothing in the real pipeline can prove where a
box should have landed, because nothing in it knows where the pedestrians
actually were. So the pipeline is fed a road whose contents are stipulated.

Every frame carries the same set of solid MARKER squares at pixel positions
this module chose. A marker is pure magenta on a grey field: a colour that
survives JPEG, that no road scene contains, and that a screenshot can be
searched for without a model. That gives the harness two ground truths at once
-- where the marker is in the frame the server received, and where the marker
is on the screen the driver is looking at -- and the whole question is whether
the box drawn between them lands on both.

The markers do not move. A moving target would fold the timestamp-alignment
question back into the geometry question, and those are two different bugs with
two different harnesses (see tools/overlay_lag.py for the other one). A static
marker is wrong in exactly one way: geometrically.

VP9/WebM, not H.264: the Chromium that Playwright installs is the open-source
build and has no proprietary decoders. The codec is irrelevant to what is being
measured -- the browser hands the same RGB to drawImage either way.
"""
import os
import subprocess
import tempfile

import numpy as np

try:
    import cv2
except Exception:                                    # pragma: no cover
    cv2 = None

# Pure magenta. Chosen so a JPEG round trip and a screenshot's colour handling
# cannot slide it into any neighbour: the test matches on "red and blue high,
# green low", which no grey road surface and no anti-aliased cyan stroke does.
MARKER_BGR = (255, 0, 255)
BG_BGR = (64, 68, 72)


def visible_crop(w: int, h: int, box_aspect: float = 16.0 / 9.0):
    """The part of the frame the camera box actually shows.

    The box is 16:9 and the media is `object-fit: cover`, so a frame of any
    other shape is centre-cropped to 16:9 before the driver sees any of it. A
    marker outside that crop is not a test of anything -- it is off screen, and
    a harness that put one there would report a missing box as a placement
    failure. So the grid is laid out in CROP fractions and converted to source
    pixels here, which also means one grid definition works at every aspect
    ratio instead of one per shape.
    """
    if w / float(h) > box_aspect:            # wider than the box: sides cropped
        cw = h * box_aspect
        return ((w - cw) / 2.0, 0.0, cw, float(h))
    ch = w / box_aspect                      # taller: top and bottom cropped
    return (0.0, (h - ch) / 2.0, float(w), ch)


# Marker centres and sizes as fractions of the VISIBLE crop, which is also
# (under `cover`) fractions of the camera box on screen. The bands the page's
# own chrome occupies are avoided deliberately: the headway HUD covers the top
# ~17% of the left fifth, the source badge and the fullscreen button the top
# ~13% of the right quarter, and the stats strip the bottom ~5%. A marker under
# any of those is half-hidden by an opaque panel, its centroid moves, and the
# run reports a geometry failure that is really a furniture collision.
_GRID = [
    # (name, cx, cy, size)  -- all fractions of the crop's WIDTH except cy
    ("westedge", 0.030, 0.500, 0.045),   # hard left: `cover` crops here first
    ("eastedge", 0.968, 0.560, 0.045),   # hard right, and not at the same height
    ("topmid",   0.500, 0.090, 0.050),
    ("upleft",   0.300, 0.235, 0.062),
    ("upright",  0.700, 0.180, 0.055),
    ("centre",   0.440, 0.455, 0.080),
    ("lowleft",  0.205, 0.720, 0.065),
    ("lowright", 0.760, 0.795, 0.070),
]


def marker_grid(w: int, h: int):
    """The stipulated road: markers in SOURCE pixels.

    Deliberately asymmetric in both axes. A grid centred on the frame passes a
    harness that has the sign of an offset backwards, and one symmetric
    left-to-right passes a mapper that mirrors x -- both of which are exactly
    the class of bug being hunted.
    """
    ox, oy, cw, ch = visible_crop(w, h)
    out = []
    for name, fx, fy, fs in _GRID:
        s = max(8, int(round(fs * cw)))
        cx = int(round(ox + fx * cw))
        cy = int(round(oy + fy * ch))
        x1 = min(max(0, cx - s // 2), w - s)
        y1 = min(max(0, cy - s // 2), h - s)
        out.append({"name": name, "box": [x1, y1, x1 + s, y1 + s]})
    return out


# The moving grid. Three markers, well separated vertically, travelling left to
# right at a constant speed across the visible crop. Constant, and inside the
# frame for the whole clip: a marker that wraps at the edge splits into two
# blobs for one frame and both halves get a box, which is a measurement
# artefact indistinguishable from the fault being looked for.
_MOTION_GRID = [
    ("high", 0.26, 0.048),
    ("mid", 0.50, 0.058),
    ("low", 0.74, 0.052),
]
_MOTION_MARGIN = 0.03            # fraction of the crop kept clear at each end


def motion_grid(w: int, h: int, seconds: float):
    """Start boxes and the constant velocity that carries them across.

    Speed is derived from the clip rather than chosen: the markers start at one
    margin and finish at the other, so the whole clip is usable and nothing has
    to wrap. What the harness needs back is the velocity, because that is what
    turns a pixel offset on the screen into a MILLISECOND of misalignment --
    the number the alignment tolerance is actually written in.
    """
    ox, oy, cw, ch = visible_crop(w, h)
    margin = _MOTION_MARGIN * cw
    out = []
    travel = None
    for name, fy, fs in _MOTION_GRID:
        s = max(8, int(round(fs * cw)))
        x1 = int(round(ox + margin))
        y1 = int(round(oy + fy * ch - s / 2.0))
        y1 = min(max(0, y1), h - s)
        span = cw - 2 * margin - s
        travel = span if travel is None else min(travel, span)
        out.append({"name": name, "box": [x1, y1, x1 + s, y1 + s]})
    v = float(travel) / max(0.1, seconds)
    return out, v


def _frame(w: int, h: int, markers, i: int, dx: int = 0):
    img = np.zeros((h, w, 3), np.uint8)
    img[:, :] = BG_BGR
    # Faint texture so the encoder is not compressing a flat field into
    # something that no longer resembles a video frame's noise floor.
    img[::16, :] = (78, 82, 86)
    img[:, ::16] = (78, 82, 86)
    # A moving stripe, well away from every marker, purely so the decoder has
    # real inter-frame change and requestVideoFrameCallback fires per frame.
    x = int((i * 7) % max(1, w - 24))
    img[h - 12:h - 4, x:x + 24] = (200, 200, 200)
    for m in markers:
        x1, y1, x2, y2 = m["box"]
        x1, x2 = x1 + dx, x2 + dx
        if x1 < 0 or x2 > w:
            continue
        img[y1:y2, x1:x2] = MARKER_BGR
    return img


def render(path: str, w: int = 1280, h: int = 720, seconds: float = 8.0,
           fps: int = 24, motion: bool = False) -> dict:
    """Write the clip and return the ground truth that goes with it.

    `motion` swaps the static grid for the moving one. The static clip isolates
    geometry by removing time from the question; the moving clip puts time back
    in on purpose, because a box drawn on the right pixels of the WRONG FRAME is
    displaced exactly like a box drawn on the wrong pixels, and from the
    passenger seat the two are the same complaint.
    """
    if cv2 is None:
        raise RuntimeError("cv2 required to render the geometry clip")
    v_px_s = 0.0
    if motion:
        markers, v_px_s = motion_grid(w, h, seconds)
    else:
        markers = marker_grid(w, h)
    n = int(round(seconds * fps))
    tmp = tempfile.mkdtemp(prefix="geomclip-")
    for i in range(n):
        dx = int(round(v_px_s * i / float(fps)))
        cv2.imwrite(os.path.join(tmp, "f%05d.png" % i),
                    _frame(w, h, markers, i, dx))
    os.makedirs(os.path.dirname(os.path.abspath(path)) or ".", exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps),
           "-i", os.path.join(tmp, "f%05d.png"),
           "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "24",
           "-pix_fmt", "yuv420p", path]
    subprocess.run(cmd, check=True)
    for f in os.listdir(tmp):
        os.unlink(os.path.join(tmp, f))
    os.rmdir(tmp)
    return {"path": path, "w": w, "h": h, "fps": fps, "seconds": seconds,
            "markers": markers, "motion": bool(motion),
            "v_px_s": round(v_px_s, 3)}


def find_markers(bgr) -> list:
    """Locate the magenta squares in a frame. Used on BOTH ground truths.

    Run against the JPEG the server received it says what the capture path did
    to the picture; run against a screenshot of the camera box it says where
    the driver sees the marker. Same function, so a disagreement between the
    two is a disagreement about the pipeline and not about two detectors.
    """
    b = bgr[:, :, 0].astype(np.int16)
    g = bgr[:, :, 1].astype(np.int16)
    r = bgr[:, :, 2].astype(np.int16)
    mask = ((r > 110) & (b > 110) & (g < 100) &
            (r - g > 60) & (b - g > 60)).astype(np.uint8)
    n, lab, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if area < 24:
            continue
        out.append({"box": [int(x), int(y), int(x + w), int(y + h)],
                    "area": int(area),
                    "center": [float(cent[i][0]), float(cent[i][1])]})
    out.sort(key=lambda m: (m["box"][1], m["box"][0]))
    return out


if __name__ == "__main__":
    import argparse
    import json
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--w", type=int, default=1280)
    ap.add_argument("--h", type=int, default=720)
    ap.add_argument("--seconds", type=float, default=8.0)
    a = ap.parse_args()
    print(json.dumps(render(a.path, a.w, a.h, a.seconds), indent=2))
