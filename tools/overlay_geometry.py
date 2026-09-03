"""overlay_geometry.py — does the box land on the thing it is a box for?

    python -m tools.overlay_geometry
    python -m tools.overlay_geometry --size 640x480 --size 1280x720
    python -m tools.overlay_geometry --keep --out runs/geom

THE BUG THIS ANSWERS
--------------------
Uploaded clips drew boxes of a plausible SIZE in an impossible PLACE: a
correctly proportioned "pedestrian" box 150 px left of the pedestrians, on a
bush. Shape right, position wrong, which is the signature of a coordinate frame
mismatch -- the frame the detector measured is not the frame the driver is
looking at -- and no amount of staring at detection output can find it, because
detection output is in the detector's frame and correct there.

So this harness stipulates the road. tools/geom_clip renders a clip whose
magenta markers are at pixel positions it chose. The clip goes through the REAL
browser path: a real upload into a real Chromium, the page's own capture, its
own POST, its own overlay canvas. Only the server is replaced, by an echo that
finds the markers in the JPEG IT ACTUALLY RECEIVED and hands those pixel boxes
straight back.

That substitution is what makes the measurement possible. A real detector
answers "where is the car"; the echo answers "where did this frame's marker
end up", so any difference between the box on the screen and the marker on the
screen is geometry and nothing else. Both are then read out of ONE screenshot
of the camera box -- not from the page's own numbers, which are the numbers
under suspicion.

WHAT IT ASSERTS
---------------
1. capture fidelity   the JPEG the server received is the clip's own size, and
                      the markers in it are within a couple of px of where the
                      clip put them. This is where a squashed or mis-sized
                      capture element is caught.
2. overlay placement  every drawn box edge is within TOL_PX of the marker's own
                      edge on screen. This is where the mapper is caught.

Both run at several aspect ratios on purpose. A 16:9 clip in a 16:9 box has no
letterbox and no crop, so it passes under a mapper that ignores object-fit
entirely -- which is why a passing 16:9 run alone proves nothing. 4:3 and
portrait crop hard, and the wedge marker sits in the part `cover` eats first.
"""
import argparse
import json
import os
import re
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tools import geom_clip                                    # noqa: E402

try:
    import cv2
except Exception:                                              # pragma: no cover
    cv2 = None

BASE = os.environ.get("RIO_HARNESS_URL", "http://127.0.0.1:8888")

# A few pixels, and the number is not arbitrary: the box stroke is 1 px wide,
# the screenshot is taken at devicePixelRatio 1, and the marker's own edge is
# an exact integer in source pixels that lands on a fractional display pixel
# after scaling. 3 px covers the rounding and nothing else -- the failure being
# hunted is 150.
TOL_PX = 3.0
# Edges get one more pixel than centres, and for a reason rather than to make a
# run go green. A centre is the midpoint of two strokes and the errors cancel;
# an edge is one stroke, drawn 1 px wide under a 6 px shadow, measured by the
# first pixel that differs from the layer-off shot -- so it reads consistently
# just outside the geometric edge. On a clip the display upscales (640x480 into
# a 928 px box) the marker's own edge is blurred by the same resampling. The
# residual is about a pixel and a half of stroke and blur; it is not an offset,
# and the fault being hunted is a hundred times larger either way.
TOL_EDGE_PX = 4.5
# Capture is exact or it is broken: the frame is drawn 1:1 off an element whose
# intrinsic size is known, so a marker that has moved at all has been resampled.
TOL_CAPTURE_PX = 2.0
# The moving clip gets a little more room, and only on the marker's WIDTH. A
# square sitting at a whole-pixel offset every frame still lands between chroma
# samples after 4:2:0 subsampling, and the JPEG the page encodes softens the
# edge again; the measured width of a 65 px square wanders by a pixel or two
# for reasons that have nothing to do with geometry. What this check is for --
# a capture that squashed or letterboxed the picture -- moves a width by tens
# of per cent, not by three.
TOL_CAPTURE_MOTION_PX = 4.0

# One frame at 24 fps. Not a local choice: config.HEADWAY_ALIGN_TOLERANCE_S,
# the overlay's own ALIGN_TOLERANCE_S and tools/overlay_lag.py all hold this
# number, and a harness with its own opinion about it would pass runs the
# others fail.
try:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import config as _config
    ALIGN_TOLERANCE_S = float(_config.HEADWAY_ALIGN_TOLERANCE_S)
except Exception:
    ALIGN_TOLERANCE_S = 1.0 / 24.0



# The lane cases, one per marker, cycled. Each is a lane drawn straight down a
# marker's centre column, and each carries a different verdict from the server
# so that one screenshot answers three questions at once: does a lane the
# pipeline stands behind get drawn, does one it has rejected on shape stay off
# the screen, and does one it barely asserts stay off too.
#
# The middle case is the one that matters. It is CONFIDENT and implausible --
# exactly the lane that used to reach the screen looking like paint -- so a
# gate built on confidence alone passes the other two and fails this one.
LANE_CASES = [
    {"score": 0.90, "plausible": True, "drawn": True,
     "why": "confident and lane-shaped"},
    {"score": 0.90, "plausible": False, "drawn": False,
     "why": "confident and NOT lane-shaped"},
    {"score": 0.10, "plausible": True, "drawn": False,
     "why": "barely asserted"},
]


def lane_case(i):
    return LANE_CASES[i % len(LANE_CASES)]


# --------------------------------------------------------------------------
# reading the screen
# --------------------------------------------------------------------------
def overlay_mask(rgb_on, rgb_off):
    """Exactly the pixels the overlay painted, and nothing else.

    Not a colour match on the finished screenshot: the camera box also contains
    the page's own chrome, and #camfull sits in the top-right corner in
    --accent-bright, which is the same blue family the boxes are stroked in. A
    scan looking for "cyan" found that button and reported a box 160 px wide
    around a marker that had none. So the overlay is isolated the only way that
    cannot confuse the two -- shot with the layer on, shot with it off through
    the page's own toggle, and the difference is the layer.
    """
    d = np.abs(rgb_on.astype(np.int16) - rgb_off.astype(np.int16)).max(axis=2)
    return d > 24


def lane_through(mask, cx, cy, tol=TOL_PX):
    """Is there an overlay stroke within `tol` of this marker's centre column?

    Not box_edges: that scans outward until it finds something, so with the
    lane absent it happily reports the NEXT marker's lane a hundred pixels away
    and the run reads as drawn-but-misplaced instead of correctly-suppressed.
    """
    h, w = mask.shape
    cy = int(round(cy))
    if not (0 <= cy < h):
        return None
    lo, hi = int(round(cx - tol)), int(round(cx + tol))
    lo, hi = max(0, lo), min(w - 1, hi)
    if hi < lo:
        return None
    hits = [x for x in range(lo, hi + 1) if mask[cy, x]]
    if not hits:
        return None
    return float(sum(hits)) / len(hits)


def box_edges(mask, cx, cy):
    """The four edges of the box around (cx, cy), found by scanning out from it.

    `mask` is the overlay-only mask from overlay_mask().

    Not a connected-component fit: the box's own label is drawn in the same
    colour a few px above its top edge and a blur can bridge the gap, which
    would drag a component's bbox up by the height of a line of text and
    manufacture a failure. Scanning outward from the marker's centre along its
    own centre row and column finds the first stroke in each direction and
    cannot see the label at all.

    Returns None if any direction has no stroke -- which is itself the answer
    when a box has been drawn 150 px away.
    """
    h, w = mask.shape
    cx, cy = int(round(cx)), int(round(cy))
    if not (0 <= cx < w and 0 <= cy < h):
        return None
    row, col = mask[cy, :], mask[:, cx]

    def first(seq, start, step, limit):
        i = start
        while 0 <= i < limit:
            if seq[i]:
                return i
            i += step
        return None

    x1 = first(row, cx, -1, w)
    x2 = first(row, cx, +1, w)
    y1 = first(col, cy, -1, h)
    y2 = first(col, cy, +1, h)
    if None in (x1, x2, y1, y2):
        return None
    return [float(x1), float(y1), float(x2), float(y2)]


# --------------------------------------------------------------------------
# the echo server, standing in for the detector
# --------------------------------------------------------------------------
def _jpeg_from_multipart(body: bytes):
    i = body.find(b"\xff\xd8\xff")
    if i < 0:
        return None
    j = body.rfind(b"\xff\xd9")
    return body[i:j + 2] if j > i else body[i:]


def _form_field(body: bytes, name: str):
    m = re.search(rb'name="' + name.encode() + rb'"\r\n\r\n(.*?)\r\n--', body, re.S)
    return m.group(1).decode("utf-8", "replace") if m else None


class Echo:
    """Finds the markers in the frame it was given and hands them back as boxes.

    Every frame is recorded, so the capture-fidelity half of the report is a
    measurement of the real POSTs rather than a second opinion about them.
    """

    def __init__(self, lanes=False, delay_ms=0):
        self.frames = []
        self.lanes = lanes
        self.delay_ms = float(delay_ms or 0)

    def handle(self, route):
        # A round trip the client has to schedule around. Free answers are not
        # a test of a presentation buffer: the whole mechanism only does
        # anything when the result lands after the frame it describes was
        # captured, so the harness has to cost something.
        if self.delay_ms:
            time.sleep(self.delay_ms / 1000.0)
        req = route.request
        body = req.post_data_buffer or b""
        jpg = _jpeg_from_multipart(body)
        if not jpg:
            route.fulfill(status=200, content_type="application/json",
                          body=json.dumps({"ok": False, "error": "no jpeg"}))
            return
        arr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        h, w = arr.shape[:2]
        found = geom_clip.find_markers(arr)
        self.frames.append({"w": w, "h": h, "bytes": len(jpg),
                            "markers": found,
                            "frame_t": _form_field(body, "frame_t")})
        objs = [{
            "box": m["box"], "label": "car", "range_m": None,
            "confirmed": True, "is_lead": False, "vulnerable": False,
        } for m in found]
        lanes, lane_scores, lane_plausible = [], [], []
        if self.lanes:
            # A vertical line straight down each marker's centre. If the lane
            # transform and the box transform ever disagree, the line misses
            # the square it was drawn through -- and the per-lane verdicts say
            # which of them should have been drawn at all.
            for i, m in enumerate(found):
                cx = (m["box"][0] + m["box"][2]) / 2.0
                lanes.append([[cx, 0], [cx, h - 1]])
                case = lane_case(i)
                lane_scores.append(case["score"])
                lane_plausible.append(case["plausible"])
        ft = _form_field(body, "frame_t")
        route.fulfill(status=200, content_type="application/json", body=json.dumps({
            "ok": True, "band": "NORMAL", "distance_m": None, "tau_s": None,
            "trend": None, "speak": None,
            "corridor": [], "lanes": lanes,
            "lane_scores": lane_scores, "lane_plausible": lane_plausible,
            "scene_objects": [] if self.lanes else objs,
            "lead_box": None, "image": {"w": w, "h": h},
            "frame_t": float(ft) if ft not in (None, "") else None,
        }))


# --------------------------------------------------------------------------
# one run
# --------------------------------------------------------------------------
PAGE_METRICS = """() => {
  const q = id => document.getElementById(id);
  const wrap = document.querySelector('.cam-wrap');
  const el = (v) => v ? {
    videoW: v.videoWidth || null, videoH: v.videoHeight || null,
    clientW: v.clientWidth, clientH: v.clientHeight,
    rect: (r => ({w: +r.width.toFixed(2), h: +r.height.toFixed(2),
                  x: +r.x.toFixed(2), y: +r.y.toFixed(2)}))(v.getBoundingClientRect()),
    objectFit: getComputedStyle(v).objectFit,
    currentTime: isFinite(v.currentTime) ? +v.currentTime.toFixed(3) : null,
    readyState: v.readyState, paused: v.paused,
  } : null;
  const c = q('overlay');
  return {
    dpr: window.devicePixelRatio,
    wrap: {clientW: wrap.clientWidth, clientH: wrap.clientHeight,
           rect: (r => ({w:+r.width.toFixed(2), h:+r.height.toFixed(2),
                         x:+r.x.toFixed(2), y:+r.y.toFixed(2)}))(wrap.getBoundingClientRect())},
    preview: el(q('preview')),
    scout: el(q('scout')),
    video: el(q('video')),
    canvas: {attrW: c.width, attrH: c.height,
             cssW: c.clientWidth, cssH: c.clientHeight,
             rect: (r => ({w:+r.width.toFixed(2), h:+r.height.toFixed(2),
                           x:+r.x.toFixed(2), y:+r.y.toFixed(2)}))(c.getBoundingClientRect())},
    source: window.RIO && window.RIO.source ? window.RIO.source.kind() : null,
  };
}"""


def run_one(pw, clip, truth, outdir, lanes=False, keep=False, headed=False,
            fullscreen=False, viewport=(1000, 900), delay_ms=0, dpr=1.0):
    from playwright.sync_api import Error as PWError            # noqa: F401
    browser = pw.chromium.launch(headless=not headed, args=[
        "--autoplay-policy=no-user-gesture-required",
        "--disable-features=PreloadMediaEngagementData,MediaEngagementBypassAutoplayPolicies",
    ])
    # A narrow viewport on purpose: the page's wide layout puts the camera box
    # in a 284 px column, and a 96 px marker lands on 21 px of screen there --
    # too few pixels for a 3 px tolerance to mean anything. The single-column
    # layout gives the box ~930 px, which is also the shape a driver's phone
    # actually renders.
    ctx = browser.new_context(viewport={"width": viewport[0], "height": viewport[1]},
                              device_scale_factor=dpr)
    page = ctx.new_page()
    echo = Echo(lanes=lanes, delay_ms=delay_ms)
    logs = []
    page.on("console", lambda m: logs.append(f"{m.type}: {m.text}"))
    page.route("**/headway_frame*", echo.handle)
    # /perceive is a second box layer on the same canvas from a different
    # endpoint. Silenced rather than echoed: two layers would put two boxes on
    # one marker and the scan would measure whichever was outermost.
    page.route("**/perceive*", lambda r: r.fulfill(
        status=200, content_type="application/json",
        body=json.dumps({"observation": "", "qwen_boxes": [], "corridor": [],
                         "lanes": [], "image": {"w": 1, "h": 1}})))

    page.goto(BASE + "/", wait_until="domcontentloaded")
    page.wait_for_timeout(600)
    page.set_input_files("#upload", clip)
    page.wait_for_function(
        "() => { const p = document.getElementById('preview');"
        " return p && p.videoWidth > 0; }", timeout=15000)
    before = page.evaluate(PAGE_METRICS)

    # A speed override, because the replay path warns without one and a warning
    # in the console during a geometry run is noise the next reader has to
    # re-diagnose.
    try:
        page.fill("#speedval", "30")
        page.dispatch_event("#speedval", "input")
    except Exception:
        pass

    # Playing a clip starts the detection loop on its own now, and #preview
    # carries `autoplay`, so by the time the upload has a videoWidth the run is
    # usually already up. Clicking Run there would TOGGLE IT OFF -- so the
    # button is only used when the automatic start did not happen (the toggle
    # switched off, a browser that refused autoplay).
    started = page.evaluate(
        "() => !!(window.RIO && RIO.headway && RIO.headway.videoRunning)")
    if not started:
        try:
            page.wait_for_function(
                "() => !!(window.RIO && RIO.headway && RIO.headway.videoRunning)",
                timeout=3000)
            started = True
        except Exception:
            started = False
    if not started:
        page.click("#runheadway")
    if fullscreen:
        # The other geometry: fullscreen drops the 16:9 lock and switches the
        # media to object-fit: contain, so the frame letterboxes instead of
        # cropping. Different offsets, different sign, same mapper.
        #
        # AFTER the run has started, and through the element's own handler: in
        # fullscreen the camera box covers the viewport, so every other control
        # on the page -- Run, and the layer toggle -- is behind it and cannot be
        # clicked in the ordinary way.
        page.evaluate("() => document.getElementById('camfull').click()")
        page.wait_for_timeout(600)
    # Long enough for the picture to pass the replay lead and for results to be
    # arriving steadily -- a screenshot taken during the lead-in would find no
    # box at all and report it as a placement failure.
    page.wait_for_function(
        "() => { const p = document.getElementById('preview');"
        " return p && p.currentTime > 1.2; }", timeout=20000)
    page.wait_for_timeout(400)
    after = page.evaluate(PAGE_METRICS)

    # Freeze the picture before reading it. A screenshot of a playing video and
    # a canvas painted for a different frame is two moments in one image; the
    # markers do not move, but the box's due-time scheduling does, and a paused
    # clip removes that variable entirely.
    page.evaluate("() => document.getElementById('preview').pause()")
    page.wait_for_timeout(500)
    page.evaluate("() => { if (window.RIO && RIO.overlay && RIO.overlay.redraw)"
                  " RIO.overlay.redraw(); }")
    page.wait_for_timeout(120)

    tag = "%dx%d%s%s%s" % (truth["w"], truth["h"], "_lanes" if lanes else "",
                           "_full" if fullscreen else "",
                           "_motion" if truth.get("motion") else "")
    # Clipped to the CANVAS, not to .cam-wrap. The canvas is the coordinate
    # space the mapper draws in, so a pixel in this image is a pixel the mapper
    # named; cropping to the wrap instead drags the box's 1 px border into the
    # frame and puts everything one pixel out for no reason worth explaining
    # twice.
    # Page coordinates, not viewport coordinates: Playwright's clip is
    # document-relative and getBoundingClientRect is not, so on a page that has
    # scrolled -- which this one does, the moment the file input is used -- the
    # two differ by the scroll offset and the screenshot comes back showing the
    # site header instead of the camera box.
    # Element screenshots of the WRAP, cropped to the canvas afterwards.
    # page.screenshot(clip=...) was tried first and is not worth the argument:
    # its rectangle and getBoundingClientRect do not agree about scrolling, the
    # two shots came back framing different parts of the page, and differencing
    # them produced a mask that said every marker was covered in overlay and
    # every box was zero pixels wide. A locator screenshot scrolls itself and
    # frames the same element both times by construction.
    shot_on = os.path.join(outdir, "screen_%s.png" % tag)
    page.locator(".cam-wrap").screenshot(path=shot_on)
    # The same frame with the layer off, through the page's own toggle. The
    # difference between the two is the overlay and only the overlay.
    page.evaluate("() => document.getElementById('layers').click()")
    page.wait_for_timeout(150)
    shot_off = os.path.join(outdir, "screen_%s_off.png" % tag)
    page.locator(".cam-wrap").screenshot(path=shot_off)
    page.evaluate("() => document.getElementById('layers').click()")
    # Where the canvas sits inside that screenshot, and how big it is. The
    # canvas is the mapper's own coordinate space, so cropping to it makes a
    # pixel in the image a pixel the mapper named -- no border, no rounding,
    # nothing to reason about later.
    canvas_in_shot = page.evaluate("""() => {
      const w = document.querySelector('.cam-wrap').getBoundingClientRect();
      const c = document.getElementById('overlay').getBoundingClientRect();
      return [c.x - w.x, c.y - w.y, c.width, c.height];
    }""")
    # Screenshots come back in DEVICE pixels. The mapper works in CSS pixels,
    # so everything measured off the image is divided back down -- a phone at
    # dpr 2 is a different canvas backing store and the same CSS geometry, and
    # conflating the two would report every box at twice its offset.
    canvas_in_shot = [v * dpr for v in canvas_in_shot]
    # Where the page's own furniture sits, in canvas pixels. The HUD panel, the
    # source badge, the fullscreen button and the stats strip are opaque and
    # sit ABOVE the media: a marker underneath one is half-hidden, its measured
    # centre moves, and that is a collision with the furniture rather than a
    # geometry fault. The grid is laid out to miss them; this is what proves it
    # did, at whatever size the layout actually came out.
    chrome = page.evaluate("""() => {
      const wrap = document.querySelector('.cam-wrap');
      const base = document.getElementById('overlay').getBoundingClientRect();
      const media = new Set(['video', 'preview', 'campreview', 'scout', 'overlay']);
      const out = [];
      wrap.querySelectorAll('*').forEach(e => {
        if (media.has(e.id)) return;
        const cs = getComputedStyle(e);
        if (cs.display === 'none' || cs.visibility === 'hidden') return;
        if (+cs.opacity === 0) return;
        const r = e.getBoundingClientRect();
        if (r.width < 2 || r.height < 2) return;
        out.push({id: e.id || e.className.toString(),
                  box: [r.x - base.x, r.y - base.y,
                        r.right - base.x, r.bottom - base.y]});
      });
      return out;
    }""")
    frozen = page.evaluate(PAGE_METRICS)
    # The page's OWN alignment measurement, over every result of the run rather
    # than over the one frame the screenshot froze: source frame timestamp
    # against the video time the box was actually painted at. The screenshot
    # offset is then a physical cross-check on this number rather than the only
    # evidence -- two ways of asking the same question, one in milliseconds and
    # one in pixels.
    lag = page.evaluate("() => (window.RIO && RIO.overlay && RIO.overlay.lagStats)"
                        " ? RIO.overlay.lagStats() : null")
    # Stop through the button, not by poking videoRunning -- it is a getter
    # and assigning to it fails silently, leaving the loop posting frames at a
    # browser that is about to close.
    try:
        page.evaluate("() => document.getElementById('runheadway').click()")
    except Exception:
        pass
    if not keep:
        ctx.close()
        browser.close()
    return {"before": before, "after": after, "frozen": frozen,
            "shot": shot_on, "shot_off": shot_off, "chrome": chrome,
            "canvas_in_shot": canvas_in_shot, "lag": lag,
            "echo": echo, "logs": logs}


def reference_map(metrics):
    """Source pixels -> camera-box pixels, derived from the page's MEASURED
    layout rather than from the page's own mapper.

    It is the same arithmetic the overlay does, which on its own would prove
    nothing -- a harness that reimplements the bug agrees with it. What makes
    it a reference is that it is checked against the screenshot first: the
    marker's centre on screen has to land where this says it does before any
    box is judged against it. If that check fails, the reference is wrong and
    the run says so instead of blaming the overlay.
    """
    pv = metrics["preview"]
    nw, nh = pv["videoW"], pv["videoH"]
    cw, ch = metrics["wrap"]["clientW"], metrics["wrap"]["clientH"]
    if not (nw and nh and cw and ch):
        return None
    fit = (pv["objectFit"] or "fill").strip()
    rx, ry = cw / float(nw), ch / float(nh)
    if fit == "contain":
        sx = sy = min(rx, ry)
    elif fit == "none":
        sx = sy = 1.0
    elif fit == "scale-down":
        sx = sy = min(1.0, min(rx, ry))
    elif fit == "fill":
        sx, sy = rx, ry
    else:
        sx = sy = max(rx, ry)
    dx, dy = (cw - nw * sx) / 2.0, (ch - nh * sy) / 2.0
    return lambda x, y: (dx + x * sx, dy + y * sy)


def analysis_interval_s(echo):
    """Median gap between the SOURCE frames that were actually analysed.

    The overlay draws the newest result whose frame the picture has reached, so
    between two analysed frames the box is legitimately up to one analysis
    interval old. That interval is the round trip, not the clip's frame rate,
    and it is measured here rather than assumed because assuming it is how a
    harness ends up excusing a real fault or failing a correct one.
    """
    ts = sorted(float(f["frame_t"]) for f in echo.frames
                if f.get("frame_t") not in (None, ""))
    if len(ts) < 3:
        return None
    gaps = [b - a for a, b in zip(ts, ts[1:]) if 0 < b - a < 2.0]
    if not gaps:
        return None
    gaps.sort()
    return gaps[len(gaps) // 2]


def align_error(rows, truth, metrics, interval_s=None):
    """Pixel offsets on a moving clip, restated as milliseconds of misalignment.

    On the static clip a displaced box is a geometry fault. On the moving clip
    it is a CLOCK fault, and pixels are the wrong unit to judge it in: the same
    30 px is a quarter of a second on a slow marker and a frame and a half on a
    fast one. Dividing by the marker's own speed on screen gives the number the
    tolerance is written in -- config.HEADWAY_ALIGN_TOLERANCE_S, one frame at
    24 fps -- and the same number tools/overlay_lag.py reports from simulation.

    A residual of up to one clip frame is structural and not a fault: the
    result describes the newest frame whose time has passed, and the picture
    has moved on by up to one frame interval since. So the budget is the
    tolerance plus one frame, and what it is really asserting is that the box
    is on the frame it was computed from rather than on one a round trip later.
    """
    ref = reference_map(metrics)
    if ref is None or not truth.get("v_px_s"):
        return None
    x0, _ = ref(0.0, 0.0)
    x1, _ = ref(1000.0, 0.0)
    v_screen = truth["v_px_s"] * (x1 - x0) / 1000.0
    if v_screen <= 0:
        return None
    step_s = interval_s or (1.0 / float(truth.get("fps") or 24))
    budget_s = ALIGN_TOLERANCE_S + step_s
    out = []
    for r in rows:
        if r.get("drawn") is None:
            out.append({"marker": r["marker_center"], "err_ms": None,
                        "ok": False, "why": r.get("why")})
            continue
        dx = (r["drawn"][0] + r["drawn"][2]) / 2.0 - r["marker_center"][0]
        err_s = abs(dx) / v_screen
        out.append({"marker": r["marker_center"],
                    "dx_px": round(dx, 2),
                    "err_ms": round(err_s * 1000, 1),
                    "ok": err_s <= budget_s})
    return {"v_screen_px_s": round(v_screen, 1),
            "analysis_interval_ms": None if interval_s is None
                                    else round(interval_s * 1000, 1),
            "budget_ms": round(budget_s * 1000, 1),
            "rows": out, "ok": all(r["ok"] for r in out) if out else False}


def measure(shot_on, shot_off, truth, metrics, chrome=(), crop=None,
            lanes=False, motion=False, dpr=1.0):
    """Read one screenshot pair: where the markers are, where the boxes are.

    The primary number is the DIRECT one -- the drawn box's centre against the
    marker's own centre in the same image -- because it assumes nothing at all
    about how either got there. Centres rather than edges: the marker is a
    solid square scaled down by the display, so its edge pixels are blended
    with the background and its measured extent erodes by a pixel or so, while
    the box's 1 px stroke stays crisp. The edge comparison is kept as well, but
    against the validated reference rather than against the eroded square.
    """
    on = cv2.imread(shot_on, cv2.IMREAD_COLOR)
    off = cv2.imread(shot_off, cv2.IMREAD_COLOR)
    if on is None or off is None or on.shape != off.shape:
        return {"n_markers": 0, "rows": [], "occluded": [],
                "why": "screenshot pair unusable"}
    if crop:
        x, y, w, h = (int(round(v)) for v in crop)
        on, off = on[y:y + h, x:x + w], off[y:y + h, x:x + w]
    if dpr and dpr != 1:
        on = cv2.resize(on, (int(round(on.shape[1] / dpr)),
                             int(round(on.shape[0] / dpr))),
                        interpolation=cv2.INTER_AREA)
        off = cv2.resize(off, (int(round(off.shape[1] / dpr)),
                               int(round(off.shape[0] / dpr))),
                         interpolation=cv2.INTER_AREA)
    mask = overlay_mask(on[:, :, ::-1], off[:, :, ::-1])
    screen = geom_clip.find_markers(off)          # the clean picture, no layer
    ref = reference_map(metrics)

    # Marker -> the ground-truth square it came from, so the reference can be
    # evaluated at the exact source coordinates the echo sent back.
    def truth_for(m):
        if ref is None:
            return None
        best, bestd = None, 1e9
        for t in truth["markers"]:
            tx = (t["box"][0] + t["box"][2]) / 2.0
            ty = (t["box"][1] + t["box"][3]) / 2.0
            px, py = ref(tx, ty)
            d = abs(px - m["center"][0]) + abs(py - m["center"][1])
            if d < bestd:
                best, bestd = t, d
        return best

    def occluded(box):
        for c in chrome:
            b = c["box"]
            if (box[0] < b[2] and box[2] > b[0]
                    and box[1] < b[3] and box[3] > b[1]):
                return c["id"]
        return None

    rows = []
    skipped = []
    for i, m in enumerate(screen):
        cx, cy = m["center"]
        under = occluded(m["box"])
        if under:
            skipped.append({"marker_screen": m["box"], "under": under})
            continue
        t = None if motion else truth_for(m)
        exp = None
        ref_center_err = None
        if t is not None:
            x1, y1 = ref(t["box"][0], t["box"][1])
            x2, y2 = ref(t["box"][2], t["box"][3])
            exp = [x1, y1, x2, y2]
            ref_center_err = round(max(abs((x1 + x2) / 2 - cx),
                                       abs((y1 + y2) / 2 - cy)), 2)
        if lanes:
            # The echo built lane i from marker i, and both sides sort the
            # markers the same way, so the index is the join.
            case = lane_case(i)
            hit = lane_through(mask, cx, cy)
            rows.append({"marker_name": t["name"] if t else None,
                         "marker_center": [round(v, 1) for v in m["center"]],
                         "case": case["why"],
                         "expected": "drawn" if case["drawn"] else "suppressed",
                         "observed": "drawn" if hit is not None else "suppressed",
                         "drawn": None if hit is None else [round(hit, 1)],
                         "center_err_px": (None if hit is None
                                           else round(abs(hit - cx), 2)),
                         "edge_err_px": None,
                         "ok": ((hit is not None) == case["drawn"]
                                and (hit is None or abs(hit - cx) <= TOL_PX))})
            continue

        drawn = box_edges(mask, cx, cy)
        row = {"marker_name": t["name"] if t else None,
               "marker_screen": [round(v, 1) for v in m["box"]],
               "marker_center": [round(v, 1) for v in m["center"]],
               "expected_screen": [round(v, 1) for v in exp] if exp else None,
               "reference_center_err_px": ref_center_err}
        if drawn is None:
            row.update({"drawn": None, "center_err_px": None, "edge_err_px": None,
                        "ok": False, "why": "no overlay stroke around marker"})
            rows.append(row)
            continue
        cerr = max(abs((drawn[0] + drawn[2]) / 2 - cx),
                   abs((drawn[1] + drawn[3]) / 2 - cy))
        # On the moving clip the marker's own source position changes frame to
        # frame, so there is no fixed truth rect to compare edges against and
        # the reference check does not apply -- align_error() judges that run.
        eerr = (None if motion or not exp
                else max(abs(drawn[i] - exp[i]) for i in range(4)))
        ok = (motion
              or (cerr <= TOL_PX
                  and (eerr is None or eerr <= TOL_EDGE_PX)
                  and (ref_center_err is None or ref_center_err <= TOL_PX)))
        row.update({"drawn": [round(v, 1) for v in drawn],
                    "center_err_px": round(cerr, 2),
                    "edge_err_px": None if eerr is None else round(eerr, 2),
                    "ok": bool(ok)})
        rows.append(row)
    return {"n_markers": len(screen), "rows": rows, "occluded": skipped,
            "reference": None if ref is None else "from measured layout"}


def capture_report(echo, truth):
    """What the server actually received, against what the clip actually is.

    On the moving clip a marker's source position is a function of the frame,
    so position is not the invariant -- SIZE is. A capture that squashed or
    letterboxed the picture changes a 61 px square into something else no
    matter which frame it came from, so that is what is checked there.
    """
    if not echo.frames:
        return {"ok": False, "why": "no frames posted", "n": 0,
                "worst_marker_err_px": None, "sizes": [],
                "expected": {"w": truth["w"], "h": truth["h"]}}
    sizes = sorted({(f["w"], f["h"]) for f in echo.frames})
    if truth.get("motion"):
        want = sorted(m["box"][2] - m["box"][0] for m in truth["markers"])
        worst, n_wrong = 0.0, 0
        for f in echo.frames:
            got = sorted(m["box"][2] - m["box"][0] for m in f["markers"])
            if len(got) != len(want):
                n_wrong += 1
                continue
            worst = max(worst, max(abs(a - b) for a, b in zip(got, want)))
        return {
            "n": len(echo.frames),
            "sizes": [{"w": w, "h": h} for w, h in sizes],
            "expected": {"w": truth["w"], "h": truth["h"]},
            "size_ok": sizes == [(truth["w"], truth["h"])],
            "measured": "marker width (the clip is moving)",
            "markers_expected": len(truth["markers"]),
            "frames_with_wrong_marker_count": n_wrong,
            "worst_marker_err_px": round(worst, 2),
            "ok": (sizes == [(truth["w"], truth["h"])] and n_wrong == 0
                   and worst <= TOL_CAPTURE_MOTION_PX),
        }
    truth_boxes = {tuple(m["box"]) for m in truth["markers"]}
    worst, worst_frame = 0.0, None
    n_wrong_count = 0
    for f in echo.frames:
        if len(f["markers"]) != len(truth["markers"]):
            n_wrong_count += 1
            continue
        for got in f["markers"]:
            best = min(max(abs(got["box"][i] - t[i]) for i in range(4))
                       for t in truth_boxes)
            if best > worst:
                worst, worst_frame = best, f
    return {
        "n": len(echo.frames),
        "sizes": [{"w": w, "h": h} for w, h in sizes],
        "expected": {"w": truth["w"], "h": truth["h"]},
        "size_ok": sizes == [(truth["w"], truth["h"])],
        "markers_expected": len(truth["markers"]),
        "frames_with_wrong_marker_count": n_wrong_count,
        "worst_marker_err_px": round(worst, 2),
        "ok": (sizes == [(truth["w"], truth["h"])] and n_wrong_count == 0
               and worst <= TOL_CAPTURE_PX),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", action="append", default=None,
                    help="WxH, repeatable. Default: 1280x720, 640x480, 720x1280")
    ap.add_argument("--out", default="runs/geom")
    ap.add_argument("--seconds", type=float, default=8.0)
    ap.add_argument("--lanes", action="store_true",
                    help="also assert the lane polyline transform")
    ap.add_argument("--keep", action="store_true", help="leave the browser open")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--fullscreen", action="store_true",
                    help="also run the fullscreen/object-fit:contain geometry")
    ap.add_argument("--motion", action="store_true",
                    help="moving markers: judges timestamp alignment, in ms")
    ap.add_argument("--echo-delay-ms", type=float, default=60.0,
                    help="latency the stand-in server answers with")
    ap.add_argument("--dpr", type=float, default=1.0,
                    help="device pixel ratio, as a phone would render it")
    ap.add_argument("--vw", type=int, default=1000)
    ap.add_argument("--vh", type=int, default=900)
    a = ap.parse_args()
    if cv2 is None:
        print("cv2 is required", file=sys.stderr)
        return 2
    sizes = a.size or ["1280x720", "640x480", "720x1280"]
    os.makedirs(a.out, exist_ok=True)

    from playwright.sync_api import sync_playwright
    report, failed = [], False
    with sync_playwright() as pw:
        for s in sizes:
            w, h = (int(v) for v in s.lower().split("x"))
            clip = os.path.join(a.out, "geom_%dx%d%s.webm"
                                % (w, h, "_motion" if a.motion else ""))
            truth = geom_clip.render(clip, w, h, a.seconds, motion=a.motion)
            modes = [False, True] if a.lanes else [False]
            fits = [False, True] if a.fullscreen else [False]
            for full in fits:
              for lanes in modes:
                r = run_one(pw, clip, truth, a.out, lanes=lanes,
                            keep=a.keep, headed=a.headed,
                            fullscreen=full, viewport=(a.vw, a.vh),
                            delay_ms=a.echo_delay_ms, dpr=a.dpr)
                cap = capture_report(r["echo"], truth)
                place = measure(r["shot"], r["shot_off"], truth, r["frozen"],
                                chrome=r["chrome"], crop=r["canvas_in_shot"],
                                lanes=lanes, motion=a.motion, dpr=a.dpr)
                interval = analysis_interval_s(r["echo"])
                align = (align_error(place["rows"], truth, r["frozen"],
                                     interval_s=interval)
                         if a.motion else None)
                lag = (r["lag"] or {}).get("replay")
                ok = (cap["ok"] and place["rows"]
                      and place["n_markers"] == len(truth["markers"])
                      and not place["occluded"]
                      and all(row["ok"] for row in place["rows"])
                      and (align is None or align["ok"]))
                failed = failed or not ok
                report.append({
                    "clip": "%dx%d" % (w, h), "layer": "lanes" if lanes else "boxes",
                    "fit": "contain (fullscreen)" if full else "cover",
                    "ok": ok, "capture": cap, "placement": place,
                    "alignment": align, "lag_replay": lag,
                    "metrics": r["frozen"], "metrics_before_run": r["before"],
                    "shot": r["shot"],
                    "console": [l for l in r["logs"]
                                if "headway" in l or "error" in l.lower()][:12],
                })

    path = os.path.join(a.out, "geometry.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2)

    for e in report:
        m = e["metrics"]
        cap = e["capture"]
        print("=" * 72)
        print("CLIP %s  layer=%s  fit=%s   %s"
              % (e["clip"], e["layer"], e["fit"],
                 "PASS" if e["ok"] else "FAIL"))
        print("  captured  : %s  (expected %dx%d)  frames=%d  worst marker err %.2f px"
              % (", ".join("%dx%d" % (s["w"], s["h"]) for s in cap.get("sizes", []))
                 or "none",
                 cap["expected"]["w"], cap["expected"]["h"], cap["n"],
                 cap["worst_marker_err_px"]))
        pv, sc = m["preview"], m["scout"]
        print("  displayed : preview video %sx%s  css %sx%s  fit=%s"
              % (pv["videoW"], pv["videoH"], pv["clientW"], pv["clientH"],
                 pv["objectFit"]))
        print("  scout     : video %sx%s  css %sx%s"
              % (sc["videoW"], sc["videoH"], sc["clientW"], sc["clientH"]))
        print("  wrap      : %sx%s   canvas attr %sx%s css %sx%s dpr %s"
              % (m["wrap"]["clientW"], m["wrap"]["clientH"],
                 m["canvas"]["attrW"], m["canvas"]["attrH"],
                 m["canvas"]["cssW"], m["canvas"]["cssH"], m["dpr"]))
        for row in e["placement"]["rows"]:
            if "case" in row:
                print("    marker@%-16s %-28s expected %-10s got %-10s%s"
                      % (row["marker_center"], row["case"], row["expected"],
                         row["observed"], "" if row["ok"] else "   <-- FAIL"))
                continue
            print("    %-6s marker@%-16s drawn %-26s centre err %-6s edge err %-6s%s"
                  % (row.get("marker_name"), row["marker_center"], row["drawn"],
                     row["center_err_px"], row["edge_err_px"],
                     "" if row["ok"] else "   <-- FAIL"))
        lg = e.get("lag_replay")
        if lg and lg.get("n"):
            print("  draw lag  : p50 %.0f ms  p95 %.0f ms  max %.0f ms  over %d "
                  "results, %d past the %.0f ms tolerance"
                  % (1000 * (lg["p50"] or 0), 1000 * (lg["p95"] or 0),
                     1000 * (lg["max"] or 0), lg["n"], lg["late"],
                     1000 * lg["tolerance_s"]))
        al = e.get("alignment")
        if al:
            print("  alignment : marker %.0f px/s on screen, analysis every "
                  "%s ms, budget %.1f ms"
                  % (al["v_screen_px_s"], al["analysis_interval_ms"],
                     al["budget_ms"]))
            for row in al["rows"]:
                print("    marker@%-16s box offset %-8s = %s ms%s"
                      % (row["marker"], row.get("dx_px"), row["err_ms"],
                         "" if row["ok"] else "   <-- FAIL"))
        for sk in e["placement"].get("occluded", []):
            print("    marker at %s hidden under page chrome (%s) -- not judged"
                  % (sk["marker_screen"], sk["under"]))
        if not e["placement"]["rows"]:
            print("    no markers found on screen")
    print("=" * 72)
    print("report: %s" % path)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
