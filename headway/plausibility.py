"""Is this box's size consistent with the distance claimed for it?

One pinhole identity, applied per detection:

    h_px = f_px * H_real / d        ->        d = f_px * H_real / h_px

A class fixes H_real inside a narrow range (a car is 1.35-1.95 m tall; nothing
on the road is 30 cm tall or 8 m tall), and the corridor already computes f_px
from the camera's horizontal FOV. So a box's PIXEL HEIGHT implies a range
window, and any depth reading outside that window is arithmetically impossible
for an object of that class -- whatever the depth model says, and however
confident it says it.

WHY THIS EXISTS
---------------
Observed on a coastal-road clip: a cluster of small boxes at the vanishing
point, labelled 2 m, 6 m and 9 m alongside 62 m and 51 m, in the same overlay,
on the same frame. Boxes ~10 px tall cannot be 2 m away -- a car at 2 m fills
the frame. Depth Anything is being asked for the median depth inside a
10x8 px ROI that contains road-end texture and horizon, and it answers with a
number, because that is what it does. depth.roi_depth's own confidence cannot
catch this: the ROI is small and internally consistent, so spread is low and
the score is high. The check has to come from OUTSIDE the depth model, and
geometry is the only thing that qualifies.

This is a measurement veto, not a detector: it never says a box is not a car,
only that a range attached to it is not believable. An implausible range is
turned into NO range, which is a state the whole pipeline already handles --
membership refuses to make a rangeless candidate the lead, and the overlay
draws "--" rather than a number.

WHAT IT DELIBERATELY DOES NOT DO
--------------------------------
It does not enforce the NEAR bound on a vertically truncated box. A vehicle
close enough to be clipped by the frame edge has a visible height smaller than
its real one, so its implied range is an OVER-estimate and the near bound would
reject exactly the vehicle that matters most. The far bound still holds for
those (true height >= visible height => true range <= the far bound), so a
truncated box is checked on the side where the inequality is still sound.

TOLERANCE
---------
TOL widens the window multiplicatively either side. It absorbs DA-V2's absolute
scale bias (~10-20%, §0 Challenge 2), camera pitch the flat-road model does not
carry, and the fact that H_real is a range and not a constant. At 1.6 it is
loose on purpose: this gate is here to catch the impossible, not to second-guess
plausible readings. Every rejection it makes should be one a person looking at
the frame would agree with.
"""
import math

# Real-world heights in metres, (min, max). A window rather than a point: the
# check is only as tight as the class's genuine size spread, and pretending a
# car is exactly 1.5 m tall would reject estate cars and vans at the edges.
CLASS_HEIGHT_M = {
    "car": (1.30, 2.10),        # hatchback roofline .. tall SUV
    "truck": (2.20, 4.20),      # box van .. artic
    "bus": (2.60, 3.80),
    "motorcycle": (1.20, 1.90),  # with rider
    "cyclist": (1.30, 2.00),     # bicycle + rider
    "pedestrian": (1.20, 2.05),  # a child is in this range; a seated adult is not
}

# Multiplicative slack either side of the geometric window. See the header.
TOL = 1.6

# --- size floor -------------------------------------------------------------
# A box shorter than this is past the range where anything downstream can use
# it: at f_px ~1100 a 20 px car is ~83 m away, and depth.py's own range_score
# has been decaying since 50 m. Detections between the hard drop floor
# (detect.MIN_BOX_PX) and this are kept and drawn, because a driver should see
# that RIO has noticed something, but they are marked UNCONFIRMED: no range is
# claimed for them and they cannot be the reason for a warning.
CONFIRM_MIN_PX = 20.0

# How close to the frame edge counts as truncated. Two pixels of slack, because
# a box clamped to the frame in detect.py lands exactly ON the edge.
EDGE_PX = 2.0


def focal_px(width_px, hfov_deg=None):
    """Pinhole focal length in pixels, from the same FOV the corridor uses.

    Imported from anchor rather than restated so there is one camera model in
    the system: if HFOV_DEG is ever measured properly for a real install, the
    corridor and this check move together.
    """
    if hfov_deg is None:
        from .anchor import HFOV_DEG as hfov_deg  # noqa: N813
    return (float(width_px) / 2.0) / math.tan(math.radians(float(hfov_deg)) / 2.0)


def box_height_px(box):
    return max(0.0, float(box[3]) - float(box[1]))


def is_truncated(box, image_h, edge_px=EDGE_PX):
    """Is this box clipped by the top or bottom of the frame?"""
    return (float(box[1]) <= edge_px
            or float(box[3]) >= float(image_h) - edge_px)


def range_window(label, box, f_px, tol=TOL):
    """Plausible range window (lo, hi) in metres for this box, or None.

    None means the class has no size on record -- an unknown label is not
    evidence of anything and must not be rejected on geometry it has no entry
    for.
    """
    dims = CLASS_HEIGHT_M.get(str(label or "").lower())
    if dims is None:
        return None
    h_px = box_height_px(box)
    if h_px <= 0.0:
        return None
    h_min, h_max = dims
    return (f_px * h_min / h_px / tol, f_px * h_max / h_px * tol)


def implied_range_m(label, box, f_px):
    """Mid-window range: what this box's SIZE says its distance is.

    Reported alongside a rejection so the log says what the geometry expected,
    not just that it disagreed.
    """
    win = range_window(label, box, f_px, tol=1.0)
    return None if win is None else (win[0] + win[1]) / 2.0


def check(label, box, depth_m, f_px, image_h=None, tol=TOL,
          confirm_min_px=CONFIRM_MIN_PX):
    """Is `depth_m` a believable range for this box? -> verdict dict.

    Keys:
        ok            may this range be used and displayed
        reason        why not, when it is not
        confirmed     is the box big enough to claim anything about at all
        h_px          the box's pixel height, the input to the whole check
        implied_m     what the box's size says the range is
        window        (lo, hi) the depth had to fall inside
        truncated     was the near bound skipped because the box is clipped

    `ok` is False for a missing depth as well as an impossible one: both mean
    "no range you may put on the screen", and every caller wants that one
    answer rather than a tri-state.
    """
    h_px = box_height_px(box)
    confirmed = h_px >= float(confirm_min_px)
    out = {
        "ok": False, "reason": "", "confirmed": bool(confirmed),
        "h_px": round(h_px, 1), "implied_m": None, "window": None,
        "truncated": False,
    }

    if depth_m is None or not math.isfinite(float(depth_m)) or float(depth_m) <= 0.0:
        out["reason"] = "no_depth"
        return out
    depth_m = float(depth_m)

    if not confirmed:
        # Not "implausible" -- unmeasurable. A 14 px box is beyond the range
        # where a depth reading means anything, so no range is claimed for it
        # rather than a wrong one being argued about.
        out["reason"] = "below_size_floor"
        return out

    win = range_window(label, box, f_px, tol=tol)
    if win is None:
        # No size on record for this class. Geometry has no opinion, so it does
        # not get a veto: the depth stands or falls on the depth model's own
        # confidence, exactly as it did before this module existed.
        out.update(ok=True, reason="no_class_size")
        return out

    lo, hi = win
    out["window"] = (round(lo, 1), round(hi, 1))
    out["implied_m"] = round(implied_range_m(label, box, f_px), 1)

    truncated = image_h is not None and is_truncated(box, image_h)
    out["truncated"] = bool(truncated)

    if depth_m > hi:
        out["reason"] = "depth_too_far_for_box"
        return out
    if depth_m < lo and not truncated:
        # The near bound is the one that catches the vanishing-point cluster:
        # a small box claiming a small distance. Skipped when the box is
        # clipped by the frame, where a small visible height is expected.
        out["reason"] = "depth_too_near_for_box"
        return out

    out["ok"] = True
    out["reason"] = "ok"
    return out
