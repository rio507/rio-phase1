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

AND THAT EXEMPTION IS WHERE A REAL DRIVE GOT THROUGH
----------------------------------------------------
Session 06af3214: a box at [0.6, 308, 640, 478] on a 640x480 frame -- the whole
width of the picture, flush with the bottom -- labelled `car`, ranged by the
depth model at 3.7 m, and held as the lead at 20 m/s for 584 frames. The height
check had no complaint: 170 visible pixels of car is a window of 2.6 m to
11.0 m, and 3.7 falls inside it. The box is the phone's view of the car's own
bonnet and dashboard.

The height was never going to catch that, because 170 px of "car" IS consistent
with a car four metres ahead. The WIDTH is what is impossible: 640 px of car,
at this focal length, is 1.6 metres away. Not 3.7.

So the same argument the header makes about height is now made about width, in
the direction where it is sound. Visible width <= true width always, so

    true range = f * W_true / w_true_px  <=  f * W_max / w_visible_px

is a FAR bound that holds whether the box is clipped or not.

Applied narrowly, and the narrowness is the point. A box's width is a poor
range estimator in general: the boxes are loose, HFOV_DEG is a guess pending
Stage 2 calibration, and a genuine lead at 50 m would be falsely vetoed by it.
At NEAR-FULL FRAME WIDTH it says something no calibration error can explain,
and there it is allowed to speak. Measured on that drive: 583 of the 584
offending frames vetoed, and the rule was not even applicable to any of the 295
frames carrying a genuine lead beyond 15 m.

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

import config

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

# Real-world WIDTHS, same idea and the same caveat: a window, not a constant.
# Only ever used as an upper bound (see wide_box_max_range_m), so what matters
# is that the maximum is generous -- a wing-mirrored pickup, a lorry with its
# load overhanging.
CLASS_WIDTH_M = {
    "car": (1.55, 2.10),        # city car .. wide-bodied SUV with mirrors
    "truck": (2.20, 2.90),
    "bus": (2.40, 2.90),
    "motorcycle": (0.60, 1.10),
    "cyclist": (0.50, 1.10),
    "pedestrian": (0.40, 0.95),
}

# Multiplicative slack either side of the geometric window. See the header.
TOL = 1.6

# How much of the frame width a box must span before its WIDTH is allowed to
# bound its range. See the header for why this is narrow on purpose, and
# config.py for the drive that set it.
WIDE_BOX_FRAC = config.HEADWAY_WIDE_BOX_FRAC

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


def box_width_px(box):
    return max(0.0, float(box[2]) - float(box[0]))


def wide_box_max_range_m(label, box, image_w, f_px, tol=TOL,
                         wide_frac=WIDE_BOX_FRAC):
    """The furthest this box can possibly be, from its width. None if it does
    not apply.

    Returns None -- meaning "no opinion" -- unless the box spans `wide_frac` of
    the frame. Everywhere else a box's width is too loose a measurement to veto
    anything with, and saying nothing is the honest answer rather than a
    generous bound nobody reads.
    """
    if image_w is None:
        return None
    w_px = box_width_px(box)
    if w_px <= 0.0 or w_px < float(wide_frac) * float(image_w):
        return None
    dims = CLASS_WIDTH_M.get(str(label or "").lower())
    if dims is None:
        return None
    return f_px * dims[1] / w_px * tol


def implied_range_m(label, box, f_px):
    """Mid-window range: what this box's SIZE says its distance is.

    Reported alongside a rejection so the log says what the geometry expected,
    not just that it disagreed.
    """
    win = range_window(label, box, f_px, tol=1.0)
    return None if win is None else (win[0] + win[1]) / 2.0


def check(label, box, depth_m, f_px, image_h=None, tol=TOL,
          confirm_min_px=CONFIRM_MIN_PX, image_w=None):
    """Is `depth_m` a believable range for this box? -> verdict dict.

    Keys:
        ok            may this range be used and displayed
        reason        why not, when it is not
        confirmed     is the box big enough to claim anything about at all
        h_px          the box's pixel height, the input to the whole check
        implied_m     what the box's size says the range is
        window        (lo, hi) the depth had to fall inside
        truncated     was the near bound skipped because the box is clipped
        wide_max_m    the furthest this box can be, from its WIDTH, when it is
                      wide enough for that to mean anything. None otherwise.

    `ok` is False for a missing depth as well as an impossible one: both mean
    "no range you may put on the screen", and every caller wants that one
    answer rather than a tri-state.
    """
    h_px = box_height_px(box)
    confirmed = h_px >= float(confirm_min_px)
    out = {
        "ok": False, "reason": "", "confirmed": bool(confirmed),
        "h_px": round(h_px, 1), "implied_m": None, "window": None,
        "truncated": False, "wide_max_m": None,
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

    # THE WIDTH BOUND, ASKED FIRST AND ANSWERED INDEPENDENTLY.
    #
    # Before the class-height window, because it does not need one: it is sound
    # for a truncated box, for an unconfirmed box, and for a class whose height
    # is not on record. On the drive that produced it, the box was `car`,
    # confirmed and untruncated at the top -- it passed every one of those and
    # was still the car's own bonnet.
    wide_max = wide_box_max_range_m(label, box, image_w, f_px, tol=tol)
    if wide_max is not None:
        out["wide_max_m"] = round(wide_max, 2)
        if depth_m > wide_max:
            # A box this wide is about two metres away or it is not a vehicle.
            out["reason"] = "depth_too_far_for_box_width"
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
