"""A predicted path, drawn on the picture — with the camera model RIO already has.

Alpamayo's trajectory comes back as 64 points at 10 Hz in metres, in the ego
frame at t0: x forward, y left, z up. To draw it on the feed it has to be
projected, and the temptation is to guess a focal length. There is no need to:
headway/anchor.py has been projecting the ego corridor through a pinhole model
since the corridor existed, the overlay already draws that corridor over the
same picture, and a ribbon drawn through a DIFFERENT camera model would sit
next to the corridor being subtly, visibly wrong.

So this imports that model's three constants and does the same arithmetic. The
selftest asserts they still match, because the failure mode of copying three
numbers is that somebody calibrates one of them and only one copy moves.

WHAT THE RIBBON IS AND IS NOT
-----------------------------
It is a picture of what a model thinks the car will do. It is flat-road, it
assumes the pitch the corridor assumes, and at 6.4 s it reaches 100 m at
motorway speed, where a degree of pitch error is metres of ground. It is drawn
faint for that reason.

It is not used for anything. Nothing reads these pixels back, no warning is
computed from them, and the metric path is carried in the corpus beside them so
a later reader can re-project it through a better camera model than this one.
"""
import math

# The one camera model on this page. Imported rather than restated so a
# calibration lands in both places at once -- and asserted in the selftest so
# that this comment cannot quietly stop being true.
from headway.anchor import CAMERA_HEIGHT_M, CAMERA_PITCH_RAD, HFOV_DEG


def camera(width_px: float, height_px: float, hfov_deg: float = None) -> dict:
    """The pinhole parameters for a frame of this size, as plain floats."""
    w = float(width_px)
    fov = float(HFOV_DEG if hfov_deg is None else hfov_deg)
    return {
        "f_px": (w / 2.0) / math.tan(math.radians(fov) / 2.0),
        "cx": w / 2.0,
        "cy": float(height_px) / 2.0,
        "h_m": float(CAMERA_HEIGHT_M),
        "pitch": float(CAMERA_PITCH_RAD),
    }


def ground_to_pixel(forward_m: float, left_m: float, cam: dict):
    """(forward, left) on the road plane -> (u, v) in pixels, or None.

    The inverse of EgoCorridor.ground_from_pixel, and deliberately written the
    same way round as EgoCorridor.polygon(), which is the code that already
    draws ground geometry onto this feed:

        z_c = h sin(theta) + L cos(theta)      depth along the optical axis
        y_c = h cos(theta) - L sin(theta)      height below the axis
        v   = cy + f * y_c / z_c
        u   = cx + f * lateral / z_c

    `left_m` is FLU (positive to the left); the camera's lateral axis points
    right, so it is negated on the way in. Getting that backwards would put a
    left turn's ribbon over the right-hand lane, which is exactly the kind of
    wrong that looks plausible.
    """
    L = float(forward_m)
    if L <= 0.5:
        # Behind the camera, or so close the projection blows up. The first
        # points of a trajectory are metres from the bumper and are simply not
        # in shot; dropping them is right and the polyline starts where it can.
        return None
    ct, st = math.cos(cam["pitch"]), math.sin(cam["pitch"])
    z_c = cam["h_m"] * st + L * ct
    if z_c <= 1e-6:
        return None
    y_c = cam["h_m"] * ct - L * st
    v = cam["cy"] + cam["f_px"] * y_c / z_c
    u = cam["cx"] + cam["f_px"] * (-float(left_m)) / z_c
    return (u, v)


def trajectory_pixels(xyz, width_px, height_px, max_range_m: float = 80.0):
    """A metric path -> the pixel polyline to stroke, oldest point first.

    Returns [[u, v], ...] with the unprojectable points removed rather than
    nulled, so the overlay can stroke it without checking every vertex. Points
    beyond `max_range_m` are dropped: past that the whole 6.4 s tail collapses
    into a couple of pixels at the horizon and adds nothing but a smear.
    """
    if not xyz:
        return []
    cam = camera(width_px, height_px)
    out = []
    for p in xyz:
        if not p or len(p) < 2:
            continue
        fwd, left = float(p[0]), float(p[1])
        if fwd > max_range_m:
            break
        uv = ground_to_pixel(fwd, left, cam)
        if uv is None:
            continue
        u, v = uv
        if not (math.isfinite(u) and math.isfinite(v)):
            continue
        out.append([round(u, 1), round(v, 1)])
    return out
