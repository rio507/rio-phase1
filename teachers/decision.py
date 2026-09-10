"""What Alpamayo's predicted path MEANS, in words, by arithmetic.

WHY THIS AND NOT A PROMPT
-------------------------
Alpamayo's unique output is a 6.4-second trajectory: 64 waypoints at 10 Hz, in
metres, in the ego frame at t0. Cosmos has nothing like it and nothing else on
this pod does either -- so it is the one row that column can fill that no other
column can.

Read as a ribbon on the glass it is atmospheric. Read as two words -- *slowing,
drifting right* -- it is a claim you can hold RIO's own deterministic state
against: the band said NORMAL and the model's own path says it would be braking.
That is the comparison the whole panel exists for.

The two words are DERIVED, not asked for. Asking the model to describe its own
trajectory would put a language model between a number and its meaning, and
the answer would be unreproducible from the row. Differencing 64 waypoints is
four lines of arithmetic, gives the same answer every time, and can be
re-derived by anyone reading the corpus a year from now.

STILL DISPLAY-ONLY
------------------
Nothing here reaches a warning, a band or a spoken line. It is a label on a
prediction, and the prediction is a picture. See teachers/__init__.py.

THE FRAME
---------
FLU at t0: x forward, y LEFT, z up (teachers/egomotion.py builds the history in
the same frame). So positive lateral displacement is a drift to the left, and
getting that backwards would put "drifting right" on a left-hand curve --
which is why the sign is asserted in both directions in the selftest.
"""
import math

# --- the thresholds, and why each is where it is --------------------------
# Longitudinal acceleration. 0.35 m/s^2 over a 6.4 s horizon is about 2 m/s of
# speed change -- roughly 8 km/h. Below that the "change" is inside the noise
# of a sampled diffusion head, and calling it slowing would put a word on the
# dashboard that flickers between keyframes on a car doing a steady speed.
ACCEL_EPS = 0.35            # m/s^2
# A predicted final speed under this is a stop rather than a slowing.
STOP_SPEED = 1.0            # m/s

# Lateral. A lane is ~3.5 m, so 1.2 m of net displacement over the horizon is
# a third of a lane: unambiguous drift, and still well short of a lane change.
DRIFT_M = 1.2               # metres of net lateral displacement
# Past this it is not a drift, it is a turn.
TURN_M = 4.0

# How much of the path each end average covers. A single first and last
# waypoint would be two samples of a noisy head; a second at each end is ten.
WINDOW_S = 1.0


def _speeds(xyz, hz):
    """Waypoint-to-waypoint speed, m/s. len(xyz) - 1 values."""
    dt = 1.0 / float(hz or 10)
    out = []
    for a, b in zip(xyz, xyz[1:]):
        dx = float(b[0]) - float(a[0])
        dy = float(b[1]) - float(a[1])
        out.append(math.hypot(dx, dy) / dt)
    return out


def describe(trajectory: dict) -> dict:
    """A predicted path -> {"longitudinal", "lateral", "text", ...} or {}.

    Every number it decided on rides back with the words, so the label can be
    argued with from the row rather than trusted.
    """
    if not trajectory:
        return {}
    xyz = trajectory.get("xyz") or []
    if len(xyz) < 8:
        return {}
    hz = float(trajectory.get("hz") or 10)
    n_win = max(2, int(round(WINDOW_S * hz)))

    v = _speeds(xyz, hz)
    if len(v) < 2 * n_win:
        n_win = max(1, len(v) // 3)
    v_early = sum(v[:n_win]) / n_win
    v_late = sum(v[-n_win:]) / n_win
    # Between the CENTRES of the two windows, not between the ends: averaging
    # over a second and then dividing by the whole horizon would understate the
    # acceleration by however wide the windows are.
    span_s = (len(v) - n_win) / hz
    accel = (v_late - v_early) / span_s if span_s > 0 else 0.0

    if v_late <= STOP_SPEED and v_early > STOP_SPEED:
        longitudinal = "stopping"
    elif accel <= -ACCEL_EPS:
        longitudinal = "slowing"
    elif accel >= ACCEL_EPS:
        longitudinal = "accelerating"
    else:
        longitudinal = "holding"

    # LATERAL. y is metres to the LEFT in the FLU frame at t0.
    y_end = float(xyz[-1][1])
    y_max = max((abs(float(p[1])) for p in xyz), default=0.0)
    if abs(y_end) >= TURN_M:
        lateral = "turning left" if y_end > 0 else "turning right"
    elif abs(y_end) >= DRIFT_M:
        lateral = "drifting left" if y_end > 0 else "drifting right"
    else:
        lateral = "straight"

    reach = float(xyz[-1][0])
    return {
        "longitudinal": longitudinal,
        "lateral": lateral,
        "text": f"{longitudinal.capitalize()}, {lateral}",
        "v_start_ms": round(v_early, 2),
        "v_end_ms": round(v_late, 2),
        "accel_ms2": round(accel, 3),
        "lateral_end_m": round(y_end, 2),
        "lateral_max_m": round(y_max, 2),
        "reach_m": round(reach, 1),
        "horizon_s": trajectory.get("horizon_s"),
        "points": len(xyz),
        # The thresholds this verdict was reached with, on the record, because
        # a label whose thresholds are not written down cannot be re-derived
        # from a corpus row.
        "thresholds": {"accel_ms2": ACCEL_EPS, "drift_m": DRIFT_M,
                       "turn_m": TURN_M, "stop_ms": STOP_SPEED,
                       "window_s": WINDOW_S},
    }


def split_physics(text: str, marker: str = "Plausibility:") -> dict:
    """Cosmos's physics answer -> {"actors", "plausibility", "implausible"}.

    The prompt asks for a line beginning with the marker precisely so this can
    be a split rather than a search for a verdict in prose. When the model does
    not produce one -- which happens -- the whole answer is the actor account
    and `plausibility` is empty, said plainly rather than guessed at.
    """
    t = (text or "").strip()
    if not t:
        return {}
    i = t.find(marker)
    if i < 0:
        return {"actors": t, "plausibility": "", "implausible": None,
                "has_verdict": False}
    actors = t[:i].strip()
    verdict = t[i + len(marker):].strip()
    low = verdict.lower()
    # WHAT COUNTS AS A FLAG. Only an explicit denial: the common answer is
    # "the scenario is physically possible", and matching on "possible" would
    # invert it. Negations first, then the bare positives.
    implausible = None
    # BARE YES/NO FIRST, AND MIND THE POLARITY. The prompt asks whether
    # anything "could not physically happen", so "No." means nothing is
    # implausible and "Yes" means something is -- the opposite of the reading a
    # naive keyword match gives. Cosmos answered exactly that way on two of ten
    # keyframes of the acceptance clip, and both came back as "no verdict"
    # until this was here.
    bare = low.strip().rstrip(".!").strip()
    if bare in ("no", "none", "nothing", "no."):
        implausible = False
    elif bare in ("yes", "yes."):
        implausible = True
    elif any(w in low for w in ("not physically", "impossible", "implausible",
                                "cannot physically", "could not physically",
                                "physically unlikely", "not plausible")):
        implausible = True
    elif any(w in low for w in ("plausible", "possible", "consistent",
                                "nothing", "no physical",
                                "no inconsistencies", "no unrealistic")):
        implausible = False
    return {"actors": actors, "plausibility": verdict,
            "implausible": implausible, "has_verdict": True}
