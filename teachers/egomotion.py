"""Where the car has just been — 1.6 s of it, in the frame it is in now.

WHY THIS EXISTS AT ALL
----------------------
Alpamayo does not take a picture and answer. It takes a picture AND the ego
vehicle's recent motion, because "should I brake" is a different question at
50 km/h closing than at 50 km/h steady, and the frames alone do not say which.
Its own loader builds 16 poses at 10 Hz ending at t0, expressed in the ego
frame AT t0 -- so the last pose is always the origin, and the path runs
backwards from it into the past. That shape is what this file produces, out of
what a phone in a windscreen cradle can actually measure.

WHAT A PHONE CAN ACTUALLY MEASURE
---------------------------------
Not position. GPS at 1 Hz with 5-10 m of noise, differenced over 1.6 s, is
mostly noise: the displacement being measured is 20 m and the error on each end
is half of that. So position is DEAD RECKONED, from two things that are good
over short windows:

  SPEED, from the server's own resolver -- OBD when there is a bus worth
  reading, GPS otherwise, and it says which. This is the same number the
  headway band rests on, deliberately: an ego history built from a different
  speed than the warning was computed from would make the corpus disagree with
  itself.

  YAW RATE, from the phone's IMU. Not from its compass, which lags and swings
  between buildings, and not from gyroscope axes read directly -- see below.

Over 1.6 s at road speeds this is accurate to well under a metre, which is far
inside what the model is sensitive to. Over a minute it would drift badly; it
is never asked to.

THE ONE PIECE OF GEOMETRY WORTH READING TWICE
---------------------------------------------
The phone's own axes are useless here, because nobody mounts a phone squarely.
A cradle tilts, a vent mount rolls, and a phone lying face-up on the passenger
seat is a perfectly ordinary thing for a test drive. Reading `rotationRate` and
calling one of its components "yaw" is therefore wrong by an unknown angle that
changes when the driver adjusts the mount.

Gravity fixes it. DeviceMotion reports acceleration both with and without
gravity, so the difference is the gravity vector in the phone's own frame --
which is, to within the car's own tilt, straight down. The car's yaw axis is
straight up. So the vehicle's yaw rate is the component of the measured
rotation-rate vector along that axis:

    yaw_rate = omega . up,    up = -g / |g|

and it is right for any mounting, including one that changes mid-drive.

WHEN THERE IS NO IMU
--------------------
A laptop, a phone whose owner declined the iOS permission prompt, a desk test.
The history is then built from speed alone with zero yaw rate -- a straight
line behind the car -- and the record says `yaw_rate: "none"`. That is a worse
input and an honest one, and a corpus row carrying it can be filtered out
later. It is not a reason to send nothing.
"""
import math
import threading
import time
from collections import deque

import config

_lock = threading.Lock()
_sessions = {}      # key -> {"imu": deque, "speed": deque, "n": int, ...}

# Gravity, only ever used to normalise. A phone in a moving car measures
# somewhere between 9.6 and 10.2 depending on the road; anything outside this
# window is not gravity and the sample is refused rather than trusted.
_G_MIN, _G_MAX = 6.0, 14.0


def _state(key):
    st = _sessions.get(key)
    if st is None:
        ring = int(getattr(config, "TEACHER_EGO_RING", 256))
        st = {"imu": deque(maxlen=ring), "speed": deque(maxlen=ring),
              "n_imu": 0, "n_speed": 0, "refused": 0, "last_t": 0.0}
        _sessions[key] = st
    return st


def yaw_rate_from_sample(rr, g):
    """(rotationRate deg/s in phone x,y,z), (gravity m/s^2 in phone x,y,z)
    -> yaw rate in rad/s about the car's vertical, or None.

    Positive is a LEFT turn, which is positive yaw in the FLU frame the rest
    of this package uses. `up` is minus gravity, so the dot product comes out
    with that sign for free.
    """
    if not rr or not g or len(rr) < 3 or len(g) < 3:
        return None
    try:
        gx, gy, gz = float(g[0]), float(g[1]), float(g[2])
        wx, wy, wz = float(rr[0]), float(rr[1]), float(rr[2])
    except (TypeError, ValueError):
        return None
    mag = math.sqrt(gx * gx + gy * gy + gz * gz)
    if not (_G_MIN <= mag <= _G_MAX):
        # Freefall, a phone being shaken, or a browser reporting nothing.
        return None
    ux, uy, uz = -gx / mag, -gy / mag, -gz / mag
    deg_s = wx * ux + wy * uy + wz * uz
    return math.radians(deg_s)


def ingest(session_key: str, samples: list, offset_s: float = 0.0) -> dict:
    """Take a batch of client IMU samples. -> a small stats dict.

    `offset_s` is added to every client timestamp to bring it into this
    server's clock -- the frame transport already measures that offset against
    this process and the client sends the number it measured, so the IMU and
    the frames end up on one ruler rather than two.

    Never raises: this is fed by a phone over a mobile link and a malformed
    batch must cost the batch, not the drive.
    """
    key = str(session_key or "default")
    now = time.time()
    max_age = float(getattr(config, "TEACHER_EGO_SAMPLE_MAX_AGE_S", 3.0))
    taken = refused = 0
    with _lock:
        st = _state(key)
        for s in (samples or []):
            if not isinstance(s, dict):
                refused += 1
                continue
            try:
                t = float(s.get("t")) + float(offset_s or 0.0)
            except (TypeError, ValueError):
                refused += 1
                continue
            # A sample from the future, or from before the window, is not a
            # sample of now. Dropped rather than carried: a bad clock offset
            # would otherwise silently shift the whole history.
            if t > now + 1.0 or (now - t) > max_age * 4:
                refused += 1
                continue
            yaw = yaw_rate_from_sample(s.get("rr"), s.get("g"))
            if yaw is None:
                refused += 1
                continue
            st["imu"].append({"t": t, "yaw_rate": yaw,
                              "rr": list(s.get("rr") or []),
                              "g": list(s.get("g") or [])})
            taken += 1
        st["n_imu"] += taken
        st["refused"] += refused
        st["last_t"] = now
    return {"taken": taken, "refused": refused}


def note_speed(session_key: str, v_ms, source: str, at: float = None) -> None:
    """One resolved speed, from the headway result that just landed.

    Called on the frame path, so it does exactly this much: a float, a string
    and an append to a bounded deque under a lock nothing else holds for long.
    """
    if v_ms is None:
        return
    key = str(session_key or "default")
    with _lock:
        st = _state(key)
        st["speed"].append({"t": float(at or time.time()),
                            "v": float(v_ms), "src": str(source or "none")})
        st["n_speed"] += 1


def _sample_at(ring, t, max_gap):
    """Nearest entry to `t`, or None if the nearest is further than max_gap.

    Nearest rather than interpolated on purpose. Speed comes in at 4-15 Hz and
    yaw rate at ~20 Hz against a 10 Hz grid, so the nearest sample is within
    half a period and interpolation would be inventing precision. The gap that
    was actually used is reported, so a reader can see when it was not.
    """
    best, best_dt = None, None
    for s in ring:
        dt = abs(s["t"] - t)
        if best_dt is None or dt < best_dt:
            best, best_dt = s, dt
    if best is None or best_dt > max_gap:
        return None, best_dt
    return best, best_dt


def history(session_key: str, t0: float) -> dict:
    """The 16-pose ego history ending at t0. -> EGO_FIELDS dict, or None.

    None only when there is no speed at all -- with no idea how fast the car
    was going there is no history to build and a straight line of zeros would
    be a lie the model would take seriously. Everything else degrades: no IMU
    means no yaw, a sparse ring means a larger max_gap_s, and both are
    reported rather than hidden.
    """
    key = str(session_key or "default")
    steps = int(getattr(config, "TEACHER_EGO_HISTORY_STEPS", 16))
    dt = float(getattr(config, "TEACHER_EGO_HISTORY_STEP_S", 0.1))
    max_age = float(getattr(config, "TEACHER_EGO_SAMPLE_MAX_AGE_S", 3.0))

    with _lock:
        st = _sessions.get(key)
        imu = list(st["imu"]) if st else []
        spd = list(st["speed"]) if st else []

    # The grid: t0-(n-1)*dt ... t0, oldest first, so the last row is t0 itself.
    grid = [t0 - (steps - 1 - i) * dt for i in range(steps)]

    # Speed is the one thing that cannot be defaulted.
    speeds, gaps = [], []
    src_speed = "none"
    for t in grid:
        s, gap = _sample_at(spd, t, max_age)
        if s is None:
            speeds.append(None)
        else:
            speeds.append(s["v"])
            src_speed = s["src"]
            gaps.append(gap)
    known = [v for v in speeds if v is not None]
    if not known:
        return None
    # Hold the nearest known value across the gaps. At 4-15 Hz against a 10 Hz
    # grid there are rarely any; when the feed has stuttered there are, and a
    # held speed is a better history than a zero.
    extrapolated = 0
    for i, v in enumerate(speeds):
        if v is None:
            extrapolated += 1
            speeds[i] = known[0] if i < len(speeds) // 2 else known[-1]

    yaws_in, src_yaw = [], "none"
    for t in grid:
        s, gap = _sample_at(imu, t, max_age)
        if s is None:
            yaws_in.append(0.0)
        else:
            yaws_in.append(s["yaw_rate"])
            src_yaw = "imu"
            gaps.append(gap)

    # --- integrate, forwards, in a world frame that starts at the oldest pose
    # Unicycle: heading turns at the measured yaw rate, position advances at
    # the measured speed along the heading. Both are held over the step, which
    # at 100 ms is the same order as the sensors' own jitter.
    xs, ys, th = [0.0], [0.0], [0.0]
    for i in range(steps - 1):
        theta = th[-1] + yaws_in[i] * dt
        v = speeds[i]
        xs.append(xs[-1] + v * math.cos(theta) * dt)
        ys.append(ys[-1] + v * math.sin(theta) * dt)
        th.append(theta)

    # --- and then re-express it in the frame at t0, which is the last pose.
    # Subtract its position, rotate by minus its heading. The last row comes
    # out (0,0,0) with yaw 0, exactly as Alpamayo's own loader produces.
    x0, y0, t0h = xs[-1], ys[-1], th[-1]
    c, s = math.cos(-t0h), math.sin(-t0h)
    xyz, yaw = [], []
    for i in range(steps):
        dx, dy = xs[i] - x0, ys[i] - y0
        xyz.append([round(dx * c - dy * s, 4), round(dx * s + dy * c, 4), 0.0])
        yaw.append(round(th[i] - t0h, 5))

    return {
        "steps": steps,
        "step_s": dt,
        "frame": "FLU_at_t0",
        "xyz": xyz,
        "yaw": yaw,
        "source": {
            "speed": src_speed,
            "yaw_rate": src_yaw,
            # Said plainly because it is the honest description of the method
            # and somebody reading the corpus should not have to infer it.
            "position": "dead_reckoned",
        },
        "quality": {
            "n_speed": len(spd), "n_imu": len(imu),
            "max_gap_s": round(max(gaps), 3) if gaps else None,
            "extrapolated": extrapolated,
            "imu": src_yaw == "imu",
        },
    }


def rot_matrices(yaw_list):
    """Yaw angles -> the (steps, 3, 3) rotation matrices Alpamayo wants.

    Kept out of history() because the service is what needs the matrices and
    the corpus is better off with the angle: one number per pose instead of
    nine, and no reader has to check whether a 3x3 is really a rotation.
    """
    out = []
    for a in yaw_list:
        c, s = math.cos(a), math.sin(a)
        out.append([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    return out


def status() -> dict:
    with _lock:
        return {k: {"imu": st["n_imu"], "speed": st["n_speed"],
                    "refused": st["refused"],
                    "buffered_imu": len(st["imu"]),
                    "buffered_speed": len(st["speed"]),
                    "idle_s": round(time.time() - st["last_t"], 1)
                              if st["last_t"] else None}
                for k, st in _sessions.items()}


def drop(session_key: str) -> bool:
    with _lock:
        return _sessions.pop(str(session_key or "default"), None) is not None


def reset_all() -> None:
    with _lock:
        _sessions.clear()
