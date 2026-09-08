"""Which speed the warning rests on, and how much to trust it.

OBD > GPS > coasted GPS > nothing. One function, one answer, and the answer
carries its own provenance so the HUD, the log and the policy all say the same
thing about it.

WHY THIS IS A FILE AND NOT THREE LINES IN live.py
--------------------------------------------------
Because "the phone streams GPS speed; headway must use it" turns out to have
four cases and three of them are about what to do when it is missing, and each
of those is a decision with a different failure mode:

  OBD          a car's own bus, when there is one. Beats the phone: it is
               measured at the wheels rather than differentiated from a
               position fix, and it does not vanish under a bridge.

  GPS          what the phone sends on every frame, with the age of the fix
               next to it. The normal case, and the case the first real drive
               ran on end to end (session 06af3214: v_host present on 1944 of
               1949 frames, tracking the navigation tracker's own speed to
               within a fraction of a metre per second).

  COASTED      a GPS fix that has gone stale but not ancient. The old code
               called this UNKNOWN and went silent. A car's speed does not
               change much in eight seconds, and a system that stops warning
               because the sky went quiet under a bridge teaches the driver
               that its silence means nothing. So the last speed is carried
               forward and every margin is WIDENED -- earlier warnings, a
               relaxed confidence floor -- because a speed we are less sure of
               is a reason to give the driver more room, not less.

  NONE         no fix at all, or one older than the coast window. This is the
               only case that silences the gap warnings, and it silences them
               because tau has no denominator, not as a policy choice.

THE MOCK MUST NOT OUTRANK THE PHONE
-----------------------------------
config.HEADWAY_OBD_SPEED_SOURCES lists which telemetry sources count as a car.
The mock and the simulation are not on it, and that is a safety property rather
than tidiness: both cheerfully report 0 mph while parked at a desk, and an
unconditional OBD-beats-GPS rule would have taken a live 20 m/s GPS fix on a
freeway and replaced it with the desk's zero, putting the whole drive in
SUPPRESSED and speaking not once.

DISAGREEMENT IS NOT A TIEBREAK
------------------------------
If OBD and a fresh GPS differ by more than
config.HEADWAY_SPEED_DISAGREEMENT_MS, neither is trusted outright: the drive
continues on the GPS value, DEGRADED, with widened margins. Picking a winner
between two sources that cannot both be right is how a system reports high
confidence in a number nobody checked.

No model, no network, no I/O beyond one telemetry read. Deterministic in its
inputs, which is what lets headway/live_selftest.py assert on it directly.
"""
import math

import config

# What a resolved speed is trusted for.
OBD = "obd"
GPS = "gps"
COASTED = "gps_coasted"
NONE = "none"


class SpeedFix:
    """One frame's answer to "how fast are we going, and how sure are we"."""

    __slots__ = ("v_ms", "source", "age_s", "degraded", "reason", "obd_ms",
                 "gps_ms")

    def __init__(self, v_ms, source, age_s, degraded, reason,
                 obd_ms=None, gps_ms=None):
        self.v_ms = v_ms
        self.source = source
        self.age_s = age_s
        # DEGRADED means "usable, and every margin widens". It never means
        # "silent" -- that is `source == NONE`, and only that.
        self.degraded = bool(degraded)
        self.reason = reason
        self.obd_ms = obd_ms
        self.gps_ms = gps_ms

    @property
    def known(self) -> bool:
        return self.v_ms is not None

    def to_log(self) -> dict:
        return {
            "v_ms": None if self.v_ms is None else round(self.v_ms, 3),
            "source": self.source,
            "age_s": None if self.age_s is None else round(self.age_s, 3),
            "degraded": self.degraded,
            "reason": self.reason,
            "obd_ms": None if self.obd_ms is None else round(self.obd_ms, 3),
            "gps_ms": None if self.gps_ms is None else round(self.gps_ms, 3),
        }


def _finite(x):
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _read_obd(reader):
    """The car's own speed, or None. Never raises."""
    if reader is None:
        return None
    try:
        row = reader()
    except Exception:
        return None
    if not row:
        return None
    if row.get("source") not in tuple(config.HEADWAY_OBD_SPEED_SOURCES):
        return None
    v = _finite(row.get("v_ms"))
    if v is None or v < 0:
        return None
    age = _finite(row.get("age_s"))
    if age is None or age > config.HEADWAY_OBD_SPEED_MAX_AGE_S:
        return None
    return {"v_ms": v, "age_s": age}


class SpeedResolver:
    """Holds one thing: the last speed that was actually believed.

    That is all a coast needs, and keeping it here rather than in the session
    means the resolver can be tested on its own with a list of frames.
    """

    def __init__(self, obd_reader=None):
        self._obd_reader = obd_reader
        self._last_v = None
        self._last_t = None
        self._last_source = None

    def reset(self):
        self._last_v = None
        self._last_t = None
        self._last_source = None

    def resolve(self, v_gps, v_gps_age_s, t) -> SpeedFix:
        """One frame. `t` is the session clock in seconds.

        Gate order is the priority order, and it is deliberate: every branch
        returns, so a speed's provenance is always the FIRST source that could
        supply it rather than whichever check happened to run last.
        """
        gps = _finite(v_gps)
        if gps is not None and gps < 0:
            gps = None
        gps_age = _finite(v_gps_age_s)
        gps_fresh = (gps is not None
                     and (gps_age is None
                          or gps_age <= config.HEADWAY_V_HOST_STALE_S))

        obd = _read_obd(self._obd_reader)

        # --- OBD, when there is a car under it ----------------------------
        if obd is not None:
            if gps_fresh and abs(obd["v_ms"] - gps) > config.HEADWAY_SPEED_DISAGREEMENT_MS:
                # Two sources that cannot both be right. Continue on the phone,
                # which is the one whose failure mode is understood, and widen
                # everything.
                return self._remember(SpeedFix(
                    gps, GPS, gps_age or 0.0, True, "obd_gps_disagree",
                    obd_ms=obd["v_ms"], gps_ms=gps), t)
            return self._remember(SpeedFix(
                obd["v_ms"], OBD, obd["age_s"], False, "obd",
                obd_ms=obd["v_ms"], gps_ms=gps), t)

        # --- the phone -----------------------------------------------------
        if gps_fresh:
            return self._remember(SpeedFix(gps, GPS, gps_age or 0.0, False,
                                           "gps", gps_ms=gps), t)

        # --- a fix that has gone quiet, but not long enough to disown -------
        if (self._last_v is not None and self._last_t is not None
                and t is not None
                and (t - self._last_t) <= config.HEADWAY_V_HOST_COAST_S):
            return SpeedFix(self._last_v, COASTED, t - self._last_t, True,
                            "coasted_last_known", gps_ms=gps)

        # --- nothing ------------------------------------------------------
        return SpeedFix(None, NONE, gps_age, True,
                        "stale_gps" if gps is not None else "no_fix",
                        gps_ms=gps)

    def _remember(self, fix: SpeedFix, t) -> SpeedFix:
        if fix.v_ms is not None and t is not None:
            self._last_v = fix.v_ms
            self._last_t = t
            self._last_source = fix.source
        return fix


def default_obd_reader():
    """The live car, read lazily so headway never imports telemetry at import.

    Returns a callable, or None where telemetry is not importable at all (a
    bench, a trimmed checkout). None means "no OBD", which the resolver already
    handles as the ordinary case.
    """
    try:
        import telemetry
    except Exception:
        return None
    return telemetry.road_speed_ms
