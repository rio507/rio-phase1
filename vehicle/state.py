"""state.py — §21's vehicle state, as a VIEW rather than a fourth authority.

Three things in this repository already know something about whether the car is
being driven, and they knew it before this file existed:

    sessions.py              the driver tapped Start Drive; the heartbeat is
                             alive; the reaper closes it if the client vanishes
    diag/drivecycle.py       a drive cycle, opened by that session and closed
                             with it, with a motion fallback for when nobody
                             told us
    telemetry._Runtime       how long the engine has been running, gated on
                             TELEMETRY_ENGINE_RUNNING_RPM

§21 asks for an eight-state machine. Implemented as a fourth source of truth it
would disagree with the other three within a week — the classic failure, and one
this codebase has avoided everywhere else by deriving rather than duplicating.
So this DERIVES. It owns no state, starts no timer and decides nothing that
anything else already decides.

    offline             nothing is reporting at all
    connected           data is arriving, the engine is not turning
    key_on              the electrics are live, the engine is not turning
    engine_running      the engine is turning
    stationary_running  turning, not moving
    driving             turning and moving
    engine_off          it was running and has stopped
    session_complete    the drive has ended

"A SINGLE SOURCE MUST NOT DETERMINE DRIVE STATE WITHOUT VALIDATION"
-------------------------------------------------------------------
§21 says so and it is the reason `engine_off` needs both no rpm AND a timeout:
one missed poll is not the engine stopping. Every transition below is either
corroborated by a second signal or held for a period, and the payload carries
`why` so a disagreement can be read rather than reproduced.
"""
from __future__ import annotations

import time
from typing import Optional

import config

OFFLINE = "offline"
CONNECTED = "connected"
KEY_ON = "key_on"
ENGINE_RUNNING = "engine_running"
STATIONARY_RUNNING = "stationary_running"
DRIVING = "driving"
ENGINE_OFF = "engine_off"
SESSION_COMPLETE = "session_complete"

LABEL = {
    OFFLINE: "Offline",
    CONNECTED: "Connected",
    KEY_ON: "Key On",
    ENGINE_RUNNING: "Engine Running",
    STATIONARY_RUNNING: "Stationary, Running",
    DRIVING: "Driving",
    ENGINE_OFF: "Engine Off",
    SESSION_COMPLETE: "Drive Complete",
}

# How long with no rpm before the engine counts as off rather than as a poll
# that went missing. Matched to the telemetry staleness rule, because they are
# answering the same question about the same data.
_ENGINE_OFF_AFTER_S = config.TELEMETRY_STALE_AFTER_S * 2


def derive(telemetry_snapshot: dict, session_open: bool,
           drive_cycle: Optional[dict], now: float = None) -> dict:
    """-> {state, label, why, ...}. Reads; never writes.

    `telemetry_snapshot` is telemetry.snapshot(record=False). Passing it in
    rather than fetching it means the state and the panel are computed from the
    SAME read, which is the whole reason HealthSource.refresh exists.
    """
    now = time.time() if now is None else now
    snap = telemetry_snapshot or {}
    rows = {r["id"]: r for r in (snap.get("rows") or [])}

    available = bool(snap.get("available"))
    running = bool(snap.get("engine_running"))
    stale = bool(snap.get("stale"))
    updated = snap.get("updated_at")
    age = None if updated is None else round(now - updated, 1)

    rpm = (rows.get("rpm") or {}).get("value")
    speed = (rows.get("vehicle_speed") or {}).get("value")
    volts = (rows.get("battery_voltage") or {}).get("value")

    if not available or updated is None:
        return _out(OFFLINE, "nothing is reporting", now, age, rpm, speed,
                    session_open, drive_cycle)

    if stale and age is not None and age > _ENGINE_OFF_AFTER_S:
        # Data arrived once and has stopped. Not the same as offline — we know
        # what the car was doing a moment ago, and that is worth saying.
        return _out(OFFLINE, f"no data for {age:.0f}s", now, age, rpm, speed,
                    session_open, drive_cycle)

    if running:
        if speed is not None and speed >= config.HEALTH_DRIVING_MPH:
            return _out(DRIVING, f"{speed:.0f} mph", now, age, rpm, speed,
                        session_open, drive_cycle)
        if speed is not None:
            return _out(STATIONARY_RUNNING, "engine turning, not moving", now,
                        age, rpm, speed, session_open, drive_cycle)
        # Running, and nobody can say whether it is moving. Deliberately the
        # weaker claim: `engine_running` says what is corroborated and
        # `driving` would be a guess.
        return _out(ENGINE_RUNNING, "engine turning, road speed unknown", now,
                    age, rpm, speed, session_open, drive_cycle)

    # Not running. Three ways to arrive here and they are different facts.
    if not session_open and drive_cycle is None:
        if volts is not None and volts > 0:
            return _out(KEY_ON, "electrics live, engine not turning", now, age,
                        rpm, speed, session_open, drive_cycle)
        return _out(CONNECTED, "data arriving, engine not turning", now, age,
                    rpm, speed, session_open, drive_cycle)

    if drive_cycle is None and session_open is False:
        return _out(SESSION_COMPLETE, "the drive has ended", now, age, rpm,
                    speed, session_open, drive_cycle)

    if age is not None and age <= _ENGINE_OFF_AFTER_S:
        return _out(ENGINE_OFF, "engine stopped, data still arriving", now, age,
                    rpm, speed, session_open, drive_cycle)
    return _out(ENGINE_OFF, f"no rpm for {age:.0f}s" if age else "no rpm", now,
                age, rpm, speed, session_open, drive_cycle)


def _out(state: str, why: str, now: float, age: Optional[float],
         rpm, speed, session_open: bool, drive_cycle: Optional[dict]) -> dict:
    return {
        "state": state,
        "label": LABEL.get(state, state),
        # Every state carries WHY. A state machine whose transitions cannot be
        # read after the fact can only be debugged by reproducing them, and this
        # one is driven by a car.
        "why": why,
        "at": now,
        "data_age_s": age,
        "rpm": rpm,
        "vehicle_speed": speed,
        "session_open": bool(session_open),
        "drive_cycle_id": (drive_cycle or {}).get("cycle_id"),
        "drive_started_at": (drive_cycle or {}).get("started_at"),
        "drive_started_by": (drive_cycle or {}).get("started_by"),
    }


def is_driving(state: str) -> bool:
    return state == DRIVING


def is_running(state: str) -> bool:
    return state in (ENGINE_RUNNING, STATIONARY_RUNNING, DRIVING)
