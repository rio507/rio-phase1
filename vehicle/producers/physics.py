"""physics.py — one engine model, feeding both the mock and the simulator.

This is telemetry.py's Holley mock, moved. Nothing about the model changed; what
changed is who can use it. It used to be the body of MockHolleyProvider.read(),
which meant the canonical simulator would have had to reimplement it — and two
engine models drift, so "the simulator produces what the mock produces" would
have become a thing you hoped rather than a thing that was true.

Now MockHolleyProvider calls it and the canonical simulator calls it, and the
identical numbers go down the direct path and the ingestion path.

CORRELATED, NOT RANDOM
----------------------
Every value below is derived from the same three driver inputs — throttle, rpm,
road speed — and from how long the engine has been running, because that is how
an engine actually works: MAP follows the throttle plate, oil pressure follows
rpm, oil temperature lags coolant, airflow follows rpm and manifold pressure, and
the alternator sags at crank and then holds ~14.2 V.

A mock that jitters fourteen independent numbers looks alive for about four
seconds and then looks wrong, because nothing agrees with anything. Worse, it
cannot exercise the insight engine at all: "fuel pressure drops during aggressive
acceleration" is only detectable if the fuel pressure is actually a function of
the throttle, and a cross-signal monitor has nothing to cross.

SCENARIOS, NOT A RANDOM WALK
----------------------------
A scenario is reproducible and a random walk is not. Every fault below crosses
its threshold on a schedule a person can wait out — the overheat reaches the warn
band in about twenty seconds and the critical band in about thirty, and then
HOLDS. A scenario that runs away to 400°F stops being a demonstration of
anything, and one that takes four minutes to show its point is one nobody watches
to the end.

WHAT IS NOT HERE
---------------
Judgement. Not a band, not a status word, not a severity. This module answers
"what is the sensor reading", exactly as a provider must, and config.py decides
what the number means. The one apparent exception — the comment saying the
overheat crosses the warn band at twenty seconds — is a note about the scenario
being useful, not a threshold this file holds.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import config

AMBIENT_F = 76.0
THERMOSTAT_F = 191.0
BARO_KPA = 99.0


def wander(sensor_id: str, t: float, amp: float) -> float:
    """Smooth, deterministic sensor noise.

    Two slow sines at frequencies derived from the channel's own name, so every
    channel wanders differently and none of them wander together. Deterministic
    because a mock you cannot get back to the state you were just looking at is
    useless for building a UI.

    crc32 rather than hash(): Python randomises string hashing per process, and
    a mock whose noise changes shape on every restart is a mock that cannot be
    used to reproduce anything.
    """
    if amp <= 0:
        return 0.0
    h = zlib.crc32(sensor_id.encode())
    f1 = 0.09 + (h % 41) / 400.0
    f2 = 0.021 + ((h >> 6) % 29) / 1100.0
    p1 = (h % 628) / 100.0
    p2 = ((h >> 11) % 628) / 100.0
    return amp * (0.62 * math.sin(t * f1 + p1) + 0.38 * math.sin(t * f2 + p2))


# --- driver inputs, per scenario -------------------------------------------

def _in_idle(t):
    return {"rpm": 880.0, "throttle": 0.8, "speed": 0.0}


def _in_warmup(t):
    # The first second and a half is the starter turning the engine over: low
    # rpm, no oil pressure yet, and the battery dragged down to ~9.8 V. It is
    # the one moment a healthy electrical system looks alarming, which is
    # exactly why the panel needs to be able to say CRANKING.
    if t < 1.6:
        return {"rpm": 255.0, "throttle": 0.0, "speed": 0.0}
    return {"rpm": 880.0 + 430.0 * math.exp(-(t - 1.6) / 45.0),
            "throttle": 1.1, "speed": 0.0}


def _in_cruise(t):
    return {"rpm": 1850.0, "throttle": 14.0, "speed": 62.0}


def _in_city(t):
    """Stop-start traffic. Load without airflow, which is the condition a
    cooling system is worst at and the one a contextual coolant monitor needs
    in order to have anything to compare a motorway against."""
    ph = t % 90.0
    if ph < 35.0:
        return {"rpm": 820.0, "throttle": 1.0, "speed": 0.0}
    if ph < 60.0:
        f = (ph - 35.0) / 25.0
        return {"rpm": 1200.0 + 900.0 * f, "throttle": 22.0, "speed": 14.0 + 16.0 * f}
    f = (ph - 60.0) / 30.0
    return {"rpm": 2100.0 - 1200.0 * f, "throttle": 3.0, "speed": 30.0 * (1.0 - f)}


def _in_aggressive(t):
    """A repeating pull: hard on the throttle, shift, coast, do it again.

    Sixteen seconds, so a whole cycle is visible without waiting and the
    conditioned baselines in insights.py collect both loaded and unloaded
    samples of every channel within a minute of the scenario being selected.
    """
    ph = t % 16.0
    if ph < 6.0:
        f = ph / 6.0
        return {"rpm": 2200.0 + 3500.0 * f,
                "throttle": 18.0 + 74.0 * min(1.0, f * 2.5),
                "speed": 35.0 + 55.0 * f}
    if ph < 8.0:
        return {"rpm": 3400.0, "throttle": 6.0, "speed": 90.0}
    f = (ph - 8.0) / 8.0
    return {"rpm": 3400.0 - 1650.0 * f, "throttle": 4.0, "speed": 90.0 - 52.0 * f}


def _in_restarts(t):
    """Repeated start events, for the start-voltage trend monitor.

    Ninety seconds apart, cranking for the first two of each, because the whole
    of that monitor's evidence is what the battery does in the second the
    starter is loading it — and one start event is not a trend.
    """
    ph = t % 90.0
    if ph < 2.0:
        return {"rpm": 240.0, "throttle": 0.0, "speed": 0.0}
    if ph < 8.0:
        return {"rpm": 1250.0, "throttle": 2.0, "speed": 0.0}
    if ph < 70.0:
        return {"rpm": 880.0, "throttle": 0.8, "speed": 0.0}
    return {"rpm": 0.0, "throttle": 0.0, "speed": 0.0}


@dataclass(frozen=True)
class Scenario:
    name: str
    label: str
    inputs: object
    # How long the engine had already been running when this scenario was
    # selected. Warm scenarios start warm; only the warm-up ones start cold.
    warm_offset_s: float = 900.0
    charging_fault: bool = False
    overheat: bool = False
    # Coolant climbing fast rather than climbing high. A rate-of-rise monitor
    # catches this a long way before any fixed ceiling does, which is the whole
    # reason that monitor exists.
    coolant_rise_f_per_min: float = 0.0
    dropout: Tuple[str, ...] = ()
    # Channels this vehicle does not have at all. Different from a dropout and
    # the difference matters: a dropout is a sensor that stopped answering, and
    # this is a PID the ECU never supported. One is a fault and one is a fact
    # about the car.
    unsupported: Tuple[str, ...] = ()
    # A channel that keeps reporting the value it last had. The nastiest sensor
    # failure there is, because every plausibility check passes.
    frozen: Tuple[str, ...] = ()
    # A channel reporting a number no sensor could produce.
    invalid: Tuple[str, ...] = ()
    # Nothing answers at all. An ECU that has stopped talking, as distinct from
    # a bridge that has stopped uploading.
    no_response: bool = False
    # Long-term fuel trim climbing away from zero, in percent per minute.
    trim_drift_pct_per_min: float = 0.0
    # Cranking voltage falling drive over drive — a battery on its way out that
    # every running-voltage reading says is fine.
    start_voltage_decay_v_per_min: float = 0.0


SCENARIOS: Tuple[Scenario, ...] = (
    Scenario("normal_idle",     "Normal Idle",    _in_idle),
    Scenario("warmup",          "Cold Warm-Up",   _in_warmup, warm_offset_s=0.0),
    Scenario("cruise",          "Cruise",         _in_cruise, warm_offset_s=1200.0),
    Scenario("city",            "City Traffic",   _in_city, warm_offset_s=1200.0),
    Scenario("aggressive",      "Aggressive",     _in_aggressive, warm_offset_s=1500.0),
    Scenario("charging_fault",  "Low Running Voltage", _in_idle, charging_fault=True),
    Scenario("overheating",     "High Coolant",   _in_cruise, warm_offset_s=1200.0,
             overheat=True),
    Scenario("coolant_rapid_rise", "Rapid Coolant Rise", _in_cruise,
             warm_offset_s=1200.0, coolant_rise_f_per_min=9.0),
    Scenario("start_voltage_decline", "Declining Start Voltage", _in_restarts,
             warm_offset_s=0.0, start_voltage_decay_v_per_min=0.22),
    Scenario("fuel_trim_drift", "Fuel Trim Drift", _in_idle,
             trim_drift_pct_per_min=1.4),
    Scenario("sensor_dropout",  "Sensor Dropout", _in_cruise, warm_offset_s=1200.0,
             dropout=("coolant_temp", "oil_pressure")),
    Scenario("unsupported_pids", "Unsupported PIDs", _in_cruise,
             warm_offset_s=1200.0, unsupported=("maf_gs", "ltft_b1", "stft_b1")),
    Scenario("frozen_signal",   "Frozen Signal",  _in_cruise, warm_offset_s=1200.0,
             frozen=("coolant_temp",)),
    Scenario("invalid_value",   "Invalid Value",  _in_cruise, warm_offset_s=1200.0,
             invalid=("coolant_temp",)),
    Scenario("ecu_no_response", "ECU No Response", _in_cruise,
             warm_offset_s=1200.0, no_response=True),
)

BY_NAME: Dict[str, Scenario] = {s.name: s for s in SCENARIOS}

DEFAULT = SCENARIOS[0].name


def resolve(name: str) -> str:
    return name if name in BY_NAME else DEFAULT


def catalogue() -> list:
    return [{"name": s.name, "label": s.label} for s in SCENARIOS]


# A value no sensor could produce, for the invalid-value scenario. Deliberately
# absurd rather than merely high: the point is to prove the range check fires,
# and a number that could plausibly be a real overheat would be testing the
# wrong thing.
_INVALID_VALUE = 8888.0


def sample(sc: Scenario, t: float,
           previous: Dict[str, Optional[float]] = None) -> Tuple[Dict[str, Optional[float]], Tuple[str, ...]]:
    """The whole engine at time `t` seconds into this scenario.

    -> (values by telemetry id, ids that did not answer)

    `previous` is the last sample, needed only by the frozen-signal scenario:
    a stuck sensor reports the value it last had, and that is impossible to
    model without knowing what that was.
    """
    previous = previous or {}
    run_s = sc.warm_offset_s + t

    inp = sc.inputs(t)
    rpm = inp["rpm"] + wander("rpm", t, 9.0)
    throttle = max(0.0, inp["throttle"] + wander("throttle_pct", t, 0.25))
    speed = max(0.0, inp["speed"] + wander("vehicle_speed", t,
                                           0.4 if inp["speed"] else 0.0))
    tp = min(1.0, throttle / 100.0)
    cranking = rpm < config.TELEMETRY_ENGINE_RUNNING_RPM and rpm > 50.0
    running = rpm >= config.TELEMETRY_ENGINE_RUNNING_RPM

    # Coolant: exponential approach to the thermostat, plus a little for load.
    # Time constant ~3.5 minutes, which is about what a small block with a 180°F
    # stat actually takes.
    load = tp
    warm_frac = 1.0 - math.exp(-max(0.0, run_s) / 210.0)
    coolant = AMBIENT_F + (THERMOSTAT_F - AMBIENT_F) * warm_frac + 7.0 * load
    if sc.overheat:
        # Something has stopped rejecting heat. Crosses the warn band at ~20 s
        # and the critical band at ~30 s, then holds.
        coolant += min(62.0, max(0.0, t * 1.6))
    if sc.coolant_rise_f_per_min:
        # Climbing fast rather than climbing high. Capped below the fixed
        # ceiling on purpose: this scenario exists to show that a rate-of-rise
        # monitor sees it long before any absolute limit does, and letting it
        # eventually trip the ceiling too would blur the demonstration.
        coolant += min(38.0, sc.coolant_rise_f_per_min * t / 60.0)
    coolant += wander("coolant_temp", t, 0.5)

    # Oil runs hotter than coolant under load and lags it badly. Twice the time
    # constant: oil is the last thing on the engine to come up to temperature.
    oil_target = coolant + 12.0 + 18.0 * load
    oil_frac = 1.0 - math.exp(-max(0.0, run_s) / 430.0)
    oil_temp = AMBIENT_F + (oil_target - AMBIENT_F) * oil_frac + wander("oil_temp", t, 0.6)

    # Oil pressure tracks rpm almost linearly until the pump reaches relief, and
    # falls off as the oil thins with heat.
    if running:
        oil_press = 34.0 + rpm * 0.0145 - max(0.0, oil_temp - 200.0) * 0.055
        oil_press = min(72.0, oil_press) + wander("oil_pressure", t, 0.7)
    else:
        # Not turning, or only being turned by the starter. Zero is the correct
        # reading and the band gate in config.py is what stops it being reported
        # as a critical fault.
        oil_press = 0.0

    # Manifold pressure follows the throttle plate, with a little more vacuum at
    # rpm when the plate is closed.
    map_kpa = (40.0 + 58.0 * (tp ** 0.55)
               - 3.5 * (1.0 - tp) * min(1.0, rpm / 2500.0))
    map_kpa = min(BARO_KPA, map_kpa) + wander("map_kpa", t, 0.6)
    if not running:
        map_kpa = BARO_KPA - 0.4

    # Commanded AFR out of the table: stoichiometric in closed loop, rich under
    # power. The wideband tracks it with a small lag and a little error, which is
    # what makes the two rows worth having separately — a wideband that has
    # drifted away from the target is the earliest sign of a fuelling problem
    # there is.
    enrich = max(0.0, (tp - 0.55)) / 0.45
    afr_target = 14.7 - 1.9 * enrich
    afr_wb = afr_target + 0.02 + wander("afr_wideband", t, 0.12) - 0.15 * enrich
    if not running:
        afr_target = None
        afr_wb = None

    # Returnless rail. Holds ~58 psi until the injectors ask for more than the
    # pump can keep up with, and then sags. This mock sags a little more than a
    # healthy system would — every band still passes, and the insight engine
    # notices anyway. That gap is the entire thesis of the feature.
    if running:
        fuel_press = 58.4 - 9.0 * max(0.0, tp - 0.45) / 0.55 + wander("fuel_pressure", t, 0.35)
    else:
        fuel_press = 0.0

    # Heat soak at rest, scrubbed away by airflow once moving.
    iat = AMBIENT_F + 2.5 + 9.0 * math.exp(-speed / 12.0) + wander("intake_air_temp", t, 0.5)

    # Calculated load, the way the standard defines it: how much air the engine
    # is drawing against how much it could draw at this rpm. On a naturally
    # aspirated engine that tracks manifold pressure almost exactly.
    if running:
        engine_load = max(0.0, min(100.0,
                                   100.0 * (map_kpa - 20.0) / (BARO_KPA - 20.0)))
        engine_load += wander("engine_load", t, 0.8)
        # Speed density: airflow rises with rpm and with manifold pressure, and
        # falls as the charge gets hotter and thinner.
        maf = (rpm * map_kpa * 4.0e-4) * (540.0 / (iat + 460.0))
        maf = max(0.0, maf + wander("maf_gs", t, 0.4))
    else:
        engine_load = None
        maf = None

    # Fuel trims. The short-term trim is the closed-loop correction the ECU is
    # applying right now and it oscillates by design — a wideband that holds
    # perfectly still is a wideband that has stopped working. The long-term trim
    # is what the ECU has LEARNED: it moves slowly, it survives a key cycle, and
    # it is the earliest number on the car that says something has changed.
    if running:
        # Open loop under power: the ECU stops trimming and follows the table,
        # so both trims go to zero. Reporting a live correction there would be
        # inventing one.
        open_loop = tp > 0.55
        drift = sc.trim_drift_pct_per_min * t / 60.0
        drift = max(-60.0, min(60.0, drift))
        ltft = 0.0 if open_loop else (1.8 + drift + wander("ltft_b1", t, 0.35))
        stft = 0.0 if open_loop else (wander("stft_b1", t, 3.2) - 0.4)
    else:
        ltft = stft = None

    # Charging system. Sags hard at crank, then holds regulated voltage with a
    # little ripple.
    if sc.charging_fault:
        # An alternator that has stopped keeping up and is slowly draining the
        # battery instead. Starts just under the warn floor and keeps going
        # down: out of band AND still heading the wrong way, which is the only
        # combination that earns the abnormal arrow.
        volts = 13.05 - 0.009 * t + wander("battery_voltage", t, 0.04)
        volts = max(11.4, volts)
    elif cranking:
        # The one channel where a fault is only visible for two seconds at a
        # time. A battery losing capacity holds its running voltage perfectly
        # and drops further every time the starter loads it, so the decay is
        # applied HERE and nowhere else — which is exactly why a start-voltage
        # monitor sees something no running-voltage band can.
        decay = sc.start_voltage_decay_v_per_min * t / 60.0
        volts = max(6.5, 9.8 - decay) + wander("battery_voltage", t, 0.15)
    elif running:
        volts = 14.2 - 0.35 * load + wander("battery_voltage", t, 0.06)
    else:
        volts = 12.6 + wander("battery_voltage", t, 0.03)

    values: Dict[str, Optional[float]] = {
        "battery_voltage": volts,
        "rpm": rpm if rpm > 50.0 else 0.0,
        "coolant_temp": coolant,
        "intake_air_temp": iat,
        "map_kpa": map_kpa,
        "maf_gs": maf,
        "throttle_pct": throttle,
        "engine_load": engine_load,
        "stft_b1": stft,
        "ltft_b1": ltft,
        "afr_target": afr_target,
        "afr_wideband": afr_wb,
        "fuel_pressure": fuel_press,
        "oil_pressure": oil_press,
        "oil_temp": oil_temp,
        "vehicle_speed": speed,
    }

    # A stuck sensor reports the value it last had. Applied after everything
    # else so the rest of the engine carries on around it, which is what makes
    # it hard: every plausibility check passes, the number is in band, and only
    # the fact that it has not MOVED gives it away.
    for sensor_id in sc.frozen:
        if previous.get(sensor_id) is not None:
            values[sensor_id] = previous[sensor_id]

    for sensor_id in sc.invalid:
        values[sensor_id] = _INVALID_VALUE

    # A PID this ECU never supported is absent, not null. The difference is the
    # whole of §32.7: absent means "this car does not have it" and belongs in
    # the capability report, null means "it should be there and is not".
    for sensor_id in sc.unsupported:
        values.pop(sensor_id, None)

    if sc.no_response:
        return {}, tuple(values)

    return values, tuple(sc.dropout)
