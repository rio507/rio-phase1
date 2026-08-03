"""registry.py — the canonical signal names, and the aliases to what RIO calls them.

    powertrain.engine.coolant_temperature   ←→   coolant_temp

WHY BOTH NAMES EXIST
--------------------
The dotted name is the contract. It is what a bridge sends, what the ingestion
API validates, what a future Jetson will send, and what makes "OBD-II PID 0105"
and "the Holley coolant channel" the same signal rather than two signals that
happen to be about the same water.

The flat name is what this codebase already runs on. `coolant_temp` is a key in
config.TELEMETRY_BANDS, in TELEMETRY_TREND_DELTA, in TELEMETRY_MODES, in
INSIGHTS_DEVIATION_DELTA, in INSIGHTS_DRIFT_DELTA, in CONDITIONS, and in four
weeks of daily baselines sitting in training_data/vehicle/baselines.json.

Renaming the internal ids to match the canonical ones would have been tidier for
about an hour. It would also have silently orphaned every one of those baselines
— the file is keyed by the old names, the drift detector reads it by key, and a
key that no longer matches does not raise, it just reports no history. Four weeks
of measured behaviour would have become "not enough data yet", and nothing would
have said so.

So: canonical on the wire, flat inside, and this file is the only place that
knows both. Every threshold in config.py stays where it is and means what it
meant.

UNITS FOLLOW THE SAME RULE
--------------------------
`unit` is RIO's canonical unit — Fahrenheit, PSI, mph — because that is what the
bands are written in. `source_unit` is what the standard actually sends, and it
is here so a decoder knows what it is converting FROM. OBD-II PID 0105 arrives
in Celsius and PID 010D in km/h; both are converted once, at the decoder, and
never again. See units.py on why that boundary is the only safe one.

A SIGNAL WITH NO INTERNAL ID
----------------------------
Perfectly legal, and it means "canonical, ingestible, stored, but not currently a
row on the dashboard". A signal with no PID is equally legal and means "this
vehicle class does not expose it over standard OBD-II" — oil pressure and a
wideband are Holley channels, and no amount of Mode 01 will produce them.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import units as U

# Groups, matching telemetry.py's so a canonical signal lands in the panel
# section a driver would look for it in.
ENGINE = "Engine"
ELECTRICAL = "Electrical"
INDUCTION = "Induction & Fuel"
DRIVELINE = "Driveline"
CHASSIS = "Chassis"


@dataclass(frozen=True)
class SignalSpec:
    """One canonical signal, and everything needed to move it about."""
    name: str                    # the dotted canonical name — the contract
    telemetry_id: str            # the flat internal id, "" if not on the panel
    unit: str                    # RIO's canonical unit for this signal
    group: str
    label: str
    obd_pid: str = ""            # "0105"; "" when standard OBD-II has no such PID
    source_unit: str = ""        # what the PID natively sends, if different
    holley: bool = False         # exposed by the Holley channel set
    # A plausible range in the CANONICAL unit. Not a warning band — those are
    # config.py's — but the bound outside which the number is not a measurement
    # at all and is marked invalid_range on ingestion.
    plausible: Optional[Tuple[float, float]] = None

    @property
    def needs_conversion(self) -> bool:
        return bool(self.source_unit) and self.source_unit != self.unit


_SPECS: Tuple[SignalSpec, ...] = (
    # --- the standard OBD-II set (§12) --------------------------------------
    SignalSpec("powertrain.engine.rpm", "rpm", U.RPM, DRIVELINE,
               "Engine RPM", obd_pid="010C", plausible=(0.0, 10000.0)),
    SignalSpec("vehicle.speed", "vehicle_speed", U.MPH, DRIVELINE,
               "Vehicle Speed", obd_pid="010D", source_unit=U.KPH,
               plausible=(0.0, 200.0)),
    SignalSpec("powertrain.engine.coolant_temperature", "coolant_temp",
               U.FAHRENHEIT, ENGINE, "Coolant Temp", obd_pid="0105",
               source_unit=U.CELSIUS, holley=True, plausible=(-40.0, 400.0)),
    SignalSpec("powertrain.engine.intake_air_temperature", "intake_air_temp",
               U.FAHRENHEIT, INDUCTION, "Intake Air Temp", obd_pid="010F",
               source_unit=U.CELSIUS, holley=True, plausible=(-40.0, 300.0)),
    SignalSpec("powertrain.engine.throttle_position", "throttle_pct",
               U.PERCENT, INDUCTION, "Throttle Position", obd_pid="0111",
               holley=True, plausible=(0.0, 100.0)),
    SignalSpec("powertrain.engine.calculated_load", "engine_load",
               U.PERCENT, INDUCTION, "Engine Load", obd_pid="0104",
               plausible=(0.0, 100.0)),
    SignalSpec("powertrain.engine.manifold_pressure", "map_kpa",
               U.KPA, INDUCTION, "MAP", obd_pid="010B", holley=True,
               plausible=(0.0, 300.0)),
    SignalSpec("powertrain.engine.mass_air_flow", "maf_gs",
               U.GRAMS_PER_SECOND, INDUCTION, "Mass Air Flow", obd_pid="0110",
               plausible=(0.0, 700.0)),
    SignalSpec("powertrain.fuel.short_term_trim_bank_1", "stft_b1",
               U.PERCENT, INDUCTION, "Short Term Fuel Trim", obd_pid="0106",
               plausible=(-100.0, 100.0)),
    SignalSpec("powertrain.fuel.long_term_trim_bank_1", "ltft_b1",
               U.PERCENT, INDUCTION, "Long Term Fuel Trim", obd_pid="0107",
               holley=True, plausible=(-100.0, 100.0)),
    SignalSpec("electrical.control_module_voltage", "battery_voltage",
               U.VOLT, ELECTRICAL, "Battery Voltage", obd_pid="0142",
               holley=True, plausible=(0.0, 30.0)),

    # --- channels standard OBD-II does not expose ---------------------------
    # Every one of these is a Holley channel or a RIO-installed sensor. A car
    # with only a standard OBD-II port will report them `unsupported`, and the
    # UI has to handle that gracefully rather than showing an empty row that
    # looks like a fault.
    SignalSpec("powertrain.engine.oil_pressure", "oil_pressure", U.PSI,
               ENGINE, "Oil Pressure", holley=True, plausible=(0.0, 150.0)),
    SignalSpec("powertrain.engine.oil_temperature", "oil_temp", U.FAHRENHEIT,
               ENGINE, "Oil Temp", holley=True, plausible=(-40.0, 400.0)),
    SignalSpec("powertrain.fuel.rail_pressure", "fuel_pressure", U.PSI,
               ENGINE, "Fuel Pressure", holley=True, plausible=(0.0, 150.0)),
    SignalSpec("powertrain.fuel.air_fuel_ratio_commanded", "afr_target",
               U.RATIO, INDUCTION, "Air Fuel Ratio", holley=True,
               plausible=(5.0, 25.0)),
    SignalSpec("powertrain.fuel.air_fuel_ratio_measured", "afr_wideband",
               U.RATIO, INDUCTION, "Wideband O₂", holley=True,
               plausible=(5.0, 25.0)),
)

# --- tire channels, generated --------------------------------------------
# Eight of them, and a typo in one corner's id is a row that silently never
# updates — the same reason telemetry.py generates rather than types them.
_CORNER_PATH = {"FL": "front_left", "FR": "front_right",
                "RL": "rear_left", "RR": "rear_right"}
_CORNER_LABEL = {"FL": "Front Left", "FR": "Front Right",
                 "RL": "Rear Left", "RR": "Rear Right"}

_TIRE_SPECS = tuple(
    s for corner, path in _CORNER_PATH.items() for s in (
        SignalSpec(f"chassis.tire.{path}.pressure", f"tire_pressure_{corner}",
                   U.PSI, CHASSIS, f"{_CORNER_LABEL[corner]} Pressure",
                   plausible=(0.0, 100.0)),
        SignalSpec(f"chassis.tire.{path}.temperature", f"tire_temp_{corner}",
                   U.FAHRENHEIT, CHASSIS, f"{_CORNER_LABEL[corner]} Temp",
                   plausible=(-40.0, 400.0)),
    )
)

SPECS: Tuple[SignalSpec, ...] = _SPECS + _TIRE_SPECS

BY_NAME: Dict[str, SignalSpec] = {s.name: s for s in SPECS}
BY_TELEMETRY_ID: Dict[str, SignalSpec] = {s.telemetry_id: s for s in SPECS
                                          if s.telemetry_id}
BY_PID: Dict[str, SignalSpec] = {s.obd_pid: s for s in SPECS if s.obd_pid}


# ---------------------------------------------------------------------------
# The lookups. Every one of them returns None rather than raising, because a
# signal nobody has registered is a thing that genuinely happens — a
# manufacturer PID, a Holley channel not yet decoded — and it must be storable
# and displayable rather than fatal.
# ---------------------------------------------------------------------------

def spec(name: str) -> Optional[SignalSpec]:
    """Canonical name -> its spec."""
    return BY_NAME.get(name)


def by_telemetry_id(telemetry_id: str) -> Optional[SignalSpec]:
    """Internal flat id -> its spec."""
    return BY_TELEMETRY_ID.get(telemetry_id)


def by_pid(pid: str) -> Optional[SignalSpec]:
    """"0105" -> its spec. The bridge's decoder table."""
    return BY_PID.get((pid or "").upper())


def canonical(telemetry_id: str) -> Optional[str]:
    """Internal flat id -> canonical dotted name."""
    s = BY_TELEMETRY_ID.get(telemetry_id)
    return s.name if s else None


def internal(name: str) -> Optional[str]:
    """Canonical dotted name -> internal flat id, or None if not on the panel."""
    s = BY_NAME.get(name)
    return (s.telemetry_id or None) if s else None


def is_known(name: str) -> bool:
    return name in BY_NAME


def obd_signals() -> List[SignalSpec]:
    """The standard-PID set, in the order a scheduler should first ask for it."""
    return [s for s in SPECS if s.obd_pid]


def holley_signals() -> List[SignalSpec]:
    return [s for s in SPECS if s.holley]


def in_range(name: str, value: Optional[float]) -> bool:
    """Is this a measurement at all?

    Not a warning band — those are config.py's and always will be. This is the
    bound outside which the number cannot be what it claims to be: a coolant
    temperature of 6000°F is a decode error, not an overheat, and calling it an
    overheat would put a fault on the dashboard for a bug in a formula.
    """
    if value is None:
        return False
    s = BY_NAME.get(name)
    if s is None or s.plausible is None:
        return True
    lo, hi = s.plausible
    return lo <= value <= hi


def to_canonical_unit(name: str, value: Optional[float],
                      from_unit: str = None) -> Optional[float]:
    """Convert an incoming value into RIO's unit for that signal.

    `from_unit` defaults to the signal's declared source unit, which is what a
    standard OBD-II decoder produces. Passing it explicitly is for a source that
    already converted, or for one whose units differ from the standard's.
    """
    s = BY_NAME.get(name)
    if s is None:
        return value
    frm = from_unit or s.source_unit or s.unit
    return U.convert(value, frm, s.unit)


def view() -> List[dict]:
    """The whole registry, for the service view and the capability report."""
    return [{
        "name": s.name,
        "telemetry_id": s.telemetry_id or None,
        "label": s.label,
        "unit": s.unit,
        "group": s.group,
        "obd_pid": s.obd_pid or None,
        "source_unit": s.source_unit or None,
        "holley": s.holley,
        "plausible_range": list(s.plausible) if s.plausible else None,
    } for s in SPECS]
