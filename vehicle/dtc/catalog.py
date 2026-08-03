"""catalog.py — decoding a diagnostic trouble code, and saying what it means.

Two jobs that are easy to confuse:

    PARSING       two bytes off the bus -> "P0171". Purely mechanical, defined
                  by SAE J2012, and complete: every possible pair of bytes
                  decodes to some code.

    EXPLAINING    "P0171" -> "the mixture is lean on bank 1". Partial, always,
                  and it must stay honest about being partial.

A catalogue that conflated them would drop codes it had no definition for, and
the codes it had no definition for are disproportionately the interesting ones —
manufacturer-specific faults are exactly where a car does something unusual.

SO AN UNKNOWN CODE IS A FIRST-CLASS CODE
----------------------------------------
It parses, it stores, it appears on the dashboard, it goes in the report, it has
a lifecycle, and it says "RIO does not have a definition for this code" instead
of a description. What it does not do is disappear. §15 asks for this in one
line and it is the single most likely thing to be got wrong, because dropping it
makes every other part of the feature simpler.

WHAT A CODE MEANS, AND WHAT IT DOES NOT
---------------------------------------
A DTC names a CONDITION the ECU observed. It does not name the component that
failed, and the difference is the whole of §16.2:

    P0171 means the mixture is lean on bank 1.
    P0171 does NOT mean the MAF is bad.

So `possible_causes` is a list, it is presented as a list, and nothing in this
codebase is allowed to promote one of its entries to a fact. `confirmed_cause`
exists on the record and is only ever filled in by a person — a mechanic, or a
repair that was validated. See vehicle/signals/provenance.py.

SEVERITY IS REVIEWED, NOT DERIVED
---------------------------------
There is no formula from code number to severity, and any that looked like one
would be wrong: P0300 (random misfire) and P0303 (cylinder 3 misfire) are
adjacent numbers and quite different conversations. Every severity below is a
judgement someone made, written down, and reviewable.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# Severity, in the same vocabulary the rest of the health layer speaks. §17.7's
# four labels map onto it exactly.
INFORMATION = "information"
WATCH = "watch"
WARNING = "warning"
URGENT = "urgent"

SEVERITY_RANK = {INFORMATION: 1, WATCH: 2, WARNING: 3, URGENT: 4}

SEVERITY_LABEL = {INFORMATION: "Information", WATCH: "Watch",
                  WARNING: "Warning", URGENT: "Urgent"}

# §17.7 gives code cards their own four words, and the conversation layer already
# had four of its own. Two independent ladders would be one too many: a code that
# reads "Urgent" on the dashboard and arrives at the announcement policy as
# "advisory" is a system disagreeing with itself in front of the driver.
#
# So this is a TRANSLATION, not a second ladder. The card shows §17.7's word; the
# health layer is handed its own, through here; and vehicle/selftest.py asserts
# the map is total, so a severity added on either side cannot silently fall
# through to a default.
HEALTH_SEVERITY = {
    INFORMATION: "informational",
    WATCH: "advisory",
    WARNING: "warning",
    URGENT: "critical",
}


def health_severity(severity: str) -> str:
    """A code's severity in the conversation layer's vocabulary."""
    return HEALTH_SEVERITY.get(severity, "informational")

# --- systems (§15.3) --------------------------------------------------------
FUEL_AIR = "fuel_and_air_metering"
FUEL_AIR_INJ = "fuel_and_air_metering_injector"
IGNITION = "ignition_or_misfire"
EMISSIONS = "auxiliary_emission_controls"
SPEED_IDLE = "vehicle_speed_and_idle_control"
COMPUTER = "computer_output_circuit"
TRANSMISSION = "transmission"
COOLING = "cooling_system"
ELECTRICAL = "electrical_and_charging"
NETWORK = "network_communication"
UNKNOWN_SYSTEM = "unknown"

SYSTEM_LABEL = {
    FUEL_AIR: "Fuel and air metering",
    FUEL_AIR_INJ: "Fuel and air metering (injector circuit)",
    IGNITION: "Ignition system or misfire",
    EMISSIONS: "Auxiliary emission controls",
    SPEED_IDLE: "Vehicle speed and idle control",
    COMPUTER: "Computer output circuit",
    TRANSMISSION: "Transmission",
    COOLING: "Cooling system",
    ELECTRICAL: "Electrical and charging",
    NETWORK: "Network communication",
    UNKNOWN_SYSTEM: "Unknown",
}


# ---------------------------------------------------------------------------
# Parsing — SAE J2012, and complete
# ---------------------------------------------------------------------------

_LETTER = ("P", "C", "B", "U")


def decode(byte_a: int, byte_b: int) -> Optional[str]:
    """Two bytes off the bus -> a code string, or None for the no-code padding.

    0x0000 is not a code: Mode 03 pads its response to a whole number of frames
    with zeroes, and reading those as "P0000" would invent a fault on a healthy
    car every time it was asked.
    """
    if byte_a == 0 and byte_b == 0:
        return None
    letter = _LETTER[(byte_a >> 6) & 0x03]
    first = (byte_a >> 4) & 0x03
    return f"{letter}{first}{byte_a & 0x0F:X}{byte_b >> 4:X}{byte_b & 0x0F:X}"


def encode(code: str) -> Optional[Tuple[int, int]]:
    """A code string -> the two bytes an ECU would send. For the mock and tests."""
    code = (code or "").strip().upper()
    if len(code) != 5 or code[0] not in _LETTER:
        return None
    try:
        first = int(code[1])
        rest = int(code[2:], 16)
    except ValueError:
        return None
    if not (0 <= first <= 3):
        return None
    byte_a = (_LETTER.index(code[0]) << 6) | (first << 4) | ((rest >> 8) & 0x0F)
    return byte_a, rest & 0xFF


def is_manufacturer_specific(code: str) -> bool:
    """P1xxx, P3xxx and their C/B/U equivalents are the manufacturer's.

    Worth knowing because it changes what RIO may honestly say: a standard code
    has a published meaning, and a manufacturer code means whatever that
    manufacturer decided. Presenting a guess at one as a definition would be
    inventing a fault.
    """
    code = (code or "").strip().upper()
    return len(code) == 5 and code[0] in _LETTER and code[1] in ("1", "3")


def is_valid(code: str) -> bool:
    code = (code or "").strip().upper()
    if len(code) != 5 or code[0] not in _LETTER or code[1] not in "0123":
        return False
    try:
        int(code[2:], 16)
    except ValueError:
        return False
    return True


# ---------------------------------------------------------------------------
# Explaining — partial, and honest about it
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CodeDefinition:
    """What is known about one code, and who it is being said to.

    `driver_explanation` and `technician_detail` are separate fields rather than
    one field written twice, for the reason tire_diag/codes.py keeps
    driver_term apart from technician_description: they are answers to different
    questions. A driver asks what this means for their day; a mechanic asks what
    the ECU measured and where to put a meter.
    """
    code: str
    description: str                     # the standard's own wording
    system: str
    severity: str
    driver_explanation: str
    technician_detail: str
    possible_causes: Tuple[str, ...]
    # Canonical signal names worth watching while this code is live. §18's
    # related-signal profile: when a code appears, these are what RIO starts
    # paying attention to.
    related_signals: Tuple[str, ...] = ()

    @property
    def system_label(self) -> str:
        return SYSTEM_LABEL.get(self.system, self.system)

    @property
    def severity_label(self) -> str:
        return SEVERITY_LABEL.get(self.severity, self.severity)


_LEAN_SIGNALS = ("powertrain.fuel.short_term_trim_bank_1",
                 "powertrain.fuel.long_term_trim_bank_1",
                 "powertrain.engine.manifold_pressure",
                 "powertrain.engine.mass_air_flow",
                 "powertrain.engine.rpm",
                 "powertrain.engine.calculated_load",
                 "powertrain.fuel.air_fuel_ratio_measured")

_COOLANT_SIGNALS = ("powertrain.engine.coolant_temperature",
                    "vehicle.speed",
                    "powertrain.engine.calculated_load",
                    "powertrain.engine.intake_air_temperature")

_MISFIRE_SIGNALS = ("powertrain.engine.rpm",
                    "powertrain.engine.calculated_load",
                    "powertrain.fuel.short_term_trim_bank_1",
                    "powertrain.fuel.long_term_trim_bank_1",
                    "electrical.control_module_voltage")

_VOLTAGE_SIGNALS = ("electrical.control_module_voltage",
                    "powertrain.engine.rpm")

_AIRFLOW_SIGNALS = ("powertrain.engine.mass_air_flow",
                    "powertrain.engine.manifold_pressure",
                    "powertrain.engine.calculated_load",
                    "powertrain.engine.throttle_position")


# (code, description, system, severity, driver, technician, causes, signals)
_DEFINITIONS = (
    # --- fuel trim and mixture ---------------------------------------------
    ("P0171", "System too lean, bank 1", FUEL_AIR, WATCH,
     "The engine is getting more air than the computer expected, so it is "
     "adding fuel to compensate.",
     "Long-term fuel trim on bank 1 has exceeded the lean threshold under the "
     "monitor's enabling conditions.",
     ("Unmetered air entering after the mass-air-flow sensor",
      "Vacuum or intake leak", "Mass-air-flow sensor reading low",
      "Fuel delivery falling short — pump, filter or injectors",
      "Exhaust leak ahead of the upstream oxygen sensor"),
     _LEAN_SIGNALS),
    ("P0174", "System too lean, bank 2", FUEL_AIR, WATCH,
     "The same lean condition as P0171, on the engine's other bank.",
     "Long-term fuel trim on bank 2 has exceeded the lean threshold.",
     ("Unmetered air entering after the mass-air-flow sensor",
      "Vacuum or intake leak", "Mass-air-flow sensor reading low",
      "Fuel delivery falling short"),
     _LEAN_SIGNALS),
    ("P0172", "System too rich, bank 1", FUEL_AIR, WATCH,
     "The engine is getting more fuel than it needs, so the computer is "
     "pulling fuel back.",
     "Long-term fuel trim on bank 1 has exceeded the rich threshold.",
     ("Leaking injector", "Fuel pressure too high",
      "Mass-air-flow sensor reading high", "Restricted air intake",
      "Failing oxygen sensor"),
     _LEAN_SIGNALS),
    ("P0175", "System too rich, bank 2", FUEL_AIR, WATCH,
     "The same rich condition as P0172, on the engine's other bank.",
     "Long-term fuel trim on bank 2 has exceeded the rich threshold.",
     ("Leaking injector", "Fuel pressure too high",
      "Mass-air-flow sensor reading high"),
     _LEAN_SIGNALS),

    # --- misfire -------------------------------------------------------------
    ("P0300", "Random or multiple cylinder misfire detected", IGNITION, WARNING,
     "The engine is not firing cleanly, and it is not confined to one "
     "cylinder.",
     "Misfire counts exceeded the threshold across more than one cylinder "
     "within the monitor's sample window.",
     ("Ignition components — plugs, coils or leads",
      "Fuel delivery falling short across the engine",
      "Vacuum leak", "Low compression", "Failing crankshaft position sensor"),
     _MISFIRE_SIGNALS),
    ("P0301", "Cylinder 1 misfire detected", IGNITION, WARNING,
     "One cylinder is not firing cleanly.",
     "Misfire counts on cylinder 1 exceeded the threshold.",
     ("Plug, coil or lead on that cylinder", "Injector on that cylinder",
      "Low compression on that cylinder", "Localised vacuum leak"),
     _MISFIRE_SIGNALS),
    ("P0302", "Cylinder 2 misfire detected", IGNITION, WARNING,
     "One cylinder is not firing cleanly.",
     "Misfire counts on cylinder 2 exceeded the threshold.",
     ("Plug, coil or lead on that cylinder", "Injector on that cylinder",
      "Low compression on that cylinder"),
     _MISFIRE_SIGNALS),
    ("P0303", "Cylinder 3 misfire detected", IGNITION, WARNING,
     "One cylinder is not firing cleanly.",
     "Misfire counts on cylinder 3 exceeded the threshold.",
     ("Plug, coil or lead on that cylinder", "Injector on that cylinder",
      "Low compression on that cylinder"),
     _MISFIRE_SIGNALS),
    ("P0304", "Cylinder 4 misfire detected", IGNITION, WARNING,
     "One cylinder is not firing cleanly.",
     "Misfire counts on cylinder 4 exceeded the threshold.",
     ("Plug, coil or lead on that cylinder", "Injector on that cylinder",
      "Low compression on that cylinder"),
     _MISFIRE_SIGNALS),

    # --- cooling -------------------------------------------------------------
    ("P0128", "Coolant thermostat below regulating temperature", COOLING, WATCH,
     "The engine is taking longer to warm up than it should.",
     "Coolant temperature failed to reach the thermostat's regulating "
     "temperature within the expected time and load.",
     ("Thermostat stuck open", "Coolant temperature sensor reading low",
      "Coolant level low"),
     _COOLANT_SIGNALS),
    ("P0217", "Engine over-temperature condition", COOLING, URGENT,
     "The engine has run hotter than it is designed to.",
     "Coolant temperature exceeded the over-temperature threshold.",
     ("Coolant level low", "Failing water pump", "Thermostat stuck closed",
      "Blocked radiator or failed cooling fan", "Head gasket failure"),
     _COOLANT_SIGNALS),
    ("P0116", "Engine coolant temperature circuit range or performance",
     COOLING, WATCH,
     "The coolant temperature reading is not behaving the way the computer "
     "expects it to.",
     "Coolant temperature signal out of expected range or rate for the "
     "operating conditions.",
     ("Coolant temperature sensor drifting", "Thermostat fault",
      "Wiring or connector fault at the sensor"),
     _COOLANT_SIGNALS),
    ("P0118", "Engine coolant temperature circuit high", COOLING, WARNING,
     "The coolant temperature signal is reading at its electrical limit, "
     "which usually means a wiring fault rather than a hot engine.",
     "Coolant temperature circuit voltage high — open circuit or sensor "
     "failure.",
     ("Open circuit in the sensor wiring", "Failed coolant temperature sensor",
      "Connector corrosion"),
     _COOLANT_SIGNALS),

    # --- airflow and pressure ------------------------------------------------
    ("P0101", "Mass air flow circuit range or performance", FUEL_AIR, WATCH,
     "The airflow reading does not agree with what the rest of the engine is "
     "doing.",
     "Mass-air-flow signal outside the expected range for the current rpm, "
     "load and throttle position.",
     ("Dirty or failing mass-air-flow sensor", "Intake leak after the sensor",
      "Restricted air filter", "Exhaust restriction"),
     _AIRFLOW_SIGNALS),
    ("P0102", "Mass air flow circuit low input", FUEL_AIR, WATCH,
     "The airflow sensor is reading at or near zero.",
     "Mass-air-flow circuit voltage low.",
     ("Disconnected or failed mass-air-flow sensor", "Wiring fault",
      "Severely restricted intake"),
     _AIRFLOW_SIGNALS),
    ("P0106", "Manifold pressure circuit range or performance", FUEL_AIR, WATCH,
     "The manifold pressure reading does not agree with the throttle "
     "position.",
     "MAP signal outside the expected range for the current throttle and rpm.",
     ("Failing MAP sensor", "Vacuum line to the sensor split or blocked",
      "Intake leak"),
     _AIRFLOW_SIGNALS),

    # --- electrical ----------------------------------------------------------
    ("P0562", "System voltage low", ELECTRICAL, WARNING,
     "The car's electrical system is running below the voltage it needs.",
     "Control module supply voltage below threshold with the engine running.",
     ("Alternator not charging", "Failing battery",
      "Loose or corroded battery connection", "Slipping or broken drive belt"),
     _VOLTAGE_SIGNALS),
    ("P0563", "System voltage high", ELECTRICAL, WARNING,
     "The charging system is putting out more voltage than it should, which "
     "damages electronics and boils the battery.",
     "Control module supply voltage above threshold.",
     ("Failed voltage regulator", "Alternator overcharging",
      "Poor earth connection"),
     _VOLTAGE_SIGNALS),
    ("P0620", "Generator control circuit", ELECTRICAL, WATCH,
     "The computer is not getting the response it expects from the charging "
     "system.",
     "Generator control circuit fault.",
     ("Alternator control circuit fault", "Wiring or connector fault",
      "Failing alternator"),
     _VOLTAGE_SIGNALS),

    # --- oxygen sensors ------------------------------------------------------
    ("P0131", "O2 sensor circuit low voltage, bank 1 sensor 1", FUEL_AIR, WATCH,
     "The upstream oxygen sensor is reading lean and not moving the way a "
     "healthy one does.",
     "Upstream O2 sensor voltage below threshold and not switching.",
     ("Failing oxygen sensor", "Exhaust leak ahead of the sensor",
      "Genuinely lean mixture", "Wiring fault"),
     _LEAN_SIGNALS),
    ("P0135", "O2 sensor heater circuit, bank 1 sensor 1", FUEL_AIR, INFORMATION,
     "The heater inside the upstream oxygen sensor is not working, so it takes "
     "longer than it should to start reading properly.",
     "O2 heater circuit resistance or current outside expected range.",
     ("Failed sensor heater element", "Blown fuse", "Wiring or connector fault"),
     _LEAN_SIGNALS),

    # --- idle and throttle ---------------------------------------------------
    ("P0505", "Idle air control system", SPEED_IDLE, WATCH,
     "The engine is having trouble holding a steady idle.",
     "Idle control system unable to maintain target idle speed.",
     ("Carbon build-up in the throttle body or idle valve",
      "Vacuum leak", "Failing idle air control valve"),
     ("powertrain.engine.rpm", "powertrain.engine.throttle_position",
      "powertrain.engine.manifold_pressure")),
    ("P0122", "Throttle position sensor circuit low", SPEED_IDLE, WARNING,
     "The throttle position signal is reading at its electrical limit.",
     "Throttle position circuit voltage low.",
     ("Failing throttle position sensor", "Wiring fault", "Connector corrosion"),
     ("powertrain.engine.throttle_position", "powertrain.engine.manifold_pressure",
      "powertrain.engine.calculated_load")),

    # --- network -------------------------------------------------------------
    ("U0100", "Lost communication with ECM/PCM", NETWORK, WARNING,
     "One of the car's computers stopped talking to the others.",
     "No CAN messages received from the engine control module within the "
     "expected interval.",
     ("Wiring or connector fault on the CAN bus", "Module power or earth fault",
      "Failed module"),
     ()),
)


def _build() -> Dict[str, CodeDefinition]:
    out = {}
    for code, desc, system, sev, driver, tech, causes, signals in _DEFINITIONS:
        out[code] = CodeDefinition(
            code=code, description=desc, system=system, severity=sev,
            driver_explanation=driver, technician_detail=tech,
            possible_causes=tuple(causes), related_signals=tuple(signals))
    return out


DEFINITIONS: Dict[str, CodeDefinition] = _build()


def unknown_definition(code: str) -> CodeDefinition:
    """A code RIO has no definition for, described honestly.

    Not a placeholder and not an error. A manufacturer-specific code is the
    ECU's own vocabulary, and the truthful thing to say about one is that the
    vehicle reported it and RIO cannot translate it — never a guess dressed as a
    definition.
    """
    manufacturer = is_manufacturer_specific(code)
    return CodeDefinition(
        code=code,
        description=("Manufacturer-specific code" if manufacturer
                     else "Unrecognised diagnostic trouble code"),
        system=UNKNOWN_SYSTEM,
        # Watch, not information: RIO does not know what it means, and "we do
        # not know" is a reason to keep looking rather than a reason to relax.
        severity=WATCH,
        driver_explanation=(
            "Your vehicle reported this code. It is specific to the "
            "manufacturer and RIO does not have a definition for it — a "
            "workshop with the right software will."
            if manufacturer else
            "Your vehicle reported this code and RIO does not have a "
            "definition for it."),
        technician_detail=("Manufacturer-specific code; refer to the "
                           "manufacturer's service information."
                           if manufacturer else
                           "No catalogue entry. Stored verbatim as reported."),
        possible_causes=(),
        related_signals=())


def get(code: str) -> CodeDefinition:
    """Always returns a definition. See unknown_definition."""
    code = (code or "").strip().upper()
    return DEFINITIONS.get(code) or unknown_definition(code)


def is_known(code: str) -> bool:
    return (code or "").strip().upper() in DEFINITIONS


def related_signals(code: str) -> Tuple[str, ...]:
    """§18's related-signal profile: what to watch while this code is live."""
    return get(code).related_signals


def view() -> List[dict]:
    """The whole catalogue, for the service view."""
    return [{
        "code": d.code,
        "description": d.description,
        "system": d.system,
        "system_label": d.system_label,
        "severity": d.severity,
        "severity_label": d.severity_label,
        "driver_explanation": d.driver_explanation,
        "technician_detail": d.technician_detail,
        "possible_causes": list(d.possible_causes),
        "related_signals": list(d.related_signals),
    } for d in sorted(DEFINITIONS.values(), key=lambda x: x.code)]
