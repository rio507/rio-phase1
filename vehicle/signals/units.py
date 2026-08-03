"""units.py — conversion, and the one place it is allowed to happen.

RIO's canonical units are the ones its thresholds are already written in:
Fahrenheit, PSI, mph, volts, kPa. That is not a preference, it is a constraint.
Every band in config.TELEMETRY_BANDS is a Fahrenheit number, four weeks of
baselines in training_data/vehicle/baselines.json are keyed and valued in those
units, and vehicle_health_policy.spoken_temp says "degrees" with no scale
because the driver's car is in °F. Switching the canonical unit to Celsius would
move every threshold in the codebase and orphan the history, in exchange for
matching an example in a specification.

So OBD-II, which is natively metric, converts HERE — at the decoder boundary,
on the way in — and everything downstream sees the units it has always seen.

WHY CONVERSION MUST NOT HAPPEN DOWNSTREAM
-----------------------------------------
Because there is no way to tell, three layers later, whether a number has been
converted already. A coolant temperature of 96 is a healthy engine in Celsius
and a stone-cold one in Fahrenheit, and both are plausible readings. The only
defence is that exactly one place is allowed to convert and it is the place
where the source is still known.

Every function is a pure scalar transform and returns None for None. That
matters more than it looks: "the sensor did not answer" has to survive a unit
conversion, and a converter that turned None into 32.0 would invent a reading.
"""
from __future__ import annotations

from typing import Optional


def c_to_f(v: Optional[float]) -> Optional[float]:
    return None if v is None else v * 9.0 / 5.0 + 32.0


def f_to_c(v: Optional[float]) -> Optional[float]:
    return None if v is None else (v - 32.0) * 5.0 / 9.0


def kph_to_mph(v: Optional[float]) -> Optional[float]:
    return None if v is None else v * 0.621371


def mph_to_kph(v: Optional[float]) -> Optional[float]:
    return None if v is None else v / 0.621371


def kpa_to_psi(v: Optional[float]) -> Optional[float]:
    return None if v is None else v * 0.1450377


def psi_to_kpa(v: Optional[float]) -> Optional[float]:
    return None if v is None else v / 0.1450377


def identity(v: Optional[float]) -> Optional[float]:
    return v


# The canonical unit strings. Deliberately spelled out rather than abbreviated:
# these travel on the wire, into a log, and one day into somebody else's parser,
# and "c" versus "C" versus "°C" is a class of bug nobody should have to have.
FAHRENHEIT = "fahrenheit"
CELSIUS = "celsius"
PSI = "psi"
KPA = "kilopascal"
VOLT = "volt"
MPH = "mile_per_hour"
KPH = "kilometre_per_hour"
RPM = "revolutions_per_minute"
PERCENT = "percent"
RATIO = "ratio"
GRAMS_PER_SECOND = "gram_per_second"
DEGREE = "degree"

# How to get from a source unit to RIO's canonical one. A pair absent from here
# is a conversion nobody has defined, and convert() refuses rather than guessing
# — an unconverted number that silently passes for a converted one is the exact
# failure this module exists to prevent.
_CONVERSIONS = {
    (CELSIUS, FAHRENHEIT): c_to_f,
    (FAHRENHEIT, CELSIUS): f_to_c,
    (KPH, MPH): kph_to_mph,
    (MPH, KPH): mph_to_kph,
    (KPA, PSI): kpa_to_psi,
    (PSI, KPA): psi_to_kpa,
}


class UnitError(ValueError):
    """A conversion nobody has defined. Never guessed at."""


def convert(value: Optional[float], frm: str, to: str) -> Optional[float]:
    """-> the value in `to`. Raises UnitError if the pair is undefined."""
    if value is None:
        return None
    if frm == to:
        return value
    fn = _CONVERSIONS.get((frm, to))
    if fn is None:
        raise UnitError(f"no defined conversion from {frm!r} to {to!r}")
    return fn(value)


def can_convert(frm: str, to: str) -> bool:
    return frm == to or (frm, to) in _CONVERSIONS
