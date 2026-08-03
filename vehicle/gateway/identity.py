"""identity.py — who is sending, and which car it is bolted to.

    {
      "gateway_id": "gateway_uuid",
      "device_name": "rio-obd-prototype-01",
      "bridge_version": "0.1.0",
      "hardware_type": "canable_2",
      "firmware_type": "candlelight",
      "vehicle_id": "vehicle_uuid"
    }

WHY A GATEWAY IS NOT A VEHICLE
------------------------------
They are separate ids because they have separate lifetimes. An adapter moves
between cars during development, a car outlives several adapters, and the whole
point of the prototype is that the temporary laptop bridge will one day be
replaced by a Jetson without the vehicle's history restarting. Folding them into
one identifier would make that migration a data migration.

WHY THE VIN IS NOT THE VEHICLE ID
---------------------------------
A VIN identifies a car to the world: it is on the windscreen, it is in insurance
records, and it is enough to look up an owner. `vehicle_id` is an opaque handle
that means something only inside RIO. The VIN, when Mode 09 returns one, is
stored against the vehicle and is never the key, never in a URL, and never in a
dashboard event. That separation is cheap to keep and impossible to add later.

CREDENTIALS ARE NOT IN HERE
---------------------------
This dataclass is the part of a gateway that is safe to log, display and put in
a heartbeat. The token lives in auth.py, is stored hashed, and never appears in
any structure this module produces — including its repr, which is why the fields
below are exactly these and no more.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import List, Optional

# Hardware and firmware RIO knows about. Unknown values are ACCEPTED and
# recorded: a gateway RIO does not recognise is still a gateway, and refusing
# registration over a spelling would make the field a gate rather than a label.
KNOWN_HARDWARE = ("canable_2", "canable_1", "canable_mkii", "jetson",
                  "simulator", "replay", "unknown")
KNOWN_FIRMWARE = ("candlelight", "slcan", "gs_usb", "none", "unknown")

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _.:-]{0,63}$")
_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,128}$")


@dataclass(frozen=True)
class GatewayIdentity:
    """Everything about a gateway that is safe to show."""
    gateway_id: str
    device_name: str
    vehicle_id: str
    bridge_version: str = "0.0.0"
    hardware_type: str = "unknown"
    firmware_type: str = "unknown"

    def to_dict(self) -> dict:
        return asdict(self)


def validate(device_name: str, vehicle_id: str,
             gateway_id: str = None) -> List[str]:
    """-> [] when a registration request is acceptable.

    Deliberately shallow. The only things worth refusing here are the ones that
    would poison a filename, a URL or a log line; everything else about a
    gateway is descriptive and a wrong value is a wrong label, not a breach.
    """
    errors = []
    if not device_name or not _NAME_RE.match(device_name):
        errors.append("device_name must be 1-64 chars of letters, digits, "
                      "space, dot, dash, colon or underscore")
    if not vehicle_id or not _ID_RE.match(vehicle_id):
        errors.append("vehicle_id must be 1-128 URL-safe characters")
    if gateway_id is not None and not _ID_RE.match(gateway_id):
        errors.append("gateway_id must be 1-128 URL-safe characters")
    return errors


def normalize_hardware(value: Optional[str]) -> str:
    v = (value or "").strip().lower().replace("-", "_")
    return v if v else "unknown"


def normalize_firmware(value: Optional[str]) -> str:
    v = (value or "").strip().lower().replace("-", "_")
    return v if v else "unknown"
