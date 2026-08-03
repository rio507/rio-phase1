"""auth.py — per-gateway credentials, heartbeats, and the rate limit.

There was no authentication anywhere in this repository before this file. Every
endpoint was open, which was correct for a dashboard talking to localhost and is
not correct for an endpoint a device in a car posts to over the internet. So this
is built from zero, and it is deliberately small: the prototype needs to know
which gateway is talking, to be able to stop trusting one, and to not be
trivially floodable. It does not need an identity provider.

WHAT IS HERE
------------
    registration    a bootstrap key admits a new gateway once
    tokens          per-gateway, stored HASHED, rotatable, revocable
    heartbeat       liveness plus the bridge's own view of itself
    rate limit      a token bucket per gateway
    idempotency     batch ids remembered, so a retry is not a second delivery

WHY THE TOKEN IS STORED HASHED
------------------------------
Because the gateway registry is a JSON file on disk beside the diagnostic
history, and a file that contains live credentials is a file that must never be
copied, attached to a bug report, or committed. Storing sha256(token) means the
file is safe to hand to somebody debugging a drive. The gateway keeps the only
copy of its own token, which is also why rotation issues a new one rather than
revealing the old.

WHY REGISTRATION NEEDS A BOOTSTRAP KEY
--------------------------------------
Otherwise the registration endpoint is an open door that mints credentials. The
key comes from the environment, never from the source tree, and when it is unset
registration is REFUSED rather than allowed — an unconfigured deployment that
accepts any device is worse than one that accepts none, because the first
failure is silent.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
Any route that asks a vehicle to do something. A gateway authenticates in order
to REPORT. There is no command channel, no queue of instructions for the bridge
to collect, and no endpoint that could become one — the read-only posture is
enforced by there being nothing to call, and vehicle/selftest.py asserts it.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import threading
import time
from typing import Dict, List, Optional, Tuple

import config

from .identity import (GatewayIdentity, normalize_firmware, normalize_hardware,
                       validate)

_lock = threading.RLock()

_PATH = os.path.join(config.VEHICLE_DIAG_DIR, "gateways.json")

# Bridge-reported states, echoed back on the dashboard. Free-form on purpose:
# these are the bridge's own view of itself and a cloud that enumerated them
# would have to be redeployed to learn a new one.
_HEARTBEAT_FIELDS = ("bridge_version", "can_interface", "can_state",
                     "network_state", "outbox_pending", "vehicle_state")


def _hash(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _now() -> float:
    return time.time()


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _blank() -> dict:
    return {"version": 1, "gateways": {}}


def _load() -> dict:
    try:
        with open(_PATH) as fh:
            data = json.load(fh)
    except Exception:
        return _blank()
    if not isinstance(data, dict) or "gateways" not in data:
        return _blank()
    return data


def _save(data: dict) -> None:
    os.makedirs(os.path.dirname(_PATH), exist_ok=True)
    tmp = _PATH + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, separators=(",", ":"))
    os.replace(tmp, _PATH)


_state = _load()

# Per-gateway token bucket, in memory only. A rate limit that survived a restart
# would punish a bridge for the cloud having been redeployed.
_buckets: Dict[str, Tuple[float, float]] = {}     # gateway_id -> (tokens, at)

# Batch ids already accepted, per gateway, newest last. In memory and bounded:
# the durable deduplication is by event_id in ingest.py, and this is only the
# cheap first line that lets an identical retry be acknowledged without being
# re-processed.
_seen_batches: Dict[str, List[str]] = {}
_SEEN_BATCH_MAX = 256


class AuthError(Exception):
    """Refused. The message is safe to return to the caller."""


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

def registration_enabled() -> bool:
    return bool(config.VEHICLE_GATEWAY_REGISTRATION_KEY)


def register(device_name: str, vehicle_id: str, registration_key: str,
             hardware_type: str = "unknown", firmware_type: str = "unknown",
             bridge_version: str = "0.0.0",
             gateway_id: str = None) -> dict:
    """Admit a gateway and issue it a token. -> {gateway_id, token, ...}

    The token is returned exactly once, here. Nothing else in the system can
    produce it again — see the module header — so a bridge that loses it must
    re-register or have its token rotated.
    """
    if not registration_enabled():
        raise AuthError("gateway registration is not configured on this server")
    if not registration_key or not hmac.compare_digest(
            str(registration_key), str(config.VEHICLE_GATEWAY_REGISTRATION_KEY)):
        raise AuthError("invalid registration key")

    errors = validate(device_name, vehicle_id, gateway_id)
    if errors:
        raise AuthError("; ".join(errors))

    with _lock:
        gid = gateway_id or ("gw_" + secrets.token_hex(8))
        token = secrets.token_urlsafe(32)
        now = _now()
        existing = _state["gateways"].get(gid) or {}
        _state["gateways"][gid] = {
            "gateway_id": gid,
            "device_name": device_name,
            "vehicle_id": vehicle_id,
            "hardware_type": normalize_hardware(hardware_type),
            "firmware_type": normalize_firmware(firmware_type),
            "bridge_version": bridge_version,
            "token_sha256": _hash(token),
            "token_issued_at": now,
            "registered_at": existing.get("registered_at", now),
            "revoked": False,
            "last_seen_at": None,
            "heartbeat": {},
            "batches_accepted": existing.get("batches_accepted", 0),
            "events_accepted": existing.get("events_accepted", 0),
            "events_rejected": existing.get("events_rejected", 0),
        }
        _save(_state)
        return {"gateway_id": gid, "token": token, "vehicle_id": vehicle_id,
                "registered_at": _state["gateways"][gid]["registered_at"],
                "token_issued_at": now}


def rotate_token(gateway_id: str) -> dict:
    """Issue a new token and invalidate the old one immediately."""
    with _lock:
        rec = _state["gateways"].get(gateway_id)
        if rec is None:
            raise AuthError("unknown gateway")
        token = secrets.token_urlsafe(32)
        rec["token_sha256"] = _hash(token)
        rec["token_issued_at"] = _now()
        _save(_state)
        return {"gateway_id": gateway_id, "token": token,
                "token_issued_at": rec["token_issued_at"]}


def revoke(gateway_id: str) -> bool:
    """Stop trusting a gateway without forgetting it existed.

    Revoked rather than deleted, on purpose: the drives it uploaded are still
    the vehicle's history, and a gateway id that vanished would leave events
    attributed to nothing.
    """
    with _lock:
        rec = _state["gateways"].get(gateway_id)
        if rec is None:
            return False
        rec["revoked"] = True
        _save(_state)
        return True


# ---------------------------------------------------------------------------
# Authentication
# ---------------------------------------------------------------------------

def authenticate(gateway_id: str, token: str) -> dict:
    """-> the gateway record. Raises AuthError on anything wrong.

    One error message for "no such gateway" and "wrong token", deliberately: a
    caller that can tell them apart can enumerate gateway ids.
    """
    if not gateway_id or not token:
        raise AuthError("missing gateway credentials")
    with _lock:
        rec = _state["gateways"].get(gateway_id)
        if rec is None or not hmac.compare_digest(rec.get("token_sha256", ""),
                                                  _hash(token)):
            raise AuthError("unknown gateway or invalid token")
        if rec.get("revoked"):
            raise AuthError("this gateway has been revoked")
        return rec


def authorize_vehicle(rec: dict, vehicle_id: str) -> None:
    """A gateway may only speak about the vehicle it was registered against."""
    if vehicle_id and rec.get("vehicle_id") != vehicle_id:
        raise AuthError("this gateway is not registered to that vehicle")


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

def check_rate(gateway_id: str, cost: float = 1.0, now: float = None) -> None:
    """A token bucket per gateway. Raises AuthError when it is empty.

    Sized in BATCHES, not events: a bridge uploading a backlog after a tunnel
    sends few large batches, and a per-event limit would punish exactly the
    recovery behaviour the outbox exists to produce.
    """
    now = _now() if now is None else now
    rate = float(config.VEHICLE_INGEST_RATE_PER_MIN) / 60.0
    burst = float(config.VEHICLE_INGEST_BURST)
    with _lock:
        tokens, at = _buckets.get(gateway_id, (burst, now))
        tokens = min(burst, tokens + (now - at) * rate)
        if tokens < cost:
            _buckets[gateway_id] = (tokens, now)
            raise AuthError("rate limit exceeded for this gateway")
        _buckets[gateway_id] = (tokens - cost, now)


def seen_batch(gateway_id: str, batch_id: str) -> bool:
    """Has this exact batch already been accepted? Cheap idempotency.

    The durable guarantee is per-event deduplication in ingest.py; this only
    lets an identical retry be acknowledged without being re-walked.
    """
    if not batch_id:
        return False
    with _lock:
        return batch_id in _seen_batches.get(gateway_id, ())


def note_batch(gateway_id: str, batch_id: str, accepted: int = 0,
               rejected: int = 0) -> None:
    with _lock:
        if batch_id:
            ring = _seen_batches.setdefault(gateway_id, [])
            ring.append(batch_id)
            if len(ring) > _SEEN_BATCH_MAX:
                del ring[:len(ring) - _SEEN_BATCH_MAX]
        rec = _state["gateways"].get(gateway_id)
        if rec is not None:
            rec["batches_accepted"] = rec.get("batches_accepted", 0) + 1
            rec["events_accepted"] = rec.get("events_accepted", 0) + accepted
            rec["events_rejected"] = rec.get("events_rejected", 0) + rejected


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

def heartbeat(gateway_id: str, payload: dict, now: float = None) -> dict:
    """Record a bridge's liveness and its own view of itself.

    The bridge reports what it BELIEVES — its CAN state, its network state, how
    many events are stuck in its outbox. The cloud records that verbatim and
    forms its own opinion separately from `last_seen_at`, because a bridge that
    thinks its network is fine and has not been heard from in five minutes is
    exactly the disagreement worth surfacing.
    """
    now = _now() if now is None else now
    with _lock:
        rec = _state["gateways"].get(gateway_id)
        if rec is None:
            raise AuthError("unknown gateway")
        rec["last_seen_at"] = now
        hb = {k: payload.get(k) for k in _HEARTBEAT_FIELDS if k in payload}
        hb["observed_at"] = payload.get("observed_at")
        hb["at"] = now
        rec["heartbeat"] = hb
        if payload.get("bridge_version"):
            rec["bridge_version"] = payload["bridge_version"]
        _save(_state)
        return public(rec, now)


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

def public(rec: dict, now: float = None) -> dict:
    """A gateway as the dashboard sees it. Never the token, never its hash."""
    now = _now() if now is None else now
    last = rec.get("last_seen_at")
    idle = None if last is None else round(now - last, 1)
    stale = idle is None or idle > config.VEHICLE_GATEWAY_STALE_S
    return {
        "gateway_id": rec.get("gateway_id"),
        "device_name": rec.get("device_name"),
        "vehicle_id": rec.get("vehicle_id"),
        "hardware_type": rec.get("hardware_type"),
        "firmware_type": rec.get("firmware_type"),
        "bridge_version": rec.get("bridge_version"),
        "registered_at": rec.get("registered_at"),
        "revoked": bool(rec.get("revoked")),
        "last_seen_at": last,
        "idle_s": idle,
        # The cloud's own opinion, formed from when it last heard rather than
        # from what the bridge claims about itself.
        "link": "lost" if stale else "connected",
        "heartbeat": rec.get("heartbeat") or {},
        "batches_accepted": rec.get("batches_accepted", 0),
        "events_accepted": rec.get("events_accepted", 0),
        "events_rejected": rec.get("events_rejected", 0),
    }


def gateways(vehicle_id: str = None) -> List[dict]:
    now = _now()
    with _lock:
        return [public(r, now) for r in _state["gateways"].values()
                if vehicle_id is None or r.get("vehicle_id") == vehicle_id]


def get(gateway_id: str) -> Optional[dict]:
    with _lock:
        rec = _state["gateways"].get(gateway_id)
        return public(rec) if rec else None


def reset_for_test(path: str = None) -> None:
    """Point the registry somewhere disposable, and forget every gateway.

    Tests only, and explicit: a suite that could mint credentials into the real
    registry would be a suite that leaves a working key behind when it fails.
    """
    global _PATH, _state
    with _lock:
        _PATH = path or os.path.join(config.VEHICLE_DIAG_DIR, "gateways.json")
        _state = _blank()
        _buckets.clear()
        _seen_batches.clear()
