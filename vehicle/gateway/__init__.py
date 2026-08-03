"""gateway — who is sending data, and whether they are allowed to.

    identity.py   the §10.5 device identity, and why a gateway is not a vehicle
    auth.py       registration, tokens, heartbeats, rate limiting, idempotency

The temporary laptop bridge and the future Jetson use the same flow. That is the
whole reason this exists in the prototype rather than later: the migration to a
Jetson is supposed to be a change of hardware, and a gateway that authenticated
differently would make it a change of contract.
"""
