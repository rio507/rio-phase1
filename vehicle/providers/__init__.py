"""providers — the sources of vehicle data, all entering the same pipeline.

    ingested.py    the push→pull join. Everything that arrives over the network
                   — live OBD-II, a passive Holley capture, a replay, the
                   simulator — lands in its buffer and is read from there.

The provider CLASSES that telemetry.snapshot() calls live in telemetry.py,
beside MockHolleyProvider and TireTelemetryProvider, because that is where the
provider interface is defined and where the house style already puts them. What
lives here is the state they read: keeping the buffer on this side means
telemetry.py imports vehicle and vehicle never imports telemetry, so there is no
cycle at import time in a module app.py loads first.
"""
