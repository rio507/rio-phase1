"""vehicle — the cloud-side vehicle data layer.

Everything between a source of vehicle data and the health intelligence that
already exists. It is called `vehicle` and not `vehicle_health` because
vehicle_health.py is a module at the repository root and has been since the
conversation layer was built; a package of that name would shadow it and break
app.py, llm_interface.py and both selftests on import.

    bridge / simulator / replay
              ↓  canonical telemetry events (signals/schema.py)
        POST /api/v1/vehicle-telemetry/batches
              ↓  ingest.py — authenticate, validate, deduplicate
        providers/ingested.py
              ↓  the SAME TelemetryProvider interface the mock uses
        telemetry.snapshot()
              ↓
        insights.py, vehicle_health.py, the dashboard, RIO

THE ONE THING THIS PACKAGE IS FOR
---------------------------------
Making "the health engine does not care where a signal came from" true rather
than aspirational. The existing pipeline is PULL: the browser polls
/vehicle/telemetry, telemetry.snapshot() asks each provider to read(), and the
insight engine is fed as a side effect. A bridge in a car is PUSH. Those are
opposite directions, and nothing in the repository connected them.

providers/ingested.py is the join. Pushed events land in a buffer; the buffer
answers read() when the poll comes round. Live OBD-II, a Holley capture, a
recorded replay and the simulator all enter through it, which means none of them
gets its own path into the health layer and none of them can drift from the
others.

WHAT IS DELIBERATELY NOT HERE
-----------------------------
Interpretation. Not a threshold, not a band, not a severity, not a sentence.
Those live where they already lived — config.py, telemetry.py, insights.py,
tire_diag/, powertrain_diag/, vehicle_health_policy.py — and a second copy in
this package would be a second answer to the same question.

READ-ONLY POSTURE
-----------------
Nothing in this package can ask a vehicle to do anything. There is no Mode 04
constant, no clear-codes route, no actuator command and no Holley transmit path
anywhere below this line, and vehicle/selftest.py asserts their absence by
parsing the source rather than by trusting this paragraph.
"""
