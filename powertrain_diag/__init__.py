"""powertrain_diag — the engine monitors, as an INSTANCE of diag/.

Not a second diagnostic system. The lifecycle, the healing, the freeze frames,
the recurrence, the communication ledger and the shadow machinery all come from
diag/, which is the same code tire_diag runs on. What is here is nine evaluation
functions, one catalogue of codes and five hooks.

    codes.py      RIO's engine codes — NOT SAE DTCs, and never presented as one
    monitors.py   the nine §23 monitors and their pure evaluation
    engine.py     the five domain hooks, bound to the generic runner
    store.py      its own files, beside the tire domain's
    selftest.py   python -m powertrain_diag.selftest

RIO CODES AND P-CODES ARE DIFFERENT CLAIMS
------------------------------------------
    RIO-ENGINE-COOLANT-OVER-LIMIT     RIO observed this
    P0217                             the vehicle reported this

Both can be about the same overheat. vehicle/dtc/ owns the second kind; this
package owns the first; and §17.8 puts them under different headings because a
driver who cannot tell them apart has been told something false by a system that
only said true things.

SHADOW MODE, MORE FIRMLY THAN THE TIRES
---------------------------------------
Every code ships speak=False, config.VEHICLE_DIAG_SHADOW_MODE is True, and
nothing here has a fast path. The tire monitors have shadow logs from real drives
behind them; these have never seen a vehicle. That difference is the whole reason
clearance became per-domain rather than one global flag.
"""
