"""dtc — diagnostic trouble codes: reading them, tracking them, explaining them.

    catalog.py    parsing (complete) and explaining (partial, and honest about it)
    lifecycle.py  the state machine, and the three axes it holds apart
    snapshot.py   what RIO was watching either side of a code appearing
    service.py    the scan schedule, and the Flagged Error Codes section

THE ONE SENTENCE THIS PACKAGE EXISTS FOR
----------------------------------------
    The check-engine light should not be the first time the driver learns
    that the ECU has noticed a problem.

Everything here follows from that. Pending codes are scanned for often, because
a pending code is the whole advantage. `early_detection` is recorded at first
sight and never recalculated, because "was this caught before the light came on"
is a fact about a moment. The snapshot reaches BACKWARDS, because the minute
before the ECU noticed is the minute nothing else in the system retained.

AND THE ONE IT MUST NOT SAY
---------------------------
    This component has failed.

A DTC names a condition, not a part. P0171 is "the mixture is lean", not "the
MAF is bad". Possible causes are a list and stay a list; `confirmed_cause` is
filled in by a person or by a validated repair, and by nothing in this package.
"""
