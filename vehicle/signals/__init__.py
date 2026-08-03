"""signals — the canonical vocabulary: names, units, quality and provenance.

Four small modules and one rule between them: a number that has crossed into RIO
carries where it came from, how much it is worth, and what it is measured in, and
never loses any of the three.

    registry.py     canonical dotted names, and the aliases to RIO's flat ids
    units.py        conversion, at the decoder boundary and nowhere else
    quality.py      how much a reading is worth
    provenance.py   who is making the claim — the ECU, or RIO
    schema.py       the event and batch on the wire, and their validation
"""
