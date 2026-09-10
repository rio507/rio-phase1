"""The teacher services — run OUTSIDE RIO's interpreter, on purpose.

Nothing in this directory may be imported by RIO. It is written for the two
isolated environments under /opt/teachers/venvs (Python 3.12, their own torch
and transformers pins, one per model), and RIO's own environment cannot import
most of it. tools/teacher_firewall_selftest.py asserts that no RIO module
imports `teachers.service`, which is the machine-checkable half of this
paragraph.

The other direction is asserted too: nothing here imports RIO. These files see
JPEG bytes, a list of numbers and a prompt, and answer with JSON. They do not
know what a session is, what a band is, or that RIO exists.
"""
