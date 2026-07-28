"""RIO headway monitoring — deterministic time-headway / gap-trend core.

Design ref: docs/headway_design.md. Stage 0 = recorded clips on a cloud pod.

Deliberately self-contained: nothing here imports the live RIO app (app.py,
vision.py, llm_interface.py), so the headway loop can be developed, run and
broken without touching the voice service.
"""
