"""tire_diag — OBD-inspired diagnostic monitors for Tire Health.

RIO Tire Health is NOT an OBD-II system, is not OBD-II compliant, and emits no
SAE powertrain DTCs. What is borrowed is the diagnostic *discipline* that makes
OBD-II trustworthy after forty years of use, applied to tire sensors:

    monitors run only under valid conditions
    not-ready is not the same as passed
    one bad observation makes a PENDING fault, never a confirmed one
    persistent evidence makes a CONFIRMED fault
    a narrowly-defined urgent condition may take a faster path
    the evidence is frozen at the moment of confirmation
    notifying the driver is a separate decision from storing the fault
    a problem is repaired only after passing verification, never after one
      good sample
    history survives a restart
    recurrence stays traceable
    the system never claims more certainty than its evidence supports

Layout, mirroring headway/:

    codes.py        the RIO diagnostic identifiers and what each one means
    monitors.py     the monitor definitions and their pure evaluation
    drivecycle.py   a drive cycle, built on the existing session infrastructure
    store.py        append-only event log + atomic state, on disk
    engine.py       the runner: samples in, monitor results and Issues out
    selftest.py     python -m tire_diag.selftest

LLM firewall: nothing in this package imports openai, llm_interface, vision,
visual_qa or app, and no monitor, lifecycle transition, severity mapping or
speech decision can read a model output. selftest.py asserts it, the same way
headway/live_selftest.py asserts it for live_policy.py.

SHADOW MODE: every monitor ships with speech_eligible=False. Detection,
persistence, freeze frames and the speech proposal RIO *would* have made are all
recorded; nothing is spoken. The single exception is the urgent fast path, which
is separately gated by fast_path_eligible and uses the pre-rendered clip
mechanism at the VEHICLE_HEALTH arbiter priority.
"""
