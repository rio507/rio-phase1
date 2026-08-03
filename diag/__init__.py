"""diag — the diagnostic discipline, with the tires taken out of it.

This package is tire_diag/ with every reference to a tire removed. Nothing here
is new behaviour: it is the machinery that was proved against TPMS sensors,
lifted so that a second subsystem can be an INSTANCE of it rather than a copy of
it. The engine monitors in powertrain_diag/ are the first such instance, and the
whole argument for doing the lift first is that a copy would drift — two
lifecycles, two healing rules, two definitions of "confirmed", and eventually
two answers to "is this still wrong?".

WHAT IS GENERIC, AND WHY EACH PIECE IS
--------------------------------------
    lifecycle       CANDIDATE -> ACTIVE -> RESOLVED
    severity        informational / advisory / warning / critical
    communication   what the DRIVER has been told, separately from both

Those three axes are independent for reasons that have nothing to do with what
is being measured. A finding can be worse and already announced; a resolved
finding can still be worth mentioning once. That is as true of a charging system
as it is of a rear-left tire.

So is everything below it: a monitor that runs only under valid conditions,
`status` (could it run) held apart from `last_result` (what did it find), one
bad reading making a PENDING condition and never a confirmed one, evidence
frozen at the moment of the decision and never rewritten, healing that requires
sustained passing evidence rather than one good sample, recurrence that stays
traceable, and history that survives a restart.

WHAT IS NOT GENERIC
-------------------
Five things, and they are exactly the five hooks DiagnosticEngine leaves open:

    _ingest              what a snapshot from this domain looks like
    _validate            what an implausible reading is here
    _system_healthy      what a whole-subsystem outage looks like
    _build_input         what a monitor of this domain is allowed to see
    _freeze_evidence     what "the conditions at the time" means here

A domain is a subclass that fills those in. tire_diag.TireDiagnosticEngine is
one; powertrain_diag.PowertrainDiagnosticEngine is the other. Neither reimplements
a single lifecycle transition.

LAYOUT
------
    codes.py        DiagnosticCode + CodeCatalog — identity and meaning
    monitors.py     the monitor contract, the gate, and the runner for one run
    store.py        Store — atomic state + append-only events, per domain
    drivecycle.py   drive cycles, built on the session infrastructure
    runner.py       DiagnosticEngine — the stateful half, domain-agnostic
    shadow.py       per-domain speech clearance
    selftest.py     python -m diag.selftest

LLM FIREWALL
------------
Nothing in this package imports openai, llm_interface, vision, visual_qa or app,
and no monitor, lifecycle transition, severity mapping or speech decision can
read a model output. diag/selftest.py asserts it, the same way
tire_diag/selftest.py already did for the tire half.

Note the one asymmetry with tire_diag: this package does not import `config`
either. A domain passes its tunables in. That is not purity for its own sake —
two domains cannot share one module-level TIRE_DIAG_* constant, and a framework
that reached for config would have to know which domain was asking.
"""
