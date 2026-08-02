# OBD-inspired tire diagnostic monitors

**RIO Tire Health is not an OBD-II system.** It is not OBD-II compliant, it
emits no SAE powertrain DTCs, and nothing here imitates a P-code. What is
borrowed is the *diagnostic discipline* that has made OBD-II trustworthy for
forty years, applied to tire sensors.

> Copy OBD-II's diagnostic discipline, not its emissions-specific
> implementation.

## What this supersedes

The vehicle-health conversation layer (`6dd4b25`) had `vehicle_health.TireSource`
read `tires.snapshot()` and turn each corner's *instantaneous* state into an
issue. One reading below the threshold became a `warning`. One missed poll became
a `critical`. That is exactly the behaviour this spec removes.

| Before | Now |
|---|---|
| `TireSource` classifies one reading | reads the diagnostic engine's ACTIVE issues |
| policy's in-memory `_seen` cooldown is the memory of what the driver was told | persisted communication ledger on the Issue |
| `MockTireProvider` returns four perfect readings instantly | a radio: 45 s intervals, sleep when parked, junk wake frames, packet loss |
| `tires.py` `_classify()` creates issues | still the dashboard's instantaneous view; no longer creates issues |

Unchanged: the router intent, the context injection, the prompt, the arbiter
tier, the endpoints, and `EngineSource` (this spec is tire-only).

---

## 1. The realistic mock came first

Every enabling condition a monitor has — *enough valid samples*, *comparable
thermal state*, *the sensor is still transmitting* — is a statement about a
stream that is sparse, late and occasionally wrong. Tuned against a stream that
answers instantly and perfectly, a monitor confirms a fault in four polls and
then never fires on real hardware.

So `tires.py` gained a `_Radio` between the scenario and the reader:

- **45 s ± 12 s report intervals**, per corner, out of lockstep
- **sleep after 15 min parked** — silence is the *normal* state of a parked car
- **junk wake-up frames** — the first 1–2 reports after a wake are tens of PSI out, in either direction
- **3 % packet loss** (45 % on a dying cell), deterministic via `crc32(corner, n)` — not `random()`, so a monitor failure is reproducible
- **fast transmit mode** at ≥3 PSI of movement, dropping to 6 s — this is why the urgent path is reachable at all
- **erratic dying-battery mode** — wandering, not drifting, which is what tells it apart from air actually leaving

Six new scenarios exercise the radio rather than the tire: `dying_sensor`,
`cold_morning`, `inflation_event`, `receiver_outage`, `leak_then_loss`,
`blowout`.

`cold_morning` is the one worth calling out: all four tires down 4 PSI together.
The absolute low-pressure monitor **should** fire — the tires really are
under-inflated. The slow-leak monitor **must not**, because nothing has left the
tire. Four corners moving together is the signature that separates weather from
a puncture, and it is why every leak judgement subtracts the peers.

---

## 2. Monitors

`tire_diag/monitors.py`. Nine, each a `MonitorDefinition` dataclass with
enabling conditions, inhibiting conditions, pending/confirmed/urgent criteria,
healing criteria, freeze-frame fields and a pure `evaluate`.

| monitor | what it claims | notable gate |
|---|---|---|
| `tire.low_pressure` | below placard − warn delta | no thermal gate: a cold tire that is 4 PSI down *is* 4 PSI down |
| `tire.critical_low_pressure` | below critical delta or absolute floor | urgent path needs moving + falling + 2 validated samples |
| `tire.slow_leak` | air has actually left | thermally comparable pair **and** peers materially more stable |
| `tire.asymmetric_loss` | one corner vs its axle peer | both corners need a comparable pair |
| `tpms.sensor_plausibility` | the sensor is lying | reads the *rejected* samples — everything else filters them out |
| `tpms.sensor_connectivity` | stopped transmitting / low cell | **moving only** — parked sensors are asleep by design |
| `tire.sensor_loss_during_decline` | lost sight of a tire that was going down | requires an ACTIVE decline on that corner |
| `tire.inflation_event` | somebody added air | parked only |
| `tpms.receiver_health` | one box fault, not four tire faults | inhibits every per-corner monitor |

### Status and last_result are two fields

```
status       could it run, and can it now?
             NOT_SUPPORTED / NOT_READY / RUNNING / READY / INHIBITED /
             DATA_UNAVAILABLE
last_result  what did it find the last time it DID run?
             PASSED / FAILED_PENDING / FAILED_CONFIRMED / None
```

Folding these into one enum is the mistake that makes a diagnostic system lie.
`NOT_READY` is not a result; a monitor that has never run has *no* result. The
sentence the spec asks for needs both fields and one enum cannot say it:

> "I do not have enough comparable readings yet to evaluate a slow leak, but the
> current pressure is not critically low."

`RUNNING` is distinct from `NOT_READY`: it has the samples and is waiting out
the window. A driver asking is owed "still watching", not "not enough data".

**The rule this enforces:** if neither pressure monitor has judged anything, the
conversation layer emits an informational *"tire pressure has not been evaluated
yet — this is not the same as the tires being fine"* issue, so the one-line
summary in every turn cannot claim an all-clear nobody established.

---

## 3. Lifecycle, severity and communication are three independent dimensions

```
lifecycle       CANDIDATE -> ACTIVE -> RESOLVED     what the diagnostic believes
severity        informational / advisory /          how bad it is
                warning / critical
communication   firstToldAt, lastToldAt,            what the DRIVER has been told
                lastSeverityTold, lastBriefedAt,
                acknowledgedAt, announcementCount,
                monitoringActive,
                resolutionMentionedAt,
                shadowProposals
```

ANNOUNCED is not a lifecycle state. Neither is CRITICAL, WORSENED or
DATA_UNAVAILABLE. This is OBD's own separation: storing a diagnostic condition
and commanding the malfunction lamp are different decisions on different
evidence. A tire can be worse *and* already announced; a resolved issue can still
be worth mentioning once.

`advisory` was added to the severity ladder for this work — a confirmed finding
that belongs on the dashboard and in a drive-start briefing but must never
interrupt a drive. A possible slow leak is the case it exists for, and before it
existed everything confirmed had to be either ignorable or worth interrupting
for.

### Validation, before any monitor sees a packet

- out of `TIRE_PLAUSIBLE_RANGE_PSI` → rejected outright
- step > `TIRE_IMPLAUSIBLE_STEP_PSI` → rejected **provisionally**
- **retraction**: if the next report agrees with the rejected one, *both* become
  valid — the step was the tire, not the radio

That retraction is load-bearing. Without it a tire going down fast never
accumulates two consecutive valid readings at the new level, and the urgent path
is unreachable by construction. Rejected samples are *kept* with `valid=False`,
because how often a sensor talks nonsense is what the plausibility monitor runs
on.

---

## 4. Freeze frames

Captured when an Issue is confirmed, and again on any material severity
increase. Never rewritten by a later reading — three weeks later the question is
not what the tire reads now, it is what we were looking at when we decided.

Each frame carries only the fields its code declares (`FF_PRESSURE`, `FF_SENSOR`,
`FF_SYSTEM`) plus identity. Raw radio identifiers are deliberately absent: a
sensor id is of no use in a sentence and the conversation layer has no business
holding one.

---

## 5. Diagnostic codes

`RIO-TIRE-LOW-PRESSURE-RL`, `RIO-TPMS-RECEIVER-UNAVAILABLE`, 37 in total.
Readable and namespaced — `RIO-<system>-<condition>-<location>` — because a code
in a log should be legible without a lookup table, which is the one thing the
P-code space is worst at.

Each maps to monitor, component, default severity, driver-facing term,
technician description, suggested action, confirmation summary, healing summary,
freeze-frame fields, `speech_eligible` and `fast_path_eligible`.

**Never shown to the driver.** RIO says *"your rear-left tire may have a slow
leak"*, not *"RIO-TIRE-POSSIBLE-LEAK-RL is active"*. `GET
/vehicle/diagnostics/catalogue` is the service view.

---

## 6. Shadow mode

**Every code ships with `speak=False`, and `TIRE_DIAG_SHADOW_MODE=True` sits over
the top of all of them.** These monitors have never seen a real drive; the tuning
data that would justify letting one interrupt a driver does not exist yet.

What still happens in shadow mode: full detection, confirmation, persistence,
freeze frames, and the *complete* announcement decision — severity, cooldown,
priority and wording. What does not happen is the speaking. The policy returns a
`proposal` instead of an `announce`, and the engine writes it to the permanent
log as `shadow_proposal` with the exact words.

A proposal **consumes the cooldown** exactly as an announcement would. Otherwise
the shadow log would show a fault proposing itself on every poll and would wildly
overstate how talkative the monitor really is.

`vehicle_health_policy.py` still imports **nothing** — it composes the proposal
and hands it back; `app.py` does the writing. The firewall is unchanged.

### What the first shadow run changed

The first drive run through this produced two proposals, and the second one was
wrong: the driver had added air nine minutes earlier, the pressure had been good
ever since, and the ten-minute reminder fired four and a half minutes before the
healing criteria finished verifying the repair. RIO would have told them to pull
over for a tire they had just dealt with.

The fix is **not** to make `REMIND_S` longer than the healing time — that is a
coincidence between two unrelated constants and it breaks silently the next time
either is tuned. Reminders are gated on `healing_runs == 0`: do not remind about
a fault that is currently passing its monitor.

That bar is deliberately lower than resolution. Resolution needs a stable period
*as well as* a run count, and it should, because one good reading is how a warm
tire on a motorway "fixes" a leak. But the cost of a wrong silence is a reminder
ten minutes later; the cost of a wrong reminder is the nag above.

`healing_runs` travels on the issue dict because the policy imports nothing and
cannot ask the engine anything. The engine clears `healing_progress` the moment
a monitor fails again, so a fault that is getting worse cannot be silenced by a
stale count.

### The one exception: the urgent fast path

Two codes carry `fast_path=True` — `CRITICAL-PRESSURE-*` and
`SENSOR-LOSS-DURING-DECLINE-*`. They speak even in shadow mode, because the
argument for a fast path is that the consequence of waiting is not recoverable.
They skip the **run count**, not the **validation**: every gate in `monitors.py`
has already been passed, and those gates are what a single malformed packet, a
wake-up frame, a receiver-wide loss or an unknown sensor cannot get through.

They play **pre-rendered clips** (`static/audio/tire_critical.mp3`,
`tire_sensor_lost.mp3`) at the existing `VEHICLE_HEALTH` arbiter priority — the
same mechanism and the same argument as the headway red tier. A pre-rendered
line cannot name a corner or a pressure, so it says what is true of all of them
and the dashboard carries the detail.

---

## 7. Drive cycles

Built on `sessions.py`, not beside it. `/session/start` opens a cycle,
`_teardown_session` closes it — including the untidy ending where the client
simply vanishes and the reaper notices. The speed heuristic
(`TIRE_DIAG_DRIVE_START_MPH`, `TIRE_DIAG_DRIVE_END_PARKED_S`) exists only for the
case nobody told us: a pod restart mid-drive, and every headless test.

Used **sparingly**. Exactly one monitor requires a drive cycle — the slow leak,
where a decline measured inside a single drive is mostly measuring the drive.
Nothing urgent waits for one. A critically low tire that sat through three drives
before being mentioned would be a design failure, not diagnostic rigour.

Ending a drive resolves nothing, clears nothing and resets no counter. A system
where the way to clear a fault is to close the tab is not a diagnostic system.

---

## 8. Healing, history and relearn

Healing needs passing runs **and** a stable period, per monitor
(`TIRE_DIAG_HEAL_RUNS`, `TIRE_DIAG_HEAL_STABLE_S`), plus
`TIRE_DIAG_HEAL_HYSTERESIS_PSI` so a pressure on the line cannot heal and re-fail
forever. A warm tire on a motorway reads a PSI or two higher than the same tire
cold; a system that resolves on that reading has observed the weather, not a
repair.

Persistence is two files under `training_data/vehicle/`:

- `tire_diag_state.json` — what is true now, rewritten whole, atomically
- `tire_diag_events.jsonl` — what happened, append-only, trimmed only by age

A missing state file means *"we have no history"*, never *"the car is fine"*.
Clearing the cache or restarting cannot mark a problem as repaired.

**Relearn** (`POST /vehicle/diagnostics/relearn`) deletes nothing. It says "stop
comparing against what came before", records who and why and the previous sample
counts, and moves trend monitors to `NOT_READY` because they genuinely are.
Absolute pressure monitoring returns the moment a reliable reading exists — a
tire at 12 PSI is at 12 PSI whether or not we have learned its new sensor.

---

## 9. Endpoints

```
GET  /vehicle/diagnostics             monitor readiness, issues, drive cycle
GET  /vehicle/diagnostics/events      the append-only record
GET  /vehicle/diagnostics/catalogue   every code and monitor, fully described
POST /vehicle/diagnostics/relearn     sensor replaced / rotated / reset
```

The engine is fed from exactly two places — `/vehicle/tires` and
`/vehicle/health/announcement` — and deliberately **not** from a conversation
turn. Reports are deduplicated by their own timestamp, so polling faster than
the sensors transmit adds no evidence; without that, "valid sample count" would
measure the poll rate.

---

## 10. Testing

```
python -m tire_diag.selftest              118 checks, offline, ~2 s
python -m tools.vehicle_health_selftest   126 checks (132 with --model)
python -m headway.selftest                284 checks
node tools/nav_selftest.js                 41 checks
```

The 18 checks the spec asks for, plus amendment 6's nineteenth, each in its own
function and each named for what it protects. All run against the realistic
mock, in synthetic time — `tires.snapshot(at=)`, `engine.observe(now=)`,
`policy.tick(t)` all take their clock from the caller, which is what makes a
thirty-minute leak window testable in milliseconds.

A note on the harness: `check(condition, label)` follows
`headway/live_selftest.py`'s argument order, and `headway/selftest.py` uses the
opposite one. Getting them the wrong way round produces a suite where every check
passes, which is how this one started.

The store is pointed at a temp directory *before* anything can write. A test
suite able to fabricate a car's fault record would be worse than no test suite.
