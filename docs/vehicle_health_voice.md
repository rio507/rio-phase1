# Vehicle Health as a conversation context — and as a check engine light

> **Superseded in part by `docs/tire_diagnostics.md`.** The conversation layer,
> router intent, priority tier, prompt and endpoints described here are all
> current. What changed is where the TIRE issues come from: `TireSource` used to
> classify one instantaneous reading, and now reads confirmed findings out of the
> OBD-inspired diagnostic engine. One reading can no longer become an issue, and
> the communication ledger that records what the driver was told is persisted on
> the Issue rather than held in the policy's memory.

RIO understands the car the same way she understands navigation and the world
out of the window: as one more context source. She answers questions about it
naturally, and she interrupts about it exactly when a modern check engine light
would — once, for something that actually matters.

Nothing here is a new speech system. Everything routes through the arbiter in
`static/rio_speech.js`, the router in `router.py` and the conversation path in
`llm_interface.py` that already existed.

---

## 1. Architecture

```
  tires.py                 telemetry.py                 (later: OBD-II,
  TireHealthProvider       TelemetryProvider(s)          RIO Connect, ECU)
       │                        │                              │
       └────────────┬───────────┴──────────────────────────────┘
                    ▼
             vehicle_health.py            the Voice Context API
        source registry → one vocabulary, one severity ladder,
        one issue list worst-first, an observation_window per issue
                    │
        ┌───────────┴────────────────────────┐
        ▼                                    ▼
  llm_interface.py                  vehicle_health_policy.py
  compact line every turn           NO IMPORTS. Severity gate,
  full structure when the           cooldowns, the decision to
  router says health question       speak, and the words
        │                                    │
        ▼                                    ▼
  RIO answers, streamed              GET /vehicle/health/announcement
  through /talk → arbiter                    │
  at CONVO (P5)                              ▼
                                     static/rio_health.js
                                     submits to the SAME arbiter
                                     at VEHICLE_HEALTH (P2)
```

Files created: `vehicle_health.py`, `vehicle_health_policy.py`,
`static/rio_health.js`, `tools/vehicle_health_selftest.py`.

Files modified: `router.py`, `llm_interface.py`, `rio_prompts.py`, `app.py`,
`config.py`, `tires.py`, `telemetry.py`, `static/rio_speech.js`,
`static/rio_nav.js`, `static/index.html`, `tools/nav_selftest.js`.

### The firewall

> **The LLM may ANSWER questions about vehicle health.
> The LLM never decides WHETHER or WHEN an announcement fires.**

`vehicle_health_policy.py` contains no import statement of any kind. Not
`config`, not `tires`, not `openai` — nothing. The clock is passed in on every
tick, so the whole policy replays from a log. This is the discipline
`headway/live_policy.py` established for collision warnings, applied to the
opposite failure: a model must not be able to raise an alarm, and must not be
able to talk itself out of one.

`tools/vehicle_health_selftest.py` asserts the import set is empty, that
`open`/`eval`/`exec`/`compile`/`__import__` never appear, and that the words
"openai", "gpt", "llm", "model", "prompt", "http" appear nowhere in the
executable half of the file (prose is exempt — the docstring has to name what it
is keeping out).

### Not about tires

The word "tire" appears in one class in `vehicle_health.py`. A subsystem is
three methods — `available()`, `state()`, `issues()` — plus a
`register_source()` call. `EngineSource` already proves it: battery, oil
pressure, coolant and fuel pressure arrive through the same interface today,
because `telemetry.py` reports them. Adding a domain changes no prompt, no
policy, no endpoint and no client code.

---

## 2. Conversation flow

```
POST /talk  (or /ask)
  └─ Whisper
  └─ _route_and_prepare()
       └─ router.classify()
            ├─ vehicle_health_question → _health_policy.note_status_request()
            │                            (one of the spec's cooldown resets)
            ├─ visual type             → visual_qa.answer()   unchanged
            └─ anything else           → conversation path    unchanged
  └─ llm_interface.generate_stream(transcript, route)
       └─ _health_block(route)
            ├─ health question → full context JSON  (~1.2k tokens)
            └─ every other turn → one line          (~110 tokens)
  └─ ElevenLabs, sentence-buffered
  └─ client: RIO.sayReply() → arbiter at CONVO (P5)
```

There is **one** router. `vehicle_health_question` is a type in the existing
`router.py`, matched by rules before the visual types (a health question must
never reach the camera: "is everything okay?" answered from a photograph of the
road is RIO describing traffic while the driver asks about a tire). It is not
visual, so it falls to the ordinary conversation path — the classification only
changes how much context the turn carries.

### The compact / full split

Every turn:

```
VEHICLE HEALTH: warning · 1 active issue · Rear Left has dropped 2.4 PSI over the last 24 hours, which looks like a slow leak.
```

A health question gets `vehicle_health.context(full=True)` — every corner, every
engine channel, every issue with its window and its suggested action. Never UI
state, in either size: no CSS class, no glyph, no poll interval, no banner.

### One behaviour change worth knowing about

`generate_stream` used to append the *composed* turn — camera observation and
all — to `history`. It now sends `history + this turn` to the model and appends
only the driver's plain words and RIO's reply.

Both injected blocks describe NOW. Left in history, twenty turns left twenty
stale snapshots behind, all unlabelled. For the camera that was untidy. For
health it is dangerous: four old pressure readings are exactly what a model
needs to invent a trend, and inventing trends is the one thing the health prompt
forbids.

---

## 3. Speech flow and priority

```
  P1  SAFETY           headway gap warnings
  P2  VEHICLE_HEALTH   critical faults only        ← inserted
  P3  TURN_NEAR        the 4-second turn
  P4  NAV              every other nav line
  P5  CONVO            RIO answering the driver
```

Below the gap warning because a collision is measured in seconds and a failing
tire in minutes. Above navigation because missing a junction costs three minutes
and missing this costs the car. Every existing relative order is unchanged;
`rio_nav.js`'s hardcoded `2`/`3` became `RIO.speech.P.TURN_NEAR` / `P.NAV`, so
the ladder now exists in exactly one place.

Announcement group is `health`, so a worse fault supersedes a lesser one
mid-sentence rather than queueing behind it. TTL 20 s — long enough to wait out
a gap warning, short enough that a stale one is dropped rather than delayed.

`/vehicle/health/voice?id=` is a **lookup, not a text-to-speech endpoint**: it
serves text out of the policy's issued table and refuses any id the policy did
not issue. The set of things RIO can ever say unprompted about the car is
bounded by `vehicle_health_policy.LINE`. Same contract as `/nav/voice` and
`/headway_voice`.

### Numbers are written to be heard

`"The rear left tire is down to twenty-six P S I."`

`nav.py` rounds a distance the way a person says it before it is ever spoken
("in 300 meters", never "in 287 meters") and composes it deterministically at
decision time. The same discipline, one step further: pressures and temperatures
come back as words, because an alert is the worst place to discover that a
synthesiser reads "29 PSI" as "two nine P S I". Voltage keeps its tenth —
12.1 V and 12.9 V are a dying battery and a healthy one.

---

## 4. Vehicle health context flow

```
GET /vehicle/health                 the normalized context. Read-only —
                                    does NOT tick the policy, so a debug tab
                                    cannot consume an announcement.
GET /vehicle/health/announcement    THE TICK. Server decides; returns an
                                    announcement or null plus the reason.
GET /vehicle/health/voice?id=       TTS by id. Refuses anything else.
GET /vehicle/health/policy          what the policy holds and why it was quiet.
```

`telemetry.snapshot()` gained `record=False`. A normal snapshot pushes the 20 s
trend ring, advances the runtime clock and feeds the insight engine — correct
once per poll at 1 Hz, wrong on a conversation turn. A driver who asked three
questions in a row would otherwise fit a slope across samples caused by having
been asked.

### Truthfulness

Every issue carries `observation_window`; the context carries `history_depth`.
The system prompt forbids exceeding either. If the window is 24 hours RIO cannot
say "over the past three weeks"; if there is no trend in the data there is no
trend. A corner with no 24-hour history is given `trend: null`, not
`"stable"` — claiming stability over a window that does not exist is the same
fabrication in a friendlier voice.

Interpreting is welcome. Extrapolating is not.

---

## 5. Example conversations

Real output from `python -m tools.vehicle_health_selftest --model`:

| scenario | driver | RIO |
|---|---|---|
| all normal | "How are my tires?" | "All good. All four are right where they should be." |
| slow leak | "How are my tires?" | "Rear left's been losing air over the past day — down a couple PSI. Not urgent, but get it checked in the next day or two." |
| slow leak | "Which tire is low?" | "Rear left. It's down a bit compared with the others, and it's been dropping over the last day." |
| critical | "Is there anything wrong?" | "Rear left is too low to keep driving on. Ease off and get air in it before going farther, or put the spare on." |
| no sensors | "Are my tires healthy?" | "I don't have tire data right now, so I can't judge them from here. Engine side looks normal." |
| all normal | "Hey." | "Hey. What's up." |

Note the third and fourth rows: "over the past day" is the honest window, and
the no-sensor answer refuses to guess rather than reporting healthy tires.

The announcement, spoken not written:

> "The rear left tire is down to twenty-six P S I. That's low enough that I
> wouldn't keep driving on it."

---

## 6. Severity and cooldown behaviour

| severity | speaks on its own | dashboard | answered if asked |
|---|---|---|---|
| informational | never | yes | yes |
| warning | never | yes | yes |
| critical | **once** | yes, until resolved | yes |

`ANNOUNCE_AT_RANK` is the single constant that decides this. Moving it to
WARNING is one edit, in the deterministic module, and nothing a conversation can
reach.

Cooldowns (`vehicle_health_policy.py`, PROVISIONAL block):

- `REMIND_S = 600` — the same unchanged fault may be mentioned again after ten
  minutes. A reminder says the same words; it is not a new alarm.
- `MIN_GAP_S = 20` — nothing at all within 20 s of the last announcement. Two
  criticals are two problems and one sentence a driver can take in.
- `RESOLVED_CLEAR_S = 60` — an issue must have been genuinely gone this long
  before its return counts as a new event. Otherwise a fault flickering across a
  threshold re-announces every other poll.
- `WORSEN_FRAC = 0.25` / `WORSEN_MIN` — a deterioration of a quarter, or a rank
  escalation, outranks the cooldown. Drift does not.
- `POST_REQUEST_QUIET_S = 90` — asking for status clears every per-issue
  cooldown **and** buys 90 s of silence. Both halves matter: without the reset a
  later deterioration sits behind a timer the driver started by being curious;
  without the quiet window RIO announces, seconds after answering, the fault she
  just described.

Every decision, spoken or not, is recorded with its reason. A silence with a
reason is as useful for tuning as an utterance — `headway/live_policy.py`'s
`voice_log` argument, and `/vehicle/health/policy` is where you read it.

---

## 7. Testing

```
python -m tools.vehicle_health_selftest            # 123 checks, fully offline
python -m tools.vehicle_health_selftest --model    # + 6 real conversation turns
node tools/nav_selftest.js                         # the arbiter, incl. the new tier
```

The Python selftest needs no network and no GPU. Eight parts:

- **A** context over the spec's six scenarios (all healthy, one low, slow leak,
  critical pressure, sensor disconnected, no sensor data), plus a check that no
  UI state reaches the conversation context.
- **B** the announcement policy, scripted straight into `VehicleHealthPolicy`
  (it is pure, so every branch is reachable exactly): announce once, stay quiet,
  remind, escalate on worsening, re-arm on resolve-and-return, hold the gap,
  never speak below the threshold — then the same driven through the mock.
- **C** truthfulness: every issue in every scenario carries a window, no window
  claims more history than exists, the prompt forbids exceeding it.
- **D** spoken form: no digit reaches the synthesiser in any line, and no line
  sounds like a diagnostic scanner.
- **E** the router, model fallback off: the spec's eight questions land on
  health, and nothing already routed elsewhere moved.
- **F** the LLM firewall.
- **G** the speech ladder, read out of the JavaScript.
- **H** dashboard/conversation sync across all nine tire scenarios.

`node tools/nav_selftest.js` is where the interrupt semantics are actually
proved — pre-emption, supersede-by-group, queue order — because that drives the
real arbiter through the code that ships. 41 checks, six of them the new tier:

```
=== arbiter — vehicle health sits between safety and navigation ===
  ok    the ladder is safety > vehicle health > near turn > nav > conversation
  ok    a critical health announcement cuts through a turn announcement
  ok    and the turn is dropped rather than resumed behind it
  ok    a gap warning still pre-empts vehicle health — never above safety
  ok    and it interrupts casual conversation, which is the whole point
  ok    a worse fault replaces the announcement still being spoken
  ok    queued behind a warning: health, then the turn, then conversation
```

Part G of the Python selftest asserts the ladder statically as well, so a
renumber cannot pass unnoticed on a machine with no node installed — which is
the normal state of the pod. `apt-get install -y nodejs` is enough; the harness
is plain ES5 with no dependencies and runs on anything from v12 up.

Driving it by hand: the tire scenario selector at the foot of the Vehicle Health
column switches the mock live, and `/vehicle/health/policy` shows what the
policy decided and why.
