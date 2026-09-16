# Weather as a context source — what the camera cannot see

RIO can see the sky. She cannot see three o'clock.

That sentence is the whole design. The camera is a genuinely good instrument for
*those clouds ahead are getting dark* and a worthless one for *it'll rain in
about twenty minutes* — and the failure when the two are confused is specific:
a vision model asked what the weather will do will **answer**, fluently, with a
number, and the driver changes their drive on the strength of it. A forecast
inferred from a photograph is a guess wearing a number.

So weather joins navigation, places and vehicle health as one more context
source with a hard boundary around it. Nothing here is a new speech system:
answers go out the conversation path that already existed, and the one
unprompted line goes through the arbiter in `static/rio_speech.js` at the tier
it already has.

---

## 1. Architecture

```
  the panel's GPS fix                 (rio_realtime.js: currentFix(),
  lat / lng / age_s                    already attached to every tool call)
         │
         ▼
  weather.py                          the only file that knows Google exists
  ├── currentConditions:lookup        ─┐
  ├── forecast/hours:lookup            ├─ one refresh, 3 billed requests
  └── forecast/days:lookup            ─┘
         │
         ▼
  get_weather_context()               the normalized context: a dozen numbers,
  cache · staleness · distance        each with its unit, plus fetched_at,
                                      fetched_for and age_s
         │
    ┌────┴─────────────────────────────┐
    ▼                                  ▼
  realtime.py                      weather_policy.py
  get_weather tool                 NO IMPORTS. Thresholds, cooldowns,
  the driver ASKED                 min-gap, suppression. The decision
                                   to volunteer, and nothing else
    │                                  │
    ▼                                  ▼
  conversation path              arbiter at P.CONVO (tier 5, the lowest)
```

Downstream of `weather.py` nothing knows Google's response shape. A different
provider is one more implementation of `get_weather_context`, which is the same
property `navigation/provider.py` has and for the same reason.

---

## 2. The two sources, and the rule where they meet

| | authority on | never |
|---|---|---|
| **camera** | what is visible from the car right now — wet road, dark cloud, spray, low sun | any number; anything later than this second |
| **Weather API** | temperature, probability, when it starts, wind, visibility, what happens next | what is out of the window |

**When they disagree, RIO says both and resolves neither.**

> The roads are wet, though the weather data isn't showing active rain here.

Not *it's raining*. Not a theory about which source is right. Two readings, both
reported; the driver puts them together perfectly well. This is asserted in
`tools/weather_selftest.py` §B against a live model rather than against the
rule text, because a rule that contains the right sentence proves nothing about
whether a model follows it.

The same boundary is written into `VISUAL_SYSTEM_PROMPT` (`rio_prompts.py`), so
the vision turn cannot produce a forecast even when weather never comes up as a
topic.

---

## 3. Honesty: staleness is absence, not a caveat

Every context carries `fetched_at`, `fetched_for` and `age_s`. `usable()`
refuses one that is older than `WEATHER_MAX_AGE_S` (900 s) or fetched more than
`WEATHER_MAX_DISTANCE_M` (15 km) from where the car is now.

A refusal **removes the data**. There is no *the forecast is a bit old, but…*
answer, because a driver hears the forecast and not the but. A confident
forecast for where the car was fifteen minutes ago is worse than no forecast:
it is indistinguishable from a good one at the moment it is spoken, and only
wrong later, when the driver is in the rain they were told to expect elsewhere.

The refresh clock (600 s quiet, 300 s when the numbers are moving, 5 km of
movement) is deliberately **tighter** than the honesty limits. The gap between
them is where a refresh gets to happen without RIO going silent mid-sentence.

Failure modes, all of which produce the same behaviour and different log lines:

| `note` | meaning |
|---|---|
| `no_fix` | no GPS, or an unreadable one |
| `stale_fix` | the fix is older than `WEATHER_MAX_FIX_AGE_S` (180 s) |
| `no_coverage` | Google has no weather here (Japan, South Korea, mainland China) |
| *exception name* | network or API failure |

In every one, RIO says she cannot pull the forecast and may still describe what
she sees. What she may not do is turn the picture into a forecast.

---

## 4. The unprompted line

`weather_policy.py` imports nothing, keeps its tunables as module constants, and
takes the clock as an argument — the same discipline as
`vehicle_health_policy.py`, for a lower-stakes version of the same rule:

> **The LLM may answer weather questions and phrase an advisory it is given.
> The LLM never decides whether an unprompted weather line happens.**

It fires on four kinds and nothing else: rain not yet started (≥60%, 5–45 min
out), visibility under two miles, thunderstorms ≥50%, wind ≥30 mph. Note the
proactive rain bar (60%) is deliberately higher than the bar used to *answer*
(30%): a coin flip is worth saying when asked and is not worth interrupting
somebody for.

Gates on top of that: per-kind cooldown (900 s), minimum gap between any two
proactive lines (120 s), a quiet window after answering a weather question
(300 s), and no submission at all while anything above CONVO holds the mouth.

**It cannot preempt safety, health or navigation, structurally.** P.CONVO is
tier 5, below SAFETY(1), VEHICLE_HEALTH(2), TURN_NEAR(3) and NAV(4). Being cut
off by all four is correct: a weather line can be asked for again, a turn
cannot.

Every decision returns a reason whether it spoke or not — a drive that
suppressed forty findings and a drive that had none are very different drives
and must not look the same in a log.

### Facts must name their own subject

A finding carries `what` alongside its `facts`. This was not in the first
version, and the demo harness caught what its absence produced:

```
facts: {"probability": 60, "condition": "Partly sunny"}
  ->  "Partly sunny ahead, with about a 60 percent chance showing."
```

Every number real, every honesty assertion passed, and the sentence means
nothing — a chance of *what*? A fact that does not name its subject is not a
fact yet. `tools/weather_selftest.py` now asserts no finding carries a bare
`probability` or `condition` key.

---

## 5. What the API actually is (verified, not assumed)

Run `python -m tools.weather_probe` to re-check any of this. Full reading and
the open licensing questions are in **LICENSING.md §3**.

- **Enabled** on the existing `GOOGLE_MAPS_API_KEY` with no project change.
- **Horizon**: hourly 240 h max (24/page); daily 10 days max.
- **Units** are a *request* parameter (`unitsSystem`), not a response property.
- **Cost**: one SKU, *Weather Usage*, billed per request regardless of endpoint.
  10,000/month free, then $0.15/1,000.
- **`fields` mask** works but — unlike Places (New) — does **not** select a SKU.
- **No alerts endpoint exists.** `alerts` is always `[]` and means **not known**,
  never "no severe weather".
- **Coverage is not global.** Japan, South Korea and mainland China 404.

---

## 6. Cost, and why route-aware weather is phase 2

Phase 1 counted against `weather.py`'s own `_needs_refresh`, stepped per minute;
phase 2 modelled on top with the assumptions named below it.

| drive | P1 requests | P1 cost | P2 requests | P2 cost | ×P1 |
|---|---:|---:|---:|---:|---:|
| 20 min urban, settled | 6 | $0.0009 | 12 | $0.0018 | 2.0× |
| 45 min mixed, settled | 21 | $0.0031 | 35 | $0.0052 | 1.7× |
| 45 min mixed, rain in window | 24 | $0.0036 | 38 | $0.0057 | 1.6× |
| 90 min motorway | 69 | $0.0103 | 189 | $0.0283 | 2.7× |
| 4 h road trip | 183 | $0.0274 | 503 | $0.0754 | 2.7× |

Phase 2 assumptions, each a dial rather than a fact: route ahead sampled every
~25 km capped at 6 points; each sample = current + hourly (2 requests — a point
40 km up the road has no use for tomorrow's sunrise); destination-at-arrival = 2
requests re-checked every 15 min; route samples on the same volatile/quiet clock.

**The finding that matters is not the money.** Phase 2 costs under 8¢ on a
four-hour road trip, and the free tier covers 384 forty-five-minute drives a
month. Two other things are the real content of this table:

1. **Movement dominates at speed.** On the motorway the 5 km distance trigger
   fires every ~3 minutes, so a 90-minute motorway drive costs eleven times a
   20-minute urban one. That is the feature working — weather must follow the
   vehicle — but it means the cost driver is *distance*, not duration, and
   phase 2 multiplies exactly the dimension that is already dominant.
2. **Fan-out is a correctness surface before it is a billing one.** Every
   sampled point needs its own `fetched_for`, its own age, and its own
   staleness refusal, or phase 1's central guarantee quietly stops holding —
   and it stops holding in the direction that is hardest to see: a confident
   forecast for a point on the route the car will reach in forty minutes, spoken
   as though it were about here.

---

## 7. Files

| file | what |
|---|---|
| `weather.py` | the service, cache, staleness, the normalized context |
| `weather_policy.py` | the unprompted gate. Imports nothing |
| `realtime.py` | `get_weather` tool + schema + the driver-facing rules |
| `rio_prompts.py` | the vision turn's weather boundary |
| `config.py` | the `WEATHER_*` block |
| `tools/weather_selftest.py` | amendment C's four checks, plus coverage |
| `tools/weather_probe.py` | re-runs amendment A's verification |
| `tools/weather_shapes.py` | every question shape, live, end to end |
| `LICENSING.md` §3 | what was verified, and what is still open |
