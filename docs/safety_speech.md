# Safety speech — split by urgency, not by category

**What changed.** Most of what RIO says about danger used to be a recording or
a template with two slots in it. Now only the lines that genuinely cannot wait
are recordings; everything else is written fresh from the structured event, in
her own voice, and never comes out word-for-word the same twice in a drive.

**What did not change.** When she speaks. Every band, cooldown, minimum gap,
reminder window, healing gate, confidence and warm-up suppression is exactly
where it was and decides exactly what it did. This is a change to the words.

---

## The split

The question asked of each line is only: **can this wait for a model and a
mouth?**

### Critical — stays pre-recorded (4 lines)

| id | line | fires on |
|---|---|---|
| `too_close` | "Too close — ease back." | tau < 2.0 s, confirmed entry (variant A) |
| `watch_distance` | "That's tight — drop back." | as above (variant B, alternated) |
| `back_off` | "Back off — now." | TTC below threshold **with the gap collapsing**, or worsening |
| `tire_critical` | "Tire's going down fast — pull over when you can." | `tire.critical_low_pressure`, urgent fast path |

A local file, already decoded by the output bus. No network, no model, no
session. **Measured at 29 ms** from the policy's decision to the first audible
sample, in a real browser.

### Everything else — phrased fresh

- headway `calm` and `escalate` (already TTS; this changes the words)
- `tire_sensor_lost` — **left the clip tier**
- all 9 `vehicle_health_policy.LINE` templates
- connection notices — none were spoken before; the event shape now supports one

Navigation's 12 imminent turn clips are **not** safety speech and are
unchanged.

### Why `tire_sensor_lost` left

Its action is *"check it by hand when you stop"*. That is not now, so it has no
business paying the price of a fixed clip — and a clip has no slots, so it
could never say which corner. Phrased, it says:

> "The rear-right sensor's gone quiet for three minutes after reading 26 PSI
> and falling — check that tire by hand at your next stop."

---

## Rewriting the critical lines

They were stale because nobody had read them out loud since they were written,
not because they were clips.

| was | now | why |
|---|---|---|
| "You're too close." | **"Too close — ease back."** | a flat assessment with nothing to do about it; now fact *and* action |
| "Watch your distance." | **"That's tight — drop back."** | the worst of them. That is a driving instructor, and the bible rules the register out by name |
| "Back off — now." | **unchanged** | reviewed and kept. Short, action first, urgency in the words. Changing it to show change would have made it worse. |
| "Pull over when it's safe — one of your tires is dangerously low and still going down." | **"Tire's going down fast — pull over when you can."** | fifteen words in the tier defined by not having time for them |

A and B **must stay equal in force**: `_next_unsafe_line` alternates them for
the same event, so a difference in severity between them would make the warning
a driver got depend on how many times they had been warned before. Both are
fact-then-action fragments of the same length.

The tire rewrite also fixed a clip the Gleam voice could not render at all —
"pull over" transcribes correctly mid-sentence where it never did
sentence-initial. See `docs/live_gpt_live.md`.

---

## How a line gets written

`safety_speech.py`. The structured event — reason code, severity, evidence,
observation window, suggested action, derived provenance — goes to
`gpt-5.6-luna`, and **what comes back is checked before it is spoken**:

- **no number that is not in the event.** Digits and words both ("nineteen"
  fails the same way "19" does)
- **provenance survives.** Derived conservatively, and the default is "RIO
  worked it out" — claiming the car reported something it did not is the worse
  error by a long way, because that is what a mechanic will look for and not
  find
- **an unconfirmed finding may not be asserted.** A pending code stays pending
- **the bible's banned list applies**
- **length is capped**

Two failures and the deterministic sentence is used instead — which is exactly
what shipped before this file existed. A line that cannot be proved honest is
not spoken. Same discipline the clips get from the render verifier, applied to
a sentence instead of an audio file.

The announcement policy stays **pure, clockless and replayable** — the model
call happens in `app.py`, which ticks it, and the result arrives as an ordinary
field on the issue. `tools/vehicle_health_selftest.py` still replays whole
sessions and asserts on exact words.

### Variation

Scoped to the drive and keyed by the issue, so it is the second mention of the
*same* fault that gets varied. What has already been said is fed back and a
word-for-word repeat is rejected by the same gate that rejects an invented
number. Measured: **3 distinct sentences in 3 firings, on every event tested.**

---

## The timing, and why phrasing happens early

| tier | p50 | p95 | budget | fits |
|---|---|---|---|---|
| **clip (critical)** | **29 ms** | 33 ms | — | unaffected |
| live, prepared — `openai_realtime` | **670 ms** | 728 ms | 2000 ms | comfortably |
| live, cold — `openai_realtime` | 1925 ms | 2037 ms | 2000 ms | **no** |
| live, prepared — `gpt_live` | 1558 ms | 3435 ms | 2400 ms | at p50 |
| live, cold — `gpt_live` | 3146 ms | 3570 ms | 2400 ms | **no** |

Writing the sentence costs 1.2–1.7 s, and the health channel allows 2000 ms
from decision to first audio. **On-demand phrasing does not fit inside that
budget; it fits inside the fallback** — which would have made this a slower way
to ship the old behaviour.

It does not have to happen then. A confirmed fault sits behind a cooldown, a
minimum gap and a healing gate before it is ever announced, and the sentence
can be written during that wait. Same move `observer.py` makes for the camera:
start describing the road now, not when she is first asked about it.

The headway coaching lines have no such lead-in — the band is entered and
confirmed in a second or two — but their event barely varies, so one line is
kept **ready at all times** and taking it starts writing the next. A pool of
one: the calm tier's cooldown is 30 s and writing takes under two.

---

## Selftests

```
  python -m tools.safety_speech_selftest             # the whole split
  python -m tools.safety_speech_selftest --offline   # no API calls
  python -m tools.safety_phrase_bench                # which model, and does it vary
  python -m tools.safety_latency                     # the table above
  python -m tools.clip_verify                        # the critical clips say what they should
```

The suppression checks are deliberately **not** re-implemented: the suite
shells out to `headway.live_selftest` and `tools.vehicle_health_selftest`
unmodified, on the principle that the way to prove something did not change is
to run its own tests untouched. Both still pass — 273 and 133 checks.

## Switching it off

`SAFETY_PHRASE_ENABLED=0` and every non-critical line is the deterministic
sentence it was before. The critical tier is not affected by that switch or by
any other: it is four files on disk.
