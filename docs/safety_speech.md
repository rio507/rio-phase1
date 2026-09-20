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

---

## Stage 4 — Eve, rendered and read back (2026-09-20)

Every pre-rendered line re-rendered in **Eve** on `grok-voice-think-fast-2.0`,
into `static/audio/eve/`, and **verified by a model that did not speak it** —
`grok-voice-transcribe-2.0` over `/v1/stt`. Rendered, not auditioned: nobody
listened to a single one of these and decided it sounded right.

**16 correct, 0 wrong, 0 unchecked, 0 missing.** Every line passed on the first
take, which is a better result than the Gleam render got and is worth recording
as a property of the voice rather than of the harness.

### The hazard that prompted the requirement

One voice previously failed on **"Pull over" sentence-initially** — and worse,
self-reported the failure as correct: it rendered *"Hey, Ava, when it's safe…"*
for *"Pull over when it's safe…"*, replacing the entire instruction with a
greeting. That is why the render-time check is the voice's own transcript **and**
an independent transcriber, and why an unverified clip does not ship.

Tested directly on Eve, three takes each:

| line | takes verbatim |
|---|---|
| `Pull over now.` | **3 / 3** |
| `Pull over when you can.` | **3 / 3** |
| `Pull over when it is safe — one of your tires is dangerously low and still going down.` | 2 / 3 |

**Sentence-initial "Pull over" is not a hazard for Eve.** The historical failure
does not reproduce: the two short imperatives are perfect, and the long line's one
miss was *not* the instruction — it came back *"one of **her** tires"* for *"one
of **your** tires"*. A pronoun substitution in the middle of a long sentence, not
a dropped action.

That is a milder failure than the original but the same lesson, and it lands on a
line that is **already not in the shipped set**: the long form was rewritten to
`tire_critical` — *"Tire's going down fast — pull over when you can."* — which
verifies clean. This retroactively justifies that rewrite on a second voice, and
it is an argument against ever putting a fifteen-word safety line back.

### No line needed rewriting

The requirement was that any line Eve cannot deliver gets rewritten and
re-verified. None of the sixteen failed, so nothing was rewritten. The only shape
that misbehaved is one the library had already moved away from.

### One difference in the verifier, stated because it makes the check harder

`/v1/stt` takes **no vocabulary prompt**. The OpenAI path supplies the whole clip
library as a hint on clips of four words or fewer — deliberately scoped, because a
hint on a long line talks the transcriber round exactly where a local error hides.
Eve's clips got no hint at all, so a two-word clip was transcribed cold. That
makes 16/16 a stronger result than the same number on the OpenAI path, and it is
why `_render_xai` treats an empty transcription as a failed attempt rather than as
"could not check": there is no hint to fall back on, so an unreadable clip is
re-rendered instead of shipped unverified.

### What is NOT done

`xai_voice` is a **render target only**. `config.VOICE_BACKENDS` deliberately does
not include it, so `VOICE_BACKEND=xai_voice` is still refused — the browser
controller for that transport does not exist (stage 3 built the playout queue, not
the session). These clips are ready for the day it does.
