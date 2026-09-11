# "What do you see?" — where the seconds were, per stage

**The complaint:** visual questions are slow again. **The suspect:** `look()`
waiting on teacher context.

**The suspect was innocent.** Teacher context fetch measured **0.0 ms** on
every call, in both question shapes, before anything was changed. It reads a
cached reading and returns; it has never inferred on demand. It is now bounded
anyway, for a reason that is real but was not the cause — see §2.

The time was in four local Qwen passes, none of which had a deadline.

---

## 1. How it was measured

`tools/look_latency.py` drives the real `/realtime/tool` endpoint with frames
flowing at 4 fps for the whole run — a driver asks while driving, and the
observer refuses to serve a description older than `OBSERVER_FRESH_S`, so a
probe that pushes a burst and then asks is measuring a stalled feed.

Two question shapes, measured separately, because they are two paths and an
average hides both. Stage numbers come back in the tool result: `look()`
records its own, `visual_qa` has recorded its own since it was written, and the
endpoint adds the teacher fetch. Nothing is re-derived.

Getting the probe honest took three corrections worth stating, because each one
produced a confident wrong answer first: frames pushed without a registered
session are refused (`unknown_session`) and never reach the ring;
`/session/start` mints an id rather than taking one; and the observer is started
by the **session mint**, not by the first question, so a probe that skips the
mint measures cold start on every run.

---

## 2. Teacher context — not the cause, bounded anyway

| | measured |
|---|---|
| context fetch, cache miss | **0.0 ms** |
| context fetch, cache hit | **0.0 ms** |
| timed out | 0 of 49 |

It never blocked. But `context_for` takes the panel's lock, which is shared
with the frame path — and "none of those critical sections is long" is an
argument, not a bound. `context_now` gives it **50 ms**
(`TEACHER_CONTEXT_TIMEOUT_MS`) and returns nothing if the lock is not free,
which is the same state as the teachers having nothing fresh: already a thing
the block expresses and RIO already knows how to work with.

`tools/look_yield_selftest.py` asserts the rule the brief asked for: **a look
with no cached reading costs the same as one with the teachers disabled**,
measured over 200 calls each and required to agree within 1 ms.

---

## 3. What it actually was

A Qwen pass on this box is a flat **2.8 s** — and flat is the word: `/perceive`
takes 2777 ms handed a 384 px image and 2795 ms handed a 1024 px one, because
`vision.observe` downscales to 512 px first. Nothing else on the visual path
did.

Four passes had no deadline:

| stage | before | what it is |
|---|---|---|
| `observe_now` | **1.0–5.8 s** | on-demand description when the cache misses |
| `resolve` (VLM) | **5.1 s** | which of several candidates they meant |
| `enrich` | **6.4 s** | colour and body style — *14 output tokens* |
| `clarify` enrichment | **11.7 s** | one attribute read **per candidate** |

Each now has a bound, and each bound has the same shape: the pass cannot be
cancelled (it is a local GPU call and Python cannot interrupt a thread), so it
is not waited on. It runs **off** the answer path, and what it produces is
filed for the next question rather than thrown away.

- **`observe_now` → `observe_soon`**, 600 ms. On a miss `look()` now prefers
  `observer.recent()` — a slightly older description, **with its age attached**
  (`OBSERVER_ANSWER_MAX_AGE_S`, 5 s). A driver would rather hear about three
  seconds ago than wait five for now, and the answer already says how old it is.
- **`enrich` → 700 ms.** The composing model is handed the crop as well, so it
  can see the colour itself. A late read still files itself, so a follow-up
  about the same car has it free.
- **`resolve` VLM → 900 ms**, and a timeout is **not** treated as ambiguity —
  see below.
- **clarification reads → 700 ms.** Late reads cost colour;
  `resolve.describe` falls back to label and lane, which is what the question
  sounded like before enrichment existed.

### A timeout is not an ambiguous question

The first version of the resolve deadline fell through to the existing
`ambiguous` return, which makes RIO ask "which one?". That is the right answer
to a cue-less question — the rule exists because Qwen once picked a third
vehicle entirely on a bare "what is that?" — and the **wrong** answer here. The
words did discriminate, geometry produced a ranking, and the only thing missing
was an opinion we could not afford to wait for. Asking the driver because our
own card was busy dresses up a latency failure as a question.

So a timed-out tie-break takes geometry's top candidate, confidence marked
down, `method="geometry_after_timeout"`. The visual suite went from 53/54 to
**54/54** on that change — it also fixed a failure that predates this work.

### And one wrong answer that was arriving quickly

`is_generic_scene_question("What car is in front of us?")` returned **True**.
Seven words, and `_NEEDS_SPECIFICS` had `building`, `shop`, `store` and `sign`
but no vehicles — so an object question was served from the running caption, in
6 runs out of 6, at 1.2 ms. It looked like the fast path triumphing and it was
a general description of the road answering a question about one car. Vehicle
nouns now sit with the others; `"What's ahead?"` still has no noun in it and is
still a scene question.

---

## 4. Before / after, per stage, with teachers running

### Scene question — "What do you see?"

| stage | before | after |
|---|---|---|
| teacher context | 0.0 ms | 0.0 ms |
| observer cache | miss → **1.0–5.8 s** | **hit, 0.0 ms** |
| **tool round trip** | **p50 967 ms · p95 5829 ms** | **p50 1.4 ms · p95 3.4 ms** |

### Object question — "What car is in front of us?"

| stage | before | after |
|---|---|---|
| router classify | 0.2 ms | 0.1 ms |
| select frame | 12 ms | 12 ms |
| resolve referent | **5106 ms** | **450 ms** |
| enrich (Qwen) | **6381 ms** | **0 ms** (bounded, off-path) |
| clarification reads | **11 662 ms** | **701 ms** |
| backend composition | 1935 ms | 1547 ms |
| **tool round trip** | **p50 13 649 ms · p95 16 006 ms** | **p50 2231 ms · p95 3371 ms** |

**Targets: scene ≤ 1 s — met at 1.4 ms. Object ≤ 3 s — met at p50 2.2 s**
(p95 3.4 s, just over).

**First audio** adds the live session speaking the result on top of the tool
round trip: measured separately at 670 ms p50 on the shipped `openai_realtime`
backend (`docs/safety_speech.md`). Scene lands well inside a second; the object
question lands at roughly 2.9 s p50.

### The other suspects, checked

- **Is the scene fast path still serving from the observer cache?** Yes, and
  faster than the 4 ms it was cited at — **1.4 ms**. The observer is healthy:
  ~1 observation/second, 0 errors. It had **not** gone cold from the yield
  rules; the cold cache in the first measurements was the probe not minting a
  session.
- **Is the object path still locate-then-read at full resolution?** It was
  locate-then-read, and the resolution was not the problem: downscaling the
  crop cut input tokens 454 → 266 and changed decode time **not at all**. The
  cost is per pass, not per pixel.
- **Is backend composition adding time?** No. 1.5–1.9 s, which is what a remote
  visual composition cost before.

---

## 5. Running it

```
  python -m tools.look_latency                  # both shapes, per stage
  python -m tools.look_latency --warm off       # measure the cold cache
  python -m tools.look_yield_selftest           # the rules, asserted
```
