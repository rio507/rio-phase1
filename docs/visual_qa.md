# Visual conversation — Phase A

**Status:** shipped, verified against real frames from a dashcam clip. Every
threshold and weight below is provisional in the same sense as the headway
ladder's: a starting point chosen to be argued with against real drives, not a
validated number. Everything that feeds a decision is logged, so they can be.

---

## The problem

Ask RIO what she can see, and she used to read out a caption:

> *"A silver vehicle is visible in the left lane."*

That sentence is not badly written. It is badly **sourced**. Qwen looked at the
frame, wrote one line, and GPT-5.5 — the model doing the talking — rewrote it.
The model holding the conversation had never seen the road. Every follow-up was
therefore a conversation about a sentence rather than about a car, and the
moment the driver asked something the caption did not contain ("what year is
it?") there was nothing to answer from.

The fix is not a better caption. It is that **the model doing the talking gets
the picture**.

---

## The split

Five components, and the division of labour is the whole design:

```
  camera ─► /headway_frame  ──────────────────────────────► 4 fps, continuous
                │
                ├─ UFLDv2          where the lane is                  ~2 ms
                ├─ RF-DETR         what is there, boxed                ~5 ms
                ├─ Depth Anything  how far away it is                  ~7 ms
                ├─ membership.py   stable track ids, which is the lead ~1 ms
                └─ framebuf        retain the frame, ~6 s of them
                        │
  driver speaks ──► whisper ──► router ──┬─ not visual ─► llm_interface (unchanged)
                                         │
                                         └─ visual ─► scene graph
                                                      reference resolution ──► track_id
                                                      best frame for that track
                                                      crop it
                                                      Qwen: colour, body style
                                                      GPT-5.5 (frame + crop + graph)
                                                          │
                                                      rio_speech @ P4 ─► ElevenLabs
```

| Component | Answers | Never does |
|---|---|---|
| RF-DETR | what is there, where, tracked | describe, or decide what matters |
| Depth Anything | how far | anything conversational |
| UFLDv2 | where the lane is | anything conversational |
| membership.py | which track, in which lane, is it the lead | consult a model |
| Qwen3-VL | what colour, what body style, which one they meant | detect, track, or speak |
| GPT-5.5 | what to say about it | measure anything, or issue a warning |

**Nothing on the perception side is ever spoken, and nothing GPT-5.5 says ever
becomes a measurement.** That is not a stylistic preference. A deterministic
safety system depends on the left-hand column, and this whole path is
downstream of it: if every file listed in this document returned nonsense, RIO
would describe the road badly and the headway warnings would be bit-for-bit
unchanged.

### Qwen is not the detector, and does not go back to being one

Qwen used to hold the headway anchor and was removed from it deliberately —
see `headway/detect.py`'s header. It cost 0.6–1.5 s a call against a 250 ms
frame budget, and everything built around it was rationing. It is used here for
two things it is genuinely good at and which are **not** on the frame path:

- **enrichment** — colour and body style of one cropped vehicle
- **reference resolution** — "the silver one on the left" → a track id, when
  geometry alone cannot separate two candidates

Both are on demand, capped, and cached. Neither runs at 4 fps.

---

## §1 Scene graph — `scene.py`

Arithmetic on state that already exists. Each object:

```json
{
  "track_id": "vehicle_27",
  "label": "car",
  "fine_label": "sedan",
  "bounding_box": [1017, 401, 1249, 509],
  "position": "right_adjacent_lane",
  "motion": "traveling_parallel",
  "depth_meters": 25.7,
  "attributes": {"color": "white"},
  "confidence": 0.93
}
```

`track_id` is `membership.py`'s own candidate id with a family prefix, so it can
always be traced back to the candidate that produced it.

**`label` and `fine_label` are separate fields, and this is the one deliberate
deviation from the spec's example** (which writes `"sports_coupe"` into
`label`). `label` is the detector's class, which a deterministic warning path
also reads; `fine_label` is Qwen's opinion. A language model's guess must not
silently overwrite a field a safety system depends on, and the model downstream
needs to know which of the two it is reading.

### Position

`membership.py` measures *how much of a box is in my lane*, which is right for a
following distance and useless for "the one on the left". Sidedness is recovered
here from the same corridor: the object's offset from the ego-lane centre, in
**lane widths**, at the row where it meets the road.

| offset (lane widths) | position |
|---|---|
| ≤ 0.5 | `ego_lane` |
| ≤ 1.5 | `left_adjacent_lane` / `right_adjacent_lane` |
| > 1.5 | `two_or_more_lanes_left` / `two_or_more_lanes_right` |

When the corridor has no bounds at that row (above the detected paint, past the
horizon) it falls back to frame thirds and says so: `left_side_of_frame` is a
claim about the picture, `left_adjacent_lane` is a claim about the road, and the
model is told which one it is getting via `position_basis`.

### Motion

A candidate keeps three depth samples for a median — the right window for a
median and the wrong one for a trend. So `SceneTracker` holds ~2.5 s of
`(t, range)` per track and takes a least-squares slope:

- `closing` / `pulling_away` — gap changing faster than 0.9 m/s
- `traveling_parallel` — inside the deadband, and we are moving
- `stationary` — gap shrinking at roughly our own road speed: it is parked and
  we are driving past it
- `unknown` — fewer than 3 samples, or under 0.6 s of span

---

## §2 Ring buffer — `framebuf.py`

~6 s, RAM only, one per session, dropped by `/session/end`.

**Why keep frames at all:** because the newest one is usually wrong. A driver
asks about a car a beat after noticing it; by the time the question has been
spoken and transcribed, the vehicle is smaller, further off-centre or half out
of shot. The frame where it was clearest went past two seconds ago.

Each entry holds the JPEG **exactly as the client sent it** — not re-encoded, so
the picture the model sees is the picture the detector measured — plus that
frame's per-object perception and the cheap half of a quality score. Sharpness
and exposure are *not* computed on the way in: they cost a decode, the 4 fps
path must not pay for a question nobody asked, and they are only ever needed for
a shortlist. They are filled in lazily and cached.

**Retention.** `RING_PERSIST` is **off**. Nothing writes a picture of the road to
disk. The buffer is six seconds long, it is not a recording, and it dies with
the drive. Turning it on writes the selected frame and crop of each answer under
`training_data/visual/<session_id>/` and records the paths in the log — an
operator's decision, taken deliberately.

---

## §3 Frame selection — `frameselect.py`

The spec's rule: *never automatically the newest frame*. Two passes, because
measuring costs a decode.

1. Score every retained frame on the cheap terms — object size, occlusion by
   other boxes, clipping at the frame edge, detector confidence, recency, glare.
2. Take the top `FRAME_SHORTLIST` (6), decode only those, and re-score with
   sharpness and exposure folded in.

| weight | term | why |
|---|---|---|
| 0.30 | size | a bigger box carries more of the detail an answer needs |
| 0.25 | quality | sharpness + exposure, once measured |
| 0.20 | clear | not overlapped by another box — a crop of two cars is a crop of neither |
| 0.10 | score | the detector's confidence in *this* frame |
| 0.10 | whole | not sliced by the frame edge; half a car reads as a different car |
| 0.05 | recent | ties go to the more recent frame |

Whole-scene questions use a different mix (quality 0.6, recency 0.3, coverage
0.1) because nothing is being singled out and "what do you see" is a question
about *now*.

Every result records `was_newest` and the full component breakdown, so the
question "did the scoring actually earn its keep" is answerable from the log.
Observed live: for an object question the selector chose a frame **3 s old**
over the newest, because the car was 35 % larger in it.

---

## §4 Qwen enrichment — `enrich.py`

One call per crop, capped at `ENRICH_MAX_OBJECTS` (3), cached per track for
`ENRICH_TTL_S` (20 s), taking the same `vision.get_handles()` lock every other
Qwen caller takes. Returns a colour and a body style, both validated against a
closed vocabulary — anything outside it is dropped rather than passed on, because
an invented colour is a detail the model downstream would state as fact.

It runs when a question turns on an attribute ("the silver one"), and for the
resolved referent. It does **not** run for a plain "what do you see": GPT-5.5 is
looking at the same frame and reads colour off it directly, so paying an 8B
decode for that would be latency spent on nothing.

Measured: ~340 ms per object, warm.

---

## §5 Request router — `router.py`

Rules first, model only when the rules are unsure. The asymmetry matters: a
visual question sent down the plain path is answered from a stale caption, which
is the failure this whole pipeline exists to remove; a non-visual question sent
down the visual path costs a frame, a crop and a multimodal call, and RIO
answers "hey" while staring at a photograph.

Phase A implements `scene_description`, `specific_object_question`,
`visual_follow_up` and `non_visual_question`. The other four spec types are
**classified anyway** and marked `phase_b`, so the log shows the real
distribution of what drivers ask before the behaviour exists.

Rules cover 18/18 of the phrasings in the router regression set with no model
call. The model fallback (~400 ms) only fires on phrasings the rules do not
recognise.

`visual_follow_up` is the type that depends on state: "what year is it" is a
follow-up only when an active referent is alive. Without one, the same words are
a question about nothing and go to the ordinary path.

---

## §6 Reference resolution — `resolve.py`

Geometry first, model second, and never the other way round.

Side, class and colour cues are parsed from the driver's own words and scored
against the graph. A stated side that does not match is heavily penalised — it
is the cue a driver is least likely to get wrong about their own field of view.
If one candidate is clear by `DECISIVE_MARGIN`, the answer is settled and **no
model is called**: the fast path is also the trustworthy one.

If a colour was named and nothing is enriched yet, the top few candidates get a
Qwen pass and are re-scored. Colour neighbours are treated as partial matches —
a silver car called "grey" or "white" is the commonest near-miss in this
vocabulary and rejecting it outright would discard the right vehicle.

Only when two candidates remain plausible does Qwen get the frame with the
candidates drawn on it as numbered boxes, and pick a number. That is the one
part of this a VLM is genuinely better at than arithmetic: the question has
become *what do they look like*.

**Ambiguity in Phase A:** when nothing separates them, the best candidate is
returned with `ambiguous` set and the alternatives listed, and that uncertainty
is passed to the answer so RIO hedges and lets the driver correct her. Asking a
clarifying question back is **Phase B**, and this flag is what it will hang off.

---

## §7 The multimodal turn — `visual_qa.py`

**Never video.** At most two images per question: the best recent full frame,
and a crop. Local perception runs continuously; GPT-5.5 is called once, when the
driver asks.

### Crops are upscaled, and this was measured

`CROP_MIN_PX` is 768. This is not cosmetic and it is not "adding detail" — it is
about how many image tokens the object gets at the far end. A high-detail image
is tiled at ~512 px, so a 400 px crop is one tile and the vehicle inside it lands
on a fraction of one.

On the white saloon in the test clip (native crop 398×246, true object 245×111),
asked "what kind of car is that on the right?", five runs each:

| crop | result |
|---|---|
| native (336 px floor) | **0/5** — "BMW 5 Series", every time |
| upscaled (768 px floor) | **5/5** — "Lexus LS 460", correct |

The pixels are identical. Only the tiling changed.

There is a real limit past which this stops being true: when the object was
genuinely tiny in the source frame, no interpolation puts a badge back, and a
model shown a smooth 768 px image of a 30 px car will read detail that was never
there. So the **true object size in the original frame** travels with every crop,
and anything under `CROP_DETAIL_LIMIT_PX` (96) is explicitly flagged as
detail-limited.

### Wide frames do not carry makes

Also measured: asked "what do you see", the model named a marque for a car 25 m
away in the wide frame — and got it wrong every time. Raising the image detail
from `auto` to `high` did not help, because the pixels are not there. Worse, the
wrong name **stuck**: the object question that followed inherited it from the
conversation history.

So the scene turn is told not to make the claim: describe vehicles by type and
colour, and leave identification to the close view that arrives if the driver
asks. After that change, three full acceptance runs gave a generic scene answer
("a white sedan"), a correct identification, and a correctly hedged year.

Object turns additionally carry `supersedes_earlier` — the close crop is better
evidence than anything said before, and without saying so the model stays loyal
to its own earlier sentence.

### Visual referent memory

```json
{"track_id": "vehicle_1", "label": "car", "fine_label": "sedan",
 "attributes": {"color": "white"}, "position": "right_adjacent_lane",
 "depth_meters": 25.7, "motion": "traveling_parallel",
 "last_best_frame_id": "f000023", "last_crop_path": null}
```

A follow-up **does not re-identify**. Re-resolving "what year is it?" from
scratch would silently switch cars the moment a nearer one appeared. The
referent also stores its last good crop, so a follow-up survives the vehicle
leaving the frame — and only a *better* crop replaces it, so a car that has since
shrunk into the distance cannot downgrade the picture the answer rests on.
Observed live: the third turn reported `crop_source: referent_memory_better`.

The referent is remembered even when generation fails and nothing was said. It
is established by what the driver **asked about**, not by whether RIO managed to
answer — and a failed turn is exactly when they are most likely to just ask
again.

TTL is 90 s, and it is dropped with the session: a car discussed on the last
drive must not be what "what year is it?" attaches to on the next one.

### One conversation, not two

A visual turn runs its own request against its own system prompt, so nothing
about it passes through `llm_interface.generate_stream`. `note_turn()` writes the
plain question and the spoken reply back into that history — without it, the
driver could ask about a car, get an answer, and find the next ordinary question
had no idea the exchange happened. The images and grounding stay out: large,
stale within seconds, and worse than useless on an unrelated later turn.

---

## §8 One mouth — `rio_speech.js`

Conversation was the last voice on the page that owned its own audio element.
That worked while it was the only thing that talked back, and stopped working
the moment navigation could speak.

```
  P1  safety      headway warnings
  P2  turn_near   the one time-critical nav line
  P3  nav         every other announcement
  P4  convo       RIO answering the driver, visual answers included   ← new
```

Conversation is **lowest, deliberately**, and the ordering is not about
importance. It is about what is still true later. A turn announced after the
junction is worthless; a gap warning three seconds late is worthless. An answer
to "what kind of car is that" can be asked for again. So conversation is the tier
that yields, and a visual answer gets cut off mid-sentence by a near-tier turn
exactly as it should be.

Group `convo`, so a second question replaces an answer still being spoken. No
TTL — an answer does not expire on a clock — but a 60 s watchdog, because a
visual answer is several sentences rather than one.

Verified in `tools/nav_selftest.js` (34 checks): near-tier pre-empts convo, safety
pre-empts convo, a pre-empted answer is dropped and never resumed, a second
question supersedes the first, and queued behind a warning the nav line goes
first and conversation last.

---

## §9 Logging and privacy

One `visual_qa` event per turn, carrying the whole **decision chain**: the
question, how it was classified and by what, which frame was chosen out of the
ring and how it scored against the newest, which track the reference resolved to
and by which method, what the crop actually contained, what went to the model,
what came back, and every stage's latency.

That chain is the only way to argue with a bad answer. "RIO described the wrong
car" has at least four distinct causes — misrouted, wrong frame, wrong track, or
the model misreading a good crop — and they are indistinguishable from the reply
alone.

**No images.** Not the frame, not the crop, not a thumbnail. Only ids, sizes and
scores. Images are written only when `RING_PERSIST` is on, and then the record
carries the path — so a review can find them if they were kept, and can see that
they were not if they weren't.

---

## §10 Knob sheet

| knob | value | what it trades |
|---|---|---|
| `RING_SECONDS` | 6.0 | how far back a better frame can be found, against RAM (~3 MB) |
| `RING_MAX_FRAMES` | 32 | hard ceiling; a fast client cannot grow the buffer |
| `RING_PERSIST` | **False** | whether any picture of the road reaches disk |
| `FRAME_MAX_AGE_S` | 4.0 | past this the scene has moved on |
| `FRAME_SHORTLIST` | 6 | frames decoded per question (~5 ms each) |
| `CROP_PAD_FRAC` | 0.6 | context around the object |
| `CROP_MIN_PX` | 768 | image tokens on the object — see §7, measured |
| `CROP_DETAIL_LIMIT_PX` | 96 | below this, the crop is flagged detail-limited |
| `ENRICH_MAX_OBJECTS` | 3 | Qwen calls per question, ~340 ms each |
| `ENRICH_TTL_S` | 20.0 | attribute cache; also what stops a recycled track id inheriting a colour |
| `REFERENT_TTL_S` | 90.0 | how long "it" keeps meaning the same car |
| `OPENAI_VISUAL_MAX_TOKENS` | 700 | at 300 with `low` effort the reasoning pass ate the budget and the reply came back empty |
| `OPENAI_VISUAL_REASONING_EFFORT` | `low` | the one thing in this product that benefits from a thinking pass |
| `MEMBER_*`, `MERGE_*` | unchanged | this document changes nothing in `headway/` |

---

## §11 Measured, on this pod

Per `/headway_frame` (unchanged by this work): **~20 ms** — lanes 2, depth 7,
detect 5, membership 1, decode 5.

Per visual question, warm:

| stage | scene | object | follow-up |
|---|---|---|---|
| route | ~0 ms (rules) | ~0 ms | ~0 ms |
| frame selection | 25 ms | 26 ms | 23 ms |
| reference resolution | — | 0.2 ms (geometry) | — (referent reused) |
| crop | — | 16 ms | 17 ms |
| Qwen enrichment | — | 340 ms | 0 ms (cached) |
| GPT-5.5 first token | 1.6 s | 1.4 s | 0.9 s |
| GPT-5.5 total | 2.7 s | 2.4 s | 2.0 s |
| **end to end** | **2.7 s** | **2.8 s** | **2.0 s** |

Add ~0.5–1 s of Whisper at the front and the first TTS chunk at the back for the
spoken path. Everything before the model call is **under 400 ms**; the answer is
dominated by generation, which is where it should be.

---

## §12 What Phase A does not do

Deliberately not built, awaiting go:

- **ambiguity clarification** — asking "the blue coupe beside us, or the black
  SUV ahead?" instead of hedging. The `ambiguous` flag and the alternatives list
  already exist and are already logged.
- **lost-object handling** — saying out loud that a car is no longer in view.
  The stored crop and `referent_visible: false` already exist; nothing speaks
  them yet.
- **unsupported-detail behaviour** — `detail_limited` is already computed and
  sent; the explicit behaviour around it is Phase B.
- **`object_comparison`, `location_or_landmark_question`, `read_visible_text`** —
  classified and marked `phase_b`, then handled as the nearest Phase A
  behaviour.

Acceptance tests 4–6 cover exactly these.

---

## §13 Running it

```
python -m tools.visual_selftest                  # acceptance 1-3, real frames
python -m tools.visual_selftest --no-model       # grounding only, no tokens
node tools/nav_selftest.js                       # the arbiter, incl. P4
python -m headway.selftest                       # the safety core, unchanged
```

Endpoints:

```
GET  /scene?session_id=…          the live scene graph
POST /ask?session_id=…            a visual question in text, answered in text
GET  /visual_state?session_id=…   buffer, retention settings, active referent
POST /talk?session_id=…           unchanged contract; now routes, and returns
                                  X-Request-Type
```
