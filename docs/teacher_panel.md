# The teacher panel — two driving models watching, and driving nothing

*Design note. Companion to `docs/visual_qa.md` (the observer and look()),
`docs/live_headway_v3.md` (the fast loop) and `docs/frame_transport.md` (how a
picture gets here).*

---

## 0. What this is, in one paragraph

Two NVIDIA autonomous-driving foundation models — **Alpamayo 1.5**
(`nvidia/Alpamayo-1.5-10B`) and **Cosmos-Reason2-8B** — read the same road RIO
is driving on, from the same frames, at the same instants, roughly every two
seconds while a drive is live. What they say is shown side by side on the
dashboard, pinned back to the RF-DETR tracks they name, and written to a corpus
on the volume. Nothing they say reaches the arbiter, the speech path, `look()`
or the observer cache. Qwen3-VL-8B is still RIO's observer and still the only
model whose sentences she may speak.

They are called *teachers* because that is what the corpus is for. Today they
are a second opinion to look at; the intent is that a labelled record of where
they and RIO agree and disagree is the raw material for a better observer.

---

## 1. Why this exists

"Was RIO right about that junction?" currently has one answer: a human watches
the clip back. That does not scale past a handful of drives, and it does not
produce anything a later training run can use.

The two models here are built for exactly that question. Alpamayo 1.5 is a
vision-language-action model with a Chain-of-Causation reasoning trace and a
6.4-second trajectory head; Cosmos-Reason2 is a physical-reasoning VLM that was
post-trained on what moves, what happens next, and whether that is plausible.
Run them on RIO's own frames, at RIO's own instants, with RIO's own
deterministic state recorded beside them, and three previously unanswerable
questions become arithmetic:

* Did the models see the hazard RIO's geometry found?
* Did they see one it missed?
* What does a good spoken line look like at an instant like this?

## 2. Why it is shadow, and what "shadow" is enforced to mean

A control that influences the thing it is measuring is not a control. If a
teacher's opinion could nudge a band, break a tie, or seed a sentence, the
corpus would be a record of the panel talking to itself and the comparison
would be worthless.

So the isolation is not a convention, it is a test.
`tools/teacher_firewall_selftest.py` walks the syntax tree of every module in
the repo and asserts:

| # | Rule | How it is checked |
|---|------|-------------------|
| A | Nothing gets in | Every module under `teachers/` may import only stdlib, `config`, `numpy`, and three camera constants from `headway.anchor`. A **whitelist**, so a new forbidden module cannot be forgotten. |
| A2 | The services know nothing about RIO | No module under `teachers/service/` imports `config`, `observer`, `framebuf`, `headway`, … |
| B | Nothing gets out | Only `app.py` may import `teachers` at all. `observer.py`, `realtime.py`, `router.py`, `visual_qa.py`, everything in `headway/` and `navigation/` may not so much as *mention* a teacher. |
| C | The one door is narrow | Every reference to the panel in `app.py` must sit inside a named allowlist of functions — the two frame handlers, the read-only endpoints, the lifespan, the teardown. |
| D | The door is one-way | The speech path may tell the panel what was said (`note_spoken`). The panel has no counterpart pointing back, and the read-only endpoints touch nothing that can speak. |
| E | The browser half | `static/rio_teachers.js` may not name `RIO.speak`, `RIO.output`, `RIO.realtime` or the arbiter's speaking methods. (`RIO.speech.onEvent` is a *listener* on a line already playing, and is the one allowed touch — it is how RIO's own words reach the corpus.) |

The test is proved to fail: adding `import observer` to `teachers/schema.py`,
or `from teachers import panel` to `observer.py`, breaks it.

## 3. Where the models live

They do not live in RIO's interpreter, and they cannot:

| | Python | torch | transformers |
|---|---|---|---|
| RIO | 3.11 | 2.11.0+cu128 | 4.57.x |
| Alpamayo 1.5 | **3.12** | **2.8.0** | **4.57.1** |
| Cosmos-Reason2 | **3.12** | **2.9.0** | **4.57.3** |

Three environments, three processes, two loopback ports (8801, 8802). RIO's own
runtime environment is untouched — the selftest asserts that importing the whole
panel pulls in no `torch`, no `transformers` and no `alpamayo1_5`.

```
/workspace/teachers/src/{alpamayo1.5,cosmos-reason2}   pinned upstream checkouts (volume)
/opt/teachers/venvs/{alpamayo,cosmos}                  the environments (container layer, rebuilt by boot.sh)
$HF_HOME/hub/models--nvidia--*                         the weights (volume)
/workspace/teachers/fp8/                               the FP8 checkpoints (volume)
/workspace/teachers/secrets.env                        the HF token — outside the git worktree, always
```

The environments are on the **container layer** on purpose: they are ~14 GB
each, they are derived from a lockfile, and `boot.sh teachers-build` rebuilds
them — exactly like the pip packages and torch wheels in steps 3 and 4. The
weights and the FP8 checkpoints are on the **volume**, because they are not
derived from anything cheap.

### Pins

| | repo | commit |
|---|---|---|
| Alpamayo weights | `nvidia/Alpamayo-1.5-10B` | `7aba8293c09993f2e125c6819df05d7fa3e873ea` |
| Alpamayo code | `NVlabs/alpamayo1.5` | `36aeb4c5938cbc2eb2aed33b22434773da4ab639` |
| Cosmos weights | `nvidia/Cosmos-Reason2-8B` | `a9fae2cf89dc64db96b12860417f0eb403013bb9` |
| Cosmos code | `nvidia-cosmos/cosmos-reason2` | `a3b4a1db4065fe13c4b1f4d2fb8605bad647f4b9` |

A teacher whose weights or inference code moved is a corpus whose newer rows
are not comparable with its older ones, and the only way to notice is to have
written the sha down. Every reading carries both.

### The gate

`nvidia/Cosmos-Reason2-8B` is a gated repository. **Alpamayo needs it too** —
its `config.json` names `vlm_name_or_path: nvidia/Cosmos-Reason2-8B` and
`ReasoningVLA` builds its tokenizer and VLM config from that repo at load time.
So one accepted licence unlocks both teachers, and without it neither loads.

## 4. The shared input

The unit is a **keyframe**: one instant, one four-frame window, one ego
history, two readings, one corpus row. Both teachers get the identical payload
— the same base64 JPEGs (the client's own bytes, never re-encoded, so the
teachers and RF-DETR are looking at the same pixels), the same `t0`, the same
prompt set.

### The window

Four frames at `t0-0.3, t0-0.2, t0-0.1, t0`. Not arbitrary: that is what
Alpamayo's own dataset loader builds (`num_frames=4, time_step=0.1`) and
therefore the temporal spacing the model was trained to read.

The ring holds whatever the transport delivered, so each slot takes the nearest
frame and the row records how far off it was:

* **8–15 fps** (the socket): worst slot error ≈ 50 ms → `exact: true`
* **4 fps** (the POST fallback): worst slot error ≈ 100 ms → `exact: false`

The tolerance is 60 ms, derived rather than tuned: past half a step the nearest
frame to one slot is nearer to its neighbour, and the "window" stops being four
evenly spaced instants. An inexact window is still sent — it is still a real
0.3 s of road — it just may not silently mean the exact one.

### The ego history

16 poses at 10 Hz ending at `t0`, expressed in the ego frame **at** `t0`, so
the last pose is always the origin. Again Alpamayo's own shape
(`num_history_steps=16`). Frame convention: **FLU** — x forward, y left, z up.

A phone cannot measure position. GPS at 1 Hz with 5–10 m of noise, differenced
over 1.6 s, is mostly noise. So position is **dead reckoned** from two things
that are good over short windows:

* **speed** — the server's own resolver (OBD > GPS), which is deliberately the
  same number the headway band rests on. An ego history built from a different
  speed than the warning was computed from would make the corpus disagree with
  itself.
* **yaw rate** — the phone's IMU, resolved **against gravity**:

  ```
  yaw_rate = ω · up,     up = −g/|g|,     g = accelerationIncludingGravity − acceleration
  ```

  Nobody mounts a phone squarely. Reading `rotationRate.alpha` and calling it
  yaw is wrong by an unknown angle that changes when the driver adjusts the
  cradle. Projecting the rotation-rate vector onto the true vertical is right
  for any mounting, including a phone lying face-up on the passenger seat.

  On iOS 13+ `DeviceMotion` needs a permission that can only be requested from
  a user gesture, so it is asked for on the **Start Drive** tap and nowhere
  else. Declining is a supported drive: the history falls back to speed alone,
  every row says `yaw_rate: "none"`, and nothing else changes.

Samples go to `POST /teachers/ego` at ~20 Hz in one batch a second —
deliberately *not* on the frame socket, whose size and cadence are being
controlled to hold frame age down.

### Alpamayo and a single forward camera

**Supported, canonical, and not an adapter.** `camera_indices=[1]` is
`camera_front_wide_120fov`; `notebooks/inference_cam_num.ipynb` runs "1 cam
(front wide)" as one of its three configurations and
`notebooks/inference_vqa.ipynb` uses a single front wide camera throughout.
`helper.create_message` builds the exact prompt scaffolding the model was
trained on for that configuration.

It is still a **degraded** input, and the model card says so: accuracy falls
with fewer cameras, most where cross-traffic matters (NVIDIA's own example is a
right turn across traffic). Every reading carries `cameras: 1` so a corpus
reader knows which it is looking at. No adapter was needed and no output is
marked non-canonical.

## 5. Cadence

| trigger | why |
|---|---|
| **event** — band change, nav maneuver, driver question, IMU jolt | the instants a review actually wants a second opinion on, and the ones a fixed timer is least likely to land on |
| **floor** — every ~2 s while a drive is live | "nothing happened" is a class the training set needs as much as the near-misses |
| **minimum gap** — 1 s, applied to every trigger | band transitions cluster; without it six keyframes are built in a second and five are dropped stale |

Frame-age gate: 1 s, enforced **twice** — once when the keyframe is built, and
again when the job is picked up by the worker. The second one is where the
seconds actually accumulate, because a job sits in a slot for as long as the
previous inference took.

**Drop, never queue.** One job in flight, one waiting, newest wins. The same
rule `rio_frames.js` and `headway_ws` run on, for the same reason: a queued
keyframe is measured against a road that has already gone past.

### The fast loop does not move

`tools/teacher_timing_selftest.py`, measured on this pod with the real
pipeline (RF-DETR, Depth-Anything, UFLDv2, tracker, filter) and both teacher
queues saturated:

| | p50 | p95 |
|---|---|---|
| hook cost, ordinary frame | 0.004 ms | 0.011 ms |
| hook cost, keyframe frame | 0.31 ms | 0.55 ms |
| pipeline, panel off | 21.35 ms | 27.03 ms |
| pipeline, panel on (queues full, evicting) | 21.20 ms | 27.99 ms |

Conditions are interleaved rather than run in two blocks, because this GPU is
shared and a run measured entirely before another run is measured against a
different machine.

## 5b. One `Conv3d`, and 99.9% of a vision-tower forward

The first thing the real weights said was that Cosmos took **56.8 s** per
keyframe. On a 2 s floor that is not a slow teacher, it is an empty column:
every keyframe would have been evicted before it was ever run.

Generation length was not the cause — it stops correctly at `<|im_end|>` after
seventeen tokens. Per-token speed was not the cause either — text-only
generation runs at 31 tok/s. A single **prefill** through the vision tower took
15.0 s, and inside that:

| | |
|---|---|
| 27 transformer blocks | 16 ms |
| positional embedding | 2 ms |
| rotary table | 0.5 ms |
| merger | 0.1 ms |
| **`patch_embed`** | **13,700 ms** |

`patch_embed` is an `nn.Conv3d` whose `kernel_size` **equals** its `stride` and
equals the whole spatial extent of each input volume: 2400 separate
`(3, 2, 16, 16)` blocks, each producing exactly one output voxel. cuDNN picks a
pathological algorithm for that shape on this card, and
`torch.backends.cudnn.benchmark = True` does not help (measured: 13.8 s).

With no overlap, no padding and no dilation, that convolution **is** a matrix
multiply, by definition — one dot product per block between the flattened block
and the flattened kernel. Reshaping the weight from `(E, C, T, P, P)` to
`(E, C*T*P*P)` and the input to match makes it one GEMM: **0.06 ms**.

| on the synthetic bench keyframe | before | after |
|---|---|---|
| Cosmos-Reason2 | 56.8 s | **2.6 s** |
| Alpamayo 1.5 | 6.1 s | **3.6 s** |

Those are bench-frame numbers, and a bench frame is a flat two-tone rectangle
that neither model has much to say about. On a **real** 1282×684 road clip the
same fix leaves Alpamayo at ~4–5 s and Cosmos at ~32 s, because a real scene
produces real answers — see §6b, which is about that 32 s rather than about
this fix.

Applied to both — Alpamayo's backbone *is* a Qwen3-VL and has the same layer —
through one shared helper, so the two teachers cannot end up with
differently-shaped vision paths.

It is **checked, not asserted**. `common.flatten_patch_embed` runs both forms
against random input at load and refuses the swap if they disagree beyond bf16
rounding; it refuses outright if the conv overlaps, pads or dilates, because
then it is a different operation. `/health` reports what it did and by how much
the forms differed, and the selftest asserts equality *first* and speed second.
A fast layer that quietly computed something else would be invisible in every
reading afterwards and would look exactly like the model being worse at
driving.

## 6. Outputs

Per keyframe, per model, **every field the model offers, verbatim**. `raw` is
the model's own output before any of this code touched it, and it is what a
training run should read; the parsed fields beside it are for the dashboard.

**Alpamayo 1.5**
* Chain-of-Causation trace and the 6.4 s trajectory — from **one**
  `sample_trajectories_from_data_with_vlm_rollout` call, because the trace and
  the path are the same act of reasoning and asking separately would produce
  two that do not correspond. 64 waypoints at 10 Hz.
* `meta_action`
* The three prompt-set answers, via `generate_text` (frames only, no ego
  history — which is why they still work on a drive with no speed signal).

**Cosmos-Reason2**
* Physical reasoning with its `<think>` trace kept separately from the answer.
* The same three prompt-set answers.
* No trajectory: the row says `null`, so the card's path control can be
  honestly disabled for that column rather than silently missing.

**The prompt set, word for word, both models:**

1. `Describe the scene.`
2. `Which single road user is the most safety-relevant to the ego vehicle right now, and why?`
3. `What should the ego vehicle pay attention to in the next 2 seconds?`

A prompt tuned for one of them would make the comparison a comparison of
prompts. The one question that is *not* shared is the reasoning question —
each model is asked for a trace of the kind it was built to produce, which is
the comparison worth making.

Plus, on every reading: `latency_ms` (the model's own time, load and JPEG
decode excluded), `precision`, `freshness_s` (t0 to the reading coming back),
`queue_ms`.

### The trajectory is display-only

It is drawn as a faint tapering ribbon and used for nothing. The projection
uses `headway/anchor.py`'s **own** pinhole constants (`teachers/project.py`), so
the ribbon and the ego corridor cannot disagree about where the road is — a
ribbon through a second camera model would sit visibly beside the corridor
being wrong. The selftest asserts the constants still match. The metric path is
kept in the corpus beside the pixels so a later reader can re-project it
through a better camera model.

## 6c. "Describe the scene." does not describe the scene

Asked the obvious question — the one NVIDIA's own VQA notebook uses — Alpamayo
returns a string out of the **scenario-label taxonomy it was trained on**:

```
"A vehicle controls loss. The vehicle driver is in distracted driving."
"Lead vehicle stops. The vehicle driver is in distracted driving."
"A vehicle changes lanes with the same direction to ego-car/another vehicle.
 Vehicles do not notice the coming vehicles when turning or changling lanes"
```

Eight of ten keyframes on the acceptance clip returned the first of those
**verbatim**, on a stretch of motorway with no lane change, no stopped lead and
no distracted driver in it. Two tells give it away, and both are in the
characters rather than the words: **non-breaking spaces** between them, and
**"changling"** — a typo carried out of a label spreadsheet. It happens on
every sample, seeded or not, and `num_samples=4` returns four taxonomy entries
rather than four descriptions.

This project has met the same failure from the other side. The observer's
prompt once ended with four example sentences and Qwen returned the first one
whatever the frame was (`rio_prompts.is_prompt_example`, `vision.parroted`).
Same shape, different memory: a prompt's examples there, a training set's
labels here — and in both cases a fluent, plausible, well-formed sentence about
nothing in the picture.

**Rewording fixes it completely.** Same model, same four frames, one question
changed:

| question | Alpamayo's answer |
|---|---|
| `Describe the scene.` | *"A vehicle controls loss. The vehicle driver is in distracted driving."* |
| `Describe the road, the traffic and the weather.` | *"The scene shows a clear day with good visibility… a multi-lane highway with a concrete divider…"* |
| `Describe what is on the road ahead.` | *"A vehicle controls loss…"* (canned again) |
| `What do you see?` | *"I see a three-lane highway, some trees, and buildings in the distance…"* |

The attention question had the same problem, more quietly: the shipped wording
returned *"The vehicle should pay attention to vehicles changing lanes"* on nine
keyframes out of ten. Asking for the hazard **and where it is** gets a real
answer from both models — and gives the actor association something to match.

So the prompt set was rechosen, A/B'd against **both** models on the same
frames, for wording that elicits perception rather than recall:

| | |
|---|---|
| scene | `Describe the road, the traffic and the weather.` |
| critical actor | `Which single road user is the most safety-relevant to the ego vehicle right now, and why?` (unchanged — it always worked) |
| attention | `What is the main hazard the ego vehicle should watch in the next two seconds, and where is it?` |

It is still **one prompt set, asked of both models word for word**, which is the
property that matters. What changed is the wording, not the symmetry.

### And a guard, so it cannot come back quietly

`teachers/canned.py` flags a reading that looks **recited rather than
observed**. It does not judge whether an answer is *true* — nothing here can,
and a checker that guessed would be worse than none. It reports two things it
can actually see:

* **markers** — non-breaking spaces, thin spaces, control characters. Prose
  generated by a language model does not contain them; text pasted from a
  spreadsheet into a label file and memorised does.
* **repetition** — the same answer, byte for byte, on three or more consecutive
  keyframes. Model-agnostic and needs no list of known phrases, which is the
  point: the next memorised taxonomy will not be one anybody has written down.

Tracked per model **and per field**, because Alpamayo's critical-actor answers
were specific and correct on the very keyframes where its scene answers were a
fixed string.

A flagged reading is still shown and still recorded — **flagging is not
filtering**, and a recited answer is real evidence about the model. The card
puts a small *recited?* next to the field label, and the corpus row carries the
structural evidence under `flags`.

## 6b. Cosmos is a thirty-second teacher, and that is the right trade

Measured on a real road clip, after the `patch_embed` fix:

| question | time | output |
|---|---|---|
| physical reasoning | 28.4 s | 4275 chars |
| scene | 2.2 s | 399 |
| critical actor | 1.0 s | 149 |
| attention | 0.9 s | 152 |

One question is 88% of the model's time, and it runs to the full 1024-token
budget. That is **not** waste: those 4275 characters *are* the physical-
reasoning trace this panel exists to collect, and truncating them to fit a 2 s
floor would be spending the GPU to produce a worse version of the one output
nobody else here can give us.

The consequence, stated plainly because it is a design choice rather than a
bug: **at the shipped 2 s floor Cosmos answers roughly one keyframe in fifteen**
and the rest are evicted — newest wins, so it is always working on the most
recent road rather than catching up through a backlog.

That matters less than it looks, because of an asymmetry worth naming: Alpamayo
is faster and gets the *same* jobs, so **every keyframe Cosmos answers is one
Alpamayo answered too**. Every Cosmos reading is therefore a comparison; there
are simply fewer of them. Over a twenty-minute drive that is roughly forty
side-by-side readings.

`config.TEACHER_COSMOS_MAX_NEW_TOKENS` is where the trade lives: lower it for
more readings and shorter traces. It is an operator's decision, and the numbers
above are in the config comment so it can be made knowingly.

**For an acceptance run**, `tools/teacher_replay.py --floor N` paces keyframes
to the slower teacher so every one of them is a comparison. That is not a
drive's cadence and the report says which it is looking at.

## 7. Actor association

Both teachers name a road user in English. The deterministic pipeline, at the
same instant, has tracked boxes with classes, ranges and lane membership.
`teachers/associate.py` is the bridge, and it is **word lists and geometry**,
not a model.

The obvious implementation is to ask an LLM which box it meant. That would make
the association as opaque as the thing it is checking, and it would not be
reproducible: re-running the corpus in a year would produce different
associations from the same rows.

Scoring, stated rather than tuned: class 0.55 (0.22 for large-vehicle
confusion, which is the most common error in both directions), side 0.28,
in-corridor 0.28 for "ahead", then range, box size and lead-lock as ordered
tie-breaks. Floor 0.45 — a class match alone passes, a pile of tie-breaks does
not.

**Abstaining is a result.** `matched: false` with a reason is the honest answer
for "a pedestrian on the far pavement" when the detector found none, and the
reasons are specific: `names_undetectable_class:traffic light` (RF-DETR has six
classes and a traffic light is not one), `opposing_traffic_not_tracked`
(nothing here knows which way a box faces), `no_track_of_class:cyclist`,
`below_match_floor`. A forced match to the nearest car would be a silent lie in
a training corpus, which is the worst place to put one.

**Two abstentions are not agreement.** Counting them as one would make the
tally strip read best when the association is working worst.

## 8. The dashboard

A new card in the centre stage, directly under the picture, above Perception —
and on a phone, directly under the picture too (`order: -1`, so it beats Drive
and Navigation). The existing Perception card is left exactly where it is, so
Qwen's running description sits on the same screen for comparison.

Two columns, `Alpamayo 1.5 · Cosmos-Reason2`, left to right always. Each
carries: Scene, Chain-of-Causation / Physical reasoning, Attention (next 2 s),
Referenced actor, freshness, latency, precision. A tally strip underneath:
keyframes processed, agreement on the critical actor, matched-track rate,
keyframes dropped. A service strip under that: which weights, at which
precision, and whether the process is answering.

### New design tokens

```
--teach-a: #a78bff   Alpamayo    (violet,  hue 262)
--teach-c: #ef6fc4   Cosmos      (magenta, hue 320)
--teach-ribbon: rgba(167,139,255,0.34)
```

Chosen for **distance from every existing hue** rather than for taste — against
`--accent` 216, `--good` 152, `--warn` 38 and `--danger` 8. A teacher is
neither a control the driver can press nor the car's opinion of itself, and it
must never read as either. They are also deliberately dimmer than the safety
colours: a shadow reading must not be the brightest thing on a driving surface.
The card selftest asserts that neither column is drawn in a safety colour.

### On the picture

The matched RF-DETR track is ringed in a **per-model style as well as colour**,
so the two are told apart by anyone who cannot separate the hues: Alpamayo a
dashed ring 6 px outside the box, Cosmos a dotted ring 2 px inside it. Both
thinner than the detector's own stroke, and neither ever in a band colour.

The ring follows the **track id**, not the box the teacher was shown — the
reading is a second or two old and the car has moved, so ringing the stale box
would put the ring beside the object.

Three switches under the Layers control in the Drive card: *Alpamayo actor*,
*Cosmos actor*, *Predicted path*. The path is **off by default** — it is a
6.4 s guess through a flat-road camera model and it should be asked for. The
whole row is hidden when the master Layers toggle is off, because a switch for
a layer that cannot be drawn is a switch that lies.

## 9. The record

`training_data/teachers/<session_id>/keyframes.jsonl`, one JSON object per
line, with the four JPEGs beside it under `frames/<kf_id>/{0..3}.jpg` (relative
paths — an absolute path in a corpus stops working the moment the corpus is
copied, which is the only thing corpora ever have done to them).

A row carries: the window (with per-frame ids, timestamps, slot offsets and
exactness), the ego history and its provenance, **both** models' complete
outputs, both associations, the RF-DETR scene at t0 (so an association can be
re-checked without the pictures), RIO's deterministic state at t0 (band, gap,
TTC, tau, trend, speed and its source, lead id, corridor source), and RIO's
spoken line if there was one.

The schema is declared in `teachers/schema.py`, versioned, and asserted by
`tools/teacher_selftest.py` against **real** rows. `validate_row()` returns a
list of complaints rather than raising, because the writer never validates — a
row that is wrong should still be written, or the evidence of the bug goes with
it.

Rows are assembled **by keyframe id**, and written when every model has
*reported* on that keyframe — answered it, failed it, or been told by its own
client that the job was thrown away. That last case is why eviction and
stale-drop report back rather than dropping silently: without it a row would
wait forever for an answer that is never coming, and the *other* model's real
reading of the same instant would go with it.

It has to work that way because **the two teachers do not stay in step**. Each
has its own queue and its own newest-wins eviction, so a fast one runs
keyframes 10, 11, 12 while a slow one runs 10 and then 13. A rule of "write
when both models' latest readings name the same keyframe" is true for the first
few seconds of a drive and almost never again — the corpus would quietly thin
to nothing.

So `models_ran` is on every row. Usually both; not always. A row with one name
still carries a real reading of a real window and is worth keeping — an
analysis that wants *comparisons* filters on `len(models_ran) == 2` rather than
discovering the difference by surprise. A model that ran and *failed* writes
`ok: false` with its error, which is data; a model that never ran writes
`error: "evicted"` or `"stale_dropped"`, which is a different fact.

~100 kB a keyframe, ~180 MB an hour at the 2 s floor. Kept, because a corpus
without the pictures is a corpus of opinions — **except** when neither teacher
produced a reading. That row is still written (it records that the panel was
down, and with which error) but its four JPEGs are not: with both services
stopped, keeping them would be 180 MB an hour of road photographs whose entire
accompanying content is "connection refused". The row says `frames_kept: false`
rather than leaving a reader to infer it from the null paths.

## 10. Running it

```bash
bash boot.sh teachers-build     # the two isolated environments (~28 GB, container layer)
bash boot.sh teachers           # start both services, wait for /health
bash boot.sh teachers-stop
bash boot.sh status             # includes both services, their weights and their VRAM

python -m tools.teacher_firewall_selftest      # the shadow guarantee
python -m tools.teacher_selftest               # shapes, refusals, isolation, the record
python -m tools.teacher_input_selftest         # input contracts, the patch-embed rewrite and the FP8 recipe, in their own venvs
python -m tools.teacher_selftest --live        # ...and one real keyframe through both
python -m tools.teacher_timing_selftest        # the fast loop does not move
python -m tools.teacher_timing_selftest --gpu-load   # ...with both models on the card
python -m tools.teacher_card_selftest          # the card, in a browser, against a fixture
python -m tools.teacher_bench --all --report /tmp/teachers.md   # VRAM and latency
python -m tools.teacher_replay --clip runs/road_clip.mp4 --keyframes 10
```

FP8, by NVIDIA's recipe:

```bash
/opt/teachers/venvs/cosmos/bin/python   -m teachers.service.quantize --model cosmos
/opt/teachers/venvs/alpamayo/bin/python -m teachers.service.quantize --model alpamayo
```

Cosmos shells out to `scripts/quantize.py` from the pinned cosmos-reason2
checkout through `uv run --script`, which resolves that script's own locked
dependency set — the recipe has pins for a reason and re-deriving it by hand is
how a quantization silently stops matching the one the model was validated
with. Alpamayo has no vendor script (it is a composite model: a Cosmos-Reason2
VLM with a diffusion action expert), so the same llmcompressor recipe is
applied to the part it applies to, with `FP8_DYNAMIC` because static
calibration needs a forward pass the composite model's signature does not
provide.

**What `precision: "fp8"` means on a reading:** the language backbone's Linear
weights are FP8. Not the vision tower, not the KV cache, and for Alpamayo not
the diffusion action expert — that stays BF16, so the predicted trajectory is
computed at full precision from a quantized trace. For Alpamayo that is 398
modules quantized and 581 excluded (176 vision tower, 397 expert, 7 action
projections, 1 `lm_head`).

Written down because "FP8" on a dashboard is otherwise a claim nobody can
check — and because the first build got it wrong. The ignore list was adapted
from NVIDIA's Cosmos recipe, where the vision tower is `visual.*`; in Alpamayo
the whole VLM is nested one level down and it is `vlm.model.visual.*`.
`compressed_tensors` matches with `re.match`, which is **anchored at the start**,
so `re:visual.*` matched nothing and the first checkpoint quantized all 176
vision-tower layers and all 397 expert layers while its own `config.json`
recorded a five-entry ignore list.

The selftest did not catch it because it simulated the matcher — `re.search`
against module names the test itself made up. It now uses
`compressed_tensors.utils.match._match_name`, the library's own function,
against the module names read out of the shipped `model.safetensors.index.json`.
A recipe check that invents its own inputs is not a check.

`boot.sh teachers` picks FP8 automatically when a checkpoint is on the volume
and BF16 otherwise, so the L40S pod loads the quantized build and this one does
not have to.

## 10b. VRAM and latency — and why FP8 is off by default

Measured on the H200 with `tools/teacher_bench.py --all`, one configuration at
a time, over a fixed synthetic keyframe, first run discarded:

| model | precision | weights | held between keyframes | peak in inference | p50 | p90 | load |
|---|---|---|---|---|---|---|---|
| Alpamayo 1.5 | bf16 | 21408 MB | 21408 MB | 21756 MB | 2857 ms | 3039 ms | 60 s |
| Alpamayo 1.5 | fp8 | 14784 MB | 14784 MB | **25932 MB** | **18905 ms** | 19286 ms | 45 s |
| Cosmos-Reason2 | bf16 | 16760 MB | 16760 MB | 17040 MB | 3633 ms | 4622 ms | 48 s |
| Cosmos-Reason2 | fp8 | 10146 MB | 10146 MB | **19798 MB** | **5906 ms** | 7075 ms | 33 s |
| **both** | **bf16** | **38168 MB** | | **38796 MB** | | | |
| **both** | **fp8** | **24930 MB** | | **45730 MB** | | | |

**FP8 makes both teachers worse here, and the weight saving is real but
useless.** The checkpoints are correct — Alpamayo's weights drop 21.4 → 14.8 GB
and Cosmos's 16.8 → 10.1 GB — but they are served through plain
`transformers` + `compressed-tensors`, which **dequantizes on the fly**. So the
peak *during inference* goes **up** (21.8 → 25.9 GB, 17.0 → 19.8 GB) and
latency goes up with it: Cosmos 1.6× slower, Alpamayo **6.6×** slower.

That is not a defect in the recipe. NVIDIA's own README says the deployment
path is vLLM (`vllm>=0.11.0`), where the FP8 kernels are fused; the recipe is
aimed there. In this serving path it is a pessimisation, so `boot.sh` now
defaults to **BF16 even when an FP8 checkpoint exists**, and FP8 is opt-in via
`TEACHERS_PRECISION=fp8`. The checkpoints stay on the volume because the moment
either teacher is served through vLLM they become the right thing to load.

Three VRAM numbers, not one, because they answer different questions — and
because reporting only the peak initially made FP8 look like it needed *more*
card than BF16 even at rest, which is true of the load and false of the drive.
`max_memory_reserved` is a high-water mark that is never reset, and loading an
FP8 checkpoint spikes hard while `compressed-tensors` materialises each layer.
The service resets the peak once the weights are down, so every number after
that describes the drive.

### The L40S answer

48 GB, of which RIO's live stack (Qwen3-VL-8B, Depth-Anything, UFLDv2, RF-DETR)
holds about 18 GB — roughly **30 GB available**. A budget is a *peak*, not a
weight: a model that holds 10 GB and balloons to 20 GB during inference needs
20 GB of card.

| | peak | fits in 30 GB? |
|---|---|---|
| both, bf16 | 38.8 GB | **no** |
| both, fp8 | 45.7 GB | **no** |
| Cosmos alone, bf16 | 17.0 GB | yes |
| Alpamayo alone, bf16 | 21.8 GB | yes |

**Both teachers do not fit on an L40S beside the live stack, at either
precision, and FP8 does not change that** — it makes it worse. What fits is
*one* teacher, at BF16. Cosmos alone leaves the most headroom (17.0 GB of 30);
Alpamayo alone fits with 8 GB spare and is the one that also predicts a
trajectory.

The panel already degrades correctly into that shape: a teacher that is not
running shows as not answering on the card, `/health` reports it degraded, and
every corpus row records `models_ran` so a one-teacher drive is
self-describing rather than silently different.

## 10c. The container layer, and the two times it filled

Twice in this build the 60 GB container layer filled completely and took the
box with it: no writable temp space, so nothing that could have diagnosed the
problem could run either, and it needed a terminal outside the harness to
clear.

Both times, `uv`. Its download cache **and** the ephemeral environment a
PEP-723 `uv run --script` resolves into live under `UV_CACHE_DIR`, which
defaults to `~/.cache/uv` — on the layer that already holds ~28 GB of teacher
venvs. One torch unpack is ~10 GB.

The first fix set `UV_CACHE_DIR` in `boot.sh` and `/workspace/env.sh`. That is
right and **was not enough**: it only reaches a process whose shell sourced one
of them, and the second time the layer filled, it filled *from a test* —
`tools/teacher_input_selftest.py` shells out to `uv` inheriting `os.environ`,
and run from a plain shell the variable was simply absent.

An environment variable somebody else has to export is a convention, not a
setting. So `teachers/paths.py` holds the values, and every subprocess in the
project builds its environment from `paths.subprocess_env()` — asserted in the
selftest against a *scrubbed* environment, which is exactly the case that
failed.

| | |
|---|---|
| uv cache + script environments | `/workspace/teachers/uv-cache` (volume) |
| weights | `/workspace/.cache/huggingface` (volume) |
| FP8 checkpoints | `/workspace/teachers/fp8` (volume) |
| pinned checkouts | `/workspace/teachers/src` (volume) |
| the two venvs | `/opt/teachers/venvs` (container layer, rebuilt by boot.sh) |
| HF token | `/workspace/teachers/secrets.env` (volume, outside the worktree) |

## 11. What is deliberately not here

* **No training.** The corpus is schema'd so it can be one later.
* **No arbitration.** A teacher never breaks a tie, seeds a sentence, or
  changes a band.
* **No second observer.** `look()` reads Qwen and only Qwen.
* **No trajectory in the loop.** The 6.4 s path is a picture.
