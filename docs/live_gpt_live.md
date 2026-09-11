# RIO on gpt-live-1 — a voice that doesn't think, and a backend that does

**Status: BUILT, MEASURED, NOT DEFAULT.** `VOICE_BACKEND` still ships as
`openai_realtime`. This backend is one env var and a restart away
(`VOICE_BACKEND=gpt_live`), every guard is behind a flag, and
`gpt-realtime-2.1` remains the fallback tier exactly as ElevenLabs does.

The decision this document exists to support is **not** "is gpt-live-1 better".
It is better at the thing it was built for and worse at one thing RIO depends
on, and both halves are measured below.

---

## The architecture

`gpt-realtime-2.1` is one model that hears, thinks, calls tools and speaks.
`gpt-live-1` is a **front end**: it listens and speaks, full duplex, and hands
anything needing a tool or a thought to a **backend text model**.

```
  driver ──audio──▶ gpt-live-1 (gleam)  ──delegation──▶ gpt-5.6-luna
                         │                                   │
                         │◀──── commentary / instructions ────┤
                         │                                   │
                         ▼                            response.event
                      speaker                                │
                                                    ┌────────┴────────┐
                                              RIO's 9 tools    (our server
                                                                and page)
```

### How delegation actually works

Session config, at `POST /v1/live/sessions` (see `live.session_config`):

```json
{"transport": {"type": "webrtc", "sdp": "..."},
 "session": {
   "model": "gpt-live-1",
   "instructions": "<the live prompt — who she is, how she holds a turn>",
   "audio": {"output": {"voice": "gleam"}},
   "delegation": {"type": "responses",
                  "responses": {"model": "gpt-5.6-luna",
                                "instructions": "<the backend prompt>",
                                "tools": [ ...RIO's nine... ],
                                "tool_choice": "auto"}}}}
```

**WebRTC only.** Every other transport is refused: *"Only the webrtc transport
is supported."* And the session config and the SDP offer arrive in **one**
request, which is why `POST /live/session` on our server proxies the
negotiation — it is the only way to keep the account key out of the browser.
One extra hop on connect, off the audio path; the media still flows
browser-to-OpenAI.

**Events, in the direction they travel:**

| event | who sends it | what it means |
|---|---|---|
| `session.started` | server | the config that was actually accepted. **Not in the HTTP response** — reading it from there reports a session with no model, no voice and no backend. |
| `session.input_transcript.delta/.done` | server | the driver, revisable and overlapping |
| `session.output_transcript.delta/.done` | server | her |
| `session.delegation.created` | server | she has decided this needs the backend |
| `response.event` | server | the backend's own stream, **wrapped**. A tool call is two levels down: `event.event.item`. |
| `response.item.create` + `response.create` | us | a tool result, and "carry on". **Both**, always — sending only the first leaves a model holding an answer nobody told it to use, which presents as RIO going quiet after a tool call. |
| `session.commentary.append` | us | something to say **in her own words** |
| `session.instructions.append` | us | a directive — including "say this exactly" |
| `session.usage.updated` | server | `{"seconds": N}`, and nothing else |

`delegation_id` is **required** on all three appends. For anything RIO says on
her own initiative it is `null` — explicitly null. Omitting the key is
`missing_required_parameter` and a car that says nothing.

### What the tool path becomes

Unchanged in substance, rerouted in shape. The backend model chooses the tool;
the call arrives wrapped in `response.event`; the page executes it with **the
same implementations the realtime path uses** (`RIO.realtime.localTools` for
nav, `POST /realtime/tool` for everything else); the result goes back as the
two-event pair. All nine tools round-trip — see `tools/live_selftest.py`.

---

## Which backend, and why

`tools/live_backend_bench.py`, RIO's own nine tool schemas and eleven scripted
utterances, **nine trials each**:

| model | routed | first token p50 | p95 | $/M in | $/M out |
|---|---|---|---|---|---|
| `gpt-5.6-terra` | 95/99 | 790 ms | **1482 ms** | 2.00 | 12.00 |
| **`gpt-5.6-luna`** | **98/99** | **790 ms** | 1564 ms | **0.20** | **1.20** |
| `gpt-5.6-sol` | 33/33\* | 1143 ms | 2520 ms | 5.00 | 30.00 |

\* *sol was dropped after the n=3 screen; it was clearly slowest.*

**Luna.** It routed better, matched Terra's median, and costs a tenth as much.
The documentation calls Terra the primary recommendation and Luna "the
cost-sensitive alternative"; on RIO's actual tool surface that ranking does not
survive measurement. Terra's misses were `nav_status` → `find_places` ("how far
is it to the Getty") and a tool call for a question that needed none.

Terra is better at the **tail** — 1482 ms against 1564. That is real, and this
repository cares about p95 more than most. It is not worth ten times the price
and four more routing misses. `GPT_LIVE_BACKEND_MODEL` is one env var.

---

## Verbatim: the part that nearly sank it

Under `openai_realtime` a dictated line is exact **by construction** —
`response.create` carries the words and the model reads them. gpt-live-1 has no
such event and the API documents **no verbatim mechanism at all**.

The obvious candidate is `session.commentary.append`, and the docs describe it
exactly: information for the model to vocalize, *which it may paraphrase*.

**It does.** `tools/live_verbatim_bench.py`, RIO's real deterministic lines,
three trials each:

| | verbatim | append → first audio |
|---|---|---|
| `session.commentary.append` | **9/21** | p50 758 ms, p95 883 ms |
| `session.instructions.append` | **21/21** | p50 1324 ms, p95 2317 ms |

What commentary did to the lines:

```
  "In half a mile, turn right onto Ocean Avenue."
    → "Half a mile up, take a right on Ocean Avenue."
  "In 300 feet, turn right onto Ocean Avenue."
    → "About 300 feet, take a right onto Ocean Avenue."
  "I've lost the sensor on the rear right tire — that corner's dark to me
   while we're moving."
    → "Hey— I'm blind on the rear right tire sensor right now, so if you…"
```

Read the second one hardest: **"about 300 feet" is a hedge added to a distance
the policy stated exactly.** The third drops "while we're moving", the
qualifier that makes the sentence true rather than alarming. These are not the
paraphrases a passenger makes; they change what was claimed.

`session.instructions.append` is a **directive**, is what the guide gives for
exact wording ("Immediately say the following disclosure exactly…"), and holds
21/21 — including the two long health announcements commentary got wrong every
single time. So deterministic speech goes through `instructions.append` and
conversation goes through `commentary.append`. See `live.verbatim_event`.

**It is still a directive obeyed, not a contract enforced.** The lines where
being wrong is dangerous — the red headway tier, the two tire fast-path lines,
the imminent turn call — do not go through the model under **any** backend.
They play a pre-rendered clip, no network in the path.

### What verbatim costs in milliseconds

570 ms at the median, 1.4 s at the tail. The per-channel budgets in `config.py`
were cut against gpt-realtime's 390–585 ms, so a 900 ms budget in front of a
p50 of 1324 does not mean "fall back if something goes wrong", it means "fall
back every time". Hence a **per-channel floor**
(`GPT_LIVE_SPEAK_TIMEOUT_FLOOR_MS_BY_CHANNEL`):

| channel | floor | why |
|---|---|---|
| nav | 2400 ms | the calls that reach dictation have room in front of the maneuver; the junction call plays its clip **first** |
| health | 2400 ms | an announcement, not an alert |
| headway | **1200 ms, unchanged** | its arbiter item carries a 2500 ms TTL — a gap measured three seconds ago is not a gap |

nav and health buy one voice with patience; headway buys timeliness and accepts
the risk of the fallback tier's voice. That is the argument the per-channel
table has always made, applied to a slower mouth.

---

## Gleam everywhere

`gleam` is a **live-session voice only**. `/v1/audio/speech` rejects it —
*"Supported values are: alloy, echo, fable, onyx, nova, shimmer, coral, verse,
ballad, ash, sage, marin, cedar"* — and gpt-live-1 serves no endpoint but
`v1/live/sessions`. So there is no request that turns a sentence into a Gleam
audio file, and without one the red headway tier would keep playing marin while
the conversation happened in Gleam.

**So a clip is rendered by opening a session and recording her**
(`live.render_speech`, `tools/render_alerts.py --backend gpt_live`). Slow — a
WebRTC negotiation per line — and it does not matter: this runs on a
workstation and the artifact is played from disk months later.

Two things make it safe, and neither is the model's cooperation:

1. **The voice's own transcript must match** before the audio is accepted.
2. **An independent transcription of the finished MP3** must match too, and a
   clip that fails is re-rendered rather than shipped.

Clips are kept **per voice** (`config.CLIP_DIRS`): `static/audio/` is still the
marin set, `static/audio/gleam/` is the new one, and the served prefix follows
`VOICE_BACKEND` (`clip_base`, carried with the session). A backend switch is
therefore a restart, not a re-render, and there is no window where the files and
the session disagree.

### Verifying a clip you cannot listen to

Two checks, and the second one is scoped, which is the whole safety argument.

The finished MP3 is transcribed **cold** — no hint — and must match. That is
right for a sentence and measurably wrong for a one-word clip: "Merge." comes
back as "March.", "Keep left." as "He left.", "Make a U-turn." as "WikiU turn.",
and the voice had said all three correctly. A verifier that refuses a good clip
six times running is not being careful; it deletes a safety clip and calls that
caution.

So for clips of **four words or fewer** the whole library is supplied as a
transcription hint — every line RIO can play, none of them privileged — and the
question becomes "which of these seventeen sentences is this?". A wrong clip
still matches the wrong row.

**Above four words there is no hint, ever.** A long line can be wrong in one
place while the rest agrees, and a hint talks the transcriber round exactly
there. `tire_critical` is the worked example: fifteen words, thirteen of them
perfect, and with the hint in place it passed on the first attempt. Without it,
it fails every time. The two words the hint was papering over were "Pull over".

`tools/clip_verify.py` re-checks a finished library on the same rule. An empty
transcription counts as **unchecked**, not wrong — the transcriber declines on
very short audio often enough that treating silence as a mismatch condemned six
good clips the render-time check had just passed.

**Final state of the Gleam library: 16 rendered and verified, 0 wrong, 1
refused.**

It earned its keep twice.

**It caught a renderer bug.** Recording started at the first *loud* frame,
which cut the attack off the first word. Fixed with a 240 ms pre-roll.

**And it caught a clip that would have shipped saying the wrong thing.** An
earlier version of `_render_live` kept the last render and printed a warning
when the transcriber would not agree, on the reasoning that it is unreliable on
short clips and the voice's own transcript had agreed every time. That
reasoning is wrong, and this is the clip that proved it:

```
  asked: "Pull over when it's safe — one of your tires is dangerously low…"
  said : "Hey, Ava, when it's safe, one of your tires is dangerously low…"
```

The voice's own transcript called that render correct. It was not — **the
instruction to pull over, the entire action the warning exists to request, had
been replaced by a greeting.** A keep-and-flag policy would have written it to
disk and played it at the worst possible moment for months. So an unverified
clip is now **refused**: the file is removed, the run continues, the exit code
is non-zero. A missing clip is not silence — `rio_speak` falls through to
dictation and then the synthesiser. A *wrong* clip has no tier underneath it.

### The one clip Gleam could not render

`tire_critical` — *"Pull over when it's safe — one of your tires is dangerously
low and still going down."* — **does not verify in Gleam**, across ten renders
and both punctuations. The opening phrase comes back as "Hold out for when it's
safe", "Paying is unsafe", "So while not safe", "Hey Alpha, let's see".

This is **voice-specific, not a pipeline bug**, and the control says so: the
shipped **marin** clip of the identical sentence transcribes perfectly first
time. Gleam's rendering of "Pull over" is not reliably intelligible to the same
transcriber that handles marin's without trouble.

It is left **unrendered on purpose**. That line falls back to dictation and then
the synthesiser, which costs a round trip and possibly a different voice on one
critical warning — and is strictly better than a clip that says something else.
Needs a human to listen before anyone decides otherwise.

So "every path is one voice" is true of sixteen of the seventeen clips, the
conversation, and all dictated lines. It is **not** true of that one warning,
and this is the report rather than a silent mix.

---

## The four guards

The claim is that gpt-live-1 handles noise, silence and interruption natively,
+30pp on Full Duplex Bench. Measured against RIO's four guards, the answer is
not the one the benchmark implies.

### Echo gate — **STAYS ON. The benchmark is wrong about this one.**

`tools/live_echo_probe.py` plays RIO's own voice back into the session the way
a phone speaker plays it back into a phone microphone. Eight trials, two
sources, four attenuations:

| attenuation | heard as the driver | she replied |
|---|---|---|
| 0 dB | "Back off now" | "Alright, I'll give you space." |
| −6 dB | "Park off now" | "Got it. Finding the nearest safe parking…" |
| −12 dB | "Back off, now" | "Okay, I'll give you space." |
| −20 dB | "Back off, now" | "Got it. Pausing interaction." |

**8/8.** Full duplex does not mean deaf to itself — it means the opposite: it
listens while it speaks *on purpose*, so it cannot treat everything arriving
during its own turn as noise. Its own WebRTC audio is cancelled by the
browser's echo canceller, which has a reference for it. **The lines that come
back here are the ones the canceller has no reference for** — a pre-rendered
clip and the TTS fallback are ordinary media playback. A new conversation model
does not change what a loudspeaker does to a microphone eight inches away.

### The other three — **OFF, structurally**

Not because the model is better at them, but because **the events they hang off
no longer exist.**

| guard | why it cannot run |
|---|---|
| sustained-speech barge-in | built on `input_audio_buffer.speech_started`, acts by `response.cancel`. gpt-live-1 emits neither and accepts neither. The model owns its turn. |
| fragment coalescing | joins two *committed* utterances. There are no commits — the migration guide's first instruction is to remove them, and transcripts arrive as revisable overlapping deltas. |
| phantom supersede | superseding means cancelling a response. There is no response to cancel. |

**Barge-in was measured working natively.** `tools/live_selftest.py --only
bargein`: she was mid-line on a long dictated sentence, the driver cut in with
"Actually, how are my tires?", and the transcript reads

```
"Let me tell you about the history of this road,Sure, checking on that. All good…"
```

She stopped at "this road,", called `vehicle_status`, and answered. That is the
thing the sustained-speech rule was built to approximate, done properly.

**Nothing is deleted.** Every guard keeps its code, its constants and its tests;
`config.guards()` decides only whether each **runs**, and each has an env
override (`RIO_GUARD_ECHO_GATE`, `RIO_GUARD_BARGE_SUSTAIN`,
`RIO_GUARD_FRAGMENT_COALESCE`, `RIO_GUARD_PHANTOM_SUPERSEDE`). A bench is not a
cabin. The first drive that disagrees needs a flag, not a revert.

---

## Latency

`tools/live_latency.py`, speech-end → first audio, tool-free questions:

| backend | n | p50 | p95 |
|---|---|---|---|
| gpt-live-1 | 12 | **782 ms** | **1468 ms** |
| gpt-realtime-2.1 | 10 | 744 ms | 1866 ms |

38 ms slower at the median, **400 ms better at the tail** — and this file cares
about the tail. Small n; treat as indicative.

One measurement note that is itself a finding: **she greets.** The first sound
on the track is "Hey. What's up." before the driver has said anything, and a
turn timed from it comes back at 98 ms — or −32 ms — for an answer that had not
been thought about yet. The bench times from the first *onset after the question
ends* and discards turns where she was still mid-greeting.

---

## Cost

`tools/live_cost.py`, the seven-question scripted drive, and a silent session:

| | drive (74–77 s, 7 turns) | idle (90 s, nobody talking) |
|---|---|---|
| gpt-live-1 + luna | **$0.0463/min** | **$0.0492/min** |
| gpt-realtime-2.1 | $0.0499/min | ~$0.000/min |

gpt-live-1's backend share is **modelled**, not reported: `session.usage.updated`
carries `seconds` and nothing else, so the same prompt, tools and utterances are
sent to the same backend over the Responses API and that usage is priced.
Labelled as modelled wherever it is printed.

**The drive column is misleading on its own, and this is the important part.**
Seven questions in 74 seconds is 5.7 turns a minute — a rate no driver sustains.
The two backends are priced at opposite ends:

- **gpt-live-1** charges $0.05/min for the session being *open*. Silence costs
  exactly what conversation costs. The idle column is the proof: **$0.0492/min
  for a car in which nobody said anything.**
- **gpt-realtime-2.1** charges tokens, and only on a turn. A quiet minute is
  very nearly free.

Break-even is around **5.5 driver turns per minute**. The scripted drive sits
just above it, which is why the drive column flatters gpt-live-1 by 7%. **A real
drive sits far below it, and there gpt-live-1 is several times more expensive** —
an hour-long drive is $3.00 of session whether or not anyone speaks.

---

## Selftests

```
  node tools/live_controller_selftest.js      # the page's controller, no browser
  python -m tools.live_selftest               # session, tools, verbatim, bargein, arbiter
  python -m tools.live_selftest --quick       # the short version
  python -m tools.clip_verify --backend gpt_live
  python -m tools.live_backend_bench          # which backend model
  python -m tools.live_verbatim_bench         # commentary vs instructions
  python -m tools.live_latency                # against gpt-realtime-2.1
  python -m tools.live_cost [--idle 90]       # $/min
  python -m tools.live_echo_probe             # does it hear itself
```

The arbiter check is deliberately **not** a live test: it shells out to
`tools/one_voice_selftest.js` unmodified, because the arbiter is not supposed to
have changed and the way to prove that is to run its own suite untouched.

---

## What is preserved

- **The deterministic/arbiter boundary.** She answers; the arbiter announces.
  `static/rio_speech.js` is untouched; priorities unchanged; 27 checks pass.
- **Verbatim dictation with its fallback tiers** — through `instructions.append`
  now, with the clip and synthesiser tiers underneath exactly as before.
- **Teacher context in `look()`** — the tool schemas are `realtime.BASE_TOOLS`,
  imported, not restated.
- **Two-tier brevity** — enforced in the tool, not the prompt, and the tool did
  not move.
- **Truthfulness and field-of-view rules** — moved into the **backend** prompt,
  which is the model that now holds the tool output. A rule enforced on the
  model that cannot act on it is not enforced.
