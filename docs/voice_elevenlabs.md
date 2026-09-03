# RIO's voice — ElevenLabs, with GPT-Realtime still doing the thinking

**Status:** implemented, measured against the real models and the real service.
The playback half runs in a browser and has not been driven in a car yet; see
*What is unproven* at the end.

---

## What changed

| | before | now |
|---|---|---|
| hearing | the live session, continuously | **unchanged** |
| thinking | `gpt-realtime-2.1` | **unchanged** |
| speaking | the same session, in `cedar` | ElevenLabs `eleven_multilingual_v2`, voice `weA4Q36twV5kwSaTEL0Q` |
| deterministic lines | dictated into the live session | ElevenLabs `eleven_flash_v2_5`, **same voice id** |
| pre-rendered clips | rendered in `cedar` | re-rendered on `eleven_multilingual_v2`, **same voice id** |

> **The conversation model was v3 first, and v3 was wrong.** The voice is a
> professional clone, and a clone is trained against a set of base models. This
> one publishes eight of them — the whole v2 family, `eleven_multilingual_v2`
> and `eleven_flash_v2_5` among them — and **no v3 model at all**. Asked for one
> outside that set the service does not refuse; it synthesises an
> approximation, so the driver hears a voice that is recognisably not the one
> that was cloned, with no error anywhere. That is only findable by listening,
> which is how it was found. `voice_selftest --live` now asks the question
> directly, against the voice's own `high_quality_base_model_ids`.

The live session is put in **text mode**. Its audio input, its voice-activity
detector, its transcription, its tools and its instructions are untouched — it
still hears the driver and still decides when they have stopped talking. What
changed is what it PRODUCES: words instead of sound, and something else says
them.

**What did not change: everything that speaks deterministically.** Headway
warnings, vehicle-health announcements and navigation instructions are still
written by policy code, still addressed by an id rather than by text, and still
arbitrated exactly as before. The red headway tier and the two tire fast-path
lines still play local files with no network in the path.

---

## The shape of it

```
  driver's audio ──► GPT-Realtime (text mode) ──► text deltas
                          │                            │
                          │ tools, VAD, transcription  │  (browser)
                          ▼                            ▼
                     unchanged                  ws://…/voice/dialogue
                                                       │
                                                  (server, holds the key)
                                                       │
                                        voice_dialogue.PhraseChunker
                                        voice_tags.sanitize
                                                       │
                                    wss://api.elevenlabs.io/v1/text-to-speech/
                                      {voice}/multi-stream-input
                                          (v3 would use text-to-dialogue)
                                                       │
                                             pcm_24000 + alignment
                                                       │
                                                       ▼
                                        rio_voice_eleven.js
                                     (Web Audio queue, 30 ms fade)
```

### Why the socket is on the server

The browser cannot hold an ElevenLabs key. Every other credential on this page
follows the same rule — routing, geocoding and TTS all go through the server,
and the one key a browser sees is the Maps render key, which can only draw a
map. A dialogue socket opened from the page would put a full-privilege
synthesis key in a devtools console.

So the relay is thin in one direction and not the other. The browser forwards
text deltas the instant they arrive and knows nothing about phrases, tags,
flushes or fallbacks. What it owns is playback, because those are facts about a
speaker: when a sample is audible, how fast RIO can be faded out, and how far
she actually got.

### Why a socket and not a request

A request cannot start until the sentence is finished. That is the entire cost
of putting a synthesiser after a model instead of inside one, and on a
three-clause answer it is most of a second the speech-to-speech path did not
pay.

Phrases go into the Text-to-Dialogue socket as they are written, audio comes
back while the model is still writing, and the driver hears the first clause
while the last one is being thought of. `flush` is what makes that true rather
than approximately true: without it the server waits for ~40 characters of its
own accord, which is a second sentence's worth of delay sitting inside the
mechanism that exists to remove delay.

### Two sockets, and the model picks

ElevenLabs streams over two different websockets, and which one a voice needs
is decided by the model:

| model | socket |
|---|---|
| `eleven_v3*` | `text-to-dialogue/stream-input` |
| everything else | `text-to-speech/{voice}/multi-stream-input` |

`voice_dialogue.py` speaks both. Everything around them is shared — the phrase
chunker, the tag gate, both fallback tiers, the keep-alive, the capacity
parking, the character accounting a resume depends on — and the difference is
four message shapes and three field names, written down once as a dialect.

**The multi-context socket is the better-behaved of the two**, and it is worth
saying why rather than filing it as a detail. Every audio frame it sends
carries the `contextId` it belongs to, and `isFinal` arrives per context. So an
utterance can be finished or abandoned **by name**: a barge-in closes one
context and the connection carries on. The dialogue socket sends audio with
nothing on it saying which turn it came from, so the only safe way to abandon a
turn there is to throw the connection away and open another — a reconnect per
barge-in, and an end-of-turn marker per *flush* that has to be counted against
flushes sent. One of those is a design; the other is a workaround for a missing
field.

Both are tested live on every `--live` run, including the one this build is not
configured for, because the transport that is switched off is the one that
rots.

---

## Measured

Speech-end to first audio — the silence a driver actually sits in. A real
spoken question fed into the session's input at real time, so the server's own
detector ends the turn exactly as it does in a car. Same question, same
instructions, same VAD settings on every row; only the mouth varies.

```
python -m tools.voice_latency --turns 10
```

8 turns per path, 2026-09-03:

| path | p50 | p95 | model's first word (p50) | **synthesis** p50 / p95 |
|---|---|---|---|---|
| `cedar` (the session speaks for itself) | 594 ms | 733 ms | — (inside the model) | — |
| **`elevenlabs_v2`** (multi-context socket) | **699 ms** | 879 ms | 278 ms | **491 / 536 ms** |
| `elevenlabs_v3` (dialogue socket) | 912 ms | 1286 ms | 547 ms | 366 / 433 ms |
| `elevenlabs_flash` (the per-utterance fallback) | 599 ms | 885 ms | 296 ms | 336 / 353 ms |
| pre-rendered clip | 0 ms | 0 ms | — | — |

**v2 costs +105 ms against cedar at the median.**

**Read the synthesis column, not the end-to-end one, when comparing the three
ElevenLabs rows.** They ran against the same GPT and got different luck from
it: v3's end-to-end p50 is 912 ms mostly because GPT took 547 ms to write its
first word in that batch against 278 ms in v2's. Subtracting the model leaves
the part this system owns, and there v2 is the slowest of the three —
**491 ms against v3's 366 and flash's 336**. That is the price of the clone
being reproduced properly, and it is ~125 ms.

v2 also returns audio in **fewer, larger chunks** than v3 (2 against 8 on the
same sentence). Playback still starts before the sentence is finished, which is
the whole point, but there is less of a head start than v3 gave.

---

## One voice everywhere

Every path names `config.ELEVENLABS_VOICE_ID`:

| what | model | why that model |
|---|---|---|
| conversation | `eleven_multilingual_v2` | the clone is reproduced properly, and it streams over the multi-context socket |
| nav / health / calm headway | `eleven_flash_v2_5` | **measured**: 168 ms to first byte against v2's 917 ms on the same line and the same voice |
| per-utterance fallback | `eleven_flash_v2_5` | same reason, plus it is already the fast path |
| pre-rendered clips | `eleven_multilingual_v2` | rendered offline, so there is no first byte to wait for — and matching the conversation exactly is the only thing left to optimise |

### Could the deterministic path use the conversation model too?

No, and it is not close. Measured on the same voice, same lines, time to first
byte over HTTP — which is the only property that path cares about, because the
sentence is already known in full before anything is called:

```
python -m tools.voice_latency --deterministic --turns 5

eleven_flash_v2_5          p50    168 ms   p95    253 ms
eleven_multilingual_v2     p50    917 ms   p95    952 ms
eleven_turbo_v2_5          p50    210 ms   p95    497 ms
```

**The conversation model costs +749 ms on a warning.** For reference, the
dictation budget this system already treats as too long to wait is 900 ms. A
driver cannot hear the difference between two readings of "Take the next left";
they can certainly hear it arrive late.

`eleven_turbo_v2_5` is the interesting third column — +42 ms over flash, and in
the same v2 family as the conversation model, so if flash ever turns out to
render the clone less faithfully than v2 does, that is the trade to look at
before giving up the latency. Both are on the voice's supported list.

**Dictation is switched off, not removed.** It exists so a warning comes out of
the same mouth as a conversation; under this backend that is already true
without dictating anything, and dictating into a text-mode session would
produce a warning as text, spoken through a socket built for prosody — slower,
out of band, and for nothing. `REALTIME_SPEECH_ENABLED` is still there and
still governs the cedar backend; the mint payload reports `speech_enabled:
false` and `rio_speak.js` falls through to the TTS endpoints, which cost
nothing to reach and are 200 ms away.

Clips are re-rendered with `python -m tools.render_alerts --force`, verified by
transcribing the finished MP3 with Whisper and re-rendering if it does not
match, and recorded in `static/audio/rendered.json` — which `tools/preflight`
compares against the configured voice, so "the clips are in RIO's voice" is a
check rather than a claim.

---

## Audio tags

v3 reads square-bracket directions: `[laughs]`, `[sighs]`, `[whispers]`,
`[exhales]`. Used once, in the right place, this is the difference between a
voice and a reading.

Used the way a model will use it if you let it, it is exactly the register the
bible bans in as many words — *NOT loud, polished, corporate, robotic, or
performative* — so `voice_tags.py` is a gate rather than a feature:

* **conversation only.** Every other channel is stripped unconditionally, and
  there is no configuration that turns that off. The failure this prevents is
  concrete: a synthesiser that does not know a tag reads the word "sighs" out
  loud at a junction.
* **one per utterance**, from a list short enough to read.
* **never the whole utterance.** A laugh with nothing to laugh about is the
  definition of performative.
* **never inside a word.** `Lin[laughs]coln` is not a direction, it is a token
  that got out sideways, and it is dropped rather than passed on.
* **unknown tags are dropped, not passed through.** Dropping and ignoring are
  different: an ignored tag gets spoken.

Nothing is discarded silently — every drop is counted with its reason
(`not_allowed`, `wrong_channel`, `misplaced`, `too_many`, `no_words`) and
printed, because "she never laughs any more" should be answerable.

The paragraph in the live instructions telling the model what it may do is
**generated from the same list the validator enforces**
(`voice_tags.instruction()`). A model told it may do something the gate will
undo is a model that keeps trying.

---

## Cancel, barge-in, resume

The arbiter is untouched. Every caller keeps the item it already had — same
priority, same group, same TTL — and conversation is still the tier that
yields. What changed is where three answers come from:

**Mute is a 30 ms fade on a gain node, and is undoable.** That is what a
barge-in does before anyone knows whether a person spoke, and the sustain gate
exists precisely to take it back when the noise turns out to have been a cough.
30 ms because a hard stop on a voice is a click, and a click in a car reads as
a fault.

**Cancel stops and flushes, and is not undoable.** `cancelGeneration()` now
does the same two things on both sides of the mouth: tell the model to stop
writing, and throw away the sound already made from what it wrote. On the
server the dialogue socket is **recycled** rather than muted — audio already
generated for a cancelled turn is on its way and there is no message that
unsends it; a fresh connection is ~90 ms, happens while the driver is talking
anyway, and removes the entire class of bug where a cancelled sentence finishes
itself over the top of the warning that stopped it.

**"How far did she get" comes from the speaker, not the model.** In text mode
the model finishes an answer seconds before the driver hears the end of it, so
a resume built from the model's text would make RIO skip every word the
interruption actually cost. The sink tracks what has been heard from the
context clock and the per-chunk alignment ElevenLabs returns, and that is what
a resume carries.

**The mouth is held until the LISTENER is done.** `response.done` is not the
end of the sentence any more. Handing the arbiter item back there would let the
next queued thing start talking over the second half of her answer — which is
not a warning pre-empting her, it is two voices at once.

Re-run the ten-turn probe under both mouths:

```
node tools/live_voice_probe.js                      # the session speaks
node tools/live_voice_probe.js --voice elevenlabs   # the sink speaks
```

Both report `ANSWERS LOST THAT SHOULD NOT HAVE BEEN 0` over the same ten turns.

---

## When it does not work

Two tiers, and they are different failures.

**Tier 1 — the line was slow.** v3 produced no byte within
`ELEVENLABS_FIRST_BYTE_BUDGET_MS` (1500), or the socket errored on this
utterance. That utterance finishes on flash, same voice, and the next one goes
back to the socket. The remainder handed to flash starts at the character count
ElevenLabs has actually produced audio for — not at what was sent to it — so
the driver hears one sentence with a change of texture in the middle rather
than a clause twice.

**The pool is full — which is neither of those.** Measured on this account
(see below): a connection costs no dialogue seat, so the socket opens normally
and the refusal arrives the first time RIO tries to *speak*, as a 1008 close
carrying `too_many_concurrent_requests`. From inside a car that is
indistinguishable from a dropped socket, and the reflex that is right for a
dropped socket is wrong here — the reconnect succeeds, the next utterance is
refused identically, and the drive spends itself reconnecting once per
sentence. So a capacity refusal **parks** the dialogue socket for
`ELEVENLABS_CAPACITY_BACKOFF_S` (60), runs every utterance on flash meanwhile,
logs it once, and quietly asks for a seat again when the back-off expires. It
is not counted as the service being down and never reaches cedar: a full pool
costs prosody, not speech.

**Tier 2 — the service is gone.** After
`ELEVENLABS_FAILURES_BEFORE_CEDAR` (2) consecutive utterances where neither the
socket nor flash produced anything, RIO takes her own voice back mid-drive: the
session is switched to audio output with `cedar` named, and it stays there.
One-way and sticky, because a voice that alternates between two people while
the network alternates is worse than either of them and the driver has no way
to interpret it. If the relay will not open at all, the drive starts on cedar
and never leaves.

Every fallback is logged with its cause, to stdout and into the drive's own
JSONL (`voice_fallback`), and counted at `/voice/status`.

**A dropped socket reconnects underneath whatever is happening.** The case that
matters is the CLEAN close — the 20-second inactivity rule, or a server restart
— because it does not raise: the async iterator simply ends. Treating that as
"the loop finished" left a stale connection in place for the rest of a drive,
with every later utterance failing its send and falling back to flash forever.
It is treated as a drop, and `tools/voice_selftest --live` closes a live socket
to prove it.

---

## Configuration

```
VOICE_BACKEND=elevenlabs                          # elevenlabs | openai_realtime
ELEVENLABS_VOICE_ID=weA4Q36twV5kwSaTEL0Q
ELEVENLABS_MODEL_CONVERSATION=eleven_multilingual_v2
ELEVENLABS_API_KEY=sk_...                         # never leaves the server
OPENAI_REALTIME_VOICE=cedar                       # what tier 2 falls back to
```

The conversation model is the only setting that chooses a transport: name an
`eleven_v3*` model and the dialogue socket is used, name anything else and the
multi-context text-to-speech socket is. `ELEVENLABS_MODEL` is still read as a
fallback for the older name. The deterministic model is not an environment
variable — it is `eleven_flash_v2_5` in `config.py`, for the measured reason
above.

The backend is **config plus a restart**, not a runtime toggle: the two paths
differ in what the live session is asked to produce, and a session that changed
its mind about that halfway through a drive would be a third path nobody had
tested. The per-utterance fallbacks are what handle trouble inside a drive.

`VOICE_BACKEND=realtime` is still accepted and means `openai_realtime`. A car
that comes up mute because a value was renamed is a bad trade for a tidier
vocabulary.

The rest lives in `config.py` under *WHOSE VOICE*: the chunker's two rules
(`ELEVENLABS_CHUNK_MIN_TOKENS`, `ELEVENLABS_CHUNK_MAX_WAIT_MS`), the keep-alive
interval, the output format, the tag policy, both fallback thresholds and
`ELEVENLABS_CAPACITY_BACKOFF_S` — how long a car stays on flash after finding
the dialogue pool full.

> **A whitespace trap worth knowing about.** A key pasted out of a browser
> arrives with a leading non-breaking space. Every call then fails with
> "Invalid API key", which reads as a revoked key and is not one, and nothing
> in the message mentions whitespace. `voice.api_key()` and
> `voice_dialogue.api_key()` strip it, and `tools/preflight` says so out loud
> — without ever printing the key.

---

## Testing

```
python -m tools.voice_selftest                       # config, chunker, tags, firewall, clips
python -m tools.voice_selftest --live                # + both real sockets and both fallback tiers
python -m tools.voice_selftest --server http://127.0.0.1:8888   # + the relay
python -m tools.voice_selftest --pool                # + a genuinely full dialogue pool
node tools/realtime_selftest.js                      # + the text-mode controller section
node tools/live_voice_probe.js --voice elevenlabs    # the ten turns, through the sink
python -m tools.voice_latency --turns 10             # the four voices, side by side
python -m tools.voice_latency --deterministic        # ...and the warning path, per model
python -m tools.render_alerts --force                # re-render the clips + manifest
python -m tools.preflight                            # is this pod configured for that voice
```

Both fallback tiers are triggered by **making the real conditions true** — a
budget that cannot be met, a socket that has gone away, a key that is refused —
rather than by calling the fallback function. A test that calls the fallback
proves the fallback runs; only a test that causes the failure proves anything
reaches it. Two real bugs were found this way: the clean-close reconnect above,
and a reader task that cancelled itself from inside its own reconnect and
abandoned it half-done.

The browser sink is tested against a stub AudioContext whose clock the test
turns by hand (`tools/voice_sink_harness.js`). Everything interesting it does
is about time — how much has been heard, whether a cancel stopped audio that
was already scheduled, whether the mouth is handed back when the model finishes
or when the listener does — and against a real context those assertions would
be about how fast the machine running them is.

---

## What is unproven

* **Playback in a real browser.** Every decision in the sink is tested against
  a stub context; the Web Audio scheduling itself, and the iOS gesture-unlock
  path around it, have not been run on a device.
* **How v2 sounds in a moving car** at road-noise levels, against flash. The
  latency difference is measured and the clone is now reproduced by a model the
  voice actually supports; whether flash — which is also on that list, and
  which every warning uses — holds the likeness as well as v2 does is a
  judgement nobody has made yet with the engine running. It is the same class
  of thing that made v3 wrong, and the only way to settle it is to listen.
* **How often the tag gate fires in practice.** The rules are tested; the
  *rate* at which a real conversation reaches for a tag it does not get is a
  property of the model and the prompt, and wants a drive's worth of
  `/voice/status` behind it.
* **More than one car at once.** The limit is measured and the refusal path is
  tested, but every measurement so far is one drive at a time. Nothing has
  driven two RIOs against this workspace simultaneously.

---

## "What do you see?" — where the time goes

The visual turn got slower when the voice changed, and the shape of it is not
what the change made obvious. Measured end to end on the current path
(`python -m tools.visual_latency --live --session auto`), with a seam at every
stage, before and after the fixes below:

| stage | before (p50/p95) | after (p50/p95) | whose time |
|---|---|---|---|
| turn end → `look()` called | 716 / 2178 ms | 704 / 1233 ms | the model |
| `look()` → tool result | 4 / 5996 ms | **4 / 3421 ms** | ours |
| tool result → first text token | 385 / 1697 ms | 607 / 766 ms | the model |
| first token → first phrase out | 101 ms | **67 ms** | ours |
| first phrase → FIRST AUDIO | 422 ms | 293 ms | the synthesiser |
| **turn end → first audio, scene questions** | ~1629 ms | **~1480 ms** | |
| turn end → first audio, all questions | 2464 / 5426 ms | 1898 / 4900 ms | |
| observer cache hits | 4/8 | **7/10** | |
| answer length | 23 / 44 words | 19 / 32 words | |

**The observer was never the problem.** It ran throughout — 304 observations
over a ten-question run, `has_record` true — and every cache hit came back in
**3–4 ms**. What was wrong was which questions reached it.

### The gate was judging the wrong sentence

`look()` decides whether a question can be answered from the running
observation by matching it against a list of scene phrasings. Those phrasings
are things *drivers* say. What arrives at the tool is what the *model* passed,
and the model paraphrases: "what's around us right now" arrives as "describe
the current scene", misses the list, and pays 2.3–6.0 s for a full remote
visual turn to produce a sentence that was already sitting in memory.

Two changes, and the second matters more than the first:

* **The driver's own transcript now travels with the tool call.** The panel is
  where Whisper's output lands, so it is the only thing that can send it, and
  the gate judges it first. A paraphrase is now the fallback rather than the
  authority.
* **The rule is the right way round.** It was a whitelist of phrasings that
  might be scene questions — a losing game, where every phrasing nobody thought
  of costs two seconds. What actually disqualifies a caption from answering is
  small, closed and already written down: the question asks about a *particular
  thing*. So a short question with no object reference in it is a scene
  question, and the whitelist survives only as the fast yes.

Three of eight scene questions were taking the full path. Now the only
questions that take it are the ones that should: "what's that car ahead",
"what's on the left", "what colour is the car in front".

### Two smaller things the numbers also named

* **She was told to say "let me look" before calling the camera.** Those words
  are composed *before* the call goes out, so the driver waits through them —
  in front of a tool that usually answers in four milliseconds. She calls
  first and speaks after now. The research tool, which really does take
  seconds, keeps its holding line.
* **Camera answers ran to 23 words at the median and 44 at p95**, against
  instructions asking for one short sentence. Instructions are guidance; the
  follow-up response after a `look` now carries
  `REALTIME_LOOK_ANSWER_MAX_TOKENS` as an actual ceiling. 19 / 32 words after.

### What did not turn out to be true

Two reasonable suspicions, both cleared by measurement rather than argument:

* **The observer is not gated on the old audio session.** It starts when the
  session is minted, which text mode still does.
* **v2's coarser chunking is not what delays the first word.** The first phrase
  reaches the socket 67 ms after the model's first token. It was worth taking
  to its floor anyway — the first phrase of an answer has nothing playing
  behind it, so it now goes at three words or 120 ms rather than five or 250 —
  but this stage was never the problem, and saying so is worth more than the
  40 ms.

### It is still above the target, and this is why

The goal was ~1.2 s to first audio on a scene question. It is ~1.48 s, and the
budget says where the rest is:

```
  ~700 ms   the model hearing the question and deciding to call the camera
     4 ms   the camera answering
  ~450 ms   the model composing a sentence from the answer
    67 ms   the chunker deciding it is worth speaking
  ~330 ms   v2 synthesising it
```

**About a second of that is two round trips to a remote model**, and roughly
0.4 s is this system's. Getting under 1.2 s means removing one of the two model
passes — speaking the observation with less of RIO's own composition on top of
it — and that is a deliberate trade against the rule this codebase has held
throughout: *she phrases it, the pipeline does not.* It is a decision about
what RIO is, not a tuning parameter, so it is written down here rather than
taken.

---

## Concurrency — measured, because it is not published

**Which pool depends on the transport, and they are different pools.** Under
`eleven_multilingual_v2` the conversation runs on the text-to-speech socket,
not the dialogue socket, so the 21-seat dialogue pool below no longer governs
the conversation path — it governs the v3 path, which is still shipped and one
config value away.

**On the text-to-speech socket: at least 20 simultaneous, and no refusal.**
Twenty multi-context requests generating *at the same time* on this account all
succeeded. That is notably not the 3 the published table gives multilingual v2
on Starter, which suggests websocket sessions are metered separately from HTTP
there too. The ceiling was not probed further: 20 already answers the only
question that matters, since a car needs one.

> A first attempt at this measured nothing and looked like a pass. It opened
> sockets one at a time and generated a short phrase on each, so no two
> requests were ever concurrent. The number only means something when the
> generations overlap, which is the easy mistake to make here and the reason
> the probe now opens every socket first and starts them together.

**On the dialogue socket, Starter holds 21 concurrent sessions.** The service
says so itself:

```
too_many_concurrent_requests: Too many concurrent requests. Your current
subscription is associated with a maximum of 21 concurrent requests
```

For scale, this tier's *published* standard concurrency is 3 on multilingual v2
and 6 on flash — so the dialogue pool really is sized differently. Either way,
neither number is one this product is going to reach: a car needs one seat.

**The seat is taken lazily, and that is the part worth knowing.** Two probes
disagreed before this was understood: thirty idle connections were all
accepted, while twenty-two that had each generated once were not. A connection
costs nothing; the seat is allocated at first synthesis and held until close.
Which means:

* the relay can open a car's socket at the start of a drive without spending
  anything, which is exactly what it does — the ~90 ms of setup is paid before
  RIO's first word rather than on it;
* a page reload that briefly leaves two sockets open for one car costs one
  seat, not two;
* and a car that is going to be refused is refused **when she first speaks**,
  not when the drive starts. That asymmetry is why the capacity path above
  exists and why it is tested by filling the pool for real rather than by
  simulating a refusal:

```
python -m tools.voice_selftest --pool
```

That test opens sessions until the service refuses, then drives a car through
it: the connection succeeds, the first utterance falls back to flash and is
recorded as a full pool rather than a dropped socket, the socket is parked
rather than reconnected, the second utterance goes straight to flash without
asking again, and the drive keeps RIO's own voice instead of reaching for
cedar. It briefly uses every dialogue seat the account has, so it is behind its
own flag rather than in the ordinary `--live` run.
