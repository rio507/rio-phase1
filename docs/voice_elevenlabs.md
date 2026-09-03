# RIO's voice — ElevenLabs v3, with GPT-Realtime still doing the thinking

**Status:** implemented, measured against the real models and the real service.
The playback half runs in a browser and has not been driven in a car yet; see
*What is unproven* at the end.

---

## What changed

| | before | now |
|---|---|---|
| hearing | the live session, continuously | **unchanged** |
| thinking | `gpt-realtime-2.1` | **unchanged** |
| speaking | the same session, in `cedar` | ElevenLabs `eleven_v3_conversational`, voice `weA4Q36twV5kwSaTEL0Q` |
| deterministic lines | dictated into the live session | ElevenLabs `eleven_flash_v2_5`, **same voice id** |
| pre-rendered clips | rendered in `cedar` | re-rendered on v3, **same voice id** |

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
                                    wss://api.elevenlabs.io/v1/
                                      text-to-dialogue/stream-input
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

---

## Measured

Speech-end to first audio — the silence a driver actually sits in. A real
spoken question fed into the session's input at real time, so the server's own
detector ends the turn exactly as it does in a car. Same question, same
instructions, same VAD settings on every row; only the mouth varies.

```
python -m tools.voice_latency --turns 10
```

10 turns per path, 2026-09-03:

| path | p50 | p95 | model's first word (p50) | synthesis p50 / p95 |
|---|---|---|---|---|
| `cedar` (the session speaks for itself) | 634 ms | 1004 ms | — (inside the model) | — |
| `elevenlabs_v3` (text mode → dialogue socket) | 717 ms | 1183 ms | 293 ms | 402 / 444 ms |
| `elevenlabs_flash` (the per-utterance fallback) | 633 ms | 872 ms | 316 ms | 325 / 369 ms |
| pre-rendered clip | 0 ms | 0 ms | — | — |

**v3 conversational costs +83 ms against cedar at the median.** The p95 column
is mostly the MODEL: a turn where GPT takes a second to write its first word is
a second late in every voice, which is why the synthesis column is reported
separately — that is the part this change owns, and it is a much tighter
distribution than the end-to-end number suggests (402 ms p50, 444 ms p95).

Flash is level with cedar end to end and has the tightest p95 of the three,
which is the reason it is the per-utterance fallback and the reason every
deterministic line uses it.

---

## One voice everywhere

Every path names `config.ELEVENLABS_VOICE_ID`:

| what | model | why that model |
|---|---|---|
| conversation | `eleven_v3_conversational` | prosody and audio tags; a socket, so it can stream |
| nav / health / calm headway | `eleven_flash_v2_5` | fastest to first byte, and a turn instruction is a line where that is the only property that matters |
| per-utterance fallback | `eleven_flash_v2_5` | same reason, plus it is already the fast path |
| pre-rendered clips | `eleven_v3_conversational` | rendered offline, so there is no first byte to wait for |

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
VOICE_BACKEND=elevenlabs                   # elevenlabs | openai_realtime
ELEVENLABS_VOICE_ID=weA4Q36twV5kwSaTEL0Q
ELEVENLABS_MODEL=eleven_v3_conversational
ELEVENLABS_API_KEY=sk_...                  # never leaves the server
OPENAI_REALTIME_VOICE=cedar                # what tier 2 falls back to
```

The backend is **config plus a restart**, not a runtime toggle: the two paths
differ in what the live session is asked to produce, and a session that changed
its mind about that halfway through a drive would be a third path nobody had
tested. The per-utterance fallbacks are what handle trouble inside a drive.

`VOICE_BACKEND=realtime` is still accepted and means `openai_realtime`. A car
that comes up mute because a value was renamed is a bad trade for a tidier
vocabulary.

The rest lives in `config.py` under *WHOSE VOICE*: the chunker's two rules
(`ELEVENLABS_CHUNK_MIN_TOKENS`, `ELEVENLABS_CHUNK_MAX_WAIT_MS`), the keep-alive
interval, the output format, the tag policy and both fallback thresholds.

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
python -m tools.voice_selftest --live                # + the real socket and both fallback tiers
python -m tools.voice_selftest --server http://127.0.0.1:8888   # + the relay
node tools/realtime_selftest.js                      # + the text-mode controller section
node tools/live_voice_probe.js --voice elevenlabs    # the ten turns, through the sink
python -m tools.voice_latency --turns 10             # the three voices, side by side
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
* **How v3 sounds in a moving car** at road-noise levels, against flash. The
  latency difference is measured; the quality difference is a judgement nobody
  has made yet with the engine running.
* **How often the tag gate fires in practice.** The rules are tested; the
  *rate* at which a real conversation reaches for a tag it does not get is a
  property of the model and the prompt, and wants a drive's worth of
  `/voice/status` behind it.
* **Concurrency at the plan level.** See below.

---

## One thing that needs a human

Dialogue sessions draw from a **separate pool** from ordinary synthesis — audio
generated over the Text-to-Dialogue socket does not count toward the standard
concurrency limit. This account is on the **Starter** tier, and the dialogue
pool size for that tier is not exposed through the API.

This build holds **exactly one** dialogue connection per live RIO session, and
closes the previous one if a page reloads mid-drive — so one car is one seat.
That is fine for one driver and is the thing to check before a second one:
someone with account access should confirm the Starter dialogue-session limit,
because the failure mode if it is 1 is not an error, it is the second car
starting every drive on cedar.
