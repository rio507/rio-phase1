"""RIO's mouth, when her voice is ElevenLabs.

WHAT THIS REPLACES, AND WHAT IT DOES NOT
----------------------------------------
The live session used to hear the driver, think, and speak, all inside one
model. Under VOICE_BACKEND=elevenlabs it still hears and still thinks — its
audio INPUT is untouched, its tools are untouched, its instructions are
untouched — but it is put in text mode, and the words it produces are
synthesised here.

Nothing deterministic comes through this file. Warnings, turns and health
announcements are written by policy code and spoken by voice.synthesize_stream
on flash, and the pre-rendered clips do not touch the network at all. This is
the conversation path, which is the tier that yields.

WHY A SOCKET AND NOT A REQUEST
------------------------------
A request cannot start until the sentence is finished. That is the whole cost
of putting a synthesiser after a model instead of inside one: the driver waits
for the model to stop writing before anything begins to speak, and on a
three-clause answer that is most of a second of silence that the speech-to-
speech path did not have.

The Text-to-Dialogue socket removes it. Phrases go in as they are produced,
audio comes back while the model is still writing, and the driver hears the
first clause while the last one is being thought of. `flush` is what makes that
true rather than approximately true: without it the server waits for ~40
characters of its own accord, which is a second sentence's worth of delay
sitting inside a mechanism that exists to remove delay.

ONE CONNECTION PER LIVE SESSION
-------------------------------
The docs are explicit that one connection is one dialogue session, and this
holds exactly one for as long as a drive lasts: opened when the panel connects,
kept alive through every silence, reconnected underneath whatever is happening
if it drops, closed when the session ends. The reconnect is not an optimisation
— a socket that has been quiet for twenty seconds is closed BY THE SERVER, and
in a car twenty seconds of quiet is the normal state of things.

WHAT HAPPENS WHEN IT DOES NOT WORK
----------------------------------
Two tiers, and they are different failures:

  the LINE was slow      v3 conversational did not produce a byte inside
                         ELEVENLABS_FIRST_BYTE_BUDGET_MS, or errored on this
                         utterance. That utterance finishes on flash — same
                         voice, flatter reading, much faster — and the next one
                         goes back to the socket.
  the SERVICE is gone    consecutive utterances have failed, or the socket
                         will not open and flash will not stream either. Then
                         RIO's voice goes back to the live session's own, and
                         stays there for the drive.

Every fallback is logged with its cause, because "she sounded different for a
minute" is not something a driver can debug and is exactly what these two tiers
look like from the passenger seat.
"""
import asyncio
import base64
import json
import os
import re
import time

import config
import voice_tags

WS_URL = "wss://api.elevenlabs.io/v1/text-to-dialogue/stream-input"
FLASH_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream"

# Causes, named once. These are what /voice/status counts and what the log
# lines say, and having them in one place is what stops "error" becoming the
# answer to every question about why RIO sounded like somebody else.
SLOW_FIRST_BYTE = "slow_first_byte"
SOCKET_ERROR = "socket_error"
SOCKET_UNAVAILABLE = "socket_unavailable"
SYNTH_ERROR = "synth_error"
SERVICE_DOWN = "service_down"


def api_key() -> str:
    """The key, with the whitespace taken off.

    Not defensiveness for its own sake: a key pasted out of a browser arrives
    with a non-breaking space in front of it, every REST call fails with
    "Invalid API key", and nothing in the message says that whitespace is the
    reason. This cost an afternoon once.
    """
    return (os.getenv("ELEVENLABS_API_KEY") or "").strip()


def voice_id() -> str:
    return (config.ELEVENLABS_VOICE_ID
            or (os.getenv("ELEVENLABS_VOICE_ID") or "").strip())


def configured() -> bool:
    return bool(api_key() and voice_id())


# ---------------------------------------------------------------------------
# WHEN A PHRASE IS WORTH SPEAKING
# ---------------------------------------------------------------------------
# The chunker is the whole difference between text mode being faster than the
# speech-to-speech path and being slower than it, and it is a pure function of
# a character stream and a clock — so it lives here as an object with no
# network in it and is tested directly.
#
# Two rules, and they cover different failures:
#
#   a clause boundary with enough words behind it     the ordinary case. RIO
#   starts speaking the first clause while the model writes the second, and the
#   seam between them falls where a person would have breathed anyway.
#
#   too long since the first unsent word arrived      the safety net. An answer
#   that opens with a long subordinate clause has no boundary for a while, and
#   a short reply — "Yeah." — may never produce one at all before the model
#   stops. Without this rule those two shapes are exactly the ones that feel
#   slow, which is to say the ones a driver notices.
#
# Never mid-tag. A `[` opens a direction to the synthesiser and splitting it
# across two messages produces two broken ones.
_BOUNDARY = re.compile(r"[.!?…](?=\s|$)|[,;:—–](?=\s)")


class PhraseChunker:
    """A character stream in, speakable phrases out."""

    def __init__(self, min_tokens=None, max_wait_ms=None):
        self.min_tokens = int(config.ELEVENLABS_CHUNK_MIN_TOKENS
                              if min_tokens is None else min_tokens)
        self.max_wait_ms = float(config.ELEVENLABS_CHUNK_MAX_WAIT_MS
                                 if max_wait_ms is None else max_wait_ms)
        self.buf = ""
        self.since = None        # when the oldest unsent character arrived

    def _tokens(self, s: str) -> int:
        return len(s.split())

    def _open_tag(self, s: str) -> bool:
        return s.rfind("[") > s.rfind("]")

    def push(self, text: str, now: float = None) -> list:
        """Add text; return whatever is ready to speak right now."""
        if not text:
            return []
        now = time.time() if now is None else now
        if self.since is None:
            self.since = now
        self.buf += text
        return self._harvest(now)

    def _harvest(self, now: float) -> list:
        out = []
        while True:
            cut = None
            for m in _BOUNDARY.finditer(self.buf):
                head = self.buf[:m.end()]
                if self._open_tag(head):
                    continue
                if self._tokens(head) >= self.min_tokens:
                    cut = m.end()
                    break
            if cut is None:
                break
            out.append(self.buf[:cut])
            self.buf = self.buf[cut:]
            self.since = now if self.buf.strip() else None
        return out

    def due(self, now: float = None) -> list:
        """The max-wait rule. Called on a tick, not on arrival.

        Returns at most one phrase: the point is to get SOMETHING moving, and
        whatever is left keeps waiting for a boundary like everything else.

        Where it cuts matters as much as that it cuts. The last clause boundary
        if there is one — a comma four words in is a real place to breathe,
        even though the token rule declined it — and otherwise the last
        complete word. NEVER mid-word: half a word is not a shorter phrase, it
        is a mispronunciation followed by a seam, and it is audible.
        """
        now = time.time() if now is None else now
        if self.since is None or not self.buf.strip():
            return []
        if (now - self.since) * 1000.0 < self.max_wait_ms:
            return []
        if self._open_tag(self.buf):
            return []           # a half-written tag is not a phrase

        cut = 0
        for m in _BOUNDARY.finditer(self.buf):
            if not self._open_tag(self.buf[:m.end()]):
                cut = m.end()
        if not cut:
            cut = self.buf.rstrip().rfind(" ") + 1
        if cut <= 0 or not self.buf[:cut].strip():
            # One long unbroken word, still arriving. Waiting for the rest of
            # it is the only thing that produces speakable audio.
            return []
        phrase, self.buf = self.buf[:cut], self.buf[cut:]
        self.since = now if self.buf.strip() else None
        return [phrase]

    def drain(self) -> list:
        """Everything left, because the model has stopped writing."""
        left = self.buf
        self.buf, self.since = "", None
        return [left] if left.strip() else []

    def pending(self) -> str:
        return self.buf


# ---------------------------------------------------------------------------
# One utterance
# ---------------------------------------------------------------------------
class _Utterance:
    """What RIO is saying now, and how much of it has actually been voiced.

    `voiced` is counted from the alignment ElevenLabs sends back, not from what
    was sent to it, and that distinction is what makes the fallback clean: when
    a socket dies mid-sentence the remainder handed to flash starts where the
    audio stopped, so the driver hears one sentence rather than a clause twice.
    """

    def __init__(self, rid: str):
        self.rid = rid
        self.chunker = PhraseChunker()
        self.sent = ""            # text handed to the socket
        self.voiced = 0           # characters ElevenLabs has produced audio for
        self.began = time.time()
        self.first_audio = None   # ms from first flush to first byte
        self.flushed_at = None
        self.ended = False        # the model has stopped writing
        self.done = False         # ...and the audio for it is finished
        # THE SOCKET ANSWERS PER FLUSH, NOT PER UTTERANCE. Every `flush` that
        # goes out eventually comes back with an `is_final_audio_for_turn`, and
        # generation lags — so the marker for the FIRST clause can easily land
        # after the model has stopped writing. Treating any one of them as "the
        # utterance is over" hands the mouth back with two clauses still to
        # play, and the next queued thing starts talking over them. Counting
        # both sides is what makes "finished" mean finished.
        self.flushes = 0
        self.finals = 0
        self.cancelled = False
        self.on_flash = False     # this one is finishing on the fallback model
        self.tags_dropped = []
        self.chunks = 0


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------
class DialogueSession:
    """Exactly one ElevenLabs dialogue socket, for exactly one live session.

    `on_audio(rid, pcm, text)` is awaited for every piece of audio produced,
    with the text that piece corresponds to — the browser needs both: the
    samples to play and the words, so that when a warning cuts RIO off it can
    say how far she actually got rather than how far the model got.

    `on_event(kind, detail)` is awaited for everything worth reporting: a
    fallback and its cause, a reconnect, the service going away.
    """

    def __init__(self, on_audio, on_event=None, voice=None, model=None,
                 fmt=None, force_flash=False):
        # Every utterance goes straight to the fallback model, with no dialogue
        # socket opened at all. Not a degraded mode and not an error: it is how
        # tools/voice_latency.py times the tier-1 destination on its own, and
        # how the selftests exercise the fallback without breaking anything.
        self.force_flash = bool(force_flash)
        self.on_audio = on_audio
        self.on_event = on_event or (lambda *a, **k: _noop())
        self.voice = voice or voice_id()
        self.model = model or config.ELEVENLABS_DIALOGUE_MODEL
        self.fmt = fmt or config.ELEVENLABS_OUTPUT_FORMAT

        self._ws = None
        self._reader = None
        self._ticker = None
        self._lock = asyncio.Lock()
        self._closed = False
        self._utt = None
        self._last_send = 0.0
        self._consecutive_failures = 0

        self.degraded = False     # tier 2: ElevenLabs is out, cedar has it
        self.stats = {"utterances": 0, "reconnects": 0, "keepalives": 0,
                      "fallbacks": {}, "audio_chunks": 0, "audio_bytes": 0,
                      "first_audio_ms": [], "tags_dropped": 0}

    # -- lifecycle ---------------------------------------------------------
    def _url(self) -> str:
        return (f"{WS_URL}?model_id={self.model}&output_format={self.fmt}"
                "&sync_alignment=true")

    async def _open(self) -> bool:
        """Open the socket and register the voice. False if it would not.

        Authentication is the `xi-api-key` HEADER and nothing else. The guide
        also documents an `xi_api_key` field in the first message; this
        endpoint rejects a message carrying it with "Invalid API key" even when
        the header is correct, which reads as a bad key and is not one.
        """
        import websockets

        if not configured():
            return False
        try:
            self._ws = await websockets.connect(
                self._url(), additional_headers={"xi-api-key": api_key()},
                max_size=None, open_timeout=10)
            await self._ws.send(json.dumps({"voices": [self.voice]}))
            self._last_send = time.time()
            return True
        except Exception as e:
            print(f"[voice] dialogue socket would not open: "
                  f"{type(e).__name__}: {str(e)[:160]}", flush=True)
            self._ws = None
            return False

    async def start(self) -> bool:
        """Open the socket when the drive starts, not when RIO first speaks.

        Ninety milliseconds of connection setup is not much, but it is ninety
        milliseconds that would otherwise land on the first thing she says,
        which is the one utterance a driver is listening for.
        """
        if self.force_flash:
            self._ticker = asyncio.create_task(self._tick_loop())
            return False
        ok = await self._open()
        if ok:
            self._reader = asyncio.create_task(self._read_loop())
        self._ticker = asyncio.create_task(self._tick_loop())
        if not ok:
            await self._fail(SOCKET_UNAVAILABLE, None)
        return ok

    async def close(self):
        self._closed = True
        for task in (self._reader, self._ticker):
            if task:
                task.cancel()
        ws, self._ws = self._ws, None
        if ws:
            try:
                await ws.send(json.dumps({"close_socket": True}))
            except Exception:
                pass
            try:
                await ws.close()
            except Exception:
                pass

    async def _reconnect(self, why: str):
        """Put a socket back underneath whatever is happening.

        Transparent by construction: the caller does not learn about it, and an
        utterance that was in flight when the old one went has already been
        handed to flash by the code that noticed. This only has to make the
        NEXT one work.
        """
        if self._closed:
            return
        ws, self._ws = self._ws, None
        if ws:
            try:
                await ws.close()
            except Exception:
                pass
        # The reader is very often the CALLER here — a socket that ends is
        # noticed by the loop reading it, and that loop then asks for a new
        # one. Cancelling it from inside itself raises CancelledError at the
        # next await and abandons the reconnect half-done, which is a session
        # that never comes back and no error anywhere saying why.
        if self._reader and self._reader is not asyncio.current_task():
            self._reader.cancel()
        self._reader = None
        if await self._open():
            self._reader = asyncio.create_task(self._read_loop())
            self.stats["reconnects"] += 1
            print(f"[voice] dialogue socket reconnected ({why})", flush=True)
            await self._emit("reconnected", {"why": why})

    # -- the wire ----------------------------------------------------------
    async def _send(self, obj: dict) -> bool:
        ws = self._ws
        if ws is None:
            return False
        try:
            await ws.send(json.dumps(obj))
            self._last_send = time.time()
            return True
        except Exception as e:
            print(f"[voice] dialogue send failed: {type(e).__name__}", flush=True)
            self._ws = None
            return False

    async def _read_loop(self):
        """Read until the socket ends, then put a new one underneath it.

        A CLEAN close is the case that matters, and it is the one that does not
        raise: the async iterator simply ends. That is what the twenty-second
        inactivity rule looks like from here, and what a server restart looks
        like, and treating it as "the loop finished" rather than as "the socket
        went" left a stale connection object in place for the rest of the
        drive — every utterance after it failed its send, fell back to flash,
        and nothing ever reconnected. The driver would have heard RIO
        permanently in the fallback voice with no error anywhere.
        """
        ws = self._ws
        if ws is None:
            return
        why = "closed"
        try:
            async for raw in ws:
                await self._on_message(json.loads(raw))
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if self._closed:
                return
            why = type(e).__name__
            print(f"[voice] dialogue socket dropped: {why}: {str(e)[:120]}",
                  flush=True)
        if self._closed or self._ws is not ws:
            return          # a deliberate close, or already replaced
        # An utterance in flight has to be rescued before the socket is
        # replaced, or the driver hears half an answer and no more.
        utt = self._utt
        if utt and not utt.done and not utt.cancelled:
            await self._fall_back(utt, SOCKET_ERROR)
        await self._reconnect(why)

    async def _on_message(self, m: dict):
        if m.get("error"):
            print(f"[voice] dialogue error: {str(m.get('message'))[:160]}",
                  flush=True)
            utt = self._utt
            if utt and not utt.done and not utt.cancelled:
                await self._fall_back(utt, SOCKET_ERROR)
            return

        audio_b64 = m.get("audio")
        if audio_b64:
            utt = self._utt
            # Audio for something that has been abandoned — cancelled by a
            # barge-in, or moved onto flash — is dropped rather than played.
            # This is the whole reason an utterance carries a flag instead of
            # the code simply stopping: the bytes are already in flight.
            if utt is None or utt.cancelled or utt.on_flash:
                return
            pcm = base64.b64decode(audio_b64)
            chars = ((m.get("alignment") or {}).get("chars")) or []
            text = "".join(chars)
            utt.voiced += len(text)
            utt.chunks += 1
            if utt.first_audio is None and utt.flushed_at:
                utt.first_audio = (time.time() - utt.flushed_at) * 1000.0
                self.stats["first_audio_ms"].append(round(utt.first_audio, 1))
                self._consecutive_failures = 0
            self.stats["audio_chunks"] += 1
            self.stats["audio_bytes"] += len(pcm)
            await self._deliver(utt.rid, pcm, text)
            return

        if m.get("is_final_audio_for_turn"):
            utt = self._utt
            if utt:
                utt.finals += 1
                await self._maybe_done(utt)

    async def _maybe_done(self, utt: "_Utterance"):
        """Is this utterance over? Said once, when both halves agree.

        The model has stopped writing AND every flush has come back. Either on
        its own is a lie: the first is true seconds before the driver hears the
        end of the sentence, and the second is true after every clause.

        On the fallback there are no flushes to wait for — the audio came over
        HTTP and the stream ending IS the end of it — so the counters are equal
        by construction and this fires as soon as `end` has run.
        """
        if utt.done or utt.cancelled or not utt.ended:
            return
        if utt.finals < utt.flushes:
            return
        utt.done = True
        await self._emit("utterance_done", {
            "rid": utt.rid, "first_audio_ms": utt.first_audio,
            "chunks": utt.chunks, "on_flash": utt.on_flash})

    # -- what the panel asks for -------------------------------------------
    async def begin(self, rid: str):
        """A new answer is starting. One utterance, one dialogue turn.

        If the last one had not finished generating, the socket is recycled on
        the way past — for the same reason a cancel recycles it. Audio comes
        back over one stream with nothing on it that says which utterance it
        belongs to; the ONLY thing that makes attribution safe is that the
        previous turn is complete, and "complete" here means every flush has
        had its end-of-turn marker. Left running, the tail of the old answer
        would be counted as the opening of the new one, and a resume would
        carry the wrong sentence.
        """
        stale = False
        async with self._lock:
            if self._utt and not self._utt.done:
                self._utt.cancelled = True
                stale = True
            self._utt = _Utterance(rid)
            self._utt.on_flash = self.force_flash
            self.stats["utterances"] += 1
        if stale and self._ws is not None:
            await self._reconnect("superseded")

    async def delta(self, rid: str, text: str):
        """More of the answer. Speak whatever this completes."""
        async with self._lock:
            utt = self._utt
            if not utt or utt.rid != rid or utt.cancelled:
                return
            clean, dropped = voice_tags.sanitize(text, voice_tags.CONVERSATION)
            if dropped:
                utt.tags_dropped.extend(dropped)
                self.stats["tags_dropped"] += len(dropped)
                for d in dropped:
                    print(f"[voice] tag dropped ({d['reason']}): {d['tag']}",
                          flush=True)
            for phrase in utt.chunker.push(clean):
                await self._speak(utt, phrase)

    async def end(self, rid: str):
        """The model has stopped writing. Say the rest and close the turn."""
        async with self._lock:
            utt = self._utt
            if not utt or utt.rid != rid or utt.cancelled:
                return
            for phrase in utt.chunker.drain():
                await self._speak(utt, phrase)
            utt.ended = True
            # NO EXTRA FLUSH HERE. Every phrase is flushed as it is sent, so by
            # this point there is nothing buffered — and a flush over an empty
            # buffer generates no audio and returns no end-of-turn marker. One
            # sent anyway leaves the counters permanently one apart, and an
            # utterance that can never report itself finished is a mouth that
            # is never handed back.
            await self._maybe_done(utt)

    async def cancel(self, rid: str):
        """Stop. A warning has the mouth, or the driver interrupted.

        The socket is recycled rather than merely muted. Audio already
        generated for this turn is on its way and there is no message that
        unsends it; a fresh connection is ninety milliseconds and removes the
        entire class of bug where a cancelled sentence finishes itself over the
        top of the warning that stopped it.

        NO LOCK on the flag. Everything else here serialises on `self._lock` so
        that phrases reach the socket in the order they were written — and a
        fallback holds that lock for as long as its HTTP stream takes, which is
        seconds. A cancel that queued behind one would be a barge-in that took
        effect after the sentence it was interrupting had finished, which is
        the opposite of what it is for. Setting a flag is atomic between
        awaits; the readers all check it before doing anything audible.
        """
        utt = self._utt
        if utt and utt.rid == rid:
            utt.cancelled = True
            utt.done = True
        if self._ws is not None:
            await self._reconnect("cancelled")

    # -- speaking, and not being able to ------------------------------------
    async def _speak(self, utt: "_Utterance", phrase: str):
        if not phrase.strip():
            return
        if utt.on_flash:
            if utt.flushed_at is None:
                utt.flushed_at = time.time()
            await self._flash(utt, phrase)
            return
        first = not utt.sent
        ok = await self._send({"inputs": [{
            "text": phrase, "voice_id": self.voice, "new_turn": first}]})
        if ok:
            ok = await self._send({"flush": True})
            if ok:
                utt.flushes += 1
        if not ok:
            await self._fall_back(utt, SOCKET_ERROR, phrase)
            return
        utt.sent += phrase
        if utt.flushed_at is None:
            utt.flushed_at = time.time()

    async def _fall_back(self, utt: "_Utterance", cause: str,
                         extra: str = ""):
        """TIER 1. Finish THIS utterance on flash, same voice.

        The remainder starts at `voiced` — the character count ElevenLabs has
        actually produced audio for — so what the driver hears is one sentence
        with a change of texture in the middle, not a clause said twice.
        """
        if utt.cancelled or utt.on_flash:
            return
        utt.on_flash = True
        self._note_fallback(cause)
        remainder = (utt.sent[utt.voiced:] + extra
                     + utt.chunker.pending() + "".join(utt.chunker.drain()))
        print(f"[voice] falling back to {config.ELEVENLABS_DETERMINISTIC_MODEL} "
              f"for this line ({cause}); {len(remainder)} chars remaining",
              flush=True)
        await self._emit("fallback", {
            "tier": "flash", "cause": cause, "rid": utt.rid,
            "model": config.ELEVENLABS_DETERMINISTIC_MODEL,
            "chars": len(remainder)})
        # Whatever the socket still owed is never coming. The counters are
        # levelled so the utterance can finish on the fallback's own terms
        # rather than waiting out markers from a turn nobody is generating.
        utt.finals = utt.flushes
        if remainder.strip():
            await self._flash(utt, remainder)
        await self._maybe_done(utt)

    async def _flash(self, utt: "_Utterance", text: str):
        """One phrase through the fast model, over HTTP, as PCM."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=20.0) as http:
                async with http.stream(
                    "POST", FLASH_URL.format(voice=self.voice),
                    params={"output_format": config.ELEVENLABS_OUTPUT_FORMAT},
                    headers={"xi-api-key": api_key()},
                    json={"text": text,
                          "model_id": config.ELEVENLABS_DETERMINISTIC_MODEL},
                ) as r:
                    if r.status_code != 200:
                        raise RuntimeError(f"flash {r.status_code}")
                    said = False
                    async for pcm in r.aiter_bytes():
                        if utt.cancelled:
                            return
                        if not pcm:
                            continue
                        if not said:
                            said = True
                            if utt.first_audio is None and utt.flushed_at:
                                utt.first_audio = (
                                    time.time() - utt.flushed_at) * 1000.0
                        utt.chunks += 1
                        self.stats["audio_chunks"] += 1
                        self.stats["audio_bytes"] += len(pcm)
                        # The whole phrase's text rides on its first piece of
                        # audio: flash gives no alignment, so the browser gets
                        # the words at the moment the sound for them starts.
                        await self._deliver(utt.rid, pcm, text if said else "")
                        text = ""
            self._consecutive_failures = 0
        except Exception as e:
            print(f"[voice] fallback synthesis failed too: "
                  f"{type(e).__name__}: {str(e)[:140]}", flush=True)
            await self._fail(SYNTH_ERROR, utt.rid)

    async def _fail(self, cause: str, rid):
        """TIER 2. Count it, and hand the drive back to cedar if it keeps up."""
        self._note_fallback(cause)
        self._consecutive_failures += 1
        if (self._consecutive_failures >= config.ELEVENLABS_FAILURES_BEFORE_CEDAR
                and not self.degraded):
            self.degraded = True
            print(f"[voice] ElevenLabs is not answering ({cause}); RIO's voice "
                  f"goes back to {config.OPENAI_REALTIME_VOICE} for this drive",
                  flush=True)
            await self._emit("fallback", {
                "tier": "cedar", "cause": SERVICE_DOWN, "rid": rid,
                "after": self._consecutive_failures,
                "voice": config.OPENAI_REALTIME_VOICE})

    def _note_fallback(self, cause: str):
        self.stats["fallbacks"][cause] = self.stats["fallbacks"].get(cause, 0) + 1

    async def _deliver(self, rid: str, pcm: bytes, text: str):
        """Hand audio to whoever is listening, and survive them not being there.

        A handoff that fails is the PAGE going away, not ElevenLabs. Letting it
        propagate lands it in whichever handler happens to be on the stack — the
        read loop reads a closed browser as a dropped dialogue socket and
        reconnects one nobody is listening to; the fallback reads it as flash
        failing and starts counting towards giving up on the service. Both are
        the wrong diagnosis of a page that simply reloaded.
        """
        try:
            await self.on_audio(rid, pcm, text)
        except Exception as e:
            print(f"[voice] audio could not reach the page: "
                  f"{type(e).__name__}", flush=True)

    async def _emit(self, kind: str, detail: dict):
        try:
            await self.on_event(kind, detail)
        except Exception:
            pass

    # -- the clock ---------------------------------------------------------
    async def _tick_loop(self):
        """Three things that are true about time rather than about events.

        The max-wait rule, the first-byte budget and the keep-alive all fire
        because nothing happened, which is precisely what an event handler
        cannot notice. 50 ms is fine for all three: the tightest of them is a
        250 ms deadline.
        """
        try:
            while not self._closed:
                await asyncio.sleep(0.05)
                await self._tick()
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[voice] tick loop stopped: {type(e).__name__}", flush=True)

    async def _tick(self):
        now = time.time()
        # FIRST, and outside the lock. A fallback holds the lock for the length
        # of its HTTP stream, and a keep-alive that waited for it would be a
        # keep-alive that missed the twenty-second window it exists for.
        if (self._ws is not None
                and (now - self._last_send) * 1000.0
                > config.ELEVENLABS_KEEPALIVE_MS):
            if await self._send({"keep_alive": True}):
                self.stats["keepalives"] += 1

        async with self._lock:
            utt = self._utt
            if utt and not utt.cancelled and not utt.ended:
                for phrase in utt.chunker.due(now):
                    await self._speak(utt, phrase)
            if (utt and not utt.cancelled and not utt.on_flash
                    and utt.flushed_at and utt.first_audio is None
                    and (now - utt.flushed_at) * 1000.0
                    > config.ELEVENLABS_FIRST_BYTE_BUDGET_MS):
                await self._fall_back(utt, SLOW_FIRST_BYTE)

    # -- diagnostics -------------------------------------------------------
    def status(self) -> dict:
        first = sorted(self.stats["first_audio_ms"])
        return {
            "open": self._ws is not None,
            "degraded": self.degraded,
            "model": self.model,
            "voice_id": self.voice,
            "output_format": self.fmt,
            "utterances": self.stats["utterances"],
            "reconnects": self.stats["reconnects"],
            "keepalives": self.stats["keepalives"],
            "audio_chunks": self.stats["audio_chunks"],
            "audio_bytes": self.stats["audio_bytes"],
            "tags_dropped": self.stats["tags_dropped"],
            "fallbacks": dict(self.stats["fallbacks"]),
            "first_audio_ms_p50": _pct(first, 50),
            "first_audio_ms_p95": _pct(first, 95),
        }


def _pct(sorted_values, p):
    if not sorted_values:
        return None
    i = min(len(sorted_values) - 1,
            max(0, int(round((p / 100.0) * (len(sorted_values) - 1)))))
    return round(sorted_values[i], 1)


async def _noop():
    return None
