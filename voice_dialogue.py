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
MULTI_URL = "wss://api.elevenlabs.io/v1/text-to-speech/{voice}/multi-stream-input"
FLASH_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice}/stream"

# How long the service holds a quiet socket open. The text-to-speech socket
# lets this be asked for (up to 180 s); the dialogue socket does not and is
# fixed at 20. Both are kept alive from here anyway — this only decides how
# much slack there is if a keep-alive is ever late.
MULTI_INACTIVITY_S = 180

# Causes, named once. These are what /voice/status counts and what the log
# lines say, and having them in one place is what stops "error" becoming the
# answer to every question about why RIO sounded like somebody else.
SLOW_FIRST_BYTE = "slow_first_byte"
SOCKET_ERROR = "socket_error"
SOCKET_UNAVAILABLE = "socket_unavailable"
SYNTH_ERROR = "synth_error"
SERVICE_DOWN = "service_down"
# Every dialogue seat in the workspace is taken. Its own cause, because it is
# its own condition: nothing is broken, nothing will get better by retrying,
# and the thing that fixes it is another car finishing a drive.
NO_SEAT = "no_dialogue_seat"

# How the service says so. It arrives two ways — as an error message on the
# socket, and as the REASON on a 1008 close — and the reason text is all a
# closed connection leaves behind, so both are matched on the same marker.
_NO_SEAT_MARKERS = ("too_many_concurrent_requests", "too many concurrent requests")


def _is_capacity(text) -> bool:
    low = str(text or "").lower()
    return any(m in low for m in _NO_SEAT_MARKERS)


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

    def __init__(self, min_tokens=None, max_wait_ms=None,
                 first_min_tokens=None, first_max_wait_ms=None):
        self._min_tokens = int(config.ELEVENLABS_CHUNK_MIN_TOKENS
                               if min_tokens is None else min_tokens)
        self._max_wait_ms = float(config.ELEVENLABS_CHUNK_MAX_WAIT_MS
                                  if max_wait_ms is None else max_wait_ms)
        # The first phrase of an answer is the only one with nothing playing
        # behind it, so it gets a lower bar. See config.
        self._first_min_tokens = int(
            config.ELEVENLABS_FIRST_CHUNK_MIN_TOKENS
            if first_min_tokens is None else first_min_tokens)
        self._first_max_wait_ms = float(
            config.ELEVENLABS_FIRST_CHUNK_MAX_WAIT_MS
            if first_max_wait_ms is None else first_max_wait_ms)
        self.spoke = False       # has anything gone out for this utterance?
        self.buf = ""
        self.since = None        # when the oldest unsent character arrived

    @property
    def min_tokens(self):
        return self._first_min_tokens if not self.spoke else self._min_tokens

    @property
    def max_wait_ms(self):
        return self._first_max_wait_ms if not self.spoke else self._max_wait_ms

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
            self.spoke = True
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
        self.spoke = True
        self.since = now if self.buf.strip() else None
        return [phrase]

    def drain(self) -> list:
        """Everything left, because the model has stopped writing."""
        left = self.buf
        self.buf, self.since = "", None
        if left.strip():
            self.spoke = True
            return [left]
        return []

    def pending(self) -> str:
        return self.buf


# ---------------------------------------------------------------------------
# TWO SOCKETS, ONE CONVERSATION
# ---------------------------------------------------------------------------
# ElevenLabs streams over two different websockets and which one a voice needs
# is decided by the MODEL, not by anything about the car:
#
#   text-to-dialogue/stream-input        eleven_v3 models, and only those
#   text-to-speech/{voice}/multi-stream-input   everything else
#
# Everything above and below this section is the same either way — the phrase
# chunker, the tag gate, both fallback tiers, the keep-alive, the capacity
# parking, the character accounting a resume depends on. What differs is the
# shape of four messages and the name of three fields, and that is exactly what
# a dialect is for: the difference is written down once, here, instead of
# spreading `if v3` through the session.
#
# The multi-context socket is the better-behaved of the two, and it is worth
# saying why rather than treating it as a detail. Every audio frame it sends
# carries the CONTEXT it belongs to, so a cancelled utterance's audio can be
# dropped by name. The dialogue socket sends audio with nothing on it that says
# which turn it came from, so the only safe way to abandon a turn there is to
# throw the whole connection away and open another. One of those is a design;
# the other is a workaround for a missing field.
class _Dialect:
    """The wire, and nothing else. No policy lives in here."""

    #: Does a cancelled utterance need the connection thrown away?
    recycle_on_cancel = True

    def __init__(self, voice: str, model: str, fmt: str):
        self.voice, self.model, self.fmt = voice, model, fmt

    def url(self) -> str:
        raise NotImplementedError

    def hello(self) -> list:
        """Messages to send the moment the socket opens."""
        raise NotImplementedError

    def begin(self, rid: str) -> list:
        return []

    def speak(self, rid: str, phrase: str, first: bool) -> list:
        raise NotImplementedError

    def finish(self, rid: str) -> list:
        """The model has stopped writing. Say the rest and end the turn."""
        return []

    def abandon(self, rid: str) -> list:
        """Give up on this utterance without ending the session."""
        return []

    def keepalive(self) -> dict:
        raise NotImplementedError

    def goodbye(self) -> dict:
        return {"close_socket": True}

    def read(self, m: dict) -> dict:
        """One wire message -> {kind, rid, pcm, text} in one vocabulary."""
        raise NotImplementedError


class _DialogueDialect(_Dialect):
    """eleven_v3, over text-to-dialogue/stream-input.

    One turn per utterance, `flush` per phrase, and an end-of-turn marker per
    flush rather than per turn — which is why the session counts both. Audio
    arrives unlabelled, so a cancel recycles the connection.
    """

    recycle_on_cancel = True

    def url(self):
        return (f"{WS_URL}?model_id={self.model}&output_format={self.fmt}"
                "&sync_alignment=true")

    def hello(self):
        return [{"voices": [self.voice]}]

    def speak(self, rid, phrase, first):
        return [{"inputs": [{"text": phrase, "voice_id": self.voice,
                             "new_turn": first}]},
                {"flush": True}]

    def keepalive(self):
        return {"keep_alive": True}

    def read(self, m):
        if m.get("error"):
            return {"kind": "error", "code": m.get("error"),
                    "message": m.get("message")}
        if m.get("audio"):
            chars = ((m.get("alignment") or {}).get("chars")) or []
            return {"kind": "audio", "rid": None,
                    "pcm": base64.b64decode(m["audio"]), "text": "".join(chars)}
        if m.get("is_final_audio_for_turn"):
            return {"kind": "flushed", "rid": None}
        return {"kind": "other"}


class _MultiContextDialect(_Dialect):
    """Everything else, over text-to-speech/{voice}/multi-stream-input.

    One CONTEXT per utterance, named for the response it belongs to. Audio
    comes back labelled with it and `isFinal` arrives per context, so an
    utterance can be finished or abandoned by name and the socket carries on —
    no counting, no recycling.

    `chunk_length_schedule` opens low on purpose. The service buffers text to
    improve quality before generating, and the first number is how long it
    waits for the FIRST piece — which is the one the driver is sitting in
    silence for. Every phrase is flushed explicitly anyway, so the schedule
    only decides what happens if a flush is ever late.
    """

    recycle_on_cancel = False

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        # ONCE PER CONNECTION, NOT ONCE PER UTTERANCE.
        #
        # The service is strict about this and says so plainly: "voice_settings
        # field must be provided in the first message and then either be not
        # provided or not change." Sending it again on the second context does
        # not warn — it closes the socket with a 1008.
        #
        # Which is exactly what it did. Every utterance after the first killed
        # the connection, fell back to flash for that line, and reconnected;
        # the drive kept speaking, in the wrong model, reconnecting once per
        # sentence. The tests did not catch it because a fallback that still
        # produces audio still passes "she said something" — see the check
        # added for it in voice_selftest.
        self._greeted = False

    def url(self):
        return (MULTI_URL.format(voice=self.voice)
                + f"?model_id={self.model}&output_format={self.fmt}"
                + f"&sync_alignment=true&inactivity_timeout={MULTI_INACTIVITY_S}")

    def hello(self):
        # A fresh connection has said nothing yet, and the settings ride on the
        # first context rather than on their own message so that opening a
        # socket costs no synthesis.
        self._greeted = False
        return []

    def _settings(self):
        return {
            "voice_settings": {"stability": 0.5, "similarity_boost": 0.75,
                               "use_speaker_boost": True},
            "generation_config": {"chunk_length_schedule": [50, 120, 160, 250]},
        }

    def begin(self, rid):
        msg = {"text": " ", "context_id": rid}
        if not self._greeted:
            self._greeted = True
            msg.update(self._settings())
        return [msg]

    def speak(self, rid, phrase, first):
        return [{"text": phrase, "context_id": rid, "flush": True}]

    def finish(self, rid):
        return [{"context_id": rid, "close_context": True}]

    def abandon(self, rid):
        # Ends the context so nothing more is generated for it. Audio already
        # committed still arrives, and is dropped by name on the way in.
        return [{"context_id": rid, "close_context": True}]

    def keepalive(self):
        # No context, so it cannot disturb an utterance in flight.
        return {"text": " "}

    def read(self, m):
        if m.get("error"):
            return {"kind": "error", "code": m.get("error"),
                    "message": m.get("message")}
        rid = m.get("contextId")
        if m.get("audio"):
            chars = ((m.get("alignment") or {}).get("chars")) or []
            return {"kind": "audio", "rid": rid,
                    "pcm": base64.b64decode(m["audio"]), "text": "".join(chars)}
        if m.get("isFinal"):
            return {"kind": "done", "rid": rid}
        return {"kind": "other"}


def dialect_for(voice: str, model: str, fmt: str) -> _Dialect:
    cls = (_DialogueDialect if config.uses_dialogue_socket(model)
           else _MultiContextDialect)
    return cls(voice, model, fmt)


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
        self.model = model or config.ELEVENLABS_CONVERSATION_MODEL
        self.fmt = fmt or config.ELEVENLABS_OUTPUT_FORMAT
        self.wire = dialect_for(self.voice, self.model, self.fmt)

        self._ws = None
        self._reader = None
        self._ticker = None
        self._lock = asyncio.Lock()
        self._closed = False
        self._utt = None
        self._last_send = 0.0
        self._consecutive_failures = 0
        # While this is in the future the dialogue socket is parked and every
        # utterance goes to flash. See ELEVENLABS_CAPACITY_BACKOFF_S.
        self._no_seat_until = 0.0

        self.degraded = False     # tier 2: ElevenLabs is out, cedar has it
        self.stats = {"utterances": 0, "reconnects": 0, "keepalives": 0,
                      "fallbacks": {}, "audio_chunks": 0, "audio_bytes": 0,
                      "first_audio_ms": [], "tags_dropped": 0}

    # -- lifecycle ---------------------------------------------------------
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
                self.wire.url(), additional_headers={"xi-api-key": api_key()},
                max_size=None, open_timeout=10)
            for msg in self.wire.hello():
                await self._ws.send(json.dumps(msg))
            self._last_send = time.time()
            return True
        except Exception as e:
            print(f"[voice] {self.model} socket would not open: "
                  f"{type(e).__name__}: {str(e)[:160]}", flush=True)
            self._ws = None
            return False

    def _parked(self) -> bool:
        """Is the dialogue socket parked because the pool was full?"""
        return time.time() < self._no_seat_until

    def _pool_name(self) -> str:
        """Which pool ran out — they are different pools, and different sizes.

        The dialogue socket draws on a dedicated pool (measured at 21 on this
        account). Every other model streams over the text-to-speech socket and
        draws on ORDINARY concurrency, which on the same account is a single
        digit. Naming the right one is the difference between a log line
        somebody can act on and one that sends them to the wrong dashboard.
        """
        return ("dialogue session" if self.wire.recycle_on_cancel
                else f"concurrent {self.model} request")

    async def _note_no_seat(self, where: str):
        """Park the socket, and say so ONCE.

        Once because this repeats per utterance otherwise, and a log that
        repeats is a log nobody reads. The drive carries on in the same voice
        on flash, which is the point: a full pool costs prosody, not speech.
        """
        first = not self._parked()
        self._no_seat_until = time.time() + config.ELEVENLABS_CAPACITY_BACKOFF_S
        # Let the connection go, whichever way the refusal arrived.
        #
        # It costs no seat to hold one — that is the whole surprise of this
        # pool — but it buys nothing either: every synthesis on it is refused
        # until a seat frees, and keeping it alive leaves TWO ways back to a
        # working socket depending on whether the service closed the connection
        # or merely answered on it. One way is easier to be sure of, and the
        # tick's seat retry is that way.
        ws, self._ws = self._ws, None
        if ws:
            try:
                await ws.close()
            except Exception:
                pass
        if not first:
            return
        self._note_fallback(NO_SEAT)
        print(f"[voice] every {self._pool_name()} in the workspace is in use "
              f"({where}); this drive runs on "
              f"{config.ELEVENLABS_DETERMINISTIC_MODEL} for the next "
              f"{config.ELEVENLABS_CAPACITY_BACKOFF_S:.0f}s", flush=True)
        await self._emit("fallback", {
            "tier": "flash", "cause": NO_SEAT, "rid": None,
            "pool": self._pool_name(),
            "model": config.ELEVENLABS_DETERMINISTIC_MODEL,
            "retry_in_s": config.ELEVENLABS_CAPACITY_BACKOFF_S})

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
                await ws.send(json.dumps(self.wire.goodbye()))
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
            print(f"[voice] {self.model} socket reconnected ({why})", flush=True)
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
            print(f"[voice] send failed on the {self.model} socket: "
                  f"{type(e).__name__}", flush=True)
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
            if self._closed or self._ws is not ws:
                # A close WE caused — the session ending, or a capacity refusal
                # that has already parked this socket. Reporting it as a drop
                # is how a deliberate act ends up in the log looking like a
                # fault, next to the line that explains what really happened.
                return
            why = type(e).__name__
            # A 1008 close carrying `too_many_concurrent_requests` is not a
            # dropped socket, and reconnecting is the wrong reflex: the new
            # connection is accepted (a connection costs no seat) and the next
            # utterance is refused exactly the same way.
            if _is_capacity(e):
                why = NO_SEAT
                await self._note_no_seat("closed by the service")
            else:
                print(f"[voice] {self.model} socket dropped: {why}: {str(e)[:120]}",
                      flush=True)
        if self._closed or self._ws is not ws:
            return          # a deliberate close, or already replaced
        # An utterance in flight has to be rescued before the socket is
        # replaced, or the driver hears half an answer and no more.
        utt = self._utt
        if utt and not utt.done and not utt.cancelled:
            await self._fall_back(utt, NO_SEAT if why == NO_SEAT else SOCKET_ERROR)
        if why == NO_SEAT:
            self._ws = None       # parked, not replaced
            return
        await self._reconnect(why)

    async def _on_message(self, raw: dict):
        """One wire message, read through the dialect and acted on once.

        Everything below is about the CAR: whose utterance this belongs to,
        whether it has been abandoned, how much of it the driver has heard.
        None of it knows which socket it came from, which is the point.
        """
        m = self.wire.read(raw)
        kind = m.get("kind")

        if kind == "error":
            capacity = _is_capacity(m.get("code")) or _is_capacity(m.get("message"))
            if capacity:
                await self._note_no_seat("refused on the socket")
            else:
                print(f"[voice] {self.model} error: {str(m.get('message'))[:160]}",
                      flush=True)
            utt = self._utt
            if utt and not utt.done and not utt.cancelled:
                await self._fall_back(utt, NO_SEAT if capacity else SOCKET_ERROR)
            return

        if kind == "audio":
            utt = self._utt
            # Audio for something that has been abandoned — cancelled by a
            # barge-in, or moved onto flash — is dropped rather than played.
            # `rid` is the strong version of that test and is available on the
            # multi-context socket; the dialogue socket sends nothing that says
            # which turn its audio is for, which is why a cancel there has to
            # throw the connection away instead.
            if utt is None or utt.cancelled or utt.on_flash:
                return
            if m.get("rid") and m["rid"] != utt.rid:
                return
            pcm, text = m["pcm"], m.get("text") or ""
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

        if kind == "flushed":
            # The dialogue socket answers per FLUSH, not per utterance. See
            # _Utterance for why both sides are counted.
            utt = self._utt
            if utt:
                utt.finals += 1
                await self._maybe_done(utt)
            return

        if kind == "done":
            # The multi-context socket answers per CONTEXT, which is per
            # utterance, so there is nothing to count.
            utt = self._utt
            if utt and (not m.get("rid") or m["rid"] == utt.rid):
                utt.finals = utt.flushes
                await self._maybe_done(utt)
            return

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
        # A stale turn only forces a new connection where audio is unlabelled.
        # With a context per utterance the old one is closed by name and the
        # socket carries on, which is a reconnect a drive does not have to pay.
        if stale and self._ws is not None and self.wire.recycle_on_cancel:
            await self._reconnect("superseded")
        elif self._ws is not None:
            if stale:
                for msg in self.wire.abandon(rid):
                    await self._send(msg)
            for msg in self.wire.begin(rid):
                await self._send(msg)

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
            # NO EXTRA FLUSH. Every phrase is flushed as it is sent, so there
            # is nothing buffered — and a flush over an empty buffer generates
            # no audio and returns no end-of-turn marker. One sent anyway
            # leaves the counters permanently one apart, and an utterance that
            # can never report itself finished is a mouth never handed back.
            #
            # What the dialect may still need is a way to say the turn is OVER,
            # which is what produces the per-context final on the
            # multi-context socket.
            if not utt.on_flash:
                for msg in self.wire.finish(utt.rid):
                    await self._send(msg)
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
        if self._ws is None:
            return
        if self.wire.recycle_on_cancel:
            await self._reconnect("cancelled")
        else:
            # Close the context and drop anything still arriving for it by
            # name. No reconnect, so a barge-in costs nothing but a message.
            for msg in self.wire.abandon(rid):
                await self._send(msg)

    # -- speaking, and not being able to ------------------------------------
    async def _speak(self, utt: "_Utterance", phrase: str):
        if not phrase.strip():
            return
        if self._parked() and not utt.on_flash:
            # The pool was full a moment ago. Going back for a refusal per
            # phrase costs a round trip and produces nothing.
            utt.on_flash = True
            utt.finals = utt.flushes
        if utt.on_flash:
            if utt.flushed_at is None:
                utt.flushed_at = time.time()
            await self._flash(utt, phrase)
            return
        first = not utt.sent
        ok = True
        for msg in self.wire.speak(utt.rid, phrase, first):
            ok = await self._send(msg)
            if not ok:
                break
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
            if await self._send(self.wire.keepalive()):
                self.stats["keepalives"] += 1

        # The back-off has run out and there is no socket. Try for a seat
        # again — quietly, between utterances, where a failure costs nothing.
        if (self._ws is None and not self._closed and not self.force_flash
                and self._no_seat_until and not self._parked()):
            self._no_seat_until = 0.0
            await self._reconnect("seat retry")

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
            "transport": ("text_to_dialogue" if self.wire.recycle_on_cancel
                          else "multi_context_tts"),
            "voice_id": self.voice,
            "output_format": self.fmt,
            # Parked on flash because the workspace's dialogue pool was full.
            # Distinct from `degraded`, which is ElevenLabs not answering at
            # all: this one is working exactly as designed, in a smaller voice.
            "no_seat": self._parked(),
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
