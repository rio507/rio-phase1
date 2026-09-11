"""A real gpt-live-1 session, driven from a script, with no browser.

Every other tool in this directory that measures the live backend uses this.
It exists for the same reason tools/live_voice_probe.js exists for the old
one: the interesting failures are in the seam between the voice layer, the
delegated backend and RIO's tools, and a seam cannot be tested by unit-testing
either side of it.

WHAT IT IS: an aiortc peer that offers audio, plays a scripted driver into the
session in real time, executes RIO's tools when the backend asks for them, and
records when the first sound came back. WebRTC because the live endpoint takes
nothing else -- "Only the webrtc transport is supported" is the API's first
answer to every other shape of request.

WHAT IT IS NOT: a cabin. It cannot tell you how often a real microphone in a
real car fires on a real bump. It tells you what the session does when it
does, which is the half that lives in this repository.
"""
import asyncio
import fractions
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv                              # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import av                                                   # noqa: E402
import httpx                                                # noqa: E402
from aiortc import RTCPeerConnection, RTCSessionDescription  # noqa: E402
from aiortc.mediastreams import MediaStreamTrack            # noqa: E402

import config                                               # noqa: E402
import live                                                 # noqa: E402

RATE = 48000          # what WebRTC/opus wants, both directions
FRAME = 960           # 20 ms
CACHE = Path(__file__).resolve().parent / "_live_cache"


def say_as_driver(text: str) -> np.ndarray:
    """One spoken question as 48 kHz mono, synthesised once and cached.

    Cached for the reason tools/voice_latency.py caches its own: the recording
    is the CONSTANT in every experiment that uses it, and re-synthesising per
    run puts a different waveform in front of the detector each time.

    Deliberately NOT in RIO's voice. This is the driver, and a driver who
    sounds exactly like her would make the echo probe meaningless.
    """
    import hashlib
    CACHE.mkdir(exist_ok=True)
    f = CACHE / f"drv_{hashlib.sha1(text.encode()).hexdigest()[:12]}.pcm"
    if f.exists() and f.stat().st_size > 1000:
        return np.frombuffer(f.read_bytes(), "<i2").astype(np.int16)
    from openai import OpenAI
    r = OpenAI().audio.speech.create(model="gpt-4o-mini-tts", voice="ash",
                                     input=text, response_format="wav")
    pcm = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", "-", "-f", "s16le", "-ac", "1",
         "-ar", str(RATE), "-"],
        input=r.content, capture_output=True).stdout
    f.write_bytes(pcm)
    return np.frombuffer(pcm, "<i2").astype(np.int16)


def silence(ms: int) -> np.ndarray:
    return np.zeros(int(RATE * ms / 1000), np.int16)


class ScriptedMic(MediaStreamTrack):
    """The cabin's microphone: whatever the script puts in it, in real time.

    Real time matters and is not an implementation detail. The turn detector
    upstream observes silence in the audio it is GIVEN and cannot observe an
    absence of audio -- the footnote in config.py's turn-end table is about
    exactly this bug -- so a track that runs as fast as it can produces
    measurements of nothing.
    """

    kind = "audio"

    def __init__(self, buf: np.ndarray):
        super().__init__()
        self.buf = buf
        self.i = 0
        self._pts = 0
        self.t0 = None
        self.marks = {}          # name -> wall clock when that sample played

    def mark_at(self, name: str, sample_index: int):
        self._pending = getattr(self, "_pending", [])
        self._pending.append((name, sample_index))

    async def recv(self):
        if self.t0 is None:
            self.t0 = time.time()
        target = self.t0 + self._pts / RATE
        now = time.time()
        if target > now:
            await asyncio.sleep(target - now)
        chunk = self.buf[self.i:self.i + FRAME]
        if len(chunk) < FRAME:
            chunk = np.concatenate([chunk, np.zeros(FRAME - len(chunk),
                                                    np.int16)])
        for name, idx in getattr(self, "_pending", []):
            if name not in self.marks and self.i + FRAME >= idx:
                self.marks[name] = time.time()
        self.i += FRAME
        f = av.AudioFrame.from_ndarray(chunk.reshape(1, -1), format="s16",
                                       layout="mono")
        f.sample_rate = RATE
        f.pts = self._pts
        f.time_base = fractions.Fraction(1, RATE)
        self._pts += FRAME
        return f


class LiveSession:
    """One session: events in, tools executed, first audio timed."""

    def __init__(self, mic: MediaStreamTrack, tool_fn=None,
                 session_override=None):
        self.mic = mic
        self.tool_fn = tool_fn
        self.session_override = session_override
        self.pc = None
        self.dc = None
        self.events = []
        self.heard = ""
        self.said = ""
        self.first_audio_t = None
        self.audio = bytearray()
        self.last_audio_t = None
        # EVERY TIME SHE STARTS TALKING, not just the first.
        #
        # `first_audio_t` alone measured the wrong thing and did it silently:
        # gpt-live-1 opens with a greeting, so the first sound on the track is
        # "Hey. What's up." and a turn timed from it comes back at 98 ms, or
        # at -32 ms, for an answer that had not been thought about yet. An
        # onset is a transition from quiet to speech, and the one that matters
        # is the first one AFTER the driver stopped.
        self.onsets = []
        self.tool_calls = []
        self.errors = []
        # WHAT THE SERVER SAYS THE SESSION IS, which is not what the HTTP
        # response says. The mint returns the SDP answer and little else; the
        # configuration it actually accepted arrives on the data channel as
        # `session.started`. Reading it from the HTTP body looked right and
        # reported a session with no model, no voice and no backend attached.
        self.session = {}

    async def open(self):
        self.pc = RTCPeerConnection()
        self.pc.addTrack(self.mic)

        @self.pc.on("track")
        def _(track):
            asyncio.ensure_future(self._pump(track))

        self.dc = self.pc.createDataChannel("oai-events")

        @self.dc.on("message")
        def _(m):
            try:
                e = json.loads(m)
            except Exception:
                return
            self._on_event(e)

        await self.pc.setLocalDescription(await self.pc.createOffer())
        sess = self.session_override or live.session_config()
        r = httpx.post(live.LIVE_SESSIONS_URL,
                       headers={"Authorization": f"Bearer {live._key()}",
                                "Content-Type": "application/json"},
                       json={"transport": {"type": "webrtc",
                                           "sdp": self.pc.localDescription.sdp},
                             "session": sess},
                       timeout=45.0)
        if r.status_code >= 400:
            raise RuntimeError(f"mint {r.status_code}: {r.text[:300]}")
        body = r.json()
        await self.pc.setRemoteDescription(RTCSessionDescription(
            sdp=body["transport"]["sdp"], type="answer"))
        self.mint_body = body
        for _ in range(120):
            if self.dc.readyState == "open":
                break
            await asyncio.sleep(0.05)
        # ...and then for the session echo, which comes over the channel.
        for _ in range(100):
            if self.session:
                break
            await asyncio.sleep(0.05)
        return self

    async def _pump(self, track):
        while True:
            try:
                fr = await track.recv()
            except Exception:
                return
            a = fr.to_ndarray().reshape(-1).astype(np.int16)
            if np.abs(a).max() > 200:
                now = time.time()
                if self.last_audio_t is None or now - self.last_audio_t > 0.35:
                    self.onsets.append(now)
                if self.first_audio_t is None:
                    self.first_audio_t = now
                self.last_audio_t = now
            if self.first_audio_t is not None:
                self.audio += a.tobytes()

    def _on_event(self, e: dict):
        self.events.append((time.time(), e))
        t = e.get("type")
        if t == "session.started":
            self.session = e.get("session") or {}
        elif t == "session.input_transcript.delta":
            self.heard += e.get("delta", "")
        elif t == "session.output_transcript.delta":
            self.said += e.get("delta", "")
        elif t == "error":
            self.errors.append(e)
        call = live.parse_tool_call(e)
        if call and self.tool_fn:
            self.tool_calls.append(call)
            out = self.tool_fn(call["name"], call["arguments"])
            for ev in live.tool_result_events(call["call_id"], out):
                self.send(ev)

    def send(self, event: dict):
        self.dc.send(json.dumps(event))

    def say_line(self, text: str, event_id="line"):
        """A deterministic line, through the event that is actually verbatim."""
        self.send(live.verbatim_event(text, event_id=event_id))

    def say_commentary(self, text: str, event_id="c"):
        """...and the other one, for the bench that compares them."""
        self.send(live.commentary_event(text, event_id=event_id))

    def onset_after(self, t: float):
        """The first time she STARTED speaking after `t`, or None.

        None is a real answer and not a missing one: it means she was already
        talking across that moment, and a turn measured through her own
        previous sentence is not a measurement of this one.
        """
        for x in self.onsets:
            if x > t:
                return x
        return None

    async def wait_quiet(self, max_s=25.0, quiet_s=0.9):
        """Until she has started and then stopped for `quiet_s`."""
        t0 = time.time()
        while time.time() - t0 < max_s:
            await asyncio.sleep(0.1)
            if self.last_audio_t and time.time() - self.last_audio_t > quiet_s:
                return True
        return False

    async def close(self):
        try:
            await self.pc.close()
        except Exception:
            pass
