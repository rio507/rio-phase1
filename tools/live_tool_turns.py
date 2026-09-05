"""live_tool_turns.py — the four questions that were failing, over a real drive.

    python -m tools.live_tool_turns --out /tmp/rio_tools.wav

WHAT THIS IS FOR
----------------
Three live failures shared one shape: plain conversation worked and anything
that needed a TOOL did not. "What do you see outside" froze with no audio and
no answer; a question that needed the reasoning model never came back with one;
asked for the directions she said the car would read them, which is the
behaviour from before nav_directions existed.

Everything about that shape lives in the gap between a tool result and a sound,
and nothing could reach it. The offline suites drive the controller with events
a test wrote. The recording tool drives a real session with its OWN copy of the
controller's logic -- which is how it once contained the very bug it was
recording. Neither could have caught it.

So this runs the SHIPPED page against the SHIPPED server:

    the session      minted over HTTP from /realtime/session, so the payload
                     the browser actually gets is the payload under test
    the controller   static/rio_realtime.js, the file itself
    the sink         static/rio_voice_eleven.js, the file itself, decoding and
                     scheduling real PCM off the real relay
    the tools        the panel's own for the route, /realtime/tool over HTTP
                     for the camera and the reasoning model
    the voice        the real /voice/dialogue socket and real ElevenLabs

and writes one WAV of what a listener would have heard, in the order and at the
times they would have heard it. A question that produced no sound leaves a
silence in the file, which is the failure it is.

Python owns the two sockets because node 18 has neither; node owns every
decision, because that is the code being tested.

ONE DIFFERENCE FROM THE CAR, stated so nobody chases it: the panel reaches the
live session over WebRTC and this reaches it over a WebSocket. The event stream
is the same, which is why the controller does not know or care -- but
`output_audio_buffer.clear`, which a cancel sends alongside `response.cancel`,
is a WebRTC-only event and comes back here as an `error`. It is noise in this
harness and correct in the browser. Every other event on both sockets is the
same one the page would see.
"""
import argparse
import asyncio
import base64
import json
import subprocess
import sys
import time
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv                              # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                               # noqa: E402
import voice_dialogue as vd                                 # noqa: E402

REPO = Path(__file__).resolve().parent.parent
RATE = config.ELEVENLABS_SAMPLE_RATE

# The driver, in someone else's voice so a listener can tell the two apart.
DRIVER_VOICE = "CwhRBWXzGAHq8TQ4Fs17"        # Roger

# The four turns, and what each one is evidence for. Every one of them is a
# question the driver actually asked and did not get an answer to.
SCRIPT = [
    {"say": "What do you see outside?",
     "want": "the camera's fast path — the observation, spoken as written",
     "budget_s": 3.0},
    {"say": "What kind of car is in front of us?",
     "want": "the full visual turn — composed by her, from a fresh look",
     "budget_s": 20.0},
    # ASKED THE WAY A DRIVER ASKS FOR RESEARCH, and asked straight after two
    # camera questions on purpose: that is the sequence that used to make the
    # reasoning model unreachable, because any look inside sixty seconds
    # refused every deep_dive that followed it whatever it was about.
    {"say": "Can you look up why carmakers switched from hydraulic to "
            "electric power steering?",
     "want": "depth — a holding line, then a real answer",
     "budget_s": 40.0},
    {"say": "Take me to the Getty.",
     "want": "start_navigation, and one line confirming it",
     "budget_s": 20.0},
    {"say": "What are the directions?",
     "want": "nav_directions, read aloud — the turn that resurfaced the old "
             "behaviour",
     "budget_s": 25.0},
    # HOW SHE TALKS ABOUT THE CAR, and it is here because the delivery of that
    # changed: the health register used to ride in every response and now
    # arrives with the data, in vehicle_status's `rules`. The thing to listen
    # for is that it is interpreted rather than recited — "about where they
    # should be", not "twenty-nine PSI" — and that nothing is claimed past the
    # window the data covers.
    {"say": "How are my tires?",
     "want": "the car, interpreted rather than recited",
     "budget_s": 25.0},
]


def silence(ms):
    return b"\x00\x00" * int(RATE * ms / 1000)


def say_as_driver(text: str) -> bytes:
    import httpx
    r = httpx.post(vd.FLASH_URL.format(voice=DRIVER_VOICE),
                   params={"output_format": f"pcm_{RATE}"},
                   headers={"xi-api-key": vd.api_key()},
                   json={"text": text,
                         "model_id": config.ELEVENLABS_DETERMINISTIC_MODEL},
                   timeout=60.0)
    r.raise_for_status()
    return r.content


class Panel:
    """The node process, and the two sockets it cannot open for itself."""

    def __init__(self, base, session_id, session):
        self.base = base
        self.session_id = session_id
        self.session = session
        self.proc = None
        self.notes = []
        self.oai = None
        self.relay = None

    async def start(self):
        self.proc = await asyncio.create_subprocess_exec(
            "node", str(REPO / "tools" / "live_tool_turns.js"),
            self.base, self.session_id, json.dumps(self.session),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=None, cwd=str(REPO))

    def tell(self, obj):
        self.proc.stdin.write((json.dumps(obj) + "\n").encode())

    async def pump_node(self):
        """Everything the page wants to send, sent."""
        while True:
            line = await self.proc.stdout.readline()
            if not line:
                return
            try:
                m = json.loads(line)
            except Exception:
                continue
            if m["k"] == "send" and self.oai:
                await self.oai.send(json.dumps(m["obj"]))
            elif m["k"] == "wire" and self.relay:
                await self.relay.send(json.dumps(m["obj"]))
            elif m["k"] == "note":
                self.notes.append(m)

    async def pump_session(self):
        async for raw in self.oai:
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            self.notes.append({"k": "wire_in", "type": ev.get("type"),
                               "ev": ev})
            self.tell({"k": "ev", "ev": ev})

    async def pump_relay(self):
        async for raw in self.relay:
            try:
                self.tell({"k": "wire", "m": json.loads(raw)})
            except Exception:
                pass


async def run(base, out_path, video, dump=None):
    import httpx
    import websockets

    from tools.visual_latency import HttpDrive

    http = httpx.AsyncClient(timeout=90.0)
    driver_track = []

    with HttpDrive(base, None, video, 0.25) as drive:
        sid = drive.session_id
        print(f"  drive {sid[:8]} — frames going in, camera warming")

        # THE SESSION THE BROWSER GETS. Over HTTP, from the running server.
        r = await http.post(f"{base}/realtime/session", params={"session_id": sid})
        session = r.json()
        if not session.get("client_secret"):
            print(f"  could not mint a session: {session}")
            return 1
        print(f"    minted: {session['model']}, "
              f"{session['output_modalities']}, "
              f"{len(session['tools'])} tools "
              f"({', '.join(session['tools'])})")
        assert session["output_modalities"] == ["text"], \
            "this drive is only interesting in text mode"

        # LET THE CAMERA GET AHEAD OF THE QUESTION. The fast path answers
        # from an observation written in the last couple of seconds; asked
        # before the loop has produced one, the same question takes the full
        # remote turn and this measures the wrong thing. Eight seconds is well
        # past the first few observations and still nothing anyone waits for.
        await asyncio.sleep(8)

        panel = Panel(base, sid, session)
        await panel.start()

        oai_url = ("wss://api.openai.com/v1/realtime?model="
                   + config.OPENAI_REALTIME_MODEL)
        ws_base = base.replace("http://", "ws://").replace("https://", "wss://")
        async with websockets.connect(
                oai_url,
                additional_headers={
                    "Authorization": "Bearer " + session["client_secret"]},
                max_size=None) as oai, \
                websockets.connect(
                    f"{ws_base}/voice/dialogue?session_id={sid}",
                    max_size=None) as relay:
            panel.oai, panel.relay = oai, relay
            pumps = [asyncio.create_task(panel.pump_node()),
                     asyncio.create_task(panel.pump_session()),
                     asyncio.create_task(panel.pump_relay())]
            await asyncio.sleep(1.5)     # sockets settle, relay says ready

            for turn in SCRIPT:
                q_pcm = say_as_driver(turn["say"])
                driver_track.append((time.perf_counter(), q_pcm))
                panel.tell({"k": "mark", "label": "ask:" + turn["say"]})
                mark = len(panel.notes)
                t_ask = time.perf_counter()

                # THE DRIVER SPEAKS, AND THE MICROPHONE KEEPS RUNNING.
                #
                # The tail used to be sized from REALTIME_VAD_SILENCE_MS, and
                # that broke the moment turn detection stopped being a silence
                # timer: a detector observes silence in the audio it is given
                # and cannot observe an absence of audio, so a tail shorter
                # than its slowest decision leaves the turn hanging. Three of
                # six questions in a drive went unanswered that way, and then
                # arrived merged into one turn — a fault entirely in this
                # file. A real cabin never stops sending.
                payload = q_pcm + silence(4000)
                step = int(RATE * 20 / 1000) * 2
                start = time.perf_counter()
                for i in range(0, len(payload), step):
                    delay = start + (i / 2) / RATE - time.perf_counter()
                    if delay > 0:
                        await asyncio.sleep(delay)
                    await oai.send(json.dumps({
                        "type": "input_audio_buffer.append",
                        "audio": base64.b64encode(payload[i:i + step]).decode()}))

                # Wait for the answer, then for the speaker to run out. A
                # script that asks the next question over the tail of the last
                # answer produces barge-ins that are the script's fault, and a
                # recording of this harness rather than of her.
                await asyncio.sleep(turn["budget_s"])
                await settle(panel, mark)
                report(turn, panel.notes[mark:], t_ask)

            # THE GATE ITSELF, asked directly, because the model choosing to
            # answer from memory and the tool refusing to run are
            # indistinguishable from the passenger seat -- and it was the
            # second one.
            gate = (await http.post(
                f"{base}/realtime/tool", params={"session_id": sid},
                json={"name": "deep_dive",
                      "arguments": {"question": "why do carmakers use "
                                                "electric power steering"}})
                    ).json()
            print(f"\n  deep_dive, seconds after two camera questions: "
                  f"ok={gate.get('ok')} "
                  f"{'REFUSED: ' + str(gate.get('reason')) if not gate.get('ok') else ''}")
            if gate.get("ok"):
                print(f"      {str(gate.get('answer'))[:200]!r}")

            panel.tell({"k": "wav", "path": str(out_path)})
            await asyncio.sleep(2.0)
            wav = [n for n in panel.notes if n.get("note") == "wav"]
            if wav:
                print(f"\n  wrote {out_path}  ({wav[0]['seconds']:.1f}s of RIO)")
                print(f"  counters: {json.dumps(wav[0]['counters'])}")
            allow = [lim for n in panel.notes
                     if n.get("type") == "rate_limits.updated"
                     for lim in (n["ev"].get("rate_limits") or [])
                     if lim.get("name") == "tokens"]
            if allow:
                low = min(allow, key=lambda l: l.get("remaining", 0))
                print(f"\n  lowest the minute's token budget got over the "
                      f"whole drive: {low.get('remaining'):,} of "
                      f"{low.get('limit'):,}")
            if dump:
                Path(dump).write_text(
                    "\n".join(json.dumps(n) for n in panel.notes))
                print(f"  every event of the drive: {dump}")
            panel.tell({"k": "bye"})
            for p in pumps:
                p.cancel()

    await http.aclose()
    return 0


async def settle(panel, mark, quiet_s=2.5, cap_s=12.0):
    """Wait until nothing has been said for a moment, or long enough."""
    deadline = time.perf_counter() + cap_s
    last = len(panel.notes)
    quiet_since = time.perf_counter()
    while time.perf_counter() < deadline:
        await asyncio.sleep(0.25)
        if len(panel.notes) != last:
            last = len(panel.notes)
            quiet_since = time.perf_counter()
        elif time.perf_counter() - quiet_since >= quiet_s:
            return


def report(turn, notes, t_ask):
    """What happened to one question, in the terms the failure was reported."""
    said = "".join(
        n["ev"].get("delta", "") for n in notes
        if n.get("k") == "wire_in"
        and n.get("type") == "response.output_text.delta")
    direct = [n for n in notes if n.get("note") == "live"
              and n["ev"].get("type") == "LIVE_DIRECT_ANSWER"]
    tools = [n for n in notes if n.get("note") == "tool"]
    fails = [n for n in notes if n.get("note") == "live"
             and n["ev"].get("type") == "LIVE_RESPONSE_FAILED"]
    heard = [n for n in notes if n.get("note") == "voice"
             and n["ev"].get("type") == "VOICE_UTTERANCE_DONE"]
    # WHAT THE MINUTE HAS LEFT, from the session itself. The ceiling the
    # instructions are sized against is not a number out of a document: the
    # session reports its own limit and what is left of it on every response,
    # and this is the only place that can watch a real drive approach it.
    limits = [lim for n in notes if n.get("type") == "rate_limits.updated"
              for lim in (n["ev"].get("rate_limits") or [])
              if lim.get("name") == "tokens"]
    transcript = [n["ev"].get("transcript") for n in notes
                  if n.get("k") == "wire_in" and n.get("type") ==
                  "conversation.item.input_audio_transcription.completed"]

    spoken = said or (direct[0]["ev"]["text"] if direct else "")
    print(f"\n  {turn['say']!r}   ({turn['want']})")
    print(f"      heard as: {transcript}")
    for t in tools:
        print(f"      tool {t['name']}({json.dumps(t.get('args'))[:80]}) "
              f"-> ok={t['ok']} path={t['path']} direct={t['direct']} "
              f"({t['ms']} ms)"
              + (f"  note={t.get('note_text')!r}" if t.get("note_text") else "")
              + (f"\n        speech={t.get('speech')!r}" if t.get("speech") else ""))
    if fails:
        for f in fails:
            print(f"      RESPONSE FAILED: {f['ev'].get('code')} "
                  f"retrying={f['ev'].get('retrying')}")
    print(f"      said: {spoken[:220]!r}")
    print(f"      utterances that reached the speaker: {len(heard)}"
          + ("   <-- SILENT" if not heard else ""))
    if limits:
        low = min(limits, key=lambda l: l.get("remaining", 0))
        print(f"      tokens left in the minute: {low.get('remaining'):,} of "
              f"{low.get('limit'):,}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://127.0.0.1:8888")
    ap.add_argument("--video", default="/workspace/ufldv2/example.mp4")
    ap.add_argument("--out", default="/tmp/rio_tools.wav")
    ap.add_argument("--dump", default=None,
                    help="write every event of the drive to this file")
    args = ap.parse_args()
    return asyncio.run(run(args.base, Path(args.out), args.video,
                           args.dump))


if __name__ == "__main__":
    raise SystemExit(main())
