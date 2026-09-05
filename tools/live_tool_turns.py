"""live_tool_turns.py — the questions that were failing, over a real drive.

    python -m tools.live_tool_turns --out /tmp/rio_tools.wav
    python -m tools.live_tool_turns --script nav --out /tmp/rio_nav.wav

TWO DRIVES, ONE RIG. `--script tools` (the default) asks the six questions
below; `--script nav` starts a route by voice, runs the simulator through it,
asks for another way and then asks her to stop — because everything the six
questions prove is about a tool ANSWERING, and none of it can hear whether the
car calls a turn out loud while it is moving.

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


# THE OTHER DRIVE THIS FILE CAN DO: a route, called out loud, changed and
# stopped by voice. Everything the six turns above prove is about a tool
# ANSWERING; none of it can hear whether the car actually says "left onto
# Lincoln" while it is moving, in her voice, or whether asking her to stop
# stops it.
#
# It is one script rather than six turns because it is one continuous event:
# the simulator runs THROUGH the questions, which is the only way to find out
# whether a reroute asked for mid-drive lands on a car that is already moving.
NAV_SCRIPT = [
    # A destination the provider resolves cleanly and unambiguously. "The
    # Santa Monica Pier" was the first choice and comes back as a wealth
    # management office of that name — a real provider quirk, and not
    # something a recording of navigation should be spending its first line
    # on.
    {"say": "Take me to Griffith Observatory.",
     "want": "start_navigation, and one line confirming it",
     "budget_s": 20.0},
    # DRIVE. Long enough for the first turns to come round at 25 mph, which is
    # what the numbers below are: sim seconds, not wall-clock guesses.
    {"drive_s": 75.0,
     "want": "the early call, the instruction and the imminent backup — out "
             "loud, in her voice, on the same id the conversation uses"},
    {"say": "Avoid the freeway.",
     "want": "reroute with avoid=highways, one line, and the car still moving",
     "budget_s": 25.0},
    {"drive_s": 30.0,
     "want": "the new generation calling its own turns, and nothing left over "
             "from the plan it replaced"},
    {"say": "Stop navigation.",
     "want": "the route off, one line, and silence after it",
     "budget_s": 20.0},
    {"drive_s": 20.0,
     "want": "nothing at all — the drive is over and the queue was emptied"},
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
        # WHETHER THE PANEL IS STILL THERE. node is where every decision in
        # this harness is made, so a node that has exited produces exactly the
        # report a mute RIO produces -- silent turn after silent turn -- and
        # the drive has to say which of the two it was looking at.
        self.died_at = None

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
                if self.died_at is None:
                    self.died_at = time.perf_counter()
                    print("\n  THE PANEL PROCESS EXITED -- its traceback is "
                          "above, and nothing below this line is evidence "
                          "about RIO.")
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
                m["t"] = time.perf_counter()
                self.notes.append(m)

    async def pump_session(self):
        async for raw in self.oai:
            try:
                ev = json.loads(raw)
            except Exception:
                continue
            self.notes.append({"k": "wire_in", "type": ev.get("type"),
                               "t": time.perf_counter(), "ev": ev})
            self.tell({"k": "ev", "ev": ev})

    async def pump_relay(self):
        async for raw in self.relay:
            try:
                self.tell({"k": "wire", "m": json.loads(raw)})
            except Exception:
                pass


async def run(base, out_path, video, dump=None, script="tools"):
    import httpx
    import websockets

    from tools.visual_latency import HttpDrive

    http = httpx.AsyncClient(timeout=90.0)
    driver_track = []
    turn_ends = []
    legs = []

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

            turns = NAV_SCRIPT if script == "nav" else SCRIPT
            driving = False
            for turn in turns:
                # A LEG OF THE DRIVE RATHER THAN A QUESTION. Nobody says
                # anything; the car is moving and the only thing being
                # measured is what it says without being asked.
                if "drive_s" in turn:
                    # STARTED HERE RATHER THAN AT THE TOP, because a
                    # simulator with no route to run along refuses politely
                    # and is never asked again. The route exists by now: the
                    # turn before this one asked for it out loud.
                    if not driving:
                        panel.tell({"k": "sim", "on": True,
                                    "tickMs": 1000, "mph": 25})
                        driving = True
                    mark = len(panel.notes)
                    t_leg = time.perf_counter()
                    await asyncio.sleep(turn["drive_s"])
                    legs.append(report_drive(turn, panel.notes[mark:], t_leg))
                    continue
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
                # WHEN THE DRIVER STOPPED TALKING. The audio is paced in real
                # time, so the last sample of speech goes on the wire exactly
                # this far into the send -- before the room tone that follows
                # it. Every turn-end latency below is measured from here, the
                # same zero tools/turn_end_bench.py uses.
                t_speech_end = start + (len(q_pcm) / 2) / RATE
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
                turn_ends.append(
                    report(turn, panel.notes[mark:], t_ask, t_speech_end))

            if script == "nav":
                panel.tell({"k": "nav_state"})
                await asyncio.sleep(0.5)
                st = [n for n in panel.notes if n.get("note") == "nav_state"]
                if st:
                    s0 = st[-1]
                    print(f"\n  the panel, after all of it: "
                          f"routing={s0.get('routing')} "
                          f"destination={s0.get('destination')!r} "
                          f"generation={s0.get('generation_id')}")
                    print(f"      tools the session is carrying: "
                          f"{s0.get('tools')}")
                    print(f"      how the deterministic lines were spoken: "
                          f"{s0.get('speak_stats')}")
                panel.tell({"k": "sim", "on": False})

            # THE GATE ITSELF, asked directly, because the model choosing to
            # answer from memory and the tool refusing to run are
            # indistinguishable from the passenger seat -- and it was the
            # second one.
            gate = None if script == "nav" else (await http.post(
                f"{base}/realtime/tool", params={"session_id": sid},
                json={"name": "deep_dive",
                      "arguments": {"question": "why do carmakers use "
                                                "electric power steering"}})
                    ).json()
            if gate is not None:
                print(f"\n  deep_dive, seconds after two camera questions: "
                      f"ok={gate.get('ok')} "
                      + ('REFUSED: ' + str(gate.get('reason'))
                         if not gate.get('ok') else ''))
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
    return verdict(turn_ends, panel.died_at is None, legs)


def verdict(turns, panel_alive=True, legs=None):
    """The whole drive in the two terms it is accepted on.

    SPOKE, because a question that produces no sound is the failure this file
    was written for, and TRUNCATED, because an answer that stops at a length
    rather than at an end is the other one. The turn-end numbers are reported
    rather than asserted: they are a handful of samples off one drive and
    tools/turn_end_bench.py is where that distribution is measured.
    """
    if not turns:
        print("\n  no turns ran")
        return 1
    silent = [t for t in turns if not t["spoke"]]
    cut = [t for t in turns if t["truncated"] or t["cutoffs"]]
    split = [t for t in turns if t["commits"] > 1]
    ends = sorted(t["committed_ms"] for t in turns
                  if t["committed_ms"] is not None)
    print(f"\n  {len(turns) - len(silent)}/{len(turns)} turns spoke, "
          f"{len(turns) - len(cut)}/{len(turns)} finished the sentence")
    for t in turns:
        print(f"      {'spoke' if t['spoke'] else 'SILENT':>6}  "
              + (f"{t['committed_ms']:>6.0f} ms" if t["committed_ms"]
                 is not None else "     -- ms")
              + f"  {t['say']!r}"
              + ("  TRUNCATED:" + ",".join(x for x in t["truncated"] if x)
                 if t["truncated"] else "")
              + (f"  CUT:{','.join(t['cutoffs'])}" if t["cutoffs"] else "")
              + (f"  {t['commits']} COMMITS" if t["commits"] > 1 else ""))
    if ends:
        p50 = ends[len(ends) // 2]
        print(f"  turn end, last sample of speech to the turn closing: "
              f"min {ends[0]:.0f} ms, median {p50:.0f} ms, "
              f"max {ends[-1]:.0f} ms  ({len(ends)} turns)")
    # WHAT THE CAR SAID WITHOUT BEING ASKED. A navigating drive whose legs are
    # silent is the failure this script exists to catch, and it is not visible
    # in the turns above: every question can be answered perfectly by a car
    # that never calls a turn.
    bad_legs = []
    if legs:
        called = sum(l["calls"] for l in legs)
        spoke = sum(l["heard"] for l in legs)
        print(f"  turns called while driving: {called}, of which "
              f"{spoke} reached the speaker")
        for l in legs:
            for t in l["texts"]:
                print(f"      {t!r}")
        if not called:
            bad_legs.append("no turns were called on any leg")
        elif not spoke:
            bad_legs.append("turns were called and none of them was heard")
        for b in bad_legs:
            print(f"      <-- {b}")

    bad = silent or cut or split or bad_legs
    if not panel_alive:
        print("  VOID -- the panel process died during the drive, so the "
              "silences above are this harness's and not hers")
        return 1
    print("  " + ("ACCEPTED" if not bad else "NOT ACCEPTED"))
    return 1 if bad else 0


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


def report_drive(turn, notes, t0):
    """One leg with nobody talking: what the car said, and in whose voice."""
    # TWO DIFFERENT QUESTIONS, and the drive that could not tell them apart was
    # counting each call twice. The planner emits one event when it DECIDES a
    # call is worth making and another when the arbiter has SAID it, and the
    # gap between those two is where a turn goes missing.
    planned = [n["ev"] for n in notes if n.get("note") == "nav"
               and n["ev"].get("call_type")
               and n["ev"].get("type") != "NAV_SPEECH_SPOKEN"]
    spoken = [n["ev"] for n in notes if n.get("note") == "nav"
              and n["ev"].get("type") == "NAV_SPEECH_SPOKEN"]
    # ...and one more: what came out of a speaker. A line the arbiter believes
    # it spoke and no listener heard is the failure a WAV exists to show.
    heard = [n for n in notes if n.get("note") == "voice"
             and n["ev"].get("type") == "VOICE_UTTERANCE_DONE"]
    audio = [n for n in notes if n.get("note") == "nav_audio"]
    other = [n["ev"]["type"] for n in notes if n.get("note") == "nav"
             and not n["ev"].get("call_type")]
    calls = spoken or planned
    print(f"\n  [driving {turn['drive_s']:.0f}s]   ({turn['want']})")
    for c in calls:
        print(f"      {c.get('call_type'):>8}  {c.get('text')!r}"
              + (f"  ({c.get('anchor_label')})" if c.get("anchor_label") else ""))
    if len(planned) != len(spoken):
        print(f"      planned {len(planned)}, spoken {len(spoken)}")
    if audio:
        print(f"      synthesised and played: {len(audio)} lines, "
              f"{sum(a['seconds'] for a in audio):.1f}s of audio")
    if other:
        print(f"      also: {', '.join(sorted(set(other)))}")
    # THE POINT OF THE LEG. A call the planner decided on and nobody heard is
    # the failure this drive exists to catch, and it is invisible in the event
    # stream alone: the planner emits SPEECH_SPOKEN when it hands the line
    # over, not when a speaker finishes it.
    # A navigation line does not go through the conversation's sink under the
    # ElevenLabs backend — it is synthesised by /nav/voice and played through
    # the element — so BOTH are counted and either one means it was audible.
    audible = len(heard) + len(audio)
    print(f"      reached a speaker: {audible}"
          + ("   <-- SILENT" if calls and not audible else ""))
    if not calls:
        print("      no turns called")
    return {"leg_s": turn["drive_s"], "calls": len(calls),
            "heard": audible,
            "texts": [c.get("text") for c in calls]}


def report(turn, notes, t_ask, t_speech_end):
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

    # HOW LONG SHE KEPT LISTENING AFTER THE QUESTION WAS OVER. `stopped` is
    # the detector's own decision and `committed` is the turn actually closing
    # -- the number tools/turn_end_bench.py reports and the one the driver
    # feels. Both are measured from the last sample of speech, not from the
    # end of the send, because the send keeps going the way a cabin does.
    def first_at(kind):
        for n in notes:
            if n.get("k") == "wire_in" and n.get("type") == kind:
                return (n["t"] - t_speech_end) * 1000
        return None
    stopped_ms = first_at("input_audio_buffer.speech_stopped")
    committed_ms = first_at("input_audio_buffer.committed")
    # ONE TURN, NOT THREE. A detector that ends the turn mid-sentence commits
    # more than once for one question, which is the failure semantic_vad is
    # here to not have.
    commits = len([n for n in notes if n.get("k") == "wire_in"
                   and n.get("type") == "input_audio_buffer.committed"])
    # AND WHETHER THE ANSWER WAS ALLOWED TO FINISH. `incomplete` with a reason
    # of max_output_tokens is an answer stopping at a length rather than at an
    # end; the controller files it as a token_cap cutoff.
    done = [n["ev"] for n in notes if n.get("k") == "wire_in"
            and n.get("type") == "response.done"]
    incomplete = [(r.get("response") or {}) for r in done
                  if (r.get("response") or {}).get("status") == "incomplete"]
    cutoffs = [n["ev"] for n in notes if n.get("note") == "live"
               and n["ev"].get("type") == "LIVE_CUTOFF"]

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
    print("      turn ended: "
          + (f"{stopped_ms:.0f} ms to the detector" if stopped_ms is not None
             else "detector never fired")
          + (f", {committed_ms:.0f} ms to the commit" if committed_ms
             is not None else ", never committed")
          + (f"   <-- {commits} COMMITS, one question cut into "
             f"{commits} turns" if commits > 1 else ""))
    for inc in incomplete:
        det = inc.get("status_details") or {}
        print(f"      TRUNCATED: response {inc.get('id')} stopped at "
              f"{det.get('reason')}")
    for c in cutoffs:
        print(f"      CUT OFF: cause={c.get('cause')} "
              f"after {c.get('said_chars')} chars"
              + (f" ({c.get('reason')})" if c.get("reason") else ""))
    return {"say": turn["say"], "stopped_ms": stopped_ms,
            "committed_ms": committed_ms, "commits": commits,
            "spoke": bool(heard), "said": spoken,
            "truncated": [(i.get("status_details") or {}).get("reason")
                          for i in incomplete],
            "cutoffs": [c.get("cause") for c in cutoffs]}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base", default="http://127.0.0.1:8888")
    ap.add_argument("--video", default="/workspace/ufldv2/example.mp4")
    ap.add_argument("--out", default="/tmp/rio_tools.wav")
    ap.add_argument("--dump", default=None,
                    help="write every event of the drive to this file")
    ap.add_argument("--script", default="tools", choices=("tools", "nav"),
                    help="tools: the six questions that were failing. "
                         "nav: one route, called out loud, rerouted and "
                         "stopped by voice.")
    args = ap.parse_args()
    return asyncio.run(run(args.base, Path(args.out), args.video,
                           args.dump, args.script))


if __name__ == "__main__":
    raise SystemExit(main())
