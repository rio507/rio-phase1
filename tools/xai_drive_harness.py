"""xai_drive_harness.py — a whole turn through the real session, sequenced properly.

    python -m tools.xai_drive_harness
    python -m tools.xai_drive_harness --say "how are my tires" --turns 2

WHY A HARNESS BEFORE A BROWSER CONTROLLER.

Stage 3's failure mode is not a crash. It is RIO cutting herself off mid-sentence,
and it took two drives to find last time. A thousand lines of browser controller
that nobody can run — no microphone, no speaker, no car — would be exactly the
"bad job that does not show up as a failing test" this stage was warned about.

So the protocol and the SEQUENCING are proven here first, against the live
endpoint, in something that runs on this machine:

  audio in      real speech, rendered by the same voice, with a tail of silence
                so server_vad can hear the utterance end
  audio out     collected and timed, with audio-end computed from the BYTES the
                way static/rio_playout.js computes it -- never from response.done
  tools         dispatched through realtime.run_tool, the same function the panel
                calls, so a tool that works here works in the car
  the gate      the next request waits for the playout queue to drain, which is
                what their own docs warn about and what rio_playout.idle() answers
  transcripts   deduplicated BY ITEM ID, because .completed arrives three times

What it cannot prove: microphone capture, speaker playback, and the arbiter
fighting for the mouth. Those are the browser's, and they are what remains.

EVERYTHING THE SESSION WILL DO QUIETLY IF MISCONFIGURED is honoured from
xai_voice.session_policy() rather than restated here, so there is one description
of a correct session and this file cannot drift from it.
"""
import argparse
import asyncio
import base64
import json
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv                                  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                                   # noqa: E402
import realtime                                                 # noqa: E402
import xai_voice                                                # noqa: E402

RATE = xai_voice.RATE
FRAME = int(RATE * 0.04) * 2          # 40 ms of PCM16, the shape a browser sends


class Playout:
    """The queue's arithmetic, in Python, matching static/rio_playout.js.

    Deliberately a reimplementation rather than a port: what is being checked is
    that the CONTRACT holds against real audio from the real vendor, and a shared
    implementation would only prove it agrees with itself. The inequality is the
    same one the node suite breaks on purpose --

        audio-end must not be announced before the last sample it was given is
        scheduled to finish

    -- and the grace is added after that instant, never subtracted.
    """

    def __init__(self, grace_s=0.06):
        self.grace = grace_s
        self.ends_at = 0.0
        self.generation_done = False
        self.bytes = 0
        self.started_at = None
        self.ended_at = None

    def push(self, n):
        now = time.time()
        start = self.ends_at if self.ends_at > now else now
        if self.started_at is None:
            self.started_at = start
        self.ends_at = start + (n / 2 / RATE)
        self.bytes += n

    def done_generating(self):
        self.generation_done = True

    def idle(self):
        if self.bytes and not self.generation_done:
            return False
        return time.time() >= self.ends_at + self.grace

    def until_idle(self):
        if self.bytes and not self.generation_done:
            return float("inf")
        return max(0.0, (self.ends_at + self.grace) - time.time())

    async def wait_idle(self, cap_s=30.0):
        """Block until the sound is over. THE GATE, and the reason it is here:
        their docs warn that a tool result followed immediately by response.create
        makes the server start the next turn over audio still playing."""
        t0 = time.time()
        while not self.idle() and time.time() - t0 < cap_s:
            left = self.until_idle()
            await asyncio.sleep(0.02 if left == float("inf") else min(left, 0.25))
        self.ended_at = time.time()
        return self.ended_at


def prerender(utterances):
    """The driver's speech, synthesised BEFORE the session opens.

    render_speech calls asyncio.run(), which raises inside a running loop -- so
    rendering mid-session failed every turn, and the harness then reported "no
    silent turns" because a turn that never ran skipped the check that would have
    caught it. Both halves of that are fixed: the audio is made up front, and a
    turn that cannot run is a FAILURE rather than a `continue`.
    """
    out = []
    for text in utterances:
        got = xai_voice.render_speech(text)
        if not got.get("ok"):
            out.append((text, None, got.get("note")))
        else:
            out.append((text, _wav_body(got["wav"]), None))
    return out


async def drive(prepared, turns, verbose):
    import websockets

    mint = xai_voice.mint_client_secret()
    print(f"  {mint['model']} / {mint['voice']}, {len(mint['tools'])} tools, "
          f"effort={(mint['session'].get('reasoning') or {}).get('effort')}, "
          f"silence tail {mint['silence_tail_ms']} ms")

    ttfa, tool_calls, transcripts, dupes = [], [], [], 0
    failed = []
    seen_ids = set()
    silent_turns = []

    async with websockets.connect(
            mint["ws_url"], subprotocols=[mint["ws_subprotocol"]],
            open_timeout=30, max_size=None) as ws:

        async def send(obj):
            await ws.send(json.dumps(obj))

        await send({"type": "session.update", "session": mint["session"]})
        while True:
            ev = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if ev.get("type") == "session.updated":
                resolved = (ev.get("session") or {}).get("model")
                print(f"  session up; the wire says it is {resolved!r}")
                break
            if ev.get("type") == "error":
                raise RuntimeError("session refused: "
                                   + json.dumps(ev.get("error"))[:300])

        for turn in range(turns):
            for text, pcm, why in prepared:
                print(f"\n  --- driver: {text!r}")
                if pcm is None:
                    # NOT A `continue`. A turn that could not be driven is a turn
                    # nobody proved anything about, and the first version of this
                    # file skipped it and then printed "no silent turns".
                    failed.append(f"{text} (could not synthesise: {why})")
                    print(f"      FAILED to synthesise the driver: {why}")
                    continue

                for i in range(0, len(pcm), FRAME):
                    await send({"type": "input_audio_buffer.append",
                                "audio": base64.b64encode(
                                    pcm[i:i + FRAME]).decode()})
                    await asyncio.sleep(0.004)
                # THE TAIL. Without it the turn never closes -- measured.
                tail = int(mint["silence_tail_ms"] / 40)
                for _ in range(tail):
                    await send({"type": "input_audio_buffer.append",
                                "audio": base64.b64encode(b"\x00" * FRAME).decode()})
                # AND NO EXPLICIT COMMIT. server_vad owns the end of the utterance;
                # committing as well suppresses the transcript entirely.

                play = Playout()
                t0 = time.time()
                first = None
                called = None
                spoke = False
                # EVERY CALL IN THE RESPONSE, not the last one. This held a single
                # `pending_call`, so a turn in which the model asked for two tools
                # left the second one with no output -- and a conversation holding a
                # function call with no output produces NO ANSWER AT ALL. That is
                # what "DEGRADED: the tool result produced no answer" was, twice, on
                # the drive of 2026-09-20: not the model declining to speak, this
                # harness not answering what it was asked. The browser's controller
                # answers every call (rio_realtime.js dispatches per
                # response.function_call_arguments.done), so a harness that answered
                # one was testing a sequence the car does not run.
                pending_calls = []

                while True:
                    try:
                        ev = json.loads(await asyncio.wait_for(ws.recv(), 45))
                    except asyncio.TimeoutError:
                        print("      <<< no further events: the session went quiet")
                        break
                    t = ev.get("type", "")

                    if t == "response.output_audio.delta":
                        raw = base64.b64decode(ev.get("delta") or b"")
                        play.push(len(raw))
                        if first is None:
                            first = (time.time() - t0) * 1000
                            ttfa.append(first)
                        spoke = True

                    elif t == "conversation.item.input_audio_transcription.completed":
                        iid = ev.get("item_id")
                        # DEDUPLICATED BY ID, DELIBERATELY. It arrives three times
                        # with one id; the binding suppressing the repeats by
                        # accident is not the same as a client that knows.
                        if iid in seen_ids:
                            dupes += 1
                        else:
                            seen_ids.add(iid)
                            transcripts.append(ev.get("transcript") or "")
                            print(f"      heard: {ev.get('transcript')!r}")

                    elif t == "response.function_call_arguments.done":
                        called = ev.get("name")
                        pending_calls.append((ev.get("name"), ev.get("call_id"),
                                              ev.get("arguments")))
                        tool_calls.append(called)
                        print(f"      tool: {called}")

                    elif t == "error":
                        print(f"      error: {json.dumps(ev.get('error'))[:200]}")

                    elif t == "response.done":
                        # GENERATION over. The SOUND may not be.
                        play.done_generating()
                        break

                # THE TAIL IS HELD HERE, not at response.done.
                left = play.until_idle()
                if left:
                    print(f"      generation done; {left:.2f} s of audio still to "
                          f"play — the mouth is NOT handed back yet")
                await play.wait_idle()

                if not spoke and not called:
                    # SILENCE WHERE AUDIO WAS EXPECTED. Three wrong conclusions
                    # came from a session that looked healthy and never spoke.
                    silent_turns.append(text)
                    print("      DEGRADED: the turn produced neither audio nor a "
                          "tool call — a session that connects, runs VAD and never "
                          "speaks must not look healthy")

                if pending_calls:
                    if len(pending_calls) > 1:
                        print(f"      (the model asked for {len(pending_calls)} "
                              f"tools in one turn: "
                              f"{[c[0] for c in pending_calls]} — every one of "
                              f"them is answered, because one left open blocks "
                              f"the whole turn)")
                    for name, call_id, args in pending_calls:
                        try:
                            out = realtime.run_tool(name, args,
                                                    session_key="harness")
                        except Exception as e:
                            out = {"ok": False, "note": f"{type(e).__name__}"}
                        print(f"      {name} -> ok={out.get('ok')} "
                              f"{str(out.get('note') or '')[:60]}")
                        await send({"type": "conversation.item.create", "item": {
                            "type": "function_call_output", "call_id": call_id,
                            "output": json.dumps(out)[:4000]}})
                    # ...AND ONLY NOW ASK FOR THE ANSWER. The gate.
                    await play.wait_idle()
                    await send({"type": "response.create"})
                    play2 = Playout()
                    t1 = time.time()
                    got_audio = False
                    while True:
                        try:
                            ev = json.loads(await asyncio.wait_for(ws.recv(), 45))
                        except asyncio.TimeoutError:
                            break
                        t = ev.get("type", "")
                        if verbose:
                            print(f"        <- {t}")
                        if t == "response.output_audio.delta":
                            play2.push(len(base64.b64decode(ev.get("delta") or b"")))
                            if not got_audio:
                                ttfa.append((time.time() - t1) * 1000)
                                got_audio = True
                        elif t == "response.output_audio_transcript.done":
                            print(f"      she said: {str(ev.get('transcript'))[:90]!r}")
                        elif t == "error":
                            # NEVER SWALLOWED. A turn that produced no answer while
                            # an error event went unprinted is how a wrong item
                            # shape reads as "the model chose not to speak" -- and
                            # the answer-after-a-tool-call is the one sequence in
                            # the drive where that mistake is invisible.
                            print(f"      WIRE ERROR: {str(ev)[:300]}")
                        elif t == "response.done":
                            play2.done_generating()
                            break
                    await play2.wait_idle()
                    if not got_audio:
                        silent_turns.append(text + " (after the tool result)")
                        print("      DEGRADED: the tool result produced no answer")

    return {"ttfa": ttfa, "tools": tool_calls, "transcripts": transcripts,
            "dupes": dupes, "silent": silent_turns, "failed": failed}


def _wav_body(wav_bytes: bytes) -> bytes:
    import io
    import wave

    with wave.open(io.BytesIO(wav_bytes), "rb") as w:
        return w.readframes(w.getnframes())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--say", action="append", default=None)
    ap.add_argument("--turns", type=int, default=1)
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    if not os.getenv("XAI_API_KEY"):
        print("XAI_API_KEY is not set")
        return 2
    said = a.say or ["What's the weather going to do later?",
                     "How are my tires doing?"]
    print(f"a drive of {len(said)} utterance(s) x {a.turns} turn(s)\n")
    prepared = prerender(said)
    bad = [t for t, pcm, _ in prepared if pcm is None]
    if bad:
        print(f"  could not synthesise: {bad}")
    try:
        r = asyncio.run(drive(prepared, a.turns, a.verbose))
    except Exception as e:
        print(f"\nFAILED {type(e).__name__}: {str(e)[:300]}")
        return 1

    print("\n" + "-" * 66)
    t = sorted(r["ttfa"])
    if t:
        print(f"  first audio  n={len(t)}  p50 {statistics.median(t):.0f}  "
              f"p95 {t[min(len(t)-1, int(len(t)*0.95))]:.0f}  max {t[-1]:.0f} ms")
    print(f"  tools called : {r['tools'] or 'none'}")
    print(f"  transcripts  : {r['transcripts']}")
    print(f"  duplicate .completed suppressed by id: {r['dupes']}")
    if r.get("failed"):
        print(f"  TURNS THAT NEVER RAN: {len(r['failed'])} — {r['failed']}")
    if r["silent"]:
        print(f"  DEGRADED TURNS: {len(r['silent'])} — {r['silent']}")
        return 1
    if r.get("failed"):
        return 1
    print("  no silent turns, no failed turns")
    return 0


if __name__ == "__main__":
    sys.exit(main())
