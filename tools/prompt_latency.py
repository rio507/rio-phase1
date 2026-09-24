"""prompt_latency.py — what a bigger prompt costs a drive, in milliseconds.

    python -m tools.prompt_latency
    python -m tools.prompt_latency --pad 0 3000 8000 18000 --n 12

WHY THIS AND NOT A TOKEN BUDGET. The per-response growth alarm in
tools/realtime_selftest.py was derived from the OpenAI realtime account's
200,000 tokens a minute. The drive session is xAI now, and xAI's speech-to-speech
endpoint has no token-per-minute limit at all: grok-voice-think-fast-2.0 is
limited by CONCURRENT SESSIONS only (docs.x.ai/developers/rate-limits), it is
billed per minute of audio rather than per token, and the session reports no
usage (response.done carries `usage: {}`). So the one thing a longer prompt can
measurably cost on this backend is the driver's wait. This measures it.

THE SESSION IS HER REAL ONE -- realtime.instructions() and the real tool
schemas -- with inert padding appended to the instructions to reach each size.
Padding rather than a real section, because the question is what SIZE costs; a
real section would also change what she says, and that is companion_bench's job.

Two numbers per size, each over N fresh sessions:
  first audio   response.create -> first audio delta, on a turn that speaks
                ("How's your day going?") -- what a driver waits for an answer
  tool call     response.create -> function_call_arguments.done, on a turn that
                searches ("Find me coffee.") -- what a driver waits before RIO
                even starts fetching
Each is measured on the FIRST turn of a session (the whole prompt is new) and on
the SECOND turn (whatever the vendor caches of it, it has cached by then).
"""
import argparse
import asyncio
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
from companion_bench import session_config                      # noqa: E402

SPEAK = "How's your day going?"
TOOL = "Find me coffee."
PAD_LINE = ("Reference note {i}: this paragraph is padding for a latency "
            "measurement. It carries no instructions and nothing in it applies "
            "to the drive.\n")


def tokens(s):
    return len(s) // 4          # the estimate realtime_selftest uses without tiktoken


def padded(base, extra_tokens):
    out, i = [base, "\n\n"], 0
    while tokens("".join(out)) - tokens(base) < extra_tokens:
        out.append(PAD_LINE.format(i=i))
        i += 1
    return "".join(out) if extra_tokens else base


async def _timed(ws, text):
    await ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": text}]}}))
    t0 = time.perf_counter()
    await ws.send(json.dumps({"type": "response.create"}))
    audio = call = None
    while True:
        ev = json.loads(await asyncio.wait_for(ws.recv(), 60))
        t = ev.get("type", "")
        if t == "response.output_audio.delta" and audio is None:
            audio = (time.perf_counter() - t0) * 1000
        elif t == "response.function_call_arguments.done" and call is None:
            call = (time.perf_counter() - t0) * 1000
        elif t == "error":
            raise RuntimeError(str(ev.get("error"))[:200])
        elif t == "response.done":
            return audio, call


async def one(instructions, tools, text):
    import websockets
    token = await asyncio.to_thread(xai_voice._ephemeral, 300)
    async with websockets.connect(
            f"{xai_voice.WS_URL}?model={config.XAI_VOICE_MODEL}",
            subprotocols=[f"xai-client-secret.{token}"],
            open_timeout=30, max_size=None) as ws:
        await ws.send(json.dumps({"type": "session.update",
                                  "session": session_config(instructions, tools)}))
        while True:
            ev = json.loads(await asyncio.wait_for(ws.recv(), 30))
            if ev.get("type") == "session.updated":
                break
            if ev.get("type") == "error":
                raise RuntimeError("session refused: " + str(ev.get("error"))[:200])
        first = await _timed(ws, text)
        # A result for the tool call, so the second turn is well formed.
        if text == TOOL:
            await ws.send(json.dumps({
                "type": "conversation.item.create",
                "item": {"type": "function_call_output", "call_id": "bench",
                         "output": json.dumps({"ok": True, "note": "bench"})}}))
        second = await _timed(ws, text)
        return first, second


def p50(xs):
    xs = [x for x in xs if x is not None]
    return statistics.median(xs) if xs else float("nan")


async def run(pads, n, conc):
    base = realtime.instructions()
    tools = [dict(t) for t in realtime.session_config()["tools"]]
    tool_tokens = tokens(json.dumps(tools))
    sem = asyncio.Semaphore(conc)
    rows = []
    for pad in pads:
        instr = padded(base, pad)

        async def go(text):
            async with sem:
                for attempt in range(3):
                    try:
                        return await one(instr, tools, text)
                    except Exception:
                        await asyncio.sleep(2 + 3 * attempt)
                return (None, None), (None, None)

        speak = await asyncio.gather(*(go(SPEAK) for _ in range(n)))
        tool = await asyncio.gather(*(go(TOOL) for _ in range(n)))
        row = {"pad": pad, "per_response": tokens(instr) + tool_tokens,
               "audio_first": p50([s[0][0] for s in speak]),
               "audio_second": p50([s[1][0] for s in speak]),
               "call_first": p50([t[0][1] for t in tool]),
               "call_second": p50([t[1][1] for t in tool]),
               "raw": {"speak": speak, "tool": tool}}
        rows.append(row)
        print(f"  {row['per_response']:>7,} tokens   first audio p50 "
              f"{row['audio_first']:>6.0f} / {row['audio_second']:>6.0f} ms   "
              f"tool call p50 {row['call_first']:>6.0f} / {row['call_second']:>6.0f} ms"
              f"   (first turn / second turn, n={n})", flush=True)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pad", type=int, nargs="+", default=[0, 3000, 8000, 18000])
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--out")
    a = ap.parse_args()
    if not os.getenv("XAI_API_KEY"):
        print("XAI_API_KEY is not set")
        return 2
    print(f"{config.XAI_VOICE_MODEL}, effort {config.XAI_VOICE_EFFORT}")
    rows = asyncio.run(run(a.pad, a.n, a.conc))
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
