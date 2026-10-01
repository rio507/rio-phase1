"""xai_single_response_probe.py — does xAI still run one response at a time?

    python -m tools.xai_single_response_probe            # xAI, asserts; exit 1 on change
    python -m tools.xai_single_response_probe --openai   # ...and the OpenAI contrast

WHY THIS IS A TEST. static/rio_xai_session.js serialises every response.create
because of what this measured on 2026-10-01 -- and none of it is in xAI's
documentation, which says nothing about concurrent responses, response.cancel,
out-of-band responses or response.metadata. A behaviour we only observed is one
the vendor can change without telling us. If they do, this fails, and the gate's
constants (settle 2 s; metadata as the orphan key) need re-deciding.

WHAT IT ASSERTS, each over several fresh sessions (one session per trial):

  echo        response.created carries the create's response.metadata
              -- the orphan claim in rio_realtime.js binds to it
  mid_drop    an item-less create sent while one is generating is DROPPED:
              the first completes, the second is never created
  mid_cancel  a conversation item + create while one is generating CANCELS
              the first and runs the second
  post_drop   an item-less create 500 ms after a COMPLETED response is dropped
  post_ok     ...and at the gate's settle (2 s) it is created
  cancel_ok   an item-less create right after a CANCELLED response is created

A "dropped" create is one with no response.created within 6 s and no error.
"""
import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv                                  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                                   # noqa: E402
import xai_voice                                                # noqa: E402
from tools.companion_bench import session_config                # noqa: E402

SETTLE_MS = 2000           # static/rio_xai_session.js settleMs default
QUIET_S = 6.0


def dictate(tag, words):
    return {"type": "response.create", "response": {
        "conversation": "none", "output_modalities": ["audio"],
        "metadata": {"rio": tag},
        "instructions": "Read exactly, word for word: " + words}}


def answer(tag):
    return {"type": "response.create", "response": {"metadata": {"rio": tag}}}


def item(text):
    return {"type": "conversation.item.create", "item": {
        "type": "message", "role": "user",
        "content": [{"type": "input_text", "text": text}]}}


LONG = ("Head northwest on Sunset Boulevard, then turn right onto Palisades "
        "Drive and continue for two miles.")


async def connect(vendor):
    import websockets
    if vendor == "xai":
        tok = await asyncio.to_thread(xai_voice._ephemeral, 300)
        ws = await websockets.connect(
            f"{xai_voice.WS_URL}?model={config.XAI_VOICE_MODEL}",
            subprotocols=[f"xai-client-secret.{tok}"], max_size=None)
        await ws.send(json.dumps({"type": "session.update",
                                  "session": session_config("You are terse.", [])}))
    else:
        ws = await websockets.connect(
            f"wss://api.openai.com/v1/realtime?model={config.OPENAI_REALTIME_MODEL}",
            additional_headers={"Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]},
            max_size=None)
        await ws.send(json.dumps({"type": "session.update", "session": {
            "type": "realtime", "output_modalities": ["audio"],
            "instructions": "You are terse.",
            "audio": {"output": {"voice": "marin",
                                 "format": {"type": "audio/pcm", "rate": 24000}}}}}))
    while True:
        ev = json.loads(await asyncio.wait_for(ws.recv(), 30))
        if ev.get("type") == "session.updated":
            return ws
        if ev.get("type") == "error":
            raise RuntimeError(str(ev)[:200])


class Wire:
    """One session: send, and read until a condition holds or it goes quiet."""

    def __init__(self, ws):
        self.ws, self.ids, self.meta = ws, {}, {}
        self.created, self.done, self.audio, self.errors = [], {}, set(), []

    def _note(self, ev):
        t, r = ev.get("type"), ev.get("response") or {}
        if t == "response.created":
            tag = (r.get("metadata") or {}).get("rio")
            self.ids[r.get("id")] = tag
            self.meta[tag] = r.get("metadata")
            self.created.append(tag)
        elif t == "response.done":
            self.done[self.ids.get(r.get("id"))] = r.get("status")
        elif t in ("response.output_audio.delta", "response.audio.delta"):
            self.audio.add(self.ids.get(ev.get("response_id")))
        elif t == "error":
            self.errors.append((ev.get("error") or {}).get("message"))

    async def send(self, obj):
        await self.ws.send(json.dumps(obj))

    async def until(self, pred, quiet=QUIET_S):
        try:
            while not pred():
                self._note(json.loads(await asyncio.wait_for(self.ws.recv(), quiet)))
        except asyncio.TimeoutError:
            pass


async def trial(vendor, shape):
    ws = await connect(vendor)
    w = Wire(ws)
    try:
        if shape in ("mid_drop", "mid_cancel"):
            await w.send(dictate("A", LONG))
            await w.until(lambda: "A" in w.audio)
            if shape == "mid_cancel":
                await w.send(item("Say hi."))
            await w.send(answer("B") if shape == "mid_cancel" else dictate("B", "Keep left."))
            await w.until(lambda: "A" in w.done and "B" in w.done)
            return {"A": w.done.get("A"), "B_created": "B" in w.created}
        if shape in ("post_drop", "post_ok"):
            await w.send(dictate("A", LONG))
            await w.until(lambda: "A" in w.done)
            await asyncio.sleep((500 if shape == "post_drop" else SETTLE_MS) / 1000)
            await w.send(dictate("B", "Keep left at the fork."))
            await w.until(lambda: "B" in w.created)
            return {"A": w.done.get("A"), "B_created": "B" in w.created}
        if shape == "cancel_ok":
            await w.send(item("Tell me about the Santa Monica pier in two sentences."))
            await w.send(answer("A"))
            await w.until(lambda: "A" in w.audio)
            rid = next(k for k, v in w.ids.items() if v == "A")
            await w.send({"type": "response.cancel", "response_id": rid})
            await w.until(lambda: "A" in w.done)
            await w.send(dictate("B", "Turn left."))
            await w.until(lambda: "B" in w.created)
            return {"A": w.done.get("A"), "B_created": "B" in w.created}
        if shape == "echo":
            await w.send(dictate("A", "Turn right."))
            await w.until(lambda: "A" in w.created or None in w.created)
            return {"echoed": w.meta.get("A") == {"rio": "A"}}
    finally:
        await ws.close()


# shape -> (trials, check on the list of results, what it means)
EXPECT = {
    "echo":       (2, lambda rs: all(r["echoed"] for r in rs),
                   "response.created echoes response.metadata"),
    "mid_drop":   (3, lambda rs: all(r["A"] == "completed" and not r["B_created"] for r in rs),
                   "item-less create while generating is dropped, first completes"),
    "mid_cancel": (3, lambda rs: all(r["A"] == "cancelled" and r["B_created"] for r in rs),
                   "item + create while generating cancels the first"),
    "post_drop":  (3, lambda rs: sum(not r["B_created"] for r in rs) >= 2,
                   "item-less create 500 ms after a completed response is dropped"),
    "post_ok":    (4, lambda rs: all(r["B_created"] for r in rs),
                   f"...and created at the gate's settle ({SETTLE_MS} ms)"),
    "cancel_ok":  (3, lambda rs: all(r["A"] == "cancelled" and r["B_created"] for r in rs),
                   "item-less create right after a cancel is created"),
}


async def run(vendor):
    failed = []
    for shape, (n, check, what) in EXPECT.items():
        rs = []
        for _ in range(n):
            try:
                rs.append(await asyncio.wait_for(trial(vendor, shape), 60))
            except Exception as e:
                rs.append({"error": f"{type(e).__name__}: {e}"[:120]})
        good = not any("error" in r for r in rs) and check(rs)
        print(f"  {'ok  ' if good else 'DIFF'}  {vendor:6} {shape:<10} {what}\n"
              f"        {rs}", flush=True)
        if not good:
            failed.append(shape)
    return failed


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--openai", action="store_true",
                    help="also run the shapes against OpenAI, for contrast "
                         "(expected to DIFFER: it runs concurrent responses)")
    a = ap.parse_args()
    print("== xAI: one response at a time (asserted)")
    failed = asyncio.run(run("xai"))
    if a.openai:
        print("\n== OpenAI: for contrast, not asserted")
        asyncio.run(run("openai"))
    print("\n" + ("xAI behaves as rio_xai_session.js assumes"
                  if not failed else
                  f"xAI CHANGED: {failed} — revisit the gate in "
                  "static/rio_xai_session.js and the orphan key in rio_realtime.js"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
