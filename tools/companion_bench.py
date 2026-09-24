"""companion_bench.py — does she answer a statement like a person, and a request at once?

    python -m tools.companion_bench --label before --out runs/companion_before.json
    python -m tools.companion_bench --label after  --out runs/companion_after.json
    python -m tools.companion_bench --compare runs/companion_before.json runs/companion_after.json

THE RULE UNDER TEST: a statement is not a request.

  request     "Find me coffee." "Take me to Century City." She acts on it
              straight away -- a tool call in the FIRST response, and no
              "do you want me to look?" in front of it.
  statement   "I'm tired." "This traffic is killing me." She answers the person
              first and OFFERS -- no tool call in the first response -- and when
              the driver says "yeah", she acts.
  chat        "Do you like movies?" No tool, and no offer required: there is
              nothing to help with.

THREE NUMBERS, per arm:

  statements  no tool on the first turn AND an offer in the reply; and, as a
              second line, whether a bare "Yeah." then produces the tool call
  requests    a tool call on the first turn with no confirmation question
  openings    across every statement reply, how many DISTINCT first-two-word
              openings. A prompt that carries an example gets copied, and a
              copy shows up here as the same opening every time.

ONE SESSION PER TRIAL, so nothing a previous utterance said can shape the next
one, and the openings count measures the prompt rather than the conversation.
The driver's words go in as TEXT, the same choice tools/xai_voice_bench.py made
and for its reason: this is about what she does with an utterance, not about
the transcriber. The session is otherwise her real one -- realtime.instructions()
and realtime.session_config()'s tools, read at run time, so an arm is whatever
the code says when it runs, and the JSON records a hash of the instructions so
two runs of the same prompt cannot be mistaken for a before and an after.

Tool calls are answered with a stub, never run: whether she CALLED is the
measurement, and a real find_places on "I'm hungry" would spend a Places query
to learn nothing.
"""
import argparse
import asyncio
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv                                  # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                                   # noqa: E402
import realtime                                                 # noqa: E402
import xai_voice                                                # noqa: E402

# THE SET. `src` says where each line came from, because most of the statements
# could NOT come from a drive log: the log records the driver's words only when
# the barge gate refused them (turn_phantom), when a turn was split
# (turn_fragment), in the old /talk path, and in search_local_news's `question`,
# which the tool asks the model to pass through verbatim. A statement that was
# answered normally left no text at all. Written lines are marked `written`;
# the three from the brief are the driver's own examples.
SET = [
    # --- requests: act on the first turn -----------------------------------
    ("request", "Can you give me directions to Century City?",
     "log 8bf74c64 turn_phantom"),
    ("request", "All right, in the meantime, can you just give me some "
                "directions to Starbucks?", "log turn_phantom"),
    ("request", "Can you start directions to a Starbucks nearby?",
     "log turn_phantom"),
    ("request", "What's the weather like?", "log turn_phantom"),
    ("request", "Okay, what are some headlines?", "log turn_phantom"),
    ("request", "Any news around here?", "log search_local_news.question"),
    ("request", "Why is traffic so bad here?", "log search_local_news.question"),
    ("request", "What do you see?", "log talk.transcript"),
    ("request", "What kind of car is that on the right?", "log talk.transcript"),
    ("request", "Is this a movie theater right here?", "log turn_phantom"),
    ("request", "What's the next direction?", "log turn_phantom"),
    ("request", "is there any basketball games playing soon?",
     "log search_local_news.question"),
    ("request", "Um, can you check on the clavicular news?",
     "log search_local_news.question"),
    ("request", "Find me coffee.", "brief"),
    ("request", "What's that building?", "brief"),
    # --- statements: answer the person, offer, act on yes -------------------
    ("statement", "I'm feeling kinda hungry.", "brief"),
    ("statement", "I'm tired.", "brief"),
    ("statement", "This traffic is killing me.", "brief"),
    ("statement", "I could really use a coffee right now.", "written"),
    ("statement", "Man, I'm running low on gas.", "written"),
    ("statement", "Ugh, I think I'm going to be late.", "written"),
    ("statement", "I haven't eaten all day.", "written"),
    ("statement", "I kind of want to see a movie tonight.", "written"),
    ("statement", "I need to stretch my legs.", "written"),
    ("statement", "My back is killing me from this drive.", "written"),
    # --- chat: no tool, and nothing to offer --------------------------------
    ("chat", "I like comedy.", "log turn_phantom"),
    ("chat", "Do you like movies?", "log talk.transcript"),
    ("chat", "Hey, what's up", "log talk.transcript"),
    ("chat", "Have you ever been to a service center?", "log talk.transcript"),
    ("chat", "Thanks for your help.", "log turn_phantom"),
]

YES = "Yeah."

# An offer is a question she asks about doing something. Deliberately loose on
# the verb and strict on the shape: it has to be a question, and it has to be
# about her doing or finding something -- "want me to", "should I", "I can ...
# if you want", "how about I". Every reply is also printed and saved, because a
# regex is a first read, not the verdict.
_OFFER = re.compile(
    r"(want me to|should i|shall i|i can .*\?|i could .*\?|how about i|"
    r"want to (?:grab|stop|find|pull|look|swing|get|hit|go)|"
    r"(?:up for|fancy|feel like) .*\?|"
    r"(?:look|find|check|search|pull up|see if).*\?|"
    r"want (?:a|some|the) .*\?)",
    re.I | re.S)

# A request answered with a question about whether to act.
_CONFIRM = re.compile(
    r"(want me to|should i|shall i|do you want|would you like|"
    r"you want me|i can .* if you)",
    re.I)


# A place, distance or time said in a first response. Nothing has come back
# from a tool at that point -- the bench never returns a result before the
# response ends -- so any of these is unsourced by construction.
_UNSOURCED = re.compile(
    r"\b(?:about|around|in|within|under|only|just|roughly)\s+"
    r"(?:a|an|\d+|one|two|three|four|five|ten|fifteen|twenty|few|couple of)\s+"
    r"(?:miles?|minutes?|km|kilomet\w*|feet|metres|meters|blocks?)\b|"
    r"\b\d+(?:\.\d+)?\s*(?:mi|km|miles?|minutes?|mins?)\b|"
    r"coming up|(?:^|[.!?]\s+)there'?s an? |right (?:up|down) the|"
    r"just (?:up|down) the",
    re.I)


def is_unsourced(text: str) -> bool:
    return bool(_UNSOURCED.search(text or ""))


def is_offer(text: str) -> bool:
    return "?" in text and bool(_OFFER.search(text))


def opening(text: str, n: int = 2) -> str:
    words = re.findall(r"[a-z']+", text.lower())
    return " ".join(words[:n])


def session_config(instructions, tools):
    cfg = {
        "type": "realtime",
        "output_modalities": ["audio"],
        "instructions": instructions,
        "tools": tools,
        "tool_choice": "auto",
        "audio": {
            "input": {
                "format": {"type": "audio/pcm", "rate": xai_voice.RATE},
                "transcription": {"model": config.XAI_STT_SESSION_MODEL},
                "turn_detection": {"type": "server_vad"},
            },
            "output": {
                "voice": config.XAI_VOICE,
                "format": {"type": "audio/pcm", "rate": xai_voice.RATE},
            },
        },
    }
    if config.XAI_VOICE_EFFORT:
        cfg["reasoning"] = {"effort": config.XAI_VOICE_EFFORT}
    return cfg


async def _turn(ws, text):
    """One driver line in, one response out: (tools called, what she said)."""
    await ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": text}]}}))
    await ws.send(json.dumps({"type": "response.create"}))
    calls, said = [], ""
    while True:
        ev = json.loads(await asyncio.wait_for(ws.recv(), 60))
        t = ev.get("type", "")
        if t == "response.output_item.done":
            it = ev.get("item") or {}
            if it.get("type") in ("function_call", "function"):
                calls.append({"name": it.get("name"),
                              "args": it.get("arguments"),
                              "call_id": it.get("call_id")})
        elif t == "response.output_audio_transcript.done":
            said = (said + " " + (ev.get("transcript") or "")).strip()
        elif t == "error":
            raise RuntimeError(str(ev.get("error"))[:300])
        elif t == "response.done":
            return calls, said


async def one(kind, text, instructions, tools):
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
                raise RuntimeError("session refused: " + str(ev.get("error"))[:300])

        calls, said = await _turn(ws, text)
        rec = {"kind": kind, "text": text, "calls": calls, "said": said}
        # THE SECOND HALF OF THE RULE: an offer is only right if "yeah" then
        # does the thing. Asked only where there was an offer to accept.
        if kind == "statement" and not calls and is_offer(said):
            calls2, said2 = await _turn(ws, YES)
            rec["yes_calls"], rec["yes_said"] = calls2, said2
        return rec


def score(recs):
    out = {}
    st = [r for r in recs if r["kind"] == "statement"]
    rq = [r for r in recs if r["kind"] == "request"]
    ch = [r for r in recs if r["kind"] == "chat"]
    out["statement_no_tool"] = (sum(not r["calls"] for r in st), len(st))
    out["statement_offered"] = (
        sum(not r["calls"] and is_offer(r["said"]) for r in st), len(st))
    offered = [r for r in st if "yes_calls" in r]
    out["statement_acts_on_yes"] = (
        sum(bool(r["yes_calls"]) for r in offered), len(offered))
    out["request_acted"] = (
        sum(bool(r["calls"]) and not _CONFIRM.search(r["said"]) for r in rq),
        len(rq))
    out["request_asked_first"] = (
        sum((not r["calls"]) and bool(_CONFIRM.search(r["said"])) for r in rq),
        len(rq))
    out["chat_no_tool"] = (sum(not r["calls"] for r in ch), len(ch))
    out["unsourced_claims"] = (sum(is_unsourced(r["said"]) for r in recs),
                               len(recs))
    said = [r["said"] for r in st if r["said"]]
    out["openings_1w"] = (len({opening(s, 1) for s in said}), len(said))
    out["openings_2w"] = (len({opening(s, 2) for s in said}), len(said))
    out["openings_3w"] = (len({opening(s, 3) for s in said}), len(said))
    return out


async def run(trials, conc):
    instructions = realtime.instructions()
    tools = [dict(t) for t in realtime.session_config()["tools"]]
    sem = asyncio.Semaphore(conc)
    jobs = [(k, t, src) for k, t, src in SET for _ in range(trials)]

    async def go(k, t, src):
        async with sem:
            for attempt in range(3):
                try:
                    r = await one(k, t, instructions, tools)
                    r["src"] = src
                    return r
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    await asyncio.sleep(2 + 3 * attempt)
            return {"kind": k, "text": t, "src": src, "calls": [], "said": "",
                    "error": err}

    t0 = time.time()
    recs = await asyncio.gather(*(go(*j) for j in jobs))
    return instructions, recs, time.time() - t0


def show(res):
    s = res["score"]
    f = lambda p: f"{p[0]}/{p[1]}"
    print(f"\n== {res['label']}  (instructions {res['instructions_sha']}, "
          f"{res['n']} turns, {res['errors']} errors, {res['seconds']:.0f}s)")
    for k, v in s.items():
        print(f"   {k:<24} {f(v)}")


def compare(a, b):
    sa, sb = a["score"], b["score"]
    print(f"\n{'':<24} {a['label']:>12} {b['label']:>12}")
    for k in sa:
        print(f"{k:<24} {sa[k][0]:>7}/{sa[k][1]:<4} {sb[k][0]:>7}/{sb[k][1]:<4}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--only", help="substring filter on the utterance text")
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        compare(*(json.load(open(p)) for p in a.compare))
        return 0
    if not os.getenv("XAI_API_KEY"):
        print("XAI_API_KEY is not set")
        return 2
    global SET
    if a.only:
        SET = [s for s in SET if a.only.lower() in s[1].lower()]
    instructions, recs, secs = asyncio.run(run(a.trials, a.conc))
    res = {"label": a.label, "model": config.XAI_VOICE_MODEL,
           "effort": config.XAI_VOICE_EFFORT,
           "instructions_sha": hashlib.sha1(
               instructions.encode()).hexdigest()[:10],
           "n": len(recs), "errors": sum("error" in r for r in recs),
           "seconds": secs, "score": score(recs), "records": recs}
    for r in recs:
        tool = ",".join(c["name"] for c in r["calls"]) or "-"
        yes = ""
        if "yes_calls" in r:
            yes = "  | yeah -> " + (",".join(c["name"] for c in r["yes_calls"]) or "-")
        print(f"[{r['kind'][:4]}] {r['text'][:44]!r:<48} tool={tool:<18} "
              f"{(r.get('error') or r['said'])[:110]!r}{yes}")
    show(res)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"   wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
