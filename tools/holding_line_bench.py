"""holding_line_bench.py — does she say something before the slow tool, and only the slow one?

    python -m tools.holding_line_bench --label before --out runs/holding_before.json
    python -m tools.holding_line_bench --compare runs/holding_before.json runs/holding_after.json

THE RULE UNDER TEST. deep_dive takes ten to twenty-five seconds; on drive
3d69ebaa (2026-10-01) it took 15.9 s and the driver heard 18.2 s of nothing,
because the calling response said 0 characters. So:

  slow      a question that goes to deep_dive (or search_local_news, the
            other slow one, twenty to fifty seconds) gets one short spoken line IN THE
            SAME RESPONSE, before the function call -- audio already flowing
            while the research runs.
  instant   find_places, look, nav_status answer in milliseconds, so nothing
            goes in front of them. This arm is the control: fixing the slow
            tool must not teach her to talk in front of the fast ones.

THE NUMBERS:

  slow_called         deep_dive was called at all (a trial that answered from
                      memory or picked another tool says nothing about the line)
  slow_spoke_first    of those, words were said BEFORE the call, in that order
  distinct_lines      how many DIFFERENT holding lines across every trial, and
                      the most common one's share. A line in the prompt gets
                      copied, and a copy shows up here as one phrase every time.
  instant_silent      instant-tool trials with nothing said before the call

ONE SESSION PER TRIAL, text in, as tools/companion_bench.py and for its reasons;
the session is realtime.instructions() and realtime.session_config()'s tools
read at run time. Tool calls are never answered: the line before the call is
the whole measurement, and the response ends where the call is made.
"""
import argparse
import asyncio
import collections
import hashlib
import json
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
from tools.companion_bench import session_config                # noqa: E402

SLOW = "slow"
INSTANT = "instant"

# Slow: questions that need looking up or thinking through, none of them a
# place near the car, the route, the weather, the news or the camera -- the
# things deep_dive's own description sends elsewhere. The first is the drive's.
SET = [
    (SLOW, "What are the current showtimes for the Laemmle Monica Film "
           "Center?", "log 3d69ebaa deep_dive.question"),
    (SLOW, "Can you look up how much a Rivian R2 costs?", "written"),
    (SLOW, "How does a heat pump in an electric car actually work? Go "
           "into detail.", "written"),
    (SLOW, "What's the current exchange rate from dollars to euros?",
     "written"),
    (SLOW, "Can you find out whether the Model Y or the Ioniq 5 has more "
           "real-world range?", "written"),
    (SLOW, "If I drive fifty miles a day, how long would a set of tires "
           "last? Work it out.", "written"),
    (INSTANT, "Find me coffee.", "brief"),
    (INSTANT, "What do you see?", "log talk.transcript"),
    (INSTANT, "What's the next turn?", "log turn_phantom"),
    (INSTANT, "Any gas stations near here?", "written"),
]

SLOW_TOOLS = {realtime.TOOL_NAME, realtime.NEWS_TOOL_NAME}
INSTANT_TOOLS = {realtime.PLACES_TOOL_NAME, realtime.LOOK_TOOL_NAME,
                 realtime.NAV_TOOL_NAME, realtime.NAV_DIRECTIONS_TOOL_NAME}


def norm(line: str) -> str:
    """What makes two holding lines the same line: words, not punctuation."""
    return " ".join(re.findall(r"[a-z']+", (line or "").lower()))


async def _turn(ws, text):
    """One driver line in -> (first tool called, words said before it, all words).

    ORDER is read from output items, not from when transcripts finish: a
    message item that was ADDED before the function_call item is speech in
    front of the call, which is the only kind that fills the wait.
    """
    await ws.send(json.dumps({
        "type": "conversation.item.create",
        "item": {"type": "message", "role": "user",
                 "content": [{"type": "input_text", "text": text}]}}))
    await ws.send(json.dumps({"type": "response.create"}))
    order, said, first_call = [], {}, None
    while True:
        ev = json.loads(await asyncio.wait_for(ws.recv(), 60))
        t = ev.get("type", "")
        if t == "response.output_item.added":
            it = ev.get("item") or {}
            order.append((it.get("type"), it.get("id"), it.get("name")))
        elif t == "response.output_audio_transcript.delta":
            iid = ev.get("item_id")
            said[iid] = said.get(iid, "") + (ev.get("delta") or "")
        elif t == "response.output_audio_transcript.done":
            iid = ev.get("item_id")
            if ev.get("transcript"):
                said[iid] = ev["transcript"]
        elif t == "error":
            raise RuntimeError(str(ev.get("error"))[:300])
        elif t == "response.done":
            break
    before, everything = [], []
    for kind, iid, name in order:
        if kind in ("function_call", "function"):
            if first_call is None:
                first_call = name
            continue
        if kind == "message" and said.get(iid):
            everything.append(said[iid])
            if first_call is None:
                before.append(said[iid])
    # Transcripts with no item_id the order can place: count as spoken, not
    # as "before" -- an unplaceable line is not evidence of the order.
    for iid, s in said.items():
        if iid not in {o[1] for o in order} and s:
            everything.append(s)
    return first_call, " ".join(before).strip(), " ".join(everything).strip()


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
        call, before, said = await _turn(ws, text)
        return {"kind": kind, "text": text, "call": call, "before": before,
                "said": said}


def score(recs):
    slow = [r for r in recs if r["kind"] == SLOW and r["call"] in SLOW_TOOLS]
    inst = [r for r in recs if r["kind"] == INSTANT and r["call"] in INSTANT_TOOLS]
    lines = [norm(r["before"]) for r in slow if r["before"]]
    common = collections.Counter(lines).most_common(1)
    return {
        "slow_called": (len(slow), sum(r["kind"] == SLOW for r in recs)),
        "slow_spoke_first": (len(lines), len(slow)),
        "distinct_lines": (len(set(lines)), len(lines)),
        "most_common_line": (common[0][1] if common else 0, len(lines)),
        "instant_silent": (sum(not r["before"] for r in inst), len(inst)),
    }


async def run(trials, conc):
    instructions = realtime.instructions()
    tools = [dict(t) for t in realtime.session_config()["tools"]]
    sem = asyncio.Semaphore(conc)
    jobs = [(k, t, src) for k, t, src in SET for _ in range(trials)]

    async def go(k, t, src):
        err = None
        async with sem:
            for attempt in range(3):
                try:
                    r = await one(k, t, instructions, tools)
                    r["src"] = src
                    return r
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                    await asyncio.sleep(2 + 3 * attempt)
            return {"kind": k, "text": t, "src": src, "call": None,
                    "before": "", "said": "", "error": err}

    t0 = time.time()
    recs = await asyncio.gather(*(go(*j) for j in jobs))
    return instructions, recs, time.time() - t0


def show(res):
    s = res["score"]
    print(f"\n== {res['label']}  (instructions {res['instructions_sha']}, "
          f"{res['n']} trials, {res['errors']} errors, {res['seconds']:.0f}s)")
    for k, v in s.items():
        print(f"   {k:<20} {v[0]}/{v[1]}")
    c = collections.Counter(norm(r["before"]) for r in res["records"]
                            if r["kind"] == SLOW and r["before"])
    print("   holding lines:")
    for line, n in c.most_common():
        print(f"     {n:>3}  {line}")


def compare(a, b):
    sa, sb = a["score"], b["score"]
    print(f"\n{'':<20} {a['label']:>12} {b['label']:>12}")
    for k in dict.fromkeys(list(sa) + list(sb)):
        f = lambda s: f"{s[k][0]:>7}/{s[k][1]:<4}" if k in s else f"{'-':>12}"
        print(f"{k:<20} {f(sa)} {f(sb)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--trials", type=int, default=5)
    ap.add_argument("--conc", type=int, default=6)
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs=2)
    a = ap.parse_args()
    if a.compare:
        x, y = (json.loads(Path(p).read_text()) for p in a.compare)
        show(x), show(y), compare(x, y)
        return 0
    instructions, recs, secs = asyncio.run(run(a.trials, a.conc))
    res = {"label": a.label,
           "instructions_sha": hashlib.sha256(instructions.encode()).hexdigest()[:12],
           "n": len(recs), "errors": sum(bool(r.get("error")) for r in recs),
           "seconds": secs, "score": score(recs), "records": recs}
    for r in recs:
        print(f"[{r['kind']:<7}] {r['call'] or '-':<14} "
              f"before={r['before']!r:.70}  | {r['text'][:50]}"
              + (f"  ERROR {r['error']}" if r.get("error") else ""))
    show(res)
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
