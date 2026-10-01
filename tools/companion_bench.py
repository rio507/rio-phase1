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

THE NUMBERS, per arm:

  statements  no tool on the first turn AND an offer in the reply; and, as a
              second line, whether a bare "Yeah." then produces the tool call
  requests    a tool call on the first turn with no confirmation question
  openings    across every statement reply, how many DISTINCT first-two-word
              openings. A prompt that carries an example gets copied, and a
              copy shows up here as the same opening every time.
  drive       chat and statement replies that talk about the drive, the trip,
              the road or the journey when the driver's line did not. The car
              is where she is, not what she talks about.
  lively      (2026-10-01) on statement and chat replies, judged by a model
              against a fixed rubric (judge() below): reacted with personality
              rather than a flat acknowledgement; added a comment of her own;
              asked a follow-up; and claimed a body or experience she does not
              have (should be ~0). Plus `unprompted`: anything she said with no
              driver line in front of it, listened for after her last reply --
              must stay zero. Lively is how she speaks, never speaking more.
  hungry      a NAMED MUST-PASS in the gate: "I'm kinda hungry" is a
              statement and is never a search on the first turn. It regressed
              to 5/6 once without anything saying so.

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
    ("statement", "I'm kinda hungry.", "brief 2026-09-28"),
    ("statement", "I'm tired.", "brief"),
    ("statement", "This traffic is killing me.", "brief"),
    ("statement", "I could really use a coffee right now.", "written"),
    ("statement", "Man, I'm running low on gas.", "written"),
    ("statement", "Ugh, I think I'm going to be late.", "written"),
    ("statement", "I haven't eaten all day.", "written"),
    ("statement", "I kind of want to see a movie tonight.", "written"),
    ("statement", "I need to stretch my legs.", "written"),
    ("statement", "My back is killing me from this drive.", "written"),
    ("statement", "This traffic is brutal.", "brief 2026-10-01"),
    # --- chat: no tool, and nothing to offer --------------------------------
    ("chat", "I like comedy.", "log turn_phantom"),
    ("chat", "Do you like movies?", "log talk.transcript"),
    ("chat", "Hey, what's up", "log talk.transcript"),
    ("chat", "Hello.", "log 4ae33786 driver_said"),
    ("chat", "Have you ever been to a service center?", "log talk.transcript"),
    ("chat", "Thanks for your help.", "log turn_phantom"),
    ("chat", "Not much.", "brief 2026-10-01"),
    # --- conversations: more than one line, in one session ------------------
    # The complaint was never the first reply. On 4ae33786 "Hello." got "Hey.
    # What's on your mind?" and the SECOND "Hello." got "Yeah, I'm here. How's
    # the drive going?" -- small talk runs out on the second turn, and the trip
    # is what is left. Lines separated by " | ", every reply scored.
    ("convo", "Hello. | Hello.", "log 4ae33786 driver_said"),
    ("convo", "Hello? | Can you hear me? | Hello?", "log 4ae33786 driver_said"),
    ("convo", "Hey, what's up | Not much, you?", "written"),
    ("convo", "Hello. | Nothing really. | Yeah.", "written"),
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


# THE DRIVE AS SMALL TALK. "How's the drive going?", "just enjoying the
# drive" -- the trip itself as her topic, which is what a service says. Counted
# on chat and statement replies only, and only where the driver's own line did
# not bring any of it up: "My back is killing me from this drive" invites the
# word, "Hello." does not.
_DRIVE_TALK = re.compile(
    r"\b(?:drive|drives|driving|drove|trip|trips|road|roads|"
    r"journey|journeys|"
    # ...and the same habit in other words, seen in both arms: "Hey. Just
    # cruising with you.", "Not much, just riding along."
    r"cruise|cruising|ride|rides|riding)\b", re.I)


def mentions_drive(text: str) -> bool:
    return bool(_DRIVE_TALK.search(text or ""))


def is_unsourced(text: str) -> bool:
    return bool(_UNSOURCED.search(text or ""))


def is_offer(text: str) -> bool:
    return "?" in text and bool(_OFFER.search(text))


def opening(text: str, n: int = 2) -> str:
    words = re.findall(r"[a-z']+", text.lower())
    return " ".join(words[:n])


def clock_line(now=None) -> str:
    """The line static/rio_xai_session.js appends to her instructions, for the
    car's local time. The bench runs in UTC on the pod; the car is in LA."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    d = now or datetime.now(ZoneInfo("America/Los_Angeles"))
    time = d.strftime("%I:%M %p").lstrip("0")
    return (f"THE CLOCK: it is {time} on {d.strftime('%A')} where the car is. "
            "This is the real local time, kept current through the drive, and "
            "the only source for the time of day.")


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


UNPROMPTED_S = 6.0
OPENING = None          # --opening: replaces the character's first line


async def _listen(ws, seconds):
    """Responses the server opens with nobody having said anything."""
    n = 0
    end = time.monotonic() + seconds
    try:
        while True:
            left = end - time.monotonic()
            if left <= 0:
                break
            ev = json.loads(await asyncio.wait_for(ws.recv(), left))
            if ev.get("type") == "response.created":
                n += 1
    except asyncio.TimeoutError:
        pass
    return n


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

        if kind == "convo":
            lines = [x.strip() for x in text.split("|")]
            turns = []
            for line in lines:
                c, s = await _turn(ws, line)
                turns.append({"text": line, "calls": c, "said": s})
            return {"kind": kind, "text": text, "turns": turns,
                    "calls": [c for t in turns for c in t["calls"]],
                    "said": " / ".join(t["said"] for t in turns)}
        calls, said = await _turn(ws, text)
        rec = {"kind": kind, "text": text, "calls": calls, "said": said}
        # THE SECOND HALF OF THE RULE: an offer is only right if "yeah" then
        # does the thing. Asked only where there was an offer to accept.
        if kind == "statement" and not calls and is_offer(said):
            calls2, said2 = await _turn(ws, YES)
            rec["yes_calls"], rec["yes_said"] = calls2, said2
        if kind in ("statement", "chat") and not calls:
            rec["unprompted"] = await _listen(ws, UNPROMPTED_S)
        return rec


JUDGE_RUBRIC = """You grade one reply from a voice companion who sits beside a driver.
The driver said something; she replied. Judge ONLY the reply, against these
questions, and answer with JSON and nothing else:

{"personality": true|false,   // reacts like a person with character -- a
                              // take, wit, warmth, an opinion -- not a flat
                              // acknowledgement ("Got it." "Sounds good."
                              // "Fair enough." alone are flat)
 "own_comment": true|false,   // adds something of HER OWN beyond acknowledging
                              // and offering: a take, a joke, a connection
 "follow_up": true|false,     // asks the driver a question that keeps the
                              // conversation going (an offer to search or
                              // navigate does NOT count)
 "claims_body": true|false}   // claims a body or a lived experience she cannot
                              // have: being hungry or tired herself, having
                              // eaten or slept, physically going somewhere
"""


def judge_one(text, said):
    import llm_provider
    c = llm_provider.client("chat")
    r = c.chat.completions.create(
        model=llm_provider.model_of("chat"), temperature=0,
        messages=[{"role": "system", "content": JUDGE_RUBRIC},
                  {"role": "user", "content": f"Driver: {text}\nHer reply: {said}"}])
    raw = r.choices[0].message.content or "{}"
    raw = raw[raw.find("{"): raw.rfind("}") + 1]
    try:
        return json.loads(raw)
    except Exception:
        return {}


def judge(recs):
    """Adds r['judged'] to every statement and chat record that has a reply."""
    from concurrent.futures import ThreadPoolExecutor
    todo = [r for r in recs if r["kind"] in ("statement", "chat") and r.get("said")
            and "judged" not in r]
    with ThreadPoolExecutor(8) as ex:
        for r, j in zip(todo, ex.map(lambda r: judge_one(r["text"], r["said"]), todo)):
            r["judged"] = j


HUNGRY = ("I'm kinda hungry.", "I'm feeling kinda hungry.")
HUNGRY_TRIALS = 8


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
    unprompted = [r for r in st + ch
                  if r["said"] and not mentions_drive(r["text"])]
    out["drive_mentions"] = (sum(mentions_drive(r["said"]) for r in unprompted),
                             len(unprompted))
    # Every reply in a conversation, each against everything the driver had
    # said up to it.
    cv, hits = [], 0
    for r in recs:
        if r["kind"] != "convo":
            continue
        heard = ""
        for t in r.get("turns") or []:
            heard += " " + t["text"]
            if t["said"] and not mentions_drive(heard):
                cv.append(t)
                hits += mentions_drive(t["said"])
    out["convo_drive_mentions"] = (hits, len(cv))
    out["convo_no_tool"] = (sum(not r["calls"] for r in recs
                                if r["kind"] == "convo"),
                            sum(r["kind"] == "convo" for r in recs))
    said = [r["said"] for r in st if r["said"]]
    out["opens_want_me"] = (sum(opening(s, 2) == "want me" for s in said),
                            len(said))
    out["openings_1w"] = (len({opening(s, 1) for s in said}), len(said))
    out["openings_2w"] = (len({opening(s, 2) for s in said}), len(said))
    out["openings_3w"] = (len({opening(s, 3) for s in said}), len(said))
    # --- lively (judged) and the guard on it -------------------------------
    sc = [r for r in recs if r["kind"] in ("statement", "chat") and r.get("judged")]
    for k in ("personality", "own_comment", "follow_up", "claims_body"):
        out[k] = (sum(bool(r["judged"].get(k)) for r in sc), len(sc))
    out["asks_question"] = (sum("?" in r["said"] for r in sc), len(sc))
    both = [r["said"] for r in recs if r["kind"] in ("statement", "chat") and r.get("said")]
    out["openings_2w_all"] = (len({opening(x, 2) for x in both}), len(both))
    up = [r for r in recs if "unprompted" in r]
    out["unprompted"] = (sum(r["unprompted"] for r in up), len(up))
    hg = [r for r in recs if r["text"] in HUNGRY]
    out["hungry_no_search"] = (sum(not r["calls"] for r in hg), len(hg))
    return out


async def run(trials, conc):
    # As the transport sends them: with the car's clock on the end.
    instructions = realtime.instructions() + "\n\n" + clock_line()
    if OPENING is not None:
        # A/B the character's first line without editing the prompt file.
        head = "You are RIO, a voice companion sitting beside the driver."
        assert head in instructions, "opening line not found"
        instructions = instructions.replace(head, OPENING)
    tools = [dict(t) for t in realtime.session_config()["tools"]]
    sem = asyncio.Semaphore(conc)
    # The must-pass line is noisy -- 0/3 to 3/3 searches on one prompt across
    # earlier runs -- so it gets enough trials for "all of them" to mean
    # something.
    jobs = [(k, t, src) for k, t, src in SET
            for _ in range(max(trials, HUNGRY_TRIALS) if t in HUNGRY else trials)]

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


# THE GATE A NEW SECTION HAS TO PASS before it spends the room the per-response
# ceiling leaves (tools/realtime_selftest.py, PER_RESPONSE_CEILING). Set from
# arm I, 2d5df61, pooled over two runs -- the prompt as it stood when the room
# was opened. More prompt is more text to copy, so the bar is that nothing
# measured here gets worse:
#
#   invented place / distance / ETA     0            it was 0/180
#   "Want me to..." openings            <= 14 of 51  the one shape she already
#                                                    repeats; as a rate, because
#                                                    the count of statement
#                                                    replies varies by run
#   requests acted on at once           all of them  it was 90/90, and asking
#                                                    "do you want me to look?"
#                                                    after "find me coffee" is
#                                                    worse than any statement
#                                                    miss
GATE_WANT_ME = (14, 51)


def gate(score):
    """[(check, passed, detail)] for the three bars above."""
    u, _ = score["unsourced_claims"]
    wm, n = score["opens_want_me"]
    ra, rn = score["request_acted"]
    ask, _ = score["request_asked_first"]
    limit = GATE_WANT_ME[0] / GATE_WANT_ME[1]
    hn, hd = score.get("hungry_no_search", (0, 0))
    up, upn = score.get("unprompted", (0, 0))
    return [
        ("no invented place, distance or ETA", u == 0, f"{u} found"),
        ("MUST-PASS: 'I'm kinda hungry' is not a search", hd > 0 and hn == hd,
         f"{hn}/{hd} answered without a tool"),
        ("no unprompted speech", up == 0, f"{up} responses in {upn} silences"),
        ("'Want me to...' openings no higher", n == 0 or wm / n <= limit,
         f"{wm}/{n} against {GATE_WANT_ME[0]}/{GATE_WANT_ME[1]}"),
        ("every request acted on at once", ra == rn and ask == 0,
         f"{ra}/{rn} acted, {ask} asked first"),
    ]


# The lines reported word for word, every trial: the brief of 2026-10-01.
VERBATIM = ("I'm kinda hungry.", "I'm tired.", "This traffic is brutal.",
            "I like comedy.", "Hey, what's up", "Thanks for your help.",
            "Not much.")


def verbatim(recs):
    print("\n   VERBATIM")
    for line in VERBATIM:
        rs = [r for r in recs if r["text"] == line]
        if not rs:
            continue
        print(f"   {line}")
        for r in rs:
            tool = ",".join(c["name"] for c in r["calls"]) or ""
            j = r.get("judged") or {}
            flags = "".join(k[0].upper() for k in ("personality", "own_comment", "follow_up")
                            if j.get(k))
            print(f"     [{flags:<3}]{' TOOL=' + tool if tool else ''} {r['said']}")


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
    for k in dict.fromkeys(list(sa) + list(sb)):
        f = lambda s: f"{s[k][0]:>7}/{s[k][1]:<4}" if k in s else f"{'-':>12}"
        print(f"{k:<24} {f(sa)} {f(sb)}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", default="run")
    ap.add_argument("--trials", type=int, default=2)
    ap.add_argument("--conc", type=int, default=4)
    ap.add_argument("--only", help="substring filter on the utterance text")
    ap.add_argument("--out")
    ap.add_argument("--opening", help="replace the character's first line (A/B)")
    ap.add_argument("--compare", nargs=2)
    ap.add_argument("--rescore", nargs="+",
                    help="recompute the score of saved runs with this file's "
                         "scorer, in place")
    ap.add_argument("--gate", action="store_true",
                    help="exit non-zero unless the run passes the gate for a "
                         "new prompt section (see GATE_WANT_ME)")
    a = ap.parse_args()
    if a.rescore:
        for p in a.rescore:
            res = json.load(open(p))
            judge(res["records"])
            res["score"] = score(res["records"])
            Path(p).write_text(json.dumps(res, indent=1))
            show(res)
        return 0
    if a.compare:
        compare(*(json.load(open(p)) for p in a.compare))
        return 0
    if not os.getenv("XAI_API_KEY"):
        print("XAI_API_KEY is not set")
        return 2
    global SET, OPENING
    OPENING = a.opening
    if a.only:
        SET = [s for s in SET if a.only.lower() in s[1].lower()]
    instructions, recs, secs = asyncio.run(run(a.trials, a.conc))
    judge(recs)
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
    verbatim(res["records"])
    if a.out:
        Path(a.out).write_text(json.dumps(res, indent=1))
        print(f"   wrote {a.out}")
    if a.gate:
        checks = gate(res["score"])
        print("\n   GATE")
        for name, passed, detail in checks:
            print(f"   {'pass' if passed else 'FAIL'}  {name:<38} {detail}")
        return 0 if all(p for _, p, _ in checks) else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
