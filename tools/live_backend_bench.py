"""Which backend model reasons for RIO, decided by measurement.

    python -m tools.live_backend_bench
    python -m tools.live_backend_bench --n 5
    python -m tools.live_backend_bench --models gpt-5.6-terra,gpt-5.6-luna

gpt-live-1 does not think. It listens, speaks, and hands the thinking to a
backend text model over Responses delegation (docs/live_delegation.md), and
WHICH model that is, is a choice with no default worth trusting: the voice
layer is identical either way, so everything the driver experiences past "she
heard me" is this model's doing.

WHAT THIS MEASURES, and why it is these two things:

  routing   Did it reach for the right tool? RIO has nine, several of which
            are near neighbours -- nav_status and nav_directions answer
            different halves of the same question, find_places and
            start_navigation both take a place name, and deep_dive is the one
            that must NOT fire for anything the camera can see. A backend that
            is fluent and picks `look` for "how far to the Getty" is worse
            than a slower one that does not, because the driver gets a
            confident answer about the wrong thing.
  latency   Time to the first token of the backend's answer. This is the part
            of speech-end -> first-audio that the backend owns; the voice
            layer's share is measured separately by tools/live_latency.py.

WHAT IT DOES NOT MEASURE: prosody, interruption, turn-taking, any of it. Those
belong to gpt-live-1 and do not change when this choice changes -- which is
precisely why this can be benched over the Responses API directly, without a
WebRTC session in the path adding a second of jitter to every trial.

The utterances are the scripted drive from tools/live_tool_turns.py, plus the
tools that script does not reach, so a model is scored on the whole tool
surface rather than the easy two-thirds of it.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv                              # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                               # noqa: E402
import realtime                                             # noqa: E402

# The candidates. Terra and Luna are what the delegation guide names; Sol is on
# the account and is benched rather than assumed away, because "the docs did
# not mention it" is not a measurement.
DEFAULT_MODELS = ["gpt-5.6-terra", "gpt-5.6-luna", "gpt-5.6-sol"]

# WHAT THE DRIVER SAYS AND WHICH TOOL IS RIGHT.
#
# `want` is None where the right answer is to reach for nothing. Those rows are
# not filler: a backend that calls a tool for "how long have you been driving
# with me" costs the driver a round trip for a sentence it already had, and
# over-reaching is the failure mode a tool-rich session actually has.
CASES = [
    ("What do you see outside?",                                  "look"),
    ("What kind of car is in front of us?",                        "look"),
    ("What's that building on the left?",                          "look"),
    ("Can you look up why carmakers switched from hydraulic to "
     "electric power steering?",                              "deep_dive"),
    ("Take me to the Getty.",                             "start_navigation"),
    ("What are the directions?",                            "nav_directions"),
    ("How far is it to the Getty?",                              "nav_status"),
    ("How are my tires?",                                    "vehicle_status"),
    ("Is there a coffee shop near here?",                       "find_places"),
    ("How long have you been driving with me?",                         None),
    ("Thanks, that's helpful.",                                         None),
]


def _tools_for_responses() -> list:
    """RIO's tools, in the shape the Responses API wants.

    The SAME schemas the realtime session sends -- imported, not retyped, for
    the reason every other copy in this repo is imported: a tool description
    that drifts between the two paths is a routing change nobody made on
    purpose. The conditional pair rides along because a bench that leaves them
    out is scoring a tool surface the car does not have.
    """
    out = []
    for t in list(realtime.BASE_TOOLS) + list(
            realtime.CONDITIONAL_TOOLS.get("routing", [])):
        spec = dict(t)
        spec.setdefault("type", "function")
        out.append(spec)
    return out


def _instructions() -> str:
    """The backend prompt.

    Per the prompting guide the split is: conversational behaviour in the live
    prompt, procedure and tool logic here. So this is RIO's tool discipline
    with the speaking style left out -- the style now belongs to gpt-live-1 and
    saying it twice would be two places to change it.
    """
    return (
        "You are the reasoning backend for RIO, an in-car assistant.\n"
        "Pick the one tool that answers what was asked, or none if you can "
        "answer from what you already know.\n"
        "- Anything the forward camera can see (the road, a car, a sign, a "
        "building, 'what's that') is `look`.\n"
        "- The route we are on is `nav_status` (where things stand) or "
        "`nav_directions` (the turn list).\n"
        "- Starting a route is `start_navigation`. Finding a business or "
        "place near the car is `find_places`.\n"
        "- The car's own sensors are `vehicle_status`.\n"
        "- `deep_dive` is only for research or multi-step reasoning with no "
        "picture and no sensor behind it.\n"
        "Answers are spoken aloud in a moving car: one or two sentences."
    )


def run_model(model: str, n: int, timeout: float = 60.0) -> dict:
    from openai import OpenAI
    cl = OpenAI(timeout=timeout)
    tools = _tools_for_responses()
    lat, hits, miss = [], 0, []
    for say, want in CASES:
        for _ in range(n):
            t0 = time.time()
            first = None
            called = None
            try:
                stream = cl.responses.create(
                    model=model, instructions=_instructions(),
                    input=[{"role": "user", "content": say}],
                    tools=tools, tool_choice="auto", stream=True)
                for ev in stream:
                    et = getattr(ev, "type", "")
                    if first is None and et in (
                            "response.output_text.delta",
                            "response.function_call_arguments.delta",
                            "response.output_item.added"):
                        first = time.time()
                    if et == "response.output_item.done":
                        it = getattr(ev, "item", None)
                        nm = getattr(it, "name", None)
                        if nm and getattr(it, "type", "") in (
                                "function_call", "function"):
                            called = nm
            except Exception as e:
                miss.append((say, want, f"ERROR {type(e).__name__}: {e}"))
                continue
            if first:
                lat.append((first - t0) * 1000.0)
            if called == want:
                hits += 1
            else:
                miss.append((say, want, called))
    total = len(CASES) * n
    return {"model": model, "routed": hits, "total": total,
            "p50": statistics.median(lat) if lat else None,
            "p95": (statistics.quantiles(lat, n=20)[18]
                    if len(lat) >= 20 else (max(lat) if lat else None)),
            "misses": miss}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3,
                    help="trials per utterance per model")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    args = ap.parse_args()
    rows = []
    for m in [x.strip() for x in args.models.split(",") if x.strip()]:
        print(f"-- {m} ...", flush=True)
        r = run_model(m, args.n)
        rows.append(r)
        p50 = f'{r["p50"]:.0f}' if r["p50"] else "--"
        p95 = f'{r["p95"]:.0f}' if r["p95"] else "--"
        print(f'   routed {r["routed"]}/{r["total"]}  '
              f'first-token p50 {p50} ms  p95 {p95} ms', flush=True)
        for say, want, got in r["misses"][:8]:
            print(f'     MISS {say!r} want={want} got={got}', flush=True)
    print()
    print(f'{"model":<18}{"routed":>10}{"p50 ms":>10}{"p95 ms":>10}')
    for r in rows:
        p50 = f'{r["p50"]:.0f}' if r["p50"] else "--"
        p95 = f'{r["p95"]:.0f}' if r["p95"] else "--"
        print(f'{r["model"]:<18}{r["routed"]}/{r["total"]:<7}{p50:>10}{p95:>10}')
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
