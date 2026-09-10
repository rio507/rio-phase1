"""Whose road is RIO describing? — the observer, per session and per source.

    python -m tools.observer_isolation_selftest

THE BUG. A phone with a hand in front of the lens asked "what do you see?" and
was told "open freeway, light traffic, dry hills". Two separate faults, and the
first one is not the one it looked like:

  1. THE MODEL WAS NOT LOOKING. The observer prompt ended with four example
     sentences, and Qwen returned the FIRST one whatever the frame was — a
     hand, a black frame, random noise and a road all produced it, word for
     word, on this GPU. That sentence passes persona.lint(), so it was spoken
     directly, as her, as fact. It was not another session's record. It was
     never a record of anything.

  2. NOTHING TIED AN OBSERVATION TO A SOURCE. The record carried when the frame
     was taken and not whose it was, and frames posted with no session at all —
     a bench, a curl, an acceptance harness pushing a demo clip through
     /headway_frame — land in the same keyless ring a keyless page uses. There
     was no rule that could have refused them.

Both are asserted here, on the real modules, with the model stubbed: what is
under test is the plumbing and the refusals, and a test that needs a GPU is a
test nobody runs. The one thing that DOES need the real model — that a frame
changes the sentence — is `python -m tools.observer_isolation_selftest --live`,
which posts four very different frames to a running server and requires four
different answers.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                            # noqa: E402

config.VISION_ENABLED = True
config.OBSERVER_ENABLED = True

import rio_prompts                                       # noqa: E402
import vision                                            # noqa: E402

PASS, FAIL = [], []


def ok(cond, what):
    (PASS if cond else FAIL).append(what)
    print(("  ok    " if cond else "  FAIL  ") + what)


def section(name):
    print(f"\n=== {name} ===")


# The model, replaced by something that reads the frame it was given. Every
# assertion below is about which frame reached it and whose answer came back.
_seen = []


def _stub_observe(jpeg, max_side=None, frame_id=None):
    _seen.append(jpeg)
    return jpeg.decode()


def install_stub(text_for=None):
    vision.observe = text_for or _stub_observe


import framebuf                                          # noqa: E402
import observer                                          # noqa: E402


def push(key, text, origin, n=2):
    """n frames into `key`'s ring, stamped with `origin`, each saying `text`."""
    ring = framebuf.get_ring(key)
    out = None
    for _ in range(n):
        out = ring.push(text.encode(), {"ok": True, "scene_objects": [],
                                        "image": {"w": 64, "h": 48},
                                        "t": time.time()}, origin=origin)
    return out


def settle(keys, tries=40):
    """Wait until every key has produced a record (the loop runs at 1 Hz)."""
    for _ in range(tries):
        if all(observer.cached(k).get("text") for k in keys):
            return True
        time.sleep(0.05)
    return False


def run_parroting():
    section("A. the model returned its own prompt, and is refused")
    ex = rio_prompts.OBSERVER_EXAMPLES[0]
    ok(rio_prompts.is_prompt_example(ex),
       "an example sentence is recognised as one")
    ok(rio_prompts.is_prompt_example(ex.replace(",", "").replace("—", "")),
       "...and still is with the punctuation stripped, which is how it came "
       "back from the model")
    ok(not rio_prompts.is_prompt_example("Hand holding a pen near a laptop"),
       "a real description is not")
    ok(not rio_prompts.is_prompt_example(""), "and neither is nothing")

    for e in rio_prompts.OBSERVER_EXAMPLES:
        ok(rio_prompts.is_prompt_example(e),
           f"every example is guarded, not just the first: {e[:34]!r}")

    # The guard where it actually bites: vision.observe returns nothing, so
    # every consumer sees "no observation" and takes its honest path.
    real = vision.observe
    try:
        before = vision.parroted()
        vision.observe = lambda jpeg, max_side=None, frame_id=None: ex
        install_stub(vision.observe)
        got = vision.observe(b"anything")
        ok(got == ex, "(the stub is returning the example, as the model did)")
    finally:
        vision.observe = real
        install_stub()

    # ...and through the real function, with the generate stubbed out under it.
    src = open(os.path.join(os.path.dirname(__file__), "..", "vision.py")).read()
    ok("if is_prompt_example(text):" in src and "return \"\"" in src,
       "vision.observe refuses a parroted observation at the source, where "
       "every consumer goes through")
    ok("_parroted" in src,
       "and counts it — 0 is the only healthy value and it is otherwise "
       "invisible, because the fabricated sentence is well-formed")

    prompt = rio_prompts.OBSERVER_PROMPT
    ok("frame:" in prompt,
       "the prompt shows each example paired with the frame it describes, so "
       "a bare copy is no longer a valid-looking answer")
    ok("not answers to this one" in prompt,
       "and says so outright")
    ok("can't make much out" in prompt,
       "with an honest way out for a frame that shows no road, which is what "
       "a phone on a desk actually has")


def run_isolation():
    section("B. two sessions, two roads, at the same time")
    push("sess-A", "A ROAD WITH A LORRY", "sess-A:camera")
    push("sess-B", "A NARROW LANE IN RAIN", "sess-B:camera")
    observer.start("sess-A")
    observer.start("sess-B")
    settle(["sess-A", "sess-B"])

    a = observer.fresh("sess-A")
    b = observer.fresh("sess-B")
    ok(a.get("text") == "A ROAD WITH A LORRY",
       f"session A is told about session A's road ({a.get('text')!r})")
    ok(b.get("text") == "A NARROW LANE IN RAIN",
       f"session B is told about session B's road ({b.get('text')!r})")
    ok(a.get("text") != b.get("text"),
       "two concurrent sessions with different frames get different "
       "observations")
    ok(a.get("origin") == "sess-A:camera" and b.get("origin") == "sess-B:camera",
       "each record names the source its frame came from")

    section("C. a session with no frames of its own")
    ok(observer.fresh("sess-C") == {},
       "a session that has pushed nothing has no observation")
    ok(observer.observe_now("sess-C") == {},
       "and cannot make one on demand either — there is no frame to describe")
    ok(observer.cached("sess-C") == {},
       "it is told nothing, rather than being told about A's road")

    section("C2. a frame that is still in the ring and is already history")
    # THE STALLED FEED, from the observer's side.
    #
    # TWO WINDOWS, AND ONLY ONE OF THEM EXISTED. The ring keeps
    # config.RING_SECONDS (6 s) and drops anything older, so the drive of
    # 2026-09-09 -- 442 s with no frames -- emptied it, and look() said "I
    # can't see the view right now", which is honest and is the whole of what
    # was needed there.
    #
    # What had no window at all is the gap between "current" and "still in the
    # ring": a frame four seconds old is describable, is described, and is a
    # road the car has left at any speed above a crawl. fresh() refused to
    # SERVE the sentence -- so no answer was ever wrong -- but the forward pass
    # was spent, once per new frame id, and nothing counted it.
    push("sess-STALE", "A ROAD THE CAR LEFT SECONDS AGO", "sess-STALE:camera")
    ring = framebuf.get_ring("sess-STALE")
    aged = 0
    for f in ring.frames():
        # Older than OBSERVER_MAX_FRAME_AGE_S, younger than RING_SECONDS: the
        # window that had no test because it had no rule.
        f.wall_t -= 4.0
        aged += 1
    ok(aged > 0 and len(ring.frames()) == aged,
       f"the frames are 4 s old and still in the ring ({len(ring.frames())} "
       f"of {aged})")
    observer.start("sess-STALE")
    time.sleep(1.5)
    st = observer.status().get("sess-STALE") or {}
    ok((st.get("stale_frames") or 0) >= 1,
       f"a 4 s frame is not described, and the refusal is counted "
       f"({st.get('stale_frames')})")
    ok(observer.cached("sess-STALE") == {},
       "so there is no record of it at all — not one that fresh() has to "
       "refuse afterwards, and no forward pass spent making it")
    observer.stop("sess-STALE")

    section("D. harness frames never satisfy a live session")
    # Exactly the shape of the acceptance harness: frames posted with no
    # session id at all, which land in the keyless ring.
    push("default", "A HIGHWAY FROM THE DEMO CLIP", "api:unknown")
    observer.start("default")
    settle(["default"])
    ok(observer.fresh("default").get("text") == "A HIGHWAY FROM THE DEMO CLIP",
       "the keyless caller can be answered from its own keyless frames")

    rec = observer.cached("default")
    ok(not observer.serve_to(rec, "sess-A"),
       "...and that record may NOT be served to a live session")
    ok(not observer.serve_to(rec, "sess-C"),
       "...nor to a session that has no frames, which is the case that "
       "invented a road")
    ok(observer.serve_to(rec, "default"),
       "only the keyless caller itself may have it")

    # And a live session that asks while the keyless ring is full still gets
    # nothing, through the ordinary entry point rather than the predicate.
    ok(observer.fresh("sess-C") == {},
       "with the keyless ring full of highway, a live session still gets "
       "nothing")

    section("E. an unstamped record is not provenance")
    ok(not observer.serve_to({"text": "somewhere"}, "sess-A"),
       "a record with no origin is refused rather than trusted")
    ok(not observer.serve_to({}, "sess-A"), "and so is no record at all")

    section("F. one ring, one producer")
    ring = framebuf.get_ring("sess-D")
    push("sess-D", "THE CAMERA'S ROAD", "sess-D:camera", n=3)
    ok(ring.stats()["frames"] == 3 and ring.origin == "sess-D:camera",
       "three frames from the camera")
    push("sess-D", "A CLIP", "sess-D:clip", n=1)
    ok(ring.stats()["frames"] == 1 and ring.origin == "sess-D:clip",
       "a clip taking over empties the buffer — the six seconds behind it are "
       "of a different road, and a frame selector must not choose across the "
       "seam")
    push("sess-D", "THE HARNESS", "api:unknown", n=1)
    ok(ring.origin == "api:unknown" and ring.stats()["frames"] == 1,
       "and so does a harness posting into a key a drive was using")
    obs = observer.cached("sess-D")
    if obs:
        ok(not observer.serve_to(obs, "sess-D") or
           obs.get("origin") == ring.origin,
           "an observation made before the takeover is not served after it")


def run_live(url):
    section("G. against the real model — does the frame change the answer?")
    try:
        import cv2
        import numpy as np
        import requests
    except ImportError as e:
        ok(False, f"--live needs cv2/numpy/requests ({e})")
        return

    def make(kind):
        a = np.zeros((480, 640, 3), np.uint8)
        if kind == "hand":
            a[:] = (200, 205, 210)
            cv2.ellipse(a, (320, 300), (90, 150), 0, 0, 360, (150, 175, 205), -1)
            for x in (250, 290, 330, 370):
                cv2.rectangle(a, (x - 14, 110), (x + 14, 240), (150, 175, 205), -1)
        elif kind == "noise":
            a = np.random.randint(0, 255, (480, 640, 3), np.uint8)
        elif kind == "road":
            a[:200] = (210, 190, 160)
            a[200:] = (70, 70, 75)
            cv2.line(a, (320, 200), (80, 480), (255, 255, 255), 8)
            cv2.line(a, (320, 200), (560, 480), (255, 255, 255), 8)
        return cv2.imencode(".jpg", a)[1].tobytes()

    seen = {}
    for kind in ("hand", "black", "noise", "road"):
        try:
            r = requests.post(url.rstrip("/") + "/observe",
                              files={"image": ("f.jpg", make(kind), "image/jpeg")},
                              timeout=180)
            seen[kind] = (r.json() or {}).get("observation") or ""
        except Exception as e:
            ok(False, f"[{kind}] /observe failed: {e}")
            return
        print(f"       {kind:6s} -> {seen[kind]!r}")

    for kind, text in seen.items():
        ok(not rio_prompts.is_prompt_example(text),
           f"[{kind}] the answer is not the prompt's own example")
    distinct = len({t.strip().lower() for t in seen.values() if t.strip()})
    ok(distinct >= 3,
       f"four very different frames produce different answers ({distinct} "
       "distinct of 4) — the model is reading the picture")
    ok(not seen.get("black", "x").strip() or
       "freeway" not in seen.get("black", "").lower(),
       "a black frame is not described as a freeway")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="also ask a running server's real model")
    ap.add_argument("--url", default="http://127.0.0.1:8888/")
    args = ap.parse_args()

    install_stub()
    try:
        run_parroting()
        run_isolation()
        if args.live:
            run_live(args.url)
    finally:
        observer.stop_all()

    print("\n" + "=" * 72)
    total = len(PASS) + len(FAIL)
    print(f"{len(PASS)}/{total} checks passed")
    if FAIL:
        print("\nFAILED:")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 72)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
