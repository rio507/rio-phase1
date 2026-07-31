"""Phase A acceptance tests for the visual conversation path.

    python -m tools.visual_selftest --video /workspace/ufldv2/example.mp4
    python -m tools.visual_selftest --frames /some/dir --start 20 --count 24

Runs the real pipeline in-process against real frames: RF-DETR detection, UFLDv2
lanes, Depth Anything ranges, the frame ring, the router, reference resolution,
Qwen enrichment and the GPT-5.5 multimodal call. No HTTP, no microphone, no
speakers -- everything between "a frame arrived" and "here are the words" is
exercised, which is the part that can be wrong.

The three checks are the spec's acceptance tests 1-3:

  1  "What do you see?"                     scene frame + grounding, natural
                                            summary, not an object list
  2  "What kind of car is that on the ..."  right vehicle grounded, crop cut
                                            from the clearest frame, both
                                            images sent, honest uncertainty
  3  "What year is it?"                     the SAME referent reused, no
                                            re-identification, crop no worse

Some of what these tests assert is mechanical and some is a judgement about
English. The mechanical half is asserted; the judgement half is printed in full
so a person can read it, because "does this sound like a person looking out of
the window" is not a thing to fake with a regex.

--no-model runs everything except the GPT-5.5 call, for checking the grounding
half without spending tokens.
"""
import argparse
import glob
import json
import os
import sys
import time

sys.path.insert(0, "/workspace/rio-phase1")

from dotenv import load_dotenv                 # noqa: E402

load_dotenv("/workspace/rio-phase1/.env")

import config                                  # noqa: E402
import framebuf                                # noqa: E402
import router                                  # noqa: E402
import scene as scene_mod                      # noqa: E402
import visual_qa                               # noqa: E402
from headway import live as headway_live       # noqa: E402

SESSION = "selftest-visual"

_results = []


def check(name, condition, detail=""):
    _results.append((bool(condition), name, detail))
    mark = "PASS" if condition else "FAIL"
    print(f"    [{mark}] {name}" + (f"  -- {detail}" if detail else ""))


def load_frames(args, limit):
    """-> [jpeg_bytes]. Either decoded from a clip or read from a directory."""
    if args.frames:
        paths = sorted(glob.glob(os.path.join(args.frames, "*.jpg")))
        return [open(p, "rb").read() for p in paths[args.start:args.start + limit]]

    import cv2

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"could not open {args.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    # Sample at the cadence the live client actually sends (~4 fps), so the
    # motion history and the candidate age-out see the spacing they were tuned
    # for rather than a clip played at its own frame rate.
    step = max(1, int(round(fps * args.dt)))
    out, i = [], 0
    while len(out) < args.start + limit:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            ok2, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if ok2:
                out.append(buf.tobytes())
        i += 1
    cap.release()
    return out[args.start:]


class Drive:
    """A drive that keeps running between questions.

    The ring is trimmed by WALL clock, because on the road frames never stop
    arriving and six seconds ago means six seconds ago. A test that fed a batch
    and then spent twenty seconds talking to a model would find the buffer
    empty by the second question -- correctly, and uselessly. So this keeps the
    drive going: a few more frames before each turn, exactly as the 4 fps loop
    would have delivered them.
    """

    def __init__(self, frames, dt, v_host):
        headway_live.reset_session(SESSION)
        framebuf.drop_ring(SESSION)
        visual_qa.drop_session(SESSION)
        self.frames = frames
        self.dt = dt
        self.v_host = v_host
        self.i = 0
        self.session = headway_live.get_session(SESSION, use_qwen=config.VISION_ENABLED)
        self.ring = framebuf.get_ring(SESSION)
        self.last = None

    def feed(self, n, quiet=False):
        t0 = time.time()
        fed = 0
        while fed < n and self.i < len(self.frames):
            jpeg = self.frames[self.i]
            self.last = self.session.process(jpeg, self.v_host, 0.1,
                                             frame_t=self.i * self.dt)
            self.ring.push(jpeg, self.last)
            self.i += 1
            fed += 1
        if not quiet:
            per = (time.time() - t0) / max(1, fed) * 1000
            print(f"  fed {fed} frames ({per:.0f} ms/frame) — "
                  f"ring {json.dumps(self.ring.stats())}")
            if self.last:
                print(f"  band={self.last['band']} lead={self.last.get('lead_id')} "
                      f"corridor={self.last['corridor_source']} "
                      f"objects={len(self.last.get('scene_objects') or [])}")
        return fed


def warm() -> None:
    """Load every model off the measured path, as the server does at startup."""
    from headway import depth as depth_mod
    from headway import detect as detect_mod
    from headway import lanes as lanes_mod

    for name, fn in (("lanes", lanes_mod.warm), ("detector", detect_mod.warm),
                     ("depth", depth_mod.warm)):
        try:
            fn()
        except Exception as e:
            print(f"  !! {name} unavailable: {e}")
    if config.ENRICH_ENABLED:
        try:
            import vision

            vision.get_handles()
        except Exception as e:
            print(f"  !! Qwen unavailable, enrichment will be skipped: {e}")


def show_graph(graph: dict) -> None:
    print("\n  SCENE GRAPH")
    for o in graph.get("objects", []):
        print(f"    {o['track_id']:12s} {o['label']:10s} {o['position']:22s} "
              f"{'' if o.get('depth_meters') is None else str(o['depth_meters']) + ' m':>9s} "
              f"{o['motion']:18s} conf={o['confidence']}"
              + (f" fine={o['fine_label']}" if o.get("fine_label") else "")
              + (f" attrs={o['attributes']}" if o.get("attributes") else ""))


def run_turn(question: str, no_model: bool):
    """One visual turn. -> VisualAnswer (unsent when no_model)."""
    print(f"\n  DRIVER: {question!r}")
    va = visual_qa.answer(SESSION, question)
    if no_model:
        # Still finalise: the referent has to be established or the follow-up
        # test has nothing to follow up on.
        va.finish()
        va.meta["reply"] = "(--no-model: GPT-5.5 not called)"
        va.reply = ""
    else:
        va.text()
    print(f"  RIO:    {va.reply or va.meta.get('reply')}")
    t = va.meta.get("timing_ms", va.timing)
    print(f"  stages: {json.dumps(t)}")
    return va


# --- the technical vocabulary a driver must never hear ----------------------
LEAKS = ["track_id", "bounding box", "bounding_box", "confidence score",
         "vehicle_", "detection model", "qwen", "chatgpt", "gpt-", "rf-detr",
         "perception system", "crop", "pixel", "json", "depth_meters",
         "object id", "scene graph"]


def leaked(text: str):
    low = (text or "").lower()
    return [w for w in LEAKS if w in low]


def test_1_scene(no_model):
    print("\n" + "=" * 72)
    print("ACCEPTANCE 1 — \"What do you see?\"")
    print("=" * 72)
    va = run_turn("What do you see?", no_model)
    m = va.meta

    check("routed as scene_description",
          m["route"]["request_type"] == router.SCENE, m["route"]["request_type"])
    kinds = [i["kind"] for i in m.get("request", {}).get("images", [])]
    check("full frame sent to the model", "frame" in kinds, str(kinds))
    check("no crop sent for a whole-scene question", "crop" not in kinds, str(kinds))
    sel = m.get("frame_selection") or {}
    check("a frame was chosen from the ring by score",
          sel.get("reason") == "ok",
          f"chose {sel.get('chosen_frame_id')} of {sel.get('n_frames')}, "
          f"newest={sel.get('was_newest')}")
    check("scene graph accompanied the image",
          bool(m.get("request", {}).get("grounding_bytes")),
          f"{m.get('request', {}).get('grounding_bytes')} bytes")
    if not no_model:
        leaks = leaked(va.reply)
        check("no internal vocabulary in the reply", not leaks, str(leaks))
        check("reply is prose, not an enumeration",
              va.reply.count("\n") <= 1 and not va.reply.lstrip().startswith(("-", "*", "1.")),
              repr(va.reply[:60]))
        check("reply is not empty", bool(va.reply.strip()))
    return va


def test_2_object(question, no_model, expect_side=None):
    print("\n" + "=" * 72)
    print(f"ACCEPTANCE 2 — {question!r}")
    print("=" * 72)
    va = run_turn(question, no_model)
    m = va.meta

    check("routed as specific_object_question",
          m["route"]["request_type"] == router.OBJECT, m["route"]["request_type"])
    res = m.get("resolution") or {}
    check("reference resolved to a tracked object",
          bool(res.get("track_id")),
          f"{res.get('track_id')} via {res.get('method')} "
          f"conf={res.get('confidence')} ambiguous={res.get('ambiguous')}")
    ref = m.get("referent") or {}
    if expect_side and res.get("track_id"):
        # Read the position off the referent rather than re-querying the live
        # graph: the graph moves on, and the question is what this turn was
        # grounded to, not what is out there now.
        got_side = scene_mod.side_of(ref.get("position") or "")
        check(f"grounded object is on the {expect_side}",
              got_side == expect_side,
              f"{res.get('track_id')} at {ref.get('position')} "
              f"({ref.get('depth_meters')} m)")
    check("a crop was cut for the object",
          m.get("crop_source") in ("fresh", "referent_memory", "referent_memory_better"),
          str(m.get("crop_source")))
    kinds = [i["kind"] for i in m.get("request", {}).get("images", [])]
    check("model received BOTH the full frame and the crop",
          "frame" in kinds and "crop" in kinds, str(kinds))
    osel = m.get("object_frame_selection") or {}
    check("crop came from the clearest frame containing it, not just the newest",
          osel.get("reason") == "ok",
          f"chose {osel.get('chosen_frame_id')} of {osel.get('n_containing')} "
          f"containing it; newest={osel.get('was_newest')}")
    crop = m.get("crop") or {}
    print(f"    crop: object {crop.get('object_px')} px, output {crop.get('output_px')} px, "
          f"upscaled={crop.get('upscaled')}")
    if not no_model:
        leaks = leaked(va.reply)
        check("no internal vocabulary in the reply", not leaks, str(leaks))
        check("reply is not empty", bool(va.reply.strip()))
    return va


def test_3_follow_up(prev, no_model):
    print("\n" + "=" * 72)
    print("ACCEPTANCE 3 — \"What year is it?\" (follow-up)")
    print("=" * 72)
    before = (prev.meta.get("referent") or {}).get("track_id")
    before_crop = (prev.meta.get("crop") or {}).get("object_px")
    va = run_turn("What year is it?", no_model)
    m = va.meta

    check("routed as visual_follow_up",
          m["route"]["request_type"] == router.FOLLOW_UP, m["route"]["request_type"])
    check("reused the active referent instead of re-identifying",
          m.get("referent_source") in ("active", "active_reconfirmed"),
          str(m.get("referent_source")))
    check("no reference resolution ran", "resolution" not in m,
          "resolution absent" if "resolution" not in m else str(m["resolution"]))
    after = (m.get("referent") or {}).get("track_id")
    check("same object as the previous turn", after == before and after is not None,
          f"{before} -> {after}")
    crop = m.get("crop") or {}
    after_crop = crop.get("object_px")
    if before_crop and after_crop:
        area_b = before_crop[0] * before_crop[1]
        area_a = after_crop[0] * after_crop[1]
        check("crop supplied is the same or better",
              area_a >= area_b or m.get("crop_source") == "referent_memory_better",
              f"{before_crop} -> {after_crop} ({m.get('crop_source')})")
    kinds = [i["kind"] for i in m.get("request", {}).get("images", [])]
    check("model received a crop of the remembered object", "crop" in kinds, str(kinds))
    if not no_model:
        # Only meaningful when something was actually said: a turn with no
        # reply is correctly not written into the conversation history.
        check("prior turn was in the conversation history",
              m.get("request", {}).get("history_turns", 0) >= 1,
              f"{m.get('request', {}).get('history_turns')} turns")
        leaks = leaked(va.reply)
        check("no internal vocabulary in the reply", not leaks, str(leaks))
    return va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="/workspace/ufldv2/example.mp4")
    ap.add_argument("--frames", default=None,
                    help="directory of .jpg frames instead of a clip")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--count", type=int, default=24,
                    help="frames fed before the first question")
    ap.add_argument("--between", type=int, default=10,
                    help="frames fed between questions — the drive continues")
    ap.add_argument("--dt", type=float, default=0.25, help="seconds between frames")
    ap.add_argument("--speed", type=float, default=25.0, help="host speed, m/s")
    ap.add_argument("--question", default="What kind of car is that on the right?")
    ap.add_argument("--expect-side", default="right", choices=["left", "right", "ahead", ""])
    ap.add_argument("--no-model", action="store_true",
                    help="skip the GPT-5.5 call; check grounding only")
    args = ap.parse_args()

    print("RIO visual conversation — Phase A acceptance tests")
    print(f"  source: {args.frames or args.video}  from frame {args.start}")
    print(f"  models: RF-DETR + UFLDv2 + Depth Anything + "
          f"{'Qwen3-VL' if config.ENRICH_ENABLED else 'no enrichment'} + "
          f"{'(skipped)' if args.no_model else config.OPENAI_VISUAL_MODEL}")

    # Warm everything BEFORE anything is timed. A cold Qwen load is ~13 s and
    # would otherwise land inside a stage measurement and be reported as the
    # cost of enrichment, which it is not — the server warms all of this at
    # startup (app.py's lifespan).
    t0 = time.time()
    warm()
    print(f"  warm: {time.time() - t0:.1f}s")

    frames = load_frames(args, args.count + 3 * args.between + 8)
    if not frames:
        raise SystemExit("no frames loaded")
    drive = Drive(frames, args.dt, args.speed)
    drive.feed(args.count)
    show_graph(visual_qa.scene_graph(SESSION))

    test_1_scene(args.no_model)
    drive.feed(args.between, quiet=True)
    prev = test_2_object(args.question, args.no_model, args.expect_side or None)
    drive.feed(args.between, quiet=True)
    test_3_follow_up(prev, args.no_model)

    print("\n" + "=" * 72)
    passed = sum(1 for ok, _, _ in _results if ok)
    for ok, name, detail in _results:
        if not ok:
            print(f"  FAILED: {name}  -- {detail}")
    print(f"  {passed}/{len(_results)} checks passed")
    print("=" * 72)
    return 0 if passed == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
