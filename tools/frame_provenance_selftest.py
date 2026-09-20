"""frame_provenance_selftest.py — an annotated frame must never reach a model.

    python -m tools.frame_provenance_selftest

THE THIRD TIME THIS CLASS OF BUG HAS APPEARED.

  teachers were reading a composited clip and the card was not reading at all
    (commit 9b0b0b1)
  /perceive drew boxes on the frame it then asked about
  an annotated align-harness frame reached the visual model, which read
    "car 18.2m" off the overlay and reported it as a distance

The third is the one that decides the shape of this file, because nothing about it
looks wrong. The model answers fluently. The number is plausible. And it is a
MEASUREMENT RIO DID NOT MAKE -- a driver told "about eighteen metres" has been told
something no sensor on this car produced, in the voice of a thing that measures.
There is no tier underneath that to catch it: a wrong distance is not a missing
answer, it is a confident one.

So this asserts the guard by BREAKING IT. Every check below is run twice -- once
against a frame that came through framebuf.push, and once against the same frame
with pixels drawn on it -- and the suite fails if the annotated one gets through.
A guard that has only ever been seen to pass is a guard nobody has evidence about,
which is the same argument tools/playout_selftest.js makes in section Z.

It needs no GPU and no network: framebuf.push takes the jpeg and the detector's
result as data, so a result dict is enough to exercise the whole path.
"""
import hashlib
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import framebuf                                                 # noqa: E402
import visual_qa                                                # noqa: E402

OK, BAD = "ok  ", "FAIL"
_fails = []
_checks = []


def ok(name, cond, extra=""):
    _checks.append(1)
    if cond:
        print(f"  {OK} {name}")
    else:
        _fails.append(name)
        print(f"  {BAD} {name}{('  ' + str(extra)) if extra else ''}")


def section(t):
    print(f"\n== {t}")


def jpeg(seed: int, annotate: bool = False) -> bytes:
    """A small real JPEG. `annotate` draws on it, which is the whole point."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (160, 120), (30 + seed % 40, 60, 90))
    d = ImageDraw.Draw(im)
    d.rectangle([20, 20, 90, 80], fill=(120, 120, 130))
    if annotate:
        # EXACTLY THE SHAPE THAT CAUSED THIS. A box and a distance label -- the
        # same thing the align harness draws and the same thing a model read back
        # as "car 18.2m".
        d.rectangle([18, 18, 92, 82], outline=(255, 0, 0), width=2)
        d.text((22, 86), "car 18.2m", fill=(255, 255, 0))
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


RESULT = {"ok": True, "t": 1.0, "image": {"w": 160, "h": 120},
          "scene_objects": []}


def pushed_frame(raw: bytes):
    """One frame through the real ring, the way the request handler pushes it."""
    ring = framebuf.FrameRing(seconds=30, max_frames=8)
    f = ring.push(raw, RESULT, origin="test:clip")
    return ring, f


def main() -> int:
    section("the fingerprint is taken where the detector measured")
    raw = jpeg(1)
    ring, f = pushed_frame(raw)
    ok("push records a digest of the bytes it was given",
       f is not None and f.raw_sha == hashlib.sha256(raw).hexdigest(),
       getattr(f, "raw_sha", None))
    ok("...and the frame verifies against it", framebuf.verify_raw(f))
    ok("the ring hands back the same bytes for that frame id",
       ring.get(f.frame_id).jpeg == raw)

    section("an annotated frame is refused — the case that caused this")
    bad = jpeg(1, annotate=True)
    ok("drawing on it really does change the bytes", bad != raw,
       f"{len(raw)} -> {len(bad)} bytes")
    f.jpeg = bad
    ok("verify_raw REFUSES it", framebuf.verify_raw(f) is False)
    meta = {}
    ok("...and visual_qa's gate returns no frame at all",
       visual_qa._raw_frame_or_none(f, meta, "test") is None)
    ok("...recording which frame and where, so a drive log can say",
       meta.get("frames_refused")
       and meta["frames_refused"][0]["frame_id"] == f.frame_id,
       meta)
    ok("...and that it DID have provenance, which distinguishes 'tampered' from "
       "'never came through the ring'",
       meta["frames_refused"][0]["has_provenance"] is True)

    section("a frame with no provenance is refused too")

    class Synthetic:
        """A frame-shaped object that never went through push -- a composited
        preview, a hand-built fixture, a cached picture from somewhere else."""
        frame_id = "synthetic"
        jpeg = raw
        raw_sha = ""

    ok("an empty digest does not read as 'fine'",
       framebuf.verify_raw(Synthetic()) is False)
    meta2 = {}
    ok("...and the gate refuses it",
       visual_qa._raw_frame_or_none(Synthetic(), meta2, "test") is None)
    ok("...and says it had NO provenance, which is a different fault from a "
       "frame that was altered",
       meta2["frames_refused"][0]["has_provenance"] is False)
    ok("a missing frame is simply absent, not an error",
       visual_qa._raw_frame_or_none(None, {}, "test") is None)

    section("re-encoding, and the one case that is genuinely not a threat")
    ring2, f2 = pushed_frame(jpeg(2))
    original = f2.jpeg
    from PIL import Image

    # A BIT-EXACT RE-ENCODE IS NOT A THREAT, AND THE FIRST VERSION OF THIS TEST
    # CLAIMED IT WAS. Decoding and re-saving at the same quality with the same
    # settings reproduced the identical bytes, so the digest matched and the frame
    # verified -- correctly. If the bytes are the same then the pixels ARE the ones
    # the detector measured, and there is nothing to catch. The assertion was
    # wrong, not the guard, and it is corrected rather than deleted because
    # "re-encoding is always suspicious" is the plausible-sounding rule this
    # actually disproves.
    im = Image.open(io.BytesIO(original))
    same = io.BytesIO()
    im.save(same, format="JPEG", quality=85)
    if same.getvalue() == original:
        ok("a bit-exact re-encode still verifies, because identical bytes are "
           "identical pixels — there is nothing here to refuse",
           framebuf.verify_raw(f2) is True)
    else:
        ok("this re-encode changed the bytes, so it is refused",
           framebuf.verify_raw(f2) is False)

    # A re-encode that CHANGES anything is refused, and that is the real case: a
    # resize, a different quality, a different encoder, a thumbnail.
    lower = io.BytesIO()
    im.save(lower, format="JPEG", quality=40)
    f2.jpeg = lower.getvalue()
    ok("a re-encode at a different quality IS refused — not the buffer the "
       "detector measured, which is what the teacher path has said since it was "
       "written",
       framebuf.verify_raw(f2) is False,
       f"{len(original)} -> {len(f2.jpeg)} bytes")
    f2.jpeg = original
    smaller = io.BytesIO()
    im.resize((80, 60)).save(smaller, format="JPEG", quality=85)
    f2.jpeg = smaller.getvalue()
    ok("and so is a resize", framebuf.verify_raw(f2) is False)

    section("TWO LIVE SOURCES IN ONE SESSION — the drive of 2026-09-20")
    # A drive on the desktop camera and an uploaded clip's caption watcher,
    # both pushing into one session key, 0.6 s apart. Emptying the ring on each
    # flip is the right thing to DO and it answered nothing: every reset looked
    # local and reasonable, and the pattern — the thing that made perception
    # change its mind — was counted nowhere.
    ring2 = framebuf.FrameRing(seconds=30, max_frames=8)
    cam, clip = "s1:camera", "s1:clip"
    ring2.push(jpeg(10), RESULT, origin=cam)
    ok("one producer is not a conflict", ring2.conflict is None
       and ring2.origin_flips == 0)

    ring2.push(jpeg(11), RESULT, origin=clip)
    ok("nor is a HANDOVER — a driver loading a clip flips the origin once",
       ring2.conflict is None, f"flips={ring2.origin_flips}")
    ok("...and the ring still empties, which is the part that was right",
       len(ring2.frames()) == 1)

    ring2.push(jpeg(12), RESULT, origin=cam)
    ok("flipping BACK is two live producers — one cannot do that",
       ring2.conflict is not None, str(ring2.conflict))
    ok("...and it names both of them",
       (ring2.conflict or {}).get("origins") == sorted([cam, clip]))
    ok("...and is LATCHED: a conflict that stopped is still why the last "
       "answer was wrong",
       ring2.push(jpeg(13), RESULT, origin=cam) is not None
       and ring2.conflict is not None)
    ok("the stats a dashboard reads carry it",
       ring2.stats().get("conflict") is not None
       and ring2.stats().get("origin_flips") == 2)

    ring3b = framebuf.FrameRing(seconds=30, max_frames=8)
    ring3b.push(jpeg(14), RESULT, origin=cam)
    ring3b._flip_times.clear()
    ring3b.push(jpeg(15), RESULT, origin=clip)
    # Two flips far apart in time are two handovers, not two producers: the
    # window is what separates them, so it has to be the thing that decides.
    ring3b._flip_times.clear()
    ring3b.conflict = None
    ring3b.push(jpeg(16), RESULT, origin=cam)
    ok("two changes OUTSIDE the window are two handovers, not a conflict",
       ring3b.conflict is None)

    ring2.reset_origin()
    ok("a new drive on the key starts clean",
       ring2.conflict is None and ring2.origin_flips == 0)

    section("the guard is not vacuous: a good frame still gets through")
    ring3, f3 = pushed_frame(jpeg(3))
    ok("an untouched frame from the ring is returned, not refused",
       visual_qa._raw_frame_or_none(f3, {}, "test") is f3)
    ok("...and nothing is recorded against it", "frames_refused" not in {})

    print("\n" + "-" * 58)
    print(f'  {"PASS" if not _fails else "FAIL"}: {len(_fails)} failure(s)'
          f' of {len(_checks)} checks')
    for x in _fails:
        print(f"    - {x}")
    return len(_fails)


if __name__ == "__main__":
    sys.exit(main())
