"""eyewindow.py — the eye reads a stretch of road, not a snapshot of one.

WHY THIS EXISTS
---------------
Cosmos-Reason2 is a video model. NVIDIA's own tooling never hands it a still:
every example in their repo is `--videos <clip> --fps 4`, and their serving
recipe passes `--media-io-kwargs '{"video": {"num_frames": -1}}'` precisely so
the caller can choose the rate. We were handing it one JPEG.

A still cannot show motion, and almost everything that matters on a road IS
motion -- closing, drifting, braking, cutting in. Asked from one frame, the
model had no way to know a gap was shrinking, so it did the only thing it
could: it guessed, and a guess dressed as an observation is the fault this
whole project keeps having to catch. The readings of 2026-09-21 invented a
pickup truck that was not there and a rule about overtaking on the median that
answered a question nobody asked.

WHAT A WINDOW IS
----------------
The last `seconds` of the session's ring, decoded, as one video. Not a frame,
not a hand-assembled bundle of frames that happens to be passed as a list: the
processor is told the fps and the duration, so Qwen3-VL's own temporal
encoding runs and the prompt carries real timestamps -- `<0.2 seconds>`,
`<1.2 seconds>` -- against which a reading about something two seconds back can
be checked. That is the property the grounding block depends on, and it comes
free only if the metadata is real.

PROVENANCE IS CHECKED PER FRAME, NOT PER WINDOW
-----------------------------------------------
framebuf.verify_raw is run on every frame that goes in. A window is a stronger
version of the same promise a single frame made -- these pixels are the pixels
the detector measured -- and it is a weaker promise if one frame in twenty-four
was re-encoded or drawn on. A frame that fails is dropped and counted, because
a model that reads "car 18.7m" off a burned-in overlay and reports it as a
distance is exactly the fault raw_sha was added for.

WHY NOT ENCODE AN MP4
---------------------
Because there is nothing to encode it for. The ring holds JPEGs; the processor
wants pixel arrays; an mp4 in between is a lossy round-trip through a codec
whose artefacts the model would then be asked to interpret. The uploaded-clip
path is different and is handled by `from_clip` below -- there the clip already
IS a video and is decoded as one, in order, rather than sampled into stills.
"""
import time

import numpy as np

import config
import framebuf

# THE RATE NVIDIA USES. Every video example in cosmos-reason2's README passes
# --fps 4, and their temporal_localization sample logs VisionConfig(fps=4.0).
# Ours is a config knob so a drive can be measured at another rate, but the
# default is theirs and changing it is a decision somebody should have to make.
DEFAULT_FPS = float(getattr(config, "EYE_WINDOW_FPS", 4.0))

# HOW MUCH ROAD. The ring is six seconds (config.RING_SECONDS) and that is the
# ceiling, not a coincidence: it was chosen so a question asked a beat late can
# still find the frame where the thing was biggest. Six seconds at 13 m/s is
# about eighty metres of road, which is long enough for a gap to visibly close
# and short enough that the near end is still now.
DEFAULT_SECONDS = float(getattr(config, "EYE_WINDOW_S", 6.0))

# THE PIXEL BUDGET FOR THE WHOLE WINDOW, not per frame -- NVIDIA's own number,
# from the VisionConfig in assets/outputs/temporal_localization.log. The
# processor spreads it across however many frames it is given, so a longer
# window is automatically a lower-resolution one. That trade is the right way
# round for a road: motion survives downscaling, and a window that blew the
# context budget would not be read at all.
TOTAL_PIXELS = int(getattr(config, "EYE_WINDOW_TOTAL_PIXELS", 3774873))

# A window with fewer frames than this is not a video. The processor's own
# floor is 4 (Qwen3VLVideoProcessor.min_frames); below it the temporal
# encoding has nothing to encode and the reading is a still in disguise, which
# is the thing this module exists to stop happening by accident.
MIN_FRAMES = 4

# BELOW THIS, THE PICTURE IS NOT CHANGING. Mean absolute luminance difference
# between sampled frames, on an 8x-decimated grey copy. A road at 13 m/s moves
# every pixel in the lower half of the frame between one sample and the next
# and sits one to two orders of magnitude above this; a paused clip, a frozen
# transport or a parked car sits under it. Used to LABEL a window, never to
# refuse one -- "the view has not changed" is a reading, not an error.
STATIC_MOTION_FLOOR = float(getattr(config, "EYE_WINDOW_STATIC_FLOOR", 0.8))


class Window:
    """A stretch of road, ready to hand to the model.

    `frames` are RGB uint8 HxWx3, oldest first. `t_rel` gives each frame's
    offset in seconds from the start of the window, which is the clock the
    model's own `<n seconds>` markers are on -- so a reading that says "at
    around two seconds" can be laid against the tracks without a second
    ruler being invented for it.
    """

    def __init__(self, frames, t_rel, ring_frames, fps, dropped_unverified=0,
                 source="ring"):
        self.frames = frames
        self.t_rel = t_rel
        self.ring_frames = ring_frames
        self.fps = float(fps)
        self.dropped_unverified = int(dropped_unverified)
        self.source = source

    def __len__(self):
        return len(self.frames)

    @property
    def span_s(self) -> float:
        """Wall-clock length of the stretch, first frame to last."""
        if len(self.t_rel) < 2:
            return 0.0
        return float(self.t_rel[-1] - self.t_rel[0])

    @property
    def end_wall_t(self) -> float:
        """When the LAST frame was taken. The window's 'now'."""
        if not self.ring_frames:
            return 0.0
        return float(getattr(self.ring_frames[-1], "wall_t", 0.0) or 0.0)

    @property
    def end_age_s(self) -> float:
        """How stale the newest frame in the window is."""
        end = self.end_wall_t
        return (time.time() - end) if end else float("inf")

    def metadata(self) -> list:
        """What the processor needs to run Qwen3-VL's temporal encoding.

        WITHOUT THIS THE WINDOW IS SILENTLY GUTTED. The video processor
        defaults to do_sample_frames=True at fps=2 and, given no metadata,
        assumes the input was shot at 24 fps -- so a 24-frame window is
        resampled down to 2 temporal groups and twenty of the frames are
        thrown away before the model sees them. Measured, not inferred: the
        same 24 frames produce grid_thw t=6 with metadata and t=2 without.
        """
        return [{
            "fps": self.fps,
            "duration": max(self.span_s, len(self.frames) / max(self.fps, 1e-6)),
            "total_num_frames": len(self.frames),
            "frames_indices": list(range(len(self.frames))),
        }]

    def structure(self):
        """How much is IN this window, and how much of it MOVES. -> (std, motion).

        Both are pixel statistics, not model output, which is what makes them
        admissible as measured state: a covered lens and a frozen transport are
        facts about the frames and neither needs a model to notice.

        Sampled at three points rather than computed over twenty-six frames.
        The question is "is this window empty" and "is this window still", and
        neither needs every frame to answer -- the cost of asking all of them
        would land on a path that runs for the length of a drive.
        """
        if not self.frames:
            return 0.0, 0.0
        idx = sorted({0, len(self.frames) // 2, len(self.frames) - 1})
        picks = [self.frames[i] for i in idx]
        grey = [f[::8, ::8, :].mean(axis=2) for f in picks]
        std = float(np.mean([g.std() for g in grey]))
        if len(grey) < 2:
            return std, 0.0
        motion = float(np.mean([np.abs(grey[i + 1] - grey[i]).mean()
                                for i in range(len(grey) - 1)]))
        return std, motion

    def to_meta(self) -> dict:
        """Everything about the window except the pixels. Safe to log and show.

        THE AGE FIELD BECOMES A SPAN. A single frame had one age; a window has
        a beginning and an end, and a card that showed only one of them would
        be describing either a moment that has passed or a stretch of road
        whose length nobody can see.
        """
        std, motion = self.structure()
        return {
            "source": self.source,
            "n_frames": len(self.frames),
            "fps": round(self.fps, 2),
            "span_s": round(self.span_s, 2),
            "end_age_s": round(self.end_age_s, 2),
            "start_age_s": round(self.end_age_s + self.span_s, 2),
            "frame_ids": [getattr(f, "frame_id", None) for f in self.ring_frames],
            "dropped_unverified": self.dropped_unverified,
            # NOTHING IN IT, and NOTHING MOVING IN IT. A blank window must
            # still be READ -- a reading that says the view is unusable is the
            # right answer and refusing before the pass makes that untestable
            # -- so the fact travels with the window and the judgement is made
            # on what the model said. See eyeread._ROAD_WORDS.
            "blank": bool(std < float(getattr(config, "LOCAL_VISION_BLANK_STD", 6.0))),
            "structure_std": round(std, 2),
            "motion": round(motion, 3),
            "static": bool(motion < STATIC_MOTION_FLOOR),
            "image": ({"w": self.frames[0].shape[1], "h": self.frames[0].shape[0]}
                      if self.frames else None),
        }


def _decode_rgb(ring_frame):
    """RingFrame -> RGB ndarray, or None if it will not decode."""
    import cv2
    bgr = ring_frame.frame_bgr()
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def from_ring(session_key: str, seconds: float = None, fps: float = None,
              min_frames: int = MIN_FRAMES):
    """The last `seconds` of this session's road, as a video. -> Window|None.

    Returns None rather than a short window when there is not enough road to
    be a video: a two-frame "window" would be read as a still, and a still is
    what this module exists to stop the eye being given.
    """
    seconds = DEFAULT_SECONDS if seconds is None else float(seconds)
    fps = DEFAULT_FPS if fps is None else float(fps)
    ring = framebuf.peek_ring(session_key)
    if ring is None:
        return None
    recent = ring.frames(max_age_s=seconds)
    if not recent:
        return None

    # PROVENANCE FIRST, DECODE SECOND. Checking the digest costs a hash of
    # bytes already in RAM; decoding costs a JPEG pass. Dropping an unverified
    # frame before it is decoded is free, and the order matters on a path that
    # runs every few seconds for the length of a drive.
    kept, dropped = [], 0
    for rf in recent:
        if not framebuf.verify_raw(rf):
            dropped += 1
            continue
        kept.append(rf)
    if len(kept) < min_frames:
        return None

    # THIN TO THE TARGET RATE HERE, rather than handing over everything and
    # letting the processor resample. The detector feeds at whatever rate the
    # page manages -- measured between 2 and 15 fps on this pod -- so "every
    # frame in the ring" is a different video every time and a window whose
    # rate moved with the transport would make two readings incomparable.
    # Picking frames by time gives the same six seconds at the same rate
    # whether the feed was fast or slow.
    picked = _thin_to_fps(kept, fps)
    if len(picked) < min_frames:
        picked = kept[-max(min_frames, len(kept)):]

    frames, ring_frames, t_rel = [], [], []
    t0 = float(getattr(picked[0], "wall_t", 0.0) or 0.0)
    for rf in picked:
        rgb = _decode_rgb(rf)
        if rgb is None:
            continue
        frames.append(rgb)
        ring_frames.append(rf)
        t_rel.append(float(getattr(rf, "wall_t", t0) or t0) - t0)
    if len(frames) < min_frames:
        return None
    return Window(frames, t_rel, ring_frames, fps, dropped, source="ring")


def _thin_to_fps(ring_frames, fps):
    """Pick the frames nearest each 1/fps tick across the span. -> list.

    Nearest rather than every-nth: the feed is not isochronous, and every-nth
    on a feed that stuttered gives a window whose real rate is not the rate the
    metadata claims -- which would put the model's timestamps somewhere other
    than where the road was.
    """
    if fps <= 0 or len(ring_frames) < 2:
        return list(ring_frames)
    t0 = float(getattr(ring_frames[0], "wall_t", 0.0) or 0.0)
    t1 = float(getattr(ring_frames[-1], "wall_t", 0.0) or 0.0)
    span = max(t1 - t0, 1e-6)
    n = max(2, int(round(span * fps)) + 1)
    step = span / max(n - 1, 1)
    out, used = [], set()
    for i in range(n):
        target = t0 + i * step
        best, best_d = None, None
        for rf in ring_frames:
            if id(rf) in used:
                continue
            d = abs(float(getattr(rf, "wall_t", 0.0) or 0.0) - target)
            if best_d is None or d < best_d:
                best, best_d = rf, d
        if best is not None:
            used.add(id(best))
            out.append(best)
    out.sort(key=lambda r: float(getattr(r, "wall_t", 0.0) or 0.0))
    return out


def from_clip(path: str, at_s: float = None, seconds: float = None,
              fps: float = None):
    """A stretch of an uploaded clip, decoded as a clip. -> Window|None.

    THE CLIP IS NOT SAMPLED INTO STILLS. It is opened once and read forward
    across the window, which is what makes the frames consecutive road rather
    than a sequence of independent seeks -- and a seek-per-frame on a clip
    whose keyframes are sparse returns the same picture repeatedly, a fault
    this repo has already measured (tools/vision_ab.clip_frame).

    `at_s` is the END of the window, defaulting to the end of the clip, so a
    window is always the road up TO a moment rather than after it.
    """
    import cv2
    seconds = DEFAULT_SECONDS if seconds is None else float(seconds)
    fps = DEFAULT_FPS if fps is None else float(fps)
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return None
    try:
        src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        dur = (n_total / src_fps) if n_total else 0.0
        end = dur if at_s is None else float(at_s)
        start = max(0.0, end - seconds)
        step = max(1, int(round(src_fps / max(fps, 1e-6))))
        first_idx = int(round(start * src_fps))
        cap.set(cv2.CAP_PROP_POS_FRAMES, first_idx)
        frames, t_rel = [], []
        idx = first_idx
        want = idx
        last_idx = int(round(end * src_fps))
        while idx <= last_idx:
            ok, bgr = cap.read()
            if not ok:
                break
            if idx >= want:
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                t_rel.append((idx - first_idx) / src_fps)
                want = idx + step
            idx += 1
        if len(frames) < MIN_FRAMES:
            return None
        return Window(frames, t_rel, [], fps, 0, source="clip")
    finally:
        cap.release()


def processor_kwargs() -> dict:
    """What the image/video processor is told, NVIDIA's values.

    `size` here is the WHOLE WINDOW's pixel budget -- longest_edge is read by
    Qwen3-VL's video processor as total pixels across every frame, which is
    why a longer window downscales itself instead of overflowing the context.
    """
    return {
        "fps": DEFAULT_FPS,
        "do_sample_frames": False,
        "size": {"shortest_edge": 4096, "longest_edge": TOTAL_PIXELS},
    }
