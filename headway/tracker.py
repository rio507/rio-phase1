"""OpenCV CSRT ROI tracker wrapper — holds the lead box between Qwen anchors.

Design ref: headway_design.md §0 Challenge 1, §1 (fast loop step 1), §2, §7.4.

This is the component the original plan was missing. Qwen runs at 0.2-0.5 Hz;
a lead closing at 20 mph relative moves ~36 m between inferences. CSRT is a
classical correlation-filter tracker (no neural net, no GPU), so it can hold the
box at 10-15 Hz between re-anchors and give Depth Anything something to sample.

CSRT specifically, not KCF: CSRT handles scale change much better, and scale
change is the dominant motion here -- a lead vehicle's box grows/shrinks as the
gap closes, which is the whole signal.
"""
import numpy as np

try:
    import cv2
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "headway.tracker needs opencv-contrib-python-headless (<5 for the "
        "stable CSRT API): pip install 'opencv-contrib-python-headless<5'"
    ) from e


def _make_csrt():
    """CSRT moved namespaces across OpenCV versions; try each known home."""
    for factory in (
        lambda: cv2.TrackerCSRT_create(),
        lambda: cv2.legacy.TrackerCSRT_create(),
        lambda: cv2.TrackerCSRT.create(),
    ):
        try:
            return factory()
        except AttributeError:
            continue
    raise RuntimeError(
        "No CSRT implementation found. Install opencv-contrib (not plain "
        "opencv-python), which is where the tracking module lives."
    )


class LeadTracker:
    """Wraps one CSRT instance and scores how much to believe it.

    OpenCV's tracker only tells you success/failure, which is far too coarse for
    §7.4 (tracker drifting onto an adjacent car). It reports success right up
    until it is confidently tracking the wrong vehicle. So `quality` is derived
    from motion plausibility instead: a real lead vehicle's box changes size and
    position smoothly, and a drift/jump event violates that.
    """

    def __init__(self, max_area_ratio_per_s: float = 8.0, max_shift_frac_per_s: float = 3.0,
                 max_aspect_ratio_change: float = 0.35):
        # Physical plausibility budgets, per second so they are frame-rate
        # independent. These are deliberately generous: a lead closing at 9 m/s
        # from 19 m changes box area ~2.7x/s, and that is perfectly legitimate
        # motion. The budget exists to catch the tracker *jumping to another
        # object*, which is a discontinuity an order of magnitude larger, not to
        # penalise fast approach.
        self.max_area_ratio_per_s = max_area_ratio_per_s
        self.max_shift_frac_per_s = max_shift_frac_per_s
        self.max_aspect_ratio_change = max_aspect_ratio_change

        self._tracker = None
        self._box = None
        self._quality = 0.0
        self._age = 0

    @property
    def box(self):
        return self._box

    @property
    def quality(self) -> float:
        return self._quality

    @property
    def age(self) -> int:
        """Frames since the last init() -- feeds the staleness term in §4 confidence."""
        return self._age

    @property
    def active(self) -> bool:
        return self._tracker is not None

    def init(self, frame: np.ndarray, box) -> None:
        """(Re-)anchor on an [x1,y1,x2,y2] box. Called by anchor.py every N frames."""
        x1, y1, x2, y2 = [float(v) for v in box]
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"degenerate box {box}")

        self._tracker = _make_csrt()
        # OpenCV wants (x, y, w, h); the rest of this module speaks corners.
        self._tracker.init(frame, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        self._box = (x1, y1, x2, y2)
        # A fresh anchor is trusted, but not blindly -- Qwen may have proposed a
        # bad box, and only the corridor check in anchor.py has vetted it.
        self._quality = 1.0
        self._age = 0

    def update(self, frame: np.ndarray, dt: float = 1.0 / 15.0):
        """Advance one frame. Returns (box, quality); box is None once tracking fails."""
        if self._tracker is None:
            return None, 0.0

        ok, xywh = self._tracker.update(frame)
        self._age += 1
        if not ok:
            self._quality = 0.0
            self._box = None
            return None, 0.0

        x, y, w, h = xywh
        new_box = (float(x), float(y), float(x + w), float(y + h))

        quality = self._score(self._box, new_box, dt) if self._box is not None else 1.0
        self._box = new_box
        # EMA rather than instantaneous: one noisy frame shouldn't zero the
        # score, but sustained implausible motion should pull it down fast.
        self._quality = 0.7 * self._quality + 0.3 * quality
        return new_box, float(np.clip(self._quality, 0.0, 1.0))

    @staticmethod
    def _anomaly_score(excess: float, budget: float) -> float:
        """1.0 while within budget, then decays exponentially past it.

        A plain linear ramp (the obvious first cut) scores *every* frame below
        1.0, so ordinary tracker jitter on a perfectly good lock accumulates into
        a low quality and trips the §4 DEGRADED floor -- which is what the first
        version of this file actually did on the synthetic clip. What is wanted
        is a discriminator, not a penalty: normal motion should be
        indistinguishable from perfect, and only a genuine discontinuity should
        collapse the score.
        """
        budget = max(float(budget), 1e-9)
        if excess <= budget:
            return 1.0
        return float(np.exp(-(excess - budget) / budget))

    def _score(self, old, new, dt: float) -> float:
        """Plausibility of the old->new box transition, 0-1."""
        dt = max(dt, 1e-3)
        ox1, oy1, ox2, oy2 = old
        nx1, ny1, nx2, ny2 = new

        old_area = max((ox2 - ox1) * (oy2 - oy1), 1e-6)
        new_area = max((nx2 - nx1) * (ny2 - ny1), 1e-6)

        # Symmetric so growing and shrinking are treated the same way.
        ratio = max(new_area / old_area, old_area / new_area)
        area_score = self._anomaly_score(
            ratio - 1.0, (self.max_area_ratio_per_s - 1.0) * dt)

        old_w = max(ox2 - ox1, 1e-6)
        old_h = max(oy2 - oy1, 1e-6)
        # Centre shift measured in box-widths: scale-free, so a distant (small)
        # box is held to the same standard as a near one.
        dx = abs((nx1 + nx2) / 2 - (ox1 + ox2) / 2) / old_w
        dy = abs((ny1 + ny2) / 2 - (oy1 + oy2) / 2) / old_h
        shift = float(np.hypot(dx, dy))
        shift_score = self._anomaly_score(shift, self.max_shift_frac_per_s * dt)

        # Aspect ratio of a given vehicle is near-constant; a sudden change means
        # the filter latched onto something else (§7.4). Not time-scaled -- a
        # vehicle's proportions shouldn't drift no matter how long you watch.
        old_ar = old_w / old_h
        new_ar = max(nx2 - nx1, 1e-6) / max(ny2 - ny1, 1e-6)
        ar_ratio = max(new_ar / old_ar, old_ar / new_ar)
        ar_score = self._anomaly_score(ar_ratio - 1.0, self.max_aspect_ratio_change)

        return float(area_score * shift_score * ar_score)

    def reset(self) -> None:
        self._tracker = None
        self._box = None
        self._quality = 0.0
        self._age = 0
