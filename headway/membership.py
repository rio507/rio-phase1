"""In-lane membership — which vehicles are in MY lane, and for how long.

Design ref: docs/live_headway_v3.md §2, extending anchor.py's corridor gate now
that UFLDv2 supplies a real ego-lane polygon rather than an assumed trapezoid.

WHAT THIS REPLACES
------------------
The corridor gate used to ask one question of one pixel: is the box's
bottom-CENTRE inside the corridor? That is cheap and wrong at the edges:

  - a wide truck straddling the line has its centre in the adjacent lane while
    half its body is in ours -- read as "not my lane", and dropped
  - a motorcycle hugging the lane edge has its centre a hand's width outside --
    read as "not my lane", and dropped
  - a car easing across the line flickers in and out frame by frame as the
    centre crosses back and forth, so the tracker never holds a lock

Membership is now the FRACTION of the box's bottom edge that lies inside the
lane, with hysteresis on that fraction and a dwell time on top. Three separate
mechanisms, because they defend against three different failures: the fraction
fixes geometry, the hysteresis fixes flicker, the dwell fixes transients.

THE FIREWALL STILL HOLDS
------------------------
Nothing here consults a model. Qwen supplies boxes and labels; every decision
below -- membership, eligibility, merge promotion, which candidate becomes the
lead -- is arithmetic on those boxes against the lane polygon. §1 unchanged.
"""
import math
from collections import deque

# --- membership, as a fraction of the box-bottom edge inside the lane -------
# Asymmetric on purpose. A candidate must be substantially in the lane to be
# adopted (0.40) but only marginally in it to be kept (0.25). A single
# threshold makes a lead drifting along the line flicker in and out of
# membership, and each flicker is a lost lock and a re-anchor.
MEMBER_ENTER_FRAC = 0.40
MEMBER_EXIT_FRAC = 0.25

# Continuous membership before a candidate may become the lead. Oncoming
# traffic swinging through the lane polygon on a bend, and adjacent cars
# clipped by a momentary lane-detection wobble, both satisfy the fraction test
# for a frame or two and neither should take the lock.
MEMBER_HOLD_S = 0.5

# --- merge-in promotion -----------------------------------------------------
# The deliberate exception. Everything above is built to make the lead lock
# STICKY; a car merging into our lane is the one case where stickiness costs
# reaction time, because by the time it satisfies MEMBER_HOLD_S it is already
# there. So a vehicle whose overlap is climbing steadily, close enough ahead to
# matter, is promoted to lead-eligible before it qualifies the slow way.
MERGE_WINDOW_S = 0.75        # trend measured over at least this long
MERGE_MIN_SLOPE = 0.25       # overlap fraction gained per second
MERGE_MIN_OVERLAP = 0.15     # ...and it must actually be touching our lane
MERGE_MAX_RANGE_M = 40.0     # ...and be close enough for the answer to matter
MERGE_MIN_SAMPLES = 3

# --- candidate association ---------------------------------------------------
# Matching this frame's detections to the candidates we already hold.
#
# IoU ALONE DOES NOT WORK AT THIS FRAME RATE. The obvious choice -- and what
# SORT/ByteTrack use -- is IoU overlap, but those run at 25-30 fps where a
# vehicle barely moves between frames. This loop runs at 4 fps, and measured on
# the winding clip a correctly-tracked motorcycle scores IoU 0.10-0.30 against
# its own previous box because the camera is turning. At MATCH_IOU=0.30 that
# minted a fresh candidate almost every frame: 10 live candidates for 2 real
# vehicles, and no candidate ever survived long enough to earn its dwell.
#
# So association is by CENTRE DISPLACEMENT measured in box-diagonals, which is
# scale-free and tolerant of the large per-frame motion 4 fps implies, with IoU
# kept only as a fast accept. The label must match exactly: on the same clip a
# motorcycle and its rider sit 0.29 diagonals apart, so distance alone would
# happily swap them.
MATCH_IOU = 0.30             # IoU this high is the same object, no question
MATCH_MAX_CENTRE_FRAC = 1.0  # ...otherwise, centre may move up to one diagonal
MATCH_MAX_AREA_RATIO = 3.0   # ...and may not change size by more than this

# Age-out, measured from the last DETECTION.
#
# This was 6.0 s when candidates came from a 5 s Qwen enumeration and were
# carried between enumerations by MOSSE micro-trackers, and even then it was
# generous. RF-DETR detects on EVERY frame, so a candidate unseen for four
# consecutive frames at 4 fps is occluded, out of shot, or was never real.
# Holding stale boxes for six seconds was a workaround for how rarely we could
# afford to look, and looking is now cheap.
CANDIDATE_MAX_UNDETECTED_S = 1.0
DEPTH_MEDIAN_N = 3           # per-candidate range smoothing (see range_m)

# How many VALID range samples a candidate needs before it may hold the lead.
#
# This closes a hole that has nothing to do with how fast a range changes.
# Measured on the winding clip: three consecutive frames had their depth refused
# as glare-flared, which correctly emptied every candidate's sample window. On
# the next frame -- still glare-corrupted, but below the refusal threshold -- a
# candidate's history was [None, None, 5.1] and it took the lead lock on that
# ONE uncorroborated sample, at 5.1 m for a real 30 m gap. A lead switch resets
# the Kalman, so the filter's outlier gate was discarded at exactly the moment
# the measurement was an outlier.
#
# A RATE bound was tried first and does not work: the bad jump (30 m -> 5 m in
# 0.24 s, 107 m/s) sits inside the legitimate distribution, whose 99.5th
# percentile is 78-84 m/s and whose max is 95-107 m/s, because ROI depth on a
# small distant box genuinely swings that hard. Corroboration separates cleanly
# where rate cannot.
#
# It costs nothing on any legitimate path, which is why 2 is enough: MEMBER_HOLD_S
# already requires 0.5 s (two frames at 4 fps) of membership, and merge promotion
# already requires three overlap samples over 0.75 s. Both are longer than this.
# What it forbids is precisely the case above -- taking the lock on the first
# reading after a blind spell.
RANGE_MIN_SAMPLES = 2
HISTORY_S = 2.0              # overlap history kept per candidate

# Only a vehicle can be a lead. Mirrors anchor.LEAD_LABELS; imported from there
# so the two cannot drift.
try:
    from .anchor import LEAD_LABELS
except ImportError:                                   # pragma: no cover
    LEAD_LABELS = {"car", "truck", "bus", "van", "motorcycle"}


def bottom_edge_overlap(box, corridor):
    """Fraction of a box's bottom edge lying inside the ego lane. -> (frac, info).

    The bottom edge is where the vehicle meets the road, so it is the only part
    of the box whose horizontal extent means anything in lane terms -- the roof
    of a lorry overhangs, the bottom edge does not. Both corridors expose
    `bounds_at_row`, so this is one interval intersection either way.
    """
    x1, _, x2, y2 = [float(v) for v in box]
    if x2 <= x1:
        return 0.0, {"reason": "degenerate_box"}
    if corridor is None:
        return 0.0, {"reason": "no_corridor"}

    bounds = corridor.bounds_at_row(y2)
    if bounds is None:
        # Above where the lane was detected, or above the horizon. Not "outside
        # the lane" -- unknown. The caller must not read 0.0 as evidence.
        return 0.0, {"reason": "no_lane_at_row"}

    xl, xr = bounds
    inside = max(0.0, min(x2, xr) - max(x1, xl))
    frac = inside / (x2 - x1)
    return frac, {
        "reason": "ok",
        "lane_px": [round(xl, 1), round(xr, 1)],
        "edge_px": [round(x1, 1), round(x2, 1)],
        "inside_px": round(inside, 1),
    }


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return float(inter / ua) if ua > 0 else 0.0


def match_cost(box, label, cand):
    """Association cost between a detection and a candidate. -> cost or None.

    None means "cannot be the same object". Lower is better; IoU >= MATCH_IOU
    short-circuits to 0.0.
    """
    if label is not None and cand.label is not None and label != cand.label:
        return None
    if iou(box, cand.box) >= MATCH_IOU:
        return 0.0

    ax1, ay1, ax2, ay2 = box
    bx1, by1, bx2, by2 = cand.box
    aw, ah = max(ax2 - ax1, 1e-6), max(ay2 - ay1, 1e-6)
    bw, bh = max(bx2 - bx1, 1e-6), max(by2 - by1, 1e-6)

    area_ratio = max((aw * ah) / (bw * bh), (bw * bh) / (aw * ah))
    if area_ratio > MATCH_MAX_AREA_RATIO:
        return None

    dx = (ax1 + ax2) / 2.0 - (bx1 + bx2) / 2.0
    dy = (ay1 + ay2) / 2.0 - (by1 + by2) / 2.0
    diag = (math.hypot(aw, ah) + math.hypot(bw, bh)) / 2.0
    frac = math.hypot(dx, dy) / max(diag, 1e-6)
    return frac if frac <= MATCH_MAX_CENTRE_FRAC else None


def _slope(samples):
    """Least-squares slope of overlap against time, per second.

    Least squares rather than (last - first) / dt: a single noisy end sample
    would otherwise decide whether a merge is declared.
    """
    n = len(samples)
    if n < 2:
        return 0.0
    mt = sum(t for t, _ in samples) / n
    mv = sum(v for _, v in samples) / n
    num = sum((t - mt) * (v - mv) for t, v in samples)
    den = sum((t - mt) ** 2 for t, _ in samples)
    return float(num / den) if den > 1e-9 else 0.0


class Candidate:
    """One tracked vehicle and its membership history."""

    __slots__ = ("id", "label", "box", "first_seen", "last_seen", "last_detected",
                 "overlap", "history", "member", "member_since", "merge_promoted",
                 "merge_t", "merge_slope", "depths", "tracker", "lost",
                 "overlap_reason", "score", "quality", "n_detections",
                 "confirmed", "vel_px", "range_reject")

    def __init__(self, cid, box, label, t):
        self.id = cid
        self.label = label
        self.box = tuple(float(v) for v in box)
        self.first_seen = float(t)
        self.last_seen = float(t)
        self.last_detected = float(t)
        self.overlap = 0.0
        self.overlap_reason = "new"
        self.history = deque()          # (t, overlap)
        self.member = False
        self.member_since = None
        self.merge_promoted = False
        self.merge_t = None
        self.merge_slope = 0.0
        self.depths = deque(maxlen=DEPTH_MEDIAN_N)
        self.tracker = None
        self.lost = False
        self.score = 0.0            # detector confidence, last detection
        self.n_detections = 1
        # Big enough for a range to be claimed for it (detect.py's size floor).
        # An unconfirmed candidate is still watched and still drawn -- it is a
        # road user we can see but cannot measure.
        self.confirmed = True
        # Per-edge box velocity in px/s, EMA-smoothed. The overlay extrapolates
        # boxes with it between detections so they do not visibly trail a moving
        # object; nothing in the warning path reads it.
        self.vel_px = (0.0, 0.0, 0.0, 0.0)
        # Why this candidate's last depth reading was refused, if it was. Set by
        # the caller's depth_fn (live.py runs the plausibility check, which needs
        # camera geometry this module deliberately does not carry).
        self.range_reject = None
        # v2 sec8's track_quality term. With RF-DETR the box comes from a fresh
        # detection every frame rather than from a correlation filter, but the
        # question the term asks is unchanged: does this box move like one
        # vehicle, or did we just jump to a different object? Same scoring
        # function CSRT used, same EMA, so the confidence maths sees the same
        # kind of number it was tuned against.
        self.quality = 1.0

    # -- range ---------------------------------------------------------------
    @property
    def range_m(self):
        """Median of the last few depth readings, or None.

        A median, not the raw sample: ROI depth on a small or partly occluded
        box is spiky, and lead selection compares candidates against each other,
        so one bad frame must not reorder them. The full Kalman stays reserved
        for the selected lead -- running one per candidate would be four
        filters' worth of state to justify for a number only used to sort.
        """
        vals = [d for d in self.depths if d is not None and math.isfinite(d)]
        if not vals:
            return None
        vals = sorted(vals)
        return float(vals[len(vals) // 2])

    @property
    def n_valid_range(self):
        """How many of the retained depth samples are usable numbers.

        Falls to zero while depth is being refused (glare, model failure), which
        is what makes RANGE_MIN_SAMPLES bite on the first frame afterwards.
        """
        return sum(1 for d in self.depths
                   if d is not None and math.isfinite(d))

    @property
    def member_age_s(self):
        return None if self.member_since is None else (self.last_seen - self.member_since)

    def eligible(self, t):
        """May this candidate hold the lead lock?"""
        if self.merge_promoted:
            return True
        return (self.member and self.member_since is not None
                and (t - self.member_since) >= MEMBER_HOLD_S)

    # -- per-frame update -----------------------------------------------------
    def update_overlap(self, frac, reason, t):
        self.overlap = float(frac)
        self.overlap_reason = reason
        self.last_seen = float(t)
        # A row with no lane detected is not evidence of anything, so it does
        # not enter the trend history -- a merge must be seen, not inferred
        # from a gap.
        if reason == "ok":
            self.history.append((float(t), float(frac)))
        while self.history and (t - self.history[0][0]) > HISTORY_S:
            self.history.popleft()

        if reason != "ok":
            return
        if self.member:
            if frac < MEMBER_EXIT_FRAC:
                self.member = False
                self.member_since = None
        else:
            if frac >= MEMBER_ENTER_FRAC:
                self.member = True
                self.member_since = float(t)

    def check_merge(self, t, allow):
        """Rising-overlap promotion. -> event dict on the frame it fires."""
        if self.merge_promoted or self.member or not allow:
            return None
        rng = self.range_m
        if rng is None or rng > MERGE_MAX_RANGE_M:
            return None
        if self.overlap < MERGE_MIN_OVERLAP:
            return None
        window = [(ts, v) for ts, v in self.history if t - ts <= MERGE_WINDOW_S * 2]
        if len(window) < MERGE_MIN_SAMPLES:
            return None
        if (window[-1][0] - window[0][0]) < MERGE_WINDOW_S:
            return None
        slope = _slope(window)
        self.merge_slope = slope
        if slope < MERGE_MIN_SLOPE:
            return None
        self.merge_promoted = True
        self.merge_t = float(t)
        return {
            "candidate_id": self.id,
            "label": self.label,
            "t": round(float(t), 3),
            "overlap": round(self.overlap, 3),
            "slope_per_s": round(slope, 3),
            "window_s": round(window[-1][0] - window[0][0], 3),
            "samples": len(window),
            "range_m": round(rng, 2),
            "box": [round(v, 1) for v in self.box],
        }

    def to_log(self):
        """Compact per-frame record. Short keys: this ships every frame."""
        return {
            "id": self.id,
            "l": self.label,
            "ov": round(self.overlap, 3),
            "m": int(self.member),
            "ms": (None if self.member_age_s is None
                   else round(self.member_age_s, 2)),
            "e": int(self.eligible(self.last_seen)),
            "mp": int(self.merge_promoted),
            "r": (None if self.range_m is None else round(self.range_m, 1)),
            "q": round(self.quality, 2),
            "nr": self.n_valid_range,
            "s": round(self.score, 2),
            "w": self.overlap_reason if self.overlap_reason != "ok" else None,
            "cf": int(self.confirmed),
            "rr": self.range_reject,
        }


class CandidateSet:
    """Every vehicle we are currently watching, and which of them is the lead.

    Two clocks feed this. Detections arrive only when Qwen enumerates (the slow
    loop); between enumerations each candidate's box is carried forward by its
    own tracker, so membership dwell and overlap trend are measured on real
    per-frame geometry rather than interpolated between two distant samples.
    """

    def __init__(self, tracker_factory=None):
        self.candidates = {}
        self._next_id = 1
        self.lead_id = None
        self.n_merge_promotions = 0
        # MOSSE for candidates, not CSRT. Measured on this pod: CSRT 10.9 ms per
        # tracker per frame, MOSSE 0.20 ms -- six candidates is 65 ms against
        # 1.2 ms, and 65 ms/frame is most of the fast loop's budget spent on
        # vehicles we are not measuring. The trade is scale: MOSSE holds
        # position but not box size, so a candidate's width goes stale between
        # enumerations. That is acceptable here and nowhere else -- the LEAD
        # keeps CSRT, because its box scale is exactly what the Kalman reads.
        self._factory = tracker_factory or _make_mosse

    # -- detections (slow loop) ----------------------------------------------
    def observe(self, detections, t, frame=None, keep_labels=None):
        """Fold in one frame's detections.

        `detections` is [(label, box)] or [(label, box, score)]. Association is
        greedy IoU against the boxes we already hold, which is all the identity
        continuity a per-frame detector needs -- at 4 fps a vehicle moves a
        fraction of its own width between frames, so successive detections of
        the same car overlap heavily and detections of different cars do not.
        This is what replaced the MOSSE micro-trackers: they existed only to
        carry a box across the seconds between Qwen enumerations, and there are
        no such gaps any more.

        `keep_labels` limits which classes become candidates. Left None, every
        detected class is kept -- membership.py wants to know a cyclist is in
        the corridor even though LEAD_LABELS will never let one be the lead.
        """
        t = float(t)
        unmatched = dict(self.candidates)
        seen = []

        for det in detections:
            label, box = det[0], det[1]
            score = float(det[2]) if len(det) > 2 else 0.0
            # detect.py's fourth field. Absent for the synthetic-clip stubs and
            # for run_clip.py's Qwen anchor, where every box is confirmed by
            # construction, so its absence means True rather than unknown.
            info = det[3] if len(det) > 3 and isinstance(det[3], dict) else {}
            if keep_labels is not None and label is not None and label not in keep_labels:
                continue
            best, best_cost = None, None
            for cid, cand in unmatched.items():
                c = match_cost(box, label, cand)
                if c is not None and (best_cost is None or c < best_cost):
                    best, best_cost = cid, c
            box = tuple(float(v) for v in box)
            if best is not None:
                cand = unmatched.pop(best)
                dt = max(t - cand.last_detected, 1e-3)
                # Same discriminator CSRT's quality used, now applied to
                # detection-to-detection continuity. A jump to another vehicle
                # collapses it; ordinary approach does not.
                from .tracker import motion_plausibility
                q = motion_plausibility(cand.box, box, dt)
                cand.quality = 0.7 * cand.quality + 0.3 * q
                # Per-edge velocity BEFORE the box is replaced, from the two
                # detections and the real time between them. EMA at 0.5: enough
                # smoothing that one jittery box does not fling the overlay's
                # extrapolation, little enough that it follows a genuine
                # acceleration within a couple of frames.
                v = tuple((b - a) / dt for a, b in zip(cand.box, box))
                cand.vel_px = tuple(0.5 * old + 0.5 * new
                                    for old, new in zip(cand.vel_px, v))
                cand.box = box
                cand.label = label or cand.label
                cand.score = score
                cand.confirmed = bool(info.get("confirmed", True))
                cand.n_detections += 1
                cand.last_detected = t
                cand.last_seen = t
                cand.lost = False
                seen.append(cand)
            else:
                cand = Candidate(self._next_id, box, label, t)
                cand.score = score
                cand.confirmed = bool(info.get("confirmed", True))
                self._next_id += 1
                self.candidates[cand.id] = cand
                seen.append(cand)
            if frame is not None:
                self._init_tracker(cand, frame)

        # Candidates this frame did not report are not deleted here -- a single
        # missed detection is normal. They age out in _evict().
        self._evict(t)
        return seen

    def _init_tracker(self, cand, frame):
        x1, y1, x2, y2 = cand.box
        if x2 - x1 < 4 or y2 - y1 < 4:
            cand.tracker, cand.lost = None, True
            return
        try:
            tr = self._factory()
            tr.init(frame, (int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
            cand.tracker, cand.lost = tr, False
        except Exception:
            cand.tracker, cand.lost = None, True

    # -- tracking (fast loop) -------------------------------------------------
    def track(self, frame, t):
        """Carry every candidate's box forward one frame."""
        for cand in list(self.candidates.values()):
            if cand.tracker is None:
                # No tracker (creation failed, or boxes are being supplied
                # directly). Not "lost" -- its box simply stops advancing, and
                # it ages out below if no enumeration confirms it again.
                continue
            try:
                ok, xywh = cand.tracker.update(frame)
            except Exception:
                ok, xywh = False, None
            if not ok:
                cand.lost = True
                continue
            x, y, w, h = xywh
            cand.box = (float(x), float(y), float(x + w), float(y + h))
            cand.last_seen = float(t)
        self._evict(t)

    # -- membership (fast loop) ----------------------------------------------
    def evaluate(self, corridor, t, depth_fn=None, allow_merge=True):
        """Recompute overlap, membership and merge promotion for every candidate.

        `depth_fn(box, label) -> metres or None` keeps this module free of the
        depth model AND of the camera geometry: the caller is what knows the
        focal length, so the caller is where a range is checked for plausibility
        against the box's size (see headway/plausibility.py). A refused range
        arrives here as None, which is a state this module already handles --
        a candidate without a range cannot be the lead.

        `allow_merge` is False when the corridor is the trapezoid, since
        promoting a merge off guessed geometry would be inventing the one event
        the driver is least able to check.
        """
        t = float(t)
        promotions = []
        for cand in self.candidates.values():
            if cand.lost:
                continue
            frac, info = bottom_edge_overlap(cand.box, corridor)
            cand.update_overlap(frac, info["reason"], t)
            if depth_fn is not None:
                out = depth_fn(cand.box, cand.label)
                # (metres, reject_reason) from live.py; a bare number from any
                # caller that does not run the plausibility check.
                if isinstance(out, tuple):
                    rng, cand.range_reject = out
                else:
                    rng, cand.range_reject = out, None
                cand.depths.append(rng)
            ev = cand.check_merge(t, allow_merge)
            if ev is not None:
                ev["corridor_source"] = getattr(corridor, "source", "static")
                promotions.append(ev)
                self.n_merge_promotions += 1
        return promotions

    # -- selection -------------------------------------------------------------
    def select_lead(self, t):
        """Nearest eligible candidate. -> (candidate|None, info).

        Nearest wins outright: a car that has moved in between us and the
        vehicle we were following IS the new lead, and the gap that matters is
        the one to it. The identity change is reported so the caller can reset
        the filter and let the state machine re-confirm from scratch, rather
        than carrying a distance history that belongs to a different car.
        """
        # LEAD_LABELS is enforced here rather than at observe(): a pedestrian or
        # cyclist in the corridor is something the system must SEE (and log),
        # it just must never be the thing a following distance is computed to.
        eligible = [c for c in self.candidates.values()
                    if not c.lost and c.label in LEAD_LABELS
                    and c.eligible(t) and c.range_m is not None
                    # ...and its range is corroborated, not a single reading
                    # taken the frame after a blind spell. See
                    # RANGE_MIN_SAMPLES.
                    and c.n_valid_range >= RANGE_MIN_SAMPLES]
        uncorroborated = [c.id for c in self.candidates.values()
                          if not c.lost and c.label in LEAD_LABELS
                          and c.eligible(t) and c.range_m is not None
                          and c.n_valid_range < RANGE_MIN_SAMPLES]
        info = {
            "n_candidates": len(self.candidates),
            "n_members": sum(1 for c in self.candidates.values()
                             if c.member and not c.lost),
            "n_eligible": len(eligible),
            "n_vulnerable": sum(1 for c in self.candidates.values()
                                if c.label not in LEAD_LABELS and not c.lost),
            "n_uncorroborated": len(uncorroborated),
        }
        if uncorroborated:
            info["uncorroborated"] = uncorroborated
        if not eligible:
            changed = self.lead_id is not None
            self.lead_id = None
            info.update({"reason": "no_eligible_candidate", "lead_changed": changed})
            return None, info

        # Nearest by median depth; overlap breaks a tie so the more solidly
        # in-lane vehicle wins when two are the same distance away.
        eligible.sort(key=lambda c: (c.range_m, -c.overlap))
        best = eligible[0]
        changed = best.id != self.lead_id
        prev = self.lead_id
        self.lead_id = best.id
        info.update({
            "reason": "ok",
            "lead_id": best.id,
            "lead_label": best.label,
            "lead_range_m": round(best.range_m, 2),
            "lead_overlap": round(best.overlap, 3),
            "lead_via_merge": bool(best.merge_promoted),
            "lead_changed": changed,
            "prev_lead_id": prev,
        })
        return best, info

    # -- housekeeping ----------------------------------------------------------
    def _evict(self, t):
        for cid, cand in list(self.candidates.items()):
            # Liveness is "an enumeration confirmed it recently", full stop. A
            # tracker failure is not grounds for deletion on its own: the
            # candidate stops being advanced and stops being selectable, but
            # its history survives so the next enumeration can re-adopt it by
            # IoU rather than restarting a merge trend from zero.
            unconfirmed = (t - cand.last_detected) > CANDIDATE_MAX_UNDETECTED_S
            if unconfirmed:
                self.candidates.pop(cid, None)
                if self.lead_id == cid:
                    self.lead_id = None

    def reset(self):
        self.candidates.clear()
        self.lead_id = None

    def to_log(self):
        return [c.to_log() for c in self.candidates.values() if not c.lost]


def _make_mosse():
    """MOSSE tracker, wherever this OpenCV build keeps it."""
    import cv2
    for factory in (
        lambda: cv2.legacy.TrackerMOSSE_create(),
        lambda: cv2.TrackerMOSSE_create(),
    ):
        try:
            return factory()
        except AttributeError:
            continue
    raise RuntimeError("No MOSSE tracker in this OpenCV build")
