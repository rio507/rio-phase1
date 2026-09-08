"""RF-DETR object detection — the per-frame candidate source.

Design ref: docs/headway_design.md §9 ("When to add RF-DETR") and §10
(production stack). This is that graduation.

WHY QWEN HAD TO GO
------------------
§9's trigger list includes "Qwen starves the fast loop below 10 Hz", and it
did. Qwen3-VL decodes at ~52 tokens/s here, so enumerating vehicles cost
0.6-1.5 s per call against a 250 ms frame budget. Everything built around that
was a workaround: an 80-token cap that deliberately truncated the reply, a 1 s
floor between calls, a 5 s candidate-refresh interval, and merge detection that
could only see a vehicle if it happened to be in the last enumeration. RF-DETR
runs in single-digit milliseconds, so all four go away and candidates are
simply detected on every frame.

Qwen is not gone from the product -- it still writes /perceive's captions and
RIO's scene commentary, which is what an 8B VLM is actually for. It is gone
from the ANCHOR PATH. Nothing a language model produces now touches lead
selection, which makes §1's firewall a property of the architecture rather than
a rule the anchor prompt has to keep promising.

Upstream: RF-DETR (github.com/roboflow/rf-detr), Apache-2.0, Roboflow.
LICENCE MATTERS HERE: the obvious alternative is YOLO, and the Ultralytics
YOLO line is AGPL-3.0, which is commercially unusable for this product. The
variant config below carries upstream's own `license` field and
`_assert_apache()` checks it at load time, in the same spirit as depth.py's
guard against the CC-BY-NC Depth Anything weights.

WHY THE PACKAGE IS IMPORTED SIDEWAYS
------------------------------------
`import rfdetr` executes a package __init__ that pulls in the whole training
harness -- COCO dataset loaders, pycocotools, albumentations and its transitive
mess. None of it is needed to run a forward pass, and albucore currently fails
outright against this numpy. So `_rfdetr_models()` installs a stub package
object carrying only __path__ and imports `rfdetr.models.lwdetr` beneath it.
That is the same call made for UFLDv2 in lanes.py: use the model, not the
training harness. Here the model is a 30 M-parameter DINOv2-backbone DETR that
would be genuinely unwise to restate by hand, so upstream's own module is used
verbatim -- only its __init__ is skipped.

The architecture config comes from upstream's `RFDETRNanoConfig`, NOT from the
`args` namespace stored inside the checkpoint: that stored namespace is missing
patch_size, num_windows and positional_encoding_size, and carries the
training-time num_classes. The checkpoint is loaded STRICTLY, so a config that
did not match the weights is an error rather than a silently half-random
network -- verified at 0 missing / 0 unexpected keys.
"""
import math
import os
import sys
import threading
import time

import numpy as np
import torch

import config

from . import plausibility

# Where rfdetr's package directory is. NOT a constant: this used to be the
# literal "/usr/local/lib/python3.12/dist-packages/rfdetr", which was true of
# the pod it was written on and false the moment that pod was rebuilt on a
# Python 3.11 image. pip had installed rfdetr correctly the whole time; the
# stub's __path__ simply pointed at a directory that no longer existed, and the
# failure surfaced as `No module named 'rfdetr.models'` — which reads like a
# package-layout change and is nothing of the sort. Resolved at run time now, so
# it follows whichever interpreter is actually running.
RFDETR_PKG = os.environ.get("RIO_RFDETR_PKG", "")   # override; "" = discover

# Nano: 30.5 M params, 384x384, 2 decoder layers. The smallest published
# variant. Small (512 px, 3 decoder layers) is the fallback if nano's recall on
# distant vehicles proves too weak -- both checkpoints are fetched, and the
# variant is one constant. Measured cost for each is in the selftest.
VARIANT = os.environ.get("RIO_DETECTOR_VARIANT", "nano")
WEIGHTS = {
    "nano": os.environ.get("RIO_DETECT_WEIGHTS",
                           "/workspace/rio-phase1/weights/rf-detr-nano.pth"),
    "small": "/workspace/rio-phase1/weights/rf-detr-small.pth",
}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# COCO class ids -> the labels the headway geometry understands. RF-DETR's head
# is 90-way COCO; these are the only classes that can matter to a headway
# decision. `bicycle` and `person` are carried because membership.py needs to
# know they are THERE (a cyclist in the corridor is a hazard) even though
# LEAD_LABELS will refuse to make either one a following target.
COCO_TO_LABEL = {
    1: "pedestrian",     # person
    2: "cyclist",        # bicycle
    3: "car",
    4: "motorcycle",
    6: "bus",
    8: "truck",
}

# Confidence floor for a detection to become a candidate at all. Deliberately
# low: membership.py is what decides whether a box matters, and it is much
# better at it than a class score is. A vehicle dropped here is invisible to
# every rule downstream, so this gate only exists to keep obvious noise out of
# the candidate set.
SCORE_MIN = 0.35

# Per class, because the classes do not fail the same way. The vulnerable ones
# sit LOWER than the vehicles on purpose: a pedestrian at 40 m is a few hundred
# pixels of a person-shaped thing and scores worse than a car of the same
# apparent size, and the cost of missing one is not symmetric with the cost of
# drawing a spurious box. Nothing downstream can promote a pedestrian into a
# following target (membership.LEAD_LABELS), so a generous floor here buys
# recall without touching the warning path.
SCORE_MIN_BY_LABEL = {
    "car": 0.35,
    "truck": 0.35,
    "bus": 0.35,
    "motorcycle": 0.30,
    "pedestrian": 0.28,
    "cyclist": 0.28,
}

# Detections below this many pixels tall are past the range where DA-V2 gives a
# usable depth anyway (§7.2), and they dominate the candidate count on an open
# motorway.
MIN_BOX_PX = 12

# Between MIN_BOX_PX and plausibility.CONFIRM_MIN_PX a detection is kept but
# marked UNCONFIRMED: drawn, so the driver can see RIO has noticed something,
# with no range claimed for it. A small box must also clear a HIGHER score to
# be emitted at all, and that is the gate aimed squarely at the vanishing
# point: road-end texture produces small boxes at low confidence, and real
# distant vehicles produce small boxes at higher confidence. Size alone would
# throw both away; score alone keeps both.
SMALL_BOX_SCORE_MIN = 0.55

# --- duplicate suppression --------------------------------------------------
# RF-DETR is set-based and is not supposed to need NMS -- the Hungarian
# matching in training is what removes duplicates, and on a well-separated
# vehicle it does. It fails where the assignment is ambiguous, and the
# vanishing point is the canonical ambiguous case: several queries land on the
# same few pixels of road-end texture, each with its own slightly different box
# and its own confidence, and none of them is suppressed. Observed as a cluster
# of 4-5 overlapping "car" boxes at the horizon on a coastal clip.
#
# So the duplicates are removed here, greedily, highest score first.
IOU_MERGE = 0.55        # two boxes of one class overlapping this much are one object
CONTAIN_FRAC = 0.80     # ...or a small box mostly swallowed by a bigger one

# Cross-class suppression applies only WITHIN this set. car/truck/bus/motorcycle
# are competing readings of one object, so two of them on the same pixels means
# one object and one wrong label. pedestrian and cyclist are deliberately NOT in
# it: a cyclist IS a person on a bicycle, the two boxes genuinely coincide, and
# suppressing one would delete a road user rather than a duplicate.
EXCLUSIVE_LABELS = frozenset({"car", "truck", "bus", "motorcycle"})

# --- the ego vehicle's own bodywork ------------------------------------------
# The camera looks out over the car it is in, and the detector will call what it
# sees a car. This is the single most dangerous false positive available to this
# system: it lands dead centre in the ego lane at essentially zero range, so it
# wins lead selection outright and reads as a stationary object a couple of
# metres ahead.
#
# TWO SHAPES, AND THE SECOND ONE COST A REAL DRIVE.
#
# The first is what a windscreen-mounted dashcam sees: a thin strip of bonnet
# across the bottom of the frame. Observed on the winding clip, `car 0.40` at
# [2, 661, 1279, 720] -- full frame width, flush with the bottom edge, aspect
# 21.6. The aspect bound was what separated it from a genuinely close vehicle,
# which is wide but also TALL.
#
# The second is what a PHONE sees, and it is not thin. A phone sits higher and
# further back, so the bonnet and the whole dashboard fill a deep band across
# the bottom of the picture. Session 06af3214: `[0.6, 308, 640, 478]` on a
# 640x480 frame, aspect 3.8, present on 584 frames -- 36% of a ten-minute drive
# -- ranged at 3.7 m, holding the lead lock at 20 m/s, and the source of twelve
# of the twenty-two warnings RIO spoke that day. It passed this gate because
# 3.8 is not 6.
#
# So aspect is no longer the discriminator; the HORIZON is, and it is a
# statement about geometry rather than a fitted number. A box spanning
# essentially the whole frame width is, by the pinhole relation, about two
# metres away -- and at two metres any vehicle is far taller than the frame, so
# its box is clipped at the top and its top edge is at y=0. A full-width box
# whose top edge never rises above the horizon is therefore not a vehicle. It
# is the thing the camera is bolted to.
#
# Checked against the drive: this catches 579 of the 584 offending frames and
# not one of the 295 frames carrying a genuine lead beyond 15 m. The remainder
# are caught by the two independent gates downstream -- the width bound in
# plausibility.py and the static-structure veto in membership.py.
BONNET_BOTTOM_FRAC = 0.98    # box bottom is flush with the frame bottom
BONNET_WIDTH_FRAC = 0.85     # ...spanning almost the whole frame
BONNET_MIN_ASPECT = 6.0      # ...and far wider than it is tall

# The second shape needs a stricter width, because it leans entirely on the
# horizon argument and that argument is overwhelming at 0.95 and merely
# suggestive at 0.85. See config.py.
SLAB_WIDTH_FRAC = config.HEADWAY_EGO_SLAB_WIDTH_FRAC
SLAB_HORIZON_SLACK_FRAC = config.HEADWAY_EGO_SLAB_HORIZON_SLACK_FRAC


def horizon_row(w, h) -> float:
    """The image row the horizon falls on, in pixels from the top.

    Same camera model the corridor uses, imported rather than restated so there
    is one of it: a ground point at infinite range projects to
    v = cy - f*tan(pitch), which at the configured pitch of zero is simply the
    middle of the frame. Imported lazily for the same reason plausibility.py
    does it -- detect.py is loaded before the corridor exists.
    """
    from .anchor import CAMERA_PITCH_RAD, HFOV_DEG
    f_px = (float(w) / 2.0) / math.tan(math.radians(HFOV_DEG) / 2.0)
    return float(h) / 2.0 - f_px * math.tan(CAMERA_PITCH_RAD)


def _is_ego_bonnet(x1, y1, x2, y2, w, h) -> bool:
    """Is this box the car this camera is bolted to? Either shape counts."""
    bw, bh = x2 - x1, max(y2 - y1, 1e-6)
    if y2 < BONNET_BOTTOM_FRAC * h:
        # Not flush with the bottom of the frame. Both shapes require it: the
        # camera cannot see its own car anywhere else.
        return False
    # Shape one: wide and flat. A dashcam's strip of bonnet.
    if bw >= BONNET_WIDTH_FRAC * w and (bw / bh) >= BONNET_MIN_ASPECT:
        return True
    # Shape two: wide and deep, and never rising above the horizon. A phone's
    # view of the bonnet and the dashboard together.
    if bw >= SLAB_WIDTH_FRAC * w:
        top_limit = horizon_row(w, h) - SLAB_HORIZON_SLACK_FRAC * h
        if y1 >= top_limit:
            return True
    return False


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0, 0.0
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    union = area_a + area_b - inter
    # Second value is the CONTAINMENT fraction of the smaller box, which is the
    # number that catches a nested duplicate: a 10 px box sitting entirely
    # inside a 40 px box has an IoU of only 0.06 and a containment of 1.0.
    return inter / union, inter / min(area_a, area_b)


def _duplicates(dets):
    """Greedy duplicate suppression. -> (kept, dropped)

    `dets` is [(label, box, score), ...]; both lists returned hold the same
    tuples, with the dropped ones carrying the id of the box that beat them so
    a quality report can say WHICH cluster collapsed rather than only how many.

    Highest score first, so the survivor of a cluster is the detection the model
    was most sure of. Two boxes are the same object when they are the same class
    and overlap past IOU_MERGE, when one is CONTAIN_FRAC swallowed by the other,
    or when they are two different EXCLUSIVE_LABELS readings of the same pixels.
    """
    order = sorted(range(len(dets)), key=lambda i: -dets[i][2])
    kept, dropped = [], []
    for i in order:
        label_i, box_i, score_i = dets[i][0], dets[i][1], dets[i][2]
        beaten_by = None
        for j in kept:
            label_j, box_j = dets[j][0], dets[j][1]
            same_class = label_i == label_j
            exclusive = (label_i in EXCLUSIVE_LABELS and label_j in EXCLUSIVE_LABELS)
            if not (same_class or exclusive):
                continue
            iou, contain = _iou(box_i, box_j)
            if iou >= IOU_MERGE or contain >= CONTAIN_FRAC:
                beaten_by = (j, round(iou, 3), round(contain, 3),
                             "iou" if iou >= IOU_MERGE else "containment")
                break
        if beaten_by is None:
            kept.append(i)
        else:
            dropped.append((i, beaten_by))
    return kept, dropped


def raw_boxes(scores, labels, boxes, w, h):
    """Model output -> ([(label, box, score)], n_bonnet_rejected).

    Road users only, no confidence gate.

    Everything that is class-mappable, big enough to be a box at all, and not
    our own bonnet. The confidence and duplicate gates are deliberately NOT
    here: they are policy, they have changed once and will change again, and
    keeping them separate is what lets one forward pass be re-gated two ways.
    """
    out, n_bonnet = [], 0
    for s, c, b in zip(scores, labels, boxes):
        name = COCO_TO_LABEL.get(int(c))
        if name is None:
            continue
        x1, y1, x2, y2 = (float(b[0]), float(b[1]), float(b[2]), float(b[3]))
        x1, x2 = max(0.0, min(x1, x2)), min(float(w), max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(float(h), max(y1, y2))
        if (x2 - x1) < 4 or (y2 - y1) < MIN_BOX_PX:
            continue
        if _is_ego_bonnet(x1, y1, x2, y2, w, h):
            n_bonnet += 1
            continue
        out.append((name, (x1, y1, x2, y2), float(s)))
    return out, n_bonnet


def gate(raw, dedupe: bool = True, score_min: float = None,
         per_class: bool = True, size_floor: bool = True) -> dict:
    """Apply the confidence and duplicate policy to raw boxes.

    `per_class=False, size_floor=False, dedupe=False` reproduces the gating
    this file shipped before any of it existed: one flat SCORE_MIN for every
    class and every box size, and whatever duplicates the model emitted. That
    combination is not used in production -- it is there so the quality harness
    can measure what the gates actually removed rather than asserting it.
    """
    kept_pre, n_small = [], 0
    for name, box, score in raw:
        h_px = box[3] - box[1]
        if per_class:
            floor = SCORE_MIN_BY_LABEL.get(name, SCORE_MIN)
        else:
            floor = SCORE_MIN
        if size_floor and h_px < plausibility.CONFIRM_MIN_PX:
            floor = max(floor, SMALL_BOX_SCORE_MIN)
        if score_min is not None:
            floor = max(floor, float(score_min))
        if score < floor:
            # Counted separately when it is the SIZE that raised the bar: that
            # is the gate the vanishing-point cluster dies on, and its rate is
            # what the quality report leads with.
            if score >= SCORE_MIN_BY_LABEL.get(name, SCORE_MIN):
                n_small += 1
            continue
        kept_pre.append((name, box, score))

    if dedupe:
        keep_idx, dropped = _duplicates(kept_pre)
        kept = [kept_pre[i] for i in keep_idx]
    else:
        kept, dropped = list(kept_pre), []

    dets = [(name, box, score,
             {"confirmed": (box[3] - box[1]) >= plausibility.CONFIRM_MIN_PX,
              "h_px": round(box[3] - box[1], 1)})
            for name, box, score in kept]

    # Largest first, which on a forward-facing camera is nearest first. The
    # membership layer re-ranks by measured depth; this only makes truncation
    # or a debug dump show the vehicles that matter at the top.
    dets.sort(key=lambda d: -((d[1][2] - d[1][0]) * (d[1][3] - d[1][1])))

    return {
        "detections": dets,
        "n_duplicates_dropped": len(dropped),
        "n_small_rejected": n_small,
        "n_unconfirmed": sum(1 for d in dets if not d[3]["confirmed"]),
        "duplicates": [{"label": kept_pre[i][0],
                        "box": [round(v, 1) for v in kept_pre[i][1]],
                        "score": round(kept_pre[i][2], 3),
                        "beaten_by": [round(v, 1) for v in kept_pre[j][1]],
                        "iou": iou, "containment": contain, "rule": rule}
                       for i, (j, iou, contain, rule) in dropped],
    }


def _score_floor(label, h_px):
    """The confidence this detection has to clear, given its class and size."""
    floor = SCORE_MIN_BY_LABEL.get(label, SCORE_MIN)
    if h_px < plausibility.CONFIRM_MIN_PX:
        floor = max(floor, SMALL_BOX_SCORE_MIN)
    return floor

_model = None
_postproc = None
_device = None
_dtype = None
_mean = None
_std = None
_resolution = None
_lock = threading.Lock()
_load_error = None


_pkg_path = None


def _pkg_dir():
    """rfdetr's package directory, found WITHOUT executing its __init__.

    find_spec on the top-level name only asks the path finders — it does not run
    the module, which is the whole point of the sideways import. (The same call
    on a SUBmodule would import the parent, so this must stay a top-level
    lookup.)

    Memoised, and that is not an optimisation. find_spec consults sys.modules
    FIRST, and once the stub is installed there it raises ValueError on the
    stub's absent __spec__ — so a second call would report rfdetr uninstalled
    while the module it points at is loaded and working. The answer is cached
    from the one call that happens before the stub exists, and the stub's own
    __path__ is the fallback for anyone who asks later.
    """
    global _pkg_path
    if RFDETR_PKG:
        return RFDETR_PKG
    if _pkg_path and os.path.isdir(_pkg_path):
        return _pkg_path
    import importlib.util

    try:
        spec = importlib.util.find_spec("rfdetr")
    except (ImportError, ValueError):
        spec = None
    locations = list(getattr(spec, "submodule_search_locations", None) or [])
    if not locations:
        # Already imported (stub or real): believe what it says about itself.
        locations = list(getattr(sys.modules.get("rfdetr"), "__path__", None) or [])
        locations = [p for p in locations if os.path.isdir(p)]
    if not locations:
        raise ImportError(
            "rfdetr is not importable by this interpreter "
            f"({sys.executable}). Install it the way boot.sh does — "
            "`pip install --no-deps rfdetr==1.5.0 supervision==0.29.1 "
            "pycocotools peft` — or point RIO_RFDETR_PKG at the package "
            "directory.")
    _pkg_path = locations[0]
    return _pkg_path


def _rfdetr_models():
    """Import rfdetr.models.lwdetr without executing the package __init__."""
    import types

    existing = sys.modules.get("rfdetr")
    if existing is None or (getattr(existing, "_rio_stub", False)
                            and not os.path.isdir((getattr(existing, "__path__", None) or [""])[0])):
        # Second half of that condition: a stub left behind by an earlier failed
        # attempt would otherwise pin the bad path for the life of the process,
        # so a fixed environment could not recover without a restart.
        stub = types.ModuleType("rfdetr")
        stub.__path__ = [_pkg_dir()]
        stub._rio_stub = True
        sys.modules["rfdetr"] = stub
    from rfdetr.models.lwdetr import PostProcess, build_model
    return build_model, PostProcess


def _config(variant: str):
    """Upstream's own config for this variant, as a plain dict."""
    _rfdetr_models()                      # ensures the stub package exists
    from rfdetr.config import RFDETRNanoConfig, RFDETRSmallConfig

    cfg_cls = {"nano": RFDETRNanoConfig, "small": RFDETRSmallConfig}[variant]
    return cfg_cls().model_dump()


def _assert_apache(cfg: dict) -> None:
    """Refuse anything that is not Apache-2.0.

    Same guard depth.py applies to the Depth Anything weights, for the same
    reason: the licence of a model in this product is a shipping constraint,
    not a footnote. It is checked here rather than trusted to whoever next
    edits VARIANT.
    """
    lic = str(cfg.get("license", "")).strip()
    if lic and lic.lower() != "apache-2.0":
        raise ValueError(
            f"Refusing to load an RF-DETR variant licensed {lic!r}: headway "
            "requires Apache-2.0. (This is also why the detector is RF-DETR "
            "and not Ultralytics YOLO, which is AGPL-3.0.)")


def _build_args(cfg: dict):
    """cfg -> the namespace build_model() expects.

    The extras below are architecture fields that live in populate_args()
    upstream rather than in the model config. They are stated explicitly
    instead of imported because importing main.py drags the training stack back
    in; the strict checkpoint load is what proves they are right.
    """
    import argparse

    cfg = dict(cfg)
    cfg.pop("license", None)
    extra = dict(
        aux_loss=True, backbone_lora=False, backbone_only=False, drop_path=0.0,
        encoder_only=False, freeze_encoder=False, num_feature_levels=1,
        position_embedding="sine", pretrained_encoder=None, rms_norm=False,
        use_cls_token=False, vit_encoder_num_layers=12, window_block_indexes=None,
        dropout=0.0, dim_feedforward=2048, decoder_norm="LN",
        shape=(cfg["resolution"], cfg["resolution"]),
        # The DINOv2 encoder's ImageNet weights are NOT downloaded: every
        # parameter is about to be overwritten by the RF-DETR checkpoint, so
        # fetching them would be a few hundred MB to throw away, and it would
        # put a network call on the load path.
        force_no_pretrain=True, pretrain_weights=None,
    )
    return argparse.Namespace(**{**cfg, **extra})


def _ensure_loaded(variant: str = None, weights_path: str = None) -> None:
    global _model, _postproc, _device, _dtype, _mean, _std, _resolution, _load_error
    if _model is not None:
        return
    if _load_error is not None:
        raise RuntimeError(_load_error)

    variant = variant or VARIANT
    path = weights_path or WEIGHTS[variant]
    if not os.path.exists(path):
        _load_error = (
            f"RF-DETR weights not found at {path}. Fetch them with "
            "`python -m tools.fetch_detector_weights`.")
        raise RuntimeError(_load_error)

    cfg = _config(variant)
    _assert_apache(cfg)
    build_model, PostProcess = _rfdetr_models()

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _dtype = torch.float16 if _device == "cuda" else torch.float32
    _resolution = int(cfg["resolution"])

    model = build_model(_build_args(cfg))
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"RF-DETR checkpoint does not match the model: {len(missing)} "
            f"missing {missing[:4]}, {len(unexpected)} unexpected {unexpected[:4]}")

    _model = model.to(device=_device, dtype=_dtype).eval()
    _postproc = PostProcess(num_select=int(cfg["num_select"]))
    _mean = torch.tensor(IMAGENET_MEAN, device=_device,
                         dtype=torch.float32).view(1, 3, 1, 1)
    _std = torch.tensor(IMAGENET_STD, device=_device,
                        dtype=torch.float32).view(1, 3, 1, 1)


def warm(variant: str = None) -> None:
    """Preload weights and pay CUDA kernel warmup off the frame path.

    ...and the compile, and the check that decides whether to use it. All of
    it here, because every second of it is a second that would otherwise be
    inside a live frame. See adopt_accel.
    """
    _ensure_loaded(variant)
    detect(np.zeros((720, 1280, 3), dtype=np.uint8))
    adopt_accel()


def available() -> bool:
    if _model is not None:
        return True
    if _load_error is not None:
        return False
    return os.path.exists(WEIGHTS.get(VARIANT, ""))


def _preprocess(frame_bgr):
    """BGR uint8 -> normalised square tensor on the device.

    Mirrors upstream predict(): to_tensor -> normalize -> resize. The order is
    upstream's (normalise before resize); it is preserved because the weights
    were validated against it, and swapping the two changes edge pixels.
    """
    t = torch.from_numpy(np.ascontiguousarray(frame_bgr)).to(_device)
    t = t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    t = t.flip(1)                                       # BGR -> RGB
    t = (t - _mean) / _std
    t = torch.nn.functional.interpolate(
        t, size=(_resolution, _resolution), mode="bilinear",
        align_corners=False, antialias=True)
    return t.to(_dtype)


# ===========================================================================
# ACCELERATION — Jetson prep, gated on seeing the same road
# ===========================================================================
# WHY, and it is not this pod. Detection here is 11.8 ms of a 67 ms frame and
# nothing is waiting on it. But the profile says its cost is almost entirely
# LAUNCH OVERHEAD -- 3.26 ms of GPU arithmetic inside 10.08 ms of elapsed time,
# spread over ~2940 aten calls per forward -- and launch overhead is exactly
# the cost that gets worse on an Orin, whose CPU is also doing the camera and
# the radio. So the work is done here, where a compiled kernel can be checked
# against a known-good answer, rather than on the target where a regression
# would look like the target being slow.
#
# WHAT THIS IS NOT. It is not a second implementation of detection. `detect()`
# is unchanged: the same preprocess, the same postprocess, the same gates. The
# only thing that moves is which callable produces pred_logits and pred_boxes.
#
# THE TRADE, STATED. torch.compile(mode="reduce-overhead") is CUDA graphs plus
# Inductor codegen, and Inductor's kernels are not the eager kernels: the raw
# tensors differ by up to 4.21 on pred_logits. A raw torch.cuda.CUDAGraph
# capture WOULD be bit-for-bit -- and cannot be captured here, because rfdetr's
# transformer builds `spatial_shapes` with torch.as_tensor(<python list>,
# device=cuda) on every forward and an unpinned host-to-device copy is not
# capturable. Patching a vendored library's internals in the path that decides
# when to warn a driver is a worse trade than a measured tolerance.
# torch.compile(backend="cudagraphs") is bit-for-bit and worth 1.09x, because
# dynamo graph-breaks on those same calls.
#
# SO THE GUARDRAIL IS RUNTIME, NOT EDITORIAL. `adopt_accel` compiles, then runs
# the compiled model and the eager model over real frames and compares THE
# DETECTIONS -- what the rest of the system receives, after the confidence
# gate, the size floor, the ego-structure gate and duplicate suppression. Same
# count, same labels, boxes and scores inside config's tolerance, or the
# compiled model is discarded and the drive runs eager. Every outcome is
# printed and readable afterwards through accel_status().
#
# tools/accel_verify.py is the same comparison over a much wider corpus, and is
# where the tolerance in config.py came from.

_accel_model = None          # the compiled callable, once it has been adopted
_accel_state = {
    "mode": None,            # what config asked for
    "active": False,         # is the compiled model the one being used
    "reason": None,          # why not, when it is not
    "compile_s": None,
    "checked_frames": 0,
    "max_box_frac": None,
    "max_score": None,
    "input_shape": None,     # the shape it was compiled for
}


def accel_status() -> dict:
    """What happened to the acceleration, for the log and the selftest."""
    return dict(_accel_state)


def _accel_frames(n):
    """Real road frames to check a compiled model against.

    Real, because a compiled kernel that agrees with eager on a grey rectangle
    has proved nothing about a road. At the drive's own frame sizes as well as
    the clip's, so the postprocess -- which is what turns a model output into a
    box in frame pixels -- is exercised at each of them.

    An empty list is a refusal, not a pass: with nothing to check against,
    adopt_accel keeps the eager model.
    """
    import cv2

    # Relative to this file, not to this pod: the whole point of the exercise
    # is a target machine, and a check that only finds its corpus at
    # /workspace is a check that silently does not run on an Orin.
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    clips = [os.path.join(here, "runs", "road_clip.mp4"),
             os.path.join(here, "runs", "night_clip.webm"),
             "/workspace/ufldv2/example.mp4"]
    sizes = [None, (640, 480), (480, 640)]
    out = []
    for path in clips:
        if len(out) >= n or not os.path.exists(path):
            continue
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 200
        for i in range(0, total, max(1, total // 4)):
            if len(out) >= n:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, f = cap.read()
            if not ok:
                break
            for size in sizes:
                if len(out) >= n:
                    break
                out.append(f if size is None
                           else cv2.resize(f, size, interpolation=cv2.INTER_AREA))
        cap.release()
    return out


def _detections_only(frame):
    """Everything a consumer of detect() actually receives, for one frame.

    The label, the box, the score AND the `confirmed` flag -- the last of those
    because it is the size floor, which decides whether a range may be claimed
    for a box at all. A compiled model that quietly moved a box across it would
    be changing behaviour while every other number looked fine.
    """
    r = detect(frame)
    w = float((r.get("image") or {}).get("w") or 1.0)
    return w, [(d[0], tuple(float(v) for v in d[1]), float(d[2]),
                bool((d[3] or {}).get("confirmed", True)) if len(d) > 3 else True)
               for d in r["detections"]]


def _arbitrary_tie(la, lb, sa, sb) -> bool:
    """Is this label disagreement a coin-toss the eager model also loses?

    EXCLUSIVE_LABELS already exists because car, truck, bus and motorcycle are
    "competing readings of one object" (see its own note); duplicate
    suppression keeps whichever scored highest, and at a small enough margin
    that is a tie-break rather than a judgement.

    THIS IS NOT A CONVENIENCE, AND IT WAS MEASURED. Verification over 180 real
    frames turned up exactly two disagreements, both of this shape:

        road_clip.mp4@108   eager `bus` 0.3797   compiled `truck` 0.3795
        road_clip.mp4@114   eager `bus` 0.3794   compiled `truck` 0.3789

    So the eager model was asked the same question again with the frame nudged
    in ways nobody could see -- one grey level brighter, one darker, re-encoded
    as JPEG at q95 and at q92, shifted by a single pixel:

        EAGER ITSELF ANSWERED DIFFERENTLY ON FIVE OF SIX PERTURBATIONS.

    Requiring the compiled model to reproduce a tie-break the eager model does
    not hold stable against a one-grey-level change is requiring bit-identity,
    which Inductor cannot give. So a disagreement between two exclusive labels
    whose scores are inside the score tolerance is allowed and COUNTED, and
    every other label disagreement still fails outright.
    """
    if la not in EXCLUSIVE_LABELS or lb not in EXCLUSIVE_LABELS:
        return False
    return abs(sa - sb) <= config.HEADWAY_ACCEL_MAX_SCORE_DELTA


def _same_road(eager, accel):
    """-> (ok, reason, max_frac_of_frame, max_score). The whole guardrail, once.

    THE DISCRETE OUTCOMES HAVE NO TOLERANCE. A detection appearing or
    vanishing, a label changing, the confirmed flag flipping: no distance in
    pixels describes any of them, so they fail outright rather than being
    measured against a number.

    The box delta is bounded TWICE, in two units, because two different things
    are being asked. As a fraction of FRAME WIDTH it detects drift, which is
    where the error lives -- the model emits normalised coordinates and the
    postprocess multiplies by the frame size, so an absolute pixel bound would
    charge a large frame for arithmetic it did not do. As a fraction of THE BOX
    it bounds behaviour: every consumer downstream reads a fraction of a box,
    and the worst case measured was a sub-pixel shift on a 20 px wide box. See
    config for both numbers and where they came from.
    """
    max_frac = 0.0        # of frame width -- the drift detector
    max_of_box = 0.0      # of the box's smaller side -- the behaviour bound
    max_score = 0.0
    for (w, a), (_, b) in zip(eager, accel):
        if len(a) != len(b):
            return False, f"{len(a)} detections became {len(b)}", max_frac, max_score
        for (la, ba, sa, ca), (lb, bb, sb, cb) in zip(a, b):
            if la != lb and not _arbitrary_tie(la, lb, sa, sb):
                return False, f"label {la!r} became {lb!r}", max_frac, max_score
            if ca != cb:
                return (False, f"the confirmed flag flipped {ca} -> {cb} on a "
                        f"{la}", max_frac, max_score)
            px = max(abs(p - q) for p, q in zip(ba, bb))
            side = max(1e-6, min(ba[2] - ba[0], ba[3] - ba[1]))
            max_frac = max(max_frac, px / max(w, 1.0))
            max_of_box = max(max_of_box, px / side)
            max_score = max(max_score, abs(sa - sb))
    if max_frac > config.HEADWAY_ACCEL_MAX_BOX_DELTA_FRAC:
        return (False, f"a box moved {max_frac:.6f} of frame width, over the "
                f"{config.HEADWAY_ACCEL_MAX_BOX_DELTA_FRAC} tolerance",
                max_frac, max_score)
    if max_of_box > config.HEADWAY_ACCEL_MAX_BOX_DELTA_OF_BOX:
        return (False, f"a box moved {max_of_box:.4f} of its own smaller side, "
                f"over the {config.HEADWAY_ACCEL_MAX_BOX_DELTA_OF_BOX} tolerance",
                max_frac, max_score)
    if max_score > config.HEADWAY_ACCEL_MAX_SCORE_DELTA:
        return (False, f"a score moved {max_score:.5f}, over the "
                f"{config.HEADWAY_ACCEL_MAX_SCORE_DELTA} tolerance",
                max_frac, max_score)
    return True, None, max_frac, max_score


def set_accel(on: bool) -> bool:
    """Switch the compiled model in or out. -> is it now active.

    Used by tools/accel_verify.py to measure both paths in one process, and by
    the selftest. A drive goes through adopt_accel, which checks first.
    """
    global _accel_model
    if not on:
        _accel_state["active"] = False
        return False
    if _accel_model is None and not _compile_model():
        return False
    _accel_state["active"] = True
    return True


def _compile_model() -> bool:
    """Compile, or say why not. Never raises."""
    global _accel_model
    if _accel_model is not None:
        return True
    if _model is None or _device != "cuda":
        _accel_state["reason"] = "no model on cuda"
        return False
    try:
        t0 = time.perf_counter()
        m = torch.compile(_model, mode="reduce-overhead")
        # Compilation is lazy; these are what actually pay for it, and paying
        # for it HERE is the point -- the alternative is a 7 second first frame.
        probe = torch.zeros((1, 3, _resolution, _resolution),
                            device=_device, dtype=_dtype)
        with torch.inference_mode():
            for _ in range(3):
                m(probe)
            torch.cuda.synchronize()
        _accel_model = m
        _accel_state["compile_s"] = round(time.perf_counter() - t0, 1)
        _accel_state["input_shape"] = (1, 3, _resolution, _resolution)
        return True
    except Exception as e:
        # Anything at all: no triton, no compiler, an OOM during autotuning, a
        # torch too old. The drive runs eager and says so.
        _accel_state["reason"] = f"compile failed: {type(e).__name__}: {e}"[:200]
        _accel_model = None
        return False


def adopt_accel() -> bool:
    """Compile the detector and use it ONLY if it sees the same road.

    Called from warm(). Returns whether the compiled model is now in use.
    Never raises: every failure path leaves the eager model running, which is
    the model that has been driven on.
    """
    mode = str(getattr(config, "HEADWAY_DETECT_ACCEL", "off")).lower()
    _accel_state["mode"] = mode
    if mode == "off":
        _accel_state["reason"] = "switched off in config"
        return False
    if not torch.cuda.is_available():
        _accel_state["reason"] = "no cuda"
        return False

    try:
        frames = _accel_frames(int(config.HEADWAY_ACCEL_VERIFY_FRAMES))
        if mode != "on" and not frames:
            # No corpus, no check, no adoption. "auto" means "on if verified",
            # and an unverifiable claim is not a verified one.
            _accel_state["reason"] = "no road clips to verify against"
            return False

        # The reference, from the model that has actually been driven on.
        set_accel(False)
        eager = [_detections_only(f) for f in frames]

        if not _compile_model():
            return False
        set_accel(True)

        if mode == "on":
            # Deliberate escape hatch for benchmarking. Never for a drive.
            print("[detect] acceleration forced on WITHOUT the identity check "
                  "(config.HEADWAY_DETECT_ACCEL='on')", flush=True)
            return True

        accel = [_detections_only(f) for f in frames]
        ok, reason, max_box, max_score = _same_road(eager, accel)
        _accel_state.update(checked_frames=len(frames),
                            max_box_frac=round(max_box, 7),
                            max_score=round(max_score, 6))
        if not ok:
            set_accel(False)
            _accel_state["reason"] = reason
            print(f"[detect] compiled model REJECTED: {reason} "
                  f"(over {len(frames)} real frames) — running eager",
                  flush=True)
            return False
        print(f"[detect] compiled model adopted: {len(frames)} real frames, "
              f"same detections, boxes within {max_box:.6f} of frame width, "
              f"scores within {max_score:.6f} "
              f"(compile {_accel_state['compile_s']}s)", flush=True)
        return True
    except Exception as e:
        set_accel(False)
        _accel_state["reason"] = f"verify failed: {type(e).__name__}: {e}"[:200]
        print(f"[detect] acceleration abandoned: {_accel_state['reason']}",
              flush=True)
        return False


# One pair of CUDA events, reused. Creating two per frame at 15 fps is
# allocation for nothing, and `_lock` already serialises every forward pass, so
# a shared pair cannot be recorded by two calls at once.
#
# `enable_timing=True` is what makes elapsed_time work; without it an event is
# only a synchronisation marker.
_events = None


def _forward_events():
    """The event pair to bracket the model pass with, or (None, None) on CPU."""
    global _events
    if not torch.cuda.is_available():
        return None, None
    if _events is None:
        _events = (torch.cuda.Event(enable_timing=True),
                   torch.cuda.Event(enable_timing=True))
    return _events


def _forward(x):
    """The model pass: compiled if one was adopted, eager otherwise.

    THE SHAPE GUARD IS NOT DECORATION. A CUDA graph is captured for exactly one
    input shape, and `mode="reduce-overhead"` recompiles rather than failing if
    it sees another -- which on a frame path means a multi-second stall inside
    a live drive the first time anything changes `_resolution`. So a shape that
    is not the compiled one goes to the eager model, once, silently, and keeps
    going there.
    """
    if _accel_state["active"] and _accel_model is not None:
        if tuple(x.shape) == _accel_state["input_shape"]:
            try:
                return _accel_model(x)
            except Exception as e:
                # A compiled model that starts throwing mid-drive gets one
                # sentence and no further chances. The eager model is the one
                # the car has been driven on.
                _accel_state["active"] = False
                _accel_state["reason"] = f"runtime: {type(e).__name__}: {e}"[:200]
                print(f"[detect] compiled model failed mid-drive "
                      f"({type(e).__name__}) — back to eager", flush=True)
        else:
            _accel_state["active"] = False
            _accel_state["reason"] = (
                f"input shape {tuple(x.shape)} is not the compiled "
                f"{_accel_state['input_shape']}")
            print(f"[detect] {_accel_state['reason']} — back to eager", flush=True)
    return _model(x)


def detect(frame, score_min: float = None, variant: str = None,
           dedupe: bool = True) -> dict:
    """Detect road users in one BGR frame.

    Args:
        score_min  extra global confidence floor. None (the default) uses
                   SCORE_MIN_BY_LABEL alone; a number is applied ON TOP of the
                   per-class floors, never below them, so a caller can ask for
                   less noise but never for a class gate looser than the table.
        dedupe     False reproduces the pre-suppression behaviour verbatim.
                   Only tools/detect_quality.py passes it, to measure the
                   before/after on the same frames.

    Returns:
        detections  [(label, (x1, y1, x2, y2), score, info), ...] in FRAME
                    pixels, nearest-looking first (largest box first). `info`
                    carries `confirmed` (the box clears the size floor, so a
                    range may be claimed for it) and `h_px`. Consumers that
                    predate it read the first three fields and are unaffected.
        n_duplicates_dropped  how many boxes were suppressed as duplicates
        n_small_rejected      how many failed the size/score floor
        timing_ms   forward / postprocess / total, in milliseconds.

                    `forward` is the GPU KERNEL time of the model pass,
                    measured with CUDA events -- not the wall clock around the
                    call. It used to be the wall clock, and on this hardware
                    that reported 0.4 ms for 9.8 ms of work: a CUDA launch
                    returns the instant the kernels are queued, so what was
                    being timed was the queueing. The number is read after the
                    `.cpu()` below, which has already synchronised past both
                    events, so this costs nothing.
    """
    t0 = time.perf_counter()
    _ensure_loaded(variant)
    h, w = frame.shape[:2]

    ev0, ev1 = _forward_events()
    with _lock:
        with torch.inference_mode():
            x = _preprocess(frame)
            if ev0 is not None:
                ev0.record()
            out = _forward(x)
            if ev1 is not None:
                ev1.record()
            if isinstance(out, tuple):
                out = {"pred_boxes": out[0], "pred_logits": out[1]}
            t_fwd = time.perf_counter()
            sizes = torch.tensor([[h, w]], device=_device)
            res = _postproc({"pred_logits": out["pred_logits"].float(),
                             "pred_boxes": out["pred_boxes"].float()},
                            target_sizes=sizes)[0]
        # This is the synchronisation point for the whole pass, which is why
        # the events above can be read straight after it.
        scores = res["scores"].detach().cpu().numpy()
        labels = res["labels"].detach().cpu().numpy()
        boxes = res["boxes"].detach().cpu().numpy()
        fwd_ms = None
        if ev0 is not None:
            try:
                fwd_ms = ev0.elapsed_time(ev1)
            except Exception:
                # A device that cannot time itself is not a reason to fail a
                # frame. The field goes to the wall-clock meaning it had.
                fwd_ms = None

    raw, n_bonnet = raw_boxes(scores, labels, boxes, w, h)
    out = gate(raw, dedupe=dedupe, score_min=score_min)
    out["n_bonnet_rejected"] = n_bonnet
    t_end = time.perf_counter()
    if fwd_ms is None:
        fwd_ms = (t_fwd - t0) * 1000

    out.update({
        # Everything the model proposed that is a road user and is not the
        # bonnet, BEFORE any confidence or duplicate gate. Carried so a caller
        # can re-gate the same frame differently without a second forward pass —
        # which is exactly what tools/detect_quality.py does to produce a
        # before/after on identical pixels.
        "raw": raw,
        "image": {"w": w, "h": h},
        "timing_ms": {
            # GPU kernel time of the model pass. See the docstring.
            "forward": round(fwd_ms, 2),
            "post": round((t_end - t_fwd) * 1000, 2),
            "total": round((t_end - t0) * 1000, 2),
        },
    })
    return out
