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
import os
import sys
import threading
import time

import numpy as np
import torch

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

# Detections below this many pixels tall are past the range where DA-V2 gives a
# usable depth anyway (§7.2), and they dominate the candidate count on an open
# motorway.
MIN_BOX_PX = 12

# --- the ego vehicle's own bonnet -------------------------------------------
# A dashcam sees the host car's bonnet across the bottom of every frame, and
# the detector will occasionally call it a car. Observed on the winding clip:
# `car 0.40` at [2, 661, 1279, 720] -- full frame width, flush with the bottom
# edge, aspect 21.6.
#
# This is the single most dangerous false positive available to this system: it
# lands dead centre in the ego lane at essentially zero range, so it would win
# lead selection outright and read as a stationary object one metre ahead. The
# corridor's MIN_RANGE_M gate is a backstop that would probably catch it, but
# "probably caught downstream" is not how this one should be handled.
#
# The signature is unmistakable and nothing real shares it: a vehicle 3 m ahead
# is wide but also TALL. The aspect bound is what separates them.
BONNET_BOTTOM_FRAC = 0.98    # box bottom is flush with the frame bottom
BONNET_WIDTH_FRAC = 0.85     # ...spanning almost the whole frame
BONNET_MIN_ASPECT = 6.0      # ...and far wider than it is tall


def _is_ego_bonnet(x1, y1, x2, y2, w, h) -> bool:
    bw, bh = x2 - x1, max(y2 - y1, 1e-6)
    return (y2 >= BONNET_BOTTOM_FRAC * h
            and bw >= BONNET_WIDTH_FRAC * w
            and (bw / bh) >= BONNET_MIN_ASPECT)

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
    """Preload weights and pay CUDA kernel warmup off the frame path."""
    _ensure_loaded(variant)
    detect(np.zeros((720, 1280, 3), dtype=np.uint8))


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


def detect(frame, score_min: float = SCORE_MIN, variant: str = None) -> dict:
    """Detect road users in one BGR frame.

    Returns:
        detections  [(label, (x1, y1, x2, y2), score), ...] in FRAME pixels,
                    nearest-looking first (largest box first)
        timing_ms   forward / postprocess / total
    """
    t0 = time.perf_counter()
    _ensure_loaded(variant)
    h, w = frame.shape[:2]

    with _lock:
        with torch.inference_mode():
            out = _model(_preprocess(frame))
            if isinstance(out, tuple):
                out = {"pred_boxes": out[0], "pred_logits": out[1]}
            t_fwd = time.perf_counter()
            sizes = torch.tensor([[h, w]], device=_device)
            res = _postproc({"pred_logits": out["pred_logits"].float(),
                             "pred_boxes": out["pred_boxes"].float()},
                            target_sizes=sizes)[0]
        scores = res["scores"].detach().cpu().numpy()
        labels = res["labels"].detach().cpu().numpy()
        boxes = res["boxes"].detach().cpu().numpy()

    dets = []
    n_bonnet = 0
    for s, c, b in zip(scores, labels, boxes):
        if s < score_min:
            continue
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
        dets.append((name, (x1, y1, x2, y2), float(s)))

    # Largest first, which on a forward-facing camera is nearest first. The
    # membership layer re-ranks by measured depth; this only makes truncation
    # or a debug dump show the vehicles that matter at the top.
    dets.sort(key=lambda d: -((d[1][2] - d[1][0]) * (d[1][3] - d[1][1])))
    t_end = time.perf_counter()

    return {
        "detections": dets,
        "n_bonnet_rejected": n_bonnet,
        "image": {"w": w, "h": h},
        "timing_ms": {
            "forward": round((t_fwd - t0) * 1000, 2),
            "post": round((t_end - t_fwd) * 1000, 2),
            "total": round((t_end - t0) * 1000, 2),
        },
    }
