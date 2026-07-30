"""UFLDv2 lane geometry — where the ego lane actually is.

Design ref: docs/live_headway_v3.md §2 (ego corridor). This replaces the
assumption at the heart of the Stage 0 corridor — "the lane is a fixed trapezoid
straight ahead of a level camera" — with a measurement of the two painted lines
the car is actually between.

Upstream: Ultra-Fast-Lane-Detection-v2 (github.com/cfzd/Ultra-Fast-Lane-Detection-v2),
MIT licence, Qin et al., "Ultra Fast Deep Lane Detection with Hybrid Anchor
Driven Ordinal Classification" (TPAMI 2022). Variant: culane_res18 — CULane,
ResNet-18 backbone, 4 lanes, row/column hybrid anchors.

WHY THE ARCHITECTURE IS RE-STATED HERE RATHER THAN IMPORTED
-----------------------------------------------------------
The upstream repo is a training harness: importing its `parsingNet` pulls in
`utils.common` -> `utils.dist_utils` -> torch.distributed init, NVIDIA DALI, a
mmcv-style Config loader and a stray `import pdb`, and its backbone calls
`torchvision.models.resnet18(pretrained=True)`, which both reaches the network
at construction time and rides a kwarg torchvision has been deprecating. None of
that survives contact with a live service.

The network itself is forty lines (ResNet-18 trunk -> 1x1 conv to 8 channels ->
two-layer MLP -> four flat prediction heads), so it is restated below and the
OFFICIAL published checkpoint is loaded into it unchanged. `selftest.py`
verifies the state dict loads with zero missing and zero unexpected keys, which
is what makes "this is really UFLDv2 and not something shaped like it" a checked
claim rather than a hope. Decoding follows `demo.py:pred2coords` exactly,
vectorised onto the GPU; `test_decode_matches_reference` pins it against a
literal transcription of the upstream loop.

ONNX was the stated fallback if the torch path fought torch 2.11. It did not:
weights_only=True loads the checkpoint cleanly and the trunk is stock
torchvision, so the torch path stays. It is the cleaner integration by a
distance — no 250 MB onnxruntime-gpu dependency, no second CUDA context
alongside Depth Anything, no export artefact to keep in sync with the weights,
and fp16 lands inside the frame budget anyway (see selftest timings).

COST
----
One forward at 320x1600, fp16, batch 1. Test-time augmentation (`forward_tta`,
a 5x batch) is deliberately NOT used: it buys a little accuracy for 5x the
compute, and this is a per-frame path.

WHAT THIS MODULE MAY AND MAY NOT DO
-----------------------------------
It reports geometry. It does not decide anything. The corridor-source switch
lives in anchor.py, the confidence contribution in state.py, and the drift
monitor here emits a *log record* and nothing else — no voice line, no steering
number, no suggestion of one. See LaneDriftMonitor.
"""
import math
import os
import threading
import time

import numpy as np
import torch

# ---------------------------------------------------------------------------
# culane_res18 configuration. Mirrors configs/culane_res18.py upstream; these
# are properties of the published checkpoint, not tunables.
# ---------------------------------------------------------------------------
BACKBONE = "18"
TRAIN_WIDTH = 1600
TRAIN_HEIGHT = 320
CROP_RATIO = 0.6            # frame is resized to H/CROP_RATIO then bottom-cropped
NUM_ROW = 72                # row anchors (near-vertical lanes)
NUM_COL = 81                # column anchors (near-horizontal lanes)
NUM_CELL_ROW = 200          # ordinal classification cells across the row
NUM_CELL_COL = 100
NUM_LANES = 4
FC_NORM = True
LOCAL_WIDTH = 1             # +/- cells averaged around the argmax, as upstream

# Row anchors span 0.42..1.0 of ORIGINAL image height, columns 0..1 of width
# (utils/common.py, CULane branch). Because 0.42 > 1 - CROP_RATIO = 0.40, every
# row anchor lands inside the cropped region, so decoded coordinates map
# straight back to full-frame pixels with no crop bookkeeping.
ROW_ANCHOR = np.linspace(0.42, 1.0, NUM_ROW)
COL_ANCHOR = np.linspace(0.0, 1.0, NUM_COL)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# CULane orders its four lanes left-to-right, and upstream's own evaluation
# reads lanes 1 and 2 off the row branch (they are the near-vertical ego
# boundaries) and lanes 0 and 3 off the column branch. We decode every lane from
# whichever branch declares it, because after a lane change the ego pair is no
# longer 1 and 2 — the pair is chosen geometrically below, never by index.
ROW_FIRST_LANES = (1, 2)

DEFAULT_WEIGHTS = os.environ.get(
    "RIO_LANE_WEIGHTS", "/workspace/rio-phase1/weights/culane_res18.pth")

# ---------------------------------------------------------------------------
# Geometry gates
# ---------------------------------------------------------------------------
# Confidence at or above this and the corridor is built from paint; below it the
# static trapezoid takes over. Worn paint, night, rain and snow all land below.
LANE_CONF_MIN = 0.55

# A lane the net only asserts over half its anchors is still a real lane (it
# just starts halfway up the frame) -- upstream's own validity gate is exactly
# "more than half the row anchors exist". So coverage earns full credit at 50%
# rather than being scored linearly.
ROW_COVERAGE_FULL = 0.5
COL_COVERAGE_FULL = 0.25
MIN_LANE_POINTS = 6

# Where "the bottom of the image" is sampled for ego-pair selection. Not the
# very last row: on most dashcams that is bonnet, wiper or vignette.
BOTTOM_REF_FRAC = 0.95

# Extrapolating a lane below its lowest detected point is fine for a few rows
# and nonsense for a hundred. Past this (as a fraction of frame height) the
# boundary is treated as absent rather than invented.
MAX_EXTRAP_FRAC = 0.20

# Plausible ego-lane width at the bottom reference row, as a fraction of frame
# width. Deliberately generous: this rejects garbage pairings, it does not
# second-guess a real lane on an unusual mounting. Outside the plateau the
# geometry factor decays to zero at the hard bounds.
WIDTH_FRAC_PLATEAU = (0.15, 1.60)
WIDTH_FRAC_HARD = (0.08, 2.50)


# ---------------------------------------------------------------------------
# Network — restated from model/model_culane.py + model/backbone.py upstream.
# Submodule names are load-bearing: they are the checkpoint's key prefixes.
# ---------------------------------------------------------------------------
class _ResNetTrunk(torch.nn.Module):
    """torchvision ResNet-18 split so layer2/3/4 are all reachable.

    Upstream's `resnet` wrapper returns (x2, x3, x4). Only x4 feeds the lane
    heads; x2/x3 exist for the optional segmentation aux head, which
    culane_res18 does not use (use_aux=False), so they are dropped here.
    """

    def __init__(self):
        super().__init__()
        import torchvision
        m = torchvision.models.resnet18(weights=None)
        self.conv1 = m.conv1
        self.bn1 = m.bn1
        self.relu = m.relu
        self.maxpool = m.maxpool
        self.layer1 = m.layer1
        self.layer2 = m.layer2
        self.layer3 = m.layer3
        self.layer4 = m.layer4

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.layer4(x)


class ParsingNet(torch.nn.Module):
    """UFLDv2 hybrid-anchor head. Four flat outputs, sliced into row/col
    location logits and row/col existence logits."""

    def __init__(self, input_height=TRAIN_HEIGHT, input_width=TRAIN_WIDTH):
        super().__init__()
        self.num_grid_row = NUM_CELL_ROW
        self.num_cls_row = NUM_ROW
        self.num_grid_col = NUM_CELL_COL
        self.num_cls_col = NUM_COL
        self.num_lane_on_row = NUM_LANES
        self.num_lane_on_col = NUM_LANES

        self.dim1 = NUM_CELL_ROW * NUM_ROW * NUM_LANES      # row locations
        self.dim2 = NUM_CELL_COL * NUM_COL * NUM_LANES      # col locations
        self.dim3 = 2 * NUM_ROW * NUM_LANES                 # row existence
        self.dim4 = 2 * NUM_COL * NUM_LANES                 # col existence
        self.total_dim = self.dim1 + self.dim2 + self.dim3 + self.dim4

        self.input_dim = input_height // 32 * input_width // 32 * 8

        self.model = _ResNetTrunk()
        self.pool = torch.nn.Conv2d(512, 8, 1)
        self.cls = torch.nn.Sequential(
            torch.nn.LayerNorm(self.input_dim) if FC_NORM else torch.nn.Identity(),
            torch.nn.Linear(self.input_dim, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, self.total_dim),
        )

    def forward(self, x):
        fea = self.pool(self.model(x))
        out = self.cls(fea.view(-1, self.input_dim))
        d1, d2, d3, d4 = self.dim1, self.dim2, self.dim3, self.dim4
        return {
            "loc_row": out[:, :d1].view(-1, self.num_grid_row, self.num_cls_row,
                                        self.num_lane_on_row),
            "loc_col": out[:, d1:d1 + d2].view(-1, self.num_grid_col, self.num_cls_col,
                                               self.num_lane_on_col),
            "exist_row": out[:, d1 + d2:d1 + d2 + d3].view(-1, 2, self.num_cls_row,
                                                           self.num_lane_on_row),
            "exist_col": out[:, -d4:].view(-1, 2, self.num_cls_col,
                                           self.num_lane_on_col),
        }


def _load_state_dict(net: torch.nn.Module, path: str) -> None:
    """Load the official checkpoint, strictly.

    Upstream loads with strict=False, which is how a renamed layer silently
    becomes a randomly-initialised one. Here anything missing or unexpected is
    an error: a lane model that quietly half-loaded would produce confident
    nonsense, and confident nonsense is the one output this system cannot
    tolerate.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    sd = ckpt.get("model", ckpt) if isinstance(ckpt, dict) else ckpt
    # Trained under DistributedDataParallel, so every key carries "module.".
    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in sd.items()}
    missing, unexpected = net.load_state_dict(sd, strict=False)
    # The aux segmentation head is in neither the checkpoint nor this model.
    if missing or unexpected:
        raise RuntimeError(
            f"UFLDv2 checkpoint does not match the model: "
            f"{len(missing)} missing {missing[:4]}, "
            f"{len(unexpected)} unexpected {unexpected[:4]}")


# ---------------------------------------------------------------------------
# Decoding — vectorised transcription of demo.py:pred2coords
# ---------------------------------------------------------------------------
def _decode_branch(loc, exist, grid_n, local_width=LOCAL_WIDTH):
    """Expected-value decode of one anchor branch.

    For each (anchor, lane) upstream takes the argmax cell, softmaxes the
    logits in a +/-local_width window around it, and reads the probability-
    weighted mean cell index (+0.5 for cell centre). Out-of-range window cells
    are dropped -- masked to -inf here rather than clamped, because clamping
    would duplicate a border cell's logit and change its softmax weight.

    Returns (pos, valid, p_exist), each (K, L) numpy: sub-cell position in cell
    units, the existence decision, and its probability.
    """
    logits = loc.permute(0, 2, 3, 1)                     # (1, K, L, G)
    max_idx = logits.argmax(-1)                          # (1, K, L)

    offs = torch.arange(-local_width, local_width + 1, device=loc.device)
    idx = max_idx.unsqueeze(-1) + offs                   # (1, K, L, w)
    in_bounds = (idx >= 0) & (idx <= grid_n - 1)
    win = torch.gather(logits, 3, idx.clamp(0, grid_n - 1))
    win = win.masked_fill(~in_bounds, float("-inf"))
    weights = win.float().softmax(-1)
    pos = (weights * idx.clamp(0, grid_n - 1).float()).sum(-1) + 0.5

    ex = exist.float()
    p_exist = ex.softmax(1)[:, 1]                        # (1, K, L)
    valid = ex.argmax(1).bool()                          # (1, K, L)

    return (pos[0].cpu().numpy(), valid[0].cpu().numpy(),
            p_exist[0].cpu().numpy())


def _lane_from_row(pos, valid, p_exist, lane, width, height):
    """One lane's polyline off the row branch, or None."""
    v = valid[:, lane]
    n_valid = int(v.sum())
    if n_valid <= NUM_ROW / 2 or n_valid < MIN_LANE_POINTS:
        return None                                       # upstream's gate
    ks = np.nonzero(v)[0]
    xs = pos[ks, lane] / (NUM_CELL_ROW - 1) * width
    ys = ROW_ANCHOR[ks] * height
    conf = _lane_confidence(n_valid / NUM_ROW, ROW_COVERAGE_FULL,
                            float(p_exist[ks, lane].mean()))
    return {"points": list(zip(xs.tolist(), ys.tolist())), "confidence": conf,
            "source": "row", "lane_index": int(lane)}


def _lane_from_col(pos, valid, p_exist, lane, width, height):
    """One lane's polyline off the column branch, or None."""
    v = valid[:, lane]
    n_valid = int(v.sum())
    if n_valid <= NUM_COL / 4 or n_valid < MIN_LANE_POINTS:
        return None                                       # upstream's gate
    ks = np.nonzero(v)[0]
    xs = COL_ANCHOR[ks] * width
    ys = pos[ks, lane] / (NUM_CELL_COL - 1) * height
    order = np.argsort(ys)                                # keep y ascending
    conf = _lane_confidence(n_valid / NUM_COL, COL_COVERAGE_FULL,
                            float(p_exist[ks, lane].mean()))
    return {"points": list(zip(xs[order].tolist(), ys[order].tolist())),
            "confidence": conf, "source": "col", "lane_index": int(lane)}


def _lane_confidence(coverage, coverage_full, certainty):
    """How much of the lane the net asserts, times how sure it is where it does."""
    cov = min(1.0, coverage / coverage_full) if coverage_full > 0 else 0.0
    return float(max(0.0, min(1.0, cov * certainty)))


def x_at_y(points, y, max_extrap):
    """Lane x at row y by linear interpolation. -> (x, extrapolated) or None.

    `points` is y-ascending. Below the lowest point the last segment is
    extended, which is what reaching the bottom of the frame usually needs;
    beyond `max_extrap` pixels the answer is refused rather than guessed.
    """
    n = len(points)
    if n < 2:
        return None
    y = float(y)
    y0, y1 = points[0][1], points[-1][1]

    if y < y0:
        if y0 - y > max_extrap:
            return None
        (xa, ya), (xb, yb) = points[0], points[1]
    elif y > y1:
        if y - y1 > max_extrap:
            return None
        (xa, ya), (xb, yb) = points[-2], points[-1]
    else:
        for i in range(n - 1):
            if points[i][1] <= y <= points[i + 1][1]:
                (xa, ya), (xb, yb) = points[i], points[i + 1]
                break
        else:
            return None
        if abs(yb - ya) < 1e-6:
            return float(xa), False
        t = (y - ya) / (yb - ya)
        return float(xa + t * (xb - xa)), False

    if abs(yb - ya) < 1e-6:
        return None
    t = (y - ya) / (yb - ya)
    return float(xa + t * (xb - xa)), True


def _width_factor(width_frac):
    """Soft plausibility of an ego-lane width, 0..1."""
    lo_hard, hi_hard = WIDTH_FRAC_HARD
    lo_ok, hi_ok = WIDTH_FRAC_PLATEAU
    if width_frac <= lo_hard or width_frac >= hi_hard:
        return 0.0
    if width_frac < lo_ok:
        return (width_frac - lo_hard) / (lo_ok - lo_hard)
    if width_frac > hi_ok:
        return (hi_hard - width_frac) / (hi_hard - hi_ok)
    return 1.0


def select_ego_pair(lanes, width, height):
    """The two lanes bracketing image centre at the bottom. -> (info, conf).

    Ego identity is decided by where the paint is, not by CULane's lane
    numbering: after a lane change or on a slip road the ego pair is no longer
    lanes 1 and 2, and trusting the index there would build the corridor around
    the wrong strip of road.
    """
    cx = width * 0.5
    y_ref = height * BOTTOM_REF_FRAC
    max_extrap = height * MAX_EXTRAP_FRAC

    left = right = None      # (x_at_ref, lane)
    for lane in lanes:
        hit = x_at_y(lane["points"], y_ref, max_extrap)
        if hit is None:
            continue
        x, extrapolated = hit
        lane["x_bottom"] = x
        lane["extrapolated"] = extrapolated
        if x < cx and (left is None or x > left[0]):
            left = (x, lane)
        elif x >= cx and (right is None or x < right[0]):
            right = (x, lane)

    if left is None or right is None:
        return {
            "ego_left": None, "ego_right": None,
            "reason": "no_bracketing_pair" if (left or right) else "no_lanes",
        }, 0.0

    xl, lane_l = left
    xr, lane_r = right
    width_frac = (xr - xl) / width
    geom = _width_factor(width_frac)

    # The pair is only as trustworthy as its weaker boundary: a corridor with
    # one confident edge and one guessed edge is a guessed corridor.
    conf = min(lane_l["confidence"], lane_r["confidence"]) * geom

    return {
        "ego_left": lane_l["points"],
        "ego_right": lane_r["points"],
        "ego_left_conf": lane_l["confidence"],
        "ego_right_conf": lane_r["confidence"],
        "x_bottom_left": xl,
        "x_bottom_right": xr,
        "y_ref": y_ref,
        "lane_width_frac": width_frac,
        "geom_factor": geom,
        "extrapolated": bool(lane_l["extrapolated"] or lane_r["extrapolated"]),
        "reason": "ok" if conf > 0 else "implausible_width",
    }, float(max(0.0, min(1.0, conf)))


# ---------------------------------------------------------------------------
# Detector
# ---------------------------------------------------------------------------
_net = None
_device = None
_dtype = None
_mean = None
_std = None
_lock = threading.Lock()
_load_error = None


def _ensure_loaded(weights_path: str = None) -> None:
    global _net, _device, _dtype, _mean, _std, _load_error
    if _net is not None:
        return
    if _load_error is not None:
        raise RuntimeError(_load_error)

    path = weights_path or DEFAULT_WEIGHTS
    if not os.path.exists(path):
        _load_error = (
            f"UFLDv2 weights not found at {path}. Fetch culane_res18.pth from "
            "the model zoo in the upstream README and set RIO_LANE_WEIGHTS.")
        raise RuntimeError(_load_error)

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    # fp16 on CUDA only, matching depth.py: on CPU it is slower than fp32 and
    # some ops have no half kernel, and the CPU path exists for unit tests.
    _dtype = torch.float16 if _device == "cuda" else torch.float32

    net = ParsingNet()
    _load_state_dict(net, path)
    _net = net.to(device=_device, dtype=_dtype).eval()

    _mean = torch.tensor(IMAGENET_MEAN, device=_device,
                         dtype=torch.float32).view(1, 3, 1, 1)
    _std = torch.tensor(IMAGENET_STD, device=_device,
                        dtype=torch.float32).view(1, 3, 1, 1)


def warm(weights_path: str = None) -> None:
    """Preload weights and pay CUDA kernel warmup off the frame path."""
    _ensure_loaded(weights_path)
    detect_lanes(np.zeros((720, 1280, 3), dtype=np.uint8))


def available() -> bool:
    """True if lane detection can run — weights present and loadable."""
    if _net is not None:
        return True
    if _load_error is not None:
        return False
    return os.path.exists(DEFAULT_WEIGHTS)


def _preprocess(frame_bgr):
    """BGR uint8 frame -> normalised (1,3,320,1600) tensor on the device.

    Upstream's transform is PIL Resize(H/CROP_RATIO, W) -> ToTensor ->
    Normalize -> bottom crop to H. Done on the GPU here so a 1280x720 frame
    costs a memcpy instead of a CPU resize; antialias=True keeps the downscale
    close to PIL's bilinear, which is what the weights were trained against.
    """
    resized_h = int(TRAIN_HEIGHT / CROP_RATIO)      # 533
    t = torch.from_numpy(np.ascontiguousarray(frame_bgr)).to(_device)
    t = t.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
    t = t.flip(1)                                   # BGR -> RGB
    t = torch.nn.functional.interpolate(
        t, size=(resized_h, TRAIN_WIDTH), mode="bilinear",
        align_corners=False, antialias=True)
    t = (t - _mean) / _std
    return t[:, :, -TRAIN_HEIGHT:, :].to(_dtype)    # bottom crop


def detect_lanes(frame, weights_path: str = None) -> dict:
    """Detect lanes in one BGR frame.

    Returns:
        lanes       list of polylines, each [(x, y), ...] in frame pixels,
                    y ascending (far to near)
        ego_left    polyline of the left ego boundary, or None
        ego_right   polyline of the right ego boundary, or None
        confidence  0..1 for the EGO PAIR specifically -- the number the
                    corridor-source switch reads. Individual lanes carry their
                    own confidence in `lane_conf`.
    """
    t0 = time.perf_counter()
    _ensure_loaded(weights_path)

    h, w = frame.shape[:2]
    with _lock:
        with torch.inference_mode():
            pred = _net(_preprocess(frame))
        t_fwd = time.perf_counter()

        row_pos, row_valid, row_pe = _decode_branch(
            pred["loc_row"], pred["exist_row"], NUM_CELL_ROW)
        col_pos, col_valid, col_pe = _decode_branch(
            pred["loc_col"], pred["exist_col"], NUM_CELL_COL)

    lanes = []
    for i in range(NUM_LANES):
        # Row anchors first for the near-vertical lanes, columns first for the
        # outer ones -- the branch upstream's own evaluation trusts per lane.
        # Either way the other branch is a fallback, so a lane the preferred
        # branch misses is not silently lost.
        order = ((_lane_from_row, row_pos, row_valid, row_pe),
                 (_lane_from_col, col_pos, col_valid, col_pe))
        if i not in ROW_FIRST_LANES:
            order = order[::-1]
        for fn, pos, valid, pe in order:
            lane = fn(pos, valid, pe, i, w, h)
            if lane is not None:
                lanes.append(lane)
                break

    ego, confidence = select_ego_pair(lanes, w, h)
    t_end = time.perf_counter()

    return {
        "lanes": [lane["points"] for lane in lanes],
        "lane_conf": [round(lane["confidence"], 3) for lane in lanes],
        "lane_index": [lane["lane_index"] for lane in lanes],
        "ego_left": ego["ego_left"],
        "ego_right": ego["ego_right"],
        "confidence": round(confidence, 3),
        "ego": ego,
        "image": {"w": w, "h": h},
        "timing_ms": {
            "forward": round((t_fwd - t0) * 1000, 2),
            "decode": round((t_end - t_fwd) * 1000, 2),
            "total": round((t_end - t0) * 1000, 2),
        },
    }


# ---------------------------------------------------------------------------
# Lane departure — ADVISORY LOGGING ONLY
# ---------------------------------------------------------------------------
# Deliberate scope limit. This monitor writes a `lane_drift` record to the
# session JSONL and returns it to the caller. It has no voice line, no audio
# clip, and no path to one: live_policy.py is where speech is decided and it is
# not wired to this. That is a product decision, not an oversight -- drift
# thresholds are meaningless until they have been scored against real drives,
# and a lane-keeping prompt that fires on a deliberate lane change teaches the
# driver to ignore the system.
#
# It is also not, and must never become, steering guidance. It reports that a
# drift happened. It does not say which way to correct, by how much, or when.
DRIFT_RATIO = 0.70          # |offset| beyond 70% of the way to a boundary
DRIFT_HOLD_S = 1.0          # ...held this long
DRIFT_REARM_RATIO = 0.50    # ...and recentred to here before it can fire again
DRIFT_MIN_CONF = LANE_CONF_MIN


class LaneDriftMonitor:
    """Sustained lateral offset within the ego lane. Logs; never speaks.

    Offset convention: 0.0 is the ego-lane centreline, +1.0 the right boundary,
    -1.0 the left. `center_bias` subtracts a fixed offset for a camera that is
    not mounted on the vehicle centreline -- a dash-corner mount reads a
    permanent lean otherwise. Left at 0.0 until real-drive logs say what it
    should be; the logged raw offset is what that calibration gets read from.
    """

    def __init__(self, ratio=DRIFT_RATIO, hold_s=DRIFT_HOLD_S,
                 rearm_ratio=DRIFT_REARM_RATIO, min_conf=DRIFT_MIN_CONF,
                 center_bias=0.0):
        # The re-arm point must sit INSIDE the drift threshold, or the two
        # bands overlap and every frame of one excursion both fires and
        # re-arms -- one drift becomes an event per frame for as long as it
        # lasts. The shipped defaults (0.70 / 0.50) are fine; this guard is for
        # whoever tunes DRIFT_RATIO down after reading the first real drive and
        # does not think to move DRIFT_REARM_RATIO with it. Caught here, at
        # construction, rather than as a flood of records after the drive.
        if not 0.0 <= float(rearm_ratio) < float(ratio):
            raise ValueError(
                f"rearm_ratio ({rearm_ratio}) must be >= 0 and strictly less "
                f"than ratio ({ratio}); otherwise the monitor re-arms inside "
                "its own drift band and reports one excursion many times")
        if float(hold_s) <= 0.0:
            raise ValueError(f"hold_s must be > 0, got {hold_s}")
        self.ratio = float(ratio)
        self.hold_s = float(hold_s)
        self.rearm_ratio = float(rearm_ratio)
        self.min_conf = float(min_conf)
        self.center_bias = float(center_bias)

        self.armed = True           # False after firing, until recentred
        self.side = None            # 'left'/'right' of the excursion in progress
        self.since_t = None         # when the current excursion started
        self.last_offset = None
        self.n_events = 0

    def reset(self) -> None:
        self.armed = True
        self.side = None
        self.since_t = None
        self.last_offset = None

    @staticmethod
    def offset(lane_result, width, height):
        """Signed lateral position in the ego lane, or None.

        +ve means the camera axis sits right of the lane centreline, i.e. the
        car is toward the right-hand boundary.
        """
        ego = (lane_result or {}).get("ego") or {}
        xl, xr = ego.get("x_bottom_left"), ego.get("x_bottom_right")
        if xl is None or xr is None:
            return None
        half = (xr - xl) / 2.0
        if half <= 1e-6:
            return None
        return float((width * 0.5 - (xl + xr) / 2.0) / half)

    def update(self, lane_result, width, height, t, confidence) -> dict:
        """One frame. Returns an event dict on the frame drift is confirmed."""
        off = self.offset(lane_result, width, height)
        if off is not None:
            off -= self.center_bias
        self.last_offset = off

        # Low-confidence lanes cannot sustain or refute an excursion. The timer
        # is dropped rather than coasted: an event confirmed across a gap of
        # unreadable paint is an event we did not actually observe.
        if off is None or confidence < self.min_conf:
            self.side = None
            self.since_t = None
            return {"offset": None, "drift": False, "held_s": 0.0,
                    "reason": "no_lane_confidence"}

        if abs(off) < self.rearm_ratio:
            self.armed = True

        side = "right" if off > 0 else "left"
        if abs(off) < self.ratio or (self.side is not None and side != self.side):
            self.side = side if abs(off) >= self.ratio else None
            self.since_t = float(t) if self.side else None
            return {"offset": round(off, 3), "drift": False, "held_s": 0.0,
                    "reason": "within_lane" if abs(off) < self.ratio
                              else "side_changed"}

        if self.since_t is None:
            self.since_t = float(t)
            self.side = side
        held = float(t) - self.since_t

        if held < self.hold_s or not self.armed:
            return {"offset": round(off, 3), "drift": False,
                    "held_s": round(held, 3),
                    "reason": "holding" if self.armed else "already_reported"}

        self.armed = False
        self.n_events += 1
        return {
            "offset": round(off, 3),
            "drift": True,
            "side": side,
            "held_s": round(held, 3),
            "threshold": self.ratio,
            "confidence": round(float(confidence), 3),
            "event_index": self.n_events,
            "reason": "drift_confirmed",
        }
