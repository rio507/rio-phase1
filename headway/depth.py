"""Depth Anything V2 Metric-Small loader — metric depth in metres.

Design ref: headway_design.md §1 (fast loop step 2), §4 (depth_conf), §10 (MVP).

LICENSE CONSTRAINT (§10): only the **Small** metric variant is Apache-2.0.
Depth-Anything-V2-Metric-{Base,Large} are CC-BY-NC-4.0 and must never be used
here. `_assert_apache_small()` enforces that at load time rather than trusting
the caller to remember.

Stage 0 runs this on an L40S in fp16 (~12 ms/frame at 810x1080). Stage 1 swaps
the backend for TensorRT on Orin; the `depth_map()` contract stays identical.
"""
import numpy as np
import torch

# Outdoor variant: trained for driving-range metric depth (0-80 m), which is the
# regime that matters here. The indoor checkpoint is cached too but is wrong for
# this application -- it compresses everything past ~10 m.
MODEL_ID = "depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf"

# Beyond this range DA-V2 flattens badly (failure mode §7.2). Depth is still
# returned, but roi_depth() confidence-gates on it.
MAX_TRUSTED_RANGE_M = 50.0

_model = None
_processor = None
_device = None


def _assert_apache_small(model_id: str) -> None:
    """Guard §10's licence constraint. Base/Large metric weights are CC-BY-NC."""
    lowered = model_id.lower()
    if "small" not in lowered:
        raise ValueError(
            f"Refusing to load {model_id!r}: headway requires the Apache-2.0 "
            "Metric-Small variant. Base/Large metric checkpoints are "
            "CC-BY-NC-4.0 and cannot ship in this product (design §10)."
        )


def _ensure_loaded() -> None:
    global _model, _processor, _device
    if _model is not None:
        return

    _assert_apache_small(MODEL_ID)
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    _device = "cuda" if torch.cuda.is_available() else "cpu"
    # fp16 on CUDA only -- fp16 on CPU is slower than fp32 and unsupported for
    # some ops, so the CPU fallback (used by unit tests) stays fp32.
    dtype = torch.float16 if _device == "cuda" else torch.float32

    try:
        _processor = AutoImageProcessor.from_pretrained(MODEL_ID, use_fast=True)
    except (TypeError, ValueError):
        # Older//slow-only processors reject use_fast; the slow path is correct,
        # just a few ms slower per frame.
        _processor = AutoImageProcessor.from_pretrained(MODEL_ID)

    _model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, dtype=dtype)
    _model = _model.to(_device).eval()


def warm() -> None:
    """Preload weights + pay CUDA kernel warmup so frame 1 of a clip isn't slow."""
    _ensure_loaded()
    dummy = np.zeros((64, 64, 3), dtype=np.uint8)
    depth_map(dummy)


def depth_map(frame: np.ndarray) -> np.ndarray:
    """BGR uint8 HxWx3 frame -> HxW float32 array of metres, same H/W as input.

    BGR because every other consumer in this module is OpenCV-native; converting
    once here keeps the colour-order question out of the call sites.
    """
    _ensure_loaded()
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"expected HxWx3 BGR frame, got shape {frame.shape}")

    h, w = frame.shape[:2]

    # Preprocessing, not inference, is the bottleneck here: on 720p the CPU-side
    # resize+normalise measured ~70 ms/frame against ~12 ms for the model itself,
    # which alone put the loop under the 10 Hz floor. Handing the processor a
    # CUDA tensor moves that work to the GPU and costs ~3 ms instead.
    if _device == "cuda":
        try:
            t = torch.from_numpy(frame).to(_device, non_blocking=True)
            t = t[:, :, [2, 1, 0]].permute(2, 0, 1).contiguous()  # BGR HWC -> RGB CHW
            inputs = _processor(images=t, return_tensors="pt", device=_device)
            inputs = {k: v for k, v in inputs.items()}
            inputs["pixel_values"] = inputs["pixel_values"].half()
        except (TypeError, ValueError):
            inputs = _cpu_preprocess(frame)
    else:
        inputs = _cpu_preprocess(frame)

    with torch.inference_mode():
        predicted = _model(**inputs).predicted_depth

    # The model emits depth at its own working resolution; resample to the frame
    # so ROI boxes in pixel coords index straight into the result. float() first
    # because bicubic interpolate is not implemented for fp16 on all backends.
    resized = torch.nn.functional.interpolate(
        predicted[:, None].float(), size=(h, w), mode="bicubic", align_corners=False
    )[0, 0]
    return resized.cpu().numpy().astype(np.float32)


def _cpu_preprocess(frame: np.ndarray) -> dict:
    """Fallback path: numpy in, tensors on `_device`.

    ascontiguousarray, not a bare ::-1 view -- the reversed view has a negative
    stride and the fast image processor calls torch.from_numpy on it, which
    rejects negative strides outright.
    """
    rgb = np.ascontiguousarray(frame[:, :, ::-1])  # BGR -> RGB
    inputs = _processor(images=rgb, return_tensors="pt")
    inputs = {k: v.to(_device) for k, v in inputs.items()}
    if _device == "cuda":
        inputs["pixel_values"] = inputs["pixel_values"].half()
    return inputs


def roi_depth(depth: np.ndarray, box, shrink: float = 0.6):
    """Median depth inside an ROI plus a depth_conf score, per §4 and §2.

    Returns (d_raw_m, depth_conf, stats). depth_conf is 0-1 and feeds
    `R = R0 / depth_conf` in the Kalman update -- high ROI variance or few valid
    pixels inflates measurement noise instead of poisoning the state.

    Median, never mean (§4): a box around a car inevitably catches road under it
    and sky/background around it, and those outliers are exactly what a mean
    would smear into the estimate.

    `shrink` samples the centre 60% of the box. The border pixels of any tracked
    box are the least reliable -- tracker slop puts background there -- so the
    centre crop is both more stable and a better sample of the vehicle body.
    """
    h, w = depth.shape[:2]
    x1, y1, x2, y2 = _clamp_box(box, w, h)
    if x2 <= x1 or y2 <= y1:
        return float("nan"), 0.0, {"valid_frac": 0.0, "n_px": 0}

    bw, bh = x2 - x1, y2 - y1
    mx, my = int(bw * (1 - shrink) / 2), int(bh * (1 - shrink) / 2)
    patch = depth[y1 + my : y2 - my, x1 + mx : x2 - mx]
    if patch.size == 0:
        patch = depth[y1:y2, x1:x2]

    finite = np.isfinite(patch)
    # 0 m and negatives are model failures, not real readings; drop them so they
    # cannot drag the median down and fake a dangerously close lead.
    valid = finite & (patch > 0.1)
    n_valid = int(valid.sum())
    if n_valid < 8:
        return float("nan"), 0.0, {"valid_frac": 0.0, "n_px": int(patch.size)}

    vals = patch[valid]
    d = float(np.median(vals))
    valid_frac = n_valid / float(patch.size)

    # Relative spread is the scale-free way to ask "is this ROI one surface or a
    # mix of car and horizon?". MAD rather than std: robust to the same outliers
    # the median is chosen to survive.
    mad = float(np.median(np.abs(vals - d)))
    rel_spread = mad / max(d, 1e-3)

    spread_score = float(np.clip(1.0 - rel_spread / 0.25, 0.0, 1.0))
    valid_score = float(np.clip(valid_frac / 0.6, 0.0, 1.0))
    # §7.2: DA-V2 flattens past ~50 m, so trust decays with range rather than
    # cutting off hard -- a cliff would make states flicker at the boundary.
    range_score = float(np.clip(1.0 - max(0.0, d - MAX_TRUSTED_RANGE_M) / 30.0, 0.1, 1.0))

    depth_conf = float(np.clip(spread_score * valid_score * range_score, 0.0, 1.0))
    stats = {
        "valid_frac": round(valid_frac, 4),
        "n_px": int(patch.size),
        "mad_m": round(mad, 3),
        "rel_spread": round(rel_spread, 4),
        "spread_score": round(spread_score, 4),
        "valid_score": round(valid_score, 4),
        "range_score": round(range_score, 4),
    }
    return d, depth_conf, stats


def _clamp_box(box, w: int, h: int):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1 = max(0, min(x1, w - 1))
    x2 = max(0, min(x2, w))
    y1 = max(0, min(y1, h - 1))
    y2 = max(0, min(y2, h))
    return x1, y1, x2, y2
