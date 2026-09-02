"""Qwen3-VL-8B vision module for RIO.
Loads the model once at first use, caches the most recent observation
so llm_interface.get_observation() can pull it cheaply per user turn.

Upgraded from Qwen2.5-VL-3B. Qwen3-VL needs transformers >= 4.57.0 and a
different model class; torch stays at 2.4.1+cu124 (4.57 only wants >= 2.2).
"""
import io
import os
import threading
import time
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import config
from rio_prompts import OBSERVER_PROMPT

MODEL_ID = "Qwen/Qwen3-VL-8B-Instruct"

# Sourced from rio_prompts.py (compiled from behavior bible v1).
# The observer is NOT RIO. It produces a short factual note that RIO reads.
TEACHER_PROMPT = OBSERVER_PROMPT


_processor = None
_model = None
_lock = threading.Lock()
_last_observation = ""
# WHEN that observation was made, and of what. The cache used to be a bare
# string, which was fine while its only reader asked "what did she last see"
# and answered a conversation turn with it. It is not fine now that a live
# answer can be served from it: a caption with no timestamp cannot be told
# apart from a current one, and describing a road the car left thirty seconds
# ago -- confidently, in the present tense -- is the failure this whole
# subsystem exists to prevent.
_last_observed_at = 0.0
_last_observed_frame = None


# Where the weights go. Pinned rather than "auto".
#
# HONEST NOTE ON WHAT THIS DOES AND DOES NOT FIX. It was changed while chasing
# a startup failure -- "Cannot copy out of meta tensor" from inside
# accelerate's dispatch_model -- and it did NOT fix it. That failure was two
# uvicorn processes loading Qwen onto the same card at once; it happens on
# "auto" and on "cuda:0" alike, and stops happening when there is one server.
#
# It is kept pinned anyway, for a smaller and separate reason: "auto" asks
# accelerate to plan a memory budget from whatever the card has free at that
# instant, and to offload whatever it thinks will not fit. On one GPU with
# 97 GB and a 17 GB model there is nothing to plan, and a plan that depends on
# timing is a plan that can differ between two identical restarts. Pinning
# removes a variable rather than a bug.
#
# If this ever has to run on a card too small for the model, that is a
# deliberate change to make here, with sharding thought about on purpose.
DEVICE_MAP = os.environ.get("RIO_QWEN_DEVICE",
                            "cuda:0" if torch.cuda.is_available() else "cpu")


def _ensure_loaded():
    global _processor, _model
    if _model is None:
        print(f"[vision] Loading Qwen3-VL-8B onto {DEVICE_MAP}...", flush=True)
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        # `dtype` replaces the deprecated `torch_dtype` kwarg in transformers 4.57.
        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE_MAP
        )
        print("[vision] Loaded.", flush=True)


def _downscale(pil, max_side: int):
    """Fit the long edge to `max_side`, or leave it alone.

    Prefill scales with vision tokens, and vision tokens scale with pixels. The
    observer's whole job is one short factual sentence about the scene, which
    does not need 720p: measured on this GPU, full frame 432 ms, 768 px 377 ms,
    512 px 360 ms, and the caption at 512 was the same sentence. That is ~17%
    off a call this now makes once a second in the background, and the smaller
    frame leaves that much more of the card for the 4 fps headway loop it runs
    beside.
    """
    if not max_side:
        return pil
    w, h = pil.size
    longest = max(w, h)
    if longest <= max_side:
        return pil
    scale = max_side / float(longest)
    return pil.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                      Image.BILINEAR)


def observe(image_bytes: bytes, max_side: int = None, frame_id=None) -> str:
    """Run VLM on a new frame, cache + return the observation."""
    global _last_observation, _last_observed_at, _last_observed_frame
    if not config.VISION_ENABLED:
        return ""
    if max_side is None:
        max_side = config.OBSERVER_MAX_SIDE_PX
    with _lock:
        _ensure_loaded()
        pil = _downscale(Image.open(io.BytesIO(image_bytes)).convert("RGB"),
                         max_side)
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": TEACHER_PROMPT},
        ]}]
        inputs = _processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(_model.device)
        out = _model.generate(**inputs, max_new_tokens=60, do_sample=False)
        text = _processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
        _last_observation = text
        _last_observed_at = time.time()
        _last_observed_frame = frame_id
        return text


def warm() -> None:
    """Preload weights and compile CUDA kernels so the first real /observe is fast.

    Without this the first frame of a drive pays ~16s of cold-load while the
    5s capture loop is already running.
    """
    if not config.VISION_ENABLED:
        return
    with _lock:
        _ensure_loaded()
        # One throwaway forward pass: loading the weights is most of the cost,
        # but the first generate() also pays kernel warmup. Deliberately does
        # NOT write _last_observation — RIO must not read a dummy frame as if
        # it were the road ahead.
        try:
            pil = Image.new("RGB", (64, 64), (0, 0, 0))
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": TEACHER_PROMPT},
            ]}]
            inputs = _processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(_model.device)
            _model.generate(**inputs, max_new_tokens=1, do_sample=False)
        except Exception as e:
            print("[vision] warm inference skipped:", e)


def get_observation() -> str:
    """Called by llm_interface on each user turn. Returns the cached observation."""
    return _last_observation if config.VISION_ENABLED else ""


def observation() -> dict:
    """The cached observation WITH its age. -> {text, at, age_s, frame_id}

    The age is the whole point. Any caller that would speak this to a driver
    has to be able to decide whether it is still true, and a bare string cannot
    be asked. `text` is empty when nothing has been observed.
    """
    if not config.VISION_ENABLED:
        return {"text": "", "at": 0.0, "age_s": None, "frame_id": None}
    at = _last_observed_at
    return {
        "text": _last_observation,
        "at": at,
        "age_s": (time.time() - at) if at else None,
        "frame_id": _last_observed_frame,
    }


def set_observation(text: str) -> None:
    """Publish an observation produced by another path (e.g. /perceive).

    Drive mode calls /perceive instead of /observe, so without this the cache
    that get_observation() serves to RIO would never refresh and she would talk
    about the road as if the last /observe frame were still current.
    """
    global _last_observation, _last_observed_at
    if text:
        _last_observation = text
        _last_observed_at = time.time()


def get_handles():
    """(processor, model, lock) for the one loaded Qwen3-VL instance.

    Lets headway.anchor ground boxes on the model the app already has resident
    instead of pulling a second ~17 GB copy. The load happens under `_lock` and
    the lock is then released before returning: callers take it themselves
    around generate(), and threading.Lock is not reentrant.
    """
    with _lock:
        _ensure_loaded()
    return _processor, _model, _lock
