"""Qwen3-VL-8B vision module for RIO.
Loads the model once at first use, caches the most recent observation
so llm_interface.get_observation() can pull it cheaply per user turn.

Upgraded from Qwen2.5-VL-3B. Qwen3-VL needs transformers >= 4.57.0 and a
different model class; torch stays at 2.4.1+cu124 (4.57 only wants >= 2.2).
"""
import io
import threading
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


def _ensure_loaded():
    global _processor, _model
    if _model is None:
        print("[vision] Loading Qwen3-VL-8B...", flush=True)
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        # `dtype` replaces the deprecated `torch_dtype` kwarg in transformers 4.57.
        _model = Qwen3VLForConditionalGeneration.from_pretrained(
            MODEL_ID, dtype=torch.bfloat16, device_map="auto"
        )
        print("[vision] Loaded.", flush=True)


def observe(image_bytes: bytes) -> str:
    """Run VLM on a new frame, cache + return the observation."""
    global _last_observation
    if not config.VISION_ENABLED:
        return ""
    with _lock:
        _ensure_loaded()
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
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
