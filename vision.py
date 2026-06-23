"""Qwen2.5-VL-3B vision module for RIO.
Loads the model once at first use, caches the most recent observation
so llm_interface.get_observation() can pull it cheaply per user turn.
"""
import io
import threading
import torch
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import config
from rio_prompts import OBSERVER_PROMPT

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

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
        print("[vision] Loading Qwen2.5-VL-3B...")
        _processor = AutoProcessor.from_pretrained(MODEL_ID)
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
        )
        print("[vision] Loaded.")


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


def get_observation() -> str:
    """Called by llm_interface on each user turn. Returns the cached observation."""
    return _last_observation if config.VISION_ENABLED else ""
