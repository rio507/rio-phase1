#!/usr/bin/env python
"""Cosmos-Reason2-8B, behind a socket.

    /opt/teachers/venvs/cosmos/bin/python -m teachers.service.cosmos_service \
        --port 8802 --precision bf16

RUN IT WITH THE COSMOS VENV. transformers 4.57.3 and torch 2.9 -- a different
set from Alpamayo's and a different set again from RIO's. Cosmos-Reason2 is a
Qwen3-VL architecture, so it loads through plain
`transformers.Qwen3VLForConditionalGeneration` with no vendor package at all;
the cosmos-reason2 repo is documentation, examples and the quantization recipe,
and its own README says the repo is not needed to run inference.

WHAT IT ASKS, AND THE ONE THING THAT IS NOT SHARED
--------------------------------------------------
The three prompt-set questions, word for word as Alpamayo gets them, plus one
more: the physical-reasoning question. That extra question is the counterpart
of Alpamayo's Chain-of-Causation trace -- each model is asked for a reasoning
trace of the kind it was built to produce, and the card puts them in the same
row because that is the comparison worth making. Asking Cosmos for a
Chain-of-Causation, or Alpamayo for physical plausibility, would be comparing
two models on one model's home ground.

THE FRAMES GO IN AS A SHORT VIDEO
---------------------------------
Cosmos-Reason2's whole subject is what is happening over time -- what is
moving, what happens next, whether that is physically plausible -- and it was
post-trained on video. So the four frames are handed to the video processor as
a 0.3 s clip at 10 fps rather than as four unrelated pictures. Same bytes, same
t0, same window as Alpamayo sees; a different arrangement of them, because the
two models read a temporal window in different ways and the point is to ask
each one properly.

If the video path is unavailable in this transformers build, it falls back to
the image-sequence form and says which was used in `frames_as`. A silently
different input would make two rows of the corpus incomparable.

THE THINKING BLOCK IS KEPT
--------------------------
Cosmos reasons inside <think>...</think> before answering. That trace is the
most interesting thing it produces and it is kept verbatim, separated from the
answer rather than stripped -- the dashboard shows the answer, the corpus keeps
both.
"""
import argparse
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from teachers.service import common                              # noqa: E402

MODEL_ID = "nvidia/Cosmos-Reason2-8B"
REVISION = "a9fae2cf89dc64db96b12860417f0eb403013bb9"
CODE_REVISION = "a3b4a1db4065fe13c4b1f4d2fb8605bad647f4b9"

# From NVIDIA's own minimal example (scripts/inference_sample.py): the vision
# tower's token budget, expressed in pixels. Held to the same numbers so a
# reading here is comparable with a reading from the model card.
PIXELS_PER_TOKEN = 32 ** 2
MIN_VISION_TOKENS = 256
MAX_VISION_TOKENS = 8192

SYSTEM = "You are a helpful assistant."

_THINK = re.compile(r"<think>(.*?)</think>", re.S)


class CosmosService(common.Service):
    name = "cosmos-reason2"

    def __init__(self, model_path, precision="bf16", depth=1,
                 max_new_tokens=1024, top_p=0.98, temperature=0.6,
                 attn="sdpa", fps=10.0):
        super().__init__(depth=depth)
        self.model_path = model_path
        self.precision = precision
        self.max_new_tokens = int(max_new_tokens)
        self.top_p = float(top_p)
        self.temperature = float(temperature)
        self.attn = attn
        self.fps = float(fps)
        self.frames_as = "video"
        self.model = None
        self.processor = None

    def load(self):
        import torch
        import transformers

        self._torch = torch
        transformers.set_seed(0)
        self.model = transformers.Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_path, dtype=torch.bfloat16, device_map="cuda",
            attn_implementation=self.attn)
        self.model.eval()
        self.processor = transformers.AutoProcessor.from_pretrained(self.model_path)
        size = {"shortest_edge": MIN_VISION_TOKENS * PIXELS_PER_TOKEN,
                "longest_edge": MAX_VISION_TOKENS * PIXELS_PER_TOKEN}
        try:
            self.processor.image_processor.size = dict(size)
            self.processor.video_processor.size = dict(size)
        except AttributeError:
            # An older processor without a separate video path. Not fatal: the
            # image path still works and `frames_as` will say so.
            self.frames_as = "images"

    def describe(self):
        return {"model": self.name, "model_id": MODEL_ID, "revision": REVISION,
                "code_revision": CODE_REVISION, "precision": self.precision,
                "frames_as": self.frames_as, "attn": self.attn}

    # -- input --------------------------------------------------------------
    def _video_metadata(self, images):
        """What the four frames ACTUALLY are, in time.

        Without this, transformers warns and defaults to 24 fps -- so a model
        asked "what is about to happen next" is told the window it is looking
        at spans 4/24 = 0.17 s when it really spans 0.3 s. That is not a
        cosmetic difference for a physical-reasoning model: every velocity it
        infers from the frames is scaled by it, and a car closing at 2 m/s
        reads as one closing at 3.6.

        Caught by the input smoke test before the weights were ever loaded,
        which is the only reason it is not a silent bias in every reading.
        """
        from transformers.video_utils import VideoMetadata

        n = len(images)
        return VideoMetadata(
            total_num_frames=n,
            fps=self.fps,
            width=images[0].width,
            height=images[0].height,
            # (n-1) intervals, not n: four frames at 10 Hz span 0.3 s, which is
            # exactly the window teachers/keyframe.py builds.
            duration=(n - 1) / self.fps if self.fps else None,
        )

    def _conversation(self, images, question):
        """Media first, then the text — the order the model was trained on.

        NVIDIA's example puts it in a comment and it is load-bearing: a
        conversation with the question before the pictures is a different
        distribution from the one this model was post-trained on, and it
        answers noticeably worse.
        """
        if self.frames_as == "video":
            media = {"type": "video", "video": list(images)}
        else:
            media = None
        content = ([media] if media else
                   [{"type": "image", "image": im} for im in images])
        return [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content + [{"type": "text",
                                                    "text": question}]},
        ]

    def _generate(self, images, question, max_new_tokens):
        import torch

        kwargs = {}
        if self.frames_as == "video":
            kwargs["fps"] = self.fps
            kwargs["video_metadata"] = [self._video_metadata(images)]
        try:
            inputs = self.processor.apply_chat_template(
                self._conversation(images, question), tokenize=True,
                add_generation_prompt=True, return_dict=True,
                return_tensors="pt", **kwargs)
        except Exception:
            # The video path is the preferred one, not a required one. Falling
            # back is better than a keyframe with no Cosmos column -- and the
            # fallback is recorded on every reading from here on, so nobody
            # compares a video row with an image row without noticing.
            self.frames_as = "images"
            inputs = self.processor.apply_chat_template(
                self._conversation(images, question), tokenize=True,
                add_generation_prompt=True, return_dict=True,
                return_tensors="pt")
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=True,
                top_p=self.top_p, temperature=self.temperature)
        trimmed = [o[len(i):] for i, o in zip(inputs.input_ids, ids)]
        return self.processor.batch_decode(
            trimmed, skip_special_tokens=True,
            clean_up_tokenization_spaces=False)[0]

    # -- inference ----------------------------------------------------------
    def infer(self, payload):
        images = common.decode_frames(payload)
        prompts = payload.get("prompts") or {}
        physical = payload.get("physical_prompt") or PHYSICAL_DEFAULT

        raw, timings = {}, {}

        # The reasoning question first: it is the longest generation and the
        # one worth having if the keyframe is abandoned part-way through.
        t = time.perf_counter()
        text = self._generate(images, physical, self.max_new_tokens)
        timings["physical_ms"] = round((time.perf_counter() - t) * 1000, 1)
        thinking, answer = split_thinking(text)
        raw["physical_reasoning"] = {"question": physical, "text": text,
                                     "thinking": thinking, "answer": answer}

        answers = {}
        for key in ("scene", "critical_actor", "attention"):
            q = prompts.get(key)
            if not q:
                continue
            t = time.perf_counter()
            out = self._generate(images, q, min(self.max_new_tokens, 768))
            th, ans = split_thinking(out)
            answers[key] = ans
            raw[key] = {"question": q, "text": out, "thinking": th,
                        "answer": ans}
            timings[key + "_ms"] = round((time.perf_counter() - t) * 1000, 1)

        return {
            "ok": True,
            "raw": raw,
            "scene": answers.get("scene", ""),
            "critical_actor": answers.get("critical_actor", ""),
            "attention": answers.get("attention", ""),
            "reasoning": answer,
            "thinking": thinking,
            "meta_action": None,
            # Cosmos predicts no trajectory. Explicitly null rather than absent
            # so the card's "Predicted path" control can be honestly disabled
            # for this column instead of silently missing.
            "trajectory": None,
            "timings_ms": timings,
        }


PHYSICAL_DEFAULT = (
    "The video shows the view from a car's forward camera. What is moving in "
    "this scene, what is about to happen next, and is that physically "
    "plausible? Think step by step."
)


def split_thinking(text):
    """<think>trace</think> answer -> (trace, answer). Never loses text.

    A model that did not think returns ("", text) -- not (text, "") -- because
    an un-tagged generation is an answer, and putting it in the trace field
    would empty the column the card actually shows.
    """
    t = text or ""
    m = _THINK.search(t)
    if not m:
        return "", t.strip()
    return m.group(1).strip(), _THINK.sub("", t).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8802)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--precision", default="bf16", choices=("bf16", "fp8"))
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--attn", default="sdpa", choices=("sdpa", "flash_attention_2"))
    ap.add_argument("--frames-as", default="video", choices=("video", "images"))
    ap.add_argument("--no-warm", action="store_true")
    args = ap.parse_args()

    svc = CosmosService(args.model, precision=args.precision, depth=args.depth,
                        max_new_tokens=args.max_new_tokens, attn=args.attn)
    svc.frames_as = args.frames_as
    common.serve(svc, host=args.host, port=args.port, warm=not args.no_warm)


if __name__ == "__main__":
    main()
