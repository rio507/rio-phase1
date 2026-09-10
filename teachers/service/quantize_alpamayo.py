#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "llmcompressor==0.9.0.4",
#   "transformers==4.57.1",
#   "torch==2.8.0",
#   "torchvision>=0.23.0",
#   "accelerate>=1.12.0",
#   "einops>=0.8.1",
#   "hydra-core>=1.3.2",
#   "pillow>=12.0.0",
#   "numpy<3",
#   "scipy>=1.11",
# ]
# ///
"""FP8 for Alpamayo 1.5 — in an environment of its own, for a hard-won reason.

    uv run --script teachers/service/quantize_alpamayo.py -o /workspace/teachers/fp8

WHY THIS IS NOT A FUNCTION IN THE SERVING VENV
-----------------------------------------------
Because putting it there destroyed that venv. `uv pip install llmcompressor`
into the Alpamayo environment resolved to the newest llmcompressor, which wants
transformers 5.x, which dragged transformers from 4.57.1 to 5.14.1 and torch
from 2.8.0+cu128 to 2.13.0+cu130 -- and the serving environment stopped being
able to import `PreTrainedModel` at all. A quantizer that breaks the thing it
is quantizing for is not a tool, it is an outage waiting for the next person
who runs it.

NVIDIA had already worked this out: their own scripts/quantize.py is a
`uv run --script` with its own locked dependency block, entirely separate from
the environment the model serves from. This is the same shape, for the model
they did not ship a script for.

The pins are not decoration. llmcompressor is pinned to 0.9.0.4 because that is
the newest version that resolves against transformers 4.57.1 -- and 4.57.1 is
not negotiable, because `alpamayo1_5` is written against it and imports
`Qwen3VLConfig` from it by name.

WHAT IS AND IS NOT QUANTIZED
----------------------------
Alpamayo 1.5 is a composite: a Cosmos-Reason2 VLM with a diffusion action
expert bolted on. The same recipe NVIDIA applies to Cosmos is applied to the
part it applies to -- the language backbone's Linear layers -- and everything
else is excluded by name:

  lm_head          the output projection. A rounding error of the parameter
                   count, and measurable quality if you round it. NVIDIA's own
                   recipe excludes it.
  visual.*         the vision tower. Same reason, and it is the part that has
                   to read a 40-metre car out of thirty pixels.
  mlp.gate         MoE routing. A router that rounds picks a DIFFERENT expert,
                   which is not a small error.
  diffusion, action_expert, action_in_proj, traj
                   the trajectory head. Small, and the only numeric output
                   either teacher produces.

FP8_DYNAMIC rather than static FP8: static calibration needs a forward pass
driven by a calibration dataset, and the composite model's forward signature is
not the one llmcompressor's VLM collator produces. Dynamic activation scales
cost a little inference speed and need no calibration data at all.

So `precision: "fp8"` on an Alpamayo reading means: the language backbone's
Linear weights are FP8. Not the vision tower, not the KV cache, and not the
diffusion expert -- the predicted trajectory is computed at full precision from
a quantized trace. Written down because "FP8" on a dashboard is otherwise a
claim nobody can check.
"""
import argparse
import os
import sys
import time

# THE PACKAGE IS PUT ON THE PATH, NOT DECLARED AS A DEPENDENCY.
#
# `alpamayo1_5` is a path dependency whose pyproject requires flash-attn, and
# flash-attn compiles from source against nvcc -- half an hour, for a module
# this script never calls, in an environment that exists to run one oneshot.
# The serving venv skips it with `--no-install-package flash-attn`; a
# PEP-723 script has no equivalent, so the checkout goes on sys.path and its
# real runtime requirements (torch, transformers, einops, hydra, numpy, scipy)
# are listed above by name. `attn_implementation="sdpa"` is what the model is
# loaded with either way.
ALPAMAYO_SRC = "/workspace/teachers/src/alpamayo1.5/src"
if os.path.isdir(ALPAMAYO_SRC):
    sys.path.insert(0, ALPAMAYO_SRC)

MODEL_ID = "nvidia/Alpamayo-1.5-10B"
REVISION = "7aba8293c09993f2e125c6819df05d7fa3e873ea"

IGNORE = [
    "re:.*lm_head",
    "re:visual.*",
    "re:model.visual.*",
    "re:.*mlp.gate$",
    "re:.*diffusion.*",
    "re:.*action_expert.*",
    "re:.*action_in_proj.*",
    "re:.*action_out_proj.*",
    "re:.*traj.*",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default="/workspace/teachers/fp8")
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--name", default="alpamayo_fp8")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    import torch
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    out = os.path.join(args.out, args.name)
    os.makedirs(out, exist_ok=True)

    # ON THE CPU, deliberately. This is a 21 GB model being rewritten; doing it
    # on the card would sit on VRAM the live stack is using, and llmcompressor
    # moves what it needs itself.
    print(f"loading {args.model} @ {args.revision[:7]} (bf16, cpu)...", flush=True)
    t0 = time.time()
    model = Alpamayo1_5.from_pretrained(
        args.model, revision=args.revision, dtype=torch.bfloat16,
        attn_implementation="sdpa")
    print(f"  loaded in {time.time() - t0:.0f}s", flush=True)

    recipe = QuantizationModifier(
        targets="Linear", scheme="FP8_DYNAMIC", ignore=IGNORE)
    print("oneshot FP8_DYNAMIC (data-free)...", flush=True)
    t0 = time.time()
    oneshot(model=model, recipe=recipe)
    print(f"  quantized in {time.time() - t0:.0f}s", flush=True)

    print(f"saving -> {out}", flush=True)
    model.save_pretrained(out, save_compressed=True)
    # The tokenizer travels with the checkpoint, or the service cannot build a
    # processor from the local path -- and a quantized model that can only be
    # loaded next to the original defeats the point of having one.
    try:
        model.tokenizer.save_pretrained(out)
    except Exception as e:
        print(f"  (tokenizer not saved: {e})")

    total = sum(os.path.getsize(os.path.join(dp, f))
                for dp, _, fs in os.walk(out) for f in fs)
    print(f"done: {total / 1024 ** 3:.1f} GB in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
