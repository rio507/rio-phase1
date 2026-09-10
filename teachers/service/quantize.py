#!/usr/bin/env python
"""FP8, by NVIDIA's recipe — for both teachers, from one file.

    /opt/teachers/venvs/<venv>/bin/python -m teachers.service.quantize \
        --model cosmos --out /workspace/teachers/fp8

WHY THIS EXISTS WHEN THE PANEL RUNS BF16
----------------------------------------
Because the pod it was built on is not the pod it lives on. This machine is an
H200 with 141 GB, where both teachers fit at BF16 next to RIO's own stack with
room to spare. The everyday pod is an L40S with 48 GB, which already holds
Qwen3-VL-8B, Depth-Anything, UFLDv2 and RF-DETR -- and 21 GB of Alpamayo plus
16 GB of Cosmos does not go into what is left.

So the FP8 variants are not an optimisation, they are the reason the panel can
exist on the pod anyone actually drives with. They are built here, measured
here, and the numbers for both precisions go in the report so the trade is a
measurement rather than a hope.

THE RECIPE IS NVIDIA'S, NOT MINE
--------------------------------
For Cosmos this shells out to `scripts/quantize.py` from the pinned
nvidia-cosmos/cosmos-reason2 checkout, through `uv run --script`, which
resolves that script's OWN locked dependency set (llmcompressor at a pinned
commit, a pinned transformers, qwen-vl-utils). That is deliberate: the recipe
has pins for a reason, they disagree with the serving environment's pins, and
re-deriving it by hand is how a quantization silently stops matching the one
the model was validated with.

For Alpamayo there is no vendor recipe -- it is a composite model, a
Cosmos-Reason2 VLM with a diffusion action expert bolted on, and NVIDIA ships
no quantization script for it. So this applies the SAME llmcompressor recipe to
the part it applies to: the language backbone's Linear layers, with the vision
tower, the lm_head and the entire diffusion expert excluded. FP8_DYNAMIC rather
than static FP8, because static calibration needs a forward pass driven by a
calibration dataset and the composite model's forward signature is not the one
llmcompressor's VLM collator produces. Dynamic activation scales cost a little
speed and need no calibration data at all.

WHAT THAT MEANS FOR A READING'S `precision` FIELD
-------------------------------------------------
"fp8" means the weights of the language backbone are FP8. It does not mean the
vision tower is, it does not mean the KV cache is, and for Alpamayo it does not
mean the diffusion expert is -- that stays BF16, so the predicted trajectory is
computed at full precision from a quantized trace. Written down here because
"FP8" on a dashboard is otherwise a claim nobody can check.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO_SRC = Path("/workspace/teachers/src")
COSMOS_REPO = REPO_SRC / "cosmos-reason2"

ALPAMAYO_ID = "nvidia/Alpamayo-1.5-10B"
COSMOS_ID = "nvidia/Cosmos-Reason2-8B"

# What may NOT be quantized, and why each one is on the list.
#   lm_head          — output projection; quantizing it costs measurable
#                      quality for a layer that is a rounding error of the
#                      parameter count. NVIDIA's own recipe excludes it.
#   visual / model.visual — the vision tower. Same reason, and it is the part
#                      that has to read a 40-metre car out of 30 pixels.
#   mlp.gate         — MoE routing. A router that rounds picks a different
#                      expert, which is not a small error.
#   diffusion / action / expert — Alpamayo only: the trajectory head. It is
#                      small, it is the only numeric output either model
#                      produces, and it is not worth a byte of what it costs.
IGNORE_COMMON = ["re:.*lm_head", "re:visual.*", "re:model.visual.*",
                 "re:.*mlp.gate$"]
IGNORE_ALPAMAYO = IGNORE_COMMON + [
    "re:.*diffusion.*", "re:.*action_expert.*", "re:.*action_in_proj.*",
    "re:.*traj.*",
]


def quantize_cosmos(out_dir: Path, precision: str, num_samples: int) -> int:
    """Straight to NVIDIA's script, in its own locked environment."""
    script = COSMOS_REPO / "scripts" / "quantize.py"
    if not script.exists():
        print(f"!! {script} is missing — is {COSMOS_REPO} checked out?")
        return 2
    cmd = ["uv", "run", "--script", str(script),
           "-o", str(out_dir), "--model", COSMOS_ID,
           "--precision", precision, "--num-samples", str(num_samples)]
    print("+ " + " ".join(cmd), flush=True)
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    return subprocess.call(cmd, cwd=str(COSMOS_REPO), env=env)


def quantize_alpamayo(out_dir: Path) -> int:
    """The same recipe, applied to the backbone of a model NVIDIA did not script.

    Run with the ALPAMAYO venv plus llmcompressor: the model class lives in the
    `alpamayo1_5` package and cannot be loaded anywhere else.
    """
    import torch
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier

    from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

    out = out_dir / "alpamayo_fp8"
    out.mkdir(parents=True, exist_ok=True)

    print(f"loading {ALPAMAYO_ID} (bf16, cpu)...", flush=True)
    model = Alpamayo1_5.from_pretrained(
        ALPAMAYO_ID, dtype=torch.bfloat16, attn_implementation="sdpa")

    recipe = QuantizationModifier(
        targets="Linear", scheme="FP8_DYNAMIC", ignore=IGNORE_ALPAMAYO)
    print("oneshot FP8_DYNAMIC (data-free)...", flush=True)
    oneshot(model=model, recipe=recipe)

    print(f"saving -> {out}", flush=True)
    model.save_pretrained(str(out), save_compressed=True)
    # The tokenizer travels with the checkpoint or the service cannot build a
    # processor from the local path, and a quantized model that can only be
    # loaded next to the original defeats the point of having it.
    try:
        model.tokenizer.save_pretrained(str(out))
    except Exception as e:
        print(f"   (tokenizer not saved: {e})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=("alpamayo", "cosmos"))
    ap.add_argument("--out", default="/workspace/teachers/fp8")
    ap.add_argument("--precision", default="fp8",
                    choices=("fp8", "fp8_dynamic", "nvfp4"),
                    help="cosmos only; alpamayo is always FP8_DYNAMIC")
    ap.add_argument("--num-samples", type=int, default=512,
                    help="cosmos only: llmcompressor calibration samples")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.model == "cosmos":
        return quantize_cosmos(out, args.precision, args.num_samples)
    return quantize_alpamayo(out)


if __name__ == "__main__":
    sys.exit(main())
