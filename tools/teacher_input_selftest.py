"""The two models' input contracts — checked in their own environments.

    python -m tools.teacher_input_selftest

WHY THIS IS A SEPARATE SUITE
----------------------------
tools/teacher_selftest.py asserts everything RIO builds: the window, the ego
history, the payload. It cannot assert what happens to that payload on the
other side of the socket, because the other side is a Python this interpreter
cannot import -- different version, different torch, different transformers.

So this one shells out. It runs a short script inside each teacher's venv and
checks the half of the contract that lives there: does the JPEG become the
tensor the model wants, does the single forward camera produce the scaffolding
the model was trained on, do four frames really go in as a 0.3 s video.

IT NEEDS NO WEIGHTS, and that is the point. The processors are ungated
(Qwen/Qwen3-VL-2B-Instruct for Alpamayo's, which is what helper.get_processor
uses anyway; Qwen/Qwen3-VL-8B-Instruct for Cosmos's, which is the model Cosmos
was finetuned from and is already on this pod). Token IDs are therefore not the
teachers' own -- shapes, scaffolding and plumbing are, and those are what break.

THE BUG THIS ALREADY CAUGHT
---------------------------
Cosmos's four frames were going in as a video with `fps=10` and no video
metadata, so transformers warned once and defaulted to 24 fps. The model was
being told a 0.3 s window spans 0.17 s -- which scales every velocity it infers
by 2.4x, in a model whose entire job is "what is about to happen next". A
warning on stderr during a load nobody was reading, and a systematic bias in
every reading afterwards. Found here, before a single weight was downloaded.
"""
import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

VENVS = {
    "alpamayo": "/opt/teachers/venvs/alpamayo/bin/python",
    "cosmos": "/opt/teachers/venvs/cosmos/bin/python",
}

PASS, FAIL = [], []


def ok(cond, what):
    (PASS if cond else FAIL).append(what)
    print(("  ok    " if cond else "  FAIL  ") + what)


def section(name):
    print(f"\n=== {name} ===")


FRAMES = '''
import base64, io, sys
sys.path.insert(0, "%s")
import numpy as np
from PIL import Image
def frames_b64(n=4):
    out = []
    for i in range(n):
        a = np.zeros((480, 640, 3), np.uint8)
        a[:240] = (170, 155, 140); a[240:] = (62, 62, 68)
        a[300:360, 280 + i * 6:400 + i * 6] = (48, 48, 172)
        b = io.BytesIO(); Image.fromarray(a).save(b, "JPEG")
        out.append(base64.b64encode(b.getvalue()).decode())
    return out
''' % REPO

ALPAMAYO = FRAMES + '''
import torch
from transformers import AutoProcessor
from alpamayo1_5 import helper
from teachers.service import common
from teachers.service.alpamayo_service import AlpamayoService, FRONT_WIDE

payload = {"frames": frames_b64(),
           "ego_history_xyz": [[-(15 - i) * 1.3, 0.0, 0.0] for i in range(16)],
           "ego_history_yaw": [0.0] * 16}
imgs = common.decode_frames(payload)
svc = AlpamayoService.__new__(AlpamayoService); svc._torch = torch
t = svc._frames_tensor(imgs)
print("SHAPE", tuple(t.shape), t.dtype)

xyz, rot, synth = AlpamayoService._ego(svc, payload)
print("EGO", tuple(xyz.shape), tuple(rot.shape), synth)
xyz2, rot2, synth2 = AlpamayoService._ego(svc, {})
print("EGOFALLBACK", tuple(xyz2.shape), synth2)

proc = AutoProcessor.from_pretrained(helper.BASE_PROCESSOR_NAME,
                                     min_pixels=helper.MIN_PIXELS,
                                     max_pixels=helper.MAX_PIXELS)
cam = torch.tensor([FRONT_WIDE], dtype=torch.long)
msgs = helper.create_message(t, camera_indices=cam, num_frames_per_camera=4)
ids = proc.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False,
                               continue_final_message=True, return_dict=True,
                               return_tensors="pt")
text = proc.apply_chat_template(msgs, tokenize=False, continue_final_message=True)
print("ROLLOUT", tuple(ids.input_ids.shape))
print("CAMNAME", "Front camera" in text)
print("FRAMEIDX", all(("frame %d" % i) in text for i in range(4)))
print("TRAJHIST", "<|traj_history_start|>" in text)
print("COTSTART", "<|cot_start|>" in text)

vq = helper.create_vqa_message(t, question="Describe the scene.",
                               camera_indices=cam, num_frames_per_camera=4)
vtext = proc.apply_chat_template(vq, tokenize=False, continue_final_message=True)
vids = proc.apply_chat_template(vq, tokenize=True, add_generation_prompt=False,
                                continue_final_message=True, return_dict=True,
                                return_tensors="pt")
print("VQA", tuple(vids.input_ids.shape))
print("VQAQ", "<|question_start|>" in vtext and "<|answer_start|>" in vtext)
print("VQANOTRAJ", "<|traj_history_start|>" not in vtext)
'''

COSMOS = FRAMES + '''
import transformers
from teachers.service import common
from teachers.service.cosmos_service import (CosmosService, PIXELS_PER_TOKEN,
    MIN_VISION_TOKENS, MAX_VISION_TOKENS, split_thinking, PHYSICAL_DEFAULT)

imgs = common.decode_frames({"frames": frames_b64()})
svc = CosmosService.__new__(CosmosService); svc.frames_as = "video"; svc.fps = 10.0
svc.processor = transformers.AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
size = {"shortest_edge": MIN_VISION_TOKENS * PIXELS_PER_TOKEN,
        "longest_edge": MAX_VISION_TOKENS * PIXELS_PER_TOKEN}
svc.processor.image_processor.size = dict(size)
svc.processor.video_processor.size = dict(size)

md = svc._video_metadata(imgs)
print("META", md.total_num_frames, md.fps, md.duration)
conv = svc._conversation(imgs, PHYSICAL_DEFAULT)
print("MEDIAFIRST", conv[1]["content"][0]["type"])

import warnings
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    ins = svc.processor.apply_chat_template(
        conv, tokenize=True, add_generation_prompt=True, return_dict=True,
        return_tensors="pt", fps=svc.fps, video_metadata=[md])
    msgs = " | ".join(str(x.message) for x in w)
print("VIDEO", tuple(ins.input_ids.shape), "pixel_values_videos" in ins)
print("NOFPSWARN", "no video metadata" not in msgs)

svc.frames_as = "images"
conv2 = svc._conversation(imgs, "Describe the scene.")
ins2 = svc.processor.apply_chat_template(conv2, tokenize=True,
    add_generation_prompt=True, return_dict=True, return_tensors="pt")
print("IMAGES", tuple(ins2.input_ids.shape), "pixel_values" in ins2)
print("THINK", split_thinking("<think>a b</think>\\nAnswer.") == ("a b", "Answer."))
print("NOTHINK", split_thinking("Plain answer.") == ("", "Plain answer."))
'''


# The FP8 recipe, checked against layer names rather than against a checkpoint.
# "FP8" on a dashboard means the language backbone and NOT the vision tower,
# the lm_head or the diffusion expert -- which is a claim about regexes, and
# regexes are exactly the thing that silently stops matching.
QUANT = '''
import json, glob, sys
sys.path.insert(0, "%s")
import torch, transformers, llmcompressor
from llmcompressor.modifiers.quantization import QuantizationModifier
from compressed_tensors.utils.match import _match_name
from teachers.service.quantize_alpamayo import IGNORE
print("ENV", torch.__version__, transformers.__version__, llmcompressor.__version__)
m = QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=IGNORE)
print("SCHEME", m.scheme)

# THE MODEL'S REAL MODULE NAMES, read out of the weight index rather than
# invented here. The first version of this check used re.search against names
# the test author chose, and passed while the actual build quantized the whole
# vision tower -- because compressed_tensors matches with re.MATCH, anchored at
# the start, and the real names are nested one level deeper than the patterns
# assumed. So: the library's own matcher, against the shipped index.
idx = glob.glob("/workspace/.cache/huggingface/hub/"
                "models--nvidia--Alpamayo-1.5-10B/snapshots/*/"
                "model.safetensors.index.json")
if not idx:
    print("NOINDEX 1")
else:
    names = sorted({n[:-7] for n in json.load(open(idx[0]))["weight_map"]
                    if n.endswith(".weight")})
    def ignored(n):
        return any(_match_name(n, t) for t in IGNORE)
    def frac(sel):
        got = [n for n in names if sel(n)]
        return len(got), sum(1 for n in got if ignored(n))
    n_vis, i_vis = frac(lambda n: ".visual." in n)
    n_exp, i_exp = frac(lambda n: n.startswith("expert."))
    n_act, i_act = frac(lambda n: n.startswith("action_"))
    n_lm,  i_lm  = frac(lambda n: "lm_head" in n)
    n_bb,  i_bb  = frac(lambda n: n.startswith("vlm.") and ".visual." not in n
                                  and "lm_head" not in n)
    print("VISION", n_vis, i_vis)
    print("EXPERT", n_exp, i_exp)
    print("ACTION", n_act, i_act)
    print("LMHEAD", n_lm, i_lm)
    print("BACKBONE", n_bb, i_bb)

# ...and NVIDIA's own Cosmos list, against Cosmos's own names. It is their file
# and it is correct -- `model.visual.*` matches there because the VLM is not
# nested. Checked anyway: it is the same class of bug one rename away, and the
# check costs a JSON read.
cidx = glob.glob("/workspace/.cache/huggingface/hub/"
                 "models--nvidia--Cosmos-Reason2-8B/snapshots/*/"
                 "model.safetensors.index.json")
if cidx:
    COSMOS_IGNORE = ["re:.*lm_head", "re:visual.*", "re:model.visual.*",
                     "re:.*mlp.gate$"]
    cn = sorted({n[:-7] for n in json.load(open(cidx[0]))["weight_map"]
                 if n.endswith(".weight")})
    def cign(n):
        return any(_match_name(n, t) for t in COSMOS_IGNORE)
    vis = [n for n in cn if ".visual." in n or n.startswith("visual.")]
    bb = [n for n in cn if n not in vis and "lm_head" not in n]
    print("COSMOSVISION", len(vis), sum(1 for n in vis if cign(n)))
    print("COSMOSBACKBONE", len(bb), sum(1 for n in bb if cign(n)))
''' % REPO

QUANT_HEADER = """# /// script
# requires-python = "==3.12.*"
# dependencies = ["llmcompressor==0.9.0.4","transformers==4.57.1","torch==2.8.0",
#                 "torchvision>=0.23.0","accelerate>=1.12.0","einops>=0.8.1",
#                 "hydra-core>=1.3.2","pillow>=12.0.0","numpy<3","scipy>=1.11"]
# ///
"""


def run_uv_script(body):
    """A PEP-723 script through `uv run`, in an environment of its own."""
    import tempfile

    env = dict(os.environ)
    env["PATH"] = os.path.expanduser("~/.local/bin") + ":" + env.get("PATH", "")
    env.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(QUANT_HEADER + body)
        path = f.name
    try:
        p = subprocess.run(["uv", "run", "--script", path], cwd=REPO, env=env,
                           capture_output=True, text=True, timeout=1200)
    finally:
        os.unlink(path)
    if p.returncode != 0:
        ok(False, f"the quantization environment failed to resolve or import"
                  f"\n{p.stderr[-1200:]}")
        return {}
    out = {}
    for line in p.stdout.splitlines():
        parts = line.split(None, 1)
        if parts and parts[0].isupper():
            out[parts[0]] = parts[1] if len(parts) > 1 else ""
    return out


def run(name, script):
    py = VENVS[name]
    if not os.path.exists(py):
        ok(False, f"{name}: no environment at {py} "
                  f"(bash boot.sh teachers-build)")
        return {}
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/workspace/.cache/huggingface")
    env["TRANSFORMERS_VERBOSITY"] = "error"
    p = subprocess.run([py, "-c", script], cwd=REPO, env=env,
                       capture_output=True, text=True, timeout=900)
    if p.returncode != 0:
        ok(False, f"{name}: the input script failed\n{p.stderr[-1200:]}")
        return {}
    out = {}
    for line in p.stdout.splitlines():
        parts = line.split(None, 1)
        if parts and parts[0].isupper():
            out[parts[0]] = parts[1] if len(parts) > 1 else ""
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()

    section("A. Alpamayo 1.5 — one forward camera, four frames, 16 poses")
    a = run("alpamayo", ALPAMAYO)
    if a:
        ok(a.get("SHAPE", "").startswith("(4, 3, 480, 640) torch.uint8"),
           f"JPEGs become the (N, 3, H, W) uint8 tensor create_message wants "
           f"({a.get('SHAPE')}) — floats here would be normalised twice")
        ok(a.get("EGO", "").startswith("(1, 1, 16, 3) (1, 1, 16, 3, 3)"),
           f"the ego history is (1,1,16,3) and (1,1,16,3,3) — the loader's own "
           f"shape ({a.get('EGO')})")
        ok("True" in a.get("EGOFALLBACK", ""),
           "a drive with no speed gets a synthesised history, FLAGGED as one "
           "rather than silently read as a stationary car")
        ok(a.get("CAMNAME") == "True",
           "the prompt names the single forward camera the way the model was "
           "trained on it (`Front camera:`) — this is the flexible-camera path, "
           "not an adapter")
        ok(a.get("FRAMEIDX") == "True",
           "with all four frame indices, which is what the scaffolding is "
           "indexed by and why an under-filled window is refused")
        ok(a.get("TRAJHIST") == "True" and a.get("COTSTART") == "True",
           "the rollout message carries the trajectory-history placeholder and "
           "opens the Chain-of-Causation")
        ok(a.get("VQAQ") == "True",
           "the VQA message uses the question/answer special tokens")
        ok(a.get("VQANOTRAJ") == "True",
           "...and carries NO trajectory history — which is why the prompt set "
           "still works on a drive with no speed signal at all")
        print(f"       rollout {a.get('ROLLOUT')} tokens, VQA {a.get('VQA')}")

    section("B. Cosmos-Reason2 — 0.3 s of road, as 0.3 s of road")
    c = run("cosmos", COSMOS)
    if c:
        ok(c.get("META", "").startswith("4 10.0 0.3"),
           f"the four frames are declared as 4 frames at 10 fps spanning 0.3 s "
           f"({c.get('META')})")
        ok(c.get("MEDIAFIRST") == "video",
           "they go in as a video, not as four unrelated pictures — Cosmos's "
           "subject is what happens over time and it was post-trained on video")
        ok("True" in c.get("VIDEO", ""),
           f"and the video processor produces pixel_values_videos "
           f"({c.get('VIDEO')})")
        ok(c.get("NOFPSWARN") == "True",
           "WITH video metadata. Without it transformers defaults to 24 fps, "
           "so a 0.3 s window is described as 0.17 s and every velocity the "
           "model infers is scaled by 2.4x")
        ok("True" in c.get("IMAGES", ""),
           f"the image-sequence fallback still works, for a build with no "
           f"video path ({c.get('IMAGES')})")
        ok(c.get("THINK") == "True" and c.get("NOTHINK") == "True",
           "the <think> trace is separated from the answer, and an untagged "
           "generation stays an ANSWER rather than becoming an empty one")

    section("D. the vision tower's first layer — same arithmetic, 13.7 s cheaper")
    pe = run("cosmos", PATCHEMBED)
    if pe:
        ok(pe.get("APPLIED", "").startswith("True"),
           f"the Conv3d patch embedding is rewritten as a linear layer "
           f"({pe.get('APPLIED')})")
        ok(pe.get("ALLCLOSE") == "True" and pe.get("SHAPE", "").startswith("True"),
           f"and computes the SAME numbers — max deviation "
           f"{pe.get('MAXDEV')} is bf16 rounding, not a different model")
        try:
            conv_ms = float(pe.get("CONVMS", "0"))
            lin_ms = float(pe.get("LINMS", "0"))
        except ValueError:
            conv_ms = lin_ms = 0.0
        ok(conv_ms > 1000 and lin_ms < 50,
           f"the shipped Conv3d takes {conv_ms:.0f} ms on one window and the "
           f"linear form takes {lin_ms:.2f} ms — 99.9% of a vision-tower "
           f"forward, and four of them per keyframe made Cosmos a 57-second "
           f"teacher on a 2-second cadence")
        ok(pe.get("REFUSED") == "True",
           "an overlapping or padded conv is NOT rewritten — the equivalence "
           "only holds for non-overlapping patches, and a wrong rewrite here "
           "would be invisible in every reading afterwards")

    section("C. the FP8 recipe reaches the backbone and nothing else")
    q = run_uv_script(QUANT)
    if q:
        env = q.get("ENV", "").split()
        ok(len(env) == 3 and env[1] == "4.57.1",
           f"the quantizer resolves against Alpamayo's OWN transformers pin "
           f"({q.get('ENV')}) — llmcompressor unpinned drags transformers to "
           f"5.x and torch to cu130, which is how the serving venv was "
           f"destroyed once already")
        ok(q.get("SCHEME") == "FP8_DYNAMIC",
           "FP8_DYNAMIC — static calibration needs a forward pass the "
           "composite model's signature does not provide")
        ok(not q.get("NOINDEX"),
           "the real weight index is available to check the patterns against")

        def pair(key):
            try:
                a, b = q.get(key, "0 0").split()
                return int(a), int(b)
            except ValueError:
                return 0, 0

        for key, what, why in (
                ("VISION", "the vision tower",
                 "it has to read a 40-metre car out of thirty pixels"),
                ("EXPERT", "the diffusion action expert",
                 "so the predicted path is computed at full precision from a "
                 "quantized trace"),
                ("ACTION", "the action projections", "same head, same reason"),
                ("LMHEAD", "the output projection",
                 "a rounding error of the parameter count, and measurable "
                 "quality if you round it")):
            n, ign = pair(key)
            ok(n > 0 and ign == n,
               f"{what}: all {n} modules excluded ({ign}/{n}) — {why}")

        n, ign = pair("BACKBONE")
        ok(n > 0 and ign == 0,
           f"the language backbone IS quantized: {n} modules, {ign} excluded")

        if "COSMOSVISION" in q:
            n, ign = pair("COSMOSVISION")
            ok(n > 0 and ign == n,
               f"NVIDIA's own Cosmos list still excludes its vision tower "
               f"({ign}/{n}) — correct there because the VLM is not nested, "
               f"and one rename from the bug above")
            n, ign = pair("COSMOSBACKBONE")
            ok(n > 0 and ign == 0,
               f"...while quantizing its backbone ({n} modules, {ign} excluded)")

    print("\n" + "=" * 72)
    total = len(PASS) + len(FAIL)
    print(f"{len(PASS)}/{total} checks passed")
    if FAIL:
        print("\nFAILED:")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 72)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
