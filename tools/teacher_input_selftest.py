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
