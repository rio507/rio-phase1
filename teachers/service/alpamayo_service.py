#!/usr/bin/env python
"""Alpamayo 1.5, behind a socket.

    /opt/teachers/venvs/alpamayo/bin/python -m teachers.service.alpamayo_service \
        --port 8801 --precision bf16

RUN IT WITH THE ALPAMAYO VENV AND NOTHING ELSE. It needs Python 3.12,
torch 2.8, transformers 4.57.1 and the `alpamayo1_5` package from
NVlabs/alpamayo1.5 -- a set RIO's own environment does not have and must not be
made to have. That is the entire reason this is a separate process.

WHAT IT ASKS THE MODEL, AND WHY IN THAT ORDER
---------------------------------------------
Four generations per keyframe:

  1. THE ROLLOUT -- `sample_trajectories_from_data_with_vlm_rollout`. The VLM
     produces its Chain-of-Causation trace, and a diffusion expert conditioned
     on that trace's hidden states produces a 6.4 s trajectory (64 waypoints at
     10 Hz). Both come out of ONE call, which is why it goes first: the trace
     and the path are the same act of reasoning and asking for them separately
     would produce two that do not correspond.

  2-4. THE PROMPT SET -- three VQA questions, word for word the ones Cosmos is
     asked, so the two columns of the card are answers to the same question.
     `generate_text` needs no ego history: VQA is frames-only, which is also
     why these still work on a drive with no speed signal at all.

THE SINGLE FORWARD CAMERA
-------------------------
Alpamayo 1.5's flexible-camera path takes any subset of its seven cameras --
`notebooks/inference_cam_num.ipynb` runs "1 cam (front wide)" as one of its
three configurations, and `notebooks/inference_vqa.ipynb` uses a single front
wide camera throughout. So a phone in a windscreen is a supported input and
NOT an adapter: camera_indices=[1] is `camera_front_wide_120fov`, the frames go
in as (1, 4, 3, H, W), and helper.create_message builds exactly the prompt
scaffolding the model was trained on.

It is still a DEGRADED input, and the model card says so: accuracy falls with
fewer cameras, most where cross-traffic matters -- a right turn across traffic
is the example NVIDIA gives. That is a limit on what the readings mean, not on
whether the path is canonical. Every reading carries `cameras: 1` so a corpus
reader knows which it is looking at.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from teachers.service import common                              # noqa: E402

MODEL_ID = "nvidia/Alpamayo-1.5-10B"
# The weights this was built and measured against. Pinned because a teacher
# whose weights moved is a corpus whose rows are no longer comparable, and the
# only way to notice is to have written the sha down.
REVISION = "7aba8293c09993f2e125c6819df05d7fa3e873ea"
# NVlabs/alpamayo1.5 at the commit whose helper.py, action space and diffusion
# expert this service calls.
CODE_REVISION = "36aeb4c5938cbc2eb2aed33b22434773da4ab639"

# `camera_front_wide_120fov` in the model's own camera_name_to_index map. A
# phone through a windscreen is a forward-facing wide camera; nothing else in
# that map describes it.
FRONT_WIDE = 1


class AlpamayoService(common.Service):
    name = "alpamayo1.5"

    def __init__(self, model_path, precision="bf16", depth=1,
                 traj_samples=1, max_new_tokens=256, top_p=0.98,
                 temperature=0.6, attn="sdpa"):
        super().__init__(depth=depth)
        self.model_path = model_path
        self.precision = precision
        self.traj_samples = int(traj_samples)
        self.max_new_tokens = int(max_new_tokens)
        self.top_p = float(top_p)
        self.temperature = float(temperature)
        self.attn = attn
        self.model = None
        self.processor = None

    # -- loading ------------------------------------------------------------
    def load(self):
        import torch
        from alpamayo1_5 import helper
        from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

        self._torch = torch
        self._helper = helper
        kw = {"dtype": torch.bfloat16, "attn_implementation": self.attn}
        if self.precision == "fp8":
            # An FP8 checkpoint carries its own quantization_config and
            # compressed-tensors materialises the layers on load; the dtype
            # asked for here is the ACTIVATION dtype, which stays bf16. Saying
            # so out loud because "fp8" is a claim about the weights only.
            kw["dtype"] = torch.bfloat16
        self.model = Alpamayo1_5.from_pretrained(self.model_path, **kw).to("cuda")
        self.model.eval()
        self.processor = helper.get_processor(self.model.tokenizer)

    def describe(self):
        return {"model": self.name, "model_id": MODEL_ID, "revision": REVISION,
                "code_revision": CODE_REVISION, "precision": self.precision,
                "cameras": 1, "attn": self.attn}

    # -- input --------------------------------------------------------------
    def _frames_tensor(self, images):
        """PIL frames -> the (N, 3, H, W) uint8 tensor helper.create_message wants.

        Same rearrange the model's own loader does ("t h w c -> t c h w"), and
        the same uint8 dtype: the processor normalises, and handing it floats
        would double-normalise into a washed-out picture the model has never
        seen.
        """
        import numpy as np
        import torch

        arr = np.stack([np.asarray(im, dtype=np.uint8) for im in images])
        return torch.from_numpy(arr).permute(0, 3, 1, 2).contiguous()

    def _ego(self, payload):
        """The 16-pose history -> the two tensors the rollout conditions on.

        A drive with no speed signal sends null, and rather than refusing the
        keyframe this synthesises a stationary history and SAYS SO in the
        response. A stationary history is not neutral -- it tells the model the
        car is stopped -- so `ego_synthetic: true` rides on every reading built
        from one, and a corpus row carrying it can be filtered out of a
        training set.
        """
        import torch

        xyz = payload.get("ego_history_xyz")
        yaw = payload.get("ego_history_yaw")
        synthetic = not (xyz and yaw)
        if synthetic:
            steps = 16
            xyz = [[0.0, 0.0, 0.0]] * steps
            yaw = [0.0] * steps
        t_xyz = torch.tensor(xyz, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        rot = []
        import math
        for a in yaw:
            c, s = math.cos(float(a)), math.sin(float(a))
            rot.append([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        t_rot = torch.tensor(rot, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        return t_xyz.to("cuda"), t_rot.to("cuda"), synthetic

    def _tokenize(self, messages):
        return self.processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False,
            continue_final_message=True, return_dict=True, return_tensors="pt")

    # -- inference ----------------------------------------------------------
    def infer(self, payload):
        import torch

        images = common.decode_frames(payload)
        frames = self._frames_tensor(images)
        cam_idx = torch.tensor([FRONT_WIDE], dtype=torch.long)
        n_per_cam = frames.shape[0]

        ego_xyz, ego_rot, ego_synthetic = self._ego(payload)
        prompts = payload.get("prompts") or {}

        raw = {}
        timings = {}

        # --- 1. the rollout: Chain-of-Causation + the 6.4 s path ------------
        messages = self._helper.create_message(
            frames, camera_indices=cam_idx, num_frames_per_camera=n_per_cam)
        inputs = self._tokenize(messages)
        model_inputs = self._helper.to_device(
            {"tokenized_data": inputs}, "cuda")
        model_inputs["ego_history_xyz"] = ego_xyz
        model_inputs["ego_history_rot"] = ego_rot

        t = time.perf_counter()
        torch.cuda.manual_seed_all(42)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            pred_xyz, pred_rot, extra = \
                self.model.sample_trajectories_from_data_with_vlm_rollout(
                    data=model_inputs, top_p=self.top_p,
                    temperature=self.temperature,
                    num_traj_samples=self.traj_samples,
                    max_generation_length=self.max_new_tokens,
                    return_extra=True)
        timings["rollout_ms"] = round((time.perf_counter() - t) * 1000, 1)

        cot = _first(extra.get("cot"))
        meta_action = _first(extra.get("meta_action"))
        raw["chain_of_causation"] = cot
        raw["meta_action"] = meta_action

        # (B, n_traj_group, n_samples, T, 3) -> the first sample's 64 waypoints.
        traj = None
        try:
            xyz = pred_xyz.detach().float().cpu().numpy()
            pts = xyz[0, 0, 0]
            traj = {
                "xyz": [[round(float(v), 3) for v in p[:3]] for p in pts],
                "hz": 10, "horizon_s": round(len(pts) / 10.0, 2),
                "frame": "FLU_at_t0", "samples": self.traj_samples,
                "pixels": None,
            }
        except Exception as e:
            raw["trajectory_error"] = f"{type(e).__name__}: {e}"

        # --- 2. the prompt set --------------------------------------------
        answers = {}
        for key in ("scene", "critical_actor", "attention"):
            q = prompts.get(key)
            if not q:
                continue
            t = time.perf_counter()
            vq = self._helper.create_vqa_message(
                frames, question=q, camera_indices=cam_idx,
                num_frames_per_camera=n_per_cam)
            vin = self._helper.to_device(
                {"tokenized_data": self._tokenize(vq)}, "cuda")
            torch.cuda.manual_seed_all(42)
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                out = self.model.generate_text(
                    data=vin, top_p=self.top_p, temperature=self.temperature,
                    num_samples=1, max_generation_length=self.max_new_tokens)
            answers[key] = _first(out.get("answer"))
            raw[key] = {"question": q, "answer": answers[key],
                        "cot": _first(out.get("cot"))}
            timings[key + "_ms"] = round((time.perf_counter() - t) * 1000, 1)

        return {
            "ok": True,
            "raw": raw,
            "scene": answers.get("scene", ""),
            "critical_actor": answers.get("critical_actor", ""),
            "attention": answers.get("attention", ""),
            # Alpamayo's reasoning IS the Chain-of-Causation trace. There is no
            # separate hidden thinking block to report, so `thinking` is null
            # rather than a copy -- a corpus reader can then tell the two
            # models' trace kinds apart without knowing which is which.
            "reasoning": cot,
            "thinking": None,
            "meta_action": meta_action,
            "trajectory": traj,
            "ego_synthetic": ego_synthetic,
            "timings_ms": timings,
        }


def _first(x):
    """extract_text_tokens returns [B, num_samples] arrays. Take [0][0], safely."""
    try:
        if x is None:
            return ""
        v = x
        while hasattr(v, "__len__") and not isinstance(v, str) and len(v):
            v = v[0]
        return str(v) if not isinstance(v, str) else v
    except Exception:
        return ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8801)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default=MODEL_ID,
                    help="HF id or a local path (an FP8 checkpoint)")
    ap.add_argument("--precision", default="bf16", choices=("bf16", "fp8"))
    ap.add_argument("--depth", type=int, default=1)
    ap.add_argument("--traj-samples", type=int, default=1)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--attn", default="sdpa", choices=("sdpa", "flash_attention_2"))
    ap.add_argument("--no-warm", action="store_true")
    args = ap.parse_args()

    svc = AlpamayoService(args.model, precision=args.precision, depth=args.depth,
                          traj_samples=args.traj_samples,
                          max_new_tokens=args.max_new_tokens, attn=args.attn)
    common.serve(svc, host=args.host, port=args.port, warm=not args.no_warm)


if __name__ == "__main__":
    main()
