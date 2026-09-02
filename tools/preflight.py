"""preflight.py — is this pod actually provisioned?

    python -m tools.preflight            # check, exit non-zero if anything is missing
    python -m tools.preflight --fix      # print the exact commands to fix it

WHY THIS EXISTS
---------------
Only /workspace survives a pod restart. Everything boot.sh installs into the
container layer -- apt packages, site-packages, the RF-DETR wheel -- is gone on
every rebuild, and boot.sh has to put it back. When boot.sh does not finish, the
pod comes up looking fine and behaving differently.

On 2026-08-02 that happened. boot.sh completed steps 1-4 and never reached step
5c, so RF-DETR was absent for a whole day: headway had no candidate source and
reported UNKNOWN, and the visual conversation's scene graph was empty. Nothing
complained, because a missing detector is deliberately non-fatal -- a pod that
comes up degraded is better than a pod that does not come up. The cost of that
choice is that "degraded" has to be something you can ASK about, and until this
file there was nothing to ask.

Reconstructing what had happened took package mtimes and dpkg timestamps,
because boot.sh's only record was scrollback in a terminal that had gone. That
is fixed at the other end (boot.sh now tees to /workspace/boot.log); this is the
half that answers the question directly.

WHAT IT CHECKS
--------------
Only things that live on the EPHEMERAL container layer and can therefore vanish
without anyone touching the repo. Weights and caches on /workspace are checked
too, because a fetch can fail, but they are not the interesting case.

Each check names what breaks when it fails, in terms of what RIO does or stops
doing -- not "scipy missing" but "no headway candidates and an empty scene
graph". A checklist that does not say what it is protecting is a checklist
people learn to skip.
"""
import argparse
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, "/workspace/rio-phase1")

REPO = Path("/workspace/rio-phase1")

_results = []


def check(ok, name, consequence, fix=""):
    _results.append((bool(ok), name, consequence, fix))
    mark = "ok  " if ok else "MISS"
    print(f"  [{mark}] {name}")
    if not ok:
        print(f"         -> {consequence}")
    return bool(ok)


def head(title):
    print(f"\n{title}")


# ---------------------------------------------------------------------------

def check_interpreter():
    head("interpreter")
    print(f"  python {sys.version.split()[0]} at {sys.executable}")
    # Not a pass/fail on its own -- it is the context for every import below.
    # The detector import was once pinned to a hardcoded python3.12 path and
    # silently died when the pod was rebuilt on 3.11, which is why this is
    # printed rather than assumed.
    return True


def check_apt():
    head("apt packages (boot.sh step 2)")
    check(shutil.which("ffmpeg"), "ffmpeg",
          "Whisper gets no audio: /talk fails on every utterance.",
          "apt-get install -y ffmpeg")
    check(shutil.which("node") or shutil.which("nodejs"), "nodejs",
          "the two JavaScript test suites cannot run: `node tools/nav_selftest.js` "
          "and `node tools/realtime_selftest.js` are the only checks the route "
          "tracker and the speech arbiter have, and both live in the browser.",
          "apt-get install -y nodejs")
    check(shutil.which("git"), "git", "no version control in the container.",
          "apt-get install -y git")


def check_core_packages():
    head("python packages (boot.sh step 3)")
    for mod, why in (
        ("fastapi", "no server at all."),
        ("uvicorn", "no server at all."),
        ("openai", "no conversation: /talk and /ask both fail."),
        ("elevenlabs", "RIO has no voice — every announcement is silent."),
        ("transformers", "no Qwen3-VL: RIO cannot see anything."),
        ("cv2", "no frame decoding, no tracker, no headway."),
    ):
        try:
            __import__(mod)
            ok = True
        except Exception:
            ok = False
        check(ok, mod, why, "pip install -r requirements.txt")


def check_torch():
    head("torch + CUDA (boot.sh step 4-5)")
    try:
        import torch
    except Exception as e:
        check(False, "torch", f"nothing that uses a model runs. ({e})",
              "see boot.sh step 4")
        return
    print(f"  torch {torch.__version__} (cuda build {torch.version.cuda})")
    ok = torch.cuda.is_available()
    check(ok, "cuda available",
          "every model falls back to CPU or fails outright.",
          "check the driver/wheel match — boot.sh step 4")
    if ok:
        try:
            x = torch.randn(256, 256, device="cuda")
            torch.cuda.synchronize()
            float((x @ x).sum())
            check(True, "cuda matmul", "")
        except Exception as e:
            # is_available() can be True while the runtime is broken by a
            # driver/wheel mismatch, so the GPU is actually touched.
            check(False, "cuda matmul",
                  f"the GPU reports present but does not work: {e}",
                  "reinstall torch — boot.sh step 4")


def check_detector():
    head("RF-DETR detector (boot.sh step 5c) — the one that went missing")
    fix = ("pip install --no-cache-dir --no-deps rfdetr==1.5.0 "
           "supervision==0.29.1 pycocotools peft && "
           "pip install --no-cache-dir --no-deps scipy")

    try:
        import scipy.optimize  # noqa: F401
        ok = True
    except Exception:
        ok = False
    # scipy is imported at IMPORT time by rfdetr.models.matcher, so without it
    # the sideways import dies and the pod has no detector at all.
    check(ok, "scipy",
          "rfdetr.models.matcher cannot import: no detector, so headway has no "
          "candidates and the visual scene graph is empty.", fix)

    try:
        from headway import detect
        detect._rfdetr_models()
        pkg = detect._pkg_dir()
        ok = True
    except Exception as e:
        pkg, ok = str(e), False
    check(ok, "rfdetr importable",
          "headway reports UNKNOWN for the whole drive and RIO can see nothing "
          "to talk about. NON-FATAL by design, which is exactly why it goes "
          "unnoticed.", fix)
    if ok:
        print(f"         from {pkg}")

    try:
        import cv2
        ok = (any(hasattr(cv2, n) for n in ("TrackerCSRT_create", "TrackerCSRT"))
              or hasattr(cv2, "legacy"))
        ver = cv2.__version__
    except Exception as e:
        ok, ver = False, str(e)
    check(ok, f"cv2 tracking API (cv2 {ver})",
          "no CSRT: the headway tracker cannot follow a lead vehicle between "
          "anchor frames. Usually means a stray opencv-python shadowed the "
          "pinned contrib-headless build.",
          "pip uninstall -y opencv-python && pip install --force-reinstall "
          "opencv-contrib-python-headless")


def check_weights():
    head("weights (persistent volume — survive a rebuild)")
    for name, path, why in (
        ("RF-DETR nano", REPO / "weights/rf-detr-nano.pth",
         "no detector candidates even with the package installed."),
        ("UFLDv2 lanes", REPO / "weights/culane_res18.pth",
         "headway falls back to the static trapezoid corridor."),
    ):
        p = Path(path)
        ok = p.exists() and p.stat().st_size > 1_000_000
        check(ok, f"{name}  {p.name}", why,
              f"cd {REPO} && python -m tools.fetch_detector_weights")

    audio = REPO / "static/audio"
    clips = sorted(p.stem for p in audio.glob("*.mp3")) if audio.exists() else []
    expected = {"back_off", "too_close", "watch_distance",
                "tire_critical", "tire_sensor_lost"}
    missing = sorted(expected - set(clips))
    check(not missing, f"pre-rendered alert clips ({len(clips)}/{len(expected)})",
          f"the fast paths fall back to a TTS round trip they exist to avoid, "
          f"or go silent: {missing}",
          "python -m tools.render_alerts")


def check_persistent():
    head("persistent volume")
    # This is a property of the SHELL preflight is running in, not of the pod:
    # it fails in any shell that has not sourced the environment, which means
    # every non-interactive one and every one that predates the last boot.sh.
    # The fix is one line, not a re-provision -- and boot.sh now keeps the value
    # in /workspace/env.sh so there is a durable file to source. It used to
    # append the export to ~/.bashrc, which lives in the container layer, so the
    # persistence lasted exactly until the next rebuild and this check came up
    # unset on every fresh pod.
    hf = os.environ.get("HF_HOME", "")
    check(hf.startswith("/workspace"), f"HF_HOME={hf or '<unset>'}",
          "anything started from THIS shell caches into the container layer and "
          "re-downloads 16 GB of Qwen3-VL on every pod start. (The uvicorn "
          "boot.sh launched is unaffected — step 7 passes HF_HOME explicitly.)",
          ". /workspace/env.sh   # written by boot.sh step 1, and sourced from "
          "~/.bashrc for new shells")

    env_file = Path("/workspace/env.sh")
    has_env = env_file.exists() and "HF_HOME" in env_file.read_text()
    check(has_env, "/workspace/env.sh present",
          "there is no durable copy of the pod's environment to source, so "
          "HF_HOME exists only in whatever shell boot.sh happened to run in.",
          "bash /workspace/boot.sh   # step 1 writes it")

    boot_log = Path("/workspace/boot.log")
    check(boot_log.exists(), "boot.log present",
          "boot.sh has not run since it learned to keep a log — if this pod is "
          "missing something, there is no record of where provisioning stopped.",
          "bash /workspace/boot.sh")

    # /workspace/boot.sh is a COPY of the repo's, and it is the one everything
    # points at, because only /workspace survives a pod restart. On 2026-08-26
    # it turned out to be three weeks stale: no RF-DETR step, no detector
    # weights, no boot log, no preflight — i.e. the exact version whose missing
    # step 5c caused the 2026-08-02 incident this file's header describes.
    #
    # Comparing them is the only check here that protects a FUTURE pod rather
    # than this one: everything else asks what is missing now, and this asks
    # whether the thing that puts it back is the current one. A comment saying
    # "remember to copy it" was what existed before, and it is what failed.
    repo_boot = REPO / "boot.sh"
    live_boot = Path("/workspace/boot.sh")
    if not repo_boot.exists():
        check(False, "boot.sh in the repo",
              "there is no source copy to compare against or to provision from.",
              "git -C /workspace/rio-phase1 checkout boot.sh")
    elif not live_boot.exists():
        check(False, "/workspace/boot.sh present",
              "the documented provisioning command points at a file that does "
              "not exist, so a rebuilt pod has nothing to run.",
              "cp /workspace/rio-phase1/boot.sh /workspace/boot.sh")
    else:
        same = repo_boot.read_bytes() == live_boot.read_bytes()
        detail = ""
        if not same:
            repo_n = len(repo_boot.read_text().splitlines())
            live_n = len(live_boot.read_text().splitlines())
            detail = f" (repo {repo_n} lines, /workspace {live_n})"
        check(same, "/workspace/boot.sh matches the repo" + detail,
              "the documented provisioning command runs a DIFFERENT script from "
              "the one in git. The last time these drifted, the copy predated "
              "RF-DETR: re-provisioning would have brought the pod up with no "
              "detector, no boot log and no preflight, and the only symptom "
              "would have been an empty scene graph.",
              "cp /workspace/rio-phase1/boot.sh /workspace/boot.sh")


def check_server():
    head("server")
    try:
        import urllib.request
        with urllib.request.urlopen("http://127.0.0.1:8888/health", timeout=2) as r:
            ok = r.status == 200
    except Exception:
        ok = False
    check(ok, "uvicorn answering on :8888",
          "RIO is not running. (Not a provisioning fault on its own — this "
          "check is here so one command answers 'is the pod up and complete'.)",
          "cd /workspace/rio-phase1 && HF_HOME=/workspace/.cache/huggingface "
          "nohup uvicorn app:app --host 0.0.0.0 --port 8888 "
          "> uvicorn.log 2>&1 &")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--fix", action="store_true",
                    help="print the commands that would repair what is missing")
    args = ap.parse_args()

    print("=" * 72)
    print("RIO preflight — what this pod is missing after a rebuild")
    print("=" * 72)

    check_interpreter()
    check_apt()
    check_core_packages()
    check_torch()
    check_detector()
    check_weights()
    check_persistent()
    check_server()

    missing = [r for r in _results if not r[0]]
    print("\n" + "=" * 72)
    print(f"{len(_results) - len(missing)}/{len(_results)} checks passed")
    if missing:
        print("\nMISSING:")
        for _, name, consequence, _fix in missing:
            print(f"  - {name}\n      {consequence}")
        if args.fix:
            print("\nTO FIX:")
            seen = set()
            for _, _, _, fix in missing:
                if fix and fix not in seen:
                    seen.add(fix)
                    print(f"  {fix}")
        else:
            print("\n  (re-run with --fix for the commands)")
    print("=" * 72)
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
