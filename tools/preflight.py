"""preflight.py — is this pod actually provisioned?

    python -m tools.preflight            # check, exit non-zero if anything is missing
    python -m tools.preflight --fix      # print the exact commands to fix it

WHY THIS EXISTS
---------------
Only /workspace survives a pod restart. Everything boot.sh installs into the
container layer -- apt packages, site-packages, the RF-DETR wheel, the Chromium
the browser suites drive -- is gone on every rebuild, and boot.sh has to put it
back. When boot.sh does not finish, the pod comes up looking fine and behaving
differently.

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


def check_browser():
    head("playwright + chromium (boot.sh step 5d) — the suites that were never run")
    fix = ("pip install --no-cache-dir playwright && "
           "python -m playwright install-deps chromium && "
           "python -m playwright install chromium")

    # WHY THIS IS HERE AT ALL. On 2026-09-09 both browser suites had been
    # unrunnable on this pod for as long as anyone had been asking for "the
    # full suite": playwright was in neither requirements.txt nor boot.sh, so
    # each exited 2 with an install hint. Nobody noticed, because a suite that
    # CANNOT RUN reads almost exactly like a suite that passes -- no failures,
    # no red, nothing in a summary line. That is the same shape as the 2026-08-02
    # detector incident in this file's header: silently degraded, and nothing to
    # ask. This is the asking.
    try:
        import playwright  # noqa: F401
        ok = True
    except Exception:
        ok = False
    check(ok, "playwright",
          "tools/output_bus_selftest.py and tools/mobile_layout_selftest.py "
          "cannot run. They are the only checks the shared audio bus and its "
          "unlock, and the mobile layout, have -- and neither can be faked in "
          "node, because what they assert is what a BROWSER does with the page.",
          fix)
    if not ok:
        return

    # Two questions, ONE driver. The module and the browser are separate
    # installs that fail separately -- `pip install playwright` alone leaves you
    # importable with nothing to drive -- and the launch is the only way to
    # catch the apt half, since install-deps is its own boot.sh command and can
    # fail on its own, leaving a binary that is present and unstartable for want
    # of libnss3 or a font. Same shape as torch above, where cuda can report
    # available while the runtime is broken. A headless launch costs ~0.1 s.
    #
    # Both asked inside a SINGLE sync_playwright(), because opening a second one
    # in the same process leaves the first connection's teardown pending and
    # prints "Task was destroyed but it is pending" and a TargetClosedError
    # traceback AFTER the summary -- noise out of a file whose entire job is a
    # checklist somebody can read.
    present, launched, detail = False, False, ""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as pw:
            present = Path(pw.chromium.executable_path).exists()
            if present:
                try:
                    browser = pw.chromium.launch()
                    detail = f" ({browser.version})"
                    browser.close()
                    launched = True
                except Exception as e:
                    # Wide enough for a path to survive: the first line of a
                    # launch failure is usually "Executable doesn't exist at
                    # <path>" or a named missing library, and a cut in the
                    # middle of either says nothing.
                    detail = (f" ({type(e).__name__}: "
                              f"{str(e).splitlines()[0][:140]})")
    except Exception as e:
        detail = f" ({type(e).__name__})"

    check(present, "chromium binary present",
          "playwright imports but has no browser to launch: both suites fail at "
          "the first line rather than reporting anything about the page.",
          "python -m playwright install chromium")
    if not present:
        return
    check(launched, "...and it launches" + detail,
          "the browser is installed but will not start — usually the system "
          "libraries, which are a separate boot.sh command from the download.",
          "python -m playwright install-deps chromium")


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
    # ASKED OF THE RENDERER rather than listed here, so a clip added to the
    # set cannot be missing from the check that exists to notice it missing.
    # A hardcoded five is how this read "9/5" the day the junction calls were
    # added -- a pass that was counting the wrong thing.
    try:
        from tools import render_alerts as _ra
        expected = (set(_ra.CLIP_LINES) | set(_ra.TIRE_CLIPS)
                    | set(_ra.IMMINENT_CLIPS))
    except Exception:
        expected = {"back_off", "too_close", "watch_distance",
                    "tire_critical", "tire_sensor_lost"}
    missing = sorted(expected - set(clips))
    check(not missing,
          f"pre-rendered clips ({len(expected) - len(missing)}/{len(expected)})",
          f"the fast paths fall back to a TTS round trip they exist to avoid, "
          f"or go silent: {missing}",
          "python -m tools.render_alerts")

    # ...and WHOSE clips they are. Five files present is not the same claim as
    # five files in the voice RIO is currently speaking in, and the second one
    # is the one that stops being true silently: change the voice id, restart,
    # and the most important sentence in the system is said by the previous
    # occupant with nothing anywhere to say so.
    try:
        from tools import render_alerts as _ra

        want = _ra.voice_signature()
        doc = _ra.manifest()
        rendered = doc.get("clips", {})
        stale = sorted(
            line for line in expected
            if (rendered.get(line, {}).get("voice") != want["voice"]
                or rendered.get(line, {}).get("backend") != want["backend"]))
        check(not stale,
              f"...and rendered in the configured voice "
              f"({want['backend']}/{want['voice'] or '<unset>'})",
              f"these clips were made by a different voice or have no record "
              f"of who made them, so the lines that matter most sound like "
              f"somebody else: {stale}",
              "python -m tools.render_alerts --force")
    except Exception as e:
        check(False, "clip voice manifest",
              f"cannot tell which voice the clips are in ({type(e).__name__})",
              "python -m tools.render_alerts --force")


def check_voice():
    head("voice (config.VOICE_BACKEND)")
    import config as _cfg

    backend = _cfg.VOICE_BACKEND
    check(backend in ("elevenlabs", "openai_realtime"),
          f"VOICE_BACKEND={backend}",
          "the backend name is not one this build knows; RIO starts with no "
          "voice at all.",
          "VOICE_BACKEND=elevenlabs|openai_realtime in .env")

    if backend != "elevenlabs":
        print(f"  (speech to speech in {_cfg.OPENAI_REALTIME_VOICE}; "
              f"ElevenLabs is the fallback only)")
        return

    # The key is never printed. What is checked is that it EXISTS and that it
    # is clean: a key pasted out of a browser carries a leading non-breaking
    # space, every call then fails with "Invalid API key", and nothing in that
    # message mentions whitespace. That failure looked exactly like a revoked
    # key and cost an afternoon.
    raw = os.environ.get("ELEVENLABS_API_KEY", "")
    check(bool(raw.strip()), "ELEVENLABS_API_KEY set",
          "RIO has no voice: conversation, warnings and turns are all silent.",
          "add it to .env")
    check(raw == raw.strip() and raw.startswith("sk_"),
          "...and free of stray whitespace",
          "the key has leading or trailing whitespace (a non-breaking space "
          "survives a copy-paste) and every call will fail as 'Invalid API "
          "key', which reads as a wrong key and is not one.",
          "re-paste the value in .env with no spaces around it")
    check(bool(_cfg.ELEVENLABS_VOICE_ID), 
          f"ELEVENLABS_VOICE_ID={_cfg.ELEVENLABS_VOICE_ID or '<unset>'}",
          "no voice id: every synthesis call is refused.",
          "add it to .env")

    for mod in ("websockets", "httpx"):
        try:
            __import__(mod)
            ok_mod = True
        except Exception:
            ok_mod = False
        check(ok_mod, mod,
              "the dialogue socket cannot be opened: RIO falls straight back "
              "to cedar for every drive." if mod == "websockets" else
              "the per-utterance fallback cannot run, so a slow v3 line is a "
              "silent line.",
              "pip install -r requirements.txt")


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


def check_teachers():
    """The shadow panel's two environments, its weights and its services.

    EVERY LINE HERE IS NON-FATAL IN SPIRIT: a pod without the teacher panel
    drives exactly as it did before the panel existed. What it cannot do is
    tell you it is missing, which is the entire reason these checks are in the
    same list as the detector's.
    """
    head("teacher panel (shadow — docs/teacher_panel.md)")
    venvs = Path("/opt/teachers/venvs")
    src = Path("/workspace/teachers/src")

    for name, mods in (("alpamayo", ("torch", "transformers", "alpamayo1_5")),
                       ("cosmos", ("torch", "transformers"))):
        py = venvs / name / "bin" / "python"
        ok = py.exists()
        if ok:
            import subprocess
            probe = "import " + ",".join(mods)
            ok = subprocess.call([str(py), "-c", probe],
                                 stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL) == 0
        check(ok, f"teacher env: {name}",
              f"the {name} service cannot start, so its column on the "
              f"Teachers card stays empty and no corpus row is written "
              f"(the drive itself is unaffected)",
              "bash /workspace/boot.sh teachers-build")

    for name, sha in (("alpamayo1.5", "36aeb4c"), ("cosmos-reason2", "a3b4a1d")):
        d = src / name
        ok = (d / ".git").exists()
        check(ok, f"teacher source: {name} @ {sha}",
              f"{name}'s pinned inference code is missing — the environment "
              f"cannot be rebuilt from it",
              "bash /workspace/boot.sh teachers-build")

    hub = Path(os.environ.get("HF_HOME", "/workspace/.cache/huggingface")) / "hub"
    for repo, gb in (("models--nvidia--Alpamayo-1.5-10B", 20),
                     ("models--nvidia--Cosmos-Reason2-8B", 15)):
        d = hub / repo
        size = 0
        if d.exists():
            for f in d.rglob("*"):
                try:
                    if f.is_file() and not f.is_symlink():
                        size += f.stat().st_size
                except OSError:
                    pass
        ok = size > gb * 0.8 * 1024 ** 3
        pretty = repo.replace("models--", "").replace("--", "/")
        check(ok, f"weights: {pretty} ({round(size / 1024 ** 3, 1)} GB)",
              f"{pretty} is not cached — that teacher cannot load. "
              f"Cosmos-Reason2 is a GATED repo and needs an HF token whose "
              f"account has accepted the licence; Alpamayo needs it too, "
              f"because it loads its tokenizer and VLM config from it.",
              "hf download " + pretty)

    tok = Path("/workspace/teachers/secrets.env")
    check(tok.exists(), "HF token for the gated Cosmos repo",
          "nvidia/Cosmos-Reason2-8B is gated; without a token NEITHER teacher "
          "loads, because Alpamayo reads its tokenizer and VLM config from "
          "that repo",
          "write HF_TOKEN=... to /workspace/teachers/secrets.env "
          "(outside the git worktree, and it must stay there)")

    import urllib.request
    for port, name in ((8801, "alpamayo1.5"), (8802, "cosmos-reason2")):
        try:
            with urllib.request.urlopen(
                    f"http://127.0.0.1:{port}/health", timeout=3) as r:
                import json as _json
                ok = bool(_json.loads(r.read().decode()).get("loaded"))
        except Exception:
            ok = False
        check(ok, f"teacher service: {name} on :{port}",
              f"{name} is not answering — its column stays empty and no "
              f"corpus row is written for any keyframe",
              "bash /workspace/boot.sh teachers")


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
    check_voice()
    check_detector()
    check_browser()
    check_weights()
    check_persistent()
    check_teachers()
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
