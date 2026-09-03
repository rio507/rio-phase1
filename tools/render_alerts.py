"""Pre-render the red-tier warning clips to static/audio/.

    python -m tools.render_alerts            # render anything missing
    python -m tools.render_alerts --force    # re-render everything
    python -m tools.render_alerts --list     # show what exists
    python -m tools.render_alerts --backend openai_realtime   # the cedar voice

RENDERED IN RIO'S OWN VOICE, by default — whichever voice that currently is.
These clips are the lines that matter most and the ones a driver hears at the
worst moment; having them arrive in a different voice from everything else RIO
says would make the most important thing she does sound like a different
product. So they come from whatever speaks her conversation, read verbatim,
once, offline — and the fast path they exist for is completely unchanged: a
local file, preloaded, no network.

OFFLINE IS WHY THE QUALITY MODEL IS USED HERE. Nothing in a car waits on this:
the render happens on a workstation, minutes at a time, and the artifact is
played months later from disk. So the ElevenLabs path renders on the SAME model
RIO converses with rather than on the fast one the live warnings use — the
argument for flash is entirely about first-byte latency, and there is no first
byte to wait for in a file that already exists.

The UNSAFE tier cannot pay a TTS round-trip. Measured on this stack an
ElevenLabs stream is 300-800 ms to first audio, and the tier exists precisely
for the situation where that is already too late — so its three lines are
rendered once, served as static files, preloaded by the browser and played
locally with no network in the path. The amber (calm) tier still uses live TTS:
it is coaching, not an alert, and a few hundred ms costs nothing there.

The words come from headway.live_policy.LINE_TEXT, so the clips and the
deterministic policy that fires them cannot drift apart.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from headway import live_policy  # noqa: E402

AUDIO_DIR = Path(__file__).resolve().parent.parent / "static" / "audio"

# WHICH VOICE THESE FILES ARE IN, written next to them.
#
# An MP3 does not say who is speaking, so "the clips are in RIO's voice" was a
# claim nobody could check — and it is exactly the claim that quietly stops
# being true the day the voice id changes and nobody re-renders. The manifest
# records what each file was made from; preflight compares it with the config
# and says so when the two have parted company, which is the whole point:
# stale clips are not a missing file, they are the wrong person saying the most
# important sentence in the system.
MANIFEST = AUDIO_DIR / "rendered.json"

# Exactly the lines whose LINE_AUDIO is a clip id rather than "tts".
CLIP_LINES = [k for k, v in live_policy.LINE_AUDIO.items() if v != "tts"]

# The tire diagnostic fast path. Two conditions are allowed to interrupt a
# driver before ordinary confirmation completes, and both are pre-rendered for
# the same reason the headway red tier is: waiting on a TTS round trip is the
# thing a fast path exists to avoid.
#
# The words live here rather than in vehicle_health_policy.LINE because they are
# fixed clips, not templates -- a pre-rendered line cannot name a corner or a
# pressure, so it says the thing that is true of all of them and the dashboard
# carries the detail. That is a real constraint of the mechanism, not a
# shortcut: a clip per corner per pressure is not a set anyone can render.
TIRE_CLIPS = {
    "tire_critical":
        "Pull over when it's safe — one of your tires is dangerously low and "
        "still going down.",
    "tire_sensor_lost":
        "I've lost the sensor on a tire that was already losing air. Check it "
        "by hand when you stop.",
}


# A clip is written once and played for months, so a wrong word in one is a
# warning that says the wrong thing every time it fires, forever. Rendering is
# therefore attempted more than once and the ARTIFACT is what gets checked --
# the finished MP3, transcribed independently, not the model's own account of
# what it said. Those are different claims, and this caught the difference:
# a first pass produced a clip Whisper heard as "check it by hand when WE stop"
# while the model's own transcript said "you". Re-rendered, it was right, and
# three further renders were all verbatim -- but a check that would have shipped
# that file is not a check.
CLIP_RENDER_ATTEMPTS = 3


def _render_realtime(text: str, tmp: Path) -> int:
    """One clip, spoken by the live model, transcoded to MP3, then verified."""
    last = ""
    for attempt in range(1, CLIP_RENDER_ATTEMPTS + 1):
        n = _render_realtime_once(text, tmp)
        heard = _transcribe(tmp)
        if not heard or _norm(heard) == _norm(text):
            return n
        last = heard
        print(f"      attempt {attempt}: not verbatim, re-rendering\n"
              f"        wanted: {text!r}\n        heard : {heard.strip()!r}")
        tmp.unlink(missing_ok=True)
    raise RuntimeError(
        f"the clip was not verbatim after {CLIP_RENDER_ATTEMPTS} attempts.\n"
        f"  asked: {text!r}\n  heard: {last!r}")


def _transcribe(path: Path) -> str:
    """What the finished file actually says, according to a different model.

    Whisper, the same transcriber every other transcript in this system comes
    from. Returns "" if transcription is unavailable, which downgrades the
    check rather than failing the render: an unverified clip in the right voice
    still beats no clip at all, and the caller says so.
    """
    try:
        from openai import OpenAI

        import config as _config

        # Handed over as a named BytesIO rather than the file object: the
        # clip is still at its ".part" path at this point (it is not put in
        # place until it has been verified), and the transcription API reads
        # the format from the name.
        import io

        buf = io.BytesIO(path.read_bytes())
        buf.name = "clip.mp3"
        return OpenAI().audio.transcriptions.create(
            model=_config.OPENAI_STT_MODEL, file=buf).text or ""
    except Exception as e:
        print(f"      (could not verify: {type(e).__name__}) ", end="")
        return ""


def _render_realtime_once(text: str, tmp: Path) -> int:
    import subprocess

    import realtime

    got = realtime.render_speech(text)
    if not got.get("ok"):
        raise RuntimeError(f"live voice returned no audio: {got.get('note')}")

    said = _norm(got.get("transcript", ""))
    if said and said != _norm(text):
        raise RuntimeError(
            "the live voice did not read the line verbatim.\n"
            f"  asked: {text!r}\n  said : {got.get('transcript')!r}")

    wav = tmp.with_suffix(".wav.part")
    wav.write_bytes(got["wav"])
    # ffmpeg rather than a Python encoder: it is already a dependency (boot.sh
    # step 2) and the browser wants MP3, which is what the clip path already
    # serves and what the preloaded <audio> elements already point at.
    try:
        proc = subprocess.run(
            # -f mp3 explicitly: the output is written to a .part file first
            # (so an interrupted run cannot leave a truncated clip for the
            # browser to preload), and ffmpeg infers format from the extension.
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
             "-codec:a", "libmp3lame", "-q:a", "4", "-f", "mp3", str(tmp)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg could not transcode the clip (exit {proc.returncode}): "
                f"{(proc.stderr or '').strip()[:300]}")
    finally:
        wav.unlink(missing_ok=True)
    return tmp.stat().st_size


def _norm(text: str) -> str:
    """Spoken-form comparison: punctuation and case are a transcriber's
    choices, the words are not."""
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (text or "").lower()).split())


def _render_elevenlabs(text: str, tmp: Path) -> int:
    """One clip in RIO's ElevenLabs voice, then checked like every other clip.

    The verification is the same one the live-voice path gets and for the same
    reason: a clip is written once and played for months, so a wrong word in
    one is a warning that says the wrong thing every time it fires, forever.
    v3 makes this MORE necessary rather than less — it is the expressive model,
    and expressiveness is exactly the thing that occasionally decides a line
    would be better with a word added to it.
    """
    last = ""
    for attempt in range(1, CLIP_RENDER_ATTEMPTS + 1):
        n = _render_elevenlabs_once(text, tmp)
        heard = _transcribe(tmp)
        if not heard or _norm(heard) == _norm(text):
            return n
        last = heard
        print(f"      attempt {attempt}: not verbatim, re-rendering\n"
              f"        wanted: {text!r}\n        heard : {heard.strip()!r}")
        tmp.unlink(missing_ok=True)
    raise RuntimeError(
        f"the clip was not verbatim after {CLIP_RENDER_ATTEMPTS} attempts.\n"
        f"  asked: {text!r}\n  heard: {last!r}")


def _render_elevenlabs_once(text: str, tmp: Path) -> int:
    import voice   # imported late: it builds an ElevenLabs client on first use

    n = 0
    with tmp.open("wb") as fh:
        for chunk in voice.synthesize_stream(
                text, backend="elevenlabs",
                model=config.ELEVENLABS_DIALOGUE_MODEL):
            fh.write(chunk)
            n += len(chunk)
    if n == 0:
        raise RuntimeError("ElevenLabs returned no audio")
    return n


def voice_signature(backend: str = None) -> dict:
    """Who these clips would be rendered by, if they were rendered now."""
    backend = backend or config.VOICE_BACKEND
    if backend == "realtime":
        backend = "openai_realtime"
    if backend == "elevenlabs":
        return {"backend": "elevenlabs",
                "voice": config.ELEVENLABS_VOICE_ID,
                "model": config.ELEVENLABS_DIALOGUE_MODEL}
    return {"backend": "openai_realtime",
            "voice": config.OPENAI_REALTIME_VOICE,
            "model": config.OPENAI_REALTIME_MODEL}


def manifest() -> dict:
    try:
        return json.loads(MANIFEST.read_text())
    except Exception:
        return {}


def _write_manifest(backend: str, rendered: list):
    """Record what was made, and from what. Merged, not replaced.

    A run with no --force re-renders nothing, and a manifest rewritten from
    that run would claim the untouched files came from today's config. Only the
    files this run actually produced get their entry updated.
    """
    sig = voice_signature(backend)
    doc = manifest()
    doc["voice"] = sig
    doc.setdefault("clips", {})
    for line, path, n, what in rendered:
        if what != "rendered":
            continue
        doc["clips"][line] = {
            **sig, "bytes": n, "at": round(time.time(), 1),
            "sha1": hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12],
        }
    MANIFEST.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def render(force: bool = False, backend: str = None) -> list:
    backend = backend or config.VOICE_BACKEND
    if backend == "realtime":
        backend = "openai_realtime"       # the old name for the same thing
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    everything = [(line, live_policy.LINE_TEXT[line]) for line in CLIP_LINES]
    everything += sorted(TIRE_CLIPS.items())
    for line, text in everything:
        path = AUDIO_DIR / f"{line}.mp3"
        if path.exists() and not force:
            out.append((line, path, path.stat().st_size, "kept"))
            continue
        # Render to a temp path and move into place, so an interrupted run can
        # never leave a truncated clip that the browser would happily preload.
        tmp = path.with_suffix(".mp3.part")
        try:
            if backend == "openai_realtime":
                n = _render_realtime(text, tmp)
            else:
                n = _render_elevenlabs(text, tmp)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(path)
        out.append((line, path, n, "rendered"))
    _write_manifest(backend, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-render existing clips")
    ap.add_argument("--list", action="store_true", help="show state and exit")
    ap.add_argument("--backend", default=None,
                    choices=["openai_realtime", "realtime", "elevenlabs"],
                    help="which voice to render in (default: config.VOICE_BACKEND)")
    args = ap.parse_args()

    if args.list:
        listing = [(k, live_policy.LINE_TEXT[k]) for k in CLIP_LINES]
        listing += sorted(TIRE_CLIPS.items())
        for line, text in listing:
            p = AUDIO_DIR / f"{line}.mp3"
            size = p.stat().st_size if p.exists() else 0
            print(f"  {line:18} {'OK ' if size else '-- '} {size:>7} B  {text!r}")
        return 0

    backend = args.backend or config.VOICE_BACKEND
    if backend == "realtime":
        backend = "openai_realtime"
    if backend == "openai_realtime" and not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set (.env)", file=sys.stderr)
        return 2
    if backend == "elevenlabs":
        import voice_dialogue

        if not voice_dialogue.configured():
            print("ELEVENLABS_API_KEY / ELEVENLABS_VOICE_ID not set (.env)",
                  file=sys.stderr)
            return 2

    if backend == "elevenlabs":
        print(f"  voice: elevenlabs {config.ELEVENLABS_VOICE_ID} "
              f"on {config.ELEVENLABS_DIALOGUE_MODEL}")
    else:
        print(f"  voice: openai_realtime ({config.OPENAI_REALTIME_VOICE})")
    for line, path, n, what in render(force=args.force, backend=backend):
        print(f"  [{what:8}] {line:16} {n:>7} B  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
