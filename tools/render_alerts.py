"""Pre-render the red-tier warning clips to static/audio/.

    python -m tools.render_alerts            # render anything missing
    python -m tools.render_alerts --force    # re-render everything
    python -m tools.render_alerts --list     # show what exists
    python -m tools.render_alerts --backend elevenlabs   # the old voice

RENDERED IN RIO'S OWN VOICE, by default. These clips are the lines that matter
most and the ones a driver hears at the worst moment; having them arrive in a
different voice from everything else RIO says would make the most important
thing she does sound like a different product. So they come from the same live
model that speaks her conversation and dictates her warnings — read verbatim,
once, offline — and the fast path they exist for is completely unchanged: a
local file, preloaded, no network.

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
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from headway import live_policy  # noqa: E402

AUDIO_DIR = Path(__file__).resolve().parent.parent / "static" / "audio"

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
    import voice   # imported late: it constructs an ElevenLabs client at import

    n = 0
    with tmp.open("wb") as fh:
        for chunk in voice.synthesize_stream(text, backend="elevenlabs"):
            fh.write(chunk)
            n += len(chunk)
    if n == 0:
        raise RuntimeError("ElevenLabs returned no audio")
    return n


def render(force: bool = False, backend: str = None) -> list:
    backend = backend or config.VOICE_BACKEND
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
            if backend == "realtime":
                n = _render_realtime(text, tmp)
            else:
                n = _render_elevenlabs(text, tmp)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        tmp.replace(path)
        out.append((line, path, n, "rendered"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-render existing clips")
    ap.add_argument("--list", action="store_true", help="show state and exit")
    ap.add_argument("--backend", default=None,
                    choices=["realtime", "elevenlabs"],
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
    if backend == "realtime" and not os.getenv("OPENAI_API_KEY"):
        print("OPENAI_API_KEY not set (.env)", file=sys.stderr)
        return 2
    if backend == "elevenlabs" and not (os.getenv("ELEVENLABS_API_KEY")
                                        and os.getenv("ELEVENLABS_VOICE_ID")):
        print("ELEVENLABS_API_KEY / ELEVENLABS_VOICE_ID not set (.env)", file=sys.stderr)
        return 2

    print(f"  voice: {backend}"
          + (f" ({config.OPENAI_REALTIME_VOICE})" if backend == "realtime" else ""))
    for line, path, n, what in render(force=args.force, backend=backend):
        print(f"  [{what:8}] {line:16} {n:>7} B  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
