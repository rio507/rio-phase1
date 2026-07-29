"""Pre-render the red-tier warning clips to static/audio/ via ElevenLabs.

    python -m tools.render_alerts            # render anything missing
    python -m tools.render_alerts --force    # re-render everything
    python -m tools.render_alerts --list     # show what exists

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
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from headway import live_policy  # noqa: E402

AUDIO_DIR = Path(__file__).resolve().parent.parent / "static" / "audio"

# Exactly the lines whose LINE_AUDIO is a clip id rather than "tts".
CLIP_LINES = [k for k, v in live_policy.LINE_AUDIO.items() if v != "tts"]


def render(force: bool = False) -> list:
    import voice   # imported late: it constructs an ElevenLabs client at import

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for line in CLIP_LINES:
        text = live_policy.LINE_TEXT[line]
        path = AUDIO_DIR / f"{line}.mp3"
        if path.exists() and not force:
            out.append((line, path, path.stat().st_size, "kept"))
            continue
        # Render to a temp path and move into place, so an interrupted run can
        # never leave a truncated clip that the browser would happily preload.
        tmp = path.with_suffix(".mp3.part")
        n = 0
        with tmp.open("wb") as fh:
            for chunk in voice.synthesize_stream(text):
                fh.write(chunk)
                n += len(chunk)
        if n == 0:
            tmp.unlink(missing_ok=True)
            raise RuntimeError(f"ElevenLabs returned no audio for {line!r}")
        tmp.replace(path)
        out.append((line, path, n, "rendered"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--force", action="store_true", help="re-render existing clips")
    ap.add_argument("--list", action="store_true", help="show state and exit")
    args = ap.parse_args()

    if args.list:
        for line in CLIP_LINES:
            p = AUDIO_DIR / f"{line}.mp3"
            size = p.stat().st_size if p.exists() else 0
            print(f"  {line:16} {'OK ' if size else '-- '} {size:>7} B  "
                  f"{live_policy.LINE_TEXT[line]!r}")
        return 0

    if not os.getenv("ELEVENLABS_API_KEY") or not os.getenv("ELEVENLABS_VOICE_ID"):
        print("ELEVENLABS_API_KEY / ELEVENLABS_VOICE_ID not set (.env)", file=sys.stderr)
        return 2

    for line, path, n, what in render(force=args.force):
        print(f"  [{what:8}] {line:16} {n:>7} B  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
