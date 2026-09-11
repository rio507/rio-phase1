"""Does gpt-live-1 mistake RIO's own voice for the driver?

    python -m tools.live_echo_probe
    python -m tools.live_echo_probe --atten 0,-6,-12,-20

THE FAILURE THIS IS ABOUT is the one config.py's barge-in section was written
for, and it is physical rather than algorithmic: a phone on a mount puts RIO's
voice out of a loudspeaker eight inches from the microphone, at driving volume,
and only SOME of what she says goes out through a renderer the echo canceller
has a reference for. The live session's own WebRTC audio is cancelled. A
pre-rendered clip and the TTS fallback are ordinary media playback, and they
come back into the microphone at full level.

WHY IT NEEDED RE-ASKING ON THIS BACKEND. gpt-live-1 is advertised as handling
background noise and interruptions natively, +30pp on Full Duplex Bench, and
the obvious reading of that is "the echo gate is now dead weight". The obvious
reading is wrong, and this is the tool that says so.

WHAT IT DOES: opens a session with nobody in the car, plays one of RIO's own
lines into the microphone at a given attenuation, and reports whether the
session transcribed it as the driver and answered it.

WHAT IT CANNOT TELL YOU: how loud the echo actually is in a particular cabin.
That is a property of the cabin. This tells you what the session does with it
at each level, which is the half that lives in this repository.
"""
import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv                              # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                               # noqa: E402
from live_harness import LiveSession, ScriptedMic, RATE, silence  # noqa: E402


def load48(path: Path) -> np.ndarray:
    out = subprocess.run(
        ["ffmpeg", "-v", "quiet", "-i", str(path), "-f", "s16le", "-ac", "1",
         "-ar", str(RATE), "-"], capture_output=True).stdout
    return np.frombuffer(out, "<i2").astype(np.int16)


def her_own_lines() -> list:
    """Clips of RIO's own voice, whichever voice this install has rendered."""
    import render_alerts as ra
    out = []
    for backend in (config.VOICE_BACKEND, "gpt_live", "openai_realtime"):
        d = ra.audio_dir(backend)
        for name in ("back_off", "too_close", "watch_distance"):
            p = d / f"{name}.mp3"
            if p.exists():
                out.append((f"{backend}/{name}", p))
        if out:
            break
    return out


async def trial(clip: np.ndarray, atten_db: float, label: str) -> bool:
    gain = 10 ** (atten_db / 20.0)
    echo = (clip.astype(np.float32) * gain).astype(np.int16)
    mic = ScriptedMic(np.concatenate([silence(1500), echo, silence(9000)]))
    s = LiveSession(mic)
    await s.open()
    await asyncio.sleep(15.0)
    await s.close()
    replied = bool(s.said.strip())
    print(f"  {label:<26} {atten_db:>5.0f} dB  "
          f"heard={s.heard.strip()[:34]!r:<38} "
          f"{'ANSWERED IT' if replied else 'ignored it'}  "
          f"{s.said.strip()[:34]!r}", flush=True)
    return replied


async def main_async(args) -> int:
    clips = her_own_lines()
    if not clips:
        print("no rendered clips to play back; run tools.render_alerts first")
        return 2
    attens = [float(x) for x in args.atten.split(",") if x.strip()]
    fired = total = 0
    for atten in attens:
        for name, path in clips[:2]:
            total += 1
            fired += await trial(load48(path), atten, name)
    print(f"\n  her own voice treated as a driver turn: {fired}/{total}")
    if fired:
        print("  => the echo gate is still load-bearing. "
              "config.GUARD_ECHO_GATE stays on.")
    else:
        print("  => the model refused every one. Worth re-reading "
              "config.GUARD_ECHO_GATE against this run.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--atten", default="0,-6,-12,-20",
                    help="attenuations in dB, comma separated")
    return asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
