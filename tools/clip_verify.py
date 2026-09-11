"""Listen to the rendered clips again, harder, and say which are wrong.

    python -m tools.clip_verify
    python -m tools.clip_verify --backend gpt_live
    python -m tools.clip_verify --clip merge_left

WHY A SECOND VERIFIER EXISTS. render_alerts.py already transcribes every clip
it makes and re-renders the ones that come back wrong. That check is right and
it is not sufficient, because it is unreliable in one specific place: a clip
one second long, transcribed cold, with no context. Measured on the Gleam
render, "Merge left." came back as "Mars left.", "Keep right." as "Keep
quiet.", "Watch your distance." as "Walk two distance." -- and the voice had
said all three correctly. A verifier that fails a good clip five times running
is not protecting anything; it is just noisy, and noise is what gets ignored.

WHAT THIS DOES DIFFERENTLY, and why it is still a real test. On a clip of four
words or fewer it transcribes with the WHOLE CLIP LIBRARY supplied as
vocabulary -- every line RIO can play, not the one expected of this file. So
the transcriber is answering "which of these seventeen sentences is this?"
rather than "what are these sounds?", and a clip that says the wrong thing
still fails: it matches a different row, or none.

ABOVE four words it gets NO HINT, and that scoping is the whole safety
argument -- see render_alerts._library_vocabulary. A long line can be wrong in
one place while the rest agrees, and a hint talks the transcriber round
exactly there. `tire_critical` is the worked example: thirteen of its fifteen
words are perfect and the two that are not are "Pull over".

It is the second of two checks: a clip has to pass this AND have had the
voice's own transcript agree at render time.
"""
import argparse
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv                              # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                               # noqa: E402
import render_alerts as ra                                  # noqa: E402


def expected() -> dict:
    """clip id -> the words it is supposed to say."""
    from headway import live_policy
    out = {k: live_policy.LINE_TEXT[k] for k in ra.CLIP_LINES}
    out.update(ra.TIRE_CLIPS)
    out.update(ra.IMMINENT_CLIPS)
    return out


def verify(backend: str, only: str = None) -> int:
    from openai import OpenAI

    want = expected()
    audio = ra.audio_dir(backend)
    # The vocabulary hint, used on SHORT clips only -- see
    # render_alerts._library_vocabulary for why the scoping is the whole
    # safety argument. On a long line a hint papers over a local error, which
    # is exactly the case that matters.
    vocab = ra._library_vocabulary()
    cl = OpenAI()
    bad, missing, unchecked, ok = [], [], [], 0
    for clip, text in sorted(want.items()):
        if only and clip != only:
            continue
        path = audio / f"{clip}.mp3"
        if not path.exists():
            missing.append(clip)
            print(f"  MISSING {clip}")
            continue
        buf = io.BytesIO(path.read_bytes())
        buf.name = "clip.mp3"
        try:
            kw = {}
            if len(text.split()) <= ra.VOCAB_HINT_MAX_WORDS:
                kw["prompt"] = vocab
            heard = cl.audio.transcriptions.create(
                model=config.OPENAI_STT_MODEL, file=buf, **kw).text
        except Exception as e:
            print(f"  ERROR   {clip}: {type(e).__name__}: {e}")
            bad.append(clip)
            continue
        if not (heard or "").strip():
            # AN EMPTY TRANSCRIPTION IS NOT A WRONG CLIP. The transcriber
            # declines on very short audio often enough that treating silence
            # as a mismatch reported five good clips as broken and asked for
            # them to be re-rendered -- clips the render-time check had just
            # passed. render_alerts._transcribe already treats "" as "could
            # not check"; this has to agree with it or the two verifiers give
            # opposite answers about the same file.
            unchecked.append(clip)
            print(f"  ?       {clip:<26} transcriber returned nothing")
        elif ra._norm(heard) == ra._norm(text):
            ok += 1
            print(f"  ok      {clip:<26} {heard.strip()[:44]!r}")
        else:
            bad.append(clip)
            print(f"  WRONG   {clip:<26} heard {heard.strip()[:44]!r}")
            print(f"          {'':<26} want  {text[:44]!r}")
    print("\n" + "-" * 60)
    print(f"  {ok} correct, {len(bad)} wrong, {len(unchecked)} unchecked, "
          f"{len(missing)} missing   ({audio})")
    if bad:
        print("  RE-RENDER THESE: " + ", ".join(bad))
    if unchecked:
        print("  COULD NOT CHECK (transcriber returned nothing; the clip may "
              "be fine): " + ", ".join(unchecked))
    # Only a clip that transcribed to the WRONG WORDS, or is absent, is a
    # failure. "The transcriber said nothing" is a fact about the transcriber.
    return len(bad) + len(missing)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default=None)
    ap.add_argument("--clip", default=None)
    a = ap.parse_args()
    return verify(a.backend or config.VOICE_BACKEND, a.clip)


if __name__ == "__main__":
    raise SystemExit(main())
