"""Pre-render the lines that cannot wait, to static/audio/.

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
from navigation import speech as nav_speech  # noqa: E402

_STATIC_AUDIO = Path(__file__).resolve().parent.parent / "static" / "audio"


def audio_dir(backend: str = None) -> Path:
    """Where THIS voice's clips are written.

    Per voice, because the clips cannot be re-made at the moment they are
    needed and a backend switch would otherwise leave the previous voice in
    the three lines that matter most. config.CLIP_DIRS holds the argument and
    the URL prefix; this resolves the same answer on disk.
    """
    base = config.CLIP_DIRS.get(backend or config.VOICE_BACKEND,
                                "/static/audio/")
    rel = base.replace("/static/audio/", "").strip("/")
    return _STATIC_AUDIO / rel if rel else _STATIC_AUDIO


# The shipped set's directory, for the callers that predate per-voice clips.
AUDIO_DIR = _STATIC_AUDIO


class ClipUnverified(RuntimeError):
    """One clip could not be proved correct. The others are unaffected."""

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


def manifest_path(backend: str = None) -> Path:
    """The record of what was rendered, next to the clips it describes.

    Per voice for the same reason the clips are: one manifest covering two
    directories would say the Gleam set came from marin the moment either was
    re-rendered.
    """
    return audio_dir(backend) / "rendered.json"

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
# ONE ENTRY, NOT TWO, SINCE 2026-09-11, and the one that left is the point of
# the split. `tire_sensor_lost` said "check it by hand when you stop" -- an
# action that is not NOW -- so it has no business paying the price of a fixed
# clip. It is phrased fresh from the event instead (safety_speech.py), which
# also lets it say the corner and the last reading, which a clip never could.
#
# The one that stays was rewritten. "Pull over when it's safe - one of your
# tires is dangerously low and still going down." is fifteen words in the tier
# defined by not having time for them, and it was also the line the Gleam voice
# could not render intelligibly -- see docs/live_gpt_live.md. The replacement
# leads with why and puts the action second, and "when you can" keeps the
# judgement where it belongs: pulling over is the driver's call, and RIO does
# not grab the wheel.
TIRE_CLIPS = {
    "tire_critical":
        "Tire's going down fast — pull over when you can.",
}

# THE IMMINENT TURN CALL, for the same reason and by the same argument.
#
# "Left here." at the junction is the most time-critical sentence RIO says that
# is not a safety warning, and while it was dictated it was also the one most
# likely to arrive in the fallback voice: it keeps the tightest budget in the
# system precisely because it cannot be late, and a tight budget is a budget
# that gets missed. Measured on a clean navigating drive, it was the only line
# of eleven that fell back.
#
# It can be a file where the other turn calls cannot, and the difference is not
# a preference: every other call names a road, and there is no set of roads to
# render. The imminent call names nothing — it is two words on purpose — so the
# whole set of sentences it can ever produce is four, and four is a set.
#
# ASKED OF navigation.speech RATHER THAN LISTED HERE. A second copy of these
# sentences is a second copy to forget, and forgetting this one means a turn
# called in the wrong voice at the moment it matters most.
IMMINENT_CLIPS = dict(nav_speech.imminent_clips())


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
        heard = _transcribe(tmp, text)
        if not heard or _norm(heard) == _norm(text):
            return n
        last = heard
        print(f"      attempt {attempt}: not verbatim, re-rendering\n"
              f"        wanted: {text!r}\n        heard : {heard.strip()!r}")
        tmp.unlink(missing_ok=True)
    raise RuntimeError(
        f"the clip was not verbatim after {CLIP_RENDER_ATTEMPTS} attempts.\n"
        f"  asked: {text!r}\n  heard: {last!r}")


def _render_live(text: str, tmp: Path) -> int:
    """One clip in Gleam, recorded off a gpt-live-1 session, then verified.

    THE SAME RETRY-AND-CHECK AS THE REALTIME PATH, and it earns its keep here
    rather than merely inheriting it. gpt-live-1 has no verbatim event: the
    line is a REQUEST, the model is documented as free to paraphrase it, and
    the only thing standing between that and a safety clip saying the wrong
    words forever is this loop and the independent transcription under it.
    """
    last = ""
    # MORE ATTEMPTS THAN THE REALTIME PATH GETS, and for a reason that is not
    # the model's reliability. A render here is a whole WebRTC session, and a
    # session can fail to come up, come up and stay silent, or come up and be
    # transcribed wrongly -- "You're too close." came back from Whisper as "The
    # chinquos.", which is a two-word clip defeating the verifier rather than
    # the voice getting it wrong. Each of those is worth another go; none of
    # them is worth aborting a seventeen-clip run.
    for attempt in range(1, CLIP_RENDER_ATTEMPTS * 2 + 1):
        try:
            n = _render_live_once(text, tmp)
        except RuntimeError as e:
            last = str(e)
            print(f"      attempt {attempt}: {e}")
            tmp.unlink(missing_ok=True)
            continue
        heard = _transcribe(tmp, text)
        if not heard or _norm(heard) == _norm(text):
            return n
        last = heard.strip()
        print(f"      attempt {attempt}: not verbatim, re-rendering\n"
              f"        wanted: {text!r}\n        heard : {heard.strip()!r}")
        tmp.unlink(missing_ok=True)
    # AN UNVERIFIED CLIP DOES NOT SHIP. EVER.
    #
    # An earlier version of this kept the last render and printed a warning,
    # on the reasoning that the transcriber is unreliable on a one-second clip
    # and the voice's own transcript had agreed every time. That reasoning is
    # wrong, and the clip that proved it is worth keeping in the comment:
    #
    #   asked: "Pull over when it's safe - one of your tires is dangerously
    #           low and still going down."
    #   said : "Hey, Ava, when it's safe, one of your tires is dangerously
    #           low and still going down."
    #
    # The voice's own transcript said that clip was correct. It was not. The
    # instruction to pull over -- the entire action the warning exists to
    # request -- had been replaced by a greeting, and a keep-and-flag policy
    # would have written it to disk and played it at the worst moment for
    # months. This is the exact failure the note above CLIP_RENDER_ATTEMPTS
    # already described, in the other direction.
    #
    # So: the file is removed, the run CONTINUES, and the caller is told. A
    # missing clip is not silence -- rio_speak falls through to dictation and
    # then to the synthesiser, which is a warning in a slightly different voice.
    # A wrong clip is a warning that says the wrong thing, and there is no tier
    # underneath that to catch it.
    tmp.unlink(missing_ok=True)
    raise ClipUnverified(
        f"not verbatim after {CLIP_RENDER_ATTEMPTS * 2} attempts; "
        f"last heard {last!r}")


def _render_live_once(text: str, tmp: Path) -> int:
    """One take: a session, the line, the audio, transcoded to MP3.

    The voice's own transcript is checked here and is NOT sufficient -- see
    _render_live, which transcribes the finished file with a different model.
    Both checks exist because each has caught the other being wrong.
    """
    import subprocess

    import live

    got = live.render_speech(text)
    if not got.get("ok"):
        raise RuntimeError(f"the live voice returned no audio: "
                           f"{got.get('note')}")
    said = _norm(got.get("transcript", ""))
    if said and said != _norm(text):
        raise RuntimeError(
            "the live voice did not read the line verbatim.\n"
            f"  asked: {text!r}\n  said : {got.get('transcript')!r}")
    wav = tmp.with_suffix(".wav.part")
    wav.write_bytes(got["wav"])
    try:
        proc = subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
             "-codec:a", "libmp3lame", "-q:a", "4", "-f", "mp3", str(tmp)],
            capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"ffmpeg could not transcode the clip "
                f"(exit {proc.returncode}): "
                f"{(proc.stderr or '').strip()[:300]}")
    finally:
        wav.unlink(missing_ok=True)
    return tmp.stat().st_size


# HOW SHORT IS "TOO SHORT TO TRANSCRIBE COLD".
#
# Four words. Below it the transcriber is measurably unreliable -- "Merge."
# came back as "March.", "Keep left." as "He left.", "Make a U-turn." as
# "WikiU turn." -- and the voice had said all three correctly. Above it, it is
# fine, and the hint below must not be used.
VOCAB_HINT_MAX_WORDS = 4


def _library_vocabulary() -> str:
    """Every line RIO can play, as a transcription hint for SHORT clips only.

    WHY THIS IS SCOPED, and the scoping is the whole safety argument.
    Supplying the library as a hint biases the transcriber towards the lines in
    it -- that is what makes it useful on a one-word clip and what makes it
    dangerous on a long one.

    On a clip of one to four words there is nowhere for an error to hide: the
    clip either is that line or is a different one, the hint does not privilege
    the expected row over the other sixteen, and a wrong clip still matches the
    wrong row.

    On a LONG line an error can be local, and a hint will paper over exactly
    the case that matters. Measured, and this is why the rule exists rather
    than the convenience: `tire_critical` is fifteen words, of which Gleam
    renders thirteen perfectly and the first two -- "Pull over" -- not at all.
    Cold, it transcribes as "Paying for when it's safe...", "Hang on while it's
    safe...", "Think of a one-inch safe...", never twice the same. WITH the
    hint it transcribes perfectly, because the transcriber has been shown the
    answer and the other thirteen words agree with it.
    
    The instruction to pull over is the entire action that warning exists to
    request. A verifier that a hint can talk round is not verifying it.
    """
    from headway import live_policy
    lines = [live_policy.LINE_TEXT[k] for k in CLIP_LINES]
    lines += list(TIRE_CLIPS.values()) + list(IMMINENT_CLIPS.values())
    return " ".join(sorted(set(lines)))


def _transcribe(path: Path, expected: str = "") -> str:
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
        # The hint, for a clip short enough that it cannot hide an error.
        kw = {}
        if expected and len(expected.split()) <= VOCAB_HINT_MAX_WORDS:
            kw["prompt"] = _library_vocabulary()
        return OpenAI().audio.transcriptions.create(
            model=_config.OPENAI_STT_MODEL, file=buf, **kw).text or ""
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


# CONTRACTIONS ARE THE TRANSCRIBER'S CHOICE TOO, and until this existed the
# verifier was failing clips over it. "You're too close." came back from
# Whisper as "You are too close." -- the voice said the right thing and the
# check said it did not, three times in a row, and the only way to pass was to
# keep re-rendering a clip that had been correct the first time.
#
# Expanded on BOTH sides rather than contracted on both: "we're" is
# unambiguous going out and "were" is not coming back.
_CONTRACTIONS = (
    ("won't", "will not"), ("can't", "cannot"), ("n't", " not"),
    ("'re", " are"), ("'ve", " have"), ("'ll", " will"), ("'m", " am"),
    ("it's", "it is"), ("that's", "that is"), ("what's", "what is"),
    ("she's", "she is"), ("he's", "he is"), ("there's", "there is"),
    ("let's", "let us"),
)


def _norm(text: str) -> str:
    """Spoken-form comparison: punctuation, case and contraction are a
    transcriber's choices, the words are not."""
    t = (text or "").lower().replace("\u2019", "'")
    for a, b in _CONTRACTIONS:
        t = t.replace(a, b)
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", t).split())


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
        heard = _transcribe(tmp, text)
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
                model=config.ELEVENLABS_CONVERSATION_MODEL):
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
    if backend == "gpt_live":
        return {"backend": "gpt_live",
                "voice": config.GPT_LIVE_VOICE,
                "model": config.GPT_LIVE_MODEL}
    if backend == "elevenlabs":
        return {"backend": "elevenlabs",
                "voice": config.ELEVENLABS_VOICE_ID,
                "model": config.ELEVENLABS_CONVERSATION_MODEL}
    return {"backend": "openai_realtime",
            "voice": config.OPENAI_REALTIME_VOICE,
            "model": config.OPENAI_REALTIME_MODEL}


def manifest(backend: str = None) -> dict:
    try:
        return json.loads(manifest_path(backend).read_text())
    except Exception:
        return {}


def _write_manifest(backend: str, rendered: list):
    """Record what was made, and from what. Merged, not replaced.

    A run with no --force re-renders nothing, and a manifest rewritten from
    that run would claim the untouched files came from today's config. Only the
    files this run actually produced get their entry updated.
    """
    sig = voice_signature(backend)
    doc = manifest(backend)
    doc["voice"] = sig
    doc.setdefault("clips", {})
    for line, path, n, what in rendered:
        if what != "rendered":
            continue
        doc["clips"][line] = {
            **sig, "bytes": n, "at": round(time.time(), 1),
            "sha1": hashlib.sha1(Path(path).read_bytes()).hexdigest()[:12],
        }
    p = manifest_path(backend)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")


def render(force: bool = False, backend: str = None) -> list:
    backend = backend or config.VOICE_BACKEND
    if backend == "realtime":
        backend = "openai_realtime"       # the old name for the same thing
    out_dir = audio_dir(backend)
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []
    everything = [(line, live_policy.LINE_TEXT[line]) for line in CLIP_LINES]
    everything += sorted(TIRE_CLIPS.items())
    everything += sorted(IMMINENT_CLIPS.items())
    for line, text in everything:
        path = out_dir / f"{line}.mp3"
        if path.exists() and not force:
            out.append((line, path, path.stat().st_size, "kept"))
            continue
        # Render to a temp path and move into place, so an interrupted run can
        # never leave a truncated clip that the browser would happily preload.
        tmp = path.with_suffix(".mp3.part")
        try:
            if backend == "openai_realtime":
                n = _render_realtime(text, tmp)
            elif backend == "gpt_live":
                n = _render_live(text, tmp)
            else:
                n = _render_elevenlabs(text, tmp)
        except ClipUnverified as e:
            # One clip nobody could prove is one clip. The other sixteen are
            # still worth having, and a run that aborts on the first hard line
            # leaves the library half-rendered in two voices.
            tmp.unlink(missing_ok=True)
            print(f"  [UNVERIFIED] {line}: {e}")
            out.append((line, path, 0, "unverified"))
            continue
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
                    choices=["openai_realtime", "realtime", "gpt_live",
                             "elevenlabs"],
                    help="which voice to render in (default: config.VOICE_BACKEND)")
    args = ap.parse_args()

    if args.list:
        listing = [(k, live_policy.LINE_TEXT[k]) for k in CLIP_LINES]
        listing += sorted(TIRE_CLIPS.items())
        listing += sorted(IMMINENT_CLIPS.items())
        for line, text in listing:
            p = AUDIO_DIR / f"{line}.mp3"
            size = p.stat().st_size if p.exists() else 0
            print(f"  {line:18} {'OK ' if size else '-- '} {size:>7} B  {text!r}")
        return 0

    backend = args.backend or config.VOICE_BACKEND
    if backend == "realtime":
        backend = "openai_realtime"
    if backend in ("openai_realtime", "gpt_live") \
            and not os.getenv("OPENAI_API_KEY"):
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
              f"on {config.ELEVENLABS_CONVERSATION_MODEL}")
    else:
        sig = voice_signature(backend)
        print(f'  voice: {sig["backend"]} ({sig["voice"]})')
    rows = render(force=args.force, backend=backend)
    for line, path, n, what in rows:
        print(f"  [{what:8}] {line:16} {n:>7} B  {path}")
    # A CLIP THAT COULD NOT BE PROVED CORRECT IS A FAILED RUN, and the exit
    # code says so. It is not fatal to the car -- rio_speak falls through to
    # dictation and then to the synthesiser for any line with no clip -- but it
    # is fatal to the claim that this voice's library is complete, and that
    # claim is what preflight and the manifest exist to check.
    unverified = [line for line, _, _, what in rows if what == "unverified"]
    if unverified:
        print(f"\n  !! {len(unverified)} clip(s) could not be verified and "
              f"were NOT written: {', '.join(unverified)}")
        print("     Those lines fall back to dictation and the synthesiser. "
              "Re-run to retry.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
