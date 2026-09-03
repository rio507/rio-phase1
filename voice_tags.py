"""Audio tags — what RIO is allowed to do with her voice, and where.

WHAT A TAG IS
-------------
Eleven v3 reads square-bracket directions in the text it is given. `[laughs]`,
`[sighs]`, `[whispers]` are not spoken; they change how the words around them
come out. Used once, in the right place, this is the difference between a voice
and a reading — RIO agreeing with a short laugh instead of the word "ha".

WHY THIS FILE IS A GATE AND NOT A FEATURE
-----------------------------------------
A model given a mechanism for performance will perform. The bible is explicit
about the register — "NOT loud, polished, corporate, robotic, or performative",
and silence as the default tone — and a tag is the single easiest way to break
that without changing a word of what she says.

So tags survive exactly one path: RIO's own conversation, one per utterance,
from a list short enough to read. Everywhere else — a turn instruction, a
health announcement, a headway warning, anything a policy wrote — they are
removed before synthesis, and there is no configuration that turns that off.

The failure this prevents is concrete. A model that emits "[sighs] Take the
next left." into a deterministic channel gets one of two outcomes depending on
the synthesiser: a navigation instruction that sighs, or a synthesiser that
does not know the tag and reads the word "sighs" out loud at a junction. Both
are the same bug, and the second one is the reason a validator that DROPS an
unknown tag is not the same thing as a synthesiser that ignores it.

WHAT "MISPLACED" MEANS
----------------------
A tag is a direction that attaches to the words next to it, so it has to sit
between them: at the start, at the end, or on its own between two words. A tag
welded into the middle of a word — `Lin[laughs]coln` — is not a direction, it
is a token that got out of the model's head sideways, and it is dropped rather
than passed on to find out what v3 makes of it.

A tag on its own with no words is dropped too. A laugh with nothing to laugh
about is the definition of performative.
"""
import re

import config

# What a tag looks like on the wire. Deliberately permissive about CONTENT —
# this pattern is how a candidate is FOUND, and the allow-list below is what
# decides whether it survives. A validator that only recognised known tags
# could not drop unknown ones; it would leave them in the text to be spoken.
_TAG = re.compile(r"\[([^\[\]\n]{1,40})\]")

# The two modes, named rather than passed as a boolean, because the call sites
# read better and because "False" is the wrong shape for "this is a warning".
CONVERSATION = "conversation"
DETERMINISTIC = "deterministic"

# Reasons a tag was dropped. One per cause: "tags were stripped" is not a
# diagnosis, and the interesting one — a model reaching for a tag on a
# deterministic channel — is invisible if every drop is counted the same.
NOT_ALLOWED = "not_allowed"          # not on the list
WRONG_CHANNEL = "wrong_channel"      # a policy line, where no tag belongs
MISPLACED = "misplaced"              # welded into a word
TOO_MANY = "too_many"                # past the per-utterance budget
NO_WORDS = "no_words"                # the tag was the whole utterance


def allowed_tags() -> tuple:
    """The list, normalised, from config. Lowercase and without brackets."""
    return tuple(str(t).strip().lower().strip("[]")
                 for t in (config.AUDIO_TAGS_ALLOWED or ()))


def _placed_well(text: str, start: int, end: int) -> bool:
    """Is this tag between words rather than inside one?

    Whitespace or an edge on the left; whitespace, an edge or punctuation on
    the right. That is the whole rule, and it is the rule the docs describe:
    a tag is placed immediately before or after the speech it modifies.
    """
    before = text[start - 1] if start > 0 else " "
    after = text[end] if end < len(text) else " "
    return (before.isspace() or before in "([{\"'—–-")\
        and (after.isspace() or after in ".,!?;:)]}\"'—–")


def sanitize(text: str, mode: str = DETERMINISTIC) -> tuple:
    """(text_to_synthesise, dropped) — every tag either kept or accounted for.

    `dropped` is a list of {tag, reason}. Nothing is silently discarded: a tag
    the model tried to use and did not get is exactly the kind of thing that is
    worth counting over a drive, and the alternative — a regex that quietly
    deletes — makes "she never laughs any more" unanswerable.

    Never raises and never returns None. This sits between a model and a
    speaker; a validator that can fail is a validator that can make RIO mute.
    """
    src = text or ""
    dropped = []

    tags_on = bool(getattr(config, "AUDIO_TAGS_ENABLED", False))
    budget = int(getattr(config, "AUDIO_TAGS_MAX_PER_UTTERANCE", 1))
    allow = allowed_tags()
    kept = 0

    # Are there any words at all once the tags come out? Decided up front,
    # because "[laughs]" alone must lose its tag rather than become the one
    # utterance where the budget is spent on nothing.
    words_only = _TAG.sub(" ", src).strip()

    out = []
    pos = 0
    for m in _TAG.finditer(src):
        out.append(src[pos:m.start()])
        pos = m.end()
        name = m.group(1).strip().lower()
        raw = m.group(0)

        if mode != CONVERSATION or not tags_on:
            dropped.append({"tag": raw, "reason": WRONG_CHANNEL})
            continue
        if name not in allow:
            dropped.append({"tag": raw, "reason": NOT_ALLOWED})
            continue
        if not _placed_well(src, m.start(), m.end()):
            dropped.append({"tag": raw, "reason": MISPLACED})
            continue
        if not words_only:
            dropped.append({"tag": raw, "reason": NO_WORDS})
            continue
        if kept >= budget:
            dropped.append({"tag": raw, "reason": TOO_MANY})
            continue

        kept += 1
        out.append(f"[{name}]")
    out.append(src[pos:])

    clean = "".join(out)
    # A removed tag leaves a hole. " ,"  and "  " are things a synthesiser
    # reads as timing, so the seam is closed rather than left for v3 to
    # interpret: this is the difference between a dropped tag being invisible
    # and a dropped tag being an audible stumble.
    clean = re.sub(r"[ \t]{2,}", " ", clean)
    clean = re.sub(r"\s+([.,!?;:])", r"\1", clean)
    clean = re.sub(r"\(\s+", "(", clean)
    # A tag at either end leaves whitespace the source did not have. That gap
    # is closed, and ONLY that gap: a phrase arriving mid-stream keeps the
    # space that joins it to the last one, or RIO runs her words together.
    if clean[:1].isspace() and not src[:1].isspace():
        clean = clean.lstrip()
    if clean[-1:].isspace() and not src[-1:].isspace():
        clean = clean.rstrip()
    return clean, dropped


def strip(text: str) -> str:
    """Every tag out, no questions. The deterministic path's one-liner."""
    return sanitize(text, DETERMINISTIC)[0]


def has_tag(text: str) -> bool:
    return bool(_TAG.search(text or ""))


def instruction() -> str:
    """What the live session is told about tags, in words.

    Generated from the same config the validator enforces, so the model is
    never told it may use a tag the gate will drop — the two halves of a rule
    that disagree is how you get a model that keeps trying.

    Returns "" when tags are off, which is not the same as saying "do not use
    tags": with the feature off there is no reason to spend instruction budget
    teaching a model about a mechanism it cannot reach.
    """
    if not config.AUDIO_TAGS_ENABLED or not allowed_tags():
        return ""
    listed = ", ".join(f"[{t}]" for t in allowed_tags())
    return (
        "HOW YOU SOUND. You may place at most ONE of these in a reply, and "
        f"only in ordinary conversation: {listed}. Put it on its own between "
        "words, never inside one.\n"
        "Use one when a person would actually do it and not otherwise. A reply "
        "is not improved by a laugh, and a driver hearing you perform an "
        "emotion you were not having is worse than a flat reading. Most "
        "replies should carry no tag at all.\n"
        "Never use one in anything about the car, the road or the route, and "
        "never use any bracketed word that is not on that list — an unknown "
        "one is dropped, not spoken.\n"
    )
