"""What RIO sounds like, as something a machine can check.

WHY THIS IS A MODULE AND NOT A PARAGRAPH
----------------------------------------
The bible's voice rules have always lived in a prompt, which means they were
enforced by a model agreeing to follow them. That was fine while every spoken
word came out of the model the prompt was given to.

It stopped being fine the moment a sentence written by the OBSERVER — a local
vision model, captioning a frame once a second — became a sentence the driver
hears in RIO's voice, spoken as hers, with no conversational model in between.
A caption is not a character. "The image shows a road with several cars on it"
is a perfectly good caption and is not something RIO would ever say.

So the register becomes a check. `lint()` returns the reasons a line is not in
RIO's voice, and an empty list means it is safe to speak as her. The visual
fast path calls it before deciding a cached line can go straight to the
speaker, and falls back to having her compose one when it fails — which makes
this the thing standing between a driver and a robot reading a photo caption.

ONE LIST, TWO USERS
-------------------
The banned words below are also what rio_prompts renders into the system
prompt. A model told one list and measured against another is a model that
gets blamed for a disagreement between two hand-typed copies; the audio tag
gate is built the same way and for the same reason.
"""
import re

# From the bible (§ "Banned (v1.1)"), verbatim, plus the assistant-isms it
# names elsewhere. These are phrases RIO does not say — most of them because
# they belong to a customer-service voice she is defined against, and the
# callsigns because they were retired.
BANNED_PHRASES = (
    "captain", "agent 507", "buddy", "champ", "boss", "joshua", "sir",
    "roger", "copy that", "be advised",
    "no problem", "happy to help", "i think", "i see", "i notice",
    "as your ai", "let me know if", "is there anything else",
    "i'm here to help", "im here to help", "great question",
    "i can help with that", "absolutely", "certainly",
)

# ...and the ones that only a CAPTION says. A vision model describes a
# photograph; RIO is looking through a windscreen. Everything here is a phrase
# that gives away which of those two is talking.
CAPTION_TELLS = (
    "the image", "this image", "the photo", "the picture", "the scene shows",
    "depicts", "in the foreground", "in the background", "appears to be",
    "there is a", "there are a", "we can see", "you can see", "visible in",
    "a view of", "shot of", "camera", "frame",
)

# A spoken sentence in a car. Long enough for "Open freeway, light traffic —
# dry hills both sides", short enough that nothing paragraph-shaped survives.
MAX_WORDS = 14

_FIRST_PERSON = re.compile(r"\b(i|i'm|im|i've|my|me)\b", re.IGNORECASE)
_SENTENCE_END = re.compile(r"[.!?](?:\s|$)")


def _norm(text: str) -> str:
    return " ".join((text or "").lower().split())


def lint(text: str) -> list:
    """Every reason this line is not in RIO's voice. Empty means it is.

    Deliberately a LIST rather than a boolean: a line that fails wants to be
    logged with the reason, because "the observer stopped sounding like her" is
    a sentence somebody will have to debug, and "it failed the lint" is not an
    answer to it.
    """
    raw = (text or "").strip()
    if not raw:
        return ["empty"]
    low = _norm(raw)
    out = []

    for phrase in BANNED_PHRASES:
        if re.search(rf"\b{re.escape(phrase)}\b", low):
            out.append(f"banned: {phrase!r}")
    for tell in CAPTION_TELLS:
        if tell in low:
            out.append(f"caption: {tell!r}")

    words = raw.split()
    if len(words) > MAX_WORDS:
        out.append(f"too long: {len(words)} words (max {MAX_WORDS})")

    # More than one sentence is a paragraph in a car. A trailing full stop is
    # not a second sentence, which is why the end of the string is trimmed
    # before counting.
    if len(_SENTENCE_END.findall(raw.rstrip(".!?").strip() + " ")) >= 1:
        out.append("more than one sentence")

    if _FIRST_PERSON.search(raw):
        out.append("first person — she is looking, not reporting")

    if raw.endswith("?"):
        out.append("a question, not an observation")

    # A model asked for one sentence sometimes answers with a list.
    if raw.count(",") >= 4 or "\n" in raw:
        out.append("a list, not a sentence")

    return out


def speakable(text: str) -> bool:
    """May this be spoken as RIO, with no model between it and the driver?"""
    return not lint(text)


def banned_words_block() -> str:
    """The banned list, rendered for the system prompt.

    Generated rather than typed so the words the model is told about and the
    words the lint enforces are provably the same ones.
    """
    quoted = ", ".join(f'"{w}"' for w in BANNED_PHRASES)
    return quoted + "."
