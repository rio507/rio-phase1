"""Is this a reading, or a recitation? — the same question vision.py asks Qwen.

WHY THIS FILE EXISTS
--------------------
Asked "Describe the scene.", Alpamayo 1.5 does not describe the scene. It
returns a string out of the scenario-label taxonomy it was trained on:

    "A vehicle controls loss. The vehicle driver is in distracted driving."
    "Lead vehicle stops. The vehicle driver is in distracted driving."
    "A vehicle changes lanes with the same direction to ego-car/another
     vehicle. Vehicles do not notice the coming vehicles when turning or
     changling lanes"

Eight of ten keyframes on the acceptance clip returned the first of those
verbatim, on scenes that had no lane change, no stopped lead and no distracted
driver in them. Reworded, the same model on the same frames describes the road
properly -- so the prompt set was changed (config.TEACHER_PROMPTS) and this
file is what stops the problem coming back silently the next time somebody
edits a question.

It is the same guard this project already runs on its own observer, from the
other side. There, the prompt ended with four example sentences and Qwen
returned the first one whatever the frame was; rio_prompts.is_prompt_example
refuses those and vision.parroted() counts them. The memory being recited is
different -- a prompt's examples there, a training set's labels here -- and the
symptom is identical: a fluent, plausible, well-formed sentence about nothing
in the picture.

WHAT IT CAN AND CANNOT SEE
--------------------------
It does NOT judge whether an answer is true. Nothing here can, and a checker
that guessed would be worse than none.

It reports two things it CAN see, and both are structural:

  MARKERS. Non-breaking spaces, thin spaces, and other exotic whitespace do not
  come out of a language model generating prose -- they come out of text that
  was pasted from a spreadsheet or a web page into a label file and memorised
  with the whitespace still in it. On the readings above, the non-breaking
  spaces are the tell that made the whole thing obvious, and they are as close
  to a fingerprint as this gets.

  REPETITION. A model that answers ten different keyframes with the same
  sentence, byte for byte, is not looking at them. That one is model-agnostic
  and needs no list of known phrases, which is exactly why it is here: the next
  memorised taxonomy will not be one anybody has written down.

A flagged reading is still SHOWN and still RECORDED -- flagging is not
filtering. What it must not do is sit on a dashboard looking like perception.
"""
import re
import unicodedata

# Whitespace that prose does not contain. U+00A0 non-breaking space is the one
# that actually appeared; the rest are its neighbours and cost nothing to add.
_EXOTIC_WS = "      　  "
_EXOTIC_RE = re.compile("[" + _EXOTIC_WS + "]")

# How many keyframes in a row may carry the identical answer before it is worth
# saying so. Two is a coincidence on a motorway where nothing changes -- three
# is the model not looking.
REPEAT_FLOOR = 3


def markers(text: str) -> list:
    """Structural tells that this text was memorised rather than generated."""
    t = text or ""
    found = []
    exotic = sorted({c for c in t if c in _EXOTIC_WS})
    if exotic:
        found.append("exotic_whitespace:" + ",".join(
            "U+%04X" % ord(c) for c in exotic))
    # A control character that is not a newline or tab is not something a
    # tokenizer emits in an answer either.
    if any(unicodedata.category(c) == "Cc" and c not in "\n\t\r" for c in t):
        found.append("control_characters")
    return found


# A sentence repeated this many times inside ONE answer is a decoding loop.
# Three is chosen high enough that a model legitimately restating a point twice
# is not caught.
LOOP_FLOOR = 3


def loop_run(text: str) -> int:
    """Longest run of an identical sentence inside one answer. -> count.

    A DIFFERENT FAULT FROM RECITATION, and worth its own check. Cosmos, asked
    for physical reasoning, produced:

        "The sedan is also further away from the ego vehicle. The sedan is also
         further away from the ego vehicle. The sedan is also further away from
         the ego vehicle. ..."

    -- five times, filling the token budget. Nothing about that is memorised;
    it is a decoding loop, which the model's shipped generation_config invites
    by setting repetition_penalty to 1.0. The sampling parameters are left as
    NVIDIA ships them (changing them is a tuning decision, and a quiet one
    would make every reading incomparable with the model card's); what is not
    left alone is a corpus row that looks like four hundred words of analysis
    and is one sentence.
    """
    parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", text or "")]
    parts = [p for p in parts if len(p) > 12]
    best = run = 0
    prev = None
    for p in parts:
        run = run + 1 if p == prev else 1
        prev = p
        best = max(best, run)
    return best


def looks_canned(text: str) -> bool:
    return bool(markers(text))


def describe(text: str, repeats: int = 0) -> dict:
    """-> the `flags` block that rides on a reading. Empty dict when clean."""
    out = {}
    m = markers(text)
    if m:
        out["markers"] = m
    if repeats >= REPEAT_FLOOR:
        out["repeated"] = int(repeats)
    loop = loop_run(text)
    if loop >= LOOP_FLOOR:
        out["loop"] = loop
    if not out:
        return out
    # TWO STRENGTHS, AND THEY ARE NOT THE SAME CLAIM.
    #
    # Markers are near-proof: a non-breaking space between two words did not
    # come out of a decoder, it came out of a spreadsheet.
    #
    # Repetition on its own is a HINT and must be labelled as one. On a
    # straight, empty motorway "Keep lane since the lane is clear ahead" three
    # keyframes running is the model being right three times, not the model
    # not looking -- and Alpamayo's Chain-of-Causation trace is drawn from a
    # small vocabulary of exactly that kind. Calling that "recited" would train
    # whoever reads this card to ignore the flag, which is the only way a flag
    # can fail.
    out["strength"] = "strong" if (m or out.get("loop")) else "weak"
    if out.get("loop") and not m:
        out["why"] = (f"one sentence repeats {out['loop']} times in a row "
                      f"inside this answer — a decoding loop, not analysis")
        return out
    if m:
        out["why"] = ("this answer carries whitespace that generated prose "
                      "does not — it looks like memorised label text")
        if repeats >= REPEAT_FLOOR:
            out["why"] += (f", and it is byte-identical to the previous "
                           f"{repeats} keyframes' answer")
    else:
        out["why"] = (f"identical to the previous {repeats} keyframes' answer. "
                      f"That is worth noticing, not worth believing: on an "
                      f"unchanging road a repeated answer may simply be right "
                      f"again")
    return out


class RepeatTracker:
    """How many keyframes in a row a model has given the identical answer.

    Per (model, field), because a model can be reciting for one question and
    perceiving for another -- which is exactly what Alpamayo does: its
    critical-actor answers were specific and correct on the same keyframes
    where its scene answers were a fixed string.
    """

    def __init__(self):
        self._last = {}     # (model, field) -> [text, count]

    def note(self, model: str, field: str, text: str) -> int:
        key = (model, field)
        t = (text or "").strip()
        if not t:
            self._last.pop(key, None)
            return 0
        prev = self._last.get(key)
        if prev and prev[0] == t:
            prev[1] += 1
        else:
            self._last[key] = [t, 1]
        return self._last[key][1]

    def reset(self):
        self._last.clear()
