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


# A WORD repeated this many times in a row is a decoding loop too, and this is
# the check that was missing. loop_run above splits on sentence boundaries, so it
# scored the worst reading of the 2026-09-20 measurement as CLEAN:
#
#     "ROAD: three lanes in one direction, asphalt, solid yellow line dividing
#      opposite lanes, interstate interstate interstate interstate ..."
#
# 216 words, 14 distinct, "interstate" x203, no full stop anywhere in it -- one
# sentence as far as loop_run was concerned, and 5721 ms of a 1 Hz observer's
# budget. Five is chosen well above anything English does on purpose; "very very
# very" is three.
WORD_LOOP_FLOOR = 5


# What separates one piece of a reading from the next. Whitespace is not enough,
# and that cost a second measurement: a reading in the terse sensor format came
# back as
#
#     ROAD: two|three|curb|curb|curb|curb|curb|curb|curb|
#
# which `str.split()` sees as TWO words -- "ROAD:" and one 48-character word --
# because the loop has no spaces in it. A model in a loop repeats whatever unit it
# is emitting, and the unit is not always space-delimited; on a format whose own
# separator is a bar, the bar is the unit. So the split is on the separators a
# reading can actually contain.
_PIECES = re.compile(r"[\s|,;:/]+")


def word_loop_run(text: str) -> int:
    """Longest run of one repeated piece. -> count (1 when nothing repeats).

    The cheap half of loop detection and the half that catches a model that has
    stopped producing sentences altogether. Deliberately case-sensitive: "the
    the" from a real stutter is two, and a decoder in a loop emits the identical
    unit tens or hundreds of times.
    """
    ws = [p for p in _PIECES.split(text or "") if p]
    best = run = 1 if ws else 0
    for i in range(1, len(ws)):
        run = run + 1 if ws[i] == ws[i - 1] else 1
        if run > best:
            best = run
    return best


# A CATALOGUE IS NOT A READING, and this is the third way a model stops
# looking without repeating itself. From the drive of 2026-09-21, on a road
# with four cars on it:
#
#     TRAFFIC: sedan|truck|car|bus|van|minibus|taxi|ambulance|fire truck|pol
#     TRAFFIC: cars, minivans, sedans, hatchbacks, SUVs, trucks, vans,
#              motorcycles, bicycles, bicycles that matter, vehicles that matter
#     TRAFFIC: cars|minivans|sedans|trucks|jeeps|mazdas|xpo|bmws|hyundais|audi
#     ROAD:    two|four|asphalt|single|straight|no|no|no|no
#
# Nothing repeats, so word_loop_run scores 1. Nothing is memorised label text,
# so markers() is empty. Every one of them is fluent, well-formed, in the right
# format -- and is the model emptying a category out of its vocabulary instead
# of describing the picture. It is the SAME FAILURE as reciting a training
# label, arrived at from the other end: there, one memorised sentence; here,
# the whole list the sentence would have come from.
#
# Six, because a real reading of a road does use short lists and they are
# short: "four, asphalt, moderate" is three, "five, two, single lane, asphalt,
# signalized" is five and is a perfectly good ROAD field. Nine one-word items
# is a vocabulary.
LIST_FLOOR = 6

# ...and the length at which an item stops being a bare category. "sedan" is a
# catalogue entry; "white sedan right lane" is an observation. Counted in
# words, per item.
LIST_ITEM_MAX_WORDS = 2

# Where a thing IS, which is what the prompt asks for and what a catalogue
# never has. A field carrying any of these is describing a scene rather than
# emptying a category, however many items it has.
_PLACE_WORDS = (
    "ahead", "behind", "left", "right", "front", "beside", "next to", "lane",
    "oncoming", "adjacent", "kerb", "curb", "verge", "shoulder", "junction",
    "crossing", "parked", "stopped", "turning", "merging", "queue", "near",
    "far", "centre", "center", "side", "opposite", "across", "up", "back",
)

# One piece of a list. Bars and commas only -- a semicolon or a colon can
# separate a field from its name, and splitting on those would make every
# reading look like a list of three.
_LIST_SPLIT = re.compile(r"\s*[|,]\s*")

# ...matched on WORD BOUNDARIES, which cost a measurement to learn: as a plain
# substring test, "back" is inside "hatchbacks" and the twelve-item vehicle
# catalogue of 2026-09-21 therefore read as a scene that said where things
# were. A place word has to be a word.
_PLACE_RE = re.compile(
    r"\b(?:" + "|".join(w.replace(" ", r"\s+") for w in _PLACE_WORDS) + r")\b",
    re.IGNORECASE)

# A field name at the front of a list item: "TRAFFIC: cars" is the item "cars"
# in the TRAFFIC field, not a different item from the "cars" after it. Without
# this, "TRAFFIC: cars|cars|cars" runs to two and passes.
_ITEM_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z _]{2,20}:\s*")


def list_run(text: str) -> int:
    """Longest run of bare catalogue items in any one field. -> count.

    A FIELD AT A TIME, because the fields are different questions and a long
    ROAD does not excuse a long TRAFFIC. Field names are found the same way
    rio_prompts.split_sensor_reading finds them -- a word in caps followed by a
    colon -- rather than imported, so this file keeps working on a teacher row
    that has no sensor fields at all.
    """
    t = (text or "").strip()
    if not t:
        return 0
    # Split into fields on "NAME:" where NAME is a bare word. Everything before
    # the first one is a field too (a reading with no names at all).
    parts = re.split(r"(?:^|[|\s])([A-Za-z][A-Za-z _]{2,20}):", t)
    chunks = [parts[0]] + parts[2::2] if len(parts) > 1 else [t]
    best = 0
    for chunk in chunks:
        chunk = (chunk or "").strip()
        if not chunk:
            continue
        if _PLACE_RE.search(chunk):
            continue                 # it says where things are: a scene
        items = [i.strip() for i in _LIST_SPLIT.split(chunk) if i.strip()]
        run = 0
        for item in items:
            if len(item.split()) <= LIST_ITEM_MAX_WORDS:
                run += 1
                best = max(best, run)
            else:
                run = 0
    return best


# A DELIMITED ITEM REPEATED, which is a loop that word_loop_run is deliberately
# too generous to catch. Its floor is five because "very very very" is three
# and prose does that on purpose. A BAR-SEPARATED FIELD DOES NOT: nothing
# legitimate writes "cars|cars|cars", and that reading reached the card on
# 2026-09-21 under a three-run that both loop checks passed.
ITEM_LOOP_FLOOR = 3


def item_loop_run(text: str) -> int:
    """Longest run of one repeated list ITEM. -> count (0 when there is no list)."""
    items = [_ITEM_NAME_RE.sub("", i.strip()).strip().lower()
             for i in _LIST_SPLIT.split(text or "") if i.strip()]
    items = [i for i in items if i]
    if len(items) < 2:
        return 0
    best = run = 1
    for i in range(1, len(items)):
        run = run + 1 if items[i] == items[i - 1] else 1
        best = max(best, run)
    return best if best > 1 else 0


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
    # ...and the same fault one level down, where a model has stopped emitting
    # sentences at all. Reported under its own key so a corpus row says WHICH
    # kind of loop it was.
    wloop = word_loop_run(text)
    if wloop >= WORD_LOOP_FLOOR:
        out["word_loop"] = wloop
    # ...and the same fault a third way: an item repeated inside a delimited
    # list, where three is already impossible on purpose.
    iloop = item_loop_run(text)
    if iloop >= ITEM_LOOP_FLOOR:
        out["item_loop"] = iloop
    # ...and the model emptying a category instead of describing the picture.
    lrun = list_run(text)
    if lrun >= LIST_FLOOR:
        out["catalogue"] = lrun
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
    out["strength"] = ("strong" if (m or out.get("loop") or out.get("word_loop")
                                    or out.get("item_loop") or out.get("catalogue"))
                       else "weak")
    if out.get("catalogue") and not m:
        out["why"] = (f"{out['catalogue']} bare categories in a row and no "
                      f"sign of where any of them is — a vocabulary, not a "
                      f"reading of this frame")
        return out
    if out.get("item_loop") and not m and not out.get("loop"):
        out["why"] = (f"one list item repeats {out['item_loop']} times — a "
                      f"decoding loop, not a reading")
        return out
    if out.get("word_loop") and not m and not out.get("loop"):
        out["why"] = (f"one word repeats {out['word_loop']} times in a row — a "
                      f"decoding loop, not a reading")
        return out
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
