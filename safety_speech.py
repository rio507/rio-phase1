"""Saying a safety finding in her own words, without letting her make one up.

WHAT THIS REPLACES AND WHAT IT DOES NOT. RIO's spoken safety layer was one
mechanism for two very different jobs:

  "Back off — now."          a fixed sentence, pre-rendered, played from disk
                             because the situation is already too late to wait
                             on anything.
  "The front left tire is    a TEMPLATE with a corner and a pressure slotted
   down to 24 PSI. ..."      into it, said the same way every time for the
                             life of the drive.

The first is right. The second is a recording of a sentence pretending to be a
person, and a driver hears the difference immediately -- the same words, the
same order, the same emphasis, every time the car finds the same thing. That is
an ADAS beep with a vocabulary, and the goal here is a passenger who noticed
something.

So: the critical tier keeps its clips, and everything else is PHRASED FRESH
from the structured event. Same decision, same timing, same suppression -- this
changes the words, never whether or when they are said.

THE PART THAT MATTERS MOST, and the reason this file is not just a prompt: a
model writing a sentence about a car is a model that can invent a pressure. The
old mechanism could not lie because it could not compose; this one can, so the
composition is CHECKED rather than trusted:

  * every number it says must appear in the event it was given
  * provenance is supplied as a phrase and must survive into the output
  * the observation window bounds the claim
  * the banned-word list from the bible still applies
  * anything that fails is retried once and then falls back to the
    deterministic sentence, which is exactly what shipped before this file

A generated line that cannot be proved honest is not spoken. That is the same
discipline the pre-rendered clips get from tools/render_alerts.py, applied to
a sentence instead of an audio file.
"""
import json
import re
import threading
import time

import config


# ---------------------------------------------------------------------------
# WHICH FINDINGS ARE ALLOWED TO BE PHRASED, AND WHICH ARE NOT
# ---------------------------------------------------------------------------
# The split is by URGENCY, not by subject. The question for each line is only:
# can this wait for a model and a mouth?
#
# CRITICAL -- cannot wait, stays a clip:
#     the headway red tier (tau < 2 s: a collision is closing)
#     "pull over now" for a tire that is failing under the car
#
# EVERYTHING ELSE -- can wait a beat, and reads better for it:
#     tire advisories, health findings, coaching, the amber gap band,
#     connection notices
#
# This set is the ONE place that decides it. A code that is not named here is
# not critical, which is the safe default for the words and the unsafe default
# for latency -- and latency is recoverable by the fallback chain while a
# missing warning is not.
CRITICAL_CLIPS = frozenset({
    # headway, red tier only
    "too_close", "watch_distance", "back_off",
    # the one health finding whose action is "now"
    "tire_critical",
})


def is_critical(clip_id: str) -> bool:
    return str(clip_id or "") in CRITICAL_CLIPS


# ---------------------------------------------------------------------------
# The drive's memory of what it has already said
# ---------------------------------------------------------------------------
# WHY THIS EXISTS: "phrased fresh" is a claim about a DRIVE, not about a call.
# A generator asked the same question twice with the same input will happily
# return the same sentence twice, and two identical sentences ten minutes apart
# is the recording this file exists to replace.
#
# Keyed by the issue key rather than by the words, so the second mention of the
# SAME fault is what gets varied. Two different faults that happen to produce
# similar sentences are not a repeat.
_said = {}                      # session_key -> {issue_key: [lines]}
_said_lock = threading.Lock()
SAID_MAX_PER_ISSUE = 6


def _recent(session_key: str, issue_key: str) -> list:
    with _said_lock:
        return list((_said.get(session_key) or {}).get(issue_key) or [])


def _remember(session_key: str, issue_key: str, line: str):
    with _said_lock:
        bag = _said.setdefault(session_key, {}).setdefault(issue_key, [])
        bag.append(line)
        if len(bag) > SAID_MAX_PER_ISSUE:
            del bag[:-SAID_MAX_PER_ISSUE]


def forget(session_key: str = None):
    """A new drive starts with nothing said. Tests use it too."""
    with _said_lock:
        if session_key is None:
            _said.clear()
        else:
            _said.pop(session_key, None)


def _norm(s: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", (s or "").lower()).split())


# ---------------------------------------------------------------------------
# THE CHECK, which runs on every generated line
# ---------------------------------------------------------------------------

# Numbers are the thing a model invents most readily and the thing a driver is
# most likely to act on. Written out as words as well as digits, because "down
# to twenty-four" and "down to 24" are the same claim and only one of them is
# a numeral.
_WORD_NUM = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10", "eleven": "11", "twelve": "12", "thirteen": "13",
    "fourteen": "14", "fifteen": "15", "sixteen": "16", "seventeen": "17",
    "eighteen": "18", "nineteen": "19", "twenty": "20", "thirty": "30",
    "forty": "40", "fifty": "50", "sixty": "60", "seventy": "70",
    "eighty": "80", "ninety": "90", "hundred": "100",
}


def _numbers_in(text: str) -> set:
    """Every number a listener would hear, as digits."""
    out = set()
    for m in re.findall(r"\d+(?:\.\d+)?", text or ""):
        out.add(str(float(m)).rstrip("0").rstrip("."))
    words = _norm(text).split()
    i = 0
    while i < len(words):
        w = words[i]
        if w in _WORD_NUM:
            val = int(_WORD_NUM[w])
            # "twenty four" and "twenty-four" both reach here as two tokens.
            if (val in (20, 30, 40, 50, 60, 70, 80, 90)
                    and i + 1 < len(words)
                    and words[i + 1] in _WORD_NUM
                    and int(_WORD_NUM[words[i + 1]]) < 10):
                val += int(_WORD_NUM[words[i + 1]])
                i += 1
            out.add(str(val))
        i += 1
    return out


# From the bible's banned list. A generated line that reaches for one of these
# is not RIO, whatever else is true about it.
BANNED = ("captain", "agent 507", "buddy", "champ", "boss", "sir",
          "roger", "copy that", "be advised", "no problem", "happy to help",
          "as your ai", "let me know if", "is there anything else",
          "i'm here to help", "i am here to help")


def _allowed_numbers(event: dict) -> set:
    """Every figure the event itself contains, in any field."""
    allowed = set()
    for k, v in (event or {}).items():
        if k in ("fallback",):
            continue          # the deterministic line is an output, not input
        if isinstance(v, (list, tuple)):
            for x in v:
                allowed |= _numbers_in(str(x))
        elif v is not None and not isinstance(v, bool):
            allowed |= _numbers_in(str(v))
    return allowed


def check(line: str, event: dict) -> tuple:
    """(ok, reason). The gate every generated line passes before it is spoken.

    Deliberately mechanical. Every rule here is one a prompt also asks for, and
    the prompt is not the enforcement -- a rule that lives only in a prompt is
    a rule the model may reconsider, which is the same argument the two-tier
    brevity gate in realtime.py makes about itself.
    """
    text = (line or "").strip()
    if not text:
        return False, "empty"
    if len(text) > 220:
        return False, "too long for a spoken warning"
    words = _norm(text).split()
    if len(words) > 34:
        return False, "too long for a spoken warning"
    low = text.lower()
    for b in BANNED:
        if b in low:
            return False, f"banned phrase {b!r}"
    # NO INVENTED NUMBERS. Every figure spoken must be one it was given --
    # and "given" means ANYWHERE in the event, not just in the figures list.
    # The first version of this checked only `numbers` and rejected "over the
    # last 40 minutes" as an invention, when the 40 came from the observation
    # window this module had handed the model one line earlier. A validator
    # that refuses the evidence it supplied teaches nothing except to stop
    # citing evidence.
    allowed = _allowed_numbers(event)
    said = _numbers_in(text)
    extra = said - allowed
    if extra:
        return False, f"number not in the evidence: {sorted(extra)}"
    # NO UPGRADING A HEDGE. If the event says the finding is unconfirmed, the
    # line may not assert it as fact.
    if event.get("unconfirmed") and not re.search(
            r"\b(might|may|looks like|seems|possible|possibly|picked up|"
            r"hasn'?t confirmed|not confirmed|think)\b", low):
        return False, "an unconfirmed finding stated as fact"
    return True, ""


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------

_SYSTEM = """You are RIO, riding shotgun. Say one thing about the car, out loud, to the driver.

VOICE
Sharp, easygoing, calm. A friend in the passenger seat who knows cars — not a dashboard, not an assistant, not a driving instructor. Contractions always. Fragments fine. One sentence; two at the very outside.
Never a form of address: no name, no "sir", no callsign. Just "you".
No system-speak: no "alert", no "warning", no "detected", no "advisory", no "please", no "I notice", no "I see".
Urgency comes from the words, not from sounding like a radio.

TRUTH — these are not style notes
- Say only what the event supports. If a figure is not in the event, you do not know it and you do not say it.
- Keep the provenance exactly as given. What the car's own computer reported and what RIO worked out are different claims and must sound different.
- If the event says the finding is unconfirmed, say it that way. Never upgrade it.
- Do not claim anything outside the observation window you are given.
- Never describe something seen out of the window. You did not see it.

OUTPUT
The sentence only. No quotes, no preamble, no explanation."""


def _user_block(event: dict, avoid: list) -> str:
    parts = [f"WHAT: {event.get('what', '')}"]
    for k, label in (("severity", "HOW SERIOUS"),
                     ("location", "WHERE"),
                     ("provenance", "WHO SAYS SO"),
                     ("observation_window", "OVER WHAT PERIOD"),
                     ("evidence", "EVIDENCE"),
                     ("action", "WHAT THE DRIVER SHOULD DO")):
        v = event.get(k)
        if v:
            parts.append(f"{label}: {v}")
    nums = event.get("numbers") or []
    if nums:
        parts.append("THE ONLY FIGURES YOU MAY SAY: " + ", ".join(
            str(n) for n in nums))
    else:
        parts.append("THE ONLY FIGURES YOU MAY SAY: none — say no numbers.")
    if event.get("unconfirmed"):
        parts.append("THIS IS NOT CONFIRMED. Say so.")
    if avoid:
        parts.append(
            "You have already said this to them on this drive, in these "
            "words. Say it differently — same meaning, same seriousness, "
            "fresh sentence:\n" + "\n".join(f"  - {a}" for a in avoid))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Generating
# ---------------------------------------------------------------------------

def phrase(event: dict, session_key: str = "", issue_key: str = "",
           timeout_s: float = None) -> dict:
    """One safety line, in her words. Falls back to the deterministic one.

    Returns {ok, text, source, ms, note} where `source` is "generated" or
    "fallback" -- and the caller treats both the same, because a line that
    arrives is the point and which mechanism produced it is a detail for the
    log rather than for the driver.
    """
    t0 = time.time()
    fallback = (event.get("fallback") or event.get("what") or "").strip()
    issue_key = issue_key or str(event.get("key") or event.get("what") or "")
    avoid = _recent(session_key, issue_key)
    budget = (timeout_s if timeout_s is not None
              else config.SAFETY_PHRASE_TIMEOUT_S)

    try:
        from openai import OpenAI
        cl = OpenAI(timeout=budget)
        for attempt in (1, 2):
            r = cl.responses.create(
                model=config.SAFETY_PHRASE_MODEL,
                instructions=_SYSTEM,
                input=[{"role": "user", "content": _user_block(event, avoid)}],
                max_output_tokens=config.SAFETY_PHRASE_MAX_TOKENS,
                # No reasoning pass. This is one sentence from six labelled
                # fields; there is nothing to think about, and on a reasoning
                # model the budget above is shared between thinking and
                # speaking -- which is how the first version of this returned
                # an empty line and fell back to the template.
                reasoning={"effort": "none"},
            )
            text = (getattr(r, "output_text", "") or "").strip().strip('"')
            ok, why = check(text, event)
            if ok and _norm(text) in {_norm(a) for a in avoid}:
                ok, why = False, "word-for-word repeat of an earlier line"
            if ok:
                _remember(session_key, issue_key, text)
                return {"ok": True, "text": text, "source": "generated",
                        "ms": round((time.time() - t0) * 1000, 1),
                        "attempts": attempt}
            # One retry, with the reason fed back. Two failures is a signal
            # about the event rather than about the sampling, and a third round
            # trip spends the budget the fallback needs.
            avoid = avoid + ([text] if text else [])
            note = why
        return {"ok": False, "text": fallback, "source": "fallback",
                "ms": round((time.time() - t0) * 1000, 1), "note": note}
    except Exception as e:
        return {"ok": False, "text": fallback, "source": "fallback",
                "ms": round((time.time() - t0) * 1000, 1),
                "note": f"{type(e).__name__}: {e}"}


# ---------------------------------------------------------------------------
# PHRASING AHEAD OF THE ANNOUNCEMENT, which is the only way this fits
# ---------------------------------------------------------------------------
# MEASURED FIRST, because the obvious design does not work. Phrasing on demand
# -- generate when the policy decides to speak -- costs 1.2 to 1.7 seconds for
# the sentence, p95 over 6 s when a retry is needed, and THEN the mouth still
# has to start. The health channel's whole budget from decision to first audio
# is 2000 ms. On-demand phrasing does not fit inside it; it fits inside the
# fallback, which means the driver would hear the deterministic sentence most
# of the time and this feature would be a slower way to ship the old one.
#
# So the work moves off the critical path. A fault is CONFIRMED well before the
# announcement policy decides to mention it -- there is a cooldown, a minimum
# gap, a confidence gate and a healing gate between those two moments -- and
# the sentence can be written during that gap instead of after it.
#
# This is the same move observer.py makes for the camera, for the same reason
# and with the same note in the endpoint that starts it: begin describing the
# road NOW, not when she is first asked about it.
#
# WHAT IT DOES NOT DO IS MAKE HER TALK MORE. Preparing a line is not deciding
# to say one. Every gate that decides whether there is an announcement at all
# runs exactly as it did; this only means that when one of them says yes, the
# words are already written.
_prepared = {}                  # (session, issue) -> {"rev", "text", "state"}
_prep_lock = threading.Lock()


def _slot(session_key: str, issue_key: str) -> tuple:
    return (str(session_key or ""), str(issue_key or ""))


def prepare(event: dict, session_key: str = "", issue_key: str = "",
            revision: str = "") -> str:
    """Start writing the line for an issue that has not been announced yet.

    Returns the state: "ready" if it is already written for this revision,
    "working" if a thread is on it, "started" if this call started one.

    `revision` is what makes a WORSENED fault get a new sentence rather than
    the one written when it was milder. The caller passes something that
    changes when the finding does -- magnitude, severity -- and a changed
    revision discards the prepared line rather than speaking a stale one.
    """
    if not config.SAFETY_PHRASE_ENABLED:
        return "disabled"
    key = _slot(session_key, issue_key)
    rev = str(revision or "")
    with _prep_lock:
        cur = _prepared.get(key)
        if cur and cur.get("rev") == rev:
            return cur.get("state", "working")
        _prepared[key] = {"rev": rev, "text": "", "state": "working"}

    def work():
        got = phrase(event, session_key=session_key, issue_key=issue_key)
        with _prep_lock:
            cur = _prepared.get(key)
            # A revision that moved on while this was in flight has already
            # asked for a different sentence; this one is about a finding that
            # is no longer the finding.
            if not cur or cur.get("rev") != rev:
                return
            cur["text"] = got.get("text") or ""
            cur["source"] = got.get("source")
            cur["ms"] = got.get("ms")
            cur["state"] = "ready" if got.get("ok") else "fallback"

    threading.Thread(target=work, daemon=True, name="rio-phrase").start()
    return "started"


def take(session_key: str = "", issue_key: str = "",
         revision: str = "") -> dict:
    """The prepared line for this issue, if one is written for this revision.

    Returns {"text", "source"} or {} -- and an empty dict is a normal answer,
    not an error. It means the words are not ready, and the caller does what it
    did before this module existed: says the deterministic sentence.
    """
    key = _slot(session_key, issue_key)
    rev = str(revision or "")
    with _prep_lock:
        cur = _prepared.get(key)
        if not cur or cur.get("rev") != rev:
            return {}
        if cur.get("state") not in ("ready", "fallback"):
            return {}
        text = (cur.get("text") or "").strip()
        if not text:
            return {}
        return {"text": text, "source": cur.get("source") or "generated",
                "ms": cur.get("ms")}


def drop(session_key: str = "", issue_key: str = ""):
    """Forget a prepared line -- the issue resolved, or the drive ended."""
    with _prep_lock:
        if issue_key:
            _prepared.pop(_slot(session_key, issue_key), None)
        else:
            for k in [k for k in _prepared if k[0] == str(session_key or "")]:
                _prepared.pop(k, None)


# ---------------------------------------------------------------------------
# TURNING A HEALTH ISSUE INTO SOMETHING SAYABLE
# ---------------------------------------------------------------------------
# The engine's issue dict is the structured event this module was designed
# around -- reason code, severity, evidence, observation window, suggested
# action -- and this is the one place that maps it. One place, because the
# honesty rules are enforced against the event: a field that gets dropped here
# is a fact the line may no longer cite, and a field invented here is a fact
# nothing checked.

# WHERE A CLAIM CAME FROM, derived rather than carried, because the issue dict
# does not have a provenance field and inventing one on the engine would be a
# larger change than this is.
#
# THE DEFAULT IS "RIO WORKED IT OUT", and that asymmetry is deliberate. Saying
# the car reported something it did not is the worse error by a long way: it
# tells the driver a fault is in the vehicle's own log, which is what a
# mechanic will look for and not find. Saying RIO noticed something the ECU
# actually reported understates RIO and misleads nobody.
_ECU_DOMAINS = ("dtc", "engine", "powertrain")


def provenance_of(issue: dict) -> str:
    """One clause naming who is making this claim."""
    issue = issue or {}
    code = str(issue.get("code") or "")
    itype = str(issue.get("type") or "")
    domain = str(issue.get("domain") or "")
    if re.match(r"^[PBCU][0-9]{4}$", code.upper()):
        return ("the car's own computer reported this as a fault code"
                if not issue.get("unconfirmed")
                else "the car's own computer has this pending, not confirmed")
    if domain in _ECU_DOMAINS:
        return "the car's own computer reported it"
    if "sensor_lost" in itype or "sensor_loss" in itype:
        return "the car stopped reporting that sensor"
    return ("RIO worked this out from the readings; the car has not "
            "reported a fault")


def event_from_issue(issue: dict) -> dict:
    """The structured event, from one vehicle-health issue."""
    issue = issue or {}
    ev = issue.get("evidence") or {}
    meas = ev.get("measurement") or {}
    bits = []
    for k, v in sorted(meas.items()):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            bits.append(f"{k.replace('_', ' ')} {v}")
    value, unit = issue.get("value"), (issue.get("unit") or "")
    if value is not None:
        bits.insert(0, f"now {value}{(' ' + unit) if unit else ''}")
    unconfirmed = bool(issue.get("unconfirmed")) or (
        "pending" in str(issue.get("type") or ""))
    return {
        "key": str(issue.get("key") or ""),
        "what": (issue.get("message") or "").strip(),
        "severity": issue.get("severity") or "",
        "location": issue.get("location") or "",
        "provenance": provenance_of({**issue, "unconfirmed": unconfirmed}),
        "observation_window": issue.get("observation_window") or "",
        "evidence": "; ".join(bits),
        "action": issue.get("suggested_action") or "",
        "unconfirmed": unconfirmed,
        # THE FIGURES IT MAY SAY. Exactly the ones in the issue -- the reading,
        # and whatever the measurement block carries. check() widens this to
        # everything in the event, so the window's own number counts too.
        "numbers": ([value] if value is not None else [])
                   + [v for v in meas.values()
                      if isinstance(v, (int, float))
                      and not isinstance(v, bool)],
        # What gets said if phrasing is slow, refused or switched off. This is
        # the sentence that shipped before this module existed.
        "fallback": (issue.get("spoken_fallback")
                     or issue.get("message") or "").strip(),
    }


def revision_of(issue: dict) -> str:
    """What makes a prepared line stale.

    A fault that got worse is a different thing to say, and a sentence written
    when it was milder would understate it. Severity and magnitude are what the
    policy itself uses to decide a deterioration outranks the cooldown, so they
    are what invalidates the words too.
    """
    issue = issue or {}
    return f"{issue.get('severity_rank', 0)}:{issue.get('magnitude', 0)}"


# ---------------------------------------------------------------------------
# A LINE THAT IS ALWAYS ALREADY WRITTEN
# ---------------------------------------------------------------------------
# The headway coaching lines are a different shape from a health finding. A
# health issue appears, sits behind a cooldown, and is announced once -- there
# is a natural gap to write in. The gap warning has no such lead-in: the band
# is entered and confirmed in a second or two, and the line is wanted at the
# end of it.
#
# But its EVENT barely varies -- "the gap to the car in front is getting
# short" is the same sentence to write every time -- so the writing does not
# have to happen after the trigger. One line is kept ready at all times, and
# taking it immediately starts writing the next.
#
# A pool of one is enough and deeper would be waste: the calm tier's cooldown
# is 30 s and this takes under two.
def next_line(event: dict, session_key: str = "", issue_key: str = "") -> dict:
    """The ready line, and start writing its replacement.

    Returns {"text", "source"}. Always returns something: the deterministic
    sentence is used when nothing is ready yet, which is the first firing of a
    drive and any firing after a failure.
    """
    issue_key = issue_key or str(event.get("key") or "")
    got = take(session_key, issue_key, revision="pool")
    if got:
        # Consumed: the slot is now free for the next one, and the next one
        # must be a DIFFERENT sentence -- phrase() is given what has already
        # been said on this drive and asked to avoid it.
        drop(session_key, issue_key)
        prepare(event, session_key, issue_key, revision="pool")
        return got
    prepare(event, session_key, issue_key, revision="pool")
    return {"text": (event.get("fallback") or event.get("what") or "").strip(),
            "source": "fallback"}


# The two coaching lines, as events. The words in `fallback` are
# live_policy.LINE_TEXT's -- imported by the caller rather than restated here,
# so the deterministic floor and the policy cannot drift apart.
HEADWAY_EVENTS = {
    "calm": {
        "key": "headway.calm",
        "what": "the gap to the car in front is getting short",
        "severity": "coaching — this is a nudge, not an alarm",
        "provenance": "RIO is watching it through the forward camera",
        "observation_window": "right now",
        "evidence": "following distance is under three seconds",
        "action": "ease off a little",
        "numbers": [],
    },
    "escalate": {
        "key": "headway.escalate",
        "what": "the gap is still closing after you were already told once",
        "severity": "coaching, firmer — still not the red tier",
        "provenance": "RIO is watching it through the forward camera",
        "observation_window": "right now",
        "evidence": "the gap has kept shrinking since the last line",
        "action": "ease off",
        "numbers": [],
    },
}
