"""vehicle_health_policy.py — whether RIO says something about the car, and what.

Design ref: the Vehicle Health conversation spec. This module is the equivalent
of a check engine light: it decides, on its own, that a fault has crossed the
line where the driver has to be told NOW, and it decides that once.

LLM FIREWALL — the whole reason this is a separate module
---------------------------------------------------------
Same discipline as headway/live_policy.py, restated because it matters here for
a different reason. The headway firewall exists so a hallucinated observation
cannot fire a collision warning. This one exists so the opposite cannot happen
either: a model that is having a conversation about the car must never be in a
position to decide whether an alarm goes off, in either direction — not to raise
one, and not to talk itself out of one.

So:

    THE LLM MAY ANSWER QUESTIONS ABOUT VEHICLE HEALTH.
    THE LLM NEVER DECIDES WHETHER OR WHEN AN ANNOUNCEMENT FIRES.

This file imports NOTHING. Not config, not tires, not telemetry, not openai —
nothing. There is no import statement anywhere below, which is a stronger claim
than "it does not call a model": there is no object in scope that could reach
one. tools/vehicle_health_selftest.py parses this file and asserts the import
set is empty, that `open`/`eval`/`exec`/`__import__` never appear, and that the
strings "openai", "llm" and "gpt" do not occur outside this docstring.

That is also why the tunables are module constants rather than config.py
entries, and why the clock is passed in on every tick: `import config` would be
a hole in exactly the wall this module is, and a policy that reads the clock
itself cannot be replayed from a log. Both are live_policy.py's choices and
neither is an accident.

THE WORDS ARE HERE TOO
----------------------
LINE below is the same device as live_policy.LINE_TEXT: the sentences RIO can
say about a mechanical fault are a fixed table in the deterministic layer, not
something generated at the moment of speaking. The set of things RIO can ever
say unprompted about the car is bounded by this table and nothing else can add
to it. /vehicle/health/voice serves text out of `text_for()` and refuses
anything that is not an id this module issued — the same contract /nav/voice
and /headway_voice already hold.

Numbers are written the way they are meant to be heard ("twenty-nine P S I",
not "29 PSI"), and rounded the way a person says them, which is nav.py's
format_distance discipline applied to pressure and temperature. "Twenty-nine
point three P S I" is a machine talking, and this is the one channel where RIO
interrupts to be listened to.

WHAT IT IS GIVEN
----------------
A list of issues from vehicle_health.py. Each one is a plain dict:

    key                 stable identity — the same fault produces the same key
                        on every tick, which is what makes "announce once" mean
                        anything
    severity            informational | warning | critical
    severity_rank       0-3, so the ordering is explicit rather than implied
    magnitude           how bad, in the issue's own units, higher is worse.
                        Comparable only against ITSELF across ticks — this is
                        the worsening test, not a cross-issue ranking
    type                which line to say
    location            spoken form of where, e.g. "rear left"
    value / unit        the number for the line, if the line takes one
    spoken_fallback     a complete sentence for a type with no template, so a
                        future domain is silent-by-omission never speechless

None of those fields is a sentence RIO says. The one that looks like it —
spoken_fallback — is composed by vehicle_health.py, which is deterministic code
against the same thresholds; it is not model output and cannot be.
"""

# ===========================================================================
# PROVISIONAL BLOCK -- every tunable value in this module lives here.
#
# Inherited from the same convention headway/live_policy.py states, and it
# still applies:
#   "PROVISIONAL prototype values. These are engineering starting points for
#    shadow-mode tuning -- NOT validated safety thresholds. Do not represent
#    them as such anywhere, ever."
# ===========================================================================

# --- severity vocabulary ---------------------------------------------------
INFORMATIONAL = "informational"
ADVISORY = "advisory"
WARNING = "warning"
CRITICAL = "critical"

# `advisory` was added with the diagnostic monitors. It is the level for a
# CONFIRMED finding that belongs on the dashboard and in a drive-start briefing
# but must never interrupt a drive -- a possible slow leak is the case it exists
# for. Without it, everything confirmed had to be either ignorable or worth
# interrupting for, and a slow leak is neither.
SEVERITY_RANK = {"normal": 0, INFORMATIONAL: 1, ADVISORY: 2, WARNING: 3,
                 CRITICAL: 4}

# The threshold at which RIO speaks without being asked. This single constant is
# the spec's "configurable severity threshold", and moving it to WARNING is how
# you would make RIO chattier — deliberately one edit, deliberately in the
# deterministic module, deliberately not something a conversation can change.
#
#   informational   never spoken. Answered if asked, and that is all.
#   warning         never spoken. Dashboard and answers only.
#   critical        spoken once, then held on the dashboard until it resolves.
ANNOUNCE_AT_RANK = SEVERITY_RANK[CRITICAL]

# --- cooldowns -------------------------------------------------------------
# One announcement is usually enough. These three numbers are the whole of
# "do not become annoying".

# How long the same unchanged fault stays quiet before RIO mentions it again.
# Ten minutes: long enough that a driver who heard it and chose to keep going is
# not nagged, short enough that a fault on a two-hour drive is raised more than
# once. A reminder is not a new alarm and it says the same words.
#
# A reminder is additionally gated on the issue NOT currently healing -- see
# R_HEALING below. That gate exists because of something the shadow log showed
# on the first drive it was run against: the driver added air, the pressure was
# good from that moment, and the reminder fired four and a half minutes before
# the healing criteria finished verifying it. RIO would have told them to stop
# for a tire they had just dealt with.
#
# The naive fix is to make this number larger than the healing time. That is the
# wrong fix: it is a coincidence between two unrelated constants, and it breaks
# silently the next time either is tuned. The right statement is the one the
# gate makes -- do not remind about a fault that is currently showing recovery.
REMIND_S = 600.0

# Nothing at all inside this window of the previous announcement, whatever it is
# about. Two criticals arriving together are two problems and one sentence the
# driver can actually take in; the second follows a beat later, not over the top
# of the first. The arbiter would supersede it anyway — this stops it being
# submitted at all, so the reason is recorded rather than silently lost.
MIN_GAP_S = 20.0

# An issue that disappears and comes back is a NEW event, not a repeat, and gets
# a fresh announcement — but only if it was genuinely gone. A fault flickering
# either side of a threshold would otherwise re-announce every few seconds,
# which is the exact failure the cooldown exists to prevent.
RESOLVED_CLEAR_S = 60.0

# How much worse an issue has to get, as a fraction of where it was, before the
# deterioration outranks the cooldown. A tire 6 PSI under target that reaches 8
# has moved 33% and is worth saying again; one that drifts from 6.0 to 6.1 has
# not. Rank escalation (warning -> critical) always counts, whatever the
# magnitude does.
WORSEN_FRAC = 0.25
# Floor for the above, so a magnitude that lives near zero cannot make every
# tick a 25% deterioration.
WORSEN_MIN = 1.0

# After the driver asks for a status report, per-issue cooldowns are cleared —
# the spec's "reset cooldown when the driver requests status" — but the mouth
# stays shut for this long regardless. Both halves matter: without the reset a
# fault that later worsens would be swallowed by a ten-minute timer the driver
# themself started; without the quiet window RIO would announce, three seconds
# after answering the question, the very thing she just answered.
POST_REQUEST_QUIET_S = 90.0

# Ring of issued announcements kept for /vehicle/health/voice to look up. Small
# on purpose: an id that is no longer in here is one RIO is no longer trying to
# say, and serving audio for it would be speaking a line that was retired.
ISSUED_MAX = 24

# ===========================================================================
# END PROVISIONAL BLOCK
# ===========================================================================

# --- reasons recorded on every decision, spoken or not ---------------------
# Same idea as live_policy's: a silence with a reason attached is as useful for
# tuning as an utterance, and both end up in the same log.
R_FIRST = "first_time"
R_WORSENED = "worsened"
R_RETURNED = "resolved_and_returned"
R_REMINDER = "reminder"
R_NOTHING = "nothing_to_say"
R_BELOW_THRESHOLD = "below_announce_threshold"
R_ALREADY = "suppressed_already_announced"
# The fault is still ACTIVE, but it is currently passing its monitor and working
# through its healing criteria. It has not been verified as repaired -- that is
# what the healing criteria are for and they are deliberately slow -- but it is
# no longer the right moment to tell somebody to pull over. A reminder here is
# the system nagging about a problem the driver has already dealt with.
R_HEALING = "suppressed_while_healing"
R_MIN_GAP = "suppressed_by_min_gap"
R_POST_REQUEST = "suppressed_after_status_request"
R_NO_LINE = "suppressed_no_line_for_type"
# Shadow mode: the decision to speak was MADE and then not carried out, because
# this monitor has never been cleared to interrupt anybody. The words are
# composed anyway and handed back as a proposal -- that proposal is the entire
# output of a shadow deployment, and it is worthless if it is not the real one.
R_SHADOW = "shadow_mode_proposal_only"

SPEAK_REASONS = (R_FIRST, R_WORSENED, R_RETURNED, R_REMINDER)


# ===========================================================================
# Numbers, written to be heard
# ===========================================================================
# nav.py rounds a distance the way a person says it before it is ever spoken
# ("in 300 meters", never "in 287 meters"). The same rule, one step further:
# these come back as words, because an alert is the worst possible place to
# discover that a synthesiser reads "29 PSI" as "twenty-nine pounds per square
# inch" on one voice and "two nine P S I" on another. What is written here is
# what is heard.

_ONES = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
         "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
         "sixteen", "seventeen", "eighteen", "nineteen")
_TENS = (None, None, "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
         "eighty", "ninety")


def spoken_int(n) -> str:
    """A whole number 0-999 in words. Hyphenated the way it is said.

    Out of range returns the digits: a four-digit pressure is a broken sensor,
    and the honest failure is an odd-sounding number rather than a wrong one.
    """
    try:
        n = int(round(float(n)))
    except (TypeError, ValueError):
        return "--"
    if n < 0:
        return "minus " + spoken_int(-n)
    if n < 20:
        return _ONES[n]
    if n < 100:
        tens, ones = divmod(n, 10)
        return _TENS[tens] + ("-" + _ONES[ones] if ones else "")
    if n < 1000:
        hundreds, rest = divmod(n, 100)
        return _ONES[hundreds] + " hundred" + (" " + spoken_int(rest) if rest else "")
    return str(n)


def spoken_psi(v) -> str:
    """"twenty-nine P S I". Whole PSI — see the module header on rounding.

    The spaces in "P S I" are not decoration. Synthesisers say the bare token
    "PSI" as a word about half the time, and "pea ess eye" is not a unit.
    """
    return spoken_int(v) + " P S I"


def spoken_temp(v) -> str:
    """"one hundred seventy-one degrees". No scale said out loud — the driver
    knows which one their car is in, and "degrees Fahrenheit" in an alert is
    three syllables of nothing."""
    return spoken_int(v) + " degrees"


def spoken_value(value, unit: str) -> str:
    if value is None:
        return ""
    u = (unit or "").strip().lower()
    if u in ("psi", "pounds", "pound"):
        return spoken_psi(value)
    if u in ("f", "°f", "c", "°c", "deg", "degrees"):
        return spoken_temp(value)
    if u in ("v", "volts", "volt"):
        # Voltage is the one channel where the tenth carries the meaning: 12.1 V
        # and 12.9 V are a dying battery and a healthy one. Said as a person
        # says it — "twelve point one volts".
        try:
            whole = int(float(value))
            tenth = int(round((float(value) - whole) * 10))
            if tenth:
                return f"{spoken_int(whole)} point {spoken_int(tenth)} volts"
            return f"{spoken_int(whole)} volts"
        except (TypeError, ValueError):
            return ""
    if not u:
        return spoken_int(value)
    return f"{spoken_int(value)} {unit}"


# ===========================================================================
# The lines
# ===========================================================================
# Fixed, here, in the deterministic layer — live_policy.LINE_TEXT's rule, and
# for the same reason. The bible governs how these are DELIVERED; it never gets
# to choose whether or what to warn.
#
# Voice notes, from docs/behavior_bible_v1.md: this is OPERATIONAL mode. Short,
# declarative, calm but alert. No preamble, no name, no "warning", no code, no
# ceremony. The driver is being interrupted, so the first four words have to
# carry the whole thing in case they miss the rest.
#
# {loc}    "rear left"
# {value}  the number, already in words
LINE = {
    # --- tires ---
    "possible_blowout":
        "Pull over when it's safe — the {loc} tire is down to {value} and dropping fast.",
    "rapid_pressure_loss":
        "Your {loc} tire is losing air fast — it's at {value}. Worth stopping to look at it.",
    "critical_low_pressure":
        "The {loc} tire is down to {value}. That's low enough that I wouldn't keep driving on it.",
    "tire_overheating":
        "The {loc} tire is running at {value}. Ease off and give it a chance to cool.",
    "tire_sensor_lost_driving":
        "I've lost the sensor on the {loc} tire — that corner's dark to me while we're moving.",
    # --- engine. Not reachable in this phase unless the ECU is in a critical
    # band; here because the whole point of the layer is that a new domain is a
    # row in this table and nothing else. ---
    "oil_pressure_critical":
        "Oil pressure's dropped to {value}. Shut it down as soon as you can safely stop.",
    "coolant_temp_critical":
        "She's running hot — {value}. Pull over before it gets worse.",
    "battery_voltage_critical":
        "The charging system isn't keeping up — {value}. Expect it not to restart.",
    "fuel_pressure_critical":
        "Fuel pressure's down to {value}. Don't be surprised if she stumbles.",
}

# A type with no line above says its supplied sentence rather than nothing. A
# future domain that lands a critical before anyone writes it a line is a design
# gap; a future domain that stays SILENT about a critical is a safety one.
FALLBACK_LINE = "{fallback}"


def compose(issue: dict) -> str:
    """The exact sentence for one issue. -> "" if there is nothing sayable.

    Pure. Given the same issue it returns the same string, which is what lets
    the selftest assert on RIO's actual words instead of on a shape.
    """
    issue = issue or {}
    template = LINE.get(issue.get("type"))
    fallback = (issue.get("spoken_fallback") or "").strip()
    if not template:
        if not fallback:
            return ""
        template = FALLBACK_LINE

    loc = (issue.get("location") or "").strip()
    text = template.replace("{loc}", loc)
    text = text.replace("{value}", spoken_value(issue.get("value"),
                                                issue.get("unit") or ""))
    text = text.replace("{fallback}", fallback)
    # A location that was not there leaves "the  tire"; collapse rather than
    # ship a double space into a synthesiser.
    while "  " in text:
        text = text.replace("  ", " ")
    return text.strip()


# ===========================================================================
# The policy
# ===========================================================================

class VehicleHealthPolicy:
    """Per-process announcement state. Pure: no I/O, no model, no clock.

    `t` is supplied by the caller on every tick (seconds, monotonic across a
    process) so the whole policy is replayable from a log — which is exactly
    what tools/vehicle_health_selftest.py does to it.

    One instance per process, held in app.py, ticked by the browser's poll. The
    single-driver assumption every other stateful thing in this codebase makes
    (`_last_talk`, nav's route registry) is made here too.
    """

    def __init__(self):
        # key -> {"rank", "magnitude", "announced_t", "last_seen_t", "gone_since"}
        self._seen = {}
        self._last_announce_t = None
        self._request_t = None
        self._counter = 0
        self._issued = []          # [(id, text)], newest last, capped
        self.log = []              # every decision, spoken or not

    @staticmethod
    def _blank(t) -> dict:
        return {"rank": 0, "magnitude": 0.0, "announced_t": None,
                "last_seen_t": t, "gone_since": None, "resolved_for": None}

    # -- main entry ---------------------------------------------------------
    def tick(self, issues, t) -> dict:
        """One poll. -> {"announce": {...}|None, "reason": str, ...}

        Gate order is deliberate and mirrors live_policy._decide_voice: the
        states where there is nothing to consider come first, so a silence is
        always attributed to the FIRST reason that applies rather than to
        whichever branch happened to run last.
        """
        t = float(t)
        issues = list(issues or [])
        self._age_out(issues, t)

        candidates = [i for i in issues
                      if int(i.get("severity_rank", 0)) >= ANNOUNCE_AT_RANK]
        if not candidates:
            return self._quiet(R_BELOW_THRESHOLD if issues else R_NOTHING, t)

        # Worst first: severity, then how bad, then key so a tie is stable
        # rather than dependent on dict ordering.
        candidates.sort(key=lambda i: (-int(i.get("severity_rank", 0)),
                                       -float(i.get("magnitude") or 0.0),
                                       str(i.get("key") or "")))

        # Every candidate is examined, not just the worst one: the worst may be
        # a fault already announced ten seconds ago while the second is brand
        # new, and the new one is the news.
        #
        # When none of them speaks, the reason reported is the WORST candidate's
        # actual reason — candidates are sorted worst-first, so that is the one
        # a person reading the log is asking about. This used to be a hardcoded
        # R_ALREADY, which was true while "already announced" was the only way
        # to stay quiet and became a lie the moment it was not: a fault
        # suppressed because it is healing would have been logged as suppressed
        # by a cooldown, and the tuning data would have said the opposite of
        # what happened.
        chosen, why = None, R_NOTHING
        for idx, issue in enumerate(candidates):
            reason = self._why_speak(issue, t)
            if reason in SPEAK_REASONS:
                chosen, why = issue, reason
                break
            if idx == 0:
                why = reason

        if chosen is None:
            return self._quiet(why, t)

        # Timing gates last, so a suppressed announcement is one that was
        # DECIDED and held rather than one that was never considered — and the
        # per-issue state is not marked, so it fires on the next tick.
        if self._request_t is not None and (t - self._request_t) < POST_REQUEST_QUIET_S:
            return self._quiet(R_POST_REQUEST, t, issue=chosen)
        if self._last_announce_t is not None and (t - self._last_announce_t) < MIN_GAP_S:
            return self._quiet(R_MIN_GAP, t, issue=chosen)

        text = compose(chosen)
        if not text:
            return self._quiet(R_NO_LINE, t, issue=chosen)

        # Shadow mode. The decision has been made in full -- severity, cooldown,
        # priority, wording -- and is then not carried out. What comes back is
        # the proposal, which is the only honest way to answer "how often would
        # this have interrupted somebody" before it interrupts anybody.
        #
        # Nothing is written here. Recording the proposal is the caller's job:
        # this module performs no I/O, and that is asserted, not assumed.
        if not chosen.get("announce_allowed", True):
            out = self._quiet(R_SHADOW, t, issue=chosen)
            out["proposal"] = {
                "issue_id": chosen.get("issue_id") or chosen.get("key"),
                "key": chosen.get("key"),
                "code": chosen.get("code"),
                "severity": chosen.get("severity"),
                "text": text,
                "would_have_fired_because": why,
                "at": t,
            }
            # A proposal is a decision that happened, so it consumes the same
            # per-issue cooldown a real announcement would. Otherwise the shadow
            # log would show a fault proposing itself on every poll and would
            # wildly overstate how talkative the monitor really is.
            self._mark(chosen, t)
            return out

        return self._fire(chosen, why, text, t)

    # -- the driver asked ---------------------------------------------------
    def note_status_request(self, t) -> None:
        """The driver asked how the car is. Called from the conversation path
        when the router classifies a vehicle-health question.

        Two effects, and they pull in opposite directions on purpose:

          Every per-issue cooldown is cleared. That is the spec's reset — after
          a report, a fault that later deteriorates must be able to speak again
          rather than sitting behind a ten-minute timer the driver started by
          being curious.

          The mouth is held shut for POST_REQUEST_QUIET_S. Without this the
          clear above would have RIO announce, seconds after answering the
          question, the exact fault she just described. That is not a check
          engine light, it is an argument.
        """
        t = float(t)
        self._request_t = t
        for st in self._seen.values():
            st["announced_t"] = None

    # -- lookup for /vehicle/health/voice -----------------------------------
    def text_for(self, announcement_id: str):
        """The words for an id this module issued, or None.

        This is what makes /vehicle/health/voice a lookup rather than a
        text-to-speech endpoint, exactly as nav.announcement_text does for
        /nav/voice. The browser addresses a decision; it never sends a sentence.
        """
        for aid, text in self._issued:
            if aid == announcement_id:
                return text
        return None

    def state(self) -> dict:
        """Observability for the panel and the log. No decisions are made here."""
        return {
            "tracked": {k: dict(v) for k, v in self._seen.items()},
            "last_announce_t": self._last_announce_t,
            "last_request_t": self._request_t,
            "issued": [aid for aid, _ in self._issued],
        }

    # -- internals ----------------------------------------------------------
    def _age_out(self, issues, t) -> None:
        """Update presence bookkeeping. An issue absent from this tick is not
        forgotten immediately — it has to stay gone for RESOLVED_CLEAR_S before
        its return counts as a new event, or a fault flickering across a
        threshold would re-announce on every other poll."""
        live = set()
        for issue in issues:
            key = str(issue.get("key") or "")
            if not key:
                continue
            live.add(key)
            st = self._seen.setdefault(key, self._blank(t))
            st["last_seen_t"] = t
            if st["gone_since"] is not None:
                # It is back. How long it was away is the thing that decides
                # whether that is news, and it is latched rather than read off
                # `gone_since` — which has to be cleared now — so a return that
                # gets held by MIN_GAP is still a return on the next tick.
                # _fire clears it; nothing else does.
                st["resolved_for"] = max(float(st.get("resolved_for") or 0.0),
                                         t - st["gone_since"])
                st["gone_since"] = None

        for key, st in self._seen.items():
            if key not in live and st["gone_since"] is None:
                st["gone_since"] = t

        # Forget an issue that has been gone long enough to count as resolved.
        # Keeping it would mean a fault that comes back an hour later is
        # compared against numbers measured before it was fixed.
        stale = [k for k, st in self._seen.items()
                 if st["gone_since"] is not None
                 and (t - st["gone_since"]) > RESOLVED_CLEAR_S * 4]
        for k in stale:
            del self._seen[k]

    def _why_speak(self, issue, t) -> str:
        """-> one of SPEAK_REASONS, or a suppression reason. Records nothing."""
        key = str(issue.get("key") or "")
        st = self._seen.get(key)
        rank = int(issue.get("severity_rank", 0))
        mag = float(issue.get("magnitude") or 0.0)

        if st is None or st["announced_t"] is None:
            # Never announced. Either genuinely new, or the driver asked for a
            # status and cleared the timers.
            return R_FIRST

        # Resolved and returned. `resolved_for` is latched by _age_out on the
        # tick the issue comes back and survives until it is acted on, so this
        # reads "it was away long enough that its return is news".
        away = st.get("resolved_for")
        if away is not None and away >= RESOLVED_CLEAR_S:
            return R_RETURNED

        # Worsened. Rank escalation always counts; magnitude has to move a
        # meaningful fraction, so drift does not become an alarm.
        if rank > int(st.get("rank", 0)):
            return R_WORSENED
        prev = float(st.get("magnitude") or 0.0)
        if mag - prev >= max(WORSEN_MIN, abs(prev) * WORSEN_FRAC):
            return R_WORSENED

        if (t - st["announced_t"]) >= REMIND_S:
            # ...unless it is currently healing. `healing_runs` is how many
            # consecutive passing monitor runs the issue has accumulated; above
            # zero means the fault is measurably better right now, whether or
            # not the healing criteria have finished proving it.
            #
            # Note this is NOT the same as the issue being resolved. It is
            # deliberately weaker: resolution needs a stable period as well as a
            # run count, and it should, because one good reading is how a warm
            # tire on a motorway "fixes" a leak. But the bar for *staying quiet*
            # is rightly lower than the bar for *declaring it repaired* — the
            # cost of a wrong silence here is a reminder the driver gets ten
            # minutes later instead, and the cost of a wrong reminder is RIO
            # telling somebody to pull over for a tire they just inflated.
            if int(issue.get("healing_runs") or 0) > 0:
                return R_HEALING
            return R_REMINDER

        return R_ALREADY

    def _mark(self, issue, t) -> None:
        """Record that a decision was reached for this issue at t.

        Shared by the real announcement and the shadow proposal, because both
        ARE decisions and both must consume the cooldown. A shadow deployment
        whose proposals were free of the cooldown would report a rate of
        interruption that the live system would never produce.
        """
        key = str(issue.get("key") or "")
        st = self._seen.setdefault(key, self._blank(t))
        st["announced_t"] = t
        st["rank"] = int(issue.get("severity_rank", 0))
        st["magnitude"] = float(issue.get("magnitude") or 0.0)
        st["gone_since"] = None
        st["resolved_for"] = None
        self._last_announce_t = t

    def _fire(self, issue, reason, text, t) -> dict:
        key = str(issue.get("key") or "")
        self._counter += 1
        # Deterministic id: no clock, no randomness, so a whole session replays
        # to the character. `#n` distinguishes a reminder from the original.
        aid = f"{key}#{self._counter}"

        self._mark(issue, t)

        self._issued.append((aid, text))
        if len(self._issued) > ISSUED_MAX:
            del self._issued[:len(self._issued) - ISSUED_MAX]

        announce = {
            "id": aid,
            "key": key,
            "issue_id": issue.get("issue_id") or key,
            "code": issue.get("code"),
            "type": issue.get("type"),
            "severity": issue.get("severity"),
            "text": text,
            "reason": reason,
            # "tts" or a pre-rendered clip id. The urgent fast path uses a clip
            # for the reason the headway red tier does: a TTS round trip is
            # 300-800 ms, and the whole argument for the fast path is that
            # waiting is what it exists to avoid.
            "audio": issue.get("audio") or "tts",
            "fast_path": bool(issue.get("fast_path")),
            "at": t,
        }
        self.log.append({"t": round(t, 3), "key": key, "spoken": True,
                         "reason": reason, "text": text})
        return {"announce": announce, "reason": reason,
                "since_last_s": None, "at": t}

    def _quiet(self, reason, t, issue=None) -> dict:
        entry = {"t": round(t, 3), "spoken": False, "reason": reason}
        if issue is not None:
            entry["key"] = issue.get("key")
        # A silence with no candidate at all is the normal state of a healthy
        # car and would fill the log with nothing. Only silences that had
        # something to say are worth recording.
        if issue is not None or reason not in (R_NOTHING, R_BELOW_THRESHOLD):
            self.log.append(entry)
            if len(self.log) > 200:
                del self.log[:len(self.log) - 200]
        return {
            "announce": None,
            "reason": reason,
            "since_last_s": (None if self._last_announce_t is None
                             else round(t - self._last_announce_t, 1)),
            "at": t,
        }
