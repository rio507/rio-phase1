"""llm_provider.py — which vendor answers a ROLE, and the only file that knows.

WHAT THIS IS FOR, in one sentence: `realtime.escalate` should ask for "the
reasoning model" and get a client, without containing the word xai, openai, grok
or gpt anywhere in it.

THE SHAPE IS DELIBERATELY THE SAME AS static/rio_provider.js. That file keeps the
live session's vendor at the browser's door; this one keeps it at the server's.
Two halves of one idea, and the reason is the same: seven code paths were spent
removing ElevenLabs, and a vendor name that spreads is what makes that cost
recur. A capability record rather than an `if` on a model id, so code downstream
asks "does this report its own cost" instead of "is this xAI".

ROLES, NOT MODELS. A role is a job RIO needs done; a model is who happens to be
doing it this week. config.py still names every model id, exactly as it did --
this file does not own them, it routes them.

    reasoning   deep_dive: research and multi-step work, reached as a TOOL, with
                a holding line in front of it. Never a second voice.
    news        localnews' retrieval and background calls. Same vendor as
                reasoning by design (no second search stack) but a different
                CALL, for the reasons in localnews.py's header.
    chat        the text conversation path (/talk, /ask).

WHAT WAS MEASURED BEFORE ANY OF THIS WAS WRITTEN, because the spec asked for
stage 1 to be measured against the current stack and the answer changes what the
code has to do. Three deep_dive questions, the shipped instructions, the shipped
45 s timeout, web_search on, n=1 each (2026-09-19):

    gpt-5.6-sol   8.2 s median   reasoning tokens 58-243
    grok-4.6     31.7 s median   reasoning tokens 1439-1954   $0.042-$0.126

Four findings came out of that, and three of them are load-bearing here:

  1. grok-4.6 REASONS MUCH HARDER BY DEFAULT and it is most of the latency.
     Measured on one question: effort low 12.5 s, default 20.5 s, high 24.6 s,
     for answers of 716, 832 and 712 characters -- the same answer, two and a
     half times the wait. So the default is wrong for this path and
     XAI_REASONING_EFFORT is "low". That is a CONFIG choice made on a
     measurement, and the measurement is one question; it wants repeating over a
     drive's worth of real ones.

  2. max_output_tokens IS NOT A CAP. Asked for 200, grok-4.6 returned 849 output
     tokens (725 of them reasoning), status "completed", incomplete_details
     None. realtime.escalate sends DEEP_ANSWER_MAX_TOKENS +
     DEEP_REASONING_MAX_TOKENS as one number precisely BECAUSE the old API
     enforced it -- and the failure path that reports "thought past the answer
     budget" keys on incomplete_details.reason == "max_output_tokens", which can
     now never fire. On this vendor deep_dive has no spend ceiling but the
     client timeout. See `enforces_token_cap` below.

  3. THE COST COMES BACK EXACT. usage.cost_in_usd_ticks, at 1e-10 USD per tick,
     reconciles to the fraction of a cent against list prices -- and it revealed
     that cached input bills at $0.50/M against $2.00 uncached. Checked:
     625 output at $6/M + 149 uncached input at $2/M + 512 cached at $0.50/M =
     $0.004304, and the API reported 43,040,000 ticks.

     This is the answer to the bug in config.NEWS_IN_COST_USD, which priced
     gpt-5.6-sol's tokens at gpt-5's tier and was wrong by 4x on input and 3x on
     output for as long as it has been quoted. An estimate derived from list
     prices is a thing that goes stale silently; a number the vendor returns is
     not. So `reports_cost` is a capability, and any caller that has it should
     prefer it to arithmetic.

  4. SERVER-SIDE SEARCH STUFFS ABOUT 5x THE CONTEXT. On the one question both
     vendors searched hardest, input was 12,824 tokens against 63,984. That is
     the "3-5x a standard completion" the migration warned about, measured, and
     it is stage 2's whole cost story rather than this file's.
"""
import threading
from typing import Optional

from openai import OpenAI

import config

_clients = {}
_lock = threading.Lock()

# ---------------------------------------------------------------------------
# A SEAM FOR TESTS, SPELLED OUT RATHER THAN LEFT TO A PRIVATE GLOBAL
# ---------------------------------------------------------------------------
# tools/realtime_selftest.py fakes the reasoning client to drive every failure
# deep_dive can have -- a timeout, a refusal, an empty answer -- without a
# network. It used to do that by assigning realtime._client, which worked because
# escalate() built its own client there.
#
# When escalate() started asking this module instead, that assignment silently
# stopped intercepting anything: the fake sat unused, the REAL API answered, and
# the suite failed on an answer that did not match the script -- having spent
# money to get it. A refactor that moves a test's seam without moving the test is
# the same class of quiet as everything else in this repo's recent history.
#
# So the seam is public and per-role. A test says which role it is faking, which
# is also the thing it actually means.
_override = {}


def override(role: str, fake) -> None:
    """Answer `role` with `fake` until cleared. Tests only."""
    vendor_of(role)                      # raises on a role nobody has defined
    _override[role] = fake


def clear_overrides() -> None:
    _override.clear()

# ---------------------------------------------------------------------------
# The vendors, as facts rather than as names to branch on
# ---------------------------------------------------------------------------
# `base_url=None` means the SDK's own default, which is OpenAI. Spelled as None
# rather than as the URL so that nothing here has to be updated when they move
# it, and so the openai entry reads as "the default" rather than as a peer.
VENDORS = {
    "openai": {
        "label": "OpenAI",
        "base_url": None,
        "key_env": "OPENAI_API_KEY",
        # Does max_output_tokens actually stop generation? The whole of
        # escalate()'s budget argument rests on this being true.
        "enforces_token_cap": True,
        # Does usage carry what the call cost? None means "no, estimate it".
        "reports_cost": None,
        # Reasoning effort values this vendor's TEXT models accept on Responses.
        # Distinct from the voice model's scale, which is high|none -- see
        # static/rio_provider.js. Two different scales on one vendor is exactly
        # the sort of thing a capability record exists to stop being folded into
        # one constant.
        "reasoning_efforts": ("none", "low", "medium", "high", "xhigh"),
        # How a server-side search shows up in the response.
        "search_count": "output_items",     # count web_search_call items
    },
    "xai": {
        "label": "xAI",
        "base_url": "https://api.x.ai/v1",
        "key_env": "XAI_API_KEY",
        # MEASURED FALSE. See finding 2 in the module docstring.
        "enforces_token_cap": False,
        # usage.cost_in_usd_ticks, at this many USD per tick.
        "reports_cost": ("cost_in_usd_ticks", 1e-10),
        "reasoning_efforts": ("low", "high"),
        # usage.num_server_side_tools_used.
        #
        # NOT because the output items are missing -- and a commit message of
        # mine said they were, wrongly. Measured on three real news calls, xAI
        # emits web_search_call items too and the two agree exactly: 15 and 15,
        # 8 and 8, 2 and 2. The old counting method would have worked.
        #
        # The usage field is still the right source, for a smaller reason than
        # the one I claimed: it is one integer the vendor computed rather than a
        # scan of an output list whose item taxonomy is theirs to change, and on
        # a 32-item response most of those items are `reasoning`. But nothing was
        # broken here, and a justification that overstates itself is how a
        # capability record stops being trustworthy.
        "search_count": "usage_field",
    },
}

# Which vendor does which job. One switch per role rather than one global, so a
# stage can be migrated and measured on its own -- which is the whole point of
# doing these in an order.
ROLE_VENDOR = {
    "reasoning": config.REASONING_VENDOR,
    "news": config.NEWS_VENDOR,
    "chat": config.CHAT_VENDOR,
}

ROLE_MODEL = {
    "reasoning": lambda: config.reasoning_model(),
    "news": lambda: config.news_model(),
    "chat": lambda: config.chat_model(),
}


def vendor_of(role: str) -> str:
    try:
        return ROLE_VENDOR[role]
    except KeyError:
        raise ValueError(f"llm_provider: unknown role {role!r} "
                         f"(have: {', '.join(sorted(ROLE_VENDOR))})")


def model_of(role: str) -> str:
    if role not in ROLE_MODEL:
        raise ValueError(f"llm_provider: unknown role {role!r}")
    return ROLE_MODEL[role]()


def capability(role: str, name: str):
    """A fact about the vendor behind this role. Raises on a name nobody has
    thought about, rather than returning None and letting a caller read that as
    'no' -- which is the difference between a record and a dict."""
    v = VENDORS[vendor_of(role)]
    if name not in v:
        raise KeyError(f"llm_provider: no capability {name!r} recorded for "
                       f"{vendor_of(role)}; add it to VENDORS rather than "
                       f"guessing at the call site")
    return v[name]


def label(role: str) -> str:
    return VENDORS[vendor_of(role)]["label"]


def client(role: str) -> OpenAI:
    """The client for a role. Cached per vendor, because a client is a connection
    pool and two of them for one vendor is two pools."""
    if role in _override:
        return _override[role]
    name = vendor_of(role)
    with _lock:
        if name not in _clients:
            v = VENDORS[name]
            kw = {}
            if v["base_url"]:
                kw["base_url"] = v["base_url"]
            key = config.vendor_key(v["key_env"])
            if key:
                kw["api_key"] = key
            _clients[name] = OpenAI(**kw)
        return _clients[name]


def reasoning_effort(role: str) -> Optional[str]:
    """The effort to ask for, or None to leave it to the vendor.

    Checked against what the vendor actually accepts, because an effort string
    the API refuses is a 400 on a tool call, which the driver hears as RIO
    declining to look something up.
    """
    want = config.reasoning_effort_for(role)
    if not want:
        return None
    allowed = VENDORS[vendor_of(role)]["reasoning_efforts"]
    if want not in allowed:
        print(f"[llm] {vendor_of(role)} does not take reasoning effort "
              f"{want!r} (has {allowed}); sending none", flush=True)
        return None
    return want


def cost_usd(role: str, usage) -> Optional[float]:
    """What the call actually cost, when the vendor says so.

    None means it did not, and the caller has to estimate -- which is what
    config's NEWS_*_COST_USD constants are for, and what went wrong with them.
    A returned figure here is the vendor's own number and needs no rates.
    """
    rec = VENDORS[vendor_of(role)]["reports_cost"]
    if not rec or usage is None:
        return None
    field, unit = rec
    ticks = getattr(usage, field, None)
    if ticks is None and isinstance(usage, dict):
        ticks = usage.get(field)
    if ticks is None:
        return None
    try:
        return round(float(ticks) * unit, 6)
    except (TypeError, ValueError):
        return None


def searches_of(role: str, resp) -> int:
    """How many server-side searches this response ran.

    Two vendors, two places to look, and the old code looked in only one: it
    counted output items of type web_search_call, which xAI does not emit.
    Counting zero searches on a call that ran five is how a per-drive budget
    stops being a budget -- see localnews._charge, whose whole point is to debit
    what was spent rather than what was asked for.
    """
    how = VENDORS[vendor_of(role)]["search_count"]
    if how == "usage_field":
        u = getattr(resp, "usage", None)
        n = getattr(u, "num_server_side_tools_used", None) if u else None
        return int(n or 0)
    return sum(1 for item in (getattr(resp, "output", None) or [])
               if getattr(item, "type", "") == "web_search_call")


def describe(role: str) -> dict:
    """For /health, the drive log and the selftests: what is answering this role."""
    v = VENDORS[vendor_of(role)]
    return {
        "role": role, "vendor": vendor_of(role), "label": v["label"],
        "model": model_of(role),
        "reasoning_effort": reasoning_effort(role),
        "enforces_token_cap": v["enforces_token_cap"],
        "reports_cost": bool(v["reports_cost"]),
    }
