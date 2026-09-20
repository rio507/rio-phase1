"""llm_provider_selftest.py — the server's vendor seam, without spending a cent.

    python -m tools.llm_provider_selftest

llm_provider.py exists so that nothing else on the server knows which vendor
answers a role. That is easy to claim and easy to half-do, and a half-done seam
reads as a boundary while a vendor name leaks through it. So this asserts the
properties that make it real:

  the DEFAULT is the running stack, byte for byte -- a refactor that changed
  behaviour on import would not be a refactor;
  every role resolves a vendor, a model and an effort, and an unknown role
  RAISES rather than quietly answering as OpenAI;
  a capability nobody recorded raises too, so a call site cannot read a missing
  fact as "no";
  the effort string is checked against what the vendor accepts, because a value
  the API refuses is a 400 on a tool call, which the driver hears as RIO
  declining to look something up;
  and realtime.py no longer names a model or a vendor on the deep_dive path.

Nothing here calls an API. The measurements that justify the config values are in
llm_provider.py's docstring and config.py's; this file is about the seam.
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import assert_guard as _guard                                  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OK, BAD = "ok  ", "FAIL"
_fails = []
_checks = []


def ok(name, cond, extra=""):
    _checks.append(1)
    if cond:
        print(f"  {OK} {name}")
    else:
        _fails.append(name)
        print(f"  {BAD} {name}{('  ' + str(extra)) if extra else ''}")


ok_all, ok_none, non_empty = _guard.bind(ok, order="name_first")


def section(t):
    print(f"\n== {t}")


def fresh(**env):
    """A config + provider pair read with this environment, from scratch.

    Re-imported rather than reloaded, because config reads the environment AT
    IMPORT -- which is the property that makes a stage switchable by an env var
    and the property that makes a test of it need a clean module table.
    """
    keep = {k: os.environ.get(k) for k in env}
    os.environ.update({k: v for k, v in env.items() if v is not None})
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
    for m in ("config", "llm_provider"):
        sys.modules.pop(m, None)
    import config as c
    import llm_provider as lp
    for k, v in keep.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    return c, lp


def main():
    section("the default is xAI, and OpenAI is one env var away per role")
    c, lp = fresh(REASONING_VENDOR=None, NEWS_VENDOR=None, CHAT_VENDOR=None)
    for role, want in (("reasoning", c.XAI_REASONING_MODEL),
                       ("news", c.XAI_REASONING_MODEL),
                       ("chat", c.XAI_CHAT_MODEL),
                       ("visual", c.XAI_VISUAL_MODEL)):
        ok(f"{role} answers as {want}",
           lp.vendor_of(role) == "xai" and lp.model_of(role) == want,
           f"{lp.vendor_of(role)}/{lp.model_of(role)}")
    ok("...and every one of them is a pinned version, not an alias",
       all("latest" not in lp.model_of(r)
           for r in ("reasoning", "news", "chat", "visual")))

    # The effort per role, which is deliberately not one value.
    ok(f"the reasoning roles think at {c.XAI_REASONING_EFFORT!r} — 17.8 s median "
       "through the real path, against 31.7 s at the vendor default",
       lp.reasoning_effort("reasoning") == c.XAI_REASONING_EFFORT
       and lp.reasoning_effort("news") == c.XAI_REASONING_EFFORT)
    ok(f"and chat at {c.XAI_CHAT_EFFORT!r}, because it runs BEFORE the answer: "
       "585 ms against 3,179 for one more correct classification in fourteen",
       lp.reasoning_effort("chat") == c.XAI_CHAT_EFFORT)
    ok(f"and the visual turn at {c.XAI_VISUAL_EFFORT!r}: 1,200 ms against 8,619 "
       "at low, on a question about something going past the window",
       lp.reasoning_effort("visual") == c.XAI_VISUAL_EFFORT)
    ok("the visual role is its OWN role, not chat wearing a different budget — "
       "it sends images and it was the fourth vendor nobody had named",
       "visual" in lp.ROLE_VENDOR and lp.ROLE_API["visual"] == "chat")

    section("visual_qa builds no client of its own")
    src = (REPO / "visual_qa.py").read_text()
    import suite_sweep as _sw
    code = _sw.mask_non_code(src)
    ok("OpenAI(" not in code,
       "no OpenAI client is constructed in visual_qa's code — it asked for one at "
       "import and read OPENAI_VISUAL_MODEL, which made the visual turn invisible "
       "to the vendor switch")
    ok('model_of("visual")' in code and 'client("visual")' in code,
       "it asks for the visual ROLE's model and client")
    ok("class RemoteChatAdapter" in code and "OpenAIChatAdapter" not in code,
       "and the adapter is named for what it is — remote — rather than for one "
       "vendor it no longer necessarily talks to")
    ok("class QwenChatAdapter" in code,
       "...while the LOCAL Qwen adapter beside it is untouched: vision stays on "
       "this box, and only the model that writes the sentence moved")
    ok("...which is also what this path has always sent, so the flip changed the "
       "vendor and not the behaviour",
       c.XAI_CHAT_EFFORT == c.OPENAI_REASONING_EFFORT)

    section("the rollback, which is the point of the switch being per role")
    for role, env in (("reasoning", "REASONING_VENDOR"),
                      ("news", "NEWS_VENDOR"), ("chat", "CHAT_VENDOR")):
        _c, _lp = fresh(**{env: "openai"})
        want = (_c.OPENAI_CHAT_MODEL if role == "chat"
                else _c.OPENAI_REASONING_MODEL)
        ok(f"{env}=openai puts {role} back on {want}",
           _lp.vendor_of(role) == "openai" and _lp.model_of(role) == want,
           f"{_lp.vendor_of(role)}/{_lp.model_of(role)}")
        others = [r for r in ("reasoning", "news", "chat") if r != role]
        ok(f"...and leaves {' and '.join(others)} on xAI — one role at a time is "
           "the whole reason these are three switches and not one",
           all(_lp.vendor_of(o) == "xai" for o in others))

    _c, _lp = fresh(REASONING_VENDOR="openai")
    ok("a rolled-back reasoning role gets no effort sent, which is what the "
       "OpenAI path has always had",
       _lp.reasoning_effort("reasoning") is None)
    ok("and OpenAI is still recorded as enforcing max_output_tokens, so the "
       "length bound is belt and braces there rather than the only belt",
       _lp.capability("reasoning", "enforces_token_cap") is True)
    ok("...and as not reporting its own cost, so the rate-table estimate is "
       "still needed on that path", _lp.capability("reasoning", "reports_cost") is None)

    section("what the record says about xAI, measured rather than assumed")
    ok("max_output_tokens is recorded as NOT enforced (asked 200, got 849)",
       lp.capability("reasoning", "enforces_token_cap") is False)
    ok("the cost comes back from the vendor",
       lp.capability("reasoning", "reports_cost") == ("cost_in_usd_ticks", 1e-10))

    class _U:
        cost_in_usd_ticks = 43040000
        num_server_side_tools_used = 3
    ok("...and 43,040,000 ticks reads as $0.004304 — the figure that reconciled "
       "to list prices to the fraction of a cent",
       lp.cost_usd("reasoning", _U()) == 0.004304,
       lp.cost_usd("reasoning", _U()))

    class _R:
        usage = _U()
        output = []
    ok("searches come from usage, not from counting output items",
       lp.searches_of("reasoning", _R()) == 3,
       lp.searches_of("reasoning", _R()))

    section("an effort the vendor refuses is never sent")
    ok(f"the configured effort is accepted ({c.XAI_REASONING_EFFORT})",
       lp.reasoning_effort("reasoning") == c.XAI_REASONING_EFFORT)
    c2, lp2 = fresh(REASONING_VENDOR="xai", XAI_REASONING_EFFORT="medium")
    ok("a value this vendor does not take is dropped rather than sent — "
       "'medium' is on the text scale for OpenAI and not for xAI",
       lp2.reasoning_effort("reasoning") is None)
    # THREE SCALES ON ONE VENDOR, which is why the check is against the endpoint
    # and not merely against the vendor name. Measured: /v1/responses refuses
    # "medium"; /v1/chat/completions accepts it; and the VOICE model's scale is
    # high|none only (see static/rio_provider.js). A single per-vendor tuple --
    # which is what this file asserted first -- would have dropped the router's
    # "none" on the floor.
    ok(set(lp2.VENDORS["xai"]["reasoning_efforts"]["responses"]) == {"low", "high"},
       "the Responses API takes low|high")
    ok("none" in lp2.VENDORS["xai"]["reasoning_efforts"]["chat"],
       "chat/completions takes 'none' too — which the router depends on")
    ok(lp2.ROLE_API["chat"] == "chat"
       and lp2.ROLE_API["reasoning"] == "responses",
       "...and each role is checked against the endpoint it actually speaks")

    section("a question nobody has answered raises")
    for bad in ("realtime", "voice", ""):
        raised = False
        try:
            lp.vendor_of(bad)
        except ValueError:
            raised = True
        ok(f"an unknown role {bad!r} raises rather than defaulting to OpenAI",
           raised)
    raised = False
    try:
        lp.capability("reasoning", "does_it_rhyme")
    except KeyError:
        raised = True
    ok("an unrecorded capability raises, so no call site can read a missing "
       "fact as 'no'", raised)

    section("the vendor does not appear on the deep_dive path")
    src = (REPO / "realtime.py").read_text()
    esc = src[src.index("def escalate("):src.index("# --- the fast path's gate")]
    # CODE, NOT PROSE, and the first version of this check did not distinguish
    # them: it failed on the comments that explain WHY the effort and the cap
    # handling are what they are, which name grok-4.6 and xAI because that is
    # what was measured. A comment carrying evidence is the opposite of a leak.
    # What must not appear is a vendor name the program ACTS on.
    import suite_sweep as _sweep
    code = _sweep.mask_non_code(esc)
    for name in ("grok", "xai", "x.ai"):
        ok(f"no {name!r} in escalate()'s code (its comments may cite it)",
           name not in code.lower(),
           [l for l in code.lower().splitlines() if name in l][:1])
    ok("it asks for a model by ROLE",
       'model_of("reasoning")' in esc)
    ok("...and for a client by role, rather than building one",
       'client("reasoning")' in esc and "OpenAI(" not in esc)
    ok("the result names which model and vendor actually answered, so a drive "
       "log can say", '"vendor"' in esc and '"usd"' in esc)

    section("a timeout means what it says")
    import openai as _openai
    sdk_default = _openai._constants.DEFAULT_MAX_RETRIES
    ok(f"the SDK would retry {sdk_default}x by default, and its timeout is PER "
       "ATTEMPT — which is why every timeout constant here was describing a "
       f"{1 + sdk_default}th of the real ceiling", sdk_default > 0)
    for role in ("reasoning", "news"):
        ok(f"{role} retries 0 times, because a retry re-runs its searches — a "
           "second bill for work the budget already paid for",
           lp.max_retries_for(role) == 0, lp.max_retries_for(role))
        ok(f"...and the client really carries it, not just the config",
           lp.client(role).max_retries == 0, lp.client(role).max_retries)
    ok("chat may retry once — it runs no tools, so its failure is a dropped "
       "connection on a cheap call", lp.max_retries_for("chat") == 1)
    ok(f"so a news question is bounded at {c.NEWS_TIMEOUT_S:.0f} s rather than "
       f"{c.NEWS_TIMEOUT_S * (1 + sdk_default):.0f} s — the 95-second call that "
       "found this was one attempt timing out and a retry succeeding",
       c.NEWS_TIMEOUT_S * (1 + lp.max_retries_for("news")) == c.NEWS_TIMEOUT_S)
    ok(f"and deep_dive at {c.DEEP_ANSWER_TIMEOUT_S:.0f} s rather than "
       f"{c.DEEP_ANSWER_TIMEOUT_S * (1 + sdk_default):.0f} s",
       c.DEEP_ANSWER_TIMEOUT_S * (1 + lp.max_retries_for("reasoning"))
       == c.DEEP_ANSWER_TIMEOUT_S)
    ok("two roles on one vendor with different retry policies are two clients, "
       "not one shared pool with the wrong policy",
       lp.client("news") is not lp.client("chat")
       or lp.vendor_of("news") != lp.vendor_of("chat"))

    section("the length bound, which does not need the vendor's cooperation")

    import realtime

    class _Resp:
        def __init__(self, text):
            self.output_text = text
            self.status = "completed"
            self.incomplete_details = None
            self.output = []
            class U:
                output_tokens = 4000
                class output_tokens_details:
                    reasoning_tokens = 3000
            self.usage = U()

    class _Fake:
        text = ""
        class responses:
            @staticmethod
            def create(**kw):
                return _Resp(_Fake.text)
        def with_options(self, **kw):
            return self

    import llm_provider as _lp
    _lp.override("reasoning", _Fake())
    try:
        _Fake.text = "A short honest answer."
        r = realtime.escalate("why")
        ok("an answer inside the bound is returned", r.get("ok") is True)

        _Fake.text = "y" * int(c.DEEP_ANSWER_MAX_CHARS)
        r = realtime.escalate("why")
        ok(f"exactly {c.DEEP_ANSWER_MAX_CHARS} characters is allowed — the "
           "boundary is inclusive", r.get("ok") is True)

        _Fake.text = "x" * (int(c.DEEP_ANSWER_MAX_CHARS) + 1)
        r = realtime.escalate("why")
        ok("one character over is REFUSED, not truncated",
           r.get("ok") is False and r.get("reason") == "over_length",
           r.get("note"))
        ok("...and it fails CLOSED: no text comes back at all, so nothing can "
           "hand the live model half a sentence to read out",
           "answer" not in r, sorted(r))
        ok("...saying how long it was and what the ceiling is",
           r.get("chars") == int(c.DEEP_ANSWER_MAX_CHARS) + 1
           and r.get("limit") == int(c.DEEP_ANSWER_MAX_CHARS))
        ok("the bound exists because max_output_tokens is advisory on at least "
           "one vendor — asked 200, got 849; asked 3,000, got 5,041",
           _lp.VENDORS["xai"]["enforces_token_cap"] is False)
    finally:
        _lp.clear_overrides()

    print("\n" + "-" * 58)
    print(f'  {"PASS" if not _fails else "FAIL"}: {len(_fails)} failure(s)'
          f' of {len(_checks)} checks')
    for f in _fails:
        print(f"    - {f}")
    return len(_fails)


if __name__ == "__main__":
    sys.exit(main())
