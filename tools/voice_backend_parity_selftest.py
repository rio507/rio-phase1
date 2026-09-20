"""voice_backend_parity_selftest.py — two wires, one drive policy.

    python -m tools.voice_backend_parity_selftest

THE FAULT THIS EXISTS FOR, AND IT WAS REAL. When the xAI mint was first written it
carried 15 fields. The OpenAI one carried 36. The 27 missing ones were the barge
gate's sustain and confirm windows, the echo-text gate, the turn policy that makes
newest-wins work, the speak deadlines per channel, the resume instruction,
max_resumes, the verbatim dictation instruction, look_answer_max_tokens, and the
clip prefix -- and NOT ONE of them would have raised an error. The browser reads
`session.barge_sustain_ms || <its own default>` for each, so a drive on the new
wire would have run with the code's defaults instead of the numbers measured on a
phone in a car, and the only symptom would have been RIO behaving slightly worse
in ways nobody could name.

It also carried no conditional_tools, which meant stop_navigation and reroute --
two of the nine live tools -- simply did not exist for the length of a drive.

So the mints do not each hold a copy any more: realtime.drive_policy() is the one
place, and this asserts that both mints carry it whole and that the tool list is
complete on both. Every field added to the drive's policy from here is therefore
carried by both wires or fails here.

NO NETWORK: the OpenAI mint calls the API to create a secret, so it is read
STATICALLY -- the field names come out of the source. That is enough, because what
is being compared is which keys exist, and it means this runs with no key and no
credits. The xAI mint needs neither (its secret is the account key, scoped by
expiry), so it is called for real.
"""
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import config                                                    # noqa: E402
import realtime                                                  # noqa: E402
import xai_voice                                                  # noqa: E402

_fails: list = []
_checks: list = []


def ok(name, cond, extra=""):
    _checks.append(1)
    if cond:
        print(f"  ok   {name}")
    else:
        _fails.append(name)
        print(f"  FAIL {name}{('  ' + str(extra)) if extra else ''}")


def section(t):
    print(f"\n== {t}")


def mint_keys_from_source() -> set:
    """The field names realtime.mint_client_secret returns, read as text."""
    src = (Path(__file__).resolve().parent.parent / "realtime.py").read_text()
    i = src.index("def mint_client_secret() -> dict:")
    j = src.index("\n\n\n", i)
    body = src[i:j]
    return set(re.findall(r'^\s{8}"([a-z_0-9]+)":', body, re.M))


def main() -> int:
    section("the shared policy is shared")
    shared = realtime.drive_policy(realtime.session_config())
    ok("drive_policy is not empty — a regex or a refactor that emptied it would "
       "make every check below pass for the wrong reason",
       len(shared) >= 20, len(shared))

    openai_keys = mint_keys_from_source() | set(shared)
    ok("the OpenAI mint's fields were found in the source",
       len(openai_keys) >= 30, len(openai_keys))

    x = xai_voice.mint_client_secret()
    # cedar_voice is a dead alias kept for one release; nothing reads it.
    missing = sorted(k for k in openai_keys
                     if k not in x and k not in ("cedar_voice",))
    ok("every field the OpenAI mint sends is sent by the xAI mint too",
       not missing, missing)

    for k in ("barge_sustain_ms", "barge_touch", "turn_policy",
              "echo_text_window_s", "speak_timeout_ms_by_channel",
              "resume_instruction", "verbatim_instruction", "max_resumes",
              "look_answer_max_tokens", "peer_disconnect_grace_ms"):
        ok(f"...including {k}, which the browser would otherwise default",
           k in x and x[k] not in (None, "", {}), x.get(k))

    section("the nine live tools exist on this wire")
    names = list(x["tools"])
    cond = [t["name"] for group in x["conditional_tools"].values()
            for t in group]
    both = set(names) | set(cond)
    required = ["look", "nav_status", "nav_directions", "start_navigation",
                "stop_navigation", "reroute", "vehicle_status", "find_places",
                "deep_dive"]
    absent = [t for t in required if t not in both]
    ok("all nine are reachable, in the session or with their precondition",
       not absent, absent)
    ok("stop_navigation and reroute ride with the route rather than the session "
       "— a tool schema is input on every response and there is nothing to stop "
       "until a route exists",
       "stop_navigation" in cond and "reroute" in cond
       and "stop_navigation" not in names)
    ok("...and the schemas travel, not just the names, because a session.update "
       "replaces the tool list whole",
       all(isinstance(t, dict) and t.get("name")
           for t in x["tool_schemas"]) and len(x["tool_schemas"]) == len(names),
       len(x["tool_schemas"]))

    section("the wire's own fields, whose absence is silent")
    for k in ("ws_url", "ws_subprotocol", "session", "silence_tail_ms"):
        ok(f"{k} is present", bool(x.get(k)), x.get(k))
    pol = x["session"]
    ok("the policy carries output_modalities — without it the session accepts "
       "everything, runs VAD and produces nothing, with no error",
       pol.get("output_modalities") == ["audio"], pol.get("output_modalities"))
    ok("...and an output voice, for the same reason",
       (pol.get("audio", {}).get("output", {}).get("voice")) == config.XAI_VOICE,
       pol.get("audio", {}).get("output", {}))
    ok("server_vad owns the end of an utterance, so the page sends no commit",
       pol["audio"]["input"]["turn_detection"]["type"] == "server_vad")

    section("the voice and its clips follow the backend, not a literal")
    ok("the mint names the configured voice",
       x["voice"] == config.XAI_VOICE and x["live_voice"] == config.XAI_VOICE,
       x["voice"])
    ok("the model is pinned to a version rather than an alias — an alias moves "
       "under a drive",
       x["model"] == config.XAI_VOICE_MODEL and "latest" not in x["model"],
       x["model"])
    # clip_base follows VOICE_BACKEND, which is what makes this worth asserting:
    # the three lines that cannot be re-rendered at the moment they fire are
    # clips, and a backend change that left the previous voice's clips in place
    # would change speaker in exactly those three.
    if config.VOICE_BACKEND == "xai_voice":
        ok("and the clip prefix is this voice's, because a backend change must "
           "not leave the previous voice in the lines that cannot be "
           "re-rendered",
           x["clip_base"] == config.CLIP_DIRS["xai_voice"], x["clip_base"])
    else:
        ok(f"(clip prefix not asserted: VOICE_BACKEND is "
           f"{config.VOICE_BACKEND}, so clip_base is that backend's — rerun "
           f"with VOICE_BACKEND=xai_voice to check it)", True)

    section("/health reports the wire it is actually on")
    ok("the model /health names is the configured backend's",
       realtime.backend_model() == (
           config.XAI_VOICE_MODEL if config.VOICE_BACKEND == "xai_voice"
           else config.GPT_LIVE_MODEL if config.VOICE_BACKEND == "gpt_live"
           else config.OPENAI_REALTIME_MODEL),
       realtime.backend_model())
    ok("...and so is the voice", realtime.backend_voice() == (
           config.XAI_VOICE if config.VOICE_BACKEND == "xai_voice"
           else config.GPT_LIVE_VOICE if config.VOICE_BACKEND == "gpt_live"
           else config.OPENAI_REALTIME_VOICE),
       realtime.backend_voice())
    ok("xai_voice is selectable at all", "xai_voice" in config.VOICE_BACKENDS,
       config.VOICE_BACKENDS)

    print("\n" + "-" * 58)
    print(f'  {"PASS" if not _fails else "FAIL"}: {len(_fails)} failure(s)'
          f" of {len(_checks)} checks")
    for f in _fails:
        print(f"    - {f}")
    return len(_fails)


if __name__ == "__main__":
    sys.exit(main())
