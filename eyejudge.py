"""eyejudge.py — the structured label, enforced by a grammar rather than asked for.

WHY THIS IS NOT A PROMPT
------------------------
MEASURED, on this pod, before this module existed:

    schema at the END of the reading prompt      prose 0 chars, 0 chars
    schema in the MIDDLE of the reading prompt   prose 0 chars, 0 chars
    no schema in the prompt at all               prose 808 chars, 246 chars

A JSON schema anywhere in a prompt to Cosmos-Reason2-2B returns the schema
and answers nothing. Moving it to a second pass fixed the reading and left
the label pass with the same problem in miniature: a model asked in prose for
an object with five keys produces a plausible object with five keys, and
"plausible" is the whole difficulty -- the last two fill-in-the-blanks this
project shipped came back populated with invented values.

Guided decoding removes the question. The grammar admits only strings that
validate against the schema, so the model cannot emit a malformed object, an
invented priority word, or a track id that is not an integer. The failure
modes that remain are the interesting ones: a WELL-FORMED object that is
wrong, which rubric.grade and the shadow counts are there to catch.

WHY IT IS TEXT-ONLY, AND WHAT THAT COSTS
-----------------------------------------
vLLM takes a video as a FILE or a URL, not as the frames sitting in the ring.
Feeding it the window would mean encoding an mp4 -- re-encoding the exact
bytes framebuf.raw_sha exists to prove were the detector's, and writing road
frames to disk against the retention posture in framebuf's header. Both are
real costs and neither is worth paying for a label.

So this pass is given the READING and the MEASURED STATE, not the pixels. The
consequence is stated rather than hidden: `priority`, `about` and
`should_speak` follow from the description and the geometry, which is where
they were always going to come from, and `occluded` becomes a judgement about
what the reading SAID about occlusion rather than about what is in the frame.
That last one is the weakest field in the schema and should be read as such.

THE SERVER IS NOT ASSUMED TO BE THERE
-------------------------------------
If vLLM is not running, `judge` returns None and the caller falls back to the
prompt-asked object. A label is worth having; an eye that stops reading the
road because a sidecar is down is not.
"""
import json
import time

import config
import rubric

# Where the label server lives. Not the RIO server and not the same process:
# vllm needs its own torch (2.13/cu130 against this venv's 2.11/cu128), so it
# is a sidecar by necessity rather than by design.
BASE_URL = str(getattr(config, "EYE_JUDGE_VLLM_URL",
                       "http://127.0.0.1:8899/v1")).rstrip("/")
MODEL = str(getattr(config, "EYE_JUDGE_VLLM_MODEL", "nvidia/Cosmos-Reason2-2B"))
TIMEOUT_S = float(getattr(config, "EYE_JUDGE_TIMEOUT_S", 30.0))

# THE PROMPT WITH NO SCHEMA IN IT. The grammar carries the shape; the prompt
# carries only what the fields MEAN. This is the distinction the measurement
# above is about, and writing the field list back in here would undo it.
INSTRUCTION = (
    "You are the situational-awareness layer for a driving co-pilot. Below "
    "is a description of the last few seconds of road, and the measurements "
    "this vehicle's own sensors took over the same seconds.\n\n"
    "Judge the situation. Be silent by default: do not elevate vehicles "
    "travelling normally in adjacent lanes, pedestrians safely on sidewalks, "
    "ordinary curves, ordinary braking, parked vehicles showing no activity, "
    "or distant objects with no plausible interaction. Something is elevated "
    "only when motion, geometry, behaviour, visibility or context creates a "
    "plausible conflict.\n\n"
    "You are not the collision-warning system. Separate deterministic "
    "systems decide whether a warning is actually given; say only whether "
    "speaking would be justified.\n\n"
    "Name the measured track numbers your judgement concerns. If it concerns "
    "none, name none.")

_available = None
_last_check = 0.0


def available(recheck_after_s: float = 30.0) -> bool:
    """Is the label server up? -> bool. Cached; never raises."""
    global _available, _last_check
    now = time.time()
    if _available is not None and (now - _last_check) < recheck_after_s:
        return _available
    _last_check = now
    try:
        import requests
        r = requests.get(BASE_URL + "/models", timeout=3)
        _available = r.status_code == 200
    except Exception:
        _available = False
    return _available


def build_prompt(reading: str, grounding_text: str = "") -> str:
    parts = [INSTRUCTION, "Description:\n" + (reading or "").strip()]
    if grounding_text:
        parts.append(grounding_text)
    return "\n\n".join(parts)


def judge(reading: str, grounding_text: str = "", schema: dict = None):
    """One grammar-constrained judgement. -> (judgement|None, info).

    Never raises: a label server that is down, slow or confused costs a
    label, and the reading it would have labelled is already in hand.
    """
    info = {"backend": "vllm", "url": BASE_URL, "ms": None, "error": None}
    if not available():
        info["error"] = "server unavailable"
        return None, info
    schema = schema or rubric.JUDGEMENT_SCHEMA
    body = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": build_prompt(reading, grounding_text)},
        ],
        # Short, because the object is short. The reasoning trace in front of
        # it is the only thing that can grow, and qwen3's parser strips it.
        "max_tokens": int(getattr(config, "EYE_JUDGEMENT_MAX_TOKENS", 512)),
        "temperature": 0.6,
        "top_p": 0.98,
        # THE GRAMMAR. This is the whole module.
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "judgement", "schema": schema,
                            "strict": True},
        },
    }
    t0 = time.time()
    try:
        import requests
        r = requests.post(BASE_URL + "/chat/completions", json=body,
                          timeout=TIMEOUT_S)
        info["ms"] = round((time.time() - t0) * 1000, 1)
        if r.status_code != 200:
            info["error"] = f"HTTP {r.status_code}: {r.text[:200]}"
            return None, info
        msg = (r.json().get("choices") or [{}])[0].get("message") or {}
        content = msg.get("content") or ""
        # The reasoning parser hands the trace back separately; keep it, for
        # the same reason the reading pass keeps its own.
        info["reasoning"] = msg.get("reasoning") or msg.get("reasoning_content") or ""
        raw = json.loads(content)
    except Exception as e:
        info["ms"] = round((time.time() - t0) * 1000, 1)
        info["error"] = f"{type(e).__name__}: {e}"
        return None, info

    # THE GRAMMAR GUARANTEES THE SHAPE, NOT THE SENSE. Still normalised here,
    # because a valid object with a track id we are not holding is exactly
    # the failure this whole stage is built to count, and it must arrive in
    # the same form as a prompt-asked one so the scoring cannot tell them
    # apart by accident.
    judgement = {
        "priority": str(raw.get("priority") or "NORMAL").upper(),
        "risk": rubric.to_target(str(raw.get("priority") or "NORMAL")),
        "should_speak": bool(raw.get("should_speak")),
        "why": str(raw.get("why") or "").strip(),
        "about": [int(v) for v in (raw.get("about") or [])
                  if isinstance(v, (int, float))],
        "occluded": bool(raw.get("occluded")),
        "source": "guided",
    }
    return judgement, info
