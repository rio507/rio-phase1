import json

from openai import OpenAI

import config
import router as request_router
import vehicle_health
from vision import get_observation

client = OpenAI()

history = [
    {
        "role": "system",
        "content": config.SYSTEM_PROMPT,
    }
]


def _health_block(route: dict) -> str:
    """What this turn is told about the car.

    Two sizes, and the router picks between them.

    EVERY turn gets one line: overall status, how many issues, and the worst one
    in a sentence. It costs about twenty tokens and it is what lets RIO answer
    "everything alright?" without a round trip, and — more importantly — stops
    her saying "sounds good, enjoy the drive" while a tire is going down.

    A turn the router classified as a vehicle-health question gets the full
    normalized structure: every corner, every channel, every issue with its
    observation window and its suggested action. That is a much larger payload
    and it is paid for only when it was actually asked for.

    Never UI state, in either size. See vehicle_health.py's header.
    """
    if not getattr(config, "VEHICLE_HEALTH_ENABLED", True):
        return ""
    try:
        report = (route or {}).get("diagnostic_report")
        if report:
            # The driver asked RIO to interrogate the car, and it has been
            # interrogated. The report's own plain-language summary is in here
            # and is DETERMINISTIC — assembled from the report's fields, not
            # generated. RIO narrates it; RIO does not replace it, because a
            # summary generated from a summary can restate a hedge as a
            # conclusion and nothing downstream would know.
            return ("DIAGNOSTIC REPORT (just run, at the driver's request). "
                    "Lead with `summary`. Codes under `confirmed_faults` and "
                    "`early_detection` are what the VEHICLE reported; anything "
                    "under `rio_observations` is what RIO inferred, and the two "
                    "must not be blurred. Possible causes are possibilities — "
                    "never state one as the cause:\n"
                    + json.dumps(report, indent=1, default=str))

        is_health = request_router.is_vehicle_health(
            (route or {}).get("request_type", ""))
        if is_health:
            ctx = vehicle_health.context(full=True)
            return ("VEHICLE HEALTH (live, from the car's own sensors — "
                    "interpret it, do not read it out; do not claim anything "
                    "outside each issue's observation_window):\n"
                    + json.dumps(ctx, indent=1))
        return vehicle_health.compact_line()
    except Exception as e:
        # A health layer that throws must not cost the driver their answer.
        # RIO simply has nothing to say about the car this turn.
        print(f"[llm] vehicle health context failed: {type(e).__name__}: {e}",
              flush=True)
        return ""


def _compose(user_text: str, route: dict) -> str:
    """The turn as the model sees it: the driver's words, plus this instant.

    Both blocks are about NOW. That is why the composed text is not what goes
    into `history` — see generate_stream.
    """
    parts = []
    scene = get_observation()
    print("RIO DEBUG SCENE:", scene)
    if scene:
        parts.append(f"CURRENT CAMERA OBSERVATION:\n{scene}")

    health = _health_block(route)
    if health:
        parts.append(health)

    if not parts:
        return user_text

    parts.append(f"DRIVER SAID:\n{user_text}")
    parts.append(
        "Use the blocks above as RIO's current context. If the driver asks what "
        "you see, answer from the camera observation. If they ask about the car "
        "itself, answer from the vehicle health data. Do not invent a different "
        "scene, and do not claim anything about the car that is not in the data.")
    return "\n\n".join(parts)


def generate_stream(user_text: str, route: dict = None):
    """One conversation turn, streamed.

    `route` is the router's decision for this utterance, or None. It is only
    used to size the vehicle-health context — the routing itself already
    happened in app.py, once, on the server, so the voice path and the text path
    cannot diverge on it.

    WHAT GOES INTO HISTORY, and why it is not what goes to the model.

    The composed text carries the camera observation and the car's health as
    they are RIGHT NOW. Appending that to `history` — which is what this
    function used to do — meant every turn left a snapshot behind, and twenty
    turns later the model was reading a transcript containing twenty different
    accounts of the road and the tires, all of them stale and none of them
    labelled as such. For the camera that was untidy. For vehicle health it is
    dangerous: a model looking at four old pressure readings has exactly what it
    needs to invent a trend, and inventing trends is the one thing the health
    prompt forbids.

    So the model is sent history + this turn's composed text, and history keeps
    the driver's actual words and RIO's actual reply. The conversation is
    remembered; the instant is not.
    """
    composed = _compose(user_text, route)

    messages = history + [{"role": "user", "content": composed}]

    stream = client.chat.completions.create(
        model=config.OPENAI_CHAT_MODEL,
        messages=messages,
        temperature=config.OPENAI_TEMPERATURE,
        max_completion_tokens=config.OPENAI_MAX_TOKENS,
        reasoning_effort=config.OPENAI_REASONING_EFFORT,
        stream=True,
    )

    full_reply = []

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            full_reply.append(delta)
            yield delta

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": "".join(full_reply)})
    _trim()


def note_turn(user_text: str, reply: str) -> None:
    """Record a turn that was answered somewhere else.

    The visual path (visual_qa.py) runs its own multimodal request against its
    own system prompt, so nothing about that turn passes through
    generate_stream. Without this the driver could ask about a car, get an
    answer, and then find that the next ordinary question had no idea the
    exchange had happened — one conversation split across two memories.

    Only the plain question and the spoken reply are kept. The images and the
    perception grounding stay out: they are large, they are stale within
    seconds, and re-sending them on an unrelated later turn would be worse than
    not having them. Which is the same rule generate_stream now applies to its
    own turns.
    """
    if not user_text or not reply:
        return
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    _trim()


def _trim() -> None:
    if len(history) > 21:
        history[:] = [history[0]] + history[-20:]
