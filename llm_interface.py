from openai import OpenAI
import config
from vision import get_observation

client = OpenAI()

history = [
    {
        "role": "system",
        "content": config.SYSTEM_PROMPT,
    }
]

def generate_stream(user_text: str):
    scene = get_observation()
    print("RIO DEBUG SCENE:", scene)

    if scene:
        user_text = f"""CURRENT CAMERA OBSERVATION:
{scene}

DRIVER SAID:
{user_text}

Use the current camera observation as RIO's visual context. If the driver asks what you see, answer from this observation. Do not invent a different scene."""

    history.append({"role": "user", "content": user_text})

    stream = client.chat.completions.create(
        model=config.OPENAI_CHAT_MODEL,
        messages=history,
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
    not having them.
    """
    if not user_text or not reply:
        return
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": reply})
    _trim()


def _trim() -> None:
    if len(history) > 21:
        history[:] = [history[0]] + history[-20:]