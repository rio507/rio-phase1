"""gpt-live-1: the voice layer, and the backend it hands the thinking to.

WHAT IS DIFFERENT HERE, in one paragraph, because everything else in this file
follows from it. realtime.py talks to ONE model that hears, thinks, calls tools
and speaks. gpt-live-1 only hears and speaks. It has no tools and no idea what
a tire is. When the driver asks something it cannot answer out of its own
mouth, it raises a DELEGATION, and a backend text model -- gpt-5.6-luna, chosen
by measurement in tools/live_backend_bench.py -- does the thinking and calls
RIO's tools through the Responses API.

So this module owns three things realtime.py does not have to separate:

  the live prompt      who she is and how she talks. SHORT: the voice layer
                       has a small context window and the prompting guide is
                       explicit that procedure does not belong in it.
  the backend prompt   tool discipline, brevity, the field-of-view rules --
                       everything that used to ride in one prompt and is now
                       read by the model that actually decides.
  the seam             minting, the SDP proxy, and the deterministic-speech
                       path, which stopped being a dictation and became a
                       request. See commentary_event().

WHAT IS NOT DIFFERENT, and is not allowed to become different: the tools are
realtime.py's, imported rather than restated; the arbiter is untouched and
still announces; the deterministic/answering boundary is exactly where it was.
A second copy of a tool schema is a routing change nobody made on purpose.
"""
import json
import os

import httpx

import config
import realtime
import rio_prompts

LIVE_SESSIONS_URL = "https://api.openai.com/v1/live/sessions"


# ---------------------------------------------------------------------------
# THE TWO PROMPTS
# ---------------------------------------------------------------------------
# The split is the prompting guide's, and it is not cosmetic: the live model
# reads its prompt on every turn of a full-duplex conversation, and the backend
# reads its own only when there is something to think about. Putting the tool
# rules in the live prompt would pay for them continuously to no purpose, and
# putting the personality in the backend would make her sound like herself only
# on the turns that happened to need a tool.

LIVE_ADDENDUM = """
# How you talk, here

You are the voice. You listen and you speak, and you may do both at once —
that is what this is for. Let the driver interrupt you; stop when they do,
and do not announce that you were interrupted.

# What you cannot do, and must never pretend to

You have no camera, no sensors, no map and no search. You cannot see the road,
the car in front, a sign, a building or the weather. You do not know the
route, the ETA, the tire pressures, or what is nearby. NONE of that is
something you may answer from your own knowledge, or guess at, or hedge about.

So: any question about what is out of the window, where we are, how the drive
is going, how the car is, what is nearby, or anything needing current facts —
HAND IT OFF, every time, without exception. That includes "what's that",
"what do you see", "what kind of car is that", "how far", "how are my tires",
"is there a coffee place near here".

Never say you cannot see something, never ask the driver to describe it, and
never answer one of those from memory. Handing it off IS the answer.

# While it is being worked on

Say nothing about the machinery. No "let me check", no "one moment", no
narrating a tool. A short natural line is fine if the answer will take a
moment; silence is also fine.

When you are given commentary to say, say it exactly as written.
""".strip()


def live_instructions() -> str:
    """The voice layer's prompt: who she is, and how she holds a turn.

    Built from the SAME bible as every other path (rio_prompts.live_prompt),
    because the alternative is a second RIO who agrees with the first one until
    somebody edits one of them.
    """
    return "\n\n".join(p for p in (rio_prompts.live_prompt(), LIVE_ADDENDUM)
                       if p)


BACKEND_ADDENDUM = """
# What you are

You are the reasoning behind RIO's voice. You do not speak to the driver; a
voice model says what you produce, aloud, in a moving car. Write what she
should SAY — no headings, no lists, no markdown, no stage directions.

# Reaching for a tool

Pick the one tool that answers what was actually asked, or none if you already
know. Do not call a tool to look busy.

- Anything the forward camera can see — the road, a car, a sign, a building,
  "what's that", "what do you see" — is `look`, and nothing else is.
- The route is `nav_status` (where it stands) or `nav_directions` (the turns).
  Starting one is `start_navigation`. A business or place near the car is
  `find_places`.
- The car itself is `vehicle_status`.
- `deep_dive` is for research and multi-step reasoning only: no picture, no
  sensor, no route behind it. It is slower and the driver feels it.

# Length

One or two sentences. This is speech, and the driver is driving. A question
that deserves more gets more only when the driver asks for more.

# Truthfulness

Say what the data supports and no more. If a tool did not return something,
you do not know it. Never invent a street, a business, a distance, a reading
or a fact, and never describe something the camera did not show you.
""".strip()


def backend_instructions() -> str:
    """The prompt for the model that thinks.

    THE FIELD-OF-VIEW AND TRUTHFULNESS RULES LIVE HERE NOW, and that is the
    one genuinely load-bearing consequence of the split. They used to be in
    the single prompt the one model read; the model that can now break them is
    this one, because it is the one holding the tool output. A rule enforced
    on the model that cannot act on it is not enforced.
    """
    return "\n\n".join(p for p in (rio_prompts.live_prompt(),
                                   BACKEND_ADDENDUM) if p)


# ---------------------------------------------------------------------------
# THE SESSION
# ---------------------------------------------------------------------------

def backend_tools() -> list:
    """RIO's tools, in the shape Responses delegation wants.

    realtime.BASE_TOOLS, imported. The conditional pair is attached the same
    way it is under the other backend -- when its precondition holds -- and by
    the same argument: two tools nobody can call are paid for on every turn.
    """
    out = []
    for t in realtime.BASE_TOOLS:
        spec = dict(t)
        spec.setdefault("type", "function")
        out.append(spec)
    return out


def conditional_tools() -> dict:
    return {name: [dict(t) for t in tools]
            for name, tools in realtime.CONDITIONAL_TOOLS.items()}


def session_config() -> dict:
    """The live session, as /v1/live/sessions wants it.

    `delegation.responses` is where the second model is named. Everything the
    backend needs to do its job -- its prompt, its tools -- rides inside that
    object rather than at the top level, which is the API's way of saying the
    two models are configured separately because they ARE separate.
    """
    return {
        "model": config.GPT_LIVE_MODEL,
        "instructions": live_instructions(),
        "audio": {"output": {"voice": config.GPT_LIVE_VOICE}},
        "delegation": {
            "type": config.GPT_LIVE_DELEGATION,
            "responses": {
                "model": config.GPT_LIVE_BACKEND_MODEL,
                "instructions": backend_instructions(),
                "tools": backend_tools(),
                "tool_choice": "auto",
                "max_output_tokens": int(config.REALTIME_TOOL_MAX_OUTPUT_TOKENS),
            },
        },
    }


def _key() -> str:
    k = (os.getenv("OPENAI_API_KEY") or "").strip()
    if not k:
        raise RuntimeError("OPENAI_API_KEY is not set")
    return k


def negotiate(sdp_offer: str, timeout: float = 45.0) -> dict:
    """Trade the browser's SDP offer for an answer, and open the session.

    THE SERVER IS IN THE MIDDLE HERE AND IS NOT UNDER THE OTHER BACKEND, which
    is worth stating because it looks like a step backwards. gpt-realtime mints
    an ephemeral client secret and the browser negotiates directly. The live
    endpoint takes the session configuration and the SDP offer in ONE request,
    so the only way to keep the account key out of the browser is to make that
    request from here.

    What it costs is one extra hop on connect, once per drive, off the audio
    path entirely -- the media still flows browser-to-OpenAI. What it buys is
    the same thing every other key in this repo buys by going through the
    server: a browser that cannot spend the account.
    """
    r = httpx.post(
        LIVE_SESSIONS_URL,
        headers={"Authorization": f"Bearer {_key()}",
                 "Content-Type": "application/json"},
        json={"transport": {"type": "webrtc", "sdp": sdp_offer},
              "session": session_config()},
        timeout=timeout)
    if r.status_code >= 400:
        return {"ok": False, "status": r.status_code,
                "error": r.text[:400]}
    body = r.json()
    return {"ok": True,
            "sdp": (body.get("transport") or {}).get("sdp"),
            "session_id": (body.get("session") or {}).get("id"),
            "expires_at": (body.get("session") or {}).get("expires_at")}


def client_config() -> dict:
    """What the page needs to know, and nothing it could decide for itself.

    The same discipline as realtime.mint_client_secret: every policy here is
    read from config.py so the browser holds no second copy of a decision.
    """
    return {
        "backend": "gpt_live",
        "model": config.GPT_LIVE_MODEL,
        "voice": config.GPT_LIVE_VOICE,
        "backend_model": config.GPT_LIVE_BACKEND_MODEL,
        "delegation": config.GPT_LIVE_DELEGATION,
        "tools": [t["name"] for t in backend_tools()],
        "tool_schemas": backend_tools(),
        "conditional_tools": conditional_tools(),
        # WHICH GUARDS THE PAGE SHOULD RUN. Measured per backend in config.py;
        # sent rather than inferred, so turning one back on is an env var on
        # the server and not a new build of the browser.
        "guards": config.guards(),
        "speech_enabled": bool(config.REALTIME_SPEECH_ENABLED),
        "speech_channels": dict(config.REALTIME_SPEECH_CHANNELS),
        "speak_timeout_ms": int(config.REALTIME_SPEAK_TIMEOUT_MS),
        "speak_timeout_ms_by_channel": {
            ch: dict(by) for ch, by
            in config.REALTIME_SPEAK_TIMEOUT_MS_BY_CHANNEL.items()},
        # The floor this backend's slower deterministic path needs. The page
        # takes the larger of this and the channel's own budget.
        "speak_timeout_floor_ms": int(config.speak_timeout_floor_ms()),
        # ...AND PER CHANNEL, because headway's floor is bounded by the TTL on
        # its own arbiter item and nav's is not. The whole table travels rather
        # than the one number, exactly as speak_timeout_ms_by_channel does.
        "speak_timeout_floor_ms_by_channel": config.speak_timeout_floors(),
        "direct_speech_timeout_ms":
            int(config.REALTIME_DIRECT_SPEECH_TIMEOUT_MS),
        "verbatim_instruction": config.GPT_LIVE_VERBATIM_INSTRUCTION,
        # WHERE THIS VOICE'S CLIPS ARE. Sent rather than assumed: the page
        # builds three different clip URLs in three different files, and a
        # backend switch that updated two of them would be a drive with the
        # previous voice in the worst three lines.
        "clip_base": config.clip_base(),
        # THE ECHO GATE'S OWN NUMBERS, sent because the gate still runs under
        # this backend and its thresholds were measured for this cabin rather
        # than for this model. Identical to the ones the realtime session
        # carries, on purpose: a drive should be comparable across backends,
        # and a gate tuned twice is a gate nobody can reason about.
        "echo_tail_ms": int(config.REALTIME_ECHO_TAIL_MS),
        "echo_text_window_s": float(config.REALTIME_ECHO_TEXT_WINDOW_S),
        "echo_text_overlap": float(config.REALTIME_ECHO_TEXT_OVERLAP),
        "echo_text_min_words": int(config.REALTIME_ECHO_TEXT_MIN_WORDS),
        "look_answer_max_tokens": int(config.look_answer_max_tokens()),
    }


# ---------------------------------------------------------------------------
# DETERMINISTIC SPEECH, WHICH IS NOW A REQUEST
# ---------------------------------------------------------------------------

# THE TWO WAYS TO PUT WORDS IN HER MOUTH, AND ONLY ONE OF THEM IS VERBATIM.
#
# This was got wrong first and the measurement caught it, which is the only
# reason the right one is here. session.commentary.append is the obvious
# candidate -- it is the event whose whole job is "here is something to say" --
# and the documentation's description of it is exact: "information for the
# model to vocalize, WHICH IT MAY PARAPHRASE".
#
# It does. Measured on RIO's own deterministic lines
# (tools/live_verbatim_bench.py), commentary.append produced:
#
#   "In half a mile, turn right onto Ocean Avenue."
#     -> "Half a mile up, take a right on Ocean Avenue."
#   "In 300 feet, turn right onto Ocean Avenue."
#     -> "About 300 feet, take a right onto Ocean Avenue."
#   "I've lost the sensor on the rear right tire - that corner's dark to me
#    while we're moving."
#     -> "Hey- I'm blind on the rear right tire sensor right now, so if you..."
#
# 9 of 21 word-perfect overall, and 0 of 9 on the long lines. The second one is
# the one to look at hardest: "about 300 feet" is a HEDGE ADDED TO A DISTANCE
# the policy stated exactly, and the third drops "while we're moving", which is
# the qualifier that makes the sentence true.
#
# session.instructions.append is the other one, and the live-conversations
# guide gives it with exactly this use: "Immediately say the following
# disclosure exactly...". It is a DIRECTIVE rather than material, and it holds:
# the same three lines that failed 0/9 through commentary came back word for
# word through instructions.
#
# So deterministic speech goes through instructions.append and conversation
# goes through commentary.append, and the difference is not a preference.

def verbatim_event(text: str, event_id: str = "rio_line",
                   delegation_id=None) -> dict:
    """Make her say this line, exactly as written.

    `delegation_id` is None for anything RIO says on her OWN initiative -- a
    turn call, a health announcement, a headway line. None is not a missing
    value here; it is the documented way to say "this is not an answer to a
    delegation", and OMITTING the key is a missing_required_parameter error and
    a car that says nothing at all. Measured, because that cost an afternoon.

    STILL NOT A GUARANTEE, and the fallback chain still matters. The API
    documents no verbatim mechanism at all, so this is a directive that is
    obeyed rather than a contract that is enforced. The lines where being
    wrong is dangerous -- the red headway tier, the tire fast path, the
    imminent turn call -- do not come through here under any backend: they play
    a pre-rendered clip, with no network and no model in the path, and under
    this backend those clips are rendered in Gleam so the voice still matches.
    """
    line = str(text or "")
    return {"type": "session.instructions.append",
            "event_id": event_id,
            "delegation_id": delegation_id,
            "content": (f"Say this out loud right now, reproducing it EXACTLY "
                        f"as written, with no additions, no omissions, no "
                        f"rewording and no introduction: \"{line}\"")}


def commentary_event(text: str, event_id: str = "rio_line",
                     delegation_id=None) -> dict:
    """Hand her something to say IN HER OWN WORDS.

    The right event for a result she should relay conversationally, and the
    wrong one for anything a policy wrote. See verbatim_event above for the
    measurement that separates them.
    """
    return {"type": "session.commentary.append",
            "event_id": event_id,
            "delegation_id": delegation_id,
            "content": str(text or "")}


def tool_result_events(call_id: str, output) -> list:
    """A finished tool call, on its way back to the backend model.

    TWO EVENTS AND NOT ONE, which is the shape of the Responses delegation
    loop: the result is added to the backend's input, and then the backend is
    asked to carry on. Sending only the first leaves a model holding an answer
    it was never told to use, which presents as RIO going quiet after a tool
    call -- the failure this pair exists to make impossible.
    """
    payload = output if isinstance(output, str) else json.dumps(output)
    return [
        {"type": "response.item.create",
         "event_id": f"tool_result_{call_id}",
         "item": {"type": "function_call_output",
                  "call_id": call_id,
                  "output": payload}},
        {"type": "response.create", "event_id": f"continue_{call_id}"},
    ]


def parse_tool_call(event: dict):
    """Pull a function call out of a delegation envelope, or return None.

    Responses delegation nests the backend's own stream inside `response.event`
    envelopes, so a tool call is two levels down and arrives with the
    delegation it belongs to. The delegation_id is carried out with it because
    the reply has to name it.
    """
    if not isinstance(event, dict) or event.get("type") != "response.event":
        return None
    inner = event.get("event") or {}
    if inner.get("type") != "response.output_item.done":
        return None
    item = inner.get("item") or {}
    if item.get("type") not in ("function", "function_call"):
        return None
    return {"call_id": item.get("call_id"),
            "name": item.get("name"),
            "arguments": item.get("arguments") or "{}",
            "delegation_id": event.get("delegation_id")}


# ---------------------------------------------------------------------------
# RENDERING A CLIP IN GLEAM, WHICH TAKES A WHOLE SESSION
# ---------------------------------------------------------------------------
# THE PROBLEM THIS SOLVES, stated plainly because it nearly sank the backend:
# Gleam exists ONLY inside a live session. It is not a text-to-speech voice --
# /v1/audio/speech rejects it ("Supported values are: alloy, echo, fable,
# onyx, nova, shimmer, coral, verse, ballad, ash, sage, marin, cedar") -- and
# gpt-live-1 serves no endpoint but v1/live/sessions. So there is no request
# that turns a sentence into a Gleam audio file.
#
# Without one, the red headway tier and the tire fast path would keep playing
# clips in marin while the conversation happened in Gleam, and "one voice
# everywhere" would quietly become two. That is the outcome worth refusing.
#
# So a clip is rendered by OPENING A SESSION AND RECORDING HER. It is slow --
# a WebRTC negotiation and a few seconds of speech per line -- and it does not
# matter at all, because this runs on a workstation and the artifact is played
# from disk months later.
#
# WHAT MAKES IT SAFE is not the model's cooperation, which is not guaranteed:
# it is that the caller checks. The transcript comes back with the audio,
# render_alerts.py transcribes the finished MP3 with a DIFFERENT model, and a
# line that did not come out word for word is re-rendered rather than shipped.
# That discipline already existed for the old backend; this backend needs it
# more, and it is the reason a model documented as free to paraphrase can still
# be trusted to cut a safety clip.

_RENDER_INSTRUCTIONS = (
    "You are a speech renderer, not an assistant. Speak exactly the text you "
    "are given, word for word, once. Never add a greeting, a comment, an "
    "acknowledgement or a question. Never rephrase.")


def render_speech(text: str, timeout_s: float = 45.0) -> dict:
    """Speak one line in Gleam through a throwaway session; return the audio.

    Returns {ok, wav, transcript, ms, seconds} -- the same shape
    realtime.render_speech returns, so tools/render_alerts.py can treat the
    two backends as one thing with a switch in front of it.
    """
    line = (text or "").strip()
    if not line:
        return {"ok": False, "note": "no text"}
    try:
        return _run_render(line, timeout_s)
    except Exception as e:
        print(f"[live] render failed: {type(e).__name__}: {e}", flush=True)
        return {"ok": False, "note": f"{type(e).__name__}: {e}"}


def _run_render(line: str, timeout_s: float) -> dict:
    import asyncio
    import fractions
    import io
    import time
    import wave

    import numpy as np
    import av
    from aiortc import RTCPeerConnection, RTCSessionDescription
    from aiortc.mediastreams import MediaStreamTrack

    rate, frame = 48000, 960

    class _Quiet(MediaStreamTrack):
        """A microphone with nobody in front of it.

        A track is still required: the session negotiates audio in both
        directions and an offer with no outbound track gets an answer with no
        inbound one.
        """

        kind = "audio"

        def __init__(self):
            super().__init__()
            self._pts = 0
            self.t0 = None

        async def recv(self):
            if self.t0 is None:
                self.t0 = time.time()
            target = self.t0 + self._pts / rate
            now = time.time()
            if target > now:
                await asyncio.sleep(target - now)
            f = av.AudioFrame.from_ndarray(
                np.zeros((1, frame), np.int16), format="s16", layout="mono")
            f.sample_rate = rate
            f.pts = self._pts
            f.time_base = fractions.Fraction(1, rate)
            self._pts += frame
            return f

    async def go() -> dict:
        t0 = time.time()
        pc = RTCPeerConnection()
        pc.addTrack(_Quiet())
        # A PRE-ROLL, because starting the recording at the first LOUD frame
        # cuts the attack off the first word. Measured, by the clip verifier
        # that exists for exactly this: "Pull over when it's safe" was
        # transcribed "Paying is unsafe", "Merge right." as "Walk right.", and
        # the three shortest clips came back as silence -- every one of them a
        # plosive or a fricative that had already happened by the time the
        # frame crossed the threshold.
        #
        # So the last few frames before the threshold are kept and prepended.
        # Twelve 20 ms frames is a quarter of a second: longer than any onset
        # and shorter than the gap before she starts.
        from collections import deque
        state = {"pcm": bytearray(), "txt": "", "last": 0.0, "first": None,
                 "pre": deque(maxlen=12)}

        @pc.on("track")
        def _(track):
            async def pump():
                while True:
                    try:
                        fr = await track.recv()
                    except Exception:
                        return
                    a = fr.to_ndarray().reshape(-1).astype(np.int16)
                    loud = np.abs(a).max() > 150
                    if loud:
                        state["last"] = time.time()
                        if state["first"] is None:
                            state["first"] = time.time()
                            for old_frame in state["pre"]:
                                state["pcm"] += old_frame
                    if state["first"] is None:
                        state["pre"].append(a.tobytes())
                    else:
                        state["pcm"] += a.tobytes()
            asyncio.ensure_future(pump())

        dc = pc.createDataChannel("oai-events")

        @dc.on("message")
        def _(m):
            try:
                e = json.loads(m)
            except Exception:
                return
            if e.get("type") == "session.output_transcript.delta":
                state["txt"] += e.get("delta", "")

        await pc.setLocalDescription(await pc.createOffer())
        sess = {"model": config.GPT_LIVE_MODEL,
                "instructions": _RENDER_INSTRUCTIONS,
                "audio": {"output": {"voice": config.GPT_LIVE_VOICE}},
                "delegation": {"type": "client"}}
        r = httpx.post(LIVE_SESSIONS_URL,
                       headers={"Authorization": f"Bearer {_key()}",
                                "Content-Type": "application/json"},
                       json={"transport": {"type": "webrtc",
                                           "sdp": pc.localDescription.sdp},
                             "session": sess},
                       timeout=timeout_s)
        if r.status_code >= 400:
            return {"ok": False, "note": f"mint {r.status_code}: "
                                         f"{r.text[:200]}"}
        await pc.setRemoteDescription(RTCSessionDescription(
            sdp=r.json()["transport"]["sdp"], type="answer"))
        for _ in range(120):
            if dc.readyState == "open":
                break
            await asyncio.sleep(0.05)
        # SHE GREETS, AND THE GREETING MUST NOT END UP IN THE CLIP.
        #
        # gpt-live-1 opens a session by saying hello -- "Hey. What's up." --
        # whatever the instructions say, because the session is a conversation
        # and that is how one starts. A recorder that begins at the first loud
        # frame therefore begins on the greeting, and what lands on disk is the
        # greeting with the line behind it. Measured: `tire_critical` came out
        # at 10.8 seconds for a six-second sentence, and an independent
        # transcription heard "Hey, Ava, when it's safe..." where the line says
        # "Pull over when it's safe...". The instruction to pull over -- the
        # whole action the warning exists to request -- was gone, and the
        # model's own transcript still called the render correct.
        #
        # So: let the greeting happen, wait for it to finish, and only then ask
        # for the line and start listening.
        settle = time.time()
        while time.time() - settle < 12.0:
            await asyncio.sleep(0.15)
            if state["last"] and time.time() - state["last"] > 1.0:
                break
            if not state["last"] and time.time() - settle > 3.0:
                break        # she never greeted; nothing to wait out
        # Everything heard so far was hers and was not asked for.
        state["pcm"] = bytearray()
        state["pre"].clear()
        state["first"] = None
        state["last"] = 0.0
        state["txt"] = ""
        asked = time.time()
        dc.send(json.dumps(verbatim_event(line, event_id="render")))
        # TWO DEADLINES, because "she never started" and "she has finished" are
        # different questions and one loop answering both returns an empty clip
        # for the first. A session that is asked and stays silent has failed,
        # and it has to say so rather than hand back zero bytes of audio.
        started = False
        while time.time() - t0 < timeout_s:
            await asyncio.sleep(0.1)
            if state["last"]:
                started = True
                # Quiet for long enough that the sentence is over rather than
                # between two of its own words.
                if time.time() - state["last"] > 0.8:
                    break
            elif time.time() - asked > 12.0:
                break
        await pc.close()
        if not started:
            return {"ok": False, "note": "asked and stayed silent"}
        pcm = bytes(state["pcm"])
        if not pcm:
            return {"ok": False, "note": "no audio"}
        # Trim the trailing quiet the stop test just waited through, so the
        # clip is the sentence and not the sentence plus a second of room.
        arr = np.frombuffer(pcm, "<i2")
        loud = np.where(np.abs(arr) > 150)[0]
        if len(loud):
            arr = arr[:min(len(arr), loud[-1] + int(rate * 0.18))]
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(rate)
            w.writeframes(arr.astype(np.int16).tobytes())
        return {"ok": True, "wav": buf.getvalue(),
                "transcript": state["txt"].strip(),
                "ms": round((time.time() - t0) * 1000, 1),
                "seconds": round(len(arr) / float(rate), 2)}

    return asyncio.run(go())
