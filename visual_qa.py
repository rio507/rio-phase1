"""Visual conversation — the turn where RIO and the driver look at the same thing.

Design ref: docs/visual_qa.md. This module is the orchestrator: it owns the
order of the stages, the conversation's visual memory, and the one call that
actually puts a photograph in front of GPT-5.5.

THE POINT
---------
Before this, "what do you see?" was answered by reading a Qwen caption out
loud, and it sounded like it: *"A silver vehicle is visible in the left lane."*
The model doing the talking had never seen the road. It was rewriting a
sentence about a picture.

So the fix is not a better caption. It is that the model doing the talking gets
the actual frame, plus a crop of the thing being asked about, plus the
measurements the local stack already has — and looks for itself.

WHAT EACH PART CONTRIBUTES
--------------------------
    RF-DETR        what is there, where, tracked with a stable id   (per frame)
    Depth Anything how far away it is                               (per frame)
    UFLDv2         where the lane is, so "left" means a lane        (per frame)
    Qwen3-VL       colour, body style, and which one they meant     (on demand)
    GPT-5.5        looks at the image and says something human      (per question)

The division is not decorative. Everything on the left of that list is a
measurement that a deterministic safety system also depends on, and none of it
is allowed to become an opinion. Everything GPT-5.5 says is an opinion, and
none of it is allowed to become a warning — see the last line of
VISUAL_SYSTEM_PROMPT, and note that this module never touches the band, the
policy, or the arbiter's safety priority.

WHAT IS NEVER SENT
------------------
Video. Not a clip, not a burst, not "the last few frames". Local perception runs
continuously and GPT-5.5 is called once per question with at most two images:
the best recent full frame, and a crop. That is the implementation rule from
the spec and it is also the only version of this that is affordable.
"""
import base64
import json
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

import config
import enrich as enrich_mod
import framebuf
import frameselect
import resolve as resolve_mod
import router
import scene as scene_mod
from rio_prompts import CLARIFY_SYSTEM_PROMPT, VISUAL_SYSTEM_PROMPT


# ---------------------------------------------------------------------------
# Chat adapter — the one swap point for the talking model
# ---------------------------------------------------------------------------
class ChatAdapter:
    """What the visual turn needs from a conversational multimodal model."""

    name = "none"

    def stream(self, system: str, messages: list):
        """Yield text deltas. `messages` uses OpenAI content-part shape."""
        raise NotImplementedError


class OpenAIChatAdapter(ChatAdapter):
    """GPT-5.5 through the OpenAI client the rest of the app already uses."""

    name = config.OPENAI_VISUAL_MODEL

    def __init__(self):
        from openai import OpenAI

        self.client = OpenAI()

    def stream(self, system: str, messages: list):
        stream = self.client.chat.completions.create(
            model=config.OPENAI_VISUAL_MODEL,
            messages=[{"role": "system", "content": system}] + messages,
            max_completion_tokens=config.OPENAI_VISUAL_MAX_TOKENS,
            reasoning_effort=config.OPENAI_VISUAL_REASONING_EFFORT,
            stream=True,
        )
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta.content or ""
            if delta:
                yield delta


class QwenChatAdapter(ChatAdapter):
    """Qwen3-VL, resident on this box, answering from the crop directly.

    THE HOP THIS REMOVES. An object question already produces the thing that
    answers it: a close crop of the object, chosen by the frame selector out of
    the ring. The crop was then sent over the network to a reasoning model,
    which looked at it and wrote a sentence — a remote round trip, measured at
    ~2 s, to describe a picture that a resident 8B multimodal model can describe
    in a few hundred milliseconds. And the realtime model rephrases whatever
    comes back anyway, so the careful prose was being paraphrased before the
    driver ever heard it.

    WHAT IT IS AND IS NOT GIVEN. The question and the picture. NOT the
    perception grounding packet: that is a page of JSON written for a large
    model that can hold it and the question at once, and an 8B model handed both
    starts answering the JSON. The detector's own label rides along in one short
    line, because "the thing in this crop is a car" is worth stating and is
    something the crop alone can be wrong about.

    The honesty rules survive the swap, because they were never the model's
    idea: a crop that cannot carry a claim is refused by the pipeline before
    this, and what reaches here is a question that the picture can answer.
    """

    name = "qwen3-vl-8b"

    SYSTEM = ("You are the eyes of a car assistant. Answer the driver's "
              "question about the picture in ONE or TWO short sentences of "
              "plain spoken English.\n"
              "ANSWER THE QUESTION THAT WAS ASKED. If they ask what a car is, "
              "name it as precisely as the picture allows — make and model if "
              "the shape or the badge tells you, otherwise the type and "
              "colour ('a white saloon'). 'A white car' is not an answer to "
              "'what kind of car is that'.\n"
              "Say only what you can actually see, and if the picture does not "
              "settle it say that plainly and briefly. No lists, no markdown, "
              "no preamble, no describing the image as an image.")

    def _split(self, messages):
        """The last user turn -> (question_text, [PIL images]).

        Only the last turn: the history is text and this model is being asked
        one visual question, not to hold a conversation. `visual_qa` keeps the
        conversation's memory itself.
        """
        import base64
        import io as _io

        from PIL import Image

        text_bits, images = [], []
        last = None
        for m in messages:
            if m.get("role") == "user":
                last = m
        parts = (last or {}).get("content")
        if isinstance(parts, str):
            return parts, []
        for part in (parts or []):
            if part.get("type") == "text":
                t = part.get("text", "")
                # The grounding packet is deliberately dropped -- see the class
                # docstring. Everything else in the turn is a short human line
                # ("Close crop of the object being asked about:") worth keeping.
                if t.startswith("PERCEPTION GROUNDING"):
                    continue
                text_bits.append(t)
            elif part.get("type") == "image_url":
                url = (part.get("image_url") or {}).get("url", "")
                if "," in url:
                    try:
                        raw = base64.b64decode(url.split(",", 1)[1])
                        images.append(Image.open(_io.BytesIO(raw)).convert("RGB"))
                    except Exception:
                        pass
        return "\n".join(text_bits).strip(), images

    def stream(self, system: str, messages: list):
        import vision

        question, images = self._split(messages)
        if not images:
            # Nothing to look at. The caller's honesty path handles an empty
            # answer; inventing one here would be the whole failure this
            # subsystem exists to prevent.
            return
        # The CROP is the last image attached and the one the question is about
        # (_build_messages appends the full frame first, then the crop), so when
        # both are present the crop goes last here too and Qwen sees it nearest
        # the question.
        images = images[-config.VISUAL_QWEN_MAX_IMAGES:]

        processor, model, lock = vision.get_handles()
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": question})
        msgs = [{"role": "system", "content": [{"type": "text", "text": self.SYSTEM}]},
                {"role": "user", "content": content}]
        with lock:
            inputs = processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt").to(model.device)
            out = model.generate(**inputs,
                                 max_new_tokens=int(config.VISUAL_QWEN_MAX_TOKENS),
                                 do_sample=False)
            text = processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True)[0].strip()
        if text:
            # One delta. This model is not streamed: it is local and short, and
            # a token-by-token trickle from it would only make the caller's
            # first-token timing look better than the driver's wait actually is.
            yield text


_chat_adapter = None
_qwen_adapter = None
_chat_lock = threading.Lock()


def get_chat_adapter(prefer: str = None) -> ChatAdapter:
    """The model that writes the answer.

    `prefer="qwen"` asks for the local one. It is a REQUEST, not a switch: if
    the local model cannot be had, this returns the remote one rather than
    failing the turn, because a slower answer is better than none.
    """
    global _chat_adapter, _qwen_adapter
    if prefer == "qwen" and config.VISION_ENABLED:
        with _chat_lock:
            if _qwen_adapter is None:
                _qwen_adapter = QwenChatAdapter()
            return _qwen_adapter
    with _chat_lock:
        if _chat_adapter is None:
            _chat_adapter = OpenAIChatAdapter()
        return _chat_adapter


def set_chat_adapter(adapter: ChatAdapter) -> None:
    global _chat_adapter
    with _chat_lock:
        _chat_adapter = adapter


# ---------------------------------------------------------------------------
# Visual memory
# ---------------------------------------------------------------------------
@dataclass
class VisualReferent:
    """The object the conversation is currently about.

    This is what makes "what year is it?" a sentence rather than a riddle. It
    holds the track id AND the last good crop, so a follow-up survives the
    vehicle leaving the frame — which is the whole of acceptance test 5, and is
    why the crop is stored now even though the behaviour that needs it lands in
    Phase B.
    """
    track_id: str
    label: str
    fine_label: Optional[str] = None
    attributes: dict = field(default_factory=dict)
    description: str = ""
    last_best_frame_id: Optional[str] = None
    last_crop_jpeg: Optional[bytes] = field(default=None, repr=False)
    last_crop_info: dict = field(default_factory=dict)
    last_crop_path: Optional[str] = None
    established_at: float = 0.0
    last_used_at: float = 0.0
    question: str = ""
    # Where it was and what it was doing when last seen. Kept on the referent
    # rather than looked up on demand because the point of a referent is to
    # outlive the object's presence in the buffer: once the car is gone, this
    # is the only record of where it was, and it is what a review reads to see
    # which vehicle an answer was actually about.
    last_position: Optional[str] = None
    last_depth_m: Optional[float] = None
    last_motion: Optional[str] = None
    # Wall clock of the last frame this object was actually seen in. What makes
    # "it's out of sight now" a statement with a number behind it rather than a
    # guess, and what stops a referent from silently going stale.
    last_seen_at: float = 0.0

    def stale(self, ttl_s: float = None) -> bool:
        ttl = config.REFERENT_TTL_S if ttl_s is None else ttl_s
        return (time.time() - self.last_used_at) > ttl

    def to_log(self) -> dict:
        return {
            "track_id": self.track_id,
            "label": self.label,
            "fine_label": self.fine_label,
            "attributes": dict(self.attributes),
            "position": self.last_position,
            "depth_meters": self.last_depth_m,
            "motion": self.last_motion,
            "last_best_frame_id": self.last_best_frame_id,
            "last_crop_path": self.last_crop_path,
            "age_s": round(time.time() - self.established_at, 1),
        }

    def observe(self, obj) -> None:
        """Refresh what is known about the object from the current graph."""
        if obj is None:
            return
        self.last_position = obj.position
        self.last_depth_m = obj.depth_m
        self.last_motion = obj.motion
        self.last_seen_at = time.time()
        if obj.fine_label:
            self.fine_label = obj.fine_label
        if obj.attributes:
            self.attributes.update(obj.attributes)

    def unseen_for(self) -> Optional[float]:
        return None if not self.last_seen_at else (time.time() - self.last_seen_at)


@dataclass
class PendingClarification:
    """A "which one?" RIO has asked and is waiting on.

    Holds the ORIGINAL question, because that is what has to be answered once
    the driver picks: they said "the black one", and what they actually want to
    know is still "what kind of car is that".
    """
    question: str
    candidates: list                      # track ids, in the order offered
    descriptions: dict = field(default_factory=dict)
    asked_at: float = 0.0
    text: str = ""
    frame_id: Optional[str] = None

    def stale(self, ttl_s: float = None) -> bool:
        ttl = config.CLARIFY_TTL_S if ttl_s is None else ttl_s
        return (time.time() - self.asked_at) > ttl

    def to_log(self) -> dict:
        return {
            "original_question": self.question,
            "candidates": list(self.candidates),
            "descriptions": dict(self.descriptions),
            "asked": self.text,
            "age_s": round(time.time() - self.asked_at, 1),
        }


class VisualSession:
    """Per-drive conversation state for the visual path."""

    def __init__(self, key: str):
        self.key = key
        self.referent: Optional[VisualReferent] = None
        self.pending: Optional[PendingClarification] = None
        self.enrichment = enrich_mod.EnrichmentCache()
        self.turns = []                 # [{"q":..., "a":..., "t":...}]
        self.lock = threading.Lock()
        self.created = time.time()

    def active_referent(self) -> Optional[VisualReferent]:
        if self.referent is None or self.referent.stale():
            return None
        return self.referent

    def pending_clarification(self) -> Optional[PendingClarification]:
        """An outstanding "which one?", if it has not lapsed.

        A lapsed question is dropped rather than kept: the next thing the driver
        says a minute later is a new utterance, and treating it as an answer to
        something they have forgotten being asked is worse than not having
        asked.
        """
        if self.pending is None:
            return None
        if self.pending.stale():
            self.pending = None
            return None
        return self.pending

    def clear_pending(self) -> None:
        self.pending = None

    def remember(self, ref: VisualReferent) -> None:
        self.referent = ref

    def add_turn(self, question: str, answer: str) -> None:
        self.turns.append({"q": question, "a": answer, "t": time.time()})
        if len(self.turns) > 24:
            self.turns[:] = self.turns[-24:]

    def recent_turns(self, n: int = None) -> list:
        n = config.VISUAL_HISTORY_TURNS if n is None else n
        return self.turns[-n:]

    def reset(self) -> None:
        self.referent = None
        self.pending = None
        self.enrichment.reset()
        self.turns.clear()


_sessions = {}
_sessions_lock = threading.Lock()


def get_session(key: str) -> VisualSession:
    k = key or "default"
    with _sessions_lock:
        s = _sessions.get(k)
        if s is None:
            s = VisualSession(k)
            _sessions[k] = s
        return s


def drop_session(key: str) -> bool:
    with _sessions_lock:
        s = _sessions.pop(key or "default", None)
    if s is not None:
        s.reset()
        return True
    return False


# ---------------------------------------------------------------------------
# The turn
# ---------------------------------------------------------------------------
def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def _nav_context() -> Optional[dict]:
    """Where the car is headed, if a route is set. Never "what's coming next"."""
    try:
        from navigation import service as navservice

        route = navservice.latest_route()
        if not route:
            return None
        return {"destination": route.destination.display_name
                               or route.destination.formatted_address,
                "note": "route destination only; progress along the route is not known here"}
    except Exception:
        return None


class VisualAnswer:
    """One visual turn, from a question to a spoken reply.

    Constructed, then `prepare()`d (everything up to the model call), then
    streamed. Split that way because /talk streams TTS off the token stream and
    still needs every preceding stage's latency recorded when the stream ends.
    """

    def __init__(self, session_id: str, question: str, route: dict = None):
        self.session_id = session_id
        self.question = question or ""
        self.route = route
        self.session = get_session(session_id)
        self.reply = ""
        self.timing = {}
        self.meta = {
            "question": self.question,
            "session_id": session_id,
            "phase": "A",
        }
        self._messages = None
        self._system = VISUAL_SYSTEM_PROMPT
        self._images = []
        self._frame = None
        self._crop = None
        self._crop_info = {}
        self._graph = None
        self._resolution = None
        self._referent = None
        # Phase B: set when this turn is RIO asking which one, rather than
        # answering. The two are different enough — different prompt, different
        # memory, no referent committed — that the flag is worth its weight.
        self._asking = None            # PendingClarification being composed
        self._compare = []             # [(referent, crop, crop_info)] for a comparison
        self._t_start = time.perf_counter()

    # -- stages -------------------------------------------------------------
    def _stage(self, name: str, t0: float) -> float:
        now = time.perf_counter()
        self.timing[name] = round((now - t0) * 1000, 1)
        return now

    def prepare(self) -> None:
        t = self._t_start

        # 1. Route the request, unless the caller already did.
        if self.route is None:
            self.route = router.classify(
                self.question,
                has_referent=self.session.active_referent() is not None,
                pending_clarification=(
                    self.session.pending_clarification() is not None))
        self.meta["route"] = self.route
        t = self._stage("route", t)

        rtype = self.route["request_type"]
        ring = framebuf.peek_ring(self.session_id)
        frames = ring.frames() if ring else []
        self.meta["ring"] = ring.stats() if ring else {"frames": 0}

        if not frames:
            # Nothing to look at. RIO says so rather than answering from a
            # remembered caption, which is the failure mode this whole path
            # exists to remove.
            self.meta["visual_unavailable"] = "no_frames"
            self._build_messages(unavailable=True)
            self._stage("prepare_total", self._t_start)
            return

        # 2. Pick a frame to reason about, and build the graph on it.
        ref_wall = time.time()
        if rtype == router.SCENE:
            frame, sel = frameselect.best_scene_frame(ring, ref_wall)
        else:
            # Reference resolution needs a graph before a per-object frame can
            # be chosen, so the newest usable frame is the working surface for
            # resolution only. The frame that ends up in front of the model is
            # chosen AFTER the track is known — see step 4.
            frame, sel = frameselect.best_scene_frame(ring, ref_wall)
        self._frame = frame
        self.meta["frame_selection"] = sel
        self._graph = self._graph_for(frame, ring)
        t = self._stage("select_frame", t)

        # 3. A comparison is two objects, and is its own path from here.
        if rtype == router.COMPARISON:
            self._prepare_comparison(ring, ref_wall)
            self._stage("prepare_total", self._t_start)
            return

        # 4. Which object is this about?
        self._referent = self._establish_referent(ring, rtype)
        t = self._stage("resolve", t)

        # 5. Nothing separated the candidates: ask, rather than guess.
        if self._should_clarify(rtype):
            self._prepare_clarification(ring)
            self._stage("clarify", t)
            self._stage("prepare_total", self._t_start)
            return

        # 6. The clearest recent frame containing THAT object, and a crop of it.
        if self._referent is not None:
            self._pick_object_frame(ring, ref_wall)
        t = self._stage("crop", t)

        # 7. Attributes for whatever is about to be described.
        self._enrich(ring)
        t = self._stage("enrich", t)

        self._build_messages()
        self._stage("prepare_total", self._t_start)

    # -- clarification ------------------------------------------------------
    def _should_clarify(self, rtype) -> bool:
        """Is the honest move to ask which one, rather than to answer?

        Four conditions, and all of them matter:

          the reference was genuinely ambiguous  -- not merely low-scoring
          there is more than one thing it could be
          this is a question ABOUT an object    -- a scene question has no
                                                   referent to be unsure about,
                                                   and a follow-up already knows
                                                   what it is discussing
          RIO is not already waiting on an answer -- asking twice in a row is
                                                    worse than guessing once
        """
        res = self._resolution
        if res is None or not res.ambiguous:
            return False
        # OBJECT only. Reading text is deliberately excluded: nothing in the
        # detector's vocabulary is a sign, so a "what does that say" reference
        # NEVER matches a tracked object and is therefore always ambiguous --
        # and asking "the white saloon, or the black one?" about a road sign is
        # not a clarifying question, it is a non sequitur. Observed exactly
        # once, which was enough.
        if rtype != router.OBJECT:
            return False
        if self.session.pending_clarification() is not None:
            self.meta["clarify_suppressed"] = "already_asked"
            return False
        candidates = [c for c in ([res.track_id] + list(res.alternatives)) if c]
        return len(candidates) >= 2

    def _prepare_clarification(self, ring) -> None:
        """Compose the "which one?" turn. Nothing is committed as the referent."""
        res = self._resolution
        offered = [c for c in ([res.track_id] + list(res.alternatives)) if c]
        offered = offered[:config.CLARIFY_MAX_CANDIDATES]

        # The candidates have to be told apart by how they LOOK, so this is
        # where enrichment earns its latency: colour is usually the only thing
        # that separates two saloons in adjacent lanes.
        enrich_mod.enrich_objects(self._frame, offered, self.session.enrichment,
                                  max_objects=len(offered))
        self._graph = self._graph_for(self._frame, ring)

        descriptions = {}
        for tid in offered:
            obj = self._graph.by_track(tid) if self._graph else None
            if obj is not None:
                descriptions[tid] = resolve_mod.describe(obj)
        # Anything that has since left the graph cannot be offered: RIO would be
        # asking about a car that is no longer there.
        offered = [t for t in offered if t in descriptions]
        if len(offered) < 2:
            self.meta["clarify_abandoned"] = "candidates_gone"
            self._build_messages()
            return

        self._asking = PendingClarification(
            question=self.question, candidates=offered, descriptions=descriptions,
            asked_at=time.time(),
            frame_id=self._frame.frame_id if self._frame else None)
        self.meta["clarification"] = {"offered": offered,
                                      "descriptions": descriptions}
        self._system = CLARIFY_SYSTEM_PROMPT
        self._build_clarify_messages(offered, descriptions)

    def _build_clarify_messages(self, offered, descriptions) -> None:
        listing = "\n".join(f"  - {descriptions[t]}" for t in offered)
        parts = [{
            "type": "text",
            "text": (f"The driver asked: {self.question}\n\n"
                     f"It could be any of these:\n{listing}\n\n"
                     "Ask which one they mean, in one short question."),
        }]
        if self._frame is not None:
            parts.append({"type": "image_url", "image_url": {
                "url": _data_url(self._frame.jpeg),
                "detail": config.VISUAL_FRAME_DETAIL}})
            self._images.append(("frame", len(self._frame.jpeg)))
        self._messages = [{"role": "user", "content": parts}]
        self.meta["request"] = {
            "model": config.OPENAI_VISUAL_MODEL,
            "images": [{"kind": k, "bytes": n} for k, n in self._images],
            "history_turns": 0,
            "grounding_bytes": len(listing),
            "reasoning_effort": config.OPENAI_VISUAL_REASONING_EFFORT,
            "purpose": "clarification",
        }

    # -- comparison ---------------------------------------------------------
    def _prepare_comparison(self, ring, ref_wall) -> None:
        """Two objects, two crops, one frame.

        Falls back to a plain object question when the driver named only one
        thing ("which one is faster?" names nothing) -- the alternative is
        inventing the second half of the comparison.
        """
        phrases = resolve_mod.split_comparison(self.question)
        self.meta["comparison"] = {"phrases": phrases}
        if not phrases or self._graph is None:
            self.meta["comparison"]["fell_back"] = "no_two_references"
            self._referent = self._establish_referent(ring, router.OBJECT)
            if self._referent is not None:
                self._pick_object_frame(ring, ref_wall)
            self._enrich(ring)
            self._build_messages()
            return

        picked = []
        used = set()
        for phrase in phrases[:config.COMPARE_MAX_OBJECTS]:
            res = resolve_mod.resolve(phrase, phrase, self._graph, self._frame,
                                      self.session.enrichment)
            if res.track_id is None or res.track_id in used:
                # Second-choice fallback: two references that resolve to the
                # same object mean one of them was matched on weaker evidence.
                alt = next((a for a in res.alternatives if a not in used), None)
                if alt is None:
                    continue
                res.track_id, res.candidate_id = alt, scene_mod.candidate_id_of(alt)
                res.info["reassigned_from_duplicate"] = True
            used.add(res.track_id)
            picked.append((phrase, res))

        self.meta["comparison"]["resolved"] = [
            {"phrase": p, **r.to_log()} for p, r in picked]
        if len(picked) < 2:
            self.meta["comparison"]["fell_back"] = "could_not_resolve_both"
            if picked:
                obj = self._graph.by_track(picked[0][1].track_id)
                if obj is not None:
                    self._referent = VisualReferent(
                        track_id=obj.track_id, label=obj.label,
                        fine_label=obj.fine_label, attributes=dict(obj.attributes),
                        established_at=time.time(), last_used_at=time.time(),
                        question=self.question)
                    self._referent.observe(obj)
                    self._pick_object_frame(ring, ref_wall)
            self._enrich(ring)
            self._build_messages()
            return

        # Enrich both, then crop both from the frame they are both in — the
        # comparison has to be of the same moment or it is not a comparison.
        enrich_mod.enrich_objects(self._frame,
                                  [r.track_id for _p, r in picked],
                                  self.session.enrichment,
                                  max_objects=len(picked))
        self._graph = self._graph_for(self._frame, ring)
        for _phrase, res in picked:
            obj = self._graph.by_track(res.track_id)
            frame_obj = (self._frame.object_by_id(res.candidate_id)
                         if self._frame else None)
            if obj is None or frame_obj is None:
                continue
            try:
                crop, info = scene_mod.crop_jpeg(self._frame.jpeg, frame_obj["box"])
            except Exception as e:
                print(f"[visual_qa] comparison crop failed: {e}", flush=True)
                continue
            self._compare.append((obj, crop, info))
        # The first one becomes the active referent, so "and how old is it?"
        # afterwards has something to attach to.
        if self._compare:
            obj = self._compare[0][0]
            ref = VisualReferent(
                track_id=obj.track_id, label=obj.label, fine_label=obj.fine_label,
                attributes=dict(obj.attributes), established_at=time.time(),
                last_used_at=time.time(), question=self.question)
            ref.observe(obj)
            self._referent = ref
        self._build_messages()

    def _graph_for(self, frame, ring):
        if frame is None:
            return None
        return scene_mod.build(
            frame.objects, frame.ego, {"w": frame.w, "h": frame.h},
            t=frame.t, wall_t=frame.wall_t, frame_id=frame.frame_id,
            tracker=ring.tracker, attributes=self.session.enrichment.all())

    def _establish_referent(self, ring, rtype) -> Optional[VisualReferent]:
        """-> the referent this turn is about, or None for a scene question."""
        active = self.session.active_referent()

        if rtype == router.SCENE:
            return None

        # The driver has just answered "which one?". Resolve within exactly the
        # set RIO offered — nothing else is a valid answer to that question —
        # and swap the original question back in, because "the black one" is
        # not what they want to know, it is which thing they want to know it
        # about.
        if rtype == router.CLARIFY_RESPONSE:
            pending = self.session.pending_clarification()
            if pending is not None:
                res = resolve_mod.resolve_among(
                    self.question, self._graph, pending.candidates,
                    self._frame, self.session.enrichment)
                self._resolution = res
                self.meta["resolution"] = res.to_log()
                self.meta["answered_clarification"] = pending.to_log()
                self.session.clear_pending()
                # The turn is now the ORIGINAL question, asked about the object
                # they picked.
                self.meta["original_question"] = pending.question
                self.meta["selection"] = self.question
                self.question = pending.question
                if res.track_id is not None and self._graph is not None:
                    obj = self._graph.by_track(res.track_id)
                    if obj is not None:
                        self.meta["referent_source"] = "clarified"
                        ref = VisualReferent(
                            track_id=obj.track_id, label=obj.label,
                            fine_label=obj.fine_label,
                            attributes=dict(obj.attributes),
                            established_at=time.time(), last_used_at=time.time(),
                            question=pending.question)
                        ref.observe(obj)
                        return ref
                self.meta["referent_source"] = "clarify_unresolved"
                return active

        # A follow-up continues the object already under discussion, and does
        # NOT re-identify: that is the whole point of holding a referent, and
        # re-resolving "what year is it?" from scratch would silently switch
        # cars the moment a nearer one appeared.
        if rtype == router.FOLLOW_UP and active is not None:
            self.meta["referent_source"] = "active"
            return active

        if self._graph is None or not self._graph.objects:
            if active is not None:
                self.meta["referent_source"] = "active_no_objects"
                return active
            self.meta["referent_source"] = "none"
            return None

        res = resolve_mod.resolve(
            self.question, self.route.get("object_reference"),
            self._graph, self._frame, self.session.enrichment)
        self._resolution = res
        self.meta["resolution"] = res.to_log()

        if res.track_id is None:
            if active is not None:
                self.meta["referent_source"] = "active_unresolved"
                return active
            self.meta["referent_source"] = "unresolved"
            return None

        obj = self._graph.by_track(res.track_id)
        if obj is None:
            self.meta["referent_source"] = "unresolved"
            return None

        # A text question that landed on a vehicle by weak scoring has not
        # found the sign; it has found the nearest car. Cropping it would put a
        # close-up of a boot lid in front of the model and invite it to read
        # words off it. The wide frame at high detail is the honest input.
        if rtype == router.READ_TEXT and res.ambiguous:
            self.meta["referent_source"] = "text_no_tracked_object"
            return None

        # A resolved object that is the SAME track as the active referent keeps
        # the referent's stored crop history rather than starting over.
        if active is not None and active.track_id == res.track_id:
            self.meta["referent_source"] = "active_reconfirmed"
            active.observe(obj)
            return active

        self.meta["referent_source"] = "resolved"
        ref = VisualReferent(
            track_id=obj.track_id, label=obj.label, fine_label=obj.fine_label,
            attributes=dict(obj.attributes),
            established_at=time.time(), last_used_at=time.time(),
            question=self.question)
        ref.observe(obj)
        return ref

    def _pick_object_frame(self, ring, ref_wall) -> None:
        """The best frame containing the referent, and the crop from it.

        Falls back to the referent's stored crop when the object is no longer in
        the ring. In Phase A that is a graceful degradation and it is reported;
        the behaviour that talks about it out loud ("it's out of sight now") is
        Phase B.
        """
        ref = self._referent
        cid = scene_mod.candidate_id_of(ref.track_id)
        frame, sel = (None, {"reason": "no_candidate_id"})
        if cid is not None:
            frame, sel = frameselect.best_frame_for(ring, cid, ref_wall)
        self.meta["object_frame_selection"] = sel

        if frame is None:
            # The object has left the buffer: overtaken, turned off, or simply
            # too far back to still be detected. This is acceptance test 5, and
            # the rule it enforces is that RIO must NOT quietly start talking
            # about a different car. The referent is kept, the last good crop is
            # what the answer is based on, and the grounding says plainly that
            # the vehicle is no longer in view.
            self.meta["referent_visible"] = False
            self.meta["referent_unseen_s"] = (
                None if ref.unseen_for() is None else round(ref.unseen_for(), 1))
            if ref.last_crop_jpeg:
                self._crop = ref.last_crop_jpeg
                self._crop_info = dict(ref.last_crop_info)
                self._crop_info["from_memory"] = True
                self.meta["crop_source"] = "referent_memory"
            else:
                self.meta["crop_source"] = "none"
            return

        self.meta["referent_visible"] = True
        # The frame the model reasons about becomes the one the object is
        # clearest in — the crop and the full frame must show the same moment,
        # or "the car on the left" in the crop is not the car on the left in
        # the scene.
        if frame.frame_id != (self._frame.frame_id if self._frame else None):
            self._frame = frame
            self._graph = self._graph_for(frame, ring)
        if self._graph is not None:
            ref.observe(self._graph.by_track(ref.track_id))
        obj = frame.object_by_id(cid)
        if obj is None:
            self.meta["crop_source"] = "none"
            return
        try:
            crop, info = scene_mod.crop_jpeg(frame.jpeg, obj["box"])
        except Exception as e:
            print(f"[visual_qa] crop failed: {type(e).__name__}: {e}", flush=True)
            self.meta["crop_source"] = "failed"
            return

        # Only replace a stored crop with a BETTER one. Acceptance test 3 asks
        # for "the same or a better crop" on a follow-up, and a car that has
        # since shrunk into the distance would otherwise downgrade the picture
        # the answer is based on.
        keep_old = (ref.last_crop_jpeg is not None
                    and ref.last_crop_info.get("object_px")
                    and _crop_area(info) < _crop_area(ref.last_crop_info))
        if keep_old:
            self._crop = ref.last_crop_jpeg
            self._crop_info = dict(ref.last_crop_info)
            self._crop_info["kept_earlier_crop"] = True
            self.meta["crop_source"] = "referent_memory_better"
        else:
            self._crop = crop
            self._crop_info = info
            self.meta["crop_source"] = "fresh"
            ref.last_crop_jpeg = crop
            ref.last_crop_info = info
            ref.last_best_frame_id = frame.frame_id
            ref.last_crop_path = framebuf.persist(
                self.session_id, f"{ref.track_id}_{frame.frame_id}_crop", crop)
        self.meta["crop"] = {k: v for k, v in self._crop_info.items()
                             if k != "from_memory"}

    def _enrich(self, ring) -> None:
        """Colour and body style for the referent, so the answer can name it."""
        if self._referent is None or self._frame is None:
            return
        got = enrich_mod.enrich_objects(
            self._frame, [self._referent.track_id], self.session.enrichment)
        e = got.get(self._referent.track_id)
        if e:
            self._referent.attributes.update(
                {k: v for k, v in (e.get("attributes") or {}).items() if v})
            self._referent.fine_label = (self._referent.fine_label
                                         or e.get("fine_label"))
        self.meta["enrichment"] = {
            "track_id": self._referent.track_id,
            "attributes": dict(self._referent.attributes),
            "fine_label": self._referent.fine_label,
            "cached": bool(got) and not e,
        }
        # Rebuild so the graph the model reads carries what was just learned.
        if self._graph is not None and e:
            self._graph = self._graph_for(self._frame, ring)

    # -- the request ---------------------------------------------------------
    def _build_messages(self, unavailable: bool = False) -> None:
        parts = []
        grounding = {"question": self.question,
                     "request_type": self.route["request_type"]}

        if unavailable:
            # Phrased as guidance rather than as a status string, because the
            # model will otherwise repeat the status string: "the drive feed
            # may not be running" is a sentence from a dashboard, not from
            # someone in the passenger seat.
            grounding["camera"] = (
                "There is no camera view available right now. Say so in one "
                "short, casual sentence — no technical detail, no apology, no "
                "mention of feeds, sessions or systems — and do not describe "
                "any road scene, because you cannot see one.")
        else:
            grounding["scene"] = self._graph.to_dict() if self._graph else None
            grounding["ego"] = self._graph.ego if self._graph else None
            if self._frame is not None:
                grounding["frame"] = {
                    "age_s": round(self._frame.age_s, 2),
                    "note": ("this frame is a moment ago, not live; it was "
                             "chosen as the clearest recent view"),
                }
            if self.route["request_type"] == router.SCENE:
                # MEASURED, and the reason this note exists: asked "what do you
                # see", the model named a make for a car forty metres away in
                # the wide frame — and got it wrong every time. Shown a close
                # crop of the same vehicle it got it right every time. The wide
                # frame simply does not carry a badge, so a make named from it
                # is invented, and worse, it sticks: the object question that
                # follows inherits the wrong answer from the conversation.
                #
                # Raising the image detail did not help, because the pixels are
                # not there. So the fix is to not make the claim.
                grounding["scene_answer_guidance"] = (
                    "This is a wide shot. Vehicles more than a few metres away "
                    "do not carry enough detail here to identify a marque or "
                    "model — describe them by type and colour (\"a white "
                    "saloon\", \"a pickup\") rather than naming a make you "
                    "cannot actually read. If the driver wants one identified "
                    "they will ask, and you will get a close view of it then.")

        if self._compare:
            grounding["comparing"] = [
                {"which": i + 1, "label": o.label, "fine_label": o.fine_label,
                 "attributes": o.attributes, "position": o.position,
                 "depth_meters": o.depth_m, "motion": o.motion}
                for i, (o, _c, _info) in enumerate(self._compare)]
            grounding["comparison_guidance"] = (
                "Two crops follow, in this order. Compare only these two, "
                "answer the question that was actually asked, and say so if the "
                "images do not settle it.")

        if self._referent is not None and not self._compare:
            r = self._referent
            visible = self.meta.get("referent_visible", True)
            grounding["referring_to"] = {
                "track_id": r.track_id, "label": r.label,
                "fine_label": r.fine_label, "attributes": r.attributes,
                "still_visible": visible,
            }
            obj = self._graph.by_track(r.track_id) if self._graph else None
            if obj is not None:
                grounding["referring_to"].update({
                    "position": obj.position, "motion": obj.motion,
                    "depth_meters": obj.depth_m,
                })
            elif not visible:
                grounding["referring_to"].update({
                    "last_known_position": r.last_position,
                    "last_known_depth_meters": r.last_depth_m,
                    "last_known_motion": r.last_motion,
                })
            if not visible:
                # Acceptance test 5. The wide shot is current and does NOT
                # contain this vehicle, while the crop does and is older —
                # without saying so, the model reconciles the two by describing
                # whatever car IS on that side of the road now.
                unseen = self.meta.get("referent_unseen_s")
                grounding["referent_no_longer_visible"] = {
                    "seconds_since_last_seen": unseen,
                    "guidance": (
                        "This vehicle is no longer in view. The close crop is "
                        "from when it was last seen; the wide shot is current "
                        "and does not contain it — do not describe a different "
                        "vehicle from the wide shot as if it were this one, and "
                        "do not switch to another car. Answer about the one you "
                        "were discussing, and mention that it is out of sight "
                        "only if it actually bears on the answer."),
                }
            if self._resolution is not None and self._resolution.ambiguous:
                grounding["reference_uncertain"] = (
                    "the local system could not be sure which object was meant; "
                    "describe what you are looking at so the driver can correct you")

        if self.route["request_type"] == router.READ_TEXT:
            # Nothing in the detector's vocabulary is a sign, so there is
            # usually no tracked object to crop and the wide frame is all there
            # is. It goes at high detail for that reason.
            grounding["read_text_guidance"] = (
                "The driver is asking what something says. Read only what is "
                "actually legible in the image, word for word. If it is too "
                "small, too far, angled away or motion-blurred to read, say "
                "that plainly and describe what you can make out instead — the "
                "shape, the colour, what kind of sign it looks like. Never "
                "reconstruct wording from what a sign of that type usually "
                "says.")

        if router.wants_fine_detail(self.question):
            # Acceptance test 6. A year, a trim, an engine or a plate is the
            # class of answer a photograph most often cannot support, and the
            # class the model is most willing to supply anyway.
            obj_px = (self._crop_info or {}).get("object_px") or [0, 0]
            limited = bool(self._crop_info) and max(obj_px) < config.CROP_DETAIL_LIMIT_PX
            grounding["fine_detail_requested"] = {
                "detail_limited": limited,
                "guidance": (
                    "The driver is asking for a specific detail — a year, a "
                    "model, a trim, a plate. Give it only if the image actually "
                    "carries it. Otherwise say what the shape and proportions "
                    "DO narrow it down to, name the range you are confident in, "
                    "and be plain about what you cannot see from this angle. A "
                    "confident wrong year is the worst answer available here; "
                    "an honest range is a good one."),
            }
        if self._crop_info:
            # The honest description of what the crop actually is. Crops are
            # upscaled so the object gets enough image tokens (see
            # config.CROP_MIN_PX), which means a smooth, sharp-looking picture
            # is NOT evidence that the detail is real. What matters is how big
            # the object was in the original frame, so that is what travels —
            # and when it was small, it is called out, because a model shown a
            # 30 px car interpolated to 768 px will otherwise read a badge off
            # the interpolation and state it as fact.
            obj_px = self._crop_info.get("object_px") or [0, 0]
            detail_limited = max(obj_px) < config.CROP_DETAIL_LIMIT_PX
            grounding["crop_quality"] = {
                "true_object_size_px": obj_px,
                "detail_limited": detail_limited,
                "note": ("this object was only a few dozen pixels in the original "
                         "frame and the crop is enlarged — fine detail such as "
                         "badges, trim or lettering is not really there"
                         if detail_limited else
                         "the crop is enlarged for legibility; the object was "
                         "captured at this size or larger in the original frame"),
                "from_earlier_frame": bool(self._crop_info.get("from_memory")
                                           or self._crop_info.get("kept_earlier_crop")),
                # A whole-scene answer is given from the wide frame, where a
                # car forty metres away is a smudge, so anything it said about
                # a make or model was a guess from a silhouette. The close crop
                # arriving now is better evidence, and without this the model
                # tends to stay loyal to its own earlier sentence rather than
                # to the picture in front of it.
                "supersedes_earlier": ("this is a much closer view than anything "
                                       "earlier in this conversation; if it "
                                       "contradicts something already said about "
                                       "this object, trust this view"),
            }

        nav_ctx = _nav_context()
        if nav_ctx:
            grounding["navigation"] = nav_ctx

        parts.append({"type": "text",
                      "text": "PERCEPTION GROUNDING (not for reading aloud):\n"
                              + json.dumps(grounding, ensure_ascii=False)})

        if self._frame is not None:
            current = ("Current road scene (the object asked about is NOT in "
                       "this one):" if self.meta.get("referent_visible") is False
                       else "Full road scene:")
            parts.append({"type": "text", "text": current})
            parts.append({"type": "image_url", "image_url": {
                "url": _data_url(self._frame.jpeg),
                "detail": (config.READ_TEXT_FRAME_DETAIL
                           if self.route["request_type"] == router.READ_TEXT
                           else config.VISUAL_FRAME_DETAIL)}})
            self._images.append(("frame", len(self._frame.jpeg)))
        for i, (obj, crop, _info) in enumerate(self._compare):
            parts.append({"type": "text", "text": f"Crop {i + 1}:"})
            parts.append({"type": "image_url", "image_url": {
                "url": _data_url(crop), "detail": config.VISUAL_CROP_DETAIL}})
            self._images.append((f"crop{i + 1}", len(crop)))
        if self._crop is not None and not self._compare:
            label = ("Close crop of the object being asked about, from when it "
                     "was last visible:" if self.meta.get("referent_visible") is False
                     else "Close crop of the object being asked about:")
            parts.append({"type": "text", "text": label})
            parts.append({"type": "image_url", "image_url": {
                "url": _data_url(self._crop),
                "detail": config.VISUAL_CROP_DETAIL}})
            self._images.append(("crop", len(self._crop)))

        parts.append({"type": "text", "text": f"The driver asks: {self.question}"})

        messages = []
        for turn in self.session.recent_turns():
            messages.append({"role": "user", "content": turn["q"]})
            messages.append({"role": "assistant", "content": turn["a"]})
        messages.append({"role": "user", "content": parts})
        self._messages = messages
        self.meta["request"] = {
            "model": config.OPENAI_VISUAL_MODEL,
            "images": [{"kind": k, "bytes": n} for k, n in self._images],
            "history_turns": len(self.session.recent_turns()),
            "grounding_bytes": len(json.dumps(grounding)),
            "reasoning_effort": config.OPENAI_VISUAL_REASONING_EFFORT,
        }

    # -- generation ----------------------------------------------------------
    def stream(self):
        """Yield reply text deltas, then finalise memory and metadata."""
        if self._messages is None:
            self.prepare()
        t0 = time.perf_counter()
        first = None
        parts = []
        prefer = self._answer_model()
        self.meta["answer_model"] = prefer
        try:
            for delta in get_chat_adapter(prefer).stream(self._system, self._messages):
                if first is None:
                    first = time.perf_counter()
                    self.timing["gpt_first_token"] = round((first - t0) * 1000, 1)
                parts.append(delta)
                yield delta
        except Exception as e:
            self.meta["error"] = f"{type(e).__name__}: {e}"
            print(f"[visual_qa] generation failed: {self.meta['error']}", flush=True)
        self.timing["gpt"] = round((time.perf_counter() - t0) * 1000, 1)
        self.reply = "".join(parts).strip()
        self.finish()

    def _answer_model(self) -> str:
        """Local or remote, for THIS turn.

        An object question has a crop, and a crop is the whole of the evidence:
        a local 8B model looking at it answers in a few hundred milliseconds
        what the remote one answered in about two seconds. A scene question has
        no crop and needs the wide frame read carefully -- including the rule
        about not naming a marque it cannot see -- so it keeps the bigger model.
        """
        setting = str(config.VISUAL_ANSWER_MODEL or "auto").lower()
        if setting in ("qwen", "openai"):
            return setting
        if self._compare or self._crop is None:
            return "openai"
        if self.route and self.route.get("request_type") == router.READ_TEXT:
            # Reading text off a sign is the one visual task where the bigger
            # model is reliably better, and a misread sign is a wrong answer
            # rather than a vague one.
            return "openai"
        return "qwen"

    def text(self) -> str:
        return "".join(self.stream())

    def finish(self) -> None:
        """Commit the turn's memory and metadata.

        Called automatically at the end of stream(). Public because a caller
        that deliberately does not generate — the self-test's --no-model mode —
        still has to leave the referent established, or the follow-up it is
        about to test has nothing to follow up on.
        """
        if self._asking is not None:
            # This turn ASKED rather than answered. No referent is committed —
            # that is the entire point: RIO does not know which one yet, and
            # recording a guess here would make the driver's answer irrelevant.
            # The question is armed only if it was actually spoken.
            if self.reply:
                self._asking.text = self.reply
                self.session.pending = self._asking
                self.meta["asked_clarification"] = self._asking.to_log()
            self.meta["is_clarification"] = True
        elif self._referent is not None:
            # Remembered even when generation failed and nothing was said. The
            # referent is established by what the driver ASKED about, not by
            # whether RIO managed to answer — and a failed turn is exactly when
            # they are most likely to just ask again, which has to land on the
            # same car.
            self._referent.last_used_at = time.time()
            self.session.remember(self._referent)
            self.meta["referent"] = self._referent.to_log()
        if self.reply:
            self.session.add_turn(self.question, self.reply)
            # One conversation, not two: the plain /talk path keeps its history
            # in llm_interface, and a visual turn that did not appear there
            # would make the next non-visual question answer as if the driver
            # had never asked about the car.
            try:
                import llm_interface

                llm_interface.note_turn(self.question, self.reply)
            except Exception:
                pass
        self.timing["total"] = round((time.perf_counter() - self._t_start) * 1000, 1)
        self.meta["timing_ms"] = dict(self.timing)
        self.meta["reply"] = self.reply
        if config.RING_PERSIST and self._frame is not None:
            self.meta["frame_path"] = framebuf.persist(
                self.session_id, f"{self._frame.frame_id}_scene", self._frame.jpeg)


def _crop_area(info: dict) -> float:
    px = (info or {}).get("object_px") or [0, 0]
    try:
        return float(px[0]) * float(px[1])
    except (TypeError, ValueError, IndexError):
        return 0.0


def answer(session_id: str, question: str, route: dict = None) -> VisualAnswer:
    """Prepare a visual turn. The caller streams it."""
    va = VisualAnswer(session_id, question, route)
    va.prepare()
    return va


def scene_graph(session_id: str) -> dict:
    """The current scene graph for a session, as the endpoint serves it."""
    ring = framebuf.peek_ring(session_id)
    if ring is None:
        return {"available": False, "reason": "no_session_buffer", "objects": []}
    frame = ring.latest()
    if frame is None:
        return {"available": False, "reason": "empty_buffer", "objects": []}
    sess = get_session(session_id)
    graph = scene_mod.build(
        frame.objects, frame.ego, {"w": frame.w, "h": frame.h},
        t=frame.t, wall_t=frame.wall_t, frame_id=frame.frame_id,
        tracker=ring.tracker, attributes=sess.enrichment.all())
    out = graph.to_dict()
    out["available"] = True
    out["frame_age_s"] = round(frame.age_s, 2)
    out["buffer"] = ring.stats()
    ref = sess.active_referent()
    out["active_referent"] = ref.to_log() if ref else None
    return out
