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
from rio_prompts import VISUAL_SYSTEM_PROMPT


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


_chat_adapter = None
_chat_lock = threading.Lock()


def get_chat_adapter() -> ChatAdapter:
    global _chat_adapter
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
        if obj.fine_label:
            self.fine_label = obj.fine_label
        if obj.attributes:
            self.attributes.update(obj.attributes)


class VisualSession:
    """Per-drive conversation state for the visual path."""

    def __init__(self, key: str):
        self.key = key
        self.referent: Optional[VisualReferent] = None
        self.enrichment = enrich_mod.EnrichmentCache()
        self.turns = []                 # [{"q":..., "a":..., "t":...}]
        self.lock = threading.Lock()
        self.created = time.time()

    def active_referent(self) -> Optional[VisualReferent]:
        if self.referent is None or self.referent.stale():
            return None
        return self.referent

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
        import nav

        route = nav.latest_route()
        if not route:
            return None
        return {"destination": route.get("label") or route.get("destination"),
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
        self._images = []
        self._frame = None
        self._crop = None
        self._crop_info = {}
        self._graph = None
        self._resolution = None
        self._referent = None
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
                has_referent=self.session.active_referent() is not None)
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

        # 3. Which object is this about?
        self._referent = self._establish_referent(ring, rtype)
        t = self._stage("resolve", t)

        # 4. The clearest recent frame containing THAT object, and a crop of it.
        if self._referent is not None:
            self._pick_object_frame(ring, ref_wall)
        t = self._stage("crop", t)

        # 5. Attributes for whatever is about to be described.
        self._enrich(ring)
        t = self._stage("enrich", t)

        self._build_messages()
        self._stage("prepare_total", self._t_start)

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
            if ref.last_crop_jpeg:
                self._crop = ref.last_crop_jpeg
                self._crop_info = dict(ref.last_crop_info)
                self._crop_info["from_memory"] = True
                self.meta["crop_source"] = "referent_memory"
                self.meta["referent_visible"] = False
            else:
                self.meta["crop_source"] = "none"
                self.meta["referent_visible"] = False
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

        if self._referent is not None:
            r = self._referent
            grounding["referring_to"] = {
                "track_id": r.track_id, "label": r.label,
                "fine_label": r.fine_label, "attributes": r.attributes,
                "still_visible": self.meta.get("referent_visible", True),
            }
            obj = self._graph.by_track(r.track_id) if self._graph else None
            if obj is not None:
                grounding["referring_to"].update({
                    "position": obj.position, "motion": obj.motion,
                    "depth_meters": obj.depth_m,
                })
            if self._resolution is not None and self._resolution.ambiguous:
                grounding["reference_uncertain"] = (
                    "the local system could not be sure which object was meant; "
                    "describe what you are looking at so the driver can correct you")
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
            parts.append({"type": "text", "text": "Full road scene:"})
            parts.append({"type": "image_url", "image_url": {
                "url": _data_url(self._frame.jpeg),
                "detail": config.VISUAL_FRAME_DETAIL}})
            self._images.append(("frame", len(self._frame.jpeg)))
        if self._crop is not None:
            parts.append({"type": "text",
                          "text": "Close crop of the object being asked about:"})
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
        try:
            for delta in get_chat_adapter().stream(VISUAL_SYSTEM_PROMPT,
                                                   self._messages):
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

    def text(self) -> str:
        return "".join(self.stream())

    def finish(self) -> None:
        """Commit the turn's memory and metadata.

        Called automatically at the end of stream(). Public because a caller
        that deliberately does not generate — the self-test's --no-model mode —
        still has to leave the referent established, or the follow-up it is
        about to test has nothing to follow up on.
        """
        if self._referent is not None:
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
