"""Qwen3-VL as an attribute reader and a reference resolver — nothing else.

Design ref: docs/visual_qa.md §5. This is the whole of Qwen's role in the
conversation path, and the boundary is deliberate.

WHAT QWEN IS FOR HERE
---------------------
Two jobs, both of them "look at this picture and tell me one small thing":

  ENRICHMENT.  What colour is this vehicle, and what body style is it? The
               detector's class vocabulary is car / truck / bus / motorcycle,
               which is right for a warning and far too coarse for a sentence.
               A colour and a body style is what turns "car" into something a
               driver would recognise as the car they meant.

  RESOLUTION.  "the silver one on the left" -> a track id. Geometry can narrow
               that down and often finishes the job on its own, but when two
               candidates are both plausible, the thing that separates them is
               what they LOOK like, and that needs eyes on the frame.

WHAT QWEN IS NOT FOR
--------------------
It does not detect, it does not track, it does not measure distance, and none
of what it returns is ever spoken. Detection is RF-DETR's, per frame, at 5 ms
(headway/detect.py); tracking is membership.py's; range is Depth Anything's.
Qwen used to hold the anchor and it was removed from that path on purpose --
see headway/detect.py's header. Nothing here puts it back.

COST, AND WHY IT RUNS ON DEMAND ONLY
------------------------------------
An 8B VLM decodes at ~52 tokens/s on this GPU, so each of these calls is a
few hundred milliseconds, and it shares the GPU with a 4 fps safety loop. So it
is never on the frame path, it is capped per request, it is cached per track,
and it is not called at all for a plain "what do you see" -- GPT-5.5 is looking
at the same frame and reads colour off it directly, so paying an 8B decode for
that would buy nothing but latency.
"""
import json
import re
import threading
import time

import config
import scene as scene_mod

# --- prompts ----------------------------------------------------------------
# Terse and strictly shaped, for the same reason perceive.py's is: decode is
# what costs, every token is serial, and this model follows a stated JSON schema
# more reliably than it follows prose.
ENRICH_PROMPT = (
    "This is a cropped photo of one road user, seen from a car's dashcam.\n"
    "Reply with ONLY this JSON and nothing else:\n"
    '{"color": "<one common colour word>", "body": "<one of: sedan, coupe, '
    'hatchback, wagon, suv, crossover, pickup, van, minivan, box_truck, '
    'semi_truck, bus, motorcycle, bicycle, other>"}\n'
    'Use "unknown" for anything you cannot see clearly. Do not guess.'
)

# Verification, not discovery. The model is handed the list the map produced and
# asked only whether each entry is on screen — with a count, because two of the
# same brand in one view is a rejection, and with a coarse side and box, because
# depth wants somewhere to sample and the map already knows which side to expect.
# Numbers rather than adjectives: "clearly visible" is not a threshold.
LANDMARK_PROMPT = (
    "This is a photo from a car's dashcam, looking forward along the road.\n"
    "For EACH of these {n} businesses, say whether its sign or storefront is "
    "visible in this image right now:\n{listing}\n"
    "Reply with ONLY a JSON array of {n} objects, in the same order:\n"
    '[{{"visible": true|false, "identity": 0.0-1.0, "clarity": 0.0-1.0, '
    '"count": <how many separate ones of that brand you can see>, '
    '"side": "left"|"right"|"ahead"}}]\n'
    "identity is how sure you are it is that exact brand. clarity is how "
    "clearly readable it is from here. Use false and 0.0 when unsure. "
    "Do not guess."
)

RESOLVE_PROMPT = (
    "A driver riding in this car said: \"{phrase}\"\n"
    "Numbered boxes mark the road users the car is tracking:\n{legend}\n"
    "Which number is the driver talking about?\n"
    'Reply with ONLY this JSON: {{"n": <number>, "sure": <true|false>}}\n'
    'Use {{"n": 0, "sure": false}} if you genuinely cannot tell which one.'
)

COLOUR_WORDS = {
    "white", "black", "silver", "grey", "gray", "red", "blue", "green",
    "yellow", "orange", "brown", "beige", "tan", "gold", "purple", "maroon",
    "navy", "cream", "bronze", "unknown",
}

BODY_WORDS = {
    "sedan", "coupe", "hatchback", "wagon", "suv", "crossover", "pickup",
    "van", "minivan", "box_truck", "semi_truck", "bus", "motorcycle",
    "bicycle", "other", "unknown",
}


# ---------------------------------------------------------------------------
# Adapter interface
# ---------------------------------------------------------------------------
class VisionAdapter:
    """What the conversation path needs from a local vision model.

    Three methods, all optional to implement usefully: an adapter that cannot
    enrich returns {}, one that cannot resolve returns None, one that cannot
    verify a landmark returns []. The pipeline degrades — to geometry-only
    reference resolution, or to navigation with no contextual anchor — rather
    than failing, which is what keeps a Qwen outage from taking visual
    conversation or turn-by-turn down with it.
    """

    name = "none"

    def enrich(self, crop_jpeg: bytes) -> dict:
        raise NotImplementedError

    def resolve(self, frame_jpeg: bytes, phrase: str, candidates: list):
        raise NotImplementedError

    def landmark(self, frame_jpeg: bytes, labels: list) -> list:
        """Which of these EXPECTED landmarks is visible in this frame?

        Navigation's only question of the camera (docs/navigation_v1.md §14).
        Note what it is not: it is not "what landmarks do you see". The map has
        already said what should be there, so this is verification of a short
        closed list — bounded, answerable, and safe to be wrong about, because
        the answer "no" simply means RIO says "Take the next left" instead.

        Returns one entry per label, in order:
            {"visible": bool, "identity": 0-1, "clarity": 0-1,
             "count": int, "side": "left"|"right"|"ahead"|None,
             "box": [x1, y1, x2, y2] | None}
        """
        return []


class QwenAdapter(VisionAdapter):
    """Qwen3-VL-8B, borrowed from the copy vision.py already has resident."""

    name = "qwen3-vl-8b"

    def _generate(self, images, prompt: str, max_new_tokens: int) -> str:
        import torch
        import vision

        processor, model, lock = vision.get_handles()
        content = [{"type": "image", "image": im} for im in images]
        content.append({"type": "text", "text": prompt})
        msgs = [{"role": "user", "content": content}]
        # The same lock every other Qwen caller takes. It serialises against
        # /perceive and against headway's anchor path, which is what stops two
        # generates stacking on the GPU while the 4 fps loop is running.
        with lock:
            inputs = processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(model.device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False)
            return processor.batch_decode(
                out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )[0].strip()

    def enrich(self, crop_jpeg: bytes) -> dict:
        import io

        from PIL import Image

        pil = Image.open(io.BytesIO(crop_jpeg)).convert("RGB")
        raw = self._generate([pil], ENRICH_PROMPT, config.ENRICH_MAX_NEW_TOKENS)
        return _parse_enrichment(raw)

    def resolve(self, frame_jpeg: bytes, phrase: str, candidates: list):
        """-> (candidate_id | None, info). `candidates` is [(cid, box, desc)]."""
        import io

        from PIL import Image

        annotated, legend = _annotate(frame_jpeg, candidates)
        pil = Image.open(io.BytesIO(annotated)).convert("RGB")
        prompt = RESOLVE_PROMPT.format(phrase=phrase, legend=legend)
        raw = self._generate([pil], prompt, 32)
        n, sure = _parse_choice(raw)
        info = {"raw": raw, "n": n, "sure": sure, "legend": legend}
        if not n or n < 1 or n > len(candidates):
            return None, info
        return candidates[n - 1][0], info

    def landmark(self, frame_jpeg: bytes, labels: list) -> list:
        """Verify a short closed list of expected landmarks in one frame.

        One generate for the whole list rather than one per label: the decode
        is what costs, the list is at most four entries, and asking about them
        together is also what lets the model report that it can see TWO Shell
        signs — which is a rejection, and would be invisible to per-label
        questions asked in isolation.
        """
        import io

        from PIL import Image

        if not labels:
            return []
        pil = Image.open(io.BytesIO(frame_jpeg)).convert("RGB")
        listing = "\n".join(f"{i + 1}. {lbl}" for i, lbl in enumerate(labels))
        prompt = LANDMARK_PROMPT.format(n=len(labels), listing=listing)
        raw = self._generate([pil], prompt, 40 + 34 * len(labels))
        return _parse_landmarks(raw, len(labels))


_adapter = None
_adapter_lock = threading.Lock()


def get_adapter() -> VisionAdapter:
    """The configured local vision adapter. One swap point for the whole path."""
    global _adapter
    with _adapter_lock:
        if _adapter is None:
            _adapter = QwenAdapter()
        return _adapter


def set_adapter(adapter: VisionAdapter) -> None:
    """Swap the local vision model (tests use this to avoid loading Qwen)."""
    global _adapter
    with _adapter_lock:
        _adapter = adapter


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def _strip_fence(text: str) -> str:
    t = (text or "").strip()
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z]*\s*", "", t)
        t = re.sub(r"\s*```$", "", t)
    return t.strip()


def _parse_enrichment(raw: str) -> dict:
    """Qwen's reply -> {"attributes": {...}, "fine_label": ...}.

    Anything outside the stated vocabularies is dropped rather than passed
    along. An invented colour is a detail the model downstream would state as
    fact, and the whole point of enrichment is to give it something it can
    trust more than its own glance, not less.
    """
    t = _strip_fence(raw)
    obj = None
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            obj = None
    if not isinstance(obj, dict):
        obj = {}
        for key in ("color", "body"):
            m = re.search(r'"%s"\s*:\s*"([^"]*)"' % key, t)
            if m:
                obj[key] = m.group(1)

    out = {"attributes": {}, "fine_label": None, "raw": raw}
    colour = str(obj.get("color") or "").strip().lower()
    if colour in COLOUR_WORDS and colour != "unknown":
        out["attributes"]["color"] = colour
    body = str(obj.get("body") or "").strip().lower().replace(" ", "_")
    if body in BODY_WORDS and body not in ("unknown", "other"):
        out["fine_label"] = body
    return out


def _parse_landmarks(raw: str, n: int) -> list:
    """The landmark reply -> a fixed-length list of observations.

    Anything malformed becomes "not visible", never a partial guess: a
    verification step that answers "probably" when it failed to parse is worse
    than one that answers "no", because "no" costs a sentence and "probably"
    costs the driver's trust.
    """
    blank = {"visible": False, "identity": 0.0, "clarity": 0.0, "count": 0,
             "side": None, "box": None}
    try:
        data = json.loads(_strip_fence(raw))
    except Exception:
        m = re.search(r"\[.*\]", _strip_fence(raw), re.S)
        if not m:
            return [dict(blank) for _ in range(n)]
        try:
            data = json.loads(m.group(0))
        except Exception:
            return [dict(blank) for _ in range(n)]
    if not isinstance(data, list):
        return [dict(blank) for _ in range(n)]
    out = []
    for i in range(n):
        item = data[i] if i < len(data) and isinstance(data[i], dict) else {}
        try:
            ident = float(item.get("identity", 0.0) or 0.0)
            clarity = float(item.get("clarity", 0.0) or 0.0)
            count = int(item.get("count", 0) or 0)
        except (TypeError, ValueError):
            ident = clarity = 0.0
            count = 0
        side = item.get("side")
        side = side.lower() if isinstance(side, str) else None
        box = item.get("box")
        if not (isinstance(box, (list, tuple)) and len(box) == 4):
            box = None
        visible = bool(item.get("visible")) and ident > 0.0
        out.append({
            "visible": visible,
            "identity": max(0.0, min(1.0, ident)),
            "clarity": max(0.0, min(1.0, clarity)),
            "count": max(count, 1 if visible else 0),
            "side": side if side in ("left", "right", "ahead") else None,
            "box": [float(v) for v in box] if box else None,
        })
    return out


def _parse_choice(raw: str):
    """-> (n, sure). n is None when the reply was not a usable choice."""
    t = _strip_fence(raw)
    m = re.search(r"\{.*\}", t, re.S)
    if m:
        try:
            obj = json.loads(m.group(0))
            n = obj.get("n")
            return (int(n) if isinstance(n, (int, float)) else None,
                    bool(obj.get("sure")))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    m = re.search(r"\b(\d{1,2})\b", t)
    return (int(m.group(1)) if m else None, False)


def _annotate(frame_jpeg: bytes, candidates: list):
    """Draw numbered boxes on a copy of the frame. -> (jpeg, legend text).

    The model has to be able to say WHICH one, and the only reliable channel for
    that is a mark it can see. A legend alone ("candidate 2 is a car on the
    left") would have it matching prose against prose, which is the failure this
    is meant to avoid -- the question is what the vehicles look like.
    """
    import cv2
    import numpy as np

    img = cv2.imdecode(np.frombuffer(frame_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return frame_jpeg, ""
    lines = []
    for i, (cid, box, desc) in enumerate(candidates, start=1):
        x1, y1, x2, y2 = [int(round(float(v))) for v in box]
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 230, 255), 3)
        label = str(i)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
        ty = max(th + 6, y1 - 6)
        cv2.rectangle(img, (x1, ty - th - 6), (x1 + tw + 10, ty + 4), (0, 230, 255), -1)
        cv2.putText(img, label, (x1 + 5, ty), cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (0, 0, 0), 3, cv2.LINE_AA)
        lines.append(f"  {i}. {desc}")
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return (buf.tobytes() if ok else frame_jpeg), "\n".join(lines)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------
class EnrichmentCache:
    """Per-session attributes, keyed by track id, with a TTL.

    A car's colour does not change, but a track id can be recycled after the
    candidate ages out of membership.py (1 s undetected), so the entry has to
    expire. The TTL is what stops a silver saloon's colour being inherited by
    whatever takes its id next.
    """

    def __init__(self, ttl_s: float = None):
        self.ttl = config.ENRICH_TTL_S if ttl_s is None else ttl_s
        self._by_track = {}
        self._lock = threading.Lock()

    def get(self, track_id: str):
        with self._lock:
            hit = self._by_track.get(track_id)
            if hit is None:
                return None
            if time.time() - hit["t"] > self.ttl:
                self._by_track.pop(track_id, None)
                return None
            return hit["value"]

    def put(self, track_id: str, value: dict) -> None:
        with self._lock:
            self._by_track[track_id] = {"t": time.time(), "value": value}

    def all(self) -> dict:
        now = time.time()
        with self._lock:
            return {k: v["value"] for k, v in self._by_track.items()
                    if now - v["t"] <= self.ttl}

    def reset(self) -> None:
        with self._lock:
            self._by_track.clear()


def enrich_objects(frame, track_ids, cache: EnrichmentCache,
                   adapter: VisionAdapter = None, max_objects: int = None) -> dict:
    """Fill in colour and body style for a few tracked objects. -> {track_id: {...}}

    Cache-first, capped, and never fatal: an enrichment failure costs an
    adjective, and the answer is still grounded by the detector and the frame
    the model can see for itself.
    """
    if not config.ENRICH_ENABLED or frame is None:
        return {}
    adapter = adapter or get_adapter()
    max_objects = config.ENRICH_MAX_OBJECTS if max_objects is None else max_objects

    out = {}
    todo = []
    for tid in track_ids:
        hit = cache.get(tid)
        if hit is not None:
            out[tid] = hit
        elif len(todo) < max_objects:
            todo.append(tid)

    for tid in todo:
        cid = scene_mod.candidate_id_of(tid)
        obj = frame.object_by_id(cid) if cid is not None else None
        if obj is None:
            continue
        t0 = time.perf_counter()
        try:
            crop, crop_info = scene_mod.crop_jpeg(frame.jpeg, obj["box"])
            value = adapter.enrich(crop)
            value["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            value["crop"] = crop_info
            cache.put(tid, value)
            out[tid] = value
        except Exception as e:
            print(f"[enrich] {tid} failed: {type(e).__name__}: {e}", flush=True)
    return out
