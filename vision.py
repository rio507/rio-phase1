"""The resident eye. One model on this card, loaded once, read by everything.

WHICH MODEL, AND WHY IT IS A ROLE. config.LOCAL_VISION_MODEL selects it:
`cosmos` (nvidia/Cosmos-Reason2-2B, the default) or `qwen`
(Qwen/Qwen3-VL-8B-Instruct, the rollback, one env var away). Both are the
`qwen3_vl` architecture -- Cosmos-Reason2 is a post-train of Qwen3-VL-2B -- so
there is one loader, one processor and one generate here, not two code paths.
See config.LOCAL_VISION_MODEL for what was measured and what the swap costs.

THE TWO MODELS ARE ASKED DIFFERENT QUESTIONS, and that is the substance of the
change rather than a detail of it:

  qwen    OBSERVER_PROMPT. The sentence RIO would say, in her register, checked
          by persona.lint() and -- if it passes -- spoken to the driver word for
          word. The local model was her voice as well as her eyes.
  cosmos  SENSOR_PROMPT. A reading: road, traffic, risk. Shown raw on the glass
          under the model's own name, handed to grok as evidence when she
          speaks, and never spoken verbatim. An instrument does not have a
          register.

WHAT RUNS ON EVERY OUTPUT, WHICHEVER MODEL IT IS
------------------------------------------------
Two guards, and they catch different lies:

  is_prompt_example   the reading came back as one of the prompt's own examples.
                      Qwen3-VL did this with a bare example list -- a hand, a
                      black frame and random noise all returned the first
                      example verbatim -- and the sentence reads perfectly.
  canned.describe     the reading is memorised label text or a decoding loop.
                      Written for the teachers, because Alpamayo 1.5 answered
                      eight of ten keyframes with a scenario label out of its
                      training taxonomy, and a 2B is MORE prone to that, not
                      less. It runs here now, on every frame, and what it flags
                      is counted and reported (`flag_rate()`).

Both are refusals of a sentence, never of the frame: an empty reading is a state
every caller already handles, and the honest slow path picks it up.
"""
import io
import os
import re
import threading
import time
import torch
from PIL import Image
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import config
from rio_prompts import (OBSERVER_PROMPT, SENSOR_PROMPT,
                         SENSOR_PROMPT_TERSE, is_prompt_example,
                         sensor_faults)
from teachers import canned

MODEL_ID = config.local_vision_model_id()

# WHICH QUESTION THIS MODEL IS ASKED. Resolved through the role, so a rollback
# moves the prompt with the weights -- a sensor prompt on Qwen would waste the
# one thing Qwen is here for, and OBSERVER_PROMPT on Cosmos would ask an
# instrument to have a voice.
def _sensor_prompt():
    """The reading's shape, which is also its latency. See
    config.LOCAL_VISION_SENSOR_PROMPT: `terse` halves the output tokens and is
    what holds the observer at 1 Hz; `full` is the comfortable-English version
    and is 1.8x slower for the same three fields."""
    if str(getattr(config, "LOCAL_VISION_SENSOR_PROMPT", "terse")) == "full":
        return SENSOR_PROMPT
    return SENSOR_PROMPT_TERSE


TEACHER_PROMPT = (OBSERVER_PROMPT if config.local_vision_speaks_directly()
                  else _sensor_prompt())

# The reasoning trace, if one comes back anyway. Cosmos-Reason2 thinks inside
# these before answering; SENSOR_PROMPT asks it not to, and this is what makes
# that a rule rather than a request. An unterminated <think> is handled too --
# see _strip_think, and the note there about what an unclosed one means.
_THINK = re.compile(r"<think>.*?</think>", re.S)
_THINK_OPEN = re.compile(r"<think>.*$", re.S)


_processor = None
_model = None
_lock = threading.Lock()
# WHO HOLDS THE MODEL, for the drive log. Every caller that takes `_lock`
# names itself here on the way in and clears it on the way out, so a headway
# row can say "the Qwen lock was held by the observer when this frame ran"
# as a field rather than as two timestamps lined up by hand. Best-effort: a
# name is a hint, the lock is the fact.
_holder = None
_held_since = 0.0


def busy():
    """-> {'held': bool, 'holder': str|None, 'held_ms': float|None}"""
    held = _lock.locked()
    return {"held": held, "holder": _holder if held else None,
            "held_ms": round((time.time() - _held_since) * 1000.0, 1)
                       if held and _held_since else None}


def note_holder(name):
    global _holder, _held_since
    _holder = name
    _held_since = time.time()


def clear_holder():
    global _holder, _held_since
    _holder = None
    _held_since = 0.0


_last_observation = ""
# WHEN that observation was made, and of what. The cache used to be a bare
# string, which was fine while its only reader asked "what did she last see"
# and answered a conversation turn with it. It is not fine now that a live
# answer can be served from it: a caption with no timestamp cannot be told
# apart from a current one, and describing a road the car left thirty seconds
# ago -- confidently, in the present tense -- is the failure this whole
# subsystem exists to prevent.
_last_observed_at = 0.0
_last_observed_frame = None
# Observations refused because the model returned one of the prompt's own
# examples instead of looking. Counted rather than silent: a number that starts
# climbing after a prompt edit is the first sign the eyes have stopped working,
# and it is otherwise invisible -- the sentence it produces is well-formed,
# in her register, and completely made up.
_parroted = 0


# Where the weights go. Pinned rather than "auto".
#
# HONEST NOTE ON WHAT THIS DOES AND DOES NOT FIX. It was changed while chasing
# a startup failure -- "Cannot copy out of meta tensor" from inside
# accelerate's dispatch_model -- and it did NOT fix it. That failure was two
# uvicorn processes loading Qwen onto the same card at once; it happens on
# "auto" and on "cuda:0" alike, and stops happening when there is one server.
#
# It is kept pinned anyway, for a smaller and separate reason: "auto" asks
# accelerate to plan a memory budget from whatever the card has free at that
# instant, and to offload whatever it thinks will not fit. On one GPU with
# 97 GB and a 17 GB model there is nothing to plan, and a plan that depends on
# timing is a plan that can differ between two identical restarts. Pinning
# removes a variable rather than a bug.
#
# If this ever has to run on a card too small for the model, that is a
# deliberate change to make here, with sharding thought about on purpose.
DEVICE_MAP = os.environ.get("RIO_QWEN_DEVICE",
                            "cuda:0" if torch.cuda.is_available() else "cpu")


def _ensure_loaded():
    global _processor, _model
    if _model is None:
        print(f"[vision] loading {MODEL_ID} ({config.LOCAL_VISION_MODEL}) "
              f"onto {DEVICE_MAP}...", flush=True)
        t0 = time.time()
        try:
            _processor = AutoProcessor.from_pretrained(MODEL_ID)
            # `dtype` replaces the deprecated `torch_dtype` kwarg in
            # transformers 4.57.
            _model = Qwen3VLForConditionalGeneration.from_pretrained(
                MODEL_ID, dtype=torch.bfloat16, device_map=DEVICE_MAP
            )
        except Exception as e:
            # THE WEIGHTS ARE NOT HERE, AND THE MESSAGE HAS TO SAY WHAT TO DO.
            #
            # This is a role now, so "the model failed to load" has a new and
            # very likely cause that a transformers traceback does not name: the
            # role points at weights this pod has never had. Cosmos-Reason2-2B
            # is a GATED repo -- a valid HF token is not enough, the account has
            # to have accepted NVIDIA's licence on that repo specifically, and
            # the failure for a token that has accepted the 8B and not the 2B is
            # a 403 forty frames deep in a stack trace.
            #
            # Named, with the rollback beside it, because the pod is still
            # useful without the local eye (the remote visual path is
            # untouched) and the person reading this log needs one line, not an
            # investigation.
            _processor = None
            print(f"[vision] CANNOT LOAD {MODEL_ID} "
                  f"(LOCAL_VISION_MODEL={config.LOCAL_VISION_MODEL}): "
                  f"{type(e).__name__}: {e}", flush=True)
            if "gated" in str(e).lower() or "403" in str(e):
                print(f"[vision] that repo is GATED: the HF account behind "
                      f"HF_TOKEN has to accept its licence at "
                      f"https://huggingface.co/{MODEL_ID} — a token that works "
                      f"for another Cosmos size does not cover this one.",
                      flush=True)
            print("[vision] the local eye is DOWN. The remote visual path is "
                  "unaffected; roll back with LOCAL_VISION_MODEL=qwen.",
                  flush=True)
            try:
                import gpu_health
                gpu_health.note("vision", e)
            except Exception:
                pass
            raise
        # WHAT IT COST TO LOAD AND WHAT IT IS HOLDING. Both printed because the
        # swap that brought this comment was justified partly on VRAM, and a
        # claim about VRAM that is not printed at startup is a claim nobody can
        # check on the machine it matters on. The 4090 in this pod has 24 GB and
        # the detector, the depth model and the lane model want their share.
        held = None
        try:
            if torch.cuda.is_available():
                held = round(torch.cuda.memory_allocated() / 1048576.0)
        except Exception:
            pass
        print(f"[vision] loaded in {time.time() - t0:.1f}s"
              + (f", {held} MiB allocated" if held is not None else ""),
              flush=True)


def _downscale(pil, max_side: int):
    """Fit the long edge to `max_side`, or leave it alone.

    Prefill scales with vision tokens, and vision tokens scale with pixels. The
    observer's whole job is one short factual sentence about the scene, which
    does not need 720p: measured on this GPU, full frame 432 ms, 768 px 377 ms,
    512 px 360 ms, and the caption at 512 was the same sentence. That is ~17%
    off a call this now makes once a second in the background, and the smaller
    frame leaves that much more of the card for the 4 fps headway loop it runs
    beside.
    """
    if not max_side:
        return pil
    w, h = pil.size
    longest = max(w, h)
    if longest <= max_side:
        return pil
    scale = max_side / float(longest)
    return pil.resize((max(1, int(w * scale)), max(1, int(h * scale))),
                      Image.BILINEAR)


def _generate_kwargs() -> dict:
    """Every argument the decode is run with, in ONE place.

    HERE BECAUSE THE NUMBERS WERE MEASURED AND THE MEASUREMENT HAS TO BE THE
    THING THAT SHIPS. warm() and observe() have to agree exactly: the static
    cache pays a one-off compile on its first call at a given shape, and a warm
    that used different arguments would move that cost onto the first frame of a
    drive. See config.LOCAL_VISION_CACHE for what was measured and what it cost.
    """
    budget = int(config.LOCAL_VISION_MAX_TOKENS)
    think = int(getattr(config, "LOCAL_VISION_THINK_BUDGET", 0) or 0)
    if think > 0 and not config.local_vision_speaks_directly():
        budget = max(budget, think)
    kw = {"max_new_tokens": budget, "do_sample": False}
    cache = str(getattr(config, "LOCAL_VISION_CACHE", "") or "").strip()
    if cache and cache != "dynamic":
        # 25.8 -> 8.1 ms/token, which is 97% of a reading. The whole fix.
        kw["cache_implementation"] = cache
    pen = float(getattr(config, "LOCAL_VISION_REPETITION_PENALTY", 1.0) or 1.0)
    if pen and pen != 1.0:
        # 1 decoding loop in 20 readings -> 0. See the config note.
        kw["repetition_penalty"] = pen
    return kw


def frame_structure(pil) -> float:
    """How much is IN this picture. -> standard deviation of its luminance.

    Asked of the frame, not of a model, because it is a property of the frame.
    128x72 is small enough to cost 2.5 ms and large enough that a lane marking
    still registers; the number it produces is two orders of magnitude apart
    between a covered lens and a road (see config.LOCAL_VISION_BLANK_STD).
    """
    import numpy as np

    small = pil.convert("L").resize((128, 72), Image.BILINEAR)
    return float(np.asarray(small, dtype=np.float32).std())


def _strip_think(text: str):
    """Remove a reasoning trace. -> (answer, had_trace, unterminated)

    SENSOR_PROMPT asks for no trace. This is what makes that a rule: a prompt is
    a request, and the model that ignores it would otherwise put four hundred
    words of deliberation on the glass where a reading goes.

    An UNTERMINATED <think> is its own case and not an answer. It means the
    token budget ran out mid-thought, so there is no answer after it to keep --
    returning the trace would publish deliberation as perception. The caller
    treats an empty string the way it treats every other refusal.
    """
    t = text or ""
    if "<think>" not in t:
        return t, False, False
    closed = _THINK.sub("", t).strip()
    if "<think>" not in closed:
        return closed, True, False
    # Opened and never closed: nothing after it to trust.
    return _THINK_OPEN.sub("", t).strip(), True, True


# WHAT THE GUARDS HAVE CAUGHT, so the claim "it reports honestly" is a number
# rather than an impression. Every one of these is a reading that was REFUSED --
# the frame is still there and the slow path still answers, which is why
# refusing is cheap and publishing a fabrication is not.
_flags = {
    # the reading was one of the prompt's own examples
    "prompt_example": 0,
    # memorised label text, exotic whitespace, or a sentence repeating inside
    # one answer (teachers.canned). The fault Alpamayo had and a 2B is more
    # prone to.
    "canned": 0,
    # identical to the previous N readings, byte for byte. A HINT, not proof:
    # on an unchanging road the same reading may simply be right again, which is
    # why it is counted apart from `canned`.
    "repeated": 0,
    # a reasoning trace came back despite the prompt
    "think_trace": 0,
    # ...and one that never closed, which is a reading that failed
    "think_unterminated": 0,
    # the instrument gave the driver an instruction (rio_prompts.sensor_faults)
    "advisory": 0,
    # the frame had nothing in it to read, and no model was asked
    "blank_frame": 0,
    "total": 0,
}
_repeats = canned.RepeatTracker()


def reading_refused(text, repeats: int = 0):
    """Is this reading a recitation or a decoding loop? -> the verdict, or None.

    THE DECISION, AS A FUNCTION SOMETHING ELSE CAN CALL. It was four words
    inside observe() -- `verdict.get("markers") or verdict.get("loop")` -- and
    on 2026-09-21 that list was one flag short of the guard it was reading.
    `word_loop` had been added to teachers/canned.py for precisely the terse
    sensor format, with its own acceptance check, and the branch that ACTS on a
    verdict never learned about it. So this reached the Perception card:

        ROAD: single|single|curb|CURB|CURB|CURB|CURB|CURB|CURB|CURB|CURB|...

    under a log line reading "reading repeated x3 (not refused)". The guard was
    right the whole time -- canned.describe scores that string word_loop 11,
    strength strong, case mixing and leading pieces and all. The consumer was
    wrong, and nothing could see it, because the suite tested the DETECTOR and
    then grepped this file for a substring.

    Two things follow, and both are the point of this function existing:

      The refusal set is `strength == "strong"`, which is canned.py's own
      statement of proof versus hint rather than a list kept in a second file.
      A new kind of loop added there is refused here the day it lands.

      It is importable, so tools/local_vision_selftest.py can run the decision
      the live path runs instead of asserting that the file contains a string.
      A test that cannot fail the way the system failed is not a test of it.

    `repeats` is how many keyframes in a row have carried this exact text (see
    canned.RepeatTracker). Repetition ALONE is weak and never refused here: on
    an unchanging road the same reading may simply be right again.
    """
    verdict = canned.describe(text or "", repeats)
    if verdict and verdict.get("strength") == "strong":
        return verdict
    return None


def flag_rate() -> dict:
    """-> the flag counts with a rate per reading. `total` is readings ATTEMPTED.

    Read this next to `parroted()`: that counter predates the swap and counts
    one of these flags. This dict is the whole set.
    """
    total = _flags["total"] or 0
    out = dict(_flags)
    out["model"] = config.local_vision_label()
    out["rate"] = {k: (round(v / total, 4) if total else None)
                   for k, v in _flags.items() if k != "total"}
    return out


def reset_flags() -> None:
    for k in _flags:
        _flags[k] = 0
    _repeats.reset()


# How long the last observe() waited for the lock before it could start.
_last_lock_wait_ms = 0.0


def last_lock_wait_ms() -> float:
    return _last_lock_wait_ms


def observe(image_bytes: bytes, max_side: int = None, frame_id=None) -> str:
    """Run VLM on a new frame, cache + return the observation."""
    global _last_observation, _last_observed_at, _last_observed_frame, _last_lock_wait_ms
    if not config.VISION_ENABLED:
        return ""
    if max_side is None:
        max_side = config.OBSERVER_MAX_SIDE_PX
    t_wait = time.time()
    with _lock:
        note_holder("observe")
        _lock_wait_ms = (time.time() - t_wait) * 1000.0
        _ensure_loaded()
        pil = _downscale(Image.open(io.BytesIO(image_bytes)).convert("RGB"),
                         max_side)
        # NOTHING IN THE FRAME MEANS NOTHING TO SAY ABOUT IT, and the model is
        # not asked. Before the forward pass, because the cheapest way to not
        # fabricate a road is to not ask a model about a picture that has no
        # road in it -- and because the pass itself is 300-1500 ms that buys
        # nothing on a blank frame.
        #
        # Cosmos-Reason2-2B answered a featureless grey frame and a frame of
        # pixel noise with the SAME confident sentence about a single-lane
        # asphalt road. See config.LOCAL_VISION_BLANK_STD for the numbers.
        try:
            structure = frame_structure(pil)
        except Exception:
            structure = None          # never fail a reading on the guard
        if structure is not None and structure < config.LOCAL_VISION_BLANK_STD:
            _flags["total"] += 1
            _flags["blank_frame"] += 1
            if _flags["blank_frame"] in (1, 10, 100):
                print(f"[vision] no reading -- the frame has nothing in it "
                      f"(structure {structure:.2f} < "
                      f"{config.LOCAL_VISION_BLANK_STD}): a covered lens, a "
                      f"dead camera or a black frame, not a road "
                      f"({_flags['blank_frame']} so far)", flush=True)
            clear_holder()
            return ""
        msgs = [{"role": "user", "content": [
            {"type": "image", "image": pil},
            {"type": "text", "text": TEACHER_PROMPT},
        ]}]
        inputs = _processor.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(_model.device)
        out = _model.generate(**inputs, **_generate_kwargs())
        raw = _processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
        )[0].strip()
        _flags["total"] += 1
        # A REASONING TRACE IS NOT A READING. Stripped, counted, and if it never
        # closed there is nothing after it to publish.
        text, had_trace, unterminated = _strip_think(raw)
        if had_trace:
            _flags["think_trace"] += 1
        if unterminated:
            _flags["think_unterminated"] += 1
            if _flags["think_unterminated"] in (1, 10, 100):
                print(f"[vision] reading refused -- reasoning trace filled the "
                      f"budget ({_flags['think_unterminated']} so far)",
                      flush=True)
            clear_holder()
            return ""
        # THE MODEL DID NOT LOOK. It completed the prompt's example list
        # instead, which is a sentence about a road that is not there and which
        # reads as a perfectly good answer -- it was written to. Refused here,
        # at the one place every consumer goes through, rather than by each of
        # them separately: an empty string is a thing every caller already
        # handles (it means "no observation"), and the honest slow path picks
        # it up. See rio_prompts.is_prompt_example for the measurement.
        if is_prompt_example(text):
            global _parroted
            _parroted += 1
            _flags["prompt_example"] += 1
            if _parroted in (1, 10, 100):
                print(f"[vision] observation refused -- prompt example verbatim "
                      f"({_parroted} so far): {text!r}", flush=True)
            clear_holder()
            return ""
        # IS THIS A READING OR A RECITATION? The guard written for the teachers,
        # run here on every frame, because the fault it was written for --
        # Alpamayo answering eight of ten keyframes with a training label -- is
        # one a 2B is more prone to rather than less. Repetition is tracked
        # across frames, which is the only way the byte-identical case is
        # visible at all.
        repeats = _repeats.note(config.LOCAL_VISION_MODEL, "reading", text)
        refusal = reading_refused(text, repeats)
        if refusal:
            _flags["canned"] += 1
            if _flags["canned"] in (1, 10, 100):
                print(f"[vision] reading refused -- {refusal.get('why')}: "
                      f"{text[:120]!r}", flush=True)
            clear_holder()
            return ""
        # Repetition alone is a hint and is NOT a refusal: on an unchanging
        # road the same reading may simply be right again. Counted so the
        # rate can be read, and left to be published.
        if repeats >= canned.REPEAT_FLOOR:
            _flags["repeated"] += 1
            if _flags["repeated"] in (1, 10, 100):
                print(f"[vision] reading repeated x{repeats} (not refused): "
                      f"{text[:80]!r}", flush=True)
        # AN INSTRUMENT DOES NOT ADVISE. A reading that tells the driver to slow
        # down has written a warning, and warnings on this car come from measured
        # geometry with a band and a lead behind them, never from a caption.
        if not config.local_vision_speaks_directly():
            advisory = sensor_faults(text)
            if advisory:
                _flags["advisory"] += 1
                if _flags["advisory"] in (1, 10, 100):
                    print(f"[vision] reading refused -- advisory "
                          f"({advisory}): {text[:120]!r}", flush=True)
                clear_holder()
                return ""
        _last_observation = text
        _last_observed_at = time.time()
        _last_observed_frame = frame_id
        _last_lock_wait_ms = _lock_wait_ms
        clear_holder()
        return text


def warm() -> None:
    """Preload weights and compile CUDA kernels so the first real /observe is fast.

    Without this the first frame of a drive pays ~16s of cold-load while the
    5s capture loop is already running.
    """
    if not config.VISION_ENABLED:
        return
    with _lock:
        _ensure_loaded()
        # One throwaway forward pass: loading the weights is most of the cost,
        # but the first generate() also pays kernel warmup. Deliberately does
        # NOT write _last_observation — RIO must not read a dummy frame as if
        # it were the road ahead.
        try:
            pil = Image.new("RGB", (64, 64), (0, 0, 0))
            msgs = [{"role": "user", "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": TEACHER_PROMPT},
            ]}]
            inputs = _processor.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=True,
                return_dict=True, return_tensors="pt",
            ).to(_model.device)
            # THE FULL BUDGET, WITH THE REAL ARGUMENTS, and not max_new_tokens=1.
            #
            # Whatever the observer's first call would pay, this pays instead.
            # That mattered most while the static KV cache was on: it compiles
            # for the shape it is first used at, MEASURED AT 36.8 s cold and
            # 16.1 s warm, and a one-token warm-up left all of it on the first
            # frame of a drive -- the one place in this system where 37 seconds
            # is unsurvivable. The static cache is off now (see
            # config.LOCAL_VISION_CACHE) and this is still the right shape: the
            # warm-up makes the same call the observer will make, whatever that
            # turns out to cost, which is why _generate_kwargs() exists.
            t_gen = time.time()
            _model.generate(**inputs, **_generate_kwargs())
            print(f"[vision] warm generate {(time.time() - t_gen):.1f}s "
                  f"({_generate_kwargs()})", flush=True)
        except Exception as e:
            print("[vision] warm inference skipped:", e)


def parroted() -> int:
    """How many observations were refused as prompt examples. 0 is the only
    healthy value; anything else means the model is not reading the frame."""
    return _parroted


def get_observation() -> str:
    """Called by llm_interface on each user turn. Returns the cached observation."""
    return _last_observation if config.VISION_ENABLED else ""


def observation() -> dict:
    """The cached observation WITH its age. -> {text, at, age_s, frame_id}

    The age is the whole point. Any caller that would speak this to a driver
    has to be able to decide whether it is still true, and a bare string cannot
    be asked. `text` is empty when nothing has been observed.
    """
    if not config.VISION_ENABLED:
        return {"text": "", "at": 0.0, "age_s": None, "frame_id": None}
    at = _last_observed_at
    return {
        "text": _last_observation,
        "at": at,
        "age_s": (time.time() - at) if at else None,
        "frame_id": _last_observed_frame,
    }


def set_observation(text: str) -> None:
    """Publish an observation produced by another path (e.g. /perceive).

    Drive mode calls /perceive instead of /observe, so without this the cache
    that get_observation() serves to RIO would never refresh and she would talk
    about the road as if the last /observe frame were still current.
    """
    global _last_observation, _last_observed_at
    if text:
        _last_observation = text
        _last_observed_at = time.time()


def model_label() -> str:
    """Which model this process actually has resident.

    ASKED, NEVER STATED, and for the reason the voice label learned on
    2026-09-20: a card that names a model from a literal is a card that lies the
    first time the model changes. /health, the perception card and the drive log
    all read this.
    """
    return config.local_vision_label()


def model_info() -> dict:
    """What is loaded, what it is being asked, and whether it may speak."""
    return {
        "role": config.LOCAL_VISION_MODEL,
        "model_id": MODEL_ID,
        "label": config.local_vision_label(),
        "loaded": _model is not None,
        "prompt": ("observer" if config.local_vision_speaks_directly()
                   else "sensor"),
        # WHETHER ITS WORDS MAY REACH A SPEAKER AS HERS. The one field that
        # says, in a form a test can assert, that the local model is a sensor.
        "speaks_directly": config.local_vision_speaks_directly(),
        "max_new_tokens": int(config.LOCAL_VISION_MAX_TOKENS),
        "flags": flag_rate(),
    }


def get_handles():
    """(processor, model, lock) for the one loaded local VLM.

    Lets headway.anchor ground boxes on the model the app already has resident
    instead of pulling a second copy. The load happens under `_lock` and the
    lock is then released before returning: callers take it themselves around
    generate(), and threading.Lock is not reentrant.
    """
    with _lock:
        _ensure_loaded()
    return _processor, _model, _lock
