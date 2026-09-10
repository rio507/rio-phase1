"""The shape of a keyframe, a reading and a corpus row — written down once.

WHY A SCHEMA MODULE AND NOT JUST DICTS
--------------------------------------
Because the point of the recording is that it is still readable in six months,
by something that is not this code. A drive recorded today has to be loadable
as a training corpus after the dashboard has been redesigned, after a model has
been swapped, and after whoever wrote it has forgotten which key held the
Chain-of-Causation. So the row format is declared here, versioned, and asserted
by tools/teacher_selftest.py against real rows -- rather than being whatever
json.dumps happened to be handed.

THE VERSION IS A PROMISE ABOUT READERS, NOT WRITERS. Bump MINOR when a field is
added (an old reader still works). Bump MAJOR when a field changes meaning or
goes away (an old reader must be told to stop). Never re-use a key for a
different thing at the same major.
"""

SCHEMA_VERSION = "1.0"

# --- one keyframe row, top level -------------------------------------------
# Every key here is REQUIRED and is asserted present by the corpus selftest.
# "Required" includes "explicitly null": a row whose ego history could not be
# built carries ego: null, and that is a fact about the drive worth training
# on. A missing key is a bug; a null value is data.
KEYFRAME_FIELDS = (
    "schema",           # str  — SCHEMA_VERSION at the time of writing
    "kf_id",            # str  — "<session>-<seq>", unique within a drive
    "seq",              # int  — monotonic per drive, no gaps
    "session_id",       # str  — the drive this belongs to
    "t0_wall",          # float — server epoch seconds of the newest frame
    "t0_session",       # float|null — headway's own session clock at t0
    "trigger",          # str  — why this instant: floor | band | nav | imu |
                        #        question | manual
    "window",           # dict — see WINDOW_FIELDS
    "ego",              # dict|null — see EGO_FIELDS
    "state",            # dict — RIO's deterministic state at t0, see STATE_FIELDS
    "spoken",           # dict|null — what RIO said near t0, if anything
    "readings",         # dict — model name -> READING_FIELDS
    "associations",     # dict — model name -> ASSOCIATION_FIELDS
    "tracks",           # list — the RF-DETR scene at t0, for the association
                        #        to be checkable without the pictures
    "models_ran",       # list — which models actually produced a reading for
                        #        this keyframe. USUALLY BOTH, and deliberately
                        #        not always: the two teachers have their own
                        #        queues and a slow one drops keyframes a fast
                        #        one runs. A row with one name here still
                        #        carries a real reading of a real window and is
                        #        worth keeping; it is just not a COMPARISON,
                        #        and an analysis that wants comparisons filters
                        #        on len(models_ran) == 2 rather than
                        #        discovering the difference by surprise.
)

# --- the 4-frame window ----------------------------------------------------
WINDOW_FIELDS = (
    "n",                # int  — how many slots were filled (4 when healthy)
    "step_s",           # float — nominal spacing, config.TEACHER_WINDOW_STEP_S
    "exact",            # bool — did every frame land inside the slot tolerance
    "spacing_error_ms", # float — worst |actual - nominal| over the window
    "frames",           # list of dicts: frame_id, wall_t, t, age_s, w, h,
                        #                slot_s (nominal offset from t0),
                        #                path (relative, or null if not kept)
    "origin",           # str  — the ring's own origin stamp: whose camera
)

# --- ego-motion history ----------------------------------------------------
# 16 poses at 10 Hz ending at t0, in the ego frame AT t0 -- so the last row is
# always the origin and identity. Frame convention is FLU: x forward, y left,
# z up, yaw about z. Stated here because it is the one thing a corpus reader
# cannot recover from the numbers.
EGO_FIELDS = (
    "steps",            # int   — 16
    "step_s",           # float — 0.1
    "frame",            # str   — "FLU_at_t0"
    "xyz",              # list[16][3] — metres
    "yaw",              # list[16]    — radians, yaw only (z-up)
    "source",           # dict  — where each ingredient came from, see below
    "quality",          # dict  — n_samples, max_gap_s, extrapolated, imu
)
# source: {"speed": "obd"|"gps"|"manual"|"none",
#          "yaw_rate": "imu"|"gps_heading"|"none",
#          "position": "dead_reckoned"|"gps"}

# --- RIO's own state at t0 -------------------------------------------------
# Copied verbatim out of the headway result that produced the keyframe. This is
# the ground truth column of the corpus: whatever the two models say, THIS is
# what RIO computed at that instant and what its warnings rested on.
STATE_FIELDS = (
    "band",             # str   — NORMAL | GETTING_UNSAFE | UNSAFE | ...
    "gap_m",            # float|null — distance_m, null when not stood behind
    "ttc_s",            # float|null
    "tau_s",            # float|null
    "trend",            # str|null
    "speed_ms",         # float|null
    "speed_source",     # str|null — obd | gps | gps_coasted | none
    "speed_degraded",   # bool|null
    "lead_id",          # int|null — the RF-DETR track holding the lead lock
    "corridor_source",  # str|null — ufld | static
    "lane_conf",        # float|null
)

# --- one model's reading ---------------------------------------------------
# EVERY FIELD THE MODEL OFFERS, VERBATIM. `raw` is the model's own output
# before any of this code touched it, and it is what a training run should read
# -- the parsed fields beside it are for the dashboard and may be wrong in ways
# the raw text is not.
READING_FIELDS = (
    "model",            # str   — "alpamayo1.5" | "cosmos-reason2"
    "model_id",         # str   — the HF repo id
    "revision",         # str   — the pinned commit sha of the weights
    "precision",        # str   — "bf16" | "fp8"
    "ok",               # bool
    "error",            # str|null
    "latency_ms",       # float — the service's own measurement, load excluded
    "queue_ms",         # float — how long the job waited on this side
    "freshness_s",      # float — t0 to the moment the reading came back
    "raw",              # dict  — every generation verbatim, keyed by what was
                        #         asked. Never trimmed, never cleaned.
    "scene",            # str   — answer to TEACHER_PROMPTS["scene"]
    "critical_actor",   # str   — answer to TEACHER_PROMPTS["critical_actor"]
    "attention",        # str   — answer to TEACHER_PROMPTS["attention"]
    "reasoning",        # str   — Alpamayo: the Chain-of-Causation trace.
                        #         Cosmos: the physical-reasoning answer.
    "thinking",         # str|null — Cosmos's <think> block. Alpamayo has no
                        #         separate one; its reasoning IS the trace.
    "meta_action",      # str|null — Alpamayo only
    "trajectory",       # dict|null — Alpamayo only: 6.4 s, 64 points at 10 Hz
                        #         {"xyz": [[x,y,z] * 64], "hz": 10,
                        #          "horizon_s": 6.4, "frame": "FLU_at_t0",
                        #          "pixels": [[u,v] ...] | null}
                        #         DISPLAY ONLY. Nothing downstream reads it.
    "gpu",              # dict  — {"vram_mb": float, "device": str}
    "extra",            # dict  — everything else the service reported, kept
                        #         verbatim. Per-stage timings, cameras: 1,
                        #         frames_as, ego_synthetic, code_revision. This
                        #         field exists so "every field the model
                        #         offers" survives the schema not having
                        #         anticipated one.
)

# --- actor association -----------------------------------------------------
# The model named a road user in words; the deterministic pipeline has tracks
# with classes and boxes. This is the bridge, and it is allowed to fail.
ASSOCIATION_FIELDS = (
    "matched",          # bool
    "track_id",         # int|null — the RF-DETR candidate id
    "label",            # str|null — that track's class
    "box",              # list|null — [x1,y1,x2,y2] in frame pixels at t0
    "range_m",          # float|null
    "score",            # float — 0..1, how well the phrase fit the track
    "reason",           # str   — why it matched, or why it did not
    "phrase",           # str   — the noun phrase that was matched on
    "wanted_class",     # str|null — the class the phrase asked for
    "wanted_side",      # str|null — left | right | ahead | null
)

MODELS = ("alpamayo1.5", "cosmos-reason2")

TRIGGERS = ("floor", "band", "nav", "imu", "question", "manual")


def blank_reading(model: str, model_id: str = "", revision: str = "",
                  precision: str = "") -> dict:
    """A reading that says nothing, honestly. Every key present, all empty.

    Used for a model that is disabled, unreachable or still loading, so the
    dashboard and the corpus both see the same shape whether or not there was
    an answer -- "no reading" is a state of the panel, not a missing column.
    """
    return {
        "model": model, "model_id": model_id, "revision": revision,
        "precision": precision, "ok": False, "error": None,
        "latency_ms": 0.0, "queue_ms": 0.0, "freshness_s": 0.0,
        "raw": {}, "scene": "", "critical_actor": "", "attention": "",
        "reasoning": "", "thinking": None, "meta_action": None,
        "trajectory": None, "gpu": {}, "extra": {},
    }


def validate_row(row: dict) -> list:
    """-> list of complaints, empty when the row is a valid corpus row.

    Deliberately a list rather than an exception: the selftest wants to report
    every fault in a row at once, and the writer never validates -- a row that
    is wrong should still be written, or the evidence of the bug goes with it.
    """
    bad = []
    for key in KEYFRAME_FIELDS:
        if key not in row:
            bad.append(f"missing top-level key: {key}")
    if row.get("schema") != SCHEMA_VERSION:
        bad.append(f"schema {row.get('schema')!r} != {SCHEMA_VERSION!r}")
    if not isinstance(row.get("models_ran"), list):
        bad.append("models_ran is not a list")
    if row.get("trigger") not in TRIGGERS:
        bad.append(f"trigger {row.get('trigger')!r} not one of {TRIGGERS}")

    win = row.get("window") or {}
    for key in WINDOW_FIELDS:
        if key not in win:
            bad.append(f"window missing: {key}")

    state = row.get("state") or {}
    for key in STATE_FIELDS:
        if key not in state:
            bad.append(f"state missing: {key}")

    ego = row.get("ego")
    if ego is not None:
        for key in EGO_FIELDS:
            if key not in ego:
                bad.append(f"ego missing: {key}")
        xyz = ego.get("xyz") or []
        if len(xyz) != ego.get("steps"):
            bad.append(f"ego xyz has {len(xyz)} rows, steps says {ego.get('steps')}")

    readings = row.get("readings") or {}
    for model in MODELS:
        if model not in readings:
            bad.append(f"readings missing model: {model}")
            continue
        for key in READING_FIELDS:
            if key not in readings[model]:
                bad.append(f"readings[{model}] missing: {key}")

    assoc = row.get("associations") or {}
    for model in MODELS:
        if model not in assoc:
            bad.append(f"associations missing model: {model}")
            continue
        for key in ASSOCIATION_FIELDS:
            if key not in assoc[model]:
                bad.append(f"associations[{model}] missing: {key}")
    return bad
