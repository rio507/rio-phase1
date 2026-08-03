"""Verification for the canonical vehicle data layer.

    python -m vehicle.selftest

Seven parts, separated by what each one is able to prove:

  A. REGISTRY — the alias in both directions, and the promise that no existing
     internal id was renamed. This is the part that would fail silently: a
     renamed id does not raise, it just stops matching four weeks of baselines
     on disk, and the drift detector reports "not enough data" forever.

  B. SCHEMA — validation, conversion at the boundary, and the three shapes that
     must survive rather than be dropped: an unknown signal, an implausible
     value, and a clock running ahead of the cloud's.

  C. GATEWAY — registration refused when unconfigured, tokens stored hashed and
     never returned twice, rotation, revocation, rate limiting, and a gateway
     kept to the vehicle it was registered against.

  D. INGEST — accepted / duplicate / rejected per event, an identical batch
     acknowledged rather than re-stored, and out-of-order arrivals not
     overwriting fresher readings.

  E. ONE PIPELINE — the claim the whole package exists to make. Ingested data
     and mock data produce the same rows, judged by the same bands, with no
     branch anywhere in between.

  F. READ-ONLY POSTURE — asserted by parsing the source, not by trusting a
     comment. No Mode 04, no clear-codes route, no actuator test, no ECU write,
     no Holley transmit, and no command channel a bridge could poll.

  G. NO SECOND SOURCE OF TRUTH — the interpretation layer stayed where it was:
     no threshold, no severity and no band lives in this package.

No GPU, no models, no network, ~1 s, so it can gate a commit.
"""
import ast
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, "/workspace/rio-phase1")

import config                                          # noqa: E402

# A bootstrap key, so registration is testable. The real one comes from the
# environment and is absent in CI, which is exactly the case part C asserts.
config.VEHICLE_GATEWAY_REGISTRATION_KEY = "selftest-bootstrap-key"

import telemetry                                       # noqa: E402
from vehicle import ingest                             # noqa: E402
from vehicle.gateway import auth                       # noqa: E402
from vehicle.providers import ingested                 # noqa: E402
from vehicle.signals import provenance as P            # noqa: E402
from vehicle.signals import quality as Q               # noqa: E402
from vehicle.signals import registry as R              # noqa: E402
from vehicle.signals import schema as S                # noqa: E402
from vehicle.signals import units as U                 # noqa: E402

ROOT = "/workspace/rio-phase1"
_results = []
_TMP = tempfile.mkdtemp(prefix="vehicle_selftest_")
auth.reset_for_test(os.path.join(_TMP, "gateways.json"))

VID = config.VEHICLE_ID


def check(condition, label, detail=""):
    _results.append((bool(condition), label, detail))
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}"
          + (f"  -- {detail}" if detail else ""))
    return bool(condition)


def head(title):
    print(f"\n{title}")


def _fresh_gateway(name="rio-obd-selftest"):
    ingest.reset_for_test()
    return auth.register(name, VID, config.VEHICLE_GATEWAY_REGISTRATION_KEY,
                         hardware_type="canable_2", firmware_type="candlelight",
                         bridge_version="0.1.0")


def _event(signal, value, at, gid, **kw):
    kw.setdefault("source_type", S.OBD2_CAN)
    st = kw.pop("source_type")
    return S.make_event(signal, value, at, st, vehicle_id=VID,
                        gateway_id=gid, **kw)


# ===========================================================================
# A. Registry
# ===========================================================================

def run_registry():
    head("A -- the alias registry: canonical on the wire, flat inside")

    check(R.canonical("coolant_temp") == "powertrain.engine.coolant_temperature",
          "flat id -> canonical name", str(R.canonical("coolant_temp")))
    check(R.internal("powertrain.engine.coolant_temperature") == "coolant_temp",
          "canonical name -> flat id",
          str(R.internal("powertrain.engine.coolant_temperature")))
    bad = [s.name for s in R.SPECS
           if s.telemetry_id and R.canonical(s.telemetry_id) != s.name]
    check(not bad, "and the round trip holds for every registered signal", str(bad))

    # THE check this file exists for. Renaming an internal id does not raise; it
    # silently orphans training_data/vehicle/baselines.json, which is keyed by
    # the old names, and the drift detector then reports "not enough data"
    # forever with nothing to say it changed.
    missing = [s.telemetry_id for s in R.SPECS
               if s.telemetry_id and s.telemetry_id not in telemetry.SPEC_BY_ID]
    check(not missing,
          "every registered flat id is still a real telemetry channel",
          str(missing))

    # ...and the other direction: a channel the panel shows that the registry
    # has never heard of cannot be ingested, which would make it mock-only.
    unregistered = [i for i in telemetry.SPEC_BY_ID
                    if R.canonical(i) is None]
    check(not unregistered,
          "and every telemetry channel has a canonical name to arrive under",
          str(unregistered))

    # Every config key that names a channel must still name one. This is the
    # same orphaning failure seen from the config side.
    keys = set(config.TELEMETRY_BANDS) | set(config.TELEMETRY_TREND_DELTA) \
        | set(config.TELEMETRY_MODES) | set(config.INSIGHTS_DEVIATION_DELTA) \
        | set(config.INSIGHTS_DRIFT_DELTA)
    tire_bases = {"tire_pressure", "tire_temp"}
    orphans = [k for k in keys
               if k not in telemetry.SPEC_BY_ID and k not in tire_bases]
    check(not orphans, "no config threshold points at a channel that is gone",
          str(orphans))

    # The seeded baselines already on disk still resolve. These are the four
    # weeks of history the drift detector runs on.
    seeded = ("battery_voltage", "coolant_temp", "oil_pressure", "fuel_pressure",
              "oil_temp", "intake_air_temp", "afr_wideband")
    lost = [k for k in seeded if k not in telemetry.SPEC_BY_ID]
    check(not lost, "and every channel the existing baselines are keyed by",
          str(lost))

    check(R.by_pid("0105").telemetry_id == "coolant_temp",
          "a standard PID resolves to a channel", str(R.by_pid("0105")))
    check(len(R.obd_signals()) >= 11,
          f"the initial OBD-II set is present ({len(R.obd_signals())} signals)")
    check(R.spec("powertrain.engine.oil_pressure").obd_pid == "",
          "and a Holley-only channel carries no PID — standard OBD-II will "
          "never produce it")


def run_units():
    head("A -- units convert once, at the boundary, and never guess")

    check(abs(U.c_to_f(100.0) - 212.0) < 1e-9, "100 C -> 212 F", str(U.c_to_f(100.0)))
    check(abs(U.kph_to_mph(100.0) - 62.1371) < 1e-3, "100 km/h -> 62.1 mph")
    check(U.c_to_f(None) is None,
          "and None survives conversion — 'the sensor did not answer' must not "
          "become 32 degrees")

    try:
        U.convert(1.0, "furlongs", U.PSI)
        ok = False
    except U.UnitError:
        ok = True
    check(ok, "an undefined conversion raises rather than guessing")

    # RIO's canonical temperature is Fahrenheit, because every band in config.py
    # is. This is the assertion that stops a future edit from quietly moving it.
    check(R.spec("powertrain.engine.coolant_temperature").unit == U.FAHRENHEIT,
          "coolant is canonically Fahrenheit, matching config.TELEMETRY_BANDS")
    check(R.spec("powertrain.engine.coolant_temperature").source_unit == U.CELSIUS,
          "and declares Celsius as what OBD-II actually sends")
    check(R.spec("vehicle.speed").unit == U.MPH
          and R.spec("vehicle.speed").source_unit == U.KPH,
          "speed is canonically mph, sent as km/h")


# ===========================================================================
# B. Schema
# ===========================================================================

def run_schema():
    head("B -- the canonical event: validated, converted, and never silently lost")

    now = time.time()
    gw = _fresh_gateway()
    gid = gw["gateway_id"]

    e, errs = S.normalize_event(
        _event("powertrain.engine.coolant_temperature", 96.0, now, gid,
               unit="celsius", source_signal="0105", source_ecu="0x7E8"), now)
    check(e is not None and not errs, "a well-formed event validates", str(errs))
    check(e and abs(e["value"] - 204.8) < 0.1,
          "96 C arrives as 204.8 F — converted once, on the way in",
          str(e["value"]) if e else "")
    check(e and e["unit"] == "fahrenheit", "and its unit is rewritten to match")
    check(e and e["observed_at"] and e["received_at"],
          "both timestamps are present — one for trends, one for latency")
    check(e and e["provenance"] == P.ECU_MEASUREMENT,
          "an OBD source defaults to ecu_measurement", e["provenance"] if e else "")

    e, errs = S.normalize_event(
        _event("powertrain.engine.coolant_temperature", 9999.0, now, gid,
               unit="celsius"), now)
    check(e is not None, "an implausible value is KEPT, not dropped")
    check(e and e["quality"] == Q.INVALID_RANGE,
          "and labelled invalid_range — 'the decoder produced nonsense at 14:22' "
          "is itself a finding", e["quality"] if e else "")

    e, errs = S.normalize_event(
        _event("manufacturer.proprietary.thing", 7.0, now, gid), now)
    check(e is not None, "an unregistered signal is stored, never discarded — "
                         "manufacturer channels are exactly this")
    check(e and e["known_signal"] is False and e["quality"] == Q.UNVERIFIED_DECODER,
          "and marked unverified rather than valid", e["quality"] if e else "")

    e, errs = S.normalize_event(
        _event("powertrain.engine.rpm", 800.0, now + 3600.0, gid), now)
    check(e is not None, "an event from the future is kept")
    check(e and e["metadata"].get("clock_skew_s"),
          "with the skew recorded rather than corrected — rewriting it would "
          "destroy the only evidence the gateway's clock is wrong",
          str(e["metadata"]) if e else "")

    e, errs = S.normalize_event({"signal": "x", "source_type": "obd2_can"}, now)
    check(e is None, "an event with no timestamp is refused outright", str(errs))
    e, errs = S.normalize_event(
        _event("powertrain.engine.rpm", 1.0, now, gid, source_type="not_a_source"),
        now)
    check(e is None, "and so is an unknown source_type", str(errs))

    check(S.from_iso(S.to_iso(now)) is not None
          and abs(S.from_iso(S.to_iso(now)) - now) < 0.01,
          "ISO-8601 round trips to the epoch floats the codebase runs on")


# ===========================================================================
# C. Gateway
# ===========================================================================

def run_gateway():
    head("C -- gateways: credentials that are real, and refused when absent")

    saved = config.VEHICLE_GATEWAY_REGISTRATION_KEY
    config.VEHICLE_GATEWAY_REGISTRATION_KEY = ""
    try:
        auth.register("x", VID, "anything")
        ok = False
    except auth.AuthError:
        ok = True
    check(ok, "with no bootstrap key configured, registration is REFUSED — an "
              "unconfigured server that admits any device fails silently")
    config.VEHICLE_GATEWAY_REGISTRATION_KEY = saved

    try:
        auth.register("x", VID, "wrong-key")
        ok = False
    except auth.AuthError:
        ok = True
    check(ok, "a wrong bootstrap key is refused")

    gw = _fresh_gateway("rio-obd-cred-test")
    gid, token = gw["gateway_id"], gw["token"]
    check(len(token) >= 32, "the issued token is long enough to be a secret",
          str(len(token)))

    rec = auth.authenticate(gid, token)
    check(rec["gateway_id"] == gid, "and it authenticates")
    check("token" not in auth.public(rec) and "token_sha256" not in auth.public(rec),
          "the public view carries no credential and no hash",
          str(sorted(auth.public(rec))))

    on_disk = open(os.path.join(_TMP, "gateways.json")).read()
    check(token not in on_disk,
          "and the token is NOT in the registry file — that file has to be safe "
          "to attach to a bug report")

    try:
        auth.authenticate(gid, "not-the-token")
        ok = False
    except auth.AuthError as e:
        ok = "unknown gateway or invalid token" in str(e)
    check(ok, "a wrong token and an unknown gateway give the SAME message — a "
              "caller that can tell them apart can enumerate gateway ids")

    rotated = auth.rotate_token(gid)
    try:
        auth.authenticate(gid, token)
        ok = False
    except auth.AuthError:
        ok = True
    check(ok, "rotation invalidates the old token immediately")
    check(auth.authenticate(gid, rotated["token"])["gateway_id"] == gid,
          "and the new one works")

    try:
        auth.authorize_vehicle(auth.authenticate(gid, rotated["token"]),
                               "some_other_vehicle")
        ok = False
    except auth.AuthError:
        ok = True
    check(ok, "a gateway may only speak about the vehicle it registered against")

    hb = auth.heartbeat(gid, {"can_state": "active", "network_state": "connected",
                              "outbox_pending": 12, "vehicle_state": "driving",
                              "bridge_version": "0.1.0"})
    check(hb["link"] == "connected", "a fresh heartbeat reads as connected")
    check(hb["heartbeat"]["outbox_pending"] == 12,
          "and the bridge's own view is recorded verbatim")

    auth.revoke(gid)
    try:
        auth.authenticate(gid, rotated["token"])
        ok = False
    except auth.AuthError as e:
        ok = "revoked" in str(e)
    check(ok, "a revoked gateway is refused")
    check(any(g["gateway_id"] == gid for g in auth.gateways()),
          "but is not forgotten — the drives it uploaded are still the "
          "vehicle's history")

    # Rate limiting, driven directly so the bucket can be emptied exactly.
    gw2 = _fresh_gateway("rio-obd-rate-test")
    t = 1_000_000.0
    spent = 0
    try:
        for _ in range(int(config.VEHICLE_INGEST_BURST) + 5):
            auth.check_rate(gw2["gateway_id"], now=t)
            spent += 1
        ok = False
    except auth.AuthError:
        ok = True
    check(ok, f"the token bucket empties after ~{int(config.VEHICLE_INGEST_BURST)} "
              f"batches at one instant", f"{spent} allowed")
    auth.check_rate(gw2["gateway_id"], now=t + 120.0)
    check(True, "and refills with time")


# ===========================================================================
# D. Ingest
# ===========================================================================

def run_ingest():
    head("D -- ingestion: per-event results, and a retry that is not a duplicate")

    gw = _fresh_gateway("rio-obd-ingest-test")
    gid, token = gw["gateway_id"], gw["token"]
    now = time.time()

    evs = [
        _event("powertrain.engine.rpm", 2150.0, now, gid, source_signal="010C"),
        _event("vehicle.speed", 100.0, now, gid, unit="kilometre_per_hour"),
        _event("powertrain.engine.coolant_temperature", 96.0, now, gid,
               unit="celsius"),
    ]
    broken = dict(evs[0])
    broken["event_id"] = S.new_id()
    broken["observed_at"] = "not a timestamp"
    batch = S.make_batch(evs + [broken], vehicle_id=VID, gateway_id=gid,
                         sent_at=now)
    res = ingest.ingest_batch(batch, gid, token)
    check(res["accepted"] == 3, "three good events stored", str(res["accepted"]))
    check(res["rejected"] == 1, "one malformed event refused", str(res["rejected"]))
    check(len(res["results"]) == 4, "with a result for every event, not a verdict "
                                    "for the batch")
    check(any(r["status"] == "rejected" and r["errors"] for r in res["results"]),
          "and the refusal says why")

    res2 = ingest.ingest_batch(batch, gid, token)
    check(res2["idempotent_replay"] is True,
          "an identical batch is acknowledged, not re-stored — a bridge that got "
          "an error for a batch it already delivered would retry forever")

    # A different batch carrying the same events: the durable guarantee.
    batch3 = S.make_batch(evs, vehicle_id=VID, gateway_id=gid, sent_at=now)
    batch3["batch_id"] = S.new_id()
    res3 = ingest.ingest_batch(batch3, gid, token)
    check(res3["duplicates"] == 3 and res3["accepted"] == 0,
          "and repeated event ids in a NEW batch are deduplicated by event_id",
          f"acc={res3['accepted']} dup={res3['duplicates']}")

    # Out of order. This is the normal consequence of an outbox working, not a
    # corner case: a bridge emptying a backlog delivers old observations while
    # the live stream continues.
    older = _event("powertrain.engine.rpm", 700.0, now - 60.0, gid)
    b4 = S.make_batch([older], vehicle_id=VID, gateway_id=gid, sent_at=now)
    res4 = ingest.ingest_batch(b4, gid, token)
    check(res4["accepted"] == 1, "a late event is accepted")
    check(res4["superseded_by_newer"] == 1,
          "and does NOT become current — otherwise every reconnection would "
          "look like the engine suddenly dropping to idle")
    cur = ingested.buffer().latest()["powertrain.engine.rpm"]
    check(cur["value"] == 2150.0, "the fresher reading still stands",
          str(cur["value"]))
    win = ingested.buffer().window(now - 120.0, now + 1.0,
                                   ["powertrain.engine.rpm"])
    check(len(win) == 2, "but the late one is in the history, where the "
                         "early-fault snapshot will need it", str(len(win)))

    try:
        ingest.ingest_batch(batch, "gw_nope", "nope")
        ok = False
    except ingest.IngestError as e:
        ok = e.status == 401
    check(ok, "bad credentials are a 401 before any event is looked at")

    big = S.make_batch([_event("powertrain.engine.rpm", 1.0, now, gid)]
                       * (config.VEHICLE_INGEST_MAX_EVENTS + 1),
                       vehicle_id=VID, gateway_id=gid)
    try:
        ingest.ingest_batch(big, gid, token)
        ok = False
    except ingest.IngestError as e:
        ok = e.status == 413
    check(ok, "an oversized batch is refused WHOLE — a half-accepted batch is "
              "the one shape an outbox cannot reason about")


# ===========================================================================
# E. One pipeline
# ===========================================================================

def run_one_pipeline():
    head("E -- ingested data and mock data are judged by the same code")

    telemetry.set_source("mock_holley")
    telemetry.set_scenario("cruise")
    mock = telemetry.snapshot(record=False)
    mock_ids = [r["id"] for r in mock["rows"]]

    gw = _fresh_gateway("rio-obd-pipeline-test")
    gid, token = gw["gateway_id"], gw["token"]
    now = time.time()

    # A coolant temperature past the warning band, arriving in Celsius from a
    # bridge. 110 C is 230 F, and config.TELEMETRY_BANDS warns at 225.
    evs = [
        _event("powertrain.engine.coolant_temperature", 110.0, now, gid,
               unit="celsius", source_signal="0105", source_ecu="0x7E8"),
        _event("powertrain.engine.rpm", 1850.0, now, gid),
        _event("vehicle.speed", 100.0, now, gid, unit="kilometre_per_hour"),
        _event("powertrain.fuel.long_term_trim_bank_1", 2.0, now, gid),
    ]
    ingest.ingest_batch(S.make_batch(evs, vehicle_id=VID, gateway_id=gid),
                        gid, token)

    telemetry.set_source("live_obd")
    live = telemetry.snapshot(record=False)
    live_ids = [r["id"] for r in live["rows"]]

    check(mock_ids == live_ids,
          "both sources produce the same rows, in the same order — the panel "
          "cannot tell which is live")

    row = next(r for r in live["rows"] if r["id"] == "coolant_temp")
    check(row["value_text"] == "230", "110 C shows as 230 F on the panel",
          row["value_text"])
    check(row["status"] == "WARNING",
          "and is judged WARNING by config.TELEMETRY_BANDS — the same band the "
          "mock is judged by", row["status"])
    check("225" in row["detail"],
          "against the same threshold, with the same wording", row["detail"])

    speed = next(r for r in live["rows"] if r["id"] == "vehicle_speed")
    check(speed["value_text"] == "62", "100 km/h shows as 62 mph", speed["value_text"])

    unseen = next(r for r in live["rows"] if r["id"] == "oil_pressure")
    check(unseen["status"] == "NO DATA",
          "a channel this source never sent reads NO DATA, not zero — 'the "
          "sensor did not answer' and 'the sensor answered zero' are different "
          "facts about the car", unseen["status"])

    cap = ingested.buffer().capability()
    check("powertrain.engine.oil_pressure" in cap["unsupported"],
          "and the capability report says it was never supported, rather than "
          "leaving an empty row that looks like a dead sensor")

    check(telemetry.set_source("mock_holley") is True,
          "switching back needs no restart")
    check(telemetry.set_source("not_a_source") is False,
          "and an unknown source changes nothing")


# ===========================================================================
# F. Read-only posture
# ===========================================================================
# Asserted by parsing the source. A comment saying "we never clear codes" is
# worth nothing the day somebody adds a helpful endpoint; a test that reads the
# files is worth something that day.

# Every spelling of the prohibited services that could plausibly appear.
FORBIDDEN_TOKENS = (
    "mode_04", "mode04", "service_04", "clear_dtc", "cleardtc", "clear_codes",
    "clearcodes", "clear_diagnostic", "reset_readiness", "reset_monitors",
    "actuator_test", "ecu_write", "write_memory", "write_did",
    "holley_send", "holley_write", "holley_command", "send_frame",
    "transmit_frame", "write_tune",
)

SCANNED_DIRS = ("vehicle",)


def _py_files(root):
    for dirpath, _, names in os.walk(os.path.join(ROOT, root)):
        if "__pycache__" in dirpath:
            continue
        for n in sorted(names):
            if n.endswith(".py"):
                yield os.path.join(dirpath, n)


def run_read_only():
    head("F -- read-only posture, asserted by reading the source")

    hits = []
    for d in SCANNED_DIRS:
        for path in _py_files(d):
            if path.endswith("selftest.py"):
                # This file holds the list of prohibited spellings and would
                # otherwise be its own only violation.
                continue
            # Executable surface only. The prose has to be able to NAME what is
            # prohibited in order to explain that it is prohibited.
            tree = ast.parse(open(path).read())
            docstrings = set()
            for node in ast.walk(tree):
                body = getattr(node, "body", None)
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)) and body \
                        and isinstance(body[0], ast.Expr) \
                        and isinstance(body[0].value, ast.Constant) \
                        and isinstance(body[0].value.value, str):
                    docstrings.add(id(body[0].value))
            surface = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Name):
                    surface.append(node.id)
                elif isinstance(node, ast.Attribute):
                    surface.append(node.attr)
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                       ast.ClassDef)):
                    surface.append(node.name)
                elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                        and id(node) not in docstrings:
                    surface.append(node.value)
            text = " ".join(surface).lower()
            for token in FORBIDDEN_TOKENS:
                if token in text:
                    hits.append(f"{os.path.relpath(path, ROOT)}: {token}")
    check(not hits, f"no prohibited service appears in {'/'.join(SCANNED_DIRS)}",
          str(hits))

    # The routes. A prohibited capability would arrive as a path, so the paths
    # are what get read.
    app_src = open(os.path.join(ROOT, "app.py")).read()
    tree = ast.parse(app_src)
    routes = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Attribute) \
                    and dec.func.attr in ("get", "post", "put", "patch", "delete"):
                for arg in dec.args:
                    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                        routes.append((dec.func.attr, arg.value))
    check(routes, f"the route table is readable ({len(routes)} routes)")

    banned_paths = [m + " " + p for m, p in routes
                    if any(w in p.lower() for w in
                           ("clear", "erase", "reset-dtc", "actuator", "write",
                            "command", "tune"))]
    check(not banned_paths, "no route offers to clear, erase, actuate or write",
          str(banned_paths))

    # A command channel would most likely appear as something a bridge POLLS.
    # There is no such route, and this is the check that would notice one.
    poll_paths = [p for m, p in routes
                  if m == "get" and "gateway" in p.lower()
                  and any(w in p.lower() for w in ("command", "task", "queue",
                                                   "pending", "instruction"))]
    check(not poll_paths, "and no route a bridge could poll for instructions",
          str(poll_paths))

    check(not any(m == "delete" for m, _ in routes),
          "no DELETE route exists at all")


# ===========================================================================
# G. No second source of truth
# ===========================================================================

def run_no_second_truth():
    head("G -- interpretation stayed where it was")

    # The bands, the severities and the words are config.py's, telemetry.py's
    # and vehicle_health_policy.py's. A copy in this package would be a second
    # answer to the same question, and the first sign of one is a threshold
    # constant appearing here.
    offenders = []
    for path in _py_files("vehicle"):
        if path.endswith("selftest.py"):
            continue
        src = open(path).read()
        for word in ("warn_high", "crit_high", "warn_low", "crit_low",
                     "SEVERITY_RANK", "ANNOUNCE_AT_RANK"):
            if word in src:
                offenders.append(f"{os.path.relpath(path, ROOT)}: {word}")
    check(not offenders, "no band or severity ladder is duplicated in vehicle/",
          str(offenders))

    # The plausible ranges in the registry are NOT bands and must not be
    # mistaken for them: they are the bound outside which a number is not a
    # measurement. Coolant's plausible ceiling has to sit well above its
    # critical band, or a genuine overheat would be discarded as a decode error.
    spec = R.spec("powertrain.engine.coolant_temperature")
    check(spec.plausible[1] > config.TELEMETRY_BANDS["coolant_temp"]["crit_high"],
          "the plausible ceiling sits above the critical band, so a real "
          "overheat is never thrown away as a decode error",
          f"{spec.plausible[1]} > {config.TELEMETRY_BANDS['coolant_temp']['crit_high']}")

    # And the package must not reach into the conversation layer.
    for path in _py_files("vehicle"):
        if path.endswith("selftest.py"):
            continue
        tree = ast.parse(open(path).read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        bad = imported & {"openai", "llm_interface", "vehicle_health",
                          "vehicle_health_policy", "app", "telemetry"}
        if bad:
            check(False, f"{os.path.relpath(path, ROOT)} imports {sorted(bad)}")
            return
    check(True, "and nothing in vehicle/ imports telemetry, the conversation "
                "layer or a model — the dependency runs one way")


# ===========================================================================

def main():
    print("=" * 74)
    print("RIO vehicle data layer — canonical schema, gateways, ingestion")
    print("=" * 74)
    print(f"  store: {_TMP}")

    for t in (run_registry, run_units, run_schema, run_gateway, run_ingest,
              run_one_pipeline, run_read_only, run_no_second_truth):
        try:
            t()
        except Exception as e:
            import traceback
            check(False, f"{t.__name__} raised", f"{type(e).__name__}: {e}")
            traceback.print_exc()

    passed = sum(1 for ok, _, _ in _results if ok)
    total = len(_results)
    print("\n" + "=" * 74)
    print(f"{passed}/{total} checks passed")
    if passed != total:
        print("\nFAILURES:")
        for ok, name, detail in _results:
            if not ok:
                print(f"  - {name}  {detail}")
    print("=" * 74)
    shutil.rmtree(_TMP, ignore_errors=True)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
