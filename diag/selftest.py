"""Verification for the generalized diagnostic framework.

    python -m diag.selftest

tire_diag/selftest.py already proves the DISCIPLINE — that one bad reading
cannot become a confirmed fault, that not-ready is never reported as passed,
that a problem is repaired only after passing verification. Those 120 checks
still run and still pass, and they are the real guarantee that this extraction
changed no behaviour.

What this file adds is the claim that could not be made before: that the
machinery is generic. It builds a domain out of nothing — three subjects, one
monitor, one code, a temporary store — and drives it through the whole lifecycle
without importing tire_diag at all. If the framework had a tire-shaped hole left
in it, this file would not run.

     1  a domain built from scratch reaches a verdict
     2  one failing run is a CANDIDATE, not an ACTIVE issue
     3  repeated failures confirm, and freeze the evidence
     4  a confirmed issue heals only on sustained passing evidence
     5  two domains keep separate records
     6  shadow clearance is per domain, and unknown domains are shadowed
     7  NO_VERDICT never overwrites a real result
     8  the gate names the FIRST reason a monitor could not judge
     9  a domain's own words reach the monitor status
    10  freeze frames carry only the fields the code declared

Plus the LLM firewall, asserted the same way tire_diag/selftest.py asserts it
for the tire half.

No config, no GPU, no models, no network, well under a second.
"""
import ast
import os
import shutil
import sys
import tempfile

sys.path.insert(0, "/workspace/rio-phase1")

from diag import codes as C            # noqa: E402
from diag import monitors as M         # noqa: E402
from diag import shadow                # noqa: E402
from diag.codes import CodeCatalog, DiagnosticCode   # noqa: E402
from diag.runner import ACTIVE, CANDIDATE, RESOLVED, DiagnosticEngine  # noqa: E402
from diag.store import Store           # noqa: E402

_results = []


def check(condition, label, detail=""):
    """(condition, label) — tire_diag/selftest.py's order, deliberately."""
    _results.append((bool(condition), label, detail))
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}"
          + (f"  -- {detail}" if detail else ""))
    return bool(condition)


def head(title):
    print(f"\n{title}")


# ---------------------------------------------------------------------------
# A domain invented for this file, and for nothing else
# ---------------------------------------------------------------------------
# Three "tanks" with a level in them. Not a real subsystem and not meant to be:
# the point is that it shares nothing with tires except the framework, so
# anything it can do the framework can do.

TANKS = ("A", "B", "C")

_CODES = {}
for _t in TANKS:
    _CODES[f"TEST-LOW-{_t}"] = DiagnosticCode(
        code=f"TEST-LOW-{_t}", monitor_id="tank.low", component_type="tank",
        subject=_t, default_severity=C.WARNING,
        driver_term="a low tank",
        technician_description="Level below the configured floor.",
        suggested_action="Fill it.",
        confirmation_summary="Two qualifying runs.",
        healing_summary="Two passing runs and a stable period.",
        freeze_frame_fields=("level", "floor", "sample_count"))

CATALOG = CodeCatalog(_CODES)


class TankSample(M.Sample):
    def __init__(self, subject, at, level=None, connected=True):
        super().__init__(subject=subject, at=at, connected=connected)
        self.level = level

    def has_primary(self) -> bool:
        return self.level is not None


def _eval_low(inp) -> M.Outcome:
    usable = M.valid_samples(inp)
    latest = usable[-1]
    floor = 20.0
    if latest.level <= floor:
        return M.Outcome(M.READY, M.FAILED_PENDING, confidence=0.8,
                         reason=f"level {latest.level:.0f} at or below {floor:.0f}",
                         detail={"level": latest.level, "floor": floor})
    return M.Outcome(M.READY, M.PASSED, confidence=0.8,
                     reason="level is above the floor",
                     detail={"level": latest.level, "floor": floor})


MONITORS = (
    M.MonitorDefinition(
        monitor_id="tank.low",
        component_type="tank",
        required_inputs=("level",),
        enabling=M.EnablingConditions(minimum_valid_samples=1),
        inhibiting_conditions=("the rig is not reporting at all",),
        pending_criteria={"at_or_below": 20.0},
        confirmed_criteria={"qualifying_runs": 2},
        healing=M.HealingCriteria(required_passing_monitor_runs=2,
                                  minimum_stable_duration_seconds=100.0),
        freeze_frame_fields=("level", "floor", "sample_count"),
        evaluate=_eval_low,
        requires_primary=True,
        default_severity=C.WARNING),
)


class TankEngine(DiagnosticEngine):
    DOMAIN = "tanks"
    SUBJECTS = TANKS
    MONITORS = MONITORS
    CATALOG = CATALOG
    MIN_RUN_SPACING_S = 5.0
    SAMPLE_MAX_AGE_S = 600.0
    CONFIRM_RUNS = {"tank.low": 2}
    NO_SUBJECT_REASON = "no gauge has ever reported on this tank"
    SYSTEM_UNHEALTHY_REASON = "the gauge rig is not reporting at all"

    def _ingest(self, snapshot, now):
        new = {}
        for subject, level in (snapshot or {}).items():
            if subject not in self.SUBJECTS:
                continue
            self._push(subject, TankSample(subject=subject, at=now, level=level))
            new[subject] = True
        return new

    def _system_healthy(self, now):
        return any(self._samples.get(s) for s in self.SUBJECTS)

    def _build_input(self, d, subject, now, ctx, system_healthy):
        return M.MonitorInput(
            subject=subject, now=now,
            samples=list(self._samples.get(subject) or []),
            moving=bool(ctx.get("moving")), speed_mph=ctx.get("speed_mph"),
            system_healthy=system_healthy,
            epoch_started_at=self._epoch_started_at(subject),
            drive_cycle_id=self.cycles.cycle_id,
            active_monitor_ids=self._active_monitor_ids(subject),
            sample_max_age_s=self.SAMPLE_MAX_AGE_S,
            no_subject_reason=self.NO_SUBJECT_REASON,
            system_unhealthy_reason=self.SYSTEM_UNHEALTHY_REASON)

    def _freeze_evidence(self, issue, d, out, now, ctx):
        return {"level": out.detail.get("level"),
                "floor": out.detail.get("floor"),
                "sample_count": len(self._samples.get(issue.get("subject")) or []),
                "a_field_no_code_asked_for": "should be filtered out"}


def _rig(tmp, name="tanks"):
    TankEngine.STORE = Store(tmp, name)
    return TankEngine(load=False)


def _step(eng, t, levels, n=1, dt=30.0):
    for _ in range(n):
        t += dt
        eng.observe(dict(levels), now=t, moving=True, speed_mph=40.0)
    return t


# ---------------------------------------------------------------------------

def test_01_a_domain_from_scratch_reaches_a_verdict(tmp):
    head("1 -- a domain built from nothing reaches a verdict")
    eng = _rig(tmp)
    t = _step(eng, 1000.0, {"A": 50.0, "B": 50.0, "C": 50.0})
    row = next(r for r in eng.monitor_view() if r["subject"] == "A")
    check(row["status"] == M.READY, "the monitor ran", row["status"])
    check(row["last_result"] == M.PASSED, "and passed on a healthy level",
          str(row["last_result"]))
    check(not eng.issues(), "with no issue raised",
          str([i["code"] for i in eng.issues()]))
    check(row.get("corner") is None and "corner" not in row,
          "and the view carries no tire vocabulary", str(sorted(row)))


def test_02_one_failure_is_a_candidate(tmp):
    head("2 -- one failing run is a CANDIDATE, not an ACTIVE issue")
    eng = _rig(tmp)
    _step(eng, 1000.0, {"A": 10.0, "B": 50.0, "C": 50.0})
    issues = eng.issues()
    check(len(issues) == 1, "exactly one issue exists", str(len(issues)))
    check(issues and issues[0]["lifecycle"] == CANDIDATE,
          "and it is a CANDIDATE", issues[0]["lifecycle"] if issues else "")
    check(not eng.issues(ACTIVE), "nothing is ACTIVE, so nothing is announceable")
    check(issues and issues[0]["subject"] == "A",
          "the issue names its subject", issues[0].get("subject") if issues else "")
    check(issues and issues[0]["domain"] == "tanks",
          "and its domain", issues[0].get("domain") if issues else "")


def test_03_repeated_failures_confirm_and_freeze(tmp):
    head("3 -- repeated qualifying failures confirm, and freeze the evidence")
    eng = _rig(tmp)
    _step(eng, 1000.0, {"A": 10.0, "B": 50.0, "C": 50.0}, n=3)
    iss = eng.issues(ACTIVE)
    check(len(iss) == 1, "the issue became ACTIVE", str(len(iss)))
    if not iss:
        return
    frames = iss[0]["freeze_frames"]
    check(len(frames) == 1, "with exactly one freeze frame", str(len(frames)))
    f = frames[0]
    check(f["capture_reason"] == "confirmed", "labelled with why it was taken",
          f["capture_reason"])
    check(f.get("level") == 10.0, "carrying the measurement that justified it",
          str(f.get("level")))
    check("a_field_no_code_asked_for" not in f,
          "and NOT the fields the code never declared", str(sorted(f)))
    check(f.get("domain") == "tanks" and f.get("issue_id"),
          "identity fields are added by the framework, not the domain")


def test_04_heals_only_on_sustained_evidence(tmp):
    head("4 -- a confirmed issue heals only on sustained passing evidence")
    eng = _rig(tmp)
    t = _step(eng, 1000.0, {"A": 10.0, "B": 50.0, "C": 50.0}, n=3)
    check(bool(eng.issues(ACTIVE)), "an issue is ACTIVE")
    t = _step(eng, t, {"A": 60.0, "B": 50.0, "C": 50.0}, n=1)
    mid = eng.issues()[0]
    check(mid["lifecycle"] == ACTIVE,
          "one good reading does not repair it", mid["lifecycle"])
    check((mid.get("healing_progress") or {}).get("passing_runs") == 1,
          "though the progress is recorded", str(mid.get("healing_progress")))
    t = _step(eng, t, {"A": 60.0, "B": 50.0, "C": 50.0}, n=8)
    done = eng.issues()[0]
    check(done["lifecycle"] == RESOLVED,
          "and it resolves once runs AND stable time are met", done["lifecycle"])
    check(done["freeze_frames"],
          "with its freeze-frame evidence retained after resolution")


def test_05_two_domains_keep_separate_records(tmp):
    head("5 -- two domains keep separate records")
    a_dir = os.path.join(tmp, "a")
    b_dir = os.path.join(tmp, "b")
    os.makedirs(a_dir, exist_ok=True)
    os.makedirs(b_dir, exist_ok=True)

    TankEngine.STORE = Store(a_dir, "alpha")
    eng_a = TankEngine(load=False)
    _step(eng_a, 1000.0, {"A": 10.0}, n=3)

    TankEngine.STORE = Store(b_dir, "beta")
    eng_b = TankEngine(load=False)
    _step(eng_b, 1000.0, {"A": 50.0}, n=3)

    check(bool(eng_a.issues(ACTIVE)), "the first domain has a confirmed issue")
    check(not eng_b.issues(ACTIVE),
          "the second has none, on the same subject name",
          str([i["code"] for i in eng_b.issues(ACTIVE)]))
    check(os.path.exists(os.path.join(a_dir, "alpha_events.jsonl")),
          "and each wrote to its own event log")
    check(os.path.exists(os.path.join(b_dir, "beta_events.jsonl")),
          "both of them")
    a_events = Store(a_dir, "alpha").read_events(limit=100)
    check(all(e.get("domain") in (None, "tanks") for e in a_events),
          "every event records which domain produced it")


def test_06_shadow_is_per_domain(tmp):
    head("6 -- shadow clearance is per domain, and unknown domains are shadowed")
    check(shadow.is_shadowed("a_domain_nobody_registered") is True,
          "an unregistered domain is shadowed — silence is the safe default")
    flag = {"on": True}
    shadow.register("selftest_domain", lambda: flag["on"])
    check(shadow.is_shadowed("selftest_domain") is True, "a registered domain reads its flag")
    flag["on"] = False
    check(shadow.is_shadowed("selftest_domain") is False,
          "and the flag is read at decision time, not frozen at import")
    check(shadow.is_shadowed("a_domain_nobody_registered") is True,
          "clearing one domain does not clear another — the whole point")

    def boom():
        raise RuntimeError("config exploded")

    shadow.register("selftest_broken", boom)
    check(shadow.is_shadowed("selftest_broken") is True,
          "and a flag that cannot be read leaves the domain shadowed")


def test_07_no_verdict_never_overwrites(tmp):
    head("7 -- a NO_VERDICT status never overwrites a real result")
    eng = _rig(tmp)
    t = _step(eng, 1000.0, {"A": 50.0}, n=2)
    row = next(r for r in eng.monitor_view() if r["subject"] == "A")
    check(row["last_result"] == M.PASSED, "the monitor has a PASSED result")

    # Age every sample out of the window without delivering a new one, then run
    # the monitor again by handing it a different subject's report.
    t += eng.SAMPLE_MAX_AGE_S + 60.0
    eng.observe({"B": 50.0}, now=t, moving=True, speed_mph=40.0)
    # A is now stale. Force a run on it by delivering a report that carries no
    # level at all: the gate must refuse rather than judge.
    eng._push("A", TankSample(subject="A", at=t, level=None))
    eng.observe({"A": None}, now=t + 1.0, moving=True, speed_mph=40.0)
    row = next(r for r in eng.monitor_view() if r["subject"] == "A")
    check(row["status"] in M.NO_VERDICT,
          "a report with no measurement leaves the monitor unable to judge",
          row["status"])
    check(row["last_result"] == M.PASSED,
          "and its previous PASSED result is untouched — 'I could not look' "
          "must never overwrite 'I looked'", str(row["last_result"]))


def test_08_gate_names_the_first_reason(tmp):
    head("8 -- the gate names the FIRST reason a monitor could not judge")
    eng = _rig(tmp)
    row = next(r for r in eng.monitor_view() if r["subject"] == "A")
    check(row["status"] == M.NOT_READY,
          "a monitor that has never run is NOT_READY", row["status"])
    check(row["last_result"] is None,
          "with NO result, because it has not judged anything",
          str(row["last_result"]))

    # Nothing has ever reported: NOT_SUPPORTED must win over every later check,
    # including the system-health one.
    out = M.run(MONITORS[0], M.MonitorInput(
        subject="A", now=100.0, samples=[], system_healthy=False,
        no_subject_reason=TankEngine.NO_SUBJECT_REASON,
        system_unhealthy_reason=TankEngine.SYSTEM_UNHEALTHY_REASON))
    check(out.status == M.NOT_SUPPORTED,
          "no source at all outranks a subsystem outage", out.status)
    check("gauge has ever reported" in out.reason,
          "and says so in the domain's own words", out.reason)


def test_09_domain_words_reach_the_status(tmp):
    head("9 -- a domain's own words reach the monitor status")
    out = M.run(MONITORS[0], M.MonitorInput(
        subject="A", now=100.0,
        samples=[TankSample(subject="A", at=100.0, level=50.0)],
        system_healthy=False,
        no_subject_reason=TankEngine.NO_SUBJECT_REASON,
        system_unhealthy_reason=TankEngine.SYSTEM_UNHEALTHY_REASON))
    check(out.status == M.INHIBITED, "a subsystem outage inhibits", out.status)
    check(out.reason == "the gauge rig is not reporting at all",
          "in the domain's words, not the framework's", out.reason)
    check("tire" not in out.reason and "receiver" not in out.reason,
          "with no tire vocabulary leaking through", out.reason)


def test_10_freeze_frames_carry_only_declared_fields(tmp):
    head("10 -- freeze frames carry only what the code declared, plus identity")
    eng = _rig(tmp)
    _step(eng, 1000.0, {"A": 10.0}, n=3)
    frames = eng.issues(ACTIVE)[0]["freeze_frames"]
    f = frames[0]
    declared = set(MONITORS[0].freeze_frame_fields)
    identity = {"issue_id", "code", "domain", "captured_at", "capture_reason",
                "triggering_monitor", "drive_cycle_id", "urgent_path", "moving",
                "monitor_detail", "confidence", "severity"}
    extra = set(f) - declared - identity
    check(not extra, "no field outside declared + identity", str(sorted(extra)))
    check(declared <= set(f), "and every declared field is present",
          str(sorted(declared - set(f))))


# ---------------------------------------------------------------------------
# LLM firewall
# ---------------------------------------------------------------------------

FORBIDDEN = ["openai", "llm_interface", "vision", "visual_qa", "app",
             "requests", "httpx", "torch", "transformers", "voice",
             "vehicle_health", "tires", "telemetry", "insights", "config"]


def test_firewall(tmp):
    head("F -- LLM firewall, and the framework's independence from any domain")
    root = os.path.dirname(os.path.abspath(__file__))
    for name in ("monitors.py", "runner.py", "codes.py", "drivecycle.py",
                 "store.py", "shadow.py"):
        path = os.path.join(root, name)
        tree = ast.parse(open(path).read())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported.add(node.module.split(".")[0])
                imported.update(a.name for a in node.names
                                if node.level and not node.module)
        bad = sorted(imported & set(FORBIDDEN))
        # `config` is in FORBIDDEN on purpose and it is the interesting one: a
        # framework that reached for config would have to know which domain was
        # asking, and two domains cannot share one TIRE_DIAG_* constant.
        check(not bad, f"{name} imports no domain and nothing that could reach "
                       f"a model", str(bad))

    # No tire vocabulary anywhere in the executable half. Prose is exempt — the
    # docstrings explain the extraction and have to name what was extracted from.
    for name in ("monitors.py", "runner.py", "codes.py"):
        tree = ast.parse(open(os.path.join(root, name)).read())
        docstrings = set()
        for node in ast.walk(tree):
            body = getattr(node, "body", None)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and body \
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
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) \
                    and id(node) not in docstrings:
                surface.append(node.value)
        text = " ".join(surface).lower()
        for word in ("tire", "tyre", "psi", "corner", "tpms", "gpt", "prompt",
                     "openai", "llm"):
            check(word not in text,
                  f"{name} never mentions {word!r} in executable code")


# ---------------------------------------------------------------------------

def main():
    print("=" * 74)
    print("RIO diagnostics — the generalized framework")
    print("=" * 74)
    tmp = tempfile.mkdtemp(prefix="diag_selftest_")
    print(f"  store: {tmp}")

    tests = [
        test_01_a_domain_from_scratch_reaches_a_verdict,
        test_02_one_failure_is_a_candidate,
        test_03_repeated_failures_confirm_and_freeze,
        test_04_heals_only_on_sustained_evidence,
        test_05_two_domains_keep_separate_records,
        test_06_shadow_is_per_domain,
        test_07_no_verdict_never_overwrites,
        test_08_gate_names_the_first_reason,
        test_09_domain_words_reach_the_status,
        test_10_freeze_frames_carry_only_declared_fields,
        test_firewall,
    ]
    for t in tests:
        try:
            t(tmp)
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
    shutil.rmtree(tmp, ignore_errors=True)
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
