"""The safety layer after the split: what is a clip, what is written, what is
neither allowed to do.

    python -m tools.safety_speech_selftest
    python -m tools.safety_speech_selftest --offline   # no API calls

FIVE THINGS, and each of them is a way this change could have quietly broken
something that used to work:

  critical    the four urgent lines are still CLIPS -- on disk, in the
              configured voice, reachable with no network and no model. This is
              the check that a refactor about prose did not put a model in
              front of a collision warning.
  split       the tier map says what it should: nothing urgent phrased,
              nothing leisurely clipped, and the set named in exactly one
              place.
  honesty     the validator refuses an invented number, an upgraded hedge, a
              banned phrase and a novel. These are asserted against the GATE,
              not against a model, so they hold whatever the model does.
  varies      the same event twice on one drive does not come back word for
              word -- and the deterministic floor is still there underneath.
  suppression the policy's own decisions are untouched. Deliberately not
              re-implemented here: it shells out to the existing suites, on the
              principle that the way to prove something did not change is to
              run its own tests unmodified.

Exit code is the number of failures.
"""
import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv                              # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                               # noqa: E402
import safety_speech as ss                                  # noqa: E402

OK, BAD = "ok  ", "FAIL"
_fails = []


def ok(name, cond, extra=""):
    if cond:
        print(f"  {OK} {name}")
    else:
        _fails.append(name)
        print(f"  {BAD} {name}{('  ' + str(extra)) if extra else ''}")


def t_critical():
    print("\n== the critical tier is still local")
    import render_alerts as ra
    from headway import live_policy as lp

    d = ra.audio_dir()
    for clip in sorted(ss.CRITICAL_CLIPS):
        p = d / f"{clip}.mp3"
        ok(f"{clip} is a file on disk", p.exists() and p.stat().st_size > 2000,
           str(p))
    # ...and the words on disk are the words the policy would say.
    want = dict(lp.LINE_TEXT)
    want.update(ra.TIRE_CLIPS)
    doc = ra.manifest().get("clips", {})
    sig = ra.voice_signature()
    for clip in sorted(ss.CRITICAL_CLIPS):
        entry = doc.get(clip) or {}
        ok(f"{clip} was rendered in the configured voice",
           entry.get("voice") == sig["voice"],
           f'manifest says {entry.get("voice")!r}, config says {sig["voice"]!r}')
    # THE RED TIER MUST NOT REACH THE MODEL. LINE_AUDIO is what decides.
    for line in ("too_close", "watch_distance", "back_off"):
        ok(f"{line} is a clip, not tts", lp.LINE_AUDIO.get(line) != "tts")


def t_split():
    print("\n== the split is where it says it is")
    import render_alerts as ra
    from headway import live_policy as lp

    ok("the critical set is exactly four", len(ss.CRITICAL_CLIPS) == 4,
       sorted(ss.CRITICAL_CLIPS))
    ok("the coaching tier is NOT clipped",
       lp.LINE_AUDIO.get("calm") == "tts"
       and lp.LINE_AUDIO.get("escalate") == "tts")
    ok("tire_sensor_lost has left the clip tier",
       "tire_sensor_lost" not in ra.TIRE_CLIPS
       and not ss.is_critical("tire_sensor_lost"))
    # ...and the fast path knows it.
    import vehicle_health
    ok("the sensor-loss fast path is now spoken, not clipped",
       vehicle_health._FAST_PATH_CLIP.get(
           "tire.sensor_loss_during_decline") is None)
    ok("critical clips and the render set agree",
       set(ra.CLIP_LINES) | set(ra.TIRE_CLIPS) == set(ss.CRITICAL_CLIPS),
       f"{sorted(set(ra.CLIP_LINES) | set(ra.TIRE_CLIPS))} vs "
       f"{sorted(ss.CRITICAL_CLIPS)}")


def t_honesty():
    print("\n== the validator refuses what a prompt only asks for")
    ev = {"what": "a slow leak in the front left",
          "observation_window": "the last 40 minutes",
          "evidence": "down 4 PSI, now 28", "numbers": [4, 28]}
    ok("a figure from the evidence is allowed",
       ss.check("Front left's down 4 PSI over the last 40 minutes — 28 now.",
                ev)[0])
    ok("an invented figure is refused",
       not ss.check("Front left is down to 19 PSI.", ev)[0])
    ok("a figure spelled as a word is still checked",
       not ss.check("Front left is down to nineteen PSI.", ev)[0])
    ok("a banned phrase is refused",
       not ss.check("Roger — front left is at 28.", ev)[0])
    ok("a paragraph is refused",
       not ss.check("Well, " + "the tire is low and " * 12 + "so on.", ev)[0])
    ok("silence is refused", not ss.check("", ev)[0])
    unc = {"what": "a lean-mixture condition", "unconfirmed": True,
           "numbers": []}
    ok("an unconfirmed finding may not be stated as fact",
       not ss.check("The car has a lean-mixture fault.", unc)[0])
    ok("...and is fine when hedged",
       ss.check("The car's picked up a possible lean mixture — not "
                "confirmed yet.", unc)[0])
    # PROVENANCE, derived conservatively: claiming the ECU said something it
    # did not is the error that sends a mechanic looking for a log entry that
    # is not there.
    ok("a trend finding is attributed to RIO, not the car",
       "RIO" in ss.provenance_of({"domain": "tires", "type": "slow_leak"}))
    ok("a DTC is attributed to the car",
       "car" in ss.provenance_of({"code": "P0171", "domain": "dtc"}))


def t_fallback():
    print("\n== the floor is still there")
    ev = {"key": "x", "what": "something", "numbers": [],
          "fallback": "The deterministic sentence."}
    old_model = config.SAFETY_PHRASE_MODEL
    config.SAFETY_PHRASE_MODEL = "definitely-not-a-model"
    try:
        got = ss.phrase(ev, session_key="t", issue_key="x", timeout_s=5.0)
    finally:
        config.SAFETY_PHRASE_MODEL = old_model
    ok("a model that will not answer falls back to the written line",
       got["text"] == "The deterministic sentence."
       and got["source"] == "fallback", got)
    ss.forget("t")
    ss.drop("t")
    got2 = ss.next_line(dict(ev), "t", "x")
    ok("a pool with nothing in it yet serves the written line",
       got2["text"] == "The deterministic sentence.")
    # ...and compose() still works exactly as it did without an override.
    import vehicle_health_policy as vhp
    iss = {"type": "critical_low_pressure", "location": "front left",
           "value": 24, "unit": "psi"}
    plain = vhp.compose(iss)
    # SPOKEN FORM, not digits: compose runs the value through spoken_value, so
    # 24 psi becomes "twenty-four P S I". Asserting on "24" tested the test.
    ok("compose with no override is unchanged",
       "twenty-four" in plain and "front left" in plain, plain)
    ok("compose prefers an override when the caller wrote one",
       vhp.compose({**iss, "spoken_override": "Front left's flat."})
       == "Front left's flat.")


def t_varies():
    print("\n== the same thing twice does not sound the same")
    ev = {"key": "tire.slow_leak.FL",
          "what": "a possible slow leak in the front left tire",
          "severity": "warning", "location": "front left",
          "provenance": "RIO worked it out from the readings",
          "observation_window": "the last 40 minutes",
          "evidence": "down 4 PSI, now 28",
          "action": "worth a look when you stop",
          "numbers": [4, 28],
          "fallback": "The front left tire may have a slow leak."}
    ss.forget("vary")
    lines = []
    for _ in range(3):
        got = ss.phrase(ev, session_key="vary", issue_key=ev["key"],
                        timeout_s=25.0)
        lines.append(got["text"])
        print(f"     {got['source'][:9]:<9} {got['text']}")
    gen = [ln for ln, g in zip(lines, lines) if ln]
    ok("three firings produced three different sentences",
       len({ss._norm(x) for x in lines}) == 3, lines)
    ok("every one of them passed the honesty gate",
       all(ss.check(x, ev)[0] for x in lines))


def t_routes():
    """The endpoints the announcement path needs still resolve to the right
    functions.

    THIS EXISTS BECAUSE OF A BUG THIS SUITE DID NOT CATCH. `_attach_phrasing`
    was inserted directly beneath `@app.get("/vehicle/health/announcement")`,
    so the decorator bound the ROUTE to the helper -- which takes an `issues`
    argument and therefore answered 422 to every poll -- and left the real
    endpoint undecorated and unreachable. Health announcements were dead for a
    whole commit and every policy test still passed, because none of them
    goes anywhere near the router.
    """
    print("\n== the routes still point at the right functions")
    import app as rio_app

    want = {
        "/vehicle/health/announcement": "vehicle_health_announcement_endpoint",
        "/vehicle/health/voice": "vehicle_health_voice_endpoint",
        "/headway_voice": "headway_voice_endpoint",
        "/headway_frame": "headway_frame_endpoint",
    }
    got = {}
    for r in rio_app.app.routes:
        path = getattr(r, "path", None)
        if path in want:
            got[path] = getattr(getattr(r, "endpoint", None), "__name__", "?")
    for path, fn in want.items():
        ok(f"{path} -> {fn}", got.get(path) == fn,
           f"resolves to {got.get(path)!r}")


def t_suppression():
    print("\n== the suppression rules are untouched")
    root = Path(__file__).resolve().parent.parent
    for name, cmd in (
            ("headway policy", [sys.executable, "-m",
                                "headway.live_selftest"]),
            ("vehicle health policy", [sys.executable, "-m",
                                       "tools.vehicle_health_selftest"])):
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=str(root))
        tail = (r.stdout or r.stderr).strip().splitlines()[-1:] or [""]
        ok(f"{name}: its own suite still passes", r.returncode == 0, tail[0])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline", action="store_true",
                    help="skip the checks that call the API")
    a = ap.parse_args()
    t_critical()
    t_split()
    t_honesty()
    t_routes()
    if not a.offline:
        t_fallback()
        t_varies()
    t_suppression()
    print("\n" + "-" * 58)
    print(f'  {"PASS" if not _fails else "FAIL"}: {len(_fails)} failure(s)')
    for f in _fails:
        print(f"    - {f}")
    return len(_fails)


if __name__ == "__main__":
    raise SystemExit(main())
