"""Whose road is it? Two sessions, two cameras, and no leaking between them.

    python -m tools.session_isolation_selftest

THE DRIVE THIS EXISTS BECAUSE OF (2026-09-16). Questions asked on a phone came
back describing a clip that had been uploaded on a desktop. Neither client had
started a drive; a live conversation does not require one; the frame ring was
keyed `session_id or "default"`; so both landed in the same buffer and the
phone was answered from somebody else's road.

Two things were wrong and both are checked here:

  the KEY     "default" was a shared bucket that any keyless client fell into.
              A tab now carries its own id whether or not a drive is running.
  the RULE    `observer.serve_to` had always checked frame provenance.
              `visual_qa` had not — it trusted the key to mean the frames were
              the asker's. One rule now, `framebuf.owns`, and every reader asks.

The point of the suite is that the second is what actually saves you. Keys can
be got wrong again; a reader that verifies provenance is wrong only once.

Exit code is the number of failures.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv                              # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import framebuf                                             # noqa: E402
import observer                                             # noqa: E402

OK, BAD = "ok  ", "FAIL"
_fails = []


def ok(name, cond, extra=""):
    if cond:
        print(f"  {OK} {name}")
    else:
        _fails.append(name)
        print(f"  {BAD} {name}{('  ' + str(extra)) if extra else ''}")


def section(t):
    print(f"\n== {t}")


def fake_result(tag):
    """A /headway_frame result shaped enough for the ring to accept it."""
    return {"ok": True, "t": time.time() % 1000,
            "image": {"w": 1280, "h": 720},
            "scene_objects": [{"id": tag, "label": tag, "box": [0, 0, 10, 10]}],
            "ego": {}, "quality": {}}


def push(key, source, tag):
    """Push a frame into `key`'s ring, stamped the way the server stamps it.

    The origin is "<visual key>:<source>" — the SAME key the ring is stored
    under. That identity is the invariant: "which buffer" and "whose road"
    disagreeing is the whole bug. `t_origin_matches_key` asserts the server
    builds it that way too, so this helper cannot drift into being kinder than
    production.
    """
    ring = framebuf.get_ring(key)
    return ring.push(b"\xff\xd8fakejpeg", fake_result(tag),
                     origin=f"{key}:{source}")


def t_rule():
    section("A. the ownership rule itself")
    cases = [
        ("a drive sees its own frames", "sess-A:camera", "sess-A", True),
        ("a drive never sees another drive's", "sess-A:camera", "sess-B", False),
        ("a tab sees its own", "client:tabA:camera", "client:tabA", True),
        ("a tab never sees another tab's", "client:tabA:camera", "client:tabB", False),
        ("a drive never sees a tab's", "client:tabA:clip", "sess-A", False),
        ("a tab never sees a drive's", "sess-A:camera", "client:tabA", False),
        ("a keyless caller sees keyless frames", "api:clip", "default", True),
        ("a drive never sees keyless frames", "api:clip", "sess-A", False),
        ("...nor does a tab", "api:clip", "client:tabA", False),
        ("unstamped provenance is refused", None, "sess-A", False),
        ("empty provenance is refused", "", "default", False),
    ]
    for name, origin, key, want in cases:
        ok(name, framebuf.owns(origin, key) is want,
           f"owns({origin!r}, {key!r})")

    # The colon in a tab key is the trap: splitting from the left reads every
    # tab's owner as "client" and hands them all the same frames.
    ok("a tab key's own colon does not collapse ownership",
       framebuf.owns("client:tabA:camera", "client:tabB") is False
       and framebuf.owns("client:tabA:camera", "client:tabA") is True)


def t_two_sessions():
    section("B. two concurrent sessions with different sources")
    framebuf.drop_ring("sess-PHONE")
    framebuf.drop_ring("sess-DESK")

    push("sess-PHONE", "camera", "phone-road")
    push("sess-DESK", "clip", "desktop-clip")

    phone = framebuf.ring_for("sess-PHONE")
    desk = framebuf.ring_for("sess-DESK")
    ok("each session has its own ring", phone is not None and desk is not None)
    ok("...and they are different objects", phone is not desk)

    pf = phone.frames() if phone else []
    df = desk.frames() if desk else []
    ok("the phone's ring holds the phone's frame",
       pf and pf[-1].objects[0]["id"] == "phone-road",
       [f.objects[0]["id"] for f in pf])
    ok("the desktop's ring holds the desktop's clip",
       df and df[-1].objects[0]["id"] == "desktop-clip",
       [f.objects[0]["id"] for f in df])
    ok("neither ring contains the other's frames",
       not any(f.objects[0]["id"] == "desktop-clip" for f in pf)
       and not any(f.objects[0]["id"] == "phone-road" for f in df))

    # The exact failure from the drive: the desktop pushed, the phone asks.
    ok("the phone may not be answered from the desktop's ring",
       framebuf.owns(desk.origin, "sess-PHONE") is False, desk.origin)
    ok("...and the desktop may not be answered from the phone's",
       framebuf.owns(phone.origin, "sess-DESK") is False, phone.origin)


def t_keyless_tabs():
    section("C. two tabs that never started a drive — the actual 2026-09-16 case")
    framebuf.drop_ring("client:tabPHONE")
    framebuf.drop_ring("client:tabDESK")
    framebuf.drop_ring("default")

    push("client:tabDESK", "clip", "desktop-clip")
    ok("the desktop tab's frames are in ITS ring, not the shared one",
       framebuf.peek_ring("client:tabDESK") is not None
       and framebuf.peek_ring("default") is None)

    ok("a phone tab with no frames of its own gets nothing",
       framebuf.ring_for("client:tabPHONE") is None)
    ok("...specifically NOT the desktop tab's clip",
       framebuf.owns("client:tabDESK:clip", "client:tabPHONE") is False)

    # And once the phone does push, it sees only its own.
    push("client:tabPHONE", "camera", "phone-road")
    mine = framebuf.ring_for("client:tabPHONE")
    ok("once the phone pushes, it sees its own road",
       mine is not None and mine.frames()[-1].objects[0]["id"] == "phone-road")


def t_no_frames_refuses():
    section("D. a session with no frames refuses rather than borrowing")
    framebuf.drop_ring("sess-EMPTY")
    framebuf.drop_ring("sess-OTHER")
    push("sess-OTHER", "camera", "somebody-elses-road")

    ok("a session that never pushed has no ring to read",
       framebuf.ring_for("sess-EMPTY") is None)

    import visual_qa

    g = visual_qa.scene_graph("sess-EMPTY")
    ok("the scene graph refuses for an empty session",
       g.get("available") is False, g.get("reason"))
    ok("...and the refusal is not somebody else's objects",
       not g.get("objects"), g.get("objects"))

    # A session pointed at a ring that exists but is foreign must say so
    # distinctly -- same sentence to the driver, different fault in the log.
    framebuf.drop_ring("sess-FOREIGN")
    ring = framebuf.get_ring("sess-FOREIGN")
    ring.push(b"\xff\xd8fake", fake_result("not-yours"), origin="sess-SOMEONE:camera")
    g2 = visual_qa.scene_graph("sess-FOREIGN")
    ok("a foreign ring is refused, and named as foreign",
       g2.get("available") is False and g2.get("reason") == "not_my_frames",
       g2.get("reason"))


def t_origin_matches_key():
    section("C2. the server stamps frames with the same key it files them under")
    import app

    # With a drive, the origin's key is the session. Without one it is the tab,
    # carried on the request — which is what the middleware exists for.
    ok("a drive's frames are stamped with the drive key",
       app._frame_origin("sess-A", "camera") == "sess-A:camera",
       app._frame_origin("sess-A", "camera"))
    ok("a tab's frames are stamped with the tab key",
       app._frame_origin(None, "camera", ) == app._visual_key(None) + ":camera")
    ok("and the stamp always owns the ring it goes in",
       framebuf.owns(app._frame_origin("sess-A", "camera"),
                     app._visual_key("sess-A")) is True)


def t_observer():
    section("E. the observer's guard still holds, on the shared rule")
    rec_a = {"text": "a road", "origin": "sess-A:camera", "at": time.time()}
    ok("its own session may be told", observer.serve_to(rec_a, "sess-A") is True)
    ok("another may not", observer.serve_to(rec_a, "sess-B") is False)
    ok("a tab may not", observer.serve_to(rec_a, "client:tabA") is False)
    ok("an unstamped record may never be served",
       observer.serve_to({"text": "x"}, "sess-A") is False)
    ok("a keyless record may not reach a drive",
       observer.serve_to({"origin": "api:clip"}, "sess-A") is False)


def t_readers_use_the_rule():
    section("F. every reader asks, rather than trusting the key")
    src = (Path(__file__).resolve().parent.parent / "visual_qa.py").read_text()
    ok("visual_qa reads through ring_for, not peek_ring alone",
       "framebuf.ring_for(" in src)
    ok("...in the answer path", src.count("framebuf.ring_for(") >= 2)
    obs = (Path(__file__).resolve().parent.parent / "observer.py").read_text()
    ok("observer delegates to the one rule", "framebuf.owns(" in obs)
    fb = (Path(__file__).resolve().parent.parent / "framebuf.py").read_text()
    ok("the rule splits the source off the RIGHT of the origin",
       'rsplit(":", 1)' in fb)


def main() -> int:
    t_rule()
    t_two_sessions()
    t_keyless_tabs()
    t_origin_matches_key()
    t_no_frames_refuses()
    t_observer()
    t_readers_use_the_rule()
    print("\n" + "-" * 58)
    print(f'  {"PASS" if not _fails else "FAIL"}: {len(_fails)} failure(s)')
    for f in _fails:
        print(f"    - {f}")
    return len(_fails)


if __name__ == "__main__":
    raise SystemExit(main())
