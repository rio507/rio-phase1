"""sensor_card_selftest.py — the glass and the live session read the same words.

    python tools/sensor_card_selftest.py            (needs the server running)

WHY THIS IS MEASURED AND NOT ASSERTED IN A COMMENT
--------------------------------------------------
The Perception card shows the resident model's reading of the road. The live
session is given that reading and composes an answer from it. Those are two
payloads, built in two files, for two consumers -- and the whole value of the
card is that what the driver reads on the glass is what RIO inferred from. If
they can drift, the card is a decoration that looks like evidence.

So this does not inspect the code. It runs the two paths against a live server
over the same frames and compares the bytes:

    /perceive           -> result["reading"]["raw"]       what the card renders
    /realtime/tool look -> result["answer"]               what the browser puts
                                                          into the session as
                                                          function_call_output

Both come off one observer record through one parser
(rio_prompts.split_sensor_reading). This is the test that would notice if that
stopped being true.

IT ALSO CHECKS THE THING THE CARD WAS BUILT FOR: that a field the model did not
write comes back marked absent rather than blank, because "no risk was
reported" and "no risk" are different sentences and only one of them is true.

Exit code 0 on success, 1 on any failure, 2 if the server is not reachable --
which is not a failure of the code under test and should not read as one.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

BASE = os.getenv("RIO_BASE", "http://127.0.0.1:8888")
KEY = os.getenv("RIO_SENSOR_KEY", "sensorcard-selftest")
CLIP = REPO / "runs" / "road_clip.mp4"

checks = 0
failures = 0


def ok(name, cond, extra=""):
    global checks, failures
    checks += 1
    if not cond:
        failures += 1
    print(("  ok   " if cond else "  FAIL ") + name + (f" — {extra}" if extra else ""),
          flush=True)


def curl(args, timeout=90):
    out = subprocess.run(["curl", "-s", "--max-time", str(timeout)] + args,
                         capture_output=True, text=True)
    try:
        return json.loads(out.stdout)
    except Exception:
        return {"_raw": out.stdout[:400], "_err": out.stderr[:200]}


def main():
    print("\n== a frame to read ==\n")
    frame = Path(os.getenv("TMPDIR", "/tmp")) / "sensor_card_frame.jpg"
    if not frame.exists():
        if not CLIP.exists():
            print(f"  no clip at {CLIP} and no frame at {frame}", flush=True)
            return 2
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-ss", "3",
                        "-i", str(CLIP), "-frames:v", "1", str(frame)],
                       check=True)
    print(f"  {frame} ({frame.stat().st_size} bytes)")

    status = curl([f"{BASE}/realtime/status?client_id={KEY}"], timeout=10)
    if "observer" not in status:
        print(f"\n  server not reachable at {BASE} — {status}", flush=True)
        return 2

    # FRAMES HAVE TO BE FLOWING for either path to be the one under test.
    # /perceive serves the observer's reading only while the headway loop is
    # running (it falls back to its own caption otherwise), and the observer
    # only describes a frame that is newer than OBSERVER_FRESH_S. So this feeds
    # continuously in the background rather than pushing a burst and hoping.
    print("\n== feeding frames, and giving the 1 Hz observer time to read one ==\n")
    feeder = subprocess.Popen(
        ["bash", "-c",
         f'end=$((SECONDS+45)); while [ $SECONDS -lt $end ]; do '
         f'curl -s -o /dev/null -X POST '
         f'"{BASE}/headway_frame?client_id={KEY}" '
         f'-F "image=@{frame}" -F "source=clip" -F "v_host=13.0"; done'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        reading = None
        look = None
        deadline = time.time() + 40
        while time.time() < deadline:
            time.sleep(3)
            perc = curl(["-X", "POST", f"{BASE}/perceive?client_id={KEY}",
                         "-F", f"image=@{frame}"])
            if (perc.get("reading") or {}).get("raw"):
                reading = perc
                break
        ok("the card is served a reading at all", bool(reading),
           "no reading inside 40 s — is the observer running?")
        if not reading:
            return 1
        print(f"\n  card raw : {reading['reading']['raw']!r}")

        look = curl(["-X", "POST", f"{BASE}/realtime/tool?client_id={KEY}",
                     "-H", "Content-Type: application/json",
                     "-d", json.dumps({"name": "look",
                                       "arguments": {"question": "what do you see"}})])
        print(f"  look     : {str(look.get('answer'))!r}")
        print(f"  path     : {look.get('path')}\n")
    finally:
        feeder.terminate()

    # ---- THE CLAIM ---------------------------------------------------------
    r = reading["reading"]
    ok("the reading names the instrument that produced it", bool(r.get("model")),
       str(r.get("model")))
    ok("...and carries its age", isinstance(r.get("age_s"), (int, float)),
       f"{r.get('age_s')} s")
    ok("...and the freshness threshold the observer itself uses",
       isinstance(r.get("fresh_s"), (int, float)), f"{r.get('fresh_s')} s")
    ok("...and says it is a sensor's reading, not her words",
       r.get("speakable") is False)

    ok("the live session took the observer path for this question",
       look.get("path") in ("observer_composed", "observer_direct"),
       str(look.get("path")))
    if look.get("path", "").startswith("observer"):
        ok("THE GLASS AND THE SESSION CARRY THE SAME BYTES",
           (look.get("answer") or "").strip() == r["raw"].strip(),
           f"card {r['raw'][:44]!r} vs session {str(look.get('answer'))[:44]!r}")
        ok("...and the session is given the same fields the card draws",
           [f["text"] for f in (look.get("reading_fields") or [])]
           == [f["text"] for f in r["fields"]])

    # ---- THE ROWS ----------------------------------------------------------
    import rio_prompts as rp
    names = [f["name"] for f in r["fields"]]
    ok("every field the prompt asks for gets a row, present or not",
       names == list(rp.SENSOR_FIELDS), str(names))
    for f in r["fields"]:
        if not f["present"]:
            ok(f"...{f['name']} is absent from this reading and is marked absent",
               f["text"] == "")
    ok("no row is both absent and full of text",
       all(f["present"] == bool(f["text"]) for f in r["fields"]))

    # Every word the model wrote is still somewhere on the card: the fields, the
    # extra, or the raw row. A renderer that silently dropped half a reading
    # would pass every check above.
    joined = " ".join([f["text"] for f in r["fields"]] + [r.get("extra") or ""])
    lost = [w for w in r["raw"].replace("|", " ").split()
            if w.rstrip(":").upper() not in rp.SENSOR_FIELDS and w not in joined]
    ok("nothing the model wrote is dropped between the wire and the rows",
       not lost, f"missing: {lost[:6]}")

    print(f"\n{checks - failures}/{checks} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
