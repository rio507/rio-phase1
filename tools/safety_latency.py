"""How long the driver waits, per safety tier.

    python -m tools.safety_latency
    python -m tools.safety_latency --arms clip
    python -m tools.safety_latency --n 5

THE QUESTION THIS ANSWERS is whether splitting the safety library by urgency
costs the critical tier anything, and what the other tier pays for being said
in her own words. Three arms, measured from the same origin -- the moment the
POLICY DECIDED there should be a line -- because that is the moment a driver's
wait starts, and every design here is a claim about what happens after it.

  clip            the critical tier. A pre-rendered file, already decoded and
                  held by the output bus, played locally. No network, no model,
                  no session. Measured in a real browser because that is the
                  only place the claim can be tested: the bus, the decode and
                  the element are browser objects, and node cannot tell you
                  whether audio actually left them.
  live prepared   the new path. The sentence was written while the fault sat
                  in the policy's cooldown, so the wait is only the mouth
                  starting.
  live cold       the same path when the sentence was NOT ready -- the fault
                  went from confirmed to announced faster than phrasing, or
                  phrasing was disabled. This is the number that says whether
                  pre-phrasing is load-bearing or a nicety.

WHAT IS NOT MEASURED HERE: whether she should have spoken at all. Every band,
cooldown, gap and gate is upstream of this file and none of it changed.
"""
import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv                              # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                               # noqa: E402
import safety_speech                                        # noqa: E402

# One critical clip and one non-critical event, so both arms are measuring the
# real thing rather than a stand-in.
CLIP_ID = "back_off"
EVENT = {
    "key": "tire.slow_leak.FL",
    "what": "a possible slow leak in the front left tire",
    "severity": "warning", "location": "front left",
    "provenance": "RIO worked this out from the pressure trend, the car has "
                  "not reported a fault",
    "observation_window": "the last 40 minutes",
    "evidence": "down 4 PSI over the window, now 28",
    "action": "worth a look when you stop",
    "numbers": [4, 28],
    "fallback": "The front left tire may have a slow leak. Worth a look when "
                "you stop.",
}


# ---------------------------------------------------------------------------
# The clip tier, in a browser
# ---------------------------------------------------------------------------
_CLIP_JS = """
async (args) => {
  const [url, n] = args;
  // Decode once, exactly as RIO.output does when it preloads the library. The
  // measurement is of PLAYING a clip that is already in memory, because that
  // is the state the car is in when a warning fires -- a decode in the hot
  // path would be a different (and much worse) number, and the bus exists so
  // that it never happens.
  const ctx = new (window.AudioContext || window.webkitAudioContext)();
  const buf = await ctx.decodeAudioData(await (await fetch(url)).arrayBuffer());
  const out = [];
  for (let i = 0; i < n; i++) {
    const an = ctx.createAnalyser();
    an.fftSize = 256;
    const data = new Float32Array(an.fftSize);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(an);
    an.connect(ctx.destination);
    const t0 = performance.now();
    src.start();
    // Poll for the first frame with anything in it -- "the speaker is making
    // a sound", not "the call returned".
    let t1 = null;
    while (performance.now() - t0 < 2000) {
      an.getFloatTimeDomainData(data);
      let peak = 0;
      for (let k = 0; k < data.length; k++) peak = Math.max(peak, Math.abs(data[k]));
      if (peak > 0.005) { t1 = performance.now(); break; }
      await new Promise(r => setTimeout(r, 1));
    }
    out.push(t1 === null ? null : t1 - t0);
    src.stop();
    await new Promise(r => setTimeout(r, 120));
  }
  return out;
}
"""


def clip_arm(url: str, n: int) -> list:
    from playwright.sync_api import sync_playwright

    clip_url = f"{url.rstrip('/')}{config.clip_base()}{CLIP_ID}.mp3"
    with sync_playwright() as p:
        b = p.chromium.launch(args=["--autoplay-policy=no-user-gesture-required",
                                    "--use-fake-ui-for-media-stream"])
        page = b.new_page()
        page.goto(url, wait_until="domcontentloaded")
        try:
            got = page.evaluate(_CLIP_JS, [clip_url, n])
        finally:
            b.close()
    return [x for x in got if x is not None]


# ---------------------------------------------------------------------------
# The spoken tiers
# ---------------------------------------------------------------------------
async def live_arm(n: int, cold: bool) -> list:
    """Decision -> first audio, for a non-critical line in her voice."""
    from live_harness import LiveSession, ScriptedMic, silence

    out = []
    for _ in range(n):
        safety_speech.forget("lat")
        safety_speech.drop("lat")
        mic = ScriptedMic(silence(40000))
        s = LiveSession(mic)
        await s.open()
        await asyncio.sleep(1.0)
        if not cold:
            # The sentence is written while the fault sits in the policy's
            # cooldown, which is the whole point: by the time the policy says
            # "speak", this has already returned.
            safety_speech.prepare(EVENT, "lat", EVENT["key"], revision="1")
            for _ in range(300):
                if safety_speech.take("lat", EVENT["key"], "1"):
                    break
                await asyncio.sleep(0.05)
        # THE ORIGIN: the policy has decided there is a line.
        t0 = time.time()
        got = safety_speech.take("lat", EVENT["key"], "1")
        if not got:
            got = safety_speech.phrase(EVENT, session_key="lat",
                                       issue_key=EVENT["key"], timeout_s=20.0)
        s.first_audio_t = None
        s.last_audio_t = None
        s.say_line(got["text"])
        while time.time() - t0 < 30:
            await asyncio.sleep(0.03)
            if s.first_audio_t:
                break
        if s.first_audio_t:
            out.append((s.first_audio_t - t0) * 1000.0)
        await s.close()
    return out


async def realtime_arm(n: int, cold: bool) -> list:
    """The same two tiers on the SHIPPED backend, which is the one that counts.

    gpt-realtime dictates a deterministic line by response.create with the
    words in it, and it is measurably faster at that than gpt-live-1's
    instructions.append -- 390-585 ms against 1324. Both are measured here
    rather than one being assumed from the other, because the whole question
    is what the driver waits, and the driver is on whichever backend is
    configured.
    """
    from openai import AsyncOpenAI

    import realtime as rt

    out = []
    cl = AsyncOpenAI()
    for _ in range(n):
        safety_speech.forget("lat")
        safety_speech.drop("lat")
        if not cold:
            safety_speech.prepare(EVENT, "lat", EVENT["key"], revision="1")
            for _ in range(300):
                if safety_speech.take("lat", EVENT["key"], "1"):
                    break
                await asyncio.sleep(0.05)
        async with cl.realtime.connect(
                model=config.OPENAI_REALTIME_MODEL) as conn:
            await conn.session.update(session={
                "type": "realtime",
                "output_modalities": ["audio"],
                "audio": {"output": {
                    "voice": config.OPENAI_REALTIME_VOICE,
                    "format": {"type": "audio/pcm", "rate": 24000}}}})
            t0 = time.time()
            got = safety_speech.take("lat", EVENT["key"], "1")
            if not got:
                got = safety_speech.phrase(EVENT, session_key="lat",
                                           issue_key=EVENT["key"],
                                           timeout_s=20.0)
            await conn.response.create(
                response=rt.verbatim_response(got["text"]))
            first = None
            async for ev in conn:
                if getattr(ev, "type", "") == "response.output_audio.delta":
                    first = time.time()
                    break
                if time.time() - t0 > 25:
                    break
            if first:
                out.append((first - t0) * 1000.0)
    return out


def report(name: str, xs: list):
    xs = [x for x in xs if x is not None]
    if not xs:
        print(f"  {name:<22} no measurements")
        return
    p95 = (statistics.quantiles(xs, n=20)[18] if len(xs) >= 20 else max(xs))
    print(f"  {name:<22} n={len(xs):<3} p50 {statistics.median(xs):8.0f} ms   "
          f"p95 {p95:8.0f} ms   min {min(xs):7.0f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--url", default="http://127.0.0.1:8888/")
    ap.add_argument("--arms", default="clip,prepared,cold")
    a = ap.parse_args()
    arms = [x.strip() for x in a.arms.split(",") if x.strip()]
    res = {}
    if "clip" in arms:
        print("-- clip (critical tier, local file, real browser)", flush=True)
        try:
            res["clip (critical)"] = clip_arm(a.url, a.n)
        except Exception as e:
            print(f"   {type(e).__name__}: {e}")
    if "prepared" in arms:
        print("-- live, sentence already written", flush=True)
        res["live, prepared"] = asyncio.run(live_arm(a.n, cold=False))
    if "cold" in arms:
        print("-- live, sentence written on demand", flush=True)
        res["live, cold"] = asyncio.run(live_arm(a.n, cold=True))
    if "rt_prepared" in arms:
        print("-- realtime, sentence already written", flush=True)
        res["realtime, prepared"] = asyncio.run(realtime_arm(a.n, cold=False))
    if "rt_cold" in arms:
        print("-- realtime, sentence on demand", flush=True)
        res["realtime, cold"] = asyncio.run(realtime_arm(a.n, cold=True))
    print("\npolicy decided -> first audio")
    for k, v in res.items():
        report(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
