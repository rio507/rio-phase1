"""turn_end_bench.py — how long RIO waits before she realises you have stopped.

    python -m tools.turn_end_bench                 # the whole sweep
    python -m tools.turn_end_bench --only "semantic_vad high"
    python -m tools.turn_end_bench --responses     # + real response.created

THE COMPLAINT: "I finish a question and she just sits there." That wait is
almost entirely one number — the silence the detector must hear before it will
call the turn over — and it is set to 700 ms because a car is not a quiet room
and a pause for breath must not end a sentence.

So the two things being measured here are the two halves of one trade, and
measuring only the first is how the pause-for-breath bug comes back:

  TURN-END LATENCY   from the last sample of speech to the turn actually
                     ending. What the driver is complaining about.
  FALSE TURN ENDS    a driver who pauses mid-sentence — "take me to... um...
                     the Getty" — and has their sentence cut in half for it.
                     What the 700 ms is buying.

Both are measured against the REAL API with the REAL session, because turn
detection happens on the server and there is nothing local to test.

WHAT THE NUMBERS ARE AND ARE NOT
--------------------------------
The audio is synthesised and the pauses are digital silence, which is the
EASIEST possible case for a silence timer: real cabin ambience sits above zero
and can hold a detector open through a gap that this would end. So the false
cut rates here are a floor for server_vad, not a ceiling — the car will be at
least this bad, plausibly worse. Read them as a comparison between configs,
which is what they are for, rather than as a prediction of a drive.

The clean utterances are complete questions a driver asks. The pausers are the
same shape of question with a hesitation spliced into the middle, at four gap
lengths that bracket the settings under test.
"""
import argparse
import asyncio
import base64
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv                              # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                               # noqa: E402
import realtime                                             # noqa: E402
import voice_dialogue as vd                                 # noqa: E402

RATE = config.ELEVENLABS_SAMPLE_RATE
DRIVER_VOICE = "CwhRBWXzGAHq8TQ4Fs17"        # Roger, as everywhere else
CACHE = Path("/tmp/rio_turn_end_cache")

# How much silence to keep sending after the driver stops. Long
# enough for the slowest decision measured (1.75 s) with room --
# and cut short the moment the turn actually ends, so a fast
# config does not pay for a slow one's allowance.
TAIL_MS = 4000

# Complete questions. Turn-end latency is measured on these.
CLEAN = [
    "What do you see?",
    "What's that car ahead?",
    "How far to the next turn?",
    "Take me to the Getty.",
    "What are the directions?",
    "How are my tires?",
    "Where's the nearest petrol station?",
    "Is anything wrong with the car?",
    "What's the road like now?",
    "Why does a turbo need an intercooler?",
]

# ...and the same driver hesitating. `gap_ms` is spliced in at the "|".
# The gaps bracket every silence setting under test, so each config is asked
# about pauses shorter than it, around it, and longer than it.
PAUSERS = [
    ("Take me to|the Getty.", 350),
    ("What's that|car ahead?", 350),
    ("Can you find me|a coffee place?", 500),
    ("How far is it|to the next turn?", 500),
    ("Take me to|the Getty Center.", 700),
    ("What do you|see out there?", 700),
    ("Navigate to|LAX please.", 900),
    ("Tell me about|that building.", 900),
]

CONFIGS = [
    ("server_vad 700 (today)", {"type": "server_vad",
        "threshold": config.REALTIME_VAD_THRESHOLD,
        "prefix_padding_ms": config.REALTIME_VAD_PREFIX_MS,
        "silence_duration_ms": 700}),
    ("server_vad 500", {"type": "server_vad",
        "threshold": config.REALTIME_VAD_THRESHOLD,
        "prefix_padding_ms": config.REALTIME_VAD_PREFIX_MS,
        "silence_duration_ms": 500}),
    ("server_vad 600", {"type": "server_vad",
        "threshold": config.REALTIME_VAD_THRESHOLD,
        "prefix_padding_ms": config.REALTIME_VAD_PREFIX_MS,
        "silence_duration_ms": 600}),
    ("server_vad 400", {"type": "server_vad",
        "threshold": config.REALTIME_VAD_THRESHOLD,
        "prefix_padding_ms": config.REALTIME_VAD_PREFIX_MS,
        "silence_duration_ms": 400}),
    ("semantic_vad low", {"type": "semantic_vad", "eagerness": "low"}),
    ("semantic_vad medium", {"type": "semantic_vad", "eagerness": "medium"}),
    ("semantic_vad high", {"type": "semantic_vad", "eagerness": "high"}),
]


def silence(ms):
    return b"\x00\x00" * int(RATE * ms / 1000)


def _trim(pcm: bytes, floor: int = 400) -> bytes:
    """Strip the near-silence a synthesiser leaves at both ends.

    It matters more than it sounds. The mark this whole file measures from is
    "the last sample of speech", and a clip that ends with 180 ms of digital
    quiet moves that mark 180 ms early — so every latency reads short by that
    much, and every spliced pause is that much longer than its label. Both
    errors flatter whatever config is being tested.
    """
    import array
    a = array.array("h")
    a.frombytes(pcm)
    lo, hi = 0, len(a)
    while lo < hi and abs(a[lo]) < floor:
        lo += 1
    while hi > lo and abs(a[hi - 1]) < floor:
        hi -= 1
    return a[lo:hi].tobytes()


def say(text: str) -> bytes:
    """The driver, synthesised once and kept — this is run many times."""
    CACHE.mkdir(exist_ok=True)
    # A STABLE key. Python hashes strings with a per-process seed, so `hash()`
    # here meant a cache that never hit once — every config in the sweep
    # re-synthesising every line, which is slow and is billed.
    import hashlib
    key = CACHE / (hashlib.sha1(text.encode()).hexdigest()[:16] + ".pcm")
    if key.exists():
        return _trim(key.read_bytes())
    import httpx
    r = httpx.post(vd.FLASH_URL.format(voice=DRIVER_VOICE),
                   params={"output_format": f"pcm_{RATE}"},
                   headers={"xi-api-key": vd.api_key()},
                   json={"text": text,
                         "model_id": config.ELEVENLABS_DETERMINISTIC_MODEL},
                   timeout=60.0)
    r.raise_for_status()
    key.write_bytes(r.content)
    return _trim(r.content)


async def feed(conn, pcm: bytes, mark_at: int = None, stop_when=None):
    """Play `pcm` into the session at the speed a person speaks it.

    Paced against an absolute schedule rather than by sleeping 20 ms at a
    time: asyncio.sleep drifts, and a measurement of when the server heard the
    end of a sentence cannot be built on a clock that runs slow.

    Returns the wall-clock instant the sample at `mark_at` went out — which is
    the end of the speech, and the instant the driver stops talking.
    """
    step = int(RATE * 20 / 1000) * 2          # 20 ms of 16-bit mono
    start = time.perf_counter()
    marked = None
    for i in range(0, len(pcm), step):
        target = start + (i / 2) / RATE
        delay = target - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        await conn.input_audio_buffer.append(
            audio=base64.b64encode(pcm[i:i + step]).decode("ascii"))
        if mark_at is not None and marked is None and i + step >= mark_at:
            marked = time.perf_counter()
        # Stop early once there is nothing left to wait for. A microphone
        # keeps streaming; this is the cheap equivalent.
        if stop_when is not None and marked is not None and stop_when():
            break
    return marked if marked is not None else time.perf_counter()


class Log:
    """Everything the session says, once, for the whole run.

    One reader for the connection rather than one per utterance. The first
    version of this started and cancelled a reader around each utterance and
    measured nothing at all: events that arrive while nobody is iterating are
    events nobody sees, and cancelling an iterator mid-turn throws away the
    very moment being measured.
    """

    def __init__(self):
        self.events = []          # (wall_clock, name)

    def since(self, t, name):
        return [w for w, n in self.events if n == name and w >= t]

    def first(self, t, name):
        got = self.since(t, name)
        return got[0] if got else None


async def run_one(conn, log, pcm, tail_ms, budget_s):
    """Feed one utterance, with its tail of silence, and time the turn end.

    THE SILENCE HAS TO BE SENT. A voice activity detector observes silence in
    the audio it is given; it cannot observe an absence of audio. The first
    version of this stopped feeding at the end of the speech and waited — and
    the turn never ended, in any config, because from the server's side the
    driver had simply stopped mid-sentence and never resumed.
    """
    mark = len(pcm)
    start = time.perf_counter()

    # THE MICROPHONE DOES NOT STOP. A detector observes silence in the audio
    # it is given and cannot observe an absence of audio, so the tail here is
    # long enough to cover the slowest decision any config makes rather than
    # sized to the fastest. Sized to server_vad's 700 ms window it looked like
    # semantic_vad was hanging on one turn in ten; it was this file running
    # out of silence to send while the detector was still thinking.
    payload = pcm + silence(tail_ms)
    committed = []

    def done():
        return bool(log.first(start, "input_audio_buffer.committed"))

    t_end = await feed(conn, payload, mark, stop_when=done)
    deadline = time.perf_counter() + budget_s
    while time.perf_counter() < deadline:
        if done():
            break
        await asyncio.sleep(0.02)
    await asyncio.sleep(0.2)
    return start, t_end


async def bench(label, td, want_responses, repeat=1, do_noise=False):
    from openai import AsyncOpenAI

    cfg = realtime.session_config()
    cfg["audio"]["input"]["turn_detection"] = dict(
        td, create_response=want_responses, interrupt_response=False)
    client = AsyncOpenAI()

    latencies, stop_latencies, resp_latencies, clean_splits = [], [], [], 0
    false_cuts, pauser_detail, by_line = 0, [], []
    noise_starts = None
    log = Log()

    async with client.realtime.connect(
            model=config.OPENAI_REALTIME_MODEL) as conn:
        await conn.session.update(session=cfg)

        async def reader():
            async for ev in conn:
                log.events.append((time.perf_counter(), ev.type))
                if ev.type == "error":
                    print(f"      ! {getattr(ev.error, 'message', '')[:110]}")

        task = asyncio.create_task(reader())
        await asyncio.sleep(0.5)

        for text in CLEAN * repeat:
            start, t_end = await run_one(conn, log, say(text), TAIL_MS, 2.0)
            stopped = log.first(start, "input_audio_buffer.speech_stopped")
            committed = log.first(start, "input_audio_buffer.committed")
            created = log.first(start, "response.created")
            if stopped:
                stop_latencies.append((stopped - t_end) * 1000)
            if committed:
                latencies.append((committed - t_end) * 1000)
            if created:
                resp_latencies.append((created - t_end) * 1000)
            if len(log.since(start, "input_audio_buffer.committed")) > 1:
                clean_splits += 1
            by_line.append((text, (committed - t_end) * 1000 if committed
                            else None))
            await asyncio.sleep(0.4)

        # REPEATED, LIKE THE CLEAN SET. Running each pauser once produced a
        # flat 0/8 for semantic_vad and it is not that good: the judgement is
        # probabilistic, and a single trial per gap measures a coin once.
        for spec, gap in PAUSERS * repeat:
            head, tail = spec.split("|", 1)
            pcm = say(head) + silence(gap) + say(tail)
            start, _ = await run_one(conn, log, pcm, TAIL_MS, 2.0)
            # More than one turn out of one sentence is the sentence being cut
            # in half. That is the whole failure, counted.
            cut = len(log.since(start, "input_audio_buffer.committed")) > 1
            false_cuts += 1 if cut else 0
            pauser_detail.append((gap, cut))
            await asyncio.sleep(0.4)

        # A QUIET CABIN, WITH NOBODY TALKING IN IT.
        #
        # server_vad takes a threshold and this one is set to 0.62 rather than
        # the default 0.5 for exactly this reason: what crosses it in a car is
        # very often RIO's own voice returning, a cough, an indicator. Every
        # crossing mutes her. semantic_vad takes no threshold at all, so what
        # it does with a noise floor is a thing to measure rather than assume.
        if do_noise:
            import random
            rng = random.Random(7)
            amp = 900                      # a quiet cabin, well under speech
            n = int(RATE * 8)
            noise = b"".join(
                int(rng.gauss(0, amp)).to_bytes(2, "little", signed=True)
                for _ in range(n))
            mark = len(log.events)
            await feed(conn, noise, None)
            await asyncio.sleep(0.5)
            noise_starts = len([1 for _, e in log.events[mark:]
                                if e == "input_audio_buffer.speech_started"])

        task.cancel()

    return {
        "label": label,
        "latency": latencies,
        "stop_latency": stop_latencies,
        "response_latency": resp_latencies,
        "clean_splits": clean_splits,
        "clean_n": len(CLEAN) * repeat,
        "by_line": by_line,
        "noise_starts": noise_starts,
        "false_cuts": false_cuts,
        "pausers": pauser_detail,
    }


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round((p / 100) * (len(xs) - 1))))]


def report(rows):
    print("\n" + "=" * 78)
    print("TURN-END LATENCY — last sample of speech to the turn ending")
    print("=" * 78)
    print(f"  {'config':<24}{'p50':>8}{'p95':>8}{'max':>8}   "
          f"{'clean':>6} {'false cuts':>11}")
    for r in rows:
        lat = r["latency"]
        if not lat:
            print(f"  {r['label']:<24}{'no turns ended':>32}")
            continue
        print(f"  {r['label']:<24}"
              f"{statistics.median(lat):>7.0f}m"
              f"{pct(lat, 95):>7.0f}m{max(lat):>7.0f}m   "
              f"{len(lat) - r['clean_splits']:>2}/{r['clean_n']:<3}"
              f"{r['false_cuts']:>7}/{len(r['pausers']):<4}")
    if any(r["response_latency"] for r in rows):
        print("\n  ...to response.created (the same wait, one hop further):")
        for r in rows:
            rl = r["response_latency"]
            if rl:
                print(f"  {r['label']:<24}{statistics.median(rl):>7.0f}m"
                      f"{pct(rl, 95):>7.0f}m")
    if any(r.get("noise_starts") is not None for r in rows):
        print("\n  a quiet cabin, eight seconds, nobody talking:")
        for r in rows:
            if r.get("noise_starts") is not None:
                print(f"  {r['label']:<24}{r['noise_starts']:>3} "
                      f"false speech starts")

    print("\n  which pauses got cut, by gap length:")
    gaps = sorted({g for _, g in PAUSERS})
    print(f"  {'config':<24}" + "".join(f"{g:>7}ms" for g in gaps))
    for r in rows:
        cells = []
        for g in gaps:
            hits = [c for gg, c in r["pausers"] if gg == g]
            cells.append(f"{sum(hits)}/{len(hits)}")
        print(f"  {r['label']:<24}" + "".join(f"{c:>9}" for c in cells))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--only", default=None,
                    help="run one config by label substring")
    ap.add_argument("--responses", action="store_true",
                    help="let turns create real responses, and time those too")
    ap.add_argument("--json", default=None, help="write the raw rows here")
    ap.add_argument("--repeat", type=int, default=1,
                    help="passes over the clean set — p95 of ten samples is "
                         "not a p95")
    ap.add_argument("--noise", action="store_true",
                    help="also count what a quiet cabin does to the detector")
    args = ap.parse_args()

    todo = [c for c in CONFIGS if not args.only or args.only in c[0]]
    rows = []
    for label, td in todo:
        print(f"  running {label} ...", flush=True)
        rows.append(asyncio.run(bench(label, td, args.responses,
                                      args.repeat, args.noise)))
    report(rows)
    if args.json:
        Path(args.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
