#!/usr/bin/env python3
"""frame_contention_probe.py — the drive's server-side load, reproduced, with stacks.

    python3 tools/frame_contention_probe.py --seconds 90
    python3 tools/frame_contention_probe.py --seconds 90 --no-observer --no-perceive

WHY THE BENCH WAS NOT ENOUGH. tools/frame_transport_bench.py measures the
transport and only the transport: it starts a session and streams frames, and
nothing else the drive runs is running. On 2026-09-17 the drive lost a frame
every ~45 s to a 4-5 s server-side stall the bench could not reproduce at all
(A and B above it: server_ms max 46 ms). The observer -- a Qwen generate every
second -- starts ONLY on a realtime-session mint (app.py:747), never on
/session/start, so it was absent from every bench run; so were the teachers'
session state and the page's /perceive every 15 s.

WHAT THIS DOES. The same session shape the page builds, minus the phone:
  1. /session/start, then POST /realtime/session?session_id=... -- the mint
     that starts the observer and the teachers for THIS session (the client
     secret it returns is never used).
  2. /headway_ws at ~10 fps from runs/road_clip.mp4, inflight <= 2, newest
     wins, exactly as rio_frames.js does it.
  3. /perceive every 15 s from a frame of the same clip, as the page does in
     headway mode.
  4. Whenever a result is more than --late-ms late, `py-spy dump` on the
     server, so the stall is read off the stack instead of inferred.
Afterwards it summarises the session's own JSONL: server_ms and depth
percentiles, frames > 200 ms, inter-row gaps > 1 s, and each stack it caught.
"""
import argparse, asyncio, json, os, shutil, subprocess, sys, time
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
import cv2
import httpx


def load_frames(clip, limit, max_side=640, quality=70):
    cap = cv2.VideoCapture(str(clip))
    out = []
    while len(out) < limit:
        ok, f = cap.read()
        if not ok:
            break
        h, w = f.shape[:2]
        s = max_side / max(h, w)
        f = cv2.resize(f, (int(w * s), int(h * s)))
        ok, buf = cv2.imencode('.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, quality])
        out.append(bytes(buf))
    if not out:
        raise SystemExit(f"no frames from {clip}")
    return out


def server_pid():
    try:
        pids = subprocess.check_output(['pgrep', '-f', 'uvicorn app:app']).decode().split()
    except Exception:
        return None
    for p in pids:
        try:
            cmd = open(f'/proc/{p}/cmdline', 'rb').read().split(b'\0')[0]
            if b'python' in cmd or b'uvicorn' in cmd:
                return int(p)
        except Exception:
            pass
    return None


def dump_stacks(pid):
    if not pid or not shutil.which('py-spy'):
        return None
    try:
        return subprocess.run(['py-spy', 'dump', '--pid', str(pid), '--nonblocking'],
                              capture_output=True, text=True, timeout=10).stdout
    except Exception as e:
        return f"py-spy failed: {e}"


def interesting(stack_text):
    """The threads that matter, trimmed: anything in a model call or a lock."""
    keep, cur = [], []
    for line in (stack_text or '').splitlines():
        if line.startswith('Thread'):
            if cur and any(k in ''.join(cur) for k in ('generate', 'depth_map', 'detect', 'detect_lanes', 'acquire', 'process (', 'observe', 'perceive', 'on_frame', 'synchronize')):
                keep.append('\n'.join(cur[:9]))
            cur = [line]
        else:
            cur.append(line)
    if cur and any(k in ''.join(cur) for k in ('generate', 'depth_map', 'detect', 'acquire', 'process (', 'observe', 'perceive', 'on_frame')):
        keep.append('\n'.join(cur[:9]))
    return keep


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='http://127.0.0.1:8888')
    ap.add_argument('--clip', default=str(REPO / 'runs' / 'road_clip.mp4'))
    ap.add_argument('--seconds', type=float, default=90.0)
    ap.add_argument('--fps', type=float, default=10.0)
    ap.add_argument('--late-ms', type=float, default=1500.0)
    ap.add_argument('--perceive-every', type=float, default=15.0)
    ap.add_argument('--no-observer', action='store_true')
    ap.add_argument('--no-perceive', action='store_true')
    args = ap.parse_args()
    import websockets

    frames = load_frames(args.clip, 400)
    base = args.base
    r = httpx.post(f'{base}/session/start', json={'metadata': {'source': 'contention_probe'}}, timeout=15)
    sid = r.json()['session_id']
    print(f'session {sid}')
    if not args.no_observer:
        r = httpx.post(f'{base}/realtime/session?session_id={sid}&client_id=probe', timeout=60)
        j = r.json()
        print(f'realtime mint: {"ok" if j.get("client_secret") else j.get("error")}  (observer + teachers started for this session)')
    pid = server_pid()
    print(f'server pid {pid}, py-spy {"present" if shutil.which("py-spy") else "ABSENT"}')

    url = base.replace('http://', 'ws://') + f'/headway_ws?session_id={sid}&client_id=probe'
    stats = {'sent': 0, 'results': 0, 'late': 0, 'skipped': 0, 'stacks': [], 'lates': []}
    inflight = {}      # seq -> sent_at
    stop = time.perf_counter() + args.seconds

    async def perceiver():
        if args.no_perceive:
            return
        async with httpx.AsyncClient(timeout=120) as c:
            i = 0
            while time.perf_counter() < stop:
                t0 = time.perf_counter()
                try:
                    r = await c.post(f'{base}/perceive?session_id={sid}',
                                     files={'image': ('f.jpg', frames[(i * 37) % len(frames)], 'image/jpeg')})
                    tm = (r.json() or {}).get('timing_ms', {})
                    print(f'  perceive #{i}: {time.perf_counter() - t0:.1f} s  qwen={tm.get("qwen")}', flush=True)
                except Exception as e:
                    print(f'  perceive #{i} failed: {e}', flush=True)
                i += 1
                await asyncio.sleep(max(0.5, args.perceive_every - (time.perf_counter() - t0)))

    async with websockets.connect(url, max_size=None) as ws:
        ready = json.loads(await ws.recv())
        print(f'ws ready, tuning max_inflight={ (ready.get("tuning") or {}).get("max_inflight") }')

        async def reader():
            while True:
                try:
                    msg = json.loads(await ws.recv())
                except Exception:
                    return
                op = msg.get('op')
                seq = msg.get('seq')
                if op in ('skip', 'error'):
                    inflight.pop(seq, None); stats['skipped'] += 1; continue
                if op:
                    continue
                sent_at = inflight.pop(seq, None)
                stats['results'] += 1

        async def watchdog():
            while time.perf_counter() < stop:
                await asyncio.sleep(0.25)
                now = time.perf_counter()
                for seq, at in list(inflight.items()):
                    if (now - at) * 1000 > args.late_ms and seq not in [l[0] for l in stats['lates']]:
                        stats['late'] += 1
                        stats['lates'].append((seq, round((now - at) * 1000)))
                        print(f'  LATE: seq {seq} waiting {round((now - at) * 1000)} ms -> dumping stacks', flush=True)
                        st = dump_stacks(pid)
                        stats['stacks'].append((seq, interesting(st)))

        rt = asyncio.create_task(reader())
        wt = asyncio.create_task(watchdog())
        pt = asyncio.create_task(perceiver())
        seq = 0
        i = 0
        period = 1.0 / args.fps
        while time.perf_counter() < stop:
            t = time.perf_counter()
            if len(inflight) < 2:
                seq += 1
                header = json.dumps({'seq': seq, 'cap_t': time.time(), 'v': 12.0, 'va': 0.1, 'src': 'clip'}).encode()
                out = len(header).to_bytes(4, 'big') + header + frames[i % len(frames)]
                await ws.send(out)
                inflight[seq] = t
                stats['sent'] += 1
                i += 1
            await asyncio.sleep(max(0, period - (time.perf_counter() - t)))
        await asyncio.sleep(1.0)
        wt.cancel(); rt.cancel(); pt.cancel()
    httpx.post(f'{base}/session/end?session_id={sid}', timeout=15)

    # ---- the session's own record ----
    path = REPO / 'training_data' / f'{sid}.jsonl'
    rows = [json.loads(l) for l in open(path)]
    hw = [(r['t'], r['payload']) for r in rows if r['kind'] == 'headway']
    def pct(xs, p):
        xs = sorted(xs); return xs[min(len(xs) - 1, int(len(xs) * p))] if xs else None
    sm = [p['server_ms'] for _, p in hw if isinstance(p.get('server_ms'), (int, float))]
    dp = [x for x in ((p.get('timing_ms') or {}).get('depth') for _, p in hw) if isinstance(x, (int, float))]
    gaps = [b - a for (a, _), (b, _) in zip(hw, hw[1:])]
    print(f'\nsent={stats["sent"]} results={stats["results"]} skipped={stats["skipped"]} late(>{args.late_ms:.0f}ms)={stats["late"]}')
    if sm:
        print(f'server_ms p50={pct(sm,.5):.0f} p90={pct(sm,.9):.0f} p99={pct(sm,.99):.0f} max={max(sm):.0f}  >200ms={sum(1 for x in sm if x>200)}')
        print(f'depth     p50={pct(dp,.5):.1f} p99={pct(dp,.99):.1f} max={max(dp):.0f}')
        print(f'inter-row gaps >1s: {sum(1 for g in gaps if g>1)}  largest: {[round(g,2) for g in sorted(gaps)[-3:]]}')
    for seq, ms in stats['lates']:
        print(f'  late seq {seq}: {ms} ms')
    for seq, threads in stats['stacks']:
        print(f'\n--- stacks at late seq {seq} ---')
        for t in threads:
            print(t)
    print(f'\nsession file: {path}')

asyncio.run(main())
