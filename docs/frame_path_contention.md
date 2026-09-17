# Frame path: the headway loop shares the GPU with a 1 Hz Qwen generate

*Parked 2026-09-17. Finding is complete; the fix needs a measurement first.*

## What the drive showed (session 0233da0d, iPhone, 2026-09-17)

Eight `FRAMES_STALLED` across 5.7 minutes, every one the same shape:
`why:no_frames_sent, inflight:1, ws_open:true, since_result_ms 3.7–4.6 s`.
The client had one frame out, the socket was open, and no result came back
for four seconds — so it rebuilt the socket (`FRAMES_WS_LOST` → `_WS_READY`,
~1 s), dropping the frame in flight and resetting the fps/inflight ramp.

The server side, from the headway rows (4073 frames, `server_ms` p50 18 /
p90 26 / p99 34):

| stall (client) | last frame before the gap | its `depth` stage | gap with no rows |
|---|---|---|---|
| 18:33:04 | idx 86, server 441 ms | 389 ms | 4.2 s |
| 18:33:50 | idx 624, server 425 ms | 384 ms | 4.3 s |
| 18:35:39 | idx 1990, server 612 ms | (decode 471) | 5.0 s |
| 18:36:22 | idx 2525, server 515 ms | 457 ms | 5.1 s |
| 18:36:47 | idx 2798, server 413 ms | 378 ms | 5.5 s |
| 18:37:35 | idx 3410, server 16 ms | — | 5.8 s |
| 18:38:17 | idx 3888, server 418 ms | 379 ms | 4.8 s |
| 18:38:35 | idx 4058, server 412 ms | 375 ms | 5.4 s |

The depth pass costs ~7.6 ms on this card. In the frame before each gap it
costs ~380–460 ms (50×), and the frame after it never completes at all: the
worker is inside `session.process()` when the client gives up and the task is
cancelled at close, so no row is ever written for it.

`MAIN_THREAD_STALL` (the 250 ms page-side detector) reported nothing across
all eight. The page was not frozen. The 2026-09-16 attribution — "the page
not running" — is refuted by the detector built to test it.

## The mechanism

* `headway/live.py` takes `_gpu_lock` around lanes, depth and detect. Nothing
  else takes that lock.
* `vision.py` holds `_lock` around every Qwen3-VL-8B generate — the observer,
  `/perceive`, and `look`. That lock is NOT `_gpu_lock`; the two run on the
  same H200 with no coordination, so a Qwen prefill (a ~100 KB image) is a
  burst of large kernels the headway kernels queue behind.
* `OBSERVER_PERIOD_S = 1.0`: while a drive is live the observer runs a
  generate **every second** (`max_new_tokens 60` at 512 px). `/perceive`
  runs another every 15 s in headway mode — "only to keep the Perception
  caption column alive" — and on this drive two of those waited **45.7 s
  and 52.7 s** in `timing_ms.qwen` (the vision lock queue behind the
  observer and two `look` calls), five more 2.7–7 s.
* Four of the eight gaps fall inside a logged long `/perceive`; the others
  fall inside the observer's cadence, which is not logged per pass.

So: a 1 Hz caption model starves the safety loop it shares a card with, the
client reads the resulting 4 s wait as a dead pipe, and rebuilds.

## What was done now (small, certain)

* `app.py` headway worker: a frame refused because the session lock is busy
  now gets `{"op":"skip","reason":"busy"}` back. It used to be evicted
  **silently**, which left the client's `inflight` pinned at 1 — a guaranteed
  4 s stall on the very next tick.

## What is parked, and what would decide it

1. **Client: a late result on an open socket is not a dead pipe.** In
   `rio_frames.js` `supervise()`, when `stalled()` fires with `mode==='ws'
   && wsOpen && inflight > 0`, release the inflight slot and keep sending
   (the server slot is newest-wins and evicts the stale frame) and note
   `FRAMES_LATE_RESULT`; rebuild only on the existing `no_results`
   (2×stallMs) branch or a closed socket. Results must be seq-aware so the
   late result does not release the next frame's slot.
2. **Server: priority.** The safety loop should not queue behind a caption.
   Candidate: run lanes/depth/detect on a high-priority CUDA stream
   (`torch.cuda.Stream(priority=-1)`), so headway kernels are scheduled
   ahead of Qwen's at kernel boundaries. Must be measured, not assumed:
   `tools/frame_transport_bench.py --seconds 40 --unshaped` alone, then with
   a `/perceive` every 2 s, then the same with the stream — compare
   `server_ms` p99/max and the count of gaps > 1 s in the session JSONL.
   If it does not move, the alternative is coordination: the observer yields
   (longer period, or skips a tick) while a headway frame is inside
   `_gpu_lock`, and `/perceive` is not requested at all in headway mode.
3. **Witnesses, cheap:** stamp every headway row with `vision_busy` (is the
   Qwen lock held, and by which caller) so the correlation above is a field
   rather than a timestamp overlay; split `/perceive`'s `qwen` timing into
   lock-wait and generate; add a 250 ms event-loop lag task on the server,
   the counterpart of the page's `MAIN_THREAD_STALL`.

None of this touches the voice path: the realtime session is browser ↔ API,
the pod's GPU is not in it. Frame contention delays tool answers (a `look`
took 4.3 s), which delays first audio; it does not stop audio.
