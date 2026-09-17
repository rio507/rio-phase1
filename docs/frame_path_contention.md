# Frame path: what shares the card with the headway loop, measured

*2026-09-17. Measured on the pod (H200), against the running server, with
`tools/frame_contention_probe.py` and `tools/frame_transport_bench.py`.*

## The drive (session 0233da0d)

Eight `FRAMES_STALLED` in 5.7 minutes, every one `inflight:1, ws_open:true,
since_result ~4 s`; the client rebuilt the socket each time and lost the frame
in flight. Before seven of the eight, the last processed frame shows the depth
pass at 375–457 ms against its 7.6 ms median, and the next frame never
completes — the worker is inside `session.process()` when the client gives up
and its task is cancelled at close, so no row is ever written for it.

The page-side `MAIN_THREAD_STALL` detector reported nothing across all eight:
the page was not frozen.

## What the measurements say

| condition | headway `server_ms` p50/p99/max | depth max | `/perceive` | observer |
|---|---|---|---|---|
| bench alone (no observer) | 28 / 42 / 46 | 15 | — | not running |
| bench + `/perceive` every 2 s | 28 / 40 / 46 | 15 | **53 s each** | not running |
| probe: observer live, perceive every 15 s (pre-fix) | 17 / 30 / 445 | **400** | **53 s, 21 s** | **3 ticks / 90 s**, lock wait up to 52.5 s |
| probe, same, after the fix | 17 / 26 / 453 | 406 | **5 ms** (caption from the observer, ~0.5 s old) | **85 ticks / 90 s**, lock wait 0 |
| headway frames while the observer's generate runs (post-fix) | 22.5 / 27 / 29 | 13 | | |

Read across:

1. **A Qwen generate does not slow the headway loop.** `/perceive` ran for
   53 s beside the bench and the loop's p99 did not move. The morning's
   "observer on the same GPU starves the loop" claim is refuted for the
   decode phase. The measured tax of the observer generating at the same
   instant as a frame is ~6 ms (22.5 vs 16.7 ms p50), no spikes.
2. **`/perceive`'s own generate is pathological under load.** 2.7 s on an
   idle card; 21–53 s while frames flow (the `SIGUSR1` stack dump shows the
   thread inside `perceive._query_qwen → generate` the whole time — not
   waiting on the lock). It holds the model lock for that long, so the
   observer — RIO's eyes — got 3 ticks in 90 s instead of ~90 and waited up
   to 52.5 s for the lock; on the drive that is why two `/perceive` calls
   took 45.7 s and 52.7 s and why `look` answers described old frames.
3. **The 400 ms depth spike is a prefill on the shared default stream.** In
   the pre-fix probe both spikes land within 400 ms of a `/perceive` start
   (768 px image → a burst of large kernels the depth kernels queue behind).
   The drive's spikes match `/perceive` and `look` starts.
4. **The teachers did not do it.** Alpamayo and Cosmos are separate
   processes on this card (18 GB and 23 GB, 3.8 s and 7.1 s generates), the
   exact shape that would freeze a frame for seconds from outside — but the
   client pauses them whenever a voice session is live (`paused()` /
   `TEACHER_PAUSE_DURING_SESSION`), their service logs show no `/infer`
   during the drive, and the corpus has no rows from today. Caveat: the pause
   has a TTL (`TEACHER_SESSION_TTL_S` = 180 s) that a 218 s conversation
   outlives.
5. **The 4–5 s gap did not reproduce** in three 90 s probe runs (0 gaps,
   0 late frames). Its cause is not established. Two candidates are now
   instrumented rather than argued: the event loop freezing (a session-row
   `write()` on the FUSE volume measured 603 ms once in 1320) and a teacher
   pass after the pause expired.

## What changed

* `/perceive` **defers to the observer while headway frames are flowing**
  (`PERCEIVE_DEFER_WHILE_FLOWING_S`, 4 s): deterministic geometry only, the
  caption is the observer's latest with its age, and the observer is started
  if none is running. Measured: 53 s → 5 ms; observer 3 → 85 ticks / 90 s.
* Session rows are written by a **writer thread**, never on the event loop;
  `end_session` flushes before closing.
* A frame refused for a busy session lock gets `skip busy` instead of a
  silence that pinned the client's `inflight`.

## Witnesses added (all in the drive's JSONL)

* every headway row: `vision_busy` at frame start and `vision_busy_end` at
  frame end — is the Qwen lock held, by whom (`observe` / `perceive` /
  `anchor`), for how long;
* `observer_tick` per generate: duration, `lock_wait_ms`;
* `/perceive` rows: `timing_ms.qwen_lock_wait`, `caption_source`,
  `caption_age_s`, `skipped`;
* `teacher_pass` per teacher inference: service, latency, queue, trigger;
* `server_loop_stall {late_ms}` from a 250 ms loop-lag task — the server's
  `MAIN_THREAD_STALL`;
* `kill -USR1 <pid>` dumps every thread's stack to `uvicorn.log`
  (`faulthandler`; py-spy cannot attach in this container).

## Still open

* A residual ~400 ms depth spike, twice per 90 s, with the Qwen lock free at
  both ends of the frame and no teacher pass. Unexplained; the witnesses
  above will place it.
* The client still rebuilds the socket on one late result. The right
  response is documented in the previous version of this file (release the
  inflight slot, keep sending, rebuild only on `no_results`); not done.
* GPU priority for the safety loop (`torch.cuda.Stream(priority=-1)`) was
  not tried: the measurement says the decode phase is not the problem and
  the prefill spike is 400 ms, so the case for it is weaker than assumed.
