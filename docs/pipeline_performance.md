# The frame pipeline, measured per stage, in both modes

**The complaint:** playing an uploaded clip on the dashboard is noticeably slow.

**The answer:** two independent causes, one of them a day old and one of them
six weeks old, and neither of them in the code the complaint pointed at.

---

## How it was measured

`tools/pipeline_probe.py` drives the **real page in a real browser** through the
real socket, and reads the page's own numbers back out. Nothing in it
re-derives a timing — a probe that computed its own would be measuring itself.

`tools/frame_transport_bench.py` already measures the transport and the server
beautifully, and cannot see the half that lives on the page: how often a
picture is actually taken, what the encode costs, what the overlay draw costs,
and what those do to each other on a shared thread. In clip mode that is where
the work is.

Both modes, same road: `clip` is a file through the real file input with
auto-detect on play; `camera` is `getUserMedia` fed by Chromium's fake device
from the same footage.

**All numbers below are loopback.** The transport leg is 2.8 ms, so these are
server and browser costs with the radio taken out. They are not comparable to
the 195 ms / 318 ms frame ages from the mobile drive log, and nothing here
claims to be an improvement on those.

---

## 1. Teacher contention — confirmed, and sharper than expected

Three conditions, same clip, same server, camera mode:

| condition | detector | depth | lanes | server total | frame age |
|---|---|---|---|---|---|
| teachers **unloaded** | 4.2 ms | 5.7 | 2.9 | 15.7 ms | 19.5 ms |
| teachers **loaded but idle** | 4.5 ms | 6.2 | 3.1 | 16.7 ms | 20.1 ms |
| teachers **loaded and inferring** | **26.2 ms** | 15.4 | 8.3 | **51.5 ms** | **55.0 ms** |

**Read the first two rows against each other before the third: being loaded
costs nothing.** Nineteen gigabytes of resident weights on the same card are
free until something runs in them. It is the *inference* that costs — **5.8× on
the detector, 3.3× on the whole server frame, 2.8× on frame age.**

That is what decides the fix. Unloading is not it: reloading is 45 seconds and
the corpus is the point of the panel. Running them **less, one at a time, and
not at all while the driver is being listened to** is.

### What the panel was actually doing

Every **2 seconds** while frames flowed, a keyframe went to **both** teachers,
**concurrently**. During a 25-second run that is ~12 keyframes × 2 models of
8–10B decoding inside a pipeline whose whole frame budget is 16 ms.

### The three rules, and where each is enforced

| rule | where | shipped value |
|---|---|---|
| **lower cadence while frames are flowing** | `teachers/panel.py` `_floor_s()` | 8 s, vs 2 s idle |
| **one teacher inferring at a time** | `teachers/client.py` `_GATE` | `TEACHER_MAX_CONCURRENT=1` |
| **pause outright while a live session is open** | `client.paused()` | on |

The existing `hold_gpu()` — "RIO is answering right now" — was already there and
is unchanged. These cover the rest: `_floor_s()` keys on **frames arriving**,
not on which button was pressed; the gate is module-level so it is shared by
both clients, because the card is shared.

**The event triggers are not gated by the floor at all.** Band change, nav
maneuver, driver question and IMU jolt still raise a keyframe immediately.
The floor is the "nothing happened" sampler, and nothing happening is the
class that can afford to be sampled four times a minute instead of thirty.

### The pause cannot leak

A counter incremented on open and decremented on close is correct exactly as
long as every close arrives — and this system already has a reaper for the
cases where it does not (flat phone, slept laptop, tab closed while hidden). A
leaked increment would pause the teachers **for the life of the process**: the
card goes quiet, no corpus row is ever written again, and nothing looks broken.

So the pause carries a **deadline** (`TEACHER_SESSION_TTL_S`, 180 s) refreshed
by the heartbeat the page already sends. If nothing keeps saying the session is
open, the teachers resume on their own. `tools/teacher_yield_selftest.py`
asserts exactly that.

---

## 2. Clip mode — a serial loop sending four times the bytes

Clip mode is **not the socket pipeline**. `runVideo()` in `static/index.html` is
its own loop: `requestVideoFrameCallback` → encode → **await** a POST to
`/headway_frame` → repeat. One frame at a time, no pipeline depth, no adaptive
rate, no adaptive quality.

And its encode was the **July 2026 one**:

| | clip (`frameFrom`) | socket (`rio_frames.js`) |
|---|---|---|
| resolution | the video's own — 1282×684 | long side capped at **640** |
| quality | fixed q0.8 | adaptive 0.40–0.78 against a 24 KB budget |
| canvas | **a new one every frame** | one, reused |
| measured bytes | **94.8 KB** | 21.4 KB |

Every consumer works at a few hundred pixels a side — the detector letterboxes
to 560, depth to 518, lanes to 800×320. The extra pixels were decoded by the
server and thrown away, and they cost the encode, the POST **and** the server's
own JPEG decode (7.5 ms against 1.8) on every single frame.

Now it reads the same tuning from the same place, so there is one answer on this
page to "how big is a frame".

### What was *not* found

The brief suspected frames might be encoded twice — once for detection, once
for the teachers. **They are not.** The keyframe builder reuses the uploaded
JPEG bytes straight off the frame ring (`teachers/keyframe.py` → `f.jpeg`), and
`RingFrame` decodes once and caches. One encode already feeds every consumer.

The scout element is also not the problem: it is a second decode of a file
already on disk, and it is what makes the boxes land on the pixels that
produced them.

---

## 3. Before / after

Both with the teacher panel **enabled and running** — i.e. the shipped state,
not a favourable one.

### Uploaded clip (auto-detect on play)

| stage | before | after | |
|---|---|---|---|
| frame bytes | 94.8 KB | **19.7 KB** | −79 % |
| effective fps | 9.96 | **14.84** | +49 % |
| encode | 13.5 ms | **9.2 ms** | |
| push (send→result) | 72.3 ms | **32.5 ms** | −55 % |
| — jpeg decode | 7.5 ms | **2.0 ms** | |
| — lanes | 9.0 ms | **5.0 ms** | |
| — depth | 21.9 ms | **11.4 ms** | |
| — detector | 29.5 ms | **9.4 ms** | −68 % |
| — server total | 67.4 ms | **27.9 ms** | −59 % |
| **frame age at detection** | **85.8 ms** | **42.2 ms** | **−51 %** |

### Live camera

| stage | before | after | |
|---|---|---|---|
| effective fps | 14.97 | 14.97 | — |
| push | 53.3 ms | **29.0 ms** | −46 % |
| — depth | 15.4 ms | **10.7 ms** | |
| — detector | 26.2 ms | **9.3 ms** | −64 % |
| — server total | 51.5 ms | **27.1 ms** | −47 % |
| **frame age at detection** | **55.0 ms** | **30.4 ms** | **−45 %** |

### …and with a live conversation open, where the teachers stop entirely

| | camera | clip |
|---|---|---|
| detector | **4.4 ms** | **4.6 ms** |
| server total | **16.4 ms** | **17.3 ms** |
| frame age | **20.0 ms** | 31.4 ms |

Within noise of the teachers-unloaded row. **While the driver may speak, the
live pipeline gets the card to itself** — which is the rule the brief asked for,
demonstrated end to end rather than asserted.

A note on clip fps: it is quantised to the clip's own frame rate, because the
loop waits for the next *presented* frame. At 24 fps the steps are 41.7 ms
wide, so the figure moves between ~10 and ~15 fps as loop work crosses a
boundary. Frame age is the number to read; fps is the number that steps.

---

## 4. What regressed, and when

**Nothing in the frame pipeline itself.** With the teachers quiet, the detector
is 4.2–4.6 ms against a known-good kernel of ~3.5 ms, and the whole server
frame is 15.7 ms against a known-good ~21 ms. The pipeline is *faster* than its
last good numbers.

Two separate things:

**Teacher contention — `89ab1ed`, 2026-09-10, one day before this
investigation.** "A teacher panel: two driving models watching, and driving
nothing." The panel was measured against `look()` when it landed (1.9× on an
observer forward pass, documented, with the answering-path yield built for it).
It was never measured against the **frame pipeline**, which runs continuously
rather than on demand, and which is where a 2-second cadence lands hardest.

**The clip encode — never regressed; it was left behind.** `frameFrom` is from
`f3336c1`, 2026-07-29, the original headway feature. The transport rebuild that
capped frames at 640 px and made quality adaptive is `2576c67`, 2026-09-08 —
and it changed `rio_frames.js` only. Clip mode kept the July encode for six
weeks because the work that replaced it was scoped to the socket, and the
clip path does not use the socket.

That second one is the more useful lesson: the fix landed in the transport, and
the mode that does not use the transport did not get it.

---

## 5. Running it

```
  python -m tools.pipeline_probe                      # both modes, per stage
  python -m tools.pipeline_probe --mode clip --seconds 40
  python -m tools.teacher_yield_selftest              # the three yield rules
  bash boot.sh teachers-stop                          # the unloaded condition
  RIO_TEACHERS_ENABLED=0 bash boot.sh restart         # the loaded-but-idle one
```

`RIO_TEACHERS_ENABLED` is new. A shadow feature that cannot be turned off
without an edit is a shadow feature nobody turns off to find out whether it is
the problem — which is exactly the measurement this needed.
