# Frame transport — what the first real drive measured, and what replaced it

Punch-list item 1. Written after session `06af3214` (iPhone, 2026-09-08,
607.8 s, 1949 frames), which is the first drive of this system on a real road
with a real phone on a real mobile link.

---

## 1. What the drive log actually says

Everything below is from `training_data/06af3214-…​.jsonl`.

| measurement | value |
|---|---|
| session span | 607.8 s |
| headway frames | 1949 (3.23 fps mean, 3.88 fps at p50) |
| inter-frame arrival | p10 223 ms · **p50 258 ms** · p90 406 ms · p99 1503 ms · max 2000 ms |
| frames more than 0.5 s apart | 103 (5.3 %) |
| frames more than 1.0 s apart | 34 |
| server processing (`total`) | **p50 23.6 ms** · p90 33.5 ms · max 1433 ms |
| — detect | p50 9.1 ms |
| — depth | p50 6.8 ms |
| — lanes | p50 3.0 ms |
| — JPEG decode | p50 3.0 ms |
| — membership | p50 1.2 ms |
| — track/filter | p50 0.3 ms |
| frame bytes (from `/perceive` on the same feed) | 71–90 KB |
| **frames carrying a capture timestamp** | **0 of 1949** |

Three conclusions, in order of how much they mattered.

**The server was never the bottleneck.** It spent 23.6 ms on a frame that
arrived every 258 ms. Nothing in the pipeline needed to get faster.

**The page's own floor was the frame rate.** `CADENCE_MS = 250` in
`static/index.html`, on a chained timeout that waited for each result. The
observed p50 of 258 ms is that floor plus scheduling jitter, which means the
whole round trip fitted inside 250 ms most of the time and the loop simply sat
there. The tail — 406 ms at p90, 1.5 s at p99 — is the round trip *escaping*
the floor, i.e. the mobile link, and 5.3 % of frames were more than half a
second apart.

**Nothing knew how old a picture was.** This is the one that mattered most and
the reason the punch list asks for a before/after number at all. `frame_t` was
an accepted form field on `/headway_frame` and the page never sent it. So the
server could measure when a frame *arrived* and never how old it *was*, and
"frame age at detection" — the number that decides whether a gap warning is
about the road the car is on or the road it was on four frames ago — did not
exist in the record. What existed was server processing time, which was fine,
and which is not the same question.

### What the log could not say, and why

Upload time per frame is **not separable from this log**. The server sees an
arrival instant and a processing duration; with no capture stamp there is no
third point to subtract. It can be bounded — the round trip was under 250 ms
for roughly two thirds of frames, and reached ~406 ms at p90 and ~1.5 s at p99
— but not measured. That gap is closed first (§2), and the before/after numbers
in §4 are measured on a bench (§3) precisely so the two transports are compared
with one ruler rather than one measured and one inferred.

---

## 2. What changed

### `cap_t` on every frame, on both transports

The capture instant, converted into the **server's** clock before it is sent.
The socket measures the offset itself (`{"op":"ping"}` → `{"op":"pong"}`,
midpoint of the round trip, keeping the lowest RTT seen). `/headway_frame`
takes the same field, so the POST path is measured identically and the
comparison is like-for-like.

Every result now carries `frame_age_ms` (capture → detection complete) plus the
three places that age was spent: `transport_ms`, `queue_ms`, `server_ms`. All
four are written into the drive's own JSONL (`sessions.log_headway`), so the
next real drive answers this question about itself.

### `/headway_ws` — a persistent binary socket

One connection for the drive. One binary message per frame:

```
uint32be  header length
bytes     UTF-8 JSON {seq, cap_t, v, va, src}
bytes     the JPEG
```

The result that comes back is the dict `/headway_frame` returns, plus the
timing fields. **No warning policy moved.** `session.process` is the same call,
`headway/live_policy.py` is untouched, and the browser still only plays what
the server told it to.

### Drop, never queue — and what "never" is doing there

*Server side*: exactly one slot for a waiting frame. A frame arriving while
another is being processed **replaces** whatever is waiting and the evicted one
is counted (`dropped`). This is strictly stronger than the POST path's
non-blocking lock, which dropped the *new* frame and carried on with the old
one — here the newest picture always wins, which is the only version of the
rule that stops frame age growing.

*Client side*: a tick that finds the pipe full, or `bufferedAmount` over
`HEADWAY_WS_BUFFER_LIMIT_BYTES`, **takes no picture at all**. It does not
encode one and it does not hold one.

### Pipeline depth is a control variable, not a constant

The first version held strictly one frame on the wire. On the link this drive
was measured over that gives `1 / (serialise + round trip)` ≈ 4 fps whatever
the loop asks for — measured at **161 ms p50 frame age and 3.85 fps, with 182
captures skipped**. Perfect freshness at the old frame rate, which is half an
answer.

So depth is allowed to reach `HEADWAY_WS_MAX_INFLIGHT` (3), and **frame age
decides**: depth is raised only while the smoothed age sits under 75 % of
target *and* captures are being skipped for want of a slot, and it collapses to
1 on the first frame over the ceiling. This is a pipeline depth, not a queue
length — the bound is hard on both sides, and the far end still holds exactly
one waiting frame.

When age climbs, the order is **depth, then rate, then quality**: a shallower
pipe costs nothing a Kalman filter can feel, and a blurrier car ahead does.

### Bytes, tuned for the link

The old path encoded at the camera's own resolution at a fixed q0.8: 71–90 KB
per frame, ~2.6 Mbit/s of uplink at 4 fps, and ~10 Mbit/s if it had ever
reached 15 — more than an LTE uplink has. Every consumer of these frames works
at a few hundred pixels a side (detector letterboxes to 560, depth to 518,
lanes to 800×320), so the long side is capped at 640 and quality walks between
0.40 and 0.78 against a 24 KB budget.

Measured cost to perception: **2.08 detections/frame at 94 KB full-resolution
vs 1.97 at 21 KB and 640 px**, on the same clip through the same detector.

All of it lives in `config.py` (`HEADWAY_WS_*`) and travels to the browser with
the socket's `ready` message, so the page holds no second copy of a number the
tests check.

### The POST path is still there

Not deprecated, not dead. It runs when `rio_frames.js` is absent, when a proxy
strips the upgrade, and after three failed socket attempts. A drive at 4 fps is
a working drive; a drive with no picture is not.

---

## 3. How the before/after was measured

`tools/frame_transport_bench.py`. Both transports, same clip, same frames, same
running server, same shaped link, back to back.

The link is **derived from the drive log rather than invented**: a ~80 KB frame
whose round trip reached 406 ms at p90, of which 24 ms was the server, leaves
~322 ms for 80 KB of uplink plus a base round trip — about **2.0 Mbit/s up at
60 ms RTT**, an ordinary LTE uplink with a voice session already on the same
radio. `--uplink-mbit` and `--rtt-ms` move it; `--unshaped` runs both with no
link model at all, which isolates the server-side floor.

Serialisation is modelled as a shared resource — two frames cannot be on the
radio at once — which is the exact property that makes queueing fatal.

---

## 4. Before / after

**Mobile link (2.0 Mbit/s up, 60 ms RTT), 25 s each:**

| | POST (shipped) | WebSocket | |
|---|---|---|---|
| frame age p50 | 486 ms | **152 ms** | −69 % |
| frame age p90 | 502 ms | **185 ms** | −63 % |
| frame age p99 | 516 ms | **193 ms** | −63 % |
| frame age max | 638 ms | **202 ms** | |
| delivered fps | 2.05 | **7.57** | 3.7× |
| bytes/frame | 90.3 KB | **21.4 KB** | −76 % |
| uplink used | 1.52 Mbit/s | **1.29 Mbit/s** | −15 % |
| settled depth | 1 (no choice) | 2 | |

**Unshaped (server-side floor), 25 s each:**

| | POST | WebSocket |
|---|---|---|
| frame age p50 | 42.7 ms | **27.6 ms** |
| frame age p90 | 48.7 ms | **28.6 ms** |
| frame age p99 | 77.7 ms | **33.3 ms** |
| delivered fps | 3.97 | **14.29** |

The headline: on the link the real drive ran over, the picture the detector
works on went from about half a second old to about a seventh of a second old,
while the frame rate nearly quadrupled and the phone sent *less* data than
before.

7.57 fps is below the 8–15 fps band the punch list asks for, and that is the
link's answer rather than the controller's: the loop asks for 15 and drops what
the radio cannot carry, because the alternative is queueing, and a queue is the
one failure this transport is not allowed to have. On the unshaped run — the
same code, a link that is not the constraint — it delivers 14.29 fps.

---

## 5. WebRTC video-track ingest (item 1b) — assessed, not built

The punch list asks for this to be assessed as the v2 path to a true 20–30 fps
stream, and says explicitly: **do not build it unless (a) proves
insufficient.** (a) did not. On the drive's own link it delivers 7.6 fps at
152 ms, and where the link allows, 14.3 fps at 28 ms. The recommendation is
therefore **not to build it now**. The assessment follows so the decision can
be revisited with numbers rather than re-derived.

### Feasibility

Real, and the pieces are already here. `aiortc` would terminate a WebRTC
session server-side; the browser already opens a peer connection for the
realtime voice session (`rio_realtime.js`), so a second `addTrack` on that same
connection is a small change on the page. The stack would be:

```
getUserMedia video track → RTCPeerConnection → aiortc MediaStreamTrack
  → recv() → av.VideoFrame → ndarray → the same session.process()
```

The pipeline behind `process()` does not care where its frames came from.

### Latency

**Better than the socket, and by less than it looks.** Real gains:

- **VP8/H.264 inter-frame coding.** A road scene at 15 fps is mostly the same
  scene; P-frames cost a fraction of an independent JPEG. Expect 4–8 KB/frame
  against the socket's 21 KB, which is the difference between the link being
  the constraint and not being it — worth roughly the gap between 7.6 fps and
  20 fps on the link measured here.
- **Congestion control that actually knows about the radio.** GCC/transport-cc
  reacts to queue build-up in milliseconds. The socket's controller reacts to a
  frame-age report, i.e. one round trip late.
- **Pacing and FEC** instead of whole frames dropped on a blip.

Against that, the parts of the 152 ms that WebRTC does **not** remove: the
~30 ms one-way link latency (it is the radio), the 24 ms of model time, and the
encode. Realistic estimate on the same link: **60–90 ms at p50 and 20–30 fps**,
against 152 ms and 7.6 fps.

There is also a latency *cost* that is easy to miss: a decoder has a jitter
buffer, and one tuned for smooth playback adds 30–100 ms of deliberate delay.
It has to be configured for low latency, and that configuration is the thing
that most often gets forgotten and silently gives back the win.

### Server complexity — the real reason not to do it yet

The socket is ~200 lines with one slot and one worker, and its failure mode is
a dropped frame. WebRTC ingest brings:

1. **A media stack in the request path.** `aiortc` pulls in `av` (FFmpeg
   bindings) and a full SRTP/ICE/DTLS implementation. The dependency surface of
   the frame path goes from "a JSON parse and a JPEG decode" to a codec.
2. **ICE.** STUN at minimum, TURN wherever a car's NAT is unfriendly, which is
   an operational service with its own credentials, bandwidth bill and failure
   modes. The socket needs a port that is already open.
3. **Decode cost, which is not free and is not obviously on the GPU.**
   `aiortc` decodes on the CPU by default. NVDEC through PyAV needs a
   hardware-accelerated FFmpeg build; getting the decoded surface to the models
   without a round trip through host memory needs more than that. Done badly
   this *adds* latency and steals CPU from an event loop that is also running
   the voice relay. Done well it is a build-and-deploy problem, and this pod's
   provisioning history (see `boot.sh`) is the argument for not adding one.
4. **The drop rule stops being ours.** The single most important property of
   the current design — newest frame wins, nothing ever queues — becomes a
   property of a decoder's frame queue and a jitter buffer, configured rather
   than written. That is a real loss of control over the one invariant that
   protects the warning's truthfulness.
5. **Frame age needs re-plumbing.** RTP timestamps are a media clock, not a
   wall clock. `cap_t` would have to travel out of band (a data-channel
   sidecar keyed by RTP timestamp) or the measurement is lost — and losing it
   is how this whole item started.

### Verdict

**Do not build.** Revisit when a real drive's own log — which now records
`frame_age_ms` on every frame — shows frame age sitting above
`HEADWAY_WS_MAX_AGE_MS` for a meaningful share of a drive with the rate already
at `HEADWAY_WS_MIN_FPS` and quality already at its floor. That is the condition
under which (a) has proved insufficient, it is measurable from the field now,
and it was not measurable before this item.

The cheaper thing to try first, if bytes turn out to be the binding
constraint: keep the socket and send **WebCodecs-encoded VP8/AV1 chunks**
instead of JPEGs. That buys most of the inter-frame coding win, keeps the one
slot, keeps `cap_t`, keeps the drop rule, and adds no server dependency at all
— the server would swap a JPEG decode for a codec decode and nothing else.
