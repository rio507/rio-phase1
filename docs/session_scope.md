# Whose is it? — session scope, and the audit after the second leak

On 2026-09-16 a drive failed twice in one way. Questions asked on a phone came
back describing a clip that had been uploaded on a desktop, and the phone's
geolocation prompt never appeared, so place search and local news answered about
somewhere else. Both are the same shape of fault: **state that belongs to one
client being read by another, and nothing on screen saying so.**

This is the second time that shape has appeared, which is why the audit in §4
exists rather than a third fix.

---

## 1. The frame leak: two bugs, and only one of them was the key

```
_visual_key(session_id)  ->  session_id or "default"
```

A live conversation does not require a drive. Two clients that both arrived
without a drive were both keyed `"default"`, so they shared one frame ring. The
desktop pushed a clip; the phone asked a question; the phone was answered.

**But the key was only half of it.** `observer.serve_to()` had always checked
frame *provenance* — it refused to serve an observation whose origin did not
name the asking session. `visual_qa` had not: it called `peek_ring(key)` and
trusted the key to mean the frames were the asker's.

That asymmetry is the real lesson. A key can be got wrong again — by a new
endpoint, a new client, a default that looks harmless. **A reader that verifies
provenance is wrong only once.**

### What changed

| | before | after |
|---|---|---|
| key without a drive | `"default"`, shared by every client | `client:<tab id>`, minted per page load |
| `visual_qa` read | `peek_ring(key)` — trusts the key | `framebuf.ring_for(key)` — verifies origin |
| `observer` read | its own copy of the rule | delegates to `framebuf.owns()` |
| the rule | in one file, used by one reader | in `framebuf`, used by every reader |

`framebuf.owns(origin, session_key)` is now the only statement of the rule:

- `"<session key>:<source>"` — answerable only to that session;
- `"api:<source>"` — a bench or harness; may satisfy a keyless caller and may
  **never** satisfy a named drive;
- absent — unknown provenance is not provenance, so it is refused.

A frame's origin is built from **the same key the ring is filed under**, so
"which buffer" and "whose road" cannot disagree. They did, and that was the bug.

One subtlety with teeth: a tab key (`client:tabA`) contains a colon, so the
source is split off the **right** of the origin. Splitting from the left reads
every tab's owner as `"client"` and hands them all the same frames — reinstating
the bug through the fix for it. `tools/session_isolation_selftest.py` pins it.

### The client id is not a session id

It never reaches `sessions.py`, is not logged as a drive, and is not persisted —
two tabs are two sources of pictures. It exists only so that *"whose pictures
are these"* has an answer before anyone presses Start Drive. It is a separate
query parameter for that reason: a fake `session_id` would sail into every
endpoint that looks a drive up and fail there instead.

---

## 2. The permission silence

The geolocation prompt never appeared, and **nothing said so** — which is the
part that cost the time, because a missing permission and a bad search look
identical from the passenger seat.

Three faults were hiding behind one silence:

1. ~~**Not a secure context.**~~ **This was my first diagnosis and the logs
   disprove it — see §2a.** It is a real failure mode and the page now detects
   it, but it is not what happened on 2026-09-16.
2. **The gesture was spent.** iOS grants a prompt only from a trusted gesture,
   and a gesture does not survive an `await`. All three requests are now
   *started* synchronously in the Start Drive handler and awaited afterwards —
   the order they are **made** in is everything; the order they resolve in does
   not matter.
3. **A refusal looked like an absence.** `startWatch()` set `lastFixAt` to *now*
   before any fix had arrived, so a drive that never received a position
   reported a **fresh** position age for its whole length. It is `null` until a
   fix actually arrives, and every reader treats that as "no fix".

## 2a. What the logs actually say, after the first diagnosis was wrong

I first concluded the phone was on a plain-http origin and built a TLS
terminator for it. **That was wrong, and the evidence was available before I
started.**

### The origin was not in the logs at all — which is the first finding

`uvicorn`'s default access log records the client address, the method, the path
and the status. **No `Host`, no `Origin`, no `X-Forwarded-*`** — zero matches
across every surviving generation. The question "was the page on a secure
origin?" was not answerable from the server, so it got inferred, and the
inference was wrong. That gap is now closed: `/session/start` records `origin`,
`host`, `forwarded_proto` and the browser's own `window.isSecureContext`
alongside the user agent it already kept.

### What the evidence does say

| evidence | reading |
|---|---|
| client addresses are all `100.64.x.x` | RFC 6598 carrier-grade NAT — the **RunPod proxy**, not a LAN. The phone reached the pod through `https://<pod>-8888.proxy.runpod.net`, which **is** a secure origin |
| six iPhone drives log `gps_error code=3 TIMEOUT` | an insecure origin raises **code 1 PERMISSION_DENIED immediately**, never a timeout |
| the only `code=1` in the whole corpus is on a **Mac** | one genuine denial, on a different machine, unrelated |
| `gps_watch_rearm` escalating 1 → 2 → 3 → 4 into coarse mode | the watch was armed and re-armed repeatedly, receiving nothing |

Two independent facts therefore rule out the secure-context theory: the origin
was a proxy https URL, and the error code was a timeout rather than a denial.

### The actual bug, which was ours

```
watchPosition(..., { timeout: 10000 })   // the browser gets 10 s
watchCfg.watchdog_s = 4.0                // we tore it down after 4
```

The watchdog exists for a good reason — a stalled GPS radio does not recover by
being asked twice, so the watch is rebuilt. But **"it was delivering and went
quiet" and "it has never delivered" were the same four-second clock**, and
before the first fix that clock is not measuring a stalled radio. It is
measuring *a human reading an iOS permission sheet and deciding to tap Allow*.

Re-arming cancels the pending request. The prompt never survives long enough to
be answered, `watchPosition` returns TIMEOUT, and the loop repeats — which is
exactly the escalation the logs show, on every one of the six drives.

**And my own "honest fix age" change made it worse before it made it better.**
Setting `lastFixAt = null` (correct in itself — a drive with no fix must not
report a fresh one) made `quiet` evaluate to `Infinity`, so the watchdog fired
on its *first* tick: a one-second prompt instead of a four-second one.

The fix separates the two clocks. `watchdog_s` (4 s) still governs a watch that
was delivering and stopped. A new `first_fix_grace_s` (45 s) governs a watch
that has never delivered, measured from when it was armed — long enough for a
person to notice a sheet, read it and tap, and it costs nothing because a watch
that is delivering never reaches that timer. `gps_watch_rearm` now carries
`first_fix`, so "never delivered" and "stopped delivering" stop looking
identical in the log.

---

### What the driver sees now

A pre-drive checklist with four states per capability — `granted`, `denied`,
`unavailable`, `needs HTTPS` — plus one sentence saying what to do, and a live
`position: 4s ago · 11 m` line. Age *and* accuracy, because "we have a position"
and "we have a position worth answering from" are different claims, and a fix
accurate to two kilometres is what "places came back far away" looks like.

**Camera and location are blocking; the microphone is not.** A drive with no
voice is degraded and still worth having. A drive with no picture or no position
*answers wrongly* rather than not at all, which is worse.

No fallback is available to fall back to: `currentFix()` returns `null` until a
real fix arrives, and the place, news and weather tools already refuse rather
than guess when `where` is absent. That was already true and is now *visible*.

---

## 3. What the tools do when they cannot see or locate

| situation | what RIO says |
|---|---|
| no frames of this session's own | she cannot see — never another client's road |
| a ring exists but belongs to someone else | same sentence; logged distinctly as `not_my_frames` |
| no position fix | place/news/weather refuse; RIO asks which area |
| location denied | the checklist blocks the drive before it starts |

The two visual cases say the same thing to the driver and different things in
the log, deliberately: one is a camera that has not started, the other is a
session boundary that leaked, and only the second is a bug.

---

## 4. The audit: what else is global, and should any of it be per-session?

Done once, because this class of fault has now appeared twice.

| state | keyed by | verdict |
|---|---|---|
| `framebuf._rings` | visual key | **was the bug.** Fixed: tab id + provenance check at every reader |
| `observer._sessions` | session key | correct, and `serve_to` verifies provenance |
| `visual_qa._sessions` | session key | correct — inherits the visual-key fix |
| `teachers/panel._sessions` | visual key | correct — inherits the same fix |
| `safety_speech._said` | session key | correct |
| `places._last` | session key | correct |
| `weather._cache` | session key | correct, and `usable()` re-checks age **and distance** independently, so even a shared entry could not be spoken for the wrong place |
| `localnews._geo_cache` | **geographic cell** | correct to share. A reverse geocode is a pure function of position — the same answer for everyone standing there |
| `localnews._news_cache` | **cell + scope + mode + topic** | correct to share, and deliberately: news about a place is the same news for every client at that place, and sharing is what makes the second question free |
| `localnews._spend` | session key | correct — the budget is per drive, not per cell |
| `navigation/service._ROUTES` | route id, global LRU of 8 | **minor, unfixed.** Two devices cannot collide on a route id, but they share one cap: a busy client can evict another's route. Noted, not urgent |
| `telemetry.*` (`_source_name`, `_providers`, `_history`, `_runtime`) | nothing — genuinely global | **correct in principle**: there is one car, and telemetry is vehicle state, not session state. See the exception below |

### The one real finding from the audit

`POST /vehicle/telemetry/source` is client-reachable and **server-wide**. Any
connected client can switch which telemetry producer the whole pipeline listens
to, for everybody, with no session scoping at all. That is the same sentence as
the camera complaint — *a second connected client must not be able to feed or
override another's source* — applied to vehicle data instead of pictures.

It is left as-is deliberately, because unlike the camera it is **correct that
there is only one**: one car has one telemetry source, and two clients
disagreeing about which one is live would be worse than one changing it. But it
is a control surface with no authentication and no per-session scope, and it
should be a deliberate decision rather than the state of things. Recorded here
so it is one.

### The rule this audit produces

Keying is not the defence. **The reader verifies provenance**, because a key is
a claim about where something was filed and provenance is a claim about who made
it — and only the second survives somebody adding a new endpoint with a sensible
default.

---

## 6. HTTPS — for the in-car deployment, NOT for the pod

> **This did not fix the 2026-09-16 drive and was not needed for it.** That pod
> is published through RunPod's HTTPS proxy, which is already a secure origin —
> see §2a. The TLS terminator is kept because it is the right answer for a
> deployment this project is heading towards and does not have yet: **a Jetson
> or a laptop in the car, on a local network, with no proxy in front of it.**
> There, `http://192.168.1.42:8888` really is a non-secure origin and really
> does make the camera, microphone and geolocation unavailable before any
> prompt is drawn.

**Which deployment you are in decides the answer, and they are not the same:**

| where | the URL | what is needed |
|---|---|---|
| RunPod pod (today) | `https://<pod>-8888.proxy.runpod.net` | **nothing** — already a secure origin |
| in-car / Jetson / LAN (later) | `https://<address>:8443` | the certificate and terminator below |

Getting this wrong sends somebody to make a certificate they do not need, which
is what happened the first time this was diagnosed. The checklist's hint reads
the hostname and names the right one of the two:

```
  # in the car, on a local network:
  bash boot.sh cert --add 192.168.1.42     # the address the phone will type
  bash boot.sh restart                     # starts TLS if a cert exists
  # phone: https://192.168.1.42:8443/      # warns once; tap through
```

### Why a terminator rather than `uvicorn --ssl-keyfile`

uvicorn serves one protocol per process and this process loads Qwen3-VL.
Turning the existing listener into an HTTPS one means either a second 16 GB
model server or no plain-HTTP dashboard — and roughly fifteen selftests, every
`curl` and every playwright run in this repository speak plain HTTP to
`127.0.0.1:8888`. They would all have to learn about certificates to go on
testing things that have nothing to do with TLS.

So `tools/tls_proxy.py` terminates TLS on **8443** and forwards plaintext to
**8888**. The edge speaks TLS, the application does not, which is how this is
done in production anyway. Both ports stay up: the phone uses one, every local
tool keeps using the other, unchanged.

It forwards **raw bytes** rather than parsing HTTP, because the dashboard's most
important connection is a WebSocket — the frame transport — and an upgrade, its
binary frames and its close handshake are all just bytes. There is no parser
here to get any of them subtly wrong. `tools/https_selftest.py` §D proves the
upgrade survives, because a terminator that quietly dropped it would take the
camera feed with it while leaving the page looking fine.

### What makes the certificate one iOS will accept

`openssl req -x509` alone produces a certificate iOS refuses *after* the driver
has already tapped through the warning, which is the most confusing possible
moment. Apple's requirements, all set by `tools/make_cert.py` and all asserted
in the selftest:

| requirement | why it bites |
|---|---|
| **Subject Alternative Names** | Apple ignores the Common Name entirely. A CN-only cert is a cert for nothing |
| **825 days or less** | for anything issued after 2019-07-01 |
| **EKU with serverAuth** | without it the certificate is rejected outright |
| **SHA-256+, EC P-256 or RSA 2048+** | weaker is refused |

The address problem is real and cannot be solved here: the server sees a
container address (`172.x`), the phone dials the host machine's LAN address, and
this script cannot know the second. Detected addresses go in, `--add` is for the
one you actually type, and a mismatch shows up as a second warning rather than a
silent failure.

### What this does not do

It does not make the certificate **trusted**. Self-signed means a warning on
first visit; tapping through is enough for a secure context, which is all the
permission prompts need. For a phone used daily, installing `cert/rio-cert.pem`
as a trusted profile removes the warning.

It does not close port 8888. Plain HTTP is still there for the loopback and
every local tool — on a machine reachable from anywhere untrusted, that is the
port to firewall.

The private key is generated per machine in seconds, so `cert/` is gitignored
entirely: there is nothing to preserve and everything to lose. Note that `chmod
600` is ineffective on this workspace's filesystem — `tools/tls_proxy.py
--check` says so rather than pretending otherwise, and the gitignore is the
protection that actually holds.

---

## 5. Files

| file | what |
|---|---|
| `framebuf.py` | `owns()` and `ring_for()` — the one ownership rule |
| `visual_qa.py` | reads through `ring_for`, reports `not_my_frames` |
| `observer.py` | delegates to `framebuf.owns` |
| `app.py` | `_visual_key` + the `client_id` middleware |
| `static/rio_permissions.js` | the gesture-initiated permission flow |
| `static/index.html` | the checklist, the fix line, the blocking gate |
| `tools/session_isolation_selftest.py` | two sessions, two sources, no leaking |
| `tools/permissions_selftest.js` | gesture ordering, four states, no fallback |
| `tools/make_cert.py` | an iOS-shaped self-signed certificate |
| `tools/tls_proxy.py` | TLS on 8443 in front of plaintext 8888 |
| `tools/https_selftest.py` | the cert, the terminator, and `isSecureContext` |
