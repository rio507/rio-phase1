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

1. **Not a secure context.** Over plain `http` on a LAN address, `getUserMedia`
   and geolocation are refused before any prompt is drawn. The page knew this
   for media (`RIO.mediaError`) and had no equivalent for position. **This is
   the most likely root cause of the live failure**: a phone opening
   `http://<ip>:8888` gets no camera and no location, silently.
2. **The gesture was spent.** iOS grants a prompt only from a trusted gesture,
   and a gesture does not survive an `await`. All three requests are now
   *started* synchronously in the Start Drive handler and awaited afterwards —
   the order they are **made** in is everything; the order they resolve in does
   not matter.
3. **A refusal looked like an absence.** `startWatch()` set `lastFixAt` to *now*
   before any fix had arrived, so a drive that never received a position
   reported a **fresh** position age for its whole length. It is `null` until a
   fix actually arrives, and every reader treats that as "no fix".

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

## 6. HTTPS, which is what actually unblocks the phone

`localhost` is a secure context by special dispensation. **A phone is never on
localhost.** Over `http://192.168.1.42:8888` a browser refuses the camera, the
microphone and geolocation *before drawing any prompt* — not as a permission the
driver denied, but as a capability never offered. That is the most likely root
cause of the live failure, and no amount of work on the permission flow fixes
it, because the flow never gets to run.

```
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
