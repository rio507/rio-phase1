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
