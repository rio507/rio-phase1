/* frame_transport_selftest.js — the frame feed, and every way it went quiet.
 *
 *   node tools/frame_transport_selftest.js
 *
 * WHAT THIS IS FOR. On 2026-09-09 (session 738fbb82) the frame push stopped
 * 40.4 s into a drive and did not resume for 442 SECONDS, on a page that was
 * demonstrably alive throughout — the conversation kept working, the GPS watch
 * kept re-arming, nav kept calling turns. The log's account of it is almost
 * entirely negative space:
 *
 *     last frame on the socket   idx 232, t=40.4
 *     next result of any kind    idx 233, t=482.7
 *     headway_ws_close           t=400.4  (the socket was OPEN for six minutes
 *                                          after the last frame)
 *     FRAMES_WS_LOST             never emitted
 *     FRAMES_STALE               never emitted
 *
 * So: not a network drop, not a server close. The capture loop stopped, with
 * the socket open, and nothing anywhere was watching for that — because
 * `schedule()` was the last statement of `tick()`, after `await once()`, and
 * an awaited promise that never settles means the next tick is never
 * scheduled. Not late: never.
 *
 * tools/frame_transport_bench.py measures how FAST this transport is against a
 * real server. This measures whether it is ALIVE, against no server at all:
 * every failure below is driven by a fake socket, a fake camera and a fake
 * clock, so the whole file runs in about a second and none of it needs a road.
 *
 * The shipped constants are seconds. Every one of them is overridable from
 * create(), and this drives them at milliseconds — the SHAPE of each decision
 * is what is under test, not the number, which is the same argument the tuning
 * table has always made.
 */
'use strict';

const path = require('path');
const frames = require(path.join(__dirname, '..', 'static', 'rio_frames.js'));

let checks = 0, failures = 0;
function ok(cond, what, detail) {
  checks++;
  if (!cond) { failures++; console.log('  FAIL  ' + what + (detail ? '  -- ' + detail : '')); }
  else console.log('  ok    ' + what + (detail ? '  -- ' + detail : ''));
}
function section(name) { console.log('\n=== ' + name + ' ==='); }
const wait = ms => new Promise(r => setTimeout(r, ms));

// ---------------------------------------------------------------------------
// The world the transport runs in: a socket, a camera and a canvas, all fake.
// ---------------------------------------------------------------------------
const sockets = [];

class FakeSocket {
  constructor(url) {
    this.url = url;
    this.readyState = 0;
    this.bufferedAmount = 0;
    this.sent = [];
    this.pings = 0;
    this.answerPings = true;
    this.closed = false;
    sockets.push(this);
    setTimeout(() => this._open(), 0);
  }
  _open() {
    if (this.closed) return;
    this.readyState = 1;
    if (this.onopen) this.onopen();
    // The server's `ready`, which is what puts the transport into ws mode.
    this._deliver({ op: 'ready', tuning: {} });
  }
  _deliver(obj) {
    if (this.onmessage) this.onmessage({ data: JSON.stringify(obj) });
  }
  send(payload) {
    if (typeof payload === 'string') {
      const msg = JSON.parse(payload);
      if (msg.op === 'ping') {
        this.pings++;
        if (this.answerPings) this._deliver({ op: 'pong', c: msg.c, s: msg.c });
      }
      return;
    }
    this.sent.push(payload);
    if (this.answerFrames !== false) {
      // A result, the way the server answers one: enough of the shape for
      // handleResult to be satisfied.
      setTimeout(() => this._deliver({ seq: this.sent.length, frame_age_ms: 120 }), 0);
    }
  }
  close() {
    if (this.closed) return;
    this.closed = true;
    this.readyState = 3;
    if (this.onclose) this.onclose();
  }
}

/* A camera element that produces pixels, until it does not. */
function fakeElement() {
  const tracks = [];
  const el = {
    videoWidth: 640, videoHeight: 480,
    srcObject: {
      getVideoTracks: () => tracks,
      getTracks: () => tracks,
    },
    _endTrack() { tracks.forEach(t => t._fire('ended')); },
  };
  const track = {
    label: 'fake camera', __rioWatched: false,
    _handlers: {},
    addEventListener(name, fn) { (this._handlers[name] = this._handlers[name] || []).push(fn); },
    _fire(name) { (this._handlers[name] || []).forEach(fn => fn()); },
  };
  tracks.push(track);
  return el;
}

/* The canvas. `hang` makes toBlob never call back, which is exactly what iOS
   does while the page is in the background — and is the bug. */
const canvasState = { hang: false, blobs: 0 };
function installDom() {
  global.document = {
    _listeners: {},
    visibilityState: 'visible',
    addEventListener(name, fn) { (this._listeners[name] = this._listeners[name] || []).push(fn); },
    _fire(name) { (this._listeners[name] || []).forEach(fn => fn()); },
    createElement() {
      return {
        width: 0, height: 0,
        getContext: () => ({ drawImage: () => {} }),
        toBlob(cb) {
          if (canvasState.hang) return;          // never calls back. Ever.
          canvasState.blobs++;
          cb({
            size: 20000,
            arrayBuffer: () => Promise.resolve(new ArrayBuffer(20000)),
          });
        },
      };
    },
  };
  global.WebSocket = FakeSocket;
  if (typeof global.TextEncoder === 'undefined') {
    global.TextEncoder = require('util').TextEncoder;
  }
}
installDom();

/* One transport, wired to the fakes, with every watchdog on a millisecond
   scale. */
function makeStream(opts) {
  opts = opts || {};
  const events = [];
  const states = [];
  const el = opts.element || fakeElement();
  const s = frames.create({
    wsUrl: 'ws://fake/headway_ws',
    postUrl: 'http://fake/headway_frame',
    element: () => (opts.elementGone ? null : el),
    speed: () => ({ v: 10, age: 0 }),
    source: () => 'camera',
    onResult: () => {},
    onEvent: (name, detail) => events.push({ name, detail }),
    onState: (state, detail) => states.push({ state, detail }),
    onTrackEnded: opts.onTrackEnded,
    tuning: { start_fps: 40, min_fps: 40, max_fps: 40 },
    stallMs: opts.stallMs || 120,
    tickBudgetMs: opts.tickBudgetMs || 60,
    pingIntervalMs: opts.pingIntervalMs || 20,
    pongMisses: opts.pongMisses || 3,
    healthyMs: opts.healthyMs === undefined ? 60 : opts.healthyMs,
    superviseMs: opts.superviseMs || 20,
    maxReconnects: opts.maxReconnects,
    reconnectBaseMs: opts.reconnectBaseMs || 20,
    openDelayMs: opts.openDelayMs === undefined ? 10 : opts.openDelayMs,
  });
  return { s, events, states, el,
           names: () => events.map(e => e.name),
           has: n => events.some(e => e.name === n) };
}

async function main() {

// ---------------------------------------------------------------------------
section('the loop cannot die — the bug that cost 442 seconds');
// ---------------------------------------------------------------------------
{
  const h = makeStream();
  h.s.start();
  await wait(150);
  const before = h.s.stats().sent;
  ok(before > 0, 'frames flow to begin with', before + ' sent');

  /* THE FAILURE, EXACTLY. canvas.toBlob stops calling back — no error, no
     rejection, just a promise that never settles. Before the fix this was the
     end of the drive. */
  canvasState.hang = true;
  await wait(200);
  const stuck = h.s.stats().sent;
  ok(stuck === before, 'nothing is sent while the encode is hung',
     before + ' -> ' + stuck);
  ok(h.has('FRAMES_TICK_OVERRUN'),
     'and the tick says so rather than disappearing');

  // ...and the camera comes back.
  canvasState.hang = false;
  await wait(200);
  ok(h.s.stats().sent > stuck,
     'the loop is still alive and resumes on its own',
     stuck + ' -> ' + h.s.stats().sent);
  h.s.stop();
}

// ---------------------------------------------------------------------------
section('the watchdog — a stall is noticed and acted on');
// ---------------------------------------------------------------------------
{
  const h = makeStream();
  h.s.start();
  await wait(100);
  canvasState.hang = true;
  await wait(400);
  ok(h.has('FRAMES_STALLED'),
     'nothing sent for stall_ms is reported as a stall');
  const stall = h.events.filter(e => e.name === 'FRAMES_STALLED')[0];
  ok(stall && stall.detail.why === 'no_frames_sent',
     '...with what it was, not just that it happened',
     stall ? stall.detail.why : 'no event');
  ok(h.states.some(x => x.state === 'reconnecting'),
     'and the driver is told the feed is interrupted');
  ok(h.s.stats().stall_recoveries > 0,
     'the recovery is counted for the drive log',
     String(h.s.stats().stall_recoveries));
  canvasState.hang = false;
  await wait(200);
  ok(h.states[h.states.length - 1].state === 'live',
     'and the banner clears itself when frames come back',
     h.states[h.states.length - 1].state);
  h.s.stop();
}

// ---------------------------------------------------------------------------
section('the socket that is open and dead');
// ---------------------------------------------------------------------------
{
  const before = sockets.length;
  // stallMs high: the stall watchdog would close this socket for its own
  // (correct) reasons long before three pings go unanswered, and what is under
  // test here is the ping watchdog on its own.
  const h = makeStream({ stallMs: 100000 });
  h.s.start();
  await wait(80);
  const sock = sockets[before];
  ok(sock && sock.pings > 0, 'pings are going out', sock ? String(sock.pings) : '-');

  /* A carrier NAT drops the flow. readyState stays OPEN forever; sends
     "succeed"; nothing ever comes back. The pings were always going out and
     nothing read the pongs. */
  sock.answerPings = false;
  await wait(400);
  ok(h.has('FRAMES_WS_UNANSWERED'),
     'three unanswered pings is a dead socket, however open it looks');
  ok(sock.closed, '...and it is closed rather than written into');
  ok(sockets.length > before + 1, 'and a new one is opened',
     (sockets.length - before) + ' sockets');
  h.s.stop();
}

// ---------------------------------------------------------------------------
section('the reconnect budget, which used to be a lifetime');
// ---------------------------------------------------------------------------
{
  // A socket that carried frames for healthyMs and then dropped did not fail
  // because the network refuses upgrades. It failed because a phone moved
  // between cells, and the budget starts again.
  const before = sockets.length;
  const h = makeStream({ healthyMs: 40, maxReconnects: 1, stallMs: 100000 });
  const dropOne = async () => {
    const live = sockets.slice(before).filter(x => !x.closed);
    if (live.length) live[live.length - 1].close();
    await wait(200);
  };
  h.s.start();
  await wait(120);                       // first socket, past healthyMs
  await dropOne();                       // reconnects -> 1, budget of 1 spent
  ok(!h.has('FRAMES_WS_GIVEN_UP'), 'the first drop is inside the budget');
  await wait(120);                       // the replacement proves itself too
  await dropOne();
  /* THE WHOLE POINT. With a lifetime counter this second drop is the end of
     the socket for the rest of the drive — MAX_RECONNECTS was 3 and nothing
     ever reset it, so three cell handovers over a long drive finished the
     journey on the POST path at about 1 fps. */
  ok(h.has('FRAMES_WS_HEALTHY_RESET'),
     'a socket that proved itself gives the budget back');
  ok(!h.has('FRAMES_WS_GIVEN_UP'),
     '...so a second drop mid-drive does not retire the transport for good');
  h.s.stop();
}

{
  // ...and a socket that never works still gives up, which is the behaviour
  // the budget existed for: reconnecting forever against a proxy that strips
  // upgrades is a battery leak with no upside.
  const h = makeStream({ healthyMs: 100000, maxReconnects: 2, stallMs: 100000 });
  h.s.start();
  await wait(40);
  for (let i = 0; i < 5; i++) {
    const live = sockets.filter(x => !x.closed);
    if (live.length) live[live.length - 1].close();
    await wait(200);
  }
  ok(h.has('FRAMES_WS_GIVEN_UP'),
     'a socket that never carries anything is given up on');
  h.s.stop();
}

// ---------------------------------------------------------------------------
section('coming back from the background');
// ---------------------------------------------------------------------------
{
  const h = makeStream();
  h.s.start();
  await wait(80);
  global.document.visibilityState = 'hidden';
  global.document._fire('visibilitychange');
  await wait(20);
  ok(h.has('FRAMES_HIDDEN'), 'the page going away is recorded');

  global.document.visibilityState = 'visible';
  global.document._fire('visibilitychange');
  await wait(40);
  ok(h.has('FRAMES_VISIBLE'), 'and coming back is too');
  /* TREATED AS A STALL WHETHER OR NOT THE CLOCK SAYS SO: iOS throttled the
     timers, may have ended the camera track, and toBlob was not calling back
     for the whole time the page was hidden. */
  const why = h.events.filter(e => e.name === 'FRAMES_STALLED')
                      .map(e => e.detail.why);
  ok(why.indexOf('became_visible') >= 0,
     'and the transport is rebuilt on the way back rather than trusted',
     JSON.stringify(why));
  h.s.stop();
}

// ---------------------------------------------------------------------------
section('the camera track ending under the page');
// ---------------------------------------------------------------------------
{
  let told = 0;
  const el = fakeElement();
  const h = makeStream({ element: el, onTrackEnded: () => { told++; } });
  h.s.start();
  await wait(80);
  el._endTrack();
  await wait(40);
  ok(h.has('FRAMES_TRACK_ENDED'),
     'a track that ends is reported rather than becoming silent skipped captures');
  ok(told === 1,
     'and the page is asked to re-acquire the camera, which is the only thing '
     + 'that can fix it', String(told));
  ok(h.states.some(x => x.state === 'lost'),
     'the driver is told the feed is gone, not merely interrupted');
  h.s.stop();
}

// ---------------------------------------------------------------------------
section('what a drive review can ask afterwards');
// ---------------------------------------------------------------------------
{
  const h = makeStream();
  h.s.start();
  await wait(120);
  const st = h.s.stats();
  for (const key of ['state', 'since_sent_ms', 'since_result_ms',
                     'stall_recoveries', 'tick_overruns', 'reconnects']) {
    ok(st[key] !== undefined, 'stats carry ' + key);
  }
  ok(st.state === 'live', 'and say the feed is live when it is', st.state);
  h.s.stop();
  ok(h.s.stats().state === 'idle', 'and idle when it is not');
}

console.log('\n' + (failures ? 'FAILED ' + failures + '/' : 'PASSED ') + checks + ' checks');
process.exit(failures ? 1 : 0);
}

main().catch(e => { console.error(e); process.exit(2); });
