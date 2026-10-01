/* route_start_order_selftest.js — her confirmation first, then the first turn.
 *
 *   node tools/route_start_order_selftest.js
 *
 * A route started by voice locks INSIDE the start_navigation tool call, so the
 * depart line used to reach the session before the tool result did -- and the
 * one-response gate, ranking a nav line above a conversational answer, then
 * held her confirmation behind it. The driver heard the first turn, then
 * "taking you to ..., about four minutes". Backwards.
 *
 * The rule (RIO.navplan.routeStartOrder, released by the controller at the
 * confirmation's AUDIO END):
 *   normal route start           confirmation, then the first turn
 *   first maneuver already       the turn first -- out of a driveway into a
 *     inside its near call         turn, there is no time to confirm first
 * and the order used is logged with why.
 *
 * Real: planner, arbiter, rio_speak, controller (and its start_navigation
 * tool), xAI transport with its gate. Stand-ins: the server, and RIO.nav's
 * attach -- a few lines that do what rio_nav.js attach does with the real
 * routeStartOrder.
 */
'use strict';

const path = require('path');
const STATIC = path.join(__dirname, '..', 'static');

let checks = 0, failures = 0;
function ok(cond, what) {
  checks++;
  if (!cond) failures++;
  console.log((cond ? '  ok    ' : '  FAIL  ') + what);
}
function section(n) { console.log('\n=== ' + n + ' ==='); }
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

global.window = global;
global.document = {
  addEventListener() {}, getElementById() { return null; },
  querySelector() { return null; }, querySelectorAll() { return []; },
  createElement() {
    return { pause() {}, addEventListener() {}, removeEventListener() {},
             style: {}, play() { return Promise.resolve(); } };
  },
};

const speech = require(path.join(STATIC, 'rio_speech.js'));
const rt = require(path.join(STATIC, 'rio_realtime.js'));
require(path.join(STATIC, 'rio_speak.js'));
const navplan = require(path.join(STATIC, 'rio_navplan.js'));
const provider = require(path.join(STATIC, 'rio_provider.js'));
const session = require(path.join(STATIC, 'rio_xai_session.js'));
const playoutMod = require(path.join(STATIC, 'rio_playout.js'));

const DEPART = 'Head north on Lincoln Blvd, then turn right onto Ocean Ave.';
const ONE_SECOND = 24000 * 2;

function route(firstAtM) {
  return {
    route_id: 'r1', generation_id: 1, duration_s: 240, total_distance_m: 2400,
    destination: { display_name: 'Laemmle Monica Film Center' },
    depart_speech: DEPART,
    maneuvers: [
      { id: 'm1', sequence: 1, type: 'TURN', direction: 'RIGHT',
        road_name: 'Ocean Ave', road_class: 'SURFACE', anchors: [],
        route_distance_position: firstAtM,
        speech: { near: 'Turn right onto Ocean Ave.', junction: 'Turn right.',
                  tiers: [{ call: 'near', at_m: 150 }, { call: 'junction', at_m: 35 }] } },
      { id: 'm2', sequence: 2, type: 'ARRIVE', direction: 'RIGHT',
        road_name: '', road_class: 'SURFACE', anchors: [],
        route_distance_position: 2400,
        speech: { arrival: 'Your destination is on the right.',
                  tiers: [{ call: 'arrival', at_m: 150 }] } },
    ],
  };
}

function fakeWS() {
  const sent = [];
  function WS() { WS.last = this; setTimeout(() => { if (this.onopen) this.onopen(); }, 0); }
  WS.prototype.send = function (s) { sent.push(JSON.parse(s)); };
  WS.prototype.close = function () {};
  WS.sent = sent;
  return WS;
}
function fakeCtx() {
  let t = 100;
  return {
    get currentTime() { return t; }, _advance: (s) => { t += s; },
    destination: {},
    createBuffer: (ch, len) => ({ duration: len / 24000, length: len,
      copyToChannel: () => {}, getChannelData: () => new Float32Array(len) }),
    createBufferSource: () => ({ buffer: null, connect: () => {}, start: () => {},
                                 stop: () => {}, onended: null }),
  };
}

async function rig(firstAtM) {
  const WS = fakeWS();
  const ctx = fakeCtx();
  const arbiter = speech.makeArbiter();
  const prov = provider.create('xai_voice');
  const tevents = [], cevents = [], navEvents = [];
  let controller = null;
  const s = session.open({
    mint: { ws_url: 'wss://example.test/v1/realtime?model=m',
            ws_subprotocol: 'xai-client-secret.tok',
            session: { type: 'realtime', output_modalities: ['audio'],
                       audio: { output: { voice: 'Eve' } } } },
    controller: { handle: (ev) => { if (controller) controller.handle(ev); } },
    provider: prov, ctx, tailGraceS: 0, playoutLib: playoutMod,
    WebSocketImpl: WS, setIntervalImpl: () => 1, clearIntervalImpl: () => {},
    onEvent: (e) => tevents.push(e),
    releaseCheck: (tag) => (controller ? controller.releaseCheck(tag) : null),
  });

  /* RIO.nav, as rio_nav.js attach() behaves: routeStartOrder decides, the
     planner is created held or not, the order is logged, releaseStart lets a
     held one speak. */
  let planner = null, hold = null;
  const nav = {
    unlock() {},
    routeToQuery(text, ropts) {
      const r = route(firstAtM);
      const order = ropts && ropts.holdStart && navplan.routeStartOrder ? navplan.routeStartOrder(r) : null;
      const held = !!(order && order.hold);
      planner = navplan.create({
        tracker: { route: r, maneuver: () => r.maneuvers[0], isPassed: () => false,
                   onEvent() {}, state: () => ({}) },
        arbiter, route: r, holdStart: held,
        audio: (c) => RIO.speak.provider({ text: c.text, channel: 'nav',
          callType: c.call_type, revalidate: c.recheck,
          element: document.createElement('audio') }),
      });
      planner.onEvent((e) => navEvents.push(e));
      if (order && !held) navEvents.push(Object.assign({ type: 'NAV_ROUTE_START_ORDER' }, order));
      if (held) hold = order;
      planner.onRouteStart();
      return Promise.resolve({ status: 'routed', destination: r.destination,
                               route: r, start_held: held });
    },
    releaseStart(why) {
      if (!hold) return false;
      navEvents.push(Object.assign({ type: 'NAV_ROUTE_START_ORDER', released_by: why }, hold));
      hold = null;
      return planner.releaseStart();
    },
  };
  global.RIO.nav = nav;

  controller = rt.createController({
    arbiter, provider: prov, nav,
    send: (o) => s.send(o),
    withdraw: (tag) => s.withdraw(tag),
    tool: (name, args) => Promise.resolve(rt.localTools[name](args)),
    audio: { mute() {}, unmute() {} },
    onEvent: (e) => cevents.push(e),
  });
  await s.connect();
  RIO.speak.reset();
  RIO.realtime = { active: () => ({
    speak: (t, o) => controller.speak(t, o),
    cancelSpeak: (tok, why) => controller.cancelSpeak(tok, why),
    speechEnabled: () => true, speakTimeout: () => 3000,
  }) };

  const creates = () => WS.sent.filter(e => e.type === 'response.create');
  const msg = (o) => s._onMessage(JSON.stringify(o));
  return {
    s, ctx, creates, cevents, navEvents, msg,
    // The server answers create `c` with `sec` seconds of audio, all generated.
    answer(c, id, words, sec) {
      msg({ type: 'response.created', response: { id, metadata: c.response.metadata } });
      msg({ type: 'response.output_audio_transcript.delta', response_id: id, delta: words });
      msg({ type: 'response.output_audio.delta', response_id: id,
            delta: Buffer.alloc(Math.round(ONE_SECOND * sec)).toString('base64') });
      msg({ type: 'response.done', response: { id, status: 'completed' } });
    },
    // Time passes for the sound: the playout drains and the gate ticks.
    play(sec) { ctx._advance(sec); s._tick(); },
    startNav() {
      controller.handle({ type: 'response.function_call_arguments.done',
        name: 'start_navigation', call_id: 'nav1',
        arguments: JSON.stringify({ destination: 'Laemmle Monica Film Center' }) });
    },
    isDepart: (c) => /Head north on Lincoln/.test((c.response && c.response.instructions) || ''),
  };
}

(async function main() {

  section('a normal route start: her confirmation, THEN the first turn');
  {
    const h = await rig(600);                 // first turn 600 m away
    h.startNav();
    await sleep(20);
    const c = h.creates();
    ok(c.length === 1 && !h.isDepart(c[0]),
       'the confirmation is asked for first — the depart line is held by the '
       + 'planner, not raced into the session');
    h.answer(c[0], 'r_confirm', 'Taking you to the Laemmle Monica Film Center, about four minutes.', 3);
    await sleep(5);
    h.play(1.0);
    await sleep(5);
    ok(h.creates().length === 1,
       'still only the confirmation while it is PLAYING — response.done is not '
       + 'the end of the sound');
    h.play(2.2);
    await sleep(20);
    h.play(2.1);                              // settle window after a completed response
    await sleep(20);
    const c2 = h.creates();
    ok(c2.length === 2 && h.isDepart(c2[1]),
       'and the first turn call goes out once her confirmation has finished playing');
    const order = h.navEvents.find(e => e.type === 'NAV_ROUTE_START_ORDER');
    ok(order && order.order === 'confirmation_first'
       && order.released_by === 'confirmation_done',
       `logged: ${order && order.order}, released by ${order && order.released_by}`);
  }

  section('an immediate first turn: the TURN first');
  {
    const h = await rig(90);                  // out of a driveway: 90 m to the turn
    h.startNav();
    await sleep(20);
    const c = h.creates();
    ok(c.length >= 1 && h.isDepart(c[0]),
       'the depart line goes first — the first turn is inside its 150 m near call');
    ok(c.length === 1, 'and the confirmation waits behind it (one response at a time)');
    const order = h.navEvents.find(e => e.type === 'NAV_ROUTE_START_ORDER');
    ok(order && order.order === 'turn_first'
       && order.why === 'first_maneuver_inside_near_call' && order.first_maneuver_m === 90,
       `logged: ${order && order.order}, ${order && order.why}, `
       + `${order && order.first_maneuver_m} m`);
    h.answer(c[0], 'r_depart', DEPART, 3);
    await sleep(5);
    h.play(3.2);
    await sleep(20);
    h.play(2.1);
    await sleep(20);
    const c2 = h.creates();
    ok(c2.length === 2 && !h.isDepart(c2[1]), 'then her confirmation');
  }

  section('a confirmation that never plays does not hold the turn for ever');
  {
    const h = await rig(600);
    h.startNav();
    await sleep(20);
    const c = h.creates();
    // The server answers with nothing audible: cancelled at once.
    h.msg({ type: 'response.created', response: { id: 'r_x', metadata: c[0].response.metadata } });
    h.msg({ type: 'response.done', response: { id: 'r_x', status: 'cancelled' } });
    await sleep(20);
    h.play(0.1);
    await sleep(20);
    ok(h.creates().some(h.isDepart),
       'a confirmation that ended without a sound releases the first turn at once');
  }

  console.log(failures ? `\nFAILED ${failures}/${checks} checks`
                       : `\nPASSED ${checks} checks`);
  process.exit(failures ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
