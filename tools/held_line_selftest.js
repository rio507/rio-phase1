/* held_line_selftest.js — a held turn call is checked again when it is released.
 *
 *   node tools/held_line_selftest.js
 *
 * On xAI one response runs at a time, so rio_xai_session.js HOLDS a dictated
 * line while another response is generating. A turn call that waited can be
 * false by the time it goes: "In half a mile" said 250 m later, or a turn the
 * car has already taken. Extending the dictation budget would only let more of
 * those through. Instead the line is validated AT RELEASE, the same way the
 * arbiter validates at dequeue:
 *
 *   still true                           -> spoken as written
 *   far call, distance moved, still      -> re-read from the route's own
 *     ahead                                 far_variants for where the car is
 *   far call, car now inside the near    -> dropped (the near call is the
 *     call                                  instruction from here)
 *   maneuver passed / not active /       -> dropped, with why
 *     route changed
 *
 * EVERYTHING REAL except the server and the GPS: the planner (rio_navplan),
 * the arbiter (rio_speech), rio_speak, the controller (rio_realtime) and the
 * xAI transport with its gate, over a recording socket and a clock the test
 * owns. The tracker is a stub so the test decides where the car is.
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

/* The route's own speech table for one maneuver, in the shape
   navigation/speech.build() writes it -- far_variants included. */
const VARIANTS = [
  [150.0, 160.0, 'In 500 feet, turn right onto Ocean Ave.'],
  [160.0, 600.0, 'In a quarter mile, turn right onto Ocean Ave.'],
  [600.0, 1005.0, 'In half a mile, turn right onto Ocean Ave.'],
  [1005.0, 1210.0, 'In three quarters of a mile, turn right onto Ocean Ave.'],
];
function route() {
  return {
    route_id: 'r1', generation_id: 1,
    timing: { speech_ttl_s: { far: 10, near: 6, junction: 2.5 } },
    maneuvers: [
      { id: 'm1', sequence: 1, type: 'TURN', direction: 'RIGHT',
        road_name: 'Ocean Ave', road_class: 'SURFACE', anchors: [],
        speech: { far: 'In half a mile, turn right onto Ocean Ave.',
                  near: 'Turn right onto Ocean Ave.', junction: 'Turn right.',
                  clips: { junction: 'turn_right' },
                  far_variants: { far: VARIANTS },
                  tiers: [{ call: 'far', at_m: 804.7 }, { call: 'near', at_m: 150 },
                          { call: 'junction', at_m: 35 }] } },
      { id: 'm2', sequence: 2, type: 'TURN', direction: 'LEFT',
        road_name: '2nd St', road_class: 'SURFACE', anchors: [],
        speech: { near: 'Turn left onto 2nd St.',
                  tiers: [{ call: 'near', at_m: 150 }] } },
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

function clipElement(played) {
  const el = {
    src: '', style: {}, muted: false, onended: null, onerror: null,
    pause() {}, addEventListener() {}, removeEventListener() {},
    play() {
      if (el.src) played.push(el.src);
      setTimeout(() => { if (el.onended) el.onended(); }, 0);
      return Promise.resolve();
    },
  };
  return el;
}

/* One car, one route, one session -- wired the way attach() wires them. */
async function rig(opts) {
  opts = opts || {};
  const WS = fakeWS();
  const ctx = fakeCtx();
  const arbiter = speech.makeArbiter();
  const prov = provider.create('xai_voice');
  const tevents = [], cevents = [], navEvents = [], silences = [];
  let controller = null;
  const s = session.open({
    mint: { ws_url: 'wss://example.test/v1/realtime?model=m',
            ws_subprotocol: 'xai-client-secret.tok',
            session: { type: 'realtime', output_modalities: ['audio'],
                       audio: { output: { voice: 'Eve' } } } },
    controller: { handle: (ev) => { if (controller) controller.handle(ev); } },
    provider: prov, ctx, tailGraceS: 0, playoutLib: playoutMod,
    WebSocketImpl: WS,
    setIntervalImpl: () => 1, clearIntervalImpl: () => {},
    onEvent: (e) => tevents.push(e),
    releaseCheck: (tag) => (controller ? controller.releaseCheck(tag) : null),
  });
  controller = rt.createController({
    arbiter, provider: prov,
    send: (o) => s.send(o),
    withdraw: (tag) => s.withdraw(tag),
    audio: { mute() {}, unmute() {} },
    onEvent: (e) => cevents.push(e),
  });
  await s.connect();
  RIO.speak.reset();
  RIO.noteSilence = (info) => silences.push(info);
  RIO.realtime = { active: () => ({
    speak: (t, o) => controller.speak(t, o),
    cancelSpeak: (tok, why) => controller.cancelSpeak(tok, why),
    speechEnabled: () => true,
    speakTimeout: () => opts.budgetMs || 300,
  }) };

  const clipsPlayed = [];
  const car = { active: 'm1', passed: {}, generation: 1 };
  const r = route();
  const tracker = {
    route: r,
    maneuver: () => r.maneuvers.find(m => m.id === car.active) || null,
    isPassed: (id) => !!car.passed[id],
    onEvent: () => {}, state: () => ({}),
  };
  const planner = navplan.create({
    tracker, arbiter, route: r,
    activeGeneration: () => car.generation,
    /* As rio_nav.audioFor builds it: the junction call plays its clip FIRST
       (dictation is its contingency), every other tier is dictated. The
       element ends the way a real one does, and the clip is counted. */
    audio: (candidate) => RIO.speak.provider({
      text: candidate.text, channel: 'nav', callType: candidate.call_type,
      revalidate: opts.noRevalidate ? null : candidate.recheck,
      clipUrl: candidate.clip ? '/static/audio/eve/' + candidate.clip + '.mp3' : null,
      clipFirst: !!candidate.clip,
      element: clipElement(clipsPlayed),
    }),
  });
  planner.onEvent((e) => navEvents.push(e));

  const creates = () => WS.sent.filter(e => e.type === 'response.create');
  return {
    s, WS, ctx, arbiter, controller, planner, car, tevents, cevents, navEvents,
    silences, creates, clipsPlayed,
    /* The server reads the most recent dictation out, start to finish. */
    hear: () => {
      const c = creates()[creates().length - 1];
      s._onMessage(JSON.stringify({ type: 'response.created',
        response: { id: 'r_dict', metadata: c.response.metadata } }));
      s._onMessage(JSON.stringify({ type: 'response.output_audio_transcript.delta',
        response_id: 'r_dict', delta: 'Turn right onto Ocean Ave.' }));
      s._onMessage(JSON.stringify({ type: 'response.done',
        response: { id: 'r_dict', status: 'completed' } }));
    },
    at: (t, m, dist, speed) => planner.onProgress({
      type: 'NAV_PROGRESS', t, maneuver_id: m, to_maneuver_m: dist,
      tta_s: dist / (speed || 25), speed_ms: speed || 25, gps_state: 'GPS_OK' }),
    /* The server is busy with something that has not claimed the mouth: a
       create for the answer to a tool result, on the wire and not yet
       created. That is the route-start shape -- and the one place a turn call
       is held by the TRANSPORT rather than waiting its turn at the arbiter
       (a patient nav line queued behind her speaking answer is checked by
       valid() at dequeue, as it always was). */
    busy: () => s.send({ type: 'response.create' }),
    // ...and the server answers it, and it ends (cancelled: no settle window).
    free: () => {
      s._onMessage(JSON.stringify({ type: 'response.created',
                                    response: { id: 'r_answer' } }));
      s._onMessage(JSON.stringify({ type: 'response.done',
                                    response: { id: 'r_answer', status: 'cancelled' } }));
      s._tick();
    },
    said: () => creates().slice(1).map(c => (c.response.instructions || '')
                                   .split('TEXT:\n').pop().trim()),
  };
}

(async function main() {

  section('still true at release -> spoken as written');
  {
    const h = await rig();
    h.busy();
    h.at(10, 'm1', 800);
    await sleep(5);
    ok(h.creates().length === 1, 'the far call is HELD — the answer to the tool result is on the wire');
    h.at(11, 'm1', 775);
    h.free();
    await sleep(5);
    ok(h.said()[0] === 'In half a mile, turn right onto Ocean Ave.',
       `released as written: still half a mile at 775 m (${h.said()[0]})`);
    const rv = h.navEvents.filter(e => e.type === 'NAV_SPEECH_REVALIDATED');
    ok(rv.length === 1 && rv[0].verdict === 'kept', 'logged: kept');
    const w = h.tevents.find(e => e.type === 'XAI_REQUEST_RELEASED');
    ok(w && w.revalidated === 'kept', 'and the create_waited row says so');
  }

  section('the distance moved, the turn is still ahead -> re-read for where the car is');
  {
    const h = await rig();
    h.busy();
    h.at(10, 'm1', 800, 30);
    await sleep(5);
    h.at(18, 'm1', 560, 30);              // eight seconds at freeway speed
    h.free();
    await sleep(5);
    ok(h.said()[0] === 'In a quarter mile, turn right onto Ocean Ave.',
       `"half a mile" at 560 m is re-read, not spoken: "${h.said()[0]}"`);
    ok(h.said()[0] && VARIANTS.some(v => v[2] === h.said()[0]),
       "...and the words are the route's own, from far_variants — nothing was "
       + 'formatted in the browser');
    const rv = h.navEvents.find(e => e.type === 'NAV_SPEECH_REVALIDATED');
    ok(rv && rv.verdict === 'regenerated' && rv.to_maneuver_m === 560,
       'logged: regenerated, at 560 m');
    ok(h.cevents.some(e => e.type === 'LIVE_DICTATION_REGENERATED'),
       'and the controller reports the change of words');
  }

  section('the car reaches the near call while the far call waits -> the near call replaces it');
  {
    const h = await rig();
    h.busy();
    h.at(10, 'm1', 800, 30);
    await sleep(5);
    h.at(19, 'm1', 140, 30);              // the near call fires at 150 m
    await sleep(5);
    ok(h.tevents.some(e => e.type === 'XAI_REQUEST_WITHDRAWN'),
       'the held far call is superseded at the arbiter and WITHDRAWN from the '
       + 'queue — it never reaches release');
    h.free();
    await sleep(5);
    ok(!h.said().some(t => /In half a mile/.test(t)),
       'no "In half a mile" with 140 m to go');
    ok(h.said()[0] === 'Turn right onto Ocean Ave.',
       `the near call is what is said (${h.said()[0]})`);
  }

  section('...and where there was no room for a near call, the far call is dropped at release');
  {
    const h = await rig();
    h.busy();
    h.at(10, 'm1', 800, 60);
    await sleep(5);
    h.at(16, 'm1', 140, 60);              // 2.3 s out: too late to begin a near call
    h.free();
    await sleep(5);
    ok(!h.said().some(t => /In half a mile/.test(t)),
       'no "In half a mile" with 140 m to go');
    const rv = h.navEvents.find(e => e.type === 'NAV_SPEECH_REVALIDATED');
    ok(rv && rv.verdict === 'dropped' && rv.why === 'inside_near_call',
       `dropped at release, and why (${rv && rv.why})`);
    ok(h.silences.some(x => /dropped_at_release/.test(x.detail || '')),
       'rio_speak stays silent — no clip, no other voice — and says so');
  }

  section('the maneuver was passed while the line waited -> dropped');
  {
    const h = await rig();
    h.busy();
    h.at(10, 'm1', 800);
    await sleep(5);
    h.car.passed.m1 = true; h.car.active = 'm2';
    h.free();
    await sleep(5);
    ok(h.creates().length === 1, 'nothing sent about a turn already taken');
    const d = h.cevents.find(e => e.type === 'LIVE_DICTATION_DROPPED');
    ok(d && d.why === 'maneuver_passed', `logged with why (${d && d.why})`);
  }

  section('the route changed while the line waited -> dropped');
  {
    const h = await rig();
    h.busy();
    h.at(10, 'm1', 800);
    await sleep(5);
    h.car.generation = 2;                 // a reroute replaced the route
    h.free();
    await sleep(5);
    ok(h.creates().length === 1, 'nothing sent about a route that is gone');
    const d = h.cevents.find(e => e.type === 'LIVE_DICTATION_DROPPED');
    ok(d && d.why === 'route_changed', `logged with why (${d && d.why})`);
    const w = h.tevents.find(e => e.type === 'XAI_REQUEST_RELEASED');
    ok(w && w.revalidated === 'dropped' && w.why === 'route_changed',
       'and the create_waited row records the drop');
  }

  section('a distance-free line moves with the car and stays true');
  {
    const h = await rig();
    h.busy();
    h.at(10, 'm1', 148);                  // the near call fires
    await sleep(5);
    h.at(12, 'm1', 100);
    h.free();
    await sleep(5);
    ok(h.said()[0] === 'Turn right onto Ocean Ave.',
       '"Turn right onto Ocean Ave." 48 m later is still the instruction');
  }

  section('the budget is not spent waiting in the queue — validity guards it instead');
  {
    const h = await rig({ budgetMs: 150 });
    h.busy();
    h.at(10, 'm1', 800);
    await sleep(400);                     // well past the 150 ms budget
    ok(!h.cevents.some(e => e.type === 'LIVE_DICTATION_FAILED'),
       'a held line with a release check does not time out while held');
    h.at(11, 'm1', 780);
    h.free();
    await sleep(5);
    ok(h.said()[0] === 'In half a mile, turn right onto Ocean Ave.',
       'and is checked and spoken at release');

    const h2 = await rig({ budgetMs: 150, noRevalidate: true });
    h2.busy();
    h2.at(10, 'm1', 800);
    await sleep(400);
    ok(h2.cevents.some(e => e.type === 'LIVE_DICTATION_FAILED'
                            && e.reason === 'timeout'),
       'a line that CANNOT be checked keeps its budget running, as before');
    h2.free();
    await sleep(5);
    ok(h2.creates().length === 1, '...and is withdrawn, never sent late');
  }

  section('a near call that goes silent above 26 mph still leaves the junction clip');
  {
    /* 15 m/s is 34 mph: 150 m to 35 m takes ~7.7 s, inside the ten-second
       junction rule. nearSpokenAt used to be set when the near call was
       QUEUED, so a near call that then went silent suppressed the junction
       call too, and the turn had no instruction at all. */
    const drive = async (h) => {
      for (let t = 11, d = 148 - 15; d > 0; t++, d -= 15) {
        h.at(t, 'm1', d, 15);
        await sleep(5);
      }
    };
    const h = await rig({ budgetMs: 150 });
    h.at(10, 'm1', 148, 15);              // the near call fires at 34 mph
    await sleep(5);
    ok(h.creates().length === 1, 'the near call is dictated');
    await sleep(300);                     // ...and the session never starts it
    ok(h.silences.some(x => x.callType === 'near' || /near/.test(x.text || '')
                            || /no_clip/.test(x.detail || '')),
       'it falls to silence: a near call names a road and has no clip');
    ok(h.navEvents.some(e => e.call_type === 'near' && e.skipped === 'not_heard'),
       'and the planner is told it was NOT heard');
    await drive(h);
    const j = h.navEvents.filter(e => e.type === 'NAV_JUNCTION_CALL');
    ok(j.some(e => e.text === 'Turn right.'),
       'THE JUNCTION CALL GOES OUT — it is now the whole instruction ('
       + (j.map(e => e.text || e.skipped).join(', ') || 'no junction event') + ')');
    ok(h.clipsPlayed.some(u => /turn_right\.mp3$/.test(u)),
       `and it is the CLIP that plays (${h.clipsPlayed.join(', ') || 'none'})`);
    ok(h.navEvents.some(e => e.type === 'NAV_SPEECH_SPOKEN' && e.call_type === 'junction'
                             && e.reason === 'spoken'),
       'spoken, at the junction');
  }

  section('...and a near call that WAS heard still suppresses it (the ten-second rule)');
  {
    const h = await rig({ budgetMs: 2000 });
    h.at(10, 'm1', 148, 15);
    await sleep(5);
    h.hear();                             // the driver hears "Turn right onto Ocean Ave."
    await sleep(5);
    ok(!h.navEvents.some(e => e.skipped === 'not_heard'), 'the near call was heard');
    for (let t = 11, d = 133; d > 0; t++, d -= 15) { h.at(t, 'm1', d, 15); await sleep(5); }
    const j = h.navEvents.filter(e => e.type === 'NAV_JUNCTION_CALL');
    ok(j.length === 1 && j[0].skipped === 'near_call_too_recent',
       `the junction call is skipped as too soon after it (${j.map(e => e.text || e.skipped).join(', ')})`);
    ok(!h.clipsPlayed.length, 'and no clip plays — one instruction, not two');
  }

  console.log(failures ? `\nFAILED ${failures}/${checks} checks`
                       : `\nPASSED ${checks} checks`);
  process.exit(failures ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
