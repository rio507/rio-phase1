/* nav_selftest.js — the navigation foundation, run without a browser or a car.
 *
 *   node tools/nav_selftest.js [route.json]
 *
 * Everything about navigation is assertable from simulated GPS and simulated
 * landmark observations, and none of the required tests needs a network, a
 * camera or a road. What is checked here:
 *
 *   1. Ordinary turn-by-turn — the route loads, the maneuver is selected,
 *      progress advances, the three speech opportunities fire in order, once
 *      each, and the maneuver is passed. This must hold with vision switched
 *      off entirely: that is the first milestone.
 *   2. Contextual navigation — an expected Shell near the turn, verified,
 *      produces "Turn left by the Shell station." And every way that can fail
 *      produces the canonical instruction instead, with no other difference.
 *   3. The things that are only wrong at the last moment — a "Right here."
 *      that was queued before the junction and dequeued after it; a line from
 *      a route generation a reroute has replaced. Both must be dropped
 *      silently at DEQUEUE, which is the only moment the answer is knowable.
 *   4. GPS health is not off-route, and a degraded fix speaks earlier, never
 *      later.
 *
 * static/rio_navcore.js, rio_navplan.js and rio_speech.js are DOM-free
 * precisely so this file can require() them: the code under test is the code
 * that ships. The server side — relations from map data, the anchor gates,
 * candidate generation, the speech table — is tools/nav_server_selftest.py.
 */
'use strict';

const path = require('path');
const fs = require('fs');

const navcore = require(path.join(__dirname, '..', 'static', 'rio_navcore.js'));
const navplan = require(path.join(__dirname, '..', 'static', 'rio_navplan.js'));
const speech = require(path.join(__dirname, '..', 'static', 'rio_speech.js'));

let checks = 0, failures = 0;
function ok(cond, what) {
  checks++;
  if (!cond) { failures++; console.log('  FAIL  ' + what); }
  else console.log('  ok    ' + what);
}
function section(name) { console.log('\n=== ' + name + ' ==='); }

// ---------------------------------------------------------------------------
// A synthetic route in the wire shape the server produces: 1200 m east, left
// onto Lincoln, 900 m north, right onto Fell, 500 m east, arrive. Vertices
// every 10 m so projection has real geometry to chew on.
// ---------------------------------------------------------------------------
const TIMING = {
  gps_stale_timeout_s: 5, gps_accuracy_limit_m: 30, gps_degraded_bias_s: 2,
  off_route_distance_m: 45, off_route_persistence: 3, reroute_debounce_s: 12,
  progress_rewind_tolerance_m: 30, maneuver_passed_eps_m: 8, arrive_radius_m: 25,
  projection_back_m: 80, projection_fwd_m: 400,
  heading_min_displacement_m: 8, heading_max_sample_age_s: 3,
  heading_min_speed_ms: 1.5, stationary_speed_ms: 0.7,
  anchor_acquisition_lead_m: 150,
  min_call_distance_m: 20, far_max_distance_m: 4000,
  // Google's ladder, as config.py sends it. Per-maneuver tiers ride on
  // `speech.tiers`; this is the table they came from.
  tier_distances_m: {
    SURFACE: { far: 804.7, near: 150.0, junction: 35.0 },
    HIGHWAY: { far: 3218.7, far_mid: 1609.3, near: 402.3, junction: 150.0 },
  },
  junction_min_gap_s: 10.0, near_min_lead_s: 4.0,
  chain_window_m: 200.0, arrival_call_m: 150.0,
  units: 'imperial',
  progress_tick_s: 0.5, clip_start_latency_s: 0.05, junction_lead_margin_m: 2.0,
  speed_floor_ms: 3, speed_nominal_ms: 11,
  duplicate_instruction_cooldown_s: 8, anchor_valid_for_m: 400,
  speech_ttl_s: { depart: 12, far: 10, far_mid: 10, near: 6, junction: 2.5,
                  arrival: 8, arrived: 8 },
  vision_enabled: true,
};

// The surface ladder, as the server writes it onto every surface maneuver.
const SURFACE_TIERS = [{ call: 'far', at_m: 804.7 },
                       { call: 'near', at_m: 150.0 },
                       { call: 'junction', at_m: 35.0 }];

function synthRoute(opts) {
  opts = opts || {};
  const lat0 = 34.0430, lng0 = -118.2673;
  const mLat = 111320, mLng = 111320 * Math.cos(lat0 * Math.PI / 180);
  const pts = [];
  const push = (x, y) => pts.push([lat0 + y / mLat, lng0 + x / mLng]);
  let x = 0, y = 0;
  push(0, 0);
  for (let d = 10; d <= 1200; d += 10) push(d, 0);            x = 1200;
  for (let d = 10; d <= 900; d += 10) push(x, d);             y = 900;
  for (let d = 10; d <= 500; d += 10) push(x + d, y);

  const iTurn1 = 120, iTurn2 = 210, iEnd = pts.length - 1;
  const anchors = opts.anchors || [];
  return {
    route_id: opts.route_id || 'synth', journey_id: 'j1',
    generation_id: opts.generation_id || 1, provider: 'fixture',
    created_at: 0, eta_epoch: 0,
    origin: { lat: pts[0][0], lng: pts[0][1] },
    destination: { display_name: 'Test Destination',
                   formatted_address: 'Test Destination, Los Angeles, CA',
                   lat: pts[iEnd][0], lng: pts[iEnd][1] },
    total_distance_m: 2600, duration_s: 236, route_length_m: 2600,
    arrival: { side: 'RIGHT' }, landmarks_state: anchors.length ? 'ready' : 'not_requested',
    depart_speech: 'Head east on Venice Boulevard, then turn left onto Lincoln Boulevard.',
    geometry: pts,
    timing: Object.assign({}, TIMING, opts.timing || {}),
    maneuvers: [
      { id: 'm0', sequence: 0, type: 'TURN', direction: 'LEFT',
        road_name: 'Lincoln Boulevard', instruction: 'Turn left onto Lincoln Boulevard',
        lat: pts[iTurn1][0], lng: pts[iTurn1][1],
        route_distance_position: 1200, polyline_index: iTurn1,
        anchors: anchors, road_class: 'SURFACE',
        speech: { far: 'In half a mile, turn left onto Lincoln Boulevard.',
                  near: 'Turn left onto Lincoln Boulevard.',
                  junction: 'Turn left.',
                  clips: { junction: 'turn_left' },
                  tiers: SURFACE_TIERS } },
      { id: 'm1', sequence: 1, type: 'TURN', direction: 'RIGHT',
        road_name: 'Fell Street', instruction: 'Turn right onto Fell Street',
        lat: pts[iTurn2][0], lng: pts[iTurn2][1],
        route_distance_position: 2100, polyline_index: iTurn2,
        anchors: [], road_class: 'SURFACE',
        speech: { far: 'In half a mile, turn right onto Fell Street.',
                  near: 'Turn right onto Fell Street.',
                  junction: 'Turn right.',
                  clips: { junction: 'turn_right' },
                  tiers: SURFACE_TIERS } },
      { id: 'm2', sequence: 2, type: 'ARRIVE', direction: 'RIGHT',
        road_name: '', instruction: 'Arrive at Test Destination',
        lat: pts[iEnd][0], lng: pts[iEnd][1],
        route_distance_position: 2600, polyline_index: iEnd,
        anchors: [], road_class: 'SURFACE',
        speech: { arrival: 'Your destination is on the right.',
                  arrived: 'You have arrived.',
                  tiers: [{ call: 'arrival', at_m: 150.0 }] } },
    ],
  };
}

const SHELL_ANCHOR = {
  anchor_id: 'm0a0', place_id: 'p_shell', label: 'Shell', spoken_label: 'the Shell station',
  type: 'gas_station', salience: 1.0, relation: 'NEAR', relation_confidence: 0.72,
  distance_to_maneuver_m: 18.2, along_delta_m: -13.6, lateral_m: 12, side: 'RIGHT',
  speech: 'Turn left by the Shell station.',
};

/* One simulated drive. Speech goes through the real arbiter; play() returns
   nothing, so the arbiter finishes it synchronously and the transcript is in
   the order a driver would have heard it. */
function drive(route, opts) {
  opts = opts || {};
  const arbiter = speech.makeArbiter();
  const tracker = navcore.create(route, opts.tracker);
  const spoken = [];
  const events = [];
  const planner = navplan.create({
    tracker: tracker, arbiter: arbiter, route: route,
    options: opts.planner,
    verify: opts.verify || null,
    audio: function (candidate) {
      return { play: function () { spoken.push(candidate.text); }, stop: function () {} };
    },
  });
  tracker.onEvent(e => { if (e.type !== 'NAV_PROGRESS') events.push(e); });
  planner.onEvent(e => events.push(e));
  // THE ROUTE-START LINE, before the first fix. rio_nav.js calls this the
  // moment a route attaches, and a harness that skips it is testing a drive
  // that begins by announcing a turn.
  if (opts.routeStart !== false) planner.onRouteStart();

  const speedMs = opts.speedMs === undefined ? 15 : opts.speedMs;
  const total = tracker.routeLength();
  let s = opts.startM || 0, t = 0;
  while (s <= total + 60 && t < 2000) {
    const p = tracker.pointAt(Math.min(s, total));
    let fix = { lat: p.lat, lng: p.lng, speed: speedMs, t: t, accuracy: 8 };
    if (opts.perturb) fix = opts.perturb(fix, s, t);
    if (fix) tracker.position(fix);
    const st = tracker.state();
    if (st.arrived || st.stopped) break;
    if (opts.after) opts.after(t, tracker, planner, arbiter, spoken);
    s += speedMs; t += 1;
  }
  return { spoken, events, tracker, planner, arbiter };
}

const spokenFor = (r, id) => r.events.filter(e => e.maneuver_id === id &&
  e.text &&
  (e.type === 'NAV_FAR_GUIDANCE' || e.type === 'NAV_CONTEXTUAL_CALL' ||
   e.type === 'NAV_JUNCTION_CALL' || e.type === 'NAV_ROUTE_START_CALL' ||
   e.type === 'NAV_ARRIVAL_CALL'));

// ---------------------------------------------------------------------------
section('normal navigation — no vision at all');
// ---------------------------------------------------------------------------
{
  const r = drive(synthRoute(), { verify: null });
  const selected = r.events.filter(e => e.type === 'NAV_MANEUVER_SELECTED');
  const passed = r.events.filter(e => e.type === 'NAV_MANEUVER_PASSED');
  const arrived = r.events.filter(e => e.type === 'NAV_ARRIVED');

  ok(selected.map(e => e.maneuver_id).join(',') === 'm0,m1,m2',
     'each maneuver is selected in turn — ' + selected.map(e => e.maneuver_id).join(','));
  ok(passed.map(e => e.maneuver_id).join(',') === 'm0,m1',
     'both turns are passed, in order, and arrival is not one of them');
  ok(arrived.length === 1 && r.events[r.events.length - 1].type === 'NAV_ARRIVED',
     'arrival fires exactly once, last');

  /* GOOGLE'S CADENCE, WORD FOR WORD, and this is the assertion the whole of
     the timing rewrite exists to make pass. Route start first — the one tier
     the old design had no slot for at all — then half a mile, then the street
     name, per turn, then the two arrival lines.

     No junction call here, and that is the >10 s rule working: at 15 m/s the
     near call lands 10 s from the turn and the junction floor 3 s from it, so
     "Turn left." would be RIO repeating herself. The slow drive below gets
     it. */
  if (process.env.NAV_DEBUG) console.log('    SPOKEN:', JSON.stringify(r.spoken, null, 1));
  ok(r.spoken.join(' | ') ===
     'Head east on Venice Boulevard, then turn left onto Lincoln Boulevard. | ' +
     'In half a mile, turn left onto Lincoln Boulevard. | ' +
     'Turn left onto Lincoln Boulevard. | ' +
     'In half a mile, turn right onto Fell Street. | ' +
     'Turn right onto Fell Street. | ' +
     'Your destination is on the right. | You have arrived.',
     'the whole drive is Google\'s cadence: the first move, then far and near '
     + 'per turn, then the two arrival lines');
  ok(r.spoken.filter(t => /\bmile|\bfeet|\bmeters/.test(t)).length === 2,
     'exactly the two far calls carry a distance, and they carry a phrase '
     + 'rather than a countdown');

  const dupes = r.spoken.filter((t, i) => r.spoken.indexOf(t) !== i);
  ok(dupes.length === 0, 'nothing is said twice' + (dupes.length ? ': ' + dupes.join(',') : ''));

  const calls = spokenFor(r, 'm0').map(e => e.call_type).join(',');
  ok(calls === 'depart,far,near',
     'the tiers fire outermost first for one maneuver — ' + calls);

  const order = spokenFor(r, 'm0').filter(e => e.to_maneuver_m !== null)
    .map(e => Math.round(e.to_maneuver_m));
  ok(order.length >= 2 && order.every((v, i) => i === 0 || v < order[i - 1]),
     'and each is closer to the turn than the last (' + order.join(' > ') + ' m)');

  /* THE LADDER WAS FOLLOWED, not merely descended. Every call is logged with
     the tier distance it was supposed to fire at next to where the car
     actually was; a review of a drive is that pair, and the drive of
     2026-09-09 could not produce it. */
  const laddered = r.events.filter(e => e.at_m && e.to_maneuver_m !== null && e.text);
  const late = laddered.filter(e => e.to_maneuver_m > e.at_m + 30);
  ok(laddered.length >= 4 && late.length === 0,
     'every call landed at or inside its tier distance ('
     + laddered.map(e => e.call_type + ' ' + Math.round(e.to_maneuver_m)
                    + '/' + Math.round(e.at_m)).join(', ') + ')');

  const st = r.tracker.state();
  ok(st.gps_state === 'GPS_OK' && st.route_state === 'ON_ROUTE',
     'a clean drive stays GPS_OK and ON_ROUTE throughout');
}

// ---------------------------------------------------------------------------
section('the voice — every line of a whole drive is hers');
// ---------------------------------------------------------------------------
{
  /* SHE IS THE NAVIGATION. On the drive of 2026-09-09 a driver asked for
     directions and was told the car would call them out; the phrasing was in
     the session instructions and in two nav tool results, and a model handed a
     third-person description of its own behaviour hands it to the driver.
     tools/nav_server_selftest.py lints those two sources. This lints the third
     one: what the planner actually says over a whole drive, which is the only
     place the sentences and the timing meet. */
  const THIRD_PERSON = [
    'the car will', "the car's", 'the vehicle will', 'the car announces',
    'the system', 'the navigation system', 'navigation will', 'the nav system',
    'the gps will', 'it will call', "you'll hear", 'will call it out',
    'will let you know', 'will tell you', 'the turn-by-turn',
  ];
  const drives = [drive(synthRoute(), { speedMs: 8 }),
                  drive(synthRoute(), { speedMs: 20 }),
                  drive(synthRoute(), { startM: 1200 - 40 })];
  const said = [].concat(...drives.map(d => d.spoken));
  const bad = said.filter(t => THIRD_PERSON.some(p => t.toLowerCase().includes(p)));
  ok(said.length > 10 && bad.length === 0,
     'no spoken line in ' + drives.length + ' whole drives (' + said.length
     + ' lines) refers to a car or a navigation in the third person'
     + (bad.length ? ': ' + JSON.stringify(bad) : ''));

  /* ...AND THE LINT CAN FAIL, which is the only reason to trust it passing. */
  ok(THIRD_PERSON.some(p => 'The car will call it out.'.toLowerCase().includes(p)),
     'and the lint catches the sentence that was really said');
}

// ---------------------------------------------------------------------------
section('the junction call — only when the near call was long enough ago');
// ---------------------------------------------------------------------------
{
  /* IN TOWN IT FIRES. At 8 m/s the near call lands 19 s from the turn and the
     junction floor 5 s from it: a driver who has spent fourteen seconds
     looking for a street sign wants "Turn left." at the mouth of it. */
  const slow = drive(synthRoute(), { speedMs: 8 });
  ok(slow.spoken.indexOf('Turn left.') >= 0 && slow.spoken.indexOf('Turn right.') >= 0,
     'at 8 m/s both junctions get their two-word confirmation');
  const gaps = slow.events.filter(e => e.type === 'NAV_JUNCTION_CALL' && e.text);
  ok(gaps.length === 2, 'exactly two of them — ' + gaps.length);

  /* AT SPEED IT DOES NOT, and it says why. Four seconds after "Turn left onto
     Lincoln Boulevard." the same instruction again is not a confirmation. */
  const fast = drive(synthRoute(), { speedMs: 20 });
  ok(fast.spoken.indexOf('Turn left.') < 0,
     'at 20 m/s it is suppressed rather than stacked on the near call');
  const skipped = fast.events.filter(e => e.type === 'NAV_JUNCTION_CALL' && !e.text);
  ok(skipped.length >= 1 && skipped[0].skipped === 'near_call_too_recent',
     'and the suppression is logged with its reason, not silently dropped — '
     + (skipped.length ? skipped[0].skipped : 'nothing logged'));
  ok(skipped.length >= 1 && typeof skipped[0].since_near_s === 'number'
     && skipped[0].since_near_s < 10,
     '...with the gap that failed the test (' 
     + (skipped.length ? skipped[0].since_near_s : '?') + ' s < 10 s)');
}

// ---------------------------------------------------------------------------
section('"then" chaining — two junctions inside 200 m are one sentence');
// ---------------------------------------------------------------------------
{
  const route = synthRoute();
  // m1 moved to 120 m past m0, and the server's chained near call written
  // onto m0 exactly as navigation/speech.build_route would write it.
  route.maneuvers[1].route_distance_position = 1320;
  route.maneuvers[1].polyline_index = 132;
  route.maneuvers[1].lat = route.geometry[132][0];
  route.maneuvers[1].lng = route.geometry[132][1];
  route.maneuvers[0].speech.near =
    'Turn left onto Lincoln Boulevard, then turn right onto Fell Street.';
  route.maneuvers[0].speech.chained_to = 'm1';

  const r = drive(route, { speedMs: 8 });
  ok(r.spoken.indexOf(
       'Turn left onto Lincoln Boulevard, then turn right onto Fell Street.') >= 0,
     'the near call carries both moves in one sentence');
  ok(r.spoken.indexOf('In half a mile, turn right onto Fell Street.') < 0,
     'and the second turn gets NO far call of its own — it has been announced');
  ok(r.spoken.indexOf('Turn right onto Fell Street.') < 0,
     '...nor a second near call forty metres later, which is the car '
     + 'repeating itself at the worst moment');
  ok(r.spoken.indexOf('Turn right.') >= 0,
     'it keeps its junction confirmation, which is the half that is still news');
}

// ---------------------------------------------------------------------------
section('normal navigation — progress is monotonic under jitter');
// ---------------------------------------------------------------------------
{
  const clean = drive(synthRoute()).spoken;
  let n = 0;
  const noisy = drive(synthRoute(), {
    perturb: (fix) => {
      n++;
      const j = ((n * 2654435761) % 1000) / 1000 - 0.5;      // deterministic
      return { lat: fix.lat + (j * 24) / 111320, lng: fix.lng + (j * 24) / 92000,
               speed: fix.speed, t: fix.t, accuracy: 8 };
    },
  });
  ok(noisy.events.filter(e => e.type === 'NAV_OFF_ROUTE_CONFIRMED').length === 0,
     'jitter inside the off-route threshold never reroutes');
  ok(noisy.spoken.join('|') === clean.join('|'),
     'the spoken sequence is identical with and without jitter');
  ok(!noisy.tracker.state().arrived === false,
     'and the jittery drive still arrives');
}

const tick = () => new Promise(r => setTimeout(r, 0));

/* Fake items for the arbiter sections: play() resolves when the test says the
   line finished, so pre-emption and queueing can be driven deliberately. */
function item(props) {
  let resolve;
  const it = Object.assign({
    play: () => new Promise(r => { resolve = r; }),
    stop: () => { if (resolve) resolve(); },
  }, props);
  it.finish = () => { if (resolve) resolve(); };
  return it;
}

const VERIFIED_SHELL = {
  anchor: {
    anchor_id: 'm0a0', label: 'Shell', type: 'gas_station',
    turn_relation_to_anchor: 'NEAR', identity_confidence: 0.93,
    relation_confidence: 0.72, visibility_confidence: 0.81, valid_for_m: 400,
  },
  reason: 'verified',
};

async function main() {

// ---------------------------------------------------------------------------
section('contextual navigation — the differentiated line');
// ---------------------------------------------------------------------------
{
  // Slow enough that the junction call survives the >10 s rule, so this
  // section can still say what happens to it after a contextual near call.
  const r = drive(synthRoute({ anchors: [SHELL_ANCHOR] }),
                  { verify: () => VERIFIED_SHELL, speedMs: 8 });
  ok(r.spoken.indexOf('Turn left by the Shell station.') >= 0,
     'the near call names the landmark — "Turn left by the Shell station."');
  ok(r.spoken.indexOf('Turn left onto Lincoln Boulevard.') < 0,
     'and REPLACES the canonical instruction rather than following it');
  ok(r.spoken.indexOf('Turn left.') >= 0,
     'the junction confirmation stays armed after a contextual call (§11C)');
  const ctx = r.events.filter(e => e.type === 'NAV_CONTEXT_ACQUISITION_STARTED');
  ok(ctx.length === 1 && ctx[0].maneuver_id === 'm0',
     'acquisition ran once, for the maneuver that had a candidate');
  ok(ctx.length === 1 && ctx[0].to_maneuver_m <= 150 + TIMING.anchor_acquisition_lead_m + 20,
     'and started about ' + (150 + TIMING.anchor_acquisition_lead_m) + ' m out (' +
     (ctx.length ? Math.round(ctx[0].to_maneuver_m) : '?') + ' m) — before the '
     + 'near call, so the anchor is in hand when the sentence is due');
  const call = r.events.filter(e => e.type === 'NAV_CONTEXTUAL_CALL' && e.maneuver_id === 'm0')[0];
  ok(call && call.anchor_label === 'Shell' && call.relation === 'NEAR',
     'the call is logged with the anchor that produced it');
  ok(r.planner.contextState('m0') === 'CALLED',
     'the context lifecycle ends at CALLED — ' + r.planner.contextState('m0'));
  ok(r.spoken.filter(t => t.indexOf('Fell') >= 0).join(' | ') ===
     'In half a mile, turn right onto Fell Street. | Turn right onto Fell Street.',
     'the maneuver with no candidate is unaffected and speaks canonically');
  ok(r.spoken.indexOf('In half a mile, turn left onto Lincoln Boulevard.')
     < r.spoken.indexOf('Turn left by the Shell station.'),
     'the far call still comes first, without a landmark in it');
}

// ---------------------------------------------------------------------------
section('contextual navigation — the verifier answering over the network');
// ---------------------------------------------------------------------------
{
  // The real verifier is an HTTP call. Its answer lands between ticks, and the
  // drive must neither wait for it nor be surprised by it.
  const arbiter = speech.makeArbiter();
  const route = synthRoute({ anchors: [SHELL_ANCHOR] });
  const tracker = navcore.create(route);
  const spoken = [];
  let resolveVerify = null;
  const planner = navplan.create({
    tracker: tracker, arbiter: arbiter, route: route,
    verify: () => new Promise(res => { resolveVerify = res; }),
    audio: (c) => ({ play: () => { spoken.push(c.text); }, stop: () => {} }),
  });
  let s = 0, t = 0;
  // Drive up to just inside the acquisition window and stop.
  while (t < 200) {
    const p = tracker.pointAt(s);
    tracker.position({ lat: p.lat, lng: p.lng, speed: 15, t: t, accuracy: 8 });
    const st = tracker.state();
    // Stop after the camera has been asked and before the near call goes out:
    // 300 m out the question is in flight, 150 m out the sentence is due.
    if (st.to_maneuver_m !== null && st.to_maneuver_m < 200) break;
    s += 15; t += 1;
  }
  ok(planner.contextState('m0') === 'ACQUIRING',
     'while the answer is outstanding the context sits in ACQUIRING — ' +
     planner.contextState('m0'));
  ok(spoken.indexOf('In half a mile, turn left onto Lincoln Boulevard.') >= 0,
     'and the far call was already spoken, unblocked by it');
  resolveVerify(VERIFIED_SHELL);
  await tick();
  ok(planner.contextState('m0') === 'VERIFIED', 'the answer moves it to VERIFIED');
  // ...and the rest of the approach uses it.
  while (t < 200) {
    const p = tracker.pointAt(s);
    tracker.position({ lat: p.lat, lng: p.lng, speed: 15, t: t, accuracy: 8 });
    if (tracker.state().maneuver.maneuver_id !== 'm0') break;
    s += 15; t += 1;
  }
  ok(spoken.indexOf('Turn left by the Shell station.') >= 0,
     'the contextual line is spoken on the approach that followed');
}

// ---------------------------------------------------------------------------
section('contextual navigation — every failure falls back, and only that');
// ---------------------------------------------------------------------------
{
  const canonical = drive(synthRoute()).spoken.join(' | ');

  const cases = [
    ['no candidate at all', synthRoute(), () => ({ anchor: null, reason: 'no_candidates' })],
    ['nothing visible', synthRoute({ anchors: [SHELL_ANCHOR] }),
     () => ({ anchor: null, reason: 'not_visible' })],
    ['identity too uncertain', synthRoute({ anchors: [SHELL_ANCHOR] }),
     () => ({ anchor: null, reason: 'no_candidate_passed',
              rejections: { m0a0: ['identity_confidence'] } })],
    ['two of the same brand in view', synthRoute({ anchors: [SHELL_ANCHOR] }),
     () => ({ anchor: null, reason: 'no_candidate_passed',
              rejections: { m0a0: ['scene_uniqueness'] } })],
    ['unstable tracking', synthRoute({ anchors: [SHELL_ANCHOR] }),
     () => ({ anchor: null, reason: 'no_candidate_passed',
              rejections: { m0a0: ['tracking_duration'] } })],
    ['camera unavailable', synthRoute({ anchors: [SHELL_ANCHOR] }),
     () => ({ anchor: null, reason: 'camera_unavailable' })],
    ['the verifier itself throws', synthRoute({ anchors: [SHELL_ANCHOR] }),
     () => { throw new Error('vision model down'); }],
    ['vision switched off', synthRoute({ anchors: [SHELL_ANCHOR] }), null],
  ];
  cases.forEach(([name, route, verify]) => {
    const r = drive(route, { verify: verify });
    ok(r.spoken.join(' | ') === canonical,
       name + ' -> the drive is word for word the canonical one');
  });

  const rejected = drive(synthRoute({ anchors: [SHELL_ANCHOR] }), {
    verify: () => ({ anchor: null, reason: 'no_candidate_passed',
                     rejections: { m0a0: ['scene_uniqueness'] } }),
  });
  ok(rejected.events.filter(e => e.type === 'NAV_ANCHOR_REJECTED').length >= 1,
     'a rejection is an event in its own right, with its reason');
  ok(rejected.events.filter(e => e.type === 'NAV_ANCHOR_CANDIDATE').length >= 1,
     'the candidate that was tried is logged too, so the pair can be read together');
  ok(rejected.events.filter(e => e.type === 'NAV_CONTEXT_ACQUISITION_STARTED').length <= 2,
     'acquisition is retried at most once before the landmark is given up on');
  ok(rejected.planner.contextState('m0') === 'EXPIRED',
     'and the context ends EXPIRED, not stuck in ACQUIRING');
}

// ---------------------------------------------------------------------------
section('contextual navigation — an anchor that goes stale before it is used');
// ---------------------------------------------------------------------------
{
  // Verified early, then the landmark goes behind a truck: the anchor outlives
  // its shelf life before the call is due, and the canonical line is used.
  const r = drive(synthRoute({ anchors: [SHELL_ANCHOR] }), {
    verify: () => ({ anchor: Object.assign({}, VERIFIED_SHELL.anchor, { valid_for_m: 20 }),
                     reason: 'verified' }),
  });
  ok(r.spoken.indexOf('Turn left by the Shell station.') < 0,
     'a stale anchor is never spoken');
  ok(r.spoken.indexOf('Turn left onto Lincoln Boulevard.') >= 0,
     'the canonical instruction goes out in its place');
  ok(r.events.some(e => e.type === 'NAV_ANCHOR_REJECTED' && e.reason === 'anchor_expired'),
     'and the expiry is logged as the rejection it is');
}

// ---------------------------------------------------------------------------
section('speech validity — checked at dequeue, not at creation');
// ---------------------------------------------------------------------------
{
  // "Turn left." is queued behind a safety warning, and the junction goes by
  // while it waits. When the mouth frees up the line is no longer true.
  let block = null;
  const safety = {
    priority: speech.P.SAFETY, group: 'headway', id: 'too_close', text: 'Too close',
    play: () => new Promise(res => { block = res; }), stop: () => { if (block) block(); },
  };
  let armed = false;
  // Slow, so the junction call survives the >10 s rule and there is something
  // for the warning to hold back in the first place.
  const r = drive(synthRoute(), {
    speedMs: 8,
    after: (t, tracker, planner, arbiter) => {
      const st = tracker.state();
      if (!armed && st.maneuver && st.maneuver.maneuver_id === 'm0'
          && st.to_maneuver_m < 60) {
        armed = true;
        arbiter.say(safety);            // takes the mouth and holds it
      }
    },
  });
  ok(r.spoken.indexOf('Turn left.') < 0,
     '"Turn left." never reaches the mouth while the warning holds it');
  const queued = r.arbiter.state().queued.map(i => i.id);
  ok(queued.indexOf('nav:m0:junction') >= 0,
     'it is sitting in the queue, created and waiting — ' + queued.join(','));

  block();                              // the warning finishes, long after the turn
  await tick();
  ok(r.spoken.indexOf('Turn left.') < 0,
     'and it is dropped rather than played after the turn has been taken');
  ok(r.events.some(e => e.type === 'NAV_SPEECH_INVALIDATED' && e.call_type === 'junction'),
     'the drop is recorded as NAV_SPEECH_INVALIDATED, not silently lost');
}

// ---------------------------------------------------------------------------
section('speech validity — a reroute invalidates the generation it replaced');
// ---------------------------------------------------------------------------
{
  const arbiter = speech.makeArbiter();
  const route1 = synthRoute({ route_id: 'r1', generation_id: 1 });
  const tracker = navcore.create(route1);
  const spoken = [];
  // The production wiring: the planner asks what generation is live NOW, and a
  // reroute REPLACES the route object rather than editing it — so a planner
  // that read its own captured route would believe generation 1 forever.
  let live = route1;
  const planner = navplan.create({
    tracker: tracker, arbiter: arbiter, route: route1,
    activeGeneration: () => live.generation_id,
    audio: (c) => ({ play: () => { spoken.push(c.text + ' @gen' + c.route_generation); },
                     stop: () => {} }),
  });
  let hold = null;
  arbiter.say({ priority: speech.P.SAFETY, group: 'headway', id: 'hw',
                play: () => new Promise(r => { hold = r; }), stop: () => { if (hold) hold(); } });

  // Drive to just before the first turn, so a nav line is created and queues
  // behind the warning...
  let s = 0, t = 0;
  while (t < 200) {
    const p = tracker.pointAt(s);
    tracker.position({ lat: p.lat, lng: p.lng, speed: 15, t: t, accuracy: 8 });
    const st = tracker.state();
    if (st.tta_s !== null && st.tta_s < 2 && st.maneuver.maneuver_id === 'm0') break;
    s += 15; t += 1;
  }
  ok(arbiter.state().queued.length > 0, 'a nav line is queued behind the warning');
  ok(spoken.length === 0, 'and nothing has been spoken — the warning has the mouth');

  // ...then the driver leaves the route, and a NEW route object — generation
  // 2 of the same journey — replaces it. The old one is untouched, exactly as
  // it is in the browser.
  live = synthRoute({ route_id: 'r2', generation_id: 2 });
  ok(route1.generation_id === 1,
     'the replaced route object is not edited — it is simply no longer current');
  hold();
  await tick();
  ok(spoken.length === 0,
     'the queued line from generation 1 is discarded once the route is replaced');
  ok(planner.state().counters.invalidated >= 1,
     'and it is counted as invalidated (' + planner.state().counters.invalidated + ')');
}

// ---------------------------------------------------------------------------
section('off route — sustained, never instantaneous, never from GPS health');
// ---------------------------------------------------------------------------
{
  const route = synthRoute();
  const tracker = navcore.create(route);
  const seen = [];
  tracker.onEvent(e => { if (e.type.indexOf('OFF_ROUTE') >= 0) seen.push(e); });
  const mLat = 111320;
  let s = 0;
  for (let i = 0; i < 3; i++) {          // two fixes off, then back on
    const p = tracker.pointAt(s);
    tracker.position({ lat: p.lat + (i < 2 ? 200 / mLat : 0), lng: p.lng,
                       speed: 20, t: i, accuracy: 8 });
    s += 20;
  }
  ok(seen.filter(e => e.type === 'NAV_OFF_ROUTE_CONFIRMED').length === 0,
     'two off-route fixes and a recovery do not confirm anything');
  ok(seen.filter(e => e.type === 'NAV_OFF_ROUTE_CANDIDATE').length === 1,
     'though the first one is reported as a candidate, which is what it is');
  ok(tracker.state().route_state === 'ON_ROUTE', 'and the state returns to ON_ROUTE');

  for (let i = 3; i < 7; i++) {
    const p = tracker.pointAt(s);
    tracker.position({ lat: p.lat + 200 / mLat, lng: p.lng, speed: 20, t: i, accuracy: 8 });
    s += 20;
  }
  const confirmed = seen.filter(e => e.type === 'NAV_OFF_ROUTE_CONFIRMED');
  ok(confirmed.length === 1, 'three consecutive off-route fixes confirm exactly once');
  ok(confirmed.length === 1 && confirmed[0].off_route_m > 45,
     'the event carries how far off the route we are (' +
     (confirmed.length ? Math.round(confirmed[0].off_route_m) + ' m' : 'n/a') + ')');
  ok(tracker.state().stopped,
     'and tracking stops on a route it knows is wrong — the provider owns the new one');
}

{
  // A fix whose own accuracy is worse than the deviation it reports is not
  // evidence of anything.
  const tracker = navcore.create(synthRoute());
  const seen = [];
  tracker.onEvent(e => { if (e.type.indexOf('OFF_ROUTE') >= 0) seen.push(e); });
  let s = 0;
  for (let i = 0; i < 6; i++) {
    const p = tracker.pointAt(s);
    tracker.position({ lat: p.lat + 60 / 111320, lng: p.lng, speed: 20, t: i, accuracy: 120 });
    s += 20;
  }
  ok(seen.length === 0,
     'a 60 m deviation on a fix accurate to ±120 m never counts towards off-route');
}

// ---------------------------------------------------------------------------
section('GPS health — stale is not off-route, degraded speaks earlier');
// ---------------------------------------------------------------------------
{
  const tracker = navcore.create(synthRoute());
  const seen = [];
  tracker.onEvent(e => seen.push(e.type));
  let s = 0;
  for (let i = 0; i < 5; i++) {
    const p = tracker.pointAt(s);
    tracker.position({ lat: p.lat, lng: p.lng, speed: 15, t: i, accuracy: 8 });
    s += 15;
  }
  // The fixes simply stop. Ten seconds of clock, no positions.
  for (let t = 5; t < 15; t++) tracker.tick(t);
  ok(tracker.state().gps_state === 'GPS_STALE', 'no fixes for ten seconds -> GPS_STALE');
  ok(tracker.state().route_state === 'ON_ROUTE',
     'and the route state is untouched — a lost fix is not a wrong turn');
  ok(seen.indexOf('NAV_OFF_ROUTE_CANDIDATE') < 0 && seen.indexOf('NAV_OFF_ROUTE_CONFIRMED') < 0,
     'nothing about off-route was even considered');
  ok(seen.filter(t => t === 'NAV_GPS_STALE').length === 1,
     'the transition is reported once, not once per tick');
}

{
  const clean = drive(synthRoute());
  const degraded = drive(synthRoute(), {
    perturb: (fix) => Object.assign({}, fix, { accuracy: 45 }),   // beyond the limit
  });
  const cleanCall = spokenFor(clean, 'm0').filter(e => e.call_type === 'near')[0];
  const degCall = spokenFor(degraded, 'm0').filter(e => e.call_type === 'near')[0];
  ok(degraded.tracker.state().gps_state === 'GPS_DEGRADED',
     'a fix worse than the accuracy limit -> GPS_DEGRADED');
  // The widening is metres of road now, not seconds added to seconds: at
  // 15 m/s a 2 s bias is 30 m of extra room on every tier.
  ok(degCall && cleanCall && degCall.to_maneuver_m >= cleanCall.to_maneuver_m,
     'the near call comes no later than it would have: ' +
     Math.round(degCall.to_maneuver_m) + ' m vs ' + Math.round(cleanCall.to_maneuver_m) + ' m');
  ok(degraded.spoken.join('|') === clean.spoken.join('|'),
     'and what is said does not change — only when');
}

// ---------------------------------------------------------------------------
section('iOS — heading and speed missing from every fix');
// ---------------------------------------------------------------------------
{
  const tracker = navcore.create(synthRoute());
  let s = 0;
  for (let t = 0; t < 12; t++) {
    const p = tracker.pointAt(s);
    // What Safari actually hands over: no speed, no heading.
    tracker.position({ lat: p.lat, lng: p.lng, speed: null, heading: null,
                       t: t, accuracy: 12 });
    s += 15;
  }
  const st = tracker.state();
  ok(st.speed_source === 'derived', 'speed is derived from consecutive fixes');
  ok(Math.abs(st.speed_ms - 15) < 1.5,
     'and lands on the real speed (' + st.speed_ms.toFixed(1) + ' m/s of 15)');
  ok(st.heading_source === 'derived', 'heading is derived from the same fixes');
  ok(Math.abs(((st.heading_deg - 90) + 540) % 360 - 180) < 8,
     'and points along the road (' + Math.round(st.heading_deg) + '° of 90°)');

  const withHeading = navcore.create(synthRoute());
  const p0 = withHeading.pointAt(0);
  withHeading.position({ lat: p0.lat, lng: p0.lng, speed: 15, heading: 88, t: 0, accuracy: 8 });
  ok(withHeading.state().heading_source === 'fix',
     'a device that DOES report heading is believed, not second-guessed');
}

{
  // Parked, with GPS noise. The derivation must refuse: a heading from two
  // fixes 3 m apart while stationary is noise with a compass rose on it.
  const tracker = navcore.create(synthRoute());
  const p = tracker.pointAt(0);
  for (let t = 0; t < 8; t++) {
    const j = ((t * 7919) % 100) / 100 - 0.5;
    tracker.position({ lat: p.lat + (j * 5) / 111320, lng: p.lng + (j * 5) / 92000,
                       speed: 0.2, heading: null, t: t, accuracy: 10 });
  }
  ok(tracker.state().heading_source !== 'derived',
     'a stationary car never derives a heading — source is "' +
     tracker.state().heading_source + '"');
}

// ---------------------------------------------------------------------------
section('starting inside the window — no line begun too late to finish');
// ---------------------------------------------------------------------------
{
  const r = drive(synthRoute(), { startM: 1200 - 40 });
  const m0 = spokenFor(r, 'm0').map(e => e.call_type);
  ok(m0.indexOf('far') < 0,
     'a route set 40 m from a turn never says "in half a mile" — the distance '
     + 'in that sentence was written at route load and is now false');
  ok(r.spoken[0] === 'Head east on Venice Boulevard, then turn left onto Lincoln Boulevard.',
     'the route-start line still goes out — it is what road we are on and '
     + 'what the first move is, and neither has stopped being true');
  ok(r.spoken.indexOf('Turn left.') >= 0,
     'and the junction gets the two-word line, because there was no room for '
     + 'a full instruction');
  ok(r.spoken.indexOf('Turn left onto Lincoln Boulevard.') < 0,
     '...which is not begun and then still playing through the turn');
  const skipped = r.events.filter(e => e.type === 'NAV_CONTEXTUAL_CALL'
                                  && e.skipped === 'no_room_to_finish');
  ok(skipped.length === 1, 'and the skip is logged with its reason');
}

// ---------------------------------------------------------------------------
section('the turn-by-turn — the list a driver asks for, not the next call');
// ---------------------------------------------------------------------------
{
  /* state() answers "what is the NEXT turn", which is what an announcement
     needs. Asked "what are the directions", RIO had nothing to read from: the
     list was in here all along and nothing exposed it. upcoming() is that,
     measured from where the car actually is. */
  const tracker = navcore.create(synthRoute({ anchors: [SHELL_ANCHOR] }));
  const p0 = tracker.pointAt(0);
  tracker.position({ lat: p0.lat, lng: p0.lng, speed: 15, t: 0 });

  const all = tracker.upcoming();
  ok(all.length === 3, 'every maneuver still ahead is listed (' + all.length + ')');
  ok(all[0].road_name === 'Lincoln Boulevard' && all[1].road_name === 'Fell Street',
     'in route order, with the road names a driver would recognise');
  ok(all[0].distance_m === 1200 && all[1].distance_m === 2100,
     'distances are measured from the car, not from the route start (' +
     all[0].distance_m + ' m, ' + all[1].distance_m + ' m)');
  ok(all[0].leg_m === 1200 && all[1].leg_m === 900,
     'and each leg is the gap from the previous turn — "then straight for ' +
     'nine hundred metres" (' + all[1].leg_m + ' m)');
  ok(all[0].anchors.length === 1 && all[0].anchors[0].label === 'Shell',
     'a maneuver carries its landmark candidates for whoever reads it');
  ok(all[2].anchors.length === 0, 'and a maneuver without one carries none');

  ok(tracker.upcoming(2).length === 2, 'a count truncates the list');
  ok(tracker.upcoming(99).length === 3, 'asking for more than exist is not an error');
  ok(tracker.upcoming(0).length === 0, 'and zero is zero, not "all"');

  /* Drive to just before the first turn: the distances have to move with the
     car, or the answer is wrong in the way nobody notices until they follow
     it. Driven in steps rather than teleported, because the projection is
     windowed on purpose — a single 1 km jump is not a fix, it is a bug, and
     the tracker treats it as one. */
  // Continues from where the last call left off: `along` is monotonic on
  // purpose, and restarting the drive from the beginning each time would be
  // asking the tracker to accept the car teleporting backwards.
  let driven = 0;
  function driveTo(s_end) {
    for (let s = driven + 20; s <= s_end; s += 20) {
      const p = tracker.pointAt(s);
      tracker.position({ lat: p.lat, lng: p.lng, speed: 15, t: s / 15 });
      driven = s;
    }
  }
  driveTo(1000);
  const mid = tracker.upcoming();
  ok(mid.length === 3 && Math.abs(mid[0].distance_m - 200) <= 15,
     'after a kilometre the next turn is ~200 m away, not 1200 (' +
     mid[0].distance_m + ' m)');
  ok(Math.abs(mid[0].leg_m - 200) <= 15,
     'the first leg is measured from the car too (' + mid[0].leg_m + ' m)');

  /* Past the first turn: it drops off the list. A driver asking for the
     directions after a turn is asking what is LEFT. */
  driveTo(1400);
  const after = tracker.upcoming();
  ok(after.length === 2 && after[0].road_name === 'Fell Street',
     'a maneuver already driven through is no longer in the directions');
}

// ---------------------------------------------------------------------------
section('the arbiter — one mouth (unchanged contracts)');
// ---------------------------------------------------------------------------
{
  const arb = speech.makeArbiter();
  const ends = [];
  const nav = item({ priority: speech.P.NAV, group: 'nav:m0', id: 'primary',
                     text: 'Take the next right', onDone: r => ends.push('nav:' + r) });
  const hw = item({ priority: speech.P.SAFETY, group: 'headway', id: 'too_close',
                    onDone: r => ends.push('hw:' + r) });
  arb.say(nav); await tick();
  ok(arb.state().speaking.id === 'primary', 'a nav line starts when nothing else is speaking');
  arb.say(hw); await tick();
  ok(arb.state().speaking.id === 'too_close', 'a safety line pre-empts it mid-sentence');
  ok(ends[0] === 'nav:preempted', 'the pre-empted nav line is dropped, not resumed');
  hw.finish(); await tick();
  ok(arb.state().speaking === null && arb.state().queued.length === 0,
     'nothing is resumed after the safety line — the mouth moves forward');
}
{
  const arb = speech.makeArbiter();
  const ends = [];
  const primary = item({ priority: speech.P.NAV, group: 'nav:m0', id: 'primary',
                         onDone: r => ends.push('primary:' + r) });
  const near = item({ priority: speech.P.TURN_NEAR, group: 'nav:m0', id: 'imminent' });
  arb.say(primary); await tick();
  arb.say(near); await tick();
  ok(ends[0] === 'primary:superseded',
     'the imminent call supersedes the primary one for the same maneuver');
  ok(arb.state().speaking.id === 'imminent', 'and takes the mouth immediately');
}
{
  const arb = speech.makeArbiter();
  const ends = [];
  const hw = item({ priority: speech.P.SAFETY, group: 'headway', id: 'hw' });
  const m0 = item({ priority: speech.P.NAV, group: 'nav:m0', id: 'm0', ttlMs: 5,
                    onDone: r => ends.push('m0:' + r) });
  const m1 = item({ priority: speech.P.NAV, group: 'nav:m1', id: 'm1' });
  arb.say(hw); await tick();
  arb.say(m0); arb.say(m1); await tick();
  ok(arb.state().queued.map(i => i.id).join(',') === 'm0,m1',
     'lower-priority lines queue behind the safety line in arrival order');
  await new Promise(r => setTimeout(r, 20));
  hw.finish(); await tick();
  ok(ends[0] === 'm0:expired',
     'a nav line that outlived its window is dropped, never played late');
  ok(arb.state().speaking && arb.state().speaking.id === 'm1', 'the still-valid line plays');
}
{
  const arb = speech.makeArbiter();
  const ends = [];
  const hw = item({ priority: speech.P.SAFETY, group: 'headway', id: 'hw' });
  const nav = item({ priority: speech.P.NAV, group: 'nav:m0', id: 'm0',
                     valid: () => false, onDone: r => ends.push('m0:' + r) });
  arb.say(hw); await tick();
  arb.say(nav); await tick();
  hw.finish(); await tick();
  ok(ends[0] === 'm0:invalid',
     'a line whose validity has lapsed is dropped at dequeue with its own reason');
}
{
  const P = speech.P;
  ok(P.SAFETY < P.VEHICLE_HEALTH && P.VEHICLE_HEALTH < P.TURN_NEAR &&
     P.TURN_NEAR < P.NAV && P.NAV < P.CONVO,
     'the ladder is safety > vehicle health > near turn > nav > conversation');
  const arb = speech.makeArbiter();
  const convo = item({ priority: P.CONVO, group: 'convo', id: 'answer' });
  const near = item({ priority: P.TURN_NEAR, group: 'nav:m3', id: 'imminent' });
  arb.say(convo); await tick();
  arb.say(near); await tick();
  ok(arb.state().speaking.id === 'imminent',
     'the imminent turn pre-empts a visual answer mid-sentence');
}
{
  const arb = speech.makeArbiter();
  const health = item({ priority: speech.P.VEHICLE_HEALTH, group: 'health', id: 'blowout' });
  const near = item({ priority: speech.P.TURN_NEAR, group: 'nav:m3', id: 'imminent' });
  arb.say(near); await tick();
  arb.say(health); await tick();
  ok(arb.state().speaking.id === 'blowout',
     'a critical health announcement still cuts through a turn announcement');
}

// ---------------------------------------------------------------------------
// Optional: the same checks against a route saved from /nav/route.
// ---------------------------------------------------------------------------
const fixture = process.argv[2];
if (fixture && fs.existsSync(fixture)) {
  section('real geometry — ' + path.basename(fixture));
  const route = JSON.parse(fs.readFileSync(fixture, 'utf8'));
  const r = drive(route, { speedMs: 12 });
  const selected = r.events.filter(e => e.type === 'NAV_MANEUVER_SELECTED');
  ok(selected.length === route.maneuvers.length,
     route.maneuvers.length + ' maneuvers, each selected once (got ' + selected.length + ')');
  ok(r.events.filter(e => e.type === 'NAV_OFF_ROUTE_CONFIRMED').length === 0,
     'driving the route exactly never goes off it');
  ok(r.events[r.events.length - 1].type === 'NAV_ARRIVED', 'the drive ends arrived');
  let bad = null;
  const seq = selected.map(e => e.sequence);
  for (let i = 1; i < seq.length; i++) if (seq[i] <= seq[i - 1]) bad = i;
  ok(bad === null, 'maneuvers are selected in route order');
  console.log('\n  the first eight lines of that drive:');
  r.spoken.slice(0, 8).forEach(t => console.log('    "' + t + '"'));
}

console.log('\n' + (failures ? 'FAILED ' + failures + '/' : 'PASSED ') + checks + ' checks');
process.exit(failures ? 1 : 0);
}

main().catch(e => { console.error(e); process.exit(2); });
