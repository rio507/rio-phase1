/* nav_selftest.js — the navigation foundation, run without a browser.
 *
 *   node tools/nav_selftest.js [route.json]
 *
 * Two things are worth asserting about turn-by-turn, and neither of them can be
 * checked by looking at a dashboard:
 *
 *   1. Announcements fire in the right ORDER — far, then mid, then near, once
 *      each, per maneuver, in maneuver order, and nothing after arrival.
 *   2. The arbiter's mouth is single — a safety line cuts in, a superseded nav
 *      line dies rather than queueing, and a stale one expires.
 *
 * The default route is synthetic and deterministic (an L: two turns then a
 * destination), so this runs offline in CI. Pass a route.json saved from
 * /nav/route to run the same checks against real Google geometry.
 *
 * static/rio_navcore.js and static/rio_speech.js are DOM-free precisely so this
 * file can require() them: the code under test here is the code that ships.
 */
'use strict';

const path = require('path');
const fs = require('fs');

const navcore = require(path.join(__dirname, '..', 'static', 'rio_navcore.js'));
const speech = require(path.join(__dirname, '..', 'static', 'rio_speech.js'));

let checks = 0, failures = 0;
function ok(cond, what) {
  checks++;
  if (!cond) { failures++; console.log('  FAIL  ' + what); }
  else console.log('  ok    ' + what);
}
function section(name) { console.log('\n=== ' + name + ' ==='); }

// ---------------------------------------------------------------------------
// A synthetic route: 1200 m east, turn left, 900 m north, turn right, 500 m
// east, arrive. Vertices every 10 m so projection has real geometry to chew on.
// ---------------------------------------------------------------------------
function synthRoute() {
  const lat0 = 37.7749, lng0 = -122.4194;
  const mLat = 111320, mLng = 111320 * Math.cos(lat0 * Math.PI / 180);
  const pts = [];
  const push = (x, y) => pts.push([lat0 + y / mLat, lng0 + x / mLng]);
  let x = 0, y = 0;
  push(x, y);
  for (let d = 10; d <= 1200; d += 10) push(x + d, y);           x += 1200;
  for (let d = 10; d <= 900; d += 10) push(x, y + d);            y += 900;
  for (let d = 10; d <= 500; d += 10) push(x + d, y);            x += 500;

  const iTurn1 = 120;            // vertex at 1200 m
  const iTurn2 = 120 + 90;       // vertex at 2100 m
  const iEnd = pts.length - 1;   // 2600 m
  const ann = (instr) => ({
    far: { text: 'In 900 meters, ' + instr.toLowerCase(), template: 'In {d}, ' + instr.toLowerCase() },
    mid: { text: 'In 350 meters, ' + instr.toLowerCase(), template: 'In {d}, ' + instr.toLowerCase() },
    near: { text: instr, template: instr },
  });
  return {
    route_id: 'synth', provider: 'web',
    destination: { label: 'Test Destination', lat: pts[iEnd][0], lng: pts[iEnd][1] },
    origin: { lat: pts[0][0], lng: pts[0][1] },
    distance_m: 2600, duration_s: 130, eta_epoch: 0,
    polyline: pts,
    tiers_s: { far: 30, mid: 12, near: 4 },
    maneuvers: [
      { index: 0, type: 'TURN_LEFT', instruction: 'Turn left onto Lincoln',
        poly_index: iTurn1, lat: pts[iTurn1][0], lng: pts[iTurn1][1], announce: ann('Turn left onto Lincoln') },
      { index: 1, type: 'TURN_RIGHT', instruction: 'Turn right onto Fell',
        poly_index: iTurn2, lat: pts[iTurn2][0], lng: pts[iTurn2][1], announce: ann('Turn right onto Fell') },
      { index: 2, type: 'ARRIVE', instruction: 'Arrive at Test Destination',
        poly_index: iEnd, lat: pts[iEnd][0], lng: pts[iEnd][1], announce: ann('Arrive at Test Destination') },
    ],
  };
}

/* Walk the route at a fixed speed, 1 Hz, exactly as the dashboard's simulate
   mode does, and collect everything the engine emitted. */
function driveRoute(route, speedMs, opts) {
  const eng = navcore.create(route, opts && opts.engine);
  const events = [];
  eng.onEvent(e => { if (e.type !== 'progress') events.push(e); });
  const total = eng.routeLength();
  let s = 0, t = 0;
  while (s <= total + 60 && t < 5000) {
    const p = eng.pointAt(Math.min(s, total));
    let fix = { lat: p.lat, lng: p.lng, speed: speedMs, t: t };
    if (opts && opts.perturb) fix = opts.perturb(fix, s);
    eng.position(fix);
    if (eng.state().arrived || eng.state().stopped) break;
    s += speedMs; t += 1;
  }
  return { events, eng };
}

// ---------------------------------------------------------------------------
section('progression — synthetic route at 20 m/s');
// ---------------------------------------------------------------------------
{
  const route = synthRoute();
  const { events } = driveRoute(route, 20);
  const approaches = events.filter(e => e.type === 'maneuver_approach');
  const completes = events.filter(e => e.type === 'maneuver_complete');
  const arrived = events.filter(e => e.type === 'arrived');

  ok(approaches.length === 9,
     'nine approach announcements — three tiers for each of three maneuvers (got ' + approaches.length + ')');

  const order = approaches.map(e => e.maneuver + ':' + e.tier).join(' ');
  ok(order === '0:far 0:mid 0:near 1:far 1:mid 1:near 2:far 2:mid 2:near',
     'tiers fire coarse-to-fine, maneuver by maneuver — ' + order);

  ok(approaches.every(e => e.tta_s <= route.tiers_s[e.tier] + 0.001),
     'no tier ever fires before its threshold');

  // ...and none fires late either, except the first announcement for a maneuver
  // that only became the next one when we were already inside its window — a
  // turn 500 m before the destination is announced the moment the turn is done,
  // at whatever time-to-arrival that is. That is the behaviour we want; the
  // check just has to know the difference.
  const firstFor = {};
  approaches.forEach(e => { if (firstFor[e.maneuver] === undefined) firstFor[e.maneuver] = e.tier; });
  ok(approaches.every(e => firstFor[e.maneuver] === e.tier ||
                           e.tta_s > route.tiers_s[e.tier] - 1.5),
     'and none fires more than one fix late (30/12/4 s at 1 Hz)');

  ok(completes.map(e => e.maneuver).join(',') === '0,1',
     'both turns complete, in order, and arrival is not one of them');

  const firstComplete = events.findIndex(e => e.type === 'maneuver_complete');
  const lastApproachM0 = events.map(e => e.type === 'maneuver_approach' && e.maneuver === 0)
                               .lastIndexOf(true);
  ok(lastApproachM0 < firstComplete,
     'a maneuver is announced before it is completed, never after');

  ok(arrived.length === 1 && events[events.length - 1].type === 'arrived',
     'arrived fires exactly once, last');

  ok(completes.every(e => e.announced_tier === 'near'),
     'every completed maneuver had been announced down to the near tier');
}

// ---------------------------------------------------------------------------
section('progression — 5 m/s: the same turns, announced later and closer');
// ---------------------------------------------------------------------------
{
  const route = synthRoute();
  const fast = driveRoute(synthRoute(), 20).events.filter(e => e.type === 'maneuver_approach');
  const slow = driveRoute(route, 5).events.filter(e => e.type === 'maneuver_approach');
  ok(slow.filter(e => e.maneuver === 0).map(e => e.tier).join(',') === 'far,mid,near',
     'all three tiers still fire at low speed');
  const fFar = fast.find(e => e.maneuver === 0 && e.tier === 'far').remaining_m;
  const sFar = slow.find(e => e.maneuver === 0 && e.tier === 'far').remaining_m;
  ok(sFar < fFar / 3,
     'time tiers scale with speed: far fires at ' + Math.round(sFar) + ' m at 5 m/s vs '
     + Math.round(fFar) + ' m at 20 m/s');
}

// ---------------------------------------------------------------------------
section('progression — route set 60 m from a turn');
// ---------------------------------------------------------------------------
{
  // Starting inside the near window must announce the turn itself, not begin a
  // countdown that is already false.
  const route = synthRoute();
  const eng = navcore.create(route);
  const seen = [];
  eng.onEvent(e => { if (e.type === 'maneuver_approach') seen.push(e.tier); });
  const p = eng.pointAt(1200 - 60);
  eng.position({ lat: p.lat, lng: p.lng, speed: 20, t: 0 });
  ok(seen.join(',') === 'near', 'fires the near tier only, not far — ' + (seen.join(',') || 'nothing'));
}

// ---------------------------------------------------------------------------
section('off route — sustained, not instantaneous');
// ---------------------------------------------------------------------------
{
  const route = synthRoute();
  const eng = navcore.create(route);
  const seen = [];
  eng.onEvent(e => { if (e.type === 'reroute') seen.push(e); });
  const mLat = 111320;
  let s = 0;
  // Two fixes 200 m off the route, then back on: a GPS blip, not a wrong turn.
  for (let i = 0; i < 3; i++) {
    const p = eng.pointAt(s);
    eng.position({ lat: p.lat + (i < 2 ? 200 / mLat : 0), lng: p.lng, speed: 20, t: i });
    s += 20;
  }
  ok(seen.length === 0, 'two off-route fixes and a recovery do not reroute');

  for (let i = 3; i < 7; i++) {
    const p = eng.pointAt(s);
    eng.position({ lat: p.lat + 200 / mLat, lng: p.lng, speed: 20, t: i });
    s += 20;
  }
  ok(seen.length === 1, 'three consecutive off-route fixes ask for one reroute');
  ok(seen.length === 1 && seen[0].off_route_m > 45,
     'the reroute event carries how far off the route we are (' +
     (seen.length ? Math.round(seen[0].off_route_m) + ' m' : 'n/a') + ')');
  ok(eng.state().stopped,
     'the engine stops announcing on a route it knows is wrong — Google owns the new one');
}

// ---------------------------------------------------------------------------
section('gps noise — ±12 m jitter must not change what is said');
// ---------------------------------------------------------------------------
{
  const clean = driveRoute(synthRoute(), 20).events.filter(e => e.type === 'maneuver_approach');
  let n = 0;
  const noisy = driveRoute(synthRoute(), 20, {
    perturb: (fix) => {
      // Deterministic pseudo-jitter, so a failure here is reproducible.
      n++;
      const j = ((n * 2654435761) % 1000) / 1000 - 0.5;   // -0.5..0.5
      return { lat: fix.lat + (j * 24) / 111320, lng: fix.lng + (j * 24) / 88000,
               speed: fix.speed, t: fix.t };
    },
  }).events;
  const noisyApp = noisy.filter(e => e.type === 'maneuver_approach');
  ok(noisy.filter(e => e.type === 'reroute').length === 0,
     'jitter inside the off-route threshold never reroutes');
  ok(noisyApp.map(e => e.maneuver + ':' + e.tier).join(' ') ===
     clean.map(e => e.maneuver + ':' + e.tier).join(' '),
     'the announcement sequence is identical with and without jitter');
}

// ---------------------------------------------------------------------------
section('speech arbiter — one mouth');
// ---------------------------------------------------------------------------
{
  // Fake items: play() resolves when the test says the line finished.
  function item(props) {
    let resolve;
    const it = Object.assign({
      play: () => new Promise(r => { resolve = r; }),
      stop: () => { if (resolve) resolve(); },
    }, props);
    it.finish = () => { if (resolve) resolve(); };
    return it;
  }
  const wait = () => new Promise(r => setTimeout(r, 0));

  return (async () => {
    {
      const arb = speech.makeArbiter();
      const ends = [];
      const nav = item({ priority: 3, group: 'nav:m0', id: 'far', text: 'In 300 meters, turn right',
                         onDone: r => ends.push('nav:' + r) });
      const hw = item({ priority: 1, group: 'headway', id: 'too_close', text: 'Too close',
                        onDone: r => ends.push('hw:' + r) });
      arb.say(nav); await wait();
      ok(arb.state().speaking.id === 'far', 'a nav line starts when nothing else is speaking');
      arb.say(hw); await wait();
      ok(arb.state().speaking.id === 'too_close', 'a safety line pre-empts a nav line mid-sentence');
      ok(ends[0] === 'nav:preempted', 'the pre-empted nav line is reported dropped, not resumed');
      hw.finish(); await wait();
      ok(arb.state().speaking === null && arb.state().queued.length === 0,
         'nothing is resumed after the safety line — the mouth moves forward');
    }

    {
      const arb = speech.makeArbiter();
      const ends = [];
      const far = item({ priority: 3, group: 'nav:m0', id: 'far', onDone: r => ends.push('far:' + r) });
      const near = item({ priority: 2, group: 'nav:m0', id: 'near', onDone: r => ends.push('near:' + r) });
      arb.say(far); await wait();
      arb.say(near); await wait();
      ok(ends[0] === 'far:superseded', 'a fresher tier supersedes the older one for the same maneuver');
      ok(arb.state().speaking.id === 'near', 'and takes the mouth immediately');
    }

    {
      const arb = speech.makeArbiter();
      const ends = [];
      const hw = item({ priority: 1, group: 'headway', id: 'hw' });
      const m0 = item({ priority: 3, group: 'nav:m0', id: 'm0', ttlMs: 5,
                        onDone: r => ends.push('m0:' + r) });
      const m1 = item({ priority: 3, group: 'nav:m1', id: 'm1', onDone: r => ends.push('m1:' + r) });
      arb.say(hw); await wait();
      arb.say(m0); arb.say(m1); await wait();
      ok(arb.state().queued.map(i => i.id).join(',') === 'm0,m1',
         'lower-priority lines queue behind the safety line in arrival order');
      await new Promise(r => setTimeout(r, 20));
      hw.finish(); await wait();
      ok(ends[0] === 'm0:expired',
         'a nav line that outlived its window is dropped, never played late');
      ok(arb.state().speaking && arb.state().speaking.id === 'm1',
         'the still-valid line plays');
    }

    {
      const arb = speech.makeArbiter();
      const ends = [];
      const hw = item({ priority: 1, group: 'headway', id: 'hw', onDone: r => ends.push('hw:' + r) });
      const m0 = item({ priority: 3, group: 'nav:m0', id: 'm0', onDone: r => ends.push('m0:' + r) });
      arb.say(hw); arb.say(m0); await wait();
      arb.clear('nav:');
      ok(ends.join(',') === 'm0:cleared', 'clear("nav:") drops nav only');
      ok(arb.state().speaking.id === 'hw', 'and leaves the safety line speaking');
    }

    // -----------------------------------------------------------------------
    // Conversation (P4). A visual answer is the longest thing RIO says and the
    // only one the driver can simply ask for again, so it is the tier that
    // yields to everything. These four checks are the whole of that claim.
    // -----------------------------------------------------------------------
    section('arbiter — conversation yields to navigation and safety');
    {
      const arb = speech.makeArbiter();
      const ends = [];
      const convo = item({ priority: speech.P.CONVO, group: 'convo', id: 'answer',
                           text: 'That looks like a C5 Corvette.',
                           onDone: r => ends.push('convo:' + r) });
      const near = item({ priority: speech.P.TURN_NEAR, group: 'nav:m3', id: 'near',
                          onDone: r => ends.push('near:' + r) });
      arb.say(convo); await wait();
      ok(arb.state().speaking.id === 'answer', 'a visual answer speaks when nothing else is');
      arb.say(near); await wait();
      ok(arb.state().speaking.id === 'near',
         'the near-tier turn pre-empts a visual answer mid-sentence');
      ok(ends[0] === 'convo:preempted',
         'the cut-off answer is dropped, never resumed after the turn');
    }
    {
      const arb = speech.makeArbiter();
      const convo = item({ priority: speech.P.CONVO, group: 'convo', id: 'answer' });
      const hw = item({ priority: speech.P.SAFETY, group: 'headway', id: 'too_close' });
      arb.say(convo); await wait();
      arb.say(hw); await wait();
      ok(arb.state().speaking.id === 'too_close',
         'a gap warning pre-empts a visual answer');
    }
    {
      const arb = speech.makeArbiter();
      const ends = [];
      const first = item({ priority: speech.P.CONVO, group: 'convo', id: 'a1',
                           onDone: r => ends.push('a1:' + r) });
      const second = item({ priority: speech.P.CONVO, group: 'convo', id: 'a2' });
      arb.say(first); await wait();
      arb.say(second); await wait();
      ok(ends[0] === 'a1:superseded' && arb.state().speaking.id === 'a2',
         'asking a second question replaces the answer still being spoken');
    }
    {
      const arb = speech.makeArbiter();
      const hw = item({ priority: speech.P.SAFETY, group: 'headway', id: 'hw' });
      const convo = item({ priority: speech.P.CONVO, group: 'convo', id: 'answer' });
      const nav = item({ priority: speech.P.NAV, group: 'nav:m0', id: 'far' });
      arb.say(hw); await wait();
      arb.say(convo); arb.say(nav); await wait();
      ok(arb.state().queued.map(i => i.id).join(',') === 'far,answer',
         'queued behind a warning, the nav line goes first and conversation last');
    }

    // -----------------------------------------------------------------------
    // Optional: the same progression checks against a real Google route.
    // -----------------------------------------------------------------------
    const fixture = process.argv[2];
    if (fixture && fs.existsSync(fixture)) {
      section('progression — real route from ' + path.basename(fixture));
      const route = JSON.parse(fs.readFileSync(fixture, 'utf8'));
      const { events } = driveRoute(route, 14);
      const app = events.filter(e => e.type === 'maneuver_approach');
      const nMan = route.maneuvers.length;
      ok(app.length === nMan * 3,
         nMan + ' maneuvers x 3 tiers = ' + (nMan * 3) + ' announcements (got ' + app.length + ')');
      let bad = null;
      for (let i = 1; i < app.length; i++) {
        if (app[i].maneuver < app[i - 1].maneuver) bad = i;
        if (app[i].maneuver === app[i - 1].maneuver &&
            ['far', 'mid', 'near'].indexOf(app[i].tier) <=
            ['far', 'mid', 'near'].indexOf(app[i - 1].tier)) bad = i;
      }
      ok(bad === null, 'real geometry: tiers still run coarse-to-fine in maneuver order');
      ok(events[events.length - 1].type === 'arrived', 'real geometry: the drive ends arrived');
      ok(events.filter(e => e.type === 'reroute').length === 0,
         'driving the route exactly never triggers a reroute');
      console.log('\n  first six announcements:');
      app.slice(0, 6).forEach(e => console.log('    m' + e.maneuver + ' ' + e.tier.padEnd(5) +
        String(Math.round(e.remaining_m)).padStart(5) + ' m  ' +
        String(e.tta_s).padStart(5) + ' s  "' + e.text + '"'));
    }

    console.log('\n' + (failures ? 'FAILED ' + failures + '/' : 'PASSED ') + checks + ' checks');
    process.exit(failures ? 1 : 0);
  })();
}
