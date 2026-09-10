/* nav_drive_replay.js — drive a REAL recorded drive back through the tracker.
 *
 *   node tools/nav_drive_replay.js [session.jsonl] [outage-session.jsonl]
 *
 * tools/nav_selftest.js drives synthetic routes at a constant speed, and every
 * check in it passed on the day navigation called not one turn on a phone. It
 * had to: the thing that was wrong was not in the route, the planner, the
 * arbiter or the validity rules. It was that the tracker only ever emitted
 * NAV_PROGRESS when a GPS fix arrived, and on a real phone the fixes stopped.
 *
 * So this is the other test: the code that ships, driven by a drive that
 * happened, including the part where the radio went quiet.
 *
 * WHAT IS REAL HERE AND WHAT IS RECONSTRUCTED
 * -------------------------------------------
 * Real, straight out of the session JSONL:
 *
 *   the route          every maneuver, its type, direction, road name, its
 *                      position along the route in metres, and the exact
 *                      sentences the server wrote for it — all logged whole in
 *                      NAV_ROUTE_STARTED.
 *   the progress       every nav call carries `to_maneuver_m` against a named
 *                      maneuver, which is a measured position along the route.
 *   the speed          the headway log carries the phone's GPS speed at ~4 Hz
 *                      for the whole drive. 1944 samples on session 06af3214.
 *   the fix timeline   NAV_GPS_STALE carries `no_fix_Ns` and NAV_GPS_OK is
 *                      only ever emitted from a real position(). Between them
 *                      the log says exactly when fixes were and were not
 *                      arriving.
 *
 * Reconstructed, and only these:
 *
 *   the polyline       the log deliberately keeps no coordinate trail (§31),
 *                      so the geometry is rebuilt as a straight line of the
 *                      route's real total length with a vertex every ten
 *                      metres. Maneuvers are pinned by their logged
 *                      `route_distance_position`, which is what navcore reads,
 *                      and the car is driven along the line — so every
 *                      distance, every TTA and every maneuver position is the
 *                      real one. What is lost is the shape of the road, which
 *                      nothing being tested here looks at.
 *   position(t)        the real GPS speed trace, integrated, then rescaled
 *                      segment by segment so it passes exactly through the
 *                      measured along-route positions above.
 *
 * WHAT IS ASSERTED
 * ----------------
 * Audio events. Not states, not internals: the NAV_EARLY_GUIDANCE,
 * NAV_CONTEXTUAL_CALL, NAV_NEAR_TURN and NAV_SPEECH_SPOKEN that a driver would
 * have heard, in order, once each, for every maneuver, ending in an arrival.
 */
'use strict';

const fs = require('fs');
const path = require('path');
const { execFileSync } = require('child_process');

const navcore = require(path.join(__dirname, '..', 'static', 'rio_navcore.js'));
const navplan = require(path.join(__dirname, '..', 'static', 'rio_navplan.js'));
const speech = require(path.join(__dirname, '..', 'static', 'rio_speech.js'));

const TD = path.join(__dirname, '..', 'training_data');
const DEFAULT_DRIVE = path.join(TD, '06af3214-ac34-488e-a13c-3a68ba888888.jsonl');
const DEFAULT_OUTAGE = path.join(TD, 'a2da65cd-6014-4d4e-bb72-aaef93bd2278.jsonl');

let checks = 0, failures = 0;
function ok(cond, what, detail) {
  checks++;
  if (!cond) { failures++; console.log('  FAIL  ' + what + (detail ? '  -- ' + detail : '')); }
  else console.log('  ok    ' + what + (detail ? '  -- ' + detail : ''));
}
function section(name) { console.log('\n=== ' + name + ' ==='); }

// ---------------------------------------------------------------------------
// Reading a session
// ---------------------------------------------------------------------------
function readSession(file) {
  const out = { nav: [], headway: [], t0: null };
  for (const line of fs.readFileSync(file, 'utf8').split('\n')) {
    if (!line.trim()) continue;
    let ev;
    try { ev = JSON.parse(line); } catch (e) { continue; }
    if (out.t0 === null) out.t0 = ev.t;
    if (ev.kind === 'nav') out.nav.push({ wall: ev.t - out.t0, p: ev.payload });
    else if (ev.kind === 'headway') out.headway.push({ wall: ev.t - out.t0, p: ev.payload });
  }
  return out;
}

/* The longest route in a session, with its maneuvers. A drive can carry
   several (this one rerouted twice and was re-destinationed once); the one
   worth replaying is the one that was actually driven to the end. */
function pickRoute(sess) {
  const starts = sess.nav.filter(e => e.p.event === 'NAV_ROUTE_STARTED'
                                 || e.p.event === 'NAV_REROUTE_COMPLETE');
  if (!starts.length) return null;
  const calls = {};
  for (const e of sess.nav) {
    if (e.p.route_id) calls[e.p.route_id] = (calls[e.p.route_id] || 0) + 1;
  }
  starts.sort((a, b) => (calls[b.p.route_id] || 0) - (calls[a.p.route_id] || 0));
  return starts[0].p;
}

// ---------------------------------------------------------------------------
// Rebuilding the route in the wire shape the browser receives
// ---------------------------------------------------------------------------
const M_PER_DEG = 111320;

function buildRoute(logged, timing) {
  const total = logged.total_distance_m;
  const o = logged.origin, d = logged.destination;

  /* A path of the route's REAL LENGTH that ENDS AT THE REAL DESTINATION.
   *
   * Both halves matter and a straight line cannot give both: a road route is
   * always longer than the crow flies, so a straight line of the right length
   * overshoots the destination — by 500 m on this route — and the tracker's
   * arrival test (which asks how far the CAR is from the destination, not how
   * far along the polyline it is) then never fires. That is not a fault in the
   * tracker; it is a fault in a fake road.
   *
   * So: an isoceles path. One bend, placed on the perpendicular bisector of
   * origin-to-destination at the height that makes the two legs sum to exactly
   * the route's length. Densified every ten metres. Every along-route
   * distance, every maneuver position and the destination are then all the
   * real ones, and the only fiction left is the shape of the road, which
   * nothing under test looks at.
   */
  const kLat = M_PER_DEG;
  const kLng = M_PER_DEG * Math.cos(o.lat * Math.PI / 180);
  const toM = p => [(p.lat - o.lat) * kLat, (p.lng - o.lng) * kLng];
  const toLL = p => ({ lat: o.lat + p[0] / kLat, lng: o.lng + p[1] / kLng });
  const D = toM(d);
  const chord = Math.hypot(D[0], D[1]) || 1;
  const half = Math.max(total, chord + 1) / 2;
  const h = Math.sqrt(Math.max(0, half * half - (chord / 2) * (chord / 2)));
  // Unit vectors along and perpendicular to the chord.
  const ux = D[0] / chord, uy = D[1] / chord;
  const apex = [D[0] / 2 - uy * h, D[1] / 2 + ux * h];

  const stepM = 10;
  const geometry = [];
  const leg = (from, to) => {
    const len = Math.hypot(to[0] - from[0], to[1] - from[1]);
    const n = Math.max(1, Math.round(len / stepM));
    for (let i = 1; i <= n; i++) {
      const f = i / n;
      const p = toLL([from[0] + (to[0] - from[0]) * f,
                      from[1] + (to[1] - from[1]) * f]);
      geometry.push([p.lat, p.lng]);
    }
  };
  geometry.push([o.lat, o.lng]);
  leg([0, 0], apex);
  leg(apex, D);
  const n = geometry.length;

  /* THE WORDS ARE REBUILT; NOTHING ELSE IS.
   *
   * The sentences in the log are the ones that were live on the day of the
   * drive, and replaying them would test a cadence that has been replaced.
   * The geometry, the speeds, the fix timeline and every maneuver position
   * stay exactly as recorded; only `speech` is asked for again, from the one
   * module that writes it (tools/nav_respeech.py over a pipe), so the replay
   * cannot drift from what a route built today would actually say. */
  const respoken = respeech(logged);
  const byId = {};
  for (const m of respoken.maneuvers) byId[m.id] = m;

  const maneuvers = logged.maneuvers.map((m, i) => ({
    id: m.id, sequence: m.sequence, type: m.type, direction: m.direction,
    road_name: m.road_name, instruction: m.instruction,
    lat: o.lat, lng: o.lng,
    polyline_index: Math.min(n - 1, Math.round(m.route_distance_position / stepM)),
    route_distance_position: m.route_distance_position,
    road_class: (byId[m.id] || {}).road_class || 'SURFACE',
    anchors: [], speech: (byId[m.id] || {}).speech || m.speech || {}
  }));
  return {
    route_id: logged.route_id, generation_id: logged.generation_id,
    journey_id: logged.journey_id, destination: d, origin: o,
    geometry: geometry, maneuvers: maneuvers,
    depart_speech: respoken.depart_speech,
    total_distance_m: total, eta_epoch: logged.eta_epoch,
    arrival: logged.arrival || { side: 'UNKNOWN' },
    timing: timing
  };
}

/* Ask navigation/speech.py what this route says today. */
function respeech(logged) {
  const payload = {
    route_id: logged.route_id, journey_id: logged.journey_id,
    generation_id: logged.generation_id,
    destination: logged.destination, arrival: logged.arrival,
    total_distance_m: logged.total_distance_m,
    // A log carries no DEPART step. The route-start line then falls back to
    // the first move on its own, which is what depart_text does with no
    // heading sentence to lead with -- and is honest about what was recorded.
    depart_instruction: logged.depart_instruction || '',
    maneuvers: logged.maneuvers,
  };
  const out = execFileSync('python3', ['-m', 'tools.nav_respeech'], {
    cwd: path.join(__dirname, '..'),
    input: JSON.stringify(payload), encoding: 'utf8',
  });
  return JSON.parse(out);
}

// ---------------------------------------------------------------------------
// Rebuilding where the car was, second by second
// ---------------------------------------------------------------------------
/* Measured along-route positions: every nav call names a maneuver and says how
   far the car was from it. maneuver.along_m - to_maneuver_m is a position. */
function alongSamples(sess, route) {
  const byId = {};
  for (const m of route.maneuvers) byId[m.id] = m.route_distance_position;
  const out = [];
  for (const e of sess.nav) {
    const p = e.p;
    if (p.route_id !== route.route_id) continue;
    if (typeof p.t !== 'number') continue;
    if (typeof p.to_maneuver_m === 'number' && byId[p.maneuver_id] !== undefined) {
      out.push({ t: p.t, along: byId[p.maneuver_id] - p.to_maneuver_m,
                 v: typeof p.speed_ms === 'number' ? p.speed_ms : null });
      continue;
    }
    /* A maneuver reported PASSED is also a measured position: the tracker only
       says so once the car is past it. And ARRIVED is the end of the route.
       These are what carry the last leg of a drive, where the calls stop and
       the log would otherwise go quiet -- on this drive m2, m3 and the arrival
       were all reported in one batch at t=600.2 with no call between them,
       which is precisely the stretch the replay has to cover. */
    if (p.event === 'NAV_MANEUVER_PASSED' && byId[p.maneuver_id] !== undefined) {
      out.push({ t: p.t, along: byId[p.maneuver_id] + 1, v: null });
    } else if (p.event === 'NAV_ARRIVED') {
      out.push({ t: p.t, along: route.total_distance_m, v: null });
    }
  }
  out.sort((a, b) => (a.t - b.t) || (a.along - b.along));
  /* One position per instant, and it is the FURTHEST one. Several events land
     on the same tick at the end of a drive -- this one reported m3 passed and
     the destination arrived at the same t=600.248 -- and taking the first
     would leave the car fifty metres short of a route it demonstrably
     finished. */
  const dedup = [];
  for (const s of out) {
    if (dedup.length && Math.abs(dedup[dedup.length - 1].t - s.t) < 1e-6) {
      if (s.along > dedup[dedup.length - 1].along) dedup[dedup.length - 1] = s;
      continue;
    }
    dedup.push(s);
  }
  // Progress is monotonic; a PASSED event landing a metre behind a call that
  // already happened is bookkeeping, not a reversal.
  for (let i = 1; i < dedup.length; i++) {
    if (dedup[i].along < dedup[i - 1].along) dedup[i].along = dedup[i - 1].along;
  }
  return dedup;
}

/* THE REAL SPEED TRACE. The headway log carries the phone's own GPS speed on
   every frame -- 1944 of 1949 frames on this drive -- which is the same fix
   stream the tracker was being given. Integrating it gives the shape of the
   drive: where it accelerated, where it sat at lights. */
function speedTrace(sess) {
  return sess.headway
    .filter(e => typeof e.p.v_host === 'number' && typeof e.p.t === 'number')
    .map(e => ({ t: e.p.t, v: e.p.v_host }))
    .sort((a, b) => a.t - b.t);
}

function speedAt(trace, t) {
  if (!trace.length) return 0;
  let lo = 0, hi = trace.length - 1;
  if (t <= trace[0].t) return trace[0].v;
  if (t >= trace[hi].t) return trace[hi].v;
  while (lo < hi - 1) {
    const mid = (lo + hi) >> 1;
    if (trace[mid].t <= t) lo = mid; else hi = mid;
  }
  const span = trace[hi].t - trace[lo].t;
  const f = span > 0 ? (t - trace[lo].t) / span : 0;
  return trace[lo].v + (trace[hi].v - trace[lo].v) * f;
}

/* Position along the route at time t: the real speed trace, integrated, then
   rescaled inside each measured segment so it lands exactly on the measured
   positions. The shape of the drive is the phone's; the anchors are the
   route's. */
function makePositionFn(samples, trace) {
  const DT = 0.25;
  // Cumulative integral of the speed trace on a fine grid.
  const t0 = samples[0].t, t1 = samples[samples.length - 1].t + 120;
  const grid = [], cum = [];
  let acc = 0;
  for (let t = t0; t <= t1 + DT; t += DT) {
    grid.push(t);
    cum.push(acc);
    acc += speedAt(trace, t) * DT;
  }
  function integral(t) {
    if (t <= grid[0]) return cum[0];
    if (t >= grid[grid.length - 1]) return cum[cum.length - 1];
    const i = Math.min(grid.length - 2, Math.floor((t - t0) / DT));
    const f = (t - grid[i]) / DT;
    return cum[i] + (cum[i + 1] - cum[i]) * f;
  }
  const last = samples[samples.length - 1];
  return function (t) {
    if (t <= samples[0].t) return samples[0].along;
    /* Past the last measured position the car does not teleport and does not
       stop: it keeps going at the speed the phone was reporting. The log's
       last nav event is not the end of the drive, it is the end of the things
       worth logging about it. */
    if (t >= last.t) return last.along + (integral(t) - integral(last.t));
    let i = 0;
    while (i < samples.length - 2 && samples[i + 1].t <= t) i++;
    const a = samples[i], b = samples[i + 1];
    const di = integral(b.t) - integral(a.t);
    const dm = b.along - a.along;
    if (di <= 0.001) {
      const f = (t - a.t) / (b.t - a.t || 1);
      return a.along + dm * f;
    }
    return a.along + dm * (integral(t) - integral(a.t)) / di;
  };
}

// ---------------------------------------------------------------------------
// The fix timeline: when the radio was and was not delivering
// ---------------------------------------------------------------------------
/* Windows during which NO fix reached the tracker, read off the log.
 *
 * NAV_GPS_STALE carries `no_fix_Ns`, so the last fix was N seconds before it.
 * NAV_GPS_OK is emitted from gpsFromFix and from nowhere else, so it marks a
 * fix arriving. Everything between the two is measured silence.
 */
function outageWindows(sess) {
  const out = [];
  let open = null;
  for (const e of sess.nav) {
    const p = e.p;
    if (p.event === 'NAV_GPS_STALE' && typeof p.t === 'number') {
      const m = /no_fix_(\d+)s/.exec(p.reason || '');
      const lastFix = p.t - (m ? parseFloat(m[1]) : 5);
      if (!open) open = { from: lastFix, to: null };
    } else if (p.event === 'NAV_GPS_OK' && typeof p.t === 'number' && open) {
      open.to = p.t;
      out.push(open);
      open = null;
    }
  }
  if (open) { open.to = Infinity; out.push(open); }
  return out;
}

// ---------------------------------------------------------------------------
// The drive
// ---------------------------------------------------------------------------
function replay(route, positionAt, trace, opts) {
  opts = opts || {};
  const outages = opts.outages || [];
  const tStart = opts.tStart, tEnd = opts.tEnd;
  const STEP = 0.5;          // the dashboard's own tick, 2 Hz
  const FIX_EVERY = 1.0;     // what a healthy iOS watch delivers

  const tracker = navcore.create(route, opts.trackerOptions || {});
  const arbiter = speech.makeArbiter();
  const heard = [];
  const events = [];

  const planner = navplan.create({
    tracker: tracker, arbiter: arbiter, route: route,
    verify: null,
    options: { vision_enabled: false },
    /* play() returns NOTHING, deliberately.
     *
     * The arbiter finishes an item synchronously when play() is not thenable
     * and on a microtask when it is. This replay loop is synchronous, so a
     * promise-returning play() would leave every line "currently speaking"
     * until the loop ended -- and each next line in the same nav:<maneuver>
     * group would supersede it. The first version of this harness recorded
     * three announcements for a five-maneuver route for exactly that reason,
     * and the drive underneath it was perfect. */
    audio: function (c) {
      return {
        play: function () {
          heard.push({ t: c.created_at, text: c.text, call: c.call_type,
                       maneuver: c.maneuver_id });
        },
        stop: function () {}
      };
    },
    activeGeneration: function () { return route.generation_id; }
  });
  tracker.onEvent(e => events.push(e));
  planner.onEvent(e => events.push(e));

  function inOutage(t) {
    for (const w of outages) if (t > w.from && t < w.to) return true;
    return false;
  }

  let lastFixT = -1e9;
  let fixes = 0, ticks = 0;
  for (let t = tStart; t <= tEnd; t += STEP) {
    const along = positionAt(t);
    if (!inOutage(t) && (t - lastFixT) >= FIX_EVERY - 1e-9) {
      lastFixT = t;
      fixes++;
      const pt = tracker.pointAt(along);
      if (pt) {
        tracker.position({ lat: pt.lat, lng: pt.lng, t: t,
                           speed: speedAt(trace, t), heading: null,
                           accuracy: 5.0 });
      }
    } else {
      ticks++;
      tracker.tick(t);
    }
    // The arbiter's queue is drained synchronously by say(); nothing here
    // needs a microtask turn because every item's play() resolves at once.
  }
  return { heard, events, tracker, fixes, ticks };
}

function callsFor(heard, id) {
  return heard.filter(h => h.maneuver === id).map(h => h.call);
}

// ---------------------------------------------------------------------------
function main() {
  const driveFile = process.argv[2] || DEFAULT_DRIVE;
  const outageFile = process.argv[3] || DEFAULT_OUTAGE;

  const sess = readSession(driveFile);
  const logged = pickRoute(sess);
  if (!logged) { console.log('no route in ' + driveFile); process.exit(2); }

  // The timing the server would send. Read from the tracker's own defaults so
  // this file holds no numbers, plus the two the punch list added.
  const timing = Object.assign({}, navcore.DEFAULTS, navplan.DEFAULTS,
                               { vision_enabled: false });

  const route = buildRoute(logged, timing);
  const samples = alongSamples(sess, route);
  const trace = speedTrace(sess);

  console.log(path.basename(driveFile));
  console.log('  route ' + route.route_id + ': ' + route.maneuvers.length
              + ' maneuvers over ' + Math.round(route.total_distance_m) + ' m');
  console.log('  ' + samples.length + ' measured along-route positions, '
              + trace.length + ' GPS speed samples');
  if (samples.length < 2) { console.log('  not enough progress in this log'); process.exit(2); }

  const tStart = samples[0].t - 2;
  const tEnd = samples[samples.length - 1].t + 60;
  const positionAt = makePositionFn(samples, trace);

  // -------------------------------------------------------------------------
  section('the recorded drive, radio healthy');
  // -------------------------------------------------------------------------
  const clean = replay(route, positionAt, trace, { tStart, tEnd });
  if (process.env.NAV_DEBUG) {
    console.log('  samples:', JSON.stringify(samples.map(s => [Math.round(s.t), Math.round(s.along)])));
    clean.events.filter(e => e.type !== 'NAV_PROGRESS')
      .forEach(e => console.log('    EV t=' + (e.t || 0).toFixed(1) + ' ' + e.type
                                + ' ' + (e.maneuver_id || '') + ' ' + (e.reason || '')));
    let last = null;
    clean.events.filter(e => e.type === 'NAV_PROGRESS').forEach(e => {
      if (last === null || e.t - last >= 5) {
        last = e.t;
        console.log('    PR t=' + e.t.toFixed(1) + ' along=' + e.along_m
                    + ' man=' + e.maneuver_id + ' to=' + e.to_maneuver_m
                    + ' tta=' + e.tta_s + ' st=' + e.maneuver_state
                    + ' v=' + e.speed_ms + (e.coasted ? ' COAST' : ''));
      }
    });
  }
  const turns = route.maneuvers.filter(m => m.type !== 'ARRIVE' && m.type !== 'DEPART');

  console.log('  ' + clean.fixes + ' fixes, ' + clean.ticks + ' fix-less ticks');
  console.log('  spoken:');
  clean.heard.forEach(h => console.log('    t=' + h.t.toFixed(1) + '  ['
                                       + h.call + '] ' + h.text));

  /* THE INSTRUCTION IS UNCONDITIONAL. The far call is not: a turn 43 m after
     the one before it is already well inside the half-mile mark when it
     becomes the active maneuver, and "in half a mile" said 43 m out is not a
     rounding, it is false. So `far` is required only where the leg leaves room
     for it, which is the planner's own rule (dist > nearAt) stated from
     outside.

     A turn that has been swallowed by the previous one's "then" clause is
     instructed too -- by that sentence -- so a chained maneuver satisfies this
     without a near call of its own. */
  const legs = {};
  let prevAlong = 0;
  for (const m of route.maneuvers) {
    legs[m.id] = m.route_distance_position - prevAlong;
    prevAlong = m.route_distance_position;
  }
  const chainedIds = new Set(route.maneuvers
    .map(m => m.speech && m.speech.chained_to).filter(Boolean));
  const tierOf = (m, call) => {
    const t = ((m.speech || {}).tiers || []).filter(x => x.call === call)[0];
    return t ? t.at_m : null;
  };
  let missing = [];
  for (const m of turns) {
    const c = callsFor(clean.heard, m.id);
    if (!c.includes('near') && !chainedIds.has(m.id)) missing.push(m.id + ':near');
    const farAt = tierOf(m, 'far'), nearAt = tierOf(m, 'near');
    if (!c.includes('far') && !chainedIds.has(m.id)
        && farAt && nearAt && legs[m.id] > farAt) {
      missing.push(m.id + ':far');
    }
  }
  ok(missing.length === 0,
     'every turn on the real route is instructed, and prepared for where there '
     + 'is room — all of it unprompted',
     missing.length ? 'missing ' + missing.join(', ')
       : turns.length + ' turns, '
         + turns.filter(m => legs[m.id] > timing.early_distance_m).length
         + ' of them with room for an early call');

  const arrivals = clean.heard.filter(h => h.call === 'arrival');
  ok(arrivals.length === 1,
     'the drive ends with exactly one arrival announcement',
     arrivals.length ? JSON.stringify(arrivals[0].text) : 'NONE — this is the '
       + 'bug the first real drive found: NAV_ARRIVED fired and nothing spoke');

  const arrivedEvents = clean.events.filter(e => e.type === 'NAV_ARRIVED');
  ok(arrivedEvents.length === 1, 'and the tracker arrived exactly once',
     arrivedEvents.length + ' NAV_ARRIVED');

  // Order: for each maneuver the calls come early -> primary -> imminent.
  let outOfOrder = null;
  const RANK = { early: 0, primary: 1, imminent: 2, arrival: 3 };
  for (const m of route.maneuvers) {
    const c = callsFor(clean.heard, m.id);
    for (let i = 1; i < c.length; i++) {
      if (RANK[c[i]] < RANK[c[i - 1]]) outOfOrder = m.id + ': ' + c.join(',');
    }
  }
  ok(outOfOrder === null, 'each maneuver\'s calls come in order', outOfOrder || '');

  let dupes = null;
  for (const m of route.maneuvers) {
    const c = callsFor(clean.heard, m.id);
    if (new Set(c).size !== c.length) dupes = m.id + ': ' + c.join(',');
  }
  ok(dupes === null, 'nothing is said twice for the same maneuver', dupes || '');

  ok(clean.events.filter(e => e.type === 'NAV_SPEECH_EXPIRED').length === 0
     && clean.events.filter(e => e.type === 'NAV_SPEECH_INVALIDATED').length === 0,
     'no line was queued and then dropped');

  // -------------------------------------------------------------------------
  section('...with the radio behaviour of the drive that called nothing');
  // -------------------------------------------------------------------------
  const bad = readSession(outageFile);
  const windows = outageWindows(bad);
  const longest = windows.reduce((a, w) =>
    Math.max(a, (isFinite(w.to) ? w.to : 0) - w.from), 0);
  console.log('  ' + path.basename(outageFile) + ': ' + windows.length
              + ' measured outage(s), longest ' + longest.toFixed(1) + ' s');
  ok(longest > 60,
     'the recorded outage is real and long', longest.toFixed(1)
       + ' s with no position update, while the page posted headway frames throughout');

  // Placed over the middle of this route's drive, so it lands across turns.
  const mid = (tStart + tEnd) / 2;
  const outages = [{ from: mid - longest / 2, to: mid + longest / 2 }];

  const coasting = replay(route, positionAt, trace, { tStart, tEnd, outages });
  const silent = replay(route, positionAt, trace, {
    tStart, tEnd, outages,
    // The tracker as it was: no dead reckoning at all, so tick() can change
    // the GPS state and nothing else.
    trackerOptions: { gps_coast_max_s: 0 }
  });

  const spokenDuring = h => h.filter(x => x.t > outages[0].from && x.t < outages[0].to);
  const during = spokenDuring(coasting.heard);
  const duringBefore = spokenDuring(silent.heard);

  console.log('  during the outage, coasting:');
  during.forEach(h => console.log('    t=' + h.t.toFixed(1) + '  [' + h.call + '] ' + h.text));
  console.log('  during the outage, as it was: '
              + (duringBefore.length ? '' : '(nothing)'));
  duringBefore.forEach(h => console.log('    t=' + h.t.toFixed(1) + '  ['
                                        + h.call + '] ' + h.text));

  ok(duringBefore.length === 0,
     'the tracker as it was says NOTHING for the whole outage — the bug, reproduced',
     longest.toFixed(0) + ' s of silence');
  ok(during.length > 0,
     'the tracker that dead-reckons goes on calling turns through it',
     during.length + ' announcement(s) the driver would otherwise not have had');

  /* AND EVERY CALL COMES NO LATER THAN IT WOULD HAVE.
   *
   * The count is now the same both ways, and that is a property of the fixed
   * distance ladder rather than a regression: a tracker that says nothing
   * through an outage still makes the near call for that maneuver once fixes
   * return -- it just makes it from inside the junction. So the thing worth
   * asserting is WHEN, not whether. Every announcement the coasting drive
   * makes lands at the same moment or earlier than the silent one's. */
  const whenOf = h => {
    const out = {};
    for (const x of h) out[x.maneuver + ':' + x.call] = x.t;
    return out;
  };
  const cw = whenOf(coasting.heard), sw = whenOf(silent.heard);
  const lost = Object.keys(sw).filter(k => cw[k] === undefined);
  const later = Object.keys(sw).filter(k => cw[k] !== undefined && cw[k] > sw[k] + 0.6);
  const earlier = Object.keys(sw).filter(k => cw[k] !== undefined && cw[k] < sw[k] - 0.6);
  ok(lost.length === 0 && later.length === 0 && earlier.length > 0,
     'and every call comes no later than it would have — some of them much earlier',
     earlier.map(k => k + ' ' + (sw[k] - cw[k]).toFixed(0) + ' s earlier').join(', ')
       + (lost.length ? '  LOST: ' + lost.join(', ') : '')
       + (later.length ? '  LATER: ' + later.join(', ') : ''));

  // The one thing coasting may never do.
  const coastedPass = coasting.events.filter(
    e => e.type === 'NAV_MANEUVER_PASSED' && e.coasted);
  ok(coastedPass.length === 0,
     'no maneuver is ever passed on dead-reckoned progress');
  const coastedProgress = coasting.events.filter(
    e => e.type === 'NAV_PROGRESS' && e.coasted);
  ok(coastedProgress.length > 0,
     'the coasted progress events are marked as such in the log',
     coastedProgress.length + ' of them');
  ok(coastedProgress.every(e => e.maneuver_state !== 'EXECUTING'
                             && e.maneuver_state !== 'PASSED'),
     'a coasted tick never claims the car is AT the junction');

  // -------------------------------------------------------------------------
  section('the cadence table — every call against the ladder it was due on');
  // -------------------------------------------------------------------------
  /* WHAT THE PUNCH LIST ASKED FOR, and what the drive of 2026-09-09 could not
     produce: each call next to the distance Google would have made it at for
     that road class. `at_m` is the tier the planner fired -- it comes off the
     route, which the server built -- and `to_maneuver_m` is where the car
     actually was. The two agree to within a tick of travel when the ladder is
     being followed and diverge visibly when it is not.

     A row with no `at_m` is a route-start line, which has no distance by
     definition: it is said before the approach exists. */
  const CALLS = ['depart', 'far', 'far_mid', 'near', 'junction', 'arrival', 'arrived'];
  /* One row per TIER EVENT. NAV_SPEECH_SPOKEN carries the same sentence a
     moment later, from the arbiter rather than the planner, and it has no
     `at_m` -- so the tier events are the ones with a distance to compare. */
  const rows = clean.events.filter(e => CALLS.indexOf(e.call_type) >= 0
                                   && typeof e.at_m === 'number'
                                   && (e.text || e.skipped));
  const legAt = {};
  let prevPos = 0;
  for (const m of route.maneuvers) {
    legAt[m.id] = m.route_distance_position - prevPos;
    prevPos = m.route_distance_position;
  }
  console.log('  man   class     call      google  actual   delta     leg   line');
  console.log('  ' + '-'.repeat(94));
  for (const e of rows) {
    const at = e.at_m;
    const got = typeof e.to_maneuver_m === 'number' ? e.to_maneuver_m : null;
    const delta = got === null ? null : (got - at);
    console.log('  ' + String(e.maneuver_id || '').padEnd(5)
      + ' ' + String(e.road_class || '-').padEnd(9)
      + ' ' + String(e.call_type).padEnd(9)
      + ' ' + (at.toFixed(0) + ' m').padStart(6)
      + ' ' + (got === null ? '     —' : (got.toFixed(0) + ' m').padStart(6))
      + ' ' + (delta === null ? '      —'
               : ((delta >= 0 ? '+' : '') + delta.toFixed(0) + ' m').padStart(7))
      + ' ' + ((legAt[e.maneuver_id] || 0).toFixed(0) + ' m').padStart(7)
      + '   ' + (e.text || ('(skipped: ' + e.skipped + ')')));
  }

  /* ONE TICK OF TRAVEL is the whole of the allowance.
   *
   * This subsumes the two distance-floor checks that used to live below it,
   * and it is a stricter statement of the same two facts. The recorded drive
   * gave an instruction at 47 m, from inside the junction, and the acceptance
   * pass of 2026-09-09 measured "Left here." landing at 24-29 m against a 35 m
   * floor -- one to two ticks late, on the one line whose entire worth is
   * WHEN it arrives. A ladder makes both of those a comparison against a
   * number the route itself carries rather than against a constant this file
   * would have to keep in step.
   *
   * The exemption is stated rather than absorbed: a maneuver whose LEG is shorter than its own
     tier does not become the active maneuver until the car is already inside
     that tier, so "landed at the tier" is not a thing that can be true of it.
     m3 on this route is 43 m after m2. Demanding it would be demanding a
     fiction; the leg column is in the table so a reader can see which rows
     that is and why. */
  const tick = r => Math.max(15, (r.speed_ms || 0) * 2);
  const measured = rows.filter(r => r.text && typeof r.to_maneuver_m === 'number');
  const judged = measured.filter(r => (legAt[r.maneuver_id] || 0) >= r.at_m + tick(r));
  const late = judged.filter(r => r.to_maneuver_m < r.at_m - tick(r));
  const worstOvershoot = judged.length
    ? Math.min(...judged.map(r => r.to_maneuver_m - r.at_m)) : 0;
  ok(judged.length > 0 && late.length === 0,
     'every call on a leg with room for it landed within one tick of the '
     + 'distance Google would have used',
     judged.length + ' of ' + measured.length + ' call(s) judged, worst '
       + 'overshoot ' + worstOvershoot.toFixed(0) + ' m'
       + (late.length ? ' — LATE: ' + late.map(r => r.call_type
           + ' ' + r.to_maneuver_m.toFixed(0) + '/' + r.at_m.toFixed(0)).join(', ') : ''));

  /* ...AND NO CALL IS EVER MADE FROM FURTHER OUT THAN ITS TIER. Early is the
     harmless direction for a warning and the wrong one for an instruction: a
     turn called before the driver can see the junction is a turn they will
     take at the wrong one. This is the check the old cadence could not have
     passed -- its early call fired on whichever of a time term and a metre
     floor came first, and on session 738fbb82 that was after the primary. */
  const early = measured.filter(r => r.to_maneuver_m > r.at_m + tick(r));
  ok(early.length === 0, 'and none from further out than its tier',
     early.length ? early.map(r => r.call_type + ' '
       + r.to_maneuver_m.toFixed(0) + '/' + r.at_m.toFixed(0)).join(', ')
       : measured.length + ' call(s) checked');

  /* THE ORDER, which is the failure the drive of 2026-09-09 actually had:
     "Coming up on a right" arrived AFTER "Take the next right" and closer to
     the junction. On a ladder that cannot happen, and this says so. */
  const backwards = [];
  const seenAt = {};
  for (const r of rows) {
    if (!r.text) continue;
    const k = r.maneuver_id;
    if (seenAt[k] !== undefined && typeof r.to_maneuver_m === 'number'
        && r.to_maneuver_m > seenAt[k] + 1) {
      backwards.push(k + ':' + r.call_type);
    }
    if (typeof r.to_maneuver_m === 'number') seenAt[k] = r.to_maneuver_m;
  }
  ok(backwards.length === 0,
     'and each maneuver\'s calls come strictly closer to it, never back out',
     backwards.length ? backwards.join(', ') : 'in order for every maneuver');

  // -------------------------------------------------------------------------
  section('the other half of the fix: the watch that watches the watch');
  // -------------------------------------------------------------------------
  /* The dead reckoning above SURVIVES a gap in the fixes. It does not close
     one, and it is bounded at twenty seconds because a guess with nothing
     under it is worse than silence -- so on the 212 s outage it covers the
     first twenty and no more. What recovers the radio is the watchdog in
     index.html, which tears the geolocation subscription down and rebuilds it.
     That lives in an inline script with a real navigator behind it and cannot
     be require()d here, so what this checks is that it is still wired: a
     source assertion, and it says so.

     The old page had none of this. One watchPosition for the life of the
     page, and an error callback that set a variable and returned. */
  const page = fs.readFileSync(path.join(__dirname, '..', 'static', 'index.html'), 'utf8');
  const glue = fs.readFileSync(path.join(__dirname, '..', 'static', 'rio_nav.js'), 'utf8');
  ok(/function\s+rearmWatch/.test(page) && /clearWatch/.test(page)
     && /armWatch\(coarse\)/.test(page),
     'the page tears the watch down and rebuilds it rather than re-asking it');
  ok(/startWatchdog/.test(page) && /watchCfg\.watchdog_s/.test(page),
     'a timer watches for the absence of fixes');
  ok(/noteGps\('gps_watch_rearm'/.test(page) && /noteGps\('gps_error'/.test(page),
     'every re-arm and every geolocation error goes into the drive log',
     'the drive that called no turns left no trace of why');
  ok(/enableHighAccuracy:\s*!coarse/.test(page),
     'and high accuracy is dropped after repeated failures');
  ok(/configureWatch\(r\.timing\)/.test(glue),
     'its numbers arrive with the route, like every other navigation timing');

  console.log('\n' + (failures ? 'FAILED ' + failures + '/' : 'PASSED ') + checks + ' checks');
  process.exit(failures ? 1 : 0);
}

main();
