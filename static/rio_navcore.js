/* rio_navcore.js — the deterministic route tracker. RIO's half of navigation.
 *
 * The provider says where the turn is. This file says where the CAR is, at
 * ~1 Hz, with no network, and everything it emits is derived from route
 * geometry and GPS. Nothing in here has ever heard of a camera.
 *
 * Why client-side, still. A turn announcement is worth nothing late. Putting a
 * round trip between "you are four seconds from the turn" and RIO saying so
 * makes the announcement's timing the network's timing. Tracking is also the
 * one part of navigation that must keep working when signal drops: the route
 * is already in hand, and following it needs no help.
 *
 * TWO STATE MACHINES, DELIBERATELY SEPARATE
 * -----------------------------------------
 *   maneuver:  UPCOMING -> APPROACHING -> IMMINENT -> EXECUTING -> PASSED
 *   GPS:       GPS_OK | GPS_DEGRADED | GPS_STALE
 *   route:     ON_ROUTE | OFF_ROUTE_CANDIDATE | OFF_ROUTE_CONFIRMED
 *
 * The maneuver machine has no VISUAL_CONTEXT state and never will. Visual
 * context has its own lifecycle in rio_navplan.js and the two run
 * independently — maneuver_state APPROACHING alongside context_state ACQUIRING
 * is the normal case, and collapsing them into one enum is how a camera
 * failure starts being able to stall a turn instruction.
 *
 * GPS health is likewise NOT off-route. A stale fix means we no longer know
 * where the car is; it does not mean the car left the route, and it must never
 * cause a reroute. Rerouting because the sky went quiet under a bridge is how
 * a navigation system loses a driver's trust in one move.
 *
 * No DOM, no fetch, no audio. It takes positions and emits events, which is
 * what lets tools/nav_selftest.js drive an entire simulated journey under node
 * against the code that ships.
 */
(function (root) {
  'use strict';

  var RAD = Math.PI / 180;
  var M_PER_DEG = 111320;

  var E = {
    MANEUVER_SELECTED: 'NAV_MANEUVER_SELECTED',
    MANEUVER_PASSED: 'NAV_MANEUVER_PASSED',
    GPS_OK: 'NAV_GPS_OK',
    GPS_DEGRADED: 'NAV_GPS_DEGRADED',
    GPS_STALE: 'NAV_GPS_STALE',
    OFF_ROUTE_CANDIDATE: 'NAV_OFF_ROUTE_CANDIDATE',
    OFF_ROUTE_CONFIRMED: 'NAV_OFF_ROUTE_CONFIRMED',
    ARRIVED: 'NAV_ARRIVED',
    PROGRESS: 'NAV_PROGRESS'
  };

  /* Fallbacks only. The real values arrive with the route, in `timing`, so the
     browser holds no navigation policy of its own — tuning happens in
     config.py and reaches the car with the next route. */
  var DEFAULTS = {
    gps_stale_timeout_s: 5.0,
    gps_accuracy_limit_m: 30.0,
    gps_degraded_bias_s: 2.0,
    off_route_distance_m: 45.0,
    off_route_persistence: 3,
    reroute_debounce_s: 12.0,
    progress_rewind_tolerance_m: 30.0,
    maneuver_passed_eps_m: 8.0,
    arrive_radius_m: 25.0,
    projection_back_m: 80.0,
    projection_fwd_m: 400.0,
    heading_min_displacement_m: 8.0,
    heading_max_sample_age_s: 3.0,
    heading_min_speed_ms: 1.5,
    stationary_speed_ms: 0.7,
    early_guidance_s: 25.0,
    near_turn_s: 2.5,
    speed_floor_ms: 3.0,
    speed_nominal_ms: 11.0
  };

  var GPS_OK = 'GPS_OK', GPS_DEGRADED = 'GPS_DEGRADED', GPS_STALE = 'GPS_STALE';
  var ON_ROUTE = 'ON_ROUTE', OFF_ROUTE_CANDIDATE = 'OFF_ROUTE_CANDIDATE',
      OFF_ROUTE_CONFIRMED = 'OFF_ROUTE_CONFIRMED';
  var UPCOMING = 'UPCOMING', APPROACHING = 'APPROACHING', IMMINENT = 'IMMINENT',
      EXECUTING = 'EXECUTING', PASSED = 'PASSED';
  var MAN_RANK = { UPCOMING: 0, APPROACHING: 1, IMMINENT: 2, EXECUTING: 3, PASSED: 4 };

  function haversineM(aLat, aLng, bLat, bLng) {
    var p1 = aLat * RAD, p2 = bLat * RAD;
    var dp = p2 - p1, dl = (bLng - aLng) * RAD;
    var h = Math.sin(dp / 2) * Math.sin(dp / 2) +
            Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) * Math.sin(dl / 2);
    return 2 * 6371008.8 * Math.asin(Math.min(1, Math.sqrt(h)));
  }

  function bearingDeg(aLat, aLng, bLat, bLng) {
    var p1 = aLat * RAD, p2 = bLat * RAD, dl = (bLng - aLng) * RAD;
    var y = Math.sin(dl) * Math.cos(p2);
    var x = Math.cos(p1) * Math.sin(p2) - Math.sin(p1) * Math.cos(p2) * Math.cos(dl);
    return (Math.atan2(y, x) / RAD + 360) % 360;
  }

  function cumulative(points) {
    var cum = [0];
    for (var i = 1; i < points.length; i++) {
      cum[i] = cum[i - 1] + haversineM(points[i - 1][0], points[i - 1][1],
                                       points[i][0], points[i][1]);
    }
    return cum;
  }

  function create(route, options) {
    var opt = {}, k;
    for (k in DEFAULTS) opt[k] = DEFAULTS[k];
    for (k in (route && route.timing) || {}) if (opt[k] !== undefined) opt[k] = route.timing[k];
    for (k in (options || {})) opt[k] = options[k];

    var points = route.geometry || [];
    var cum = cumulative(points);
    var routeLen = cum.length ? cum[cum.length - 1] : 0;

    /* A maneuver's along-route position is read off the vertex the provider
       pinned it to, never recomputed from its lat/lng: two junctions 20 m apart
       would otherwise be indistinguishable to a nearest-point search, and the
       tracker would announce the wrong one. */
    var mans = (route.maneuvers || []).map(function (m) {
      var idx = Math.max(0, Math.min(points.length - 1, m.polyline_index || 0));
      return {
        id: m.id, sequence: m.sequence, type: m.type, direction: m.direction,
        road_name: m.road_name, instruction: m.instruction,
        lat: m.lat, lng: m.lng, polyline_index: idx,
        along_m: (typeof m.route_distance_position === 'number')
                 ? m.route_distance_position : cum[idx],
        anchors: m.anchors || [], speech: m.speech || {},
        state: UPCOMING
      };
    });

    var listeners = [];
    var vertex = 0;              // last projected vertex — the search anchor
    var along = 0;               // route progress, metres. Monotonic.
    var manIdx = 0;
    var lastFix = null;          // {lat,lng,t,accuracy}
    var headingFix = null;       // last fix far enough back to derive a heading from
    var gpsState = GPS_OK, gpsSince = 0, lastFixT = null;
    var routeState = ON_ROUTE, offRun = 0, lastOffM = 0;
    var arrived = false, stopped = false, firstFix = true;
    var speed = null, speedSource = 'none';
    var heading = null, headingSource = 'none';
    var lastSelected = null;
    var nowT = 0;

    function emit(type, payload) {
      var ev = { t: nowT, route_id: route.route_id, generation_id: route.generation_id };
      for (var kk in (payload || {})) ev[kk] = payload[kk];
      ev.type = type;      // last, so no payload key can shadow the event kind
      for (var i = 0; i < listeners.length; i++) {
        try { listeners[i](ev); } catch (e) { /* a listener must not stop the drive */ }
      }
    }

    function manPublic(m, extra) {
      var out = {
        maneuver_id: m.id, sequence: m.sequence, maneuver_type: m.type,
        direction: m.direction, road_name: m.road_name, instruction: m.instruction,
        maneuver_state: m.state, along_m: Math.round(m.along_m * 10) / 10,
        n_anchor_candidates: (m.anchors || []).length
      };
      for (var kk in (extra || {})) out[kk] = extra[kk];
      return out;
    }

    /* Nearest point on the route, searched in a window around where we already
       are. Local flat-earth metres: over a few hundred metres the error is
       centimetres and the inner loop stays arithmetic. The window is what stops
       a route that crosses itself from snapping the car onto a later leg. */
    function project(lat, lng, full) {
      var kLng = M_PER_DEG * Math.cos(lat * RAD);
      var lo = 0, hi = points.length - 2;
      if (!full) {
        lo = vertex;
        while (lo > 0 && cum[vertex] - cum[lo] < opt.projection_back_m) lo--;
        hi = vertex;
        var fwd = Math.max(opt.projection_fwd_m, (speed || 0) * 30);
        while (hi < points.length - 2 && cum[hi] - cum[vertex] < fwd) hi++;
      }
      var best = null;
      for (var i = lo; i <= hi; i++) {
        var ax = (points[i][1] - lng) * kLng, ay = (points[i][0] - lat) * M_PER_DEG;
        var bx = (points[i + 1][1] - lng) * kLng, by = (points[i + 1][0] - lat) * M_PER_DEG;
        var dx = bx - ax, dy = by - ay;
        var len2 = dx * dx + dy * dy;
        var t = len2 > 0 ? -(ax * dx + ay * dy) / len2 : 0;
        t = t < 0 ? 0 : (t > 1 ? 1 : t);
        var px = ax + t * dx, py = ay + t * dy;
        var d2 = px * px + py * py;
        if (!best || d2 < best.d2) {
          best = { d2: d2, i: i, along: cum[i] + t * Math.sqrt(len2) };
        }
      }
      if (!best) return { vertex: 0, along: 0, offM: 0 };
      return { vertex: best.i, along: best.along, offM: Math.sqrt(best.d2) };
    }

    /* --- GPS health ------------------------------------------------------
       Three states, and none of them is an opinion about the route. */
    function setGps(next, why) {
      if (next === gpsState) return;
      gpsState = next;
      gpsSince = nowT;
      emit(next === GPS_OK ? E.GPS_OK : (next === GPS_STALE ? E.GPS_STALE : E.GPS_DEGRADED),
           { gps_state: next, reason: why || null,
             accuracy_m: lastFix ? lastFix.accuracy : null });
    }

    function gpsFromFix(fix) {
      var acc = (typeof fix.accuracy === 'number' && isFinite(fix.accuracy))
                ? fix.accuracy : null;
      if (acc !== null && acc > opt.gps_accuracy_limit_m) {
        setGps(GPS_DEGRADED, 'accuracy_' + Math.round(acc) + 'm');
      } else {
        setGps(GPS_OK, null);
      }
    }

    /* --- speed and heading ------------------------------------------------
       Browser Geolocation on iOS routinely reports speed: null and heading:
       null, so both are derived when they are missing — and only when the
       derivation actually means something. A heading computed from two fixes
       3 m apart while parked is not a heading, it is noise with a compass
       rose drawn on it. */
    function speedFrom(fix) {
      if (typeof fix.speed === 'number' && isFinite(fix.speed) && fix.speed >= 0) {
        return { v: fix.speed, src: fix.speedSource || 'fix' };
      }
      if (lastFix && typeof fix.t === 'number' && typeof lastFix.t === 'number') {
        var dt = fix.t - lastFix.t;
        if (dt > 0.2 && dt < 10) {
          return { v: haversineM(lastFix.lat, lastFix.lng, fix.lat, fix.lng) / dt,
                   src: 'derived' };
        }
      }
      return { v: opt.speed_nominal_ms, src: 'nominal' };
    }

    function headingFrom(fix, v) {
      if (typeof fix.heading === 'number' && isFinite(fix.heading) && fix.heading >= 0) {
        headingFix = { lat: fix.lat, lng: fix.lng, t: fix.t };
        return { h: fix.heading, src: 'fix' };
      }
      if (!headingFix) {
        headingFix = { lat: fix.lat, lng: fix.lng, t: fix.t };
        return { h: heading, src: heading === null ? 'none' : headingSource };
      }
      var dt = fix.t - headingFix.t;
      var d = haversineM(headingFix.lat, headingFix.lng, fix.lat, fix.lng);
      var accOk = !(typeof fix.accuracy === 'number' && isFinite(fix.accuracy)) ||
                  fix.accuracy <= opt.gps_accuracy_limit_m;
      if (dt > 0 && dt <= opt.heading_max_sample_age_s &&
          d >= opt.heading_min_displacement_m &&
          v >= opt.heading_min_speed_ms && v > opt.stationary_speed_ms && accOk) {
        var h = bearingDeg(headingFix.lat, headingFix.lng, fix.lat, fix.lng);
        headingFix = { lat: fix.lat, lng: fix.lng, t: fix.t };
        return { h: h, src: 'derived' };
      }
      if (dt > opt.heading_max_sample_age_s) {
        // The reference sample went stale without producing a heading; start
        // again from here rather than deriving from an old position.
        headingFix = { lat: fix.lat, lng: fix.lng, t: fix.t };
      }
      return { h: heading, src: heading === null ? 'none' : headingSource };
    }

    /* --- maneuver state --------------------------------------------------
       Time to the maneuver decides the state, with a distance floor so a
       crawling car still reaches IMMINENT at the junction. States never run
       backwards for a maneuver: a tracker that can un-imminent a turn is a
       tracker that can announce it twice. */
    function stateFor(remaining, tta) {
      if (remaining <= opt.maneuver_passed_eps_m) return EXECUTING;
      var nearS = opt.near_turn_s + (gpsState === GPS_DEGRADED ? opt.gps_degraded_bias_s : 0);
      var earlyS = opt.early_guidance_s + (gpsState === GPS_DEGRADED ? opt.gps_degraded_bias_s : 0);
      if (tta <= nearS) return IMMINENT;
      if (tta <= earlyS) return APPROACHING;
      return UPCOMING;
    }

    function selectManeuver() {
      var m = manIdx < mans.length ? mans[manIdx] : null;
      if (m && lastSelected !== m.id) {
        lastSelected = m.id;
        emit(E.MANEUVER_SELECTED, manPublic(m, {
          remaining_maneuvers: mans.length - manIdx,
          anchor_candidates: (m.anchors || []).map(function (a) {
            return { anchor_id: a.anchor_id, label: a.label, relation: a.relation,
                     relation_confidence: a.relation_confidence };
          })
        }));
      }
      return m;
    }

    function api() {
      return {
        route: route,
        events: E,

        onEvent: function (fn) { if (typeof fn === 'function') listeners.push(fn); },

        /* Clock-only update. GPS staleness is the absence of fixes, so it can
           only be noticed by something that runs when nothing arrives. The
           dashboard ticks this at 1 Hz. */
        tick: function (t) {
          if (typeof t === 'number') nowT = t;
          if (stopped || arrived) return gpsState;
          if (lastFixT !== null && (nowT - lastFixT) > opt.gps_stale_timeout_s) {
            setGps(GPS_STALE, 'no_fix_' + Math.round(nowT - lastFixT) + 's');
          }
          return gpsState;
        },

        /* One position fix: {lat, lng, t (seconds), speed?, heading?, accuracy?} */
        position: function (fix) {
          if (stopped || arrived || !points.length || !fix) return null;
          nowT = (typeof fix.t === 'number') ? fix.t : nowT;
          lastFixT = nowT;

          gpsFromFix(fix);
          var sp = speedFrom(fix);
          speed = sp.v; speedSource = sp.src;
          var hd = headingFrom(fix, sp.v);
          heading = hd.h; headingSource = hd.src;

          var pr = project(fix.lat, fix.lng, firstFix);

          /* Progress is monotonic under noise. A fix that projects behind
             where we have already been is jitter unless it is far enough back
             to be a real reversal — and a real wrong turn shows up as
             off-route, not as rewind. Only a confirmed reroute resets
             progress, and it does so by replacing the whole tracker. */
          var rewound = false;
          if (!firstFix && pr.along < along - opt.progress_rewind_tolerance_m) {
            pr.along = along; pr.vertex = vertex; rewound = true;
          } else if (!firstFix && pr.along < along) {
            pr.along = along; pr.vertex = vertex;
          }
          firstFix = false;
          vertex = pr.vertex;
          along = pr.along;
          lastOffM = pr.offM;
          lastFix = { lat: fix.lat, lng: fix.lng, t: nowT,
                      accuracy: (typeof fix.accuracy === 'number') ? fix.accuracy : null };

          /* --- off route ---------------------------------------------------
             Distance from the polyline plus persistence, and nothing else. A
             deviation smaller than the fix's own stated uncertainty is not
             evidence of anything, so it does not count towards the run. */
          var acc = lastFix.accuracy;
          var credible = pr.offM > opt.off_route_distance_m &&
                         (acc === null || pr.offM > acc);
          if (credible) {
            offRun++;
            if (offRun >= opt.off_route_persistence) {
              if (routeState !== OFF_ROUTE_CONFIRMED) {
                routeState = OFF_ROUTE_CONFIRMED;
                stopped = true;   // stop announcing a route we know is wrong
                emit(E.OFF_ROUTE_CONFIRMED, {
                  off_route_m: Math.round(pr.offM * 10) / 10,
                  fixes: offRun, lat: fix.lat, lng: fix.lng,
                  from_maneuver_id: manIdx < mans.length ? mans[manIdx].id : null
                });
              }
              return { off_route: true, route_state: routeState };
            }
            if (routeState === ON_ROUTE) {
              routeState = OFF_ROUTE_CANDIDATE;
              emit(E.OFF_ROUTE_CANDIDATE, {
                off_route_m: Math.round(pr.offM * 10) / 10, fixes: offRun,
                accuracy_m: acc
              });
            }
          } else {
            offRun = 0;
            routeState = ON_ROUTE;
          }

          /* --- maneuvers now behind us ------------------------------------ */
          while (manIdx < mans.length &&
                 along > mans[manIdx].along_m + opt.maneuver_passed_eps_m) {
            var done = mans[manIdx];
            done.state = PASSED;
            manIdx++;
            lastSelected = null;
            if (done.type !== 'ARRIVE') {
              emit(E.MANEUVER_PASSED, manPublic(done, {
                remaining_maneuvers: mans.length - manIdx
              }));
            }
          }

          /* --- arrival ----------------------------------------------------- */
          var toEnd = routeLen - along;
          var destGap = haversineM(fix.lat, fix.lng,
                                   route.destination.lat, route.destination.lng);
          if (manIdx >= mans.length ||
              (toEnd <= opt.arrive_radius_m && destGap <= opt.arrive_radius_m * 4)) {
            arrived = true;
            emit(E.ARRIVED, {
              destination: route.destination,
              arrival_side: (route.arrival && route.arrival.side) || 'UNKNOWN',
              gap_m: Math.round(destGap * 10) / 10
            });
            return { arrived: true };
          }

          var man = selectManeuver();
          var remaining = man.along_m - along;
          var vEff = Math.max(opt.speed_floor_ms, sp.v);
          var tta = remaining / vEff;
          var next = stateFor(remaining, tta);
          if (MAN_RANK[next] > MAN_RANK[man.state]) man.state = next;

          emit(E.PROGRESS, {
            along_m: Math.round(along * 10) / 10,
            remaining_m: Math.round((routeLen - along) * 10) / 10,
            off_route_m: Math.round(pr.offM * 10) / 10,
            route_state: routeState, gps_state: gpsState,
            maneuver_id: man.id, maneuver_type: man.type, direction: man.direction,
            instruction: man.instruction, road_name: man.road_name,
            maneuver_state: man.state,
            to_maneuver_m: Math.round(remaining * 10) / 10,
            tta_s: Math.round(tta * 10) / 10,
            speed_ms: Math.round(sp.v * 100) / 100, speed_source: sp.src,
            heading_deg: heading === null ? null : Math.round(heading * 10) / 10,
            heading_source: headingSource,
            rewound: rewound,
            eta_epoch: route.eta_epoch
          });

          return { along_m: along, maneuver: man, to_maneuver_m: remaining,
                   tta_s: tta, maneuver_state: man.state, gps_state: gpsState,
                   route_state: routeState };
        },

        /* Position on the geometry at a given distance along it — the
           simulator's only geometry, kept here so a simulated drive and a real
           one project through exactly the same points. */
        pointAt: function (m) {
          if (!points.length) return null;
          if (m <= 0) return { lat: points[0][0], lng: points[0][1], done: false };
          if (m >= routeLen) {
            var e = points[points.length - 1];
            return { lat: e[0], lng: e[1], done: true };
          }
          var lo = 0, hi = cum.length - 1;
          while (lo < hi - 1) {
            var mid = (lo + hi) >> 1;
            if (cum[mid] <= m) lo = mid; else hi = mid;
          }
          var seg = cum[hi] - cum[lo];
          var f = seg > 0 ? (m - cum[lo]) / seg : 0;
          return {
            lat: points[lo][0] + (points[hi][0] - points[lo][0]) * f,
            lng: points[lo][1] + (points[hi][1] - points[lo][1]) * f,
            done: false
          };
        },

        maneuver: function () { return manIdx < mans.length ? mans[manIdx] : null; },

        /* Every maneuver still ahead of the car, in order.
         *
         * state() answers "what is the NEXT turn", which is what an
         * announcement needs and all it needs. A driver asking "what are the
         * directions" is asking a different question, and until this existed
         * the only honest answer was that RIO could not read them: the list
         * was in here the whole time and nothing exposed it.
         *
         * Distances are measured from where the car is NOW, not from the start
         * of the route, because the question is always asked mid-drive. `leg_m`
         * is the gap from the previous maneuver in this list, which is what
         * turns a list of turns into a set of directions -- "then left on
         * Sunset, then straight for six miles".
         *
         * `count` omitted or null gives the whole remaining route.
         */
        upcoming: function (count) {
          var out = [];
          var limit = (count === undefined || count === null)
            ? mans.length : Math.max(0, count);
          var prevAlong = along;
          for (var i = manIdx; i < mans.length && out.length < limit; i++) {
            var m = mans[i];
            out.push(manPublic(m, {
              // Ahead of the car. Clamped at zero: the current maneuver can be
              // a metre behind the projection while still being the one we are
              // driving into, and a negative distance in an answer is worse
              // than a rounded-down one.
              distance_m: Math.max(0, Math.round(m.along_m - along)),
              leg_m: Math.max(0, Math.round(m.along_m - prevAlong)),
              // The map's landmark candidates for this turn, unfiltered and
              // unranked here -- whoever reads them decides what to say. Kept
              // whole rather than reduced to a label, because the relation is
              // what makes the sentence true.
              anchors: m.anchors || []
            }));
            prevAlong = m.along_m;
          }
          return out;
        },
        maneuverById: function (id) {
          for (var i = 0; i < mans.length; i++) if (mans[i].id === id) return mans[i];
          return null;
        },
        /* Speech validity asks this, at dequeue, about a maneuver that may
           have been passed while the line sat in the queue (§25). */
        isPassed: function (id) {
          var m = this.maneuverById(id);
          return !m || m.state === PASSED;
        },

        state: function () {
          var man = manIdx < mans.length ? mans[manIdx] : null;
          var remaining = man ? man.along_m - along : 0;
          var vEff = Math.max(opt.speed_floor_ms, speed || 0);
          return {
            route_id: route.route_id, generation_id: route.generation_id,
            destination: route.destination, eta_epoch: route.eta_epoch,
            along_m: along, route_length_m: routeLen,
            remaining_m: Math.max(0, routeLen - along),
            maneuver: man ? manPublic(man) : null,
            to_maneuver_m: man ? remaining : null,
            tta_s: man ? remaining / vEff : null,
            maneuvers_left: Math.max(0, mans.length - manIdx),
            maneuver_state: man ? man.state : null,
            gps_state: gpsState, gps_state_since: gpsSince,
            route_state: routeState, off_route_m: lastOffM,
            speed_ms: speed, speed_source: speedSource,
            heading_deg: heading, heading_source: headingSource,
            arrived: arrived, stopped: stopped
          };
        },

        routeLength: function () { return routeLen; },
        stop: function () { stopped = true; }
      };
    }

    return api();
  }

  root.RIO = root.RIO || {};
  root.RIO.navcore = {
    create: create, haversineM: haversineM, bearingDeg: bearingDeg,
    DEFAULTS: DEFAULTS, EVENTS: E,
    STATES: {
      gps: [GPS_OK, GPS_DEGRADED, GPS_STALE],
      route: [ON_ROUTE, OFF_ROUTE_CANDIDATE, OFF_ROUTE_CONFIRMED],
      maneuver: [UPCOMING, APPROACHING, IMMINENT, EXECUTING, PASSED]
    }
  };

  if (typeof module !== 'undefined' && module.exports) module.exports = root.RIO.navcore;
})(typeof window !== 'undefined' ? window : globalThis);
