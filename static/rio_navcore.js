/* rio_navcore.js — maneuver progression. The half of navigation RIO owns.
 *
 * Google computes the route. This file watches the host position against that
 * route's polyline and decides three things, at ~1 Hz, with no network:
 *
 *   which maneuver is next, how close it is, and whether that is close enough
 *   to say something about.
 *
 * Why client-side. A turn announcement is worth nothing late. Putting a
 * round-trip between "you are 4 seconds from the turn" and RIO saying so means
 * the announcement's timing is the network's timing. Progression is also the
 * one part of navigation that must survive losing signal — the route is already
 * in hand, and following it needs no help.
 *
 * Tiers are TIME, not distance
 * ----------------------------
 * far 30 s / mid 12 s / near 4 s of travel at the current speed. 300 m of open
 * highway and 300 m of downtown are the same distance and completely different
 * warnings; seconds are what a driver actually needs to act. The spoken distance
 * is then the true remaining distance at the moment the tier fires (the server
 * formats it), so a slow approach says "in 90 meters" and a fast one says "in
 * 500 meters" from the same tier.
 *
 * Only the finest applicable tier fires, and tiers never run backwards for a
 * maneuver: a route set 100 m before a turn gets "turn right onto Lincoln", not
 * a countdown that starts too late to be true.
 *
 * No DOM, no fetch, no audio: it takes positions and emits events. That is what
 * lets tools/nav_selftest.js run a whole simulated drive under node and assert
 * the announcement order.
 */
(function (root) {
  'use strict';

  var RAD = Math.PI / 180;
  var M_PER_DEG = 111320;

  var DEFAULTS = {
    // Tier thresholds, seconds of time-to-maneuver. Overridden by the route's
    // own tiers_s so the server stays the single source of the policy.
    tiers: { far: 30, mid: 12, near: 4 },
    // Below this, time-to-maneuver stops meaning anything: at 0.2 m/s every
    // maneuver is hours away and nothing is ever announced — including the turn
    // you are creeping towards in traffic. Treated as a floor, not a speed.
    vFloorMs: 3.0,
    // Used only when there is no speed at all (a fix with no speed field and no
    // previous fix to difference). Logged as speed_source:"nominal" so a
    // surprising announcement can be explained afterwards.
    vNominalMs: 11.0,
    // A maneuver is behind you once you are this far past its point.
    completeEpsM: 8.0,
    // Off-route: sustained, not instantaneous. One bad fix in a tunnel is not a
    // wrong turn, and a reroute that fires on noise is worse than no reroute.
    offRouteM: 45.0,
    offRouteFixes: 3,
    rerouteCooldownS: 10.0,
    // Arrival: within this of the final point, or past it.
    arriveM: 25.0,
    // How far back/forward along the route the projection search looks. Bounded
    // so a route that crosses itself cannot snap the position onto a later leg.
    searchBackM: 80.0,
    searchFwdM: 400.0,
  };

  function haversineM(aLat, aLng, bLat, bLng) {
    var p1 = aLat * RAD, p2 = bLat * RAD;
    var dp = p2 - p1, dl = (bLng - aLng) * RAD;
    var h = Math.sin(dp / 2) * Math.sin(dp / 2) +
            Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) * Math.sin(dl / 2);
    return 2 * 6371008.8 * Math.asin(Math.min(1, Math.sqrt(h)));
  }

  function cumulative(points) {
    var cum = [0];
    for (var i = 1; i < points.length; i++) {
      cum[i] = cum[i - 1] + haversineM(points[i - 1][0], points[i - 1][1],
                                       points[i][0], points[i][1]);
    }
    return cum;
  }

  var TIER_ORDER = ['far', 'mid', 'near'];   // coarse -> fine; never runs back

  function create(route, options) {
    var opt = {};
    var k;
    for (k in DEFAULTS) opt[k] = DEFAULTS[k];
    for (k in (options || {})) opt[k] = options[k];
    if (route && route.tiers_s) {
      opt.tiers = {
        far: route.tiers_s.far, mid: route.tiers_s.mid, near: route.tiers_s.near,
      };
    }

    var points = route.polyline || [];
    var cum = cumulative(points);
    var routeLen = cum.length ? cum[cum.length - 1] : 0;

    // The maneuver's distance along the route is read off the polyline vertex
    // the server pinned it to, not recomputed from its lat/lng: two junctions
    // 20 m apart would otherwise be indistinguishable to a nearest-point search.
    var mans = (route.maneuvers || []).map(function (m) {
      var idx = Math.max(0, Math.min(points.length - 1, m.poly_index || 0));
      return {
        index: m.index, type: m.type, instruction: m.instruction,
        lat: m.lat, lng: m.lng, poly_index: idx, along_m: cum[idx],
        announce: m.announce || {},
        firedRank: 0,          // 0 none, 1 far, 2 mid, 3 near
      };
    });

    var listeners = [];
    var vertex = 0;            // last projected vertex, the search anchor
    var along = 0;             // distance travelled along the route, metres
    var manIdx = 0;            // next maneuver
    var lastFix = null;        // {lat, lng, t}
    var offRouteRun = 0;
    // Far in the past, not 0: the cooldown must not swallow the FIRST reroute
    // of a drive whose clock starts near zero (every simulated drive, and any
    // real one timed from page load).
    var lastRerouteT = -1e9;
    var arrived = false;
    var stopped = false;
    var firstFix = true;
    var lastSpeed = null;
    var lastSpeedSource = 'none';
    var lastOffRouteM = 0;

    function emit(type, payload) {
      var ev = { t: (payload && payload.t) || 0, route_id: route.route_id };
      for (var kk in payload) ev[kk] = payload[kk];
      ev.type = type;   // last, so no payload key can ever shadow the event kind
      for (var i = 0; i < listeners.length; i++) {
        try { listeners[i](ev); } catch (e) { /* a listener must not stop the drive */ }
      }
    }

    function manPublic(m, extra) {
      // maneuver_type, not type: the event's own `type` is the event kind, and
      // a payload key that shadows it silently turns maneuver_approach into
      // TURN_LEFT for every listener downstream.
      var out = {
        maneuver: m.index, maneuver_type: m.type, instruction: m.instruction,
        along_m: Math.round(m.along_m * 10) / 10,
      };
      for (var kk in (extra || {})) out[kk] = extra[kk];
      return out;
    }

    /* Nearest point on the route to (lat,lng), searched in a window around the
       last known position. Local flat-earth metres: over a few hundred metres
       the error is centimetres, and it keeps the inner loop to arithmetic. */
    function project(lat, lng, full) {
      var kLng = M_PER_DEG * Math.cos(lat * RAD);
      var lo = 0, hi = points.length - 2;
      if (!full) {
        lo = vertex;
        while (lo > 0 && cum[vertex] - cum[lo] < opt.searchBackM) lo--;
        hi = vertex;
        var fwd = Math.max(opt.searchFwdM, (lastSpeed || 0) * 30);
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
          best = { d2: d2, i: i, t: t, along: cum[i] + t * Math.sqrt(len2) };
        }
      }
      if (!best) return { vertex: 0, along: 0, offM: 0 };
      return { vertex: best.i, along: best.along, offM: Math.sqrt(best.d2) };
    }

    function tierFor(ttaS) {
      if (ttaS <= opt.tiers.near) return 'near';
      if (ttaS <= opt.tiers.mid) return 'mid';
      if (ttaS <= opt.tiers.far) return 'far';
      return null;
    }

    function speedFrom(fix) {
      if (typeof fix.speed === 'number' && isFinite(fix.speed) && fix.speed >= 0) {
        return { v: fix.speed, src: fix.speedSource || 'fix' };
      }
      if (lastFix && typeof fix.t === 'number' && typeof lastFix.t === 'number') {
        var dt = fix.t - lastFix.t;
        if (dt > 0.2 && dt < 10) {
          var d = haversineM(lastFix.lat, lastFix.lng, fix.lat, fix.lng);
          return { v: d / dt, src: 'derived' };
        }
      }
      return { v: opt.vNominalMs, src: 'nominal' };
    }

    return {
      route: route,
      onEvent: function (fn) { if (typeof fn === 'function') listeners.push(fn); },

      /* One position fix. {lat, lng, speed?, t (seconds), accuracy?} */
      position: function (fix) {
        if (stopped || arrived || !points.length || !fix) return null;

        var sp = speedFrom(fix);
        lastSpeed = sp.v;
        lastSpeedSource = sp.src;

        var pr = project(fix.lat, fix.lng, firstFix);
        // Never run backwards on noise: a fix that projects behind where we have
        // already been is jitter unless it is far enough back to be a real
        // reversal (a missed turn shows up as off-route, not as rewind).
        if (!firstFix && pr.along < along - 30) {
          pr.along = along;
          pr.vertex = vertex;
        }
        firstFix = false;
        vertex = pr.vertex;
        along = pr.along;
        lastOffRouteM = pr.offM;
        lastFix = { lat: fix.lat, lng: fix.lng, t: fix.t };

        // --- off route -----------------------------------------------------
        if (pr.offM > opt.offRouteM) {
          offRouteRun++;
          if (offRouteRun >= opt.offRouteFixes &&
              (fix.t - lastRerouteT) > opt.rerouteCooldownS) {
            lastRerouteT = fix.t;
            offRouteRun = 0;
            // We do not repair the route. Google owns routing; this event asks
            // for a new one and the engine stops announcing until it arrives.
            stopped = true;
            emit('reroute', {
              t: fix.t, reason: 'off_route',
              off_route_m: Math.round(pr.offM * 10) / 10,
              lat: fix.lat, lng: fix.lng,
              from_maneuver: manIdx < mans.length ? manIdx : null,
            });
            return { offRoute: true };
          }
        } else {
          offRouteRun = 0;
        }

        // --- maneuvers now behind us ---------------------------------------
        while (manIdx < mans.length && along > mans[manIdx].along_m + opt.completeEpsM) {
          var done = mans[manIdx];
          manIdx++;
          if (done.type === 'ARRIVE') break;   // arrival is announced below
          emit('maneuver_complete', manPublic(done, {
            t: fix.t, announced_tier: TIER_ORDER[done.firedRank - 1] || null,
            remaining_maneuvers: mans.length - manIdx,
          }));
        }

        // --- arrival -------------------------------------------------------
        var toEnd = routeLen - along;
        var destGap = haversineM(fix.lat, fix.lng,
                                 route.destination.lat, route.destination.lng);
        if (manIdx >= mans.length || (toEnd <= opt.arriveM && destGap <= opt.arriveM * 4)) {
          arrived = true;
          var last = mans.length ? mans[mans.length - 1] : null;
          emit('arrived', {
            t: fix.t, lat: fix.lat, lng: fix.lng,
            destination: route.destination,
            gap_m: Math.round(destGap * 10) / 10,
            announced_tier: last ? (TIER_ORDER[last.firedRank - 1] || null) : null,
          });
          return { arrived: true };
        }

        // --- approach tiers --------------------------------------------------
        var man = mans[manIdx];
        var remaining = man.along_m - along;
        var vEff = Math.max(opt.vFloorMs, sp.v);
        var tta = remaining / vEff;
        var tier = tierFor(tta);
        if (tier) {
          var rank = TIER_ORDER.indexOf(tier) + 1;
          if (rank > man.firedRank) {
            man.firedRank = rank;
            emit('maneuver_approach', manPublic(man, {
              t: fix.t, tier: tier,
              remaining_m: Math.round(remaining * 10) / 10,
              tta_s: Math.round(tta * 10) / 10,
              speed_ms: Math.round(sp.v * 100) / 100,
              speed_source: sp.src,
              // The precomputed line, so the log records what RIO meant to say
              // even if the voice path failed.
              text: (man.announce[tier] || {}).text || man.instruction,
            }));
          }
        }

        emit('progress', {
          t: fix.t, along_m: Math.round(along * 10) / 10,
          remaining_m: Math.round((routeLen - along) * 10) / 10,
          off_route_m: Math.round(pr.offM * 10) / 10,
          maneuver: man.index, maneuver_type: man.type, instruction: man.instruction,
          to_maneuver_m: Math.round(remaining * 10) / 10,
          tta_s: Math.round(tta * 10) / 10,
          speed_ms: Math.round(sp.v * 100) / 100,
          speed_source: sp.src,
          eta_epoch: route.eta_epoch,
        });
        return { along: along, maneuver: man.index, tta: tta };
      },

      /* Position on the polyline at a given distance along it — the simulator's
         only geometry, kept here so simulated and real drives project through
         exactly the same points. */
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
          done: false,
        };
      },

      state: function () {
        return {
          route_id: route.route_id, along_m: along, route_length_m: routeLen,
          maneuver: manIdx < mans.length ? manPublic(mans[manIdx]) : null,
          maneuvers_left: Math.max(0, mans.length - manIdx),
          off_route_m: lastOffRouteM, speed_ms: lastSpeed,
          speed_source: lastSpeedSource, arrived: arrived, stopped: stopped,
        };
      },

      routeLength: function () { return routeLen; },
      stop: function () { stopped = true; },
    };
  }

  root.RIO = root.RIO || {};
  root.RIO.navcore = { create: create, haversineM: haversineM, DEFAULTS: DEFAULTS };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = { create: create, haversineM: haversineM, DEFAULTS: DEFAULTS };
  }
})(typeof window !== 'undefined' ? window : globalThis);
