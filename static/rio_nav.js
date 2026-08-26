/* rio_nav.js — the NAVIGATION panel: routing, the bus, the map, voice, sim.
 *
 * Layering, deliberately kept visible:
 *
 *   /nav/route            the provider computes the route      — navigation/
 *   rio_navcore.js        we track the car against it          — no DOM
 *   rio_navplan.js        we decide what is worth saying       — no DOM
 *   rio_speech.js         one mouth, priority-arbitrated       — no DOM
 *   this file             everything that touches the page
 *
 * Everything goes onto RIO.bus. Three things listen: the panel (paints), the
 * logger (POSTs to /nav/event, kind "nav" in the session JSONL), and the
 * reroute handler. None of them knows about the others, which is what makes
 * the next provider — or a heads-up display, or a passenger screen — an extra
 * subscriber rather than an edit to the tracking code.
 *
 * GPS comes from the ONE Geolocation watch the headway loop already owns. A
 * second watchPosition would double the fixes, halve the battery, and give the
 * two systems subtly different ideas of where the car is.
 */
(function () {
  'use strict';

  window.RIO = window.RIO || {};

  /* ---------------------------------------------------------------------
     Event bus. Deliberately tiny: subscribe by event type or "*".
     --------------------------------------------------------------------- */
  RIO.bus = RIO.bus || (function () {
    var subs = {};
    return {
      on: function (type, fn) {
        (subs[type] = subs[type] || []).push(fn);
        return function () {
          subs[type] = (subs[type] || []).filter(function (f) { return f !== fn; });
        };
      },
      emit: function (type, payload) {
        var ev = payload || {};
        ev.type = type;
        var lists = (subs[type] || []).concat(subs['*'] || []);
        for (var i = 0; i < lists.length; i++) {
          try { lists[i](ev); } catch (e) { console.error('[nav] subscriber', e); }
        }
      },
    };
  })();

  document.addEventListener('DOMContentLoaded', function () {
    var $ = function (id) { return document.getElementById(id); };

    var elStatus = $('navempty');
    var elDest = $('navdest');
    var elSuggest = $('navsuggest');
    var elGo = $('navgo');
    var elClear = $('navclear');
    var elSummary = $('navsummary');
    var elEta = $('naveta');
    var elDist = $('navdistance');
    var elLeft = $('navremaining');
    var elMan = $('navman');
    var elManIcon = $('navmanicon');
    var elManText = $('navmantext');
    var elManDist = $('navmandist');
    var elStateRow = $('navstate');
    var elManState = $('navmanstate');
    var elGpsState = $('navgpsstate');
    var elRouteState = $('navroutestate');
    var elCtxState = $('navctxstate');
    var elAnchor = $('navanchor');
    var elMapBox = $('navmap');
    var elMapIdle = $('navmapidle');
    var elSim = $('navsim');
    var elSimSpeed = $('navsimspeed');
    var elOrigin = $('navorigin');

    var MPH_TO_MS = 0.44704;

    var tracker = null, planner = null, route = null;
    var lastFix = null;
    var sim = { timer: null, s: 0, ms: 0 };
    var map = null, mapLine = null, mapDest = null, mapHost = null, mapsFailed = false;
    var routing = false, lastRerouteAt = -1e9, clockS = 0;
    // What the driver actually asked for, kept verbatim so a reroute asks for
    // the same PLACE. Re-resolving the label of a place picked from a list can
    // land on a different one of the same name.
    var lastRequest = null;

    function status(text) { if (elStatus) elStatus.textContent = text; }
    function nowS() { return (typeof performance !== 'undefined' ? performance.now() : Date.now()) / 1000; }

    /* ---------------------------------------------------------------------
       Announcement audio. Its own element, unlocked in a user gesture like
       every other audio path on this page (iOS will not play from a timer
       otherwise), and fetched as a blob rather than streamed so the exact
       sentence comes back with it on X-Nav-Text — the panel then shows the
       words being spoken instead of its own guess at them.

       Note what is NOT sent: the sentence. The request is (route, maneuver,
       call type, anchor) and the server looks it up in the route's own table.
       --------------------------------------------------------------------- */
    var navAudio = new Audio();
    navAudio.preload = 'auto';
    var unlocked = false;

    function unlock() {
      if (unlocked) return;
      unlocked = true;
      try {
        navAudio.muted = true;
        var p = navAudio.play();
        if (p && p.then) {
          p.then(function () { navAudio.pause(); navAudio.currentTime = 0; navAudio.muted = false; })
           .catch(function () { navAudio.muted = false; });
        } else { navAudio.muted = false; }
      } catch (e) { navAudio.muted = false; }
    }

    function audioFor(candidate) {
      var ctl = (typeof AbortController !== 'undefined') ? new AbortController() : null;
      var stopped = false;
      var cleanup = null;
      return {
        play: function () {
          var url = '/nav/voice?route_id=' + encodeURIComponent(candidate.route_id)
                  + '&m=' + encodeURIComponent(candidate.maneuver_id)
                  + '&call=' + encodeURIComponent(candidate.call_type)
                  + (candidate.anchor_id ? '&anchor=' + encodeURIComponent(candidate.anchor_id) : '');
          return fetch(url, ctl ? { signal: ctl.signal } : undefined)
            .then(function (r) {
              if (!r.ok) throw new Error('nav voice ' + r.status);
              var said = r.headers.get('X-Nav-Text');
              if (said) {
                try { RIO.bus.emit('speaking', { text: decodeURIComponent(said) }); } catch (e) {}
              }
              return r.blob();
            })
            .then(function (blob) {
              if (stopped) return;
              return new Promise(function (resolve, reject) {
                var url2 = URL.createObjectURL(blob);
                var settled = false;
                var done = function (fn) {
                  if (settled) return;
                  settled = true;
                  navAudio.onended = navAudio.onerror = null;
                  URL.revokeObjectURL(url2);
                  fn();
                };
                // A pre-empted line never reaches 'ended', so stop() has to be
                // the one that releases the blob.
                cleanup = function () { done(resolve); };
                navAudio.onended = function () { done(resolve); };
                navAudio.onerror = function () { done(reject); };
                navAudio.src = url2;
                var p = navAudio.play();
                if (p && p.catch) p.catch(function (e) { done(function () { reject(e); }); });
              });
            });
        },
        stop: function () {
          stopped = true;
          if (ctl) { try { ctl.abort(); } catch (e) {} }
          try { navAudio.pause(); } catch (e) {}
          if (cleanup) { cleanup(); cleanup = null; }
        },
      };
    }

    /* Landmark verification. The candidate list is NOT sent — the server takes
       it from the route, so the browser cannot introduce a landmark the map
       never looked up. A failure here resolves to null, which is an ordinary
       outcome: the primary call goes out with the canonical sentence. */
    function verifyAnchor(request) {
      if (!RIO.sessionId) return Promise.resolve(null);
      return fetch(RIO.url('/nav/anchor/verify'), {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ route_id: request.route_id,
                               generation_id: request.generation_id,
                               maneuver_id: request.maneuver_id }),
      }).then(function (r) { return r.json(); })
        .then(function (j) { return j || null; })
        .catch(function () { return null; });
    }

    /* ---------------------------------------------------------------------
       Panel painting
       --------------------------------------------------------------------- */
    function fmtDistance(m) {
      if (!isFinite(m)) return '--';
      if (m >= 1000) return (m / 1000).toFixed(m >= 10000 ? 0 : 1) + ' km';
      if (m >= 100) return Math.round(m / 10) * 10 + ' m';
      return Math.round(m) + ' m';
    }

    function fmtClock(epoch) {
      var d = new Date(epoch * 1000);
      var p = function (n) { return String(n).padStart(2, '0'); };
      return p(d.getHours()) + ':' + p(d.getMinutes());
    }

    function fmtDuration(s) {
      if (!isFinite(s)) return '--';
      var m = Math.round(s / 60);
      if (m < 60) return m + ' min';
      return Math.floor(m / 60) + ' h ' + (m % 60) + ' min';
    }

    // A glyph per maneuver family. Text, not icons: the panel is monospace and
    // an arrow that reads at a glance beats a sprite sheet.
    function glyph(type, direction) {
      var t = String(type || ''), d = String(direction || '');
      if (t === 'ARRIVE') return '◉';
      if (t === 'UTURN') return d === 'LEFT' ? '↶' : '↷';
      if (t === 'ROUNDABOUT') return '↻';
      if (t === 'MERGE' || t === 'RAMP' || t === 'FORK' || t === 'KEEP') return '↗';
      if (d === 'LEFT') return '←';
      if (d === 'RIGHT') return '→';
      return '↑';
    }

    function paintRoute() {
      if (!route) {
        if (elSummary) elSummary.style.display = 'none';
        if (elMan) elMan.style.display = 'none';
        if (elStateRow) elStateRow.style.display = 'none';
        if (elAnchor) elAnchor.style.display = 'none';
        return;
      }
      var first = route.maneuvers[0];
      if (elSummary) elSummary.style.display = '';
      if (elEta) elEta.textContent = fmtClock(route.eta_epoch);
      if (elDist) elDist.textContent = fmtDistance(route.total_distance_m);
      if (elLeft) elLeft.textContent = fmtDuration(route.duration_s);
      if (elMan) elMan.style.display = '';
      if (elStateRow) elStateRow.style.display = '';
      if (elManIcon) elManIcon.textContent = glyph(first && first.type, first && first.direction);
      if (elManText) elManText.textContent = first ? first.instruction : '';
      if (elManDist) elManDist.textContent = '';
    }

    function paintProgress(ev) {
      if (elManIcon) elManIcon.textContent = glyph(ev.maneuver_type, ev.direction);
      if (elManText) elManText.textContent = ev.instruction || '';
      if (elManDist) elManDist.textContent = fmtDistance(ev.to_maneuver_m);
      if (elLeft) elLeft.textContent = fmtDistance(ev.remaining_m) + ' left';
      if (elEta) {
        // ETA re-based on the remaining distance at the speed being driven,
        // rather than the provider's estimate frozen at route time.
        var v = Math.max(3, ev.speed_ms || 0);
        elEta.textContent = fmtClock(Date.now() / 1000 + ev.remaining_m / v);
      }
      paintStates();
    }

    function paintStates() {
      if (!tracker) return;
      var st = tracker.state();
      var ps = planner ? planner.state() : null;
      if (elManState) elManState.textContent = st.maneuver_state || '—';
      if (elGpsState) elGpsState.textContent = (st.gps_state || '').replace('GPS_', '') || '—';
      if (elRouteState) elRouteState.textContent = (st.route_state || '').replace('OFF_ROUTE_', 'OFF/') || '—';
      if (elCtxState) elCtxState.textContent = (ps && ps.context_state) || 'INACTIVE';
      if (elAnchor) {
        var a = ps && ps.anchor;
        if (a) {
          elAnchor.style.display = '';
          elAnchor.innerHTML = '';
          var b = document.createElement('b');
          b.textContent = a.label;
          elAnchor.appendChild(b);
          elAnchor.appendChild(document.createTextNode(
            ' · ' + a.relation.toLowerCase().replace('_', ' ') +
            ' · id ' + a.identity_confidence.toFixed(2) +
            ' · vis ' + a.visibility_confidence.toFixed(2) +
            ' · rel ' + a.relation_confidence.toFixed(2)));
        } else if (ps && ps.context_reason) {
          elAnchor.style.display = '';
          elAnchor.textContent = 'no anchor · ' + ps.context_reason;
        } else {
          elAnchor.style.display = 'none';
        }
      }
    }

    /* ---------------------------------------------------------------------
       Map. Optional by design: if the Maps JavaScript API is unavailable —
       no key, API not enabled, no network — everything else still works and
       the box says so. A route you can hear is the product; the map is a
       convenience.
       --------------------------------------------------------------------- */
    var DARK_STYLE = [
      { elementType: 'geometry', stylers: [{ color: '#0a0f18' }] },
      { elementType: 'labels.text.stroke', stylers: [{ color: '#04060c' }] },
      { elementType: 'labels.text.fill', stylers: [{ color: '#5f7f9d' }] },
      { featureType: 'poi', stylers: [{ visibility: 'off' }] },
      { featureType: 'transit', stylers: [{ visibility: 'off' }] },
      { featureType: 'road', elementType: 'geometry', stylers: [{ color: '#14202e' }] },
      { featureType: 'road', elementType: 'labels.text.fill', stylers: [{ color: '#6f97b8' }] },
      { featureType: 'road.highway', elementType: 'geometry', stylers: [{ color: '#1b2c3e' }] },
      { featureType: 'water', elementType: 'geometry', stylers: [{ color: '#050a12' }] },
      { featureType: 'administrative', elementType: 'geometry.stroke', stylers: [{ color: '#1d3348' }] },
    ];

    var mapsPromise = null;
    function loadMaps() {
      if (mapsPromise) return mapsPromise;
      mapsPromise = new Promise(function (resolve, reject) {
        if (window.google && window.google.maps) return resolve(window.google.maps);
        var key = window.RIO_MAPS_KEY || '';
        if (!key) return reject(new Error('no maps key injected — load the page from / not /static'));
        window.__rioMapsReady = function () { resolve(window.google.maps); };
        var s = document.createElement('script');
        s.src = 'https://maps.googleapis.com/maps/api/js?key=' + encodeURIComponent(key)
              + '&callback=__rioMapsReady&loading=async';
        s.async = true;
        s.onerror = function () { reject(new Error('maps script failed to load')); };
        document.head.appendChild(s);
      });
      return mapsPromise;
    }

    function drawRoute() {
      if (!route || mapsFailed) return;
      loadMaps().then(function (gm) {
        if (elMapIdle) elMapIdle.style.display = 'none';
        if (!map) {
          map = new gm.Map(elMapBox, {
            center: { lat: route.origin.lat, lng: route.origin.lng },
            zoom: 14, disableDefaultUI: true, gestureHandling: 'greedy',
            styles: DARK_STYLE, backgroundColor: '#04060c',
          });
        }
        var path = route.geometry.map(function (p) { return { lat: p[0], lng: p[1] }; });
        if (mapLine) mapLine.setMap(null);
        mapLine = new gm.Polyline({
          path: path, map: map, strokeColor: '#5fb3e8', strokeOpacity: 0.95, strokeWeight: 5,
        });
        if (mapDest) mapDest.setMap(null);
        mapDest = new gm.Marker({
          position: { lat: route.destination.lat, lng: route.destination.lng }, map: map,
          icon: { path: gm.SymbolPath.CIRCLE, scale: 6, fillColor: '#6ee7c7',
                  fillOpacity: 1, strokeColor: '#04060c', strokeWeight: 2 },
        });
        var b = new gm.LatLngBounds();
        path.forEach(function (p) { b.extend(p); });
        map.fitBounds(b, 24);
      }).catch(function (e) {
        mapsFailed = true;
        if (elMapIdle) {
          elMapIdle.style.display = '';
          elMapIdle.textContent = 'Map Offline';
          elMapIdle.title = String(e && e.message || e);
        }
        console.warn('[nav] map unavailable:', e);
      });
    }

    function moveHost(lat, lng) {
      if (!map || !window.google || !window.google.maps) return;
      var gm = window.google.maps;
      var pos = { lat: lat, lng: lng };
      if (!mapHost) {
        mapHost = new gm.Marker({
          position: pos, map: map, zIndex: 10,
          icon: { path: gm.SymbolPath.CIRCLE, scale: 7, fillColor: '#ffb84d',
                  fillOpacity: 1, strokeColor: '#04060c', strokeWeight: 2 },
        });
      } else {
        mapHost.setPosition(pos);
      }
      map.panTo(pos);
    }

    function clearMap() {
      if (mapLine) { mapLine.setMap(null); mapLine = null; }
      if (mapDest) { mapDest.setMap(null); mapDest = null; }
      if (mapHost) { mapHost.setMap(null); mapHost = null; }
    }

    /* ---------------------------------------------------------------------
       Position in, events out
       --------------------------------------------------------------------- */
    function onPosition(fix) {
      lastFix = fix;
      clockS = fix.t;
      moveHost(fix.lat, fix.lng);
      if (tracker) tracker.position(fix);
    }

    // The single Geolocation watch lives in the headway block; nav subscribes
    // to it. Simulated positions win while a simulation is running, so a desk
    // test is not fought over by a real (stationary) fix.
    if (RIO.headway && RIO.headway.onPosition) {
      RIO.headway.onPosition(function (pos) {
        if (sim.timer) return;
        var c = pos.coords;
        onPosition({
          lat: c.latitude, lng: c.longitude,
          // iOS routinely reports both of these as null. The tracker derives
          // them from consecutive fixes when it can, and says which it used.
          speed: (typeof c.speed === 'number' && isFinite(c.speed)) ? c.speed : null,
          heading: (typeof c.heading === 'number' && isFinite(c.heading)) ? c.heading : null,
          accuracy: c.accuracy,
          t: nowS(),
        });
      });
    }

    /* GPS staleness is the ABSENCE of fixes, so something has to run when
       nothing arrives. One second, and it does nothing else. */
    setInterval(function () {
      if (!tracker) return;
      clockS = sim.timer ? clockS : nowS();
      var before = tracker.state().gps_state;
      tracker.tick(clockS);
      if (tracker.state().gps_state !== before) paintStates();
    }, 1000);

    function attach(r) {
      route = r;
      tracker = RIO.navcore.create(r);
      planner = RIO.navplan.create({
        tracker: tracker, arbiter: RIO.speech, route: r,
        audio: audioFor, verify: verifyAnchor,
      });
      tracker.onEvent(function (ev) { RIO.bus.emit(ev.type, ev); });
      planner.onEvent(function (ev) { RIO.bus.emit(ev.type, ev); });
      paintRoute();
      drawRoute();
      RIO.bus.emit('NAV_ROUTE_ATTACHED', {
        route_id: r.route_id, generation_id: r.generation_id,
        journey_id: r.journey_id, destination: r.destination,
        n_maneuvers: r.maneuvers.length, landmarks_state: r.landmarks_state,
      });
      status('Route set · ' + (r.destination.display_name || r.destination.formatted_address));
      // Tracking needs fixes whether or not a drive is running, and the watch
      // is shared, so asking for it twice is free.
      if (RIO.headway && RIO.headway.startWatch) RIO.headway.startWatch();
      // A fix already in hand starts tracking immediately rather than at the
      // next GPS tick, so a route set 200 m from a turn announces it now.
      if (lastFix && !sim.timer) tracker.position(lastFix);
    }

    /* ---------------------------------------------------------------------
       Routing
       --------------------------------------------------------------------- */
    function geocode(text) {
      return fetch('/nav/geocode?q=' + encodeURIComponent(text))
        .then(function (r) { return r.json(); })
        .then(function (j) { return j.error ? null : j; });
    }

    function currentOrigin() {
      // 1. an explicit override (desk testing: "route from somewhere else")
      var manual = elOrigin && elOrigin.value.trim();
      if (manual) {
        var pair = manual.split(',');
        if (pair.length === 2 && isFinite(parseFloat(pair[0])) && isFinite(parseFloat(pair[1]))) {
          return Promise.resolve({ lat: parseFloat(pair[0]), lng: parseFloat(pair[1]) });
        }
        return geocode(manual).then(function (g) {
          if (!g) throw new Error('could not find that start point');
          return { lat: g.lat, lng: g.lng };
        });
      }
      // 2. the live GPS watch
      if (lastFix) return Promise.resolve({ lat: lastFix.lat, lng: lastFix.lng });
      // 3. one direct ask, for the case where no drive has started yet
      return new Promise(function (resolve, reject) {
        if (!navigator.geolocation) return reject(new Error('no GPS on this device'));
        navigator.geolocation.getCurrentPosition(
          function (p) { resolve({ lat: p.coords.latitude, lng: p.coords.longitude }); },
          function () { reject(new Error('no position — allow location, or set a start point')); },
          { enableHighAccuracy: true, timeout: 10000, maximumAge: 30000 });
      });
    }

    function setRoute(opts) {
      if (routing) return Promise.resolve();
      routing = true;
      if (!opts.reroute_of) lastRequest = opts;
      status(opts.reroute_of ? 'Off route · rerouting…' : 'Routing…');
      return currentOrigin().then(function (origin) {
        return fetch(RIO.url('/nav/route'), {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            lat: origin.lat, lng: origin.lng,
            destination: opts.destination || '', place_id: opts.place_id || '',
            label: opts.label || '', reroute_of: opts.reroute_of || null,
            reason: opts.reason || null,
          }),
        });
      }).then(function (r) { return r.json(); })
        .then(function (j) {
          if (j.error) throw new Error(j.error);
          attach(j);
        })
        .catch(function (e) {
          status('No route · ' + (e && e.message ? e.message : e));
          RIO.bus.emit('NAV_ROUTE_FAILED', { error: String(e && e.message || e),
                                             destination: opts.destination || opts.label || '' });
        })
        .then(function () { routing = false; });
    }

    /* Free text in, one destination or a question out. RIO does not silently
       pick between two plausible readings of "the Getty". */
    function routeToQuery(text) {
      status('Finding …');
      return fetch(RIO.url('/nav/destination'), {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q: text, lat: lastFix ? lastFix.lat : null,
                               lng: lastFix ? lastFix.lng : null }),
      }).then(function (r) { return r.json(); })
        .then(function (j) {
          if (j.status === 'ambiguous') {
            offerDestinations(j.candidates);
            return;
          }
          if (j.status !== 'resolved') {
            status('No route · could not find "' + text + '"');
            return;
          }
          var d = j.destination;
          return setRoute({ place_id: d.provider_place_id || '',
                            destination: d.provider_place_id ? '' : d.formatted_address,
                            label: d.display_name || d.formatted_address });
        })
        .catch(function (e) { status('No route · ' + (e && e.message || e)); });
    }

    function clearRoute(reason) {
      stopSim();
      if (planner) planner.stop();
      if (tracker) tracker.stop();
      tracker = null; planner = null; route = null; lastRequest = null;
      RIO.speech.clear('nav:');
      clearMap();
      paintRoute();
      status(reason || 'No Route Set');
    }

    /* Reroute: the provider is asked for a new route from where we actually
       are. We never patch the old one — a "shortest path back to the polyline"
       invented here is exactly the kind of route RIO has no business
       inventing. Debounced, because a reroute that fires on a flap is worse
       than no reroute. */
    RIO.bus.on('NAV_OFF_ROUTE_CONFIRMED', function (ev) {
      if (!route) return;
      var debounce = (route.timing && route.timing.reroute_debounce_s) || 12;
      if ((clockS - lastRerouteAt) < debounce) return;
      lastRerouteAt = clockS;
      var prev = route.route_id;
      RIO.speech.clear('nav:');
      if (planner) planner.stop();
      if (tracker) tracker.stop();
      setRoute({
        destination: (lastRequest && lastRequest.destination) || '',
        place_id: (lastRequest && lastRequest.place_id) || '',
        label: (lastRequest && lastRequest.label) || '',
        reroute_of: prev, reason: 'off_route',
      });
    });

    RIO.bus.on('NAV_ARRIVED', function () {
      status('Arrived · ' + (route ? (route.destination.display_name || '') : ''));
      stopSim();
      if (tracker) tracker.stop();
      if (elManDist) elManDist.textContent = '';
    });

    RIO.bus.on('NAV_PROGRESS', paintProgress);
    RIO.bus.on('NAV_ANCHOR_VERIFIED', paintStates);
    RIO.bus.on('NAV_ANCHOR_REJECTED', paintStates);

    /* ---------------------------------------------------------------------
       Session log. Everything except NAV_PROGRESS, which is a 1 Hz UI tick and
       would bury the events that matter in the JSONL.
       --------------------------------------------------------------------- */
    var NOT_LOGGED = { NAV_PROGRESS: 1 };
    RIO.bus.on('*', function (ev) {
      if (!RIO.sessionId) return;
      if (String(ev.type).indexOf('NAV_') !== 0 || NOT_LOGGED[ev.type]) return;
      // Route start, reroute completion and anchor verification are written
      // server-side, where the whole route is in hand; these would be thinner
      // duplicates.
      if (ev.type === 'NAV_ROUTE_ATTACHED' || ev.type === 'NAV_ANCHOR_VERIFIED' ||
          ev.type === 'NAV_ANCHOR_REJECTED') return;
      var payload = {};
      for (var k in ev) if (k !== 'type') payload[k] = ev[k];
      try {
        fetch(RIO.url('/nav/event'), {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ event: ev.type, payload: payload }),
          keepalive: true,
        }).catch(function () {});
      } catch (e) { /* logging must never break the drive */ }
    });

    /* ---------------------------------------------------------------------
       Destination input + autocomplete
       --------------------------------------------------------------------- */
    var suggestTimer = null, suggestions = [];

    function paintSuggestions(list) {
      suggestions = list || [];
      if (!elSuggest) return;
      elSuggest.innerHTML = '';
      if (!suggestions.length) { elSuggest.style.display = 'none'; return; }
      suggestions.forEach(function (s) {
        var d = document.createElement('div');
        d.className = 'nav-sug';
        d.innerHTML = '<b></b><i></i>';
        d.querySelector('b').textContent = s.main || s.text;
        d.querySelector('i').textContent = s.secondary || '';
        d.addEventListener('click', function () {
          elDest.value = s.text;
          paintSuggestions([]);
          unlock();
          setRoute({ place_id: s.place_id, label: s.main || s.text });
        });
        elSuggest.appendChild(d);
      });
      elSuggest.style.display = '';
    }

    if (elDest) {
      elDest.disabled = false;
      elDest.addEventListener('input', function () {
        var q = elDest.value.trim();
        if (suggestTimer) clearTimeout(suggestTimer);
        if (q.length < 3) { paintSuggestions([]); return; }
        // Debounced: autocomplete is billed per keystroke otherwise.
        suggestTimer = setTimeout(function () {
          var u = '/nav/suggest?q=' + encodeURIComponent(q);
          if (lastFix) u += '&lat=' + lastFix.lat + '&lng=' + lastFix.lng;
          fetch(u).then(function (r) { return r.json(); })
                  .then(function (j) {
                    paintSuggestions((j.suggestions || []).map(function (c) {
                      return { place_id: c.provider_place_id, text: c.formatted_address,
                               main: c.display_name, secondary: c.formatted_address };
                    }));
                  })
                  .catch(function () { paintSuggestions([]); });
        }, 250);
      });
      elDest.addEventListener('keydown', function (e) {
        if (e.key !== 'Enter') return;
        e.preventDefault();
        paintSuggestions([]);
        unlock();
        if (elDest.value.trim()) routeToQuery(elDest.value.trim());
      });
      document.addEventListener('click', function (e) {
        if (elSuggest && !elSuggest.contains(e.target) && e.target !== elDest) paintSuggestions([]);
      });
    }

    if (elGo) {
      elGo.addEventListener('click', function () {
        unlock();
        paintSuggestions([]);
        if (elDest && elDest.value.trim()) routeToQuery(elDest.value.trim());
      });
    }
    if (elClear) elClear.addEventListener('click', function () { clearRoute(); });

    /* ---------------------------------------------------------------------
       Simulate drive — the desk mode.
       Walks the host position along the route's own geometry at a set speed,
       through the same onPosition() a real fix goes through. Nothing about the
       tracking, the planner, the arbiter or the logging knows the difference,
       which is the point: what you hear at the desk is what you get in the car.
       --------------------------------------------------------------------- */
    function startSim() {
      if (!tracker || !route) { status('Set a route first'); return; }
      unlock();
      stopSim(true);
      var mph = parseFloat(elSimSpeed && elSimSpeed.value) || 30;
      sim.ms = Math.max(1, mph * MPH_TO_MS);
      sim.s = 0;
      clockS = 0;
      if (elSim) { elSim.textContent = 'Stop Sim'; elSim.setAttribute('aria-pressed', 'true'); }
      RIO.bus.emit('NAV_SIM_START', { speed_ms: Math.round(sim.ms * 100) / 100, mph: mph,
                                      route_id: route.route_id });
      sim.timer = setInterval(function () {
        if (!tracker) { stopSim(); return; }
        var p = tracker.pointAt(sim.s);
        if (!p) { stopSim(); return; }
        clockS += 1;
        onPosition({ lat: p.lat, lng: p.lng, speed: sim.ms, speedSource: 'sim',
                     accuracy: 5, t: clockS });
        if (p.done) { stopSim(); return; }
        sim.s += sim.ms;   // one second of travel per tick
      }, 1000);
    }

    function stopSim(quiet) {
      if (sim.timer) {
        clearInterval(sim.timer);
        sim.timer = null;
        if (!quiet) RIO.bus.emit('NAV_SIM_END', { along_m: Math.round(sim.s) });
      }
      if (elSim) { elSim.textContent = 'Simulate Drive'; elSim.setAttribute('aria-pressed', 'false'); }
    }

    if (elSim) {
      elSim.addEventListener('click', function () {
        if (sim.timer) stopSim(); else startSim();
      });
    }

    /* ---------------------------------------------------------------------
       Public surface. Swapping in an embedded or offline NavigationProvider is
       a server-side change; nothing on this page needs to know which one
       answered.
       --------------------------------------------------------------------- */
    /* RIO asked "which Getty?" out loud; the panel has to show which ones, or
       the question has no answer the driver can give. */
    function offerDestinations(candidates) {
      status('Which one?');
      paintSuggestions((candidates || []).map(function (c) {
        return { place_id: c.provider_place_id, text: c.formatted_address,
                 main: c.display_name, secondary: c.formatted_address };
      }));
    }

    RIO.nav = {
      setRoute: setRoute,
      routeToQuery: routeToQuery,
      offerDestinations: offerDestinations,
      clearRoute: clearRoute,
      simulate: startSim,
      stopSimulation: stopSim,
      unlock: unlock,
      get route() { return route; },
      state: function () {
        if (!tracker) return null;
        var st = tracker.state();
        st.context = planner ? planner.state() : null;
        return st;
      },
    };

    paintRoute();
    status('No Route Set');
  });
})();
