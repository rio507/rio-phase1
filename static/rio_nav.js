/* rio_nav.js — the NAVIGATION panel: provider, bus, map, voice, dev simulator.
 *
 * Layering, deliberately kept visible:
 *
 *   /nav/route  (server)      Google computes the route          — nav.py
 *   rio_navcore.js            we compute progression from GPS    — no DOM
 *   rio_speech.js             one mouth, priority-arbitrated     — no DOM
 *   this file                 everything that touches the page
 *
 * Nav events all go onto RIO.bus. Three things listen: the panel (paints), the
 * announcer (speaks, through the arbiter), and the logger (POSTs to /nav/event,
 * kind "nav" in the session JSONL). None of them knows about the others, which
 * is what makes the next provider — or a heads-up display, or a passenger
 * screen — an extra subscriber rather than an edit to the progression code.
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
          // One broken subscriber must not take navigation down with it.
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
    var elMapBox = $('navmap');
    var elMapIdle = $('navmapidle');
    var elSim = $('navsim');
    var elSimSpeed = $('navsimspeed');
    var elOrigin = $('navorigin');

    var MPH_TO_MS = 0.44704;

    // Tier -> how RIO treats the announcement. The near tier is the only nav
    // line that outranks other nav lines, and the TTLs are the window in which
    // each is still true: a "near" line 3 s stale is a turn already missed.
    //
    // Named, not numbered. These used to be the literals 2 and 3, which was
    // fine until a tier was inserted above them and the numbers meant something
    // else — the ladder lives in rio_speech.js and this file should not hold a
    // second, silent copy of it.
    var TIER_RULES = {
      far: { priority: RIO.speech.P.NAV, ttlMs: 12000 },
      mid: { priority: RIO.speech.P.NAV, ttlMs: 6000 },
      near: { priority: RIO.speech.P.TURN_NEAR, ttlMs: 3000 },
    };

    var engine = null;         // rio_navcore engine for the active route
    var route = null;
    var lastFix = null;        // {lat, lng, t, speed}
    var sim = { timer: null, s: 0, ms: 0 };
    var map = null, mapLine = null, mapDest = null, mapHost = null, mapsFailed = false;
    var routing = false;
    // What the driver actually asked for, kept verbatim so a reroute asks for
    // the same PLACE. Re-geocoding the label of a place picked from
    // autocomplete can land on a different one of the same name.
    var lastRequest = null;

    function status(text) { if (elStatus) elStatus.textContent = text; }

    /* ---------------------------------------------------------------------
       Announcement audio. Its own element, unlocked in a user gesture like
       every other audio path on this page (iOS will not play from a timer
       otherwise), and fetched as a blob rather than streamed so the exact
       sentence comes back with it on X-Nav-Text — the panel then shows the
       words being spoken instead of its own guess at them.
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

    function announce(ev) {
      if (!route) return;
      var rules = TIER_RULES[ev.tier] || TIER_RULES.far;
      var ctl = (typeof AbortController !== 'undefined') ? new AbortController() : null;
      var stopped = false;
      var cleanup = null;      // set once the audio blob exists; see stop()

      RIO.speech.say({
        priority: rules.priority,
        group: 'nav:m' + ev.maneuver,
        id: 'nav:m' + ev.maneuver + ':' + ev.tier,
        text: ev.text,                    // the precomputed line, until the server's comes back
        ttlMs: rules.ttlMs,
        meta: { maneuver: ev.maneuver, tier: ev.tier },
        play: function () {
          var url = '/nav/voice?route_id=' + encodeURIComponent(route.route_id)
                  + '&m=' + ev.maneuver + '&tier=' + encodeURIComponent(ev.tier)
                  + '&dist_m=' + Math.round(ev.remaining_m);
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
        onDone: function (reason) {
          // What actually came out of the mouth, as opposed to what the
          // progression engine asked for. The gap between the two is the whole
          // reason the arbiter is worth having a log of.
          RIO.bus.emit('speech', {
            maneuver: ev.maneuver, tier: ev.tier, reason: reason, text: ev.text,
          });
        },
      });
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
    function glyph(type) {
      var t = String(type || '');
      if (t.indexOf('LEFT') >= 0) return t.indexOf('U_TURN') >= 0 ? '↶' : '←';
      if (t.indexOf('RIGHT') >= 0) return t.indexOf('U_TURN') >= 0 ? '↷' : '→';
      if (t === 'ARRIVE') return '◉';
      if (t.indexOf('MERGE') >= 0 || t.indexOf('RAMP') >= 0 || t.indexOf('FORK') >= 0) return '↗';
      if (t.indexOf('ROUNDABOUT') >= 0) return '↻';
      return '↑';
    }

    function paintRoute() {
      if (!route) {
        if (elSummary) elSummary.style.display = 'none';
        if (elMan) elMan.style.display = 'none';
        return;
      }
      if (elSummary) elSummary.style.display = '';
      if (elEta) elEta.textContent = fmtClock(route.eta_epoch);
      if (elDist) elDist.textContent = fmtDistance(route.distance_m);
      if (elLeft) elLeft.textContent = fmtDuration(route.duration_s);
      if (elMan) elMan.style.display = '';
      if (elManIcon) elManIcon.textContent = glyph(route.maneuvers[0] && route.maneuvers[0].type);
      if (elManText) elManText.textContent = route.maneuvers[0] ? route.maneuvers[0].instruction : '';
      if (elManDist) elManDist.textContent = '';
    }

    function paintProgress(ev) {
      if (elManIcon) elManIcon.textContent = glyph(ev.maneuver_type);
      if (elManText) elManText.textContent = ev.instruction || '';
      if (elManDist) elManDist.textContent = fmtDistance(ev.to_maneuver_m);
      if (elLeft) elLeft.textContent = fmtDistance(ev.remaining_m) + ' left';
      if (elEta && route) {
        // ETA re-based on the remaining distance at the speed being driven,
        // rather than Google's static estimate frozen at route time.
        var v = Math.max(3, ev.speed_ms || 0);
        elEta.textContent = fmtClock(Date.now() / 1000 + ev.remaining_m / v);
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
        var path = route.polyline.map(function (p) { return { lat: p[0], lng: p[1] }; });
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
      moveHost(fix.lat, fix.lng);
      if (!engine) return;
      engine.position(fix);
    }

    // The single Geolocation watch lives in the headway block; nav subscribes
    // to it. Simulated positions win while a simulation is running, so a desk
    // test is not fought over by a real (stationary) fix.
    if (RIO.headway && RIO.headway.onPosition) {
      RIO.headway.onPosition(function (pos) {
        if (sim.timer) return;
        onPosition({
          lat: pos.coords.latitude, lng: pos.coords.longitude,
          speed: (typeof pos.coords.speed === 'number' && isFinite(pos.coords.speed))
                 ? pos.coords.speed : null,
          accuracy: pos.coords.accuracy,
          t: performance.now() / 1000,
        });
      });
    }

    function attach(r) {
      route = r;
      engine = RIO.navcore.create(r);
      engine.onEvent(function (ev) { RIO.bus.emit(ev.type, ev); });
      paintRoute();
      drawRoute();
      RIO.bus.emit('route_set', {
        route_id: r.route_id, destination: r.destination, distance_m: r.distance_m,
        duration_s: r.duration_s, n_maneuvers: r.maneuvers.length,
        reroute_of: r.reroute_of || null,
      });
      status('Route set · ' + r.destination.label);
      // Progression needs fixes whether or not a drive is running, and the
      // watch is shared, so asking for it twice is free.
      if (RIO.headway && RIO.headway.startWatch) RIO.headway.startWatch();
      // A fix already in hand starts progression immediately rather than at the
      // next GPS tick, so a route set 200 m from a turn announces it now.
      if (lastFix && !sim.timer) engine.position(lastFix);
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
      status('Routing…');
      return currentOrigin().then(function (origin) {
        return fetch(RIO.url('/nav/route'), {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            lat: origin.lat, lng: origin.lng,
            destination: opts.destination || '', place_id: opts.place_id || '',
            label: opts.label || '', reroute_of: opts.reroute_of || null,
          }),
        });
      }).then(function (r) { return r.json(); })
        .then(function (j) {
          if (j.error) throw new Error(j.error);
          attach(j);
        })
        .catch(function (e) {
          status('No route · ' + (e && e.message ? e.message : e));
          RIO.bus.emit('route_failed', { error: String(e && e.message || e),
                                         destination: opts.destination || opts.label || '' });
        })
        .then(function () { routing = false; });
    }

    function clearRoute(reason) {
      stopSim();
      if (engine) engine.stop();
      engine = null; route = null; lastRequest = null;
      RIO.speech.clear('nav:');
      clearMap();
      paintRoute();
      status(reason || 'No Route Set');
    }

    /* Reroute: Google gets asked for a new route from where we actually are.
       We never patch the old one — a "shortest path back to the polyline"
       invented here is exactly the kind of route RIO has no business
       inventing. */
    RIO.bus.on('reroute', function (ev) {
      if (!route) return;
      var dest = route.destination;
      var prev = route.route_id;
      status('Off route · rerouting…');
      RIO.speech.clear('nav:');
      if (engine) engine.stop();
      engine = null;
      setRoute({
        destination: (lastRequest && lastRequest.destination) || dest.label,
        place_id: (lastRequest && lastRequest.place_id) || '',
        label: dest.label,
        reroute_of: prev,
      });
    });

    RIO.bus.on('arrived', function () {
      status('Arrived · ' + (route ? route.destination.label : ''));
      stopSim();
      if (engine) engine.stop();
      if (elManDist) elManDist.textContent = '';
    });

    RIO.bus.on('maneuver_approach', announce);
    RIO.bus.on('progress', paintProgress);

    /* ---------------------------------------------------------------------
       Session log. Everything except `progress`, which is a 1 Hz UI tick and
       would bury the events that matter in the JSONL.
       --------------------------------------------------------------------- */
    var LOGGED = {
      route_set: 1, route_failed: 1, maneuver_approach: 1, maneuver_complete: 1,
      reroute: 1, arrived: 1, speech: 1, sim_start: 1, sim_end: 1,
    };
    RIO.bus.on('*', function (ev) {
      if (!LOGGED[ev.type] || !RIO.sessionId) return;
      // route_set is written server-side by /nav/route with the full maneuver
      // list; this one would be a thinner duplicate.
      if (ev.type === 'route_set') return;
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
       Destination input + Places autocomplete
       --------------------------------------------------------------------- */
    var suggestTimer = null, suggestions = [];

    function paintSuggestions(list) {
      suggestions = list || [];
      if (!elSuggest) return;
      elSuggest.innerHTML = '';
      if (!suggestions.length) { elSuggest.style.display = 'none'; return; }
      suggestions.forEach(function (s, i) {
        var d = document.createElement('div');
        d.className = 'nav-sug';
        d.innerHTML = '<b></b><i></i>';
        d.querySelector('b').textContent = s.main || s.text;
        d.querySelector('i').textContent = s.secondary || '';
        d.addEventListener('click', function () {
          elDest.value = s.text;
          paintSuggestions([]);
          unlock();
          setRoute({ place_id: s.place_id, label: s.text });
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
                  .then(function (j) { paintSuggestions(j.suggestions); })
                  .catch(function () { paintSuggestions([]); });
        }, 250);
      });
      elDest.addEventListener('keydown', function (e) {
        if (e.key !== 'Enter') return;
        e.preventDefault();
        paintSuggestions([]);
        unlock();
        // Enter routes to what was typed — autocomplete is a convenience, not a
        // gate. A plain address geocodes server-side.
        if (elDest.value.trim()) setRoute({ destination: elDest.value.trim() });
      });
      document.addEventListener('click', function (e) {
        if (elSuggest && !elSuggest.contains(e.target) && e.target !== elDest) paintSuggestions([]);
      });
    }

    if (elGo) {
      elGo.addEventListener('click', function () {
        unlock();
        paintSuggestions([]);
        if (elDest && elDest.value.trim()) setRoute({ destination: elDest.value.trim() });
      });
    }
    if (elClear) elClear.addEventListener('click', function () { clearRoute(); });

    /* ---------------------------------------------------------------------
       Simulate drive — the desk mode.
       Walks the host position along the route's own polyline at a set speed,
       through the same onPosition() a real fix goes through. Nothing about the
       progression, the tiers, the arbiter or the logging knows the difference,
       which is the point: what you hear at the desk is what you get in the car.
       --------------------------------------------------------------------- */
    function startSim() {
      if (!engine || !route) { status('Set a route first'); return; }
      unlock();
      stopSim(true);
      var mph = parseFloat(elSimSpeed && elSimSpeed.value) || 30;
      sim.ms = Math.max(1, mph * MPH_TO_MS);
      sim.s = 0;
      if (elSim) { elSim.textContent = 'Stop Sim'; elSim.setAttribute('aria-pressed', 'true'); }
      RIO.bus.emit('sim_start', { speed_ms: Math.round(sim.ms * 100) / 100, mph: mph,
                                  route_id: route.route_id });
      sim.timer = setInterval(function () {
        if (!engine) { stopSim(); return; }
        var p = engine.pointAt(sim.s);
        if (!p) { stopSim(); return; }
        onPosition({ lat: p.lat, lng: p.lng, speed: sim.ms, speedSource: 'sim',
                     t: performance.now() / 1000 });
        if (p.done) { stopSim(); return; }
        sim.s += sim.ms;   // one second of travel per tick
      }, 1000);
    }

    function stopSim(quiet) {
      if (sim.timer) {
        clearInterval(sim.timer);
        sim.timer = null;
        if (!quiet) RIO.bus.emit('sim_end', { along_m: Math.round(sim.s) });
      }
      if (elSim) { elSim.textContent = 'Simulate Drive'; elSim.setAttribute('aria-pressed', 'false'); }
    }

    if (elSim) {
      elSim.addEventListener('click', function () {
        if (sim.timer) stopSim(); else startSim();
      });
    }

    /* ---------------------------------------------------------------------
       Public surface. `provider` is named for what it is: swapping in an
       embedded or offline NavigationProvider means another object with these
       four methods, and nothing else on the page changes.
       --------------------------------------------------------------------- */
    RIO.nav = {
      provider: 'web',
      setRoute: setRoute,
      clearRoute: clearRoute,
      simulate: startSim,
      stopSimulation: stopSim,
      unlock: unlock,
      get route() { return route; },
      state: function () { return engine ? engine.state() : null; },
    };

    paintRoute();
    status('No Route Set');
  });
})();
