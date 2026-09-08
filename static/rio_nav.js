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
    /* One typing session: everything between starting to type a destination
       and picking one. RIO mints an opaque id for it; the server hands that to
       whichever provider is configured and Google turns it into an
       autocomplete session token. The browser never holds a provider token —
       same rule as the API key, applied to the thing the key is spent on.

       The id is minted on the first keystroke and dropped the moment a
       destination is resolved, because a session held past that point would be
       attached to the NEXT thing the driver types. */
    var suggestSession = null;
    // What the driver actually asked for, kept verbatim so a reroute asks for
    // the same PLACE. Re-resolving the label of a place picked from a list can
    // land on a different one of the same name.
    var lastRequest = null;

    function status(text) { if (elStatus) elStatus.textContent = text; }

    function openSuggestSession() {
      if (!suggestSession) {
        suggestSession = 'ac_' + Math.random().toString(36).slice(2, 10) +
                         Date.now().toString(36);
      }
      return suggestSession;
    }
    function endSuggestSession() { suggestSession = null; }

    /* One server suggestion -> one row, or nothing.
     *
     * The `or nothing` is the point. This mapping is the seam where the panel
     * and the endpoint have to agree on field names, and when they silently
     * disagreed the dropdown filled with blank rows that routed to `undefined`
     * — visibly broken, but only once you clicked. An entry without an id or
     * without something to read is dropped here instead, so a future drift
     * shows as "no suggestions" and the driver simply submits what they typed.
     */
    function toSuggestion(c) {
      if (!c) return null;
      var id = c.provider_place_id || '';
      var main = c.display_name || '';
      var secondary = c.formatted_address || '';
      if (!id || !(main || secondary)) return null;
      return { place_id: id, main: main || secondary, secondary: secondary,
               text: secondary || main };
    }
    function toSuggestions(list) {
      var out = [];
      (list || []).forEach(function (c) {
        var s = toSuggestion(c);
        if (s) out.push(s);
      });
      return out;
    }
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

    /* THE IMMINENT CALL, PRELOADED, ONE ELEMENT PER SENTENCE.
     *
     * Same mechanism the red headway tier has used all along and for the same
     * reason: an element that already holds decoded audio turns "say this now"
     * into a play() call, with no network, no queue and no deadline to miss.
     *
     * It matters here because this is the line that cannot be late. Dictated,
     * it carried the tightest budget in the system — 900 ms, because a turn
     * call at the junction is past the junction a moment later — and carrying
     * the tightest budget made it the likeliest line to miss it. On a clean
     * navigating drive it was the only one of eleven that fell back, and it
     * was the one where falling back costs the most.
     *
     * The set is small and fixed (four sentences, none of them naming a road),
     * so all of it is held. Lazily built rather than listed: the server names
     * the clip on the candidate, so a sentence this file has never heard of
     * still gets an element the first time it is called for. */
    var clipEls = {};
    function clipElement(id) {
      if (!clipEls[id]) {
        var a = new Audio('/static/audio/' + id + '.mp3');
        a.preload = 'auto';
        clipEls[id] = a;
        if (unlocked) unlockOne(a);
      }
      return clipEls[id];
    }

    var unlocked = false;

    // iOS Safari plays an element from a timer only once it has been played
    // inside a real user gesture, and EVERY element needs its own. A clip that
    // was never unlocked is a turn call that is silent at the junction, which
    // is the one failure this whole path exists to prevent.
    /* PRIMED ON SILENCE, NEVER ON THE TURN CALL ITSELF.
       Muting an element and playing its real contents to unlock it was audible
       on iOS -- the audio session switches route at that moment and the front
       of the file gets out. On this path that is a turn instruction announced
       at the kerb because a route was attached, which is worse than the safety
       clip at Start Drive that made it visible. See RIO.silentAudioUrl. */
    function unlockOne(a) {
      var real = a.getAttribute('src') || '';
      try {
        a.muted = true;
        if (root.RIO && root.RIO.silentAudioUrl) {
          a.src = root.RIO.silentAudioUrl();
          // Assigning .src does not unload the file already on the element;
          // load() does. Without it the element can still start the turn call
          // it was holding. See primeSilently in index.html.
          a.load();
        }
        var restore = function () {
          try {
            a.pause();
            a.muted = false;
            if (real) { a.src = real; a.load(); }
            else { a.removeAttribute('src'); }
          } catch (e) { a.muted = false; }
        };
        var p = a.play();
        if (p && p.then) p.then(restore).catch(restore);
        else restore();
      } catch (e) { a.muted = false; }
    }

    function unlock() {
      if (unlocked) return;
      unlocked = true;
      // The shared output wants the same gesture every element here does.
      try { if (root.RIO && root.RIO.output) root.RIO.output.unlock(); } catch (e) {}
      unlockOne(navAudio);
      Object.keys(clipEls).forEach(function (id) { unlockOne(clipEls[id]); });
    }

    /* ...and warmed at the moment a route attaches, rather than at the moment
       the turn arrives. Decoding an MP3 the first time it is played is exactly
       the delay the file exists to avoid, and a route is minutes of warning
       that these four sentences are going to be needed. */
    function warmClips(route) {
      var seen = {};
      ((route && route.maneuvers) || []).forEach(function (m) {
        var id = m.speech && m.speech.clips && m.speech.clips.imminent;
        if (id && !seen[id]) { seen[id] = true; clipElement(id); }
      });
      return Object.keys(seen).length;
    }

    function audioFor(candidate) {
      /* One voice for every line RIO says.

         The sentence itself still comes from the ROUTE'S OWN TABLE — the
         browser holds the text only because the server put it there — and it
         is now dictated to the live session word for word rather than
         synthesised separately. The fallback is the same /nav/voice endpoint
         as before, addressed by (route, maneuver, call, anchor), which still
         refuses anything that does not resolve to a line on a live route.

         The arbiter item around this is untouched: same priority, same group,
         same TTL, same pre-emption by anything that matters more. */
      var url = '/nav/voice?route_id=' + encodeURIComponent(candidate.route_id)
              + '&m=' + encodeURIComponent(candidate.maneuver_id)
              + '&call=' + encodeURIComponent(candidate.call_type)
              + (candidate.anchor_id ? '&anchor=' + encodeURIComponent(candidate.anchor_id) : '');
      /* THE FILE, WHERE THERE IS ONE, AND THE MOUTH BEHIND IT.
         Only the imminent call has a clip, and for it the ordinary ladder is
         upside down: dictation is the contingency and the pre-rendered file is
         the path. Everything else keeps the order it had. */
      var clipId = candidate.clip || null;
      return RIO.speak.provider({
        text: candidate.text || '',
        channel: 'nav',
        clipUrl: clipId ? '/static/audio/' + clipId + '.mp3' : null,
        clipFirst: !!clipId,
        clipElement: clipId ? clipElement(clipId) : null,
        // WHICH OF THE FOUR CALLS THIS IS, and it decides how long the line
        // waits for her voice before being synthesised instead. "Left here."
        // at the junction cannot be late; the early call, seconds out, can.
        // Already on the candidate — it is the /nav/voice address above.
        callType: candidate.call_type,
        ttsUrl: url,
        element: navAudio,
      });
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
        // The MESSAGE, not the Error. A stack here says where the promise
        // was rejected, which is this function every time and has never been
        // the question — the question is which of the three reasons it was
        // (no key, script blocked, script failed), and that is the message.
        // Logged once per page: mapsFailed is set above and drawRoute returns
        // early after it. The panel's own tile carries the same text.
        console.warn('[nav] map unavailable:', String(e && e.message || e));
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
    /* ...and the tick DOES SOMETHING NOW. It used to be able to change the GPS
       state and nothing else, which is why a quiet radio was a silent
       navigation system: the speech planner is driven by NAV_PROGRESS and
       position() was the only thing that emitted one. tick() dead-reckons
       along the route inside its coast window and emits progress from there,
       so the turns keep being called while the fix is being recovered.

       Twice a second rather than once. The coast is an extrapolation and the
       calls it makes are timed against a junction; a one-second granularity
       puts up to a second of error into "Left here". */
    setInterval(function () {
      if (!tracker) return;
      clockS = sim.timer ? clockS : nowS();
      var before = tracker.state().gps_state;
      tracker.tick(clockS);
      if (tracker.state().gps_state !== before) paintStates();
    }, 500);

    function attach(r) {
      route = r;
      // Decode the junction calls NOW, not at the junction. A route is minutes
      // of notice that these four sentences are coming, and decoding an MP3
      // the first time it is played is exactly the delay the file exists to
      // avoid.
      warmClips(r);
      tracker = RIO.navcore.create(r);
      planner = RIO.navplan.create({
        tracker: tracker, arbiter: RIO.speech, route: r,
        audio: audioFor, verify: verifyAnchor,
        // The live generation, read from whatever route is active NOW. After a
        // reroute this planner is no longer the current one, and anything it
        // left in the arbiter's queue must fail validity on its own rather
        // than relying on the queue having been cleared.
        activeGeneration: function () { return route ? route.generation_id : -1; },
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
      //
      // The watchdog's numbers travel with the route like every other timing
      // value, so how long the page waits for a fix before rebuilding the
      // watch is tuned in config.py and reaches the car with the next route.
      // See the note on startWatch in index.html for why that watchdog exists:
      // two hundred and twelve seconds of a real drive with no fix and no turn
      // called, on a page that was demonstrably alive throughout.
      if (RIO.headway && RIO.headway.configureWatch) {
        RIO.headway.configureWatch(r.timing);
      }
      if (RIO.headway && RIO.headway.startWatch) RIO.headway.startWatch();
      // A fix already in hand starts tracking immediately rather than at the
      // next GPS tick, so a route set 200 m from a turn announces it now.
      if (lastFix && !sim.timer) tracker.position(lastFix);
      // A SIMULATED DRIVE CARRIES ON, FROM THE START OF WHAT IT IS NOW ON.
      // The simulator's position is a distance along the route, and a route
      // that has just been attached begins where the car is — so keeping the
      // old distance would teleport the car a mile up a road it has not
      // driven. Real GPS needs none of this; it reports where the car is and
      // the projection follows.
      if (sim.timer) sim.s = 0;
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

    /* The one way a route is ever loaded, whoever asked for it: the box, a
       suggestion picked from the list, a reroute, or RIO herself on the
       driver's word.

       It resolves to an OUTCOME — {ok, route} or {ok, error} — rather than to
       nothing. The panel never needed that (it reads the result off the page
       it has just painted), but a caller who has to say out loud whether the
       car is now going somewhere does, and giving this one a return value is
       cheaper than giving her a second way to start a route. */
    function setRoute(opts) {
      if (routing) return Promise.resolve({ ok: false, error: 'already routing' });
      routing = true;
      if (!opts.reroute_of) lastRequest = opts;
      status(!opts.reroute_of ? 'Routing…'
             : (opts.reason === 'driver_request' ? 'Rerouting…'
                                                 : 'Off route · rerouting…'));
      return currentOrigin().then(function (origin) {
        return fetch(RIO.url('/nav/route'), {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            lat: origin.lat, lng: origin.lng,
            // WHICH WAY THE CAR IS POINTING, when the watch has told us. A
            // reroute computed without it is free to begin with a U-turn the
            // driver did not ask for and cannot make; with it the provider
            // routes from the lane we are actually in. Omitted rather than
            // guessed when there is no fix — a heading of 0 is due north, not
            // "unknown".
            heading: (lastFix && typeof lastFix.heading === 'number')
              ? lastFix.heading : null,
            destination: opts.destination || '', place_id: opts.place_id || '',
            label: opts.label || '', reroute_of: opts.reroute_of || null,
            reason: opts.reason || null, session: opts.session || null,
            // What the driver asked to keep off: 'highways', 'tolls',
            // 'ferries'. The server sends only what the provider actually
            // supports and says what it dropped.
            avoid: opts.avoid || [],
          }),
        });
      }).then(function (r) { return r.json(); })
        .then(function (j) {
          if (j.error) throw new Error(j.error);
          attach(j);
          return { ok: true, route: j };
        })
        .catch(function (e) {
          status('No route · ' + (e && e.message ? e.message : e));
          RIO.bus.emit('NAV_ROUTE_FAILED', { error: String(e && e.message || e),
                                             destination: opts.destination || opts.label || '' });
          return { ok: false, error: String(e && e.message || e) };
        })
        .then(function (outcome) { routing = false; return outcome; });
    }

    /* Free text in, one destination or a question out. RIO does not silently
       pick between two plausible readings of "the Getty".

       This is the whole of "go somewhere", and there is deliberately only one
       of it. The destination box calls it on Enter; RIO calls it when the
       driver tells her to take them somewhere. Same resolution, same
       ambiguity question, same route load, same tracker — a spoken
       destination is not a second path through this file, it is this path
       with a different caller.

       Resolves to what happened, for that caller's sake:
         { status: 'routed',    destination, route }
         { status: 'ambiguous', query, candidates }   // and the list is shown
         { status: 'not_found', query }
         { status: 'failed',    error }
       The panel ignores it and reads the page. RIO has to say it out loud. */
    function routeToQuery(text) {
      status('Finding …');
      // The fallback path, and the one that has to keep working when
      // autocomplete does not: whatever is in the box is resolved through the
      // same provider, and an ambiguous answer still asks rather than guesses.
      var session = suggestSession;
      endSuggestSession();
      return fetch(RIO.url('/nav/destination'), {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ q: text, lat: lastFix ? lastFix.lat : null,
                               lng: lastFix ? lastFix.lng : null,
                               session: session }),
      }).then(function (r) { return r.json(); })
        .then(function (j) {
          if (j.status === 'ambiguous') {
            // RIO is asking which one; the answer is a selection, and a
            // selection is the end of a typing session that no longer has an
            // id. The candidates carry place ids, which is all the resolution
            // needs.
            offerDestinations(j.candidates);
            return { status: 'ambiguous', query: j.query || text,
                     candidates: (j.candidates || []).slice(0, 3) };
          }
          if (j.status !== 'resolved') {
            status('No route · could not find "' + text + '"');
            return { status: 'not_found', query: j.query || text };
          }
          var d = j.destination;
          return setRoute({ place_id: d.provider_place_id || '',
                            destination: d.provider_place_id ? '' : d.formatted_address,
                            label: d.display_name || d.formatted_address })
            .then(function (res) {
              if (res && res.ok) {
                return { status: 'routed', destination: d, route: res.route };
              }
              // Resolved to a real place and still could not be routed to:
              // no fix to start from, or the provider refused. Not the same
              // answer as "I could not find it", and never dressed up as one.
              return { status: 'failed', destination: d,
                       error: (res && res.error) || 'could not build a route' };
            });
        })
        .catch(function (e) {
          status('No route · ' + (e && e.message || e));
          return { status: 'failed', error: String(e && e.message || e) };
        });
    }

    function clearRoute(reason) {
      endSuggestSession();
      stopSim();
      if (planner) planner.stop();
      if (tracker) tracker.stop();
      tracker = null; planner = null; route = null; lastRequest = null;
      RIO.speech.clear('nav:');
      clearMap();
      paintRoute();
      status(reason || 'No Route Set');
    }

    /* STOPPING, on purpose. The one path, whoever asked: the Clear button on
       the panel and `stop_navigation` in RIO's voice both arrive here.

       clearRoute already does everything that matters — the tracker and the
       planner are stopped, the queue is emptied of anything addressed 'nav:',
       and `route` going null takes activeGeneration() to -1, so an
       announcement that was already past the queue and in flight fails its own
       validity check at dequeue rather than playing over a driver who has just
       said they know the way. This adds the one thing a spoken stop needs that
       a button press does not: what was stopped, so the confirmation can name
       it, and a line in the drive's log saying who ended the route. */
    function stopRoute(reason) {
      var was = route;
      if (!was) return { ok: true, was_navigating: false };
      var dest = was.destination || {};
      clearRoute('Navigation off');
      RIO.bus.emit('NAV_STOPPED', {
        route_id: was.route_id, journey_id: was.journey_id,
        generation_id: was.generation_id,
        destination: dest.display_name || dest.formatted_address || '',
        reason: reason || 'driver',
      });
      return { ok: true, was_navigating: true,
               destination: dest.display_name || dest.formatted_address || '' };
    }

    /* REROUTING BECAUSE THE DRIVER ASKED, which is a different event from
       rerouting because the car left the route, and is deliberately not the
       same function.

       Off-route rerouting is the tracker noticing and the debounce protecting
       a flapping fix from a routing bill. This is a command: it runs on the
       word, to the SAME destination object (never a re-resolution of its
       label), and it may carry preferences the automatic path never has.

       ATOMIC, and that is the whole shape of it. The old tracker keeps running
       while the provider is asked — it is still the truth until something
       replaces it — and is stopped only once attach() has put a new route,
       tracker and planner in place. A reroute that fails leaves the car
       navigating the route it was already on, rather than navigating nothing
       because it asked a question and got an error. */
    function rerouteNow(opts) {
      opts = opts || {};
      if (!route) return Promise.resolve({ status: 'no_route' });
      if (routing) return Promise.resolve({ status: 'busy' });
      var prev = route;
      var oldTracker = tracker, oldPlanner = planner;
      lastRerouteAt = clockS;      // shares the anti-flap clock with off-route
      return setRoute({
        destination: (lastRequest && lastRequest.destination) || '',
        place_id: (lastRequest && lastRequest.place_id) || '',
        label: (lastRequest && lastRequest.label) || '',
        reroute_of: prev.route_id, reason: 'driver_request',
        avoid: opts.avoid || [],
      }).then(function (res) {
        if (!res || !res.ok) {
          return { status: 'failed', error: (res && res.error) || 'reroute failed',
                   destination: prev.destination };
        }
        // The replacement has happened. Now, and only now, the old plan is
        // torn down and anything it queued is dropped: the new generation is
        // live, so nothing addressed to the old one can be true any more.
        if (oldPlanner) oldPlanner.stop();
        if (oldTracker) oldTracker.stop();
        RIO.speech.clear('nav:');
        RIO.bus.emit('NAV_REROUTED', {
          route_id: res.route.route_id, journey_id: res.route.journey_id,
          from_route_id: prev.route_id,
          from_generation_id: prev.generation_id,
          generation_id: res.route.generation_id,
          avoid: opts.avoid || [], reason: 'driver_request',
        });
        return { status: 'rerouted', route: res.route };
      });
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
    var suggestTimer = null, suggestions = [], suggestSeq = 0;

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
          // The selection closes the typing session, and carries its id so the
          // provider can close it too.
          var session = suggestSession;
          endSuggestSession();
          setRoute({ place_id: s.place_id, label: s.main || s.text,
                     session: session });
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
        var session = openSuggestSession();
        var typedAt = ++suggestSeq;
        // Debounced. Autocomplete is billed per request otherwise, and a
        // driver types faster than a round trip.
        suggestTimer = setTimeout(function () {
          var u = '/nav/suggest?q=' + encodeURIComponent(q)
                + '&session=' + encodeURIComponent(session);
          if (lastFix) u += '&lat=' + lastFix.lat + '&lng=' + lastFix.lng;
          fetch(u).then(function (r) { return r.json(); })
                  .then(function (j) {
                    // A slow reply for a prefix the driver has already typed
                    // past would repopulate the list with older matches.
                    if (typedAt !== suggestSeq) return;
                    paintSuggestions(toSuggestions(j.suggestions));
                  })
                  .catch(function () {
                    // Autocomplete is down. Not an error the driver should see:
                    // the box still works, it just does not predict.
                    paintSuggestions([]);
                  });
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
    if (elClear) elClear.addEventListener('click', function () { stopRoute('panel'); });

    /* ---------------------------------------------------------------------
       Simulate drive — the desk mode.
       Walks the host position along the route's own geometry at a set speed,
       through the same onPosition() a real fix goes through. Nothing about the
       tracking, the planner, the arbiter or the logging knows the difference,
       which is the point: what you hear at the desk is what you get in the car.
       --------------------------------------------------------------------- */
    /* `opts.tickMs` is the desk-test knob, and it is safe for one reason
       worth stating: the simulated clock advances ONE SECOND PER TICK
       whatever the wall clock is doing, so every time-based decision
       downstream — the duplicate cooldown, the gap the imminent call needs
       after the primary, a line's TTL — sees exactly the drive it would have
       seen at 1 Hz. A suite can therefore run a whole route in a second and
       still be measuring the real timing. The button passes nothing and gets
       a second a second, which is the only setting a car ever uses. */
    function startSim(opts) {
      if (!tracker || !route) { status('Set a route first'); return; }
      unlock();
      stopSim(true);
      var tickMs = Math.max(1, (opts && opts.tickMs) || 1000);
      var mph = (opts && opts.mph) ||
                parseFloat(elSimSpeed && elSimSpeed.value) || 30;
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
      }, tickMs);
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
      paintSuggestions(toSuggestions(candidates));
    }

    RIO.nav = {
      setRoute: setRoute,
      routeToQuery: routeToQuery,
      offerDestinations: offerDestinations,
      clearRoute: clearRoute,
      stopRoute: stopRoute,
      reroute: rerouteNow,
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

      /* The turn-by-turn, from the SAME tracker state() answers from.
       *
       * Deliberately not a second read of the route object: the route knows
       * where its maneuvers are, only the tracker knows where the car is, and
       * "how far to the turn after next" is a question about both. Reading the
       * route directly would produce distances measured from the start of the
       * drive, which is the kind of answer that is wrong in a way nobody
       * notices until they are following it.
       *
       * `count` omitted gives the whole remaining route.
       */
      directions: function (count) {
        if (!tracker || !route) return null;
        var st = tracker.state();
        return {
          destination: route.destination,
          route_id: route.route_id,
          generation_id: route.generation_id,
          landmarks_state: route.landmarks_state || null,
          total_maneuvers: (route.maneuvers || []).length,
          remaining_m: st.remaining_m,
          eta_epoch: route.eta_epoch,
          route_state: st.route_state,
          gps_state: st.gps_state,
          arrived: !!st.arrived,
          maneuvers: tracker.upcoming(count),
        };
      },
    };

    paintRoute();
    status('No Route Set');
  });
})();
