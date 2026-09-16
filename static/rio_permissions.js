/* rio_permissions.js — asking for the camera, the microphone and the position,
 * once, inside the tap, and saying plainly what was refused.
 *
 * THE DRIVE THIS EXISTS BECAUSE OF (2026-09-16). On a phone, the geolocation
 * prompt never appeared. Nothing said so. Place search came back with somewhere
 * far away and "news around here" resolved to the wrong area, and from the
 * passenger seat that is indistinguishable from the search being bad — which is
 * where the time went looking.
 *
 * Three separate faults were hiding behind one silence:
 *
 *   NOT A SECURE CONTEXT. Over plain http on a LAN address, `getUserMedia` and
 *   geolocation are refused by the browser before any prompt is drawn. The page
 *   already knew this for media (RIO.mediaError) and had no equivalent for
 *   position, so a phone opened on http://<ip>:8888 failed silently and looked
 *   like a permission the driver had dismissed.
 *
 *   THE GESTURE WAS SPENT. iOS grants a prompt only from a trusted gesture, and
 *   a gesture does not survive an `await`. Anything asked for after the first
 *   network round trip in the Start Drive handler is asked for outside the tap.
 *   So all three are STARTED synchronously here and awaited afterwards — the
 *   ordering that matters is when each call is made, not when it resolves.
 *
 *   A REFUSAL LOOKED LIKE AN ABSENCE. `startWatch()` returned quietly when the
 *   geolocation object was missing, and set `lastFixAt` to now before any fix
 *   had arrived — so a drive that never got a position reported a FRESH
 *   position age. A denied permission has to be a visible state, not a value
 *   that happens never to change.
 *
 * WHAT THIS DOES NOT DO: it never decides what happens next. It reports four
 * states per capability and the drive decides which are fatal. Camera and
 * position are blocking (see index.html); the microphone is not, because a
 * drive with no voice is degraded and still worth having.
 *
 *   granted      the browser said yes
 *   denied       the driver or the browser said no
 *   unavailable  the API is not there at all
 *   insecure     the page is not a secure context, so it cannot even be asked
 *
 * Pure logic and browser APIs, no DOM: the panel renders the states and this
 * file never touches the page, which is what lets the selftest drive it.
 */
(function (root) {
  'use strict';

  var STATES = ['unknown', 'granted', 'denied', 'unavailable', 'insecure'];

  function makePermissions(nav, win) {
    nav = nav || root.navigator;
    win = win || root;

    var state = { camera: 'unknown', mic: 'unknown', geo: 'unknown' };
    var detail = { camera: null, mic: null, geo: null };
    var lastFix = null;          // {lat, lng, accuracy_m, at}
    var listeners = [];
    var asked = false;

    function secure() {
      // `isSecureContext` is the browser's own answer and is what both APIs
      // actually gate on. localhost counts; a LAN IP over http does not.
      return win.isSecureContext !== false;
    }

    function emit() {
      var snap = snapshot();
      for (var i = 0; i < listeners.length; i++) {
        try { listeners[i](snap); } catch (e) { /* a listener must not block a drive */ }
      }
    }

    function set(name, value, why) {
      if (STATES.indexOf(value) < 0) value = 'unknown';
      state[name] = value;
      detail[name] = why || null;
      emit();
    }

    /* The position, kept here rather than in the watcher, because "have we
       ever had a fix" is a permission question before it is a navigation one.
       An age is only meaningful once a real fix has arrived: before that there
       is no age, and reporting one is how a drive with no position looked like
       a drive with a fresh one. */
    function noteFix(pos) {
      var c = pos && (pos.coords || pos);
      if (!c) return;
      var lat = (typeof c.latitude === 'number') ? c.latitude : c.lat;
      var lng = (typeof c.longitude === 'number') ? c.longitude : c.lng;
      if (typeof lat !== 'number' || typeof lng !== 'number') return;
      if (!isFinite(lat) || !isFinite(lng)) return;
      lastFix = {
        lat: lat, lng: lng,
        accuracy_m: (typeof c.accuracy === 'number' && isFinite(c.accuracy))
          ? c.accuracy : null,
        at: Date.now(),
      };
      if (state.geo !== 'granted') set('geo', 'granted');
      else emit();
    }

    function fix() {
      if (!lastFix) return null;
      return {
        lat: lastFix.lat, lng: lastFix.lng,
        accuracy_m: lastFix.accuracy_m,
        age_s: Math.max(0, (Date.now() - lastFix.at) / 1000),
      };
    }

    function snapshot() {
      return {
        camera: state.camera, mic: state.mic, geo: state.geo,
        detail: { camera: detail.camera, mic: detail.mic, geo: detail.geo },
        secure: secure(),
        asked: asked,
        fix: fix(),
        /* WHAT THE DRIVE IS ALLOWED TO DO. Named here so the answer is one
           thing rather than a condition re-derived at each call site. */
        canSee: state.camera === 'granted',
        canLocate: state.geo === 'granted' && !!lastFix,
      };
    }

    /* ---- the ask, and why it is shaped like this ------------------------
     *
     * MUST be called from inside a real user gesture. All three requests are
     * STARTED before the first `await`, because the gesture is spent by the
     * first one to yield and iOS will refuse whatever is asked for after that.
     * The promises are collected and settled afterwards; the order they
     * RESOLVE in does not matter and the order they are MADE in is everything.
     */
    function request(opts) {
      opts = opts || {};
      asked = true;

      if (!secure()) {
        // Nothing can be asked for. Say so as its own state rather than
        // letting three prompts fail one after another with no explanation.
        set('camera', 'insecure', 'page is not a secure context');
        set('mic', 'insecure', 'page is not a secure context');
        set('geo', 'insecure', 'page is not a secure context');
        return Promise.resolve(snapshot());
      }

      var jobs = [];
      var md = nav.mediaDevices;

      // 1. Camera.
      if (!md || !md.getUserMedia) {
        set('camera', 'unavailable', 'no mediaDevices');
      } else if (opts.skipCamera) {
        // A clip is loaded: the drive has a picture and does not need the
        // camera. Asking anyway would train the driver to dismiss prompts.
        set('camera', 'granted', 'clip source; camera not required');
      } else {
        jobs.push(md.getUserMedia({ video: opts.video || true })
          .then(function (stream) {
            set('camera', 'granted');
            // Handed to the caller if it wants it; otherwise released at once.
            // Holding an unused camera open is a battery cost and a red dot.
            if (typeof opts.onCameraStream === 'function') {
              try { opts.onCameraStream(stream); return; } catch (e) {}
            }
            stopStream(stream);
          })
          .catch(function (e) {
            set('camera', nameToState(e), errText(e));
          }));
      }

      // 2. Microphone.
      if (!md || !md.getUserMedia) {
        set('mic', 'unavailable', 'no mediaDevices');
      } else {
        jobs.push(md.getUserMedia({ audio: true })
          .then(function (stream) {
            set('mic', 'granted');
            if (typeof opts.onMicStream === 'function') {
              try { opts.onMicStream(stream); return; } catch (e) {}
            }
            stopStream(stream);
          })
          .catch(function (e) {
            set('mic', nameToState(e), errText(e));
          }));
      }

      // 3. Position. `getCurrentPosition` is what draws the prompt; the
      // long-lived watch is started by the caller once this says granted.
      if (!nav.geolocation || !nav.geolocation.getCurrentPosition) {
        set('geo', 'unavailable', 'no geolocation API');
      } else {
        jobs.push(new Promise(function (resolve) {
          var settled = false;
          var done = function () { if (settled) return true; settled = true; return false; };
          try {
            nav.geolocation.getCurrentPosition(
              function (pos) {
                if (done()) return;
                noteFix(pos);
                set('geo', 'granted');
                resolve();
              },
              function (err) {
                if (done()) return;
                // code 1 PERMISSION_DENIED, 2 POSITION_UNAVAILABLE, 3 TIMEOUT.
                // Only the first is a refusal; the other two are a phone that
                // has not got a fix YET, which is not the driver saying no and
                // must not be reported as though it were.
                var code = err && err.code;
                set('geo', code === 1 ? 'denied' : 'unknown',
                    errText(err) || ('geolocation code ' + code));
                resolve();
              },
              { enableHighAccuracy: true,
                timeout: opts.geoTimeoutMs || 12000,
                maximumAge: 0 });
          } catch (e) {
            set('geo', 'unavailable', errText(e));
            resolve();
          }
        }));
      }

      return Promise.all(jobs).then(snapshot, snapshot);
    }

    function stopStream(stream) {
      try {
        var tracks = stream && stream.getTracks ? stream.getTracks() : [];
        for (var i = 0; i < tracks.length; i++) tracks[i].stop();
      } catch (e) {}
    }

    function nameToState(e) {
      var n = (e && (e.name || e.code)) || '';
      if (n === 'NotAllowedError' || n === 'PermissionDeniedError') return 'denied';
      if (n === 'NotFoundError' || n === 'DevicesNotFoundError') return 'unavailable';
      if (n === 'NotReadableError' || n === 'TrackStartError') return 'unavailable';
      return 'denied';
    }

    function errText(e) {
      if (!e) return null;
      return String(e.message || e.name || e).slice(0, 120);
    }

    /* A best-effort read WITHOUT prompting, for painting the checklist before
       the driver has pressed anything. Permissions.query is not everywhere and
       does not cover the camera on every browser, so an unknown stays unknown
       rather than being guessed at. */
    function probe() {
      if (!secure()) {
        set('camera', 'insecure'); set('mic', 'insecure'); set('geo', 'insecure');
        return Promise.resolve(snapshot());
      }
      if (!nav.permissions || !nav.permissions.query) {
        return Promise.resolve(snapshot());
      }
      var want = [['camera', 'camera'], ['mic', 'microphone'], ['geo', 'geolocation']];
      var jobs = want.map(function (pair) {
        return nav.permissions.query({ name: pair[1] })
          .then(function (res) {
            if (res.state === 'granted') set(pair[0], 'granted');
            else if (res.state === 'denied') set(pair[0], 'denied');
            // 'prompt' stays unknown: it has not been decided yet, and
            // colouring it either way would be a claim nobody made.
            if (typeof res.addEventListener === 'function') {
              res.addEventListener('change', function () {
                if (res.state === 'granted') set(pair[0], 'granted');
                else if (res.state === 'denied') set(pair[0], 'denied');
                else set(pair[0], 'unknown');
              });
            }
          })
          .catch(function () { /* unsupported name: leave it unknown */ });
      });
      return Promise.all(jobs).then(snapshot, snapshot);
    }

    return {
      STATES: STATES,
      request: request,
      probe: probe,
      snapshot: snapshot,
      noteFix: noteFix,
      fix: fix,
      secure: secure,
      onChange: function (fn) { if (typeof fn === 'function') listeners.push(fn); },
      // Tests drive the whole thing through a fake navigator.
      _set: set,
      _reset: function () {
        state = { camera: 'unknown', mic: 'unknown', geo: 'unknown' };
        detail = { camera: null, mic: null, geo: null };
        lastFix = null; asked = false;
      },
    };
  }

  root.RIO = root.RIO || {};
  root.RIO.makePermissions = makePermissions;
  if (!root.RIO.permissions) root.RIO.permissions = makePermissions();

  if (typeof module === 'object' && module.exports) {
    module.exports = { makePermissions: makePermissions };
  }
}(typeof self !== 'undefined' ? self : this));
