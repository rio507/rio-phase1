/* The permission flow, under node, with a fake navigator.
 *
 *     node tools/permissions_selftest.js
 *
 * THE DRIVE THIS EXISTS BECAUSE OF (2026-09-16). On a phone the geolocation
 * prompt never appeared. Place search came back with somewhere far away, "news
 * round here" resolved to the wrong area, and nothing on screen said why —
 * which is the part that cost the time, because a missing permission and a bad
 * search look identical from the passenger seat.
 *
 * Four things are asserted here and each is a way that silence comes back:
 *
 *   GESTURE      all three requests are STARTED before the first await. A
 *                gesture does not survive one, and anything asked for after
 *                the first network round trip is asked for outside the tap.
 *                This is checked by counting calls synchronously, which is the
 *                only way to check it without a phone.
 *   VISIBLE      a denial is a STATE, not an absence. Every refusal path ends
 *                in granted/denied/unavailable/insecure and never in silence.
 *   INSECURE     over plain http nothing can even be asked. That is its own
 *                state rather than three prompts failing one after another.
 *   NO FALLBACK  canLocate is false until a real fix has arrived, and the age
 *                of a fix that never came is not a number.
 *
 * Node, not a browser, on purpose: what is being tested is the ORDER of calls
 * and the mapping of failures to states, and both are invisible in a browser
 * that has already granted everything.
 *
 * Exit code is the number of failures.
 */
'use strict';

const path = require('path');

const root = {};
global.self = root;
require(path.join(__dirname, '..', 'static', 'rio_permissions.js'));
const { makePermissions } = root.RIO;

let fails = 0;
function ok(name, cond, extra) {
  if (cond) { console.log('  ok   ' + name); }
  else { fails++; console.log('  FAIL ' + name + (extra !== undefined ? '  ' + extra : '')); }
}
function section(t) { console.log('\n== ' + t); }

/* A navigator that records WHEN each request was made, and never settles until
   told to. The recording is the point: a promise that resolves later is fine,
   a call that is made later is the bug. */
function fakeNav(opts) {
  opts = opts || {};
  const calls = [];
  const settle = {};
  return {
    calls,
    settle,
    nav: {
      mediaDevices: opts.noMedia ? undefined : {
        getUserMedia(c) {
          calls.push(c && c.audio ? 'mic' : 'camera');
          return new Promise((res, rej) => {
            settle[c && c.audio ? 'mic' : 'camera'] = { res, rej };
          });
        },
      },
      geolocation: opts.noGeo ? undefined : {
        getCurrentPosition(succ, err) {
          calls.push('geo');
          settle.geo = { res: succ, rej: err };
        },
      },
      permissions: opts.permissions,
    },
  };
}

function stream() {
  let stopped = 0;
  return { getTracks: () => [{ stop: () => { stopped++; } }], stopped: () => stopped };
}

// ---------------------------------------------------------------------------
section('A. all three are asked inside the gesture, before any await');
{
  const f = fakeNav();
  const p = makePermissions(f.nav, { isSecureContext: true });
  const promise = p.request();
  // Synchronously after request() returns: every call must already be made.
  ok('camera asked synchronously', f.calls.indexOf('camera') >= 0, f.calls.join(','));
  ok('microphone asked synchronously', f.calls.indexOf('mic') >= 0, f.calls.join(','));
  ok('location asked synchronously', f.calls.indexOf('geo') >= 0, f.calls.join(','));
  ok('all three before anything resolved', f.calls.length === 3, f.calls.join(','));
  ok('and in the order the spec names', f.calls.join(',') === 'camera,mic,geo',
     f.calls.join(','));

  f.settle.camera.res(stream());
  f.settle.mic.res(stream());
  f.settle.geo.res({ coords: { latitude: 34.02, longitude: -118.49, accuracy: 12 } });
  promise.then((snap) => {
    section('B. a granted run reports what it got');
    ok('camera granted', snap.camera === 'granted', snap.camera);
    ok('mic granted', snap.mic === 'granted', snap.mic);
    ok('location granted', snap.geo === 'granted', snap.geo);
    ok('canSee is true', snap.canSee === true);
    ok('canLocate is true once a fix has arrived', snap.canLocate === true);
    ok('the fix carries its accuracy', snap.fix && snap.fix.accuracy_m === 12,
       JSON.stringify(snap.fix));
    ok('...and an age', snap.fix && typeof snap.fix.age_s === 'number');
    runDenied();
  });
}

function runDenied() {
  section('C. a denial is a state, never a silence');
  const f = fakeNav();
  const p = makePermissions(f.nav, { isSecureContext: true });
  const promise = p.request();
  const denied = new Error('denied'); denied.name = 'NotAllowedError';
  const missing = new Error('none'); missing.name = 'NotFoundError';
  f.settle.camera.rej(missing);
  f.settle.mic.rej(denied);
  f.settle.geo.rej({ code: 1, message: 'User denied Geolocation' });
  promise.then((snap) => {
    ok('a missing camera reads as unavailable', snap.camera === 'unavailable', snap.camera);
    ok('a refused microphone reads as denied', snap.mic === 'denied', snap.mic);
    ok('a refused location reads as denied', snap.geo === 'denied', snap.geo);
    ok('nothing is left at unknown', [snap.camera, snap.mic, snap.geo]
       .every((x) => x !== 'unknown'), JSON.stringify(snap));
    ok('each denial carries a reason', !!snap.detail.mic && !!snap.detail.geo,
       JSON.stringify(snap.detail));
    ok('canSee is false', snap.canSee === false);
    ok('canLocate is false', snap.canLocate === false);
    ok('and there is no fix to fall back to', snap.fix === null);
    runTimeout();
  });
}

function runTimeout() {
  section('D. a timeout is NOT a refusal');
  const f = fakeNav();
  const p = makePermissions(f.nav, { isSecureContext: true });
  const promise = p.request();
  f.settle.camera.res(stream());
  f.settle.mic.res(stream());
  // code 3 is TIMEOUT, code 2 POSITION_UNAVAILABLE: a phone that has not got a
  // fix yet, which is not the driver saying no and must not be recorded as it.
  f.settle.geo.rej({ code: 3, message: 'Timeout expired' });
  promise.then((snap) => {
    ok('a geolocation timeout is not "denied"', snap.geo !== 'denied', snap.geo);
    ok('...it stays unknown, so it can still arrive', snap.geo === 'unknown', snap.geo);
    ok('but location answers are still refused meanwhile',
       snap.canLocate === false);
    runInsecure();
  });
}

function runInsecure() {
  section('E. an insecure page cannot even ask, and says so');
  const f = fakeNav();
  const p = makePermissions(f.nav, { isSecureContext: false });
  p.request().then((snap) => {
    ok('nothing was requested at all', f.calls.length === 0, f.calls.join(','));
    ok('camera reads as insecure', snap.camera === 'insecure', snap.camera);
    ok('mic reads as insecure', snap.mic === 'insecure', snap.mic);
    ok('location reads as insecure', snap.geo === 'insecure', snap.geo);
    ok('canSee false', snap.canSee === false);
    ok('canLocate false', snap.canLocate === false);
    ok('and the page knows it is not secure', snap.secure === false);
    runNoApi();
  });
}

function runNoApi() {
  section('F. a browser missing the APIs entirely');
  const f = fakeNav({ noMedia: true, noGeo: true });
  const p = makePermissions(f.nav, { isSecureContext: true });
  p.request().then((snap) => {
    ok('camera unavailable', snap.camera === 'unavailable', snap.camera);
    ok('mic unavailable', snap.mic === 'unavailable', snap.mic);
    ok('location unavailable', snap.geo === 'unavailable', snap.geo);
    ok('nothing throws', true);
    runFix();
  });
}

function runFix() {
  section('G. no fix means no age, and no borrowing');
  const f = fakeNav();
  const p = makePermissions(f.nav, { isSecureContext: true });
  ok('before anything, there is no fix', p.fix() === null);
  ok('...and canLocate is false', p.snapshot().canLocate === false);

  // A watch delivering a position is what makes a fix real -- not arming it.
  p.noteFix({ coords: { latitude: 1, longitude: 2, accuracy: 8 } });
  const fx = p.fix();
  ok('a delivered fix becomes available', fx !== null);
  ok('...with an accuracy', fx && fx.accuracy_m === 8);
  ok('...and marks location granted', p.snapshot().geo === 'granted');
  ok('...so canLocate becomes true', p.snapshot().canLocate === true);

  // Garbage must not become a fix.
  p._reset();
  p.noteFix({ coords: { latitude: NaN, longitude: 2 } });
  ok('a NaN position is not a fix', p.fix() === null);
  p.noteFix(null);
  ok('a null position is not a fix', p.fix() === null);
  p.noteFix({ coords: { latitude: 5, longitude: 6 } });
  ok('a fix with no accuracy is still a fix', p.fix() !== null);
  ok('...and reports accuracy as unknown rather than zero',
     p.fix().accuracy_m === null, String(p.fix().accuracy_m));

  section('H. changes are announced, so the checklist can repaint');
  let seen = 0;
  const p2 = makePermissions(fakeNav().nav, { isSecureContext: true });
  p2.onChange(() => { seen++; });
  p2._set('geo', 'denied');
  ok('a state change notifies', seen > 0, String(seen));
  const before = seen;
  p2.noteFix({ coords: { latitude: 1, longitude: 2, accuracy: 5 } });
  ok('so does a new fix', seen > before, String(seen));

  done();
}

function done() {
  console.log('\n' + '-'.repeat(58));
  console.log('  ' + (fails ? 'FAIL' : 'PASS') + ': ' + fails + ' failure(s)');
  process.exit(fails);
}
