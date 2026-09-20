/* rio_source.js — WHICH PICTURE IS RIO LOOKING AT. One owner, one answer.
 *
 * The bug this exists to end: a driver uploads a clip, presses Start Drive (or
 * starts a live conversation), and the page opens the phone's camera instead.
 * The clip is still on screen, the overlay still draws on it, and the frames
 * being analysed are of the driver's face.
 *
 * It happened because "where do frames come from" was never a decision
 * anything owned. It was implied, three times over, by three code paths that
 * each independently called getUserMedia the moment they needed a picture:
 *
 *   startDrive()        the drive loop
 *   startLiveFrames()   the live session's own feed, added so RIO can see
 *                       during a conversation with no drive running — and the
 *                       one that introduced the regression, because it starts
 *                       "its own feed" without asking whether a source already
 *                       exists
 *   the manual frame button
 *
 * Three callers, no state, and therefore no way for an upload to be heard by
 * any of them. So the state is explicit here and there is exactly one function
 * that acquires a picture. A caller does not decide what to open; it asks for
 * the feed and is given whatever the source is.
 *
 * SOURCE is camera | clip | none:
 *
 *   clip    a file the driver uploaded. Set by the upload handler, cleared
 *           only by an explicit "use camera" — never implicitly, and never by
 *           a start path that would rather have a camera.
 *   camera  the rear-facing camera (see FACING). The default.
 *   none    no clip and no getUserMedia on this page at all (an insecure
 *           context, usually http:// on a phone). A real answer, not an
 *           error: RIO cannot see, and the look tool says so.
 *
 * Kept in a file of its own rather than inline in the page because this is the
 * decision that was wrong, and a decision that can only be tested by opening a
 * browser is a decision that gets tested by opening a browser once.
 */
(function (root) {
  'use strict';

  var CAMERA = 'camera', CLIP = 'clip', NONE = 'none';

  // Rear, always, unless something deliberately changes it. A dashcam view of
  // the driver's face is not a road, and every consumer of these frames --
  // headway, the corridor, the depth model, the look tool -- assumes forward.
  var FACING = 'environment';

  var state = {
    kind: CAMERA,          // what the driver has CHOSEN
    clipUrl: '',
    clipName: '',
    facing: FACING,
  };
  var subs = [];

  /* THE ONE LIVE PIPELINE, AND EVERY HAND ON IT.
   *
   * startFeed() used to acquire, unconditionally, once per caller. Three
   * callers meant three feeds were possible and the owner knew about none of
   * them: on 2026-09-20 a drive and a clip's caption watcher fed the same
   * session half a second apart, 640x480 against 1920x1080, and perception
   * answered from whichever frame arrived last.
   *
   * So there is ONE feed object for the page and callers take a HANDLE on it.
   * A second caller does not get a second camera; it gets the same one, and
   * stopFeed() releases the device only when the last hand comes off. Two
   * active sources are not unlikely here, they are unrepresentable: there is
   * one variable and it holds one feed.
   *
   * `holders` is a list of names rather than a count because "who is still
   * holding the camera" is the question asked when it will not turn off, and
   * a number cannot answer it. */
  var liveFeed = null;
  var faults = [];            // loud, kept, and readable by the page and tests

  function fault(code, detail) {
    var rec = { code: code, detail: detail || {}, at: Date.now() };
    faults.push(rec);
    if (faults.length > 20) faults.shift();
    try {
      if (root.console && console.error) {
        console.error('[source] ' + code + ' ' + JSON.stringify(rec.detail));
      }
    } catch (e) {}
    /* AND INTO THE DRIVE'S OWN RECORD. A fault that exists only in a console
       nobody has open is a fault nobody has. RIO.pageMark is the page's own
       session-log hook; absent (a test, an older page), the console line above
       is all there is and that is still better than silence. */
    try {
      var m = root.RIO && (root.RIO.pageMark || root.RIO.mark);
      if (typeof m === 'function') m('SOURCE_' + code, rec.detail);
    } catch (e) {}
    return rec;
  }

  function doc() {
    return root.document && root.document.getElementById ? root.document : null;
  }

  function byId(id) {
    var d = doc();
    return d ? d.getElementById(id) : null;
  }

  function mediaAvailable() {
    return !!(root.navigator && root.navigator.mediaDevices
              && root.navigator.mediaDevices.getUserMedia);
  }

  /* What the source IS, which is not always what was chosen: a page with no
     getUserMedia has no camera to fall back to, and saying "camera" there
     would make every start path fail at the same place for the same reason
     without ever saying why. */
  function kind() {
    if (state.kind === CLIP && state.clipUrl) return CLIP;
    return mediaAvailable() ? CAMERA : NONE;
  }

  function name() {
    if (kind() === CLIP) return state.clipName || 'clip';
    if (kind() === CAMERA) {
      return state.facing === 'user' ? 'FRONT CAM' : 'REAR CAM';
    }
    return 'NO SOURCE';
  }

  function label() {
    return kind() === CLIP ? 'CLIP: ' + name() : name();
  }

  /* The element frames are captured FROM. Not the element that happens to be
     visible: with a clip loaded the live <video> is empty and drawing it would
     produce black frames that look exactly like a dark road. */
  function element() {
    return kind() === CLIP ? byId('preview') : byId('video');
  }

  function emit() {
    var snapshot = { kind: kind(), name: name(), label: label(),
                     facing: state.facing };
    for (var i = 0; i < subs.length; i++) {
      try { subs[i](snapshot); } catch (e) { /* a listener must not break a source change */ }
    }
  }

  function onChange(fn) {
    if (typeof fn === 'function') {
      subs.push(fn);
      try { fn({ kind: kind(), name: name(), label: label(), facing: state.facing }); }
      catch (e) {}
    }
  }

  /* An upload. This is the ONLY thing that makes a clip the source, and it is
     sticky on purpose: the next Start Drive uses it, and so does the next live
     session, until the driver says otherwise. */
  function setClip(url, fileName) {
    state.kind = CLIP;
    state.clipUrl = String(url || '');
    state.clipName = String(fileName || '') || 'clip';
    /* AND THE FEED THAT IS ALREADY RUNNING MOVES WITH IT.
       Choosing a clip while a drive is on the camera used to change the answer
       for everything EXCEPT the loop that was already holding a camera feed --
       which is one of the two ways a session ends up with two sources. The
       choice is the choice: the camera is released here, now, and every holder
       of the feed is looking at the clip on its next capture. */
    retarget();
    emit();
    return kind();
  }

  /* ...and the only thing that takes it away. Deliberately explicit: a clip
     that vanished because some other part of the page wanted a camera is the
     bug this file exists for, running in the opposite direction. */
  function useCamera() {
    /* THE ELEMENT IS EMPTIED HERE, by the owner, and not only by whichever
       button happened to call this.
       Since startFeed() adopts a clip it finds on the screen (see reconcile),
       leaving a src on the element would make the next acquisition put the
       clip straight back -- silently undoing the one explicit choice this
       module has. "The way back is explicit" has to mean the way back stays. */
    var p = byId('preview');
    if (p) {
      try { p.pause(); } catch (e) {}
      try {
        p.src = '';
        if (typeof p.removeAttribute === 'function') p.removeAttribute('src');
        if (typeof p.load === 'function') p.load();
      } catch (e) {}
    }
    /* ...AND THE DRIVER'S SELECTION, which is now evidence in its own right.
       clipLoaded() reads the upload input, so leaving the file in it would make
       the next acquisition adopt the clip again and the camera unopenable —
       turning the one explicit choice in this module into one that cannot be
       made. */
    var up = byId('upload');
    if (up) { try { up.value = ''; } catch (e) {} }
    state.kind = CAMERA;
    state.clipUrl = '';
    state.clipName = '';
    retarget();
    emit();
    return kind();
  }

  /* THE SOURCE CHANGED UNDER A RUNNING FEED.
   *
   * Nothing is torn down and rebuilt: the holders keep the same feed object,
   * whose `element` is a getter that follows the owner (see makeFeed), so a
   * drive, a conversation and the caption loop all move together. What DOES
   * happen here is the release of whatever the old source held -- a camera
   * stays on otherwise, which on a desk is a lit indicator and in the log is a
   * second producer waiting to be resumed.
   *
   * Asynchronous in the clip -> camera direction only, because that one needs
   * a device. Camera -> clip is immediate, which is the direction that matters:
   * the clip must win the instant it is chosen. */
  function retarget() {
    if (!liveFeed) return Promise.resolve(null);
    var want = kind();
    if (liveFeed.kind === want) return Promise.resolve(liveFeed);
    var from = liveFeed.kind;
    releaseDevice(liveFeed);
    liveFeed.kind = want;
    liveFeed.stream = null;
    if (want === CLIP) {
      playClip();
      return Promise.resolve(liveFeed);
    }
    if (want === NONE) return Promise.resolve(liveFeed);
    return acquireCamera().then(function (stream) {
      if (liveFeed) liveFeed.stream = stream;
      return liveFeed;
    }, function (e) {
      fault('RETARGET_FAILED', { from: from, to: want,
                                 error: (e && e.name) || String(e) });
      return liveFeed;
    });
  }

  function isClip() { return kind() === CLIP; }

  /* WHAT A CAMERA IS ASKED FOR, IN ONE PLACE.
   *
   * The permission probe in the Start Drive tap used to ask for bare
   * `video: true` -- it read `RIO.source.videoConstraints`, which did not
   * exist, and fell through to the default. `video: true` is whatever the
   * device feels like: 640x480 on a desktop webcam, the FRONT camera on a
   * phone. So the probe opened one camera and startFeed then opened a
   * different one, which is two acquisitions for one picture and two different
   * pictures to choose between. Both ask through here now. */
  function videoConstraints() {
    return { facingMode: state.facing };
  }

  function playClip() {
    var clip = byId('preview');
    if (!clip) return Promise.resolve(null);
    // The clip must be RUNNING for frames to advance; a paused element hands
    // back the same frame forever, which reads downstream as a stopped car.
    var played = null;
    try { played = clip.play(); } catch (e) { played = null; }
    return Promise.resolve(played)
      .catch(function () { /* autoplay refusal is not fatal: the driver can press play */ })
      .then(function () { return clip; });
  }

  function acquireCamera() {
    return root.navigator.mediaDevices
      .getUserMedia({ video: videoConstraints(), audio: false })
      .then(function (stream) {
        var v = byId('video');
        if (v) {
          v.srcObject = stream;
          try {
            var p = v.play();
            if (p && p.catch) p.catch(function () {});
          } catch (e) {}
        }
        return stream;
      });
  }

  function releaseDevice(feed) {
    if (feed && feed.stream) {
      try { feed.stream.getTracks().forEach(function (t) { t.stop(); }); }
      catch (e) {}
    }
    if (feed && feed.kind === CAMERA) {
      var v = byId('video');
      if (v) v.srcObject = null;
    }
    if (feed && feed.kind === CLIP) {
      var c = byId('preview');
      if (c) { try { c.pause(); } catch (e) {} }
    }
  }

  /* The feed handed to callers. `element` is a GETTER, not a value, and that
     is the whole of how a source change reaches a loop that is already
     running: rio_frames.js asks for the element on every frame, so the next
     capture after an upload comes from the clip without anybody restarting
     anything. It was a fixed value, captured at acquisition, and a drive that
     had one could not be told about anything. */
  function makeFeed(k, stream) {
    var feed = { kind: k, stream: stream || null, error: null, holders: [] };
    try {
      Object.defineProperty(feed, 'element', {
        enumerable: true,
        get: function () {
          return feed.kind === CLIP ? byId('preview')
               : feed.kind === CAMERA ? byId('video') : null;
        },
      });
    } catch (e) {
      feed.element = k === CLIP ? byId('preview') : byId('video');
    }
    return feed;
  }

  /* WHAT IS ON THE SCREEN, RECONCILED WITH WHAT THE OWNER BELIEVES.
   *
   * A clip that is loaded in the page but not registered here is the exact
   * state the driver described on 2026-09-20: a clip on the glass, Start Drive
   * pressed, and the camera opened anyway -- because every start path asks
   * `kind()`, and `kind()` only knows what setClip() was told. One missed call
   * (an upload handler that threw before it got there, a clip put on the
   * element by anything other than the upload input, a cached older page)
   * and the owner is confidently wrong.
   *
   * So the acquisition does not trust the record alone: if #preview has a
   * source in it, the driver has put a picture on the screen, and that is the
   * selected source whatever this module last recorded. Adopting it is loud --
   * something failed to say so, and that is worth a line -- and it happens
   * BEFORE any device is opened, which is what makes "the camera is never
   * opened over a clip" true rather than intended. */
  /* WHAT THE DRIVER CHOSE, from the one place that cannot have been lost.
   *
   * `#upload`.files holds the File the driver picked in the dialog. It is the
   * browser's own record of the choice, it is set before any of this page's
   * JavaScript runs, and it survives every handler that was supposed to act on
   * it failing -- which the element's `src` does not, because something has to
   * put a src there.
   *
   * THE DRIVE OF 2026-09-20, 20:10:09: DRIVE_SOURCE recorded `camera`, faults
   * empty, so reconcile() looked at #preview and found no src. Either no clip
   * was re-selected after the hard reload twelve seconds earlier, or one was
   * and nothing registered it. The element could not tell those apart. This
   * can: if the input holds a video, the driver chose a clip, full stop. */
  function chosenClipFile() {
    var up = byId('upload');
    var f = up && up.files && up.files.length ? up.files[0] : null;
    if (!f) return null;
    var type = String(f.type || '');
    if (type.indexOf('video/') !== 0) return null;      // a still is not a source
    return f;
  }

  /* IS A CLIP LOADED? Asked of every piece of evidence there is, because the
     answer gates a device and a wrong `no` opens a camera over the driver's
     picture. In order of directness: what this module recorded, what is in the
     element, and what the driver actually picked. */
  function clipLoaded() {
    if (state.kind === CLIP && state.clipUrl) return 'record';
    var clip = byId('preview');
    var src = clip ? (clip.currentSrc || clip.src || '') : '';
    if (src) return 'element';
    if (chosenClipFile()) return 'upload_input';
    return null;
  }

  function reconcile() {
    if (kind() === CLIP) return;
    var why = clipLoaded();
    if (!why) return;

    var clip = byId('preview');
    var src = clip ? (clip.currentSrc || clip.src || '') : '';
    var name_ = state.clipName || 'clip';

    /* THE FILE IS STILL THERE, so the clip can be rebuilt rather than merely
       reported. A File in the input is all URL.createObjectURL needs, so a
       driver whose upload handler died still gets a drive on their clip
       instead of a camera and an apology. */
    if (!src) {
      var f = chosenClipFile();
      if (!f || !(root.URL && root.URL.createObjectURL)) return;
      try { src = root.URL.createObjectURL(f); } catch (e) { return; }
      name_ = f.name || name_;
      if (clip) {
        clip.src = src;
        try { if (typeof clip.load === 'function') clip.load(); } catch (e) {}
      }
    }

    fault('ADOPTED_VISIBLE_CLIP', { src: String(src).slice(0, 80),
                                    was: kind(), evidence: why,
                                    name: name_ });
    state.kind = CLIP;
    state.clipUrl = String(src);
    state.clipName = name_;
    retarget();
    emit();
  }

  /* ---- THE GUARD -------------------------------------------------------
   *
   * "Assert that no getUserMedia call can happen while a clip is the adopted
   * source, and make the violation fail loudly rather than reconcile after the
   * fact."
   *
   * Reconciling is a repair, and a repair runs in the one path that remembered
   * to ask. THREE places on this page request a camera -- this module, the
   * permission probe in the Start Drive tap, and the manual frame button -- and
   * the history of this file is callers who each independently decided they
   * needed a picture. A fourth will be written.
   *
   * So the rule is enforced at the only place all of them go through: the
   * browser's own getUserMedia. A request for VIDEO while a clip is loaded is
   * refused here, before any device is touched, and the refusal is a fault with
   * the caller's stack in it. Audio is untouched -- the microphone must work on
   * a clip drive exactly as it does on a camera one, and a guard that took the
   * microphone away would be a worse bug than the one it fixes.
   *
   * This is deliberately a patch on the platform call and not a convention. A
   * convention is what was in place: `skipCamera: RIO.source.kind() === 'clip'`,
   * correct, and evaluated by one caller out of three.
   */
  var originalGUM = null;

  function wantsVideo(constraints) {
    if (!constraints) return false;
    var v = constraints.video;
    return !!v;                    // true, or a constraints object
  }

  function installGuard() {
    if (originalGUM) return true;
    var md = root.navigator && root.navigator.mediaDevices;
    if (!md || typeof md.getUserMedia !== 'function') return false;
    originalGUM = md.getUserMedia.bind(md);
    md.getUserMedia = function (constraints) {
      if (wantsVideo(constraints)) {
        var why = clipLoaded();
        if (why) {
          var detail = { evidence: why, constraints: describe(constraints),
                         from: callerLine() };
          fault('CAMERA_REFUSED_CLIP_LOADED', detail);
          var err = new Error('RIO.source: a clip is the source (' + why
                              + ') — no camera may be opened. '
                              + 'See static/rio_source.js.');
          err.name = 'NotAllowedError';
          return Promise.reject(err);
        }
      }
      return originalGUM(constraints);
    };
    return true;
  }

  function uninstallGuard() {
    var md = root.navigator && root.navigator.mediaDevices;
    if (originalGUM && md) md.getUserMedia = originalGUM;
    originalGUM = null;
  }

  function describe(c) {
    try { return JSON.stringify(c).slice(0, 120); } catch (e) { return '?'; }
  }

  /* WHO ASKED. A refusal that does not name the caller is a refusal somebody
     has to go and find, and the whole point of the guard is that the caller is
     something nobody has thought about yet. */
  function callerLine() {
    try {
      var lines = String(new Error().stack || '').split('\n');
      for (var i = 0; i < lines.length; i++) {
        var l = lines[i];
        if (l.indexOf('rio_source.js') >= 0) continue;
        if (/at (Object\.)?(installGuard|getUserMedia|callerLine)/.test(l)) continue;
        if (/^\s*at /.test(l)) return l.trim().slice(0, 160);
      }
    } catch (e) {}
    return null;
  }

  /* THE ONE ACQUISITION. Every start path calls this and none of them calls
     getUserMedia.
     -> Promise<{kind, element, stream, error}>
     `stream` is null for a clip, and that is not a failure: there is nothing
     to stop afterwards and nothing to release.

     A SECOND CALLER GETS THE SAME FEED. It does not get a second camera and it
     does not get a second capture pipeline: it gets a handle on the one that
     is already open, and the device is released when the last handle is
     dropped. `who` names the caller so that "what is still holding the camera"
     has an answer. */
  function startFeed(who) {
    reconcile();
    var k = kind();
    var name_ = String(who || 'unnamed');

    if (liveFeed) {
      /* Already running. The kinds cannot disagree -- retarget() moves the one
         feed whenever the source changes -- but if they ever did, that is two
         sources by definition and the feed is moved rather than duplicated. */
      if (liveFeed.kind !== k) {
        fault('FEED_KIND_DRIFT', { feed: liveFeed.kind, source: k,
                                   holders: liveFeed.holders.slice() });
        return retarget().then(function () {
          liveFeed.holders.push(name_);
          return liveFeed;
        });
      }
      liveFeed.holders.push(name_);
      /* A NEW HAND ON A STOPPED CLIP STILL NEEDS IT MOVING. The feed being
         live is not the same fact as frames advancing: a clip paused by the
         driver, or by the end of the last holder's use of it, hands back one
         frame forever, and downstream that is a stopped car rather than a
         stopped clip. */
      if (liveFeed.kind === CLIP) {
        return playClip().then(function () { return liveFeed; });
      }
      return Promise.resolve(liveFeed);
    }

    if (k === CLIP) {
      var clip = byId('preview');
      if (!clip) {
        return Promise.resolve({ kind: NONE, element: null, stream: null,
                                 error: 'clip element missing' });
      }
      return playClip().then(function () {
        liveFeed = makeFeed(CLIP, null);
        liveFeed.holders.push(name_);
        return liveFeed;
      });
    }

    if (k === NONE) {
      return Promise.resolve({ kind: NONE, element: null, stream: null,
                               error: 'no camera on this page' });
    }

    return acquireCamera().then(function (stream) {
      liveFeed = makeFeed(CAMERA, stream);
      liveFeed.holders.push(name_);
      return liveFeed;
    });
  }

  /* One source per session, asserted rather than assumed. -> {ok, kind,
     holders, faults}. The page puts this on the Feed chip and the suites
     assert on it; anything that is not `ok` is a state this module was built
     to make impossible, so it is reported as a fault and not as a value. */
  function liveState() {
    return {
      ok: !liveFeed || liveFeed.kind === kind(),
      live: liveFeed ? 1 : 0,
      kind: liveFeed ? liveFeed.kind : null,
      source: kind(),
      holders: liveFeed ? liveFeed.holders.slice() : [],
      /* COMPACT ON PURPOSE. This goes into the DRIVE_SOURCE mark, whose note is
         truncated at 700 characters — and a truncated JSON note is not a
         shorter answer, it is an unparseable one. A code and the evidence
         behind it is what the log needs; the full records, stack lines and all,
         stay on RIO.source.faults() for the console and the suites. */
      faults: faults.map(function (f) {
        return { code: f.code, evidence: f.detail.evidence || null };
      }),
      fault_count: faults.length,
    };
  }

  /* THE CAMERA CAME BACK, OR IT DID NOT.
   *
   * A MediaStreamTrack can end under a running page: another app takes the
   * camera, iOS revokes it for a phone call, the device is unplugged. The
   * element keeps its srcObject and reports videoWidth 0 from then on, so
   * every capture silently returns nothing and the drive has no pictures with
   * nothing anywhere saying why.
   *
   * This is the ONLY thing that can fix that, and it lives here for the same
   * reason startFeed does: one acquisition, one file. It re-runs the same
   * request against the same facing mode and re-attaches the element.
   *
   * A clip needs none of it -- there is no track to lose -- and says so by
   * resolving true without touching anything.
   */
  function reacquire() {
    var k = kind();
    if (k !== CAMERA) return Promise.resolve(k !== NONE);
    var v = byId('video');
    // Release what is left before asking again: iOS will hand back the same
    // dead track otherwise, and a second live track on one element is two
    // camera pipelines for one picture.
    try {
      if (v && v.srcObject && v.srcObject.getTracks) {
        v.srcObject.getTracks().forEach(function (t) { try { t.stop(); } catch (e) {} });
      }
    } catch (e) {}
    if (v) v.srcObject = null;
    /* Straight to the device, not through startFeed(): this is the SAME feed
       coming back, not a new hand on it, and going through startFeed would
       return the live feed untouched (it is still there — it is its track that
       died) and never re-acquire anything. */
    return acquireCamera().then(function (stream) {
      if (liveFeed && liveFeed.kind === CAMERA) liveFeed.stream = stream;
      return !!stream;
    }, function () { return false; });
  }

  /* Release one HANDLE on the feed. The device goes when the last one does.
   *
   * A clip feed owns no device, so this is nearly a no-op for one — which is
   * why callers can call it unconditionally. What it is never allowed to do is
   * take the camera away from a loop that is still using it: the conversation
   * ending while a drive runs used to be guarded against at every call site
   * ("if (!RIO.driving)"), which is the same decision made in three places by
   * three callers who cannot see each other. It is made here now. */
  function stopFeed(feed) {
    if (!feed) return;
    if (feed !== liveFeed) {
      // A feed from before a retarget, or a caller's own object. Release
      // anything it still holds and leave the live one alone.
      releaseDevice(feed);
      return;
    }
    if (feed.holders.length) feed.holders.pop();
    if (feed.holders.length) return;        // somebody is still looking
    releaseDevice(feed);
    liveFeed = null;
  }

  root.RIO = root.RIO || {};
  root.RIO.source = {
    CAMERA: CAMERA, CLIP: CLIP, NONE: NONE,
    kind: kind, name: name, label: label, element: element,
    isClip: isClip, facing: function () { return state.facing; },
    videoConstraints: videoConstraints,
    /* IS A CLIP LOADED, on all the evidence — and it returns WHICH evidence,
       because "the record says so" and "the driver picked a file nothing acted
       on" are the same answer to the gate and different faults. */
    clipLoaded: clipLoaded,
    installGuard: installGuard, uninstallGuard: uninstallGuard,
    /* Called by a start path BEFORE it asks for permissions, because the
       permission probe opens a camera of its own and the decision about
       whether to ask for one at all depends on this. startFeed() calls it
       too; it is idempotent. */
    reconcile: reconcile,
    setClip: setClip, useCamera: useCamera,
    startFeed: startFeed, stopFeed: stopFeed, reacquire: reacquire,
    onChange: onChange,
    // One source per session, asked rather than assumed. See liveState().
    liveState: liveState,
    faults: function () { return faults.slice(); },
    // Tests, and the panel's own reset paths.
    _reset: function () {
      state.kind = CAMERA; state.clipUrl = ''; state.clipName = '';
      liveFeed = null; faults = [];
    },
  };

  /* INSTALLED AT LOAD, not at the first acquisition. The call this guards is
     reachable from anywhere in the page from the moment the page has a DOM,
     and a guard that arms when the owner is first ASKED is a guard that is
     absent for exactly the callers that never ask it anything. */
  installGuard();

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = root.RIO.source;
  }
})(typeof window !== 'undefined' ? window : globalThis);
