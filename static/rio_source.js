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
    emit();
    return kind();
  }

  /* ...and the only thing that takes it away. Deliberately explicit: a clip
     that vanished because some other part of the page wanted a camera is the
     bug this file exists for, running in the opposite direction. */
  function useCamera() {
    var p = byId('preview');
    if (p) {
      try { p.pause(); } catch (e) {}
    }
    state.kind = CAMERA;
    state.clipUrl = '';
    state.clipName = '';
    emit();
    return kind();
  }

  function isClip() { return kind() === CLIP; }

  /* THE ONE ACQUISITION. Every start path calls this and none of them calls
     getUserMedia.
     -> Promise<{kind, element, stream, error}>
     `stream` is null for a clip, and that is not a failure: there is nothing
     to stop afterwards and nothing to release. */
  function startFeed() {
    var k = kind();

    if (k === CLIP) {
      var clip = byId('preview');
      if (!clip) {
        return Promise.resolve({ kind: NONE, element: null, stream: null,
                                 error: 'clip element missing' });
      }
      // The clip must be RUNNING for frames to advance; a paused element hands
      // back the same frame forever, which reads downstream as a stopped car.
      var played = null;
      try { played = clip.play(); } catch (e) { played = null; }
      return Promise.resolve(played)
        .catch(function () { /* autoplay refusal is not fatal: the driver can press play */ })
        .then(function () {
          return { kind: CLIP, element: clip, stream: null, error: null };
        });
    }

    if (k === NONE) {
      return Promise.resolve({ kind: NONE, element: null, stream: null,
                               error: 'no camera on this page' });
    }

    return root.navigator.mediaDevices
      .getUserMedia({ video: { facingMode: state.facing }, audio: false })
      .then(function (stream) {
        var v = byId('video');
        if (v) {
          v.srcObject = stream;
          try {
            var p = v.play();
            if (p && p.catch) p.catch(function () {});
          } catch (e) {}
        }
        return { kind: CAMERA, element: v, stream: stream, error: null };
      });
  }

  /* Release whatever startFeed acquired. A clip feed owns nothing, so this is
     a no-op for one — which is why callers can call it unconditionally. */
  function stopFeed(feed) {
    if (feed && feed.stream) {
      try { feed.stream.getTracks().forEach(function (t) { t.stop(); }); }
      catch (e) {}
    }
    if (feed && feed.kind === CAMERA) {
      var v = byId('video');
      if (v) v.srcObject = null;
    }
  }

  root.RIO = root.RIO || {};
  root.RIO.source = {
    CAMERA: CAMERA, CLIP: CLIP, NONE: NONE,
    kind: kind, name: name, label: label, element: element,
    isClip: isClip, facing: function () { return state.facing; },
    setClip: setClip, useCamera: useCamera,
    startFeed: startFeed, stopFeed: stopFeed,
    onChange: onChange,
    // Tests, and the panel's own reset paths.
    _reset: function () { state.kind = CAMERA; state.clipUrl = ''; state.clipName = ''; },
  };

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = root.RIO.source;
  }
})(typeof window !== 'undefined' ? window : globalThis);
