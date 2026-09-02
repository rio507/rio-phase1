/* source_selftest.js — which picture is RIO looking at?
 *
 *   node tools/source_selftest.js
 *
 * THE BUG. A driver uploads a clip, presses Start Drive, and the page opens
 * the phone's camera instead. The clip stays on screen, the overlay keeps
 * drawing on it, and the frames being analysed are of the driver's face. The
 * same thing happened when a live conversation was started, and that path is
 * where the regression came in: it was written as "the live session starts its
 * own feed", and a feed of its own is exactly what it must not have.
 *
 * WHAT IS ACTUALLY TESTED. static/rio_source.js owns the decision and performs
 * the one acquisition, so the behaviour is reachable from node with a stubbed
 * DOM and a stubbed navigator: with a clip loaded, nothing may reach
 * getUserMedia, and the element handed back must be the clip's. Without one,
 * the camera must be requested, rear-facing.
 *
 * That alone would pass while the page went on calling getUserMedia by itself,
 * so the second half of this file reads index.html and asserts that the start
 * paths go through the owner -- the check that would have caught the original
 * bug, since the bug WAS a start path acquiring its own feed.
 */
'use strict';

const fs = require('fs');
const path = require('path');

let checks = 0, failures = 0;
function ok(cond, what) {
  checks++;
  if (!cond) { failures++; console.log('  FAIL  ' + what); }
  else console.log('  ok    ' + what);
}
function section(name) { console.log('\n=== ' + name + ' ==='); }

/* A DOM with exactly the two elements that matter, and a navigator that counts
   every camera request and records what it was asked for. */
function stubPage() {
  const els = {
    video: { id: 'video', srcObject: null, videoWidth: 1280, videoHeight: 720,
             play: () => Promise.resolve() },
    preview: { id: 'preview', src: '', paused: true,
               videoWidth: 1280, videoHeight: 720,
               play() { this.paused = false; return Promise.resolve(); },
               pause() { this.paused = true; } },
  };
  const asked = [];
  global.window = {
    document: { getElementById: (id) => els[id] || null },
    navigator: {
      mediaDevices: {
        getUserMedia: (c) => {
          asked.push(c);
          return Promise.resolve({
            getTracks: () => [{ stop() { this.stopped = true; } }],
          });
        },
      },
    },
    RIO: {},
  };
  global.document = global.window.document;
  global.navigator = global.window.navigator;
  delete require.cache[require.resolve(
    path.join(__dirname, '..', 'static', 'rio_source.js'))];
  const src = require(path.join(__dirname, '..', 'static', 'rio_source.js'));
  return { src, els, asked };
}

(async function main() {
  // -------------------------------------------------------------------------
  section('no clip — the camera is the source, and it is the REAR one');
  {
    const { src, els, asked } = stubPage();
    ok(src.kind() === 'camera', 'a fresh page is on the camera');
    ok(src.label() === 'REAR CAM', 'and says so: ' + src.label());
    ok(src.element() === els.video, 'frames come from the live element');

    const feed = await src.startFeed();
    ok(asked.length === 1, 'starting a feed requests the camera exactly once');
    ok(asked[0] && asked[0].video && asked[0].video.facingMode === 'environment',
       'rear-facing — a view of the driver is not a road ('
       + JSON.stringify(asked[0]) + ')');
    ok(feed.kind === 'camera' && feed.element === els.video && !!feed.stream,
       'and the feed is the camera, with a stream to release later');
    ok(els.video.srcObject === feed.stream,
       'attached to the live element, which is what the page shows');
  }

  // -------------------------------------------------------------------------
  section('a clip is loaded — NOTHING may open a camera');
  {
    const { src, els, asked } = stubPage();
    src.setClip('blob:abc', 'coastal.mp4');

    ok(src.kind() === 'clip', 'uploading makes the clip the source');
    ok(src.label() === 'CLIP: coastal.mp4',
       'named in the HUD so the state is visible: ' + src.label());
    ok(src.element() === els.preview, 'frames come from the clip element');

    // START DRIVE.
    const drive = await src.startFeed();
    ok(asked.length === 0,
       'starting a drive opens NO camera — this is the bug, in one assertion');
    ok(drive.kind === 'clip' && drive.element === els.preview,
       'the drive feeds from the clip');
    ok(drive.stream === null,
       'and owns no stream, because there is nothing to release');
    ok(els.preview.paused === false,
       'the clip is playing — a paused element hands back one frame forever, '
       + 'which reads downstream as a stopped car');

    // A LIVE SESSION, started separately, exactly as toggleLive does.
    els.preview.pause();
    const live = await src.startFeed();
    ok(asked.length === 0,
       'starting a live session opens NO camera either — the path the '
       + 'regression came in through');
    ok(live.element === els.preview, 'it feeds from the same clip');
    ok(els.preview.paused === false, 'and starts it playing again');

    // ...and ending either one leaves the clip exactly where it was.
    src.stopFeed(live);
    ok(src.kind() === 'clip' && src.element() === els.preview,
       'stopping a feed does not silently drop the clip');
    ok(asked.length === 0, 'still no camera, after all of that');
  }

  // -------------------------------------------------------------------------
  section('the way back is explicit');
  {
    const { src, els, asked } = stubPage();
    src.setClip('blob:abc', 'coastal.mp4');
    ok(src.isClip(), 'clip loaded');

    src.useCamera();
    ok(src.kind() === 'camera', 'and only useCamera() takes it away');
    ok(els.preview.paused, 'stopping the clip as it goes');
    const feed = await src.startFeed();
    ok(asked.length === 1 && feed.kind === 'camera',
       'after which the camera is the source again');
  }

  // -------------------------------------------------------------------------
  section('no camera and no clip is an answer, not a crash');
  {
    const { src } = stubPage();
    delete global.window.navigator.mediaDevices;
    ok(src.kind() === 'none',
       'an insecure context has no camera to fall back to');
    ok(src.label() === 'NO SOURCE', 'and says so rather than lying');
    const feed = await src.startFeed();
    ok(feed.kind === 'none' && !!feed.error && !feed.element,
       'a feed request comes back with a reason (' + feed.error + ')');

    // A clip still works there, which is the whole point of a desk test.
    global.window.navigator.mediaDevices = undefined;
    src.setClip('blob:abc', 'coastal.mp4');
    ok(src.kind() === 'clip',
       'and a clip is still a source with no camera on the page at all');
  }

  // -------------------------------------------------------------------------
  section('subscribers see every change');
  {
    const { src } = stubPage();
    const seen = [];
    src.onChange(s => seen.push(s.label));
    ok(seen.length === 1 && seen[0] === 'REAR CAM',
       'a subscriber is told the current state immediately');
    src.setClip('blob:x', 'coastal.mp4');
    src.useCamera();
    ok(seen.join(' -> ') === 'REAR CAM -> CLIP: coastal.mp4 -> REAR CAM',
       'and every change after it (' + seen.join(' -> ') + ')');
  }

  // -------------------------------------------------------------------------
  section('the page asks the owner rather than deciding for itself');
  {
    const html = fs.readFileSync(
      path.join(__dirname, '..', 'static', 'index.html'), 'utf8');

    ok(/<script src="\/static\/rio_source\.js">/.test(html),
       'the page loads the source owner');

    // Every video getUserMedia in the page, with its surrounding line, so a
    // new one shows up here rather than in a car.
    const videoGrabs = [];
    const re = /navigator\.mediaDevices\.getUserMedia\(([^;]*)/g;
    let m;
    while ((m = re.exec(html)) !== null) {
      const arg = m[1].replace(/\s+/g, ' ').slice(0, 90);
      if (/audio: true/.test(arg) && !/video/.test(arg)) continue;   // the microphone
      videoGrabs.push(arg);
    }
    ok(videoGrabs.length === 1,
       'exactly one video camera request left in the page — the manual '
       + 'one-off frame button (' + videoGrabs.length + ' found)');
    ok(videoGrabs.every(a => /facingMode/.test(a)),
       'and it is rear-facing like the rest: ' + videoGrabs.join(' | '));

    const drive = html.slice(html.indexOf('async function startDrive'),
                             html.indexOf('async function endDrive'));
    ok(/RIO\.source\.startFeed\(\)/.test(drive),
       'Start Drive asks the owner for a feed');
    ok(!/getUserMedia/.test(drive),
       'and opens no camera of its own');

    const liveFeed = html.slice(html.indexOf('async function startLiveFrames'),
                                html.indexOf('function stopLiveFrames'));
    ok(/RIO\.source\.startFeed\(\)/.test(liveFeed),
       'the live session asks the owner too');
    ok(!/getUserMedia/.test(liveFeed),
       'and no longer starts a feed of its own — the regression, asserted');

    ok(/RIO\.source\.setClip\(/.test(html),
       'an upload sets the clip as the source');
    ok(/id="camsource"/.test(html) && /id="usecamera"/.test(html),
       'the source is shown in the HUD and can be handed back to the camera');
  }

  console.log('\n' + (failures ? 'FAILED ' + failures + '/' : 'PASSED ')
              + checks + ' checks');
  process.exit(failures ? 1 : 0);
})();
