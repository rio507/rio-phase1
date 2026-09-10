/* rio_frame_encoder.js — the JPEG encode, off the main thread.
 *
 * WHY THIS FILE EXISTS. The frame loop was doing all of this on the main
 * thread, ten to fifteen times a second, for the whole of a drive:
 *
 *     drawImage(video -> canvas)      a scale and a copy
 *     canvas.toBlob('image/jpeg')     the encode, and the expensive part
 *     blob.arrayBuffer()              another copy
 *     new Uint8Array(4 + h + body)    and another
 *
 * On session 738fbb82 that was ~35 KB a frame at about 6 fps sustained. None
 * of it is slow in isolation; all of it is on the same thread as the media
 * pipeline that feeds the <audio> element the loopback plays out of, and a
 * main thread that stalls for 20 ms at the wrong moment is a jitter buffer
 * that runs dry. Which is a crackle.
 *
 * So the encode moves here. `createImageBitmap(videoEl)` on the main thread is
 * a GPU-side copy that transfers to a worker with no serialisation at all;
 * everything after it — the scale, the encode, the byte copy — happens on this
 * thread, and the main thread gets back an ArrayBuffer it already owns.
 *
 * ONE CANVAS, REUSED. Allocating an OffscreenCanvas per frame is a megabyte of
 * garbage a second on a phone, and the collector for it runs wherever it likes.
 *
 * NO POLICY HERE, and none is possible: this file is handed a bitmap, a size
 * and a quality, and hands back bytes. Every decision about which frame to take
 * and how good it should be stays in rio_frames.js, where the measurements
 * that drive it live.
 */
'use strict';

var canvas = null;
var ctx2d = null;

function ensure(w, h) {
  if (!canvas) {
    canvas = new OffscreenCanvas(w, h);
    ctx2d = canvas.getContext('2d');
  } else if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
  return ctx2d;
}

self.onmessage = function (e) {
  var msg = e.data || {};
  if (msg.op === 'ping') {
    // The main thread proving this worker is alive before it starts relying
    // on it. A worker that failed to load looks exactly like one that is slow.
    self.postMessage({ op: 'pong' });
    return;
  }
  if (msg.op !== 'encode') return;

  var bitmap = msg.bitmap;
  var seq = msg.seq;
  try {
    var c = ensure(msg.w, msg.h);
    c.drawImage(bitmap, 0, 0, msg.w, msg.h);
    // The bitmap is ours now and holds GPU memory until it is closed.
    try { bitmap.close(); } catch (x) {}
    canvas.convertToBlob({ type: 'image/jpeg', quality: msg.quality })
      .then(function (blob) { return blob.arrayBuffer(); })
      .then(function (bytes) {
        self.postMessage({ op: 'frame', seq: seq, bytes: bytes }, [bytes]);
      })
      .catch(function (err) {
        self.postMessage({ op: 'error', seq: seq, error: String(err) });
      });
  } catch (err) {
    try { if (bitmap) bitmap.close(); } catch (x) {}
    self.postMessage({ op: 'error', seq: seq, error: String(err) });
  }
};
