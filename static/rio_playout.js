/* rio_playout.js — when her voice actually STOPS, on a transport that will not say.
 *
 * WHAT THIS REPLACES, AND WHY IT IS NOT A PORT.
 *
 * On WebRTC the API tells us: output_audio_buffer.stopped arrives at the real end
 * of her audio, and rio_realtime.js holds the mouth until it does. That hold is
 * commit a05d233 -- "the mouth was handed back seconds before the sound stopped,
 * and every mute in that window had no owner". response.done is the end of
 * GENERATION; under speech-to-speech the audio streams on for seconds after it.
 *
 * On a WebSocket transport there is no such event (xAI documents
 * output_audio_buffer.* as WebRTC/SIP only) and we render the audio ourselves. So
 * the end of the sound stops being something we are told and becomes something we
 * know -- and the only honest place to know it is the queue we are feeding.
 *
 * THE FAILURE THIS IS BUILT AGAINST IS NOT A CRASH.
 *
 * Getting this wrong does not throw, does not log, and does not fail a test that
 * was written optimistically. It sounds like RIO cutting herself off mid-sentence,
 * and it took two drives to find last time. The specific mechanism: hand the mouth
 * back while audio is still playing, and the next detector firing meets
 * `speaking === null`, mutes the tail with no pendingBarge to classify or absorb
 * it, and never unmutes.
 *
 * So the contract here is stated as an inequality rather than as an event, because
 * an inequality is a thing a test can violate on purpose:
 *
 *      audio-end for a response must not fire before
 *      (the moment its last sample is scheduled to finish).
 *
 * tools/playout_selftest.js asserts that BY BREAKING IT -- it runs a deliberately
 * wrong implementation that ends at generation-done and requires the suite to go
 * red. A test that only passes against the correct version cannot tell you it
 * would have caught the bug.
 *
 * NO WEB AUDIO IN HERE. `now` and `schedule` are injected. The browser passes an
 * AudioContext's clock and its buffer-source scheduling; node passes a fake clock
 * and a list. That is the same reason createController takes an injected
 * transport: the decisions are testable without a microphone, a speaker or a car.
 */
'use strict';
(function (root) {

  /* Bytes of PCM16 -> seconds. One channel, because that is what both vendors
     send and what the sink plays; a stereo stream would halve this and the
     resulting audio-end would fire twice as late, so it is asserted rather than
     assumed by the caller. */
  function pcm16Seconds(byteLength, sampleRate, channels) {
    var ch = channels || 1;
    return byteLength / 2 / ch / sampleRate;
  }

  /* opts:
       sampleRate   of the incoming PCM
       now()        seconds, monotonic. An AudioContext's currentTime in the
                    browser; a number a test can advance in node.
       schedule(chunk, atSeconds)  hand audio to the output, to begin at `at`.
                    May be omitted entirely — the ledger is the point, and a
                    caller that only wants to know WHEN is not made to render.
       tailGraceS   how long after the computed end to wait before declaring the
                    sound over. See the comment on it below; it is not a fudge.
       onAudioEnd(responseId, detail)
       onAudioStart(responseId)
  */
  function createPlayout(opts) {
    opts = opts || {};
    var sampleRate = opts.sampleRate || 24000;
    var channels = opts.channels || 1;
    var now = opts.now || function () { return Date.now() / 1000; };
    var schedule = opts.schedule || null;
    /* A SMALL GRACE, AND IT IS NOT A GUESS AT THE ANSWER.
     *
     * The queue knows exactly when the last sample it was GIVEN finishes. What it
     * cannot know is whether the output device has actually flushed it -- an
     * AudioContext reports currentTime against its own clock, and the hardware
     * sits some buffer behind. Ending at exactly the computed instant is right in
     * arithmetic and a few milliseconds early in a cabin, and "a few milliseconds
     * early" is the direction that reopens a05d233.
     *
     * So the grace is added AFTER the computed end and never subtracted from it.
     * It can make the mouth late; it cannot make it early. That asymmetry is the
     * whole design and the selftest pins it in both directions. */
    var tailGraceS = opts.tailGraceS === undefined ? 0.06 : opts.tailGraceS;

    /* Per response: when its audio is scheduled to finish, whether generation is
       over, and whether we have already declared it ended. */
    var live = {};          /* responseId -> row */
    var order = [];         /* responseIds, in the order they started */

    function row(rid) {
      if (!live[rid]) {
        live[rid] = {
          id: rid,
          bytes: 0,
          /* The moment the audio handed over so far will finish. NOT "now plus
             what is left": a response whose deltas arrive slower than real time
             has a playhead in the past, and the next chunk must start at `now`
             rather than extend a stale end. */
          endsAt: 0,
          startedAt: null,
          generationDone: false,
          ended: false,
          endedBy: null,
          chunks: 0,
        };
        order.push(rid);
      }
      return live[rid];
    }

    function push(rid, byteLength) {
      var r = row(rid);
      if (r.ended) return r;                 // audio after we called it over
      var t = now();
      var startAt = r.endsAt > t ? r.endsAt : t;
      if (r.startedAt === null) {
        r.startedAt = startAt;
        if (opts.onAudioStart) {
          try { opts.onAudioStart(rid); } catch (e) {}
        }
      }
      var dur = pcm16Seconds(byteLength, sampleRate, channels);
      r.endsAt = startAt + dur;
      r.bytes += byteLength;
      r.chunks++;
      if (schedule) {
        try { schedule(rid, startAt, dur); } catch (e) {}
      }
      return r;
    }

    /* response.done. GENERATION is over; the sound is not. This is the exact
       moment the old code handed the mouth back, and the exact moment this must
       not. */
    function generationDone(rid) {
      var r = row(rid);
      r.generationDone = true;
      return r;
    }

    /* Is this response's sound over? True only when generation has finished AND
       the computed end (plus the grace) has passed. Both halves are required:
       a response still generating may hand over more audio, and a response whose
       audio is still scheduled is still audible. */
    function drained(rid, at) {
      var r = live[rid];
      if (!r) return true;
      if (!r.generationDone) return false;
      return (at === undefined ? now() : at) >= r.endsAt + tailGraceS;
    }

    /* Call this from a timer or an audio callback. Declares audio-end for every
       response whose sound is over, once each, and returns how many it ended.
       Idempotent by design: a caller that polls at 20 ms and a caller that polls
       at 500 ms must reach the same ledger. */
    function tick() {
      var t = now();
      var ended = 0;
      for (var i = 0; i < order.length; i++) {
        var rid = order[i];
        var r = live[rid];
        if (!r || r.ended) continue;
        if (!drained(rid, t)) continue;
        r.ended = true;
        r.endedBy = 'drained';
        r.endedAt = t;
        ended++;
        if (opts.onAudioEnd) {
          try {
            opts.onAudioEnd(rid, { reason: 'drained', at: t,
                                   scheduled_end: r.endsAt,
                                   bytes: r.bytes, chunks: r.chunks });
          } catch (e) {}
        }
      }
      return ended;
    }

    /* THE LOCAL EQUIVALENT OF output_audio_buffer.clear, and it is better than
       the event it replaces: a barge-in stops the sound HERE, with no round trip
       to a server that then has to tell us it stopped. The response is ended as
       CANCELLED rather than drained, because the two are different facts and the
       mute ledger distinguishes them (see `ended_by` in rio_realtime.js). */
    function flush(rid, why) {
      var out = [];
      var ids = rid ? [rid] : order.slice();
      var t = now();
      for (var i = 0; i < ids.length; i++) {
        var r = live[ids[i]];
        if (!r || r.ended) continue;
        r.ended = true;
        r.endedBy = 'cancelled:' + (why || 'flush');
        r.endedAt = t;
        r.endsAt = Math.min(r.endsAt, t);
        out.push(r.id);
        if (opts.onAudioEnd) {
          try {
            opts.onAudioEnd(r.id, { reason: r.endedBy, at: t,
                                    scheduled_end: r.endsAt,
                                    bytes: r.bytes, chunks: r.chunks });
          } catch (e) {}
        }
      }
      return out;
    }

    /* MAY THE NEXT REQUEST GO OUT YET?
     *
     * xAI's own migration note, verbatim: "If your client immediately sends
     * conversation.item.create (with the function result) followed by
     * response.create, the server starts generating the next response right away
     * — even if the client is still playing audio from the previous turn. This
     * causes overlapping audio."
     *
     * That is rio_realtime.js:3265->3276 exactly, and it has never hurt us
     * because the SERVER owned playout, so "still playing" was not a client-side
     * state we could get wrong. The moment we own the queue, it is — and it is
     * the same queue that knows the tail, which is why these are one piece of
     * work and not two.
     *
     * Deliberately NOT a promise: a caller that has a deadline (a dictated line,
     * an arbiter TTL) has to be able to ask and then decide, rather than being
     * made to wait. `waitFor` below is the convenience for the ones that can.
     */
    function idle() {
      var t = now();
      for (var i = 0; i < order.length; i++) {
        var r = live[order[i]];
        if (r && !r.ended) {
          if (!r.generationDone) return false;
          if (t < r.endsAt + tailGraceS) return false;
        }
      }
      return true;
    }

    /* How long until idle(), in seconds. 0 when already idle, Infinity when a
       response is still generating and no end can be computed yet -- which a
       caller must treat as "ask again", not as "never". */
    function untilIdle() {
      var t = now();
      var worst = 0;
      for (var i = 0; i < order.length; i++) {
        var r = live[order[i]];
        if (!r || r.ended) continue;
        if (!r.generationDone) return Infinity;
        var left = (r.endsAt + tailGraceS) - t;
        if (left > worst) worst = left;
      }
      return worst;
    }

    function state() {
      var rows = {};
      for (var i = 0; i < order.length; i++) {
        var r = live[order[i]];
        if (!r) continue;
        rows[r.id] = {
          bytes: r.bytes, chunks: r.chunks, started_at: r.startedAt,
          scheduled_end: r.endsAt, generation_done: r.generationDone,
          ended: r.ended, ended_by: r.endedBy,
          ended_at: r.endedAt === undefined ? null : r.endedAt,
        };
      }
      return { sample_rate: sampleRate, channels: channels,
               tail_grace_s: tailGraceS, idle: idle(),
               until_idle_s: untilIdle(), responses: rows };
    }

    function forget(rid) {
      delete live[rid];
      var i = order.indexOf(rid);
      if (i >= 0) order.splice(i, 1);
    }

    return {
      push: push,
      generationDone: generationDone,
      tick: tick,
      flush: flush,
      drained: drained,
      idle: idle,
      untilIdle: untilIdle,
      state: state,
      forget: forget,
    };
  }

  var api = { createPlayout: createPlayout, pcm16Seconds: pcm16Seconds };

  root.RIO = root.RIO || {};
  root.RIO.playout = api;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
