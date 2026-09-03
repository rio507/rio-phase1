/* voice_sink_harness.js — a speaker and a wire, with a clock you can turn.
 *
 * The interesting things rio_voice_eleven.js does are all about TIME. How much
 * of a sentence has been heard. Whether a cancel stopped audio that was
 * already scheduled. Whether the mouth is handed back when the model finishes
 * or when the listener does. None of that is checkable against a real
 * AudioContext in a test: a real one advances on its own, at its own rate,
 * and the assertions would be about how fast the machine running them is.
 *
 * So the context here is a stub with a clock the test moves by hand. It is not
 * a mock of Web Audio — it implements the four calls the sink actually makes
 * and records what was asked of it, which is what the tests are about.
 *
 * Shared between tools/realtime_selftest.js and tools/live_voice_probe.js
 * deliberately: the probe drives the same ten turns through the same sink the
 * unit tests use, so "the cancel path still behaves" is one claim with two
 * kinds of evidence rather than two claims that could disagree.
 */
'use strict';

/* A context whose clock only moves when a test says so. */
function fakeAudio() {
  const state = {
    t: 0,
    sources: [],        // every buffer ever scheduled
    stops: 0,           // ...and how many were stopped early
    ramps: [],          // { to, at, overS } — the fades
    closed: false,
  };
  const gain = {
    gain: {
      value: 0,
      cancelScheduledValues() {},
      setValueAtTime(v) { gain.gain.value = v; },
      linearRampToValueAtTime(v, at) {
        state.ramps.push({ to: v, at: at, overS: at - state.t });
        gain.gain.value = v;
      },
    },
    connect() {},
  };
  const ctx = {
    get currentTime() { return state.t; },
    state: 'running',
    destination: {},
    createGain() { return gain; },
    createBuffer(channels, length, rate) {
      // ONE array, handed back every time. A fresh one per call would throw
      // away everything written into the buffer, which is exactly the shape a
      // real AudioBuffer does not have — and it made the decode look wrong
      // when the decode was right.
      const data = new Float32Array(length);
      return { duration: length / rate, numberOfChannels: channels,
               length: length, sampleRate: rate,
               getChannelData() { return data; } };
    },
    createBufferSource() {
      const src = {
        buffer: null,
        connect() {},
        start(at) { src.at = at; state.sources.push(src); },
        stop() { src.stoppedAt = state.t; state.stops++; },
      };
      return src;
    },
    resume() {},
    close() { state.closed = true; },
  };
  return {
    ctx: ctx,
    gain: gain,
    state: state,
    /* Move the clock. Everything scheduled before `t` has been heard. */
    advance(seconds) { state.t += seconds; return state.t; },
  };
}

/* A relay socket that never leaves the process. */
function fakeTransport() {
  const sock = {
    readyState: 1,
    sent: [],
    closed: false,
    send(text) { sock.sent.push(JSON.parse(text)); },
    close() { sock.closed = true; if (sock.onclose) sock.onclose(); },
    /* What the server would have said. */
    deliver(obj) {
      if (sock.onmessage) sock.onmessage({ data: JSON.stringify(obj) });
    },
    ops(op) { return sock.sent.filter((m) => m.op === op); },
  };
  return sock;
}

/* Base64 PCM for `seconds` of (silent) audio at the sink's rate.
 *
 * Silence rather than a tone because nothing here listens: what the sink does
 * with a buffer is a function of its LENGTH, and the samples exist only so the
 * decode path runs on something real. */
function pcm(seconds, rate) {
  rate = rate || 24000;
  const samples = Math.max(1, Math.round(seconds * rate));
  const bytes = Buffer.alloc(samples * 2);
  return bytes.toString('base64');
}

/* An open sink over both fakes, ready to be spoken through. */
function openSink(eleven, opts) {
  opts = opts || {};
  const audio = fakeAudio();
  const wire = fakeTransport();
  const events = [];
  const sink = eleven.createSink({
    transport: wire,
    context: audio.ctx,
    sampleRate: opts.sampleRate || 24000,
    fadeMs: opts.fadeMs,
    onEvent: (ev) => events.push(ev),
  });
  const opened = sink.open();
  wire.deliver({ op: 'ready', open: true, model: 'eleven_v3_conversational',
                 voice_id: 'test-voice', sample_rate: opts.sampleRate || 24000 });
  return { sink, wire, audio, events, opened,
           /* One piece of audio from the relay, with the words it is for. */
           say(rid, text, seconds) {
             wire.deliver({ op: 'audio', rid: rid, seq: wire.sent.length,
                            text: text, pcm: pcm(seconds || 1.0,
                                                 opts.sampleRate || 24000) });
           },
           done(rid) {
             wire.deliver({ op: 'event', kind: 'utterance_done',
                            detail: { rid: rid } });
           } };
}

module.exports = { fakeAudio, fakeTransport, pcm, openSink };
