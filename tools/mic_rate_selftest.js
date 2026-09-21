/* mic_rate_selftest.js — the driver's voice arrives at the rate it was sent at.
 *
 *   node tools/mic_rate_selftest.js
 *
 * WHAT THIS DEFENDS, AND WHAT IT COST TO FIND.
 *
 * xai_voice.mint_client_secret opens the session with
 * audio.input.format = {type: 'audio/pcm', rate: 24000}. That number is not a
 * preference: it is what the far end will decode every byte the microphone
 * sends as. static/rio_xai_session.js captures that microphone on the shared
 * output AudioContext, which rio_output.js builds at 48000 because her voice
 * has to pass through the echo canceller with everything else RIO says.
 *
 * Nothing reconciled the two. The pump cut 48 kHz audio into 960-sample pieces
 * -- 960 being 40 ms AT 24 kHz -- and shipped them raw, so every word the
 * driver said reached grok-transcribe at twice its length and an octave down.
 * The drive of 2026-09-21 is that fault in numbers: four barge detections,
 * three responses, cutoffs_total 0, and those three answers 15, 24 and 15
 * characters long, which is the length of "Didn't catch that." She was not
 * silent and she was not cut off. She was answering a tape played at half
 * speed, which is the only thing a transcriber can make of one.
 *
 * A duration check alone would not have caught it -- the old path shipped every
 * sample it was given, so no audio was LOST. What was wrong was the rate those
 * samples claimed to be at, and the only way to see that is to ask what the far
 * end HEARS. So section 1 sends a known tone and measures the frequency at the
 * far end's rate, and section 2 runs the old arithmetic through the same
 * assertion and fails the suite if it passes.
 */
'use strict';

const xs = require('../static/rio_xai_session.js');

const RATE = xs.RATE;
const FRAME = Math.round(RATE * xs.FRAME_MS / 1000);

let checks = 0, failures = 0;
function ok(name, cond, extra) {
  checks++;
  if (!cond) failures++;
  console.log((cond ? '  ok   ' : '  FAIL ') + name + (extra ? ' — ' + extra : ''));
}

/* Goertzel energy at one frequency, and the peak over a band. Cheaper than an
   FFT and enough: the question is "which tone is this", not "what is the whole
   spectrum". */
function tone(x, f, sr) {
  const w = 2 * Math.PI * f / sr, c = 2 * Math.cos(w);
  let s1 = 0, s2 = 0;
  for (let i = 0; i < x.length; i++) { const s = x[i] + c * s1 - s2; s2 = s1; s1 = s; }
  return Math.sqrt(s1 * s1 + s2 * s2 - c * s1 * s2) / x.length;
}
function peak(x, sr, lo, hi) {
  let best = lo, bv = -1;
  for (let f = lo; f <= hi; f++) { const v = tone(x, f, sr); if (v > bv) { bv = v; best = f; } }
  return best;
}

function sine(sr, seconds, freq) {
  const n = Math.round(sr * seconds);
  const x = new Float32Array(n);
  for (let i = 0; i < n; i++) x[i] = Math.sin(2 * Math.PI * freq * i / sr);
  return x;
}

/* The pump's own loop, driven the way a ScriptProcessorNode drives it: 4096
   samples at a time, with whatever did not make a whole frame carried over. */
function pumped(sig, srcRate) {
  const ratio = srcRate / RATE;
  let buf = new Float32Array(0), pos = 0;
  const out = [];
  for (let off = 0; off < sig.length; off += 4096) {
    const ch = sig.subarray(off, Math.min(off + 4096, sig.length));
    const g = new Float32Array(buf.length + ch.length);
    g.set(buf, 0); g.set(ch, buf.length); buf = g;
    for (;;) {
      const got = xs.frameAtRate(buf, pos, FRAME, ratio);
      if (!got) break;
      pos = got.pos;
      const drop = Math.floor(pos);
      if (drop > 0) { buf = buf.slice(drop); pos -= drop; }
      out.push(got.out);
    }
  }
  const flat = new Float32Array(out.length * FRAME);
  out.forEach((f, i) => flat.set(f, i * FRAME));
  return flat;
}

console.log(`\n== 1. what the far end decodes, at its declared ${RATE} Hz\n`);

const FREQ = 440, SECONDS = 2.0;
// Every rate an AudioContext realistically hands out, plus RATE itself.
for (const sr of [48000, 44100, 32000, 24000, 16000]) {
  const sent = pumped(sine(sr, SECONDS, FREQ), sr);
  const heardS = sent.length / RATE;
  const naiveS = Math.round(sr * SECONDS) / RATE;      // what the old path gave
  const heardHz = peak(sent, RATE, FREQ - 220, FREQ + 220);
  // One 40 ms frame may be held in the buffer; anything more is drift.
  ok(`a ${sr} Hz microphone: ${SECONDS.toFixed(3)} s of speech lasts `
     + `${heardS.toFixed(3)} s at the far end`,
     Math.abs(heardS - SECONDS) <= (xs.FRAME_MS / 1000) + 1e-6,
     `raw would have been ${naiveS.toFixed(3)} s`);
  ok(`   ...and a ${FREQ} Hz tone is still ${heardHz} Hz`,
     Math.abs(heardHz - FREQ) <= 2,
     `raw would have been ${Math.round(FREQ * RATE / sr)} Hz`);
}

/* ALIASING, which is the reason this resamples rather than picking every Nth
   sample, and the number that chose the filter.

   A tone above the far end's Nyquist cannot be represented there. What must NOT
   happen is that it comes back as a loud tone INSIDE the speech band, on top of
   the consonants a transcriber works from. Three filters were measured against
   this exact tone while the fix was being written:

       every Nth sample     the fault itself
       box, one period      67%   <- "an average is surely enough"
       triangle, two        30%
       windowed sinc        0.7%  <- what ships

   The ceiling is 1%, which the first two fail and the third passes with room.
   It is written as a fraction of a REAL tone at that frequency rather than as
   an absolute, so it stays meaningful if the kernel's passband gain moves. */
{
  const sr = 48000, f = 15000;            // above 24 kHz audio's 12 kHz Nyquist
  const sent = pumped(sine(sr, 1.0, f), sr);
  const folded = 2 * (RATE / 2) - f;      // 9000 Hz, where naive decimation puts it
  const at9k = tone(sent, folded, RATE);
  const ref = tone(pumped(sine(sr, 1.0, folded), sr), folded, RATE);
  ok(`a ${f} Hz tone does not come back as a ${folded} Hz one`,
     at9k < ref * 0.01, `alias ${(at9k / ref * 100).toFixed(1)}% of a real tone`);
}

console.log('\n== 2. ...and the assertion can fail: the arithmetic that shipped\n');

/* THE BUG, RUN THROUGH THE SAME CHECK. The old pump sliced the source buffer
   into FRAME-sample pieces and sent them unchanged, whatever rate they were
   captured at. If this passes, section 1 is not measuring anything. */
function pumpedRaw(sig) {
  const n = Math.floor(sig.length / FRAME) * FRAME;
  return sig.subarray(0, n);
}
{
  const sr = 48000;
  const raw = pumpedRaw(sine(sr, SECONDS, FREQ));
  const heardS = raw.length / RATE;
  const heardHz = peak(raw, RATE, FREQ - 260, FREQ + 260);
  const durationCaught = Math.abs(heardS - SECONDS) > (xs.FRAME_MS / 1000);
  const pitchCaught = Math.abs(heardHz - FREQ) > 2;
  ok('the old path IS caught by the duration check',
     durationCaught, `${SECONDS.toFixed(3)} s arrived as ${heardS.toFixed(3)} s`);
  ok('...and by the pitch check',
     pitchCaught, `${FREQ} Hz arrived as ${heardHz} Hz`);
}

console.log(`\n${checks - failures}/${checks} checks passed`);
if (failures) { console.log(`${failures} FAILED`); process.exit(1); }
