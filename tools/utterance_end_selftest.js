/* utterance_end_selftest.js — how much she wrote, how much was heard, who stopped it.
 *
 *   node tools/utterance_end_selftest.js
 *
 * THE QUESTION. "She stopped mid-sentence, repeatedly, while the session was
 * healthy." Asked of the drive of 2026-09-17 the log could answer for TWO
 * responses -- the two that classified their own cut-off -- and for none of
 * the others, because nothing recorded what a response generated, when its
 * audio started and stopped, or the five mutes that never named themselves.
 *
 * TWO MECHANISMS FOUND BY READING, both in the tail -- the seconds between
 * `response.done` (generation over) and the end of the audio, in which this
 * file used to believe nothing was playing:
 *
 *   NOISE MUTE   a response the server created for an EMPTY transcript is
 *                cancelled on sight, and used to be MUTED on sight. Under
 *                speech-to-speech the element is one stream, so the mute
 *                landed on the tail of the answer before it. Twice on that
 *                drive, five seconds apart, once across "Didn't catch that."
 *   TAIL BARGE   a detector firing into the tail found `speaking` null,
 *                muted, created no pendingBarge, and so was never absorbed
 *                or classified. The rest of the answer stayed muted.
 *
 * WHAT THIS ASSERTS
 * -----------------
 *   1. A whole answer is one LIVE_UTTERANCE_END: generated chars, audio ms,
 *      heard_frac 1, ended_by completed.
 *   2. A cancelled answer names who cancelled it, with the chars it had.
 *   3. A noise reply created during the tail no longer mutes the tail.
 *   4. ...but IS muted if its own audio starts before the cancel lands.
 *   5. With the tail held (WebRTC), a blip in the tail is absorbed and
 *      unmuted; a sustained barge in the tail is classified, not swallowed.
 *   6. A mute that outlives the audio is reported: muted_at_end, ended_by
 *      muted, with the reason that muted it.
 *   7. The tail is not held forever if the end never arrives.
 */
'use strict';

const path = require('path');
const STATIC = path.join(__dirname, '..', 'static');
let checks = 0, failures = 0;
function ok(cond, what) {
  checks++;
  if (!cond) failures++;
  console.log((cond ? '  ok    ' : '  FAIL  ') + what);
}
function section(n) { console.log('\n=== ' + n + ' ==='); }
const tick = (ms) => new Promise(r => setTimeout(r, ms || 0));
if (typeof global.atob !== 'function') {
  global.atob = (b64) => Buffer.from(b64, 'base64').toString('binary');
}
if (typeof global.AbortController !== 'function') {
  global.AbortController = class { constructor() { this.signal = {}; } abort() {} };
}
const speech = require(path.join(STATIC, 'rio_speech.js'));
const rt = require(path.join(STATIC, 'rio_realtime.js'));

function session(opts) {
  opts = opts || {};
  const arbiter = speech.makeArbiter();
  const sent = [], events = [];
  const audio = { muted: false, log: [] };
  let mic = -60, out = -60;
  const controller = rt.createController({
    arbiter, send: (o) => sent.push(o),
    tool: () => Promise.resolve({ ok: true }),
    audio: { mute: () => { audio.muted = true; audio.log.push('mute'); },
             unmute: () => { audio.muted = false; audio.log.push('unmute'); } },
    onEvent: (e) => events.push(e),
    bargeSustainMs: 30, bargeConfirmMs: 60,
    bargeOnsetGuardMs: 0, bargeEchoMarginDb: 0,
    echoTailMs: 600,
    holdTail: !!opts.holdTail, tailFallbackMs: opts.tailFallbackMs || 15000,
    levels: () => ({ mic, out }),
  });
  let n = 0, items = 0;
  const h = {
    sent, events, audio, controller,
    ev: (t) => events.filter(e => e.type === t),
    ends: () => events.filter(e => e.type === 'LIVE_UTTERANCE_END'),
    cancels: () => sent.filter(e => e.type === 'response.cancel'),
    commit() { const id = 'item_' + (++items); controller.handle({ type: 'input_audio_buffer.committed', item_id: id }); return id; },
    created() { const id = 'resp_' + (++n); controller.handle({ type: 'response.created', response: { id } }); return id; },
    delta(id, w) { controller.handle({ type: 'response.output_audio_transcript.delta', response_id: id, delta: w }); },
    transcriptDone(id, t) { controller.handle({ type: 'response.output_audio_transcript.done', response_id: id, transcript: t }); },
    audioStart(id) { controller.handle({ type: 'output_audio_buffer.started', response_id: id }); },
    audioStop(id) { controller.handle({ type: 'output_audio_buffer.stopped', response_id: id }); },
    audioCleared(id) { controller.handle({ type: 'output_audio_buffer.cleared', response_id: id }); },
    done(id, status, reason) {
      const r = { id, status: status || 'completed' };
      if (reason) r.status_details = { type: status, reason };
      controller.handle({ type: 'response.done', response: r });
    },
    transcript(text, itemId) {
      controller.handle({ type: 'conversation.item.input_audio_transcription.completed',
                          transcript: text, item_id: itemId || null });
    },
    speechStart() { controller.handle({ type: 'input_audio_buffer.speech_started' }); },
    speechStop() { controller.handle({ type: 'input_audio_buffer.speech_stopped' }); },
    speaking: () => (controller.state().speaking
                     ? { responseId: controller.state().response_id } : null),
  };
  return h;
}

(async function main() {

  section('a whole answer, accounted for');
  {
    const h = session();
    const id = h.created();
    h.audioStart(id);
    h.delta(id, 'It is Mount Fuji, ');
    h.delta(id, 'about sixty miles ahead.');
    await tick(40);
    h.transcriptDone(id, 'It is Mount Fuji, about sixty miles ahead.');
    h.done(id, 'completed');
    h.audioStop(id);
    const e = h.ends();
    ok(e.length === 1, `one LIVE_UTTERANCE_END for one response (${e.length})`);
    ok(e[0] && e[0].generated_chars === 42,
       `carrying what was written (${e[0] && e[0].generated_chars} chars)`);
    ok(e[0] && e[0].ended_by === 'completed' && e[0].early === false,
       `ended by completion, not early (${e[0] && e[0].ended_by})`);
    ok(e[0] && e[0].heard_frac === 1 && e[0].audio_ms >= 40,
       `the driver heard all of it (heard_frac ${e[0] && e[0].heard_frac}, audio ${e[0] && e[0].audio_ms} ms)`);
  }

  section('a cancelled answer names who cancelled it');
  {
    const h = session();
    const id = h.created();
    h.audioStart(id);
    h.delta(id, 'The weather is fine, ');
    h.speechStart();                     // the detector fires
    await tick(50);                      // past the 30 ms sustain: cancelled
    h.audioCleared(id);
    h.done(id, 'cancelled', 'client_cancelled');
    const e = h.ends();
    ok(e.length === 1 && e[0].ended_by === 'cancelled:barge',
       `ended_by names the gate (${e[0] && e[0].ended_by})`);
    ok(e[0] && e[0].generated_chars === 21 && e[0].early === true,
       `with the chars it had written when it was stopped (${e[0] && e[0].generated_chars})`);
  }

  section('a noise reply in the tail does not mute the tail');
  {
    const h = session();
    const a = h.created();
    h.audioStart(a);
    h.delta(a, 'Turn left in two hundred metres onto Palisades Drive, then');
    h.done(a, 'completed');              // generation over; audio still streaming
    // The road roars: an EMPTY transcript. The server creates a reply for it.
    h.transcript('', 'item_noise');
    const mutesBefore = h.audio.log.filter(x => x === 'mute').length;
    const b = h.created();               // the reply to nothing
    ok(h.cancels().some(c => c.response_id === b),
       'the reply to nothing is cancelled by id, as before');
    ok(h.audio.log.filter(x => x === 'mute').length === mutesBefore && !h.audio.muted,
       'and the element is NOT muted -- the mute used to land on the tail of '
       + 'the answer still coming out of the speaker');
    h.audioStop(a);
    const ea = h.ends().find(e => e.response_id === a);
    ok(ea && ea.ended_by === 'completed' && ea.muted_ms === 0,
       `the answer's tail reached the driver whole (muted ${ea && ea.muted_ms} ms)`);
  }

  section('...but a silenced reply whose audio starts anyway is muted then');
  {
    const h = session();
    h.transcript('', 'item_noise');
    const b = h.created();
    ok(!h.audio.muted, 'not muted at creation');
    h.audioStart(b);                     // the cancel lost the race
    ok(h.audio.muted, 'muted the moment its own audio starts');
    ok(h.sent.some(e => e.type === 'output_audio_buffer.clear'), 'and the buffer is cleared');
    h.audioStop(b);
    ok(!h.audio.muted, 'and unmuted when that audio ends');
  }

  section('the tail, held: a blip is absorbed, a barge is classified');
  {
    const h = session({ holdTail: true });
    const a = h.created();
    h.audioStart(a);
    h.delta(a, 'The tallest mountain in Japan, ');
    h.done(a, 'completed');
    ok(h.speaking() && h.speaking().responseId === a,
       'after response.done the mouth is still hers -- the audio is still playing');
    // A blip in the tail: the detector fires and stops inside the sustain.
    h.speechStart();
    ok(h.audio.muted, 'muted instantly, as always');
    h.speechStop();
    ok(!h.audio.muted, 'and UNMUTED when the blip ends -- it used to stay muted, '
       + 'because with `speaking` null there was no pendingBarge to absorb against');
    ok(h.ev('LIVE_BARGE_ABSORBED').length === 1, 'and counted as an absorbed blip');
    h.audioStop(a);
    ok(!h.speaking(), 'the mouth is released when the sound stops');
    const ea = h.ends().find(e => e.response_id === a);
    ok(ea && ea.ended_by === 'completed' && ea.mute_reasons.indexOf('barge') === 0,
       `reported whole, with the blip's mute named (${ea && ea.mute_reasons})`);
  }
  {
    const h = session({ holdTail: true });
    const a = h.created();
    h.audioStart(a);
    h.delta(a, 'Fuji is about ');
    h.done(a, 'completed');
    h.speechStart();
    await tick(50);                      // sustained
    ok(h.cancels().length >= 1 && h.sent.some(e => e.type === 'output_audio_buffer.clear'),
       'a sustained barge in the tail clears the audio -- a real interruption '
       + 'is honoured in the tail exactly as it is mid-generation');
    h.audioCleared(a);
    h.speechStop();
    await tick(80);                      // no transcript follows
    ok(h.ev('LIVE_CUTOFF').some(c => c.cause === 'false_barge_in'),
       'and is CLASSIFIED when no words follow -- a cut-off with a name, '
       + 'instead of a mute nobody recorded');
  }

  section('a mute that outlives the audio is reported as such');
  {
    const h = session();
    const a = h.created();
    h.audioStart(a);
    h.delta(a, 'Head east, then ');
    h.done(a, 'completed');
    h.controller.handle({ type: 'input_audio_buffer.speech_started' }); // tail barge, no hold
    await tick(20);
    h.audioStop(a);                      // the sound ends while still muted
    const ea = h.ends().find(e => e.response_id === a);
    ok(ea && ea.muted_at_end === true && ea.ended_by === 'muted',
       `the driver did not hear the end: muted_at_end, ended_by ${ea && ea.ended_by}`);
    ok(ea && ea.muted_ms >= 15 && ea.heard_frac < 1 && ea.early === true,
       `with the muted time charged (${ea && ea.muted_ms} ms, heard_frac ${ea && ea.heard_frac})`);
  }

  section('the tail is not held forever');
  {
    const h = session({ holdTail: true, tailFallbackMs: 40 });
    const a = h.created();
    h.audioStart(a);
    h.delta(a, 'x');
    h.done(a, 'completed');
    ok(!!h.speaking(), 'held after response.done');
    await tick(80);
    ok(!h.speaking(), 'released by the fallback when .stopped never arrives');
    ok(h.ev('LIVE_TAIL_TIMEOUT').length === 1, 'and the timeout is an event');
  }

  console.log(failures ? `\nFAILED ${failures}/${checks} checks`
                       : `\nPASSED ${checks} checks`);
  process.exit(failures ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
