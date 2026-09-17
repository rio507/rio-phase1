/* barge_gate_selftest.js — what the gate says happened, against what did.
 *
 *   node tools/barge_gate_selftest.js
 *
 * THE DRIVE. 2026-09-17, session 0233da0d. Eleven `turn_phantom` events, every
 * one of them carrying the driver's actual words:
 *
 *   "Hello."  "What do you see?"  "Are my tires good?"
 *   "Can you give me directions to the Grand Canyon?"  ...
 *
 * all with `why: "no_confirmed_barge"` and `speaking: true`. Read at face
 * value that says the echo gate refused eleven real interruptions, which is a
 * frightening number and is not what happened.
 *
 * WHAT ACTUALLY HAPPENED is that transcription loses the race with the model.
 * The server commits the utterance, creates a response for it, and the WORDS
 * arrive afterwards -- by which time she is already answering them. The gate
 * looks for a confirmed barge-in behind a transcript that was never an
 * interruption, finds none, and files an ordinary question as a refusal.
 *
 * `selfAnswered` already knew: the transcript carries the same item_id the
 * response was bound to. It was consulted for the supersede, for the turn
 * counter and for the noise buffer, and NOT for the phantom tally -- which is
 * the one output anybody reads. It also set `real = false` thirty lines below
 * a comment explaining, at length, why that must not happen.
 *
 * THE SECOND FAULT, in the same gate. supersedeGate's first branch is
 * commented "the path a genuine second question takes while she is waiting on
 * a tool call". It could not be reached while she was waiting on a tool call:
 * `speaking` is set from `response.created`, and a response sits open --
 * deliberating, or writing a function call -- before it makes any sound. That
 * drive has responses open with no audio for 12.7 s and 37.3 s, and for every
 * one of those seconds a question asked into the silence was refused as a
 * possible echo of a voice that had not spoken.
 *
 * THE THIRD is that none of this was countable. `false_barge_in` counts the
 * gate stopping her for something that was not a person. Nothing counted the
 * opposite -- a suppression that a real person then spoke through -- so every
 * drive could only ever argue for tightening the threshold.
 *
 * WHAT THIS ASSERTS
 * -----------------
 *   1. A transcript for the utterance already being answered is NOT a
 *      phantom, does not supersede, and stays `real` so the barge block can
 *      still classify it.
 *   2. A genuine second question, arriving under a different item, IS still
 *      refused while she is actually audible. The fix must not open the gate.
 *   3. ...and is ALLOWED while a response is open but has made no sound.
 *   4. A suppression that a real transcript arrives behind is reported as a
 *      missed barge-in, with the margin that refused it.
 *   5. Her own words coming back are still not a missed barge-in.
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

/* THE PHONE COLUMN, which is the only column this bug lives in: the onset
   guard and the level meter are both off on a desk, and with them off the
   gate has nothing to refuse anything with. Values from config.py. */
function phone(opts) {
  opts = opts || {};
  const arbiter = speech.makeArbiter();
  const sent = [];
  const events = [];
  const audio = { muted: false };
  let mic = -60, out = -60;
  const controller = rt.createController({
    arbiter, send: (o) => sent.push(o),
    tool: () => Promise.resolve({ ok: true }),
    audio: { mute: () => { audio.muted = true; },
             unmute: () => { audio.muted = false; } },
    onEvent: (e) => events.push(e),
    bargeSustainMs: opts.bargeSustainMs === undefined ? 600 : opts.bargeSustainMs,
    bargeConfirmMs: 1500,
    bargeOnsetGuardMs: opts.onsetGuardMs === undefined ? 400 : opts.onsetGuardMs,
    bargeEchoMarginDb: 6,
    bargeEchoFloorDb: -50,
    echoTailMs: 600,
    levels: () => ({ mic, out }),
  });
  let n = 0, items = 0;
  return {
    arbiter, sent, events, audio, controller,
    ev: (t) => events.filter(e => e.type === t),
    evTypes: () => events.map(e => e.type),
    level(m, o) { mic = m; out = o; },
    /* One driver utterance, committed by the server, as the wire delivers it:
       the commit carries the item id the transcription will arrive under. */
    commit() {
      const id = 'item_' + (++items);
      controller.handle({ type: 'input_audio_buffer.committed', item_id: id });
      return id;
    },
    respond() {
      const id = 'resp_' + (++n);
      controller.handle({ type: 'response.created', response: { id } });
      return id;
    },
    audioFor(id, words) {
      controller.handle({ type: 'response.output_audio_transcript.delta',
                          response_id: id, delta: words || 'x' });
    },
    transcript(text, itemId) {
      controller.handle({
        type: 'conversation.item.input_audio_transcription.completed',
        transcript: text, item_id: itemId || null });
    },
    speechStart() { controller.handle({ type: 'input_audio_buffer.speech_started' }); },
    speechStop() { controller.handle({ type: 'input_audio_buffer.speech_stopped' }); },
    turns: () => controller.state().cutoffs,
  };
}

(async function main() {

  section('the eleven — a transcript for the answer already in flight');
  {
    const h = phone();
    // The driver asks. The server commits it and opens the answer before the
    // transcription comes back, which is the ordinary order.
    const item = h.commit();
    const rid = h.respond();
    h.audioFor(rid, 'The mountain ahead');     // she is audibly answering
    const before = h.ev('LIVE_TURN_SUPERSEDED').length;

    h.transcript('Look at that mountain over there. Wow.', item);

    ok(h.ev('LIVE_TURN_PHANTOM').length === 0,
       'it is NOT filed as a phantom — this is the driver being answered, and '
       + 'eleven of these were the whole of that drive\'s tally');
    ok(h.ev('LIVE_TURN_SELF').length === 1,
       'it is filed as the turn already being answered');
    ok(h.ev('LIVE_TURN_SUPERSEDED').length === before,
       'and it supersedes nothing — she is answering these very words');
  }

  section('a genuine second question, while she is audibly speaking');
  {
    const h = phone();
    const first = h.commit();
    const rid = h.respond();
    h.audioFor(rid, 'It is Mount Fuji, the tallest');
    // A DIFFERENT utterance: new item, never committed as the one being
    // answered. No detector firing, so nothing confirmed a person.
    h.transcript('Are my tires good?', 'item_other');
    const ph = h.ev('LIVE_TURN_PHANTOM');
    ok(ph.length === 1,
       'still refused while her voice is actually in the room — the fix must '
       + 'not open the gate');
    ok(ph[0] && ph[0].why === 'no_confirmed_barge',
       `for the reason it always gave (${ph[0] && ph[0].why})`);
    ok(ph[0] && ph[0].self_answered === false,
       'and the log now says it was NOT already being answered, which is the '
       + 'difference between this case and the eleven');
  }

  section('...and a second question while a tool call is running');
  {
    const h = phone();
    h.commit();
    const rid = h.respond();
    h.audioFor(rid, 'Let me check');
    /* THE SHAPE THE WIRE ACTUALLY DELIVERS. A tool call ENDS the response --
       the model emits the call and the response completes -- and the tool
       then runs for seconds with no response open at all. So this case was
       always reachable through the first branch, and the drive's two silent
       responses were a different thing: open, never having made a sound. */
    h.controller.handle({ type: 'response.done',
                          response: { id: rid, status: 'completed' } });
    await tick(650);                       // the tool is running; 600 ms tail spent
    h.transcript('Actually, what about the weather?', 'item_other');
    ok(h.ev('LIVE_TURN_PHANTOM').length === 0,
       'allowed — nothing of hers has been in the room for longer than the '
       + 'echo tail, so there is nothing for this to be an echo of');
  }

  section('a response that has never made a sound at all');
  {
    const h = phone();
    h.commit();
    h.respond();                           // open, deliberating, silent
    await tick(650);
    h.transcript('Can you give me directions to the Grand Canyon?', 'item_other');
    ok(h.ev('LIVE_TURN_PHANTOM').length === 0,
       'allowed too — this is the 12.7 s and 37.3 s case from the drive, where '
       + '`speaking` was true and she had said nothing');
  }

  section('the gate\'s false negatives — a suppression somebody spoke through');
  {
    const h = phone();
    const rid = h.respond();
    h.audioFor(rid, 'The road ahead is');
    /* PAST HER OPENING SYLLABLE. Inside the 400 ms onset guard the gate does
       not decide at all, it defers -- so a test that fires the detector
       immediately measures the guard and never reaches the level test. */
    await tick(450);
    // The microphone is quieter than the speaker: the level test calls it her.
    h.level(-31.3, -25.1);                 // margin -6.2 dB against a required +6
    h.speechStart();
    const sup = h.ev('LIVE_ECHO_SUPPRESSED');
    ok(sup.length === 1 && sup[0].reason === 'level_margin',
       `the gate suppresses it (margin ${sup[0] && sup[0].margin_db} dB, `
       + `required ${sup[0] && sup[0].required_db} dB)`);

    // ...and then a real transcript arrives. There was somebody there.
    h.transcript('Are my tires good?', 'item_other');
    const miss = h.ev('LIVE_BARGE_MISSED');
    ok(miss.length === 1,
       'a real transcript behind it is recorded as a MISSED barge-in — the '
       + 'number that did not exist');
    ok(miss[0] && miss[0].margin_db === -6.2 && miss[0].required_db === 6,
       `carrying the margin that refused it (${miss[0] && miss[0].margin_db} dB `
       + `against ${miss[0] && miss[0].required_db} dB), which is the only `
       + 'honest argument for moving that threshold');
    ok(miss[0] && typeof miss[0].since_suppress_ms === 'number',
       'and how long after the suppression the words arrived');
  }

  section('...but her own voice coming back is not one');
  {
    const h = phone();
    const rid = h.respond();
    h.audioFor(rid, 'It is a torii gate at the end of the street.');
    await tick(450);                       // past the onset guard
    h.level(-36.6, -35.3);
    h.speechStart();
    ok(h.ev('LIVE_ECHO_SUPPRESSED').length === 1, 'suppressed');
    // The transcriber writes down what it heard: her.
    h.transcript('It is a torii gate at the end of the street.', 'item_other');
    ok(h.ev('LIVE_BARGE_MISSED').length === 0,
       'her words back through the phone are not a person the gate missed — '
       + 'that would make the false-negative count useless');
  }

  section('a suppression long past is not blamed for a later turn');
  {
    const h = phone();
    const rid = h.respond();
    h.audioFor(rid, 'Checking.');
    await tick(450);                       // past the onset guard
    h.level(-36.6, -35.3);
    h.speechStart();
    ok(h.ev('LIVE_ECHO_SUPPRESSED').length === 1, 'suppressed');
    await tick(1600);                      // past the confirm window
    h.transcript('What is the first direction?', 'item_other');
    ok(h.ev('LIVE_BARGE_MISSED').length === 0,
       'a transcript from a later turn does not become evidence against it');
  }

  console.log(failures ? `\nFAILED ${failures}/${checks} checks`
                       : `\nPASSED ${checks} checks`);
  process.exit(failures ? 1 : 0);
})().catch(e => { console.error(e); process.exit(1); });
