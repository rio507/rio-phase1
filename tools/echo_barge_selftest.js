/* echo_barge_selftest.js — RIO does not interrupt herself, and still yields.
 *
 *   node tools/echo_barge_selftest.js
 *
 * THE BUG. On a phone, RIO's own voice out of the loudspeaker reached the
 * microphone, the turn detector called it speech, and she cut her own answer
 * off. Five long answers into an empty car cost five answers — measured, by
 * `node tools/echo_barge_probe.js`, which is the diagnostic half of this pair
 * and reports rather than judges.
 *
 * WHAT IS ASSERTED HERE, and the second half matters as much as the first:
 *
 *   A. A phone session with her own voice coming back gets ZERO false
 *      barge-ins across five long answers.
 *   B. A driver who really does talk over her still stops her — same session,
 *      same gate, an utterance that is louder than the speaker and does not
 *      stop when she pauses.
 *   C. The desktop numbers and the desktop behaviour are untouched. Nothing in
 *      this change is allowed to alter what a laptop does, and the way to be
 *      sure is to run the same scenarios with the desktop column and get the
 *      old answers.
 *   D. The gate degrades to the old behaviour whenever its evidence is
 *      missing: no meter, a meter that throws, output below the floor.
 *   E. Every deterministic line has a mouth even when the shared output is
 *      down — the fallbacks in rio_speak.js are real, not decorative.
 *
 * The thresholds come from config.py, so this tests what is shipped. Timings
 * are scaled down by SCALE: what is being checked is the shape of each
 * decision, and the shape is the same at 60 ms as at 600.
 */
'use strict';

const fs = require('fs');
const path = require('path');

const REPO = path.join(__dirname, '..');
const STATIC = path.join(REPO, 'static');

if (typeof global.atob !== 'function') {
  global.atob = (b64) => Buffer.from(b64, 'base64').toString('binary');
}

const rt = require(path.join(STATIC, 'rio_realtime.js'));
const speech = require(path.join(STATIC, 'rio_speech.js'));

let checks = 0, failures = 0;
function ok(cond, what) {
  checks++;
  if (!cond) { failures++; console.log('  FAIL  ' + what); }
  else console.log('  ok    ' + what);
}
function section(name) { console.log('\n=== ' + name + ' ==='); }

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

/* The real numbers, and the factor they are run at. A 600 ms sustain and a
   1500 ms confirmation over five answers is half a minute of test for a
   decision that is made in one branch; SCALE keeps the ratios and drops the
   wall clock. */
const SCALE = 10;
/* Zero stays zero: a desk has no onset guard at all, and "1 ms of guard" is a
   different configuration from "none". */
const scaled = (ms) => (ms > 0 ? Math.max(1, Math.round(ms / SCALE)) : 0);

function configInts() {
  const src = fs.readFileSync(path.join(REPO, 'config.py'), 'utf8');
  const read = (name, dflt) => {
    const m = new RegExp('^' + name + '\\s*=\\s*(-?\\d+)', 'm').exec(src);
    return m ? parseInt(m[1], 10) : dflt;
  };
  return {
    sustain: read('REALTIME_BARGE_SUSTAIN_MS', 300),
    sustainTouch: read('REALTIME_BARGE_SUSTAIN_MS_TOUCH', 600),
    confirm: read('REALTIME_BARGE_CONFIRM_MS', 1500),
    onset: read('REALTIME_BARGE_ONSET_GUARD_MS', 0),
    onsetTouch: read('REALTIME_BARGE_ONSET_GUARD_MS_TOUCH', 400),
    margin: read('REALTIME_BARGE_ECHO_MARGIN_DB', 0),
    marginTouch: read('REALTIME_BARGE_ECHO_MARGIN_DB_TOUCH', 6),
    floor: read('REALTIME_BARGE_ECHO_FLOOR_DB', -50),
  };
}

const CFG = configInts();

/* ---------------------------------------------------------------------------
   A cabin, as two numbers.

   `out` is what the page is rendering and `mic` is what the microphone hears.
   Echo is the loudspeaker arriving back attenuated — the mic tracks the
   output, below it. A driver is a source of their own: louder than the
   speaker, and still there when she pauses.
   --------------------------------------------------------------------------- */
function cabin() {
  const c = {
    out: -100,          // dBFS being rendered
    mic: -100,          // dBFS at the microphone
    reads: 0,
    /* RIO speaking, at driving volume. */
    rioSpeaks(db) { c.out = db === undefined ? -12 : db; c.echoOnly(); },
    rioSilent() { c.out = -100; c.mic = -100; },
    /* Her voice returning through the cabin: 8 dB down on what produced it,
       which is generous — a phone loudspeaker into its own microphone is
       usually closer. */
    echoOnly() { c.mic = c.out - 8; },
    /* A driver talking over her: above the speaker, not below it. */
    driverSpeaks(overBy) { c.mic = c.out + (overBy === undefined ? 12 : overBy); },
    levels() { c.reads++; return { mic: c.mic, out: c.out }; },
  };
  return c;
}

/* One controller, wired the way connect() wires it for the device asked for. */
function session(device, room, extra) {
  const arbiter = speech.makeArbiter();
  const sent = [];
  const events = [];
  const audio = { muted: false, mutes: 0, unmutes: 0 };
  const touch = device === 'touch';
  const cfg = Object.assign({
    arbiter: arbiter,
    send: (obj) => sent.push(obj),
    tool: () => Promise.resolve({ ok: true }),
    audio: {
      mute: () => { audio.muted = true; audio.mutes++; },
      unmute: () => { audio.muted = false; audio.unmutes++; },
    },
    onEvent: (ev) => events.push(ev),
    bargeSustainMs: scaled(touch ? CFG.sustainTouch : CFG.sustain),
    bargeConfirmMs: scaled(CFG.confirm),
    bargeOnsetGuardMs: scaled(touch ? CFG.onsetTouch : CFG.onset),
    bargeEchoMarginDb: touch ? CFG.marginTouch : CFG.margin,
    bargeEchoFloorDb: CFG.floor,
    levels: room ? room.levels : null,
    maxResumes: 1,
    resumeInstruction: 'RESUME>>',
  }, extra || {});
  const controller = rt.createController(cfg);
  return {
    arbiter, sent, events, audio, controller, room,
    cutoffs: () => controller.state().cutoffs,
    counters: () => controller.state().counters,
    evTypes: () => events.map(e => e.type),
    /* She starts an answer and her voice reaches the speaker. */
    async begin(rid) {
      controller.handle({ type: 'response.created', response: { id: rid } });
      await sleep(2);
      controller.handle({ type: 'response.output_audio_transcript.delta',
                          response_id: rid, delta: 'Answering. ' });
      if (room && room.rioSpeaks) room.rioSpeaks();
      await sleep(2);
    },
    async say(rid, words) {
      controller.handle({ type: 'response.output_audio_transcript.delta',
                          response_id: rid, delta: words });
      await sleep(2);
    },
    async finish(rid) {
      controller.handle({ type: 'response.done',
                          response: { id: rid, status: 'completed' } });
      if (room && room.rioSilent) room.rioSilent();
      await sleep(2);
    },
    /* The detector firing, and stopping. */
    started() { controller.handle({ type: 'input_audio_buffer.speech_started' }); },
    stopped() { controller.handle({ type: 'input_audio_buffer.speech_stopped' }); },
    transcript(text) {
      controller.handle({ type: 'conversation.item.input_audio_transcription.completed',
                          transcript: text });
    },
  };
}

/* Her own voice, firing the detector, for `ms` — the phone case. Nothing else
   in the cabin, so no transcript ever follows. */
async function echoBurst(h, ms) {
  h.room.echoOnly();
  h.started();
  await sleep(ms);
  h.stopped();
  await sleep(4);
}

/* A driver talking over her: louder than the speaker, and long enough to mean
   it. The transcript follows, because somebody actually spoke. */
async function driverInterrupts(h, ms, text) {
  h.room.driverSpeaks();
  h.started();
  await sleep(ms);
  h.stopped();
  await sleep(4);
  h.transcript(text === undefined ? 'wait, what about the exit' : text);
  await sleep(4);
}

async function main() {

// ---------------------------------------------------------------------------
section('A. a phone, five long answers, nobody in the car');
// ---------------------------------------------------------------------------
{
  const room = cabin();
  const h = session('touch', room);
  // Long enough to be past any gate if the gate were only about time: each
  // burst outlasts the touch sustain window.
  const burst = scaled(CFG.sustainTouch) + scaled(400);
  for (let i = 1; i <= 5; i++) {
    const rid = 'r' + i;
    await h.begin(rid);
    await echoBurst(h, burst);
    await h.say(rid, 'still talking. ');
    await echoBurst(h, burst);
    await h.finish(rid);
    await sleep(scaled(CFG.confirm) + 10);
  }
  const c = h.cutoffs();
  ok(c.false_barge_in === 0,
     'zero false barge-ins across five answers (was 5 before the gate; got ' +
     c.false_barge_in + ')');
  ok(c.barge_in === 0, 'and nothing was blamed on a driver who was not there');
  ok(h.counters().responses === 5, 'all five answers were spoken');
  ok(h.counters().echo_suppressed >= 10,
     'every firing was suppressed as echo (' +
     h.counters().echo_suppressed + ' of 10)');
  ok(h.audio.mutes === 0,
     'she was never muted — the answer plays through without a gap');
  ok(room.reads > 0, 'the meter was actually consulted (' + room.reads + ' reads)');
}

// ---------------------------------------------------------------------------
section('B. the same phone, and this time the driver really does talk');
// ---------------------------------------------------------------------------
{
  const room = cabin();
  const h = session('touch', room);
  await h.begin('r1');
  await echoBurst(h, scaled(200));               // her own voice first
  ok(h.cutoffs().false_barge_in === 0, 'the echo costs nothing');

  await driverInterrupts(h, scaled(CFG.sustainTouch) + scaled(300));
  ok(h.audio.mutes >= 1, 'a driver over the top of her mutes her instantly');
  ok(h.cutoffs().barge_in === 1,
     'and is classified as a real barge-in once the transcript lands');
  ok(h.cutoffs().false_barge_in === 0, 'not as echo');
  const cancel = h.sent.filter(e => e.type === 'response.cancel');
  ok(cancel.length === 1, 'her generation was cancelled, not just muted');
}

// ---------------------------------------------------------------------------
section('C. a driver who cuts in on her opening word');
// ---------------------------------------------------------------------------
/* The onset guard is the part of this change most able to break a real
   interruption, so it gets its own section: a driver talking over her FIRST
   SYLLABLE must still stop her. Deferred, never discarded. */
{
  const room = cabin();
  const h = session('touch', room);
  await h.begin('r1');
  // Inside the guard window, and it keeps going past it.
  room.driverSpeaks();
  h.started();
  await sleep(scaled(CFG.onsetTouch) + scaled(CFG.sustainTouch) + scaled(300));
  ok(h.evTypes().indexOf('LIVE_BARGE_DEFERRED') >= 0,
     'the decision was held over her opening syllable, not taken');
  ok(h.counters().onset_deferred === 1,
     'and then taken, once the guard expired and the speech was still going');
  ok(h.audio.mutes >= 1, 'she is muted — a driver at the start still stops her');
  h.stopped();
  await sleep(4);
  h.transcript('no, the other one');
  await sleep(4);
  ok(h.cutoffs().barge_in === 1, 'classified as the real interruption it was');
}
{
  const room = cabin();
  const h = session('touch', room);
  await h.begin('r1');
  // ...and a blip inside the guard that stops on its own costs nothing at all.
  room.echoOnly();
  h.started();
  await sleep(scaled(CFG.onsetTouch) / 2);
  h.stopped();
  await sleep(scaled(CFG.confirm) + 10);
  ok(h.audio.mutes === 0 && h.cutoffs().false_barge_in === 0,
     'a blip inside the guard that stops on its own costs nothing');
  ok(h.counters().echo_suppressed === 1, 'and is counted as what it was');
}

// ---------------------------------------------------------------------------
section('D. a desk is a different room, and nothing about it changed');
// ---------------------------------------------------------------------------
{
  // The desktop column, with no meter at all — which is what a desktop
  // session gets, because makeMeter is only built when a margin is configured.
  const h = session('desktop', null);
  const p = h.controller.state().policy;
  ok(p.barge_sustain_ms === scaled(CFG.sustain),
     'desktop sustain is the value it always was (' + CFG.sustain + ' ms)');
  ok(p.barge_onset_guard_ms === 0, 'no onset guard on a desk');
  ok(p.barge_echo_margin_db === 0, 'no level test on a desk');
  ok(p.echo_meter === false, 'and no meter is built at all');

  await h.begin('r1');
  await h.say('r1', 'the answer so far. ');
  h.started();
  await sleep(scaled(CFG.sustain) + scaled(200));
  ok(h.audio.mutes === 1, 'the detector still mutes her instantly');
  h.stopped();
  await sleep(scaled(CFG.confirm) + 20);
  ok(h.cutoffs().false_barge_in === 1,
     'a sustained firing with no transcript is still a false barge-in');
  ok(h.sent.filter(e => e.type === 'response.create' &&
                   /^RESUME>>/.test((e.response || {}).instructions || '')).length === 1,
     'and the answer it cost is still resumed');
}
{
  // The same events, this time under a phone: the difference is the whole
  // point, so it is asserted against the desk case rather than on its own.
  const room = cabin();
  const h = session('touch', room);
  await h.begin('r1');
  await h.say('r1', 'the answer so far. ');
  room.echoOnly();
  h.started();
  await sleep(scaled(CFG.sustainTouch) + scaled(200));
  h.stopped();
  await sleep(scaled(CFG.confirm) + 20);
  ok(h.cutoffs().false_barge_in === 0 && h.audio.mutes === 0,
     'the identical event stream costs a phone nothing');
}

// ---------------------------------------------------------------------------
section('E. the gate needs evidence, and without it it stands down');
// ---------------------------------------------------------------------------
{
  // No meter on a touch device — Web Audio refused, or the track never
  // arrived. The level test must not silently swallow interruptions.
  const h = session('touch', null);
  await h.begin('r1');
  h.started();
  await sleep(scaled(CFG.onsetTouch) + scaled(CFG.sustainTouch) + scaled(200));
  h.stopped();
  await sleep(scaled(CFG.confirm) + 20);
  ok(h.cutoffs().false_barge_in === 1,
     'with no meter, a sustained firing is treated exactly as it was before');
  ok(h.audio.mutes >= 1, 'she is still muted — absence of evidence is not echo');
}
{
  // A meter that throws on every read is the same situation, arrived at
  // differently.
  const h = session('touch', { levels: () => { throw new Error('no analyser'); } });
  await h.begin('r1');
  h.started();
  await sleep(scaled(CFG.onsetTouch) + scaled(CFG.sustainTouch) + scaled(200));
  h.stopped();
  await sleep(scaled(CFG.confirm) + 20);
  ok(h.cutoffs().false_barge_in === 1,
     'a meter that throws is a meter that says nothing, not one that says echo');
}
{
  // Output below the floor: she is not making a sound, so nothing the
  // microphone hears can be her.
  const room = cabin();
  const h = session('touch', room);
  await h.begin('r1');
  room.out = CFG.floor - 20;          // below the floor
  room.mic = CFG.floor - 22;          // quiet, and still below the output
  h.started();
  await sleep(scaled(CFG.onsetTouch) + scaled(CFG.sustainTouch) + scaled(200));
  h.stopped();
  await sleep(scaled(CFG.confirm) + 20);
  ok(h.cutoffs().false_barge_in === 1,
     'below the floor the level test is skipped — nothing that quiet echoes');
}

// ---------------------------------------------------------------------------
section('F. a warning still cuts through everything, echo gate or not');
// ---------------------------------------------------------------------------
{
  /* The one thing the gate must never do is make RIO harder to interrupt for
     the systems that outrank the driver. A dictation was never yielded to a
     barge-in and still is not; and a P1 warning taking the mouth is not a
     barge-in at all, so no part of this change is on its path. */
  const room = cabin();
  const h = session('touch', room);
  await h.begin('r1');
  const before = h.audio.mutes;
  h.arbiter.say({ priority: speech.P.SAFETY, group: 'headway', id: 'w1',
                  text: 'too close', maxMs: 2000,
                  play: () => new Promise(() => {}), stop: () => {} });
  await sleep(10);
  ok(h.audio.mutes > before,
     'a gap warning takes the mouth from her immediately — unchanged');
  ok(h.cutoffs().preempted === 1,
     'and is recorded as a pre-emption, not as anything to do with echo');
}

// ---------------------------------------------------------------------------
section('G. every line still has a mouth when the shared output is down');
// ---------------------------------------------------------------------------
/* rio_speak.js now tries RIO.output first for clips and for synthesised
   lines. The fallback to the element is what plays the warning when the bus is
   not there, and a fallback that is never exercised is a fallback that does
   not work. */
{
  delete require.cache[require.resolve(path.join(STATIC, 'rio_speak.js'))];
  /* rio_speak.js binds `root` at load: `window` if there is one, otherwise the
     global. Node has neither until one is made, so the stub page is built
     BEFORE the require and every bus in this section is installed on it. */
  global.window = global.window || {};
  const page = global.window;
  delete page.RIO;
  const speak = require(path.join(STATIC, 'rio_speak.js'));

  /* An <audio> element, as far as this file is concerned: something with a
     src, a play() and the two callbacks it settles on. */
  function element() {
    const el = {
      src: '', muted: true, played: 0, onended: null, onerror: null,
      play() { el.played++; return Promise.resolve(); },
      pause() {},
    };
    return el;
  }

  /* Play one line and let whatever ends up playing it finish. The element path
     settles on `onended`, and which path is taken is the thing under test, so
     the callback is waited for rather than assumed to be installed already. */
  async function playLine(opts) {
    const p = speak.provider(opts);
    const done = p.play();
    for (let i = 0; i < 40 && !opts.element.onended; i++) await sleep(2);
    if (opts.element.onended) opts.element.onended();
    await done;
  }

  const el = element();
  await playLine({ text: 'left here', channel: 'nav',
                   clipUrl: '/static/audio/left_here.mp3',
                   clipFirst: true, clipElement: el, element: el });
  ok(el.played === 1, 'with no bus, the clip plays on its element as it always did');
  ok(speak.stats().bus_missed >= 1, 'and the miss is counted rather than hidden');

  // ...and a bus that says it is not ready is not asked to play anything.
  let asked = 0;
  page.RIO.output = { ready: () => false,
                      playUrl: () => { asked++; throw new Error('nope'); } };
  const el2 = element();
  await playLine({ text: 'right here', channel: 'nav',
                   clipUrl: '/static/audio/right_here.mp3',
                   clipFirst: true, clipElement: el2, element: el2 });
  ok(el2.played === 1 && asked === 0,
     'a bus that is not ready is not asked to play anything');

  // ...and a bus that accepts the line and then fails still ends in audio.
  let fallbacks = 0;
  page.RIO.output = {
    ready: () => true,
    playUrl: () => ({ play: () => Promise.reject(new Error('decode failed')),
                      abort: () => {} }),
    noteFallback: () => { fallbacks++; },
  };
  const el3 = element();
  await playLine({ text: 'left here', channel: 'nav',
                   clipUrl: '/static/audio/left_here.mp3',
                   clipFirst: true, clipElement: el3, element: el3 });
  ok(el3.played === 1,
     'a bus that fails mid-play hands the line back to the element');
  ok(fallbacks === 1, 'and the fallback is counted where the bus can see it');

  // A working bus plays the line and the element is never touched.
  let onBus = 0;
  page.RIO.output = {
    ready: () => true,
    playUrl: () => ({ play: () => { onBus++; return Promise.resolve(); },
                      abort: () => {} }),
    noteFallback: () => {},
  };
  const el4 = element();
  const p4 = speak.provider({ text: 'left here', channel: 'nav',
                              clipUrl: '/static/audio/left_here.mp3',
                              clipFirst: true, clipElement: el4, element: el4 });
  await p4.play();
  ok(onBus === 1 && el4.played === 0,
     'and when the bus works, the line goes out through it and nowhere else');
  delete page.RIO.output;
}

// ---------------------------------------------------------------------------
section('H. the policy travels, and the page picks its own column');
// ---------------------------------------------------------------------------
{
  const src = fs.readFileSync(path.join(STATIC, 'rio_realtime.js'), 'utf8');
  ok(/session\.barge_touch/.test(src) && /session\.barge_desktop/.test(src),
     'both columns are read off the session rather than written in the page');
  ok(/function isTouchDevice/.test(src),
     'and the device is decided in the browser, where it is known');
  const py = fs.readFileSync(path.join(REPO, 'realtime.py'), 'utf8');
  ok(/"barge_touch"/.test(py) && /"barge_desktop"/.test(py),
     'the server sends both columns with the session');
  ok(CFG.sustainTouch > CFG.sustain,
     'a phone waits longer than a desk before an answer is cancelled (' +
     CFG.sustainTouch + ' vs ' + CFG.sustain + ' ms)');
  ok(CFG.onsetTouch > 0 && CFG.marginTouch > 0,
     'and has both an onset guard and a level margin');
  ok(CFG.onset === 0 && CFG.margin === 0,
     'while a desk has neither — the desktop path is byte for byte what it was');
}

// ---------------------------------------------------------------------------
section('I. one output, and the paths that are still outside it');
// ---------------------------------------------------------------------------
{
  const out = fs.readFileSync(path.join(STATIC, 'rio_output.js'), 'utf8');
  ok(/createMediaStreamDestination/.test(out) && /RTCPeerConnection/.test(out),
     'the shared output renders through a peer-connection loopback, which is ' +
     'the path iOS cancels');
  ok(/bus\.connect\(c\.destination\)/.test(out),
     'and starts wired straight out, so audio exists before the loopback does');
  const eleven = fs.readFileSync(path.join(STATIC, 'rio_voice_eleven.js'), 'utf8');
  ok(/shared\.node\(\)/.test(eleven),
     'the synthesiser renders into the shared output rather than the speaker');
  ok(/out \|\| ctx\.destination/.test(eleven),
     '...and falls back to the speaker if there is no shared output');
  const spk = fs.readFileSync(path.join(STATIC, 'rio_speak.js'), 'utf8');
  ok(/b\.playUrl\(url\)/.test(spk) && /b\.playBlob\(blob\)/.test(spk),
     'clips and synthesised lines both go through it');
}

console.log('\n' + (failures ? 'FAILED ' + failures + ' of ' + checks
                             : 'PASSED ' + checks + ' checks'));
process.exit(failures ? 1 : 0);
}

main().catch(e => { console.error(e); process.exit(1); });
