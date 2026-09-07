/* echo_barge_probe.js — is RIO interrupting herself on a phone?
 *
 *   node tools/echo_barge_probe.js
 *   node tools/echo_barge_probe.js --answers 5 --burst 700
 *   node tools/echo_barge_probe.js --device desktop
 *
 * THE REPORT. On iPhone, RIO's own voice out of the speaker flips the session
 * to "listening" and mutes her mid-answer. Two things have to be established
 * before anything is changed, and this file establishes both:
 *
 *   A. WHICH OUTPUT PATHS iOS CAN CANCEL. Safari's echo canceller is part of
 *      the WebRTC capture path: it has a reference signal for audio it renders
 *      itself, and none for audio the page plays some other way. So the
 *      question "does RIO's voice come back into the microphone" has a
 *      different answer per path, and this reads every path out of the source
 *      rather than out of anyone's memory of it.
 *
 *   B. WHAT THAT COSTS. The real controller from static/rio_realtime.js, five
 *      long answers, nobody in the car saying anything, and the detector
 *      firing on her own voice the way it does on a phone. Every LIVE_CUTOFF
 *      is logged with its cause. A false_barge_in here is an answer the driver
 *      would have had to ask for again.
 *
 * The thresholds come out of config.py, so this measures what is shipped
 * rather than what is written here. --device touch (the default) uses the
 * touch column if there is one; --device desktop uses the desktop column, so
 * the two can be compared with one flag.
 *
 * This is a PROBE, not an acceptance test: it reports numbers and exits 0
 * whatever they are. The pass/fail version is tools/echo_barge_selftest.js.
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

function arg(name, dflt) {
  const i = process.argv.indexOf('--' + name);
  return i >= 0 && process.argv[i + 1] ? process.argv[i + 1] : dflt;
}

const sleep = (ms) => new Promise(r => setTimeout(r, ms));

// ---------------------------------------------------------------------------
// The numbers under test, read from where they actually live.
// ---------------------------------------------------------------------------
function configInts() {
  const src = fs.readFileSync(path.join(REPO, 'config.py'), 'utf8');
  const read = (name, dflt) => {
    const m = new RegExp('^' + name + '\\s*=\\s*(\\d+)', 'm').exec(src);
    return m ? parseInt(m[1], 10) : dflt;
  };
  return {
    sustain: read('REALTIME_BARGE_SUSTAIN_MS', 300),
    sustainTouch: read('REALTIME_BARGE_SUSTAIN_MS_TOUCH', null),
    confirm: read('REALTIME_BARGE_CONFIRM_MS', 1500),
    onsetTouch: read('REALTIME_BARGE_ONSET_GUARD_MS_TOUCH', null),
    marginTouch: read('REALTIME_BARGE_ECHO_MARGIN_DB_TOUCH', null),
  };
}

// ---------------------------------------------------------------------------
// A. Where RIO's voice comes out, and whether iOS can cancel it
//
// Three renderers, and only one of them is inside the echo canceller:
//
//   webrtc   a remote MediaStream on an <audio> element. Rendered by the same
//            WebRTC stack that owns the capture, so it IS the reference signal
//            the canceller subtracts. This is the shipped conversational voice
//            under VOICE_BACKEND=openai_realtime.
//   element  an <audio> element playing a URL or a blob. Ordinary media
//            playback: the canceller has no reference for it.
//   webaudio an AudioContext rendering to ctx.destination. Same again.
//
// Classified from the construct, not from a list kept by hand -- a new
// `new Audio()` anywhere under static/ shows up here the day it is written.
// ---------------------------------------------------------------------------
const PATH_KINDS = [
  /* A camera preview is a MediaStream on an element too, and it renders no
     audio at all. Anything called `video` is the picture, not the voice. */
  { kind: 'video', covered: null, skip: true,
    re: /^\s*(?:var\s+)?(?:\w+\.)?(?:v|video|el)\b\w*\.srcObject\s*=/i,
    what: 'a camera preview — no audio' },
  { kind: 'webrtc', covered: true,
    re: /^\s*(?:var\s+)?(\w+(?:\.\w+)*)\.srcObject\s*=\s*(?!null)/,
    what: 'MediaStream on an element — rendered by WebRTC' },
  { kind: 'webaudio', covered: false,
    re: /\.connect\(\s*(\w+)\.destination\s*\)/,
    what: 'AudioContext → destination' },
  { kind: 'element', covered: false,
    re: /new Audio\s*\(/,
    what: 'a fresh <audio> element' },
  { kind: 'element', covered: false,
    re: /^\s*(?:var\s+)?\w*[eE]l\w*\.src\s*=\s*(?!null)/,
    what: 'a URL or blob on an <audio> element' },
];

function auditPaths() {
  const files = fs.readdirSync(STATIC)
    .filter(f => /\.js$/.test(f))
    .concat(['index.html'])
    .map(f => path.join(STATIC, f))
    .filter(f => fs.existsSync(f));
  const found = [];
  for (const file of files) {
    const lines = fs.readFileSync(file, 'utf8').split('\n');
    lines.forEach((line, i) => {
      // Comments describe these constructs constantly in this codebase; only
      // code counts.
      const code = line.replace(/^\s*(\/\/|\*|\/\*).*$/, '');
      if (!code.trim()) return;
      for (const k of PATH_KINDS) {
        if (k.re.test(code)) {
          if (!k.skip) {
            found.push({ file: path.basename(file), line: i + 1, kind: k.kind,
                         covered: k.covered, what: k.what, code: code.trim() });
          }
          break;
        }
      }
    });
  }
  return found;
}

// ---------------------------------------------------------------------------
// B. A phone-shaped session: five long answers, an empty car
// ---------------------------------------------------------------------------
/* WHAT THE PHONE ACTUALLY DOES, and what is being reproduced here.
 *
 * The turn detector is the model's, upstream, and it fires on whatever reaches
 * the microphone. On a phone in a car that includes RIO, so a
 * `speech_started` arrives shortly after she starts talking and a
 * `speech_stopped` arrives when the echo drops below the threshold again. No
 * transcript ever follows, because nobody said anything.
 *
 * That is the whole shape of the bug and it needs no audio to reproduce: the
 * events are the same events, and what they cost is decided entirely by the
 * controller's gate. */
/* THE CABIN, as two numbers: what the page is rendering, and what the
   microphone hears. Echo is the loudspeaker arriving back attenuated -- the
   microphone tracks the output from BELOW, which is the whole basis of the
   level test. --no-meter runs the same session with no measurement at all,
   which is what the code did before this existed and what a browser with no
   Web Audio still does. */
function cabin() {
  const c = {
    out: -100, mic: -100, reads: 0,
    rioSpeaks() { c.out = -12; c.mic = -20; },
    rioSilent() { c.out = -100; c.mic = -100; },
    levels() { c.reads++; return { mic: c.mic, out: c.out }; },
  };
  return c;
}

function phoneSession(opts) {
  const arbiter = speech.makeArbiter();
  const sent = [];
  const events = [];
  const audio = { muted: false, mutes: 0, unmutes: 0 };
  const levels = opts.levels || null;
  const controller = rt.createController({
    arbiter: arbiter,
    send: (obj) => sent.push(obj),
    tool: () => Promise.resolve({ ok: true }),
    audio: {
      mute: () => { audio.muted = true; audio.mutes++; },
      unmute: () => { audio.muted = false; audio.unmutes++; },
    },
    onEvent: (ev) => events.push(ev),
    bargeSustainMs: opts.sustain,
    bargeConfirmMs: opts.confirm,
    bargeOnsetGuardMs: opts.onsetGuard,
    bargeEchoMarginDb: opts.marginDb,
    levels: levels,
    maxResumes: opts.maxResumes === undefined ? 1 : opts.maxResumes,
    resumeInstruction: 'RESUME>>',
  });
  return { arbiter, sent, events, audio, controller, room: opts.room || null };
}

/* One answer, spoken over `speakMs`, with the detector firing on her own voice
   `bursts` times inside it. Nobody else is in the car: no transcription events
   are ever delivered, which is exactly what the transcriber does with echo it
   was never asked to transcribe. */
async function speakOneAnswer(h, n, plan, log) {
  const rid = 'r' + n;
  h.controller.handle({ type: 'response.created', response: { id: rid } });
  await sleep(10);
  // Her audio starts a beat after the response opens, and the echo is a beat
  // after that.
  h.controller.handle({ type: 'response.output_audio_transcript.delta',
                        response_id: rid, delta: 'This is answer ' + n + '. ' });
  if (h.room) h.room.rioSpeaks();
  let spent = 0;
  for (let b = 0; b < plan.bursts; b++) {
    await sleep(plan.gapMs);
    spent += plan.gapMs;
    if (h.controller.state().speaking === null && b > 0) break;
    log.push({ t: Date.now(), what: 'echo burst ' + (b + 1) + ' of answer ' + n });
    h.controller.handle({ type: 'input_audio_buffer.speech_started' });
    await sleep(plan.burstMs);
    spent += plan.burstMs;
    h.controller.handle({ type: 'input_audio_buffer.speech_stopped' });
    // She keeps talking underneath it, so the transcript keeps growing.
    h.controller.handle({ type: 'response.output_audio_transcript.delta',
                          response_id: rid, delta: 'more words. ' });
  }
  await sleep(Math.max(0, plan.speakMs - spent));
  h.controller.handle({ type: 'response.done',
                        response: { id: rid, status: 'completed' } });
  if (h.room) h.room.rioSilent();
  // Long enough for the confirmation window to close on anything pending.
  await sleep(plan.confirmMs + 120);
}

async function main() {
  const cfg = configInts();
  const device = arg('device', 'touch');
  const answers = parseInt(arg('answers', '5'), 10);
  const burstMs = parseInt(arg('burst', '700'), 10);

  console.log('=== A. how RIO\'s audio is played, and what iOS can cancel ===\n');
  const found = auditPaths();
  const covered = found.filter(f => f.covered);
  const bare = found.filter(f => !f.covered);
  const pad = (s, n) => (s + ' '.repeat(n)).slice(0, n);
  console.log('  ' + pad('WHERE', 34) + pad('RENDERER', 10) + 'iOS AEC');
  console.log('  ' + '-'.repeat(60));
  for (const f of found.slice().sort((a, b) => (a.covered === b.covered ? 0 : a.covered ? -1 : 1))) {
    console.log('  ' + pad(f.file + ':' + f.line, 34) + pad(f.kind, 10) +
                (f.covered ? 'covered' : 'NOT CANCELLED') + '   ' + f.what);
  }
  console.log('\n  ' + covered.length + ' path(s) inside the canceller, ' +
              bare.length + ' outside it.');
  console.log('  Everything outside it returns to the microphone at full ' +
              'level while she speaks,\n  and the turn detector upstream ' +
              'cannot tell it from a driver.');

  console.log('\n=== B. five long answers, nobody talking ===\n');
  const sustain = (device === 'touch' && cfg.sustainTouch) || cfg.sustain;
  const onsetGuard = (device === 'touch' && cfg.onsetTouch) || 0;
  const marginDb = (device === 'touch' && cfg.marginTouch) || 0;
  console.log('  device        ' + device);
  console.log('  sustain       ' + sustain + ' ms' +
              (cfg.sustainTouch ? '  (desktop ' + cfg.sustain + ')' : ''));
  console.log('  confirm       ' + cfg.confirm + ' ms');
  console.log('  onset guard   ' + (onsetGuard ? onsetGuard + ' ms' : 'none'));
  console.log('  echo margin   ' + (marginDb ? marginDb + ' dB' : 'none'));
  console.log('  echo bursts   ' + burstMs + ' ms, 2 per answer, nobody speaking\n');

  const noMeter = process.argv.indexOf('--no-meter') >= 0;
  const room = (device === 'touch' && !noMeter) ? cabin() : null;
  console.log('  meter         ' + (room ? 'on (modelled cabin)' : 'off'));
  const h = phoneSession({ sustain: sustain, confirm: cfg.confirm,
                           onsetGuard: onsetGuard, marginDb: marginDb,
                           room: room, levels: room ? room.levels : null });
  const log = [];
  const plan = { bursts: 2, burstMs: burstMs, gapMs: 300, speakMs: 2400,
                 confirmMs: cfg.confirm };
  const t0 = Date.now();
  for (let i = 1; i <= answers; i++) await speakOneAnswer(h, i, plan, log);

  const cutoffs = h.controller.state().cutoffs;
  const counters = h.controller.state().counters;
  const kinds = {};
  h.events.filter(e => e.type === 'LIVE_CUTOFF').forEach(e => {
    kinds[e.cause] = (kinds[e.cause] || 0) + 1;
  });

  console.log('  LIVE_CUTOFF, by cause:');
  const causes = Object.keys(cutoffs);
  let total = 0;
  causes.forEach(c => {
    total += cutoffs[c];
    if (cutoffs[c]) console.log('    ' + pad(c, 18) + cutoffs[c]);
  });
  if (!total) console.log('    (none)');
  console.log('\n  answers spoken            ' + counters.responses);
  console.log('  detector fired            ' + counters.barge_ins);
  console.log('  absorbed under the gate   ' + counters.blips_absorbed);
  console.log('  suppressed as echo        ' +
              (counters.echo_suppressed === undefined ? 'n/a (no gate yet)'
                                                      : counters.echo_suppressed));
  console.log('  FALSE BARGE-INS           ' + (cutoffs.false_barge_in || 0) +
              '   <-- answers lost to her own voice');
  console.log('  audio muted               ' + h.audio.mutes + ' times' +
              (h.audio.mutes ? ' (each one is RIO going quiet mid-word)' : ''));
  console.log('\n  ' + ((Date.now() - t0) / 1000).toFixed(1) + 's of session.');
  console.log('\n  Read this as: ' + (cutoffs.false_barge_in || 0) +
              ' of ' + answers + ' answers were cut off by an empty car.');
}

main().catch(e => { console.error(e); process.exit(1); });
