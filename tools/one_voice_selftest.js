/* one_voice_selftest.js — one utterance, one mouth.
 *
 *   node tools/one_voice_selftest.js
 *
 * THE BUG. Starting navigation on a phone, the turn-by-turn came out twice at
 * once: ElevenLabs reading the line while the live session read the same line
 * in marin. Two voices, same words, same junction.
 *
 * IT WAS NOT THE ELEVENLABS PATH BEING RE-ENABLED. That path is the FALLBACK
 * and it is supposed to be reachable — a warning never waits on a cloud call it
 * is not getting. What was wrong is what happened when it fired:
 *
 *   rio_speak.js asks the live session to dictate the line and gives it a
 *   budget (900 ms for the call at the junction). The budget expires, it gives
 *   up and synthesises instead. But the response it asked for is still in
 *   flight: `response.cancel` at that moment names nothing, because the
 *   response does not exist yet. It gets created a beat later, belongs to
 *   nobody, claims the mouth as an ordinary answer, and reads the turn out
 *   underneath the synthesiser reading the turn.
 *
 * So the fix is not to remove a path, it is to make abandoning one final: the
 * disowned response is cancelled BY ID the moment it exists and muted while it
 * dies. What this file asserts is the invariant that follows — exactly one
 * mouth per utterance, in every ordering of those events.
 *
 * Driven against the REAL controller from static/rio_realtime.js and the REAL
 * provider from static/rio_speak.js, with the wire and the audio stubbed: the
 * whole bug is a race between a timer and a message, and both are injectable.
 */
'use strict';

const fs = require('fs');
const path = require('path');

const REPO = path.join(__dirname, '..');
const STATIC = path.join(REPO, 'static');

if (typeof global.atob !== 'function') {
  global.atob = (b64) => Buffer.from(b64, 'base64').toString('binary');
}
/* The element path hands a blob to the browser through an object URL. Node has
   neither, and what is under test is which mouth opened, not the plumbing
   inside one -- so both are stubbed rather than avoided. */
global.URL = global.URL || {};
if (typeof global.URL.createObjectURL !== 'function') {
  global.URL.createObjectURL = () => 'blob:fake';
  global.URL.revokeObjectURL = () => {};
}
// rio_speak.js binds `root` at load. Build the page first so every stub in
// this file lands on the object it actually reads.
global.window = global.window || {};
const page = global.window;

const rt = require(path.join(STATIC, 'rio_realtime.js'));
const speech = require(path.join(STATIC, 'rio_speech.js'));
const speak = require(path.join(STATIC, 'rio_speak.js'));

let checks = 0, failures = 0;
function ok(cond, what) {
  checks++;
  if (!cond) { failures++; console.log('  FAIL  ' + what); }
  else console.log('  ok    ' + what);
}
function section(name) { console.log('\n=== ' + name + ' ==='); }
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

/* ---------------------------------------------------------------------------
   A car with one speaker in it.

   Every mouth on the page reports here, and the ledger is the whole point: two
   overlapping entries is the bug, whatever produced them.
   --------------------------------------------------------------------------- */
function cabin() {
  const spoken = [];        // { mouth, text, start, end }
  let open = 0, worst = 0;
  return {
    spoken,
    begin(mouth, text) {
      open++;
      worst = Math.max(worst, open);
      const entry = { mouth, text, done: false };
      spoken.push(entry);
      return entry;
    },
    end(entry) { if (entry && !entry.done) { entry.done = true; open--; } },
    overlapped() { return worst > 1; },
    mouths() { return spoken.map(s => s.mouth); },
    texts() { return spoken.map(s => s.text); },
  };
}

/* One live session, on a fake wire, with the audio element and the network
   both reporting into the cabin. */
function session(room, opts) {
  opts = opts || {};
  const arbiter = speech.makeArbiter();
  const sent = [];
  const events = [];
  const audio = { muted: false, mutes: 0, unmutes: 0 };
  const controller = rt.createController({
    arbiter: arbiter,
    send: (obj) => sent.push(obj),
    tool: () => Promise.resolve({ ok: true }),
    audio: {
      mute: () => { audio.muted = true; audio.mutes++; },
      unmute: () => { audio.muted = false; audio.unmutes++; },
    },
    onEvent: (ev) => events.push(ev),
    verbatimInstruction: 'READ:\n',
    speakTimeoutMs: opts.speakTimeoutMs === undefined ? 30 : opts.speakTimeoutMs,
    bargeSustainMs: 4,
    bargeConfirmMs: 8,
  });

  /* THE SESSION'S OWN VOICE. A dictated response produces audio, and that
     audio is what the driver hears — so when a response starts speaking, the
     cabin hears it, unless the element is muted at that instant. */
  let current = null;
  function responseSpeaks(rid, text) {
    if (audio.muted) return null;               // silenced before it was heard
    current = room.begin('session', text);
    return current;
  }
  function responseEnds() { room.end(current); current = null; }

  return { arbiter, sent, events, audio, controller, responseSpeaks, responseEnds,
           types: () => sent.map(e => e.type),
           cancels: () => sent.filter(e => e.type === 'response.cancel') };
}

/* The page's `live` handle, exactly as connect() builds it for rio_speak. */
function installLive(h) {
  page.RIO = page.RIO || {};
  page.RIO.realtime = {
    active: () => ({
      speak: (text, o) => h.controller.speak(text, o),
      speakTimeout: () => undefined,
      speechEnabled: () => true,
    }),
  };
}

/* The synthesiser, as a URL that either answers or does not. Every fetch is
   recorded: "no ElevenLabs request is made during a nav drive" is a claim
   about this list. */
function installFetch(room, opts) {
  opts = opts || {};
  const calls = [];
  page.fetch = global.fetch = function (url, init) {
    calls.push(String(url));
    if (opts.fail) return Promise.reject(new Error('tts down'));
    /* Deliberately NOT a mouth. Asking the synthesiser for bytes is a request;
       what the driver hears is the element playing them, and conflating the
       two would count one utterance as two. The list is what proves "no
       ElevenLabs request during a nav drive"; the cabin proves how many voices
       were open. */
    return Promise.resolve({
      ok: true,
      headers: { get: () => null },
      blob: () => Promise.resolve({
        _fake: true, arrayBuffer: () => Promise.resolve(new ArrayBuffer(8)),
      }),
    });
  };
  return calls;
}

/* An <audio> element that reports what it plays. */
function element(room) {
  const el = {
    src: '', muted: true, played: 0, onended: null, onerror: null,
    entry: null,
    play() {
      el.played++;
      el.entry = room.begin('elevenlabs-element', el.src || '(preloaded)');
      return Promise.resolve();
    },
    pause() {},
    finish() { if (el.onended) { room.end(el.entry); el.onended(); } },
  };
  return el;
}

async function main() {

// ---------------------------------------------------------------------------
section('A. the ordinary nav call: dictated, and nothing else speaks');
// ---------------------------------------------------------------------------
{
  const room = cabin();
  const h = session(room);
  installLive(h);
  const calls = installFetch(room);
  const el = element(room);

  const p = speak.provider({ text: 'Right turn coming up onto Cloverfield',
                             channel: 'nav', callType: 'early',
                             ttsUrl: '/nav/voice?call=early', element: el });
  const done = p.play();
  await sleep(2);
  // The session picks the line up and reads it.
  h.controller.handle({ type: 'response.created', response: { id: 'd1' } });
  await sleep(1);
  const heard = h.responseSpeaks('d1', 'Right turn coming up onto Cloverfield');
  h.controller.handle({ type: 'response.output_audio_transcript.done',
                        response_id: 'd1',
                        transcript: 'Right turn coming up onto Cloverfield' });
  h.controller.handle({ type: 'response.done',
                        response: { id: 'd1', status: 'completed' } });
  h.responseEnds(heard);
  await done;

  ok(room.mouths().join(',') === 'session',
     'exactly one mouth spoke, and it was hers (' + room.mouths().join(',') + ')');
  ok(calls.length === 0,
     'no ElevenLabs request was made at all during the call');
  ok(el.played === 0, 'and the synthesiser element was never touched');
  ok(speak.stats().doubled === 0, 'nothing doubled');
}

// ---------------------------------------------------------------------------
section('B. the budget expires — and the line she was asked for turns up anyway');
// ---------------------------------------------------------------------------
/* This is the phone. The response.create is in flight when the timer fires, so
   the cancel names nothing; the response is created a beat later. Before the
   fix this is the moment ElevenLabs and marin both started talking. */
{
  speak.reset();
  const room = cabin();
  const h = session(room, { speakTimeoutMs: 20 });
  installLive(h);
  const calls = installFetch(room);
  const el = element(room);

  const p = speak.provider({ text: 'Left here', channel: 'nav',
                             callType: 'imminent',
                             ttsUrl: '/nav/voice?call=imminent', element: el });
  const done = p.play();
  await sleep(60);                       // past the budget: it has given up
  ok(calls.length === 1, 'the synthesiser was asked for the line');

  // ...and now the response she was asked for is created.
  h.controller.handle({ type: 'response.created', response: { id: 'late1' } });
  await sleep(2);
  const heard = h.responseSpeaks('late1', 'Left here');
  ok(heard === null,
     'her voice does not reach the cabin — the disowned response was muted ' +
     'before it could speak');
  ok(h.audio.muted === true, 'the element is muted while it dies');
  const byId = h.sent.filter(e => e.type === 'response.cancel' && e.response_id);
  ok(byId.length === 1 && byId[0].response_id === 'late1',
     'and it is cancelled BY ID, which the bare cancel at the timeout could ' +
     'not do because the response did not exist yet');
  ok(h.controller.state().counters.orphans_silenced === 1,
     'counted as a silenced orphan');

  h.controller.handle({ type: 'response.done',
                        response: { id: 'late1', status: 'cancelled' } });
  await sleep(2);
  ok(h.audio.muted === false, 'and the mouth is given back afterwards');

  el.finish();
  await done;
  ok(room.mouths().indexOf('session') < 0,
     'the session never spoke this line (' + room.mouths().join(',') + ')');
  ok(room.mouths().length === 1,
     'exactly one mouth in the whole exchange (' + room.mouths().join(',') + ')');
  ok(!room.overlapped(), 'at no point were two mouths open at once');
  ok(speak.stats().doubled === 0, 'and nothing was recorded as doubled');
}

// ---------------------------------------------------------------------------
section('C. she started speaking, and only then did the budget expire');
// ---------------------------------------------------------------------------
/* The other ordering, and the one that must NOT fall back: her voice is
   already out of the speaker, so synthesising the same sentence underneath it
   is the two-voices bug however the timer feels about it. */
{
  speak.reset();
  const room = cabin();
  const h = session(room, { speakTimeoutMs: 40 });
  installLive(h);
  const calls = installFetch(room);
  const el = element(room);

  const p = speak.provider({ text: 'Take this exit', channel: 'nav',
                             callType: 'imminent',
                             ttsUrl: '/nav/voice?call=imminent', element: el });
  const done = p.play();
  await sleep(2);
  h.controller.handle({ type: 'response.created', response: { id: 'd2' } });
  await sleep(1);
  // Her audio actually starts. This is the event that stands the fallback
  // down, and the difference between it and `response.created` is the whole
  // of the race: asked-for is not speaking.
  h.controller.handle({ type: 'output_audio_buffer.started', response_id: 'd2' });
  const heard = h.responseSpeaks('d2', 'Take this exit');
  await sleep(80);                       // ...and the budget goes by

  ok(calls.length === 0,
     'no synthesiser request was made — she is already saying it');
  ok(el.played === 0, 'and no element played underneath her');
  h.controller.handle({ type: 'response.done',
                        response: { id: 'd2', status: 'completed' } });
  h.responseEnds(heard);
  await done;
  ok(room.mouths().join(',') === 'session', 'one mouth, hers');
  ok(!room.overlapped(), 'no overlap');
}

// ---------------------------------------------------------------------------
section('D. no session at all: the synthesiser is the mouth, alone');
// ---------------------------------------------------------------------------
{
  speak.reset();
  const room = cabin();
  page.RIO.realtime = { active: () => null };
  const calls = installFetch(room);
  const el = element(room);
  const p = speak.provider({ text: 'Left at the roundabout', channel: 'nav',
                             ttsUrl: '/nav/voice?call=early', element: el });
  const done = p.play();
  for (let i = 0; i < 40 && !el.onended; i++) await sleep(2);
  el.finish();
  await done;
  ok(calls.length === 1, 'the line is synthesised — a turn is still called');
  ok(!room.overlapped(), 'and it is the only voice');
}

// ---------------------------------------------------------------------------
section('E. the pre-rendered clip still bypasses everything');
// ---------------------------------------------------------------------------
/* The imminent call's whole point: a decoded buffer and a play(), no network,
   no mouth to wait for. Nothing in this change may have put a cloud call back
   in its path. */
{
  speak.reset();
  const room = cabin();
  const h = session(room);
  installLive(h);
  const calls = installFetch(room);
  const el = element(room);
  const p = speak.provider({ text: 'Left here', channel: 'nav',
                             callType: 'imminent',
                             clipUrl: '/static/audio/left_here.mp3',
                             clipFirst: true, clipElement: el, element: el,
                             ttsUrl: '/nav/voice?call=imminent' });
  const done = p.play();
  for (let i = 0; i < 40 && !el.onended; i++) await sleep(2);
  el.finish();
  await done;
  ok(calls.length === 0,
     'the clip plays with no request of any kind in its path');
  ok(!room.overlapped(), 'one mouth');
  ok(h.sent.filter(e => e.type === 'response.create').length === 0,
     'and the session was not asked to dictate it either');
}

// ---------------------------------------------------------------------------
section('F. the invariant, stated where it can be checked at runtime');
// ---------------------------------------------------------------------------
{
  const src = fs.readFileSync(path.join(STATIC, 'rio_speak.js'), 'utf8');
  ok(/if \(dictationStarted\) \{/.test(src),
     'rio_speak refuses to synthesise a line her voice has already started');
  ok(/doubled\+\+/.test(src),
     'and counts it if that ever happens, rather than playing it');
  const rtSrc = fs.readFileSync(path.join(STATIC, 'rio_realtime.js'), 'utf8');
  ok(/silenceOrphan/.test(rtSrc) && /response_id: responseId/.test(rtSrc),
     'the controller cancels a disowned dictation by id');
  ok(/orphans_silenced/.test(rtSrc), 'and counts what it silenced');
}

console.log('\n' + (failures ? 'FAILED ' + failures + ' of ' + checks
                             : 'PASSED ' + checks + ' checks'));
process.exit(failures ? 1 : 0);
}

main().catch(e => { console.error(e); process.exit(1); });
