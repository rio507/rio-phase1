/* rio_provider.js — the one place a vendor's event names are allowed to exist.
 *
 * WHY THIS FILE, AND WHAT IT IS NOT.
 *
 * createController() in rio_realtime.js is 4,700 lines and every decision RIO
 * makes about a live turn is in it: the arbiter, the supersede, the barge gate,
 * the echo gate, the resume budget, the mute ledger. All of it is keyed on
 * event names off the data channel. Adding a second vendor by branching inside
 * that switch is how a controller acquires two personalities that agree until
 * somebody edits one of them — which is the thing seven code paths of
 * ElevenLabs removal were spent on, and it is not being paid for twice.
 *
 * So the vendor stops at the door. This module holds:
 *
 *   a NAME MAP        vendor event name -> the name the controller speaks
 *   a CAPABILITY      what the transport and the model can and cannot do, as
 *   RECORD            facts the controller can ask about, so that no code
 *                     downstream ever asks "which vendor is this?"
 *
 * It is NOT a second controller. rio_live.js is the other road — a smaller
 * controller that re-implements the echo gate and drops the rest — and that was
 * right for gpt-live-1, where the events the other guards hung off genuinely do
 * not exist. It is the wrong road for a vendor whose events are mostly the SAME
 * NAMES, because it would fork every guard to gain nothing.
 *
 * THE CANONICAL VOCABULARY IS TODAY'S, DELIBERATELY.
 *
 * The names the controller speaks are the OpenAI Realtime names it already
 * speaks. That is not a claim that they are the right names or a permanent
 * home; it is the observation that renaming them costs a rewrite of
 * rio_realtime.js and ~4,000 lines of tools/realtime_selftest.js, and buys
 * nothing a map does not buy. `canonical` here means "what the controller
 * already calls this", and the openai_realtime profile is therefore EMPTY —
 * which is the property that makes the current stack byte-identical through
 * this file and is asserted rather than assumed
 * (tools/provider_adapter_selftest.js).
 *
 * WHAT `UNKNOWN` IS FOR, because it is the whole reason this is a record and
 * not a pile of booleans. Some of what we need to know about a provider is not
 * in its documentation and cannot be learned without a session. A boolean
 * cannot hold that: `false` would make the controller quietly take the
 * degraded path, and `true` would make it quietly take the wrong one. So those
 * fields are UNKNOWN, `probes()` lists them, and `require()` refuses to hand
 * out a profile whose unknowns matter for the path about to be taken. An
 * unmeasured fact fails loudly here or it fails silently in a car.
 */
'use strict';
(function (root) {

  /* An honest third value. Not null and not undefined: a missing key is a
     field nobody thought about, and this is a field somebody thought about and
     could not answer. */
  var UNKNOWN = '__unknown__';

  function isUnknown(v) { return v === UNKNOWN; }

  /* ---------------------------------------------------------------------
     THE CANONICAL EVENTS — every name createController's switch responds to.

     Kept as a list rather than derived, because its job is to be compared
     against the switch by a test: an event the controller handles and this
     list has forgotten is an event no provider can ever be mapped ONTO, and
     the symptom would be a guard that silently never fires. The selftest reads
     the switch out of the source and asserts the two agree in both directions.
     --------------------------------------------------------------------- */
  var CANONICAL = [
    'response.created',
    'response.done',
    'input_audio_buffer.committed',
    'input_audio_buffer.speech_started',
    'input_audio_buffer.speech_stopped',
    'output_audio_buffer.started',
    'output_audio_buffer.stopped',
    'output_audio_buffer.cleared',
    'response.output_text.delta',
    'response.output_text.done',
    'response.output_audio_transcript.delta',
    'response.output_audio_transcript.done',
    'response.function_call_arguments.done',
    'conversation.item.input_audio_transcription.completed',
    'conversation.item.input_audio_transcription.failed',
    'error',
  ];

  var CANONICAL_SET = {};
  for (var ci = 0; ci < CANONICAL.length; ci++) CANONICAL_SET[CANONICAL[ci]] = true;

  /* A mapping target meaning "the controller must never see this". Distinct
     from an absent mapping, which means "pass it through untouched": the
     default for an unrecognised event is to pass it, because the controller's
     switch already ignores what it does not know and a provider that adds an
     event should not need a code change here to be harmless. DROP is for the
     events where passing through would be WRONG rather than merely useless. */
  var DROP = '__drop__';

  /* ---------------------------------------------------------------------
     PROFILES
     --------------------------------------------------------------------- */

  /* `label` IS DISPLAY TEXT AND IS NOT A MODEL ID. It names a model so that a
     panel and a drive log can say what this session is, and it is the only
     place in the browser where a model's name appears at all. Nothing connects
     with it, nothing compares against it, and it must never become the thing
     that does: every id RIO actually uses is named in config.py and travels
     with the session (see mint_client_secret), which is what makes swapping one
     a config change rather than a search through the codebase. A label that has
     gone stale is a cosmetic bug; a label that something started connecting
     with would be a second source of truth for the most important string in the
     system. */
  var PROFILES = {

    /* ----------------------------------------------------------------
       The stack as it runs today: browser WebRTC to gpt-realtime-2.1.

       `events` is empty and must stay empty. This is not laziness, it is the
       definition of canonical — see the header. A mapping appearing here would
       mean the canonical vocabulary had drifted from the controller, and the
       selftest treats a non-empty map here as a failure.
       ---------------------------------------------------------------- */
    openai_realtime: {
      label: 'OpenAI Realtime (gpt-realtime-2.1), browser WebRTC',
      transport: 'webrtc',
      events: {},
      caps: {
        /* --- what the event stream provides ------------------------- */

        // output_audio_buffer.started/.stopped/.cleared. The mouth-hold's
        // evidence. See commit a05d233: the mouth was handed back seconds
        // before the sound stopped, and every mute in that window had no owner.
        audioBufferEvents: true,
        // ...and the client event that flushes it server-side.
        audioBufferClear: true,
        // conversation.item.input_audio_transcription.completed carries the
        // item_id the utterance was committed under. This is what the
        // self-supersede fix binds on; see rio_realtime.js ~1002 and ~2742,
        // and the drive of 2026-09-17 for what its absence costs.
        transcriptionItemId: true,
        // .failed is emitted: the branch that classifies a barge-in when the
        // transcriber could not make words out of the audio.
        transcriptionFailed: true,
        // rate_limits.updated. Nothing in the driving path reads it; it is how
        // tools/live_tool_turns.py measured TPM, which is where
        // realtime_selftest's growth alarm gets its 200,000 from.
        rateLimits: true,
        // Is the input transcription stream cumulative (each event restating
        // the whole transcript, with corrections) rather than incremental?
        // Only matters to a consumer of the deltas, and the controller is not
        // one — it waits for .completed. Recorded because that is luck, not
        // design, and the next person to want a live transcript needs to know.
        transcriptionCumulative: false,

        /* --- who owns the sound ------------------------------------- */

        // WebRTC plays the remote track. We never see a PCM frame, never
        // schedule playout, and cannot get the tail wrong.
        clientOwnsPlayout: false,
        // Where the belief "her audio has stopped" comes from.
        //   'server'         the API said so (output_audio_buffer.stopped)
        //   'client_playout' our own queue drained
        //   'none'           nothing knows; the tail cannot be held
        // This is the field that records the QUALITY of the evidence, not just
        // whether the tail is held — see holdTail() below.
        tailEvidence: 'server',
        // Does the transport give the browser's echo canceller a reference for
        // her own voice? config.py REALTIME_ECHO_* and every barge threshold
        // were tuned with this true ("Its own WebRTC audio is cancelled").
        echoCancellationReference: true,
        // Must a tool result wait for playout to drain before response.create?
        // No: the server owns playout, so it cannot be talked over by its own
        // scheduling.
        toolResultGate: false,

        /* --- how a fixed line gets said ----------------------------- */

        // 'response.create'       the words are the request; exact by
        //                         construction
        // 'force_message'         a synthesised item, no model involved
        // 'instructions.append'   a directive the model obeys (measured, not
        //                         guaranteed)
        verbatim: 'response.create',
        // Can a line be made to survive a driver talking over it? Under
        // response.create, no: a dictated line competes like any other.
        uninterruptibleLines: false,
        // A pronunciation layer applied before synthesis, transcript
        // preserved. None here — the "Pull over" failure was handled by
        // choosing a different voice.
        pronunciation: null,

        /* --- what survives a broken connection ---------------------- */

        // Server-side conversation replay on reconnect.
        resumption: null,
        // Is there a transport state between "connected" and "dead" that is
        // worth gracing? WebRTC has one and it is why
        // REALTIME_PEER_DISCONNECT_GRACE_MS exists: `disconnected` is a state
        // a connection passes through, not a teardown.
        peerGrace: true,

        /* --- model controls ----------------------------------------- */

        // Effort settings selectable on the voice model itself.
        reasoningEffort: null,
        // Does parallel tool calling need pinning explicitly? We have never
        // set it either way (there is no parallel_tool_calls anywhere in the
        // repo), so today we inherit a default. Recorded as a known gap rather
        // than a property of this vendor.
        parallelToolCalls: 'inherited',
      },
    },

    /* ----------------------------------------------------------------
       xAI grok-voice over WebSocket.

       NOT YET EXERCISED. Every value here is from xAI's published docs, and
       the two that the docs do not answer are UNKNOWN rather than guessed.
       The account this repo holds a key for is credit-blocked
       (team_blocked: true on /v1/api-key), so nothing below has been seen on a
       wire. `require('xai_voice')` will say so.

       TRANSPORT IS WEBSOCKET, AND THAT IS THE WHOLE COST OF THE MIGRATION.
       xAI documents output_audio_buffer.* as WebRTC/SIP only, ephemeral tokens
       (POST /v1/realtime/client_secrets, passed as xai-client-secret.<token>
       in sec-websocket-protocol) for WebSocket only, and its own "WebRTC
       Agent" example is a self-hosted Node gateway relaying PCM16 — browser
       WebRTC to your server, WebSocket from your server to xAI. There is no
       documented SDP exchange with api.x.ai. So the audio-buffer events are
       not merely renamed, they are unreachable, and the mouth-hold has to be
       reconstructed from a playout queue we own.
       ---------------------------------------------------------------- */
    xai_voice: {
      label: 'xAI grok-voice-think-fast-2.0, WebSocket',
      transport: 'websocket',
      events: {
        /* THE ONE RENAME, AND IT IS A DROP.
         *
         * xAI replaces conversation.item.input_audio_transcription.delta with
         * .updated, which is CUMULATIVE WITH CORRECTIONS rather than
         * incremental. The controller consumes neither: it waits for
         * .completed, which xAI still emits. So the honest mapping is not a
         * rename onto a canonical delta — there is no canonical delta — it is
         * DROP.
         *
         * Passing it through instead would be harmless today (the switch would
         * ignore it) and a trap tomorrow: the first person to add a live
         * transcript readout would wire it to a delta handler and append a
         * cumulative stream, producing the transcript three times over. The
         * drop is where that gets explained, and `cumulativeTranscript` on the
         * instance keeps the last one for whoever wants it.
         */
        'conversation.item.input_audio_transcription.updated': DROP,
      },
      caps: {
        // WebRTC/SIP only, and we have no reachable WebRTC. Set false per the
        // standing instruction: if it is not documented and does not answer a
        // probe, treat it as absent and design for WebSocket. `probes()` keeps
        // the question open — see webrtcEndpoint below.
        audioBufferEvents: false,
        audioBufferClear: false,

        /* THE UNKNOWN THAT DECIDES WHETHER A TURN CAN GO SILENT.
         *
         * xAI's event reference names .completed but does not publish its
         * payload. If it carries no item_id, the binding at
         * rio_realtime.js:2742 is false on every turn and we are back in
         * commit ee0a909: the transcription races the model, loses, arrives
         * looking like a brand new question, supersedes the turn it belongs
         * to, aborts the look() running for it and leaves the driver in
         * silence. It reproduced 3 of 3 on visual turns.
         *
         * Binding on ordinal instead is the heuristic rio_realtime.js ~1002
         * rejects by name, and it is not shipping as a silent fallback. */
        transcriptionItemId: UNKNOWN,

        // Documented as not emitted. The handler that files a barge-in as
        // transcription_failed goes dead; those turns fall through the
        // bargeConfirmMs timer and land as false_barge_in instead. Same
        // action — classified, not resumed — different label.
        transcriptionFailed: false,
        // Documented as not emitted. Two consumers, both probes
        // (tools/live_tool_turns.py). The casualty is TPM: realtime_selftest
        // documents 200,000 as measured off this event, and both the growth
        // alarm and AFFORDABLE_PER_RESPONSE derive from it. It stops being
        // measurable and the constant keeps asserting OpenAI's budget.
        rateLimits: false,
        transcriptionCumulative: true,

        // We read response.output_audio.delta and feed an AudioContext
        // ourselves. Everything in the two rows below follows from this one.
        clientOwnsPlayout: true,
        // Not the API's word for it any more — our own queue draining. The
        // tail is still held, on weaker evidence, and this is the field that
        // says so out loud. The a05d233 mute ledger now measures our
        // arithmetic as well as the road.
        tailEvidence: 'client_playout',
        // No WebRTC means no reference signal for her own voice. Every
        // REALTIME_ECHO_* threshold and the whole barge gate were tuned with
        // one. This was not on the migration's blocker list and it lands on
        // the echo gate, which rio_live.js calls the one guard that must never
        // be missing (8/8 self-answers from full level to -20 dB).
        echoCancellationReference: false,
        /* THEIR DOCS DESCRIBE OUR CODE. "If your client immediately sends
           conversation.item.create (with the function result) followed by
           response.create, the server starts generating the next response
           right away — even if the client is still playing audio from the
           previous turn. This causes overlapping audio." That is
           rio_realtime.js:3265->3276 verbatim. It has never hurt us because
           the server owned playout; the moment we own it, it does. Same
           queue as tailEvidence: one piece of work, not two. */
        toolResultGate: true,

        // conversation.item.create with item.type 'force_message':
        // synthesised verbatim, no model in the path.
        verbatim: 'force_message',
        // interruptible: false. Note what it MEANS before using it: caller
        // audio is DROPPED during playback, so a driver speaking over the line
        // is not heard at all rather than merely ignored. Commit 0bdec46 —
        // eleven refused barge-ins were eleven questions being answered — is
        // the one that says which way we care. Red tier and the imminent turn
        // call only.
        uninterruptibleLines: true,
        // `replace`: phrase -> spoken form, applied pre-TTS, case-insensitive,
        // whole-word. Transcript PRESERVED, which is the part that matters:
        // the echo gate's textual test compares her transcript against what
        // comes back through the cabin, and a mechanism that rewrote
        // transcripts would poison it.
        pronunciation: 'replace',

        // resumption.enabled + the server-assigned conversation_id on
        // reconnect. Replaces nothing we have — we lose the conversation on a
        // real drop today — but it overlaps our own resume, which carries what
        // was HEARD rather than what was generated. Two resume mechanisms that
        // disagree about what she said is worse than one.
        resumption: 'conversation_id',
        // A WebSocket is open or it is gone. There is no state between them to
        // grace, so REALTIME_PEER_DISCONNECT_GRACE_MS has nothing to time.
        peerGrace: false,

        // 'high' | 'none', defaulting to HIGH — on the path with our tightest
        // deadlines (dictation start measured 418-1277 ms against a 900 ms
        // budget, itself cut against gpt-realtime's 390-585 ms). It does not
        // replace deep_dive: that is a tool doing web research behind a
        // holding line, not a thinking setting on the voice.
        reasoningEffort: ['high', 'none'],
        parallelToolCalls: 'must_pin',
      },
      /* What is not known, why it matters, and what would settle it. Read by
         probes() and by require(); the shape is deliberately enough to write
         the probe from. */
      unknowns: {
        transcriptionItemId: {
          why: 'the self-supersede binding; without it a tool turn can go '
             + 'silent (commit ee0a909, drive of 2026-09-17)',
          settle: 'open a session with audio.input.transcription.model = '
                + '"grok-transcribe", speak one utterance, and assert the '
                + 'item_id on input_audio_buffer.committed, '
                + '...transcription.updated and ...transcription.completed '
                + 'are all equal. About 60 seconds of audio.',
          blocks: ['supersede'],
        },
        webrtcEndpoint: {
          why: 'if xAI does terminate WebRTC, audioBufferEvents and '
             + 'echoCancellationReference both flip true and stage 3 is a '
             + 'base-URL change rather than a playout rewrite',
          settle: 'check the REST reference for an SDP endpoint, then probe '
                + 'it directly. Absent and unanswering means absent.',
          blocks: [],
        },
      },
    },
  };

  /* ---------------------------------------------------------------------
     A PROVIDER INSTANCE
     --------------------------------------------------------------------- */

  /* Which provider a session is on.
   *
   * `name` comes from the server with the session (config.VOICE_BACKEND), for
   * the same reason the dictation and barge policies do: the browser holding a
   * second opinion about which vendor it is talking to is a drive that
   * disagrees with its own logs. There is no default — a mint that forgot to
   * say is a bug worth hearing about, not a reason to guess openai_realtime
   * and half-work.
   */
  function create(name, opts) {
    opts = opts || {};
    var profile = PROFILES[name];
    if (!profile) {
      throw new Error('rio_provider: unknown provider ' + JSON.stringify(name)
                      + ' (have: ' + Object.keys(PROFILES).join(', ') + ')');
    }

    var events = profile.events || {};
    var caps = profile.caps || {};
    var unknowns = profile.unknowns || {};

    /* Counted, not logged. The ingress runs on every event of every drive —
       response.output_audio.delta arrives at audio rate — so this is the one
       place in the system where a console.log per call would be a real cost.
       The panel reads these; nothing prints them. */
    var stats = { seen: 0, renamed: 0, dropped: 0, unmapped: {} };

    /* The last cumulative input transcript, for a provider that sends one.
       Kept because it is the only evidence available if transcriptionItemId
       turns out false: .updated is cumulative, so the last one before
       .completed is the same words. It gives text, not identity — which is
       exactly why it is not a fallback for the binding, and is recorded
       instead of being quietly wired up as one. */
    var cumulativeTranscript = null;

    function capability(key) {
      return Object.prototype.hasOwnProperty.call(caps, key) ? caps[key]
                                                             : undefined;
    }

    /* ONE EVENT, ON ITS WAY IN.
     *
     * Returns the event the controller should handle, or null for "nothing
     * happened". Three paths, and the first one is the hot one:
     *
     *   no mapping     the SAME OBJECT back, untouched. No allocation, no
     *                  copy, no property write. This is every event on the
     *                  current stack and every audio delta on any stack, and
     *                  it is why the openai_realtime map is empty rather than
     *                  a list of identities.
     *   DROP           null.
     *   a rename       a shallow copy with the new type. Copied rather than
     *                  mutated because handle() is also called directly by
     *                  ~4,000 lines of tests holding literals they then
     *                  inspect, and a normaliser that edited its input would
     *                  make those literals lie.
     */
    function normalise(ev) {
      if (!ev || typeof ev.type !== 'string') return ev || null;
      stats.seen++;
      var mapped = events[ev.type];

      if (mapped === undefined) {
        /* Unrecognised is PASSED, not dropped: the controller's switch already
           ignores what it does not know, and a provider adding an event should
           not need a change here to be harmless. Counted once per name so a
           drive can show what arrived that nobody has looked at. */
        if (!CANONICAL_SET[ev.type]) {
          stats.unmapped[ev.type] = (stats.unmapped[ev.type] || 0) + 1;
        }
        return ev;
      }

      if (mapped === DROP) {
        stats.dropped++;
        if (ev.type === 'conversation.item.input_audio_transcription.updated'
            && typeof ev.transcript === 'string') {
          cumulativeTranscript = ev.transcript;
        }
        return null;
      }

      stats.renamed++;
      var out = {};
      for (var k in ev) {
        if (Object.prototype.hasOwnProperty.call(ev, k)) out[k] = ev[k];
      }
      out.type = mapped;
      return out;
    }

    /* MAY THE MOUTH WAIT FOR THE SOUND TO STOP?
     *
     * The question the controller used to answer with a hardcoded `true` and a
     * comment reading "WebRTC sends output_audio_buffer.stopped". That comment
     * was correct and was also the only place the dependency was written down.
     *
     * True whenever SOMETHING knows when the audio ended — the API saying so,
     * or our own playout queue draining. False only when nothing does, in
     * which case holding the tail would mean holding it on a timer, and a
     * response that never ends is worse than one that ends early.
     */
    function holdTail() {
      var ev = capability('tailEvidence');
      return ev === 'server' || ev === 'client_playout';
    }

    /* Everything unresolved that matters, as a list. Empty is the point. */
    function probes() {
      var out = [];
      for (var key in unknowns) {
        if (!Object.prototype.hasOwnProperty.call(unknowns, key)) continue;
        if (!isUnknown(caps[key]) && caps[key] !== undefined) continue;
        var u = unknowns[key];
        out.push({ capability: key, why: u.why, settle: u.settle,
                   blocks: (u.blocks || []).slice() });
      }
      return out;
    }

    /* Would taking `path` rely on something nobody has measured?
     *
     * `blocks` in the unknowns table names the paths each unknown gates, so
     * this is a lookup rather than a judgement. A caller that asks is a caller
     * that gets told; the alternative — an UNKNOWN read as falsy by an `if` —
     * is the degraded path taken silently, which is the failure this whole
     * record exists to prevent. */
    function blockedPaths() {
      var seen = {}, out = [];
      probes().forEach(function (p) {
        p.blocks.forEach(function (b) {
          if (!seen[b]) { seen[b] = true; out.push(b); }
        });
      });
      return out;
    }

    return {
      name: name,
      label: profile.label,
      transport: profile.transport,
      normalise: normalise,
      capability: capability,
      /* A copy. The record is read in a lot of places and a caller that could
         edit it would be a second source of truth with no file of its own. */
      capabilities: function () {
        var out = {};
        for (var k in caps) {
          if (Object.prototype.hasOwnProperty.call(caps, k)) out[k] = caps[k];
        }
        return out;
      },
      holdTail: holdTail,
      probes: probes,
      blockedPaths: blockedPaths,
      unknown: function (key) { return isUnknown(caps[key]); },
      cumulativeTranscript: function () { return cumulativeTranscript; },
      stats: function () {
        return { seen: stats.seen, renamed: stats.renamed,
                 dropped: stats.dropped,
                 unmapped: JSON.parse(JSON.stringify(stats.unmapped)) };
      },
    };
  }

  /* WHICH PROVIDER A VOICE BACKEND IMPLIES, WHICH IS NOT THE SAME QUESTION.
   *
   * config.VOICE_BACKEND names WHO SPEAKS. This module is about WHOSE EVENT
   * STREAM this is, and the two come apart immediately:
   *
   *   openai_realtime   OpenAI realtime session, audio out       -> openai_realtime
   *   elevenlabs        OpenAI realtime session in TEXT mode, an
   *                     ElevenLabs sink doing the speaking       -> openai_realtime
   *   gpt_live          not this file at all; rio_live.js owns
   *                     the gpt-live-1 event stream             -> (unsupported here)
   *
   * So a session that says `elevenlabs` is still an OpenAI event stream, and a
   * resolver that passed the backend name straight through would have gone
   * looking for an `elevenlabs` profile and thrown on a configuration that
   * works today. Mapping rather than assuming, and the table is here rather
   * than in connect() because this file is the one place allowed to know what
   * a vendor is called.
   *
   * THIS SHOULD EVENTUALLY COME FROM THE SERVER. Every other policy the
   * browser applies travels with the session precisely so there is no second
   * copy to drift (see the dictation and barge policies). This is a second
   * copy. It is a small one and it is in one function, but the right home for
   * it is mint_client_secret alongside voice_backend, and the day a fourth
   * backend arrives is the day that stops being a nicety.
   */
  var BY_VOICE_BACKEND = {
    openai_realtime: 'openai_realtime',
    elevenlabs: 'openai_realtime',
    xai_voice: 'xai_voice',
  };

  function forVoiceBackend(backend) {
    var name = BY_VOICE_BACKEND[backend];
    if (!name) {
      throw new Error('rio_provider: no event provider for voice backend '
                      + JSON.stringify(backend));
    }
    return create(name);
  }

  var api = {
    create: create,
    forVoiceBackend: forVoiceBackend,
    voiceBackends: function () { return Object.keys(BY_VOICE_BACKEND); },
    UNKNOWN: UNKNOWN,
    DROP: DROP,
    CANONICAL: CANONICAL.slice(),
    /* The profile table itself, for tests and for a panel that wants to show
       what this drive is running on. */
    profiles: function () { return Object.keys(PROFILES); },
    profile: function (n) { return PROFILES[n]; },
  };

  root.RIO = root.RIO || {};
  root.RIO.provider = api;

  if (typeof module !== 'undefined' && module.exports) {
    module.exports = api;
  }
})(typeof window !== 'undefined' ? window : globalThis);
