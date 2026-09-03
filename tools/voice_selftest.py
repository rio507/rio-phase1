"""voice_selftest.py — RIO's voice, checked from both ends.

    python -m tools.voice_selftest              # everything with no network
    python -m tools.voice_selftest --live       # ...and the real socket too
    python -m tools.voice_selftest --live --server http://127.0.0.1:8888

WHAT IS WORTH CHECKING HERE, AND WHAT IS NOT
--------------------------------------------
Not "does ElevenLabs work". It does, and if it stops the drive falls back and
says so. What is worth checking is the set of decisions this repository added
around it, because every one of them is invisible from the passenger seat and
wrong in a way that sounds like something else:

  a phrase boundary in the wrong place        sounds like a bad connection
  a tag that survived onto a warning          sounds like a broken synthesiser
  a tag that was dropped when it should not   sounds like a flat reading
  a cancel that does not stop the audio       sounds like two voices at once
  a resume from the MODEL's text, not the     sounds like RIO skipping the part
    listener's                                  the driver never heard
  a fallback that does not fire               sounds like silence
  a fallback that fires and says nothing      sounds like silence, twice

The live half exercises the real socket against the real service, including
both fallback tiers — triggered by making the real conditions true (a budget
that cannot be met, a socket that has gone away, a key that is refused) rather
than by calling the fallback function directly. A test that calls the fallback
proves the fallback runs; only a test that causes the failure proves anything
reaches it.
"""
import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv                              # noqa: E402

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

import config                                               # noqa: E402
import realtime                                             # noqa: E402
import voice                                                # noqa: E402
import voice_dialogue as vd                                 # noqa: E402
import voice_tags                                           # noqa: E402

REPO = Path(__file__).resolve().parent.parent
PASS, FAIL = [], []

# Enough to fill this workspace's dialogue pool, with room over the measured 21
# so the test discovers the limit rather than encoding it. If the plan changes
# and the pool grows past this, the fill stops short and the check that says so
# fails loudly instead of the section passing on a pool that is not full.
POOL_PROBE_CAP = 30


def ok(cond, what):
    (PASS if cond else FAIL).append(what)
    print(f"  {'ok  ' if cond else 'FAIL'}  {what}")
    return bool(cond)


def section(name):
    print(f"\n=== {name} ===")


# ---------------------------------------------------------------------------
# A. The session RIO is given
# ---------------------------------------------------------------------------
def run_session():
    section("A. text mode — the session produces words, and still hears audio")

    cfg = realtime.session_config()
    text = config.VOICE_BACKEND == "elevenlabs"
    ok(cfg["output_modalities"] == (["text"] if text else ["audio"]),
       f"the session is asked for {cfg['output_modalities']} under "
       f"VOICE_BACKEND={config.VOICE_BACKEND}")

    # The half that must NOT have changed. Text mode is about her mouth; every
    # one of these is about her ears, and a regression in any of them is a
    # session that stops noticing the driver.
    audio_in = cfg["audio"]["input"]
    ok(audio_in["transcription"]["model"] == config.OPENAI_STT_MODEL,
       "audio input is untouched: the same Whisper transcribes the driver")
    td = audio_in["turn_detection"]
    ok(td["type"] == "server_vad"
       and abs(td["threshold"] - float(config.REALTIME_VAD_THRESHOLD)) < 1e-9
       and td["silence_duration_ms"] == int(config.REALTIME_VAD_SILENCE_MS),
       "...and the cabin-tuned detector, at the same numbers")
    ok(td["interrupt_response"] is False,
       "...and interruption is still the browser's decision, not the server's")
    ok(cfg["audio"]["output"]["voice"] == config.OPENAI_REALTIME_VOICE,
       f"cedar is still named ({config.OPENAI_REALTIME_VOICE}), so the tier-2 "
       "fallback is a modality switch and not a first-time configuration")
    ok(int(cfg["max_output_tokens"]) == int(config.REALTIME_MAX_RESPONSE_TOKENS),
       "the ceiling on a spoken answer is unchanged")

    # What travels to the browser with the session.
    payload_keys = ("voice_backend", "cedar_voice", "voice_sample_rate",
                    "output_modalities", "speech_enabled")
    src = Path(REPO / "realtime.py").read_text()
    ok(all(f'"{k}"' in src for k in payload_keys),
       "the mint payload carries the backend, the cedar voice and the sample "
       "rate, so the page holds no copy of any of them")


def run_dictation_policy():
    section("B. one voice — deterministic lines, and who says them")

    # Under ElevenLabs the deterministic lines are NOT dictated. That is the
    # point rather than a gap: /nav/voice and friends already synthesise on the
    # same voice id, on the model that is fastest to first byte.
    if config.VOICE_BACKEND == "elevenlabs":
        ok(realtime.mint_client_secret.__doc__ is not None, "mint is documented")
        src = Path(REPO / "realtime.py").read_text()
        ok('config.VOICE_BACKEND != "elevenlabs"' in src,
           "dictation is switched off under the ElevenLabs backend, so a "
           "warning goes to the synthesiser directly instead of waiting out a "
           "dictation budget it is never going to meet")

    # WHICH SOCKET, decided by the model and by nothing else.
    conv = config.ELEVENLABS_CONVERSATION_MODEL
    wire = vd.dialect_for("v", conv, "pcm_24000")
    ok(config.uses_dialogue_socket(conv) == conv.startswith("eleven_v3"),
       f"the transport follows the model: {conv} -> "
       f"{type(wire).__name__}")
    ok(isinstance(wire, (vd._DialogueDialect, vd._MultiContextDialect)),
       "and it is one of the two dialects, not a third path")
    v3 = vd.dialect_for("v", "eleven_v3_conversational", "pcm_24000")
    ok(isinstance(v3, vd._DialogueDialect) and v3.recycle_on_cancel,
       "v3 still routes to the dialogue socket, which recycles on a cancel "
       "because its audio carries nothing that says which turn it is for")
    v2 = vd.dialect_for("v", "eleven_multilingual_v2", "pcm_24000")
    ok(isinstance(v2, vd._MultiContextDialect) and not v2.recycle_on_cancel,
       "and v2 to the multi-context socket, which does not — a context can be "
       "closed by name, so a barge-in costs one message rather than a "
       "connection")

    ok(config.ELEVENLABS_DETERMINISTIC_MODEL == "eleven_flash_v2_5",
       f"deterministic speech uses {config.ELEVENLABS_DETERMINISTIC_MODEL}, "
       "the fastest thing to first byte")
    ok(config.ELEVENLABS_VOICE_ID and
       voice.voice_id() == config.ELEVENLABS_VOICE_ID == vd.voice_id(),
       f"and the SAME voice id as the conversation ({config.ELEVENLABS_VOICE_ID})"
       " — which is the whole of 'one voice everywhere'")

    # The controller must refuse dictation in text mode immediately rather than
    # letting a warning wait out a budget.
    js = Path(REPO / "static/rio_realtime.js").read_text()
    ok("if (sink) return Promise.reject(new Error('text_mode'));" in js,
       "the page refuses a dictation in text mode at once, so rio_speak falls "
       "through to the synthesiser without costing the warning a delay")


# ---------------------------------------------------------------------------
# C. The chunker
# ---------------------------------------------------------------------------
def run_chunker():
    section("C. phrasing — when a half-written answer is worth speaking")

    c = vd.PhraseChunker(min_tokens=5, max_wait_ms=250)
    got = []
    t = 0.0
    for piece in ["Traffic's", " thinning out", " ahead,", " and the sun's",
                  " about to do", " something nice.", " Want me to",
                  " hold this lane?"]:
        got += c.push(piece, now=t)
        t += 0.02
        got += c.due(now=t)
    got += c.drain()
    ok(len(got) >= 2, f"a two-sentence answer goes out in pieces ({len(got)})")
    ok("".join(got).replace("  ", " ").strip().startswith("Traffic's thinning"),
       "and the pieces reassemble into exactly what the model wrote")
    ok(all(p.strip() for p in got), "with no empty phrase sent to synthesise")

    # The max-wait rule, and where it is allowed to cut.
    c = vd.PhraseChunker(min_tokens=5, max_wait_ms=250)
    c.push("Because the road you're on bends left in about", now=0.0)
    ok(c.due(now=0.1) == [], "nothing goes out before the max-wait expires")
    late = c.due(now=0.4)
    ok(len(late) == 1, "after it, something does — a long clause with no "
       "boundary in it is exactly the shape that felt slow")
    ok(late and late[0].endswith(" "),
       f"and it cuts at a word boundary, never mid-word ({late[0][-12:]!r})")

    # A single unbroken word is not a phrase.
    c = vd.PhraseChunker(min_tokens=5, max_wait_ms=50)
    c.push("Interchange", now=0.0)
    ok(c.due(now=1.0) == [],
       "one long word still arriving is held rather than synthesised in halves")

    # A tag is never split across two messages.
    c = vd.PhraseChunker(min_tokens=1, max_wait_ms=50)
    c.push("Well [lau", now=0.0)
    ok(c.due(now=1.0) == [], "a half-written audio tag is not a phrase")
    out = c.push("ghs] that's one way to put it.", now=1.0)
    ok(any("[laughs]" in p for p in out + c.drain()),
       "and arrives whole once the model finishes writing it")


# ---------------------------------------------------------------------------
# D. The tag validator
# ---------------------------------------------------------------------------
def run_tags():
    section("D. audio tags — allowed in one place, and nowhere else")

    keep, dropped = voice_tags.sanitize("[laughs] Yeah, that's the one.",
                                        voice_tags.CONVERSATION)
    ok(keep.startswith("[laughs]") and not dropped,
       "a tag from the list survives in conversation")

    keep, dropped = voice_tags.sanitize("[sighs] Take the next left.",
                                        voice_tags.DETERMINISTIC)
    ok("[" not in keep and dropped[0]["reason"] == voice_tags.WRONG_CHANNEL,
       "the same tag on a navigation line is dropped, with the reason recorded")
    ok(keep == "Take the next left.",
       f"and the line is left clean, not left with the hole ({keep!r})")

    keep, dropped = voice_tags.sanitize("[shouts] Yeah.", voice_tags.CONVERSATION)
    ok("[" not in keep and dropped[0]["reason"] == voice_tags.NOT_ALLOWED,
       "a tag that is not on the list is DROPPED, not passed through to find "
       "out whether the synthesiser speaks it")

    keep, _ = voice_tags.sanitize("Lin[laughs]coln Boulevard.",
                                  voice_tags.CONVERSATION)
    ok(keep == "Lincoln Boulevard.",
       "a tag welded into the middle of a word is dropped as misplaced")

    keep, dropped = voice_tags.sanitize("[laughs] Yeah, [sighs] and then some.",
                                        voice_tags.CONVERSATION)
    ok(keep.count("[") == config.AUDIO_TAGS_MAX_PER_UTTERANCE
       and dropped[0]["reason"] == voice_tags.TOO_MANY,
       f"at most {config.AUDIO_TAGS_MAX_PER_UTTERANCE} per utterance — the "
       "budget is what keeps this from becoming a performance")

    keep, dropped = voice_tags.sanitize("[laughs]", voice_tags.CONVERSATION)
    ok(not keep.strip() and dropped[0]["reason"] == voice_tags.NO_WORDS,
       "a tag with no words to modify is dropped: a laugh with nothing to "
       "laugh about is the definition of performative")

    # The deterministic synthesiser strips unconditionally, whatever it is
    # handed. This is the belt to the validator's braces and the one that
    # actually protects a driver at a junction.
    src = Path(REPO / "voice.py").read_text()
    ok("voice_tags.strip(text)" in src,
       "voice.synthesize_stream strips tags from every line it is given, "
       "whatever the caller believed about it")

    # ...and the model is told the same list the gate enforces.
    inst = realtime.instructions()
    if config.VOICE_BACKEND == "elevenlabs" and config.AUDIO_TAGS_ENABLED:
        listed = all(f"[{t}]" in inst for t in voice_tags.allowed_tags())
        ok(listed, "the live instructions name exactly the allowed tags, "
                   "generated from the same config the validator reads")
        ok("Most replies should carry no tag at all" in inst,
           "and say plainly that most replies carry none — the bible's "
           "silence discipline, applied to a mechanism that invites the "
           "opposite")
    else:
        ok("[laughs]" not in inst,
           "with tags unreachable, the model is not told about them at all")


# ---------------------------------------------------------------------------
# E. The firewall
# ---------------------------------------------------------------------------
def _imports_of(path):
    import ast

    names = set()
    for node in ast.walk(ast.parse(Path(path).read_text())):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def run_firewall():
    section("E. firewall — the synthesiser cannot decide to speak")

    # The new modules are a MOUTH. Nothing here may reach the code that decides
    # a driver gets interrupted, in either direction.
    unreachable = {"navigation", "headway", "vehicle_health_policy",
                   "tires", "telemetry", "realtime", "llm_interface"}
    for rel in ("voice_dialogue.py", "voice_tags.py", "voice.py"):
        hits = sorted(_imports_of(REPO / rel) & unreachable)
        ok(not hits, f"{rel} cannot reach anything that decides to speak"
                     + (f" (found {hits})" if hits else ""))

    ok(_imports_of(REPO / "voice_tags.py") <= {"re", "config"},
       "voice_tags.py is a regex and a config read — it is safe to import from "
       "the live session, which is why the model's instructions and the gate "
       "can be generated from one list")

    live = _imports_of(REPO / "realtime.py")
    ok("voice" not in live and "voice_dialogue" not in live,
       "realtime.py still cannot reach a synthesiser: it produces words and "
       "has no way to make a sound")
    ok("voice_tags" in live,
       "...but it does import the tag list, because the paragraph telling the "
       "model what it may do is generated from the same list that enforces it")

    # The deterministic endpoints are still id-addressed, not text-addressed.
    app = Path(REPO / "app.py").read_text()
    for endpoint in ("/nav/voice", "/headway_voice", "/vehicle/health/voice"):
        ok(endpoint in app, f"{endpoint} still exists")
    ok('if live_policy.LINE_AUDIO.get(line) != "tts"' in app,
       "/headway_voice still refuses anything that is not in the policy's own "
       "table — a deterministic channel, not a text-to-speech endpoint")


# ---------------------------------------------------------------------------
# F. The clips
# ---------------------------------------------------------------------------
def run_clips():
    section("F. the fast path — local files, in the right voice")

    from tools import render_alerts as ra

    expected = {"back_off", "too_close", "watch_distance",
                "tire_critical", "tire_sensor_lost"}
    present = {p.stem for p in (REPO / "static/audio").glob("*.mp3")}
    ok(expected <= present,
       f"all {len(expected)} pre-rendered clips are on disk "
       f"({len(expected & present)}/{len(expected)})")

    want = ra.voice_signature()
    doc = ra.manifest().get("clips", {})
    stale = sorted(c for c in expected
                   if doc.get(c, {}).get("voice") != want["voice"])
    ok(not stale,
       f"and every one records the configured voice ({want['voice']})"
       + (f"; stale: {stale}" if stale else ""))
    ok(all(doc.get(c, {}).get("model") == want["model"] for c in expected),
       f"rendered on {want['model']} — the quality model, because nothing in a "
       "car waits on a file that already exists")

    # Runtime playback is untouched: still a local file, still no network.
    js = Path(REPO / "static/rio_speak.js").read_text()
    ok("playClip" in js and "element.src = url" in js,
       "the runtime path still plays them as local files with nothing in "
       "front of them")


# ---------------------------------------------------------------------------
# G. The live socket, and both ways of losing it
# ---------------------------------------------------------------------------
async def _drive(session, text, rid="r1", wait=20.0, events=None):
    """Say one line through a session and wait for the LISTENER to be done.

    Waiting for `utterance_done` rather than for a fixed interval, because that
    event is the thing under test everywhere it appears: it is the moment the
    arbiter gets the mouth back, and a sleep long enough to make it show up is
    a test that cannot tell "finished" from "still generating".
    """
    heard = {"pcm": 0, "text": "", "chunks": 0, "first": None, "done": False}
    t0 = time.time()

    async def on_audio(_rid, pcm, said):
        if heard["first"] is None:
            heard["first"] = (time.time() - t0) * 1000.0
        heard["pcm"] += len(pcm)
        heard["chunks"] += 1
        heard["text"] += said or ""

    mark = len(events) if events is not None else 0
    session.on_audio = on_audio
    await session.begin(rid)
    for word in text.split(" "):
        await session.delta(rid, word + " ")
        await asyncio.sleep(0.02)
    await session.end(rid)
    deadline = time.time() + wait
    while time.time() < deadline:
        if events is not None and any(
                k == "utterance_done" and (d or {}).get("rid") == rid
                for k, d in events[mark:]):
            heard["done"] = True
            break
        await asyncio.sleep(0.05)
    return heard


def run_voice_support():
    """Does this voice actually support the models we point at it?

    THE CHECK THAT WOULD HAVE CAUGHT THE LAST ONE. RIO's voice is a
    professional clone, and a clone is trained against a set of base models —
    ask for one outside that set and the service does not refuse, it
    synthesises an approximation. The result is a voice that is recognisably
    not the one that was cloned, arriving with no error anywhere, which is a
    thing you can only find by listening.

    The set is published per voice as `high_quality_base_model_ids`, so it is a
    question with an answer rather than a matter of taste. This voice lists the
    v2 family and no v3 model at all, which is precisely why the conversation
    path moved off v3.
    """
    section("B2. the voice supports the models we point at it")

    import urllib.request

    try:
        v = json.load(urllib.request.urlopen(urllib.request.Request(
            f"https://api.elevenlabs.io/v1/voices/{vd.voice_id()}",
            headers={"xi-api-key": vd.api_key()}), timeout=25))
    except Exception as e:
        ok(False, f"could not read the voice ({type(e).__name__})")
        return
    supported = set(v.get("high_quality_base_model_ids") or [])
    ok(bool(supported), f"{v.get('name')!r} lists {len(supported)} base models")
    for label, model in (("conversation", config.ELEVENLABS_CONVERSATION_MODEL),
                         ("deterministic", config.ELEVENLABS_DETERMINISTIC_MODEL)):
        ok(model in supported,
           f"the {label} model ({model}) is one this voice was cloned against"
           + ("" if model in supported else
              f" — it is NOT, and the service will approximate rather than "
              f"refuse. Supported: {sorted(supported)}"))


async def run_live():
    wire = vd.dialect_for(vd.voice_id(), config.ELEVENLABS_CONVERSATION_MODEL,
                          config.ELEVENLABS_OUTPUT_FORMAT)
    section(f"G. the real socket ({type(wire).__name__}) — opening, staying "
            "open, coming back")

    events = []

    async def on_event(kind, detail):
        events.append((kind, detail))

    s = vd.DialogueSession(on_audio=lambda *a: asyncio.sleep(0),
                           on_event=on_event)
    opened = await s.start()
    ok(opened, f"the socket for {config.ELEVENLABS_CONVERSATION_MODEL} opens")
    if not opened:
        await s.close()
        return

    heard = await _drive(s, "Traffic is thinning out ahead, and the sun is "
                            "about to do something nice over the ridge.",
                         events=events)
    ok(heard["done"], "the utterance runs to completion and says so")
    ok(heard["pcm"] > 0, f"audio comes back ({heard['pcm']} bytes)")
    ok(heard["chunks"] > 1,
       f"in pieces rather than in one lump ({heard['chunks']} chunks) — which "
       "is what lets playback start before the sentence is finished")
    ok(heard["first"] is not None and heard["first"] < 2000,
       f"first audio {heard['first']:.0f} ms after the first word was written")
    ok("Traffic" in heard["text"],
       "and each piece carries the words it is for, so the page can say how "
       "far she actually got")
    # The mouth comes back, and comes back at the END. A multi-clause answer
    # produces one flush per clause and one final marker per flush; declaring
    # the utterance over on the first marker would hand the arbiter its item
    # back with two clauses still to play.
    done = [d for k, d in events if k == "utterance_done"]
    ok(len(done) == 1 and done[0].get("rid") == "r1",
       f"the utterance reports done exactly once ({len(done)}), after every "
       "flush has come back rather than after the first one")

    # KEEP-ALIVE. The server closes a socket after 20 s of silence, and a car
    # is silent for most of a drive.
    was = s.stats["keepalives"]
    old_ms = config.ELEVENLABS_KEEPALIVE_MS
    config.ELEVENLABS_KEEPALIVE_MS = 200
    try:
        await asyncio.sleep(1.0)
    finally:
        config.ELEVENLABS_KEEPALIVE_MS = old_ms
    ok(s.stats["keepalives"] > was,
       f"keep-alives go out during a silence ({s.stats['keepalives'] - was} in "
       "a second at a 200 ms interval)")
    after = await _drive(s, "Still here, still listening.", rid="r2",
                         events=events)
    ok(after["pcm"] > 0, "and the socket is still usable afterwards")

    # RECONNECT. Kill the wire under it and check the next line still speaks.
    before = s.stats["reconnects"]
    await s._ws.close()
    await asyncio.sleep(1.5)
    ok(s.stats["reconnects"] > before,
       "a dropped socket is replaced underneath whatever is happening")
    again = await _drive(s, "Back again, as if nothing had happened.",
                         rid="r3", events=events)
    ok(again["pcm"] > 0, "and the next thing she says comes out normally")

    # A CANCEL, and what it costs. This is the behaviour that differs between
    # the two transports and the reason the multi-context one is preferred: a
    # context can be closed by name, so a barge-in is one message rather than a
    # connection thrown away and rebuilt.
    reconnects_before = s.stats["reconnects"]
    await s.begin("r4")
    await s.delta("r4", "This one is going to be interrupted part way ")
    await asyncio.sleep(0.6)
    await s.cancel("r4")
    await asyncio.sleep(1.5)
    if s.wire.recycle_on_cancel:
        ok(s.stats["reconnects"] > reconnects_before,
           "on the dialogue socket a cancel recycles the connection, because "
           "its audio carries nothing that says which turn it belongs to")
    else:
        ok(s.stats["reconnects"] == reconnects_before,
           "a cancel closes the context and leaves the connection alone — no "
           "reconnect on the barge-in path at all")
        ok(s._ws is not None, "...and the socket is still open")
    dropped = {"n": 0}

    async def count(rid, pcm, text):
        dropped["n"] += len(pcm)

    s.on_audio = count
    await asyncio.sleep(1.5)
    ok(dropped["n"] == 0,
       "and nothing more from the cancelled utterance reaches the page, so a "
       "warning is not spoken over by the sentence it interrupted")

    after_cancel = await _drive(s, "And she carries straight on.", rid="r5",
                                events=events)
    ok(after_cancel["pcm"] > 0, "the next answer speaks normally afterwards")

    await s.close()


async def run_other_transport():
    """The transport this build is NOT configured for, exercised anyway.

    Both dialects are shipped and either is one config value away, so the one
    that is switched off is the one that rots. This runs a real session over
    whichever socket the conversation model does not use, so a refactor that
    breaks it fails here rather than on the day somebody switches back.
    """
    other = ("eleven_v3_conversational"
             if not config.uses_dialogue_socket()
             else "eleven_multilingual_v2")
    section(f"G2. the other transport, kept alive ({other})")

    events = []

    async def on_event(kind, detail):
        events.append((kind, detail))

    s = vd.DialogueSession(on_audio=lambda *a: asyncio.sleep(0),
                           on_event=on_event, model=other)
    opened = await s.start()
    ok(opened, f"a session on {other} opens")
    if not opened:
        await s.close()
        return
    ok(config.uses_dialogue_socket(other) == s.wire.recycle_on_cancel,
       f"over {type(s.wire).__name__}, which is the dialect its model needs")
    heard = await _drive(s, "The other socket still works.", rid="o1",
                         events=events)
    ok(heard["pcm"] > 0, f"and it speaks ({heard['pcm']} bytes)")
    ok(heard["done"], "and reports the utterance finished, so a mouth held on "
                      "this transport still comes back")
    await s.close()


async def run_fallbacks():
    section("H. both fallback tiers, triggered by the real conditions")

    # TIER 1a: the line was slow. Triggered by making the budget unreachable
    # rather than by calling the fallback — what is being checked is that
    # something reaches it.
    events = []

    async def on_event(kind, detail):
        events.append((kind, detail))

    old_budget = config.ELEVENLABS_FIRST_BYTE_BUDGET_MS
    config.ELEVENLABS_FIRST_BYTE_BUDGET_MS = 1
    s = vd.DialogueSession(on_audio=lambda *a: asyncio.sleep(0),
                           on_event=on_event)
    try:
        await s.start()
        heard = await _drive(s, "This line will not make its budget.", rid="f1",
                             events=events)
    finally:
        config.ELEVENLABS_FIRST_BYTE_BUDGET_MS = old_budget
        await s.close()
    fb = [d for k, d in events if k == "fallback"]
    ok(any(d.get("cause") == vd.SLOW_FIRST_BYTE for d in fb),
       "a line that misses the first-byte budget falls back, cause recorded "
       f"({[d.get('cause') for d in fb]})")
    ok(any(d.get("model") == config.ELEVENLABS_DETERMINISTIC_MODEL for d in fb),
       f"onto {config.ELEVENLABS_DETERMINISTIC_MODEL}, on the same voice")
    ok(heard["pcm"] > 0,
       f"and the driver still hears the line ({heard['pcm']} bytes) — a "
       "fallback that goes quiet is not a fallback")
    ok(any(k == "utterance_done" for k, _ in events),
       "and the utterance still reports done, so the mouth comes back — a "
       "fallback line waits for no marker the socket is never going to send")

    # TIER 1b: the socket went away mid-utterance.
    events.clear()
    s = vd.DialogueSession(on_audio=lambda *a: asyncio.sleep(0),
                           on_event=on_event)
    await s.start()
    await s.begin("f2")
    await s.delta("f2", "The wire is about to go, ")
    s._ws = None                       # the send fails, exactly as it would
    heard = await _drive(s, "but the sentence still finishes.", rid="f2",
                     events=events)
    await s.close()
    ok(any(d.get("cause") == vd.SOCKET_ERROR
           for k, d in events if k == "fallback"),
       "a socket that goes away mid-sentence falls back for that utterance")
    ok(heard["pcm"] > 0, "and the rest of the sentence is still spoken")

    # TIER 2: ElevenLabs is not answering at all. The key is refused, so
    # neither the socket nor flash can produce anything, and after
    # ELEVENLABS_FAILURES_BEFORE_CEDAR consecutive failures RIO takes her own
    # voice back for the drive.
    events.clear()
    real = vd.api_key
    vd.api_key = lambda: "sk_not_a_key"
    try:
        s = vd.DialogueSession(on_audio=lambda *a: asyncio.sleep(0),
                              on_event=on_event, force_flash=True)
        await s.start()
        for i in range(config.ELEVENLABS_FAILURES_BEFORE_CEDAR + 1):
            await _drive(s, "Nobody is going to hear this.", rid=f"d{i}",
                         wait=1.0, events=events)
        degraded = s.degraded
        await s.close()
    finally:
        vd.api_key = real
    cedar = [d for k, d in events if k == "fallback" and d.get("tier") == "cedar"]
    ok(degraded, "with ElevenLabs refusing everything, the session gives up on "
                 "it rather than retrying into silence for the rest of a drive")
    ok(cedar, "and says so, once, with a cause "
              f"({cedar[0].get('cause') if cedar else None})")
    ok(cedar and cedar[0].get("voice") == config.OPENAI_REALTIME_VOICE,
       f"naming the voice the drive continues in ({config.OPENAI_REALTIME_VOICE})")
    ok(len(cedar) == 1,
       "exactly once — a voice that flickers between two people because the "
       "network is flickering is worse than either of them")


async def run_no_seat():
    """The workspace's dialogue pool, filled for real, and what RIO does then.

    THE DIALOGUE POOL SPECIFICALLY, which means this runs against a v3 session
    whatever the conversation model happens to be. The two transports have two
    different capacity stories and this is the one with a hard, reachable
    limit: measured at 21 on this account (Starter), against 20+ simultaneous
    on the text-to-speech socket without a refusal.

    MEASURED first, then asserted. The seat is taken LAZILY — a connection
    costs nothing; the refusal arrives the first time a session tries to
    synthesise, as a 1008 close carrying `too_many_concurrent_requests`.

    That shape is the reason this test exists. From inside a car it looks
    exactly like a dropped socket, and the reflex that is right for a dropped
    socket — reconnect — is wrong here: the new connection is accepted and the
    next utterance is refused the same way, once per sentence, for the rest of
    the drive.
    """
    section("I. a full dialogue pool — the one refusal that is not a fault")

    import websockets

    v3 = "eleven_v3_conversational"
    url = (vd.WS_URL + f"?model_id={v3}"
           f"&output_format={config.ELEVENLABS_OUTPUT_FORMAT}")
    hdr = {"xi-api-key": vd.api_key()}
    hogs = []

    async def hog():
        ws = await websockets.connect(url, additional_headers=hdr,
                                      max_size=None, open_timeout=25)
        await ws.send(json.dumps({"voices": [vd.voice_id()]}))
        await ws.send(json.dumps({"inputs": [
            {"text": "Holding a seat.", "voice_id": vd.voice_id(),
             "new_turn": True}]}))
        await ws.send(json.dumps({"flush": True}))
        return ws

    events = []

    async def on_event(kind, detail):
        events.append((kind, detail))

    try:
        # Fill it. Sequentially, so the count means something, and stopping the
        # moment the service says it is full rather than guessing the number.
        for _ in range(POOL_PROBE_CAP):
            try:
                hogs.append(await hog())
                await asyncio.sleep(0.4)
            except Exception:
                break
        ok(len(hogs) >= 2, f"held {len(hogs)} dialogue sessions to fill the pool")

        s = vd.DialogueSession(on_audio=lambda *a: asyncio.sleep(0),
                               on_event=on_event, model=v3)
        opened = await s.start()
        ok(opened, "a car still CONNECTS with the pool full — a connection "
                   "costs no seat, so nothing fails until she speaks")

        heard = await _drive(s, "This is the first thing she says.", rid="p1",
                             events=events, wait=25.0)
        ok(any(d.get("cause") == vd.NO_SEAT
               for k, d in events if k == "fallback"),
           "the first utterance is refused, and it is recorded as a full pool "
           "rather than as a dropped socket")
        ok(heard["pcm"] > 0,
           f"the driver still hears the line ({heard['pcm']} bytes), in the "
           "same voice on the fast model")
        ok(s.status()["transport"] == "text_to_dialogue",
           "the session under test is on the dialogue socket, which is the "
           "transport this pool belongs to")
        ok(s.status()["no_seat"] is True,
           "the dialogue socket is PARKED rather than reconnected — going back "
           "for a refusal once per sentence is the failure this avoids")
        ok(s._ws is None, "...and no connection is held while it is parked")

        before = len([1 for k, d in events
                      if k == "fallback" and d.get("cause") == vd.NO_SEAT])
        await _drive(s, "And this is the second.", rid="p2", events=events,
                     wait=25.0)
        after = len([1 for k, d in events
                     if k == "fallback" and d.get("cause") == vd.NO_SEAT])
        ok(after == before,
           "the second utterance goes straight to the fast model without "
           "asking again, and without a second log line about it")
        ok(not s.degraded,
           "and a full pool is NOT the service being down: RIO keeps her own "
           "voice rather than handing the drive back to cedar")
        await s.close()
    finally:
        for ws in hogs:
            try:
                await ws.close()
            except Exception:
                pass


def run_relay(base: str):
    section("I. the relay — the page never sees the key")

    import json
    import urllib.request

    try:
        with urllib.request.urlopen(base + "/voice/status", timeout=10) as r:
            st = json.load(r)
    except Exception as e:
        print(f"  (server not reachable at {base}: {type(e).__name__})")
        return
    ok(st.get("backend") == config.VOICE_BACKEND,
       f"/voice/status reports the backend ({st.get('backend')})")
    ok(st.get("voice_id") == config.ELEVENLABS_VOICE_ID,
       f"and the voice id ({st.get('voice_id')})")
    ok(st.get("conversation_model") == config.ELEVENLABS_CONVERSATION_MODEL
       and st.get("deterministic_model") == config.ELEVENLABS_DETERMINISTIC_MODEL,
       f"both models: {st.get('conversation_model')} for conversation, "
       f"{st.get('deterministic_model')} for everything deterministic")
    ok(st.get("configured") is True, "and that the key is present")
    body = json.dumps(st)
    ok(os.getenv("ELEVENLABS_API_KEY", "no-key-set") not in body,
       "the key itself is nowhere in the response")

    page = Path(REPO / "static").glob("*.js")
    leaked = [p.name for p in page
              if "ELEVENLABS_API_KEY" in p.read_text()]
    ok(not leaked, f"and nowhere in anything the browser downloads {leaked}")

    asyncio.run(_relay_drive(base))


async def _relay_drive(base: str):
    """Speak one line through the RELAY, exactly as the page would.

    The unit tests above drive DialogueSession directly, which proves the
    synthesis. This proves the join: a browser holding no key at all sends text
    deltas over a local socket and gets samples back, which is the arrangement
    the whole server-side design exists for.
    """
    import json

    import websockets

    url = base.replace("https://", "wss://").replace("http://", "ws://") \
              + "/voice/dialogue"
    got = {"audio": 0, "chunks": 0, "words": "", "ready": None}
    try:
        async with websockets.connect(url, max_size=None, open_timeout=10) as ws:
            ready = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
            got["ready"] = ready
            await ws.send(json.dumps({"op": "begin", "rid": "s1"}))
            for word in "Traffic is thinning out ahead, and the sun is about "\
                        "to do something nice.".split(" "):
                await ws.send(json.dumps({"op": "delta", "rid": "s1",
                                          "text": word + " "}))
                await asyncio.sleep(0.02)
            await ws.send(json.dumps({"op": "end", "rid": "s1"}))
            deadline = time.time() + 15
            while time.time() < deadline:
                m = json.loads(await asyncio.wait_for(ws.recv(), timeout=15))
                if m.get("op") == "audio":
                    got["chunks"] += 1
                    got["audio"] += len(m.get("pcm") or "")
                    got["words"] += m.get("text") or ""
                if m.get("op") == "event" and m.get("kind") == "utterance_done":
                    break
    except Exception as e:
        ok(False, f"the relay socket failed: {type(e).__name__}: {str(e)[:120]}")
        return

    ok((got["ready"] or {}).get("op") == "ready"
       and (got["ready"] or {}).get("open") is True,
       "the relay opens a dialogue session for the page and says so")
    ok(got["ready"].get("voice_id") == config.ELEVENLABS_VOICE_ID
       and got["ready"].get("sample_rate") == config.ELEVENLABS_SAMPLE_RATE,
       f"telling it the voice and the sample rate "
       f"({got['ready'].get('sample_rate')} Hz), and nothing else it does not "
       "need")
    ok(got["chunks"] > 1,
       f"text deltas come back as {got['chunks']} pieces of audio, so playback "
       "starts before the sentence is finished")
    ok(got["audio"] > 10000, f"with real samples in them ({got['audio']} b64 chars)")
    ok("Traffic" in got["words"],
       "and every piece carries the words it is for, which is how the page "
       "answers 'how far did she get' when a warning cuts her off")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--live", action="store_true",
                    help="also open the real dialogue socket")
    ap.add_argument("--server", default=None,
                    help="check the relay against a running server")
    ap.add_argument("--pool", action="store_true",
                    help="fill the workspace's dialogue pool for real and "
                         "check what a car does with none left (slow, and "
                         "briefly uses every seat the account has)")
    args = ap.parse_args()

    run_session()
    run_dictation_policy()
    run_chunker()
    run_tags()
    run_firewall()
    run_clips()
    if args.live:
        run_voice_support()
        asyncio.run(run_live())
        asyncio.run(run_other_transport())
        asyncio.run(run_fallbacks())
    if args.pool:
        asyncio.run(run_no_seat())
    if args.server:
        run_relay(args.server.rstrip("/"))

    print("\n" + "=" * 72)
    total = len(PASS) + len(FAIL)
    print(f"{len(PASS)}/{total} checks passed")
    if FAIL:
        print("\nFAILED:")
        for f in FAIL:
            print(f"  - {f}")
    print("=" * 72)
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
