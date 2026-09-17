#!/usr/bin/env python3
"""drive_voice_report.py — why she stopped talking, from the drive's own log.

    python3 tools/drive_voice_report.py training_data/<session>.jsonl
    python3 tools/drive_voice_report.py --latest

WHY THIS EXISTS. On 2026-09-17 the complaint was "her voice cut out repeatedly
on a mobile drive, and navigation was running". The drive log held the answer
and could not be asked it: the events were there, spread across three streams
(`live` cut-offs from the browser, `nav` from the planner, `mark` from the
page), and reading them meant lining three clocks up by hand and inferring the
rest. This turns that reading into a command.

WHAT IT REFUSES TO DO is guess. Every line it prints is a field somebody wrote
down. Where the log cannot say something -- and it could not say several things
before the instrumentation that ships with this file -- it says so, by name,
rather than filling the hole with the most likely story. A tally that invents
its own missing rows is worse than no tally, because the number is the whole
reason it exists.

THE FIVE CAUSES a driver experiences as one, kept apart:

  nav contention   a deterministic line refused the mouth because another one
                   held it, or given up on with nothing underneath it
  preemption       something that matters more took the mouth mid-answer
  refusal          the API would not produce the response at all -- the
                   token-per-minute ceiling lands here
  connection       the transport went away
  false barge-in   the gate stopped her for something that turned out not to
                   be a person
"""
import json, sys, os, glob, collections, datetime

# Cut-off kinds that are an answer the driver LOST, against the ones that are
# the machinery working. `bus_health` is a heartbeat, not a fault; a phantom
# that was self-answered is a transcript the gate refused while the response
# for it was already in flight.
LOST = {"cutoff", "dictation_refused", "response_failed", "transport_lost",
        "direct_speech_failed", "resume_failed"}
# Not failures: the detector firing, it being held over her opening syllable,
# a transcript arriving after the answer to it started, and the bus heartbeat.
HEARTBEAT = {"bus_health", "spoke", "barge_detected", "barge_deferred",
             "turn_self"}

CAUSE_OF = {
    "barge_missed": "gate refused a person",
    "dictation_refused": "nav contention",
    "response_failed": "rate limit / token cap",
    "transport_lost": "connection",
    "session_error": "session error",
}
CUTOFF_CAUSE = {
    "false_barge_in": "false barge-in",
    "barge_in": "driver barge-in (correct)",
    "preempted": "arbiter preemption",
    "token_cap": "rate limit / token cap",
    "transport": "connection",
}


def load(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                continue
    return rows


def hhmmss(t):
    return datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S.%f")[:-3]


def classify(rows):
    """Every voice event, with the cause the log itself records."""
    out = []
    for r in rows:
        k, p = r.get("kind"), (r.get("payload") or {})
        if k == "live" and p.get("event") == "cutoff":
            kind = p.get("kind")
            if kind in HEARTBEAT:
                continue
            cause = p.get("cause") or kind
            label = CUTOFF_CAUSE.get(cause) or CAUSE_OF.get(kind) or cause
            out.append((r["t"], kind, label, p))
        elif k == "mark" and p.get("tag") == "VOICE_SILENT":
            try:
                note = json.loads(p.get("note") or "{}")
            except ValueError:
                note = {}
            out.append((r["t"], "voice_silent",
                        "nav contention" if note.get("reason") == "busy"
                        else "line not spoken: %s" % note.get("reason"), note))
    return out


def nav_events(rows):
    return [(r["t"], (r["payload"] or {})) for r in rows if r.get("kind") == "nav"]


def near_maneuver(t, navs, window=8.0):
    """Was a maneuver call within `window` seconds either side of this event?

    The honest form of "do cut-offs cluster around maneuvers" when the events
    themselves do not carry nav state. Events recorded after the instrumentation
    added on 2026-09-17 carry `nav.to_maneuver_m` and do not need this.
    """
    for nt, p in navs:
        if p.get("event", "").endswith("_CALL") and abs(nt - t) <= window:
            return p.get("call_type")
    return None


def latency(rows):
    got = [(r["t"], r["payload"]) for r in rows
           if r.get("kind") == "live"
           and (r["payload"] or {}).get("kind") == "spoke"]
    return got


def main(argv):
    if not argv or argv[0] == "--latest":
        files = sorted(glob.glob("training_data/*.jsonl"), key=os.path.getmtime)
        if not files:
            print("no session logs in training_data/")
            return 1
        path = files[-1]
    else:
        path = argv[0]
    rows = load(path)
    if not rows:
        print("empty log: %s" % path)
        return 1

    t0, t1 = rows[0]["t"], rows[-1]["t"]
    print("drive %s" % os.path.basename(path))
    print("  %s -> %s  (%.0f s, %d events)"
          % (hhmmss(t0), hhmmss(t1), t1 - t0, len(rows)))

    navs = nav_events(rows)
    route = [p for _, p in navs if p.get("event") == "NAV_ROUTE_STARTED"]
    if route:
        rt = [t for t, p in navs if p.get("event") == "NAV_ROUTE_STARTED"][0]
        dest = (route[0].get("destination") or {}).get("display_name")
        print("  route to %s from %s" % (dest, hhmmss(rt)))
    else:
        print("  no route this drive")

    events = classify(rows)
    print("\n--- why she stopped talking (%d events) ---" % len(events))
    tally = collections.Counter(lbl for _, _, lbl, _ in events)
    if not tally:
        print("  nothing recorded")
    for label, n in tally.most_common():
        print("  %-28s %d" % (label, n))

    print("\n--- timeline ---")
    for t, kind, label, p in events:
        nav = p.get("nav") if isinstance(p, dict) else None
        if isinstance(nav, dict) and nav.get("to_maneuver_m") is not None:
            where = "  [%.0f m to %s]" % (nav["to_maneuver_m"],
                                          nav.get("maneuver_id") or "maneuver")
        else:
            call = near_maneuver(t, navs)
            where = "  [near %s call]" % call if call else ""
        extra = ""
        for key in ("text", "why", "reason", "detail", "holder", "code",
                    "said_chars", "self_answered"):
            v = p.get(key) if isinstance(p, dict) else None
            if v not in (None, "", False):
                extra += " %s=%s" % (key, json.dumps(v)[:60])
        print("  %s  %-22s %-26s%s%s"
              % (hhmmss(t), kind, label, extra.strip(), where))

    lat = latency(rows)
    print("\n--- turn end -> first audio ---")
    if not lat:
        print("  NOT RECORDED in this log. The `spoke` event that carries it")
        print("  was added on 2026-09-17; drives before it have tool durations")
        print("  and no measurement of what the driver actually waited.")
    else:
        by = collections.defaultdict(list)
        for _, p in lat:
            if p.get("wait_ms") is not None:
                by[p.get("turn_kind") or "?"].append(p["wait_ms"])
        for kind, xs in sorted(by.items()):
            xs.sort()
            print("  %-14s n=%-3d  p50=%-6d p90=%-6d max=%d"
                  % (kind, len(xs), xs[len(xs) // 2],
                     xs[min(len(xs) - 1, int(len(xs) * 0.9))], xs[-1]))
        adj = [p["wait_ms"] for _, p in lat
               if p.get("wait_ms") is not None
               and isinstance(p.get("nav"), dict)
               and (p["nav"].get("to_maneuver_m") or 1e9) < 300]
        rest = [p["wait_ms"] for _, p in lat
                if p.get("wait_ms") is not None
                and not (isinstance(p.get("nav"), dict)
                         and (p["nav"].get("to_maneuver_m") or 1e9) < 300)]
        if adj:
            print("  within 300 m of a maneuver: n=%d mean=%.0f ms"
                  % (len(adj), sum(adj) / len(adj)))
            print("  elsewhere:                 n=%d mean=%.0f ms"
                  % (len(rest), sum(rest) / len(rest) if rest else 0))
        else:
            print("  no answers were given within 300 m of a maneuver")

    # THE ECHO GATE, JUDGED IN BOTH DIRECTIONS. A drive that counts only
    # false_barge_in can argue for one thing: tightening. `barge_missed` is
    # the same threshold being wrong the other way, and until it existed there
    # was no evidence that could have argued against a tightening.
    def kinds(*names):
        return [(r["t"], r["payload"]) for r in rows
                if r.get("kind") == "live"
                and (r["payload"] or {}).get("kind") in names]

    fired = kinds("barge_detected")
    missed = kinds("barge_missed")
    false_pos = [p for _, p in kinds("cutoff") if p.get("cause") == "false_barge_in"]
    suppressed = kinds("echo_suppressed")
    if fired or missed or suppressed or false_pos:
        print("\n--- the echo gate ---")
        print("  detector fired        %d" % len(fired))
        print("  suppressed as her     %d" % len(suppressed))
        print("  false barge-in        %d   (stopped her for nobody)" % len(false_pos))
        print("  MISSED barge-in       %d   (refused a person)" % len(missed))
        if not fired and not missed:
            print("  note: barge_detected/barge_missed were added on "
                  "2026-09-17. A drive")
            print("  older than that cannot say whether the detector fired at "
                  "all.")
        for _, p in missed:
            print("    margin %s dB against %s dB required, %s ms before %s"
                  % (p.get("margin_db"), p.get("required_db"),
                     p.get("since_suppress_ms"), json.dumps(p.get("text"))[:50]))

    phantoms = [(r["t"], r["payload"]) for r in rows
                if r.get("kind") == "live"
                and (r["payload"] or {}).get("kind") == "turn_phantom"]
    if phantoms:
        unknown = [p for _, p in phantoms if p.get("self_answered") is None]
        lost = [p for _, p in phantoms if p.get("self_answered") is False]
        print("\n--- refused transcripts (%d) ---" % len(phantoms))
        if unknown:
            print("  %d of them predate `self_answered`, so whether the "
                  "question" % len(unknown))
            print("  was answered anyway cannot be read off this log.")
        else:
            print("  %d were already being answered; %d were not"
                  % (len(phantoms) - len(lost), len(lost)))
    # THE PAGE AND THE SESSION. When she stopped because the session ended
    # or the page went away, no answer above applies, and before 2026-09-17
    # the log could not say which of those had happened either.
    def marks(*tags):
        out = []
        for r in rows:
            p = r.get("payload") or {}
            if r.get("kind") == "mark" and p.get("tag") in tags:
                try:
                    note = json.loads(p.get("note") or "{}")
                except ValueError:
                    note = {}
                out.append((r["t"], p["tag"], note))
        return out

    started = [r["t"] for r in rows if r.get("kind") == "live"
               and (r["payload"] or {}).get("event") == "session_started"]
    ended = kinds("session_ended")
    lastlive = [r["t"] for r in rows if r.get("kind") == "live"
                and (r["payload"] or {}).get("event") == "cutoff"]
    vis = marks("PAGE_VISIBILITY", "FRAMES_HIDDEN", "FRAMES_VISIBLE")
    wake = marks("WAKE_LOCK")
    stalls = marks("MAIN_THREAD_STALL")
    audio = kinds("audio_interrupted", "audio_resumed", "mic_state", "peer_state")
    endrow = [r for r in rows if r.get("kind") == "session_end"]
    print("\n--- the page and the session ---")
    for t in started:
        print("  %s  voice session started" % hhmmss(t))
    if ended:
        for t, p in ended:
            a = p.get("audio_state") or {}
            print("  %s  voice session ENDED: %s  (age %ss%s)"
                  % (hhmmss(t), p.get("reason"), p.get("age_s"),
                     ", page hidden" if p.get("hidden") else ""))
            if a:
                print("            mouth paused=%s mic muted=%s peer=%s"
                      % (a.get("element_paused"), a.get("mic_muted"), a.get("peer")))
    elif started:
        print("  voice session end: NOT RECORDED (session_ended was added "
              "2026-09-17).")
        if lastlive:
            print("  last live event %s -- the end can only be placed after it."
                  % hhmmss(lastlive[-1]))
    hidden_at = None
    for t, tag, note in vis:
        state = note.get("state") if tag == "PAGE_VISIBILITY" else (
            "hidden" if tag == "FRAMES_HIDDEN" else "visible")
        if state != "visible":
            hidden_at = t
            extra = ""
            if tag == "PAGE_VISIBILITY":
                a = note.get("audio") or {}
                extra = "  live=%s bus=%s mic_muted=%s mouth_paused=%s wake_lock=%s" % (
                    note.get("live"), note.get("bus"), a.get("mic_muted"),
                    a.get("element_paused"), note.get("wake_lock"))
            print("  %s  page HIDDEN%s" % (hhmmss(t), extra))
        else:
            dur = "" if hidden_at is None else " after %.0f s" % (t - hidden_at)
            hidden_at = None
            print("  %s  page visible%s" % (hhmmss(t), dur))
    if vis and not any(tag == "PAGE_VISIBILITY" for _, tag, _ in vis):
        print("  (FRAMES_* only: the audio-session snapshot on PAGE_VISIBILITY "
              "was added 2026-09-17)")
    for t, tag, note in wake:
        print("  %s  wake lock %s%s" % (hhmmss(t), note.get("state"),
              (" (" + note["error"] + ")") if note.get("error") else ""))
    if not wake and started:
        print("  wake lock: NOT RECORDED (WAKE_LOCK was added 2026-09-17)")
    if stalls:
        print("  main-thread stalls: %d (largest %s ms)"
              % (len(stalls), max(n.get("late_ms") or 0 for _, _, n in stalls)))
    else:
        print("  main-thread stalls: none reported by the 250 ms detector")
    for t, p in audio:
        print("  %s  %-18s %s%s" % (hhmmss(t), p.get("kind"),
              p.get("state") or "", "  (page hidden)" if p.get("hidden") else ""))
    for r in endrow:
        print("  %s  drive log closed: %s"
              % (hhmmss(r["t"]), (r["payload"] or {}).get("reason")))

    selfs = [r for r in rows if r.get("kind") == "live"
             and (r["payload"] or {}).get("kind") == "turn_self"]
    if selfs:
        print("\n--- transcripts that arrived after the answer began (%d) ---"
              % len(selfs))
        print("  Ordinary: transcription loses the race with the model. Before")
        print("  2026-09-17 these were counted as refused barge-ins.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
