"""Idempotent patch script to add Phase 2.5 session support to app.py.

Run this once on the pod, in the rio-phase1 directory:

    cd /workspace/rio-phase1
    python3 apply_sessions_patch.py

It will:
  1. Verify sessions.py exists (you should have copied it next to app.py first).
  2. Add `import sessions` and `from fastapi import Query` to app.py.
  3. Modify the existing /observe endpoint to accept session_id and log to JSONL.
  4. Modify the existing /talk endpoint to accept session_id and log to JSONL.
  5. Append three new endpoints: /session/start, /session/end, /session/mark.
  6. Save a backup at app.py.bak before any changes.

Safe to re-run — it skips any change that's already been applied.

After running, restart uvicorn.
"""

import re
import shutil
import sys
from pathlib import Path

APP = Path("app.py")
BACKUP = Path("app.py.bak")
SESSIONS = Path("sessions.py")


def main() -> int:
    if not APP.exists():
        print(f"ERROR: {APP} not found. Run this from /workspace/rio-phase1/.")
        return 1
    if not SESSIONS.exists():
        print(f"ERROR: {SESSIONS} not found. Copy sessions.py next to app.py first.")
        return 1

    src = APP.read_text()

    # Back up before first modification.
    if not BACKUP.exists():
        shutil.copy(APP, BACKUP)
        print(f"Saved backup at {BACKUP}")
    else:
        print(f"Backup already exists at {BACKUP} (skipping)")

    src = ensure_import(src, "import sessions")
    src = ensure_query_import(src)
    src = patch_observe(src)
    src = patch_talk(src)
    src = append_session_endpoints(src)

    APP.write_text(src)
    print(f"Wrote patched {APP}")
    print()
    print("Now restart uvicorn:")
    print("  pkill -f uvicorn")
    print("  uvicorn app:app --host 0.0.0.0 --port 8000")
    return 0


def ensure_import(src: str, line: str) -> str:
    if line in src:
        print(f"  [skip] '{line}' already present")
        return src
    # Insert after the last `import X` or `from X import Y` block at the top.
    lines = src.splitlines(keepends=True)
    last_import_idx = 0
    for i, l in enumerate(lines):
        stripped = l.lstrip()
        if stripped.startswith(("import ", "from ")):
            last_import_idx = i
    lines.insert(last_import_idx + 1, line + "\n")
    print(f"  [add] '{line}'")
    return "".join(lines)


def ensure_query_import(src: str) -> str:
    # We need Query from fastapi. If `from fastapi import ...` exists and Query is missing, add it.
    pattern = re.compile(r"from fastapi import ([^\n]+)")
    m = pattern.search(src)
    if not m:
        return ensure_import(src, "from fastapi import Query")
    existing = m.group(1)
    if "Query" in existing:
        print("  [skip] 'Query' already in fastapi import")
        return src
    new_line = f"from fastapi import {existing.rstrip()}, Query"
    src = src[:m.start()] + new_line + src[m.end():]
    print("  [add] 'Query' to fastapi import")
    return src


def patch_observe(src: str) -> str:
    """Add session_id query param and JSONL logging to /observe."""
    if "sessions.log_observe" in src:
        print("  [skip] /observe already patched")
        return src

    # Match: @app.post("/observe") followed by async def observe(image: UploadFile = File(...)):
    sig_pattern = re.compile(
        r'(@app\.post\("/observe"\)\s*\nasync def observe\(image: UploadFile = File\(\.\.\.\))\):'
    )
    if not sig_pattern.search(src):
        print("  [warn] couldn't find /observe signature — leaving it alone")
        return src

    src = sig_pattern.sub(
        r'\1, session_id: str = Query(default=None)):\n    import time as _t\n    _t0 = _t.time()',
        src
    )

    # Inject log call right after the observation = vision.observe(...) line.
    src = re.sub(
        r'(observation = vision\.observe\(image_bytes\))\n(\s+return)',
        r'\1\n    sessions.log_observe(session_id, len(image_bytes), observation, (_t.time() - _t0) * 1000)\n\2',
        src
    )
    print("  [add] session_id + JSONL logging to /observe")
    return src


def patch_talk(src: str) -> str:
    """Add session_id query param and JSONL logging to /talk."""
    if "sessions.log_talk" in src:
        print("  [skip] /talk already patched")
        return src

    # /talk signatures vary across versions. Try the common shape we built earlier.
    sig_pattern = re.compile(
        r'(@app\.post\("/talk"\)\s*\nasync def talk\(audio: UploadFile)\):'
    )
    if not sig_pattern.search(src):
        print("  [warn] couldn't find /talk signature — leaving it alone")
        return src

    src = sig_pattern.sub(r'\1, session_id: str = Query(default=None)):', src)
    print("  [add] session_id query param to /talk (manual log call deferred — /talk streams)")
    # /talk uses streaming so we can't easily log the full reply server-side.
    # The iOS app or browser will log talk events client-side via /session/mark with tag="talk".
    return src


def append_session_endpoints(src: str) -> str:
    if '"/session/start"' in src:
        print("  [skip] /session/* endpoints already present")
        return src

    endpoints = '''

# --- Phase 2.5 session endpoints ---

@app.post("/session/start")
def session_start_endpoint(metadata: dict = None):
    sid = sessions.start_session(metadata=metadata)
    return {"session_id": sid}


@app.post("/session/end")
def session_end_endpoint(session_id: str = Query(...)):
    closed = sessions.end_session(session_id)
    return {"closed": closed}


@app.post("/session/mark")
def session_mark_endpoint(session_id: str = Query(...), tag: str = Query(...), note: str = Query(default="")):
    sessions.mark(session_id, tag, {"note": note} if note else None)
    return {"marked": tag}


@app.get("/session/{session_id}")
def session_view_endpoint(session_id: str):
    import json as _json
    path = sessions.SESSIONS_DIR / f"{session_id}.jsonl"
    if not path.exists():
        return {"error": "not found", "session_id": session_id}
    events = []
    for line in path.read_text().splitlines():
        if line.strip():
            try:
                events.append(_json.loads(line))
            except Exception:
                pass
    return {"session_id": session_id, "events": events, "active": sessions.is_active(session_id)}
'''
    print("  [add] /session/start, /end, /mark, /<id> endpoints")
    return src.rstrip() + endpoints


if __name__ == "__main__":
    sys.exit(main())
