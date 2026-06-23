

import time
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile, Query, Body
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
from openai import OpenAI

import config
import voice
import llm_interface
import vision

app = FastAPI()
client = OpenAI()


@app.get("/")
def index():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    return {"status": "ok", "service": "rio-phase1"}


@app.post("/talk")
async def talk(audio: UploadFile, session_id: str = Query(default=None)):
    t0 = time.time()

    audio_bytes = await audio.read()

    input_path = "/tmp/rio_input.webm"
    with open(input_path, "wb") as f:
        f.write(audio_bytes)

    with open(input_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model=config.OPENAI_STT_MODEL,
            file=f,
        ).text

    t1 = time.time()

    print({
        "transcript": transcript,
        "whisper_seconds": round(t1 - t0, 3),
    })

    def streamer():
        buffer = ""

        for token in llm_interface.generate_stream(transcript):
            buffer += token

            if any(buffer.rstrip().endswith(p) for p in [".", "!", "?"]):
                for chunk in voice.synthesize_stream(buffer):
                    yield chunk
                buffer = ""

        if buffer.strip():
            for chunk in voice.synthesize_stream(buffer):
                yield chunk

    return StreamingResponse(streamer(), media_type="audio/mpeg")

from fastapi import File
import sessions

@app.post("/observe")
async def observe(image: UploadFile = File(...), session_id: str = Query(default=None)):
    import time as _t
    _t0 = _t.time()
    image_bytes = await image.read()
    observation = vision.observe(image_bytes)
    sessions.log_observe(session_id, len(image_bytes), observation, (_t.time() - _t0) * 1000)
    return {"observation": observation}

# --- Phase 2.5 session endpoints ---

@app.post("/session/start")
def session_start_endpoint(metadata: dict = Body(default=None, embed=True)):
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
