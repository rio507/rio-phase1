

import time
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile
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
async def talk(audio: UploadFile):
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

@app.post("/observe")
async def observe(image: UploadFile = File(...)):
    image_bytes = await image.read()
    observation = vision.observe(image_bytes)
    return {"observation": observation}
