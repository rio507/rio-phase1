

import time
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse, HTMLResponse
from openai import OpenAI

import config
import voice
import llm_interface
import vision

app = FastAPI()
client = OpenAI()


@app.get("/", response_class=HTMLResponse)
def index():
    return """
<!DOCTYPE html>
<html>
<head>
  <title>RIO Phase 1</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body {
      background: #050505;
      color: white;
      font-family: -apple-system, BlinkMacSystemFont, sans-serif;
      display: flex;
      align-items: center;
      justify-content: center;
      min-height: 100vh;
      margin: 0;
    }

    .wrap {
      text-align: center;
      max-width: 420px;
      padding: 24px;
    }

    h1 {
      font-size: 56px;
      letter-spacing: -2px;
      margin-bottom: 8px;
    }

    p {
      opacity: 0.65;
      line-height: 1.5;
    }

    button {
      margin-top: 28px;
      width: 170px;
      height: 170px;
      border-radius: 999px;
      border: 1px solid rgba(255,255,255,0.2);
      background: radial-gradient(circle at top, #222, #000);
      color: white;
      font-size: 18px;
      cursor: pointer;
    }

    button.recording {
      background: radial-gradient(circle at top, #7a1010, #200000);
    }

    audio {
      margin-top: 24px;
      width: 100%;
    }

    #status {
      margin-top: 20px;
      font-size: 14px;
      opacity: 0.7;
    }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>RIO</h1>
    <p>Your AI driving companion. Press and 
hold to talk.</p>

    <button id="recordBtn">Hold to Talk</button>
<button id="frameBtn" style="position:fixed;bottom:20px;right:20px;padding:14px 24px;background:#2563eb;color:white;border:none;border-radius:12px;cursor:pointer;font-size:16px;z-index:9999;">Send Frame</button>
<video id="video" autoplay playsinline muted style="display:none;"></video>
<div id="observation" style="position:fixed;bottom:80px;right:20px;max-width:380px;background:#1f1f23;color:#eee;padding:12px;border-radius:8px;font-size:12px;font-family:monospace;display:none;white-space:pre-wrap;z-index:9999;"></div> <div id="status">Idle</div>
    <audio id="audio" controls autoplay></audio>
  </div>

  <script>
    let mediaRecorder;
    let chunks = [];
    let stream;

    const btn = document.getElementById("recordBtn");
    const status = document.getElementById("status");
    const audio = document.getElementById("audio");

    async function initMic() {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    }

    async function startRecording() {
      if (!stream) {
        await initMic();
      }

      chunks = [];
      mediaRecorder = new MediaRecorder(stream, { mimeType: "audio/webm" });

      mediaRecorder.ondataavailable = e => {
        if (e.data.size > 0) chunks.push(e.data);
      };

      mediaRecorder.onstop = async () => {
        status.innerText = "Thinking...";

        const blob = new Blob(chunks, { type: "audio/webm" });
        const formData = new FormData();
        formData.append("audio", blob, "audio.webm");

        const response = await fetch("/talk", {
          method: "POST",
          body: formData
        });

        const audioBlob = await response.blob();
        const url = URL.createObjectURL(audioBlob);
        audio.src = url;
        await audio.play();

        status.innerText = "Idle";
      };

      mediaRecorder.start();
      btn.classList.add("recording");
      status.innerText = "Listening...";
    }

    function stopRecording() {
      if (mediaRecorder && mediaRecorder.state === "recording") {
        mediaRecorder.stop();
      }
      btn.classList.remove("recording");
    }

    btn.addEventListener("mousedown", startRecording);
    btn.addEventListener("mouseup", stopRecording);

    btn.addEventListener("touchstart", e => {
      e.preventDefault();
      startRecording();
    });

    btn.addEventListener("touchend", e => {
      e.preventDefault();
      stopRecording();
    });
const frameBtn = document.getElementById('frameBtn');
const observationBox = document.getElementById('observation');

if (frameBtn) {
  frameBtn.addEventListener('click', async () => {
    try {
      observationBox.style.display = 'block';
      observationBox.textContent = 'Opening camera...';

      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      const video = document.getElementById('video');
      video.srcObject = stream;

      await video.play();
      await new Promise(resolve => setTimeout(resolve, 1000));
      observationBox.textContent = 'Capturing frame...';

      const canvas = document.createElement('canvas');
      canvas.width = video.videoWidth || 1280;
      canvas.height = video.videoHeight || 720;

      const ctx = canvas.getContext('2d');
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      stream.getTracks().forEach(t => t.stop());

      canvas.toBlob(async (blob) => {
        observationBox.textContent = 'Analyzing...';

        const fd = new FormData();
        fd.append('image', blob, 'frame.jpg');

        const r = await fetch('/observe', {
          method: 'POST',
          body: fd
        });

        const j = await r.json();

        observationBox.textContent = j.observation || JSON.stringify(j, null, 2);
      }, 'image/jpeg', 0.85);

    } catch (err) {
      console.error(err);
      observationBox.style.display = 'block';
      observationBox.textContent = 'Frame error: ' + err.message;
    }
  });
} 
 </script>
</body>
</html>
    """


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
