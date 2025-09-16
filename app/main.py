import os
import re
import math
import uuid
import time
import shutil
import tempfile
import subprocess
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Speech-to-text
from faster_whisper import WhisperModel

# Text analysis
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI(title="Interview Auditor API", version="1.0")

# Allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Config
WHISPER_MODEL_SIZE = "base"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_whisper_model = None
_lm_tokenizer = None
_lm_model = None

SUPPORTED_AUDIO = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg"}
SUPPORTED_VIDEO = {".mp4", ".mov", ".mkv", ".webm"}
AUDIO_SAMPLE_RATE = 16000


def load_whisper():
    global _whisper_model
    if _whisper_model is None:
        _whisper_model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE, compute_type="int8")
    return _whisper_model


def load_lm():
    global _lm_tokenizer, _lm_model
    if _lm_model is None:
        name = "distilgpt2"
        _lm_tokenizer = AutoTokenizer.from_pretrained(name)
        _lm_model = AutoModelForCausalLM.from_pretrained(name).to(DEVICE)
        _lm_model.eval()
    return _lm_tokenizer, _lm_model


def extract_audio_if_needed(in_path: str) -> str:
    ext = os.path.splitext(in_path)[1].lower()
    if ext in SUPPORTED_AUDIO:
        return in_path
    if ext in SUPPORTED_VIDEO:
        out_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}.wav")
        cmd = [
            "ffmpeg", "-y", "-i", in_path,
            "-vn", "-acodec", "pcm_s16le", "-ac", "1", "-ar", str(AUDIO_SAMPLE_RATE), out_path
        ]
        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        return out_path
    raise ValueError("Unsupported file type")


def transcribe(audio_path: str) -> Dict[str, Any]:
    model = load_whisper()
    segments, info = model.transcribe(audio_path, beam_size=5)
    transcript = " ".join([seg.text.strip() for seg in segments])
    return {
        "text": transcript,
        "duration": info.duration,
        "language": info.language
    }


def compute_perplexity(text: str) -> float:
    tokenizer, model = load_lm()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(model.device)
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    return math.exp(loss.item())


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    tmp_dir = tempfile.mkdtemp(prefix="auditor_")
    in_path = os.path.join(tmp_dir, file.filename)
    with open(in_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        audio_path = extract_audio_if_needed(in_path)
        asr = transcribe(audio_path)
        text = asr["text"].strip()

        if not text:
            return {"ok": True, "message": "No speech detected", "transcript": "", "metrics": None}

        ppl = compute_perplexity(text)
        score = max(0, min(100, 100 - (ppl / 2)))

        return {
            "ok": True,
            "transcript": text,
            "metrics": {
                "duration_sec": asr["duration"],
                "language": asr["language"],
                "words": len(re.findall(r'\w+', text))
            },
            "ai_likeness": {
                "score": round(score, 1),
                "features": {"perplexity": ppl}
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
