import os
import re
import math
import uuid
import shutil
import tempfile
import subprocess
from typing import Dict, Any
import numpy as np

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Speech-to-text
from faster_whisper import WhisperModel

# Text analysis
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI(title="Interview Auditor API", version="2.0")

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


def compute_burstiness(text: str) -> float:
    sentences = re.split(r'[.!?]', text)
    lengths = [len(s.split()) for s in sentences if s.strip()]
    if len(lengths) < 2:
        return 0.0
    return np.std(lengths) / (np.mean(lengths) + 1e-6)


def compute_repetition(text: str) -> float:
    words = re.findall(r'\w+', text.lower())
    if not words:
        return 0.0
    unique = set(words)
    return 1 - (len(unique) / len(words))


def compute_word_diversity(text: str) -> float:
    words = re.findall(r'\w+', text.lower())
    if not words:
        return 0.0
    return len(set(words)) / len(words)


def compute_filler_ratio(text: str) -> float:
    fillers = {"um", "uh", "like", "you know", "ah", "er", "hmm"}
    words = re.findall(r'\w+', text.lower())
    if not words:
        return 0.0
    filler_count = sum(word in fillers for word in words)
    return filler_count / len(words)


from sentence_transformers import SentenceTransformer, util

_semantic_model = None

def load_semantic_model():
    global _semantic_model
    if _semantic_model is None:
        _semantic_model = SentenceTransformer("all-MiniLM-L6-v2")  # lightweight
    return _semantic_model


def compute_semantic_coherence(text: str) -> float:
    model = load_semantic_model()
    sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
    if len(sentences) < 2:
        return 1.0  # trivially coherent
    embeddings = model.encode(sentences, convert_to_tensor=True)
    sims = []
    for i in range(len(embeddings) - 1):
        sims.append(util.cos_sim(embeddings[i], embeddings[i+1]).item())
    return max(0, min(1, sum(sims) / len(sims)))  # avg similarity


def compute_ai_likeness(text: str) -> Dict[str, Any]:
    ppl = compute_perplexity(text)
    burstiness = compute_burstiness(text)
    repetition = compute_repetition(text)
    diversity = compute_word_diversity(text)
    fillers = compute_filler_ratio(text)
    coherence = compute_semantic_coherence(text)

    # Weighted scoring
    score = (
        (100 - min(ppl / 2, 100)) * 0.5   # perplexity
        + coherence * 100 * 0.2           # semantic coherence
        + (1 - repetition) * 100 * 0.1    # repetition penalty
        + diversity * 100 * 0.1           # diversity
        + (1 - fillers) * 100 * 0.1       # filler penalty
    )

    return {
        "score": round(max(0, min(100, score)), 1),
        "features": {
            "perplexity": ppl,
            "burstiness": burstiness,
            "repetition_rate": repetition,
            "diversity": diversity,
            "filler_ratio": fillers,
            "coherence": coherence
        }
    }



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

        ai_metrics = compute_ai_likeness(text)

        return {
            "ok": True,
            "transcript": text,
            "metrics": {
                "duration_sec": asr["duration"],
                "language": asr["language"],
                "words": len(re.findall(r'\w+', text))
            },
            "ai_likeness": ai_metrics
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"ok": False, "error": str(e)})
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
