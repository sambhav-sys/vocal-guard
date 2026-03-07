"""
Vocal Guard - Deepfake Audio Detector
FastAPI Backend | AWS-Compatible
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import uvicorn
import numpy as np
import io
import os
import logging

from detector import VocalGuardDetector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Vocal Guard API",
    description="Real-time deepfake audio detection using Mel-spectrogram analysis",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector = VocalGuardDetector()

# Serve frontend static files
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
app.mount("/static", StaticFiles(directory=frontend_path), name="static")


@app.get("/")
async def root():
    return FileResponse(os.path.join(frontend_path, "index.html"))


@app.get("/health")
async def health_check():
    return {"status": "healthy", "model": "VocalGuard v3.0", "ready": True}


@app.post("/analyze")
async def analyze_audio(file: UploadFile = File(...)):
    """
    Analyze audio file for deepfake detection.
    Accepts WAV/WebM audio, returns prediction with confidence.
    """
    try:
        if not file.content_type or not any(
            ct in file.content_type for ct in ["audio", "octet-stream", "webm", "wav", "ogg"]
        ):
            logger.warning(f"Received content type: {file.content_type}")

        audio_bytes = await file.read()

        if len(audio_bytes) < 100:
            raise HTTPException(status_code=400, detail="Audio too short or empty")

        logger.info(f"Processing audio chunk: {len(audio_bytes)} bytes")

        result = detector.predict(audio_bytes)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Analysis error: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")


@app.post("/analyze-stream")
async def analyze_stream(file: UploadFile = File(...)):
    """
    Lightweight endpoint optimized for real-time streaming analysis.
    Returns faster results with less feature detail.
    """
    try:
        audio_bytes = await file.read()
        if len(audio_bytes) < 50:
            return {"label": "unknown", "confidence": 0, "processing_ms": 0}

        result = detector.predict_fast(audio_bytes)
        return result

    except Exception as e:
        logger.error(f"Stream analysis error: {e}")
        return {"label": "unknown", "confidence": 0.5, "processing_ms": 0}


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
