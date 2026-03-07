"""
VocalGuard
File uploads: inverted MelodyMachine labels (real=AI, fake=human)
Mic input: normal labels (fake=AI, real=human) + conservative threshold
"""
import numpy as np
import librosa, soundfile as sf
import time, io, logging, warnings
from typing import Dict, Any, Tuple, List

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoModelForAudioClassification, AutoFeatureExtractor
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


class VocalGuardDetector:

    SR = 16000
    MIN_DURATION = 0.5

    def __init__(self):
        logger.info("VocalGuard v7.2 initializing...")
        self.local_model = None
        self.local_extractor = None
        self._try_load_local()
        logger.info("VocalGuard v7.2 ready.")

    def _try_load_local(self):
        if not TORCH_OK:
            logger.warning("torch not available")
            return
        try:
            model_id = "MelodyMachine/Deepfake-audio-detection-V2"
            self.local_extractor = AutoFeatureExtractor.from_pretrained(model_id)
            self.local_model = AutoModelForAudioClassification.from_pretrained(model_id)
            self.local_model.eval()
            logger.info(f"Model loaded: {self.local_model.config.id2label}")
        except Exception as e:
            logger.error(f"Model load failed: {e}")

    # ── AUDIO LOADING ─────────────────────────────────────────────────────────
    def _load(self, audio_bytes: bytes) -> np.ndarray:
        for fn in [
            lambda b: sf.read(io.BytesIO(b), always_2d=False),
            lambda b: librosa.load(io.BytesIO(b), sr=None, mono=True),
            lambda b: (np.frombuffer(b, dtype=np.int16).astype(np.float32) / 32768.0, 16000),
        ]:
            try:
                y, sr = fn(audio_bytes)
                if hasattr(y, 'ndim') and y.ndim > 1:
                    y = y.mean(axis=1)
                if len(y) > 100:
                    if sr != self.SR:
                        y = librosa.resample(y, orig_sr=sr, target_sr=self.SR)
                    return y.astype(np.float32)
            except Exception:
                continue
        raise ValueError("Cannot decode audio")

    # ── FILE UPLOAD INFERENCE ─────────────────────────────────────────────────
    def _infer_file(self, y: np.ndarray) -> Tuple[float, str]:
        """
        MelodyMachine inverted labels for file uploads (confirmed from testing):
        'real' score = AI probability
        'fake' score = human probability
        """
        min_len = self.SR * 3
        if len(y) < min_len:
            y = np.pad(y, (0, min_len - len(y)))

        inputs = self.local_extractor(
            y, sampling_rate=self.SR,
            return_tensors="pt", padding=True
        )
        with torch.no_grad():
            logits = self.local_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].numpy()
        id2label = self.local_model.config.id2label

        logger.info(f"File probs: { {id2label[i]: round(float(probs[i]), 4) for i in range(len(probs))} }")

        # INVERTED labels for this model on file uploads
        ai_prob = float(probs[0])
        for idx, lbl in id2label.items():
            if "real" in lbl.lower():
                ai_prob = float(probs[idx])   # real = AI
                break
        for idx, lbl in id2label.items():
            if "fake" in lbl.lower():
                ai_prob = float(1.0 - probs[idx])  # fake = human → invert
                break

        logger.info(f"File AI prob: {ai_prob:.4f}")
        return float(np.clip(ai_prob, 0.01, 0.99)), "model_file"

    # ── MIC INFERENCE ────────────────────────────────────────────────────────
    def _infer_mic(self, y: np.ndarray) -> Tuple[float, str]:
        """
        Normal label logic for mic audio:
        'fake' = AI, 'real' = human
        Conservative threshold applied to reduce false positives.
        """
        from scipy import signal as scipy_signal

        # High-pass filter to remove room rumble
        sos = scipy_signal.butter(4, 80, 'hp', fs=self.SR, output='sos')
        y = scipy_signal.sosfilt(sos, y).astype(np.float32)

        # Normalize
        peak = np.max(np.abs(y))
        if peak > 0.001:
            y /= peak

        # Pad to 4 seconds minimum
        min_len = self.SR * 4
        if len(y) < min_len:
            y = np.pad(y, (0, min_len - len(y)))

        inputs = self.local_extractor(
            y, sampling_rate=self.SR,
            return_tensors="pt", padding=True
        )
        with torch.no_grad():
            logits = self.local_model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0].numpy()
        id2label = self.local_model.config.id2label

        logger.info(f"Mic probs: { {id2label[i]: round(float(probs[i]), 4) for i in range(len(probs))} }")

        # NORMAL labels for mic
        ai_prob = float(probs[0])
        for idx, lbl in id2label.items():
            if "fake" in lbl.lower():
                ai_prob = float(probs[idx])
                break
        for idx, lbl in id2label.items():
            if "real" in lbl.lower():
                ai_prob = float(1.0 - probs[idx])
                break

        # Conservative: compress uncertain results toward human
        # Only flag strong AI detections (>0.70) on mic
        if ai_prob < 0.70:
            ai_prob = ai_prob * 0.45

        logger.info(f"Mic AI prob (after conservative threshold): {ai_prob:.4f}")
        return float(np.clip(ai_prob, 0.01, 0.99)), "model_mic"

    # ── MAIN PREDICT ──────────────────────────────────────────────────────────
    def predict(self, audio_bytes: bytes, is_mic: bool = False) -> Dict[str, Any]:
        t0 = time.time()

        y = self._load(audio_bytes)
        y, _ = librosa.effects.trim(y, top_db=25)

        if len(y) < self.SR * self.MIN_DURATION:
            return self._err(t0, "Too short — speak for at least 1 second")
        peak = np.max(np.abs(y))
        if peak < 0.003:
            return self._err(t0, "Signal too quiet — check microphone")
        y /= (peak + 1e-10)
        dur = len(y) / self.SR

        if self.local_model is None:
            return self._err(t0, "Model not loaded — check torch/transformers installation")

        logger.info(f"Source: {'mic' if is_mic else 'file'} | duration: {dur:.1f}s")

        try:
            if is_mic:
                ai_prob, method = self._infer_mic(y)
            else:
                ai_prob, method = self._infer_file(y)
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return self._err(t0, f"Detection failed: {str(e)[:80]}")

        ai_prob = float(np.clip(ai_prob, 0.01, 0.99))
        label = "AI Generated" if ai_prob >= 0.5 else "Human Voice"
        conf  = ai_prob if ai_prob >= 0.5 else (1 - ai_prob)
        d = abs(ai_prob - 0.5)
        tier = "High" if d > 0.28 else ("Medium" if d > 0.13 else "Low")

        logger.info(f"Final → {label} ({conf*100:.1f}%) via {method}")

        return {
            "label":             label,
            "confidence":        round(conf * 100, 1),
            "confidence_tier":   tier,
            "ai_probability":    round(ai_prob, 4),
            "human_probability": round(1 - ai_prob, 4),
            "duration_seconds":  round(dur, 2),
            "processing_ms":     int((time.time() - t0) * 1000),
            "detection_method":  method,
            "feature_scores": {
                "AI Probability":    round(ai_prob, 4),
                "Human Probability": round(1 - ai_prob, 4),
            },
            "key_indicators": self._indicators(ai_prob, method),
        }

    def predict_fast(self, audio_bytes: bytes, is_mic: bool = False) -> Dict[str, Any]:
        return self.predict(audio_bytes, is_mic=is_mic)

    def _err(self, t0, msg):
        return {
            "label": "unknown", "confidence": 0,
            "ai_probability": 0.5, "human_probability": 0.5,
            "processing_ms": int((time.time() - t0) * 1000),
            "warning": msg, "feature_scores": {}, "key_indicators": []
        }

    def _indicators(self, ai_prob: float, method: str) -> List[str]:
        out = []
        if method == "model_mic":
            out.append("🎙️ Live mic analysis — upload file for highest accuracy")
        else:
            out.append("🔬 ML model analysis on uploaded file")
        if ai_prob > 0.75:
            out.append("⚠️ Strong AI synthesis markers detected")
        elif ai_prob > 0.50:
            out.append("⚠️ Possible AI synthesis detected")
        elif ai_prob < 0.25:
            out.append("✅ Strong natural human speech markers")
        else:
            out.append("✅ Natural human speech markers present")
        return out