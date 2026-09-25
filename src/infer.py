"""
infer.py
--------
Bitta audio fayl uchun "haqiqiy" yoki "deepfake" bashorati.
Keyinchalik bu funksiya Telegram bot ichida ham xuddi shunday chaqiriladi.

Ishlatish:
    python infer.py path/to/audio.wav
"""

import sys
import os
import numpy as np
from tensorflow import keras

from features import extract_mel_spectrogram

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
_model = None


def get_model():
    global _model
    if _model is None:
        path = os.path.join(MODELS_DIR, "final_model.keras")
        if not os.path.exists(path):
            raise RuntimeError("Model topilmadi — avval train.py ni ishga tushiring.")
        _model = keras.models.load_model(path)
    return _model


def predict(audio_path: str) -> dict:
    feat = extract_mel_spectrogram(audio_path)
    feat = np.expand_dims(feat, axis=0)  # batch dimensiyasi
    model = get_model()
    score = float(model.predict(feat, verbose=0)[0][0])
    label = "DEEPFAKE (sun'iy ovoz)" if score >= 0.5 else "HAQIQIY (real ovoz)"
    confidence = score if score >= 0.5 else 1 - score
    return {
        "label": label,
        "deepfake_probability": round(score * 100, 2),
        "confidence": round(confidence * 100, 2),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Foydalanish: python infer.py path/to/audio.wav")
        sys.exit(1)

    result = predict(sys.argv[1])
    print(f"\nFayl: {sys.argv[1]}")
    print(f"Natija: {result['label']}")
    print(f"Deepfake ehtimoli: {result['deepfake_probability']}%")
    print(f"Ishonch darajasi: {result['confidence']}%")
