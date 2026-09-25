"""
generate_dummy_data.py
-----------------------
DIQQAT: Bu haqiqiy ovoz namunalari EMAS! Bu faqat butun pipeline'ni
(feature extraction -> dataset -> CNN -> train -> evaluate -> infer)
xatosiz ishlashini tekshirish uchun sun'iy ravishda yaratilgan sinov fayllari.

Haqiqiy loyiha uchun buning o'rniga quyidagilardan foydalaning:
    - ASVspoof 2019/2021 dataset: https://datashare.ed.ac.uk/handle/10283/3336
    - "Fake-or-Real (FoR)" dataset: Kaggle'da mavjud
    - O'zingiz diktafonda yozgan real ovoz + biror TTS/voice-clone xizmatida
      yaratilgan sun'iy ovoz namunalari

Bu skript "real" sifatida — tabiiy nutqqa xos, ko'p garmonikali va biroz
"notekis" signal, "fake" sifatida esa — juda "tekis", tsikllanuvchi,
sun'iy TTS ovoziga xos ba'zi statistik xususiyatlarni taqlid qiluvchi
signal generatsiya qiladi. Bu ikkisi orasidagi farq HAQIQIY deepfake
farqidan ancha sodda, shu sababli bu yerda erishilgan "aniqlik" haqiqiy
loyihadagi natijani anglatmaydi — bu FAQAT kod ishlashini tekshirish uchun.
"""

import os
import numpy as np
import soundfile as sf

SR = 16000
DURATION = 4.0
N_SAMPLES_PER_CLASS = 60

REAL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "real")
FAKE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "fake")


def make_real_like(seed: int) -> np.ndarray:
    """Tabiiy nutqqa o'xshatilgan signal: bir nechta harmonika + jitter + shimmer + nafas shovqini."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)

    f0 = 120 + rng.uniform(-10, 10)                  # asosiy pitch (odam ovozi ~100-200Hz)
    jitter = rng.normal(0, 1.5, size=t.shape).cumsum() * 0.002   # pitch tebranishi (tabiiy)
    signal = np.zeros_like(t)
    for harmonic in range(1, 6):
        amp = 1.0 / harmonic
        shimmer = 1 + rng.normal(0, 0.05, size=t.shape)          # amplituda tebranishi
        signal += amp * shimmer * np.sin(2 * np.pi * harmonic * (f0 + jitter) * t)

    # nafas/shovqin qatlami (tabiiy ovozda doim mavjud)
    breath_noise = rng.normal(0, 0.03, size=t.shape)
    envelope = 0.6 + 0.4 * np.sin(2 * np.pi * 0.5 * t) ** 2       # tabiiy amplituda o'zgarishi
    signal = signal * envelope + breath_noise

    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.8
    return signal.astype(np.float32)


def make_fake_like(seed: int) -> np.ndarray:
    """Sun'iy TTS ovoziga xos ba'zi statistik belgilarni taqlid qiluvchi signal:
    juda barqaror pitch, nafas shovqini yo'q, notabiiy tekis amplituda."""
    rng = np.random.default_rng(seed + 10000)
    t = np.linspace(0, DURATION, int(SR * DURATION), endpoint=False)

    f0 = 150 + rng.uniform(-5, 5)   # deyarli barqaror pitch (tabiiy jitter kam)
    signal = np.zeros_like(t)
    for harmonic in range(1, 6):
        amp = 1.0 / harmonic
        signal += amp * np.sin(2 * np.pi * harmonic * f0 * t)

    # yuqori chastotali "vocoder-simon" artefakt (TTS vokoderlariga xos)
    artifact = 0.05 * np.sin(2 * np.pi * 6000 * t)
    envelope = 0.75  # deyarli o'zgarmas amplituda (notabiiy tekislik)
    signal = signal * envelope + artifact

    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.8
    return signal.astype(np.float32)


def main():
    os.makedirs(REAL_DIR, exist_ok=True)
    os.makedirs(FAKE_DIR, exist_ok=True)

    print(f"[generate] {N_SAMPLES_PER_CLASS} ta 'real' va {N_SAMPLES_PER_CLASS} ta 'fake' sinov fayli yaratilmoqda...")
    for i in range(N_SAMPLES_PER_CLASS):
        sf.write(os.path.join(REAL_DIR, f"real_{i:03d}.wav"), make_real_like(i), SR)
        sf.write(os.path.join(FAKE_DIR, f"fake_{i:03d}.wav"), make_fake_like(i), SR)

    print(f"[tayyor] Fayllar joylashgan joy:\n  {REAL_DIR}\n  {FAKE_DIR}")
    print("\nDIQQAT: bular sun'iy sinov signallari, haqiqiy ovoz namunalari emas!")
    print("Pipeline'ni tekshirib bo'lgach, ularni haqiqiy dataset bilan almashtiring.")


if __name__ == "__main__":
    main()
