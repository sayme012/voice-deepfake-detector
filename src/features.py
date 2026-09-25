"""
features.py
------------
Audio fayllardan CNN uchun mel-spektrogramma xususiyatlarini ajratib olish.

Nazariy asos:
- Ovozni vaqt domenidan chastota domeniga o'tkazish uchun Short-Time Fourier
  Transform (STFT) ishlatiladi.
- Mel shkalasi inson qulog'i chastotani qanday qabul qilishini taqlid qiladi
  (past chastotalarda aniqroq, yuqori chastotalarda kamroq aniq) — shu sababli
  nutq/ovoz tahlilida standart hisoblanadi.
- Natijaviy mel-spektrogramma 2D "rasm" ko'rinishida bo'lib, uni to'g'ridan-to'g'ri
  CNN ga rasm sifatida berish mumkin.
"""

import numpy as np
import librosa

SAMPLE_RATE = 16000       # ASVspoof va aksariyat nutq datasetlari 16kHz ishlatadi
DURATION = 4.0            # har bir namunani 4 soniyagacha kesamiz/to'ldiramiz
N_MELS = 128               # mel-band'lar soni (spektrogramma balandligi)
N_FFT = 1024               # FFT oynasi kattaligi
HOP_LENGTH = 256           # oynalar orasidagi qadam
TARGET_FRAMES = int(np.ceil(SAMPLE_RATE * DURATION / HOP_LENGTH))  # spektrogramma kengligi


def load_audio(path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Audio faylni yuklaydi, mono va berilgan sample rate'ga o'tkazadi."""
    y, _ = librosa.load(path, sr=sr, mono=True)
    return y


def fix_length(y: np.ndarray, sr: int = SAMPLE_RATE, duration: float = DURATION) -> np.ndarray:
    """Audio uzunligini belgilangan davomiylikka moslaydi (kesish yoki nol bilan to'ldirish)."""
    target_len = int(sr * duration)
    if len(y) > target_len:
        # markazdan kesib olamiz (boshlanishi/oxiri sukut bo'lishi mumkin)
        start = (len(y) - target_len) // 2
        y = y[start:start + target_len]
    elif len(y) < target_len:
        pad = target_len - len(y)
        y = np.pad(y, (pad // 2, pad - pad // 2), mode="constant")
    return y


def extract_mel_spectrogram(path: str) -> np.ndarray:
    """
    Audio fayldan log-mel-spektrogramma qaytaradi, shape=(N_MELS, TARGET_FRAMES, 1).
    CNN uchun tayyor "rasm" formatida.
    """
    y = load_audio(path)
    y = fix_length(y)

    mel = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
    )
    log_mel = librosa.power_to_db(mel, ref=np.max)

    # kenglikni TARGET_FRAMES ga moslaymiz (kichik farqlar bo'lishi mumkin)
    if log_mel.shape[1] < TARGET_FRAMES:
        pad = TARGET_FRAMES - log_mel.shape[1]
        log_mel = np.pad(log_mel, ((0, 0), (0, pad)), mode="constant", constant_values=log_mel.min())
    else:
        log_mel = log_mel[:, :TARGET_FRAMES]

    # -1..1 oralig'iga normallashtirish (CNN o'qitish uchun barqaror)
    log_mel = (log_mel - log_mel.mean()) / (log_mel.std() + 1e-8)

    return log_mel[..., np.newaxis].astype(np.float32)


def extract_classic_features(path: str) -> np.ndarray:
    """
    Oddiy (klassik ML: Random Forest/SVM) model uchun qisqa xususiyatlar vektori:
    MFCC statistikasi + pitch + spectral xususiyatlar.
    Bu 'sodda versiya'ni tez sinash uchun ham foydali.
    """
    y = load_audio(path)
    y = fix_length(y)

    mfcc = librosa.feature.mfcc(y=y, sr=SAMPLE_RATE, n_mfcc=20)
    spec_centroid = librosa.feature.spectral_centroid(y=y, sr=SAMPLE_RATE)
    spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=SAMPLE_RATE)
    zcr = librosa.feature.zero_crossing_rate(y)

    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C7"), sr=SAMPLE_RATE
    )
    f0 = f0[~np.isnan(f0)]
    pitch_mean = float(np.mean(f0)) if len(f0) else 0.0
    pitch_std = float(np.std(f0)) if len(f0) else 0.0

    feats = np.concatenate([
        mfcc.mean(axis=1), mfcc.std(axis=1),
        spec_centroid.mean(axis=1), spec_bandwidth.mean(axis=1),
        zcr.mean(axis=1),
        [pitch_mean, pitch_std],
    ])
    return feats.astype(np.float32)
