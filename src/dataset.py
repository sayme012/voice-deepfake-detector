"""
dataset.py
----------
data/real/ va data/fake/ papkalaridagi audio fayllardan CNN uchun
train/val/test to'plamlarini tayyorlaydi.

Kutilayotgan papka strukturasi:
    data/
      real/   -> haqiqiy inson ovozi fayllari (.wav)
      fake/   -> sun'iy/klonlangan ovoz fayllari (.wav)

ASVspoof yoki boshqa dataset ishlatsangiz, shunchaki fayllarni shu ikki
papkaga (label bo'yicha) joylashtirsangiz kifoya — kod qolganini avtomatik
bajaradi.
"""

import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split

from features import extract_mel_spectrogram

REAL_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "real")
FAKE_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "fake")

LABEL_REAL = 0   # 0 = haqiqiy ovoz
LABEL_FAKE = 1   # 1 = deepfake/sun'iy ovoz


def _list_audio(folder: str):
    exts = ("*.wav", "*.flac", "*.mp3", "*.ogg")
    files = []
    for e in exts:
        files.extend(glob.glob(os.path.join(folder, e)))
    return sorted(files)


def build_dataset(test_size: float = 0.15, val_size: float = 0.15, seed: int = 42):
    """
    Barcha audio fayllarni o'qib, mel-spektrogramma xususiyatlariga aylantiradi
    va train/val/test qismlarga bo'ladi (stratified, ya'ni ikkala klass ham
    har uch to'plamda proporsional taqsimlanadi).
    """
    real_files = _list_audio(REAL_DIR)
    fake_files = _list_audio(FAKE_DIR)

    if len(real_files) == 0 or len(fake_files) == 0:
        raise RuntimeError(
            f"Audio fayllar topilmadi!\n"
            f"  real/: {len(real_files)} ta fayl ({REAL_DIR})\n"
            f"  fake/: {len(fake_files)} ta fayl ({FAKE_DIR})\n"
            f"Iltimos, .wav fayllarni shu papkalarga joylashtiring "
            f"(yoki avval generate_dummy_data.py skriptini ishga tushiring)."
        )

    paths = real_files + fake_files
    labels = [LABEL_REAL] * len(real_files) + [LABEL_FAKE] * len(fake_files)

    print(f"[dataset] Real: {len(real_files)} ta, Fake: {len(fake_files)} ta, Jami: {len(paths)} ta fayl")
    print("[dataset] Mel-spektrogrammalar hisoblanmoqda...")

    X, y = [], []
    for i, (p, lab) in enumerate(zip(paths, labels)):
        try:
            feat = extract_mel_spectrogram(p)
            X.append(feat)
            y.append(lab)
        except Exception as e:
            print(f"  [ogohlantirish] {p} o'qib bo'lmadi: {e}")
        if (i + 1) % 50 == 0:
            print(f"  ...{i + 1}/{len(paths)}")

    X = np.stack(X)
    y = np.array(y, dtype=np.int32)

    # avval test'ni ajratamiz, keyin qolganini train/val ga bo'lamiz
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_ratio, stratify=y_train, random_state=seed
    )

    print(f"[dataset] Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


if __name__ == "__main__":
    build_dataset()
