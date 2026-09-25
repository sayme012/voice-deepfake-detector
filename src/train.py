"""
train.py
--------
Modelni data/real va data/fake papkalaridagi audio fayllar asosida o'qitadi.

Ishga tushirish:
    cd src
    python train.py
"""

import os
import numpy as np
from tensorflow import keras

from dataset import build_dataset
from model import build_cnn
from features import N_MELS, TARGET_FRAMES

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def main(epochs: int = 25, batch_size: int = 16):
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = build_dataset()

    model = build_cnn(input_shape=(N_MELS, TARGET_FRAMES, 1))
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=6, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            os.path.join(MODELS_DIR, "best_model.keras"),
            monitor="val_auc", mode="max", save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    # test to'plamida yakuniy baholash
    test_loss, test_acc, test_auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[natija] Test accuracy: {test_acc:.4f} | Test AUC: {test_auc:.4f}")

    np.savez(os.path.join(MODELS_DIR, "test_split.npz"), X_test=X_test, y_test=y_test)
    model.save(os.path.join(MODELS_DIR, "final_model.keras"))
    print(f"[saqlandi] {MODELS_DIR}/final_model.keras va best_model.keras")

    return history


if __name__ == "__main__":
    main()
