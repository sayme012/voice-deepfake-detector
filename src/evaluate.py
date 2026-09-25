"""
evaluate.py
-----------
Modelni test to'plamida baholaydi va soha standarti bo'lgan
EER (Equal Error Rate) ko'rsatkichini hisoblaydi.

EER nima?
    - False Acceptance Rate (FAR): deepfake'ni "haqiqiy" deb noto'g'ri qabul qilish darajasi
    - False Rejection Rate (FRR): haqiqiy ovozni "deepfake" deb noto'g'ri rad etish darajasi
    - Bu ikkisi teng bo'lgan nuqtadagi xato darajasi = EER
    - EER qancha past bo'lsa, model shuncha yaxshi (0% = ideal, 50% = tasodifiy taxmin)
"""

import os
import numpy as np
from tensorflow import keras
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, classification_report

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")


def compute_eer(y_true, y_scores):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    fnr = 1 - tpr
    # FAR (fpr) va FRR (fnr) eng yaqin kesishgan nuqtani topamiz
    idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[idx] + fnr[idx]) / 2
    eer_threshold = thresholds[idx]
    return eer, eer_threshold


def main():
    model_path = os.path.join(MODELS_DIR, "final_model.keras")
    split_path = os.path.join(MODELS_DIR, "test_split.npz")

    if not os.path.exists(model_path) or not os.path.exists(split_path):
        raise RuntimeError("Avval train.py ni ishga tushiring — model va test split topilmadi.")

    model = keras.models.load_model(model_path)
    data = np.load(split_path)
    X_test, y_test = data["X_test"], data["y_test"]

    y_scores = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_scores >= 0.5).astype(int)

    auc = roc_auc_score(y_test, y_scores)
    eer, eer_thr = compute_eer(y_test, y_scores)

    print("=" * 50)
    print(f"AUC-ROC        : {auc:.4f}")
    print(f"EER            : {eer * 100:.2f}%  (chegara={eer_thr:.3f})")
    print("=" * 50)
    print("Confusion matrix (qator=haqiqiy, ustun=bashorat) [0=real, 1=fake]:")
    print(confusion_matrix(y_test, y_pred))
    print()
    print(classification_report(y_test, y_pred, target_names=["real", "fake"]))


if __name__ == "__main__":
    main()
