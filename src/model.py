"""
model.py
--------
Mel-spektrogramma "rasm"lari ustida ishlaydigan CNN arxitekturasi.

Nazariy asos:
- Spektrogramma vaqt (x-o'q) va chastota (y-o'q) o'lchamlariga ega 2D matritsa,
  shuning uchun oddiy rasm-klassifikatsiya CNN'lari (Conv2D + MaxPooling)
  to'g'ridan-to'g'ri qo'llanadi.
- Bir nechta Conv2D qatlami past darajadagi patternlardan (chekka/artifakt)
  yuqori darajadagi patternlarga (masalan "notabiiy formant o'tishlari")
  o'tishni o'rganadi.
- Oxirida Dense qatlam ikkilik klassifikatsiya qiladi: 0 = haqiqiy, 1 = deepfake.
"""

from tensorflow import keras
from tensorflow.keras import layers


def build_cnn(input_shape=(128, 251, 1)) -> keras.Model:
    model = keras.Sequential([
        layers.Input(shape=input_shape),

        layers.Conv2D(16, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(1, activation="sigmoid"),   # 0..1 -> deepfake ehtimoli
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


if __name__ == "__main__":
    m = build_cnn()
    m.summary()
