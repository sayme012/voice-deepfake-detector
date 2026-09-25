# Voice Deepfake Detector

Ovozli xabar/faylni tahlil qilib, bu **haqiqiy inson ovozi**mi yoki **sun'iy
(AI/TTS/voice-clone bilan yasalgan) ovoz**mi ekanini aniqlaydigan tizim.
Mel-spektrogramma + CNN yondashuvi asosida qurilgan, natijalar Telegram bot
orqali ham olinadi.

## Loyiha tuzilishi

```
voice-deepfake-detector/
├── data/
│   ├── real/              # haqiqiy ovoz fayllari (.wav)
│   └── fake/               # deepfake/sun'iy ovoz fayllari (.wav)
├── models/                 # o'qitilgan modellar shu yerga saqlanadi
├── src/
│   ├── features.py         # audio -> mel-spektrogramma / MFCC
│   ├── dataset.py           # train/val/test to'plamlarini tayyorlash
│   ├── model.py              # CNN arxitekturasi
│   ├── train.py               # modelni o'qitish
│   ├── evaluate.py            # EER, AUC, confusion matrix
│   ├── infer.py                # bitta fayl uchun bashorat
│   ├── bot.py                   # Telegram bot integratsiyasi
│   └── generate_dummy_data.py   # pipeline'ni sinash uchun SINTETIK data
└── requirements.txt
```

## 1-qadam: O'rnatish

```bash
pip install -r requirements.txt
```

## 2-qadam: Ma'lumotlar (eng muhim qadam!)

Bu loyihada kod **to'liq tayyor va sinovdan o'tgan**, lekin haqiqiy o'qitish
uchun sizga **haqiqiy dataset** kerak. Ikki variant bor:

### A) Tez sinov uchun (kod ishlayotganini tekshirish)

```bash
cd src
python generate_dummy_data.py
```

Bu sun'iy signal generatsiya qiladi (haqiqiy ovoz emas!) — faqat pipeline
xatosiz ishlashini ko'rish uchun. **Bu bilan o'qitilgan model real hayotda
ishlamaydi.**

### B) Haqiqiy loyiha uchun (tavsiya etiladi)

Quyidagi ochiq datasetlardan birini yuklab oling va fayllarni
`data/real/` va `data/fake/` papkalariga joylashtiring:

| Dataset | Havola | Izoh |
|---|---|---|
| ASVspoof 2019 LA | datashare.ed.ac.uk/handle/10283/3336 | Soha standarti, ~120k fayl |
| Fake-or-Real (FoR) | Kaggle: `mohammedabdeldayem/the-fake-or-real-dataset` | Ishlatish oson, kichikroq |
| WaveFake | github.com/RUB-SysSec/WaveFake | 6 xil TTS modeldan fake namunalar |

Yoki o'zingiz kichik dataset yarating:
- **Real**: o'zingiz yoki do'stlaringiz ovozini diktafon/telefon bilan yozib oling (kamida 100-200 ta qisqa, 3-5 soniyalik namuna)
- **Fake**: bir xil matnlarni ElevenLabs, Coqui TTS yoki boshqa voice-clone xizmati orqali sintez qiling

## 3-qadam: Modelni o'qitish

```bash
cd src
python train.py
```

Bu quyidagilarni bajaradi:
1. `data/real` va `data/fake` dan barcha audio fayllarni o'qiydi
2. Har birini mel-spektrogrammaga aylantiradi (128×250 "rasm")
3. Train/val/test (70/15/15) qismlarga bo'ladi
4. CNN modelni o'qitadi (EarlyStopping bilan — ortiqcha o'qishning oldini oladi)
5. `models/final_model.keras` va `models/best_model.keras` saqlaydi

## 4-qadam: Baholash

```bash
python evaluate.py
```

Chiqadigan asosiy ko'rsatkich — **EER (Equal Error Rate)**: sohada standart
metrik, qancha past bo'lsa shuncha yaxshi. ASVspoof kabi real datasetda
yaxshi CNN model odatda 3-8% EER ga erishadi.

## 5-qadam: Bitta faylni tekshirish

```bash
python infer.py path/to/audio.wav
```

## 6-qadam: Telegram bot sifatida ishga tushirish

```bash
echo "BOT_TOKEN=sizning_tokeningiz" > .env
python bot.py
```

Botga ovozli xabar yuborsangiz, u avtomatik yuklab oladi, tahlil qiladi va
natijani qaytaradi.

## Nazariy asos (qisqacha)

1. **Mel-spektrogramma**: audio to'lqinni chastota-vaqt "rasmi"ga aylantiradi
2. **CNN**: rasmdagi patternlarni (notabiiy formant o'tishlari, yuqori
   chastota artefaktlari, notekis prosodiya) avtomatik o'rganadi
3. **EER**: False Accept va False Reject darajalari teng bo'lgan nuqtadagi
   xato foizi — modelning haqiqiy sifatini ko'rsatadi (oddiy accuracy'dan
   ko'ra ishonchliroq, chunki class balansiga kam bog'liq)

## Keyingi qadamlar (loyihani kengaytirish uchun g'oyalar)

- Data augmentation: fon shovqini, turli codec siqishlar qo'shish (real
  hayotda audio ko'pincha siqilgan bo'ladi — WhatsApp, Telegram va h.k.)
- Transfer learning: Wav2Vec2 yoki YAMNet kabi tayyor audio-embedding
  modellardan foydalanish (kichik datasetda ham yaxshi natija beradi)
- Ansambl: CNN (spektrogramma) + klassik ML (MFCC statistikasi) natijalarini
  birlashtirish
- Real-time: uzluksiz audio oqimni kichik oynalarga bo'lib tahlil qilish
