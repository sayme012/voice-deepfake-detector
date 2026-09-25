"""
bot.py
------
Telegram bot: foydalanuvchi ovozli xabar yoki audio fayl yuborsa,
model uni tahlil qilib "haqiqiy" yoki "deepfake" ekanini aytadi.

Ishga tushirishdan oldin:
    1. .env fayl yarating: BOT_TOKEN=your_telegram_bot_token
    2. python train.py orqali modelni tayyorlang (models/final_model.keras paydo bo'lishi kerak)

Ishga tushirish:
    python bot.py
"""

import os
import asyncio
import logging
import tempfile

from aiogram import Bot, Dispatcher, F
from aiogram.types import Message
from aiogram.filters import CommandStart
from dotenv import load_dotenv

from infer import predict

load_dotenv()
BOT_TOKEN = os.getenv("BOT_TOKEN")

logging.basicConfig(level=logging.INFO)
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


@dp.message(CommandStart())
async def start_handler(message: Message):
    await message.answer(
        "🎙 <b>Voice Deepfake Detector</b>\n\n"
        "Menga ovozli xabar (voice) yoki audio fayl yuboring — "
        "men uni tahlil qilib, bu haqiqiy odam ovozimi yoki "
        "sun'iy (AI bilan yasalgan) ovozmi ekanini aytib beraman.\n\n"
        "⚠️ Bu — o'quv/kurs ishi loyihasi, natijalarni 100% ishonchli deb qabul qilmang.",
        parse_mode="HTML",
    )


async def _handle_audio(message: Message, file_id: str):
    status = await message.answer("🔎 Tahlil qilinmoqda...")

    with tempfile.TemporaryDirectory() as tmp:
        local_path = os.path.join(tmp, "audio.ogg")
        file = await bot.get_file(file_id)
        await bot.download_file(file.file_path, destination=local_path)

        try:
            result = predict(local_path)
        except Exception as e:
            await status.edit_text(f"❌ Xatolik: {e}")
            return

    emoji = "🤖" if "DEEPFAKE" in result["label"] else "✅"
    text = (
        f"{emoji} <b>Natija:</b> {result['label']}\n"
        f"📊 Deepfake ehtimoli: <b>{result['deepfake_probability']}%</b>\n"
        f"🎯 Ishonch darajasi: <b>{result['confidence']}%</b>"
    )
    await status.edit_text(text, parse_mode="HTML")


@dp.message(F.voice)
async def voice_handler(message: Message):
    await _handle_audio(message, message.voice.file_id)


@dp.message(F.audio)
async def audio_handler(message: Message):
    await _handle_audio(message, message.audio.file_id)


async def main():
    if not BOT_TOKEN:
        raise RuntimeError("BOT_TOKEN topilmadi — .env faylga BOT_TOKEN=... qo'shing")
    print("Bot ishga tushdi...")
    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
