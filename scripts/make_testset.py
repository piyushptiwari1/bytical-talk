"""Generate a multi-voice unseen-voice test set for the HuBERT-vs-AVE generalization
experiment. Diverse accents/genders (US/GB/IN, M/F) that the model never trained on
(training audio was a single Indian-English voice). 16 kHz mono WAV via ffmpeg."""
import asyncio, os
import edge_tts

OUT = os.path.expanduser("~/synctalk_testset")
os.makedirs(OUT, exist_ok=True)

VOICES = {
    "us_m":  "en-US-GuyNeural",
    "us_f":  "en-US-JennyNeural",
    "gb_m":  "en-GB-RyanNeural",
    "in_f":  "en-IN-NeerjaNeural",   # closest to training accent (control)
    "in_m":  "en-IN-PrabhatNeural",
}

SCRIPTS = {
    "s1": ("Thank you for choosing us. We understand that protecting your family "
           "is your top priority, and we're here to help every step of the way."),
    "s2": ("Let me walk you through the claim process. It only takes a few minutes, "
           "and our team will handle the rest so you can focus on what matters."),
}

async def synth(voice, text, mp3_path):
    await edge_tts.Communicate(text, voice).save(mp3_path)

async def main():
    made = []
    for sk, text in SCRIPTS.items():
        for vk, voice in VOICES.items():
            base = f"{vk}_{sk}"
            mp3 = os.path.join(OUT, base + ".mp3")
            await synth(voice, text, mp3)
            made.append(mp3)
    print(f"MADE {len(made)} mp3s in {OUT}")
    for w in made:
        print(" ", os.path.basename(w))

asyncio.run(main())
