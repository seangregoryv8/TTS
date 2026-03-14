# XTTS basic live reader
#
# conda activate tortoise
# cd "C:\Users\gauth\OneDrive\Desktop\GitHub\CART498-GenAI\TTS"
# pip install TTS sounddevice
# python tts_cli_player_basic.py

import time
from pathlib import Path

import numpy as np
import sounddevice as sd
import torch
from TTS.api import TTS

from tts_cli_config import (
    LANGUAGE,
    MODEL_NAME,
    NARRATOR_FILES,
    get_output_sample_rate,
    resolve_existing_files,
)


def resolve_files(paths):
    return resolve_existing_files(paths)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Device: {device}")

    tts = TTS(MODEL_NAME, progress_bar=False).to(device)
    sample_rate = get_output_sample_rate(tts, default=24000)
    print(f"Output sample rate: {sample_rate}")
    speaker_wavs = resolve_files(NARRATOR_FILES)

    print("Type text and press Enter. Type 'quit' to exit.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            print("Exiting.")
            break

        print("Generating...")
        wav = tts.tts(text=text, speaker_wav=speaker_wavs, language=LANGUAGE)
        wav_np = np.asarray(wav, dtype=np.float32)
        out_dir = Path("outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"basicv2_{time.strftime('%Y%m%d_%H%M%S')}.wav"
        if getattr(tts, "synthesizer", None) is not None:
            tts.synthesizer.save_wav(wav=wav_np, path=str(out_path))
        print(f"Saved: {out_path}")
        sd.play(wav_np, samplerate=sample_rate)
        sd.wait()
        print("Done.")


if __name__ == "__main__":
    main()
