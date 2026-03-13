# XTTS basic live reader
#
# conda activate tortoise
# cd "C:\Users\gauth\OneDrive\Desktop\GitHub\CART498-GenAI\TTS"
# pip install TTS sounddevice
# python tts_cli_player_basic.py

from pathlib import Path
import time

import numpy as np
import sounddevice as sd
import torch
from TTS.api import TTS

MODEL_NAME = "tts_models/multilingual/multi-dataset/xtts_v2"
LANGUAGE = "en"

VOICE_DIR = Path(r"C:\Users\chief\OneDrive\Documents\GitHub\tortoise-tts\tortoise\voices")
NARRATOR_FILES = [
    VOICE_DIR / "train_dotrice" / "1.wav",
    VOICE_DIR / "train_dotrice" / "2.wav",
]

# XTTS defaults in this repo are quite "creative" (high temperature, low repetition_penalty).
# These settings tend to be more stable and keep short prompts (like "Hello") short and coherent.
INFERENCE_KWARGS = {
    "temperature": 0.25,
    "top_p": 0.85,
    "top_k": 50,
    "do_sample": False,
    "repetition_penalty": 10.0,
    "length_penalty": 1.0,
    "max_new_tokens": 40,
}


def log(message: str) -> None:
    timestamp = time.strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def resolve_files(paths):
    existing = [str(p) for p in paths if p.exists()]
    if not existing:
        raise FileNotFoundError("No narrator speaker wav files found. Update NARRATOR_FILES.")
    return existing


def try_get_model_device(tts: TTS) -> str:
    try:
        model = getattr(getattr(tts, "synthesizer", None), "tts_model", None)
        if model is None:
            return "unknown"
        params = list(model.parameters())
        if not params:
            return "unknown"
        return str(params[0].device)
    except Exception:
        return "unknown"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    log(f"Python Torch: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    log(f"Device: {device}")
    if torch.cuda.is_available():
        try:
            log(f"CUDA device count: {torch.cuda.device_count()}")
            log(f"CUDA device 0: {torch.cuda.get_device_name(0)}")
        except Exception as exc:
            log(f"CUDA device query failed: {exc}")

    log(f"Loading model: {MODEL_NAME}")
    load_start = time.perf_counter()
    tts = TTS(MODEL_NAME, progress_bar=True).to(device)
    load_s = time.perf_counter() - load_start
    log(f"Model loaded in {load_s:.1f}s (model device: {try_get_model_device(tts)})")
    sample_rate = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", 24000)
    log(f"Output sample rate: {sample_rate}")
    speaker_wavs = resolve_files(NARRATOR_FILES)
    log(f"Inference settings: {INFERENCE_KWARGS}")

    log("Type text and press Enter. Type 'quit' to exit.")
    while True:
        try:
            text = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            log("Exiting.")
            break
        if not text:
            continue
        if text.lower() in {"quit", "exit"}:
            log("Exiting.")
            break

        log("Generating...")
        gen_start = time.perf_counter()
        wav = tts.tts(text=text, speaker_wav=speaker_wavs, language=LANGUAGE, **INFERENCE_KWARGS)
        gen_s = time.perf_counter() - gen_start
        wav_np = np.asarray(wav, dtype=np.float32)
        log(f"Generated in {gen_s:.1f}s; audio {wav_np.size/sample_rate:.2f}s; playing...")
        sd.play(wav_np, samplerate=sample_rate)
        sd.wait()
        log("Done.")


if __name__ == "__main__":
    main()
