"""
voice_input.py — Local Speech-to-Text using faster-whisper
==========================================================
Records audio from the microphone and transcribes it locally.
The Whisper model is loaded ONCE at startup, not per request.

Usage:
    from voice_input import transcribe_audio, listen_and_transcribe
"""

import os
import uuid
import logging
from dotenv import load_dotenv

# ─── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()
logger = logging.getLogger(__name__)

# Whisper model size: "tiny", "base", "small", "medium", "large-v2"
# Smaller = faster but less accurate. "base" is a good balance for CPU.
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "base")

# Audio recording settings
SAMPLE_RATE = 16000  # 16kHz is required by Whisper
TEMP_AUDIO_DIR = os.path.join(os.path.dirname(__file__), "temp_audio")

# ─── Global Whisper Model Instance ─────────────────────────────────────────────
_whisper_model = None


def _get_whisper_model():
    """
    Load the Whisper model once and cache it globally.
    Downloads automatically on first use (~150MB for 'base').
    After first download, runs fully offline.
    """
    global _whisper_model

    if _whisper_model is not None:
        return _whisper_model

    try:
        from faster_whisper import WhisperModel
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "faster-whisper is not installed. "
            "Install backend/requirements.txt to enable speech-to-text."
        ) from exc

    logger.info("Loading Whisper model: %s", WHISPER_MODEL_SIZE)
    logger.info("First run will download the model, then it works offline")

    _whisper_model = WhisperModel(
        WHISPER_MODEL_SIZE,
        device="cpu",
        compute_type="int8",  # Use int8 quantization for faster CPU inference
    )

    logger.info("Whisper model loaded successfully!")
    return _whisper_model


def record_audio(duration: int = 5, sample_rate: int = SAMPLE_RATE,
                 output_path: str = None) -> str:
    """
    Record audio from the default microphone.

    Args:
        duration: Recording duration in seconds.
        sample_rate: Audio sample rate (16000 for Whisper).
        output_path: Path to save the WAV file. Defaults to temp_audio/input.wav.

    Returns:
        Path to the saved WAV file.
    """
    try:
        import sounddevice as sd
        from scipy.io import wavfile
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "sounddevice and scipy are required for microphone recording. "
            "Install backend/requirements.txt to enable this endpoint."
        ) from exc

    # Ensure temp directory exists
    os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)

    import tempfile
    if output_path is None:
        fd, output_path = tempfile.mkstemp(prefix="input_", suffix=".wav", dir=TEMP_AUDIO_DIR)
        os.close(fd)

    logger.info("Recording for %d seconds...", duration)

    # Record audio from microphone
    audio_data = sd.rec(
        int(duration * sample_rate),
        samplerate=sample_rate,
        channels=1,        # Mono audio
        dtype="int16",
    )
    sd.wait()  # Wait until recording is complete

    # Save as WAV file
    wavfile.write(output_path, sample_rate, audio_data)
    logger.info("Audio saved to: %s", output_path)

    return output_path


def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe an audio file using faster-whisper (runs locally).

    Args:
        audio_path: Path to the audio file (WAV, MP3, etc.).

    Returns:
        Transcribed text as a string.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    model = _get_whisper_model()

    # Transcribe the audio
    segments, info = model.transcribe(audio_path, beam_size=5)

    # Combine all segments into one text string
    transcribed_text = " ".join(segment.text.strip() for segment in segments)

    logger.info("Detected language: %s (probability: %.2f)", info.language, info.language_probability)
    logger.info("Transcribed: %s...", transcribed_text[:100])

    return transcribed_text


def listen_and_transcribe(duration: int = 5) -> str:
    """
    Convenience function: record from microphone and transcribe.

    Args:
        duration: Recording duration in seconds.

    Returns:
        Transcribed text.
    """
    audio_path = record_audio(duration=duration)
    try:
        text = transcribe_audio(audio_path)
    finally:
        # Clean up temp file
        try:
            os.remove(audio_path)
        except OSError:
            pass

    return text
