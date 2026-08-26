"""
voice_output.py — Local Text-to-Speech using pyttsx3
=====================================================
Speaks text through the laptop speakers using a fully offline TTS engine.
No internet required. No API calls.
This helper is standalone and is not wired into the FastAPI handlers; the
supported application playback path is browser SpeechSynthesis.

Usage:
    from voice_output import speak_text
"""

import os
import re
import logging
import pyttsx3
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

TTS_RATE = int(os.getenv("TTS_RATE", "175"))
TTS_VOLUME = float(os.getenv("TTS_VOLUME", "0.9"))


def speak_text(text: str) -> None:
    """
    Speak the given text aloud using the system's TTS engine.

    Args:
        text: The text to speak. Empty or None text is safely ignored.
    """
    # Handle empty input gracefully
    if not text or not text.strip():
        logger.debug("Nothing to speak (empty text).")
        return

    try:
        # Initialize TTS engine
        # pyttsx3 uses the OS-level TTS (SAPI5 on Windows, espeak on Linux)
        engine = pyttsx3.init()

        # Optional: adjust speech rate (default ~200 words/min)
        engine.setProperty("rate", TTS_RATE)

        # Optional: adjust volume (0.0 to 1.0)
        engine.setProperty("volume", TTS_VOLUME)

        logger.info("Speaking: %s...", text[:80])
        engine.say(text)
        engine.runAndWait()

    except Exception as e:
        logger.error("TTS error: %s", e)

def speak_stream(text_generator):
    """
    Consumes a generator of text chunks, builds up sentences, and speaks them sequentially.
    Runs its own pyttsx3 engine loop to avoid thread issues.
    """
    try:
        import pythoncom
        pythoncom.CoInitialize()
    except ImportError:
        pass

    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", TTS_RATE)
        engine.setProperty("volume", TTS_VOLUME)
        
        tokens_list = []
        for token in text_generator:
            tokens_list.append(token)
            buffer = "".join(tokens_list)
            
            # Look for sentence boundaries or pauses (comma, colon, etc.)
            while True:
                match = re.search(r'([.?!,;:\n]+)(\s+)', buffer)
                if match:
                    idx = match.end()
                    sentence = buffer[:idx].strip()
                    tokens_list = [buffer[idx:]]
                    buffer = "".join(tokens_list)
                    
                    if len(sentence) > 1: # Avoid speaking empty/single chars
                        engine.say(sentence)
                        engine.runAndWait()
                else:
                    break
                    
        buffer = "".join(tokens_list) if locals().get('tokens_list') else ""
        # Speak any remaining buffer at the end
        if buffer.strip():
            engine.say(buffer.strip())
            engine.runAndWait()
            
    except Exception as e:
        logger.error("Streaming TTS error: %s", e)
