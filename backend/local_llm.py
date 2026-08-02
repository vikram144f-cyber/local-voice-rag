"""
local_llm.py — Local LLM using llama-cpp-python
================================================
Loads a GGUF model from disk and provides generate/stream functions.
The model is loaded ONCE at startup, not on every request.

Usage:
    from local_llm import generate_response, stream_response
"""

import os
import logging
from dotenv import load_dotenv
from llama_cpp import Llama

# ─── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()
logger = logging.getLogger(__name__)

# Place your .gguf model file in the models/ folder.
# Update this path if your file has a different name.
_model_filename = os.getenv("MODEL_FILENAME", "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", _model_filename)

# Context window size (how many tokens the model can see at once)
N_CTX = int(os.getenv("N_CTX", "4096"))

# Number of CPU threads to use for inference (use most cores for speed)
N_THREADS = max(1, (os.cpu_count() or 4) - 1)

# Max tokens to generate per response
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "256"))

# Sampling parameters
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.1"))
LLM_TOP_K = int(os.getenv("LLM_TOP_K", "40"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.95"))
LLM_REPEAT_PENALTY = float(os.getenv("LLM_REPEAT_PENALTY", "1.1"))

# Stop tokens — Llama 3.1 uses <|eot_id|> to signal end of response
STOP_TOKENS = ["<|eot_id|>", "<|end_of_text|>", "</s>", "<|end|>", "<|endoftext|>"]

# ─── Global Model Instance ────────────────────────────────────────────────────
# Loaded once when this module is first imported
_llm = None


def load_llm():
    """
    Initialize the Llama model from the GGUF file.
    Called automatically on first use. Returns the model instance.
    """
    global _llm

    if _llm is not None:
        return _llm

    # Check if model file exists
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"GGUF model not found at: {MODEL_PATH}\n"
            f"Please download a GGUF model and place it in the 'models/' folder.\n"
            f"Example: models/phi-3-mini-4k-instruct-q4.gguf"
        )

    logger.info("Loading model from: %s", MODEL_PATH)
    logger.info("Using %d CPU threads, context window: %d", N_THREADS, N_CTX)

    _llm = Llama(
        model_path=MODEL_PATH,
        n_ctx=N_CTX,
        n_threads=N_THREADS,
        n_gpu_layers=-1,      # Offload to GPU if supported by the local library (-1 means all layers)
        n_batch=512,          # Faster prompt processing
        use_mmap=True,
        use_mlock=False,
        verbose=False,        # Set to True for debug output
    )

    logger.info("Model loaded successfully!")
    return _llm


def _wrap_prompt(prompt: str) -> str:
    """
    Wrap a raw prompt in the correct chat format (Llama 3 or Mistral).
    This tells the model exactly where the system/user turn ends
    and where the assistant turn begins, so it stops properly.
    """
    if "mistral" in MODEL_PATH.lower():
        return f"[INST] {prompt} [/INST]"
        
    return (
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "You are a helpful AI assistant. Answer concisely and accurately. "
        "When you are done answering, stop immediately.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
    )


def generate_response(prompt: str, max_tokens: int = MAX_TOKENS, temperature: float = LLM_TEMPERATURE) -> str:
    """
    Generate a complete response from the local LLM (non-streaming).

    Args:
        prompt: The full prompt string (including context and question).
        max_tokens: Maximum number of tokens to generate.
        temperature: Controls randomness (0.0 = deterministic, 1.0 = creative).

    Returns:
        The generated text as a string.
    """
    llm = load_llm()
    formatted = _wrap_prompt(prompt)

    result = llm(
        formatted,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=LLM_TOP_K,
        top_p=LLM_TOP_P,
        repeat_penalty=LLM_REPEAT_PENALTY,
        stop=STOP_TOKENS,
    )

    return result["choices"][0]["text"]


def stream_response(prompt: str, max_tokens: int = MAX_TOKENS, temperature: float = LLM_TEMPERATURE):
    """
    Stream response tokens from the local LLM one at a time.
    Yields plain text chunks compatible with FastAPI StreamingResponse.

    Args:
        prompt: The full prompt string (including context and question).
        max_tokens: Maximum number of tokens to generate.
        temperature: Controls randomness.

    Yields:
        Individual token strings as they are generated.
    """
    llm = load_llm()
    formatted = _wrap_prompt(prompt)

    stream = llm(
        formatted,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=LLM_TOP_K,
        top_p=LLM_TOP_P,
        repeat_penalty=LLM_REPEAT_PENALTY,
        stop=STOP_TOKENS,
        stream=True,
    )

    for chunk in stream:
        token_text = chunk["choices"][0]["text"]
        if token_text:
            yield token_text
