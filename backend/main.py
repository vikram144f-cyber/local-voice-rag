"""
main.py — FastAPI Backend for Local RAG Assistant
==================================================
Endpoints:
  GET  /api/files              — List uploaded files
  POST /api/upload             — Upload a PDF
  DELETE /api/files/{filename} — Delete a file and rebuild index
  POST /api/chat               — Text-based RAG chat (streaming)
  POST /api/rebuild            — Rebuild vectorstore (after embedding model change)
  POST /api/voice/transcribe   — Record + transcribe audio on backend machine
  POST /api/voice/upload-audio — Receive audio from browser, transcribe, run RAG
  POST /api/voice/chat         — Record on backend, transcribe, RAG, speak answer

No Ollama. No cloud APIs. Everything runs locally.
"""

import os
import json
import logging
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional

load_dotenv()
logger = logging.getLogger(__name__)

# ─── Configuration ─────────────────────────────────────────────────────────────
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:5173").split(",")
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))

from rag_core import (
    process_upload,
    delete_file_and_rebuild,
    rebuild_all_files,
    load_registry,
    load_vectorstore,
    retrieve_context,
    build_prompt,
    get_embeddings,
    UPLOADS_DIR
)
from local_llm import stream_response, load_llm
from voice_input import transcribe_audio, listen_and_transcribe, TEMP_AUDIO_DIR


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Warm embeddings, vectorstore, and LLM on startup."""
    logger.info("Warming embeddings and vectorstore...")
    get_embeddings()
    load_vectorstore()
    try:
        load_llm()
        logger.info("LLM ready.")
    except FileNotFoundError as e:
        logger.warning("LLM not loaded: %s", e)
    yield


# ─── App Setup ─────────────────────────────────────────────────────────────────
app = FastAPI(title="Local RAG Voice Assistant API", lifespan=lifespan)

# Allow React frontend to communicate with FastAPI
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure temp_audio directory exists for voice features
os.makedirs(TEMP_AUDIO_DIR, exist_ok=True)


@app.get("/health")
def health_check():
    """Health check endpoint for monitoring and uptime verification."""
    return {"status": "ok"}


def _build_sources_payload(context_docs: list) -> list:
    sources_data = []
    for doc in context_docs:
        page = doc.metadata.get("page", "?")
        sources_data.append({
            "source_file": doc.metadata.get("source_file", "unknown"),
            "page": page + 1 if isinstance(page, int) else page,
            "snippet": doc.page_content[:200].replace("\n", " ") + "..."
        })
    return sources_data


def _stream_answer(prompt: str):
    """Yield status marker then LLM tokens."""
    yield "__STATUS__generating__\n"  # parsed as __STATUS__<name>__
    try:
        for token in stream_response(prompt):
            yield token
    except Exception as e:
        logger.error("LLM streaming error: %s", e)
        yield f"\n\n**Error:** {str(e)}"


# ─── Request Models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    query: str
    model: Optional[str] = None  # Ignored — using local GGUF model
    file_filter: str = "All Files"
    top_k: int = 5
    messages: list = []  # Chat history (for context if needed)


# ─── File Management Endpoints ────────────────────────────────────────────────

@app.get("/api/files")
def list_files():
    """List all uploaded PDF files."""
    return {"files": load_registry()}


@app.post("/api/upload")
def upload_file(file: UploadFile = File(...)):
    """Upload a PDF file, chunk it, embed it, and add to FAISS."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Sanitize filename to prevent path traversal attacks
    safe_name = os.path.basename(file.filename)
    if not safe_name:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    file_path = os.path.join(UPLOADS_DIR, safe_name)
    import shutil
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
        
    if os.path.getsize(file_path) > MAX_UPLOAD_SIZE_MB * 1024 * 1024:
        os.remove(file_path)
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_UPLOAD_SIZE_MB}MB.",
        )

    try:
        chunks = process_upload(file_path, safe_name)
        return {"message": "File processed successfully", "chunks": chunks, "filename": safe_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/files/{filename}")
def delete_file(filename: str):
    """Delete a file and rebuild the FAISS index from remaining files."""
    # Sanitize filename to prevent path traversal attacks
    safe_name = os.path.basename(filename)
    if not safe_name or safe_name != filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    try:
        delete_file_and_rebuild(safe_name)
        return {"message": f"Deleted {safe_name} and rebuilt index."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Rebuild Endpoint ─────────────────────────────────────────────────────────

@app.post("/api/rebuild")
def rebuild_index():
    """
    Rebuild the FAISS vectorstore from all uploaded PDFs.
    REQUIRED after switching embedding models (e.g., Ollama → HuggingFace).
    Old indexes are NOT compatible with new embeddings.
    """
    try:
        rebuild_all_files()
        return {"message": "Vectorstore rebuilt successfully with new embeddings."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Text Chat Endpoint (Streaming) ───────────────────────────────────────────

@app.post("/api/chat")
def chat(request: ChatRequest):
    """
    RAG chat endpoint:
    1. Load FAISS vectorstore
    2. Retrieve relevant context chunks
    3. Build prompt with context
    4. Stream response from local LLM (llama-cpp-python)
    """
    # 1. Retrieve Context
    vectorstore = load_vectorstore()
    context_docs = retrieve_context(vectorstore, request.query, request.file_filter, request.top_k)

    # 2. Build Prompt
    prompt = build_prompt(request.query, context_docs)

    # 3. Stream Response
    def generate():
        sources_data = _build_sources_payload(context_docs)
        yield f"__SOURCES__{json.dumps(sources_data)}__SOURCES__\n"
        yield from _stream_answer(prompt)

    return StreamingResponse(generate(), media_type="text/plain")


# ─── Voice Endpoints ──────────────────────────────────────────────────────────

@app.post("/api/voice/transcribe")
def voice_transcribe(duration: int = Query(default=5, ge=1, le=30)):
    """
    Record audio from the backend machine's microphone and transcribe it.
    Only works when the backend is running on the same machine as the user.

    Query params:
        duration: Recording duration in seconds (default: 5, max: 30)
    """
    try:
        text = listen_and_transcribe(duration=duration)
        return {"query": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")


@app.post("/api/voice/upload-audio")
def voice_upload_audio(
    audio: UploadFile = File(...),
    file_filter: str = "All Files",
    top_k: int = 5,
):
    """
    Receive audio from browser MediaRecorder, transcribe it, run RAG pipeline,
    and stream the answer. TTS is handled in the browser.
    """
    # 1. Save uploaded audio to temp file
    audio_path = os.path.join(TEMP_AUDIO_DIR, f"upload_{os.path.basename(audio.filename)}")
    try:
        with open(audio_path, "wb") as f:
            f.write(audio.file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save audio: {str(e)}")

    # 2. Transcribe the audio
    try:
        transcribed_text = transcribe_audio(audio_path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
    finally:
        # Clean up temp file
        if os.path.exists(audio_path):
            try:
                os.remove(audio_path)
            except OSError:
                pass

    if not transcribed_text.strip():
        raise HTTPException(status_code=400, detail="Could not transcribe any speech from the audio.")

    # 3. Run RAG pipeline (same as /api/chat)
    vectorstore = load_vectorstore()
    context_docs = retrieve_context(vectorstore, transcribed_text, file_filter, top_k)
    prompt = build_prompt(transcribed_text, context_docs)

    # 4. Stream response
    def generate():
        yield f"__QUERY__{transcribed_text}__QUERY__\n"
        sources_data = _build_sources_payload(context_docs)
        yield f"__SOURCES__{json.dumps(sources_data)}__SOURCES__\n"
        yield from _stream_answer(prompt)

    return StreamingResponse(generate(), media_type="text/plain")


@app.post("/api/voice/chat")
def voice_chat(
    duration: int = Query(default=5, ge=1, le=30),
    file_filter: str = Query(default="All Files"),
    top_k: int = Query(default=5, ge=1, le=10),
    speak: bool = Query(default=True),
):
    """
    Full voice chat: record on backend → transcribe → RAG → stream answer → speak.
    Only works when backend runs on the same machine as the user.

    Query params:
        duration: Recording duration in seconds
        file_filter: Filter to specific file
        top_k: Number of context chunks
        speak: Whether to speak the answer via TTS
    """
    # 1. Record and transcribe
    try:
        transcribed_text = listen_and_transcribe(duration=duration)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recording/transcription failed: {str(e)}")

    if not transcribed_text.strip():
        raise HTTPException(status_code=400, detail="Could not transcribe any speech.")

    # 2. Run RAG pipeline
    vectorstore = load_vectorstore()
    context_docs = retrieve_context(vectorstore, transcribed_text, file_filter, top_k)
    prompt = build_prompt(transcribed_text, context_docs)

    # 3. Stream response
    def generate():
        yield f"__QUERY__{transcribed_text}__QUERY__\n"
        sources_data = _build_sources_payload(context_docs)
        yield f"__SOURCES__{json.dumps(sources_data)}__SOURCES__\n"
        yield from _stream_answer(prompt)

    return StreamingResponse(generate(), media_type="text/plain")
