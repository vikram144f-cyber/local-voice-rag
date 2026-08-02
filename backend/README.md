# Local Voice-Based RAG Assistant

A fully local, voice-enabled Retrieval-Augmented Generation (RAG) chatbot.  
**No Ollama. No cloud APIs. No external endpoints.** Everything runs on your laptop.

---

## Architecture

```
INDEXING PHASE:
  PDF documents
  → PyPDFLoader (extract text)
  → RecursiveCharacterTextSplitter (chunk text)
  → HuggingFaceEmbeddings / sentence-transformers (embed locally)
  → FAISS vectorstore (store on disk)

TEXT QUERY PHASE:
  User text query
  → same local embedding model
  → FAISS similarity search
  → retrieved context chunks
  → prompt template
  → llama-cpp-python (local GGUF model)
  → streamed answer + source chunks

VOICE QUERY PHASE:
  Browser microphone
  → MediaRecorder (browser API)
  → audio blob sent to backend
  → faster-whisper (local STT)
  → transcribed text query
  → same RAG pipeline as above
  → text answer streamed to frontend
  → pyttsx3 (local TTS, optional)
  → speaker output
```

---

## Model Files Needed

### 1. LLM (GGUF format) — **Manual download required**
Place a `.gguf` model file in `backend/models/`.  
Default expected path: `backend/models/phi-3-mini-4k-instruct-q4.gguf`

If your file has a different name, update `MODEL_PATH` in `backend/local_llm.py`.

**Recommended models:**
- [Phi-3 Mini 4K Instruct Q4](https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf) (~2.4GB)
- [Mistral 7B Instruct Q4_K_M](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF) (~4.1GB)

### 2. Embedding Model — **Auto-downloads on first run** (~90MB)
`sentence-transformers/all-MiniLM-L6-v2` is downloaded automatically.

### 3. Whisper STT Model — **Auto-downloads on first run** (~150MB)
`faster-whisper` base model is downloaded automatically.

> **After the first run, the app works fully offline.**

---

## Setup

### Backend

```bash
cd backend
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### Frontend

```bash
cd frontend
npm install
```

---

## Running

### Start Backend
```bash
cd backend
venv\Scripts\activate
uvicorn main:app --reload
```
Backend runs at: `http://localhost:8000`

### Start Frontend
```bash
cd frontend
npm run dev
```
Frontend runs at: `http://localhost:5173`

---

## Important: Rebuild Vectorstore After Migration

If you previously used OllamaEmbeddings, the old FAISS indexes are **NOT compatible** with the new HuggingFace embeddings.

**Rebuild via API:**
```bash
curl -X POST http://localhost:8000/api/rebuild
```

**Or** use the "REBUILD INDEX" button in the sidebar.

**Or** simply re-upload your PDF files.

---

## Module Guide

| Module | Purpose |
|--------|---------|
| `rag_core.py` | PDF loading, chunking, embeddings, FAISS operations, retrieval, prompt building |
| `local_llm.py` | Loads GGUF model via llama-cpp-python, provides generate/stream functions |
| `voice_input.py` | Records audio (sounddevice), transcribes (faster-whisper) |
| `voice_output.py` | Text-to-speech via pyttsx3 (offline, uses OS TTS engine) |
| `main.py` | FastAPI app — all API endpoints (files, chat, voice, rebuild) |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/files` | List uploaded files |
| POST | `/api/upload` | Upload a PDF |
| DELETE | `/api/files/{filename}` | Delete a file and rebuild index |
| POST | `/api/chat` | Text RAG chat (streaming) |
| POST | `/api/rebuild` | Rebuild vectorstore |
| POST | `/api/voice/transcribe` | Record + transcribe on backend |
| POST | `/api/voice/upload-audio` | Receive browser audio, transcribe, RAG |
| POST | `/api/voice/chat` | Full voice pipeline on backend |

---

## Known Limitations

1. **First run requires internet** — to download embedding model and Whisper model. After that, fully offline.
2. **GGUF model must be downloaded manually** — not auto-downloaded.
3. **TTS (pyttsx3) plays on the server machine** — if backend is remote, TTS won't be heard by the user.
4. **llama-cpp-python on Windows** — may require Visual C++ Build Tools for compilation. If `pip install` fails, install [Visual C++ Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/).
5. **CPU-only inference** — LLM runs on CPU which is slower than GPU. Expect 5-15 seconds per response depending on model size and hardware.
6. **Voice recording in browser** — requires HTTPS or localhost. Won't work over plain HTTP on a remote server.
