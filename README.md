# Local Voice-Based RAG Assistant

A fully local, voice-enabled Retrieval-Augmented Generation (RAG) chatbot and web application.  
**No cloud APIs. No external endpoints. 100% private and offline-capable.** Everything runs locally on your machine.

---

## ✨ Key Features

- **Voice & Text Interaction**: Ask questions using voice via browser microphone or type queries directly.
- **Local Speech-to-Text**: Transcription powered by `faster-whisper` running locally.
- **Local Retrieval-Augmented Generation**: PDF ingestion and chunking with local FAISS vector embeddings (`sentence-transformers`).
- **Local LLM Inference**: Fully local GGUF model execution using `llama-cpp-python` with streaming token responses.
- **Local Text-to-Speech**: Optional TTS output via `pyttsx3`.
- **Modern Web Interface**: Built with React, Vite, and custom CSS for an interactive user experience.

---

## 🏗️ Architecture & Workflow

```
+-----------------------------------------------------------------------------+
|                                  FRONTEND                                   |
|   [ Browser Mic / Audio Blob ]  <========>  [ React + Vite UI (App.jsx) ]   |
+-----------------------------------------------------------------------------+
                                       |
                     HTTP / REST / Multipart Form Data
                                       v
+-----------------------------------------------------------------------------+
|                                  BACKEND                                    |
|   1. Voice Input    -->  faster-whisper (Local STT)                         |
|   2. Document Store -->  PyPDFLoader -> FAISS Vector Store                  |
|   3. Retrieval      -->  HuggingFace Embeddings -> Context Lookup          |
|   4. LLM Generation -->  llama-cpp-python (Local GGUF Model) -> Streaming   |
|   5. Voice Output   -->  pyttsx3 (Local TTS)                                |
+-----------------------------------------------------------------------------+
```

---

## 📁 Repository Structure

```text
rag-project/
├── backend/                # Python FastAPI Backend & RAG Pipeline
│   ├── models/             # Directory for local .gguf LLM models (git-ignored)
│   ├── vectorstore/        # Local FAISS vector index (git-ignored)
│   ├── uploads/            # Uploaded PDFs and documents (git-ignored)
│   ├── local_llm.py        # Local LLM wrapper & streaming generation
│   ├── rag_core.py         # Document ingestion, embedding, & FAISS indexing
│   ├── voice_input.py      # Local Whisper STT transcription
│   ├── voice_output.py     # Local TTS speech generation
│   ├── main.py             # FastAPI REST API endpoints
│   ├── requirements.txt    # Python dependencies
│   └── .env.example        # Environment variable template
├── frontend/               # React + Vite Web Frontend
│   ├── src/                # React UI components & API layer
│   ├── package.json        # Node dependencies & npm scripts
│   └── vite.config.js      # Vite build configuration
├── .gitignore              # Repository ignore rules
└── README.md               # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Backend Setup

1. **Navigate to the backend directory:**
   ```bash
   cd backend
   ```
2. **Create and activate a Python virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
4. **Download an LLM model (`.gguf`):**
   - Place your desired `.gguf` model file inside `backend/models/` (e.g., `phi-3-mini-4k-instruct-q4.gguf`).
   - If using a different model filename, update `MODEL_PATH` in `backend/local_llm.py` or specify it in your `.env` file.
5. **Configure environment variables (optional):**
   - Copy `.env.example` to `.env`:
     ```bash
     cp .env.example .env
     ```
6. **Start the FastAPI server:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

---

### 2. Frontend Setup

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend
   ```
2. **Install Node dependencies:**
   ```bash
   npm install
   ```
3. **Start the development server:**
   ```bash
   npm run dev
   ```
4. **Open the App:**
   - Navigate to `http://localhost:5173` in your web browser.

---

## 🔒 Security & Privacy

This repository is configured with a comprehensive `.gitignore` to ensure that:
- No sensitive API keys or `.env` files are accidentally committed.
- No local user documents, PDF uploads, or vector store indices (`FAISS`, `uploads/`) are tracked.
- No large LLM weights (`*.gguf`, `*.bin`, `models/`) are pushed to version control.
- No IDE or agent customization directories (`.agents/`, `.gemini/`) are published.

---

## 📝 License

This project is licensed under the MIT License.
