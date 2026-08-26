import os
import json
import logging
from dotenv import load_dotenv
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ─── Configuration ─────────────────────────────────────────────────────────────
load_dotenv()
logger = logging.getLogger(__name__)

UPLOADS_DIR = os.getenv("UPLOADS_DIR", "uploads")
VECTORSTORE_DIR = os.getenv("VECTORSTORE_DIR", "vectorstore")
REGISTRY_FILE = os.getenv("REGISTRY_FILE", "uploaded_files.json")

# Local embedding model — downloads automatically on first use (~90MB), then runs offline.
# NOTE: After switching from OllamaEmbeddings to HuggingFaceEmbeddings, you MUST rebuild
#       the vectorstore. Old FAISS indexes are NOT compatible with the new embeddings.
#       Use POST /api/rebuild or re-upload all PDFs.
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Ensure directories exist
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(VECTORSTORE_DIR, exist_ok=True)

# ─── File Registry ─────────────────────────────────────────────────────────────
def load_registry():
    if os.path.exists(REGISTRY_FILE):
        try:
            with open(REGISTRY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return []
    return []

def save_registry(registry):
    with open(REGISTRY_FILE, "w") as f:
        json.dump(registry, f, indent=4)

# ─── Vectorstore ─────────────────────────────────────────────────────────────
_embeddings = None
_vectorstore_cache = None

def invalidate_vectorstore_cache():
    global _vectorstore_cache
    _vectorstore_cache = None

def get_embeddings():
    """Create local HuggingFace embeddings."""
    global _embeddings
    if _embeddings is None:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("Initializing embeddings on device: %s", device)
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": True}
        )
    return _embeddings

def load_vectorstore():
    global _vectorstore_cache
    if _vectorstore_cache is not None:
        return _vectorstore_cache
    # The index location is server configuration; no API request accepts a filesystem path.
    if not os.path.exists(os.path.join(VECTORSTORE_DIR, "index.faiss")):
        return None
    try:
        _vectorstore_cache = FAISS.load_local(
            folder_path=VECTORSTORE_DIR,
            embeddings=get_embeddings(),
            allow_dangerous_deserialization=True,
        )
        return _vectorstore_cache
    except Exception as e:
        logger.error("Failed to load vectorstore: %s", e)
        return None

# ─── RAG Pipeline Core ────────────────────────────────────────────────────────
def process_upload(file_path: str, filename: str):
    """Process a single PDF and add to FAISS."""
    global _vectorstore_cache
    # 1. Extract Text
    loader = PyPDFLoader(file_path)
    docs = loader.load()

    # Inject source file name into metadata
    for doc in docs:
        doc.metadata["source_file"] = filename

    # 2. Chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = text_splitter.split_documents(docs)

    # 3. Embed & Store
    embeddings = get_embeddings()
    vs = load_vectorstore()

    if vs is None:
        vs = FAISS.from_documents(chunks, embeddings)
    else:
        vs.add_documents(chunks)
    
    vs.save_local(VECTORSTORE_DIR)
    invalidate_vectorstore_cache()
    _vectorstore_cache = vs

    # 4. Update Registry
    registry = load_registry()
    if filename not in registry:
        registry.append(filename)
        save_registry(registry)
    
    return len(chunks)

def delete_file_and_rebuild(filename: str):
    """Delete a file and completely rebuild FAISS from remaining files."""
    file_path = os.path.join(UPLOADS_DIR, filename)
    if os.path.exists(file_path):
        os.remove(file_path)
    
    registry = load_registry()
    if filename in registry:
        registry.remove(filename)
        save_registry(registry)
        
    rebuild_all_files()

def rebuild_all_files():
    if os.path.exists(VECTORSTORE_DIR):
        shutil.rmtree(VECTORSTORE_DIR)
    os.makedirs(VECTORSTORE_DIR, exist_ok=True)
    
    registry = load_registry()
    embeddings = get_embeddings()
    
    all_chunks = []
    valid_registry = []
    
    for filename in registry:
        file_path = os.path.join(UPLOADS_DIR, filename)
        if not os.path.exists(file_path):
            continue
            
        valid_registry.append(filename)
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source_file"] = filename
            
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
    
    save_registry(valid_registry)
    
    global _vectorstore_cache
    invalidate_vectorstore_cache()
    if all_chunks:
        vs = FAISS.from_documents(all_chunks, embeddings)
        vs.save_local(VECTORSTORE_DIR)
        _vectorstore_cache = vs

# ─── Chat Helpers ─────────────────────────────────────────────────────────────
def retrieve_context(vectorstore, query: str, file_filter: str = None, top_k: int = 5):
    if vectorstore is None:
        return []
        
    search_kwargs = {"k": top_k}
    if file_filter and file_filter != "All Files":
        search_kwargs["filter"] = {"source_file": file_filter}
        
    try:
        retriever = vectorstore.as_retriever(search_kwargs=search_kwargs)
        return retriever.invoke(query)
    except Exception as e:
        logger.error("Error retrieving context: %s", e)
        return []

def build_prompt(query: str, context_docs: list, context_char_limit: int = 900) -> str:
    if not context_docs:
        return (
            "STRICT DOCUMENT ASSISTANT RULE:\n"
            "No context from the selected document matches this question.\n"
            "Do NOT answer from general knowledge.\n"
            "You MUST reply ONLY with this exact phrase and nothing else:\n"
            "I can only answer questions based on the provided documents.\n\n"
            f"User Question: {query}"
        )
        
    context_text = ""
    for i, doc in enumerate(context_docs, 1):
        source = doc.metadata.get('source_file', 'unknown')
        snippet = doc.page_content[:context_char_limit]
        context_text += f"[{source}]\n{snippet}\n\n"
        
    return (
        "STRICT DOCUMENT ASSISTANT INSTRUCTIONS:\n"
        "1. Check if the topic of the User Question appears anywhere in the DOCUMENT CONTEXT below.\n"
        "2. IF AND ONLY IF the topic is present in the DOCUMENT CONTEXT below (even as code or brief mention), you may explain it and answer the question.\n"
        "3. IF the topic of the User Question is NOT present in the DOCUMENT CONTEXT below, you MUST refuse to answer and reply ONLY with this exact phrase:\n"
        "I can only answer questions based on the provided documents.\n\n"
        "Do NOT answer questions about topics from other documents or general knowledge if they are missing from the CONTEXT below.\n\n"
        f"--- DOCUMENT CONTEXT ---\n{context_text}\n"
        f"User Question: {query}"
    )
