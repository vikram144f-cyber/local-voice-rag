import time
import os
import requests
import json
import logging

# Change dir to backend to load models correctly
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from rag_core import get_embeddings, load_vectorstore, retrieve_context
from local_llm import load_llm, generate_response

print("--- PROFILING BACKEND PERFORMANCE ---")

# 1. Embedding Generation Speed
print("\n1. Profiling Embedding Generation...")
start = time.time()
embeddings = get_embeddings()
load_time = time.time() - start
print(f"  Embedding model load time: {load_time:.4f}s")

text_to_embed = "This is a sample text to measure the embedding generation speed." * 10
start = time.time()
emb = embeddings.embed_query(text_to_embed)
embed_time = time.time() - start
print(f"  Embedding generation time (single query): {embed_time:.4f}s")

# 2. FAISS Retrieval Speed
print("\n2. Profiling FAISS Retrieval...")
start = time.time()
vectorstore = load_vectorstore()
vs_load_time = time.time() - start
print(f"  Vectorstore load time: {vs_load_time:.4f}s")

if vectorstore:
    start = time.time()
    docs = retrieve_context(vectorstore, "test query")
    retrieval_time = time.time() - start
    print(f"  Retrieval time: {retrieval_time:.4f}s")
else:
    print("  Vectorstore not found.")

# 3. LLM Response Time
print("\n3. Profiling LLM (llama-cpp-python)...")
start = time.time()
llm = load_llm()
llm_load_time = time.time() - start
print(f"  LLM load time: {llm_load_time:.4f}s")

prompt = "Hello, what is 2+2?"
start = time.time()
res = generate_response(prompt, max_tokens=10)
gen_time = time.time() - start
print(f"  Generation time (10 tokens): {gen_time:.4f}s")

# 4. FastAPI Streaming Response Latency (TTFT)
# Assuming the server is running on http://localhost:8000
print("\n4. Profiling FastAPI Streaming (TTFT)...")
try:
    start = time.time()
    response = requests.post(
        "http://localhost:8000/api/chat",
        json={"query": "Test query for TTFT", "top_k": 5, "file_filter": "All Files"},
        stream=True
    )
    first_token_time = None
    for chunk in response.iter_content(chunk_size=None):
        if chunk:
            first_token_time = time.time() - start
            break
    if first_token_time:
        print(f"  Time to First Token (TTFT): {first_token_time:.4f}s")
    else:
        print("  No tokens received.")
except Exception as e:
    print(f"  FastAPI request failed: {e}")

print("\n--- PROFILING COMPLETE ---")
