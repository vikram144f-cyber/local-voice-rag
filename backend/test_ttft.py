import time
import requests

print("--- PROFILING FASTAPI TTFT ---")
try:
    start = time.time()
    response = requests.post(
        "http://localhost:8000/api/chat",
        json={"query": "Hello?", "top_k": 5, "file_filter": "All Files"},
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
