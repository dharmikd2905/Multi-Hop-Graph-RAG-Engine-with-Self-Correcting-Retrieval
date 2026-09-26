"""
End-to-end smoke-test script: ingests sample_data/microservices_architecture.md
and runs the PRD's canonical multi-hop query against the running API.

Usage (with the API already running on :8000):
    python scripts/demo.py
"""
import json
import sys
from pathlib import Path

import httpx

API_BASE = "http://localhost:8000"
SAMPLE_DOC = Path(__file__).resolve().parents[1] / "sample_data" / "microservices_architecture.md"


def main() -> None:
    text = SAMPLE_DOC.read_text()

    print("==> Ingesting sample document...")
    resp = httpx.post(
        f"{API_BASE}/api/v1/ingest",
        json={"doc_id": "microservices_demo", "source_name": "microservices_architecture.md", "text": text},
        timeout=120,
    )
    resp.raise_for_status()
    print(json.dumps(resp.json(), indent=2))

    query = "How does Component A indirectly affect Component C through Component B?"
    print(f"\n==> Running multi-hop query: {query!r}")
    resp = httpx.post(f"{API_BASE}/api/v1/query", json={"query": query}, timeout=180)
    resp.raise_for_status()
    result = resp.json()

    print("\n--- Answer ---")
    print(result["answer"])
    print("\n--- Evaluation ---")
    print(json.dumps(result["evaluation"], indent=2))
    print(f"\nRetries used: {result['retries_used']}")
    print("\n--- Top retrieved contexts (by Composite Relevance Index) ---")
    for c in result["contexts"][:5]:
        print(f"  CRI={c['composite_relevance_index']:.4f}  hops={c['graph_hops_used']}  chunk={c['chunk_id']}")


if __name__ == "__main__":
    try:
        main()
    except httpx.ConnectError:
        print("Could not connect to the API. Start it first with: uvicorn app.main:app --reload")
        sys.exit(1)
