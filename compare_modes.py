import sys

from app.retrieval import search

question = " ".join(sys.argv[1:])
for mode in ("vector", "bm25", "hybrid", "rerank"):
    print(f"\n=== {mode.upper()} ===")
    for rank, p in enumerate(search(question, top_k=5, mode=mode), start=1):
        flat = " ".join(p.text.split())
        print(f"{rank}. {p.title}, pages {p.page_start}-{p.page_end} (chunk {p.chunk_id}, score {p.score:.3f})")
        print("   " + flat[:110] + "...")