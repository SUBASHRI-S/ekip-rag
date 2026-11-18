import csv
import os
import sys
import time
from pathlib import Path

from app.retrieval import search

QUESTIONS = Path("data/processed/eval_answerable.csv")
TAG = os.environ.get("EVAL_TAG", "default")
RESULTS = Path(f"data/processed/eval_results_{TAG}.csv")
DETAILS = Path(f"data/processed/eval_details_{TAG}.csv")
TOP_K = 10

# (search mode, whether to use the lender-type filter)
CONFIGS = [
    ("vector", False),
    ("vector", True),
    ("bm25", True),
    ("hybrid", True),
    ("rerank", True),
    ("rerank", False),
]
# `python evaluate.py rerank` runs only the named modes, with the lender filter on
if sys.argv[1:]:
    CONFIGS = [c for c in CONFIGS if c[0] in sys.argv[1:] and c[1]]


def page_hit(passage, row):
    """Lenient: right document, and the pages overlap the expected pages."""
    return (
        passage.file_name == row["file_name"]
        and passage.page_start <= int(row["page_end"])
        and passage.page_end >= int(row["page_start"])
    )


def chunk_hit(passage, row):
    """Strict: exactly the chunk the question was written from."""
    return bool(row.get("gold_chunk_id")) and passage.chunk_id == int(row["gold_chunk_id"])


def first_rank(passages, test, row):
    for rank, passage in enumerate(passages, start=1):
        if test(passage, row):
            return rank
    return None


def share(ranks, n, limit):
    return round(sum(1 for r in ranks if r and r <= limit) / n, 3)


def mrr(ranks, n):
    return round(sum(1 / r for r in ranks if r) / n, 3)


def main():
    with open(QUESTIONS, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    print(f"{n} questions, {len(CONFIGS)} settings\n")

    summary, details = [], []
    for mode, use_filter in CONFIGS:
        label = f"{mode} + lender filter" if use_filter else f"{mode}, no filter"
        page_ranks, chunk_ranks = [], []
        started = time.time()
        for row in rows:
            passages = search(row["question"], top_k=TOP_K, mode=mode, auto_filter=use_filter)
            page_rank = first_rank(passages, page_hit, row)
            chunk_rank = first_rank(passages, chunk_hit, row)
            page_ranks.append(page_rank)
            chunk_ranks.append(chunk_rank)
            top = passages[0] if passages else None
            details.append(
                {
                    "setting": label,
                    "id": row["id"],
                    "page_rank": page_rank or "",
                    "chunk_rank": chunk_rank or "",
                    "gold_chunk": row.get("gold_chunk_id", ""),
                    "top1_chunk": top.chunk_id if top else "",
                    "question": row["question"],
                }
            )
        summary.append(
            {
                "setting": label,
                "page_recall@5": share(page_ranks, n, 5),
                "chunk_hit@1": share(chunk_ranks, n, 1),
                "chunk_recall@5": share(chunk_ranks, n, 5),
                "chunk_recall@10": share(chunk_ranks, n, 10),
                "chunk_mrr@10": mrr(chunk_ranks, n),
                "sec_per_question": round((time.time() - started) / n, 2),
            }
        )
        print(f"done: {label}", flush=True)

    for path, data in ((RESULTS, summary), (DETAILS, details)):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)

    print()
    print(f"{'setting':26} {'pageR@5':>8} {'chunkH@1':>9} {'chunkR@5':>9} {'chunkR@10':>10} {'chunkMRR':>9} {'sec/q':>6}")
    for r in summary:
        print(
            f"{r['setting']:26} {r['page_recall@5']:8.3f} {r['chunk_hit@1']:9.3f} {r['chunk_recall@5']:9.3f} "
            f"{r['chunk_recall@10']:10.3f} {r['chunk_mrr@10']:9.3f} {r['sec_per_question']:6.2f}"
        )

    best = "rerank + lender filter"
    print(f"\nQuestions where '{best}' has no exact-chunk hit in the top 5:")
    for d in details:
        if d["setting"] == best and (not d["chunk_rank"] or int(d["chunk_rank"]) > 5):
            print(f"  {d['id']}: {d['question'][:100]}")
            print(f"       gold chunk {d['gold_chunk']}, top result chunk {d['top1_chunk']}")


if __name__ == "__main__":
    main()