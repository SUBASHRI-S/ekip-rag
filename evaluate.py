import csv
import os
import sys
import time
from pathlib import Path

from app.retrieval import search

QUESTIONS = Path(os.environ.get("EVAL_QUESTIONS", "data/processed/eval_answerable.csv"))
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
# `python evaluate.py bm25 rerank` runs only the named modes, with the lender filter on
if sys.argv[1:]:
    CONFIGS = [c for c in CONFIGS if c[0] in sys.argv[1:] and c[1]]


def exact_hit(passage, row):
    """Strictest: exactly the chunk the question was written from."""
    return bool(row.get("gold_chunk_id")) and passage.chunk_id == int(row["gold_chunk_id"])


def near_hit(passage, row):
    """The source chunk, or its neighbour in the same document (their text overlaps)."""
    return (
        bool(row.get("gold_chunk_id"))
        and passage.file_name == row["file_name"]
        and abs(passage.chunk_id - int(row["gold_chunk_id"])) <= 1
    )


def page_hit(passage, row):
    """Most lenient: right document, and the pages overlap the expected pages."""
    return (
        passage.file_name == row["file_name"]
        and passage.page_start <= int(row["page_end"])
        and passage.page_end >= int(row["page_start"])
    )


RULES = {"exact": exact_hit, "near": near_hit, "page": page_hit}


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
    print(f"{n} questions from {QUESTIONS.name}, {len(CONFIGS)} settings\n")

    summary, details = [], []
    for mode, use_filter in CONFIGS:
        label = f"{mode} + lender filter" if use_filter else f"{mode}, no filter"
        ranks = {name: [] for name in RULES}
        started = time.time()
        for row in rows:
            passages = search(row["question"], top_k=TOP_K, mode=mode, auto_filter=use_filter)
            found = {name: first_rank(passages, test, row) for name, test in RULES.items()}
            for name in RULES:
                ranks[name].append(found[name])
            details.append(
                {
                    "setting": label,
                    "id": row["id"],
                    "exact_rank": found["exact"] or "",
                    "near_rank": found["near"] or "",
                    "page_rank": found["page"] or "",
                    "gold_chunk": row.get("gold_chunk_id", ""),
                    "top1_chunk": passages[0].chunk_id if passages else "",
                    "question": row["question"],
                }
            )
        summary.append(
            {
                "setting": label,
                "R@5 exact": share(ranks["exact"], n, 5),
                "R@5 near": share(ranks["near"], n, 5),
                "R@5 page": share(ranks["page"], n, 5),
                "Hit@1 exact": share(ranks["exact"], n, 1),
                "Hit@1 near": share(ranks["near"], n, 1),
                "MRR near": mrr(ranks["near"], n),
                "sec/q": round((time.time() - started) / n, 2),
            }
        )
        print(f"done: {label}", flush=True)

    for path, data in ((RESULTS, summary), (DETAILS, details)):
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(data[0].keys()))
            writer.writeheader()
            writer.writerows(data)

    columns = ["R@5 exact", "R@5 near", "R@5 page", "Hit@1 exact", "Hit@1 near", "MRR near", "sec/q"]
    print()
    print(f"{'setting':26}" + "".join(f"{c:>12}" for c in columns))
    for r in summary:
        print(f"{r['setting']:26}" + "".join(f"{r[c]:12.3f}" for c in columns))
    print(f"\nFull per-question results: {DETAILS}")


if __name__ == "__main__":
    main()