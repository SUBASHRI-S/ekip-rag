import os
import sys
import csv
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

def is_hit(passage, row):
    """Right document, and the passage's pages overlap the expected pages."""
    return (
        passage.file_name == row["file_name"]
        and passage.page_start <= int(row["page_end"])
        and passage.page_end >= int(row["page_start"])
    )


def first_hit_rank(passages, row):
    for rank, passage in enumerate(passages, start=1):
        if is_hit(passage, row):
            return rank
    return None


def main():
    with open(QUESTIONS, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    n = len(rows)
    print(f"{n} questions, {len(CONFIGS)} settings\n")

    summary, details = [], []
    for mode, use_filter in CONFIGS:
        label = f"{mode} + lender filter" if use_filter else f"{mode}, no filter"
        ranks = []
        started = time.time()
        for row in rows:
            passages = search(row["question"], top_k=TOP_K, mode=mode, auto_filter=use_filter)
            rank = first_hit_rank(passages, row)
            ranks.append(rank)
            top = passages[0] if passages else None
            details.append(
                {
                    "setting": label,
                    "id": row["id"],
                    "first_hit_rank": rank or "",
                    "gold": f"{row['file_name']} {row['page_start']}-{row['page_end']}",
                    "top1": f"{top.file_name} {top.page_start}-{top.page_end}" if top else "",
                    "question": row["question"],
                }
            )
        seconds = (time.time() - started) / n
        result = {
            "setting": label,
            "hit@1": round(sum(1 for r in ranks if r == 1) / n, 3),
            "recall@5": round(sum(1 for r in ranks if r and r <= 5) / n, 3),
            "recall@10": round(sum(1 for r in ranks if r) / n, 3),
            "mrr@10": round(sum(1 / r for r in ranks if r) / n, 3),
            "sec_per_question": round(seconds, 2),
        }
        summary.append(result)
        print(f"done: {label}", flush=True)

    with open(RESULTS, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        writer.writeheader()
        writer.writerows(summary)
    with open(DETAILS, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(details[0].keys()))
        writer.writeheader()
        writer.writerows(details)

    print()
    print(f"{'setting':28} {'hit@1':>6} {'rec@5':>6} {'rec@10':>7} {'mrr@10':>7} {'sec/q':>6}")
    for r in summary:
        print(
            f"{r['setting']:28} {r['hit@1']:6.3f} {r['recall@5']:6.3f} "
            f"{r['recall@10']:7.3f} {r['mrr@10']:7.3f} {r['sec_per_question']:6.2f}"
        )

    best = "rerank + lender filter"
    print(f"\nQuestions where '{best}' found no hit in the top 5:")
    for d in details:
        if d["setting"] == best and (not d["first_hit_rank"] or int(d["first_hit_rank"]) > 5):
            print(f"  {d['id']}: {d['question'][:110]}")
            print(f"       expected {d['gold']}, top result {d['top1']}")


if __name__ == "__main__":
    main()