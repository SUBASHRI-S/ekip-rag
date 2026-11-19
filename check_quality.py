import csv
import sys

from app.retrieval import search

# Minimum acceptable Recall@5 (hybrid + lender filter, page rule).
# Measured: 0.932 on the original questions, 0.795 on the everyday-wording questions.
# Each threshold allows losing at most 2 of the 44 questions.
THRESHOLDS = {
    "data/processed/eval_answerable.csv": 0.88,
    "data/processed/eval_paraphrased.csv": 0.75,
}


def is_hit(passage, row):
    return (
        passage.file_name == row["file_name"]
        and passage.page_start <= int(row["page_end"])
        and passage.page_end >= int(row["page_start"])
    )


failed = False
for path, minimum in THRESHOLDS.items():
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    hits = sum(
        1
        for row in rows
        if any(is_hit(p, row) for p in search(row["question"], top_k=5, mode="hybrid"))
    )
    recall = hits / len(rows)
    status = "OK" if recall >= minimum else "FAILED"
    print(f"{path.split('/')[-1]}: Recall@5 = {recall:.3f} ({hits}/{len(rows)}), minimum {minimum} -> {status}")
    failed = failed or recall < minimum

sys.exit(1 if failed else 0)