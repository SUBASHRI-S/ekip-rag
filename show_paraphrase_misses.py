import csv
from pathlib import Path

DETAILS = "data/processed/eval_details_paraphrased.csv"
QUESTIONS = "data/processed/eval_paraphrased.csv"

with open(QUESTIONS, encoding="utf-8-sig") as f:
    original = {r["id"]: r["original_question"] for r in csv.DictReader(f)}
with open(DETAILS, encoding="utf-8-sig") as f:
    details = list(csv.DictReader(f))

lines = []
for setting in ("hybrid + lender filter", "rerank + lender filter"):
    misses = [
        d for d in details
        if d["setting"] == setting and (not d["near_rank"] or int(d["near_rank"]) > 5)
    ]
    lines.append(f"\n===== {setting}: {len(misses)} questions with no near hit in the top 5 =====")
    for d in misses:
        lines.append(f"\n{d['id']}")
        lines.append(f"   original:    {original.get(d['id'], '?')}")
        lines.append(f"   paraphrased: {d['question']}")

text = "\n".join(lines)
print(text)
Path("data/processed/paraphrase_misses.txt").write_text(text, encoding="utf-8")