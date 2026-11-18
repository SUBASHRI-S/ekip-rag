import csv
import time
from pathlib import Path

from app.llm import generate

SOURCE = Path("data/processed/eval_answerable.csv")
OUT = Path("data/processed/eval_paraphrased.csv")

PROMPT = """Rewrite the question below the way an ordinary customer or a new employee \
would ask it, in everyday words.

Rules:
1. Keep the meaning, and keep the type of lender (bank, NBFC or small finance bank). Keep numbers and names.
2. Do not copy more than two words in a row from the original question, apart from the lender type and any numbers. Replace technical terms with everyday ones.
3. Do not add any fact that is not in the original question.
Reply with only the rewritten question.

Original question: {question}"""


def main():
    with open(SOURCE, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = list(reader.fieldnames) + ["original_question"]

    done = {}
    if OUT.exists():
        with open(OUT, encoding="utf-8-sig") as f:
            done = {r["id"]: r for r in csv.DictReader(f)}

    results = []
    for row in rows:
        if row["id"] in done:
            results.append(done[row["id"]])
            continue
        rewritten = generate(PROMPT.format(question=row["question"])).strip().strip('"')
        new_row = dict(row)
        new_row["original_question"] = row["question"]
        new_row["question"] = rewritten
        results.append(new_row)
        print(f"{row['id']}: {rewritten}", flush=True)
        with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(results + [r for r in rows if r["id"] not in {x["id"] for x in results}][:0])
        time.sleep(6)  # stay under the free tier's rate limit


if __name__ == "__main__":
    main()