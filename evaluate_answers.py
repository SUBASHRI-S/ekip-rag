import csv
import re
import sys
import time
from pathlib import Path

from app.rag import LLM_DOWN, answer_question

DIR = Path("data/processed")
ANSWERABLE = DIR / "eval_answerable.csv"
UNANSWERABLE = DIR / "eval_unanswerable.csv"
OUT = DIR / "eval_answers.csv"
FIELDS = ["id", "kind", "question", "answerable", "cited", "cites_gold", "answer", "sources", "gold"]


def read_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def is_gold(passage, row):
    return (
        passage.file_name == row["file_name"]
        and passage.page_start <= int(row["page_end"])
        and passage.page_end >= int(row["page_start"])
    )


def load_done():
    if not OUT.exists():
        return {}
    with open(OUT, encoding="utf-8-sig") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def summarize():
    rows = list(load_done().values())
    unanswerable = [r for r in rows if r["kind"] == "unanswerable"]
    answerable = [r for r in rows if r["kind"] == "answerable"]
    print()
    if unanswerable:
        refused = sum(1 for r in unanswerable if r["answerable"] == "False")
        print(f"Unanswerable questions correctly refused: {refused} of {len(unanswerable)}")
        for r in unanswerable:
            if r["answerable"] == "True":
                print(f"   ANSWERED (should have refused) {r['id']}: {r['question']}")
    if answerable:
        wrongly_refused = [r for r in answerable if r["answerable"] == "False"]
        answered = [r for r in answerable if r["answerable"] == "True"]
        print(f"Answerable questions wrongly refused: {len(wrongly_refused)} of {len(answerable)}")
        print(f"Answers that include a citation: {sum(1 for r in answered if r['cited'])} of {len(answered)}")
        grounded = sum(1 for r in answered if r["cites_gold"] == "True")
        print(f"Answers that cite a passage from the correct place: {grounded} of {len(answered)}")


def main():
    which = sys.argv[1] if len(sys.argv) > 1 else "both"
    jobs = []
    if which in ("both", "unanswerable"):
        jobs += [("unanswerable", r) for r in read_rows(UNANSWERABLE)]
    if which in ("both", "answerable"):
        jobs += [("answerable", r) for r in read_rows(ANSWERABLE)]

    done = load_done()
    new_file = not OUT.exists()
    with open(OUT, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        for kind, row in jobs:
            if row["id"] in done:
                continue  # already answered in an earlier run
            result = answer_question(row["question"])
            if result.answer == LLM_DOWN:
                print(f"{row['id']}: the LLM was busy, will retry on the next run")
                continue
            cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", result.answer)})
            cited = [n for n in cited if 1 <= n <= len(result.passages)]
            gold, cites_gold = "", ""
            if kind == "answerable":
                gold = f"{row['file_name']} {row['page_start']}-{row['page_end']}"
                cites_gold = any(is_gold(result.passages[n - 1], row) for n in cited)
            writer.writerow(
                {
                    "id": row["id"],
                    "kind": kind,
                    "question": row["question"],
                    "answerable": result.answerable,
                    "cited": " ".join(str(n) for n in cited),
                    "cites_gold": cites_gold,
                    "answer": result.answer,
                    "sources": " | ".join(
                        f"[{i}] {p.file_name} {p.page_start}-{p.page_end}"
                        for i, p in enumerate(result.passages, start=1)
                    ),
                    "gold": gold,
                }
            )
            f.flush()
            print(f"{row['id']}: answerable={result.answerable}", flush=True)
            time.sleep(1)
    summarize()


if __name__ == "__main__":
    main()
    