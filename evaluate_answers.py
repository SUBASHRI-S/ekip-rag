import csv
import re
import sys
import time
from pathlib import Path

from app.rag import LLM_DOWN, answer_question, extract_citations

DIR = Path("data/processed")
ANSWERABLE = DIR / "eval_answerable.csv"
UNANSWERABLE = DIR / "eval_unanswerable.csv"
OUT = DIR / "eval_answers.csv"
FIELDS = ["id", "kind", "question", "answerable", "cited", "cites_gold", "answer", "sources", "gold"]
SOURCE_PATTERN = re.compile(r"\[(\d+)\] (\S+) (\d+)-(\d+)")


def read_rows(path):
    with open(path, encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_done():
    if not OUT.exists():
        return {}
    return {r["id"]: r for r in read_rows(OUT)}


def parse_sources(text):
    return [(m.group(2), int(m.group(3)), int(m.group(4))) for m in SOURCE_PATTERN.finditer(text)]


def parse_gold(text):
    name, pages = text.split(" ")
    start, end = pages.split("-")
    return name, int(start), int(end)


def rescore_all():
    """Recalculate the citation columns of every saved answer."""
    rows = list(load_done().values())
    for r in rows:
        sources = parse_sources(r["sources"])
        cited = extract_citations(r["answer"], len(sources))
        r["cited"] = " ".join(str(n) for n in cited)
        if r["kind"] == "answerable" and r["gold"]:
            name, start, end = parse_gold(r["gold"])
            r["cites_gold"] = str(
                any(
                    sources[n - 1][0] == name and sources[n - 1][1] <= end and sources[n - 1][2] >= start
                    for n in cited
                )
            )
        else:
            r["cites_gold"] = ""
    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


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
        with_citation = [r for r in answered if r["cited"]]
        grounded = [r for r in answered if r["cites_gold"] == "True"]
        print(f"Answerable questions wrongly refused: {len(wrongly_refused)} of {len(answerable)}")
        print(f"Answers that include a citation: {len(with_citation)} of {len(answered)}")
        print(f"Answers that cite a passage from the correct place: {len(grounded)} of {len(answered)}")
        if with_citation:
            print(f"   (of the answers that cite anything, {len(grounded)} of {len(with_citation)} cite a correct place)")


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
            sources = " | ".join(
                f"[{i}] {p.file_name} {p.page_start}-{p.page_end}"
                for i, p in enumerate(result.passages, start=1)
            )
            gold = f"{row['file_name']} {row['page_start']}-{row['page_end']}" if kind == "answerable" else ""
            writer.writerow(
                {
                    "id": row["id"],
                    "kind": kind,
                    "question": row["question"],
                    "answerable": result.answerable,
                    "cited": "",
                    "cites_gold": "",
                    "answer": result.answer,
                    "sources": sources,
                    "gold": gold,
                }
            )
            f.flush()
            print(f"{row['id']}: answerable={result.answerable}", flush=True)
            time.sleep(6)  # stay under the free tier's rate limit
    rescore_all()
    summarize()


if __name__ == "__main__":
    main()