import csv
from pathlib import Path

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Chunk

CANDIDATES = Path("data/processed/candidates.csv")
QUESTIONS = Path("data/processed/eval_answerable.csv")


def normalise(text):
    return " ".join(text.split())


with SessionLocal() as session:
    by_text = {normalise(text): chunk_id for chunk_id, text in session.execute(select(Chunk.id, Chunk.text))}

with open(CANDIDATES, encoding="utf-8-sig") as f:
    passage_of = {r["id"]: r["passage"] for r in csv.DictReader(f)}

with open(QUESTIONS, encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fields = list(reader.fieldnames)

if "gold_chunk_id" not in fields:
    fields.append("gold_chunk_id")

missing = []
for row in rows:
    chunk_id = by_text.get(normalise(passage_of.get(row["id"], "")))
    row["gold_chunk_id"] = chunk_id or ""
    if not chunk_id:
        missing.append(row["id"])

with open(QUESTIONS, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print(f"{len(rows) - len(missing)} of {len(rows)} questions linked to their source chunk")
if missing:
    print("Not linked:", ", ".join(missing))