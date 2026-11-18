import csv

from sqlalchemy import func, select

from app.db import SessionLocal
from app.models import Chunk

with open("data/processed/eval_unanswerable.csv", encoding="utf-8-sig") as f:
    rows = list(csv.DictReader(f))

with SessionLocal() as session:
    for row in rows:
        count = session.scalar(
            select(func.count()).select_from(Chunk).where(Chunk.text.ilike(f"%{row['check_term']}%"))
        )
        note = "OK" if count == 0 else "CHECK BY HAND"
        print(f"{row['id']} [{row['type']}] '{row['check_term']}' appears in {count} chunks -> {note}")