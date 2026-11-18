import csv
import random
import time
from pathlib import Path

from openai import APIError
from sqlalchemy import select

from app.db import SessionLocal
from app.llm import generate
from app.models import Chunk, Document

COUNT = 50
PER_DOCUMENT_LIMIT = 3
OUT = Path("data/processed/candidates.csv")
LENDER_NAMES = {
    "commercial_bank": "commercial banks",
    "nbfc": "NBFCs",
    "small_finance_bank": "small finance banks",
}

PROMPT = """You are helping build a test set for a question answering system about \
Reserve Bank of India (RBI) Master Directions.

Read the passage below and write ONE question that a compliance officer could ask \
and that this passage answers fully.

Rules:
1. Paraphrase. Do not copy any phrase of four or more words from the passage.
2. The question must mention the type of lender: {lender}.
3. The question must make sense on its own. Never write "in this passage" or "above".
4. Ask about a specific rule, limit, condition or definition.
Reply with only the question.

Passage:
{passage}"""


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    random.seed(42)
    rows_out = []
    with SessionLocal() as session:
        rows = session.execute(
            select(Chunk, Document)
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.token_count >= 400)
        ).all()
        random.shuffle(rows)
        per_document = {}
        for chunk, document in rows:
            if len(rows_out) == COUNT:
                break
            if per_document.get(document.file_name, 0) >= PER_DOCUMENT_LIMIT:
                continue
            passage = " ".join(chunk.text.split())
            lender = LENDER_NAMES[document.entity_type]
            try:
                question = generate(PROMPT.format(lender=lender, passage=passage)).strip().strip('"')
            except APIError as error:
                print("skipped one chunk:", error)
                continue
            per_document[document.file_name] = per_document.get(document.file_name, 0) + 1
            rows_out.append(
                {
                    "id": f"c{len(rows_out) + 1:02d}",
                    "keep": "",
                    "question": question,
                    "lender": document.entity_type,
                    "file_name": document.file_name,
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "passage": passage,
                }
            )
            print(f"{len(rows_out)}/{COUNT}: {question}", flush=True)
            time.sleep(1)

    with open(OUT, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        writer.writeheader()
        writer.writerows(rows_out)
    print("Saved", OUT)


if __name__ == "__main__":
    main()