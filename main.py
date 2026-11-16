import json
from collections import Counter

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()


class DocumentOut(BaseModel):
    name: str
    pages: int


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/documents", response_model=list[DocumentOut])
def list_documents():
    counts = Counter()
    with open("data/processed/pages.jsonl", encoding="utf-8") as f:
        for line in f:
            counts[json.loads(line)["document"]] += 1
    return [DocumentOut(name=name, pages=pages) for name, pages in sorted(counts.items())]