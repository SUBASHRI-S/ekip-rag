import json
from collections import Counter

from fastapi import FastAPI
from pydantic import BaseModel, Field

from app.rag import answer_question, format_pages

app = FastAPI(title="EKIP RAG")


class DocumentOut(BaseModel):
    name: str
    pages: int


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class SourceOut(BaseModel):
    number: int
    title: str
    file_name: str
    entity_type: str
    pages: str
    distance: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    answerable: bool
    sources: list[SourceOut]


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


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    result = answer_question(request.question)
    sources = [
        SourceOut(
            number=number,
            title=p.title,
            file_name=p.file_name,
            entity_type=p.entity_type,
            pages=format_pages(p),
            distance=round(p.distance, 3),
            text=p.text,
        )
        for number, p in enumerate(result.passages, start=1)
    ]
    return ChatResponse(answer=result.answer, answerable=result.answerable, sources=sources)