from dataclasses import dataclass

from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from app.chunking import MODEL_NAME
from app.db import SessionLocal
from app.models import Chunk, Document

# the bge model expects this prefix on questions, but not on stored passages
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

_model = None


def get_model():
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


@dataclass
class Passage:
    chunk_id: int
    file_name: str
    title: str
    entity_type: str
    page_start: int
    page_end: int
    text: str
    distance: float


def search(question: str, top_k: int = 5) -> list[Passage]:
    """Return the top_k passages closest in meaning to the question."""
    vector = get_model().encode(QUERY_PREFIX + question, normalize_embeddings=True).tolist()
    with SessionLocal() as session:
        distance = Chunk.embedding.cosine_distance(vector)
        rows = session.execute(
            select(Chunk, Document, distance.label("distance"))
            .join(Document, Chunk.document_id == Document.id)
            .order_by(distance)
            .limit(top_k)
        ).all()
        return [
            Passage(
                chunk_id=chunk.id,
                file_name=document.file_name,
                title=document.title,
                entity_type=document.entity_type,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                text=chunk.text,
                distance=float(dist),
            )
            for chunk, document, dist in rows
        ]