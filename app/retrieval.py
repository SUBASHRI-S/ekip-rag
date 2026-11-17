import os
import re
from collections import defaultdict
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from app.chunking import MODEL_NAME
from app.db import SessionLocal
from app.models import Chunk, Document

# the bge model expects this prefix on questions, but not on stored passages
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "and", "or", "is", "are", "for",
    "on", "by", "be", "as", "at", "with", "that", "this", "it",
}
POOL_SIZE = 50  # how many candidates each search contributes to the merge
RRF_K = 60

_model = None
_bm25_index = None


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
    distance: float  # how far the passage is from the question in meaning (smaller = closer)
    score: float = 0.0  # the ranking score of the search mode used (bigger = better)


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, drop very common words."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def embed_query(question: str) -> list[float]:
    return get_model().encode(QUERY_PREFIX + question, normalize_embeddings=True).tolist()


def get_bm25_index():
    """Build the keyword index once from all chunks in the database."""
    global _bm25_index
    if _bm25_index is None:
        with SessionLocal() as session:
            rows = session.execute(select(Chunk.id, Chunk.text).order_by(Chunk.id)).all()
        ids = [row[0] for row in rows]
        _bm25_index = (ids, BM25Okapi([tokenize(row[1]) for row in rows]))
    return _bm25_index


def vector_ranking(vector: list[float], k: int) -> list[int]:
    with SessionLocal() as session:
        distance = Chunk.embedding.cosine_distance(vector)
        return list(session.scalars(select(Chunk.id).order_by(distance).limit(k)))


def bm25_ranking(question: str, k: int) -> list[tuple[int, float]]:
    ids, bm25 = get_bm25_index()
    scores = bm25.get_scores(tokenize(question))
    best = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)[:k]
    return [(ids[i], float(scores[i])) for i in best if scores[i] > 0]


def fuse(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal Rank Fusion: merge several ranked lists of chunk ids."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] += 1 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def load_passages(chunk_ids: list[int], vector: list[float], scores: dict) -> list[Passage]:
    if not chunk_ids:
        return []
    with SessionLocal() as session:
        distance = Chunk.embedding.cosine_distance(vector)
        rows = session.execute(
            select(Chunk, Document, distance.label("distance"))
            .join(Document, Chunk.document_id == Document.id)
            .where(Chunk.id.in_(chunk_ids))
        ).all()
        by_id = {chunk.id: (chunk, document, float(dist)) for chunk, document, dist in rows}
        passages = []
        for chunk_id in chunk_ids:  # keep the ranked order
            chunk, document, dist = by_id[chunk_id]
            passages.append(
                Passage(
                    chunk_id=chunk.id,
                    file_name=document.file_name,
                    title=document.title,
                    entity_type=document.entity_type,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    text=chunk.text,
                    distance=dist,
                    score=scores.get(chunk_id, 1 - dist),
                )
            )
        return passages

RERANK_POOL = 30  # how many candidates the reranker re-reads
RERANKER_NAME = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")

_reranker = None


def get_reranker():
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder

        _reranker = CrossEncoder(RERANKER_NAME, max_length=512)
    return _reranker

def rerank(question: str, passages: list[Passage], top_k: int) -> list[Passage]:
    """Re-score each (question, passage) pair together and keep the best top_k."""
    if not passages:
        return []
    pairs = [(question, " ".join(p.text.split())) for p in passages]
    scores = get_reranker().predict(pairs)
    ranked = sorted(zip(passages, scores), key=lambda item: item[1], reverse=True)[:top_k]
    for passage, score in ranked:
        passage.score = float(score)
    return [passage for passage, _ in ranked]

def search(question: str, top_k: int = 5, mode: str = "rerank") -> list[Passage]:
    """mode is 'vector', 'bm25', 'hybrid' or 'rerank'."""
    if mode not in ("vector", "bm25", "hybrid", "rerank"):
        raise ValueError("mode must be 'vector', 'bm25', 'hybrid' or 'rerank'")
    vector = embed_query(question)

    if mode == "vector":
        ids, scores = vector_ranking(vector, top_k), {}
    elif mode == "bm25":
        ranked = bm25_ranking(question, top_k)
        ids, scores = [cid for cid, _ in ranked], dict(ranked)
    else:
        keyword_ids = [cid for cid, _ in bm25_ranking(question, POOL_SIZE)]
        fused = fuse([vector_ranking(vector, POOL_SIZE), keyword_ids])
        fused = fused[:top_k] if mode == "hybrid" else fused[:RERANK_POOL]
        ids, scores = [cid for cid, _ in fused], dict(fused)

    passages = load_passages(ids, vector, scores)
    if mode == "rerank":
        passages = rerank(question, passages, top_k)
    return passages