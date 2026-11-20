import os
import re
from collections import defaultdict
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sqlalchemy import func, select

from app.chunking import MODEL_NAME
from app.db import SessionLocal
from app.models import Chunk, Collection, Document

# the bge model expects this prefix on questions, but not on stored passages
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "and", "or", "is", "are", "for",
    "on", "by", "be", "as", "at", "with", "that", "this", "it",
}
POOL_SIZE = 50  # how many candidates each search contributes to the merge
RRF_K = 60
RERANK_POOL = int(os.environ.get("RERANK_POOL", "10"))  # candidates the reranker re-reads
RERANKER_NAME = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-base")
DEFAULT_COLLECTION = "RBI Master Directions"

ENTITY_PATTERNS = [
    ("small_finance_bank", r"\bsmall finance banks?\b|\bsfbs?\b"),
    ("nbfc", r"\bnbfcs?\b|\bnon[- ]banking financial compan(?:y|ies)\b"),
    ("commercial_bank", r"\bcommercial banks?\b"),
]

_model = None
_indexes = {}
_reranker = None


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


@dataclass
class Index:
    size: int
    ids: list
    entity_of: list
    document_of: list
    bm25: BM25Okapi
    entities: set


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, drop very common words."""
    return [w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS]


def detect_entity_types(question: str) -> list[str] | None:
    """Find which lender types the question names. None means it names none."""
    text = question.lower()
    found = [name for name, pattern in ENTITY_PATTERNS if re.search(pattern, text)]
    return found or None


def embed_query(question: str) -> list[float]:
    return get_model().encode(QUERY_PREFIX + question, normalize_embeddings=True).tolist()


def resolve_collection(collection_id):
    """No collection given means the RBI collection."""
    if collection_id is not None:
        return collection_id
    with SessionLocal() as session:
        return session.scalar(select(Collection.id).where(Collection.name == DEFAULT_COLLECTION))


def get_collection_name(collection_id):
    with SessionLocal() as session:
        return session.scalar(select(Collection.name).where(Collection.id == collection_id))


def in_collection(collection_id):
    if collection_id is None:
        return Document.collection_id.is_(None)
    return Document.collection_id == collection_id


def get_index(collection_id):
    """The keyword index of one collection. Rebuilt when its number of chunks changes."""
    with SessionLocal() as session:
        size = session.scalar(
            select(func.count())
            .select_from(Chunk)
            .join(Document, Chunk.document_id == Document.id)
            .where(in_collection(collection_id))
        )
        cached = _indexes.get(collection_id)
        if cached is not None and cached.size == size:
            return cached
        if not size:
            return None
        rows = session.execute(
            select(Chunk.id, Chunk.text, Document.entity_type, Document.id)
            .join(Document, Chunk.document_id == Document.id)
            .where(in_collection(collection_id))
            .order_by(Chunk.id)
        ).all()
    index = Index(
        size=size,
        ids=[r[0] for r in rows],
        entity_of=[r[2] for r in rows],
        document_of=[r[3] for r in rows],
        bm25=BM25Okapi([tokenize(r[1]) for r in rows]),
        entities={r[2] for r in rows},
    )
    _indexes[collection_id] = index
    return index


def vector_ranking(vector, k, collection_id, entity_types=None, document_ids=None) -> list[int]:
    with SessionLocal() as session:
        distance = Chunk.embedding.cosine_distance(vector)
        query = (
            select(Chunk.id)
            .join(Document, Chunk.document_id == Document.id)
            .where(in_collection(collection_id))
        )
        if entity_types:
            query = query.where(Document.entity_type.in_(entity_types))
        if document_ids:
            query = query.where(Document.id.in_(document_ids))
        return list(session.scalars(query.order_by(distance).limit(k)))


def bm25_ranking(question, k, index, entity_types=None, document_ids=None) -> list[tuple[int, float]]:
    scores = index.bm25.get_scores(tokenize(question))
    order = sorted(range(len(index.ids)), key=lambda i: scores[i], reverse=True)
    result = []
    for i in order:
        if scores[i] <= 0 or len(result) == k:
            break
        if entity_types and index.entity_of[i] not in entity_types:
            continue
        if document_ids and index.document_of[i] not in document_ids:
            continue
        result.append((index.ids[i], float(scores[i])))
    return result


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


def search(
    question: str,
    top_k: int = 5,
    mode: str = "hybrid",
    entity_types: list[str] | None = None,
    auto_filter: bool = True,
    collection_id: int | None = None,
    document_ids: list[int] | None = None,
) -> list[Passage]:
    """mode is 'vector', 'bm25', 'hybrid' or 'rerank'.

    The search runs inside one collection (the RBI one if none is given).
    The lender-type filter is only used when the collection has documents of that type.
    """
    if mode not in ("vector", "bm25", "hybrid", "rerank"):
        raise ValueError("mode must be 'vector', 'bm25', 'hybrid' or 'rerank'")
    cid = resolve_collection(collection_id)
    index = get_index(cid)
    if index is None:
        return []
    if entity_types is None and auto_filter:
        detected = detect_entity_types(question) or []
        entity_types = [t for t in detected if t in index.entities] or None
    vector = embed_query(question)

    if mode == "vector":
        ids, scores = vector_ranking(vector, top_k, cid, entity_types, document_ids), {}
    elif mode == "bm25":
        ranked = bm25_ranking(question, top_k, index, entity_types, document_ids)
        ids, scores = [chunk_id for chunk_id, _ in ranked], dict(ranked)
    else:
        keyword_ids = [c for c, _ in bm25_ranking(question, POOL_SIZE, index, entity_types, document_ids)]
        fused = fuse([vector_ranking(vector, POOL_SIZE, cid, entity_types, document_ids), keyword_ids])
        fused = fused[:top_k] if mode == "hybrid" else fused[:RERANK_POOL]
        ids, scores = [chunk_id for chunk_id, _ in fused], dict(fused)

    passages = load_passages(ids, vector, scores)
    if mode == "rerank":
        passages = rerank(question, passages, top_k)
    return passages