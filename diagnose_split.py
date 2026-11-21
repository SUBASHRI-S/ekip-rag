import os
import sys

os.environ["RERANK_POOL"] = "30"  # for this diagnosis only: let the reranker read 30 candidates

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Chunk, Collection, Document
from app.rag import split_question
from app.retrieval import search


def flat(text):
    return " ".join(text.lower().split())


collection_name = sys.argv[1]
phrase = sys.argv[2].lower()
question = " ".join(sys.argv[3:])

with SessionLocal() as session:
    collection_id = session.scalar(select(Collection.id).where(Collection.name == collection_name))
    rows = session.execute(
        select(Chunk.id, Document.file_name, Chunk.page_start, Chunk.page_end, Chunk.text)
        .join(Document, Chunk.document_id == Document.id)
        .where(Document.collection_id == collection_id)
    ).all()

holders = {r[0]: (r[1], r[2], r[3]) for r in rows if phrase in flat(r[4])}
print(f'Chunks in "{collection_name}" that contain "{phrase}": {len(holders)}')
for chunk_id, (name, start, end) in holders.items():
    print(f"   chunk {chunk_id}: {name} pages {start}-{end}")

parts = split_question(question)
print("\nThe question was split into:")
for part in parts:
    print("  -", part)


def rank_of(passages):
    for rank, p in enumerate(passages, start=1):
        if p.chunk_id in holders:
            return rank
    return None


print("\nWhere does a chunk containing the phrase rank?")
for query in [question] + parts:
    print(f"\n{query}")
    for mode in ("hybrid", "rerank"):
        ranked = search(query, top_k=30, mode=mode, collection_id=collection_id)
        found = rank_of(ranked)
        print(f"   {mode:7}: " + (f"rank {found} of {len(ranked)}" if found else f"not in the top {len(ranked)}"))