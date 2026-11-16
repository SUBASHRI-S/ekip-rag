import sys

from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from app.chunking import MODEL_NAME
from app.db import SessionLocal
from app.models import Chunk, Document

# the bge model expects this prefix on questions, but not on the stored passages
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

question = " ".join(sys.argv[1:]) or "What is the maximum loan to value ratio for gold loans?"
model = SentenceTransformer(MODEL_NAME)
query_vector = model.encode(QUERY_PREFIX + question, normalize_embeddings=True).tolist()

with SessionLocal() as session:
    distance = Chunk.embedding.cosine_distance(query_vector)
    rows = session.execute(
        select(Chunk, Document.file_name, distance.label("distance"))
        .join(Document, Chunk.document_id == Document.id)
        .order_by(distance)
        .limit(5)
    ).all()

print("Question:", question)
for chunk, file_name, dist in rows:
    print(f"\n{file_name}, pages {chunk.page_start}-{chunk.page_end}, distance {dist:.3f}")
    print(chunk.text[:300].replace("\n", " "))