from sqlalchemy import select

from app.db import Base, SessionLocal, engine
from app.models import EMBEDDING_DIMENSIONS, Chunk, Document

Base.metadata.create_all(engine)  # creates the tables if they don't exist

# remove leftovers from an earlier run
with SessionLocal() as session:
    for old in session.scalars(select(Document).where(Document.file_name == "fake.pdf")):
        session.delete(old)
    session.commit()

# insert one fake document with one fake chunk
with SessionLocal() as session:
    doc = Document(
        file_name="fake.pdf", title="Fake document",
        entity_type="nbfc", topic="test", page_count=1,
    )
    doc.chunks.append(
        Chunk(
            chunk_index=0, page_start=1, page_end=1,
            text="Hello vector world", token_count=3,
            embedding=[0.1] * EMBEDDING_DIMENSIONS,
        )
    )
    session.add(doc)
    session.commit()

# read it back, then delete it
with SessionLocal() as session:
    chunk = session.scalars(select(Chunk)).first()
    print(chunk.document.file_name, "|", chunk.text, "|", len(chunk.embedding), "numbers")
    session.delete(chunk.document)
    session.commit()