import re
import sys

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Chunk, Document

phrase = sys.argv[1]

with SessionLocal() as session:
    rows = session.execute(
        select(Chunk, Document.file_name)
        .join(Document, Chunk.document_id == Document.id)
        .where(Chunk.text.ilike(f"%{phrase}%"))
        .order_by(Document.file_name, Chunk.chunk_index)
    ).all()

print(f'"{phrase}" appears in {len(rows)} chunks\n')
for chunk, file_name in rows:
    flat = " ".join(chunk.text.split())
    for match in re.finditer(re.escape(phrase), flat, flags=re.IGNORECASE):
        start = max(0, match.start() - 150)
        print(f"{file_name}, pages {chunk.page_start}-{chunk.page_end}, chunk {chunk.id}")
        print("   ..." + flat[start:match.end() + 250] + "...\n")
        break  # one example per chunk