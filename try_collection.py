import sys

from sqlalchemy import select

from app.db import SessionLocal
from app.models import Collection
from app.rag import answer_question, format_pages

name = sys.argv[1]
question = " ".join(sys.argv[2:])
with SessionLocal() as session:
    collection_id = session.scalar(select(Collection.id).where(Collection.name == name))
if collection_id is None:
    sys.exit(f"No collection named {name!r}")

result = answer_question(question, collection_id=collection_id)
print("Question:", question)
print("Answerable:", result.answerable)
print("\nAnswer:\n" + result.answer)
print("\nSources:")
for number, p in enumerate(result.passages, start=1):
    print(f"  [{number}] {p.file_name}, pages {format_pages(p)} (distance {p.distance:.3f})")