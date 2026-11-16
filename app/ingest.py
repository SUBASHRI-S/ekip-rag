import json
from collections import defaultdict
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sqlalchemy import select

from app.chunking import MODEL_NAME, chunk_pages
from app.cleaning import clean_page_text
from app.db import SessionLocal
from app.models import Chunk, Document

PAGES_FILE = Path("data/processed/pages.jsonl")
SOURCES_FILE = Path("data/SOURCES.md")
ENTITY_TYPES = {"cb": "commercial_bank", "nbfc": "nbfc", "sfb": "small_finance_bank"}


def read_sources():
    """Read data/SOURCES.md into {file_name: {title, updated_as_on, source_url}}."""
    sources = {}
    for line in SOURCES_FILE.read_text(encoding="utf-8").splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) != 5 or not cells[0].endswith(".pdf"):
            continue
        file_name, title, _issued, updated, url = cells
        sources[file_name] = {
            "title": title,
            "updated_as_on": None if updated == "-" else updated,
            "source_url": url,
        }
    return sources


def read_pages():
    pages = defaultdict(list)
    with open(PAGES_FILE, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            pages[record["document"]].append(
                (record["page"], clean_page_text(record["text"]))
            )
    return pages


def entity_and_topic(file_name):
    # "rbi_nbfc_credit_facilities.pdf" -> ("nbfc", "credit_facilities")
    parts = file_name.removesuffix(".pdf").split("_")
    return ENTITY_TYPES[parts[1]], "_".join(parts[2:])


def ingest():
    sources = read_sources()
    pages_by_document = read_pages()
    model = SentenceTransformer(MODEL_NAME)

    with SessionLocal() as session:
        already_loaded = set(session.scalars(select(Document.file_name)))

        for file_name, pages in sorted(pages_by_document.items()):
            if file_name in already_loaded:
                print(f"skip    {file_name} (already in the database)")
                continue

            chunks = chunk_pages(pages)
            vectors = model.encode(
                [c.text for c in chunks],
                batch_size=32,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            entity_type, topic = entity_and_topic(file_name)
            info = sources.get(file_name, {})

            document = Document(
                file_name=file_name,
                title=info.get("title", file_name),
                entity_type=entity_type,
                topic=topic,
                updated_as_on=info.get("updated_as_on"),
                source_url=info.get("source_url"),
                page_count=len(pages),
            )
            for chunk, vector in zip(chunks, vectors):
                document.chunks.append(
                    Chunk(
                        chunk_index=chunk.chunk_index,
                        page_start=chunk.page_start,
                        page_end=chunk.page_end,
                        text=chunk.text.replace("\x00", ""),
                        token_count=chunk.token_count,
                        embedding=vector.tolist(),
                    )
                )
            session.add(document)
            session.commit()  # one document at a time, so a crash loses little
            print(f"loaded  {file_name}: {len(chunks)} chunks", flush=True)


if __name__ == "__main__":
    ingest()