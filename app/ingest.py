import argparse
import json
from collections import defaultdict
from pathlib import Path

import pymupdf
from sqlalchemy import select

from app.chunking import chunk_pages
from app.cleaning import clean_page_text
from app.db import SessionLocal
from app.models import Chunk, Collection, Document
from app.retrieval import DEFAULT_COLLECTION, get_model

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
            pages[record["document"]].append((record["page"], clean_page_text(record["text"])))
    return pages


def entity_and_topic(file_name):
    # "rbi_nbfc_credit_facilities.pdf" -> ("nbfc", "credit_facilities")
    parts = file_name.removesuffix(".pdf").split("_")
    return ENTITY_TYPES[parts[1]], "_".join(parts[2:])


def document_exists(collection_name, file_name):
    with SessionLocal() as session:
        found = session.scalar(
            select(Document.id)
            .join(Collection, Document.collection_id == Collection.id)
            .where(Collection.name == collection_name, Document.file_name == file_name)
        )
        return found is not None


def store_document(
    collection_name, file_name, pages, title=None, entity_type="uploaded",
    topic="uploaded", updated_as_on=None, source_url=None,
):
    """Chunk, embed and save one document. Returns (document_id, chunk_count), or None if it exists."""
    if document_exists(collection_name, file_name):
        return None
    chunks = chunk_pages(pages)
    if not chunks:
        raise ValueError(f"No readable text found in {file_name}. Scanned PDFs are not supported.")
    vectors = get_model().encode(
        [c.text for c in chunks], batch_size=32, normalize_embeddings=True, show_progress_bar=False
    )
    with SessionLocal() as session:
        collection = session.scalar(select(Collection).where(Collection.name == collection_name))
        if collection is None:
            collection = Collection(name=collection_name)
            session.add(collection)
            session.flush()
        document = Document(
            collection_id=collection.id,
            file_name=file_name,
            title=title or file_name,
            entity_type=entity_type,
            topic=topic,
            updated_as_on=updated_as_on,
            source_url=source_url,
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
        session.commit()
        return document.id, len(chunks)


def ingest_pdf(path, collection_name):
    """Read one PDF file and store it in a collection."""
    path = Path(path)
    with pymupdf.open(path) as doc:
        pages = [(number, clean_page_text(page.get_text())) for number, page in enumerate(doc, start=1)]
    title = path.stem.replace("_", " ").replace("-", " ").strip()
    return store_document(collection_name, path.name, pages, title=title)


def ingest_rbi():
    """The 21 RBI documents, from the saved page text and SOURCES.md."""
    sources = read_sources()
    for file_name, pages in sorted(read_pages().items()):
        entity_type, topic = entity_and_topic(file_name)
        info = sources.get(file_name, {})
        result = store_document(
            DEFAULT_COLLECTION, file_name, pages,
            title=info.get("title", file_name), entity_type=entity_type, topic=topic,
            updated_as_on=info.get("updated_as_on"), source_url=info.get("source_url"),
        )
        print(f"skip    {file_name}" if result is None else f"loaded  {file_name}: {result[1]} chunks", flush=True)

def ingest_pdf_bytes(data: bytes, file_name: str, collection_name: str):
    """Store a PDF that arrived as raw bytes (an upload). Returns (document_id, chunks) or None."""
    file_name = Path(file_name).name  # never trust a path sent by a browser
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        pages = [(number, clean_page_text(page.get_text())) for number, page in enumerate(doc, start=1)]
    title = Path(file_name).stem.replace("_", " ").replace("-", " ").strip()
    return store_document(collection_name, file_name, pages, title=title)


def main():
    parser = argparse.ArgumentParser(description="Load documents into the database.")
    parser.add_argument("pdfs", nargs="*", help="PDF files to add (leave empty to load the RBI set)")
    parser.add_argument("--collection", help="collection name, required when PDF files are given")
    args = parser.parse_args()

    if not args.pdfs:
        ingest_rbi()
        return
    if not args.collection:
        parser.error("--collection is required when you give PDF files")
    for pdf in args.pdfs:
        try:
            result = ingest_pdf(pdf, args.collection)
        except ValueError as error:
            print(error)
            continue
        print(f"skip    {pdf} (already in this collection)" if result is None
              else f"loaded  {pdf}: {result[1]} chunks", flush=True)


if __name__ == "__main__":
    main()