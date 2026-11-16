import json
from collections import defaultdict

from app.chunking import chunk_pages
from app.cleaning import clean_page_text

pages_by_document = defaultdict(list)
with open("data/processed/pages.jsonl", encoding="utf-8") as f:
    for line in f:
        record = json.loads(line)
        cleaned = clean_page_text(record["text"])
        pages_by_document[record["document"]].append((record["page"], cleaned))

all_chunks = []
for name, pages in sorted(pages_by_document.items()):
    chunks = chunk_pages(pages)
    print(f"{name}: {len(chunks)} chunks")
    all_chunks.extend((name, c) for c in chunks)

sizes = [c.token_count for _, c in all_chunks]
print()
print("Total chunks:", len(all_chunks))
print("Tokens per chunk (smallest / average / largest):",
      min(sizes), round(sum(sizes) / len(sizes)), max(sizes))

name, sample = all_chunks[100]
print()
print(f"Sample chunk from {name}, pages {sample.page_start}-{sample.page_end}:")
print(sample.text[:700])