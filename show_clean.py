import json
from app.cleaning import clean_page_text

with open("data/processed/pages.jsonl", encoding="utf-8") as f:
    pages = [json.loads(line) for line in f]

page = pages[300]
print("---- BEFORE ----")
print(page["text"][:600])
print("---- AFTER ----")
print(clean_page_text(page["text"])[:600])

kept = sum(1 for p in pages if clean_page_text(p["text"]))
print()
print(f"Pages with content after cleaning: {kept} of {len(pages)}")