import sys

from app.retrieval import search

question = sys.argv[1]
keywords = [k.lower() for k in sys.argv[2:]]

passages = search(question, top_k=5)
with open("data/processed/last_passages.txt", "w", encoding="utf-8") as f:
    for number, p in enumerate(passages, start=1):
        found = [k for k in keywords if k in p.text.lower()]
        print(f"[{number}] {p.title}, pages {p.page_start}-{p.page_end} | contains: {found}")
        f.write(f"===== [{number}] {p.title}, pages {p.page_start}-{p.page_end} =====\n{p.text}\n\n")

print("\nFull text saved in data/processed/last_passages.txt")