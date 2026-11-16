import json
from collections import Counter

with open("data/processed/pages.jsonl", encoding="utf-8") as f:
    pages = [json.loads(line) for line in f]

print("Pages with almost no text:")
for p in pages:
    if len(p["text"].strip()) < 100:
        print(" ", p["document"], "page", p["page"], "->", repr(p["text"].strip()[:60]))

# a line that appears on many pages is probably a header or footer
line_counts = Counter()
for p in pages:
    lines = set(l.strip() for l in p["text"].splitlines() if l.strip())
    for line in lines:
        line_counts[line] += 1

print()
print("Most repeated lines (likely headers or footers):")
for line, count in line_counts.most_common(10):
    print(f"  {count:5d} pages: {line[:70]}")