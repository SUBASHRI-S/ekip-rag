import json
from pathlib import Path
import fitz  # PyMuPDF

raw_dir = Path("data/raw")
out_dir = Path("data/processed")
out_dir.mkdir(parents=True, exist_ok=True)

pdf_files = sorted(raw_dir.glob("*.pdf"))
total_pages = 0
total_chars = 0
near_empty_pages = []

with open(out_dir / "pages.jsonl", "w", encoding="utf-8") as out:
    for pdf_path in pdf_files:
        doc = fitz.open(pdf_path)
        doc_chars = 0
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text()
            doc_chars += len(text)
            if len(text.strip()) < 100:
                near_empty_pages.append((pdf_path.name, page_number))
            record = {"document": pdf_path.name, "page": page_number, "text": text}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"{pdf_path.name}: {len(doc)} pages, {doc_chars} characters")
        total_pages += len(doc)
        total_chars += doc_chars

print()
print("Documents:", len(pdf_files))
print("Total pages:", total_pages)
print("Total characters:", total_chars)
print("Pages with almost no text:", len(near_empty_pages))