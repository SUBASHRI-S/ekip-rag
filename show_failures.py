import csv
from pathlib import Path

with open("data/processed/eval_answers.csv", encoding="utf-8-sig") as f:
    rows = [r for r in csv.DictReader(f) if r["kind"] == "answerable"]

lines = []


def show(title, selected):
    lines.append(f"\n===== {title}: {len(selected)} =====")
    for r in selected:
        lines.append(f"\n{r['id']}: {r['question']}")
        lines.append(f"   gold:    {r['gold']}")
        lines.append(f"   cited:   {r['cited'] or 'none'}")
        lines.append(f"   sources: {r['sources']}")
        lines.append(f"   answer:  {' '.join(r['answer'].split())[:300]}")


show("Wrongly refused", [r for r in rows if r["answerable"] == "False"])
show("Answered without any citation", [r for r in rows if r["answerable"] == "True" and not r["cited"]])
show(
    "Cited, but not the correct place",
    [r for r in rows if r["answerable"] == "True" and r["cited"] and r["cites_gold"] != "True"],
)

text = "\n".join(lines)
print(text)
Path("data/processed/failures.txt").write_text(text, encoding="utf-8")