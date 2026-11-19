import csv
from pathlib import Path

PATH = Path("data/processed/eval_paraphrased.csv")

FIXES = {
    "c23": "When a project starts commercial operation, what should an NBFC do with the additional provisions it had to set aside for that project loan under paragraph 118?",
    "c34": "What stress tests must NBFCs run on their securitisation deals?",
    "c40": "What is the highest amount a commercial bank can collect on an account-payee cheque made out to a cooperative credit society's account, and on what condition?",
    "c42": "What restriction does the RBI put on a commercial bank's term loans for projects run by public-sector units?",
    "c49": "At the end of the review period, what additional provision must a commercial bank make for the restructured accounts described in paragraphs 61 and 62?",
}

with open(PATH, encoding="utf-8-sig") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fields = reader.fieldnames

for row in rows:
    if row["id"] in FIXES:
        print(f"{row['id']}\n   was: {row['question']}\n   now: {FIXES[row['id']]}\n")
        row["question"] = FIXES[row["id"]]

with open(PATH, "w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)

print("Pairs I have not reviewed yet - please check that meaning and lender type match:")
for row in rows:
    if row["id"] in ("c01", "c04"):
        print(f"{row['id']}\n   original:    {row['original_question']}\n   paraphrased: {row['question']}\n")