import sys

from app.rag import answer_question, format_pages

question = " ".join(sys.argv[1:]) or "How long is the cooling-off period for a digital loan?"
result = answer_question(question)

print("Question:", result.question)
print("Answerable:", result.answerable)
print("\nAnswer:\n" + result.answer)

print("\nSources:")
for number, p in enumerate(result.passages, start=1):
    print(f"  [{number}] {p.title}, pages {format_pages(p)} (distance {p.distance:.3f})")