import re
from dataclasses import dataclass

from openai import APIError

from app.llm import generate
from app.retrieval import Passage, search

NOT_FOUND = "The documents do not contain enough information to answer this."
LLM_DOWN = (
    "The answer service is busy right now. "
    "The most relevant passages are listed below."
)

SYSTEM_PROMPT = (
    "You answer questions about Reserve Bank of India (RBI) Master Directions "
    "using ONLY the numbered passages provided.\n"
    "Rules:\n"
    "1. Use only the passages. Never use outside knowledge or guess.\n"
    "2. After every statement, cite the passage number in plain square brackets, "
    "like [1] or [2][3]. Use no other citation style, and no markdown "
    "(no **bold**, no headings).\n"
    "3. The passages come from rules for different types of lender (commercial banks, "
    "NBFCs, small finance banks). Say which type of lender each statement applies to. "
    "If the question does not say which lender, and the rules differ, explain the difference.\n"
    "4. Include every number, limit, condition and exception the passages give for the "
    "question, for example different limits for different loan sizes or durations.\n"
    f'5. If the passages do not contain the answer, reply exactly: "{NOT_FOUND}"\n'
    "6. Keep the answer short and in plain English."
)


def format_pages(passage: Passage) -> str:
    if passage.page_start == passage.page_end:
        return str(passage.page_start)
    return f"{passage.page_start}-{passage.page_end}"


def build_prompt(question: str, passages: list[Passage]) -> str:
    blocks = []
    for number, p in enumerate(passages, start=1):
        blocks.append(
            f"[{number}] Source: {p.title} ({p.file_name}), pages {format_pages(p)}\n{p.text}"
        )
    return "Passages:\n\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}\n\nAnswer:"


def extract_citations(answer: str, max_number: int) -> list[int]:
    """Numbers of the passages an answer cites, in any bracket style the model uses."""
    numbers = set()
    for inside in re.findall(r"[\[【]([^\]】]*)[\]】]", answer):
        for digits in re.findall(r"\d+", inside):
            if 1 <= int(digits) <= max_number:
                numbers.add(int(digits))
    return sorted(numbers)


def clean_answer(text: str) -> str:
    """Rewrite citations like 【1†note】 as [1] and remove **bold** markers."""
    text = text.replace("**", "")
    return re.sub(r"【\s*(\d+)[^】]*】", r"[\1]", text)


@dataclass
class RagAnswer:
    question: str
    answer: str
    answerable: bool
    passages: list[Passage]

def answer_question(question: str, top_k: int = 5, mode: str = "hybrid") -> RagAnswer:
    passages = search(question, top_k=top_k, mode=mode)
    try:
        text = clean_answer(generate(build_prompt(question, passages), system=SYSTEM_PROMPT).strip())
    except APIError:
        text = LLM_DOWN  # the search still worked, so show the passages anyway
    answerable = text not in (NOT_FOUND, LLM_DOWN)
    return RagAnswer(question=question, answer=text, answerable=answerable, passages=passages)