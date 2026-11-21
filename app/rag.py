import os
import re
import time
from dataclasses import dataclass

from openai import APIError
from app import retrieval

from app.llm import generate
from app.retrieval import (
    DEFAULT_COLLECTION,
    Passage,
    detect_entity_types,
    get_collection_name,
    resolve_collection,
    search,
    search_balanced,
)

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

SYSTEM_PROMPT_GENERAL = (
    "You answer questions using ONLY the numbered passages provided. "
    "They come from documents the user uploaded.\n"
    "Rules:\n"
    "1. Use only the passages. Never use outside knowledge or guess.\n"
    "2. After every statement, cite the passage number in plain square brackets, "
    "like [1] or [2][3]. Use no other citation style, and no markdown "
    "(no **bold**, no headings).\n"
    "3. When the passages come from more than one document, say which document each "
    "statement comes from, and point out where the documents agree or differ.\n"
    "4. Include every number, limit, condition and exception the passages give for the question.\n"
    "5. If the passages answer only part of the question, answer that part and say plainly "
    "which part the passages do not cover.\n"
    f'6. Only if the passages answer none of the question, reply exactly: "{NOT_FOUND}"\n'
    "7. Keep the answer short and in plain English."
)

SPLIT_PROMPT = (
    "Does the question below ask for two or more separate facts? If it does, rewrite it as "
    "separate, self-contained short questions, one per line, at most three, with no numbering "
    "or bullets. Keep every name, abbreviation and technical term from the original question "
    "in the part where it belongs (for example NBFC, SPE, credit card). "
    "If it asks for only one thing, repeat it unchanged on one line. "
    "Reply with only the questions.\n\nQuestion: {question}"
)

MULTI_PART = re.compile(r"\b(and|both|compare|comparison|difference|differ|versus|vs)\b", re.IGNORECASE)


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


def needs_splitting(question: str) -> bool:
    """A cheap check, so simple questions never cost an extra LLM call."""
    return bool(MULTI_PART.search(question)) or question.count("?") > 1


def split_question(question: str) -> list[str]:
    """Break a multi-part question into up to three short questions."""
    if not needs_splitting(question):
        return [question]
    try:
        reply = generate(SPLIT_PROMPT.format(question=question)).strip()
    except APIError:
        return [question]
    parts = [re.sub(r"^[\s\-*\d.)]+", "", line).strip() for line in reply.splitlines()]
    parts = [p for p in parts if len(p) > 8][:3]
    return parts or [question]


def retrieve_for_parts(parts: list[str], cid: int, per_part: int = 5, limit: int = 10) -> list[Passage]:
    """Search each part with the reranker over 30 candidates, then merge best-first without repeats."""
    previous = retrieval.RERANK_POOL
    retrieval.RERANK_POOL = 30
    try:
        lists = [search(part, top_k=per_part, mode="rerank", collection_id=cid) for part in parts]
    finally:
        retrieval.RERANK_POOL = previous
    merged, seen = [], set()
    for rank in range(per_part):
        for results in lists:
            if rank < len(results) and results[rank].chunk_id not in seen:
                seen.add(results[rank].chunk_id)
                merged.append(results[rank])
    return merged[:limit]


@dataclass
class RagAnswer:
    question: str
    answer: str
    answerable: bool
    passages: list[Passage]
    trace: dict


def answer_question(
    question: str, top_k: int = 5, mode: str = "hybrid", collection_id: int | None = None
) -> RagAnswer:
    started = time.perf_counter()
    cid = resolve_collection(collection_id)
    is_rbi = get_collection_name(cid) == DEFAULT_COLLECTION
    lender_filter = detect_entity_types(question) if is_rbi else None

    parts, balanced = [question], False
    if is_rbi:
        passages = search(question, top_k=top_k, mode=mode, collection_id=cid)
    else:
        parts = split_question(question)
        if len(parts) > 1:
          passages = retrieve_for_parts(parts, cid)
        else:
            passages, balanced = search_balanced(
                question, top_k=top_k, mode=mode, collection_id=cid
            )
    search_seconds = time.perf_counter() - started

    llm_started = time.perf_counter()
    try:
        prompt_rules = SYSTEM_PROMPT if is_rbi else SYSTEM_PROMPT_GENERAL
        text = clean_answer(generate(build_prompt(question, passages), system=prompt_rules).strip())
    except APIError:
        text = LLM_DOWN  # the search still worked, so show the passages anyway
    llm_seconds = time.perf_counter() - llm_started

    trace = {
        "mode": mode,
        "collection_id": cid,
        "sub_questions": parts if len(parts) > 1 else None,
        "balanced_across_documents": balanced,
        "lender_filter": lender_filter,
        "search_seconds": round(search_seconds, 3),
        "llm_seconds": round(llm_seconds, 3),
        "total_seconds": round(time.perf_counter() - started, 3),
        "llm_model": os.environ.get("LLM_MODEL"),
        "retrieved": [
            {
                "rank": rank,
                "chunk_id": p.chunk_id,
                "score": round(p.score, 4),
                "distance": round(p.distance, 4),
            }
            for rank, p in enumerate(passages, start=1)
        ],
    }
    answerable = text not in (NOT_FOUND, LLM_DOWN)
    return RagAnswer(
        question=question, answer=text, answerable=answerable, passages=passages, trace=trace
    )