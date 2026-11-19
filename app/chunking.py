import os
from bisect import bisect_right
from dataclasses import dataclass

from transformers import AutoTokenizer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
CHUNK_TOKENS = int(os.environ.get("CHUNK_TOKENS", "500"))
OVERLAP_TOKENS = int(os.environ.get("OVERLAP_TOKENS", "100"))

_tokenizer = None


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _tokenizer.model_max_length = 10_000_000  # we cut the text ourselves
    return _tokenizer


@dataclass
class ChunkData:
    chunk_index: int
    page_start: int
    page_end: int
    text: str
    token_count: int


def chunk_pages(pages, chunk_tokens=CHUNK_TOKENS, overlap_tokens=OVERLAP_TOKENS):
    """pages: list of (page_number, cleaned_text). Returns a list of ChunkData."""
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap must be smaller than the chunk size")

    # 1. join the pages into one text and remember where each page starts
    full_text = ""
    page_starts = []
    page_numbers = []
    for page_number, text in pages:
        if not text:
            continue
        page_starts.append(len(full_text))
        page_numbers.append(page_number)
        full_text += text + "\n\n"
    if not full_text:
        return []

    def page_at(position):
        return page_numbers[bisect_right(page_starts, position) - 1]

    # 2. turn the text into tokens, keeping each token's position in the text
    encoded = get_tokenizer()(
        full_text, add_special_tokens=False, return_offsets_mapping=True
    )
    offsets = encoded["offset_mapping"]
    word_ids = encoded.word_ids()
    total = len(offsets)

    def starts_a_word(i):
        return i == 0 or i >= total or word_ids[i] != word_ids[i - 1]

    # 3. slide a window over the tokens
    step = chunk_tokens - overlap_tokens
    chunks = []
    start = 0
    while start < total:
        while start < total and not starts_a_word(start):
            start += 1  # never begin in the middle of a word
        if start >= total:
            break
        end = min(start + chunk_tokens, total)
        while end > start + 1 and not starts_a_word(end):
            end -= 1  # never end in the middle of a word
        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]
        chunks.append(
            ChunkData(
                chunk_index=len(chunks),
                page_start=page_at(char_start),
                page_end=page_at(char_end - 1),
                text=full_text[char_start:char_end],
                token_count=end - start,
            )
        )
        if end >= total:
            break
        start += step
    return chunks