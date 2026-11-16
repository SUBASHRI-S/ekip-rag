import pytest

from app.chunking import chunk_pages, get_tokenizer


def make_text(n_words):
    return " ".join(f"word{i}" for i in range(n_words))


def test_empty_pages_give_no_chunks():
    assert chunk_pages([(1, ""), (2, "")]) == []


def test_short_text_is_one_chunk():
    chunks = chunk_pages([(1, "Short text.")])
    assert len(chunks) == 1
    assert chunks[0].text == "Short text."
    assert chunks[0].page_start == chunks[0].page_end == 1


def test_chunks_fit_the_embedding_model_limit():
    tokenizer = get_tokenizer()
    chunks = chunk_pages([(1, make_text(3000))])
    assert len(chunks) > 3
    for chunk in chunks:
        assert chunk.token_count <= 500
        assert len(tokenizer(chunk.text)["input_ids"]) <= 512


def test_neighbouring_chunks_overlap():
    chunks = chunk_pages([(1, make_text(3000))])
    assert chunks[1].text[:40] in chunks[0].text


def test_chunks_remember_their_pages():
    pages = [(1, "alpha " * 600), (2, "omega " * 600)]
    chunks = chunk_pages(pages)
    assert chunks[0].page_start == 1
    assert chunks[-1].page_end == 2
    assert any(c.page_start == 1 and c.page_end == 2 for c in chunks)


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_pages([(1, "text")], chunk_tokens=100, overlap_tokens=100)