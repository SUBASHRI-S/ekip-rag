from app.rag import build_prompt, clean_answer, extract_citations, format_pages
from app.retrieval import Passage


def make_passage(start, end, text="Some rule text."):
    return Passage(
        chunk_id=1, file_name="rbi_nbfc_kyc.pdf", title="NBFC - Know Your Customer",
        entity_type="nbfc", page_start=start, page_end=end, text=text, distance=0.2,
    )


def test_single_page_and_page_range_formatting():
    assert format_pages(make_passage(5, 5)) == "5"
    assert format_pages(make_passage(5, 7)) == "5-7"


def test_prompt_numbers_passages_and_contains_question():
    prompt = build_prompt("What is KYC?", [make_passage(1, 2, "First."), make_passage(3, 3, "Second.")])
    assert "[1] Source: NBFC - Know Your Customer" in prompt
    assert "[2] Source:" in prompt
    assert "pages 1-2" in prompt
    assert prompt.endswith("Question: What is KYC?\n\nAnswer:")


def test_extract_citations_accepts_different_bracket_styles():
    assert extract_citations("Yes [1][3].", 5) == [1, 3]
    assert extract_citations("Yes \u30102\u3011 and [4, 5]", 5) == [2, 4, 5]
    assert extract_citations("Effective [2025] only", 5) == []


def test_clean_answer_normalises_citations_and_removes_bold():
    assert clean_answer("The **limit** is 5%\u30101\u2020note\u3011.") == "The limit is 5%[1]."