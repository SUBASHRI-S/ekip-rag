from app.rag import build_prompt, format_pages
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