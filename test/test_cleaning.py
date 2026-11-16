from app.cleaning import clean_page_text


def test_removes_leading_page_number():
    text = "149\n\nAnnex - I\nData to be submitted"
    assert clean_page_text(text) == "Annex - I\nData to be submitted"


def test_joins_clause_markers_with_their_sentence():
    text = "12\n(2)\n(a)\nAn NBFC shall report."
    assert clean_page_text(text) == "(2) (a) An NBFC shall report."


def test_drops_table_of_contents_dots():
    text = "3\nIntroduction ............ 4\nChapter I\n(1)\nPreliminary"
    result = clean_page_text(text)
    assert "...." not in result
    assert "(1) Preliminary" in result


def test_page_with_only_a_number_becomes_empty():
    assert clean_page_text("32\n") == ""


def test_deleted_placeholder_page_becomes_empty():
    text = "150\nAnnex - II\n32[*****]\n32 Deleted with effect from April 1, 2026"
    assert clean_page_text(text) == ""