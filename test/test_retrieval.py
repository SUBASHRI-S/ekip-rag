from app.retrieval import detect_entity_types, fuse, tokenize


def test_tokenize_lowercases_splits_and_drops_common_words():
    assert tokenize("The Cooling-off period!") == ["cooling", "off", "period"]


def test_fuse_rewards_chunks_ranked_high_by_both_lists():
    ranked = fuse([[1, 2, 3], [3, 1, 4]])
    assert [chunk_id for chunk_id, _ in ranked] == [1, 3, 2, 4]


def test_detects_one_lender_type():
    assert detect_entity_types("What is the LTV for gold loans given by NBFCs?") == ["nbfc"]
    assert detect_entity_types("KYC rules for small finance banks") == ["small_finance_bank"]
    assert detect_entity_types("Do commercial banks need a cooling-off period?") == ["commercial_bank"]


def test_no_lender_type_means_no_filter():
    assert detect_entity_types("What is a lending service provider?") is None
    assert set(detect_entity_types("Compare NBFCs and commercial banks")) == {"nbfc", "commercial_bank"}