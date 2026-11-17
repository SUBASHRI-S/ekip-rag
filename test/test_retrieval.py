from app.retrieval import fuse, tokenize


def test_tokenize_lowercases_splits_and_drops_common_words():
    assert tokenize("The Cooling-off period!") == ["cooling", "off", "period"]


def test_fuse_rewards_chunks_ranked_high_by_both_lists():
    ranked = fuse([[1, 2, 3], [3, 1, 4]])
    assert [chunk_id for chunk_id, _ in ranked] == [1, 3, 2, 4]