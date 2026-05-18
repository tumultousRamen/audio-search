from audio_search.search import rrf


def test_rrf_basic():
    lists = [
        ["A", "B", "C", "D"],   # rank: A=0, B=1, C=2, D=3
        ["B", "A", "E"],        # rank: B=0, A=1, E=2
    ]
    fused = rrf(lists, k=60, top_k=5)
    # A and B both appear in both lists; A in rank 0 and 1, B in rank 1 and 0
    # Both should be at the top with the same score, then C, D, E in some order
    ids = [doc_id for doc_id, _ in fused]
    assert set(ids[:2]) == {"A", "B"}, ids
    assert "E" in ids

    score_a = dict(fused)["A"]
    score_b = dict(fused)["B"]
    # 1/(60+0) + 1/(60+1) for A; 1/(60+1) + 1/(60+0) for B → equal
    assert abs(score_a - score_b) < 1e-9


def test_rrf_single_source():
    fused = rrf([["X", "Y", "Z"]], top_k=2)
    assert [d for d, _ in fused] == ["X", "Y"]


def test_rrf_empty():
    assert rrf([], top_k=5) == []
    assert rrf([[]], top_k=5) == []
