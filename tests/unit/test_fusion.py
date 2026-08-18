from neuralake.core.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_single_list():
    results = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    fused = reciprocal_rank_fusion(results)
    assert fused[0][0] == "a"
    assert fused[1][0] == "b"
    assert fused[2][0] == "c"


def test_rrf_two_lists_overlap():
    list1 = [("a", 0.9), ("b", 0.7), ("c", 0.5)]
    list2 = [("b", 0.95), ("a", 0.6), ("d", 0.4)]
    fused = reciprocal_rank_fusion(list1, list2)
    ids = [x[0] for x in fused]
    assert "a" in ids
    assert "b" in ids
    assert "d" in ids


def test_rrf_empty():
    fused = reciprocal_rank_fusion([])
    assert fused == []


def test_rrf_boosts_overlap():
    list1 = [("a", 0.9), ("b", 0.8)]
    list2 = [("a", 0.9), ("c", 0.8)]
    fused = reciprocal_rank_fusion(list1, list2)
    assert fused[0][0] == "a"
