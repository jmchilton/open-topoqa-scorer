"""Hermetic tests for restrict_to_labels — no benchmark data / mkdssp needed."""

from open_topoqa_scorer.benchmark import DecoyLabel, restrict_to_labels


def _lab(target, model):
    return DecoyLabel(target=target, model=model, dockq=0.5, capri=1, pdb_path="")


def test_restrict_drops_superset_extras_and_keeps_order():
    # featurize_subset may hand back a cached superset (e.g. all of HAF2); after excluding a
    # target the caller must filter to the requested set or the exclusion silently no-ops.
    kept = [_lab("A", "m1"), _lab("B", "m1"), _lab("A", "m2"), _lab("7ALA", "m1")]
    graphs = ["gA1", "gB1", "gA2", "g7"]
    labels = [_lab("A", "m1"), _lab("A", "m2"), _lab("B", "m1")]  # 7ALA excluded
    g, k = restrict_to_labels(graphs, kept, labels)
    assert len(g) == 3  # the 7ALA extra was dropped
    assert [l.target for l in k] == ["A", "B", "A"]  # order follows graphs/kept, not labels
    assert g == ["gA1", "gB1", "gA2"]
    assert all((l.target, l.model) != ("7ALA", "m1") for l in k)


def test_restrict_empty_inputs():
    assert restrict_to_labels([], [], [_lab("A", "m1")]) == ([], [])


def test_restrict_matches_on_target_and_model_pair():
    # same model name under two targets must not cross-match
    kept = [_lab("A", "m1"), _lab("B", "m1")]
    graphs = ["gA", "gB"]
    g, k = restrict_to_labels(graphs, kept, [_lab("B", "m1")])
    assert g == ["gB"] and [(l.target, l.model) for l in k] == [("B", "m1")]
