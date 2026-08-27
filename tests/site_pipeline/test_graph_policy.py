from sfm_diagnosis.site_pipeline.graph import (
    articulation_points,
    bridge_edges,
    connected_components,
    replacement_paths,
    star_overlap,
)
from sfm_diagnosis.site_pipeline.policy import (
    Candidate,
    diagnostic_select,
    reinforce_bridges,
    assign_roles,
)


def test_graph_diagnosis_and_alternate_paths():
    edges = [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "e")]
    assert connected_components(["a", "b", "c", "d", "e"], edges) == (("a", "b", "c", "d", "e"),)
    assert articulation_points(edges) == {"a"}
    assert bridge_edges(edges) == {("a", "e")}
    assert replacement_paths(edges, ("a", "b")) == 1
    assert replacement_paths(edges, ("a", "e")) == 0


def test_star_overlap_is_symmetric_and_uses_neighbors():
    graph = {"a": {"b", "c"}, "b": {"a", "c", "d"}, "c": {"a", "b"}}
    assert star_overlap(graph, "a", "b") == 1
    assert star_overlap(graph, "b", "a") == 1


def test_selection_keeps_connectivity_and_does_not_promote_retrieval_alone():
    candidates = [
        Candidate("a", coverage={"north"}, geometry=0.9, connectivity=0.1, cost=1),
        Candidate("b", coverage={"south"}, geometry=0.1, connectivity=0.9, cost=1),
        Candidate(
            "retrieval-only",
            coverage={"south"},
            geometry=1.0,
            connectivity=1.0,
            cost=1,
            is_retrieval_only=True,
        ),
    ]
    result = diagnostic_select(candidates, required_coverage={"north", "south"}, budget=2)
    assert {item.id for item in result.selected} == {"a", "b"}
    assert "retrieval-only" not in {item.id for item in result.selected}


def test_required_coverage_cannot_be_starved_by_high_connectivity_candidates():
    candidates = [
        Candidate("north-a", coverage={"north"}, connectivity=100, cost=1),
        Candidate("north-b", coverage={"north"}, connectivity=99, cost=1),
        Candidate("south", coverage={"south"}, connectivity=0, cost=1),
    ]

    result = diagnostic_select(
        candidates,
        required_coverage={"north", "south"},
        budget=2,
        require_connected=False,
    )

    assert {item.id for item in result.selected} == {"north-a", "south"}
    assert result.warnings == ()


def test_selection_can_fill_a_target_cost_after_coverage_is_satisfied():
    candidates = [
        Candidate("north-a", coverage={"north"}, connectivity=3, cost=1),
        Candidate("south", coverage={"south"}, connectivity=2, cost=1),
        Candidate("north-b", coverage={"north"}, connectivity=1, cost=1),
    ]

    result = diagnostic_select(
        candidates,
        required_coverage={"north", "south"},
        budget=3,
        target_cost=3,
        require_connected=False,
    )

    assert {item.id for item in result.selected} == {"north-a", "north-b", "south"}
    assert result.cost == 3
    assert result.warnings == ()


def test_low_parallax_cannot_override_verified_bridge_and_reentry():
    bridge = Candidate(
        "bridge",
        coverage={"south"},
        geometry=0.1,
        connectivity=0.2,
        parallax=0.5,
        verified_bridge=True,
        articulation=True,
        alignment_evaluated=True,
    )
    ordinary = Candidate(
        "ordinary", coverage={"north"}, geometry=0.9, connectivity=0.1, parallax=4.0
    )
    roles = assign_roles([bridge, ordinary])
    assert roles["bridge"].role == "BRIDGE"
    assert roles["bridge"].risk == "HIGH"
    result = reinforce_bridges(
        [bridge], [Candidate("alternate", coverage={"south"}, geometry=0.5, connectivity=0.6)]
    )
    assert result.rounds == 1
    assert [c.id for c in result.added] == ["alternate"]


def test_bridge_without_diagnostic_alignment_evidence_stays_undecided():
    bridge = Candidate(
        "bridge",
        verified_bridge=True,
        articulation=True,
        parallax=0.5,
        alignment_evaluated=False,
    )
    decision = assign_roles([bridge])["bridge"]
    assert decision.role is None
    assert decision.base_map == "BLOCKED"
