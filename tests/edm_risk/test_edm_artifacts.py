import json
from pathlib import Path

from sfm_diagnosis.edm_risk.edm_artifacts import (
    load_edm_results,
    load_river_edm_results,
)
from sfm_diagnosis.edm_risk.edm_loo import EDMQueryResult


def test_river_edm_loader_uses_outer_and_nested_strict_conjunction(tmp_path: Path) -> None:
    registrations = tmp_path / "registrations"
    registrations.mkdir()
    rows = [
        {
            "query_name": "S4/q0.jpg",
            "video": "S4",
            "source_second": 0,
            "status": "DIRECT_STRONG",
            "decision": {"status": "REJECT_UNVERIFIED_SUPPORT"},
            "raw_matches": 500,
            "lifted_correspondences": 80,
            "inlier_observations": [{"point3d_id": 42, "reference_name": "S1/a.jpg"}],
            "retrieved_names": ["S1/a.jpg"],
            "metrics": {"inlier_ratio": 0.5, "reprojection_p90": 2.0},
            "pose": [[1, 0, 0, -1], [0, 1, 0, -2], [0, 0, 1, -3], [0, 0, 0, 1]],
        },
        {
            "query_name": "S4/q1.jpg",
            "video": "S4",
            "source_second": 1,
            "status": "DIRECT_STRONG",
            "decision": {"status": "ACCEPT"},
            "raw_matches": 600,
            "lifted_correspondences": 90,
            "inlier_observations": [{"point3d_id": 43, "reference_name": "S2/b.jpg"}],
            "retrieved_names": ["S2/b.jpg"],
            "metrics": {"inlier_ratio": 0.6, "reprojection_p90": 1.8},
            "pose": [[1, 0, 0, -2], [0, 1, 0, -3], [0, 0, 1, -4], [0, 0, 0, 1]],
        },
    ]
    (registrations / "S4.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )

    loaded = load_river_edm_results(registrations, evidence_mode="external-heldout")

    assert [row.success for row in loaded] == [False, True]
    assert loaded[0].registration_success is True
    assert loaded[0].raw_matches == 500
    assert loaded[0].valid_2d3d == 80
    assert loaded[0].point_ids == (42,)
    assert loaded[0].estimated_position == (1.0, 2.0, 3.0)
    assert loaded[0].ground_truth_source == "SELF_FIT_NO_ABSOLUTE_GT"


def test_generic_loader_accepts_canonical_edm_result_bundle(tmp_path: Path) -> None:
    expected = EDMQueryResult(
        query_id="OTHER_SITE/q0.jpg",
        session_id="OTHER_SITE",
        timestamp=2.0,
        success=False,
        registration_success=False,
        estimated_position=(1.0, 2.0, 3.0),
        estimated_yaw_deg=30.0,
        estimated_pitch_deg=-10.0,
    )
    path = tmp_path / "canonical.json"
    path.write_text(json.dumps({"results": [expected.to_dict()]}), encoding="utf-8")

    loaded = load_edm_results(path, evidence_mode="strict", artifact_format="auto")

    assert loaded == [EDMQueryResult(**{**expected.__dict__, "loo_mode": "strict"})]
