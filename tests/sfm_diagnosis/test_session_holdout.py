from sfm_diagnosis.session_holdout import (
    cross_session_observations,
    session_from_image_name,
    summarize_holdout_results,
)
from sfm_diagnosis.edm_risk.edm_loo import EDMQueryResult


def test_session_id_is_the_first_path_component() -> None:
    assert session_from_image_name("P1190119/pts_000067567500_frame_001620.jpg") == "P1190119"
    assert session_from_image_name("solo.jpg") == "solo.jpg"


def test_cross_session_filter_drops_points_seen_only_in_the_query_session() -> None:
    tracks = {
        1: ("P1180118/a.jpg", "P1180118/b.jpg"),
        2: ("P1180118/a.jpg", "P1190119/c.jpg", "P1200120/d.jpg"),
    }
    observations = (((10.0, 20.0), 1), ((30.0, 40.0), 2))

    kept = cross_session_observations(
        "P1180118/a.jpg",
        observations,
        tracks,
        min_other_session_views=1,
    )

    assert kept == (((30.0, 40.0), 2),)


def test_holdout_summary_separates_bridge_only_from_triangulating_queries() -> None:
    results = [
        EDMQueryResult(
            query_id="P1180118/bridge.jpg",
            session_id="P1180118",
            timestamp=0.0,
            success=False,
            registration_success=True,
            valid_2d3d=0,
            ransac_inliers=0,
            loo_mode="reference-exclusion",
        ),
        EDMQueryResult(
            query_id="P1190119/core.jpg",
            session_id="P1190119",
            timestamp=1.0,
            success=True,
            registration_success=True,
            valid_2d3d=40,
            ransac_inliers=28,
            loo_mode="reference-exclusion",
        ),
    ]

    summary = summarize_holdout_results(results, min_candidates=8)

    assert summary["queries"] == 2
    assert summary["calibration_eligible"] == 1
    assert summary["bridge_or_empty"] == 1
    assert summary["eligible_success_rate"] == 1.0
    assert summary["by_session"]["P1190119"]["eligible"] == 1
