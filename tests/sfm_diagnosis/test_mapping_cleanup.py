import numpy as np

from sfm_diagnosis.mapping_cleanup import (
    classify_registered_cameras,
    consecutive_rotation_segments,
    frame_index_from_name,
    names_in_frame_range,
    parse_gluemap_conversion_log,
    leftover_coverage_report,
)
from sfm_diagnosis.edm_risk.edm_loo import EDMQueryResult


def test_frame_index_and_seam_selection() -> None:
    names = [
        "P1190119/pts_000067567500_frame_001620.jpg",
        "P1190119/pts_000080080000_frame_001920.jpg",
        "P1180118/pts_000000000000_frame_000000.jpg",
    ]
    assert frame_index_from_name(names[0]) == 1620
    assert names_in_frame_range(names, session="P1190119", lo=1620, hi=1740) == [names[0]]


def test_registered_cameras_split_bridge_only_from_triangulating() -> None:
    report = classify_registered_cameras(
        registered=("P1180118/a.jpg", "P1180118/b.jpg", "P1190119/c.jpg"),
        bridge_only=("P1180118/b.jpg",),
        selected=("P1180118/a.jpg", "P1180118/b.jpg", "P1180118/c.jpg", "P1190119/c.jpg"),
    )
    assert report["registered"] == 3
    assert report["bridge_only"] == 1
    assert report["selected_bridge_only"] == 1
    assert report["triangulating"] == 2
    assert report["selected_unregistered"] == 1
    assert report["by_session"]["P1180118"]["triangulating"] == 1
    assert report["by_session"]["P1180118"]["unregistered"] == 1


def test_conversion_log_reads_stage_counts() -> None:
    log = """
[info] Number of images: 418
[info] Total number of valid virtual points after selection: 712840
[info] Established 28543 tracks from 303572 observations
[info] Final: 226840 real tracks (1913481 obs), 908 virtual tracks (5142 obs)
"""
    rates = parse_gluemap_conversion_log(log)
    assert rates["input_images"] == 418
    assert rates["virtual_points"] == 712840
    assert rates["established_tracks"] == 28543
    assert rates["established_observations"] == 303572
    assert rates["final_real_tracks"] == 226840
    assert rates["final_real_observations"] == 1913481


def test_consecutive_rotation_flags_low_baseline_pairs() -> None:
    names = ["P1180118/a.jpg", "P1180118/b.jpg", "P1180118/c.jpg"]
    centers = np.array([[0.0, 0.0, 0.0], [0.001, 0.0, 0.0], [1.0, 0.0, 0.0]])
    yaw = np.radians(20.0)
    r_yaw = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0.0], [np.sin(yaw), np.cos(yaw), 0.0], [0.0, 0.0, 1.0]]
    )
    rotations = np.stack([np.eye(3), r_yaw, r_yaw])
    flagged = consecutive_rotation_segments(
        names, centers, rotations, min_rotation_deg=10.0, max_translation=0.05
    )
    assert flagged == [
        {
            "first": "P1180118/a.jpg",
            "second": "P1180118/b.jpg",
            "rotation_deg": flagged[0]["rotation_deg"],
            "translation": flagged[0]["translation"],
        }
    ]
    assert flagged[0]["rotation_deg"] > 10.0
    assert flagged[0]["translation"] < 0.05


def test_leftover_coverage_is_not_a_calibration_label() -> None:
    results = [
        EDMQueryResult(
            query_id="P1670167/a.jpg",
            session_id="P1670167",
            timestamp=0.0,
            success=False,
            registration_success=True,
        )
    ]
    report = leftover_coverage_report(
        results,
        mapping_sessions=("P1180118", "P1190119", "P1200120"),
        roles={"P1670167": "NEW_SUBMAP"},
    )
    assert report["artifact_type"] == "LEFTOVER_COVERAGE_ONLY"
    assert report["usable_as_failure_probability_labels"] is False
    assert report["by_session"]["P1670167"]["success_rate"] == 0.0
    assert report["by_session"]["P1670167"]["role"] == "NEW_SUBMAP"
