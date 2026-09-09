from __future__ import annotations

import numpy as np

from sfm_diagnosis.site_pipeline.deployment_localizer import DEFAULT_THRESHOLDS
from sfm_diagnosis.viewpoint_policy import (
    FROZEN_PNP_THRESHOLDS,
    align_occupied_bins,
    bank_occupancy_cutoff,
    decide_query_status,
    filter_reference_bank,
    load_intersection_cells,
    occupied_bins,
    occupancy_percentile_threshold,
    position_in_intersection,
    prefer_side_looking_references,
    reference_observation_occupancy,
    resolve_megaloc_bank,
    never_weaker_than_frozen,
    viewpoint_should_abstain,
    write_sideview_bank,
)


def test_occupied_bins_counts_unique_cells() -> None:
    uv = np.array([[10.0, 10.0], [11.0, 11.0], [90.0, 10.0]], dtype=float)
    count, frac = occupied_bins(uv, width=100, height=100, grid=10)
    assert count == 2
    assert frac == 0.02


def test_occupied_bins_empty() -> None:
    count, frac = occupied_bins(np.zeros((0, 2)), width=100, height=100, grid=30)
    assert count == 0
    assert frac == 0.0


def test_reference_observation_occupancy_uses_image_points() -> None:
    xy = np.array([[100.0, 100.0], [2000.0, 100.0], [100.0, 1400.0]], dtype=float)
    count, frac = reference_observation_occupancy(xy, width=2688, height=1512, grid=30)
    assert count >= 3
    assert 0.0 < frac <= 1.0


def test_prefer_side_looking_drops_empty_views_from_pool() -> None:
    ranked = (0, 1, 2, 3, 4)
    occupied = (2, 80, 3, 90, 70)
    selected, used_fallback = prefer_side_looking_references(
        ranked, occupied, min_occupied_bins=20, top_k=3
    )
    assert selected == (1, 3, 4)
    assert used_fallback is False


def test_prefer_side_looking_falls_back_when_all_empty() -> None:
    ranked = (0, 1, 2)
    occupied = (1, 2, 0)
    selected, used_fallback = prefer_side_looking_references(
        ranked, occupied, min_occupied_bins=20, top_k=2
    )
    assert selected == (0, 1)
    assert used_fallback is True


def test_viewpoint_should_abstain_on_empty_image_support() -> None:
    assert viewpoint_should_abstain(
        occupancy_4x4=2, hull_coverage=0.04, occupied_frac_30=0.01
    )
    assert not viewpoint_should_abstain(
        occupancy_4x4=8, hull_coverage=0.2, occupied_frac_30=0.2
    )


def test_decide_query_status_abstains_wrong_viewpoint() -> None:
    assert (
        decide_query_status(
            registration_success=True,
            strong_success=False,
            viewpoint_ok=False,
        )
        == "ABSTAINED"
    )
    assert (
        decide_query_status(
            registration_success=True,
            strong_success=True,
            viewpoint_ok=True,
        )
        == "LOCALIZED_STRONG"
    )
    assert (
        decide_query_status(
            registration_success=True,
            strong_success=False,
            viewpoint_ok=True,
        )
        == "POSE_ESTIMATED_WEAK"
    )
    assert (
        decide_query_status(
            registration_success=False,
            strong_success=False,
            viewpoint_ok=True,
        )
        == "ABSTAINED"
    )


def test_frozen_thresholds_never_relax() -> None:
    weaker = {
        "strong_inliers": 10,
        "minimum_inlier_ratio": 0.05,
        "minimum_hull_coverage": 0.01,
        "minimum_occupancy_4x4": 1,
        "minimum_positive_depth_ratio": 0.5,
        "maximum_reprojection_p90_px": 20.0,
    }
    frozen = never_weaker_than_frozen(weaker, in_intersection=False)
    assert frozen["strong_inliers"] == FROZEN_PNP_THRESHOLDS["strong_inliers"]
    assert frozen["minimum_inlier_ratio"] == FROZEN_PNP_THRESHOLDS["minimum_inlier_ratio"]
    assert frozen["maximum_reprojection_p90_px"] == FROZEN_PNP_THRESHOLDS["maximum_reprojection_p90_px"]
    intersection = never_weaker_than_frozen(weaker, in_intersection=True)
    assert intersection == dict(FROZEN_PNP_THRESHOLDS)
    assert FROZEN_PNP_THRESHOLDS == DEFAULT_THRESHOLDS
    stronger = {**weaker, "strong_inliers": 120, "maximum_reprojection_p90_px": 2.0}
    tightened = never_weaker_than_frozen(stronger, in_intersection=True)
    assert tightened["strong_inliers"] == 120
    assert tightened["maximum_reprojection_p90_px"] == 2.0


def test_position_in_intersection_uses_radius() -> None:
    cells = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    assert position_in_intersection((0.05, 0.0, 0.0), cells, radius=0.12)
    assert not position_in_intersection((0.5, 0.0, 0.0), cells, radius=0.12)


def test_filter_reference_bank_drops_empty_side_views() -> None:
    names = ("a.jpg", "b.jpg", "c.jpg", "d.jpg")
    occupied = (10, 500, 20, 600)
    kept, dropped = filter_reference_bank(names, occupied, min_occupied_bins=40)
    assert kept == (1, 3)
    assert dropped == ("a.jpg", "c.jpg")


def test_align_occupied_bins_follows_selected_order() -> None:
    aligned = align_occupied_bins(
        ("a.jpg", "b.jpg", "c.jpg"),
        (11, 22, 33),
        ("c.jpg", "a.jpg"),
    )
    assert aligned == (33, 11)


def test_bank_occupancy_cutoff_uses_percentile_floor() -> None:
    occupied = (17, 40, 431, 489, 685, 806, 900)
    assert occupancy_percentile_threshold(occupied, 10) >= 17
    assert bank_occupancy_cutoff(occupied, percentile=10, floor=40) >= 40


def test_resolve_megaloc_bank_prefers_sideview(tmp_path) -> None:
    (tmp_path / "megaloc_references.npy").write_bytes(b"full")
    (tmp_path / "megaloc_references.names.json").write_text("[]")
    descriptors, names, kind = resolve_megaloc_bank(tmp_path)
    assert kind == "full"
    assert descriptors.name == "megaloc_references.npy"
    (tmp_path / "megaloc_references_sideview.npy").write_bytes(b"side")
    (tmp_path / "megaloc_references_sideview.names.json").write_text("[]")
    descriptors, names, kind = resolve_megaloc_bank(tmp_path)
    assert kind == "sideview"
    assert names.name == "megaloc_references_sideview.names.json"


def test_load_intersection_cells_reads_positions(tmp_path) -> None:
    path = tmp_path / "fim_lwtl_intersection_cells.json"
    path.write_text(
        '{"cell_positions": [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]}', encoding="utf-8"
    )
    cells = load_intersection_cells(path)
    assert cells.shape == (2, 3)
    assert list(cells[0]) == [0.0, 1.0, 2.0]
    assert load_intersection_cells(None).shape == (0, 3)


def test_write_sideview_bank_persists_kept_and_dropped(tmp_path) -> None:
    names = ("empty.jpg", "side.jpg")
    descriptors = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    summary = write_sideview_bank(
        tmp_path, names, descriptors, (5, 80), min_occupied_bins=40
    )
    assert summary["kept_count"] == 1
    assert summary["dropped_count"] == 1
    kept_names = __import__("json").loads(
        (tmp_path / "megaloc_references_sideview.names.json").read_text()
    )
    assert kept_names == ["side.jpg"]
    dropped = __import__("json").loads(
        (tmp_path / "megaloc_references_dropped.json").read_text()
    )
    assert dropped["dropped"] == [{"image_name": "empty.jpg", "occupied_bins": 5}]
    loaded = np.load(tmp_path / "megaloc_references_sideview.npy")
    assert loaded.shape == (1, 2)
    assert loaded[0, 1] == 1.0
