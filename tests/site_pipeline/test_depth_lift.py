from __future__ import annotations

import numpy as np
import pytest

from river_map_quality.official_edm_adapter import (
    LiftedMatch,
    UnmappedPair,
    deduplicate_lifted_matches,
    lift_unmapped_with_depth,
)


IDENTITY_W2C = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def _unmapped(query_xy=(10.0, 10.0), reference_xy=(10.0, 10.0), confidence=0.9) -> UnmappedPair:
    return UnmappedPair(
        query_xy=query_xy,
        reference_xy=reference_xy,
        reference_name="ref.jpg",
        confidence=confidence,
    )


def test_constant_depth_unprojects_principal_point() -> None:
    depth = np.full((10, 10), 5.0, dtype=np.float32)
    lifted = lift_unmapped_with_depth(
        unmapped=(_unmapped(),),
        depth=depth,
        native_wh=(20, 20),
        fx=100.0,
        fy=100.0,
        cx=10.0,
        cy=10.0,
        cam_from_world_3x4=IDENTITY_W2C,
        median_sparse_z=5.0,
    )
    assert len(lifted) == 1
    assert lifted[0].point3d_id is None
    assert lifted[0].xyz is not None
    assert lifted[0].xyz == pytest.approx((0.0, 0.0, 5.0), abs=1e-5)
    assert lifted[0].lift_distance_px == 0.0


def test_depth_outside_sparse_z_band_is_dropped() -> None:
    depth = np.full((10, 10), 5.0, dtype=np.float32)
    lifted = lift_unmapped_with_depth(
        unmapped=(_unmapped(),),
        depth=depth,
        native_wh=(20, 20),
        fx=100.0,
        fy=100.0,
        cx=10.0,
        cy=10.0,
        cam_from_world_3x4=IDENTITY_W2C,
        median_sparse_z=100.0,
    )
    assert lifted == ()


def _dense(query_xy, *, confidence, xyz) -> LiftedMatch:
    return LiftedMatch(
        query_xy=query_xy,
        point3d_id=None,
        reference_name="dense.jpg",
        confidence=confidence,
        lift_distance_px=0.0,
        xyz=xyz,
    )


def _sparse(query_xy, *, point3d_id, confidence) -> LiftedMatch:
    return LiftedMatch(
        query_xy=query_xy,
        point3d_id=point3d_id,
        reference_name="sparse.jpg",
        confidence=confidence,
        lift_distance_px=0.4,
    )


def test_deduplicate_keeps_one_dense_at_same_query_pixel() -> None:
    result = deduplicate_lifted_matches(
        (
            _dense((5.0, 5.0), confidence=0.4, xyz=(0.0, 0.0, 1.0)),
            _dense((5.0, 5.0), confidence=0.9, xyz=(0.0, 0.0, 2.0)),
        ),
        query_conflict_distance_px=1.0,
    )
    assert len(result.matches) == 1
    assert result.matches[0].xyz == (0.0, 0.0, 2.0)
    assert result.duplicate_point3d_match_count == 1


def test_deduplicate_prefers_sparse_over_nearby_dense() -> None:
    result = deduplicate_lifted_matches(
        (
            _dense((5.2, 5.0), confidence=0.99, xyz=(1.0, 2.0, 3.0)),
            _sparse((5.0, 5.0), point3d_id=42, confidence=0.1),
        ),
        query_conflict_distance_px=1.0,
    )
    assert len(result.matches) == 1
    assert result.matches[0].point3d_id == 42
    assert result.matches[0].xyz is None
    assert result.conflicting_query_match_count == 1
