from pathlib import Path

import numpy as np

from sfm_diagnosis.edm_risk.actloc_provider import (
    CachedActLocProvider,
    DisabledActLocProvider,
    build_actloc_provider,
    rotation_matrices_to_scalar_first_quaternions,
    world_yaw_pitch_to_actloc_cell,
)


class FakeProvider:
    source = "fake"
    fingerprint = "fake-v1"

    def __init__(self):
        self.calls = 0

    def score_grid(self, position):
        self.calls += 1
        return np.arange(108, dtype=float).reshape(6, 18) / 107.0

    def score(self, position, yaw_deg, pitch_deg):
        row, column = world_yaw_pitch_to_actloc_cell(yaw_deg, pitch_deg)
        return float(self.score_grid(position)[row, column])


def test_disabled_provider_never_blocks_fast_diagnosis() -> None:
    provider = DisabledActLocProvider(reason="checkpoint unavailable")
    assert provider.score(np.zeros(3), 0.0, 0.0) is None
    assert provider.source == "disabled"


def test_cached_provider_runs_backend_once_per_position_and_persists(tmp_path: Path) -> None:
    backend = FakeProvider()
    cached = CachedActLocProvider(backend=backend, cache_dir=tmp_path)

    first = cached.score(np.array([1.0, 2.0, 3.0]), 0.0, 0.0)
    second = cached.score(np.array([1.0, 2.0, 3.0]), 40.0, -20.0)
    reloaded = CachedActLocProvider(backend=FakeProvider(), cache_dir=tmp_path)
    third = reloaded.score(np.array([1.0, 2.0, 3.0]), 0.0, 0.0)

    assert first is not None and second is not None and third == first
    assert backend.calls == 1


def test_world_yaw_pitch_maps_to_official_six_by_eighteen_locmap() -> None:
    assert world_yaw_pitch_to_actloc_cell(0.0, 0.0) == (3, 9)
    assert world_yaw_pitch_to_actloc_cell(20.0, -20.0) == (4, 10)


def test_missing_official_runtime_falls_back_without_crashing(tmp_path: Path) -> None:
    provider = build_actloc_provider(
        mode="official",
        source_root=tmp_path / "missing-source",
        checkpoint=tmp_path / "missing.pth",
        fail_open=True,
    )
    assert provider.source == "disabled"


def test_actloc_quaternions_use_canonical_scalar_first_order_without_new_scipy_api() -> None:
    quaternions = rotation_matrices_to_scalar_first_quaternions(np.eye(3)[None])
    assert np.allclose(quaternions, [[1.0, 0.0, 0.0, 0.0]])
