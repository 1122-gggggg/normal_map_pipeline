import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


RUNNER = Path(__file__).parents[2] / "examples" / "run_official_actloc_grid.py"
spec = importlib.util.spec_from_file_location("official_actloc_grid", RUNNER)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_orientation_grid_is_released_6x18_grid():
    elevations, azimuths = runner.actloc_orientation_grid()
    assert elevations.tolist() == [-60, -40, -20, 0, 20, 40]
    assert azimuths.tolist() == list(range(-180, 180, 20))


def test_coordinate_scale_factor_is_positive_and_recorded_in_attestation():
    parser = runner.parse_args
    import sys

    old = sys.argv
    try:
        sys.argv = [
            "runner",
            "--sfm-dir",
            ".",
            "--risk-map",
            ".",
            "--checkpoint",
            ".",
            "--source-root",
            ".",
            "--output-dir",
            ".",
        ]
        args = parser()
    finally:
        sys.argv = old
    assert args.coordinate_scale_factor == 1.0


def test_attestation_rejects_coordinate_scale_factor_change():
    identity = {
        "checkpoint_sha256": "c",
        "source_revision": "s",
        "runner_sha256": "r",
        "map": {
            "risk_map_sha256": "m",
            "model_file_sha256": {},
            "coordinate_scale": "x",
            "coordinate_scale_factor": 1.0,
            "grid_positions": 1,
        },
        "torch": {"version": "t", "cuda": "c", "compute_capability": [1, 2], "architectures": []},
        "flash_attention": {"version": "v", "module": "mod", "wheel_sha256": None},
        "orientation_grid": {"elevations_deg": [-60], "azimuths_deg": [-180]},
    }
    changed = {**identity, "map": {**identity["map"], "coordinate_scale_factor": 2.0}}
    assert not runner.attestations_match(identity, changed)


def test_to_builtin_recursively_handles_numpy_values():
    value = {np.int64(3): (np.float32(1.25), np.array([np.int32(2), 4]))}
    assert runner.to_builtin(value) == {"3": [1.25, [2, 4]]}


def test_positions_from_risk_map_is_unique_sorted(tmp_path):
    path = tmp_path / "risk.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {"position": [1, 2, 3]},
                    {"position": [0, 4, 2]},
                    {"position": [1, 2, 3]},
                ]
            }
        )
    )
    assert runner.positions_from_risk_map(path).tolist() == [[0.0, 4.0, 2.0], [1.0, 2.0, 3.0]]


def test_resume_records_rejects_mismatched_or_noncontiguous_partial(tmp_path):
    positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"positions": [{"index": 1, "position": [0, 0, 0]}]}))
    with pytest.raises(ValueError, match="non-contiguous"):
        runner.resume_records(path, positions)

    path.write_text(json.dumps({"positions": [{"index": 0, "position": [9, 0, 0]}]}))
    with pytest.raises(ValueError, match="grid differs"):
        runner.resume_records(path, positions)


def test_resume_records_rejects_malformed_record(tmp_path):
    positions = np.asarray([[0.0, 0.0, 0.0]])
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"positions": [{"index": 0}]}))
    with pytest.raises(ValueError, match="malformed"):
        runner.resume_records(path, positions)


def test_resume_records_fails_closed_for_extra_records(tmp_path):
    positions = np.asarray([[0.0, 0.0, 0.0]])
    path = tmp_path / "partial.json"
    path.write_text(
        json.dumps(
            {
                "positions": [
                    {"index": 0, "position": [0, 0, 0]},
                    {"index": 1, "position": [1, 0, 0]},
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="more records"):
        runner.resume_records(path, positions)


def test_resume_records_rejects_stable_attestation_mismatch(tmp_path):
    positions = np.asarray([[0.0, 0.0, 0.0]])
    path = tmp_path / "partial.json"
    path.write_text(
        json.dumps(
            {
                "attestation": {"checkpoint_sha256": "old"},
                "positions": [{"index": 0, "position": [0, 0, 0]}],
            }
        )
    )
    expected = {"checkpoint_sha256": "new", "command": ["different"], "started_at_utc": "new"}
    with pytest.raises(ValueError, match="attestation mismatch"):
        runner.resume_records(path, positions, expected)


def test_attestation_full_match_allows_both_missing_wheel_hashes():
    identity = {
        "checkpoint_sha256": "c",
        "source_revision": "s",
        "runner_sha256": "r",
        "map": {
            "risk_map_sha256": "m",
            "model_file_sha256": {},
            "coordinate_scale": "x",
            "coordinate_scale_factor": 1.0,
            "grid_positions": 1,
        },
        "torch": {"version": "t", "cuda": "c", "compute_capability": [1, 2], "architectures": []},
        "flash_attention": {"version": "v", "module": "mod", "wheel_sha256": None},
        "orientation_grid": {"elevations_deg": [-60], "azimuths_deg": [-180]},
    }
    assert runner.attestations_match(identity, dict(identity))
    one_sided = dict(identity)
    one_sided["flash_attention"] = dict(identity["flash_attention"], wheel_sha256="w")
    assert not runner.attestations_match(identity, one_sided)
    assert not runner.attestations_match(
        {k: v for k, v in identity.items() if k != "runner_sha256"}, identity
    )


def test_final_json_is_not_written_when_npz_fails(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(runner.np, "savez_compressed", fail)
    with pytest.raises(OSError):
        runner.write_final_artifacts(tmp_path / "final.json", tmp_path / "final.npz", {}, [])
    assert not (tmp_path / "final.json").exists()
