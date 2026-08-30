from __future__ import annotations

from pathlib import Path

import numpy as np
import pycolmap
import pytest

from sfm_diagnosis.site_pipeline.observation_dedup import deduplicate_model


def _write_synthetic_model(
    path: Path, *, short_unique_track: bool = False
) -> tuple[Path, int, tuple[int, ...]]:
    """Create a tiny model containing a deliberately duplicated image track."""
    options = pycolmap.SyntheticDatasetOptions()
    options.num_rigs = 1
    options.num_frames_per_rig = 3
    options.num_points3D = 20
    options.track_length = 3
    reconstruction = pycolmap.synthesize_dataset(options)
    point_id, point = next(iter(reconstruction.points3D.items()))
    expected_image_ids = tuple(sorted({int(row.image_id) for row in point.track.elements}))
    original_elements = list(point.track.elements)
    element = original_elements[0]
    if short_unique_track:
        removed = original_elements[-1]
        reconstruction.delete_observation(removed.image_id, removed.point2D_idx)
        expected_image_ids = expected_image_ids[:-1]
    image = reconstruction.images[element.image_id]
    original = np.asarray(image.points2D[element.point2D_idx].xy, dtype=np.float64)
    duplicate_index = image.num_points2D()
    image.points2D.append(pycolmap.Point2D(original + np.asarray([1.0, 0.0])))
    reconstruction.add_observation(
        point_id, pycolmap.TrackElement(element.image_id, duplicate_index)
    )
    path.mkdir()
    reconstruction.write_binary(str(path))
    return path, point_id, expected_image_ids


def test_deduplicates_same_image_observations_by_reprojection_error(tmp_path: Path) -> None:
    input_model, point_id, expected_image_ids = _write_synthetic_model(tmp_path / "input")
    output_model = tmp_path / "output"
    original_xyz = pycolmap.Reconstruction(str(input_model)).point3D(point_id).xyz.copy()

    receipt = deduplicate_model(input_model, output_model)

    result = pycolmap.Reconstruction(str(output_model))
    point = result.point3D(point_id)
    assert point.xyz == pytest.approx(original_xyz)
    assert tuple(sorted(element.image_id for element in point.track.elements)) == (
        expected_image_ids
    )
    assert result.num_cameras() == 1
    assert result.num_reg_images() == 3
    assert receipt["removed_duplicate_observations"] == 1


def test_deletes_points_below_minimum_unique_image_track_length(tmp_path: Path) -> None:
    input_model, point_id, _ = _write_synthetic_model(tmp_path / "input", short_unique_track=True)
    output_model = tmp_path / "output"

    receipt = deduplicate_model(input_model, output_model, minimum_unique_images=3)

    result = pycolmap.Reconstruction(str(output_model))
    assert not result.exists_point3D(point_id)
    assert receipt["removed_short_tracks"] == 1


def test_deduplication_fails_closed_when_output_exists(tmp_path: Path) -> None:
    input_model, _, _ = _write_synthetic_model(tmp_path / "input")
    output_model = tmp_path / "output"
    output_model.mkdir()

    with pytest.raises(FileExistsError):
        deduplicate_model(input_model, output_model)
