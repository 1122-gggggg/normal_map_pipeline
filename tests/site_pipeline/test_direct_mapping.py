from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from sfm_diagnosis.site_pipeline.direct_mapping import (
    DirectMappingRequest,
    DirectMappingRuntime,
    export_original_rgb_ply,
    localization_reference_rows,
    prepare_direct_run,
    run_direct_mapping,
)


def test_prepare_direct_run_uses_all_sources_and_writes_pose_only_bridges(
    tmp_path: Path, monkeypatch
) -> None:
    corpus = tmp_path / "mapping"
    corpus.mkdir()
    sources = [corpus / "A.MP4", corpus / "B.MP4"]
    for source in sources:
        source.write_bytes(source.name.encode())
    intrinsics = tmp_path / "intrinsics.json"
    intrinsics.write_text(
        json.dumps(
            {
                "image_width": 1280,
                "image_height": 720,
                "K": [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]],
                "images_are_undistorted": True,
            }
        ),
        encoding="utf-8",
    )

    def fake_discover(root, _config):
        return {
            "schema_version": 2,
            "artifact_type": "SITE_CORPUS_MANIFEST",
            "site_name": "fixture",
            "corpus_root": str(root),
            "sources": [
                {
                    "source_id": f"vid_{index}",
                    "video_id": f"vid_{index}",
                    "source_kind": "video",
                    "path": str(source),
                    "relative_path": source.name,
                    "sha256": f"sha-{index}",
                    "evaluation_role": "MAPPING",
                    "probe_status": "OK",
                    "width": 1280,
                    "height": 720,
                }
                for index, source in enumerate(sources)
            ],
        }

    def fake_analyze(source, *, video_id, session_id, fps, intrinsics):
        assert Path(source) in sources
        assert fps == 4.0
        assert intrinsics.shape == (3, 3)
        classes = ("parallax", "hover", "pure_rotation", "parallax")
        return [
            {
                "video_id": video_id,
                "session_id": session_id,
                "frame_index": index,
                "timestamp": float(index),
                "source_uri": str(source),
                "motion_class": motion_class,
                "turn_event": motion_class == "pure_rotation",
                "corrupt": False,
                "severe_blur": False,
                "severe_motion_blur": False,
                "exposure_failed": False,
                "duplicate_previous": False,
                "black_fraction": 0.0,
                "white_fraction": 0.0,
            }
            for index, motion_class in enumerate(classes)
        ]

    def fake_extract(_source, requests):
        outputs = {}
        for request in requests:
            output = Path(request["output"])
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b"jpeg")
            outputs[int(request["source_frame_index"])] = output
        return outputs

    monkeypatch.setattr(
        "sfm_diagnosis.site_pipeline.direct_mapping.discover_corpus", fake_discover
    )
    monkeypatch.setattr(
        "sfm_diagnosis.site_pipeline.direct_mapping.analyze_frames", fake_analyze
    )
    monkeypatch.setattr(
        "sfm_diagnosis.site_pipeline.direct_mapping.extract_frames", fake_extract
    )
    request = DirectMappingRequest(
        site_name="fixture",
        corpus_root=corpus,
        run_dir=tmp_path / "run",
        intrinsics_path=intrinsics,
    )

    result = prepare_direct_run(request)

    keyframes = [
        json.loads(line)
        for line in (request.run_dir / "artifacts/keyframes/keyframes.jsonl")
        .read_text()
        .splitlines()
    ]
    receipt = json.loads((request.run_dir / "receipts/direct_preprocessing.json").read_text())
    forced = (request.run_dir / "inputs/forced_pairs.txt").read_text().splitlines()
    assert result["source_count"] == 2
    assert {row["video_id"] for row in keyframes} == {"vid_0", "vid_1"}
    assert sum(row["mapping_mode"] == "POSE_ONLY" for row in keyframes) == 2
    assert len(forced) == 4
    assert receipt["validation"] == "NONE"
    assert receipt["graph_checks"] == "SKIPPED_BY_REQUEST"


def test_export_original_rgb_ply_never_recolors_source_points(tmp_path: Path, monkeypatch) -> None:
    class Map:
        points_xyz = np.array([[1.0, 2.0, 3.0], [-1.0, 0.5, 8.0]])
        point_rgb = np.array([[0, 1, 2], [250, 128, 64]], dtype=np.uint8)

    monkeypatch.setattr(
        "sfm_diagnosis.site_pipeline.direct_mapping.load_gluemap", lambda _path: Map()
    )
    output = tmp_path / "original_rgb.ply"

    receipt = export_original_rgb_ply(tmp_path / "model", output)

    material = output.read_bytes()
    header, vertices = material.split(b"end_header\n", 1)
    first = struct.unpack("<fffBBB", vertices[:15])
    second = struct.unpack("<fffBBB", vertices[15:30])
    assert b"element vertex 2" in header
    assert first[3:] == (0, 1, 2)
    assert second[3:] == (250, 128, 64)
    assert receipt["recolored_points"] == 0


def test_localization_references_exclude_pose_only_and_zero_observation_images() -> None:
    class Point:
        def __init__(self, has_point):
            self._has_point = has_point

        def has_point3D(self):
            return self._has_point

    class Image:
        def __init__(self, name, observations):
            self.name = name
            self.points2D = [Point(True) for _ in range(observations)]

    class Reconstruction:
        images = {
            1: Image("v1/a.jpg", 4),
            2: Image("v1/turn.jpg", 0),
            3: Image("v2/empty.jpg", 0),
        }

    keyframes = {
        "v1/a.jpg": {
            "image_uri": "/images/a.jpg",
            "video_id": "v1",
            "mapping_mode": "TRIANGULATE",
        },
        "v1/turn.jpg": {
            "image_uri": "/images/turn.jpg",
            "video_id": "v1",
            "mapping_mode": "POSE_ONLY",
        },
        "v2/empty.jpg": {
            "image_uri": "/images/empty.jpg",
            "video_id": "v2",
            "mapping_mode": "TRIANGULATE",
        },
    }

    rows = localization_reference_rows(Reconstruction(), keyframes)

    assert rows == [
        {
            "image_name": "v1/a.jpg",
            "image_path": "/images/a.jpg",
            "video_id": "v1",
            "observations": 4,
        }
    ]


def test_run_direct_mapping_records_unvalidated_outputs(tmp_path: Path, monkeypatch) -> None:
    corpus = tmp_path / "mapping"
    corpus.mkdir()
    (corpus / "A.MP4").write_bytes(b"video")
    intrinsics = tmp_path / "intrinsics.json"
    intrinsics.write_text(
        json.dumps(
            {
                "image_width": 1280,
                "image_height": 720,
                "K": [[900.0, 0.0, 640.0], [0.0, 900.0, 360.0], [0.0, 0.0, 1.0]],
                "images_are_undistorted": True,
            }
        ),
        encoding="utf-8",
    )
    base_config = tmp_path / "gluemap.json"
    base_config.write_text("{}", encoding="utf-8")
    gluemap_root = tmp_path / "gluemap"
    (gluemap_root / "gluemap").mkdir(parents=True)
    megaloc_source = tmp_path / "megaloc/source"
    megaloc_source.mkdir(parents=True)
    megaloc_checkpoint = tmp_path / "megaloc/model.safetensors"
    megaloc_checkpoint.write_bytes(b"model")
    request = DirectMappingRequest(
        site_name="fixture",
        corpus_root=corpus,
        run_dir=tmp_path / "run",
        intrinsics_path=intrinsics,
    )
    runtime = DirectMappingRuntime(
        gluemap_root=gluemap_root,
        base_config_path=base_config,
        workspace_root=tmp_path / "workspace",
        megaloc_source=megaloc_source,
        megaloc_checkpoint=megaloc_checkpoint,
    )

    def fake_prepare(req, *, resume):
        (req.run_dir / "inputs").mkdir(parents=True)
        (req.run_dir / "artifacts/keyframes/images").mkdir(parents=True)
        (req.run_dir / "artifacts/keyframes/keyframes.jsonl").write_text("", encoding="utf-8")
        (req.run_dir / "inputs/corpus_manifest.json").write_text("{}", encoding="utf-8")
        (req.run_dir / "inputs/forced_pairs.txt").write_text("", encoding="utf-8")
        return {"status": "PREPARED", "source_count": 1}

    def fake_worker(payload):
        output = Path(payload["payload"]["output_model"])
        output.mkdir(parents=True)
        return {"status": "completed", "outputs": [str(output)], "pair_source": "native"}

    monkeypatch.setattr(
        "sfm_diagnosis.site_pipeline.direct_mapping.prepare_direct_run", fake_prepare
    )
    monkeypatch.setattr(
        "sfm_diagnosis.site_pipeline.direct_mapping.run_gluemap_adapter", fake_worker
    )
    monkeypatch.setattr(
        "sfm_diagnosis.site_pipeline.direct_mapping.export_original_rgb_ply",
        lambda _model, output: {
            "path": str(output),
            "vertices": 10,
            "recolored_points": 0,
        },
    )
    monkeypatch.setattr(
        "sfm_diagnosis.site_pipeline.direct_mapping.build_localization_package",
        lambda **_kwargs: {"references": 8, "descriptor_shape": [8, 8448]},
    )

    receipt = run_direct_mapping(request, runtime, resume=False)

    assert receipt["status"] == "MAP_BUILT_UNVALIDATED_ALL_INPUTS"
    assert receipt["validation"] == "NONE"
    assert receipt["graph_checks"] == "SKIPPED_BY_REQUEST"
    assert receipt["mapping"]["pair_source"] == "native"
    assert receipt["rgb_ply"]["recolored_points"] == 0
