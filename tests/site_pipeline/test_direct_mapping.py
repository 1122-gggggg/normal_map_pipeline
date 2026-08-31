from __future__ import annotations

import json
from pathlib import Path

from sfm_diagnosis.site_pipeline.direct_mapping import (
    DirectMappingRequest,
    prepare_direct_run,
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
