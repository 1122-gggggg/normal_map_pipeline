from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sfm_diagnosis.site_pipeline.config import PipelineConfig
from sfm_diagnosis.site_pipeline.handlers import (
    _execute_tiered_loo,
    _diagnosis_keyframes,
    graph_build_stage,
    pre_sfm_diagnosis_stage,
    role_assignment_stage,
    segment_keyframe_stage,
    reinforcement_stage,
)
from sfm_diagnosis.site_pipeline.loo import LOOTarget
from sfm_diagnosis.site_pipeline.pipeline import StageContext


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _context(run: Path, stage: str, outputs: tuple[str, ...]) -> StageContext:
    return StageContext(
        run,
        stage,
        PipelineConfig.from_dict({"site_name": "test"}),
        tuple(run / value for value in outputs),
        "fingerprint",
    )


def test_post_sfm_diagnosis_uses_only_the_stage_selection(tmp_path: Path) -> None:
    selection = tmp_path / "artifacts/selection/diagnostic_selection.json"
    selection.parent.mkdir(parents=True)
    selection.write_text(json.dumps({"selected_keyframes": ["selected"]}))
    keyframes = [
        {"keyframe_id": "selected", "segment_id": "active"},
        {"keyframe_id": "candidate-pool", "segment_id": "unmapped"},
    ]

    filtered = _diagnosis_keyframes(
        _context(tmp_path, "stage09_post_sfm_diagnosis", ("unused.json",)),
        keyframes,
    )

    assert filtered == [keyframes[0]]


def test_tiered_loo_can_be_explicitly_skipped_without_running_mapper(tmp_path: Path) -> None:
    context = StageContext(
        tmp_path,
        "stage13_final_diagnosis",
        PipelineConfig.from_dict(
            {"site_name": "test", "resources": {"tiered_loo_enabled": False}}
        ),
        (tmp_path / "unused.json",),
        "fingerprint",
    )

    results = _execute_tiered_loo(
        context,
        tmp_path / "model-does-not-need-to-exist",
        (LOOTarget("video", "v1", "all_mapping_videos_exact"),),
        [],
        [],
    )

    assert results == [
        {
            "kind": "video",
            "target_id": "v1",
            "reason": "all_mapping_videos_exact",
            "status": "NOT_EVALUATED_DISABLED_BY_CONFIG",
            "warnings": ["LOO_NOT_EVALUATED"],
        }
    ]


def test_graph_build_uses_verified_geometry_only_and_aggregates_four_levels(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/keyframes/keyframes.jsonl",
        [
            {"keyframe_id": "a", "segment_id": "s1", "video_id": "v1", "session_id": "day1"},
            {"keyframe_id": "b", "segment_id": "s2", "video_id": "v2", "session_id": "day1"},
            {"keyframe_id": "c", "segment_id": "s3", "video_id": "v3", "session_id": "day2"},
        ],
    )
    _write_jsonl(
        tmp_path / "artifacts/pairs/geometry.jsonl",
        [
            {"image_i": "a", "image_j": "b", "admission": "VERIFIED", "inliers_F": 100},
            {"image_i": "b", "image_j": "c", "admission": "CANDIDATE", "inliers_F": 1000},
        ],
    )
    output = "artifacts/graphs/graph_bundle.json"

    graph_build_stage(_context(tmp_path, "stage05_graph_build", (output,)))

    payload = json.loads((tmp_path / output).read_text())
    assert payload["frame"]["edges"] == [["a", "b"]]
    assert payload["segment"]["edges"] == [["s1", "s2"]]
    assert payload["video"]["edges"] == [["v1", "v2"]]
    assert payload["session"]["edges"] == []
    assert payload["excluded_candidate_pairs"] == 1


def test_segment_keyframe_stage_reextracts_an_image_that_disagrees_with_prior_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    image = tmp_path / "artifacts/keyframes/images/v/frame_00000000.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"tampered")
    _write_jsonl(
        tmp_path / "artifacts/sanitization/frames.jsonl",
        [
            {
                "frame_id": "v:00000000",
                "video_id": "v",
                "session_id": "s",
                "frame_index": 0,
                "timestamp": 0.0,
                "source_pts_seconds": 0.0,
                "source_frame_index": 0,
                "source_uri": "/immutable/raw.mp4",
                "status": "CANDIDATE",
                "evaluation_role": "MAPPING",
                "motion_class": "parallax",
            }
        ],
    )
    _write_jsonl(
        tmp_path / "artifacts/keyframes/keyframes.jsonl",
        [
            {
                "keyframe_id": "v:00000000",
                "image_uri": str(image),
                "image_sha256": hashlib.sha256(b"original").hexdigest(),
            }
        ],
    )
    artifact = {
        "path": str(image.resolve()),
        "size": len(b"original"),
        "sha256": hashlib.sha256(b"original").hexdigest(),
    }
    encoded = json.dumps([artifact], sort_keys=True, separators=(",", ":"), default=str).encode()
    receipt = tmp_path / "receipts/stage02_segment_keyframes.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps(
            {
                "referenced_artifacts": {
                    "count": 1,
                    "fingerprint": hashlib.sha256(encoded).hexdigest(),
                    "artifacts": [artifact],
                }
            }
        ),
        encoding="utf-8",
    )
    calls = []

    def restore(_source, requests):
        calls.extend(requests)
        for request in requests:
            Path(request["output"]).write_bytes(b"restored-from-raw")

    monkeypatch.setattr("sfm_diagnosis.site_pipeline.frame_analyzer.extract_frames", restore)

    segment_keyframe_stage(
        _context(
            tmp_path,
            "stage02_segment_keyframes",
            (
                "artifacts/keyframes/segments.jsonl",
                "artifacts/keyframes/keyframes.jsonl",
            ),
        )
    )

    assert len(calls) == 1
    assert image.read_bytes() == b"restored-from-raw"
    manifest = json.loads((tmp_path / "artifacts/keyframes/keyframes.jsonl").read_text())
    assert manifest["image_sha256"] == hashlib.sha256(b"restored-from-raw").hexdigest()


def test_segment_keyframe_stage_reuses_an_image_verified_by_prior_receipt(
    tmp_path: Path, monkeypatch
) -> None:
    image = tmp_path / "artifacts/keyframes/images/v/frame_00000000.jpg"
    image.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(b"verified")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    _write_jsonl(
        tmp_path / "artifacts/sanitization/frames.jsonl",
        [
            {
                "frame_id": "v:00000000",
                "video_id": "v",
                "session_id": "s",
                "frame_index": 0,
                "timestamp": 0.0,
                "source_pts_seconds": 0.0,
                "source_frame_index": 0,
                "source_uri": "/immutable/raw.mp4",
                "status": "CANDIDATE",
                "evaluation_role": "MAPPING",
                "motion_class": "parallax",
            }
        ],
    )
    artifact = {"path": str(image.resolve()), "size": image.stat().st_size, "sha256": digest}
    encoded = json.dumps([artifact], sort_keys=True, separators=(",", ":"), default=str).encode()
    receipt = tmp_path / "receipts/stage02_segment_keyframes.json"
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(
        json.dumps(
            {
                "referenced_artifacts": {
                    "count": 1,
                    "fingerprint": hashlib.sha256(encoded).hexdigest(),
                    "artifacts": [artifact],
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sfm_diagnosis.site_pipeline.frame_analyzer.extract_frames",
        lambda *_args, **_kwargs: pytest.fail("verified image should be reused"),
    )

    segment_keyframe_stage(
        _context(
            tmp_path,
            "stage02_segment_keyframes",
            (
                "artifacts/keyframes/segments.jsonl",
                "artifacts/keyframes/keyframes.jsonl",
            ),
        )
    )

    assert image.read_bytes() == b"verified"


def test_pre_sfm_diagnosis_reports_bridge_as_high_importance_and_high_risk(tmp_path: Path) -> None:
    graph_path = tmp_path / "artifacts/graphs/graph_bundle.json"
    graph_path.parent.mkdir(parents=True)
    graph_path.write_text(
        json.dumps(
            {
                "segment": {
                    "nodes": ["a", "b", "c"],
                    "edges": [["a", "b"], ["b", "c"]],
                }
            }
        )
    )
    output = "artifacts/diagnosis/pre_sfm.json"

    pre_sfm_diagnosis_stage(_context(tmp_path, "stage06_pre_sfm_diagnosis", (output,)))

    payload = json.loads((tmp_path / output).read_text())
    assert payload["articulation_nodes"] == ["b"]
    assert all(
        edge["importance"] == "HIGH" and edge["risk"] == "HIGH" for edge in payload["bridges"]
    )
    assert all(edge["replacement_paths"] == 0 for edge in payload["bridges"])


def test_role_and_reinforcement_keep_low_parallax_bridge_then_require_reshoot_if_unstable(
    tmp_path: Path,
) -> None:
    _write_jsonl(
        tmp_path / "artifacts/diagnosis/post_sfm_segments.jsonl",
        [
            {
                "segment_id": "bridge",
                "pre_sfm_role": "BRIDGE_CANDIDATE",
                "verified_bridge": True,
                "articulation": True,
                "parallax": 0.7,
                "geometry": 0.2,
                "alignment_consistent": True,
                "alignment_evaluated": True,
                "unstable": False,
            }
        ],
    )
    roles_output = "artifacts/selection/roles.jsonl"
    role_assignment_stage(_context(tmp_path, "stage10_role_assignment", (roles_output,)))
    role = json.loads((tmp_path / roles_output).read_text())
    assert role["post_sfm_role"] == "BRIDGE"
    assert role["mapping_mode"] == "POSE_ONLY"
    assert role["risk"] == "HIGH"
    assert role["legacy_session_role"] == "BASE_SUPPORT"
    assert (tmp_path / "artifacts/selection/legacy_session_roles.json").is_file()

    # Mark the same unique bridge unstable; with no alternate candidates the
    # final decision must be blocked rather than force-merged.
    post_path = tmp_path / "artifacts/diagnosis/post_sfm_segments.jsonl"
    row = json.loads(post_path.read_text())
    row["unstable"] = True
    post_path.write_text(json.dumps(row) + "\n")
    _write_jsonl(
        tmp_path / "artifacts/keyframes/segments.jsonl",
        [{"segment_id": "bridge", "video_id": "v", "session_id": "s"}],
    )
    _write_jsonl(
        tmp_path / "artifacts/keyframes/keyframes.jsonl",
        [{"keyframe_id": "k", "segment_id": "bridge", "video_id": "v"}],
    )
    _write_jsonl(tmp_path / "artifacts/pairs/geometry.jsonl", [])
    graph_bundle = tmp_path / "artifacts/graphs/graph_bundle.json"
    graph_bundle.parent.mkdir(parents=True, exist_ok=True)
    graph_bundle.write_text(json.dumps({"segment": {"nodes": ["bridge"], "edges": []}}))
    diagnostic_selection = tmp_path / "artifacts/selection/diagnostic_selection.json"
    diagnostic_selection.write_text(
        json.dumps({"selected_segments": ["bridge"], "selected_keyframes": ["k"]})
    )
    metadata = tmp_path / "inputs/metadata.csv"
    metadata.parent.mkdir(parents=True, exist_ok=True)
    metadata.write_text("source_id\n")
    corpus = tmp_path / "inputs/corpus_manifest.json"
    corpus.write_text(json.dumps({"sources": []}))
    reinforcement_stage(
        _context(
            tmp_path,
            "stage11_reinforcement",
            (
                "decisions/final_build_decision.json",
                "decisions/final_selection.json",
                "products/candidate_pool/manifest.jsonl",
                "products/weak_region_reshoot_plan.json",
            ),
        )
    )
    decision = json.loads((tmp_path / "decisions/final_build_decision.json").read_text())
    assert decision["approval_allowed"] is False
    assert "RESHOOT_REQUIRED" in decision["issues"]


def test_candidate_pool_keeps_segments_omitted_from_diagnostic_mapping(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/keyframes/segments.jsonl",
        [
            {"segment_id": "active", "video_id": "v", "session_id": "s"},
            {"segment_id": "never-mapped", "video_id": "v", "session_id": "s"},
        ],
    )
    _write_jsonl(
        tmp_path / "artifacts/keyframes/keyframes.jsonl",
        [
            {"keyframe_id": "a", "segment_id": "active", "video_id": "v"},
            {"keyframe_id": "b", "segment_id": "never-mapped", "video_id": "v"},
        ],
    )
    _write_jsonl(tmp_path / "artifacts/pairs/geometry.jsonl", [])
    _write_jsonl(
        tmp_path / "artifacts/diagnosis/post_sfm_segments.jsonl",
        [{"segment_id": "active", "geometry": 0.9}],
    )
    _write_jsonl(
        tmp_path / "artifacts/selection/roles.jsonl",
        [{"segment_id": "active", "post_sfm_role": "CORE", "mapping_mode": "TRIANGULATE"}],
    )
    graph = tmp_path / "artifacts/graphs/graph_bundle.json"
    graph.parent.mkdir(parents=True)
    graph.write_text(json.dumps({"segment": {"nodes": ["active", "never-mapped"], "edges": []}}))
    selection = tmp_path / "artifacts/selection/diagnostic_selection.json"
    selection.write_text(json.dumps({"selected_segments": ["active"], "selected_keyframes": ["a"]}))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "metadata.csv").write_text("source_id\n")
    (inputs / "corpus_manifest.json").write_text(json.dumps({"sources": []}))
    outputs = (
        "decisions/final_build_decision.json",
        "decisions/final_selection.json",
        "products/candidate_pool/manifest.jsonl",
        "products/weak_region_reshoot_plan.json",
    )

    reinforcement_stage(_context(tmp_path, "stage11_reinforcement", outputs))

    pool = [json.loads(line) for line in (tmp_path / outputs[2]).read_text().splitlines()]
    assert [row["segment_id"] for row in pool] == ["never-mapped"]
    assert pool[0]["keyframe_ids"] == ["b"]
    final_selection = json.loads((tmp_path / outputs[1]).read_text())
    assert final_selection["active_segments"] == ["active"]


def test_reinforcement_blocks_an_empty_final_mapping_selection(tmp_path: Path) -> None:
    _write_jsonl(
        tmp_path / "artifacts/keyframes/segments.jsonl",
        [{"segment_id": "excluded", "video_id": "v", "session_id": "s"}],
    )
    _write_jsonl(
        tmp_path / "artifacts/keyframes/keyframes.jsonl",
        [{"keyframe_id": "k", "segment_id": "excluded", "video_id": "v"}],
    )
    _write_jsonl(tmp_path / "artifacts/pairs/geometry.jsonl", [])
    _write_jsonl(
        tmp_path / "artifacts/diagnosis/post_sfm_segments.jsonl",
        [{"segment_id": "excluded", "geometry": 0.0, "registered_ratio": 0.0}],
    )
    _write_jsonl(
        tmp_path / "artifacts/selection/roles.jsonl",
        [{"segment_id": "excluded", "post_sfm_role": None, "mapping_mode": "EXCLUDE"}],
    )
    graph = tmp_path / "artifacts/graphs/graph_bundle.json"
    graph.parent.mkdir(parents=True)
    graph.write_text(json.dumps({"segment": {"nodes": ["excluded"], "edges": []}}))
    selection = tmp_path / "artifacts/selection/diagnostic_selection.json"
    selection.write_text(json.dumps({"selected_segments": ["excluded"], "selected_keyframes": ["k"]}))
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    (inputs / "metadata.csv").write_text("source_id\n")
    (inputs / "corpus_manifest.json").write_text(json.dumps({"sources": []}))
    outputs = (
        "decisions/final_build_decision.json",
        "decisions/final_selection.json",
        "products/candidate_pool/manifest.jsonl",
        "products/weak_region_reshoot_plan.json",
    )

    reinforcement_stage(_context(tmp_path, "stage11_reinforcement", outputs))

    decision = json.loads((tmp_path / outputs[0]).read_text())
    assert decision["approval_allowed"] is False
    assert decision["issues"] == [
        "NO_ACTIVE_KEYFRAMES",
        "NO_ACTIVE_SEGMENTS",
        "NO_VERIFIED_FINAL_PAIRS",
    ]
