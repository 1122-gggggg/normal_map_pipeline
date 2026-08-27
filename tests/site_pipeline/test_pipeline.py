from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from sfm_diagnosis.site_pipeline.config import CANONICAL_STAGES_V2, PipelineConfig
from sfm_diagnosis.site_pipeline.pipeline import (
    ApprovalRequired,
    SitePipeline,
    StageContext,
    StageOutcome,
)


def _write_metadata(path: Path, *, complete: bool) -> None:
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    for index, row in enumerate(rows):
        if row["source_kind"] == "archive":
            continue
        row.update(
            {
                "session_id": f"session-{index}",
                "route": "A-B",
                "direction": "forward",
                "camera_mode": "rectilinear",
                "crop_state": "none",
                "stabilization_state": "off",
                "intrinsics_group": "camera-a",
            }
        )
        if not complete:
            row["route"] = ""
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def _handlers() -> dict[str, object]:
    handlers = {}
    for name in CANONICAL_STAGES_V2[1:]:

        def handler(context: StageContext, *, stage_name: str = name) -> StageOutcome:
            outputs = context.expected_outputs
            for output in outputs:
                output.parent.mkdir(parents=True, exist_ok=True)
                if output.suffix:
                    output.write_text(
                        json.dumps({"stage": stage_name, "fingerprint": context.fingerprint})
                        + "\n",
                        encoding="utf-8",
                    )
                else:
                    output.mkdir(parents=True, exist_ok=True)
            if stage_name == "stage11_reinforcement":
                inputs = {
                    "final_selection": context.run_dir / "decisions/final_selection.json",
                    "roles": context.run_dir / "artifacts/selection/roles.jsonl",
                    "keyframes": context.run_dir / "artifacts/keyframes/keyframes.jsonl",
                    "pair_geometry": context.run_dir / "artifacts/pairs/geometry.jsonl",
                    "diagnostic_selection": context.run_dir
                    / "artifacts/selection/diagnostic_selection.json",
                    "metadata": context.run_dir / "inputs/metadata.csv",
                    "corpus_manifest": context.run_dir / "inputs/corpus_manifest.json",
                    "post_sfm_diagnosis": context.run_dir / "artifacts/diagnosis/post_sfm.json",
                }
                decision = {
                    "approval_allowed": True,
                    "issues": [],
                    "input_hashes": {
                        key: hashlib.sha256(path.read_bytes()).hexdigest()
                        for key, path in inputs.items()
                    },
                }
                context.expected_outputs[0].write_text(json.dumps(decision) + "\n")
            return StageOutcome(outputs=outputs, details={"implemented_by": "fake"})

        handlers[name] = handler
    return handlers


def test_pipeline_config_exposes_all_sixteen_canonical_stages() -> None:
    config = PipelineConfig.from_dict({"site_name": "test-site"})

    assert len(CANONICAL_STAGES_V2) == 16
    assert CANONICAL_STAGES_V2[0] == "stage00_inventory"
    assert CANONICAL_STAGES_V2[-1] == "stage15_publish"
    assert config.max_reinforcement_rounds == 2
    assert config.allow_unknown_metadata is False


def test_explicit_unknown_metadata_mode_allows_a_resolution_group(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "map.MP4").write_bytes(b"video")
    pipeline = SitePipeline(
        PipelineConfig.from_dict(
            {"site_name": "test", "allow_unknown_metadata": True}
        ),
        handlers=_handlers(),
    )
    initialized = pipeline.initialize(corpus, tmp_path / "run")
    metadata_path = initialized.run_dir / "inputs/metadata.csv"
    rows = list(csv.DictReader(metadata_path.open(encoding="utf-8")))
    rows[0].update(
        {
            "session_id": "unknown-source-session",
            "width": "1920",
            "height": "1080",
            "route": "unknown",
            "direction": "unknown",
            "camera_mode": "unknown",
            "crop_state": "unknown",
            "stabilization_state": "unknown",
            "intrinsics_group": "unknown-1920x1080",
        }
    )
    with metadata_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    result = pipeline.run(
        initialized.run_dir,
        to_stage="stage03_retrieval",
    )

    assert result.status == "COMPLETED"
    assert (result.run_dir / "receipts/stage03_retrieval.json").is_file()


def test_initialize_builds_hash_inventory_and_duplicate_archive_provenance(
    tmp_path: Path,
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    video = corpus / "map.MP4"
    video.write_bytes(b"immutable-video")
    sequence = corpus / "holdout_frames"
    sequence.mkdir()
    (sequence / "000000.jpg").write_bytes(b"frame-zero")
    (sequence / "000001.jpg").write_bytes(b"frame-one")
    archive = corpus / "holdout.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("holdout/000000.jpg", b"frame-zero")
        bundle.writestr("holdout/000001.jpg", b"frame-one")
    before = hashlib.sha256(video.read_bytes()).hexdigest()
    config = PipelineConfig.from_dict(
        {
            "site_name": "test-site",
            "heldout_patterns": ["holdout_frames"],
            "hash_contents": True,
        }
    )

    result = SitePipeline(config, handlers=_handlers()).initialize(corpus, tmp_path / "run")

    assert result.status == "WAITING_FOR_METADATA"
    manifest = json.loads((result.run_dir / "inputs/corpus_manifest.json").read_text())
    assert {row["source_kind"] for row in manifest["sources"]} == {
        "video",
        "image_sequence",
        "archive",
    }
    sequence_row = next(
        row for row in manifest["sources"] if row["source_kind"] == "image_sequence"
    )
    archive_row = next(row for row in manifest["sources"] if row["source_kind"] == "archive")
    assert sequence_row["evaluation_role"] == "HOLDOUT"
    assert archive_row["duplicate_of"] == sequence_row["source_id"]
    assert hashlib.sha256(video.read_bytes()).hexdigest() == before
    assert (result.run_dir / "ledger.sqlite").is_file()


def test_metadata_gate_allows_sanitization_but_blocks_retrieval(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "map.MP4").write_bytes(b"video")
    pipeline = SitePipeline(PipelineConfig.from_dict({"site_name": "test"}), handlers=_handlers())
    initialized = pipeline.initialize(corpus, tmp_path / "run")
    _write_metadata(initialized.run_dir / "inputs/metadata.csv", complete=False)

    result = pipeline.run(initialized.run_dir, to_stage="stage02_segment_keyframes")

    assert result.status == "WAITING_FOR_METADATA"
    assert (result.run_dir / "receipts/stage01_sanitization.json").is_file()
    assert (result.run_dir / "receipts/stage02_segment_keyframes.json").is_file()
    blocked = pipeline.run(
        result.run_dir, from_stage="stage03_retrieval", to_stage="stage03_retrieval"
    )
    assert blocked.status == "WAITING_FOR_METADATA"
    assert not (result.run_dir / "receipts/stage03_retrieval.json").exists()


def test_final_mapping_requires_hash_bound_human_approval_and_resumes(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "map.MP4").write_bytes(b"video")
    pipeline = SitePipeline(PipelineConfig.from_dict({"site_name": "test"}), handlers=_handlers())
    initialized = pipeline.initialize(corpus, tmp_path / "run")
    _write_metadata(initialized.run_dir / "inputs/metadata.csv", complete=True)

    diagnostic = pipeline.run(initialized.run_dir, to_stage="stage11_reinforcement")

    assert diagnostic.status == "APPROVAL_REQUIRED"
    decision = diagnostic.run_dir / "decisions/final_build_decision.json"
    decision_sha = hashlib.sha256(decision.read_bytes()).hexdigest()
    with pytest.raises(ApprovalRequired):
        pipeline.run(
            diagnostic.run_dir,
            from_stage="stage12_final_mapping",
            to_stage="stage12_final_mapping",
        )

    approval = pipeline.approve_final(diagnostic.run_dir, decision_sha, approver="tester")
    assert json.loads(approval.read_text())["decision_sha256"] == decision_sha
    geometry = diagnostic.run_dir / "artifacts/pairs/geometry.jsonl"
    original_geometry = geometry.read_bytes()
    geometry.write_text('{"changed": true}\n')
    with pytest.raises(ApprovalRequired, match="input changed"):
        pipeline.run(
            diagnostic.run_dir,
            from_stage="stage12_final_mapping",
            to_stage="stage12_final_mapping",
        )
    geometry.write_bytes(original_geometry)
    completed = pipeline.run(
        diagnostic.run_dir,
        from_stage="stage12_final_mapping",
        to_stage="stage15_publish",
    )
    assert completed.status == "COMPLETED"
    assert (completed.run_dir / "products/selection_manifest.csv").exists()

    decision.write_text('{"changed": true}\n', encoding="utf-8")
    with pytest.raises(ApprovalRequired, match="does not match"):
        pipeline.run(
            diagnostic.run_dir,
            from_stage="stage12_final_mapping",
            to_stage="stage12_final_mapping",
            force=True,
        )


def test_stage_receipts_cache_by_inputs_and_config(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "map.MP4").write_bytes(b"video")
    pipeline = SitePipeline(PipelineConfig.from_dict({"site_name": "test"}), handlers=_handlers())
    initialized = pipeline.initialize(corpus, tmp_path / "run")
    _write_metadata(initialized.run_dir / "inputs/metadata.csv", complete=True)

    first = pipeline.run(
        initialized.run_dir,
        from_stage="stage01_sanitization",
        to_stage="stage02_segment_keyframes",
    )
    second = pipeline.run(
        initialized.run_dir,
        from_stage="stage01_sanitization",
        to_stage="stage02_segment_keyframes",
    )

    assert [stage.status for stage in first.stages] == ["COMPLETED", "COMPLETED"]
    assert [stage.status for stage in second.stages] == ["CACHED", "CACHED"]


def test_stage_cache_invalidates_when_an_output_is_modified(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "map.MP4").write_bytes(b"video")
    pipeline = SitePipeline(PipelineConfig.from_dict({"site_name": "test"}), handlers=_handlers())
    initialized = pipeline.initialize(corpus, tmp_path / "run")
    _write_metadata(initialized.run_dir / "inputs/metadata.csv", complete=True)
    pipeline.run(
        initialized.run_dir,
        from_stage="stage01_sanitization",
        to_stage="stage01_sanitization",
    )
    output = initialized.run_dir / "artifacts/sanitization/frames.jsonl"
    output.write_text('{"tampered": true}\n', encoding="utf-8")

    rerun = pipeline.run(
        initialized.run_dir,
        from_stage="stage01_sanitization",
        to_stage="stage01_sanitization",
    )

    assert rerun.stages[0].status == "COMPLETED"
    assert json.loads(output.read_text())["stage"] == "stage01_sanitization"


def test_stage_cache_invalidates_when_implementation_digest_changes(
    tmp_path: Path, monkeypatch
) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "map.MP4").write_bytes(b"video")
    pipeline = SitePipeline(PipelineConfig.from_dict({"site_name": "test"}), handlers=_handlers())
    initialized = pipeline.initialize(corpus, tmp_path / "run")
    _write_metadata(initialized.run_dir / "inputs/metadata.csv", complete=True)
    pipeline.run(
        initialized.run_dir,
        from_stage="stage01_sanitization",
        to_stage="stage01_sanitization",
    )
    monkeypatch.setattr(
        "sfm_diagnosis.site_pipeline.pipeline._implementation_fingerprint",
        lambda: "changed-implementation",
    )

    rerun = pipeline.run(
        initialized.run_dir,
        from_stage="stage01_sanitization",
        to_stage="stage01_sanitization",
    )

    assert rerun.stages[0].status == "COMPLETED"


def test_run_rejects_a_raw_source_that_changed_after_inventory(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    video = corpus / "map.MP4"
    video.write_bytes(b"immutable-video")
    pipeline = SitePipeline(PipelineConfig.from_dict({"site_name": "test"}), handlers=_handlers())
    initialized = pipeline.initialize(corpus, tmp_path / "run")
    _write_metadata(initialized.run_dir / "inputs/metadata.csv", complete=True)
    video.write_bytes(b"tampered--video")

    with pytest.raises(RuntimeError, match="immutable corpus source changed"):
        pipeline.run(
            initialized.run_dir,
            from_stage="stage01_sanitization",
            to_stage="stage01_sanitization",
        )


def test_stage02_cache_validates_referenced_keyframe_images(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "map.MP4").write_bytes(b"video")
    handlers = _handlers()

    def stage02(context: StageContext) -> StageOutcome:
        image = context.run_dir / "artifacts/keyframes/images/v/frame.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        image.write_bytes(b"original-keyframe")
        segments, keyframes = context.expected_outputs
        segments.parent.mkdir(parents=True, exist_ok=True)
        segments.write_text('{"segment_id": "v:S01"}\n', encoding="utf-8")
        keyframes.write_text(
            json.dumps(
                {
                    "keyframe_id": "v:00000000",
                    "segment_id": "v:S01",
                    "image_uri": str(image),
                    "image_sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return StageOutcome(context.expected_outputs)

    handlers["stage02_segment_keyframes"] = stage02
    pipeline = SitePipeline(PipelineConfig.from_dict({"site_name": "test"}), handlers=handlers)
    initialized = pipeline.initialize(corpus, tmp_path / "run")
    _write_metadata(initialized.run_dir / "inputs/metadata.csv", complete=True)
    pipeline.run(
        initialized.run_dir,
        from_stage="stage01_sanitization",
        to_stage="stage02_segment_keyframes",
    )
    image = initialized.run_dir / "artifacts/keyframes/images/v/frame.jpg"
    image.write_bytes(b"tampered-keyframe")

    rerun = pipeline.run(
        initialized.run_dir,
        from_stage="stage02_segment_keyframes",
        to_stage="stage02_segment_keyframes",
    )

    assert rerun.stages[0].status == "COMPLETED"
    assert image.read_bytes() == b"original-keyframe"


def test_blocked_reinforcement_decision_cannot_be_human_approved(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "map.MP4").write_bytes(b"video")
    handlers = _handlers()

    def blocked(context: StageContext) -> StageOutcome:
        for output in context.expected_outputs:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text("{}\n")
        context.expected_outputs[0].write_text(
            json.dumps(
                {
                    "approval_allowed": False,
                    "issues": ["RESHOOT_REQUIRED"],
                    "input_hashes": {},
                }
            )
            + "\n"
        )
        return StageOutcome(context.expected_outputs)

    handlers["stage11_reinforcement"] = blocked
    pipeline = SitePipeline(PipelineConfig.from_dict({"site_name": "test"}), handlers=handlers)
    initialized = pipeline.initialize(corpus, tmp_path / "run")
    _write_metadata(initialized.run_dir / "inputs/metadata.csv", complete=True)

    result = pipeline.run(initialized.run_dir, to_stage="stage11_reinforcement")
    decision = result.run_dir / "decisions/final_build_decision.json"

    assert result.status == "BLOCKED"
    with pytest.raises(ApprovalRequired, match="blocked"):
        pipeline.approve_final(
            result.run_dir,
            hashlib.sha256(decision.read_bytes()).hexdigest(),
            approver="operator",
        )


def test_holdout_keyframe_cannot_enter_any_mapping_stage(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "map.MP4").write_bytes(b"map")
    (corpus / "holdout.MP4").write_bytes(b"holdout")
    pipeline = SitePipeline(
        PipelineConfig.from_dict({"site_name": "test", "heldout_patterns": ["holdout.MP4"]}),
        handlers=_handlers(),
    )
    initialized = pipeline.initialize(corpus, tmp_path / "run")
    _write_metadata(initialized.run_dir / "inputs/metadata.csv", complete=True)
    pipeline.run(initialized.run_dir, to_stage="stage02_segment_keyframes")
    manifest = json.loads((initialized.run_dir / "inputs/corpus_manifest.json").read_text())
    holdout_id = next(
        row["video_id"] for row in manifest["sources"] if row["evaluation_role"] == "HOLDOUT"
    )
    keyframes = initialized.run_dir / "artifacts/keyframes/keyframes.jsonl"
    keyframes.write_text(
        json.dumps(
            {
                "keyframe_id": "leak",
                "segment_id": "holdout-segment",
                "video_id": holdout_id,
                "evaluation_role": "HOLDOUT",
            }
        )
        + "\n"
    )

    result = pipeline.run(
        initialized.run_dir,
        from_stage="stage03_retrieval",
        to_stage="stage03_retrieval",
    )

    assert result.status == "BLOCKED"
    assert "hold-out" in result.stages[-1].reason


def test_external_adapter_failure_is_persisted_as_blocked_json_state(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "map.MP4").write_bytes(b"map")
    handlers = _handlers()

    def broken(_context: StageContext) -> StageOutcome:
        raise RuntimeError("worker failed")

    handlers["stage01_sanitization"] = broken
    pipeline = SitePipeline(PipelineConfig.from_dict({"site_name": "test"}), handlers=handlers)
    initialized = pipeline.initialize(corpus, tmp_path / "run")

    result = pipeline.run(
        initialized.run_dir,
        from_stage="stage01_sanitization",
        to_stage="stage01_sanitization",
    )

    assert result.status == "BLOCKED"
    assert result.stages[-1].status == "FAILED"
    failure = result.run_dir / "receipts/stage01_sanitization.failed.json"
    assert json.loads(failure.read_text())["reason"] == "worker failed"


def test_inventory_rejects_symlinked_raw_source(tmp_path: Path) -> None:
    outside = tmp_path / "outside.MP4"
    outside.write_bytes(b"outside")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "linked.MP4").symlink_to(outside)
    pipeline = SitePipeline(PipelineConfig.from_dict({"site_name": "test"}))

    with pytest.raises(ValueError, match="symlinks"):
        pipeline.initialize(corpus, tmp_path / "run")
