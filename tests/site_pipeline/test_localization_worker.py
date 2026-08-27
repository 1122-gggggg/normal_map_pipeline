from __future__ import annotations

import json
import csv
import pytest

from sfm_diagnosis.edm_risk.edm_loo import EDMQueryResult, EDMReferenceIndex
from sfm_diagnosis.site_pipeline.localization_worker import run_adapter_request


class Provider:
    fingerprint = "fake"

    def build_reference_index(self, *, excluded_sessions, strict):
        return EDMReferenceIndex(
            index_id="index",
            reference_sessions=frozenset({"map"}),
            session_only_landmarks_removed=strict,
            appearance_descriptors_rebuilt=strict,
        )

    def localize(self, query, index):
        return EDMQueryResult(
            query.query_id,
            query.session_id,
            query.timestamp,
            True,
            True,
            ransac_inliers=100,
            reference_ids=("map/frame.jpg",),
            loo_mode="strict",
        )


def test_localization_worker_runs_strict_loo_and_exports_reference_layer(tmp_path) -> None:
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"query_id": "q1", "session_id": "holdout", "timestamp": 0.0}) + "\n"
    )
    roles = tmp_path / "roles.jsonl"
    roles.write_text(
        json.dumps(
            {
                "segment_id": "s1",
                "post_sfm_role": "UPDATE_ONLY",
                "base_map": "EXCLUDE",
                "localization": "INCLUDE",
            }
        )
        + "\n"
    )
    validation, references = tmp_path / "validation.json", tmp_path / "references.jsonl"

    result = run_adapter_request(
        {
            "config": {"query_manifest": str(queries), "loo_mode": "strict"},
            "payload": {
                "roles": str(roles),
                "output_validation": str(validation),
                "output_references": str(references),
            },
        },
        provider_factory=lambda _: Provider(),
    )

    assert result["strict_loo"] is True
    assert json.loads(validation.read_text())["strict_loo"] is True
    assert json.loads(references.read_text())["post_sfm_role"] == "UPDATE_ONLY"


def test_localization_worker_rejects_query_session_not_declared_holdout(tmp_path) -> None:
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        json.dumps({"query_id": "q1", "session_id": "mapping", "timestamp": 0.0}) + "\n"
    )
    roles = tmp_path / "roles.jsonl"
    roles.write_text("{}\n")
    corpus = tmp_path / "corpus.json"
    corpus.write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "source_id": "h",
                        "video_id": "h",
                        "path": "/holdout.mp4",
                        "evaluation_role": "HOLDOUT",
                    }
                ]
            }
        )
    )
    metadata = tmp_path / "metadata.csv"
    with metadata.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["source_id", "session_id"])
        writer.writeheader()
        writer.writerow({"source_id": "h", "session_id": "holdout"})

    with pytest.raises(RuntimeError, match="not declared hold-outs"):
        run_adapter_request(
            {
                "config": {"query_manifest": str(queries)},
                "payload": {
                    "roles": str(roles),
                    "corpus": str(corpus),
                    "metadata": str(metadata),
                    "output_validation": str(tmp_path / "validation.json"),
                    "output_references": str(tmp_path / "references.jsonl"),
                },
            },
            provider_factory=lambda _: Provider(),
        )
