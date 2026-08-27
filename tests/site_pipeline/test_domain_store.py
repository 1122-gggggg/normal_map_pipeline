import sqlite3
from dataclasses import FrozenInstanceError

import pytest

from sfm_diagnosis.site_pipeline.domain import (
    DataStatus,
    PostSfmRole,
    ArtifactRef,
    SegmentRecord,
    content_id,
    derive_status,
    legacy_session_role,
)
from sfm_diagnosis.site_pipeline.store import Ledger


def test_enums_and_immutable_deterministic_records():
    assert DataStatus.ACTIVE_CORE.value == "ACTIVE_CORE"
    assert content_id(b"abc") == content_id(b"abc")
    rec = SegmentRecord(video_id="v", session_id="s", segment_id="seg", source_uri="x")
    with pytest.raises(FrozenInstanceError):
        rec.segment_id = "other"


def test_status_derivation_and_legacy_projection():
    assert derive_status(PostSfmRole.CORE) is DataStatus.ACTIVE_CORE
    assert derive_status(PostSfmRole.BRIDGE) is DataStatus.ACTIVE_BRIDGE
    assert derive_status(PostSfmRole.UPDATE_ONLY) is DataStatus.INACTIVE_UPDATE
    assert derive_status(PostSfmRole.REJECT) is DataStatus.INACTIVE_REJECT
    assert legacy_session_role(PostSfmRole.CORE) == "BASE_CORE"
    assert legacy_session_role(PostSfmRole.BRIDGE) == "BASE_SUPPORT"


def test_ledger_append_only_artifacts_and_exports(tmp_path):
    db = tmp_path / "ledger.sqlite"
    ledger = Ledger(db)
    ledger.add_raw_source("v1", "/data/v.mp4", "sha-v", {"fps": 30})
    ledger.add_segment(SegmentRecord("v1", "s1", "seg1", "/data/v.mp4"), run_id="r1")
    ledger.add_artifact("r1", "matches", ArtifactRef("artifacts/a.npz", "sha-a"))
    ledger.add_role("r1", "seg1", PostSfmRole.CORE, reason="good geometry")
    assert ledger.get_segments("r1")[0]["segment_id"] == "seg1"
    assert ledger.export_jsonl("r1", "segments").read_text().count("seg1") == 1
    assert ledger.export_csv("r1", "roles").read_text().splitlines()[0].startswith("run_id,")
    with pytest.raises(sqlite3.IntegrityError):
        ledger.connection.execute("UPDATE raw_sources SET source_uri='x' WHERE video_id='v1'")
    with pytest.raises(sqlite3.IntegrityError):
        ledger.connection.execute("DELETE FROM raw_sources WHERE video_id='v1'")


def test_re_registering_content_id_with_changed_metadata_is_not_silent(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite")
    ledger.register_source({"source_id": "v1", "path": "/data/v.mp4", "sha256": "sha", "fps": 24})
    with pytest.raises(sqlite3.IntegrityError):
        ledger.register_source(
            {"source_id": "v1", "path": "/data/v.mp4", "sha256": "sha", "fps": 30}
        )
