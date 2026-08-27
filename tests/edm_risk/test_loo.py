from pathlib import Path

import pytest

from sfm_diagnosis.edm_risk.edm_loo import (
    EDMLOOMode,
    EDMLOORunner,
    EDMQuery,
    EDMQueryResult,
    EDMReferenceIndex,
)


class FakeProvider:
    fingerprint = "fake-edm-v1"

    def __init__(self, *, leak: bool = False):
        self.leak = leak
        self.localize_calls = 0
        self.exclusions = []

    def build_reference_index(self, *, excluded_sessions, strict):
        self.exclusions.append((frozenset(excluded_sessions), strict))
        heldout = next(iter(excluded_sessions))
        sessions = {"S1", "S2"} - set(excluded_sessions)
        if self.leak:
            sessions.add(heldout)
        return EDMReferenceIndex(
            index_id=f"index-without-{heldout}",
            reference_sessions=frozenset(sessions),
            session_only_landmarks_removed=strict,
            appearance_descriptors_rebuilt=strict,
        )

    def localize(self, query, index):
        self.localize_calls += 1
        reference_session = sorted(index.reference_sessions)[0]
        return EDMQueryResult(
            query_id=query.query_id,
            session_id=query.session_id,
            timestamp=query.timestamp,
            success=True,
            registration_success=True,
            raw_matches=100,
            valid_2d3d=80,
            ransac_inliers=60,
            inlier_ratio=0.75,
            reprojection_p90=2.0,
            reference_ids=(f"{reference_session}/ref.jpg",),
        )


def _queries():
    return {
        "S1": [EDMQuery("S1/q0.jpg", "S1", 0.0)],
        "S2": [EDMQuery("S2/q0.jpg", "S2", 0.0)],
    }


def test_strict_loo_excludes_session_and_reuses_cached_results(tmp_path: Path) -> None:
    provider = FakeProvider()
    runner = EDMLOORunner(provider=provider, cache_dir=tmp_path)

    first = runner.run(_queries(), mode=EDMLOOMode.STRICT)
    second = runner.run(_queries(), mode=EDMLOOMode.STRICT)

    assert first.strict_loo is True
    assert first.pseudo_loo is False
    assert provider.exclusions == [(frozenset({"S1"}), True), (frozenset({"S2"}), True)] * 2
    assert provider.localize_calls == 2
    assert second.cache_hits == 2


def test_loo_fails_closed_when_reference_index_contains_heldout_session(tmp_path: Path) -> None:
    runner = EDMLOORunner(provider=FakeProvider(leak=True), cache_dir=tmp_path)
    with pytest.raises(RuntimeError, match="held-out session"):
        runner.run(_queries(), mode=EDMLOOMode.REFERENCE_EXCLUSION)


def test_reference_exclusion_is_labeled_pseudo_loo(tmp_path: Path) -> None:
    result = EDMLOORunner(provider=FakeProvider(), cache_dir=tmp_path).run(
        _queries(), mode=EDMLOOMode.REFERENCE_EXCLUSION
    )
    assert result.strict_loo is False
    assert result.pseudo_loo is True
