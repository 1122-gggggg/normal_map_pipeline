from sfm_diagnosis.edm_risk.edm_loo import EDMQueryResult
from sfm_diagnosis.edm_risk.fusion import (
    apply_calibrated_risk,
    build_training_samples,
    enrich_rows_with_ambiguity,
    enrich_rows_with_temporal,
)
from sfm_diagnosis.edm_risk.schema import RiskClass, SpatialDiagnostic


def _rows():
    return [
        SpatialDiagnostic(
            position=(0.0, 0.0, 0.0),
            yaw_deg=0.0,
            pitch_deg=0.0,
            visible_landmarks=80,
            effective_landmarks=100.0,
            fim_logdet=10.0,
        ),
        SpatialDiagnostic(
            position=(0.0, 0.0, 0.0),
            yaw_deg=180.0,
            pitch_deg=0.0,
            visible_landmarks=40,
            effective_landmarks=20.0,
            fim_logdet=2.0,
        ),
    ]


def test_edm_queries_align_to_nearest_grid_orientation_and_keep_session_group() -> None:
    result = EDMQueryResult(
        query_id="S4/q0",
        session_id="S4",
        timestamp=0.0,
        success=False,
        registration_success=True,
        estimated_position=(0.0, 0.0, 0.0),
        estimated_yaw_deg=170.0,
        estimated_pitch_deg=0.0,
    )
    samples, receipt = build_training_samples(_rows(), [result])

    assert len(samples) == 1
    assert samples[0].group == "S4"
    assert samples[0].failure is True
    assert samples[0].features["fim_logdet"] == 2.0
    assert receipt["unmatched_queries"] == 0


def test_apply_calibration_populates_probability_then_position_class() -> None:
    class FakeCalibrator:
        def predict_proba(self, features):
            return [0.05, 0.9]

    rows = _rows()
    apply_calibrated_risk(rows, FakeCalibrator())

    assert [row.failure_probability for row in rows] == [0.05, 0.9]
    assert {row.risk_class for row in rows} == {RiskClass.VIEW_DIRECTION_SENSITIVE}


def test_empirical_features_exclude_the_query_session_to_prevent_leakage() -> None:
    results = [
        EDMQueryResult(
            query_id="S1/q0",
            session_id="S1",
            timestamp=0.0,
            success=False,
            registration_success=True,
            ransac_inliers=10,
            estimated_position=(0.0, 0.0, 0.0),
            estimated_yaw_deg=0.0,
            estimated_pitch_deg=0.0,
        ),
        EDMQueryResult(
            query_id="S2/q0",
            session_id="S2",
            timestamp=0.0,
            success=True,
            registration_success=True,
            ransac_inliers=50,
            estimated_position=(0.0, 0.0, 0.0),
            estimated_yaw_deg=0.0,
            estimated_pitch_deg=0.0,
        ),
    ]

    samples, _ = build_training_samples(_rows(), results)
    by_id = {sample.query_id: sample for sample in samples}

    assert by_id["S1/q0"].features["empirical_edm_success_rate"] == 1.0
    assert by_id["S1/q0"].features["empirical_pnp_inliers"] == 50.0
    assert by_id["S2/q0"].features["empirical_edm_success_rate"] == 0.0
    assert by_id["S2/q0"].features["empirical_pnp_inliers"] == 10.0


def test_temporal_and_ambiguity_evidence_enrich_the_aligned_grid_row() -> None:
    results = [
        EDMQueryResult(
            query_id=f"S1/q{i}",
            session_id="S1",
            timestamp=float(i),
            success=success,
            registration_success=success,
            estimated_position=(0.0, 0.0, 0.0),
            estimated_yaw_deg=0.0,
            estimated_pitch_deg=0.0,
        )
        for i, success in enumerate((False, False, True))
    ]
    rows = _rows()

    enrich_rows_with_temporal(rows, results)
    enrich_rows_with_ambiguity(
        rows,
        results,
        {
            "S1/q0": {
                "num_pose_modes": 2,
                "support_margin": 0.1,
                "mode_translation_separation": 4.0,
                "mode_rotation_separation_deg": 15.0,
                "mode_entropy": 0.9,
                "loo_translation_jump": 2.0,
                "loo_rotation_jump_deg": 8.0,
            }
        },
    )

    assert rows[0].max_consecutive_failures == 2
    assert rows[0].consecutive_failure_probability == 0.5
    assert rows[0].recovery_time == 2.0
    assert rows[0].num_pose_modes == 2
    assert rows[0].mode_support_margin == 0.1
    assert rows[0].loo_translation_jump == 2.0
