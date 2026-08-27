from sfm_diagnosis.edm_risk.classifier import RiskThresholds, classify_spatial_risk
from sfm_diagnosis.edm_risk.schema import RiskClass, SpatialDiagnostic


def _row(yaw: float, probability: float) -> SpatialDiagnostic:
    return SpatialDiagnostic(
        position=(0.0, 0.0, 0.0),
        yaw_deg=yaw,
        pitch_deg=0.0,
        failure_probability=probability,
        visible_landmarks=20,
        effective_landmarks=10.0,
        parallax_p10_deg=0.5,
        fim_lambda_min=1e-6,
        fim_condition=1e9,
    )


def test_view_direction_sensitive_recommends_reorientation_not_recapture() -> None:
    rows = [_row(0.0, 0.05), _row(180.0, 0.9)]
    classify_spatial_risk(rows, RiskThresholds())

    assert {row.risk_class for row in rows} == {RiskClass.VIEW_DIRECTION_SENSITIVE}
    assert all(row.recommended_action == "REORIENT_CAMERA" for row in rows)
    assert all(row.recommended_yaw_deg == 0.0 for row in rows)


def test_dead_zone_requires_every_reasonable_direction_to_fail() -> None:
    rows = [_row(0.0, 0.9), _row(180.0, 0.85)]
    classify_spatial_risk(rows, RiskThresholds())

    assert {row.risk_class for row in rows} == {RiskClass.DEAD_ZONE}
    assert all(row.recommended_action == "SUPPLEMENTAL_CAPTURE" for row in rows)
    assert "LOW_FIM_EIGENVALUE" in rows[0].primary_failure_causes
    assert "LOW_PARALLAX" in rows[0].primary_failure_causes


def test_stable_low_failure_position_is_good() -> None:
    rows = [_row(0.0, 0.05), _row(180.0, 0.1)]
    classify_spatial_risk(rows, RiskThresholds())
    assert {row.risk_class for row in rows} == {RiskClass.GOOD}
