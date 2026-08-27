from sfm_diagnosis.edm_risk.schema import RiskClass, SpatialDiagnostic


def test_spatial_diagnostic_serializes_required_unknown_contract() -> None:
    row = SpatialDiagnostic(position=(1.0, 2.0, 3.0), yaw_deg=90.0, pitch_deg=0.0)

    payload = row.to_dict()

    assert payload["position"] == [1.0, 2.0, 3.0]
    assert payload["risk_class"] == RiskClass.UNKNOWN.value
    assert payload["failure_probability"] is None
    assert payload["actloc_score"] is None
    assert payload["fim_lambda_min"] is None
    assert payload["edm_loo_success_rate"] is None
    assert payload["primary_failure_causes"] == []
    assert payload["recommended_action"] is None
    assert payload["visibility_source"] == "frustum_only"
    assert payload["occlusion_verified"] is False
