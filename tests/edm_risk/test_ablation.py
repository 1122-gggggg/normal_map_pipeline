from sfm_diagnosis.edm_risk.ablation import run_ablation, select_variant_samples
from sfm_diagnosis.edm_risk.calibration import (
    RiskCalibrationConfig,
    RiskTrainingSample,
)


def _samples() -> list[RiskTrainingSample]:
    samples = []
    for group_index in range(4):
        for index in range(12):
            failure = index >= 6
            samples.append(
                RiskTrainingSample(
                    query_id=f"S{group_index}/q{index}",
                    group=f"S{group_index}",
                    failure=failure,
                    features={
                        "effective_landmarks": 20.0 if failure else 200.0,
                        "parallax_p10_deg": 0.2 if failure else 4.0,
                        "fim_logdet": 1.0 if failure else 9.0,
                        "actloc_score": 0.1 if failure else 0.9,
                        "empirical_edm_success_rate": 0.1 if failure else 0.9,
                        "mode_entropy": 0.9 if failure else 0.1,
                    },
                )
            )
    return samples


def test_ablation_runs_requested_a_to_f_with_group_safe_metrics() -> None:
    report = run_ablation(
        _samples(),
        config=RiskCalibrationConfig(splitter="group_kfold", folds=2),
    )

    assert list(report["variants"]) == ["A", "B", "C", "D", "E", "F"]
    assert all(
        variant["group_leakage_detected"] is False
        for variant in report["variants"].values()
    )
    assert all("roc_auc" in variant["metrics"] for variant in report["variants"].values())
    assert report["variants"]["F"]["feature_count"] >= report["variants"]["A"]["feature_count"]


def test_variant_d_keeps_empirical_edm_and_excludes_actloc() -> None:
    selected = select_variant_samples(_samples(), "D")

    assert "empirical_edm_success_rate" in selected[0].features
    assert "fim_logdet" in selected[0].features
    assert "actloc_score" not in selected[0].features
