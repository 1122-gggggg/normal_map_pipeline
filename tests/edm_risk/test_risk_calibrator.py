import numpy as np

from sfm_diagnosis.edm_risk.calibration import (
    RiskCalibrationConfig,
    RiskCalibrator,
    RiskTrainingSample,
)


def _samples() -> list[RiskTrainingSample]:
    samples = []
    for group_index in range(6):
        for index in range(20):
            weak = index >= 10
            samples.append(
                RiskTrainingSample(
                    query_id=f"S{group_index}/q{index}",
                    group=f"S{group_index}",
                    failure=weak,
                    features={
                        "fim_logdet": 2.0 if weak else 10.0,
                        "effective_landmarks": 20.0 if weak else 200.0,
                        "mode_entropy": 0.9 if weak else 0.1,
                    },
                )
            )
    return samples


def test_logistic_calibration_is_group_safe_and_selects_recall_threshold() -> None:
    result = RiskCalibrator(
        RiskCalibrationConfig(
            model_type="logistic",
            splitter="group_kfold",
            folds=3,
            target_recall=0.95,
            random_state=7,
        )
    ).fit(_samples())

    assert result.group_leakage_detected is False
    assert result.metrics["roc_auc"] > 0.95
    assert result.metrics["pr_auc"] > 0.95
    assert result.metrics["recall"] >= 0.95
    assert result.metrics["false_negative_rate"] <= 0.05
    assert len(result.out_of_fold_probabilities) == len(_samples())
    assert all(set(row["train_groups"]).isdisjoint(row["test_groups"]) for row in result.folds)


def test_hist_gradient_boosting_uses_same_group_contract() -> None:
    result = RiskCalibrator(
        RiskCalibrationConfig(
            model_type="hist_gradient_boosting",
            splitter="leave_one_group_out",
            target_recall=0.90,
        )
    ).fit(_samples())
    probabilities = result.calibrator.predict_proba(
        [{"fim_logdet": 1.0, "effective_landmarks": 10.0, "mode_entropy": 1.0}]
    )
    assert probabilities.shape == (1,)
    assert 0.0 <= probabilities[0] <= 1.0


def test_max_fnr_is_converted_to_target_recall() -> None:
    config = RiskCalibrationConfig(model_type="logistic", max_fnr=0.02)
    assert np.isclose(config.resolved_target_recall, 0.98)


def test_calibration_drops_features_that_are_missing_for_every_training_sample() -> None:
    samples = [
        RiskTrainingSample(
            query_id=sample.query_id,
            group=sample.group,
            failure=sample.failure,
            features={**sample.features, "never_observed": None},
        )
        for sample in _samples()
    ]

    result = RiskCalibrator(RiskCalibrationConfig(splitter="group_kfold", folds=3)).fit(samples)

    assert "never_observed" not in result.feature_names
