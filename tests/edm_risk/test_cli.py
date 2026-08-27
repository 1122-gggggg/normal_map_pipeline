from sfm_diagnosis.edm_risk.cli import build_parser, parse_pitch_values


def test_pitch_values_cli_accepts_negative_comma_separated_values() -> None:
    assert parse_pitch_values("-30,0,30") == (-30.0, 0.0, 30.0)


def test_fast_cli_exposes_matchability_and_fim_noise_controls() -> None:
    args = build_parser().parse_args(
        [
            "--map",
            "map",
            "--output",
            "out",
            "--matchability",
            "events.jsonl",
            "--pixel-sigma",
            "1.5",
            "--translation-scale",
            "2.0",
            "--actloc-mode",
            "fallback",
            "--actloc-cache",
            "actloc-cache",
            "--color-images",
            "mapping-images",
        ]
    )
    assert str(args.matchability) == "events.jsonl"
    assert args.pixel_sigma == 1.5
    assert args.translation_scale == 2.0
    assert args.actloc_mode == "fallback"
    assert str(args.actloc_cache) == "actloc-cache"
    assert str(args.color_images) == "mapping-images"


def test_full_cli_exposes_edm_loo_and_group_safe_calibration_controls() -> None:
    args = build_parser().parse_args(
        [
            "--map",
            "map",
            "--output",
            "out",
            "--mode",
            "full",
            "--edm-results",
            "heldout",
            "--loo-mode",
            "strict",
            "--calibration-model",
            "hist_gradient_boosting",
            "--splitter",
            "group_kfold",
            "--folds",
            "4",
            "--target-recall",
            "0.97",
            "--risk-feature-set",
            "D",
        ]
    )

    assert str(args.edm_results) == "heldout"
    assert args.loo_mode == "strict"
    assert args.calibration_model == "hist_gradient_boosting"
    assert args.splitter == "group_kfold"
    assert args.folds == 4
    assert args.target_recall == 0.97
    assert args.risk_feature_set == "D"
