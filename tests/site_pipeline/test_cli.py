from __future__ import annotations

from sfm_diagnosis.site_pipeline.cli import build_parser


def test_cli_has_init_run_status_approve_export_and_backfill_commands() -> None:
    parser = build_parser()

    assert (
        parser.parse_args(
            ["init", "--config", "site.toml", "--corpus", "raw", "--output", "run"]
        ).command
        == "init"
    )
    assert (
        parser.parse_args(["run", "--run", "run", "--to-stage", "stage11_reinforcement"]).command
        == "run"
    )
    assert parser.parse_args(["status", "--run", "run"]).command == "status"
    assert (
        parser.parse_args(
            [
                "approve-final",
                "--run",
                "run",
                "--decision-sha",
                "a" * 64,
                "--approver",
                "operator",
            ]
        ).command
        == "approve-final"
    )
    assert parser.parse_args(["export", "--run", "run"]).command == "export"
    assert (
        parser.parse_args(["backfill", "--run", "run", "--legacy-run", "legacy"]).command
        == "backfill"
    )
    assert (
        parser.parse_args(["normalize-selection", "--run", "run"]).command
        == "normalize-selection"
    )
    assert (
        parser.parse_args(
            [
                "enrich-bridge",
                "--run",
                "run",
                "--keyframe",
                "frame-a",
                "--attempt-name",
                "repair-1",
            ]
        ).command
        == "enrich-bridge"
    )
    assert (
        parser.parse_args(["materialize-layers", "--run", "run"]).command
        == "materialize-layers"
    )
    assert (
        parser.parse_args(["validate-layers", "--run", "run"]).command
        == "validate-layers"
    )
    assert (
        parser.parse_args(
            [
                "fuse-localization",
                "--run",
                "run",
                "--robust-validation",
                "robust.json",
                "--dense-validation",
                "dense.json",
            ]
        ).command
        == "fuse-localization"
    )
