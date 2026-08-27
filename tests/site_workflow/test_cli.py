from sfm_diagnosis.site_workflow.cli import build_parser


def test_site_workflow_cli_accepts_multiple_videos_and_resume_bounds() -> None:
    args = build_parser().parse_args(
        [
            "--config",
            "site.toml",
            "--videos",
            "a.mp4",
            "b.mp4",
            "--output",
            "run",
            "--from-stage",
            "mapping",
            "--to-stage",
            "risk_diagnosis",
        ]
    )

    assert [path.name for path in args.videos] == ["a.mp4", "b.mp4"]
    assert args.from_stage == "mapping"
    assert args.to_stage == "risk_diagnosis"
