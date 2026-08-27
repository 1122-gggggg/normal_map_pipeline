import json
import sys
from pathlib import Path

import pytest

from sfm_diagnosis.site_workflow import SiteWorkflow, WorkflowConfig


def _config(tmp_path: Path, *, missing_mapping_output: bool = False) -> WorkflowConfig:
    selection_code = (
        "from pathlib import Path; "
        "p=Path(r'{selection_manifest}'); p.parent.mkdir(parents=True,exist_ok=True); "
        "p.write_text('{\"selected\": 2}')"
    )
    mapping_code = (
        "from pathlib import Path; "
        "p=Path(r'{map_model}'); p.mkdir(parents=True,exist_ok=True); "
        + ("" if missing_mapping_output else "(p/'points3D.bin').write_bytes(b'map')")
    )
    return WorkflowConfig.from_dict(
        {
            "site_name": "other_site",
            "heldout_patterns": ["b.mp4"],
            "stages": [
                {
                    "name": "selection",
                    "command": [sys.executable, "-c", selection_code],
                    "produces": ["{selection_manifest}"],
                },
                {
                    "name": "mapping",
                    "command": [sys.executable, "-c", mapping_code],
                    "requires": ["{selection_manifest}"],
                    "produces": ["{map_model}/points3D.bin"],
                },
            ],
            "paths": {
                "selection_manifest": "selection/manifest.json",
                "map_model": "map/model",
            },
        }
    )


def test_site_workflow_runs_ordered_stages_and_caches_verified_outputs(tmp_path: Path) -> None:
    videos = [tmp_path / "a.mp4", tmp_path / "b.mp4"]
    for video in videos:
        video.write_bytes(b"video")
    run_dir = tmp_path / "run"
    workflow = SiteWorkflow(_config(tmp_path))

    first = workflow.run(videos, run_dir)
    second = workflow.run(videos, run_dir)

    assert [stage.name for stage in first.stages] == ["inventory", "selection", "mapping"]
    assert [stage.status for stage in first.stages] == ["completed", "completed", "completed"]
    assert [stage.status for stage in second.stages] == ["cached", "cached", "cached"]
    manifest = json.loads((run_dir / "inputs" / "videos.json").read_text())
    assert [Path(row["path"]).name for row in manifest["videos"]] == ["a.mp4", "b.mp4"]
    assert [row["role"] for row in manifest["videos"]] == ["mapping", "heldout"]
    assert (run_dir / "workflow_receipt.json").is_file()


def test_site_workflow_fails_closed_when_stage_does_not_produce_contract(tmp_path: Path) -> None:
    video = tmp_path / "a.mp4"
    video.write_bytes(b"video")

    with pytest.raises(RuntimeError, match="did not produce"):
        SiteWorkflow(_config(tmp_path, missing_mapping_output=True)).run(
            [video], tmp_path / "run"
        )
