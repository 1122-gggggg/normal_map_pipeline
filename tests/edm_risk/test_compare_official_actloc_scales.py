import importlib.util
import json
from pathlib import Path

import numpy as np

TOOL = Path(__file__).parents[2] / "examples" / "compare_official_actloc_scales.py"
spec = importlib.util.spec_from_file_location("compare_scales", TOOL)
tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tool)


def artifact(scale, shift=0):
    rows = []
    for i in range(3):
        scores = np.zeros((6, 18)).tolist()
        scores[0][i + shift] = 0.9
        rows.append({"position": [i, 0, 0], "class0_softmax": scores, "crop_point_count": 10 + i})
    return {
        "status": "COMPLETE",
        "attestation": {
            "map": {"coordinate_scale_factor": scale},
            "source_revision": "s",
            "checkpoint_sha256": "c",
            "runner_sha256": "r",
        },
        "positions": rows,
    }


def test_compare_reports_scale_sensitivity(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    a.write_text(json.dumps(artifact(1.0)))
    b.write_text(json.dumps(artifact(2.0, 1)))
    result = tool.compare([a, b])
    assert result["scales"] == [1.0, 2.0]
    assert result["common_position_count"] == 3
    assert result["best_direction_changes"][0]["changed_positions"] == 3
    assert "not calibrated" in result["warning"]


def test_load_complete_rejects_partial(tmp_path):
    path = tmp_path / "partial.json"
    path.write_text(json.dumps({"status": "PARTIAL", "positions": []}))
    try:
        tool.load_complete(path)
    except ValueError:
        pass
    else:
        raise AssertionError("partial artifact must be rejected")


def test_shuffled_rows_align_and_ties_use_average_ranks(tmp_path):
    a = artifact(1.0)
    b = artifact(2.0)
    b["positions"] = list(reversed(b["positions"]))
    for i, row in enumerate(a["positions"]):
        row["class0_softmax"] = np.zeros((6, 18)).tolist()
        row["class0_softmax"][0][0] = [0.9, 0.9, 0.1][i]
    for i, row in enumerate(b["positions"]):
        row["class0_softmax"] = np.zeros((6, 18)).tolist()
        row["class0_softmax"][0][0] = [0.1, 0.9, 0.9][i]
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    result = tool.compare([pa, pb])
    assert result["spearman_correlation"][0][1] == 1.0
    assert [r["position"] for r in result["position_best_ranks"]] == [
        [0, 0, 0],
        [1, 0, 0],
        [2, 0, 0],
    ]
    assert [r["ranks"][0] for r in result["position_best_ranks"]] == [1.5, 1.5, 3.0]


def test_identity_ignores_volatile_fields_but_rejects_checkpoint_mismatch(tmp_path):
    a, b = artifact(1.0), artifact(2.0)
    b["attestation"].update(
        {"command": ["different"], "started_at_utc": "later", "output_dir": "/other"}
    )
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))
    tool.compare([pa, pb])
    b["attestation"]["checkpoint_sha256"] = "different"
    pb.write_text(json.dumps(b))
    import pytest

    with pytest.raises(ValueError, match="identity"):
        tool.compare([pa, pb])


def test_top_positions_follow_the_same_descending_scores_as_overlap(tmp_path):
    a, b = artifact(1.0), artifact(2.0)
    for payload in (a, b):
        for score, row in zip((0.1, 0.9, 0.5), payload["positions"]):
            row["class0_softmax"] = np.zeros((6, 18)).tolist()
            row["class0_softmax"][0][0] = score
    pa, pb = tmp_path / "a.json", tmp_path / "b.json"
    pa.write_text(json.dumps(a))
    pb.write_text(json.dumps(b))

    result = tool.compare([pa, pb], top_quantile=0.34)

    assert result["top_positions"][0] == [[1, 0, 0], [2, 0, 0]]
