from __future__ import annotations

from pathlib import Path

import pytest

from sfm_diagnosis.site_pipeline.config import PipelineConfig
from sfm_diagnosis.site_pipeline.pipeline import StageContext, StageOutcome
from sfm_diagnosis.site_pipeline import handlers
from sfm_diagnosis.site_pipeline.mapping_optimization import (
    MappingOptimizationContract,
    optimization_adapter_payload,
    validate_optimizer_result,
)


def test_mapping_optimization_contract_requires_all_configured_methods() -> None:
    contract = MappingOptimizationContract.from_mapping(
        {
            "enabled": True,
            "required_methods": [
                "robust_filter_sweep",
                "connectivity_aware_rescue",
                "connector_frame_rewire",
                "fixed_pose_retriangulation",
            ],
        }
    )

    assert contract.enabled is True
    assert contract.required_methods == (
        "robust_filter_sweep",
        "connectivity_aware_rescue",
        "connector_frame_rewire",
        "fixed_pose_retriangulation",
    )


def test_mapping_optimization_payload_is_bound_to_run_inputs(tmp_path: Path) -> None:
    run = tmp_path / "run"
    dense = run / "artifacts/mapping/optimization/source/model"
    output = run / "artifacts/mapping/final/model"
    contract = MappingOptimizationContract.from_mapping({"enabled": True})

    payload = optimization_adapter_payload(run, dense, output, contract)

    assert payload["input_model"] == str(dense)
    assert payload["output_model"] == str(output)
    assert payload["selection"].endswith("decisions/final_selection.json")
    assert payload["pair_geometry"].endswith("artifacts/pairs/geometry.jsonl")
    assert payload["corpus_manifest"].endswith("inputs/corpus_manifest.json")
    assert payload["comparison"].endswith("artifacts/mapping/optimization/comparison.json")


def test_mapping_optimization_result_fails_closed_without_a_gate_passing_winner() -> None:
    contract = MappingOptimizationContract.from_mapping({"enabled": True})

    with pytest.raises(RuntimeError, match="geometry gate"):
        validate_optimizer_result(
            {
                "status": "completed",
                "winner": {"geometry_gate_pass": False},
                "methods_completed": list(contract.required_methods),
            },
            contract,
        )


def test_mapping_optimization_result_requires_every_declared_method() -> None:
    contract = MappingOptimizationContract.from_mapping({"enabled": True})

    with pytest.raises(RuntimeError, match="missing required methods"):
        validate_optimizer_result(
            {
                "status": "completed",
                "winner": {"geometry_gate_pass": True, "model": "/model"},
                "methods_completed": ["robust_filter_sweep"],
            },
            contract,
        )


def test_stage12_maps_to_dense_then_runs_the_optimizer_when_enabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    final_model = run / "artifacts/mapping/final/model"
    context = StageContext(
        run,
        "stage12_final_mapping",
        PipelineConfig.from_dict(
            {
                "site_name": "test",
                "mapping_optimization": {"enabled": True},
            }
        ),
        (final_model,),
        "fingerprint",
    )
    calls: list[tuple[str, Path]] = []

    def fake_mapper(dense_context: StageContext, adapter_name: str) -> StageOutcome:
        dense_model = dense_context.expected_outputs[0]
        dense_model.mkdir(parents=True)
        calls.append((adapter_name, dense_model))
        return StageOutcome(dense_context.expected_outputs, {"adapter": adapter_name})

    def fake_optimizer(
        stage_context: StageContext,
        dense_model: Path,
        contract: MappingOptimizationContract,
    ) -> StageOutcome:
        final_model.mkdir(parents=True)
        calls.append(("mapping_optimizer", dense_model))
        return StageOutcome(stage_context.expected_outputs, {"winner": "connected"})

    monkeypatch.setattr(handlers, "_run_adapter", fake_mapper)
    monkeypatch.setattr(handlers, "_run_mapping_optimizer", fake_optimizer)

    outcome = handlers.final_mapping_stage(context)

    expected_dense = run / "artifacts/mapping/optimization/source/model"
    assert calls == [("final_mapper", expected_dense), ("mapping_optimizer", expected_dense)]
    assert outcome.outputs == (final_model,)
    assert outcome.details["optimization_enabled"] is True


def test_stage12_keeps_the_existing_mapper_path_when_optimization_is_disabled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = tmp_path / "run"
    final_model = run / "artifacts/mapping/final/model"
    context = StageContext(
        run,
        "stage12_final_mapping",
        PipelineConfig.from_dict({"site_name": "test"}),
        (final_model,),
        "fingerprint",
    )
    calls: list[str] = []

    def fake_mapper(stage_context: StageContext, adapter_name: str) -> StageOutcome:
        calls.append(adapter_name)
        return StageOutcome(stage_context.expected_outputs, {"adapter": adapter_name})

    monkeypatch.setattr(handlers, "_run_adapter", fake_mapper)

    outcome = handlers.final_mapping_stage(context)

    assert calls == ["final_mapper"]
    assert outcome.outputs == (final_model,)
