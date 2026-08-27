from sfm_diagnosis.edm_risk.edm_loo import EDMQueryResult
from sfm_diagnosis.edm_risk.temporal import CausalTemporalEvaluator, summarize_temporal


def _result(query: str, timestamp: float, success: bool) -> EDMQueryResult:
    return EDMQueryResult(
        query_id=query,
        session_id="S1",
        timestamp=timestamp,
        success=success,
        registration_success=success,
    )


def test_temporal_evaluation_never_uses_future_frames() -> None:
    evaluator = CausalTemporalEvaluator(window_size=3)
    first = evaluator.update(_result("q0", 0.0, False))
    second = evaluator.update(_result("q1", 1.0, False))
    evaluator.update(_result("q2", 2.0, True))

    assert first.history_query_ids == ("q0",)
    assert first.window_success is False
    assert second.history_query_ids == ("q0", "q1")
    assert second.consecutive_failure_length == 2


def test_temporal_summary_reports_failure_bursts_and_recovery() -> None:
    results = [
        _result("q0", 0.0, True),
        _result("q1", 1.0, False),
        _result("q2", 2.0, False),
        _result("q3", 3.0, True),
        _result("q4", 4.0, False),
    ]
    summary = summarize_temporal(results, window_size=3, consecutive_lengths=(2, 3))

    assert summary.failure_burst_count == 2
    assert summary.max_consecutive_failures == 2
    assert summary.recovery_time_median == 2.0
    assert summary.consecutive_failure_probability[2] > 0.0
