import pytest

from marker_qc.models import Decision, InspectionStage
from marker_qc.qc import WorkflowError, evaluate_decision, next_stage


def test_first_and_second_stage_decisions() -> None:
    assert evaluate_decision(InspectionStage.FIRST, True, True, True) is Decision.APPROVED_STAGE1
    assert evaluate_decision(InspectionStage.SECOND, True, True, True) is Decision.APPROVED_STAGE2
    assert evaluate_decision(InspectionStage.FIRST, False, True, True) is Decision.REJECTED
    assert evaluate_decision(InspectionStage.FIRST, True, False, True) is Decision.REJECTED
    assert evaluate_decision(InspectionStage.FIRST, False, False, False, questionable_requested=True) is Decision.QUESTIONABLE


def test_next_stage_and_finished_state() -> None:
    assert next_stage([]) is InspectionStage.FIRST
    assert next_stage([Decision.APPROVED_STAGE1]) is InspectionStage.SECOND
    assert next_stage([Decision.QUESTIONABLE]) is InspectionStage.SECOND
    with pytest.raises(WorkflowError):
        next_stage([Decision.REJECTED])
    with pytest.raises(WorkflowError):
        next_stage([Decision.APPROVED_STAGE1, Decision.APPROVED_STAGE2])


def test_questionable_is_not_allowed_on_second_stage() -> None:
    with pytest.raises(WorkflowError):
        evaluate_decision(InspectionStage.SECOND, True, True, True, questionable_requested=True)

