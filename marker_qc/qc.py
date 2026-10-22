from __future__ import annotations

from marker_qc.models import Decision, InspectionStage


class WorkflowError(RuntimeError):
    pass


def evaluate_decision(
    stage: InspectionStage,
    dimensions_ok: bool,
    ends_ok: bool,
    shape_ok: bool,
    *,
    questionable_requested: bool = False,
) -> Decision:
    if questionable_requested:
        if stage is not InspectionStage.FIRST:
            raise WorkflowError("Статус «Под вопросом» допустим только на первом этапе ОТК.")
        return Decision.QUESTIONABLE
    passed = dimensions_ok and ends_ok and shape_ok
    if not passed:
        return Decision.REJECTED
    return Decision.APPROVED_STAGE1 if stage is InspectionStage.FIRST else Decision.APPROVED_STAGE2


def next_stage(previous_decisions: list[Decision]) -> InspectionStage:
    if not previous_decisions:
        return InspectionStage.FIRST
    if len(previous_decisions) > 1:
        raise WorkflowError("Оба этапа для маркера уже завершены.")
    first = previous_decisions[0]
    if first in (Decision.APPROVED_STAGE1, Decision.QUESTIONABLE):
        return InspectionStage.SECOND
    raise WorkflowError("Маркер уже получил окончательный статус «Брак».")

