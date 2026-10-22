from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any


Point = tuple[float, float]


class InspectionStage(StrEnum):
    FIRST = "FIRST"
    SECOND = "SECOND"


class Decision(StrEnum):
    APPROVED_STAGE1 = "APPROVED_STAGE1"
    APPROVED_STAGE2 = "APPROVED_STAGE2"
    REJECTED = "REJECTED"
    QUESTIONABLE = "QUESTIONABLE"

    @property
    def russian(self) -> str:
        return {
            self.APPROVED_STAGE1: "Одобрено ОТК 1",
            self.APPROVED_STAGE2: "Одобрено ОТК 2",
            self.REJECTED: "Брак",
            self.QUESTIONABLE: "Под вопросом",
        }[self]


@dataclass(frozen=True)
class MeasurementPoints:
    l1: Point
    l2: Point
    w1: Point
    w2: Point
    wc: Point

    def as_json(self) -> dict[str, list[float]]:
        return {name: [float(p[0]), float(p[1])] for name, p in asdict(self).items()}

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "MeasurementPoints":
        return cls(**{key: (float(value[0]), float(value[1])) for key, value in data.items()})


@dataclass(frozen=True)
class MeasurementResult:
    length_mm: float
    width_mm: float
    length_ok: bool
    width_ok: bool
    dimensions_ok: bool
    points: MeasurementPoints
    scale_mm_per_px: float
    automatic: bool
    confidence: float
    diagnostics: tuple[str, ...] = ()


@dataclass(frozen=True)
class Calibration:
    camera_index: int
    camera_backend: str
    camera_name: str
    resolution: tuple[int, int]
    roi: tuple[int, int, int, int]
    scale_mm_per_px: float
    calibrated_at: str
    reference_distance_mm: float
    calibration_points: tuple[Point, Point]

    @classmethod
    def create(
        cls,
        camera_index: int,
        camera_backend: str,
        camera_name: str,
        resolution: tuple[int, int],
        roi: tuple[int, int, int, int],
        scale_mm_per_px: float,
        reference_distance_mm: float,
        calibration_points: tuple[Point, Point],
    ) -> "Calibration":
        return cls(
            camera_index=camera_index,
            camera_backend=camera_backend,
            camera_name=camera_name,
            resolution=resolution,
            roi=roi,
            scale_mm_per_px=scale_mm_per_px,
            calibrated_at=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            reference_distance_mm=reference_distance_mm,
            calibration_points=calibration_points,
        )

    def matches(self, camera_index: int, backend: str, resolution: tuple[int, int]) -> bool:
        return (
            self.camera_index == camera_index
            and self.camera_backend == backend
            and self.resolution == resolution
            and self.scale_mm_per_px > 0
        )


@dataclass(frozen=True)
class InspectionDraft:
    batch_code: str
    marker_code: str
    operator_name: str
    stage: InspectionStage
    result: MeasurementResult
    ends_ok: bool
    shape_ok: bool
    decision: Decision
    note: str = ""

