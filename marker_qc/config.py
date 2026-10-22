from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict, dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from marker_qc.models import Calibration


Roi = tuple[int, int, int, int]


def camera_mode_key(camera_index: int, backend: str, resolution: tuple[int, int]) -> str:
    return f"{int(camera_index)}|{backend}|{int(resolution[0])}x{int(resolution[1])}"


def full_frame_roi(resolution: tuple[int, int]) -> Roi:
    return 0, 0, int(resolution[0]), int(resolution[1])


def validate_roi(roi: Roi, resolution: tuple[int, int]) -> Roi:
    x, y, width, height = map(int, roi)
    frame_width, frame_height = map(int, resolution)
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > frame_width or y + height > frame_height:
        raise ValueError("Область поиска выходит за границы кадра.")
    return x, y, width, height


@dataclass(frozen=True)
class CameraProfile:
    index: int
    backend: str
    resolution: tuple[int, int]
    name: str = "Камера"


@dataclass(frozen=True)
class Norms:
    length_nominal_mm: str = "7.00"
    length_tolerance_mm: str = "0.10"
    width_nominal_mm: str = "1.15"
    width_tolerance_mm: str = "0.05"

    def _bounds(self, nominal: str, tolerance: str) -> tuple[Decimal, Decimal]:
        try:
            n, t = Decimal(nominal), Decimal(tolerance)
        except InvalidOperation as exc:
            raise ValueError("Нормативы должны быть корректными десятичными числами.") from exc
        if t < 0:
            raise ValueError("Допуск не может быть отрицательным.")
        return n - t, n + t

    @property
    def length_bounds(self) -> tuple[Decimal, Decimal]:
        return self._bounds(self.length_nominal_mm, self.length_tolerance_mm)

    @property
    def width_bounds(self) -> tuple[Decimal, Decimal]:
        return self._bounds(self.width_nominal_mm, self.width_tolerance_mm)

    def check_length(self, value_mm: float | str | Decimal) -> bool:
        value = Decimal(str(value_mm))
        low, high = self.length_bounds
        return low <= value <= high

    def check_width(self, value_mm: float | str | Decimal) -> bool:
        value = Decimal(str(value_mm))
        low, high = self.width_bounds
        return low <= value <= high


@dataclass
class ProcessingSettings:
    min_area_ratio: float = 0.003
    max_area_ratio: float = 0.45
    secondary_object_ratio: float = 0.25
    min_aspect_ratio: float = 2.5
    border_margin_px: int = 2


@dataclass
class AppSettings:
    norms: Norms = field(default_factory=Norms)
    processing: ProcessingSettings = field(default_factory=ProcessingSettings)
    calibration: Calibration | None = None
    search_rois: dict[str, Roi] = field(default_factory=dict)
    last_camera: CameraProfile | None = None
    last_operator: str = ""
    last_batch: str = ""

    def roi_for(self, camera_index: int, backend: str, resolution: tuple[int, int]) -> Roi:
        key = camera_mode_key(camera_index, backend, resolution)
        roi = self.search_rois.get(key)
        if roi is None:
            return full_frame_roi(resolution)
        try:
            return validate_roi(roi, resolution)
        except ValueError:
            return full_frame_roi(resolution)

    def set_roi(self, camera_index: int, backend: str, resolution: tuple[int, int], roi: Roi) -> None:
        self.search_rois[camera_mode_key(camera_index, backend, resolution)] = validate_roi(roi, resolution)


class SettingsStore:
    def __init__(self, path: Path):
        self.path = Path(path)

    def load(self) -> AppSettings:
        if not self.path.exists():
            return AppSettings()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
            calibration = raw.get("calibration")
            if calibration:
                calibration["resolution"] = tuple(calibration["resolution"])
                calibration["roi"] = tuple(calibration["roi"])
                calibration["calibration_points"] = tuple(tuple(p) for p in calibration["calibration_points"])
            search_rois = {
                str(key): tuple(map(int, value))
                for key, value in raw.get("search_rois", {}).items()
            }
            last_camera = raw.get("last_camera")
            if last_camera:
                last_camera["resolution"] = tuple(last_camera["resolution"])
            calibration_model = Calibration(**calibration) if calibration else None
            # Однократная совместимая миграция: поле ROI старой калибровки
            # копируется в новое независимое хранилище, но далее не является
            # источником области поиска.
            if calibration_model and not search_rois:
                key = camera_mode_key(
                    calibration_model.camera_index,
                    calibration_model.camera_backend,
                    calibration_model.resolution,
                )
                try:
                    search_rois[key] = validate_roi(calibration_model.roi, calibration_model.resolution)
                except ValueError:
                    pass
            return AppSettings(
                norms=Norms(**raw.get("norms", {})),
                processing=ProcessingSettings(**raw.get("processing", {})),
                calibration=calibration_model,
                search_rois=search_rois,
                last_camera=CameraProfile(**last_camera) if last_camera else None,
                last_operator=str(raw.get("last_operator", "")),
                last_batch=str(raw.get("last_batch", "")),
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            broken = self.path.with_suffix(self.path.suffix + ".broken")
            try:
                os.replace(self.path, broken)
            except OSError:
                pass
            raise RuntimeError(
                f"Файл настроек повреждён. Он сохранён как {broken.name}. "
                "Приложение запустится со стандартными настройками."
            ) from exc

    def save(self, settings: AppSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(settings)
        handle, temp_name = tempfile.mkstemp(prefix="settings_", suffix=".json", dir=self.path.parent)
        try:
            with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.path)
        except Exception:
            try:
                os.unlink(temp_name)
            except OSError:
                pass
            raise

    def update(self, settings: AppSettings, **changes: Any) -> AppSettings:
        for key, value in changes.items():
            setattr(settings, key, value)
        self.save(settings)
        return settings

