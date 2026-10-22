from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from marker_qc.config import ProcessingSettings
from marker_qc.models import MeasurementPoints


class ProcessingError(RuntimeError):
    """Ошибка распознавания, которая требует исправить кадр или ручной проверки."""


@dataclass(frozen=True)
class ProcessingResult:
    roi: np.ndarray
    gray: np.ndarray
    mask: np.ndarray
    contour_overlay: np.ndarray
    points: MeasurementPoints
    confidence: float
    diagnostics: tuple[str, ...]
    roi_rect: tuple[int, int, int, int] = (0, 0, 0, 0)


def crop_roi(frame: np.ndarray, roi: tuple[int, int, int, int]) -> np.ndarray:
    if frame is None or frame.size == 0:
        raise ProcessingError("Получен пустой кадр.")
    x, y, width, height = map(int, roi)
    frame_h, frame_w = frame.shape[:2]
    if width <= 0 or height <= 0 or x < 0 or y < 0 or x + width > frame_w or y + height > frame_h:
        raise ProcessingError("Рабочая область выходит за границы текущего кадра.")
    return frame[y : y + height, x : x + width].copy()


def _is_frame_component(x: int, y: int, width: int, height: int, area: int, shape: tuple[int, int]) -> bool:
    h, w = shape
    touches = x == 0 or y == 0 or x + width >= w or y + height >= h
    full_frame = x == 0 and y == 0 and width >= 0.90 * w and height >= 0.90 * h
    # На реальной оснастке по краям ROI видна тонкая синяя/тёмная рамка.
    # Она может распасться на несколько узких компонентов, но не должна
    # считаться маркером. Крупный объект, дошедший до края, не игнорируется.
    thin_edge_artifact = (
        touches
        and area / float(h * w) < 0.02
        and (width < 0.20 * w or height < 0.20 * h)
    )
    return full_frame or thin_edge_artifact


def _clean_candidate(binary: np.ndarray, settings: ProcessingSettings) -> tuple[np.ndarray, tuple[str, ...]]:
    src = (binary > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(src, connectivity=8)
    h, w = src.shape
    image_area = float(h * w)
    candidates: list[tuple[int, int, float]] = []
    touches_border: list[int] = []
    for label in range(1, count):
        x, y, ww, hh, area = map(int, stats[label])
        ratio = area / image_area
        if _is_frame_component(x, y, ww, hh, area, (h, w)):
            continue
        component = np.zeros_like(src)
        component[labels == label] = 255
        contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        rect_sides = sorted(cv2.minAreaRect(max(contours, key=cv2.contourArea))[1])
        aspect = rect_sides[1] / max(rect_sides[0], 1.0)
        # Фон и детали оснастки могут быть крупными, но кандидатом считается
        # только вытянутый объект, похожий на маркер.
        marker_like = aspect >= settings.min_aspect_ratio
        if x <= settings.border_margin_px or y <= settings.border_margin_px or x + ww >= w - settings.border_margin_px or y + hh >= h - settings.border_margin_px:
            if ratio >= settings.min_area_ratio and marker_like:
                touches_border.append(label)
            continue
        if settings.min_area_ratio <= ratio <= settings.max_area_ratio and marker_like:
            candidates.append((label, area, aspect))
    if touches_border:
        raise ProcessingError("Объект касается границы рабочей области. Переместите маркер внутрь ROI.")
    if not candidates:
        raise ProcessingError("Маркер не найден или его площадь слишком мала.")
    candidates.sort(key=lambda item: item[1] * min(item[2], 8.0), reverse=True)
    if len(candidates) > 1 and candidates[1][1] >= candidates[0][1] * settings.secondary_object_ratio:
        raise ProcessingError("В полном кадре обнаружено несколько подходящих объектов.")
    label, area, aspect = candidates[0]
    ratio = area / image_area
    if ratio > settings.max_area_ratio:
        raise ProcessingError("Выделенный объект занимает слишком большую часть рабочей области.")
    mask = np.zeros_like(src, dtype=np.uint8)
    mask[labels == label] = 255
    diagnostics = (f"Площадь объекта: {area} px ({ratio:.2%} области поиска)", f"Отношение сторон кандидата: {aspect:.2f}")
    return mask, diagnostics


def _segment(gray: np.ndarray, settings: ProcessingSettings) -> tuple[np.ndarray, tuple[str, ...]]:
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 41, 7
    )
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    errors: list[str] = []
    scored: list[tuple[float, np.ndarray, tuple[str, ...], str]] = []
    for name, raw in (("Оцу", otsu), ("адаптивный", adaptive)):
        candidate = cv2.morphologyEx(raw, cv2.MORPH_CLOSE, kernel_close, iterations=2)
        candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel_open, iterations=1)
        try:
            cleaned, diagnostics = _clean_candidate(candidate, settings)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contour = max(contours, key=cv2.contourArea)
            area = max(cv2.contourArea(contour), 1.0)
            perimeter = cv2.arcLength(contour, True)
            compactness = min(1.0, 4.0 * np.pi * area / max(perimeter * perimeter, 1.0))
            rect = cv2.minAreaRect(contour)
            sides = sorted(rect[1])
            aspect = sides[1] / max(sides[0], 1.0)
            if aspect < settings.min_aspect_ratio:
                raise ProcessingError("Форма выделенного объекта не похожа на вытянутый маркер.")
            score = min(1.0, aspect / 8.0) * 0.7 + min(1.0, compactness / 0.35) * 0.3
            scored.append((score, cleaned, diagnostics + (f"Сегментация: {name}; отношение сторон: {aspect:.2f}",), name))
        except ProcessingError as exc:
            # Конкурирующий крупный объект нельзя «скрыть» выбором другой
            # пороговой маски: это небезопасно для решения ОТК.
            if "несколько" in str(exc):
                raise
            errors.append(f"{name}: {exc}")
    if not scored:
        raise ProcessingError("Сегментация не подтверждена. " + " ".join(errors))
    scored.sort(key=lambda item: item[0], reverse=True)
    _, mask, diagnostics, _ = scored[0]
    return mask, diagnostics


def _ordered_axis(contour_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    center = contour_xy.mean(axis=0)
    centered = contour_xy - center
    covariance = np.cov(centered.T)
    values, vectors = np.linalg.eigh(covariance)
    major = vectors[:, int(np.argmax(values))]
    if major[0] < 0:
        major = -major
    normal = np.array([-major[1], major[0]], dtype=float)
    # В координатах изображения положительное направление нормали должно смотреть вниз.
    if normal[1] < 0:
        normal = -normal
    return center, major, normal, centered


def _width_points_original_semantics(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Воспроизвести смысл X2/Y2, X4/Y4 и C исходного MATLAB без опасных индексов."""
    foreground = (mask > 0).astype(np.uint8)
    h, w = foreground.shape
    tx = np.count_nonzero(foreground, axis=0).astype(float)
    ty = np.count_nonzero(foreground, axis=1).astype(float)
    tx_max, ty_max = float(tx.max()), float(ty.max())
    if tx_max <= 0 or ty_max <= 0:
        raise ProcessingError("Пустая маска при поиске точек ширины.")

    x2: int | None = None
    for x in range(min(int(round(0.52 * w)), w - 1), 5, -1):
        if tx[x] > tx_max - 2 and 6 * tx[x] - float(tx[x - 6 : x].sum()) > 4:
            x2 = x
    if x2 is None:
        candidates = np.flatnonzero((tx > tx_max - 2) & (np.arange(w) <= 0.60 * w))
        if candidates.size == 0:
            raise ProcessingError("Не удалось определить W1 по проекции контура.")
        x2 = int(candidates.min())
    y_values = np.flatnonzero(foreground[:, x2])
    if y_values.size == 0:
        raise ProcessingError("Столбец W1 не пересекает маркер.")
    w1 = np.array([x2, int(y_values.min())], dtype=float)

    y4: int | None = None
    for y in range(0, max(0, h - 6)):
        if ty[y] > ty_max - 2 and 6 * ty[y] - float(ty[y + 1 : y + 7].sum()) > 4:
            y4 = y
    if y4 is None:
        candidates = np.flatnonzero(ty > ty_max - 2)
        if candidates.size == 0:
            raise ProcessingError("Не удалось определить W2 по проекции контура.")
        y4 = int(candidates[-1])
    x_values = np.flatnonzero(foreground[y4])
    if x_values.size == 0:
        raise ProcessingError("Строка W2 не пересекает маркер.")
    w2 = np.array([int(x_values.max()), y4], dtype=float)

    left_edges = np.zeros_like(foreground, dtype=bool)
    left_edges[:, 1:] = (foreground[:, 1:] > 0) & (foreground[:, :-1] == 0)
    yy, xx = np.nonzero(left_edges)
    if xx.size == 0:
        raise ProcessingError("Не удалось найти внутренний край для WC.")
    edge_points = np.column_stack((xx, yy)).astype(float)
    balance = np.abs(np.linalg.norm(edge_points - w1, axis=1) - np.linalg.norm(edge_points - w2, axis=1))
    wc = edge_points[int(np.argmin(balance))]
    return w1, w2, wc


def find_measurement_points(mask: np.ndarray) -> tuple[MeasurementPoints, float, tuple[str, ...]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ProcessingError("Не найден контур маркера.")
    contour = max(contours, key=cv2.contourArea)
    points = contour[:, 0, :].astype(float)
    if len(points) < 20:
        raise ProcessingError("Контур маркера содержит недостаточно точек.")
    center, major, normal, centered = _ordered_axis(points)
    along = centered @ major
    across = centered @ normal
    span = float(np.ptp(along))
    thickness = float(np.ptp(across))
    if span <= 0 or thickness <= 0 or span / thickness < 2.5:
        raise ProcessingError("Геометрия контура вырожденна или объект недостаточно вытянут.")

    # L1/L2 — наиболее удалённая пара точек контура: тот же смысл прямой хорды,
    # что у X1/Y1 и X3/Y3 в исходном MATLAB-коде.
    hull = cv2.convexHull(contour)[:, 0, :].astype(float)
    distances = np.sum((hull[:, None, :] - hull[None, :, :]) ** 2, axis=2)
    i, j = np.unravel_index(int(np.argmax(distances)), distances.shape)
    l1, l2 = hull[i], hull[j]
    if float((l1 - center) @ major) > float((l2 - center) @ major):
        l1, l2 = l2, l1

    band = max(3.0, span * 0.07)
    left_indices = np.flatnonzero(along <= along.min() + band)
    right_indices = np.flatnonzero(along >= along.max() - band)
    if left_indices.size == 0 or right_indices.size == 0:
        raise ProcessingError("Не удалось выделить торцы маркера.")
    # Знак собственных векторов PCA неоднозначен. Рассматриваем обе стороны
    # контура и выбираем хорду, дающую физическую высоту дуги. Это сохраняет
    # W1/W2 на одной стороне маркера и делает результат одинаковым при 0/180°.
    width_candidates: list[tuple[float, np.ndarray, np.ndarray, np.ndarray]] = []
    for selector in (np.argmin, np.argmax):
        w1 = points[left_indices[int(selector(across[left_indices]))]]
        w2 = points[right_indices[int(selector(across[right_indices]))]]
        chord = w2 - w1
        chord_squared = float(chord @ chord)
        if chord_squared <= 0:
            continue
        relative = points - w1
        chord_length = chord_squared ** 0.5
        signed = (chord[0] * relative[:, 1] - chord[1] * relative[:, 0]) / chord_length
        position = (relative @ chord) / chord_squared
        middle = np.flatnonzero((position >= 0.30) & (position <= 0.70))
        if middle.size < 3:
            continue
        wc_index = middle[int(np.argmax(np.abs(signed[middle])))]
        wc = points[wc_index]
        width_candidates.append((abs(float(signed[wc_index])), w1, w2, wc))
    if not width_candidates:
        raise ProcessingError("Не удалось определить точки ширины.")
    width_px, w1, w2, wc = max(width_candidates, key=lambda item: item[0])
    chord = w2 - w1
    chord_length = float(np.linalg.norm(chord))

    warnings: list[str] = []
    if width_px < 0.03 * span:
        raise ProcessingError("Точка ширины WC определена ненадёжно.")
    confidence = min(1.0, 0.55 + min(span / thickness, 10.0) * 0.025 + min(width_px / thickness, 3.0) * 0.05)
    result = MeasurementPoints(
        l1=tuple(map(float, l1)), l2=tuple(map(float, l2)),
        w1=tuple(map(float, w1)), w2=tuple(map(float, w2)), wc=tuple(map(float, wc)),
    )
    diagnostics = (
        f"Длина контура по главной оси: {span:.1f} px",
        f"Расстояние WC до W1–W2: {width_px:.1f} px",
        *warnings,
    )
    return result, confidence, diagnostics


def draw_contour_overlay(roi: np.ndarray, mask: np.ndarray, points: MeasurementPoints) -> np.ndarray:
    overlay = roi.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(overlay, contours, -1, (0, 180, 0), 2)
    l1, l2, w1, w2, wc = (tuple(map(int, p)) for p in (points.l1, points.l2, points.w1, points.w2, points.wc))
    cv2.line(overlay, l1, l2, (0, 0, 255), 2)
    cv2.line(overlay, w1, w2, (255, 0, 0), 2)
    cv2.line(overlay, wc, _project_to_line(wc, w1, w2), (255, 0, 0), 2)
    for point in (l1, l2):
        _draw_hollow_point(overlay, point, (0, 0, 255))
    for point in (w1, w2, wc):
        _draw_hollow_point(overlay, point, (255, 0, 0))
    return overlay


def _draw_hollow_point(image: np.ndarray, point: tuple[int, int], color: tuple[int, int, int]) -> None:
    cv2.circle(image, point, 6, color, 2, cv2.LINE_AA)
    cv2.drawMarker(image, point, color, cv2.MARKER_CROSS, 5, 1, cv2.LINE_AA)


def _project_to_line(point: tuple[int, int], a: tuple[int, int], b: tuple[int, int]) -> tuple[int, int]:
    p, aa, bb = np.array(point, float), np.array(a, float), np.array(b, float)
    vector = bb - aa
    denominator = float(vector @ vector)
    if denominator == 0:
        return a
    projection = aa + vector * float((p - aa) @ vector) / denominator
    return int(round(projection[0])), int(round(projection[1]))


def process_frame(
    frame: np.ndarray,
    roi_rect: tuple[int, int, int, int] | None,
    settings: ProcessingSettings,
) -> ProcessingResult:
    if frame is None or frame.size == 0:
        raise ProcessingError("Получен пустой кадр.")
    frame_height, frame_width = frame.shape[:2]
    roi_rect = roi_rect or (0, 0, frame_width, frame_height)
    x, y, width, height = map(int, roi_rect)
    roi = crop_roi(frame, (x, y, width, height))
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    mask, segmentation_diagnostics = _segment(gray, settings)
    local_points, confidence, point_diagnostics = find_measurement_points(mask)
    points = MeasurementPoints(**{
        name: (point[0] + x, point[1] + y)
        for name, point in {
            "l1": local_points.l1, "l2": local_points.l2,
            "w1": local_points.w1, "w2": local_points.w2, "wc": local_points.wc,
        }.items()
    })
    overlay = frame.copy()
    overlay[y:y + height, x:x + width] = draw_contour_overlay(roi, mask, local_points)
    return ProcessingResult(
        roi=roi,
        gray=gray,
        mask=mask,
        contour_overlay=overlay,
        points=points,
        confidence=confidence,
        diagnostics=segmentation_diagnostics + point_diagnostics,
        roi_rect=(x, y, width, height),
    )

