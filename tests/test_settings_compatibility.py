import json

from marker_qc.config import SettingsStore


def test_legacy_calibration_with_roi_remains_valid(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({
        "calibration": {
            "camera_index": 1,
            "camera_backend": "DirectShow",
            "camera_name": "Legacy",
            "resolution": [640, 480],
            "roi": [100, 50, 400, 300],
            "scale_mm_per_px": 0.0175,
            "calibrated_at": "2026-01-01T00:00:00+03:00",
            "reference_distance_mm": 10.0,
            "calibration_points": [[10, 10], [581, 10]],
        },
        "last_operator": "Старый оператор",
        "last_batch": "Старая партия",
    }, ensure_ascii=False), encoding="utf-8")
    settings = SettingsStore(path).load()
    assert settings.calibration is not None
    assert settings.calibration.roi == (100, 50, 400, 300)
    assert settings.calibration.matches(1, "DirectShow", (640, 480))
    assert settings.roi_for(1, "DirectShow", (640, 480)) == (100, 50, 400, 300)


def test_roi_is_saved_separately_without_changing_calibration(tmp_path) -> None:
    path = tmp_path / "settings.json"
    store = SettingsStore(path)
    settings = store.load()
    settings.set_roi(2, "DirectShow", (1920, 1080), (100, 50, 1600, 800))
    calibration_before = settings.calibration
    store.save(settings)
    loaded = store.load()
    assert loaded.roi_for(2, "DirectShow", (1920, 1080)) == (100, 50, 1600, 800)
    assert loaded.roi_for(2, "DirectShow", (640, 480)) == (0, 0, 640, 480)
    assert loaded.calibration == calibration_before
