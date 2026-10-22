from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AppPaths:
    root: Path
    settings: Path
    database: Path
    images: Path
    logs: Path

    @classmethod
    def resolve(cls, root: Path | None = None) -> "AppPaths":
        if root is None:
            override = os.environ.get("MARKER_QC_DATA_DIR")
            if override:
                root = Path(override)
            else:
                local = os.environ.get("LOCALAPPDATA")
                root = Path(local) / "MarkerQC" if local else Path.home() / ".marker_qc"
        root = Path(root).expanduser().resolve()
        paths = cls(
            root=root,
            settings=root / "settings.json",
            database=root / "marker_qc.sqlite3",
            images=root / "images",
            logs=root / "logs",
        )
        paths.images.mkdir(parents=True, exist_ok=True)
        paths.logs.mkdir(parents=True, exist_ok=True)
        return paths

