# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_dynamic_libs

opencv_binaries = collect_dynamic_libs("cv2")

a = Analysis(
    ["run_marker_qc.py"],
    pathex=[],
    binaries=opencv_binaries,
    datas=[],
    hiddenimports=["numpy", "cv2", "sqlite3", "openpyxl"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["tkinter", "matplotlib", "IPython", "notebook"],
    noarchive=False,
    optimize=1,
)

foreign_icu = {"icuuc.dll", "icudt78.dll"}
a.binaries = [entry for entry in a.binaries if entry[0].lower().replace("\\", "/").split("/")[-1] not in foreign_icu]

pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="MarkerQC-onefile",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

