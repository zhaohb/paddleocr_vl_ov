# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for PaddleOCR-VL OpenVINO Desktop APP
#
# Notes:
# - Prefer onedir: large runtime deps (OpenVINO + Qt)
# - We intentionally collect openvino submodules to avoid runtime import issues.

from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

hiddenimports = []
hiddenimports += collect_submodules("openvino")

hiddenimports += collect_submodules("transformers")

datas = []
datas += collect_data_files("openvino", include_py_files=True)
datas += collect_data_files("transformers", include_py_files=True)

SPECDIR = Path(globals().get("SPECPATH", ".")).resolve()
ROOT = SPECDIR.parent  # .../paddleocr_vl_ov
ENTRY = str(SPECDIR / "main.py")

# 打包本地 markdown 预览 assets（离线渲染依赖）
ASSETS_DIR = SPECDIR / "assets"
if ASSETS_DIR.exists():
    datas += [(str(ASSETS_DIR), "assets")]

# 可选：如果你提供了 `client_app/assets/app.ico`，则用于 Windows exe 图标
ICON_PATH = ASSETS_DIR / "app.ico"
ICON_FILE = str(ICON_PATH) if ICON_PATH.exists() else None

a = Analysis(
    [ENTRY],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="PaddleOCRVL-APP",
    icon=ICON_FILE,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="PaddleOCRVL-APP",
)


