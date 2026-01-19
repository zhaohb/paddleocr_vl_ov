# -*- mode: python ; coding: utf-8 -*-
#
# PyInstaller spec for PaddleOCR-VL OpenVINO Desktop APP
#
# Notes:
# - For a single-file executable (onefile), remove COLLECT and let EXE bundle everything.
# - We intentionally collect openvino submodules to avoid runtime import issues.
# - IMPORTANT (size): avoid collecting full `transformers` submodules and `include_py_files=True` datas,
#   otherwise the exe size will explode.

from pathlib import Path
from PyInstaller.utils.hooks import collect_submodules, collect_data_files

block_cipher = None

hiddenimports = []
hiddenimports += collect_submodules("openvino")

#
# transformers / torch / torchvision are huge.
# Let Analysis include what is actually imported by our code, and only add a few
# known dynamic-import submodules here (keep this list small for exe size).
#
hiddenimports += collect_submodules("transformers.models.ernie4_5")
# transformers 会在 AutoTokenizer/AutoConfig 内部通过 importlib 动态导入这些模块；
# 不显式加入会导致打包后运行时报 ModuleNotFoundError。
hiddenimports += collect_submodules("transformers.models.ernie4_5_moe")

datas = []
#
# Only collect non-.py package data. Python modules will be handled by PYZ/Analysis.
#
datas += collect_data_files("openvino", include_py_files=False)

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
    excludes=[
        # Big ML stacks that are not required for this desktop app runtime.
        # (Reduce size + avoid problematic runtime hooks)
        "tensorflow",
        "tensorflow_cpu",
        "jax",
        "jaxlib",
        "flax",
        "datasets",
        "pyarrow",
        "bitsandbytes",
        "numba",
        "llvmlite",
        "scipy",
        "sklearn",
        "librosa",
        "nltk",
        "nltk_data",
        "ctranslate2",
        "googleapiclient",
        "paddle",
        "paddlex",
        # We don't use these in the GUI runtime; exclude to shrink.
        "matplotlib",
        "seaborn",
        "pandas",
        "IPython",
        "pytest",
        "nbformat",
        "openpyxl",
        "sqlalchemy",
        "zmq",
        "jedi",
        "parso",
        "black",
        "gradio",
        "fastapi",
        "uvicorn",

        # Model download ecosystem (very large).
        # Keep `modelscope` (for snapshot_download / auto-download), but exclude other heavy toolchains.
        "diffusers",
        "peft",
        "onnxruntime",
        "timm",
        "boto3",
        "botocore",
        "skimage",
        "shapely",
        "emoji",
        "soundfile",
        "imageio",
        "imageio_ffmpeg",
        "grpc",
        "opentelemetry",

        # Not used by our GUI (PySide6). Exclude to avoid bundling Tcl/Tk runtime.
        "tkinter",
        "_tkinter",
        "gi",

        # NNCF is optional; if it's not used in your updated runtime path, exclude to shrink.
        "nncf",
    ],
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


