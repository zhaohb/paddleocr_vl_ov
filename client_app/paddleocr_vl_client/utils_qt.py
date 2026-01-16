from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Optional

from PySide6.QtGui import QImage


def pil_image_to_qimage(pil_image) -> QImage:
    """
    兼容 PIL.Image → QImage 的转换。
    """
    bio = BytesIO()
    pil_image.save(bio, format="PNG")
    data = bio.getvalue()
    img = QImage.fromData(data)
    return img


def pick_first_existing(*paths: Optional[str]) -> Optional[str]:
    for p in paths:
        if not p:
            continue
        if Path(p).exists():
            return p
    return None


