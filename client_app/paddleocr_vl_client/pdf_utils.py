from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .i18n import normalize_lang, t


@dataclass(frozen=True)
class RenderedPage:
    page_index: int
    image_path: Path


def is_pdf(path: Path) -> bool:
    return path.suffix.lower() == ".pdf"


def render_pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 300, lang: Optional[str] = None) -> List[RenderedPage]:
    """
    使用 PyMuPDF 渲染 PDF 每页为 PNG，返回渲染后的图片路径列表。
    - 为了减少依赖复杂度，避免 pdf2image/poppler，改用 PyMuPDF。
    """
    try:
        import fitz  # PyMuPDF
    except Exception as e:  # pragma: no cover
        raise RuntimeError(t("pdf.err.pymupdf_missing", normalize_lang(lang))) from e

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    pages: List[RenderedPage] = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_path = out_dir / f"page_{i+1:04d}.png"
        pix.save(str(img_path))
        pages.append(RenderedPage(page_index=i, image_path=img_path))
    doc.close()
    return pages


