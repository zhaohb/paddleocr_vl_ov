from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal, Dict, Any, List


Device = Literal["CPU", "GPU", "NPU", "AUTO"]
VlmDevice = Literal["CPU", "GPU", "AUTO"]
LayoutPrecision = Literal["fp16", "fp32", "combined_fp16", "combined_fp32"]
TaskType = Literal["ocr", "table", "chart", "formula"]


@dataclass(frozen=True)
class PipelineInitConfig:
    layout_model_path: Optional[str]
    vlm_model_path: Optional[str]
    cache_dir: Optional[str]
    vlm_device: VlmDevice
    layout_device: Device
    layout_precision: LayoutPrecision
    llm_int4_compress: bool
    vision_int8_quant: bool
    llm_int8_compress: bool
    llm_int8_quant: bool


@dataclass(frozen=True)
class PredictConfig:
    use_layout_detection: bool
    layout_threshold: float
    max_new_tokens: int


@dataclass
class TaskItem:
    input_path: Path
    status: str = "pending"  # pending/running/done/error
    # 不开启 layout detection 时，使用该类型作为整图的 prompt_label（ocr/table/chart/formula）
    task_type: TaskType = "ocr"
    # 本次运行是否启用 layout detection（由 UI 在运行开始时写入，用于历史记录展示）
    used_layout_detection: Optional[bool] = None
    output_dir: Optional[Path] = None
    error: Optional[str] = None
    summary: Optional[str] = None
    markdown_text: Optional[str] = None
    json_text: Optional[str] = None
    # 用于 UI 展示的可视化图片（以临时文件路径形式保存）
    vis_image_path: Optional[Path] = None
    # 预览输入：图片任务直接指向自身；PDF 任务指向渲染后的某一页 png
    preview_input_image_path: Optional[Path] = None
    # PDF 渲染页目录（用于多页预览）
    pdf_pages_dir: Optional[Path] = None


@dataclass(frozen=True)
class HistoryTask:
    """
    历史任务记录（仅记录成功任务）。
    用于 UI 展示与落盘持久化。
    """

    finished_at: str  # ISO-8601 string
    input_path: str
    file_type: str
    # 仅当关闭 layout detection 时记录/展示 task_type；否则为 None
    task_type: Optional[TaskType]
    output_dir: str
    summary: str = ""


def sanitize_stem(p: Path) -> str:
    # 适配 Windows 文件名
    invalid = '<>:"/\\|?*'
    stem = p.stem
    for ch in invalid:
        stem = stem.replace(ch, "_")
    return stem.strip() or "output"


def summarize_result_dict(res: Dict[str, Any]) -> str:
    parsing_cnt = len(res.get("parsing_res_list", []) or [])
    table_cnt = len(res.get("table_res_list", []) or [])
    w = res.get("width", "N/A")
    h = res.get("height", "N/A")
    page_index = res.get("page_index", "N/A")
    page_count = res.get("page_count", "N/A")
    return f"page {page_index}/{page_count}, size={w}x{h}, blocks={parsing_cnt}, tables={table_cnt}"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


SUPPORTED_IMAGE_EXTS: List[str] = [".png", ".jpg", ".jpeg", ".bmp", ".webp"]
SUPPORTED_DOC_EXTS: List[str] = [".pdf"]


