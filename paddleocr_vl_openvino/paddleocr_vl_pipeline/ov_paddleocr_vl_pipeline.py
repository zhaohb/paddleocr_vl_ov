"""
OpenVINO 版本的 PaddleOCR-VL Pipeline 实现
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
from functools import partial
import random
from PIL import Image, ImageDraw, ImageFont
import openvino as ov
import os
import logging

# 尝试导入 modelscope，如果未安装则给出提示
try:
    from modelscope import snapshot_download
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False
    logging.warning("modelscope not installed. Auto-download feature will be disabled. Install with: pip install modelscope")

# 尝试导入 modelscope，如果未安装则给出提示
try:
    from modelscope import snapshot_download
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False
    logging.warning("modelscope not installed. Auto-download feature will be disabled. Install with: pip install modelscope")

# 导入布局检测相关函数
from ..pp_doclayoutv2.ov_pp_layoutv2_infer import (
    preprocess_image_doclayout,
    postprocess_detections_detr,
    postprocess_detections_paddle_nms,
    LayoutDetectionResult,
)
from ..pp_doclayoutv2.result import LayoutAnalysisResult

# 导入 VLM 模型
from ..paddleocr_vl.ov_paddleocr_vl import OVPaddleOCRVLForCausalLM

# 导入图像处理
from ..paddleocr_vl.image_processing_paddleocr_vl import PaddleOCRVLImageProcessor
from ..paddleocr_vl.uilts import (
    convert_otsl_to_html,
    crop_margin,
    filter_overlap_boxes,
    merge_blocks,
    post_process_for_spotting,
    tokenize_figure_of_table,
    truncate_repetitive_content,
    untokenize_figure_of_table,
)

# 图像标签定义（参考 PaddleX）
BLOCK_LABEL_MAP = {
    "image_labels": ["image", "figure"],
}

def gather_imgs(original_img: np.ndarray, layout_det_objs: List[Dict]) -> List[Dict]:
    """
    从布局检测结果中提取图像区域
    
    Args:
        original_img: 原始图像（BGR 格式）
        layout_det_objs: 布局检测结果列表，每个元素包含 label, coordinate, score 等字段
    
    Returns:
        List[Dict]: 提取的图像列表，每个元素包含 path, img, coordinate, score
    """
    imgs_in_doc = []
    for det_obj in layout_det_objs:
        if det_obj["label"] in BLOCK_LABEL_MAP["image_labels"]:
            label = det_obj["label"]
            x_min, y_min, x_max, y_max = list(map(int, det_obj["coordinate"]))
            img_path = f"imgs/img_in_{label}_box_{x_min}_{y_min}_{x_max}_{y_max}.jpg"
            # 从 BGR 图像中提取区域并转换为 RGB（PIL Image 需要 RGB）
            img = Image.fromarray(original_img[y_min:y_max, x_min:x_max, ::-1])
            imgs_in_doc.append(
                {
                    "path": img_path,
                    "img": img,
                    "coordinate": (x_min, y_min, x_max, y_max),
                    "score": det_obj["score"],
                }
            )
    return imgs_in_doc

# 可视化顺序标签（参考 PaddleX）
VISUALIZE_ORDE_LABELS = [
    "text",
    "formula",
    "inline_formula",
    "display_formula",
    "algorithm",
    "reference",
    "reference_content",
    "content",
    "abstract",
    "paragraph_title",
    "doc_title",
    "vertical_text",
    "ocr",
    "number",
    "footnote",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "aside_text",
]

# PaddleX doc_vl: JSON 的 order 生成需要跳过的 label 列表
SKIP_ORDER_LABELS = [
    "figure_title",
    "vision_footnote",
    "image",
    "chart",
    "table",
    "header",
    "header_image",
    "footer",
    "footer_image",
    "footnote",
    "aside_text",
]


def _sorted_layout_blocks_paddlex_style(blocks: List["PaddleOCRVLBlock"], image_width: int) -> List["PaddleOCRVLBlock"]:
    """
    对齐 PaddleX 1.5 `layout_parsing/utils.py::sorted_layout_boxes` 的阅读顺序排序：
    - 先按 (y1, x1) 粗排
    - 再按左右栏（left/right）分桶，提升双栏文本的阅读顺序一致性

    Args:
        blocks: PaddleOCRVLBlock 列表
        image_width: 页面宽度

    Returns:
        排序后的 blocks
    """
    if not blocks or len(blocks) <= 1:
        return blocks

    w = float(image_width) if image_width else 0.0
    if w <= 0:
        return sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))

    # Sort on y first then x (PaddleX: sorted(res, key=lambda x: (y, x)))
    _boxes = sorted(blocks, key=lambda b: (b.bbox[1], b.bbox[0]))

    new_res: List[PaddleOCRVLBlock] = []
    res_left: List[PaddleOCRVLBlock] = []
    res_right: List[PaddleOCRVLBlock] = []

    i = 0
    while True:
        if i >= len(_boxes):
            break

        bbox = _boxes[i].bbox
        x1, y1, x2, y2 = bbox

        # PaddleX 左栏判定
        if x1 < w / 4 and x2 < 3 * w / 5:
            res_left.append(_boxes[i])
            i += 1
        # PaddleX 右栏判定
        elif x1 > 2 * w / 5:
            res_right.append(_boxes[i])
            i += 1
        else:
            # 碰到中间跨栏块：先把左右栏按 y 合并，再加入该块
            new_res += res_left
            new_res += res_right
            new_res.append(_boxes[i])
            res_left = []
            res_right = []
            i += 1

    # Flush remaining
    res_left = sorted(res_left, key=lambda b: b.bbox[1])
    res_right = sorted(res_right, key=lambda b: b.bbox[1])
    if res_left:
        new_res += res_left
    if res_right:
        new_res += res_right

    return new_res

# 格式化函数（参考 PaddleX）
def format_title_func(block):
    """格式化标题"""
    import re
    title = block.content
    # 简单的标题格式化
    title = title.rstrip(".")
    level = title.count(".") + 1 if "." in title else 1
    return f"#{'#' * level} {title}".replace("-\n", "").replace("\n", " ")

def format_para_title_func(block):
    """
    段落标题格式化（对齐 PaddleX `format_para_title_func` 的用途）。
    这里保持实现简洁：统一用二级标题渲染。
    """
    content = getattr(block, "content", "")
    return f"## {content}".replace("-\n", "").replace("\n", " ")

def format_centered_by_html(string):
    """HTML 居中格式化"""
    return f'<div style="text-align: center;">{string}</div>'.replace("-\n", "").replace("\n", " ") + "\n"

def format_text_plain_func(block):
    """纯文本格式化"""
    return block.content

def format_image_scaled_by_html_func(block, original_image_width):
    """缩放图像 HTML 格式化"""
    if block.image:
        image_path = block.image["path"]
        image_width = block.image["img"].width
        scale = int(image_width / original_image_width * 100)
        return '<img src="{}" alt="Image" width="{}%" />'.format(
            image_path.replace("-\n", "").replace("\n", " "), scale
        )
    return ""

def format_image_plain_func(block):
    """纯图像格式化"""
    if block.image:
        image_path = block.image["path"]
        return "![]({})".format(image_path.replace("-\n", "").replace("\n", " "))
    return ""

def format_table_center_func(block):
    tabel_content = block.content

    tabel_content = tabel_content.replace(
        "<table>", "<table border=1 style='margin: auto; word-wrap: break-word;'>"
    )

    tabel_content = tabel_content.replace(
        "<th>", "<th style='text-align: center; word-wrap: break-word;'>"
    )
    tabel_content = tabel_content.replace(
        "<td>", "<td style='text-align: center; word-wrap: break-word;'>"
    )

    return tabel_content

def simplify_table_func(table_code):
    """简化表格函数"""
    return "\n" + table_code.replace("<html>", "").replace("</html>", "").replace(
        "<body>", ""
    ).replace("</body>", "")

def format_first_line_func(block, templates, format_func, spliter):
    """格式化首行"""
    from functools import partial
    lines = block.content.split(spliter)
    for idx in range(len(lines)):
        line = lines[idx]
        if line.strip() == "":
            continue
        if line.lower() in templates:
            lines[idx] = format_func(line)
        break
    return spliter.join(lines)

def merge_formula_and_number(formula, formula_number):
    formula = formula.replace("$$", "")
    merge_formula = r"{} \tag*{{{}}}".format(formula, formula_number)
    return f"$${merge_formula}$$"

def fix_latex_syntax(text):
    """
    修复常见的 LaTeX 语法错误，特别是 VLM 模型生成的错误格式
    
    Args:
        text: 包含 LaTeX 公式的文本
    
    Returns:
        修复后的文本
    """
    import re
    
    # 修复 \inS, \inR, \inN 等错误（应该是 \in S, \in \mathbb{R}, \in \mathbb{N}）
    # 匹配模式：\in[A-Z]（如 \inS, \inR, \inN）
    def fix_in_symbol(match):
        full_match = match.group(0)
        letter = full_match[-1]  # 获取最后一个字母
        
        # 特殊处理：R -> \mathbb{R}, N -> \mathbb{N}, Z -> \mathbb{Z}, Q -> \mathbb{Q}, C -> \mathbb{C}
        if letter in ['R', 'N', 'Z', 'Q', 'C']:
            return f"\\in \\mathbb{{{letter}}}"
        else:
            # 其他情况：\inS -> \in S
            return f"\\in {letter}"
    
    # 使用正则表达式查找并替换
    # 匹配 $...$ 或 $$...$$ 中的内容
    def fix_in_formula(match):
        formula_content = match.group(1)
        # 修复 \in[A-Z] 模式
        formula_content = re.sub(r'\\in([A-Z])', fix_in_symbol, formula_content)
        return f"${formula_content}$"
    
    # 修复行内公式 $...$
    text = re.sub(r'\$([^$]+)\$', fix_in_formula, text)
    
    # 修复块级公式 $$...$$
    def fix_in_display_formula(match):
        formula_content = match.group(1)
        # 修复 \in[A-Z] 模式
        formula_content = re.sub(r'\\in([A-Z])', fix_in_symbol, formula_content)
        return f"$${formula_content}$$"
    
    text = re.sub(r'\$\$([^$]+)\$\$', fix_in_display_formula, text)
    
    return text

def format_chart2table_func(block):
    lines_list = block.content.split("\n")
    # get header and rows
    header = lines_list[0].split("|")
    rows = [line.split("|") for line in lines_list[1:]]
    # construct html table
    html = "<table border=1 style='margin: auto; width: max-content;'>\n"
    html += (
        "  <thead><tr>"
        + "".join(
            f"<th style='text-align: center;'>{cell.strip()}</th>" for cell in header
        )
        + "</tr></thead>\n"
    )
    html += "  <tbody>\n"
    for row in rows:
        html += (
            "    <tr>"
            + "".join(
                f"<td style='text-align: center;'>{cell.strip()}</td>" for cell in row
            )
            + "</tr>\n"
        )
    html += "  </tbody>\n"
    html += "</table>"
    return html

def build_handle_funcs_dict(
    *,
    text_func,
    image_func,
    chart_func,
    table_func,
    formula_func,
    seal_func,
):
    """
    Build a dictionary mapping block labels to their formatting functions.

    Args:
        text_func: Function to format text blocks.
        image_func: Function to format image blocks.
        chart_func: Function to format chart blocks.
        table_func: Function to format table blocks.
        formula_func: Function to format formula blocks.
        seal_func: Function to format seal blocks.

    Returns:
        dict: A mapping from block label to handler function.
    """
    return {
        "paragraph_title": format_para_title_func,
        "abstract_title": format_title_func,
        "reference_title": format_title_func,
        "content_title": format_title_func,
        "doc_title": lambda block: f"# {block.content}".replace("-\n", "").replace(
            "\n", " "
        ),
        "table_title": text_func,
        "figure_title": text_func,
        "chart_title": text_func,
        "vision_footnote": lambda block: block.content.replace("\n\n", "\n").replace(
            "\n", "\n\n"
        ),
        "text": lambda block: block.content.replace("\n\n", "\n").replace("\n", "\n\n"),
        "ocr": lambda block: block.content.replace("\n\n", "\n").replace("\n", "\n\n"),
        "vertical_text": lambda block: block.content.replace("\n\n", "\n").replace(
            "\n", "\n\n"
        ),
        "reference_content": lambda block: block.content.replace("\n\n", "\n").replace(
            "\n", "\n\n"
        ),
        "abstract": partial(
            format_first_line_func,
            templates=["摘要", "abstract"],
            format_func=lambda l: f"## {l}\n",
            spliter=" ",
        ),
        "content": lambda block: block.content.replace("-\n", "  \n").replace(
            "\n", "  \n"
        ),
        "image": image_func,
        "chart": chart_func,
        "formula": formula_func,
        "display_formula": formula_func,
        "inline_formula": formula_func,
        "table": table_func,
        "reference": partial(
            format_first_line_func,
            templates=["参考文献", "references"],
            format_func=lambda l: f"## {l}",
            spliter="\n",
        ),
        "algorithm": lambda block: block.content.strip("\n"),
        "seal": seal_func,
        "spotting": lambda block: block.content,
        "number": format_text_plain_func,
        "footnote": format_text_plain_func,
        "header": format_text_plain_func,
        "header_image": image_func,
        "footer": format_text_plain_func,
        "footer_image": image_func,
        "aside_text": format_text_plain_func,
    }


def get_show_color(label: str, order_label=False):
    """获取显示颜色"""
    if order_label:
        label_colors = {
            "doc_title": (255, 248, 220, 100),
            "doc_title_text": (255, 239, 213, 100),
            "paragraph_title": (102, 102, 255, 100),
            "sub_paragraph_title": (102, 178, 255, 100),
            "vision": (153, 255, 51, 100),
            "vision_title": (144, 238, 144, 100),
            "vision_footnote": (144, 238, 144, 100),
            "normal_text": (153, 0, 76, 100),
            "cross_layout": (53, 218, 207, 100),
            "cross_reference": (221, 160, 221, 100),
        }
    else:
        label_colors = {
            "paragraph_title": (102, 102, 255, 100),
            "doc_title": (255, 248, 220, 100),
            "table_title": (255, 255, 102, 100),
            "figure_title": (102, 178, 255, 100),
            "chart_title": (221, 160, 221, 100),
            "vision_footnote": (144, 238, 144, 100),
            "text": (153, 0, 76, 100),
            "vertical_text": (153, 0, 76, 100),
            "inline_formula": (153, 0, 76, 100),
            "formula": (0, 255, 0, 100),
            "display_formula": (0, 255, 0, 100),
            "abstract": (255, 239, 213, 100),
            "content": (40, 169, 92, 100),
            "seal": (158, 158, 158, 100),
            "table": (204, 204, 0, 100),
            "image": (153, 255, 51, 100),
            "figure": (153, 255, 51, 100),
            "chart": (216, 191, 216, 100),
            "reference": (229, 255, 204, 100),
            "reference_content": (229, 255, 204, 100),
            "algorithm": (255, 250, 240, 100),
        }
    default_color = (158, 158, 158, 100)
    return label_colors.get(label, default_color)

# 完整的结果类实现（参考 PaddleX）
class PaddleOCRVLBlock(object):
    """PaddleOCRVL Block Class"""

    def __init__(
        self,
        label,
        bbox,
        content="",
        group_id=None,
        polygon_points=None,
        global_block_id=None,
        global_group_id=None,
    ) -> None:
        """
        Initialize a PaddleOCRVLBlock object.

        Args:
            label (str): Label assigned to the block.
            bbox (list): Bounding box coordinates of the block.
            content (str, optional): Content of the block. Defaults to an empty string.
        """
        self.label = label
        self.bbox = list(map(int, bbox))
        self.content = content
        self.image = None
        self.polygon_points = polygon_points
        self.group_id = group_id
        self.global_block_id = global_block_id
        self.global_group_id = global_group_id

    def __str__(self) -> str:
        """
        Return a string representation of the block.
        """
        _str = f"\n\n#################\nlabel:\t{self.label}\nbbox:\t{self.bbox}\ncontent:\t{self.content}\n#################"
        return _str

    def __repr__(self) -> str:
        """
        Return a string representation of the block.
        """
        _str = f"\n\n#################\nlabel:\t{self.label}\nbbox:\t{self.bbox}\ncontent:\t{self.content}\n#################"
        return _str


class PaddleOCRVLResult(dict):
    """
    PaddleOCRVLResult class for holding and formatting OCR/VL parsing results.
    参考 PaddleX 的完整实现
    """

    def __init__(self, data) -> None:
        """
        Initializes a new instance of the class with the specified data.

        Args:
            data: The input data for the parsing result.
        """
        super().__init__(data)
        self._save_funcs = []
        markdown_ignore_labels = self.get("model_settings", {}).get(
            "markdown_ignore_labels", []
        )
        # 用于可视化（layout_order_res）——不影响 JSON 的 block_order 规则
        self.visualize_order_labels = [
            label for label in VISUALIZE_ORDE_LABELS if label not in markdown_ignore_labels
        ]
        # 对齐 PaddleX：JSON 的 block_order 需要跳过这些 labels
        self.skip_order_labels = [label for label in SKIP_ORDER_LABELS + markdown_ignore_labels]

    def _get_input_fn(self):
        """获取输入文件名"""
        import time
        import random
        if self.get("input_path", None) is None:
            timestamp = int(time.time())
            random_number = random.randint(1000, 9999)
            fp = f"{timestamp}_{random_number}"
            return Path(fp).name
        fp = self["input_path"]
        return Path(fp).name

    def _to_img(self) -> dict:
        """
        自包含版本（无 Paddle/PaddleX 依赖）：
        - 若上游结果对象/字典提供了 `img`，则透传收集；
        - spotting 可视化仅用 PIL 绘制（不依赖 SIMFANG_FONT / draw_box_txt_fine / get_minarea_rect）。
        """
        res_img_dict: dict = {}
        model_settings = self.get("model_settings", {})

        # doc_preprocessor_res
        if model_settings.get("use_doc_preprocessor", False):
            dpr = self.get("doc_preprocessor_res")
            if isinstance(dpr, dict) and isinstance(dpr.get("img"), dict):
                res_img_dict.update(dpr["img"])
            elif hasattr(dpr, "img") and isinstance(getattr(dpr, "img"), dict):
                res_img_dict.update(getattr(dpr, "img"))
            elif isinstance(dpr, list):
                for idx, item in enumerate(dpr):
                    if hasattr(item, "img") and isinstance(getattr(item, "img"), dict):
                        for k, v in getattr(item, "img").items():
                            res_img_dict[f"{k}_{idx}"] = v

        # layout_det_res
        if model_settings.get("use_layout_detection", False):
            ldr = self.get("layout_det_res")
            if isinstance(ldr, dict) and isinstance(ldr.get("img"), dict):
                if "res" in ldr["img"]:
                    res_img_dict["layout_det_res"] = ldr["img"]["res"]
            elif hasattr(ldr, "img"):
                img = getattr(ldr, "img")
                if isinstance(img, dict) and "res" in img:
                    res_img_dict["layout_det_res"] = img["res"]
            elif isinstance(ldr, list):
                for idx, item in enumerate(ldr):
                    if hasattr(item, "img"):
                        img = getattr(item, "img")
                        if isinstance(img, dict) and "res" in img:
                            res_img_dict[f"layout_det_res_{idx}"] = img["res"]

        # spotting 可视化：左侧画框/多边形，右侧写文字
        spotting_res = self.get("spotting_res")
        if spotting_res and not isinstance(spotting_res, list):
            boxes = spotting_res.get("rec_polys", [])
            txts = spotting_res.get("rec_texts", [])
            output_img = (self.get("doc_preprocessor_res") or {}).get("output_img")
            if output_img is not None:
                image_bgr = output_img[:, :, ::-1]
                h, w = image_bgr.shape[0:2]
                img_left = Image.fromarray(image_bgr)
                img_right = Image.new("RGB", (w, h), (255, 255, 255))
                draw_left = ImageDraw.Draw(img_left)
                draw_right = ImageDraw.Draw(img_right)

                try:
                    font_size = int(0.018 * int(w)) + 2
                    font = ImageFont.truetype("arial.ttf", font_size, encoding="utf-8")
                except Exception:
                    font = ImageFont.load_default()

                random.seed(0)
                for box, txt in zip(boxes, txts):
                    try:
                        color = (
                            random.randint(0, 255),
                            random.randint(0, 255),
                            random.randint(0, 255),
                        )
                        if isinstance(txt, tuple):
                            txt = txt[0]
                        pts = np.array(box, dtype=np.int32)
                        if pts.ndim == 2 and pts.shape[1] == 2:
                            pts_list = [(int(x), int(y)) for x, y in pts.tolist()]
                        else:
                            pts_list = [(int(x), int(y)) for x, y in np.array(box).reshape(-1, 2).tolist()]

                        draw_left.polygon(pts_list, outline=color, width=3)
                        # 文本位置：取最小 (x,y)
                        xs = [p[0] for p in pts_list]
                        ys = [p[1] for p in pts_list]
                        tx, ty = (min(xs), min(ys)) if xs and ys else (0, 0)
                        draw_right.text((tx, ty), str(txt), fill=color, font=font)
                    except Exception:
                        continue

                img_left = Image.blend(Image.fromarray(image_bgr), img_left, 0.5)
                img_show = Image.new("RGB", (w * 2, h), (255, 255, 255))
                img_show.paste(img_left, (0, 0, w, h))
                img_show.paste(img_right, (w, 0, w * 2, h))
                res_img_dict["spotting_res_img"] = img_show

        return res_img_dict

    def _to_json(self, *args, **kwargs) -> dict:
        """
        自包含版本（无 Paddle/PaddleX 依赖）：
        - 返回结构对齐本工程历史：`{\"res\": data}`；
        - 递归将 numpy 类型转换为 JSON 友好类型；
        - 支持 `keep_img=True`（将 block.image 原样输出；默认不输出）。
        """
        _keep_img = bool(kwargs.pop("keep_img", False))

        def _to_jsonable(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            if isinstance(obj, dict):
                return {k: _to_jsonable(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_to_jsonable(x) for x in obj]
            return obj

        data = {}
        data["input_path"] = self.get("input_path")
        data["page_index"] = self.get("page_index")
        data["page_count"] = self.get("page_count")
        data["width"] = self.get("width")
        data["height"] = self.get("height")
        model_settings = self.get("model_settings", {})
        data["model_settings"] = model_settings
        use_seal_recognition = model_settings.get("use_seal_recognition", False)
        if model_settings.get("format_block_content", False):
            original_image_width = data["width"]
            use_ocr_for_image_block = model_settings.get("use_ocr_for_image_block", False)
            format_text_func = lambda block: format_centered_by_html(
                format_text_plain_func(block)
            )
            format_image_func = lambda block: format_centered_by_html(
                format_image_scaled_by_html_func(
                    block,
                    original_image_width=original_image_width,
                    show_ocr_content=use_ocr_for_image_block,
                ),
                remove_symbol=not use_ocr_for_image_block,
            )

            format_seal_func = lambda block: format_centered_by_html(
                format_image_scaled_by_html_func(
                    block,
                    original_image_width=original_image_width,
                    show_ocr_content=True,
                ),
                remove_symbol=use_seal_recognition,
            )

            if model_settings.get("use_chart_recognition", False):
                format_chart_func = format_chart2table_func
            else:
                format_chart_func = format_image_func

            if not model_settings.get("use_layout_detection", False):
                format_seal_func = format_text_func

            format_table_func = lambda block: "\n" + format_table_center_func(block)
            format_formula_func = lambda block: block.content

            handle_funcs_dict = build_handle_funcs_dict(
                text_func=format_text_func,
                image_func=format_image_func,
                chart_func=format_chart_func,
                table_func=format_table_func,
                formula_func=format_formula_func,
                seal_func=format_seal_func,
            )

        parsing_res_list = self.get("parsing_res_list", [])
        parsing_res_list_json = []
        order_index = 1
        for idx, parsing_res in enumerate(parsing_res_list):
            label = parsing_res.label
            if label not in self.skip_order_labels:
                order = order_index
                order_index += 1
            else:
                order = None
            res_dict = {
                "block_label": parsing_res.label,
                "block_content": parsing_res.content,
                "block_bbox": parsing_res.bbox,
                "block_id": idx,
                "block_order": order,
                "group_id": (
                    parsing_res.group_id if parsing_res.group_id is not None else idx
                ),
            }
            if (
                hasattr(parsing_res, "global_block_id")
                and parsing_res.global_block_id is not None
            ):
                res_dict["global_block_id"] = parsing_res.global_block_id
            if (
                hasattr(parsing_res, "global_group_id")
                and parsing_res.global_group_id is not None
            ):
                res_dict["global_group_id"] = parsing_res.global_group_id
            if parsing_res.polygon_points is not None:
                res_dict["block_polygon_points"] = _to_jsonable(parsing_res.polygon_points)

            if _keep_img and parsing_res.image is not None:
                res_dict["image"] = _to_jsonable(parsing_res.image)

            if model_settings.get("format_block_content", False):
                if handle_funcs_dict.get(parsing_res.label):
                    res_dict["block_content"] = handle_funcs_dict[parsing_res.label](
                        parsing_res
                    )
                else:
                    res_dict["block_content"] = parsing_res.content

            parsing_res_list_json.append(res_dict)
        data["parsing_res_list"] = parsing_res_list_json
        spotting_res = self.get("spotting_res")
        if spotting_res is not None:
            data["spotting_res"] = _to_jsonable(spotting_res)

        if model_settings.get("use_doc_preprocessor", False):
            dpr = self.get("doc_preprocessor_res")
            if isinstance(dpr, dict) and isinstance(dpr.get("json"), dict):
                data["doc_preprocessor_res"] = dpr["json"].get("res")
            elif hasattr(dpr, "json") and isinstance(getattr(dpr, "json"), dict):
                data["doc_preprocessor_res"] = getattr(dpr, "json").get("res", getattr(dpr, "json"))
            else:
                data["doc_preprocessor_res"] = _to_jsonable(dpr)

        if model_settings.get("use_layout_detection", False):
            ldr = self.get("layout_det_res")
            if isinstance(ldr, dict) and isinstance(ldr.get("json"), dict):
                data["layout_det_res"] = ldr["json"].get("res")
            elif hasattr(ldr, "json") and isinstance(getattr(ldr, "json"), dict):
                data["layout_det_res"] = getattr(ldr, "json").get("res", getattr(ldr, "json"))
            else:
                data["layout_det_res"] = _to_jsonable(ldr)

        return {"res": data}

    def _to_markdown(self, pretty=True, show_formula_number=False) -> dict:
        """
        Save the parsing result to a Markdown file.

        Args:
            pretty (Optional[bool]): whether to pretty markdown by HTML, default by True.
            show_formula_number (bool): whether to show formula numbers.

        Returns:
            dict: Markdown information with text and images.
        """
        model_settings = self.get("model_settings", {})

        # 对齐 PaddleX：使用 width（可能为 list）作为原图宽度基准
        use_ocr_for_image_block = model_settings.get("use_ocr_for_image_block", False)
        use_seal_recognition = model_settings.get("use_seal_recognition", False)
        if isinstance(self.get("width"), list):
            original_image_width = self.get("width")[0]
        else:
            original_image_width = self.get("width")

        if pretty:
            format_text_func = lambda block: format_centered_by_html(format_text_plain_func(block))
            format_image_func = lambda block: format_centered_by_html(
                format_image_scaled_by_html_func(block, original_image_width=original_image_width)
            )
            format_seal_func = lambda block: format_centered_by_html(
                format_image_scaled_by_html_func(block, original_image_width=original_image_width)
            )
        else:
            format_text_func = lambda block: block.content
            format_image_func = lambda block: format_image_plain_func(block)
            format_seal_func = lambda block: format_image_plain_func(block)

        format_chart_func = (
            format_chart2table_func if model_settings.get("use_chart_recognition", False) else format_image_func
        )

        # 对齐 PaddleX：若不使用 layout detection，则 seal 走文本（避免输出图片占位）
        if not model_settings.get("use_layout_detection", False):
            format_seal_func = format_text_func

        if pretty:
            format_table_func = lambda block: "\n" + format_table_center_func(block)
        else:
            format_table_func = lambda block: simplify_table_func("\n" + block.content)

        format_formula_func = lambda block: block.content

        handle_funcs_dict = build_handle_funcs_dict(
            text_func=format_text_func,
            image_func=format_image_func,
            chart_func=format_chart_func,
            table_func=format_table_func,
            formula_func=format_formula_func,
            seal_func=format_seal_func,
        )
        for label in model_settings.get("markdown_ignore_labels", []):
            handle_funcs_dict.pop(label, None)

        markdown_content = ""
        markdown_info = {}
        markdown_info["markdown_images"] = {}
        parsing_res_list = self.get("parsing_res_list", [])
        for idx, block in enumerate(parsing_res_list):
            label = block.label
            if block.image is not None:
                markdown_info["markdown_images"][block.image["path"]] = block.image[
                    "img"
                ]
            handle_func = handle_funcs_dict.get(label, None)
            if (
                show_formula_number
                and (label == "display_formula" or label == "formula")
                and idx != len(parsing_res_list) - 1
            ):
                next_block = parsing_res_list[idx + 1]
                next_block_label = next_block.label
                if next_block_label == "formula_number":
                    block.content = merge_formula_and_number(
                        block.content, next_block.content
                    )
            if handle_func:
                markdown_content += (
                    "\n\n" + handle_func(block)
                    if markdown_content
                    else handle_func(block)
                )

        markdown_info["page_index"] = self.get("page_index")
        markdown_info["input_path"] = self.get("input_path")
        markdown_info["markdown_texts"] = markdown_content
        for img in self.get("imgs_in_doc", []):
            markdown_info["markdown_images"][img["path"]] = img["img"]

        return markdown_info

    @property
    def json(self) -> dict:
        """Property to get the JSON representation of the result."""
        return self._to_json()

    @property
    def img(self) -> dict:
        """Property to get the image representation of the result."""
        return self._to_img()

    @property
    def markdown(self) -> dict:
        """Property to get the markdown representation of the result."""
        return self._to_markdown()

    def save_to_json(self, save_path, indent=4, ensure_ascii=False):
        """Save the JSON representation of the object to a file."""
        import json
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        fn = self._get_input_fn()
        json_path = save_path / f"{Path(fn).stem}_res.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self._to_json(), f, indent=indent, ensure_ascii=ensure_ascii)
        print(f"JSON saved to: {json_path}")

    def save_to_img(self, save_path, *args, **kwargs):
        """
        Save the image representation of the result to files.
        
        Args:
            save_path: The path to save the image(s). If the save path does not end with .jpg or .png, 
                      it appends the input path's stem and suffix to the save path.
            *args: Additional positional arguments that will be passed to the image writer.
            **kwargs: Additional keyword arguments that will be passed to the image writer.
        """
        import mimetypes
        
        def _is_image_file(file_path):
            mime_type, _ = mimetypes.guess_type(str(file_path))
            return mime_type is not None and mime_type.startswith("image/")
        
        img_dict = self._to_img()
        if not _is_image_file(save_path):
            fn = Path(self._get_input_fn())
            suffix = fn.suffix if _is_image_file(fn) else ".png"
            stem = fn.stem
            base_save_path = Path(save_path)
            base_save_path.mkdir(parents=True, exist_ok=True)
            for key in img_dict:
                if img_dict[key] is not None:
                    img_path = base_save_path / f"{stem}_{key}{suffix}"
                    self._save_image(img_path.as_posix(), img_dict[key], *args, **kwargs)
        else:
            if len(img_dict) > 1:
                import logging
                logging.warning(
                    f"The result has multiple img files need to be saved. But the `save_path` has been specified as `{save_path}`!"
                )
            # 保存第一个非 None 的图片
            for key, img in img_dict.items():
                if img is not None:
                    self._save_image(save_path, img, *args, **kwargs)
                    break

    def save_to_markdown(self, save_path, pretty=True, show_formula_number=False, *args, **kwargs):
        """
        Save the markdown representation of the result to a file.
        
        Args:
            save_path: 保存路径（目录或文件路径）
            pretty: 是否使用 HTML 美化 markdown
            show_formula_number: 是否显示公式编号
            *args: Additional positional arguments for saving.
            **kwargs: Additional keyword arguments for saving.
        """
        def _is_markdown_file(file_path) -> bool:
            """Check if a file is a markdown file based on its extension or MIME type."""
            markdown_extensions = {".md", ".markdown", ".mdown", ".mkd"}
            _, ext = os.path.splitext(str(file_path))
            if ext.lower() in markdown_extensions:
                return True
            import mimetypes
            mime_type, _ = mimetypes.guess_type(str(file_path))
            return mime_type == "text/markdown"
        
        import os
        import mimetypes
        
        if not _is_markdown_file(save_path):
            fn = Path(self._get_input_fn())
            suffix = fn.suffix if _is_markdown_file(fn) else ".md"
            stem = fn.stem
            base_save_path = Path(save_path)
            save_path = base_save_path / f"{stem}{suffix}"
            self.save_path = save_path
        else:
            self.save_path = save_path
        
        self._save_data(
            self._save_markdown_text,
            self._save_image,
            self.save_path,
            self._to_markdown(pretty=pretty, show_formula_number=show_formula_number),
            *args,
            **kwargs,
        )
    
    def _save_data(
        self,
        save_mkd_func,
        save_img_func,
        save_path,
        data,
        *args,
        **kwargs,
    ):
        """Internal method to save markdown and image data.
        
        Args:
            save_mkd_func: Function to save markdown text.
            save_img_func: Function to save image data.
            save_path: The base path where the data will be saved.
            data: The markdown data to save.
            *args: Additional positional arguments for saving.
            **kwargs: Additional keyword arguments for saving.
        """
        MARKDOWN_SAVE_KEYS = ["markdown_texts"]
        save_path = Path(save_path)
        if data is None:
            return
        for key, value in data.items():
            if key in MARKDOWN_SAVE_KEYS:
                save_mkd_func(save_path.as_posix(), value, *args, **kwargs)
            if isinstance(value, dict):
                base_save_path = save_path.parent
                for img_path, img_data in value.items():
                    save_img_func(
                        (base_save_path / img_path).as_posix(),
                        img_data,
                        *args,
                        **kwargs,
                    )
    
    def _save_markdown_text(self, out_path, text, *args, **kwargs):
        """Save markdown text to file."""
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write(text)
    
    def _save_image(self, out_path, img_data, *args, **kwargs):
        """Save image data to file."""
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        if isinstance(img_data, Image.Image):
            img_data.save(out_path)
        elif isinstance(img_data, np.ndarray):
            Image.fromarray(img_data).save(out_path)
        else:
            # 如果 img_data 是其他类型，尝试转换
            try:
                if hasattr(img_data, 'save'):
                    img_data.save(out_path)
                else:
                    print(f"Warning: Cannot save image of type {type(img_data)}")
            except Exception as e:
                print(f"Warning: Failed to save image {out_path}: {e}")

    def print(self):
        """Print the result."""
        print(f"Input: {self.get('input_path')}")
        print(f"Page: {self.get('page_index')}/{self.get('page_count')}")
        print(f"Size: {self.get('width')}x{self.get('height')}")
        parsing_res_list = self.get("parsing_res_list", [])
        print(f"Blocks: {len(parsing_res_list)}")
        for i, block in enumerate(parsing_res_list):
            print(f"\nBlock {i+1}:")
            print(f"  Label: {block.label}")
            print(f"  BBox: {block.bbox}")
            content_preview = block.content[:100] + "..." if len(block.content) > 100 else block.content
            print(f"  Content: {content_preview}")


class PaddleOCRVL:
    """
    OpenVINO 版本的 PaddleOCR-VL Pipeline
    使用 OpenVINO 进行布局检测和 VLM 推理
    """
    
    # ModelScope 模型 ID
    LAYOUT_MODEL_ID = "zhaohb/PP-DocLayoutV2-ov"
    VLM_MODEL_ID = "zhaohb/PaddleOCR-Vl-OV"
    
    def __init__(
        self,
        layout_model_path: Optional[str] = None,
        vlm_model_path: Optional[str] = None,
        vlm_device: str = "CPU",
        layout_device: str = "NPU",
        use_layout_detection: bool = True,
        use_chart_recognition: bool = True,
        use_seal_recognition: bool = False,
        use_ocr_for_image_block: bool = False,
        merge_layout_blocks: bool = True,
        markdown_ignore_labels: Optional[List[str]] = None,
        cache_dir: Optional[str] = None,
        layout_precision: str = "fp16",
        llm_int4_compress: bool = False,
        vision_int8_quant: bool = True,
        llm_int8_compress: bool = True,
        llm_int8_quant: bool = True,
    ):
        """
        初始化 PaddleOCR-VL Pipeline
        
        Args:
            layout_model_path: 布局检测模型路径（OpenVINO IR .xml 文件），如果为 None 则自动下载
            vlm_model_path: VLM 模型路径（包含 vision.xml, vision_mlp.xml, llm_stateful.xml 等的目录），如果为 None 则自动下载
            vlm_device: VLM 模型推理设备 ("CPU", "GPU", "AUTO")
            layout_device: 布局检测模型（PP-DocLayoutV2）推理设备，默认 "NPU" ("CPU", "GPU", "NPU", "AUTO")
            use_layout_detection: 是否使用布局检测
            use_chart_recognition: 是否使用图表识别
            merge_layout_blocks: 是否合并布局块
            markdown_ignore_labels: Markdown 输出中忽略的标签列表
            cache_dir: ModelScope 模型缓存目录，如果为 None 则使用默认缓存目录
            layout_precision: 布局检测模型精度选择，选项: "fp16", "fp32", "combined_fp16", "combined_fp32"
                - "fp16": FP16 精度模型（更快，内存占用更低）
                - "fp32": FP32 精度模型（更准确，默认）
                - "combined_fp16": FP16 合并模型（合并了 batch size 和 boxes 节点）
                - "combined_fp32": FP32 合并模型（合并了 batch size 和 boxes 节点）
                注意：如果指定了 layout_model_path 为具体的 .xml 文件路径，此参数将被忽略
        """
        self.vlm_device = vlm_device
        self.layout_device = layout_device
        self.use_layout_detection = use_layout_detection
        self.use_chart_recognition = use_chart_recognition
        # 对齐 PaddleX：seal / image-block OCR 的默认开关需要保存在 pipeline 上
        self.use_seal_recognition = use_seal_recognition
        self.use_ocr_for_image_block = use_ocr_for_image_block
        self.merge_layout_blocks = merge_layout_blocks
        self.markdown_ignore_labels = markdown_ignore_labels or [
            "number", "footnote", "header", "header_image",
            "footer", "footer_image", "aside_text"
        ]
        self.cache_dir = cache_dir
        self.layout_precision = layout_precision
        
        # 验证 precision 参数
        valid_precisions = ["fp16", "fp32", "combined_fp16", "combined_fp32"]
        if layout_precision not in valid_precisions:
            raise ValueError(
                f"Unsupported layout_precision: {layout_precision}. "
                f"Supported options: {valid_precisions}"
            )
        
        # 自动下载或验证模型路径
        if layout_model_path is None:
            if not MODELSCOPE_AVAILABLE:
                raise ImportError("modelscope is required for auto-download. Install with: pip install modelscope")
            print(f"📥 自动下载布局检测模型: {self.LAYOUT_MODEL_ID} (precision: {layout_precision})")
            layout_model_path = self._download_layout_model()
        else:
            layout_model_path = self._ensure_layout_model(layout_model_path)
        
        if vlm_model_path is None:
            if not MODELSCOPE_AVAILABLE:
                raise ImportError("modelscope is required for auto-download. Install with: pip install modelscope")
            print(f"📥 自动下载 VLM 模型: {self.VLM_MODEL_ID}")
            vlm_model_path = self._download_vlm_model()
        else:
            vlm_model_path = self._ensure_vlm_model(vlm_model_path)
        
        self.layout_model_path = layout_model_path
        self.vlm_model_path = vlm_model_path
        
        # 初始化 OpenVINO Core
        self.core = ov.Core()
        
        # 加载布局检测模型
        if self.use_layout_detection:
            self._load_layout_model()
        
        # 加载 VLM 模型
        self._load_vlm_model(llm_int4_compress=llm_int4_compress, vision_int8_quant=vision_int8_quant, llm_int8_compress=llm_int8_compress, llm_int8_quant=llm_int8_quant)
        
        # 不需要单独初始化图像处理器，VLM 模型内部会处理
    
    def _download_layout_model(self) -> str:
        """下载布局检测模型"""
        if not MODELSCOPE_AVAILABLE:
            raise ImportError("modelscope is required for auto-download. Install with: pip install modelscope")
        
        print(f"正在从 ModelScope 下载布局检测模型: {self.LAYOUT_MODEL_ID}")
        model_dir = snapshot_download(self.LAYOUT_MODEL_ID, cache_dir=self.cache_dir)
        model_dir = Path(model_dir)
        xml_files: List[Path] = []
        
        # 根据 precision 选择对应的模型文件
        precision_map = {
            "fp16": "pp_doclayoutv2_f16.xml",
            "fp32": "pp_doclayoutv2_f32.xml",
            "combined_fp16": "pp_doclayoutv2_f16_combined.xml",
            "combined_fp32": "pp_doclayoutv2_f32_combined.xml",
        }
        
        model_filename = precision_map.get(self.layout_precision)
        model_path = model_dir / model_filename if model_filename else None
        
        # 如果指定的精度文件不存在，尝试查找其他可用的模型文件
        if model_path is None or not model_path.exists():
            print(f"⚠️  指定的精度模型文件不存在: {model_filename if model_filename else 'N/A'}")
            # 查找所有 .xml 文件
            xml_files = list(model_dir.glob("*.xml"))
            if not xml_files:
                raise FileNotFoundError(
                    f"在下载的模型目录中未找到 .xml 文件: {model_dir}\n"
                    f"layout_precision={self.layout_precision}"
                )

            # 优先选择合并版本（combined_*）
            combined_files = [f for f in xml_files if "combined" in f.name]
            if combined_files:
                model_path = combined_files[0]
                print(f"⚠️  使用找到的合并模型: {model_path.name}")
            else:
                # 否则使用第一个找到的文件
                model_path = xml_files[0]
                print(f"⚠️  使用找到的模型: {model_path.name}")
        else:
            print(f"✅ 使用指定的精度模型: {model_filename}")
        
        # 检查对应的 .bin 文件是否存在
        bin_path = model_path.with_suffix(".bin")
        if not bin_path.exists():
            raise FileNotFoundError(f"对应的 .bin 文件不存在: {bin_path}")
        
        print(f"✅ 布局检测模型已下载到: {model_path}")
        return str(model_path)
    
    def _download_vlm_model(self) -> str:
        """下载 VLM 模型"""
        if not MODELSCOPE_AVAILABLE:
            raise ImportError("modelscope is required for auto-download. Install with: pip install modelscope")
        
        print(f"正在从 ModelScope 下载 VLM 模型: {self.VLM_MODEL_ID}")
        model_dir = snapshot_download(self.VLM_MODEL_ID, cache_dir=self.cache_dir)
        
        # 验证必要的文件是否存在
        required_files = ["vision.xml", "llm_stateful.xml", "llm_embd.xml"]
        model_path = Path(model_dir)
        missing_files = []
        for file_name in required_files:
            if not (model_path / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            raise FileNotFoundError(
                f"在下载的模型目录中缺少必要的文件: {missing_files}\n"
                f"模型目录: {model_dir}"
            )
        
        print(f"✅ VLM 模型已下载到: {model_dir}")
        return str(model_dir)
    
    def _ensure_layout_model(self, model_path: str) -> str:
        """确保布局检测模型存在，如果不存在则下载"""
        model_path_obj = Path(model_path)
        
        # 如果是目录，根据 precision 查找对应的 .xml 文件
        if model_path_obj.is_dir():
            # 根据 precision 优先级搜索
            precision_map = {
                "fp16": ["pp_doclayoutv2_f16.xml", "*.xml"],
                "fp32": ["pp_doclayoutv2_f32.xml", "*.xml"],
                "combined_fp16": ["pp_doclayoutv2_f16_combined.xml", "pp_doclayoutv2_f16.xml", "*.xml"],
                "combined_fp32": ["pp_doclayoutv2_f32_combined.xml", "pp_doclayoutv2_f32.xml", "*.xml"],
            }
            
            search_patterns = precision_map.get(self.layout_precision, ["*.xml"])
            xml_file = None
            
            for pattern in search_patterns:
                if pattern == "*.xml":
                    xml_files = list(model_path_obj.glob(pattern))
                    if xml_files:
                        xml_file = xml_files[0]
                    break
                else:
                    candidate = model_path_obj / pattern
                    if candidate.exists():
                        xml_file = candidate
                        break
            
            if xml_file is None:
                print(f"⚠️  在指定目录中未找到匹配的 .xml 文件，尝试自动下载: {model_path}")
                return self._download_layout_model()
            
            # 检查对应的 .bin 文件是否存在
            bin_path = xml_file.with_suffix(".bin")
            if not bin_path.exists():
                print(f"⚠️  对应的 .bin 文件不存在: {bin_path}，尝试自动下载")
                return self._download_layout_model()
            
            return str(xml_file)
        
        # 如果是文件路径，检查文件是否存在
        if not model_path_obj.exists():
            print(f"⚠️  模型文件不存在，尝试自动下载: {model_path}")
            return self._download_layout_model()
        
        # 如果指定了具体的 .xml 文件路径，直接使用（忽略 precision 参数）
        if model_path_obj.suffix.lower() == ".xml":
            bin_path = model_path_obj.with_suffix(".bin")
            if not bin_path.exists():
                print(f"⚠️  对应的 .bin 文件不存在: {bin_path}，尝试自动下载")
                return self._download_layout_model()
            return model_path
        
        return model_path
    
    def _ensure_vlm_model(self, model_path: str) -> str:
        """确保 VLM 模型存在，如果不存在则下载"""
        model_path_obj = Path(model_path)
        
        if not model_path_obj.exists():
            print(f"⚠️  模型目录不存在，尝试自动下载: {model_path}")
            return self._download_vlm_model()
        
        # 验证必要的文件是否存在
        required_files = ["vision.xml", "llm_stateful.xml", "llm_embd.xml"]
        missing_files = []
        for file_name in required_files:
            if not (model_path_obj / file_name).exists():
                missing_files.append(file_name)
        
        if missing_files:
            print(f"⚠️  模型目录中缺少必要的文件 {missing_files}，尝试自动下载")
            return self._download_vlm_model()
        
        return model_path
    
    def _load_layout_model(self):
        """加载布局检测模型"""
        model = self.core.read_model(self.layout_model_path)
        
        # 添加预处理
        prep = ov.preprocess.PrePostProcessor(model)
        prep.input("image").tensor().set_layout(ov.Layout("NCHW"))
        prep.input("image").preprocess().scale([255, 255, 255])
        model = prep.build()
        
        # 编译模型（使用 layout_device）
        self.layout_compiled_model = self.core.compile_model(model, self.layout_device)
        self.layout_request = self.layout_compiled_model.create_infer_request()
    
    def _load_vlm_model(self, llm_int4_compress=False, vision_int8_quant=True, llm_int8_compress=True, llm_int8_quant=True):
        """加载 VLM 模型"""
        self.vlm_model = OVPaddleOCRVLForCausalLM(
            core=self.core,
            ov_model_path=self.vlm_model_path,
            device=self.vlm_device,
            llm_int4_compress=llm_int4_compress, 
            vision_int8_quant=vision_int8_quant, 
            llm_int8_compress=llm_int8_compress, 
            llm_int8_quant=llm_int8_quant, 
        )
    
    def predict(
        self,
        input: Union[str, List[str], np.ndarray, List[np.ndarray]],
        use_layout_detection: Optional[bool] = None,
        use_chart_recognition: Optional[bool] = None,
        use_seal_recognition: Optional[bool] = None,
        use_ocr_for_image_block: Optional[bool] = None,
        layout_threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, tuple]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        layout_shape_mode: Optional[str] = "auto",
        max_new_tokens: Optional[int] = None,
        prompt_label: str = "ocr",
        **kwargs,
    ):
        """
        预测文档解析结果
        
        Args:
            input: 输入图像路径、图像路径列表、numpy 数组或 numpy 数组列表
            use_layout_detection: 是否使用布局检测（覆盖初始化设置）
            layout_threshold: 布局检测阈值
            layout_nms: 是否使用 NMS
            layout_unclip_ratio: 坐标扩展比例
            layout_merge_bboxes_mode: 布局框合并模式
            max_new_tokens: 最大生成 token 数
            **kwargs: 其他参数
        
        Yields:
            PaddleOCRVLResult: 解析结果对象
        """
        # 确定是否使用布局检测
        if use_layout_detection is None:
            use_layout_detection = self.use_layout_detection

        # 对齐 PaddleX：None 表示使用初始化默认值
        if use_chart_recognition is None:
            use_chart_recognition = self.use_chart_recognition

        # 对齐 PaddleX：None 表示使用初始化默认值
        if use_seal_recognition is None:
            use_seal_recognition = self.use_seal_recognition

        # 对齐 PaddleX：None 表示使用初始化默认值
        if use_ocr_for_image_block is None:
            use_ocr_for_image_block = self.use_ocr_for_image_block

        # layout shape mode default
        if layout_shape_mode is None:
            layout_shape_mode = "auto"

        # 对齐 PaddleX：关闭 layout_detection 时，prompt_label 会影响开启的识别分支
        if not use_layout_detection and isinstance(prompt_label, str):
            if prompt_label.lower() == "seal":
                use_seal_recognition = True
            elif prompt_label.lower() == "chart":
                use_chart_recognition = True
        
        # 处理输入
        if isinstance(input, str):
            inputs = [input]
        elif isinstance(input, np.ndarray):
            inputs = [input]
        elif isinstance(input, list):
            inputs = input
        else:
            raise ValueError(f"Unsupported input type: {type(input)}")
        
        # 处理每个输入
        for idx, inp in enumerate(inputs):
            # 读取图像
            if isinstance(inp, str):
                image = cv2.imread(inp)
                input_path = inp
            elif isinstance(inp, np.ndarray):
                image = inp
                input_path = None
            else:
                raise ValueError(f"Unsupported input item type: {type(inp)}")
            
            if image is None:
                raise ValueError(f"Failed to load image: {inp}")
            
            # 执行 CV 处理（布局检测）
            results_cv = self._process_cv(
                image,
                input_path,
                use_layout_detection=use_layout_detection,
                layout_threshold=layout_threshold,
                layout_nms=layout_nms,
                layout_unclip_ratio=layout_unclip_ratio,
                layout_merge_bboxes_mode=layout_merge_bboxes_mode,
                layout_shape_mode=layout_shape_mode,
                prompt_label=prompt_label,
            )
            
            # 执行 VLM 处理（布局解析）
            result = self._process_vlm(
                results_cv,
                max_new_tokens=max_new_tokens,
                use_chart_recognition=use_chart_recognition,
                use_seal_recognition=use_seal_recognition,
                use_ocr_for_image_block=use_ocr_for_image_block,
                layout_shape_mode=layout_shape_mode,
            )
            
            yield result
    
    def _process_cv(
        self,
        image: np.ndarray,
        input_path: Optional[str],
        page_index: Optional[int] = None,
        use_layout_detection: bool = True,
        layout_threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, tuple]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        layout_shape_mode: str = "auto",
        prompt_label: str = "ocr",
    ):
        """
        处理计算机视觉部分（布局检测）
        参考 PaddleX 的实现，确保功能一致
        
        Args:
            image: 输入图像（BGR 格式）
            input_path: 输入路径
            page_index: 页面索引
            use_layout_detection: 是否使用布局检测
            layout_threshold: 布局检测阈值
            layout_nms: 是否使用 NMS
            layout_unclip_ratio: 坐标扩展比例
            layout_merge_bboxes_mode: 布局框合并模式
            prompt_label: 不使用布局检测时的默认标签（默认 "ocr"）
        
        Returns:
            dict: 包含布局检测结果的字典，格式与 PaddleX 一致
        """
        # 文档预处理（这里简化处理，直接使用原图）
        # 如果后续需要文档预处理（如方向校正、去弯曲等），可以在这里添加
        doc_preprocessor_image = image.copy()
        doc_preprocessor_res = {"output_img": doc_preprocessor_image}
        
        # 布局检测
        if use_layout_detection and self.use_layout_detection:
            # 执行布局检测
            layout_det_res = self._layout_detection(
                doc_preprocessor_image,
                threshold=layout_threshold or 0.5,
                layout_nms=layout_nms if layout_nms is not None else True,
                layout_unclip_ratio=layout_unclip_ratio or [1.0, 1.0],
                layout_merge_bboxes_mode=layout_merge_bboxes_mode,
                layout_shape_mode=layout_shape_mode,
            )

            # 提取文档中的图像（参考 PaddleX 的 gather_imgs）
            imgs_in_doc = gather_imgs(doc_preprocessor_image, layout_det_res["boxes"])
            
            # 设置 input_path 和 page_index
            layout_det_res["input_path"] = input_path
            layout_det_res["page_index"] = page_index
        else:
            # 如果不使用布局检测，创建全图框（参考 PaddleX 的实现）
            h, w = doc_preprocessor_image.shape[:2]
            layout_det_res = {
                "input_path": input_path,
                "page_index": page_index,
                "boxes": [
                    {
                        "cls_id": 0,
                        "label": prompt_label.lower(),
                        "score": 1.0,
                        "coordinate": [0, 0, w, h],
                    }
                ],
            }
            # 不使用布局检测时，不提取图像
            imgs_in_doc = []
        
        # 创建 LayoutDetectionResult 对象并获取 json 和 img
        import os
        layout_det_result_obj = LayoutDetectionResult(
            input_path=os.path.abspath(input_path) if input_path else None,
            boxes=layout_det_res["boxes"],
            page_index=page_index,
            input_img=doc_preprocessor_image
        )
        
        # 将 layout 可视化结果图保存到 output 目录（用户需求）。
        # 同时保留环境变量覆盖：PADDLEOCR_VL_DEBUG_SAVE_DIR=/path/to/dir
        try:
            save_dir = os.environ.get("PADDLEOCR_VL_DEBUG_SAVE_DIR", "").strip()
            if not save_dir:
                # 默认保存到 paddleocr_vl_ov/output（与现有测试脚本输出目录保持一致）
                save_dir = str(Path(__file__).resolve().parents[2] / "output")

            # 避免同名覆盖：加入 page_index
            base_name = Path(layout_det_result_obj._get_input_fn()).stem
            page_tag = f"_page_{int(page_index):04d}" if page_index is not None else ""
            out_file = Path(save_dir) / f"{base_name}{page_tag}_layout_res.png"
            layout_det_result_obj.save_to_img(save_path=out_file.as_posix())
        except Exception:
            pass
        
        return {
            "input_path": input_path,
            "page_index": page_index,
            "page_count": 1,
            "doc_preprocessor_image": doc_preprocessor_image,
            "doc_preprocessor_res": doc_preprocessor_res,
            "layout_det_results": [layout_det_res],
            "imgs_in_doc": [imgs_in_doc],
        }
    
    def _layout_detection(
        self,
        image: np.ndarray,
        threshold: Union[float, dict] = 0.5,
        layout_nms: bool = True,
        layout_unclip_ratio: Union[float, tuple] = None,
        layout_merge_bboxes_mode: str = None,
        layout_shape_mode: str = "auto",
    ):
        """
        执行布局检测
        
        Args:
            image: 输入图像（BGR 格式）
            threshold: 检测阈值
            layout_nms: 是否使用 NMS
            layout_unclip_ratio: 坐标扩展比例
            layout_merge_bboxes_mode: 布局框合并模式
        
        Returns:
            dict: 布局检测结果
        """
        orig_h, orig_w = image.shape[:2]
        
        # 预处理
        input_blob, scale_h, scale_w = preprocess_image_doclayout(image)
        
        # 准备输入
        input_tensors = self.layout_compiled_model.inputs
        input_data = {}
        
        # breakpoint()
        for inp in input_tensors:
            inp_name = inp.get_any_name()
            if inp_name == "im_shape":
                input_data[inp_name] = np.array([800, 800], dtype=np.float32)[np.newaxis, ...]
            elif inp_name == "image":
                input_data[inp_name] = input_blob
            elif inp_name == "scale_factor":
                input_data[inp_name] = np.array([[scale_h, scale_w]], dtype=np.float32)
        
        # 如果输入名称不匹配，按顺序分配
        if len(input_data) != len(input_tensors):
            input_data = {}
            input_data[input_tensors[0].get_any_name()] = np.array([800, 800], dtype=np.float32)[np.newaxis, ...]
            input_data[input_tensors[1].get_any_name()] = input_blob
            input_data[input_tensors[2].get_any_name()] = np.array([[scale_h, scale_w]], dtype=np.float32)
        
        # 创建 OpenVINO Tensor 对象
        input_tensors_ov = {}
        for inp in input_tensors:
            inp_name = inp.get_any_name()
            data = input_data[inp_name]
            input_tensors_ov[inp_name] = ov.Tensor(data)
        
        # 执行推理
        infer_result = self.layout_compiled_model(input_tensors_ov)
        output_tensors = self.layout_compiled_model.outputs
            # Extract output results
        output = []
        for out in output_tensors:
            output_tensor = infer_result[out]
            output_data = output_tensor.data
            output.append(output_data)
        
        # Post-processing
        if layout_unclip_ratio is None:
            layout_unclip_ratio = [1.0, 1.0]
        
        # Choose postprocess based on output shapes.
        out0 = np.array(output[0]) if len(output) > 0 else None
        out1 = np.array(output[1]) if len(output) > 1 else None
        if out0 is not None and out0.ndim == 2 and out0.shape[0] == 300 and out0.shape[1] in (6, 7) and out1 is not None and out1.size >= 1:
            # PaddleDetection exported (already NMS-ed) outputs
            results = postprocess_detections_paddle_nms(
                output,
                orig_h=orig_h,
                orig_w=orig_w,
                threshold=threshold,
                layout_nms=layout_nms,
                layout_unclip_ratio=layout_unclip_ratio,
                layout_merge_bboxes_mode=layout_merge_bboxes_mode,
                layout_shape_mode=layout_shape_mode,
            )
        else:
            # Fallback to DETR-style postprocess (older models)
            if output[0].ndim == 3:
                output[0] = np.squeeze(output[0], axis = 0)

            results = postprocess_detections_detr(
                output, scale_h, scale_w, orig_h, orig_w,
                threshold=threshold,
                layout_nms=layout_nms,
                layout_unclip_ratio=layout_unclip_ratio,
                layout_merge_bboxes_mode=layout_merge_bboxes_mode,
                layout_shape_mode=layout_shape_mode,
            )
        
        result_obj = LayoutAnalysisResult(
            {
                "input_path": None,
                "page_index": None,
                "input_img": image,
                "boxes": results,
            }
        )
        
        return result_obj
    
    def _get_label_name(self, cls_id: int) -> str:
        """获取标签名称"""
        label_list = [
            "abstract", "algorithm", "aside_text", "chart", "content", "display_formula",
            "doc_title", "figure_title", "footer", "footer_image", "footnote", "formula_number",
            "header", "header_image", "image", "inline_formula", "number", "paragraph_title",
            "reference", "reference_content", "seal", "table", "text", "vertical_text", "vision_footnote"
        ]
        if 0 <= cls_id < len(label_list):
            return label_list[cls_id]
        return "unknown"
    
    def _process_vlm(
        self,
        results_cv: dict,
        max_new_tokens: Optional[int] = None,
        use_chart_recognition: Optional[bool] = None,
        use_seal_recognition: Optional[bool] = None,
        use_ocr_for_image_block: Optional[bool] = None,
        layout_shape_mode: str = "auto",
    ):
        """
        处理视觉语言模型部分（布局解析）
        
        Args:
            results_cv: CV 处理结果
            max_new_tokens: 最大生成 token 数
        
        Returns:
            PaddleOCRVLResult: 解析结果对象
        """
        (
            input_path,
            page_index,
            page_count,
            doc_preprocessor_image,
            doc_preprocessor_res,
            layout_det_results,
            imgs_in_doc,
        ) = (
            results_cv["input_path"],
            results_cv["page_index"],
            results_cv["page_count"],
            results_cv["doc_preprocessor_image"],
            results_cv["doc_preprocessor_res"],
            results_cv["layout_det_results"],
            results_cv["imgs_in_doc"],
        )
        
        # 获取布局解析结果
        parsing_res_lists, table_res_lists, spotting_res_lists, imgs_in_doc = self.get_layout_parsing_results(
            [doc_preprocessor_image],
            layout_det_results,
            imgs_in_doc,
            max_new_tokens=max_new_tokens or 4096,
            use_chart_recognition=bool(use_chart_recognition) if use_chart_recognition is not None else self.use_chart_recognition,
            use_seal_recognition=bool(use_seal_recognition) if use_seal_recognition is not None else self.use_seal_recognition,
            use_ocr_for_image_block=bool(use_ocr_for_image_block) if use_ocr_for_image_block is not None else self.use_ocr_for_image_block,
            layout_shape_mode=layout_shape_mode,
        )
        
        # 组装结果
        parsing_res_list = parsing_res_lists[0] if parsing_res_lists else []
        table_res_list = table_res_lists[0] if table_res_lists else []
        spotting_res = spotting_res_lists[0] if spotting_res_lists else {}
        # Align with PaddleX doc_vl pipeline:
        # PaddleX does NOT reorder parsing_res_list here; it relies on layout_det_res["boxes"]
        # being already in the desired reading order.
        
        single_img_res = {
            "input_path": input_path,
            "page_index": page_index,
            "page_count": page_count,
            "width": doc_preprocessor_image.shape[1],
            "height": doc_preprocessor_image.shape[0],
            "doc_preprocessor_res": doc_preprocessor_res,
            "layout_det_res": layout_det_results[0] if layout_det_results else None,
            "table_res_list": table_res_list,
            "parsing_res_list": parsing_res_list,
            # 对齐 PaddleX：将每页 spotting_res 回填到 single_img_res
            "spotting_res": spotting_res,
            "imgs_in_doc": imgs_in_doc[0] if imgs_in_doc else [],
            "model_settings": {
                "use_doc_preprocessor": False,
                "use_layout_detection": self.use_layout_detection,
                "use_chart_recognition": bool(use_chart_recognition) if use_chart_recognition is not None else self.use_chart_recognition,
                "use_seal_recognition": bool(use_seal_recognition) if use_seal_recognition is not None else self.use_seal_recognition,
                "use_ocr_for_image_block": bool(use_ocr_for_image_block) if use_ocr_for_image_block is not None else self.use_ocr_for_image_block,
                "format_block_content": False,
                "merge_layout_blocks": self.merge_layout_blocks,
                "markdown_ignore_labels": self.markdown_ignore_labels,
                "return_layout_polygon_points": False if layout_shape_mode == "rect" else True,
            },
        }
        
        return PaddleOCRVLResult(single_img_res)
    
    def get_layout_parsing_results(
        self,
        images: List[np.ndarray],
        layout_det_results: List[dict],
        imgs_in_doc: List[List],
        max_new_tokens: int = 4096,
        use_chart_recognition: bool = False,
        use_seal_recognition: bool = False,
        use_ocr_for_image_block: bool = False,
        layout_shape_mode: str = "auto",
    ):
        """
        获取布局解析结果（参考 PaddleX 的实现，确保逻辑一致）
        
        Args:
            images: 图像列表
            layout_det_results: 布局检测结果列表
            imgs_in_doc: 文档中的图像列表
            max_new_tokens: 最大生成 token 数
        
        Returns:
            tuple: (parsing_res_lists, table_res_lists, spotting_res_list, imgs_in_doc)
        """
        # Align with PaddleX doc_vl pipeline:
        # - group VLM requests by (min_pixels, max_pixels)
        # - allow spotting branch
        has_spotting = False
        drop_figures_set = set()
        default_min_pixels = 112896
        default_max_pixels = 1003520

        batch_dict_by_pixel = {}
        id2pixel_key_map = {}
        image_path_to_obj_map = {}

        # 对齐 PaddleX：
        # - vis_image_labels: 用于决定是否给 block_info.image 赋值（与是否进入 VLM 无关）
        # - image_labels: 用于决定是否跳过 VLM（在 image_labels 内的 label 不进入 VLM，作为纯图片块）
        vis_image_labels = ["image", "header_image", "footer_image", "seal"]
        image_labels = [] if use_ocr_for_image_block else ["image", "header_image", "footer_image"]
        # 对齐 PaddleX：chart 分支由入参决定（predict 里 None->self 默认）
        if not use_chart_recognition:
            image_labels += ["chart"]
            vis_image_labels += ["chart"]
        # 对齐 PaddleX：seal 分支由入参决定（predict 里 None->self 默认）
        if not use_seal_recognition:
            image_labels += ["seal"]
        
        blocks = []
        for i, (image, layout_det_res, imgs_in_doc_for_img) in enumerate(zip(images, layout_det_results, imgs_in_doc)):
            layout_det_res = filter_overlap_boxes(layout_det_res, layout_shape_mode=layout_shape_mode)
            boxes = layout_det_res["boxes"]
            # 对齐 PaddleX：不在 parsing 阶段额外重排 boxes。
            # PaddleX 直接使用 layout_det_res["boxes"] 的原始顺序（仅做 filter_overlap_boxes），
            # 后续 crop/merge 都依赖该顺序，从而保证 parsing_res_list 的块顺序一致。
            
            # 裁剪图像区域
            blocks_for_img = self._crop_by_boxes(image, boxes, layout_shape_mode=layout_shape_mode)

            # # 保存裁剪图像为 jpg（便于调试/核对）
            # try:
            #     crop_dir = Path("output") / "cropped_blocks"
            #     crop_dir.mkdir(parents=True, exist_ok=True)
            #     for j, block in enumerate(blocks_for_img):
            #         block_img = block.get("img")
            #         if block_img is None:
            #             continue
            #         if not hasattr(block_img, "shape"):
            #             continue
            #         if getattr(block_img, "size", 0) == 0:
            #             continue

            #         label = str(block.get("label", "unknown"))
            #         safe_label = "".join(ch if ch.isalnum() or ch in ("_", "-", ".") else "_" for ch in label)

            #         # 文件名：页序号_块序号_label.jpg（不包含坐标）
            #         out_path = crop_dir / f"{i:03d}_{j:03d}_{safe_label}.jpg"

            #         img_to_save = block_img
            #         # 确保 uint8
            #         if isinstance(img_to_save, np.ndarray) and img_to_save.dtype != np.uint8:
            #             img_to_save = np.clip(img_to_save, 0, 255).astype(np.uint8)
            #         cv2.imwrite(out_path.as_posix(), img_to_save)
            # except Exception:
            #     # 保存失败不影响主流程
            #     pass
            
            # 合并布局块（如果需要）
            if self.merge_layout_blocks:
                blocks_for_img = merge_blocks(
                    blocks_for_img,
                    non_merge_labels=image_labels + ["table"],
                    layout_shape_mode=layout_shape_mode,
                )
            
            blocks.append(blocks_for_img)
            
            # 准备 VLM 输入（参考 PaddleX doc_vl）
            for j, block in enumerate(blocks_for_img):
                block_img = block["img"]
                block_label = block["label"]
                
                if block_label not in image_labels and block_img is not None:
                    figure_token_map = {}
                    text_prompt = "OCR:"
                    min_pixels = default_min_pixels
                    max_pixels = default_max_pixels
                    drop_figures = []
                    
                    if block_label == "table":
                        text_prompt = "Table Recognition:"
                        block_img, figure_token_map, drop_figures = tokenize_figure_of_table(
                            block_img, block["box"], imgs_in_doc_for_img
                        )
                    elif block_label == "chart" and use_chart_recognition:
                        text_prompt = "Chart Recognition:"
                    elif "formula" in block_label and block_label != "formula_number":
                        text_prompt = "Formula Recognition:"
                        crop_img = crop_margin(block_img)
                        try:
                            w0, h0, _ = crop_img.shape
                            if w0 > 2 and h0 > 2:
                                block_img = crop_img
                        except Exception:
                            pass
                    elif block_label == "spotting":
                        text_prompt = "Spotting:"
                        has_spotting = True

                    pixel_key = (min_pixels, max_pixels)
                    if pixel_key not in batch_dict_by_pixel:
                        batch_dict_by_pixel[pixel_key] = {
                            "images": [],
                            "queries": [],
                            "figure_token_maps": [],
                            "vlm_block_ids": [],
                            "curr_vlm_block_idx": 0,
                        }
                    batch_dict_by_pixel[pixel_key]["images"].append(block_img)
                    batch_dict_by_pixel[pixel_key]["queries"].append(text_prompt)
                    batch_dict_by_pixel[pixel_key]["figure_token_maps"].append(figure_token_map)
                    batch_dict_by_pixel[pixel_key]["vlm_block_ids"].append((i, j))
                    id2pixel_key_map[(i, j)] = pixel_key
                    
                    drop_figures_set.update(drop_figures)  # 参考 PaddleX 第 277 行

        # VLM 推理：按 (min_pixels, max_pixels) 分桶（对齐 PaddleX doc_vl）
        for pixel_key in batch_dict_by_pixel:
            min_pixels, max_pixels = pixel_key
            images_bucket = batch_dict_by_pixel[pixel_key]["images"]
            queries_bucket = batch_dict_by_pixel[pixel_key]["queries"]
            # Spotting 分支不强制覆盖 min/max_pixels（与 PaddleX 行为一致）
            vlm_kwargs = {"max_new_tokens": max_new_tokens}
            if not has_spotting:
                vlm_kwargs["min_pixels"] = min_pixels
                vlm_kwargs["max_pixels"] = max_pixels

            batch_results = self._vlm_predict(
                images_bucket,
                queries_bucket,
                **vlm_kwargs,
            )
            batch_dict_by_pixel[pixel_key]["vlm_results"] = batch_results
        
        # 组装解析结果
        parsing_res_lists = []
        table_res_lists = []
        spotting_res_list = []
        table_blocks = []
        
        for i, blocks_for_img in enumerate(blocks):
            parsing_res_list = []
            table_res_list = []
            spotting_res = {}
            
            for j, block in enumerate(blocks_for_img):
                block_img = block["img"]
                block_bbox = block["box"]
                block_label = block["label"]
                block_content = ""
                figure_token_map = {}

                if (i, j) in id2pixel_key_map:
                    pixel_key = id2pixel_key_map[(i, j)]
                    pixel_info = batch_dict_by_pixel[pixel_key]
                    curr_vlm_block_idx = pixel_info["curr_vlm_block_idx"]
                    assert curr_vlm_block_idx < len(pixel_info["vlm_block_ids"]) and pixel_info["vlm_block_ids"][curr_vlm_block_idx] == (i, j)
                    vl_rec_result = pixel_info["vlm_results"][curr_vlm_block_idx]
                    block_img4vl = pixel_info["images"][curr_vlm_block_idx]
                    figure_token_map = pixel_info["figure_token_maps"][curr_vlm_block_idx]
                    curr_vlm_block_idx += 1
                    pixel_info["curr_vlm_block_idx"] = curr_vlm_block_idx

                    vl_rec_result["image"] = block_img4vl
                    result_str = vl_rec_result.get("result", "")
                    if result_str is None:
                        result_str = ""
                    
                    # 处理重复内容（对齐 PaddleX doc_vl：table=5000 else 50）
                    min_count = 5000 if block_label == "table" else 50
                    result_str = truncate_repetitive_content(result_str, min_count=min_count)
                    
                    # 处理公式格式（参考 PaddleX 第 338-350 行）
                    if ("\\(" in result_str and "\\)" in result_str) or (
                        "\\[" in result_str and "\\]" in result_str
                    ):
                        result_str = result_str.replace("$", "")
                        result_str = (
                            result_str.replace("\\(", " $ ")
                            .replace("\\)", " $")
                            .replace("\\[\\[", "\\[")
                            .replace("\\]\\]", "\\]")
                            .replace("\\[", " $$ ")
                            .replace("\\]", " $$ ")
                        )
                        if block_label == "formula_number":
                            result_str = result_str.replace("$", "")
                    
                    # 修复 LaTeX 语法错误（修复 \inS, \inR 等常见错误）
                    result_str = fix_latex_syntax(result_str)
                    
                    # 处理表格（参考 PaddleX 第 351-357 行）
                    if block_label == "table":
                        html_str = convert_otsl_to_html(result_str)
                        if html_str != "":
                            result_str = html_str

                    if block_label == "spotting":
                        h_, w_ = block_img.shape[:2]
                        result_str, spotting_res = post_process_for_spotting(result_str, w_, h_)
                    
                    block_content = result_str
                
                block_info = PaddleOCRVLBlock(
                    label=block_label,
                    bbox=block_bbox,
                    content=block_content,
                    group_id=block.get("group_id", None),
                    polygon_points=block.get("polygon_points", None),
                )
                
                # 设置图片信息（对齐 PaddleX doc_vl：构造 image_path_to_obj_map + drop_figures_set）
                # 当 block_label 在 image_labels 中且 block_img 不为 None 时，设置 block_info.image
                if block_label in vis_image_labels and block_img is not None:
                    x_min, y_min, x_max, y_max = list(map(int, block_bbox))
                    img_path = f"imgs/img_in_{block_label}_box_{x_min}_{y_min}_{x_max}_{y_max}.jpg"
                    image_path_to_obj_map[img_path] = block_info
                    # 如果图片在 drop_figures_set 中，跳过这个 block（参考 PaddleX 第 370-379 行）
                    if img_path not in drop_figures_set:
                        # 转换 BGR 到 RGB（如果 block_img 是 NumPy 数组）
                        if isinstance(block_img, np.ndarray):
                            block_img_rgb = cv2.cvtColor(block_img, cv2.COLOR_BGR2RGB)
                            block_info.image = {
                                "path": img_path,
                                "img": Image.fromarray(block_img_rgb),
                            }
                        elif isinstance(block_img, Image.Image):
                            block_info.image = {
                                "path": img_path,
                                "img": block_img,
                            }
                    else:
                        # 如果图片在 drop_figures_set 中，跳过这个 block（参考 PaddleX 第 379 行）
                        continue
                
                parsing_res_list.append(block_info)
                if block_label == "table":
                    table_blocks.append({"figure_token_map": figure_token_map, "block": block_info})
            
            # 对齐 PaddleX：table 内容里回填图片 token（需要 image_path_to_obj_map）
            for blk_info in table_blocks:
                blk = blk_info["block"]
                ftm = blk_info["figure_token_map"]
                blk.content = untokenize_figure_of_table(blk.content, ftm, image_path_to_obj_map)

            parsing_res_lists.append(parsing_res_list)
            table_res_lists.append(table_res_list)
            spotting_res_list.append(spotting_res)
        
        # 对齐 PaddleX（见提交：support save block_id/block_order、concatenate_pages/restructure_pages）：
        # 为每个 block 分配全局 id，且当 group_id 为空时使用 global_block_id 作为默认 group_id。
        global_block_id = 0
        for one_page_parsing in parsing_res_lists:
            for blk in one_page_parsing:
                if getattr(blk, "global_block_id", None) is None:
                    blk.global_block_id = global_block_id
                if getattr(blk, "global_group_id", None) is None:
                    blk.global_group_id = global_block_id
                if getattr(blk, "group_id", None) is None:
                    blk.group_id = global_block_id
                global_block_id += 1

        # 对齐 PaddleX：跨页 merge_table 后回写 global_group_id（见 PaddleX layout_parsing/merge_table.py）
        # 仅当存在多页时才尝试合并；best-effort，不影响主流程。
        if len(parsing_res_lists) > 1:
            try:
                parsing_res_lists = self._merge_tables_across_pages_paddlex(parsing_res_lists)
            except Exception as e:
                logging.warning(f"merge_tables_across_pages failed: {e}")

        return parsing_res_lists, table_res_lists, spotting_res_list, imgs_in_doc

    def _merge_tables_across_pages_paddlex(self, pages: List[List["PaddleOCRVLBlock"]]) -> List[List["PaddleOCRVLBlock"]]:
        """
        对齐 PaddleX `paddlex/inference/pipelines/layout_parsing/merge_table.py::merge_tables_across_pages`：
        - 如果相邻两页的表格满足“跨页续表”条件，则将当前页表格内容合并到上一页表格；
        - 回写：`curr_block.content = ""` 且 `curr_block.global_group_id = prev_block.global_block_id`；
        - 最后对 global_group_id 做链式归一化（指向最终的 group root）。
        """
        from bs4 import BeautifulSoup  # 已在环境验证存在

        def full_to_half(text: str) -> str:
            result = []
            for char in text:
                code = ord(char)
                if 0xFF01 <= code <= 0xFF5E:
                    result.append(chr(code - 0xFEE0))
                else:
                    result.append(char)
            return "".join(result)

        def calculate_table_total_columns(soup):
            rows = soup.find_all("tr")
            if not rows:
                return 0
            max_cols = 0
            occupied = {}
            for row_idx, row in enumerate(rows):
                col_idx = 0
                cells = row.find_all(["td", "th"])
                if row_idx not in occupied:
                    occupied[row_idx] = {}
                for cell in cells:
                    while col_idx in occupied[row_idx]:
                        col_idx += 1
                    colspan = int(cell.get("colspan", 1))
                    rowspan = int(cell.get("rowspan", 1))
                    for r in range(row_idx, row_idx + rowspan):
                        if r not in occupied:
                            occupied[r] = {}
                        for c in range(col_idx, col_idx + colspan):
                            occupied[r][c] = True
                    col_idx += colspan
                    max_cols = max(max_cols, col_idx)
            return max_cols

        def calculate_row_columns(row):
            return sum(int(cell.get("colspan", 1)) for cell in row.find_all(["td", "th"]))

        def calculate_visual_columns(row):
            return len(row.find_all(["td", "th"]))

        def detect_table_headers(soup1, soup2, max_header_rows=5):
            rows1 = soup1.find_all("tr")
            rows2 = soup2.find_all("tr")
            min_rows = min(len(rows1), len(rows2), max_header_rows)
            header_rows = 0
            headers_match = True
            for i in range(min_rows):
                cells1 = rows1[i].find_all(["td", "th"])
                cells2 = rows2[i].find_all(["td", "th"])
                if len(cells1) != len(cells2):
                    headers_match = header_rows > 0
                    break
                match = True
                for c1, c2 in zip(cells1, cells2):
                    text1 = "".join(full_to_half(c1.get_text()).split())
                    text2 = "".join(full_to_half(c2.get_text()).split())
                    if text1 != text2 or int(c1.get("colspan", 1)) != int(c2.get("colspan", 1)):
                        match = False
                        break
                if match:
                    header_rows += 1
                else:
                    headers_match = header_rows > 0
                    break
            if header_rows == 0:
                headers_match = False
            return header_rows, headers_match

        def check_rows_match(soup1, soup2):
            rows1 = soup1.find_all("tr")
            rows2 = soup2.find_all("tr")
            if not rows1 or not rows2:
                return False
            last_row = rows1[-1]
            header_count, _ = detect_table_headers(soup1, soup2)
            first_data_row = rows2[header_count] if len(rows2) > header_count else None
            if not first_data_row:
                return False
            last_cols = calculate_row_columns(last_row)
            first_cols = calculate_row_columns(first_data_row)
            last_visual = calculate_visual_columns(last_row)
            first_visual = calculate_visual_columns(first_data_row)
            return last_cols == first_cols or last_visual == first_visual

        def is_skippable(block, allowed_labels):
            continue_keywords = ["continue", "continued", "cont'd", "续", "cont‘d", "續"]
            if block.label in allowed_labels:
                return True
            b_text = str(getattr(block, "text", "") or "").lower()
            b_fig_title = str(getattr(block, "figure_title", "") or "").lower()
            b_doc_title = str(getattr(block, "doc_title", "") or "").lower()
            b_para_title = str(getattr(block, "paragraph_title", "") or "").lower()
            full_content = f"{b_text} {b_fig_title} {b_doc_title} {b_para_title}"
            if any(kw in full_content for kw in continue_keywords):
                return True
            return False

        def can_merge_tables(prev_page, prev_block, curr_page, curr_block):
            x0, y0, x1, y1 = prev_block.bbox
            prev_width = x1 - x0
            x2, y2, x3, y4 = curr_block.bbox
            curr_width = x3 - x2
            if curr_width == 0 or prev_width == 0:
                return False, None, None
            if abs(curr_width - prev_width) / min(curr_width, prev_width) >= 0.1:
                return False, None, None

            prev_index = prev_page.index(prev_block)
            allowed_follow = all(
                b.label in ["footer", "vision_footnote", "number", "footnote", "footer_image", "seal"]
                for b in prev_page[prev_index + 1 :]
            )
            if not allowed_follow:
                return False, None, None

            curr_index = curr_page.index(curr_block)
            curr_allowed_labels = ["header", "header_image", "number", "seal"]
            allowed_before = all(is_skippable(b, curr_allowed_labels) for b in curr_page[:curr_index])
            if not allowed_before:
                return False, None, None

            html_prev = prev_block.content
            html_curr = curr_block.content
            if not html_prev or not html_curr:
                return False, None, None
            soup_prev = BeautifulSoup(html_prev, "html.parser")
            soup_curr = BeautifulSoup(html_curr, "html.parser")

            total_cols_prev = calculate_table_total_columns(soup_prev)
            total_cols_curr = calculate_table_total_columns(soup_curr)
            tables_match = total_cols_prev == total_cols_curr
            rows_match = check_rows_match(soup_prev, soup_curr)
            return (tables_match or rows_match), soup_prev, soup_curr

        def perform_table_merge(soup_prev, soup_curr):
            header_count, _ = detect_table_headers(soup_prev, soup_curr)
            rows_prev = soup_prev.find_all("tr")
            rows_curr = soup_curr.find_all("tr")
            for row in rows_curr[header_count:]:
                row.extract()
                rows_prev[-1].parent.append(row)
            return str(soup_prev)

        # main merge loop (same as PaddleX)
        for i in range(len(pages) - 1, 0, -1):
            page_curr = pages[i]
            page_prev = pages[i - 1]

            curr_block = next((b for b in page_curr if b.label == "table"), None)
            prev_block = next((b for b in reversed(page_prev) if b.label == "table"), None)

            if curr_block and prev_block:
                can_merge, soup_prev, soup_curr = can_merge_tables(page_prev, prev_block, page_curr, curr_block)
            else:
                can_merge = False

            if can_merge:
                merged_html = perform_table_merge(soup_prev, soup_curr)
                prev_block.content = merged_html
                prev_block_global_id = prev_block.global_block_id
                curr_block.content = ""
                curr_block.global_group_id = prev_block_global_id

        # normalize global_group_id chain (same as PaddleX)
        all_blocks = [block for page in pages for block in page]
        for page in pages:
            for block in page:
                if block.global_block_id != block.global_group_id:
                    block.global_group_id = all_blocks[block.global_group_id].global_group_id

        return pages
    
    def _vlm_predict(
        self,
        block_imgs: List[np.ndarray],
        text_prompts: List[str],
        max_new_tokens: int = 4096,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        **kwargs,
    ):
        """
        使用 VLM 模型进行预测（参考 torch_ov_test.py 的 OpenVINO 推理方式）
        
        Args:
            block_imgs: 图像块列表
            text_prompts: 文本提示列表
            max_new_tokens: 最大生成 token 数
        
        Returns:
            list: VLM 预测结果列表
        """
        results = []
        
        # 准备 generation_config
        generation_config = {
            "bos_token_id": self.vlm_model.tokenizer.bos_token_id,
            "eos_token_id": self.vlm_model.tokenizer.eos_token_id,
            "pad_token_id": self.vlm_model.tokenizer.pad_token_id,
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }

        for idx, (block_img, text_prompt) in enumerate(zip(block_imgs, text_prompts)):
            # 转换图像格式
            if isinstance(block_img, np.ndarray):
                if len(block_img.shape) == 2:
                    block_img = cv2.cvtColor(block_img, cv2.COLOR_GRAY2RGB)
                elif block_img.shape[2] == 3:
                    block_img = cv2.cvtColor(block_img, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(block_img)
            else:
                pil_image = block_img
            
            # # 保存 pil_image 为图片
            # import os
            # output_dir = "output"
            # os.makedirs(output_dir, exist_ok=True)
            # save_path = os.path.join(output_dir, f"pil_image_{idx}.png")
            # pil_image.save(save_path)
            # print(f"Saved pil_image to: {save_path}")
            # # breakpoint()
            pil_image = pil_image.resize((1200, 800), Image.Resampling.LANCZOS)
            
            # 准备输入消息（与 torch_ov_test.py 一致）
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_image},
                        {"type": "text", "text": text_prompt},
                    ]
                }
            ]
            
            try:
                # 使用 chat 方法进行推理（与 torch_ov_test.py 一致）
                response, history = self.vlm_model.chat(
                    messages=messages,
                    generation_config=generation_config
                )
                result_str = response
            except Exception as e:
                # 如果 VLM 推理失败，返回空字符串
                print(f"Warning: VLM inference failed: {e}")
                result_str = ""
            
            # print("result_str: ", result_str)
            results.append({"result": result_str})
        
        return results
    
    def _crop_by_boxes(
        self, image: np.ndarray, boxes: List[dict], layout_shape_mode: str = "auto"
    ) -> List[dict]:
        """
        根据框裁剪图像
        
        Args:
            image: 输入图像
            boxes: 框列表
        
        Returns:
            list: 裁剪后的图像块列表
        """
        blocks = []
        h, w = image.shape[:2]
        
        for box_info in boxes:
            coordinate = box_info["coordinate"]
            xmin, ymin, xmax, ymax = map(int, coordinate)
            
            # 确保坐标在图像范围内
            xmin = max(0, min(xmin, w))
            ymin = max(0, min(ymin, h))
            xmax = max(xmin, min(xmax, w))
            ymax = max(ymin, min(ymax, h))
            
            if xmax > xmin and ymax > ymin:
                img_crop = image[ymin:ymax, xmin:xmax].copy()
                out_info = {
                    "img": img_crop,
                    "label": box_info["label"],
                    "box": [xmin, ymin, xmax, ymax],
                    "score": box_info["score"],
                    # Align with PaddleX schema: keep polygon points for downstream json output
                    "polygon_points": box_info.get("polygon_points", None),
                }

                # 当 layout_shape_mode != "rect" 且存在 polygon_points 时，仍保留 polygon_points 供下游使用，
                # 但不对裁剪图像做“多边形外填充为白色”的 mask 处理（按需求：不填充）。
                if layout_shape_mode != "rect" and "polygon_points" in box_info:
                    out_info["polygon_points"] = box_info.get("polygon_points", None)

                blocks.append(out_info)
        
        return blocks
    
    
    def _calculate_projection_overlap_ratio(self, bbox1, bbox2, direction="horizontal"):
        """计算投影重叠比例（参考 PaddleX）"""
        start_index, end_index = (1, 3) if direction == "vertical" else (0, 2)
        intersection_start = max(bbox1[start_index], bbox2[start_index])
        intersection_end = min(bbox1[end_index], bbox2[end_index])
        overlap = intersection_end - intersection_start
        if overlap <= 0:
            return 0
        ref_width = max(bbox1[end_index], bbox2[end_index]) - min(
            bbox1[start_index], bbox2[start_index]
        )
        return overlap / ref_width if ref_width > 0 else 0.0
    
    def _calculate_overlap_ratio(self, bbox1, bbox2, mode="union"):
        """计算重叠比例（参考 PaddleX）"""
        bbox1 = np.array(bbox1)
        bbox2 = np.array(bbox2)

        x_min_inter = np.maximum(bbox1[0], bbox2[0])
        y_min_inter = np.maximum(bbox1[1], bbox2[1])
        x_max_inter = np.minimum(bbox1[2], bbox2[2])
        y_max_inter = np.minimum(bbox1[3], bbox2[3])

        inter_width = np.maximum(0, x_max_inter - x_min_inter)
        inter_height = np.maximum(0, y_max_inter - y_min_inter)

        inter_area = inter_width * inter_height

        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

        if mode == "union":
            ref_area = bbox1_area + bbox2_area - inter_area
        elif mode == "small":
            ref_area = np.minimum(bbox1_area, bbox2_area)
        elif mode == "large":
            ref_area = np.maximum(bbox1_area, bbox2_area)
        else:
            raise ValueError(
                f"Invalid mode {mode}, must be one of ['union', 'small', 'large']."
            )

        if ref_area == 0:
            return 0.0

        return inter_area / ref_area
    
    def _to_pil_image(self, img):
        """转换为 PIL Image"""
        if isinstance(img, Image.Image):
            return img
        return Image.fromarray(img)
    
    def _to_np_array(self, img):
        """转换为 numpy array"""
        if isinstance(img, Image.Image):
            return np.array(img)
        return img
    
    def _calc_merged_wh(self, images):
        """计算合并后的宽高（参考 PaddleX）"""
        widths = [self._to_pil_image(img).width for img in images]
        heights = [self._to_pil_image(img).height for img in images]
        w = max(widths)
        h = sum(heights)
        return w, h
    
    def _merge_images(self, images, aligns="center"):
        """合并图像（参考 PaddleX）"""
        if not images:
            return None
        if len(images) == 1:
            return self._to_np_array(images[0])
        if isinstance(aligns, str):
            aligns = [aligns] * (len(images) - 1)
        if len(aligns) != len(images) - 1:
            raise ValueError("The length of aligns must be len(images) - 1")
        merged = self._to_pil_image(images[0])
        for i in range(1, len(images)):
            img2 = self._to_pil_image(images[i])
            align = aligns[i - 1]
            w = max(merged.width, img2.width)
            h = merged.height + img2.height
            new_img = Image.new("RGB", (w, h), (255, 255, 255))
            if align == "center":
                x1 = (w - merged.width) // 2
                x2 = (w - img2.width) // 2
            elif align == "right":
                x1 = w - merged.width
                x2 = w - img2.width
            else:  # left
                x1 = x2 = 0
            new_img.paste(merged, (x1, 0))
            new_img.paste(img2, (x2, merged.height))
            merged = new_img
        return self._to_np_array(merged)
    
    def close(self):
        """
        关闭模型，尽可能释放 OpenVINO / VLM 相关资源。
        """
        # VLM
        try:
            vlm = getattr(self, "vlm_model", None)
            if vlm is not None:
                for m in ("close", "release"):
                    fn = getattr(vlm, m, None)
                    if callable(fn):
                        try:
                            fn()
                        except Exception:
                            pass
                        break
        except Exception:
            pass
        try:
            self.vlm_model = None
        except Exception:
            pass

        # Layout (optional)
        for name in ("layout_request", "layout_compiled_model"):
            try:
                obj = getattr(self, name, None)
                if obj is not None:
                    for m in ("close", "release"):
                        fn = getattr(obj, m, None)
                        if callable(fn):
                            try:
                                fn()
                            except Exception:
                                pass
                            break
            except Exception:
                pass
            try:
                setattr(self, name, None)
            except Exception:
                pass

        try:
            self.core = None
        except Exception:
            pass

