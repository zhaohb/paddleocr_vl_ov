"""
OpenVINO 版本的 PaddleOCR-VL Pipeline 实现
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
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

# 导入 VLM 模型
from ..paddleocr_vl.ov_paddleocr_vl import OVPaddleOCRVLForCausalLM

# 导入图像处理
from ..paddleocr_vl.image_processing_paddleocr_vl import PaddleOCRVLImageProcessor

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

# 格式化函数（参考 PaddleX）
def format_title_func(block):
    """格式化标题"""
    import re
    title = block.content
    # 简单的标题格式化
    title = title.rstrip(".")
    level = title.count(".") + 1 if "." in title else 1
    return f"#{'#' * level} {title}".replace("-\n", "").replace("\n", " ")

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
    """表格居中格式化"""
    tabel_content = block.content
    tabel_content = tabel_content.replace(
        "<table>", "<table border=1 style='margin: auto; width: max-content;'>"
    )
    tabel_content = tabel_content.replace("<th>", "<th style='text-align: center;'>")
    tabel_content = tabel_content.replace("<td>", "<td style='text-align: center;'>")
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
    """合并公式和公式编号"""
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
    """图表转表格格式化"""
    lines_list = block.content.split("\n")
    header = lines_list[0].split("|")
    rows = [line.split("|") for line in lines_list[1:]]
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
    """构建处理函数字典"""
    from functools import partial
    return {
        "paragraph_title": format_title_func,
        "abstract_title": format_title_func,
        "reference_title": format_title_func,
        "content_title": format_title_func,
        "doc_title": lambda block: f"# {block.content}".replace("-\n", "").replace("\n", " "),
        "table_title": text_func,
        "figure_title": text_func,
        "chart_title": text_func,
        "vision_footnote": lambda block: block.content.replace("\n\n", "\n").replace("\n", "\n\n"),
        "text": lambda block: block.content.replace("\n\n", "\n").replace("\n", "\n\n"),
        "ocr": lambda block: block.content.replace("\n\n", "\n").replace("\n", "\n\n"),
        "vertical_text": lambda block: block.content.replace("\n\n", "\n").replace("\n", "\n\n"),
        "reference_content": lambda block: block.content.replace("\n\n", "\n").replace("\n", "\n\n"),
        "abstract": partial(
            format_first_line_func,
            templates=["摘要", "abstract"],
            format_func=lambda l: f"## {l}\n",
            spliter=" ",
        ),
        "content": lambda block: block.content.replace("-\n", "  \n").replace("\n", "  \n"),
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
    """PaddleOCRVL Block Class（参考 PaddleX 实现）"""

    def __init__(self, label, bbox, content="", group_id=None) -> None:
        """
        Initialize a PaddleOCRVLBlock object.

        Args:
            label (str): Label assigned to the block.
            bbox (list): Bounding box coordinates of the block.
            content (str, optional): Content of the block. Defaults to an empty string.
            group_id: Group ID for the block.
        """
        self.label = label
        self.bbox = list(map(int, bbox))
        self.content = content
        self.image = None
        self.group_id = group_id

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
        self.visualize_order_labels = [
            label
            for label in VISUALIZE_ORDE_LABELS
            if label not in markdown_ignore_labels
        ]

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
        Convert the parsing result to a dictionary of images.

        Returns:
            dict: Keys are names, values are numpy arrays (images).
        """
        res_img_dict = {}
        model_settings = self.get("model_settings", {})
        if model_settings.get("use_doc_preprocessor", False):
            doc_preprocessor_res = self.get("doc_preprocessor_res", {})
            if isinstance(doc_preprocessor_res, dict) and "img" in doc_preprocessor_res:
                for key, value in doc_preprocessor_res["img"].items():
                    res_img_dict[key] = value
        if model_settings.get("use_layout_detection", False):
            layout_det_res = self.get("layout_det_res")
            if layout_det_res and isinstance(layout_det_res, dict) and "img" in layout_det_res:
                res_img_dict["layout_det_res"] = layout_det_res["img"].get("res")

        # for layout ordering image
        doc_preprocessor_res = self.get("doc_preprocessor_res", {})
        output_img = doc_preprocessor_res.get("output_img")
        if output_img is not None:
            image = Image.fromarray(output_img[:, :, ::-1])
            draw = ImageDraw.Draw(image, "RGBA")
            font_size = int(0.018 * int(image.width)) + 2
            try:
                font = ImageFont.truetype("arial.ttf", font_size, encoding="utf-8")
            except:
                font = ImageFont.load_default()
            parsing_result = self.get("parsing_res_list", [])

            order_index = 0
            for block in parsing_result:
                bbox = block.bbox
                label = block.label
                fill_color = get_show_color(label, False)
                draw.rectangle(bbox, fill=fill_color)
                if label in self.visualize_order_labels:
                    text_position = (bbox[2] + 2, bbox[1] - font_size // 2)
                    if int(image.width) - bbox[2] < font_size:
                        text_position = (
                            int(bbox[2] - font_size * 1.1),
                            bbox[1] - font_size // 2,
                        )
                    draw.text(text_position, str(order_index + 1), font=font, fill="red")
                    order_index += 1

            res_img_dict["layout_order_res"] = image

        return res_img_dict

    def _to_json(self) -> dict:
        """
        Converts the object's data to a JSON dictionary.

        Returns:
            dict: A dictionary containing the object's data in JSON format.
        """
        import copy
        data = {}
        data["input_path"] = self.get("input_path")
        data["page_index"] = self.get("page_index")
        data["page_count"] = self.get("page_count")
        data["width"] = self.get("width")
        data["height"] = self.get("height")
        model_settings = self.get("model_settings", {})
        data["model_settings"] = model_settings
        
        if model_settings.get("format_block_content", False):
            doc_preprocessor_res = self.get("doc_preprocessor_res", {})
            output_img = doc_preprocessor_res.get("output_img")
            original_image_width = output_img.shape[1] if output_img is not None else 500
            format_text_func = lambda block: format_centered_by_html(
                format_text_plain_func(block)
            )
            format_image_func = lambda block: format_centered_by_html(
                format_image_scaled_by_html_func(
                    block,
                    original_image_width=original_image_width,
                )
            )

            if model_settings.get("use_chart_recognition", False):
                format_chart_func = format_chart2table_func
            else:
                format_chart_func = format_image_func

            format_seal_func = format_image_func
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
            if label in self.visualize_order_labels:
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
            if model_settings.get("format_block_content", False):
                if handle_funcs_dict.get(parsing_res.label):
                    res_dict["block_content"] = handle_funcs_dict[parsing_res.label](
                        parsing_res
                    )
                else:
                    res_dict["block_content"] = parsing_res.content

            parsing_res_list_json.append(res_dict)
        data["parsing_res_list"] = parsing_res_list_json
        
        if model_settings.get("use_doc_preprocessor", False):
            doc_preprocessor_res = self.get("doc_preprocessor_res", {})
            if isinstance(doc_preprocessor_res, dict) and "json" in doc_preprocessor_res:
                data["doc_preprocessor_res"] = doc_preprocessor_res["json"].get("res")
        if model_settings.get("use_layout_detection", False):
            layout_det_res = self.get("layout_det_res")
            if layout_det_res and isinstance(layout_det_res, dict) and "json" in layout_det_res:
                data["layout_det_res"] = layout_det_res["json"].get("res")
        
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
        doc_preprocessor_res = self.get("doc_preprocessor_res", {})
        output_img = doc_preprocessor_res.get("output_img")
        original_image_width = output_img.shape[1] if output_img is not None else 500

        if pretty:
            format_text_func = lambda block: format_centered_by_html(
                format_text_plain_func(block)
            )
            format_image_func = lambda block: format_centered_by_html(
                format_image_scaled_by_html_func(
                    block,
                    original_image_width=original_image_width,
                )
            )
        else:
            format_text_func = lambda block: block.content
            format_image_func = format_image_plain_func

        model_settings = self.get("model_settings", {})
        format_chart_func = (
            format_chart2table_func
            if model_settings.get("use_chart_recognition", False)
            else format_image_func
        )

        if pretty:
            format_table_func = lambda block: "\n" + format_table_center_func(block)
        else:
            format_table_func = lambda block: simplify_table_func("\n" + block.content)

        format_formula_func = lambda block: block.content
        format_seal_func = format_image_func

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
        layout_threshold: Optional[Union[float, dict]] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, tuple]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
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
                prompt_label=prompt_label,
            )
            
            # 执行 VLM 处理（布局解析）
            result = self._process_vlm(
                results_cv,
                max_new_tokens=max_new_tokens,
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
            )
            
            # 过滤重叠框
            layout_det_res = self._filter_overlap_boxes(layout_det_res)
            
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
        
        # NOTE: PaddleX 不会在这里强制落盘保存可视化图片。
        # 之前硬编码保存到 "output" 会导致多 PDF/多页结果互相覆盖（例如 page_0001_res.png 重复）。
        # 仅在显式设置环境变量时保存，便于调试。
        debug_save_dir = os.environ.get("PADDLEOCR_VL_DEBUG_SAVE_DIR", "").strip()
        if debug_save_dir:
            try:
                layout_det_result_obj.save_to_img(save_path=debug_save_dir)
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
        result = self.layout_compiled_model(input_tensors_ov)
        
        # 提取输出结果
        output = []
        output_tensors = self.layout_compiled_model.outputs
        for out in output_tensors:
            output_tensor = result[out]
            output.append(output_tensor.data)
        
        # 后处理：根据输出形状选择后处理函数
        out0 = np.array(output[0]) if len(output) > 0 else None
        out1 = np.array(output[1]) if len(output) > 1 else None
        if out0 is not None and out0.ndim == 2 and out0.shape[0] == 300 and out0.shape[1] in (6, 7) and out1 is not None and out1.size >= 1:
            # PaddleDetection exported (already NMS-ed) outputs
            boxes = postprocess_detections_paddle_nms(
                output,
                orig_h=orig_h,
                orig_w=orig_w,
                threshold=threshold,
                layout_nms=layout_nms,
                layout_unclip_ratio=layout_unclip_ratio,
                layout_merge_bboxes_mode=layout_merge_bboxes_mode,
            )
        else:
            # Fallback to DETR-style postprocess (older models)
            # Handle 3D arrays with batch dimension of 1: squeeze the first dimension
            if output[0].ndim == 3:
                output[0] = np.squeeze(output[0], axis=0)
            if len(output) > 1 and output[1].ndim == 3:
                output[1] = np.squeeze(output[1], axis=0)
            
        boxes = postprocess_detections_detr(
            output,
            scale_h,
            scale_w,
            orig_h,
            orig_w,
            threshold=threshold,
            layout_nms=layout_nms,
            layout_unclip_ratio=layout_unclip_ratio,
            layout_merge_bboxes_mode=layout_merge_bboxes_mode,
        )
        
        # 转换为结果格式
        # postprocess_detections_detr 可能返回字典列表（restructured_boxes）或空列表
        if len(boxes) == 0:
            layout_det_res = {
                "input_path": None,
                "page_index": None,
                "boxes": [],
            }
        elif isinstance(boxes[0], dict):
            # 如果已经是字典格式（restructured_boxes 返回的），直接使用
            layout_det_res = {
                "input_path": None,
                "page_index": None,
                "boxes": boxes,
            }
        else:
            # 如果是 numpy 数组格式，转换为字典格式
            layout_det_res = {
                "input_path": None,
                "page_index": None,
                "boxes": [
                    {
                        "cls_id": int(box[0]),
                        "label": self._get_label_name(int(box[0])),
                        "score": float(box[1]),
                        "coordinate": [float(box[2]), float(box[3]), float(box[4]), float(box[5])],
                    }
                    for box in boxes
                ],
            }
        
        return layout_det_res
    
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
        parsing_res_lists, table_res_lists, imgs_in_doc = self.get_layout_parsing_results(
            [doc_preprocessor_image],
            layout_det_results,
            imgs_in_doc,
            max_new_tokens=max_new_tokens or 4096,
        )
        
        # 组装结果
        parsing_res_list = parsing_res_lists[0] if parsing_res_lists else []
        table_res_list = table_res_lists[0] if table_res_lists else []
        
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
            "imgs_in_doc": imgs_in_doc[0] if imgs_in_doc else [],
            "model_settings": {
                "use_doc_preprocessor": False,
                "use_layout_detection": self.use_layout_detection,
                "use_chart_recognition": self.use_chart_recognition,
                "format_block_content": False,
                "merge_layout_blocks": self.merge_layout_blocks,
                "markdown_ignore_labels": self.markdown_ignore_labels,
            },
        }
        
        return PaddleOCRVLResult(single_img_res)
    
    def get_layout_parsing_results(
        self,
        images: List[np.ndarray],
        layout_det_results: List[dict],
        imgs_in_doc: List[List],
        max_new_tokens: int = 4096,
    ):
        """
        获取布局解析结果（参考 PaddleX 的实现，确保逻辑一致）
        
        Args:
            images: 图像列表
            layout_det_results: 布局检测结果列表
            imgs_in_doc: 文档中的图像列表
            max_new_tokens: 最大生成 token 数
        
        Returns:
            tuple: (parsing_res_lists, table_res_lists, imgs_in_doc)
        """
        blocks = []
        block_imgs = []
        text_prompts = []
        vlm_block_ids = []
        figure_token_maps = []
        drop_figures_set = set()  # 参考 PaddleX 第 239 行
        
        image_labels = ["image", "header_image", "footer_image", "seal"]
        if not self.use_chart_recognition:
            image_labels.append("chart")
        
        for i, (image, layout_det_res, imgs_in_doc_for_img) in enumerate(
            zip(images, layout_det_results, imgs_in_doc)
        ):
            boxes = layout_det_res["boxes"]
            
            # 裁剪图像区域
            blocks_for_img = self._crop_by_boxes(image, boxes)
            
            # 合并布局块（如果需要）
            if self.merge_layout_blocks:
                blocks_for_img = self._merge_blocks(
                    blocks_for_img, non_merge_labels=image_labels + ["table"]
                )
            
            blocks.append(blocks_for_img)
            
            # 准备 VLM 输入（参考 PaddleX 第 254-277 行）
            for j, block in enumerate(blocks_for_img):
                block_img = block["img"]
                block_label = block["label"]
                
                if block_label not in image_labels and block_img is not None:
                    figure_token_map = {}
                    text_prompt = "OCR:"
                    drop_figures = []
                    
                    if block_label == "table":
                        text_prompt = "Table Recognition:"
                        # 对于 table，需要处理表格中的图片（参考 PaddleX 第 261-267 行）
                        try:
                            from ..paddleocr_vl.uilts import (
                                tokenize_figure_of_table,
                            )
                            block_img, figure_token_map, drop_figures = (
                                tokenize_figure_of_table(
                                    block_img, block["box"], imgs_in_doc_for_img
                                )
                            )
                        except ImportError:
                            # 如果无法导入，使用空实现
                            pass
                    elif block_label == "chart" and self.use_chart_recognition:
                        text_prompt = "Chart Recognition:"
                    elif "formula" in block_label and block_label != "formula_number":
                        text_prompt = "Formula Recognition:"
                        # 对于 formula，裁剪边距（参考 PaddleX 第 272 行）
                        try:
                            from ..paddleocr_vl.uilts import (
                                crop_margin,
                            )
                            block_img = crop_margin(block_img)
                        except ImportError:
                            # 如果无法导入，跳过裁剪
                            pass
                    
                    block_imgs.append(block_img)
                    text_prompts.append(text_prompt)
                    figure_token_maps.append(figure_token_map)
                    vlm_block_ids.append((i, j))
                    drop_figures_set.update(drop_figures)  # 参考 PaddleX 第 277 行
        
        # 打印 image 大小、label 和 query（参考 pipeline.py）
        for idx, (block_img, text_prompt, (i, j)) in enumerate(zip(block_imgs, text_prompts, vlm_block_ids)):
            block_label = blocks[i][j]["label"]
            if hasattr(block_img, 'shape'):
                img_size = block_img.shape
            elif hasattr(block_img, 'size'):
                img_size = block_img.size
            else:
                img_size = "unknown"
            # print(f"[VLM Input {idx}] Image size: {img_size}, Label: {block_label}, Query: {text_prompt}")
        
        # VLM 推理
        if block_imgs:
            vl_rec_results = self._vlm_predict(
                block_imgs,
                text_prompts,
                max_new_tokens=max_new_tokens,
            )
        else:
            vl_rec_results = []
        
        # 组装解析结果
        parsing_res_lists = []
        table_res_lists = []
        curr_vlm_block_idx = 0
        
        for i, blocks_for_img in enumerate(blocks):
            parsing_res_list = []
            table_res_list = []
            
            for j, block in enumerate(blocks_for_img):
                block_img = block["img"]
                block_bbox = block["box"]
                block_label = block["label"]
                block_content = ""
                
                if curr_vlm_block_idx < len(vlm_block_ids) and vlm_block_ids[curr_vlm_block_idx] == (i, j):
                    vl_rec_result = vl_rec_results[curr_vlm_block_idx]
                    figure_token_map = figure_token_maps[curr_vlm_block_idx]
                    block_img4vl = block_imgs[curr_vlm_block_idx]
                    curr_vlm_block_idx += 1
                    vl_rec_result["image"] = block_img4vl  # 参考 PaddleX 第 333 行
                    result_str = vl_rec_result.get("result", "")
                    if result_str is None:
                        result_str = ""
                    
                    # 处理重复内容（参考 PaddleX 第 337 行）
                    try:
                        from ..paddleocr_vl.uilts import (
                            truncate_repetitive_content,
                        )
                        result_str = truncate_repetitive_content(result_str)
                    except ImportError:
                        pass
                    
                    # 处理公式格式（参考 PaddleX 第 338-350 行）
                    if ("\\(" in result_str and "\\)" in result_str) or (
                        "\\[" in result_str and "\\]" in result_str
                    ):
                        result_str = result_str.replace("$", "")
                        result_str = (
                            result_str.replace("\\(", " $ ")
                            .replace("\\)", " $ ")
                            .replace("\\[", " $$ ")
                            .replace("\\]", " $$ ")
                        )
                        if block_label == "formula_number":
                            result_str = result_str.replace("$", "")
                    
                    # 修复 LaTeX 语法错误（修复 \inS, \inR 等常见错误）
                    result_str = fix_latex_syntax(result_str)
                    
                    # 处理表格（参考 PaddleX 第 351-357 行）
                    if block_label == "table":
                        try:
                            from ..paddleocr_vl.uilts import (
                                convert_otsl_to_html,
                                untokenize_figure_of_table,
                            )
                            html_str = convert_otsl_to_html(result_str)
                            if html_str != "":
                                result_str = html_str
                            result_str = untokenize_figure_of_table(
                                result_str, figure_token_map
                            )
                        except ImportError:
                            pass
                    
                    block_content = result_str
                
                block_info = PaddleOCRVLBlock(
                    label=block_label,
                    bbox=block_bbox,
                    content=block_content,
                    group_id=block.get("group_id", None),
                )
                
                # 设置图片信息（参考 PaddleX 的实现，第 367-379 行）
                # 当 block_label 在 image_labels 中且 block_img 不为 None 时，设置 block_info.image
                image_labels = ["image", "header_image", "footer_image", "seal"]
                if not self.use_chart_recognition:
                    image_labels.append("chart")
                
                if block_label in image_labels and block_img is not None:
                    x_min, y_min, x_max, y_max = list(map(int, block_bbox))
                    img_path = f"imgs/img_in_{block_label}_box_{x_min}_{y_min}_{x_max}_{y_max}.jpg"
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
            
            parsing_res_lists.append(parsing_res_list)
            table_res_lists.append(table_res_list)
        
        return parsing_res_lists, table_res_lists, imgs_in_doc
    
    def _vlm_predict(
        self,
        block_imgs: List[np.ndarray],
        text_prompts: List[str],
        max_new_tokens: int = 4096,
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
    
    def _crop_by_boxes(self, image: np.ndarray, boxes: List[dict]) -> List[dict]:
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
        
        for box in boxes:
            coordinate = box["coordinate"]
            xmin, ymin, xmax, ymax = map(int, coordinate)
            
            # 确保坐标在图像范围内
            xmin = max(0, min(xmin, w))
            ymin = max(0, min(ymin, h))
            xmax = max(xmin, min(xmax, w))
            ymax = max(ymin, min(ymax, h))
            
            if xmax > xmin and ymax > ymin:
                cropped = image[ymin:ymax, xmin:xmax].copy()
                blocks.append({
                    "img": cropped,
                    "label": box["label"],
                    "box": [xmin, ymin, xmax, ymax],
                    "score": box["score"],
                })
        
        return blocks
    
    def _filter_overlap_boxes(self, layout_det_res: dict) -> dict:
        """
        过滤重叠框（完整实现，与 PaddleX 功能一致）
        
        Args:
            layout_det_res: 布局检测结果
        
        Returns:
            dict: 过滤后的布局检测结果
        """
        from copy import deepcopy
        
        # 辅助函数：计算边界框面积
        def calculate_bbox_area(bbox):
            x1, y1, x2, y2 = map(float, bbox)
            area = abs((x2 - x1) * (y2 - y1))
            return area
        
        # 辅助函数：计算重叠比例（使用 small 模式）
        def calculate_overlap_ratio(bbox1, bbox2):
            x_min_inter = max(bbox1[0], bbox2[0])
            y_min_inter = max(bbox1[1], bbox2[1])
            x_max_inter = min(bbox1[2], bbox2[2])
            y_max_inter = min(bbox1[3], bbox2[3])
            inter_width = max(0, x_max_inter - x_min_inter)
            inter_height = max(0, y_max_inter - y_min_inter)
            inter_area = inter_width * inter_height
            bbox1_area = calculate_bbox_area(bbox1)
            bbox2_area = calculate_bbox_area(bbox2)
            # 使用 small 模式：取两个框面积的最小值作为参考
            ref_area = min(bbox1_area, bbox2_area)
            return inter_area / ref_area if ref_area > 0 else 0.0
        
        layout_det_res_filtered = deepcopy(layout_det_res)
        
        # 排除 reference 标签的框
        boxes = [
            box for box in layout_det_res_filtered["boxes"] if box["label"] != "reference"
        ]
        dropped_indexes = set()
        
        # 遍历所有框对，检查重叠
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if i in dropped_indexes or j in dropped_indexes:
                    continue
                
                overlap_ratio = calculate_overlap_ratio(
                    boxes[i]["coordinate"], boxes[j]["coordinate"]
                )
                
                # 如果重叠比例 > 0.7，需要处理
                if overlap_ratio > 0.7:
                    box_area_i = calculate_bbox_area(boxes[i]["coordinate"])
                    box_area_j = calculate_bbox_area(boxes[j]["coordinate"])
                    
                    # 特殊情况：如果一个是 image 标签，另一个不是，则跳过
                    if (
                        (boxes[i]["label"] == "image" or boxes[j]["label"] == "image")
                        and boxes[i]["label"] != boxes[j]["label"]
                    ):
                        continue
                    
                    # 保留面积较大的框，丢弃面积较小的框
                    if box_area_i >= box_area_j:
                        dropped_indexes.add(j)
                    else:
                        dropped_indexes.add(i)
        
        # 过滤掉被标记为丢弃的框
        layout_det_res_filtered["boxes"] = [
            box for idx, box in enumerate(boxes) if idx not in dropped_indexes
        ]
        
        return layout_det_res_filtered
    
    def _merge_blocks(self, blocks: List[dict], non_merge_labels: List[str]) -> List[dict]:
        """
        合并布局块
        参考 PaddleX 的 merge_blocks 实现，确保功能完全一致
        
        Args:
            blocks: 图像块列表
            non_merge_labels: 不合并的标签列表
        
        Returns:
            list: 合并后的图像块列表
        """
        # 分离需要合并和不需要合并的块
        blocks_to_merge = []
        non_merge_blocks = {}
        for idx, block in enumerate(blocks):
            if block["label"] in non_merge_labels:
                non_merge_blocks[idx] = block
            else:
                blocks_to_merge.append((idx, block))

        merged_groups = []
        current_group = []
        current_indices = []
        current_aligns = []

        def is_aligned(a1, a2):
            return abs(a1 - a2) <= 5

        def get_alignment(block_bbox, prev_bbox):
            if is_aligned(block_bbox[0], prev_bbox[0]):
                return "left"
            elif is_aligned(block_bbox[2], prev_bbox[2]):
                return "right"
            else:
                return "center"

        def overlapwith_other_box(block_idx, prev_idx, blocks):
            prev_bbox = blocks[prev_idx]["box"]
            block_bbox = blocks[block_idx]["box"]
            x1 = min(prev_bbox[0], block_bbox[0])
            y1 = min(prev_bbox[1], block_bbox[1])
            x2 = max(prev_bbox[2], block_bbox[2])
            y2 = max(prev_bbox[3], block_bbox[3])
            min_box = [x1, y1, x2, y2]
            for idx, other_block in enumerate(blocks):
                if idx in [block_idx, prev_idx]:
                    continue
                other_bbox = other_block["box"]
                if self._calculate_overlap_ratio(min_box, other_bbox) > 0:
                    return True
            return False

        for i, (idx, block) in enumerate(blocks_to_merge):
            if not current_group:
                current_group = [block]
                current_indices = [idx]
                current_aligns = []
                continue

            prev_idx, prev_block = blocks_to_merge[i - 1]
            prev_bbox = prev_block["box"]
            prev_label = prev_block["label"]
            block_bbox = block["box"]
            block_label = block["label"]

            iou_h = self._calculate_projection_overlap_ratio(block_bbox, prev_bbox, "horizontal")
            is_cross = (
                iou_h == 0
                and block_label == "text"
                and block_label == prev_label
                and block_bbox[0] > prev_bbox[2]
                and block_bbox[1] < prev_bbox[3]
                and block_bbox[0] - prev_bbox[2]
                < max(prev_bbox[2] - prev_bbox[0], block_bbox[2] - block_bbox[0]) * 0.3
            )
            is_updown_align = (
                iou_h > 0
                and block_label in ["text"]
                and block_label == prev_label
                and block_bbox[3] >= prev_bbox[1]
                and abs(block_bbox[1] - prev_bbox[3])
                < max(prev_bbox[3] - prev_bbox[1], block_bbox[3] - block_bbox[1]) * 0.5
                and (
                    is_aligned(block_bbox[0], prev_bbox[0])
                    ^ is_aligned(block_bbox[2], prev_bbox[2])
                )
                and overlapwith_other_box(idx, prev_idx, blocks)
            )
            if is_cross:
                align_mode = "center"
            elif is_updown_align:
                align_mode = get_alignment(block_bbox, prev_bbox)
            else:
                align_mode = None

            if is_cross or is_updown_align:
                current_group.append(block)
                current_indices.append(idx)
                current_aligns.append(align_mode)
            else:
                merged_groups.append((current_indices, current_aligns))
                current_group = [block]
                current_indices = [idx]
                current_aligns = []
        if current_group:
            merged_groups.append((current_indices, current_aligns))

        group_ranges = []
        for group_indices, aligns in merged_groups:
            start, end = min(group_indices), max(group_indices)
            group_ranges.append((start, end, group_indices, aligns))

        result_blocks = []
        used_indices = set()
        idx = 0
        while idx < len(blocks):
            group_found = False
            for start, end, group_indices, aligns in group_ranges:
                if idx == start and all(i not in used_indices for i in group_indices):
                    group_found = True
                    imgs = [blocks[i]["img"] for i in group_indices]
                    merge_aligns = aligns if aligns else []
                    w, h = self._calc_merged_wh(imgs)
                    aspect_ratio = h / w if w != 0 else float("inf")
                    if aspect_ratio >= 3:
                        for j, block_idx in enumerate(group_indices):
                            block = blocks[block_idx].copy()
                            block["img"] = blocks[block_idx]["img"]
                            block["merge_aligns"] = None
                            result_blocks.append(block)
                            used_indices.add(block_idx)
                    else:
                        merged_img = self._merge_images(imgs, merge_aligns)
                        for j, block_idx in enumerate(group_indices):
                            block = blocks[block_idx].copy()
                            block["img"] = merged_img if j == 0 else None
                            block["merge_aligns"] = merge_aligns if j == 0 else None
                            block["group_id"] = group_indices[0]
                            result_blocks.append(block)
                            used_indices.add(block_idx)
                    insert_list = []
                    for n_idx in range(start + 1, end):
                        if n_idx in non_merge_blocks:
                            insert_list.append(n_idx)
                    for n_idx in insert_list:
                        result_blocks.append(non_merge_blocks[n_idx])
                        used_indices.add(n_idx)
                    idx = end + 1
                    break
            if group_found:
                continue
            if idx in non_merge_blocks and idx not in used_indices:
                result_blocks.append(non_merge_blocks[idx])
                used_indices.add(idx)
            idx += 1
        return result_blocks
    
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

