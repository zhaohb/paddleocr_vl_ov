"""
Lightweight PaddleOCR-VL utilities (vendored from PaddleX)

Why this file exists:
- `paddleocr_vl_openvino/paddleocr_vl_pipeline/ov_paddleocr_vl_pipeline.py` originally imported
  helper functions from `paddlex.inference.pipelines.paddleocr_vl.uilts`.
- When packaging to exe, that optional dependency may be missing or partially collected, which makes
  table outputs degrade to raw OTSL tags (e.g. `<fcel>`, `<nl>`) and split image blocks.

This module vendors only the minimal set of utilities that PaddleOCR-VL pipeline uses:
- `tokenize_figure_of_table`, `untokenize_figure_of_table`
- `convert_otsl_to_html` (+ required OTSL helpers)
- `crop_margin`
- `truncate_repetitive_content` (+ required repetition helpers)

The implementations are copied from PaddleX's `uilts.py` with minimal modifications:
- Removed PaddleX-only imports and pydantic models; replaced with dataclasses.
- Kept behavior consistent to ensure Python run and packaged exe outputs match.
"""

from __future__ import annotations

import html
import itertools
import math
import re
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional

import numpy as np

try:
    # Optional dependency; only used by merge_images helpers.
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


# ---------------- Geometry helpers (aligned to PaddleX doc_vl) ----------------


def calculate_bbox_area(bbox: Union[List[float], Tuple[float, float, float, float], np.ndarray]) -> float:
    x1, y1, x2, y2 = map(float, bbox)
    return abs((x2 - x1) * (y2 - y1))


def calculate_overlap_ratio(
    bbox1: Union[List[float], Tuple[float, float, float, float], np.ndarray],
    bbox2: Union[List[float], Tuple[float, float, float, float], np.ndarray],
    mode: str = "union",
) -> float:
    x1a, y1a, x2a, y2a = map(float, bbox1)
    x1b, y1b, x2b, y2b = map(float, bbox2)

    xi1 = max(x1a, x1b)
    yi1 = max(y1a, y1b)
    xi2 = min(x2a, x2b)
    yi2 = min(y2a, y2b)

    inter_w = max(0.0, xi2 - xi1)
    inter_h = max(0.0, yi2 - yi1)
    inter = inter_w * inter_h

    area_a = max(0.0, (x2a - x1a)) * max(0.0, (y2a - y1a))
    area_b = max(0.0, (x2b - x1b)) * max(0.0, (y2b - y1b))

    if mode == "union":
        denom = area_a + area_b - inter
    elif mode == "small":
        denom = min(area_a, area_b)
    elif mode == "large":
        denom = max(area_a, area_b)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    return float(inter / denom) if denom > 0 else 0.0


def calculate_projection_overlap_ratio(bbox1, bbox2, axis: str = "horizontal") -> float:
    """
    Calculate 1D overlap ratio on horizontal/vertical projections.
    Used by merge_blocks heuristic in PaddleX doc_vl.
    """
    x1a, y1a, x2a, y2a = map(float, bbox1)
    x1b, y1b, x2b, y2b = map(float, bbox2)

    if axis == "horizontal":
        a1, a2 = x1a, x2a
        b1, b2 = x1b, x2b
    elif axis == "vertical":
        a1, a2 = y1a, y2a
        b1, b2 = y1b, y2b
    else:
        raise ValueError("axis must be 'horizontal' or 'vertical'")

    inter = max(0.0, min(a2, b2) - max(a1, b1))
    denom = min(max(0.0, a2 - a1), max(0.0, b2 - b1))
    return float(inter / denom) if denom > 0 else 0.0


def make_valid(poly):
    # Keep behavior aligned to PaddleX: fix invalid polygons by buffering.
    if not poly.is_valid:
        poly = poly.buffer(0)
    return poly


def calculate_polygon_overlap_ratio(
    polygon1: List[Tuple[int, int]],
    polygon2: List[Tuple[int, int]],
    mode: str = "union",
) -> Optional[float]:
    """
    PaddleX `uilts.py` uses shapely here.
    为了避免强依赖（尤其是打包/精简环境），shapely 作为可选依赖：
    - 可用：返回 [0,1] overlap ratio
    - 不可用：返回 None，让调用方退化为 bbox overlap 逻辑（不要返回 0.0，避免误判）
    """
    try:
        from shapely.geometry import Polygon  # type: ignore
    except Exception:
        return None

    poly1 = make_valid(Polygon(polygon1))
    poly2 = make_valid(Polygon(polygon2))
    intersection = poly1.intersection(poly2).area
    union = poly1.union(poly2).area
    if mode == "union":
        denom = union
    elif mode == "small":
        denom = min(poly1.area, poly2.area)
    elif mode == "large":
        denom = max(poly1.area, poly2.area)
    else:
        raise ValueError(f"Unknown mode: {mode}")
    return float(intersection / denom) if denom > 0 else 0.0


def filter_overlap_boxes(
    layout_det_res: Dict[str, List[Dict]], layout_shape_mode: str
) -> Dict[str, List[Dict]]:
    """
    对齐 PaddleX doc_vl 版本 `uilts.py::filter_overlap_boxes`：
    - 删除重叠框（按 small-overlap/bbox area）
    - 对 inline_formula 做特殊处理
    - 当 layout_shape_mode != rect 时，优先用 polygon overlap（二者差异较大时不删除）
    """
    layout_det_res_filtered = deepcopy(layout_det_res)
    boxes = [box for box in layout_det_res_filtered["boxes"] if box["label"] != "reference"]
    dropped_indexes = set()

    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]["coordinate"]
        w, h = x2 - x1, y2 - y1
        if w < 6 or h < 6:
            dropped_indexes.add(i)
        for j in range(i + 1, len(boxes)):
            if i in dropped_indexes or j in dropped_indexes:
                continue
            overlap_ratio = calculate_overlap_ratio(boxes[i]["coordinate"], boxes[j]["coordinate"], "small")
            if boxes[i]["label"] == "inline_formula" or boxes[j]["label"] == "inline_formula":
                if overlap_ratio > 0.5:
                    if boxes[i]["label"] == "inline_formula":
                        dropped_indexes.add(i)
                    if boxes[j]["label"] == "inline_formula":
                        dropped_indexes.add(j)
                    continue
            if overlap_ratio > 0.7:
                if layout_shape_mode != "rect" and "polygon_points" in boxes[i]:
                    poly_overlap_ratio = calculate_polygon_overlap_ratio(
                        boxes[i]["polygon_points"], boxes[j]["polygon_points"], "small"
                    )
                    if poly_overlap_ratio is not None and poly_overlap_ratio < 0.7:
                        continue
                box_area_i = calculate_bbox_area(boxes[i]["coordinate"])
                box_area_j = calculate_bbox_area(boxes[j]["coordinate"])
                if {boxes[i]["label"], boxes[j]["label"]} & {"image", "table", "seal", "chart"} and boxes[i]["label"] != boxes[j]["label"]:
                    continue
                if box_area_i >= box_area_j:
                    dropped_indexes.add(j)
                else:
                    dropped_indexes.add(i)

    layout_det_res_filtered["boxes"] = [box for idx, box in enumerate(boxes) if idx not in dropped_indexes]
    return layout_det_res_filtered


def to_pil_image(img):
    if Image is None:
        raise ImportError("PIL is required for merge_images")
    if isinstance(img, Image.Image):
        return img
    return Image.fromarray(img)


def to_np_array(img):
    if Image is not None and isinstance(img, Image.Image):
        return np.array(img)
    return img


def calc_merged_wh(images):
    widths = [to_pil_image(img).width for img in images]
    heights = [to_pil_image(img).height for img in images]
    return max(widths), sum(heights)


def merge_images(images, aligns="center", layout_shape_mode="auto"):
    """
    PaddleX doc_vl: merge images vertically with alignment.
    """
    if not images:
        return None
    if len(images) == 1:
        return to_np_array(images[0])
    if isinstance(aligns, str):
        aligns = [aligns] * (len(images) - 1)
    if len(aligns) != len(images) - 1:
        raise ValueError("The length of aligns must be len(images) - 1")

    merged = to_pil_image(images[0])
    for i in range(1, len(images)):
        img2 = to_pil_image(images[i])
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
        else:
            x1 = x2 = 0
        new_img.paste(merged, (x1, 0))
        new_img.paste(img2, (x2, merged.height))
        merged = new_img
    return to_np_array(merged)


def merge_blocks(blocks, non_merge_labels, layout_shape_mode="auto"):
    """
    PaddleX doc_vl: merge text blocks into groups by cross/up-down alignment heuristics.
    """
    blocks_to_merge = []
    non_merge_blocks = {}
    for idx, block in enumerate(blocks):
        if block["label"] in non_merge_labels:
            non_merge_blocks[idx] = block
        else:
            blocks_to_merge.append((idx, block))

    merged_groups = []
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

    def overlapwith_other_box(block_idx, prev_idx, all_blocks):
        prev_bbox = all_blocks[prev_idx]["box"]
        block_bbox = all_blocks[block_idx]["box"]
        x1 = min(prev_bbox[0], block_bbox[0])
        y1 = min(prev_bbox[1], block_bbox[1])
        x2 = max(prev_bbox[2], block_bbox[2])
        y2 = max(prev_bbox[3], block_bbox[3])
        min_box = [x1, y1, x2, y2]
        for k, other_block in enumerate(all_blocks):
            if k in [block_idx, prev_idx] or other_block["label"] not in non_merge_labels:
                continue
            other_bbox = other_block["box"]
            if calculate_overlap_ratio(min_box, other_bbox) > 0:
                return True
        return False

    for i, (idx, block) in enumerate(blocks_to_merge):
        if not current_indices:
            current_indices = [idx]
            current_aligns = []
            continue

        prev_idx, prev_block = blocks_to_merge[i - 1]
        prev_bbox = prev_block["box"]
        prev_label = prev_block["label"]
        block_bbox = block["box"]
        block_label = block["label"]

        iou_h = calculate_projection_overlap_ratio(block_bbox, prev_bbox, "horizontal")
        is_cross = (
            iou_h == 0
            and block_label == "text"
            and block_label == prev_label
            and block_bbox[0] > prev_bbox[2]
            and block_bbox[1] < prev_bbox[3]
            and (block_bbox[0] - prev_bbox[2])
            < max(prev_bbox[2] - prev_bbox[0], block_bbox[2] - block_bbox[0]) * 0.3
        )
        is_updown_align = (
            iou_h > 0
            and block_label in ["text"]
            and block_label == prev_label
            and block_bbox[3] >= prev_bbox[1]
            and abs(block_bbox[1] - prev_bbox[3])
            < max(prev_bbox[3] - prev_bbox[1], block_bbox[3] - block_bbox[1]) * 0.5
            and (is_aligned(block_bbox[0], prev_bbox[0]) ^ is_aligned(block_bbox[2], prev_bbox[2]))
            and overlapwith_other_box(idx, prev_idx, blocks)
        )

        if is_cross:
            align_mode = "center"
        elif is_updown_align:
            align_mode = get_alignment(block_bbox, prev_bbox)
        else:
            align_mode = None

        if is_cross or is_updown_align:
            current_indices.append(idx)
            current_aligns.append(align_mode)
        else:
            merged_groups.append((current_indices, current_aligns))
            current_indices = [idx]
            current_aligns = []

    if current_indices:
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
                w, h = calc_merged_wh(imgs)
                aspect_ratio = h / w if w != 0 else float("inf")
                if aspect_ratio >= 3:
                    for block_idx in group_indices:
                        block = blocks[block_idx].copy()
                        block["merge_aligns"] = None
                        result_blocks.append(block)
                        used_indices.add(block_idx)
                else:
                    merged_img = merge_images(imgs, merge_aligns, layout_shape_mode)
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


# ---------------- Table figure tokenization ----------------


def paint_token(image: np.ndarray, box, token_str: str) -> np.ndarray:
    """
    Fill a rectangular area in the image with a white background and write the given token string.

    Args:
        image: Image to paint on (BGR).
        box: (x1, y1, x2, y2) coordinates of rectangle.
        token_str: Token string to write.
    """
    import cv2

    def get_optimal_font_scale(text, font_face, square_size, fill_ratio=0.9):
        # The scale is greater than 0.2 and less than 10.
        left, right = 0.2, 10
        optimal_scale = left
        w = h = 0
        while right - left > 1e-2:
            mid = (left + right) / 2
            (w, h), _ = cv2.getTextSize(text, font_face, mid, thickness=1)
            if w < square_size * fill_ratio and h < square_size * fill_ratio:
                optimal_scale = mid
                left = mid
            else:
                right = mid
        return optimal_scale, w, h

    x1, y1, x2, y2 = [int(v) for v in box]
    box_w = x2 - x1
    box_h = y2 - y1

    img = image.copy()
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=-1)

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness_scale_ratio = 4
    font_scale, text_w, text_h = get_optimal_font_scale(token_str, font, min(box_w, box_h), fill_ratio=0.9)
    font_thickness = max(1, math.floor(font_scale * thickness_scale_ratio))

    text_x = x1 + (box_w - text_w) // 2
    text_y = y1 + (box_h + text_h) // 2

    cv2.putText(
        img,
        token_str,
        (text_x, text_y),
        font,
        font_scale,
        (0, 0, 0),
        font_thickness,
        lineType=cv2.LINE_AA,
    )
    return img


def tokenize_figure_of_table(table_block_img: np.ndarray, table_box, figures: List[Dict]) -> Tuple[np.ndarray, Dict[str, str], List[str]]:
    """
    Replace figures in a table area with tokens, return new image and token map.

    Args:
        table_block_img: Table image (BGR).
        table_box: Table bounding box [x_min, y_min, x_max, y_max].
        figures: List of figure dicts (must contain 'coordinate', 'path').

    Returns:
        - New table image
        - Token-to-img HTML map
        - List of figure paths dropped
    """

    def gen_random_map(num: int) -> List[int]:
        exclude_digits = {"0", "1", "9"}
        seq: List[int] = []
        i = 0
        while len(seq) < num:
            if not (set(str(i)) & exclude_digits):
                seq.append(i)
            i += 1
        return seq

    import random

    random.seed(1024)
    token_map: Dict[str, str] = {}
    table_x_min, table_y_min, table_x_max, table_y_max = table_box
    drop_idxes: List[int] = []
    random_map = gen_random_map(len(figures))
    random.shuffle(random_map)
    for figure_id, figure in enumerate(figures):
        figure_x_min, figure_y_min, figure_x_max, figure_y_max = figure["coordinate"]
        if (
            figure_x_min >= table_x_min
            and figure_y_min >= table_y_min
            and figure_x_max <= table_x_max
            and figure_y_max <= table_y_max
        ):
            drop_idxes.append(figure_id)
            # The figure is too small to be tokenized/recognized when shortest length < 25.
            if min(figure_x_max - figure_x_min, figure_y_max - figure_y_min) < 25:
                continue
            draw_box = [
                figure_x_min - table_x_min,
                figure_y_min - table_y_min,
                figure_x_max - table_x_min,
                figure_y_max - table_y_min,
            ]
            token_str = "[F" + str(random_map[figure_id]) + "]"
            table_block_img = paint_token(table_block_img, draw_box, token_str)
            # Align with PaddleX doc_vl: store path (not html tag). HTML is built in untokenize.
            token_map[token_str] = figure["path"]
    drop_figures = [f["path"] for i, f in enumerate(figures) if i in drop_idxes]
    return table_block_img, token_map, drop_figures


def untokenize_figure_of_table(
    table_res_str: str,
    figure_token_map: Dict[str, str],
    image_path_to_obj_map: Optional[Dict[str, object]] = None,
) -> str:
    """
    Replace tokens in a string with their HTML image equivalents.
    """

    def repl(match):
        token_id = match.group(1)
        token = f"[F{token_id}]"
        img_path = figure_token_map.get(token, None)
        if not img_path:
            return match.group(0)

        img_block = image_path_to_obj_map.get(img_path) if image_path_to_obj_map else None
        image_info = '<img src="{}" alt="Image"" />'.format(img_path.replace("-\n", "").replace("\n", " "))
        if img_block is not None:
            ocr_content = getattr(img_block, "content", "")
            if ocr_content:
                image_info += "\n\n" + str(ocr_content) + "\n\n"
        return image_info

    pattern = r"\[F(\d+)\]"
    return re.sub(pattern, repl, table_res_str)


# ---------------- Formula crop margin ----------------


def crop_margin(img: np.ndarray) -> np.ndarray:
    import cv2

    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    max_val = gray.max()
    min_val = gray.min()

    if max_val == min_val:
        return img

    data = (gray - min_val) / (max_val - min_val) * 255
    data = data.astype(np.uint8)

    _, binary = cv2.threshold(data, 200, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(binary)

    if coords is None:
        return img

    x, y, w, h = cv2.boundingRect(coords)
    cropped = img[y : y + h, x : x + w]

    return cropped


# ---------------- OTSL -> HTML (table) ----------------


@dataclass
class TableCell:
    row_span: int = 1
    col_span: int = 1
    start_row_offset_idx: int = 0
    end_row_offset_idx: int = 1
    start_col_offset_idx: int = 0
    end_col_offset_idx: int = 1
    text: str = ""
    column_header: bool = False
    row_header: bool = False
    row_section: bool = False


@dataclass
class TableData:
    table_cells: List[TableCell]
    num_rows: int
    num_cols: int

    @property
    def grid(self) -> List[List[TableCell]]:
        table_data: List[List[TableCell]] = [
            [
                TableCell(
                    text="",
                    start_row_offset_idx=i,
                    end_row_offset_idx=i + 1,
                    start_col_offset_idx=j,
                    end_col_offset_idx=j + 1,
                )
                for j in range(self.num_cols)
            ]
            for i in range(self.num_rows)
        ]
        for cell in self.table_cells:
            for i in range(min(cell.start_row_offset_idx, self.num_rows), min(cell.end_row_offset_idx, self.num_rows)):
                for j in range(min(cell.start_col_offset_idx, self.num_cols), min(cell.end_col_offset_idx, self.num_cols)):
                    table_data[i][j] = cell
        return table_data


# OTSL tag constants
OTSL_NL = "<nl>"
OTSL_FCEL = "<fcel>"
OTSL_ECEL = "<ecel>"
OTSL_LCEL = "<lcel>"
OTSL_UCEL = "<ucel>"
OTSL_XCEL = "<xcel>"

NON_CAPTURING_TAG_GROUP = "(?:<fcel>|<ecel>|<nl>|<lcel>|<ucel>|<xcel>)"
OTSL_FIND_PATTERN = re.compile(f"{NON_CAPTURING_TAG_GROUP}.*?(?={NON_CAPTURING_TAG_GROUP}|$)", flags=re.DOTALL)


def otsl_extract_tokens_and_text(s: str):
    pattern = r"(" + r"|".join([OTSL_NL, OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]) + r")"
    tokens = re.findall(pattern, s)
    text_parts = re.split(pattern, s)
    text_parts = [token for token in text_parts if token.strip()]
    return tokens, text_parts


def otsl_parse_texts(texts, tokens):
    split_word = OTSL_NL
    split_row_tokens = [list(y) for x, y in itertools.groupby(tokens, lambda z: z == split_word) if not x]
    table_cells: List[TableCell] = []
    r_idx = 0
    c_idx = 0

    # Ensure matrix completeness
    if split_row_tokens:
        max_cols = max(len(row) for row in split_row_tokens)
        for row in split_row_tokens:
            while len(row) < max_cols:
                row.append(OTSL_ECEL)
        new_texts = []
        text_idx = 0
        for row in split_row_tokens:
            for token in row:
                new_texts.append(token)
                if text_idx < len(texts) and texts[text_idx] == token:
                    text_idx += 1
                    if text_idx < len(texts) and texts[text_idx] not in [
                        OTSL_NL,
                        OTSL_FCEL,
                        OTSL_ECEL,
                        OTSL_LCEL,
                        OTSL_UCEL,
                        OTSL_XCEL,
                    ]:
                        new_texts.append(texts[text_idx])
                        text_idx += 1
            new_texts.append(OTSL_NL)
            if text_idx < len(texts) and texts[text_idx] == OTSL_NL:
                text_idx += 1
        texts = new_texts

    def count_right(tokens_, c_idx_, r_idx_, which_tokens):
        span = 0
        c_idx_iter = c_idx_
        while tokens_[r_idx_][c_idx_iter] in which_tokens:
            c_idx_iter += 1
            span += 1
            if c_idx_iter >= len(tokens_[r_idx_]):
                return span
        return span

    def count_down(tokens_, c_idx_, r_idx_, which_tokens):
        span = 0
        r_idx_iter = r_idx_
        while tokens_[r_idx_iter][c_idx_] in which_tokens:
            r_idx_iter += 1
            span += 1
            if r_idx_iter >= len(tokens_):
                return span
        return span

    for i, text in enumerate(texts):
        cell_text = ""
        if text in [OTSL_FCEL, OTSL_ECEL]:
            row_span = 1
            col_span = 1
            right_offset = 1
            if text != OTSL_ECEL:
                cell_text = texts[i + 1]
                right_offset = 2

            next_right_cell = texts[i + right_offset] if i + right_offset < len(texts) else ""
            next_bottom_cell = ""
            if r_idx + 1 < len(split_row_tokens):
                if c_idx < len(split_row_tokens[r_idx + 1]):
                    next_bottom_cell = split_row_tokens[r_idx + 1][c_idx]

            if next_right_cell in [OTSL_LCEL, OTSL_XCEL]:
                col_span += count_right(split_row_tokens, c_idx + 1, r_idx, [OTSL_LCEL, OTSL_XCEL])
            if next_bottom_cell in [OTSL_UCEL, OTSL_XCEL]:
                row_span += count_down(split_row_tokens, c_idx, r_idx + 1, [OTSL_UCEL, OTSL_XCEL])

            table_cells.append(
                TableCell(
                    text=cell_text.strip(),
                    row_span=row_span,
                    col_span=col_span,
                    start_row_offset_idx=r_idx,
                    end_row_offset_idx=r_idx + row_span,
                    start_col_offset_idx=c_idx,
                    end_col_offset_idx=c_idx + col_span,
                )
            )
        if text in [OTSL_FCEL, OTSL_ECEL, OTSL_LCEL, OTSL_UCEL, OTSL_XCEL]:
            c_idx += 1
        if text == OTSL_NL:
            r_idx += 1
            c_idx = 0
    return table_cells, split_row_tokens


def export_to_html(table_data: TableData) -> str:
    nrows = table_data.num_rows
    ncols = table_data.num_cols
    if len(table_data.table_cells) == 0:
        return ""
    body = ""
    grid = table_data.grid
    for i in range(nrows):
        body += "<tr>"
        for j in range(ncols):
            cell = grid[i][j]
            rowspan, rowstart = (cell.row_span, cell.start_row_offset_idx)
            colspan, colstart = (cell.col_span, cell.start_col_offset_idx)
            if rowstart != i or colstart != j:
                continue
            content = html.escape(cell.text.strip())
            celltag = "th" if cell.column_header else "td"
            opening_tag = f"{celltag}"
            if rowspan > 1:
                opening_tag += f' rowspan=\"{rowspan}\"'
            if colspan > 1:
                opening_tag += f' colspan=\"{colspan}\"'
            body += f"<{opening_tag}>{content}</{celltag}>"
        body += "</tr>"
    body = f"<table>{body}</table>"
    return body


def otsl_pad_to_sqr_v2(otsl_str: str) -> str:
    assert isinstance(otsl_str, str)
    otsl_str = otsl_str.strip()
    if OTSL_NL not in otsl_str:
        return otsl_str + OTSL_NL
    lines = otsl_str.split(OTSL_NL)
    row_data = []
    for line in lines:
        if not line:
            continue
        raw_cells = OTSL_FIND_PATTERN.findall(line)
        if not raw_cells:
            continue
        total_len = len(raw_cells)
        min_len = 0
        for i, cell_str in enumerate(raw_cells):
            if cell_str.startswith(OTSL_FCEL):
                min_len = i + 1
        row_data.append({"raw_cells": raw_cells, "total_len": total_len, "min_len": min_len})
    if not row_data:
        return OTSL_NL
    global_min_width = max(row["min_len"] for row in row_data) if row_data else 0
    max_total_len = max(row["total_len"] for row in row_data) if row_data else 0
    search_start = global_min_width
    search_end = max(global_min_width, max_total_len)
    min_total_cost = float("inf")
    optimal_width = search_end

    for width in range(search_start, search_end + 1):
        current_total_cost = sum(abs(row["total_len"] - width) for row in row_data)
        if current_total_cost < min_total_cost:
            min_total_cost = current_total_cost
            optimal_width = width

    repaired_lines = []
    for row in row_data:
        cells = row["raw_cells"]
        current_len = len(cells)
        if current_len > optimal_width:
            new_cells = cells[:optimal_width]
        else:
            padding = [OTSL_ECEL] * (optimal_width - current_len)
            new_cells = cells + padding
        repaired_lines.append("".join(new_cells))
    return OTSL_NL.join(repaired_lines) + OTSL_NL


def convert_otsl_to_html(otsl_content: str) -> str:
    """
    Convert OTSL-v1.0 string to HTML.
    Only 6 tags allowed: <fcel>, <ecel>, <nl>, <lcel>, <ucel>, <xcel>.
    """
    otsl_content = otsl_pad_to_sqr_v2(otsl_content)
    tokens, mixed_texts = otsl_extract_tokens_and_text(otsl_content)
    table_cells, split_row_tokens = otsl_parse_texts(mixed_texts, tokens)
    table_data = TableData(
        num_rows=len(split_row_tokens),
        num_cols=(max(len(row) for row in split_row_tokens) if split_row_tokens else 0),
        table_cells=table_cells,
    )
    return export_to_html(table_data)


# ---------------- Repetition truncation ----------------


def find_shortest_repeating_substring(s: str) -> Union[str, None]:
    n = len(s)
    for i in range(1, n // 2 + 1):
        if n % i == 0:
            substring = s[:i]
            if substring * (n // i) == s:
                return substring
    return None


def find_repeating_suffix(s: str, min_len: int = 8, min_repeats: int = 5) -> Union[Tuple[str, str, int], None]:
    for i in range(len(s) // (min_repeats), min_len - 1, -1):
        unit = s[-i:]
        if s.endswith(unit * min_repeats):
            count = 0
            temp_s = s
            while temp_s.endswith(unit):
                temp_s = temp_s[:-i]
                count += 1
            start_index = len(s) - (count * i)
            return s[:start_index], unit, count
    return None


def truncate_repetitive_content(
    content: str,
    line_threshold: int = 10,
    char_threshold: int = 10,
    min_len: int = 10,
    min_count: int = 3000,
) -> str:
    # Align with PaddleX: short strings skip truncation
    if len(content) < min_count:
        return content
    stripped_content = content.strip()
    if not stripped_content:
        return content

    # Priority 1: Phrase-level suffix repetition in long single lines.
    if "\n" not in stripped_content and len(stripped_content) > 100:
        suffix_match = find_repeating_suffix(stripped_content, min_len=8, min_repeats=5)
        if suffix_match:
            prefix, repeating_unit, count = suffix_match
            if len(repeating_unit) * count > len(stripped_content) * 0.5:
                return prefix

    # Priority 2: Full-string character-level repetition (e.g., 'ababab')
    if "\n" not in stripped_content and len(stripped_content) > min_len:
        repeating_unit = find_shortest_repeating_substring(stripped_content)
        if repeating_unit:
            count = len(stripped_content) // len(repeating_unit)
            if count >= char_threshold:
                return repeating_unit

    # Priority 3: Line-level repetition (e.g., same line repeated many times)
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    if not lines:
        return content
    total_lines = len(lines)
    if total_lines < line_threshold:
        return content
    line_counts = Counter(lines)
    most_common_line, count = line_counts.most_common(1)[0]
    if count >= line_threshold and (count / total_lines) >= 0.8:
        return most_common_line

    return content


ANNOT_TEXT_RE = re.compile(r"<\|TEXT_START\|>(.*?)<\|TEXT_END\|>", re.S)
LOC_BLOCK_RE = re.compile(r"<\|LOC_BEGIN\|>(.*?)<\|LOC_END\|>", re.S)
LOC_ITEM_RE = re.compile(r"<\|LOC_(\d+)\|>")
LOC_TOKEN_RE = re.compile(r"<\|LOC_(\d+)\|>")


def post_process_for_spotting(input_str: str, w: int, h: int) -> Tuple[str, Dict[str, List]]:
    """
    PaddleX: Post-process spotting output string to extract recognized texts and polygons.
    Returns (result_str, spotting_res).
    """
    assert isinstance(input_str, str)

    texts = ANNOT_TEXT_RE.findall(input_str)
    loc_blocks = LOC_BLOCK_RE.findall(input_str)

    rec_polys = []
    rec_texts = []

    n = min(len(texts), len(loc_blocks))
    for i in range(n):
        txt = texts[i].strip()
        loc_items = LOC_ITEM_RE.findall(loc_blocks[i])
        if len(loc_items) < 8:
            continue
        vals = list(map(int, loc_items[:8]))
        pts = [(vals[j], vals[j + 1]) for j in range(0, 8, 2)]
        pts = [(p[0] / 1000.0 * w, p[1] / 1000.0 * h) for p in pts]
        rec_polys.append(pts)
        rec_texts.append(txt)

    if not rec_polys or not rec_texts:
        matches = list(LOC_TOKEN_RE.finditer(input_str))
        last_end = 0
        i = 0
        while i + 7 < len(matches):
            group = matches[i : i + 8]
            vals = [int(m.group(1)) for m in group]
            pts = [(vals[j], vals[j + 1]) for j in range(0, 8, 2)]
            pts = [(p[0] / 1000.0 * w, p[1] / 1000.0 * h) for p in pts]
            text_span = input_str[last_end : group[0].start()]
            txt = text_span.strip()
            rec_texts.append(txt)
            rec_polys.append(pts)
            last_end = group[-1].end()
            i += 8

    result_str = "\n\n".join(rec_texts)
    spotting_res = {"rec_polys": rec_polys, "rec_texts": rec_texts}
    return result_str, spotting_res

