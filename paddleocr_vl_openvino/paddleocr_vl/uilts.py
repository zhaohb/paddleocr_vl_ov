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
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional

import numpy as np


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
            token_map[token_str] = f'<img src="{figure["path"]}" >'
    drop_figures = [f["path"] for i, f in enumerate(figures) if i in drop_idxes]
    return table_block_img, token_map, drop_figures


def untokenize_figure_of_table(table_res_str: str, figure_token_map: Dict[str, str]) -> str:
    """
    Replace tokens in a string with their HTML image equivalents.
    """

    def repl(match):
        token_id = match.group(1)
        token = f"[F{token_id}]"
        return figure_token_map.get(token, match.group(0))

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


def truncate_repetitive_content(content: str, line_threshold: int = 10, char_threshold: int = 10, min_len: int = 10) -> str:
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


