import openvino as ov
import cv2
import numpy as np
import json
import os
import argparse
from typing import Dict, List, Optional
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import mimetypes
import logging

# -----------------------------------------------------------------------------
# 直接导入子模块并直接调用
# - 作为包导入时：相对导入
# - 直接运行该脚本时：回退到绝对导入（修复 "attempted relative import with no known parent package"）
# -----------------------------------------------------------------------------
try:
    from . import processors, result, utils  # type: ignore
except ImportError:  # pragma: no cover
    import importlib
    import sys
    from pathlib import Path as _Path

    # 将 `.../pp_doclayoutv2` 的父目录加入 sys.path，使 `pp_doclayoutv2.*` 可被绝对导入
    _this_dir = _Path(__file__).resolve().parent
    _pkg_parent = str(_this_dir.parent)
    if _pkg_parent not in sys.path:
        sys.path.insert(0, _pkg_parent)

    processors = importlib.import_module("pp_doclayoutv2.processors")
    result = importlib.import_module("pp_doclayoutv2.result")
    utils = importlib.import_module("pp_doclayoutv2.utils")

# -----------------------------------------------------------------------------
# Backward-compat exports (after `result` is imported)
# -----------------------------------------------------------------------------
# Keep an alias for older imports (e.g. `from pp_doclayoutv2 import LayoutDetectionResult`).
LayoutDetectionResult = result.LayoutAnalysisResult

# Try importing modelscope, show warning if not installed
try:
    from modelscope import snapshot_download
    MODELSCOPE_AVAILABLE = True
except ImportError:
    MODELSCOPE_AVAILABLE = False
    logging.warning("modelscope not installed. Auto-download feature will be disabled. Install with: pip install modelscope")

# ModelScope model ID
LAYOUT_MODEL_ID = "zhaohb/PP-DocLayoutV2-ov"

def preprocess_image_doclayout(image, target_input_size=(800, 800)):
    """
    Preprocess image: BGR->RGB conversion, resize, normalization, HWC->CHW transformation.
    
    Args:
        image: Input image in BGR format (OpenCV format)
        target_input_size: Target input size (width, height), default (800, 800)
    
    Returns:
        tuple: (input_blob, scale_h, scale_w) where:
            - input_blob: Preprocessed image tensor [1, C, H, W]
            - scale_h: Height scaling factor
            - scale_w: Width scaling factor
    """
    orig_h, orig_w = image.shape[:2]
    target_w, target_h = target_input_size
    scale_h = target_h / orig_h
    scale_w = target_w / orig_w

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb_image, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
    input_blob = resized.astype(np.float32)
    input_blob = input_blob.transpose(2, 0, 1)[np.newaxis, ...]

    return input_blob, scale_h, scale_w

def center_to_corners_format(boxes):
    """
    Convert bounding boxes from center format (cx, cy, w, h) to corner format (xmin, ymin, xmax, ymax).
    
    Args:
        boxes: Bounding boxes in center format, shape can be [N, 4] or [N, M, 4]
    
    Returns:
        np.ndarray: Bounding boxes in corner format
    """
    if len(boxes.shape) == 2:
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    else:
        cx, cy, w, h = boxes[:, :, 0], boxes[:, :, 1], boxes[:, :, 2], boxes[:, :, 3]
    xmin, ymin = cx - w / 2.0, cy - h / 2.0
    xmax, ymax = cx + w / 2.0, cy + h / 2.0
    return np.stack([xmin, ymin, xmax, ymax], axis=-1)

def nms(boxes, iou_same=0.6, iou_diff=0.98):
    """兼容旧调用点：直接转调 utils.nms。"""
    return utils.nms(boxes, iou_same=iou_same, iou_diff=iou_diff)

def is_contained(box1, box2):
    """
    Check if box1 is contained within box2 (with 90% overlap threshold).
    
    Args:
        box1: First bounding box [class_id, score, x1, y1, x2, y2]
        box2: Second bounding box [class_id, score, x1, y1, x2, y2]
    
    Returns:
        bool: True if box1 is contained in box2, False otherwise
    """
    _, _, x1, y1, x2, y2 = box1
    _, _, x1_p, y1_p, x2_p, y2_p = box2
    box1_area = (x2 - x1) * (y2 - y1)
    xi1 = max(x1, x1_p)
    yi1 = max(y1, y1_p)
    xi2 = min(x2, x2_p)
    yi2 = min(y2, y2_p)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    intersect_area = inter_width * inter_height
    return (intersect_area / box1_area >= 0.9) if box1_area > 0 else False

def check_containment(boxes, formula_index=None, category_index=None, mode=None):
    """
    Check containment relationships between bounding boxes.
    
    Args:
        boxes: Array of bounding boxes [class_id, score, x1, y1, x2, y2, ...]
        formula_index: Optional formula class index to filter
        category_index: Optional category index for mode-based filtering
        mode: Optional mode ("large" or "small") for category-specific filtering
    
    Returns:
        tuple: (contains_other, contained_by_other) arrays indicating containment relationships
    """
    n = len(boxes)
    contains_other = np.zeros(n, dtype=int)
    contained_by_other = np.zeros(n, dtype=int)
    
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if formula_index is not None:
                if boxes[i][0] == formula_index and boxes[j][0] != formula_index:
                    continue
            if category_index is not None and mode is not None:
                if mode == "large" and boxes[j][0] == category_index:
                    if is_contained(boxes[i], boxes[j]):
                        contained_by_other[i] = 1
                        contains_other[j] = 1
                if mode == "small" and boxes[i][0] == category_index:
                    if is_contained(boxes[i], boxes[j]):
                        contained_by_other[i] = 1
                        contains_other[j] = 1
            else:
                if is_contained(boxes[i], boxes[j]):
                    contained_by_other[i] = 1
                    contains_other[j] = 1
    
    return contains_other, contained_by_other


def draw_box(img, boxes):
    """兼容旧调用点：直接转调 result.draw_box。"""
    return result.draw_box(img, boxes)

def _sort_boxes_reading_order_like_paddlex(boxes: List[Dict], same_line_y_thresh: int = 10) -> List[Dict]:
    """
    对齐 PaddleX/PP 系列常用的“阅读顺序”排序：
    - 先按左上角 (ymin, xmin) 排序
    - 若相邻框 y 差 < same_line_y_thresh 认为同一行，再按 xmin 做局部交换

    说明：
    - PaddleX 的 `update_order_index()` 只会按当前列表顺序编号，不会排序；
      PaddleOCR 的 LayoutDetection 输出通常已经按阅读顺序排列。
    - 这里用矩形框 `coordinate=[xmin,ymin,xmax,ymax]` 的 (xmin,ymin) 作为排序基准。
    """
    if not boxes:
        return boxes

    sorted_boxes = sorted(boxes, key=lambda b: (b.get("coordinate", [0, 0, 0, 0])[1], b.get("coordinate", [0, 0, 0, 0])[0]))
    _boxes = list(sorted_boxes)
    num_boxes = len(_boxes)
    for i in range(num_boxes - 1):
        for j in range(i, -1, -1):
            yj = _boxes[j].get("coordinate", [0, 0, 0, 0])[1]
            yj1 = _boxes[j + 1].get("coordinate", [0, 0, 0, 0])[1]
            xj = _boxes[j].get("coordinate", [0, 0, 0, 0])[0]
            xj1 = _boxes[j + 1].get("coordinate", [0, 0, 0, 0])[0]
            if abs(yj1 - yj) < same_line_y_thresh and xj1 < xj:
                _boxes[j], _boxes[j + 1] = _boxes[j + 1], _boxes[j]
            else:
                break
    return _boxes

def _sort_boxes_and_masks_by_reading_order(
    boxes_np: np.ndarray,
    masks: Optional[np.ndarray],
    same_line_y_thresh: int = 10,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """
    在进行 mask->polygon 的提取前，先把 boxes（以及对应 masks）按阅读顺序排序。

    为什么需要这个？
    - PaddleX 的 `extract_polygon_points_by_masks(..., layout_shape_mode="auto")` 内部会使用
      `pre_poly = polygon_points[-1]`（上一框的 polygon）参与几何判断。
    - 因此“框的遍历顺序”会影响 auto 模式下最终选择 polygon/quad/rect。
    - PaddleOCR / PaddleX 在不同后端上，原始输出框顺序可能不同（score/NMS实现差异等），
      导致 auto 分支结果不一致（表现为某些框 polygon_points 点数不同）。

    这里对齐 Paddle 常见阅读顺序：先按 (ymin, xmin) 排序，再对 y 差 < 阈值的相邻框做 x 校正。
    """
    if boxes_np is None or boxes_np.size == 0:
        return boxes_np, masks
    if boxes_np.ndim != 2 or boxes_np.shape[1] < 6:
        return boxes_np, masks

    # primary sort by ymin then xmin
    ymin = boxes_np[:, 3]
    xmin = boxes_np[:, 2]
    order = np.lexsort((xmin, ymin))

    # local swap for same-line boxes (y close) but x inverted
    order_list = list(map(int, order.tolist()))
    for i in range(len(order_list) - 1):
        for j in range(i, -1, -1):
            a = order_list[j]
            b = order_list[j + 1]
            if abs(float(boxes_np[b, 3]) - float(boxes_np[a, 3])) < same_line_y_thresh and float(boxes_np[b, 2]) < float(boxes_np[a, 2]):
                order_list[j], order_list[j + 1] = order_list[j + 1], order_list[j]
            else:
                break

    order = np.array(order_list, dtype=np.int64)
    boxes_sorted = boxes_np[order]
    if masks is None:
        return boxes_sorted, None
    masks_sorted = masks[order] if hasattr(masks, "__getitem__") and len(masks) >= len(order) else masks
    return boxes_sorted, masks_sorted

"""
结果类型使用本目录的 `result.LayoutAnalysisResult`（自包含实现），与 PaddleX 的 Result 行为保持一致：
- 内部字段：input_path/page_index/input_img/boxes
- 提供 save_to_img / save_to_json
"""

def postprocess_detections_paddle_nms(
    output,
    orig_h,
    orig_w,
    threshold=0.5,
    layout_nms=False,
    layout_unclip_ratio=None,
    layout_merge_bboxes_mode=None,
    layout_shape_mode: str = "auto",
    filter_overlap_boxes: bool = True,
    skip_order_labels: Optional[List[str]] = None,
):
    """
    Postprocess PaddleDetection-style exported outputs.

    Common output format from Paddle export (already NMS-ed in graph):
      - fetch_name_0: [Nmax, 6] or [Nmax, 7]
        If 7 cols: [image_id, class_id, score, x1, y1, x2, y2]
        If 6 cols: [class_id, score, x1, y1, x2, y2]
      - fetch_name_1: [1] int32, bbox_num (valid number of boxes)

    Returns:
      list[dict] in the same format as restructured_boxes()
    """
    if not isinstance(output, (list, tuple)) or len(output) < 1:
        raise ValueError("output must be a list/tuple with at least one output tensor")

    out0 = np.array(output[0])
    out1 = np.array(output[1]) if len(output) > 1 and output[1] is not None else None
    masks = np.array(output[2]) if len(output) > 2 and output[2] is not None else None

    if out0.ndim != 2 or out0.shape[1] < 6:
        raise ValueError(f"Unsupported Paddle NMS output shape: {out0.shape}")

    # valid count
    if out1 is not None and out1.size > 0:
        num = int(out1.reshape(-1)[0])
        num = max(0, min(num, out0.shape[0]))
    else:
        num = out0.shape[0]

    det = out0[:num]
    if masks is not None and masks.shape[0] >= num:
        masks = masks[:num]
    if det.size == 0:
        return []

    # -------- Decode NMS table robustly (column order differs across Paddle export versions) --------
    def _score_mapping(cls_col, score_col, coord_cols):
        # Heuristic score for a candidate mapping
        cls_col = cls_col.astype(np.float32)
        score_col = score_col.astype(np.float32)
        coord_cols = coord_cols.astype(np.float32)

        # class ids should be integer-like and within a reasonable range
        cls_int_like = np.mean(np.abs(cls_col - np.round(cls_col)) < 1e-3)
        cls_max = np.max(cls_col) if cls_col.size else 0.0
        cls_min = np.min(cls_col) if cls_col.size else 0.0

        # scores should mostly be in [0, 1.5]
        score_in_range = np.mean((score_col >= -0.01) & (score_col <= 1.5))
        score_std = float(np.std(score_col))

        # coords should be finite
        finite = np.mean(np.isfinite(coord_cols))
        # coords should not be all tiny if pixel coords; allow normalized [0,1] too
        coord_max = float(np.max(coord_cols)) if coord_cols.size else 0.0

        s = 0.0
        s += 2.0 * cls_int_like
        s += 1.0 * (1.0 if (0 <= cls_min and cls_max <= 200) else 0.0)
        s += 3.0 * score_in_range
        s += 1.0 * (1.0 if score_std > 1e-5 else 0.0)
        s += 1.0 * finite
        s += 1.0 * (1.0 if coord_max > 1.5 else 0.5)  # prefer pixel coords but don't kill normalized
        return s

    def _try_pattern(det_arr):
        # returns (cls, score, coords_xyxy, pattern_score)
        n, c = det_arr.shape
        candidates = []

        # patterns: (cls_idx, score_idx, coord_idxs(list of 4))
        # Common Paddle patterns:
        candidates.append((1, 2, [3, 4, 5, 6]))  # [img_id, cls, score, x0,y0,x1,y1]
        candidates.append((0, 1, [2, 3, 4, 5]))  # [cls, score, x0,y0,x1,y1, ...]
        candidates.append((2, 1, [3, 4, 5, 6]))  # [img_id, score, cls, x0,y0,x1,y1]
        candidates.append((0, 2, [3, 4, 5, 6]))  # [cls, img_id, score, x0,y0,x1,y1]
        candidates.append((6, 5, [1, 2, 3, 4]))  # [img_id?, x0,y0,x1,y1, score, cls]
        candidates.append((5, 4, [0, 1, 2, 3]))  # [x0,y0,x1,y1, score, cls, ...]
        candidates.append((0, 6, [2, 3, 4, 5]))  # [cls, ..., x0,y0,x1,y1, score]
        candidates.append((1, 6, [2, 3, 4, 5]))  # [img_id, ..., x0,y0,x1,y1, score]

        best = None
        best_s = -1e9

        for cls_idx, score_idx, coord_idxs in candidates:
            if max([cls_idx, score_idx] + coord_idxs) >= c:
                continue
            cls_col = det_arr[:, cls_idx]
            score_col = det_arr[:, score_idx]
            coord_cols = det_arr[:, coord_idxs]

            s = _score_mapping(cls_col, score_col, coord_cols)

            if s > best_s:
                best_s = s
                best = (cls_col, score_col, coord_cols, s, (cls_idx, score_idx, coord_idxs))
        return best

    # breakpoint()
    if det.shape[1] >= 7:
        best = _try_pattern(det)
        if best is None:
            raise RuntimeError(f"Unable to decode NMS output table with shape {det.shape}")
        cls, score, coords, _, used = best
        # print(f"[DEBUG] Using mapping cls={used[0]}, score={used[1]}, coords={used[2]}")
    else:
        # 6-column most common: [cls, score, x0,y0,x1,y1]
        cls = det[:, 0]
        score = det[:, 1]
        coords = det[:, 2:]

    # --- Preserve optional order_id / order_score columns (if present) ---
    # PaddleX `LayoutAnalysisProcess.apply()` supports:
    # - shape[1] == 7: new ordered object detection (extra col = order_id)
    # - shape[1] == 8: ordered object detection (extra cols = order_id, order_score)
    # If the raw NMS table contains such columns, we keep them and pass to post.apply()
    # so processors.py can sort by them (see processors.py:999-1015).
    order_id = None
    order_score = None
    try:
        if det.shape[1] >= 7:
            cls_idx, score_idx, coord_idxs = used
            used_cols = set([int(cls_idx), int(score_idx)] + [int(x) for x in coord_idxs])
            extra_idxs = [i for i in range(int(det.shape[1])) if i not in used_cols]

            def _is_int_like(arr: np.ndarray) -> bool:
                arr = arr.astype(np.float32)
                return float(np.mean(np.abs(arr - np.round(arr)) < 1e-3)) >= 0.95

            def _is_score_like(arr: np.ndarray) -> bool:
                arr = arr.astype(np.float32)
                return float(np.mean((arr >= -0.01) & (arr <= 1.5))) >= 0.95

            def _is_constant_zero(arr: np.ndarray) -> bool:
                arr = arr.astype(np.float32)
                return float(np.max(np.abs(arr))) < 1e-6

            if len(extra_idxs) == 1:
                col = det[:, extra_idxs[0]]
                # Typical 7-col Paddle NMS table includes img_id (usually all zeros for single image).
                # Ignore constant-zero column; otherwise if it's integer-like, treat as order_id.
                if (not _is_constant_zero(col)) and _is_int_like(col):
                    order_id = col
            elif len(extra_idxs) >= 2:
                cols = [(i, det[:, i]) for i in extra_idxs]
                int_like = [(i, c) for i, c in cols if _is_int_like(c) and not _is_constant_zero(c)]
                score_like = [(i, c) for i, c in cols if _is_score_like(c)]
                if int_like:
                    order_id = int_like[0][1]
                if score_like:
                    # pick a score-like column different from order_id if possible
                    for _, c in score_like:
                        if order_id is None:
                            order_score = c
                            break
                        if not np.array_equal(c, order_id):
                            order_score = c
                            break
    except Exception:
        order_id = None
        order_score = None

    # coords may be xyxy in pixels or normalized [0,1]; normalize to pixel xyxy
    coords = coords.astype(np.float32)
    if np.max(coords) <= 2.0:  # normalized-ish
        # assume coords = [x0,y0,x1,y1] normalized
        coords[:, 0] *= float(orig_w)
        coords[:, 2] *= float(orig_w)
        coords[:, 1] *= float(orig_h)
        coords[:, 3] *= float(orig_h)

    # ensure x0<x1, y0<y1
    x0 = np.minimum(coords[:, 0], coords[:, 2])
    y0 = np.minimum(coords[:, 1], coords[:, 3])
    x1 = np.maximum(coords[:, 0], coords[:, 2])
    y1 = np.maximum(coords[:, 1], coords[:, 3])
    coords = np.stack([x0, y0, x1, y1], axis=1)

    # Build boxes for downstream postprocess:
    # - base: [cls, score, x0,y0,x1,y1]
    # - optional: append order_id / order_score to enable processors.py ordering (shape==7/8)
    if order_id is not None and order_score is not None:
        boxes = np.column_stack([cls, score, coords, order_id, order_score]).astype(np.float32)
    elif order_id is not None:
        boxes = np.column_stack([cls, score, coords, order_id]).astype(np.float32)
    else:
        boxes = np.column_stack([cls, score, coords]).astype(np.float32)
    # breakpoint()

    # Align with PaddleX: round bbox coords to int-like values early
    # (PaddleX does np.round(...).astype(int) before further filtering/NMS)
    if boxes.size > 0:
        boxes[:, 2:6] = np.round(boxes[:, 2:6]).astype(np.float32)

    # # 尽量对齐 Paddle 的 auto polygon 决策：在提取 polygon 前先排序 boxes/masks
    # if masks is not None:
    #     boxes, masks = _sort_boxes_and_masks_by_reading_order(boxes, masks)

    label_list = ["abstract", "algorithm", "aside_text", "chart", "content", "display_formula",
                  "doc_title", "figure_title", "footer", "footer_image", "footnote", "formula_number",
                  "header", "header_image", "image", "inline_formula", "number", "paragraph_title",
                  "reference", "reference_content", "seal", "table", "text", "vertical_text", "vision_footnote"]

    # 对齐 PaddleX：当模型输出不含 masks 时，强制使用 rect
    effective_layout_shape_mode = layout_shape_mode
    if masks is None:
        if layout_shape_mode not in ("rect", "auto"):
            logging.warning(
                "The model you are using does not support polygon output, but the "
                f"layout_shape_mode is specified as {layout_shape_mode}, which will be set to 'rect'"
            )
        effective_layout_shape_mode = "rect"

    post = processors.LayoutAnalysisProcess(labels=label_list, scale_size=[800, 800])
    # breakpoint()
    out_boxes = post.apply(
        boxes=boxes,
        img_size=(orig_w, orig_h),
        threshold=threshold,
        layout_nms=layout_nms,
        layout_unclip_ratio=layout_unclip_ratio,
        layout_merge_bboxes_mode=layout_merge_bboxes_mode,
        masks=masks,
        layout_shape_mode=effective_layout_shape_mode,
    )

    if filter_overlap_boxes:
        out_boxes = processors.filter_boxes(out_boxes, effective_layout_shape_mode)

    if skip_order_labels is None:
        skip_order_labels = processors.SKIP_ORDER_LABELS
    # 对齐 PaddleX：不在这里额外重排 boxes。
    # PaddleX 侧 `update_order_index()` 只按当前列表顺序编号，不负责排序；
    # `LayoutAnalysisProcess.apply(...)` 的输出顺序即为后续 parsing_res_list 的基础顺序，
    # 若这里再做 (y,x) 阅读顺序排序，会导致 block_id/显示顺序与 PaddleX 不一致（例如 header/number 对调）。
    out_boxes = processors.update_order_index(out_boxes, skip_order_labels)
    return out_boxes

def postprocess_detections_detr(
    output,
    scale_h,
    scale_w,
    orig_h,
    orig_w,
    threshold=0.5,
    layout_nms=False,
    layout_unclip_ratio=None,
    layout_merge_bboxes_mode=None,
    layout_shape_mode: str = "auto",
    filter_overlap_boxes: bool = True,
    skip_order_labels: Optional[List[str]] = None,
):
    """
    Postprocess DETR-style detection outputs.
    
    Handles DETR model outputs which typically have shape [300, 8] or separate logits and boxes outputs.
    Converts center-format boxes to corner format, applies threshold filtering, NMS, and other post-processing.
    
    Args:
        output: Model output, can be list/tuple of tensors or single tensor
        scale_h: Height scaling factor from preprocessing
        scale_w: Width scaling factor from preprocessing
        orig_h: Original image height
        orig_w: Original image width
        threshold: Detection confidence threshold (float or dict)
        layout_nms: Whether to apply NMS
        layout_unclip_ratio: Box expansion ratio(s)
        layout_merge_bboxes_mode: Box merging mode ("union", "large", "small", or dict)
    
    Returns:
        list: List of detection dictionaries with keys: cls_id, label, score, coordinate
    """
    if isinstance(output, (list, tuple)):
        output0, output1 = output[0], output[1] if len(output) > 1 else None
        masks = output[2] if len(output) > 2 else None
    else:
        output0, output1, masks = output, None, None
    
    output0 = np.array(output0)
    if output1 is not None:
        output1 = np.array(output1)
    if masks is not None:
        masks = np.array(masks)
    
    # Handle 3D arrays with batch dimension of 1: squeeze the first dimension
    if len(output0.shape) == 3 and output0.shape[0] == 1:
        output0 = np.squeeze(output0, axis=0)
    if output1 is not None and len(output1.shape) == 3 and output1.shape[0] == 1:
        output1 = np.squeeze(output1, axis=0)
    
    label_list = ["abstract", "algorithm", "aside_text", "chart", "content", "display_formula",
                  "doc_title", "figure_title", "footer", "footer_image", "footnote", "formula_number",
                  "header", "header_image", "image", "inline_formula", "number", "paragraph_title",
                  "reference", "reference_content", "seal", "table", "text", "vertical_text", "vision_footnote"]
    num_classes = len(label_list)
    
    if len(output0.shape) == 2 and output0.shape[0] == 300 and output0.shape[1] == 8:
        boxes = output0.copy()
    elif len(output0.shape) >= 2 and output0.shape[-2] == 300 and output1 is not None:
        logits = output0[0] if len(output0.shape) == 3 else output0
        pred_boxes = output1[0] if len(output1.shape) == 3 else output1
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        if logits.shape[1] == num_classes + 1:
            probs = probs[:, :-1]
        
        scores = np.max(probs, axis=1)
        labels = np.argmax(probs, axis=1)
        boxes_corners = center_to_corners_format(pred_boxes)
        boxes_pixel = boxes_corners * 800.0
        
        boxes = np.zeros((300, 8), dtype=np.float32)
        boxes[:, 0] = labels
        boxes[:, 1] = scores
        boxes[:, 2:6] = boxes_pixel
    elif len(output0.shape) == 2 and output0.shape[0] == 300 and output0.shape[1] >= 2:
        boxes = np.zeros((300, 8), dtype=np.float32)
        boxes[:, 0] = output0[:, 0] if output0.shape[1] > 0 else 0
        boxes[:, 1] = output0[:, 1] if output0.shape[1] > 1 else 0.0
        
        if output1 is not None and len(output1.shape) == 2 and output1.shape[0] == 300:
            if output1.shape[1] == 4:
                boxes_corners = center_to_corners_format(output1)
                boxes[:, 2:6] = boxes_corners * 800.0
            elif output1.shape[1] >= 4:
                boxes[:, 2:6] = output1[:, :4]
        elif output0.shape[1] >= 6:
            boxes[:, 2:6] = output0[:, 2:6]
    else:
        raise ValueError(f"Unable to process output format, output[0] shape: {output0.shape}")

    # Align with PaddleX: round bbox coords to int-like values early
    # (PaddleX does np.round(...).astype(int) before further filtering/NMS)
    if boxes.size > 0 and boxes.shape[1] >= 6:
        boxes[:, 2:6] = np.round(boxes[:, 2:6]).astype(np.float32)

    # # 尽量对齐 Paddle 的 auto polygon 决策：在提取 polygon 前先排序 boxes/masks
    # if masks is not None:
    #     boxes, masks = _sort_boxes_and_masks_by_reading_order(boxes, masks)

    # 对齐 PaddleX：当模型输出不含 masks 时，强制使用 rect
    effective_layout_shape_mode = layout_shape_mode
    if masks is None:
        if layout_shape_mode not in ("rect", "auto"):
            logging.warning(
                "The model you are using does not support polygon output, but the "
                f"layout_shape_mode is specified as {layout_shape_mode}, which will be set to 'rect'"
            )
        effective_layout_shape_mode = "rect"
    
    post = processors.LayoutAnalysisProcess(labels=label_list, scale_size=[800, 800])
    # breakpoint()
    out_boxes = post.apply(
        boxes=boxes,
        img_size=(orig_w, orig_h),
        threshold=threshold,
        layout_nms=layout_nms,
        layout_unclip_ratio=layout_unclip_ratio,
        layout_merge_bboxes_mode=layout_merge_bboxes_mode,
        masks=masks,
        layout_shape_mode=effective_layout_shape_mode,
    )

    if filter_overlap_boxes:
        out_boxes = processors.filter_boxes(out_boxes, effective_layout_shape_mode)

    if skip_order_labels is None:
        skip_order_labels = processors.SKIP_ORDER_LABELS
    # 对齐 PaddleX：不在这里额外重排 boxes（原因同 postprocess_detections_paddle_nms）。
    out_boxes = processors.update_order_index(out_boxes, skip_order_labels)
    return out_boxes

def _download_model_from_modelscope(model_id=LAYOUT_MODEL_ID, cache_dir=None, precision="fp32"):
    """
    Download model from ModelScope.
    
    Args:
        model_id: ModelScope model ID
        cache_dir: Cache directory, uses default cache directory if None
        precision: Model precision, options: "fp16", "fp32", "combined_fp16", "combined_fp32"
    
    Returns:
        str: Path to downloaded model file (.xml)
    """
    if not MODELSCOPE_AVAILABLE:
        raise ImportError("modelscope is required for auto-download. Install with: pip install modelscope")
    
    print(f"📥 Downloading model from ModelScope: {model_id}")
    model_dir = snapshot_download(model_id, cache_dir=cache_dir)
    model_dir = Path(model_dir)
    
    # Select model file based on precision
    precision_map = {
        "fp16": "pp_doclayoutv2_f16.xml",
        "fp32": "pp_doclayoutv2_f32.xml",
        "combined_fp16": "pp_doclayoutv2_f16_combined.xml",
        "combined_fp32": "pp_doclayoutv2_f32_combined.xml",
    }
    
    if precision not in precision_map:
        raise ValueError(f"Unsupported precision type: {precision}. Supported options: {list(precision_map.keys())}")
    
    model_filename = precision_map[precision]
    model_path = model_dir / model_filename
    
    # If specified precision file doesn't exist, try to find other available model files
    if not model_path.exists():
        print(f"⚠️  Specified precision model file not found: {model_filename}")
        # Search for all .xml files
        xml_files = list(model_dir.glob("*.xml"))
        if not xml_files:
            raise FileNotFoundError(f"No .xml files found in downloaded model directory: {model_dir}")
        
        # Prefer combined versions
        combined_files = [f for f in xml_files if "combined" in f.name]
        if combined_files:
            model_path = combined_files[0]
            print(f"⚠️  Using found combined model: {model_path.name}")
        else:
            # Otherwise use the first found file
            model_path = xml_files[0]
            print(f"⚠️  Using found model: {model_path.name}")
    else:
        print(f"✅ Using specified precision model: {model_filename}")
    
    # Check if corresponding .bin file exists
    bin_path = model_path.with_suffix(".bin")
    if not bin_path.exists():
        raise FileNotFoundError(f"Corresponding .bin file not found: {bin_path}")
    
    print(f"✅ Model download completed: {model_path}")
    return str(model_path)

def _get_model_path(model_path, cache_dir=None, precision="fp32"):
    """
    Get model path, automatically download if not exists.
    
    Args:
        model_path: Model path (.xml file), automatically downloads if None or file doesn't exist
        cache_dir: ModelScope cache directory
        precision: Model precision, options: "fp16", "fp32", "combined_fp16", "combined_fp32"
    
    Returns:
        str: Model file path
    """
    # Auto-download if model_path is None or empty string
    if model_path is None or model_path == "" or model_path.lower() == "none":
        return _download_model_from_modelscope(cache_dir=cache_dir, precision=precision)
    
    model_path = Path(model_path)
    
    # Auto-download if file doesn't exist
    if not model_path.exists():
        print(f"⚠️  Model file does not exist: {model_path}, attempting auto-download...")
        return _download_model_from_modelscope(cache_dir=cache_dir, precision=precision)
    
    # If a directory is specified, search for corresponding .xml file based on precision
    if model_path.is_dir():
        # Search based on precision priority
        precision_map = {
            "fp16": ["pp_doclayoutv2_f16.xml", "*.xml"],
            "fp32": ["pp_doclayoutv2_f32.xml", "*.xml"],
            "combined_fp16": ["pp_doclayoutv2_f16_combined.xml", "pp_doclayoutv2_f16.xml", "*.xml"],
            "combined_fp32": ["pp_doclayoutv2_f32_combined.xml", "pp_doclayoutv2_f32.xml", "*.xml"],
        }
        
        search_patterns = precision_map.get(precision, ["*.xml"])
        xml_file = None
        
        for pattern in search_patterns:
            if pattern == "*.xml":
                xml_files = list(model_path.glob(pattern))
                if xml_files:
                    xml_file = xml_files[0]
                    break
            else:
                candidate = model_path / pattern
                if candidate.exists():
                    xml_file = candidate
                    break
        
        if xml_file is None:
            print(f"⚠️  No matching .xml file found in specified directory: {model_path}, attempting auto-download...")
            return _download_model_from_modelscope(cache_dir=cache_dir, precision=precision)
        
        # Check if corresponding .bin file exists
        bin_path = xml_file.with_suffix(".bin")
        if not bin_path.exists():
            print(f"⚠️  Corresponding .bin file not found: {bin_path}, attempting auto-download...")
            return _download_model_from_modelscope(cache_dir=cache_dir, precision=precision)
        
        return str(xml_file)
    
    # If it's a file, check if it's a .xml file
    if model_path.suffix.lower() != ".xml":
        print(f"⚠️  Specified file is not a .xml file: {model_path}, attempting auto-download...")
        return _download_model_from_modelscope(cache_dir=cache_dir, precision=precision)
    
    # Check if corresponding .bin file exists
    bin_path = model_path.with_suffix(".bin")
    if not bin_path.exists():
        print(f"⚠️  Corresponding .bin file not found: {bin_path}, attempting auto-download...")
        return _download_model_from_modelscope(cache_dir=cache_dir, precision=precision)
    
    return str(model_path)

def paddle_ov_doclayout(model_path, image_path, output_dir, device="GPU", threshold=0.5, 
                        layout_nms=True, layout_unclip_ratio=None, layout_merge_bboxes_mode=None, 
                        cache_dir=None, precision="fp32", layout_shape_mode: str = "auto"):
    """
    Perform layout detection inference using OpenVINO.
    
    Args:
        model_path: OpenVINO IR model path (.xml file), automatically downloads if None
        image_path: Input image path
        output_dir: Output directory for saving results
        device: Inference device ("CPU", "GPU", "NPU", "AUTO")
        threshold: Detection confidence threshold (float or dict)
        layout_nms: Whether to enable NMS (Non-Maximum Suppression)
        layout_unclip_ratio: Box coordinate expansion ratio(s)
        layout_merge_bboxes_mode: Layout box merging mode ("union", "large", "small", or dict)
        cache_dir: ModelScope model cache directory, uses default if None
        precision: Model precision, options: "fp16", "fp32", "combined_fp16", "combined_fp32"
                   - "fp16": FP16 precision model (faster, lower memory usage)
                   - "fp32": FP32 precision model (more accurate, default)
                   - "combined_fp16": FP16 combined model (merged batch size and boxes nodes)
                   - "combined_fp32": FP32 combined model (merged batch size and boxes nodes)
        layout_shape_mode: Layout polygon shape mode, one of {"auto","rect","quad","poly"}.
            - auto: 根据 mask 几何关系自动在 polygon/quad/rect 间选择（与 PaddleX 默认一致）
            - rect: 强制矩形（4点）
            - quad: 强制四边形（4点，最小外接旋转矩形）
            - poly: 强制多边形（N点，点数可能随 mask 细节变化）
    
    Returns:
        LayoutDetectionResult: Detection result object
    """
    # Get or download model path
    model_path = _get_model_path(model_path, cache_dir=cache_dir, precision=precision)
    
    # Initialize OpenVINO Core
    core = ov.Core()
    
    # Load model (.xml file will automatically find corresponding .bin file)
    model = core.read_model(model_path)

    # Merge preprocessing into model
    prep = ov.preprocess.PrePostProcessor(model)
    prep.input("image").tensor().set_layout(ov.Layout("NCHW"))
    prep.input("image").preprocess().scale([255, 255, 255])

    if device == "NPU":
        prep.input("im_shape").model().set_layout(ov.Layout('N...'))
        prep.input("scale_factor").model().set_layout(ov.Layout('N...'))
        prep.input("image").model().set_layout(ov.Layout('NCHW'))

    model = prep.build()

    # Set batch to make static
    if device == "NPU":
        ov.set_batch(model, 1)
    
    # Compile model
    compiled_model = core.compile_model(model, device)
    
    # Get input and output information
    input_tensors = compiled_model.inputs
    output_tensors = compiled_model.outputs
    
    print(f"Model input count: {len(input_tensors)}")
    for i, inp in enumerate(input_tensors):
        shape_str = str(inp.partial_shape) if inp.partial_shape.is_dynamic else str(inp.shape)
        print(f"  Input {i}: {inp.get_any_name()}, shape: {shape_str}, type: {inp.element_type}")
    
    print(f"Model output count: {len(output_tensors)}")
    for i, out in enumerate(output_tensors):
        shape_str = str(out.partial_shape) if out.partial_shape.is_dynamic else str(out.shape)
        print(f"  Output {i}: {out.get_any_name()}, shape: {shape_str}, type: {out.element_type}")
    
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Unable to read image: {image_path}")
    
    orig_h, orig_w = image.shape[:2]
    input_blob, scale_h, scale_w = preprocess_image_doclayout(image)
    
    # Prepare input data
    input_data = {}
    for inp in input_tensors:
        inp_name = inp.get_any_name()
        if inp_name == "im_shape":
            input_data[inp_name] = np.array([[orig_h, orig_w]], dtype=np.float32)
        elif inp_name == "image":
            input_data[inp_name] = input_blob
        elif inp_name == "scale_factor":
            # IMPORTANT: For PP-DocLayoutV3-Preview exported graph, boxes are already in original-image coords.
            # Passing resize ratios here causes an extra rescale inside the graph and makes boxes look stretched.
            # Use [1.0, 1.0] unless you verify your model expects otherwise.
            input_data[inp_name] = np.array([[1.0, 1.0]], dtype=np.float32)
        else:
            pass
    
    # If input names don't match, assign by order
    if len(input_data) != len(input_tensors):
        input_data = {}
        input_data[input_tensors[0].get_any_name()] = np.array([[orig_h, orig_w]], dtype=np.float32)
        input_data[input_tensors[1].get_any_name()] = input_blob
        input_data[input_tensors[2].get_any_name()] = np.array([[1.0, 1.0]], dtype=np.float32)

    # Create OpenVINO Tensor objects
    input_tensors_ov = {}
    for inp in input_tensors:
        inp_name = inp.get_any_name()
        data = input_data[inp_name]
        input_tensors_ov[inp_name] = ov.Tensor(data)
    
    # Execute inference
    infer_result = compiled_model(input_tensors_ov)

    # breakpoint()
    
    # Extract output results
    output = []
    for out in output_tensors:
        output_tensor = infer_result[out]
        output_data = output_tensor.data
        output.append(output_data)
    # breakpoint()
    
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
    
    # Create result object (align with PaddleX/PaddleOCR style)
    result_obj = result.LayoutAnalysisResult(
        {
            "input_path": os.path.abspath(image_path),
            "page_index": None,
            "input_img": image,
            "boxes": results,
        }
    )
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Match previous naming: <stem>_res.<suffix>
    stem = Path(image_path).stem
    suffix = Path(image_path).suffix or ".png"
    result_obj.save_to_img(save_path=str(output_dir / f"{stem}_res{suffix}"))
    result_obj.save_to_json(save_path=str(output_dir / "res.json"))
    
    return result_obj

def main():
    """Main function: parse command-line arguments and execute inference."""
    parser = argparse.ArgumentParser(description="PP-DocLayoutV2 OpenVINO Inference Script")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="OpenVINO IR model path (.xml file), automatically downloads from ModelScope if None or not specified"
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="ModelScope model cache directory, uses default cache directory if None"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="fp32",
        choices=["fp16", "fp32", "combined_fp16", "combined_fp32"],
        help="Model precision selection (default: fp32)\n"
             "  fp16: FP16 precision model (faster, lower memory usage)\n"
             "  fp32: FP32 precision model (more accurate, default)\n"
             "  combined_fp16: FP16 combined model (merged batch size and boxes nodes)\n"
             "  combined_fp32: FP32 combined model (merged batch size and boxes nodes)"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="Input image path"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_ov",
        help="Output directory for saving results (default: ./output_ov)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        choices=["CPU", "GPU", "NPU", "AUTO"],
        help="Inference device (default: GPU)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Detection confidence threshold (default: 0.5)"
    )
    parser.add_argument(
        "--layout_shape_mode",
        type=str,
        default="auto",
        choices=["auto", "rect", "quad", "poly"],
        help="Layout polygon shape mode (default: auto). "
             "auto: auto select polygon/quad/rect; rect: force 4-pt rectangle; "
             "quad: force 4-pt rotated quad; poly: force polygon (N-pt)."
    )
    
    args = parser.parse_args()
    
    # Process model_path: convert empty string to None
    model_path = args.model_path if args.model_path and args.model_path.lower() != "none" else None
    
    # Execute inference (using original default values)
    result_obj = paddle_ov_doclayout(
        model_path=model_path,
        image_path=args.image_path,
        output_dir=args.output_dir,
        device=args.device,
        threshold=args.threshold,
        layout_nms=True,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None,
        cache_dir=args.cache_dir,
        precision=args.precision,
        layout_shape_mode=args.layout_shape_mode,
    )
    
    print(f"\nDetection completed! Results saved to: {args.output_dir}")
    print(f"Detected {len(result_obj.get('boxes', []))} valid boxes")
    
    return result_obj

if __name__ == '__main__':
    main()