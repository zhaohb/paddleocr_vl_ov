"""
PP-DocLayoutV2 (OpenVINO) utils.

去掉 PaddleX 依赖，提供 processors/result 需要的最小工具函数。
参考 PaddleX 1.5 `paddlex/inference/models/layout_analysis/utils.py` 的常量定义，
并补充 OpenVINO 工程内需要的 `nms/check_containment` 等函数。
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

STATIC_SHAPE_MODEL_LIST = ["PP-DocLayoutV2", "PP-DocLayoutV3"]


def _iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
    """IoU for xyxy boxes."""
    ax1, ay1, ax2, ay2 = map(float, a)
    bx1, by1, bx2, by2 = map(float, b)

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = area_a + area_b - inter
    return float(inter / denom) if denom > 0 else 0.0


def nms(boxes: np.ndarray, iou_same: float = 0.6, iou_diff: float = 0.98) -> List[int]:
    """
    NMS for boxes with shape [N,6] = [cls_id, score, x1, y1, x2, y2]
    Returns indices of kept boxes.
    """
    if boxes is None or boxes.size == 0:
        return []
    scores = boxes[:, 1]
    order = np.argsort(scores)[::-1]
    kept: List[int] = []

    while order.size > 0:
        i = int(order[0])
        kept.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        i_cls = boxes[i, 0]
        i_box = boxes[i, 2:6]

        new_rest = []
        for j in rest:
            j = int(j)
            j_cls = boxes[j, 0]
            thr = iou_same if int(i_cls) == int(j_cls) else iou_diff
            if _iou_xyxy(i_box, boxes[j, 2:6]) < thr:
                new_rest.append(j)
        order = np.array(new_rest, dtype=np.int64)

    return kept


def _is_contained(inner_xyxy: np.ndarray, outer_xyxy: np.ndarray, min_cover: float = 0.9) -> bool:
    """Check inner is covered by outer using intersection/inner_area."""
    x1, y1, x2, y2 = map(float, inner_xyxy)
    X1, Y1, X2, Y2 = map(float, outer_xyxy)
    ix1 = max(x1, X1)
    iy1 = max(y1, Y1)
    ix2 = min(x2, X2)
    iy2 = min(y2, Y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    inner_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inner_area <= 0:
        return False
    return (inter / inner_area) >= float(min_cover)


def check_containment(
    boxes_cls_score_xyxy: np.ndarray,
    formula_index: Optional[int] = None,
    category_index: Optional[int] = None,
    mode: str = "large",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    轻量版 containment check（用于 layout_merge_bboxes_mode）。

    Returns:
        contains_other: [N] int (1 if box contains any other)
        contained_by_other: [N] int (1 if box is contained by any other)
    """
    if boxes_cls_score_xyxy is None or boxes_cls_score_xyxy.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.int32)

    n = int(boxes_cls_score_xyxy.shape[0])
    contains_other = np.zeros(n, dtype=np.int32)
    contained_by_other = np.zeros(n, dtype=np.int32)

    for i in range(n):
        cls_i = int(boxes_cls_score_xyxy[i, 0])
        if category_index is not None and cls_i != int(category_index):
            continue
        if formula_index is not None and cls_i == int(formula_index):
            # 与 PaddleX 兼容：formula 不参与 containment 合并的判定主体
            continue
        box_i = boxes_cls_score_xyxy[i, 2:6]
        for j in range(n):
            if i == j:
                continue
            cls_j = int(boxes_cls_score_xyxy[j, 0])
            if category_index is not None and cls_j != int(category_index):
                continue
            box_j = boxes_cls_score_xyxy[j, 2:6]
            if _is_contained(box_j, box_i):
                contains_other[i] = 1
                contained_by_other[j] = 1

    return contains_other, contained_by_other

