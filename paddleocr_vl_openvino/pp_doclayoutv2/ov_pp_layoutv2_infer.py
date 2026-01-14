import openvino as ov
import cv2
import numpy as np
import json
import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import mimetypes
import logging

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

def iou(box1, box2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: First bounding box [x1, y1, x2, y2]
        box2: Second bounding box [x1, y1, x2, y2]
    
    Returns:
        float: IoU value between 0 and 1
    """
    x1, y1, x2, y2 = box1
    x1_p, y1_p, x2_p, y2_p = box2
    
    x1_i = max(x1, x1_p)
    y1_i = max(y1, y1_p)
    x2_i = min(x2, x2_p)
    y2_i = min(y2, y2_p)
    
    inter_area = max(0, x2_i - x1_i + 1) * max(0, y2_i - y1_i + 1)
    box1_area = (x2 - x1 + 1) * (y2 - y1 + 1)
    box2_area = (x2_p - x1_p + 1) * (y2_p - y1_p + 1)
    
    return inter_area / float(box1_area + box2_area - inter_area)

def nms(boxes, iou_same=0.6, iou_diff=0.98):
    """
    Non-maximum suppression (NMS) to remove duplicate detections.
    
    Args:
        boxes: Detection boxes with format [class_id, score, x1, y1, x2, y2, ...]
        iou_same: IoU threshold for boxes of the same class
        iou_diff: IoU threshold for boxes of different classes
    
    Returns:
        list: Indices of selected boxes after NMS
    """
    scores = boxes[:, 1]
    indices = np.argsort(scores)[::-1]
    selected_boxes = []
    
    while len(indices) > 0:
        current = indices[0]
        current_box = boxes[current]
        current_class = current_box[0]
        current_coords = current_box[2:]
        
        selected_boxes.append(current)
        indices = indices[1:]
        
        filtered_indices = []
        for i in indices:
            box = boxes[i]
            box_class = box[0]
            box_coords = box[2:]
            iou_value = iou(current_coords, box_coords)
            threshold = iou_same if current_class == box_class else iou_diff
            
            if iou_value < threshold:
                filtered_indices.append(i)
        indices = filtered_indices
    
    return selected_boxes

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

def unclip_boxes(boxes, unclip_ratio=None):
    """
    Expand bounding box coordinates by specified ratios.
    
    Args:
        boxes: Bounding boxes array [class_id, score, x1, y1, x2, y2, ...]
        unclip_ratio: Expansion ratio(s). Can be:
            - None: No expansion
            - dict: {class_id: (width_ratio, height_ratio)}
            - tuple/list: (width_ratio, height_ratio) for all boxes
    
    Returns:
        np.ndarray: Expanded bounding boxes
    """
    if unclip_ratio is None:
        return boxes
    
    if isinstance(unclip_ratio, dict):
        expanded_boxes = []
        for box in boxes:
            class_id, score, x1, y1, x2, y2 = box
            if class_id in unclip_ratio:
                width_ratio, height_ratio = unclip_ratio[class_id]
                width = x2 - x1
                height = y2 - y1
                new_w = width * width_ratio
                new_h = height * height_ratio
                center_x = x1 + width / 2
                center_y = y1 + height / 2
                new_x1 = center_x - new_w / 2
                new_y1 = center_y - new_h / 2
                new_x2 = center_x + new_w / 2
                new_y2 = center_y + new_h / 2
                expanded_boxes.append([class_id, score, new_x1, new_y1, new_x2, new_y2])
            else:
                expanded_boxes.append(box)
        return np.array(expanded_boxes)
    else:
        widths = boxes[:, 4] - boxes[:, 2]
        heights = boxes[:, 5] - boxes[:, 3]
        new_w = widths * unclip_ratio[0]
        new_h = heights * unclip_ratio[1]
        center_x = boxes[:, 2] + widths / 2
        center_y = boxes[:, 3] + heights / 2
        new_x1 = center_x - new_w / 2
        new_y1 = center_y - new_h / 2
        new_x2 = center_x + new_w / 2
        new_y2 = center_y + new_h / 2
        return np.column_stack((boxes[:, 0], boxes[:, 1], new_x1, new_y1, new_x2, new_y2))

def restructured_boxes(boxes, labels, img_size):
    """
    Restructure boxes into a standardized output format with clipping to image boundaries.
    
    Args:
        boxes: Bounding boxes array [class_id, score, x1, y1, x2, y2]
        labels: List of label names corresponding to class IDs
        img_size: Image size (width, height)
    
    Returns:
        list: List of dictionaries with keys: cls_id, label, score, coordinate
    """
    box_list = []
    w, h = img_size
    
    for box in boxes:
        xmin, ymin, xmax, ymax = box[2:]
        xmin = max(0, xmin)
        ymin = max(0, ymin)
        xmax = min(w, xmax)
        ymax = min(h, ymax)
        if xmax <= xmin or ymax <= ymin:
            continue
        box_list.append({
            "cls_id": int(box[0]),
            "label": labels[int(box[0])],
            "score": float(box[1]),
            "coordinate": [xmin, ymin, xmax, ymax],
        })
    
    return box_list

def draw_box(img, boxes):
    """
    Draw bounding boxes and labels on the image.
    
    Args:
        img: PIL Image object
        boxes: List of detection boxes, each containing 'label', 'coordinate', and 'score'
    
    Returns:
        PIL.Image: Image with drawn bounding boxes and labels
    """
    try:
        font_size = int(0.018 * img.width) + 2
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",  # Microsoft YaHei
            "C:/Windows/Fonts/simhei.ttf",  # SimHei (Bold)
            "C:/Windows/Fonts/simsun.ttc",  # SimSun (Song)
        ]
        font = None
        for fp in font_paths:
            if os.path.exists(fp):
                try:
                    font = ImageFont.truetype(fp, font_size, encoding="utf-8")
                    break
                except:
                    continue
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    draw_thickness = int(max(img.size) * 0.002)
    draw = ImageDraw.Draw(img)
    color_list = [
        (128, 64, 128), (232, 35, 244), (70, 70, 70), (156, 102, 102), (153, 153, 190),
        (153, 153, 153), (30, 170, 250), (0, 220, 220), (35, 142, 107), (152, 251, 152),
        (180, 130, 70), (60, 20, 220), (0, 0, 255), (142, 0, 0), (70, 0, 0),
        (100, 60, 0), (90, 0, 0), (230, 0, 0), (32, 11, 119), (0, 74, 111), (81, 0, 81)
    ]
    font_colors = [
        (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
        (255, 255, 255), (255, 255, 255), (0, 0, 0), (255, 255, 255), (0, 0, 0),
        (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255),
        (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255), (255, 255, 255)
    ]
    
    label2color = {}
    label2fontcolor = {}
    
    for i, dt in enumerate(boxes):
        label, bbox, score = dt["label"], dt["coordinate"], dt["score"]
        
        if label not in label2color:
            color_index = i % len(color_list)
            label2color[label] = color_list[color_index]
            label2fontcolor[label] = font_colors[color_index]
        
        color = label2color[label]
        font_color = label2fontcolor[label]
        
        if len(bbox) == 4:
            xmin, ymin, xmax, ymax = bbox
            rectangle = [
                (xmin, ymin),
                (xmin, ymax),
                (xmax, ymax),
                (xmax, ymin),
                (xmin, ymin),
            ]
        elif len(bbox) == 8:
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
            rectangle = [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)]
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)
        else:
            continue
        
        draw.line(rectangle, width=draw_thickness, fill=color)
        text = "{} {:.2f}".format(label, score)
        try:
            left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
            tw, th = right - left, bottom - top + 4
        except:
            tw, th = draw.textsize(text, font=font)
            th += 4
        
        if ymin < th:
            draw.rectangle([(xmin, ymin), (xmin + tw + 4, ymin + th + 1)], fill=color)
            draw.text((xmin + 2, ymin - 2), text, fill=font_color, font=font)
        else:
            draw.rectangle([(xmin, ymin - th), (xmin + tw + 4, ymin + 1)], fill=color)
            draw.text((xmin + 2, ymin - th - 2), text, fill=font_color, font=font)
    
    return img

class LayoutDetectionResult:
    """
    Layout detection result class for storing and managing detection results.
    
    Attributes:
        input_path: Path to the input image
        boxes: List of detected bounding boxes
        page_index: Optional page index for multi-page documents
        input_img: Optional original input image (BGR format)
    """
    
    def __init__(self, input_path, boxes, page_index=None, input_img=None):
        self.input_path = input_path
        self.boxes = boxes
        self.page_index = page_index
        self.input_img = input_img
    
    def _get_input_fn(self):
        if self.input_path is None:
            import time
            import random
            timestamp = int(time.time())
            random_number = random.randint(1000, 9999)
            return f"{timestamp}_{random_number}"
        return Path(self.input_path).name
    
    def _to_json(self):
        def _format_data_for_json(obj):
            if isinstance(obj, np.float32):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return [_format_data_for_json(item) for item in obj.tolist()]
            elif isinstance(obj, Path):
                return obj.as_posix()
            elif isinstance(obj, dict):
                return {k: _format_data_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [_format_data_for_json(i) for i in obj]
            else:
                return obj
        
        data = {
            "input_path": self.input_path,
            "page_index": self.page_index,
            "boxes": _format_data_for_json(self.boxes)
        }
        return {"res": data}
    
    def save_to_json(self, save_path, indent=4, ensure_ascii=False):
        def _is_json_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type is not None and mime_type == "application/json"
        
        json_data = self._to_json()
        
        if not _is_json_file(save_path):
            fn = Path(self._get_input_fn())
            stem = fn.stem
            base_save_path = Path(save_path)
            for key in json_data:
                save_path = base_save_path / f"{stem}_{key}.json"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                with open(save_path, 'w', encoding='utf-8') as f:
                    json.dump(json_data[key], f, indent=indent, ensure_ascii=ensure_ascii)
        else:
            if len(json_data) > 1:
                import logging
                logging.warning(
                    f"The result has multiple json files need to be saved. But the `save_path` has been specified as `{save_path}`!"
                )
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(json_data[list(json_data.keys())[0]], f, indent=indent, ensure_ascii=ensure_ascii)
    
    def _to_img(self):
        if self.input_img is None:
            raise ValueError("input_img is required for _to_img")
        img_rgb = cv2.cvtColor(self.input_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_vis = draw_box(img_pil, self.boxes)
        
        return {"res": img_vis}
    
    def save_to_img(self, save_path):
        def _is_image_file(file_path):
            mime_type, _ = mimetypes.guess_type(file_path)
            return mime_type is not None and mime_type.startswith("image/")
        
        img = self._to_img()
        
        if not _is_image_file(save_path):
            fn = Path(self._get_input_fn())
            stem = fn.stem
            suffix = fn.suffix if _is_image_file(fn) else ".png"
            base_save_path = Path(save_path)
            for key in img:
                save_path = base_save_path / f"{stem}_{key}{suffix}"
                save_path.parent.mkdir(parents=True, exist_ok=True)
                img[key].save(save_path)
        else:
            if len(img) > 1:
                import logging
                logging.warning(
                    f"The result has multiple img files need to be saved. But the `save_path` has been specified as `{save_path}`!"
                )
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            img[list(img.keys())[0]].save(save_path)



def postprocess_detections_paddle_nms(output, orig_h, orig_w, threshold=0.5,
                                     layout_nms=False, layout_unclip_ratio=None, layout_merge_bboxes_mode=None):
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

    if out0.ndim != 2 or out0.shape[1] < 6:
        raise ValueError(f"Unsupported Paddle NMS output shape: {out0.shape}")

    # valid count
    if out1 is not None and out1.size > 0:
        num = int(out1.reshape(-1)[0])
        num = max(0, min(num, out0.shape[0]))
    else:
        num = out0.shape[0]

    det = out0[:num]
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
        coords = det[:, 2:6]

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

    boxes = np.column_stack([cls, score, coords]).astype(np.float32)

    label_list = ["abstract", "algorithm", "aside_text", "chart", "content", "display_formula",
                  "doc_title", "figure_title", "footer", "footer_image", "footnote", "formula_number",
                  "header", "header_image", "image", "inline_formula", "number", "paragraph_title",
                  "reference", "reference_content", "seal", "table", "text", "vertical_text", "vision_footnote"]

    # Threshold filtering
    if isinstance(threshold, float):
        boxes = boxes[(boxes[:, 1] > threshold) & (boxes[:, 0] > -1)]
    elif isinstance(threshold, dict):
        filtered = []
        for cat_id in np.unique(boxes[:, 0]).astype(int):
            cat_boxes = boxes[boxes[:, 0] == cat_id]
            th = float(threshold.get(int(cat_id), 0.5))
            filtered.append(cat_boxes[(cat_boxes[:, 1] > th) & (cat_boxes[:, 0] > -1)])
        boxes = np.vstack(filtered) if filtered else np.array([], dtype=np.float32)

    if boxes.size == 0:
        return []

    # Optional NMS (usually unnecessary because model already did NMS, but kept for compatibility)
    if layout_nms and boxes.shape[0] > 1:
        selected_indices = nms(boxes[:, :6], iou_same=0.6, iou_diff=0.98)
        boxes = np.array(boxes[selected_indices], dtype=np.float32)

    # Optional merge/containment logic (kept as-is)
    if layout_merge_bboxes_mode:
        formula_index = label_list.index("formula") if "formula" in label_list else None
        if isinstance(layout_merge_bboxes_mode, str):
            assert layout_merge_bboxes_mode in ["union", "large", "small"]
            if layout_merge_bboxes_mode != "union":
                contains_other, contained_by_other = check_containment(boxes[:, :6], formula_index)
                if layout_merge_bboxes_mode == "large":
                    boxes = boxes[contained_by_other == 0]
                elif layout_merge_bboxes_mode == "small":
                    boxes = boxes[(contains_other == 0) | (contained_by_other == 1)]
        elif isinstance(layout_merge_bboxes_mode, dict):
            keep_mask = np.ones(len(boxes), dtype=bool)
            for category_index, layout_mode in layout_merge_bboxes_mode.items():
                assert layout_mode in ["union", "large", "small"]
                if layout_mode == "union":
                    continue
                contains_other, contained_by_other = check_containment(
                    boxes[:, :6], formula_index, category_index, mode=layout_mode
                )
                if layout_mode == "large":
                    keep_mask &= contained_by_other == 0
                elif layout_mode == "small":
                    keep_mask &= (contains_other == 0) | (contained_by_other == 1)
            boxes = boxes[keep_mask]
        else:
            raise ValueError("layout_merge_bboxes_mode must be str or dict")

    if boxes.size == 0:
        return []

    # Optional unclip
    if layout_unclip_ratio:
        if isinstance(layout_unclip_ratio, float):
            layout_unclip_ratio = (layout_unclip_ratio, layout_unclip_ratio)
        elif isinstance(layout_unclip_ratio, (tuple, list)):
            assert len(layout_unclip_ratio) == 2
        elif isinstance(layout_unclip_ratio, dict):
            pass
        else:
            raise ValueError(f"Unsupported layout_unclip_ratio type: {type(layout_unclip_ratio)}")
        boxes = unclip_boxes(boxes, layout_unclip_ratio)

    # Convert to dict list (clip to original image)
    return restructured_boxes(boxes, label_list, (orig_w, orig_h))

def postprocess_detections_detr(output, scale_h, scale_w, orig_h, orig_w, threshold=0.5,
                                layout_nms=False, layout_unclip_ratio=None, layout_merge_bboxes_mode=None):
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
    else:
        output0, output1 = output, None
    
    output0 = np.array(output0)
    if output1 is not None:
        output1 = np.array(output1)
    
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
    
    # Step 1: Threshold filtering
    if isinstance(threshold, float):
        expect_boxes = (boxes[:, 1] > threshold) & (boxes[:, 0] > -1)
        boxes = boxes[expect_boxes, :]
    elif isinstance(threshold, dict):
        category_filtered_boxes = []
        for cat_id in np.unique(boxes[:, 0]):
            category_boxes = boxes[boxes[:, 0] == cat_id]
            category_threshold = threshold.get(int(cat_id), 0.5)
            selected_indices = (category_boxes[:, 1] > category_threshold) & (category_boxes[:, 0] > -1)
            category_filtered_boxes.append(category_boxes[selected_indices])
        boxes = np.vstack(category_filtered_boxes) if category_filtered_boxes else np.array([])
    
    if boxes.size == 0:
        return []
    
    if layout_nms:
        selected_indices = nms(boxes[:, :6], iou_same=0.6, iou_diff=0.98)
        boxes = np.array(boxes[selected_indices])
    
    filter_large_image = True
    if filter_large_image and len(boxes) > 1 and boxes.shape[1] in [6, 8]:
        if orig_w > orig_h:
            area_thres = 0.82
        else:
            area_thres = 0.93
        image_index = label_list.index("image") if "image" in label_list else None
        img_area = orig_w * orig_h
        filtered_boxes = []
        for box in boxes:
            label_index, score, xmin, ymin, xmax, ymax = box[:6]
            if label_index == image_index:
                xmin = max(0, xmin)
                ymin = max(0, ymin)
                xmax = min(orig_w, xmax)
                ymax = min(orig_h, ymax)
                box_area = (xmax - xmin) * (ymax - ymin)
                if box_area <= area_thres * img_area:
                    filtered_boxes.append(box)
            else:
                filtered_boxes.append(box)
        if len(filtered_boxes) == 0:
            filtered_boxes = boxes
        boxes = np.array(filtered_boxes)
    
    if layout_merge_bboxes_mode:
        formula_index = label_list.index("formula") if "formula" in label_list else None
        if isinstance(layout_merge_bboxes_mode, str):
            assert layout_merge_bboxes_mode in [
                "union",
                "large",
                "small",
            ], f"The value of `layout_merge_bboxes_mode` must be one of ['union', 'large', 'small'], but got {layout_merge_bboxes_mode}"
            
            if layout_merge_bboxes_mode == "union":
                pass
            else:
                contains_other, contained_by_other = check_containment(
                    boxes[:, :6], formula_index
                )
                if layout_merge_bboxes_mode == "large":
                    boxes = boxes[contained_by_other == 0]
                elif layout_merge_bboxes_mode == "small":
                    boxes = boxes[(contains_other == 0) | (contained_by_other == 1)]
        elif isinstance(layout_merge_bboxes_mode, dict):
            keep_mask = np.ones(len(boxes), dtype=bool)
            for category_index, layout_mode in layout_merge_bboxes_mode.items():
                assert layout_mode in [
                    "union",
                    "large",
                    "small",
                ], f"The value of `layout_merge_bboxes_mode` must be one of ['union', 'large', 'small'], but got {layout_mode}"
                if layout_mode == "union":
                    pass
                else:
                    if layout_mode == "large":
                        contains_other, contained_by_other = check_containment(
                            boxes[:, :6],
                            formula_index,
                            category_index,
                            mode=layout_mode,
                        )
                        keep_mask &= contained_by_other == 0
                    elif layout_mode == "small":
                        contains_other, contained_by_other = check_containment(
                            boxes[:, :6],
                            formula_index,
                            category_index,
                            mode=layout_mode,
                        )
                        keep_mask &= (contains_other == 0) | (contained_by_other == 1)
            boxes = boxes[keep_mask]
    
    if boxes.size == 0:
        return []
    
    if boxes.shape[1] == 8:
        sorted_idx = np.lexsort((-boxes[:, 7], boxes[:, 6]))
        sorted_boxes = boxes[sorted_idx]
        boxes = sorted_boxes[:, :6]
    
    if layout_unclip_ratio:
        if isinstance(layout_unclip_ratio, float):
            layout_unclip_ratio = (layout_unclip_ratio, layout_unclip_ratio)
        elif isinstance(layout_unclip_ratio, (tuple, list)):
            assert (
                len(layout_unclip_ratio) == 2
            ), f"The length of `layout_unclip_ratio` should be 2."
        elif isinstance(layout_unclip_ratio, dict):
            pass
        else:
            raise ValueError(
                f"The type of `layout_unclip_ratio` must be float, Tuple[float, float] or Dict[int, Tuple[float, float]], but got {type(layout_unclip_ratio)}."
            )
        boxes = unclip_boxes(boxes, layout_unclip_ratio)
    
    if boxes.shape[1] == 6:
        boxes = restructured_boxes(boxes, label_list, (orig_w, orig_h))
    elif boxes.shape[1] == 10:
        raise ValueError("Rotated boxes are not supported")
    else:
        raise ValueError(
            f"The shape of boxes should be 6 or 10, instead of {boxes.shape[1]}"
        )
    
    return boxes

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
                        cache_dir=None, precision="fp32"):
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
    result = compiled_model(input_tensors_ov)
    
    # Extract output results
    output = []
    for out in output_tensors:
        output_tensor = result[out]
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
            layout_merge_bboxes_mode=layout_merge_bboxes_mode
        )
    
    # Create result object
    result_obj = LayoutDetectionResult(
        input_path=os.path.abspath(image_path),
        boxes=results,
        page_index=None,
        input_img=image
    )
    
    # Save results
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_obj.save_to_img(save_path=str(output_dir))
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
        precision=args.precision
    )
    
    print(f"\nDetection completed! Results saved to: {args.output_dir}")
    print(f"Detected {len(result_obj.boxes)} valid boxes")
    
    return result_obj

if __name__ == '__main__':
    main()