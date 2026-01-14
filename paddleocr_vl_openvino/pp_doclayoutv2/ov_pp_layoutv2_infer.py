import openvino as ov
import cv2
import numpy as np
import json
import os
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import mimetypes

def preprocess_image_doclayout(image, target_input_size=(800, 800)):
    """预处理图像：BGR->RGB, resize, 归一化, HWC->CHW"""
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
    """中心点格式转角点格式"""
    if len(boxes.shape) == 2:
        cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    else:
        cx, cy, w, h = boxes[:, :, 0], boxes[:, :, 1], boxes[:, :, 2], boxes[:, :, 3]
    xmin, ymin = cx - w / 2.0, cy - h / 2.0
    xmax, ymax = cx + w / 2.0, cy + h / 2.0
    return np.stack([xmin, ymin, xmax, ymax], axis=-1)

def iou(box1, box2):
    """计算IoU"""
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
    """非极大值抑制"""
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
    """检查包含关系"""
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
    """检查框包含关系"""
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
    """扩展检测框坐标"""
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
    """格式化输出"""
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
    """绘制检测框"""
    try:
        font_size = int(0.018 * img.width) + 2
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
            "C:/Windows/Fonts/simhei.ttf",  # 黑体
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
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
    """布局检测结果类"""
    
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

def postprocess_detections_detr(output, scale_h, scale_w, orig_h, orig_w, threshold=0.5,
                                layout_nms=False, layout_unclip_ratio=None, layout_merge_bboxes_mode=None):
    """后处理DETR检测结果"""
    if isinstance(output, (list, tuple)):
        output0, output1 = output[0], output[1] if len(output) > 1 else None
    else:
        output0, output1 = output, None
    
    output0 = np.array(output0)
    if output1 is not None:
        output1 = np.array(output1)
    
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
        raise ValueError(f"无法处理输出格式，output[0] 形状 {output0.shape}")
    
    # 步骤1: 阈值过滤
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

def paddle_ov_doclayout(model_path, image_path, output_dir, device="GPU", threshold=0.5, 
                        layout_nms=True, layout_unclip_ratio=None, layout_merge_bboxes_mode=None):
    """
    使用 OpenVINO 进行布局检测推理
    
    Args:
        model_path: OpenVINO IR 模型路径 (.xml 文件)
        image_path: 输入图像路径
        output_dir: 输出保存目录
        device: 推理设备 ("CPU", "GPU", "AUTO")
        threshold: 检测阈值
        layout_nms: 是否启用 NMS
        layout_unclip_ratio: 坐标扩展比例
        layout_merge_bboxes_mode: 布局框合并模式
    
    Returns:
        LayoutDetectionResult: 检测结果对象
    """
    # 初始化 OpenVINO Core
    core = ov.Core()
    
    # 加载模型（.xml 文件会自动找到对应的 .bin 文件）
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

    # 编译模型
    compiled_model = core.compile_model(model, device)
    
    # 获取输入和输出信息
    input_tensors = compiled_model.inputs
    output_tensors = compiled_model.outputs
    
    print(f"模型输入数量: {len(input_tensors)}")
    for i, inp in enumerate(input_tensors):
        shape_str = str(inp.partial_shape) if inp.partial_shape.is_dynamic else str(inp.shape)
        print(f"  输入 {i}: {inp.get_any_name()}, shape: {shape_str}, type: {inp.element_type}")
    
    print(f"模型输出数量: {len(output_tensors)}")
    for i, out in enumerate(output_tensors):
        shape_str = str(out.partial_shape) if out.partial_shape.is_dynamic else str(out.shape)
        print(f"  输出 {i}: {out.get_any_name()}, shape: {shape_str}, type: {out.element_type}")
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"无法读取 {image_path}")
    
    orig_h, orig_w = image.shape[:2]
    input_blob, scale_h, scale_w = preprocess_image_doclayout(image)
    
    # 准备输入数据
    input_data = {}
    for inp in input_tensors:
        inp_name = inp.get_any_name()
        if inp_name == "im_shape":
            input_data[inp_name] = np.array([800, 800], dtype=np.float32)[np.newaxis, ...]
        elif inp_name == "image":
            input_data[inp_name] = input_blob
        elif inp_name == "scale_factor":
            input_data[inp_name] = np.array([[scale_h, scale_w]], dtype=np.float32)
        else:
            pass
    
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
    result = compiled_model(input_tensors_ov)
    
    # 提取输出结果
    output = []
    for out in output_tensors:
        output_tensor = result[out]
        output_data = output_tensor.data
        output.append(output_data)
    
    # 后处理
    if layout_unclip_ratio is None:
        layout_unclip_ratio = [1.0, 1.0]
    
    results = postprocess_detections_detr(
        output, scale_h, scale_w, orig_h, orig_w,
        threshold=threshold,
        layout_nms=layout_nms,
        layout_unclip_ratio=layout_unclip_ratio,
        layout_merge_bboxes_mode=layout_merge_bboxes_mode
    )
    
    # 创建结果对象
    result_obj = LayoutDetectionResult(
        input_path=os.path.abspath(image_path),
        boxes=results,
        page_index=None,
        input_img=image
    )
    
    # 保存结果
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_obj.save_to_img(save_path=str(output_dir))
    result_obj.save_to_json(save_path=str(output_dir / "res.json"))
    
    return result_obj

def main():
    """主函数：解析命令行参数并执行推理"""
    parser = argparse.ArgumentParser(description="PP-DocLayoutV2 OpenVINO 推理脚本")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="OpenVINO IR 模型路径 (.xml 文件)"
    )
    parser.add_argument(
        "--image_path",
        type=str,
        required=True,
        help="输入图像路径"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output_ov",
        help="输出保存目录 (默认: ./output_ov)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="GPU",
        choices=["CPU", "GPU", "NPU", "AUTO"],
        help="推理设备 (默认: GPU)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="检测阈值 (默认: 0.5)"
    )
    
    args = parser.parse_args()
    
    # 执行推理（使用原来的默认值）
    result_obj = paddle_ov_doclayout(
        model_path=args.model_path,
        image_path=args.image_path,
        output_dir=args.output_dir,
        device=args.device,
        threshold=args.threshold,
        layout_nms=True,
        layout_unclip_ratio=None,
        layout_merge_bboxes_mode=None
    )
    
    print(f"\n检测完成！结果已保存到: {args.output_dir}")
    print(f"检测到 {len(result_obj.boxes)} 个有效框")
    
    return result_obj

if __name__ == '__main__':
    main()