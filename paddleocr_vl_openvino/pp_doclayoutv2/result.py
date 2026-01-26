# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import json
import mimetypes
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


def draw_box(img: Image.Image, boxes: List[dict]) -> Image.Image:
    """
    Args:
        img (PIL.Image.Image): PIL image
        boxes (list): a list of dictionaries representing detection box information.
    Returns:
        img (PIL.Image.Image): visualized image
    """
    font_size = int(0.018 * int(img.width)) + 2
    # Windows 环境优先用雅黑等；否则 fallback 默认字体
    font = None
    for fp in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"):
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size, encoding="utf-8")
                break
            except Exception:
                continue
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None

    draw_thickness = int(max(img.size) * 0.002)
    draw = ImageDraw.Draw(img)
    label2color = {}
    # 固定调色板（去掉 PaddleX colormap 依赖）
    color_list = [
        (128, 64, 128), (232, 35, 244), (70, 70, 70), (156, 102, 102), (153, 153, 190),
        (153, 153, 153), (30, 170, 250), (0, 220, 220), (35, 142, 107), (152, 251, 152),
        (180, 130, 70), (60, 20, 220), (0, 0, 255), (142, 0, 0), (70, 0, 0),
        (100, 60, 0), (90, 0, 0), (230, 0, 0), (32, 11, 119), (0, 74, 111), (81, 0, 81),
    ]

    for i, dt in enumerate(boxes):
        label, bbox, score = dt["label"], dt["coordinate"], dt["score"]
        if label not in label2color:
            color_index = i % len(color_list)
            label2color[label] = color_list[color_index]
        color = tuple(label2color[label])
        font_color = (255, 255, 255)

        if len(bbox) == 4:
            # draw bbox of normal object detection
            xmin, ymin, xmax, ymax = bbox
            rectangle = [
                (xmin, ymin),
                (xmin, ymax),
                (xmax, ymax),
                (xmax, ymin),
                (xmin, ymin),
            ]
        else:
            raise ValueError(
                f"Only support bbox format of [xmin,ymin,xmax,ymax] or [x1,y1,x2,y2,x3,y3,x4,y4], got bbox of shape {len(bbox)}."
            )

        # draw bbox
        draw.line(
            rectangle,
            width=draw_thickness,
            fill=color,
        )

        # draw label
        text = "{} {:.2f}".format(dt["label"], float(score))
        if tuple(map(int, PIL.__version__.split("."))) <= (10, 0, 0):
            tw, th = draw.textsize(text, font=font)
        else:
            left, top, right, bottom = draw.textbbox((0, 0), text, font)
            tw, th = right - left, bottom - top + 4
        if ymin < th:
            draw.rectangle([(xmin, ymin), (xmin + tw + 4, ymin + th + 1)], fill=color)
            draw.text((xmin + 2, ymin - 2), text, fill=font_color, font=font)
        else:
            draw.rectangle([(xmin, ymin - th), (xmin + tw + 4, ymin + 1)], fill=color)
            draw.text((xmin + 2, ymin - th - 2), text, fill=font_color, font=font)

        text_position = (bbox[2] + 2, bbox[1] - font_size // 2)
        if int(img.width) - bbox[2] < font_size:
            text_position = (
                int(bbox[2] - font_size * 1.1),
                bbox[1] - font_size // 2,
            )
        # 对齐 PaddleX：优先显示阅读顺序 order；若不存在则回退到 boxes 下标（i+1）
        show_id = dt.get("order", None)
        if show_id is None:
            show_text = str(i + 1)
        else:
            try:
                show_text = str(int(show_id))
            except Exception:
                show_text = str(show_id)
        draw.text(text_position, show_text, font=font, fill="red")

    return img


def restore_to_draw_masks(img_size, boxes):
    """
    Restores extracted masks to the original shape and draws them on a blank image.

    """
    if cv2 is None:
        raise ImportError("cv2 is required for draw_mask (opencv-contrib-python)")

    restored_masks = []

    for i, box_info in enumerate(boxes):
        restored_mask = np.zeros(img_size, dtype=np.uint8)
        polygon = np.array(box_info["polygon_points"], dtype=np.int32)
        polygon = polygon.reshape((-1, 1, 2))  # shape: (N, 1, 2)
        cv2.fillPoly(restored_mask, [polygon], 1)
        restored_masks.append(restored_mask)

    return np.array(restored_masks)


def draw_mask(im, boxes, img_size):
    """
    Args:
        im (PIL.Image.Image): PIL image
        boxes (list): a list of dicts representing detection box information.
    Returns:
        img (PIL.Image.Image): visualized image
    """
    # 固定调色板（去掉 PaddleX colormap 依赖）
    color_list = [
        (128, 64, 128), (232, 35, 244), (70, 70, 70), (156, 102, 102), (153, 153, 190),
        (153, 153, 153), (30, 170, 250), (0, 220, 220), (35, 142, 107), (152, 251, 152),
        (180, 130, 70), (60, 20, 220), (0, 0, 255), (142, 0, 0), (70, 0, 0),
        (100, 60, 0), (90, 0, 0), (230, 0, 0), (32, 11, 119), (0, 74, 111), (81, 0, 81),
    ]
    alpha = 0.5

    im = np.array(im).astype("float32")
    clsid2color = {}

    np_masks = restore_to_draw_masks(img_size, boxes)
    im_h, im_w = im.shape[:2]
    np_masks = np_masks[:, :im_h, :im_w]

    # draw mask
    for i, mask in enumerate(np_masks):
        clsid = int(boxes[i]["cls_id"])
        if clsid not in clsid2color:
            color_index = i % len(color_list)
            clsid2color[clsid] = np.array(color_list[color_index])
        color_mask = clsid2color[clsid]
        idx = np.nonzero(mask)
        im[idx[0], idx[1], :] = (1.0 - alpha) * im[
            idx[0], idx[1], :
        ] + alpha * color_mask

    img = Image.fromarray(np.uint8(im))
    font_size = int(0.018 * img.width) + 2
    font = None
    for fp in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/simhei.ttf", "C:/Windows/Fonts/simsun.ttc"):
        if os.path.exists(fp):
            try:
                font = ImageFont.truetype(fp, font_size, encoding="utf-8")
                break
            except Exception:
                continue
    if font is None:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
    draw = ImageDraw.Draw(img)
    label2color = {}

    for i, box_info in enumerate(boxes):
        label = box_info["label"]
        score = box_info["score"]
        if label not in label2color:
            color_index = i % len(color_list)
            label2color[label] = color_list[color_index]
        color = tuple(label2color[label])
        font_color = (255, 255, 255)

        polygon_points = box_info["polygon_points"]
        # 选点策略对齐 PaddleX：分别取离图像左上角/右上角最近的点
        image_left_top = (0, 0)
        image_right_top = (img.width, 0)
        left_top = min(
            polygon_points,
            key=lambda p: (
                (p[0] - image_left_top[0]) ** 2 + (p[1] - image_left_top[1]) ** 2
            ),
        )
        right_top = min(
            polygon_points,
            key=lambda p: (
                (p[0] - image_right_top[0]) ** 2 + (p[1] - image_right_top[1]) ** 2
            ),
        )

        # label
        text = "{} {:.2f}".format(label, score)
        if tuple(map(int, PIL.__version__.split("."))) <= (10, 0, 0):
            tw, th = draw.textsize(text, font=font)
        else:
            left, top, right, bottom = draw.textbbox((0, 0), text, font)
            tw, th = right - left, bottom - top + 4
        lx, ly = left_top
        if ly < th:
            draw.rectangle([(lx, ly), (lx + tw + 4, ly + th + 1)], fill=color)
            draw.text((lx + 2, ly - 2), text, fill=font_color, font=font)
        else:
            draw.rectangle([(lx, ly - th), (lx + tw + 4, ly + 1)], fill=color)
            draw.text((lx + 2, ly - th - 2), text, fill=font_color, font=font)

        # order
        order = box_info.get("order", None)
        if order:
            order_text = str(order)
            rx, ry = right_top
            text_position = (rx + 2, ry - font_size // 2)
            if int(img.width) - rx < font_size:
                text_position = (
                    int(rx - font_size * 1.1),
                    ry - font_size // 2,
                )
            draw.text(text_position, order_text, font=font, fill="red")

    return img


class LayoutAnalysisResult(dict):
    """
    自包含结果类型（去掉 BaseCVResult/JsonMixin 依赖）。
    期望字段：
    - input_path
    - page_index
    - input_img (BGR np.ndarray)
    - boxes (list[dict])
    """

    def _to_img(self) -> dict:
        boxes = self.get("boxes", [])
        input_img = self.get("input_img", None)
        if input_img is None:
            raise ValueError("input_img is required for visualization")
        image = Image.fromarray(input_img[..., ::-1])
        ori_img_size = list(image.size)[::-1]
        # Align with PaddleX intent: if polygon_points exist, visualize as masks (tilted polygons).
        # Some pipelines may have polygon_points for only a subset; fill missing ones with rect corners
        # so we can still draw masks consistently.
        has_any_poly = any(
            isinstance(b, dict) and b.get("polygon_points", None) is not None for b in (boxes or [])
        )
        if has_any_poly:
            filled_boxes = []
            for b in boxes:
                if not isinstance(b, dict):
                    continue
                if b.get("polygon_points", None) is None:
                    coord = b.get("coordinate", None) or [0, 0, 0, 0]
                    x1, y1, x2, y2 = coord
                    bb = dict(b)
                    bb["polygon_points"] = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    filled_boxes.append(bb)
                else:
                    filled_boxes.append(b)
            image = draw_mask(image, filled_boxes, ori_img_size)
        else:
            image = draw_box(image, boxes)
        return {"res": image}

    def _to_json(self) -> dict:
        def _format_data_for_json(obj):
            # numpy scalar
            if isinstance(obj, np.generic):
                return obj.item()
            # numpy array
            if isinstance(obj, np.ndarray):
                return [_format_data_for_json(x) for x in obj.tolist()]
            # pathlib path
            if isinstance(obj, Path):
                return obj.as_posix()
            # dict / list / tuple
            if isinstance(obj, dict):
                return {k: _format_data_for_json(v) for k, v in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_format_data_for_json(x) for x in obj]
            return obj

        data = copy.deepcopy(self)
        data.pop("input_img", None)
        data = _format_data_for_json(data)
        return {"res": data}

    @property
    def img(self) -> dict:
        return self._to_img()

    @property
    def json(self) -> dict:
        return self._to_json()

    def save_to_img(self, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        img_dict = self._to_img()
        img_dict["res"].save(save_path)

    def save_to_json(self, save_path, indent: int = 4, ensure_ascii: bool = False):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(self._to_json()["res"], f, indent=indent, ensure_ascii=ensure_ascii)
