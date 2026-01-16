from __future__ import annotations

import json
import threading
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional, Tuple, Any

from PySide6.QtCore import QObject, QThread, Signal

from .pdf_utils import is_pdf, render_pdf_to_images
from .types import (
    PipelineInitConfig,
    PredictConfig,
    TaskItem,
    ensure_dir,
    sanitize_stem,
    summarize_result_dict,
    SUPPORTED_IMAGE_EXTS,
)
from .utils_qt import pil_image_to_qimage


_PIPELINE_LOCK = threading.Lock()
_PIPELINE_INSTANCE = None
_PIPELINE_FINGERPRINT: Optional[str] = None


def _pipeline_fingerprint(cfg: PipelineInitConfig) -> str:
    """
    用于判断是否可复用同一个 Pipeline（模型实例）。
    只要初始化相关配置不变，就复用已加载的模型，避免重复下载/初始化。
    """
    d = asdict(cfg)
    # 对路径/字符串做 strip 归一化，避免空格导致误判
    for k, v in list(d.items()):
        if isinstance(v, str):
            d[k] = v.strip()
    return json.dumps(d, ensure_ascii=False, sort_keys=True)


class WorkerSignals(QObject):
    log = Signal(str)
    progress = Signal(int, int)  # current, total
    task_started = Signal(int)  # task_index
    task_finished = Signal(int)  # task_index
    task_failed = Signal(int, str)  # task_index, error
    preview_ready = Signal(int, object, str, str)  # task_index, QImage, markdown, json_text


class InferenceWorker(QThread):
    """
    在后台线程中执行：初始化 pipeline + 逐任务推理。
    """

    def __init__(
        self,
        tasks: List[TaskItem],
        output_root: Path,
        init_cfg: PipelineInitConfig,
        pred_cfg: PredictConfig,
        run_indices: Optional[List[int]] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.signals = WorkerSignals()
        self._tasks = tasks
        self._output_root = output_root
        self._init_cfg = init_cfg
        self._pred_cfg = pred_cfg
        self._run_indices = run_indices[:] if run_indices else None
        self._stop_requested = False

        self._pipeline = None

    def request_stop(self) -> None:
        self._stop_requested = True

    def _log(self, msg: str) -> None:
        self.signals.log.emit(msg)

    def _init_pipeline(self) -> None:
        from paddleocr_vl_openvino.paddleocr_vl_pipeline import PaddleOCRVL

        cfg = self._init_cfg
        fp = _pipeline_fingerprint(cfg)

        global _PIPELINE_INSTANCE, _PIPELINE_FINGERPRINT
        with _PIPELINE_LOCK:
            if _PIPELINE_INSTANCE is not None and _PIPELINE_FINGERPRINT == fp:
                self._pipeline = _PIPELINE_INSTANCE
                self._log("✅ 复用已加载的 Pipeline（模型不再重复初始化）")
                return

            # 配置发生变化或首次加载：重新初始化并替换缓存
            self._log("初始化 Pipeline ...（首次/配置变化时可能会下载模型）")
            pipe = PaddleOCRVL(
                layout_model_path=cfg.layout_model_path or None,
                vlm_model_path=cfg.vlm_model_path or None,
                cache_dir=cfg.cache_dir or None,
                vlm_device=cfg.vlm_device,
                layout_device=cfg.layout_device,
                layout_precision=cfg.layout_precision,
                llm_int4_compress=cfg.llm_int4_compress,
                vision_int8_quant=cfg.vision_int8_quant,
                llm_int8_compress=cfg.llm_int8_compress,
                llm_int8_quant=cfg.llm_int8_quant,
            )
            _PIPELINE_INSTANCE = pipe
            _PIPELINE_FINGERPRINT = fp
            self._pipeline = pipe
            self._log("✅ Pipeline 初始化完成")

    def _pick_vis_image(self, result) -> Optional[object]:
        img_dict = getattr(result, "img", None)
        if not isinstance(img_dict, dict) or not img_dict:
            return None
        if img_dict.get("layout_order_res") is not None:
            return img_dict["layout_order_res"]
        for _, v in img_dict.items():
            if v is not None:
                return v
        return None

    def _extract_texts(self, result) -> Tuple[str, str]:
        markdown_text = ""
        markdown_info = getattr(result, "markdown", None)
        if isinstance(markdown_info, dict):
            markdown_text = markdown_info.get("markdown_texts", "") or ""

        json_text = ""
        json_info = getattr(result, "json", None)
        if isinstance(json_info, dict):
            payload = json_info.get("res", json_info)
            json_text = json.dumps(payload, ensure_ascii=False, indent=2)
        else:
            try:
                json_text = json.dumps(json_info, ensure_ascii=False, indent=2)
            except Exception:
                json_text = str(json_info)

        return markdown_text, json_text

    def _predict_one_image(self, image_path: Path, prompt_label: str = "ocr"):
        assert self._pipeline is not None
        gen = self._pipeline.predict(
            str(image_path),
            use_layout_detection=self._pred_cfg.use_layout_detection,
            layout_threshold=self._pred_cfg.layout_threshold,
            max_new_tokens=self._pred_cfg.max_new_tokens,
            prompt_label=prompt_label,
        )
        output = list(gen)
        if not output:
            raise RuntimeError("未检测到任何内容")
        return output[0]

    def _predict_task(self, task: TaskItem, task_out_dir: Path) -> Tuple[str, str, Optional[object], str, Optional[Any]]:
        """
        返回：markdown_text, json_text, vis_pil_image, summary, result_obj(可选)
        """
        in_path = task.input_path
        if is_pdf(in_path):
            # PDF：渲染为多页图片，再逐页推理，最后拼接 markdown
            pages_dir = task_out_dir / "pages"
            ensure_dir(pages_dir)
            pages = render_pdf_to_images(in_path, pages_dir)
            task.pdf_pages_dir = pages_dir
            if pages:
                task.preview_input_image_path = pages[0].image_path
            md_parts: List[str] = []
            json_pages: List[dict] = []
            first_vis = None
            for p in pages:
                if self._stop_requested:
                    raise RuntimeError("用户停止")
                self._log(f"  - PDF Page {p.page_index + 1}/{len(pages)}: {p.image_path.name}")
                prompt_label = "ocr" if self._pred_cfg.use_layout_detection else (task.task_type or "ocr")
                res = self._predict_one_image(p.image_path, prompt_label=prompt_label)
                md, js = self._extract_texts(res)
                md_parts.append(f"<!-- page {p.page_index + 1} -->\n\n{md}".strip())
                try:
                    json_pages.append(json.loads(js))
                except Exception:
                    json_pages.append({"raw": js})
                if first_vis is None:
                    first_vis = self._pick_vis_image(res)

            markdown_text = "\n\n---\n\n".join([s for s in md_parts if s])
            json_text = json.dumps({"pages": json_pages}, ensure_ascii=False, indent=2)
            summary = f"pdf pages={len(pages)}"
            return markdown_text, json_text, first_vis, summary, None

        # 普通图片
        task.pdf_pages_dir = None
        task.preview_input_image_path = in_path
        prompt_label = "ocr" if self._pred_cfg.use_layout_detection else (task.task_type or "ocr")
        res = self._predict_one_image(in_path, prompt_label=prompt_label)
        markdown_text, json_text = self._extract_texts(res)
        vis_pil = self._pick_vis_image(res)
        try:
            summary = summarize_result_dict(dict(res))
        except Exception:
            summary = "done"
        return markdown_text, json_text, vis_pil, summary, res

    def run(self) -> None:  # noqa: C901
        # 只执行指定索引（重跑所选），或执行 pending 任务（默认运行模式）
        if self._run_indices is not None:
            runnable_indices = [i for i in self._run_indices if 0 <= i < len(self._tasks)]
        else:
            runnable_indices = [i for i, t in enumerate(self._tasks) if t.status not in ("done", "error")]
        total = len(runnable_indices)
        if total == 0:
            self._log("没有待执行（pending）的任务，直接结束。")
            self.signals.progress.emit(0, 0)
            return

        try:
            self._init_pipeline()
        except Exception as e:
            self._log("❌ Pipeline 初始化失败：")
            self._log(str(e))
            self._log(traceback.format_exc())
            for i in runnable_indices:
                self.signals.task_failed.emit(i, f"Pipeline 初始化失败: {e}")
            return

        ensure_dir(self._output_root)

        for step_idx, idx in enumerate(runnable_indices):
            if self._stop_requested:
                self._log("已停止")
                break

            self.signals.progress.emit(step_idx, total)
            self.signals.task_started.emit(idx)
            task = self._tasks[idx]
            try:
                task.status = "running"
                name = sanitize_stem(task.input_path)
                task_out_dir = self._output_root / name
                ensure_dir(task_out_dir)
                task.output_dir = task_out_dir

                self._log(f"开始处理：{task.input_path}")
                md, js, vis_pil, summary, res_obj = self._predict_task(task, task_out_dir)

                # 结果写盘：
                # - 非 PDF：优先调用项目自带的 save_to_markdown/save_to_json（会同时保存 imgs/ 资源）
                # - PDF：使用兜底的 result.md/result.json
                md_path = None
                json_path = None
                if res_obj is not None:
                    try:
                        # pretty=False：尽量输出标准 markdown，便于 Qt QTextBrowser 渲染
                        res_obj.save_to_markdown(task_out_dir, pretty=False)
                        res_obj.save_to_json(task_out_dir)
                        # save_to_markdown/save_to_json 的文件名基于输入 stem
                        stem = task.input_path.stem
                        md_path = task_out_dir / f"{stem}.md"
                        json_path = task_out_dir / f"{stem}_res.json"
                    except Exception as e:
                        # 回退到简单写盘
                        self._log(f"⚠️ 使用 save_to_* 失败，回退到直接写盘: {e}")

                if md_path is None:
                    md_path = task_out_dir / "result.md"
                    md_path.write_text(md or "", encoding="utf-8")
                if json_path is None:
                    json_path = task_out_dir / "result.json"
                    json_path.write_text(js or "{}", encoding="utf-8")

                # 回读生成的文本（保证 UI 侧看到的是最终文件内容）
                try:
                    md = md_path.read_text(encoding="utf-8")
                except Exception:
                    pass
                try:
                    js = json_path.read_text(encoding="utf-8")
                except Exception:
                    pass

                # 可视化输出
                qimg = None
                if vis_pil is not None:
                    qimg = pil_image_to_qimage(vis_pil)
                    vis_path = task_out_dir / "vis.png"
                    try:
                        vis_pil.save(vis_path)
                        task.vis_image_path = vis_path
                    except Exception:
                        pass

                task.status = "done"
                task.summary = summary
                task.markdown_text = md
                task.json_text = js

                if qimg is not None:
                    self.signals.preview_ready.emit(idx, qimg, md, js)
                else:
                    self.signals.preview_ready.emit(idx, None, md, js)

                self._log(f"✅ 完成：{task.input_path} -> {task_out_dir}")
                self.signals.task_finished.emit(idx)
            except Exception as e:
                task.status = "error"
                task.error = f"{e}\n{traceback.format_exc()}"
                self._log(f"❌ 失败：{task.input_path}")
                self._log(task.error)
                self.signals.task_failed.emit(idx, str(e))

        self.signals.progress.emit(total, total)


