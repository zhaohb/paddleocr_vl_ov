"""
PDF -> 图片 -> PaddleOCR-VL Pipeline 跨页动态批处理识别脚本

用法:
    python pdf_ocr.py --pdf input.pdf
    python pdf_ocr.py --pdf input.pdf --early-stop-ratio 0.7
"""

import argparse
import time
from pathlib import Path

import cv2
import fitz  # PyMuPDF


def render_pdf_to_images(pdf_path: Path, out_dir: Path, dpi: int = 300):
    """将 PDF 每一页渲染为 PNG 图片，返回图片路径列表。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf_path))
    image_paths = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img_path = out_dir / f"page_{i + 1:04d}.png"
        pix.save(str(img_path))
        image_paths.append(img_path)
        print(f"  [PDF] 第 {i + 1}/{doc.page_count} 页 -> {img_path}")
    doc.close()
    return image_paths


def main():
    parser = argparse.ArgumentParser(
        description="PDF 跨页动态批处理 OCR（统一 batch + early-stop backfill）"
    )
    parser.add_argument("--pdf", type=str, required=True, help="输入 PDF 文件路径")
    parser.add_argument("--output", type=str, default="pdf_output", help="输出目录（默认 pdf_output）")
    parser.add_argument("--dpi", type=int, default=100, help="PDF 渲染 DPI（默认 100）")
    parser.add_argument("--device", type=str, default="GPU", help="VLM 推理设备：CPU/GPU/AUTO")
    parser.add_argument("--layout-device", type=str, default="CPU", help="Layout 检测设备（默认 CPU）")
    parser.add_argument("--vlm-batch-size", type=int, default=40, help="VLM batch size（默认 40）")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="最大生成 token 数（默认 1024）")
    parser.add_argument("--layout-model-path", type=str, default=None, help="Layout 模型路径（None 自动下载）")
    parser.add_argument("--vlm-model-path", type=str, default=None, help="VLM 模型路径（None 自动下载）")
    parser.add_argument("--layout-threshold", type=float, default=0.4, help="Layout 检测阈值（默认 0.4）")
    parser.add_argument("--window-pages", type=int, default=10,
                        help="每次跨页批处理的最大页数（默认 10，0 表示全部页一起处理）")
    parser.add_argument("--early-stop-ratio", type=float, default=0.0,
                        help="Batch early termination ratio (0=disabled, 0.7=stop when 70%% done, default 0)")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        print(f"错误: 找不到 PDF 文件: {pdf_path}")
        return 1

    output_dir = Path(args.output) / pdf_path.stem
    images_dir = output_dir / "images"

    # ── 1. PDF 转图片 ──────────────────────────────────────────────────────
    print("=" * 80)
    print(f"步骤 1: 将 PDF 转为图片 (DPI={args.dpi})")
    print("=" * 80)
    t0 = time.time()
    image_paths = render_pdf_to_images(pdf_path, images_dir, dpi=args.dpi)
    n_pages = len(image_paths)
    print(f"\n共 {n_pages} 页，转换耗时: {time.time() - t0:.2f} 秒")

    # ── 2. 初始化 Pipeline ─────────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("步骤 2: 初始化 PaddleOCR-VL Pipeline")
    print("=" * 80)

    from paddleocr_vl_openvino.paddleocr_vl_pipeline import PaddleOCRVL
    from paddleocr_vl_openvino.paddleocr_vl_pipeline.ov_paddleocr_vl_pipeline import (
        PaddleOCRVLResult,
    )

    t0 = time.time()
    pipeline = PaddleOCRVL(
        layout_model_path=args.layout_model_path,
        vlm_model_path=args.vlm_model_path,
        vlm_device=args.device,
        layout_device=args.layout_device,
        layout_precision="fp16",
        llm_int4_compress=False,
        vision_int8_quant=False,
        llm_int8_compress=False,
        llm_int8_quant=False,
    )
    print(f"模型初始化耗时: {time.time() - t0:.2f} 秒")

    import math
    from concurrent.futures import ThreadPoolExecutor, Future

    window = args.window_pages if args.window_pages > 0 else n_pages
    n_windows = math.ceil(n_pages / window)

    # 判断 CV 和 VLM 是否在同一 GPU 上——若是则串行，否则并行
    def _normalize_device(d):
        d = d.strip().upper()
        if d in ("CPU", "AUTO"):
            return d
        # GPU / GPU.0 / GPU.1 ...
        return d if "." in d else d + ".0"

    cv_device_norm = _normalize_device(args.layout_device)
    vlm_device_norm = _normalize_device(args.device)
    cv_parallel = (cv_device_norm != vlm_device_norm)
    if not cv_parallel:
        print(f"  [注意] CV ({args.layout_device}) 与 VLM ({args.device}) "
              f"在同一设备，CV 改为串行模式")
    else:
        print(f"  [注意] CV ({args.layout_device}) 与 VLM ({args.device}) "
              f"在不同设备，CV 使用并行预取")

    # ── CV 检测函数（可在后台线程运行）──────────────────────────────────────
    def run_cv_for_window(win_start, win_end):
        """对 [win_start, win_end) 范围的页面执行布局检测，返回 CV 结果 dict。"""
        _t_cv_wall = time.time()
        win_pages_local = image_paths[win_start:win_end]

        images, layouts, imgs_doc, paths, preproc_res = [], [], [], [], []
        blocks = 0
        for local_idx, img_path in enumerate(win_pages_local):
            global_idx = win_start + local_idx
            image = cv2.imread(str(img_path))
            if image is None:
                raise ValueError(f"无法加载图片: {img_path}")

            _t = time.time()
            results_cv = pipeline._process_cv(
                image,
                str(img_path),
                page_index=global_idx,
                use_layout_detection=True,
                layout_threshold=args.layout_threshold,
            )
            _t = time.time() - _t

            n_blk = len(results_cv["layout_det_results"][0].get("boxes", []))
            blocks += n_blk
            print(f"  [CV] 第 {global_idx + 1}/{n_pages} 页: {n_blk} 个块, {_t:.3f}s")

            images.append(results_cv["doc_preprocessor_image"])
            layouts.append(results_cv["layout_det_results"][0])
            imgs_doc.append(results_cv["imgs_in_doc"][0])
            paths.append(str(img_path))
            preproc_res.append(results_cv["doc_preprocessor_res"])

        return {
            "images": images, "layouts": layouts, "imgs_doc": imgs_doc,
            "paths": paths, "preproc_res": preproc_res, "blocks": blocks,
            "t_cv_elapsed": time.time() - _t_cv_wall,
        }

    # ── 3–5. 流水线处理：窗口 N 的 VLM 与窗口 N+1 的 CV 并行 ─────────────
    result_dir = output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    t_cv_total = 0.0
    t_vlm_total = 0.0
    total_blocks = 0
    page_block_counts = []

    executor = ThreadPoolExecutor(max_workers=1) if cv_parallel else None

    t_process_start = time.time()  # 挂钟起点

    # 预取第一个窗口的 CV
    print("\n" + "=" * 80)
    print(f"窗口 1/{n_windows}: 第 1–{min(window, n_pages)} 页 "
          f"(共 {min(window, n_pages)} 页, batch_size={args.vlm_batch_size})")
    print("=" * 80)
    t_cv_start = time.time()
    cv_result = run_cv_for_window(0, min(window, n_pages))
    t_cv_win = time.time() - t_cv_start
    t_cv_total += t_cv_win
    print(f"  [CV] 窗口小计: {min(window, n_pages)} 页, {cv_result['blocks']} 块, {cv_result['t_cv_elapsed']:.2f}s")

    for win_idx in range(n_windows):
        win_start = win_idx * window
        win_end = min(win_start + window, n_pages)
        win_size = win_end - win_start

        # cv_result 已经准备好（第一轮是同步预取，后续轮次来自并行预取）
        win_images = cv_result["images"]
        win_layout_det_results = cv_result["layouts"]
        win_imgs_in_doc = cv_result["imgs_doc"]
        win_input_paths = cv_result["paths"]
        win_doc_preprocessor_res = cv_result["preproc_res"]
        win_blocks = cv_result["blocks"]
        total_blocks += win_blocks
        page_block_counts.extend(
            len(l.get("boxes", [])) for l in win_layout_det_results
        )

        # 启动下一个窗口的 CV 预取（并行模式）或跳过（串行模式）
        next_cv_future: Future = None
        if win_idx + 1 < n_windows and cv_parallel:
            next_start = (win_idx + 1) * window
            next_end = min(next_start + window, n_pages)
            print(f"\n  [预取] 后台启动窗口 {win_idx + 2}/{n_windows} 的 CV "
                  f"(第 {next_start + 1}–{next_end} 页)")
            next_cv_future = executor.submit(run_cv_for_window, next_start, next_end)

        # ── 4. 统一 VLM 推理（所有块一起进 batch，early-stop 自动处理快慢差异）
        t_vlm_start = time.time()

        print(f"  [VLM] {win_blocks} 块, bs={args.vlm_batch_size}, "
              f"early_stop={args.early_stop_ratio:.2f}")

        _t = time.time()
        parsing_res_lists, table_res_lists, spotting_res_lists, imgs_in_doc_out = (
            pipeline.get_layout_parsing_results(
                win_images,
                win_layout_det_results,
                win_imgs_in_doc,
                max_new_tokens=args.max_new_tokens,
                use_chart_recognition=pipeline.use_chart_recognition,
                use_seal_recognition=pipeline.use_seal_recognition,
                use_ocr_for_image_block=pipeline.use_ocr_for_image_block,
                layout_shape_mode="auto",
                vlm_batch_size=args.vlm_batch_size,
                early_stop_ratio=args.early_stop_ratio,
            )
        )
        print(f"  [VLM] {win_blocks} 块, {time.time() - _t:.2f}s")

        t_vlm_win = time.time() - t_vlm_start
        t_vlm_total += t_vlm_win
        print(f"  [VLM] 窗口小计: {win_blocks} 块, {t_vlm_win:.2f}s")

        # ── 5. 按页恢复结果并保存 ─────────────────────────────────────────
        for i in range(win_size):
            global_idx = win_start + i
            parsing_res_list = parsing_res_lists[i] if i < len(parsing_res_lists) else []
            table_res_list = table_res_lists[i] if i < len(table_res_lists) else []
            spotting_res = spotting_res_lists[i] if i < len(spotting_res_lists) else {}
            imgs_page = imgs_in_doc_out[i] if i < len(imgs_in_doc_out) else []

            single_img_res = {
                "input_path": win_input_paths[i],
                "page_index": global_idx,
                "page_count": n_pages,
                "width": win_images[i].shape[1],
                "height": win_images[i].shape[0],
                "doc_preprocessor_res": win_doc_preprocessor_res[i],
                "layout_det_res": win_layout_det_results[i],
                "table_res_list": table_res_list,
                "parsing_res_list": parsing_res_list,
                "spotting_res": spotting_res,
                "imgs_in_doc": imgs_page,
                "model_settings": {
                    "use_doc_preprocessor": False,
                    "use_layout_detection": pipeline.use_layout_detection,
                    "use_chart_recognition": pipeline.use_chart_recognition,
                    "use_seal_recognition": pipeline.use_seal_recognition,
                    "use_ocr_for_image_block": pipeline.use_ocr_for_image_block,
                    "format_block_content": False,
                    "merge_layout_blocks": pipeline.merge_layout_blocks,
                    "markdown_ignore_labels": pipeline.markdown_ignore_labels,
                    "return_layout_polygon_points": True,
                },
            }

            result = PaddleOCRVLResult(single_img_res)
            result.save_to_json(save_path=str(result_dir))
            result.save_to_markdown(save_path=str(result_dir))
            print(f"  [保存] 第 {global_idx + 1}/{n_pages} 页: {len(parsing_res_list)} 个块")

        # 窗口处理完毕，释放本窗口数据
        del win_images, win_layout_det_results, win_imgs_in_doc
        del win_doc_preprocessor_res, parsing_res_lists, table_res_lists
        del spotting_res_lists, imgs_in_doc_out

        # 等待下一窗口 CV 完成（并行模式）或同步执行（串行模式）
        if next_cv_future is not None:
            cv_result = next_cv_future.result()
            t_next_cv = cv_result["t_cv_elapsed"]
            t_cv_total += t_next_cv
            next_win_size = min((win_idx + 2) * window, n_pages) - (win_idx + 1) * window
            print(f"\n" + "=" * 80)
            print(f"窗口 {win_idx + 2}/{n_windows}: 第 {(win_idx+1)*window + 1}–"
                  f"{min((win_idx+2)*window, n_pages)} 页 "
                  f"(共 {next_win_size} 页, batch_size={args.vlm_batch_size})")
            print("=" * 80)
            print(f"  [CV] 窗口小计: {next_win_size} 页, {cv_result['blocks']} 块, "
                  f"{t_next_cv:.2f}s (与上轮 VLM 并行)")
        elif win_idx + 1 < n_windows:
            # 串行模式：VLM 完成后再做下一窗口的 CV
            next_start = (win_idx + 1) * window
            next_end = min(next_start + window, n_pages)
            next_win_size = next_end - next_start
            print(f"\n" + "=" * 80)
            print(f"窗口 {win_idx + 2}/{n_windows}: 第 {next_start + 1}–"
                  f"{next_end} 页 "
                  f"(共 {next_win_size} 页, batch_size={args.vlm_batch_size})")
            print("=" * 80)
            cv_result = run_cv_for_window(next_start, next_end)
            t_next_cv = cv_result["t_cv_elapsed"]
            t_cv_total += t_next_cv
            print(f"  [CV] 窗口小计: {next_win_size} 页, {cv_result['blocks']} 块, "
                  f"{t_next_cv:.2f}s")

    if executor is not None:
        executor.shutdown(wait=False)

    # ── 6. 汇总统计 ───────────────────────────────────────────────────────
    t_process_wall = time.time() - t_process_start  # 实际挂钟耗时
    t_overlap = t_cv_total + t_vlm_total - t_process_wall  # 并行重叠部分
    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)
    print(f"总页数: {n_pages}  (窗口大小: {window} 页, 共 {n_windows} 个窗口)")
    print(f"总块数: {total_blocks}")
    print(f"布局检测 (CV):  {t_cv_total:.2f} 秒 (含并行)")
    print(f"VLM 推理:       {t_vlm_total:.2f} 秒")
    print(f"CV/VLM 重叠:    {max(t_overlap, 0):.2f} 秒")
    print(f"总识别耗时:     {t_process_wall:.2f} 秒 (实际)")
    print(f"平均每页:       {t_process_wall / n_pages:.2f} 秒")
    if total_blocks > 0:
        print(f"平均每块 VLM:   {t_vlm_total / total_blocks:.3f} 秒")
    # 批处理填充率统计
    bs = args.vlm_batch_size
    batches_cross_page = sum(
        math.ceil(sum(page_block_counts[w * window: min((w + 1) * window, n_pages)]) / bs)
        for w in range(n_windows)
    ) if bs > 0 else total_blocks
    batches_per_page = sum(
        max(1, math.ceil(n / bs)) for n in page_block_counts
    ) if bs > 0 else total_blocks
    print(f"跨页批次数:     {batches_cross_page} (逐页模式: {batches_per_page})")
    saved_pct = (1 - batches_cross_page / max(batches_per_page, 1)) * 100
    print(f"批次节省:       {saved_pct:.1f}%")
    print(f"图片目录: {images_dir}")
    print(f"结果目录: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
