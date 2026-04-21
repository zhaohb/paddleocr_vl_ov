"""
PDF -> 图片 -> PaddleOCR-VL Pipeline 批量识别脚本

用法:
    python pdf_ocr.py --pdf input.pdf
    python pdf_ocr.py --pdf input.pdf --output result --dpi 200 --device GPU
"""

import argparse
import time
from pathlib import Path

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
    parser = argparse.ArgumentParser(description="PDF 转图片并调用 PaddleOCR-VL Pipeline 识别")
    parser.add_argument("--pdf", type=str, required=True, help="输入 PDF 文件路径")
    parser.add_argument("--output", type=str, default="pdf_output", help="输出目录（默认 pdf_output）")
    parser.add_argument("--dpi", type=int, default=100, help="PDF 渲染 DPI（默认 100）")
    parser.add_argument("--device", type=str, default="GPU", help="VLM 推理设备：CPU/GPU/AUTO")
    parser.add_argument("--layout-device", type=str, default="CPU", help="Layout 检测设备（默认 CPU）")
    parser.add_argument("--vlm-batch-size", type=int, default=30, help="VLM batch size（默认 30）")
    parser.add_argument("--max-new-tokens", type=int, default=1024, help="最大生成 token 数（默认 1024）")
    parser.add_argument("--layout-model-path", type=str, default=None, help="Layout 模型路径（None 自动下载）")
    parser.add_argument("--vlm-model-path", type=str, default=None, help="VLM 模型路径（None 自动下载）")
    parser.add_argument("--layout-threshold", type=float, default=0.4, help="Layout 检测阈值（默认 0.4）")
    args = parser.parse_args()

    pdf_path = Path(args.pdf)
    if not pdf_path.is_file():
        print(f"错误: 找不到 PDF 文件: {pdf_path}")
        return 1

    output_dir = Path(args.output) / pdf_path.stem
    images_dir = output_dir / "images"

    # ── 1. PDF 转图片 ──
    print("=" * 80)
    print(f"步骤 1: 将 PDF 转为图片 (DPI={args.dpi})")
    print("=" * 80)
    t0 = time.time()
    image_paths = render_pdf_to_images(pdf_path, images_dir, dpi=args.dpi)
    print(f"\n共 {len(image_paths)} 页，转换耗时: {time.time() - t0:.2f} 秒")

    # ── 2. 初始化 Pipeline ──
    print("\n" + "=" * 80)
    print("步骤 2: 初始化 PaddleOCR-VL Pipeline")
    print("=" * 80)

    from paddleocr_vl_openvino.paddleocr_vl_pipeline import PaddleOCRVL

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

    # ── 3. 逐页识别 ──
    print("\n" + "=" * 80)
    print("步骤 3: 逐页识别")
    print("=" * 80)

    result_dir = output_dir / "results"
    result_dir.mkdir(parents=True, exist_ok=True)

    total_start = time.time()
    for idx, img_path in enumerate(image_paths):
        page_num = idx + 1
        print(f"\n--- 第 {page_num}/{len(image_paths)} 页: {img_path.name} ---")

        page_start = time.time()
        results = list(pipeline.predict(
            str(img_path),
            vlm_batch_size=args.vlm_batch_size,
            max_new_tokens=args.max_new_tokens,
            layout_threshold=args.layout_threshold,
        ))
        page_time = time.time() - page_start

        for res in results:
            res.print()
            res.save_to_json(save_path=str(result_dir))
            res.save_to_markdown(save_path=str(result_dir))

        print(f"第 {page_num} 页识别耗时: {page_time:.2f} 秒")

    total_time = time.time() - total_start
    print("\n" + "=" * 80)
    print("完成!")
    print("=" * 80)
    print(f"总页数: {len(image_paths)}")
    print(f"总识别耗时: {total_time:.2f} 秒")
    print(f"平均每页: {total_time / len(image_paths):.2f} 秒")
    print(f"图片目录: {images_dir}")
    print(f"结果目录: {result_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
