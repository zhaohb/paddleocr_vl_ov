import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, default="./test_images/doc_test.png", help="输入图片路径")
    parser.add_argument("--output", type=str, default="output", help="输出目录")
    parser.add_argument("--device", type=str, default="GPU", help="设备：CPU/GPU/NPU/AUTO")
    parser.add_argument(
        "--layout-only",
        action="store_true",
        help="只做 layout detection（不加载 VLM，不做 OCR/VLM 推理）",
    )
    parser.add_argument(
        "--layout-precision",
        type=str,
        default="fp16",
        help="layout 模型精度：fp16/fp32/combined_fp16/combined_fp32（layout-only / 完整 pipeline 两种模式都生效）",
    )
    parser.add_argument(
        "--layout-threshold",
        type=float,
        default=0.3,
        help="layout 置信度阈值（layout-only / 完整 pipeline 两种模式都生效）",
    )
    args = parser.parse_args()

    if args.layout_only:
        # 仅布局检测（推荐用于：只想拿到版面框/类型，不需要 VLM 输出）
        from paddleocr_vl_openvino.pp_doclayoutv2 import paddle_ov_doclayout

        print("开始 layout detection...")
        result = paddle_ov_doclayout(
            model_path=None,  # None 表示自动下载/自动选择
            image_path=args.image,
            output_dir=args.output,
            device=args.device,
            threshold=args.layout_threshold,
            precision=args.layout_precision,
        )
        print(f"完成：检测到 {len(result.boxes)} 个区域，输出目录：{args.output}")
        return 0

    # 完整 Pipeline：layout + VLM（OCR / 解析）
    from paddleocr_vl_openvino.paddleocr_vl_pipeline import PaddleOCRVL

    pipeline = PaddleOCRVL(
        layout_model_path="./PP-DocLayoutV3-0125-ov",  
        vlm_model_path="./ov_paddleocr_vl_model", 
        vlm_device=args.device,
        layout_device=args.device,
        layout_precision=args.layout_precision,
        llm_int4_compress=False,
        vision_int8_quant=False,
        llm_int8_compress=False,
        llm_int8_quant=False,
    )

    print("开始识别...")
    output = pipeline.predict(
        args.image,
        layout_threshold=0.3,
        layout_shape_mode="auto",
        use_chart_recognition=False,
        use_seal_recognition=False,
    )

    for res in output:
        # res.print()
        res.save_to_json(save_path=args.output)
        res.save_to_img(save_path=args.output)
        res.save_to_markdown(save_path=args.output, pretty=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())