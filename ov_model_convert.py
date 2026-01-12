import argparse

from ov_paddleocr_vl import PaddleOCR_VL_OV


def parse_args():
    """Parse CLI arguments for PaddleOCR_VL_OV export."""
    parser = argparse.ArgumentParser(
        description="Convert PaddleOCR-VL model to OpenVINO format.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 使用默认参数转换模型
  python ov_model_convert.py

  # 指定模型路径
  python ov_model_convert.py --pretrained_model_path ./PaddleOCR-VL --ov_model_path ./ov_output

  # 使用GPU设备并启用INT8压缩
  python ov_model_convert.py --device gpu --llm_int8_compress --vision_int8_quant

  # 使用GPU设备并启用INT4压缩（更小的模型，但可能影响精度）
  python ov_model_convert.py --device gpu --llm_int4_compress --vision_int8_quant

  # 完整参数示例（INT8压缩）
  python ov_model_convert.py \\
    --pretrained_model_path ./PaddleOCR-VL \\
    --ov_model_path ./ov_paddleocr_vl_model \\
    --device gpu \\
    --llm_int8_compress \\
    --vision_int8_quant

  # 完整参数示例（INT4压缩）
  python ov_model_convert.py \\
    --pretrained_model_path ./PaddleOCR-VL \\
    --ov_model_path ./ov_paddleocr_vl_model \\
    --device gpu \\
    --llm_int4_compress \\
    --vision_int8_quant

注意:
  - 模型转换可能需要较长时间，请耐心等待
  - 确保有足够的磁盘空间存储转换后的模型
  - GPU设备需要安装相应的OpenVINO GPU插件
        """,
        add_help=True,  # 显式启用帮助选项（默认已启用）
    )
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default="./PaddleOCR-VL",
        help="原始PaddleOCR-VL预训练模型路径 (默认: %(default)s)",
    )
    parser.add_argument(
        "--ov_model_path",
        type=str,
        default="./ov_paddleocr_vl_model",
        help="导出的OpenVINO模型保存路径 (默认: %(default)s)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "gpu"],
        help="OpenVINO编译使用的设备 (默认: %(default)s)",
    )
    llm_compress_group = parser.add_mutually_exclusive_group()
    llm_compress_group.add_argument(
        "--llm_int4_compress",
        action="store_true",
        help="启用LLM部分的INT4压缩（可大幅减少模型大小，但可能影响精度）",
    )
    llm_compress_group.add_argument(
        "--llm_int8_compress",
        action="store_true",
        help="启用LLM部分的INT8压缩（可减少模型大小，但可能影响精度）",
    )
    parser.add_argument(
        "--vision_int8_quant",
        action="store_true",
        help="启用视觉编码器的INT8量化（可提升推理速度）",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # 显示压缩选项信息
    if args.llm_int4_compress:
        print("ℹ️  使用 INT4 压缩（模型更小，但可能影响精度）")
    elif args.llm_int8_compress:
        print("ℹ️  使用 INT8 压缩（平衡模型大小和精度）")
    else:
        print("ℹ️  未启用LLM压缩（使用原始精度模型）")
    
    paddleocr_vl_ov = PaddleOCR_VL_OV(
        pretrained_model_path=args.pretrained_model_path,
        ov_model_path=args.ov_model_path,
        device=args.device,
        llm_int4_compress=args.llm_int4_compress,
        llm_int8_compress=args.llm_int8_compress,
        vision_int8_quant=args.vision_int8_quant,
    )
    paddleocr_vl_ov.export_vision_to_ov()


if __name__ == "__main__":
    main()
