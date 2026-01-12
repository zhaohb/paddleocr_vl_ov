from paddleocr_vl_pipeline.ov_paddleocr_vl_pipeline import PaddleOCRVL

# 初始化（支持自动下载模型）
# 方式1: 自动下载（如果模型路径为 None）
pipeline = PaddleOCRVL(
    layout_model_path=None,  # 自动下载
    vlm_model_path=None,      # 自动下载
    vlm_device="GPU", 
    layout_device="GPU",
    llm_int4_compress=False,
    vision_int8_quant=True,
    llm_int8_compress=True,
    llm_int8_quant=True,
)

# # 方式2: 使用本地模型路径（如果路径不存在会自动下载）
# pipeline = PaddleOCRVL(
#     layout_model_path="./pp_doclayoutv2_f16.xml",  # 如果文件不存在会自动下载
#     vlm_model_path="./ov_paddleocr_vl_model",  # 如果目录不存在会自动下载
#     vlm_device="GPU", 
#     layout_device="GPU",
# )

# 预测
print("开始识别...")
output = pipeline.predict("./test_images/doc_test.png")  

# 处理结果
for res in output:
    res.print()
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")