from nncf import compress_weights, CompressWeightsMode

import openvino as ov

core = ov.Core()

model_path = "/home/benchmark/xkd/XiaoMi/PaddleOCR-VL/paddleocr_vl_ov/ov-PaddleOCR-VL-model/llm_stateful.xml"
# model_path = "/home/benchmark/xkd/XiaoMi/PaddleOCR-VL/paddleocr_vl_ov/ov-PaddleOCR-VL-model/vision.xml"
# model_save_path = model_path.replace(".xml","_int8.xml")
model_save_path = model_path.replace(".xml","_int4.xml")

ov_model = core.read_model(model_path)

compressed_model = compress_weights(ov_model, mode=CompressWeightsMode.INT4_SYM)

ov.save_model(compressed_model, model_save_path)
print("[OV INFO] model compress success")
