from PIL import Image
import torch
import openvino as ov
from transformers import AutoModelForCausalLM, AutoProcessor
from ov_paddleocr_vl import PaddleOCR_VL_OV, OVPaddleOCRVLForCausalLM


# paddleocr_vl_ov = PaddleOCR_VL_OV(pretrained_model_path="./PaddleOCR-VL", ov_model_path="./ov_paddleocr_vl_model", device="cpu", llm_int4_compress=True, vision_int8_quant=False)
# paddleocr_vl_ov.export_vision_to_ov()

# ---- Settings ----
model_path = "./PaddleOCR-VL"
image_path = "./paddle_ocr_vl.png"
task = "ocr" # Options: 'ocr' | 'table' | 'chart' | 'formula'
# ------------------

DEVICE = "cpu"

PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

image = Image.open(image_path).convert("RGB")
image = image.resize((300, 150), Image.Resampling.LANCZOS)

model = AutoModelForCausalLM.from_pretrained(
    model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
).to(DEVICE).eval()
processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

messages = [
    {"role": "user",         
     "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": PROMPTS[task]},
        ]
    }
]
inputs = processor.apply_chat_template(
    messages, 
    tokenize=True, 
    add_generation_prompt=True, 	
    return_dict=True,
    return_tensors="pt"
).to(DEVICE)

# print(processor.apply_chat_template.__module__)  # 查看模块
# print(processor.apply_chat_template.__qualname__)  # 查看完整路径
# print(type(processor.tokenizer))  # 查看 tokenizer 类型

# # transformers.processing_utils
# # ProcessorMixin.apply_chat_template
# # <class 'transformers.models.llama.tokenization_llama_fast.LlamaTokenizerFast'>

# breakpoint()
outputs = model.generate(**inputs, max_new_tokens=1024, use_cache=True)
outputs = processor.batch_decode(outputs, skip_special_tokens=True)[0]
print("\n" + "="*60)
print("📄 Transformers OCR 识别结果:")
print("="*60)
print(outputs)
print("="*60 + "\n")

# breakpoint()

llm_infer_list = []
vision_infer = []
core = ov.Core()
paddleocr_vl_model = OVPaddleOCRVLForCausalLM(core=core, ov_model_path="./ov_paddleocr_vl_model", device="GPU", llm_int4_compress=False, vision_int8_quant=False, llm_int8_quant=False, llm_infer_list=llm_infer_list, vision_infer=vision_infer)
version = ov.get_version()
print("OpenVINO version \n", version)
print('\n')

generation_config = {
    "bos_token_id": paddleocr_vl_model.tokenizer.bos_token_id,
    "eos_token_id": paddleocr_vl_model.tokenizer.eos_token_id,
    "pad_token_id": paddleocr_vl_model.tokenizer.pad_token_id,
    "max_new_tokens": 1024,
    "do_sample": False,
}
response, history = paddleocr_vl_model.chat(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"], pixel_values=inputs["pixel_values"], image_grid_thw=inputs["image_grid_thw"], generation_config=generation_config)
print("\n" + "="*60)
print("📄 openVINO OCR 识别结果:")
print("="*60)
print(response)
print("="*60 + "\n")