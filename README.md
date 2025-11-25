# PaddleOCR-VL + OpenVINO

![PaddleOCR-VL Demo](./gradio_paddleocr_vl.gif)

This repo demonstrates how to convert the [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) model to OpenVINO IR, validate parity between the original PyTorch model and OpenVINO, and launch an interactive Gradio demo.

---

## Update Notes
### 2025/11/25
1. Paddleocr-VL model supports using openvino to accelerate the inference process. Currently only verified on windows.
2. LLM testing shows that enabling INT4 compression is not recommended—it tends to produce incorrect results.

## Running Guide

### Environment Setup

```bash
conda create -n paddleocr_vl_ov python=3.12
conda activate paddleocr_vl_ov
pip install -r requirements.txt
```
*Replace the Python version above if needed. The `requirements.txt` shipped in this repo already pins the versions verified for both Torch and OpenVINO pipelines.*

### 1. Convert to OpenVINO IR

#### 1.1 Replace the Hugging Face modeling file
Use the optimized implementation from this repo to overwrite the file inside the model you downloaded from Hugging Face:

```bash
cp modeling_paddleocr_vl.py <原始PaddleOCR-VL预训练模型路径>/modeling_paddleocr_vl.py
```

#### 1.2 Run the conversion script

```bash
python ov_model_convert.py \
  --pretrained_model_path ..\test\PaddleOCR-VL \
  --ov_model_path ..\test\ov_paddleocr_vl_model
```

---

### 2. Verify Torch vs. OpenVINO outputs

```bash
python torch_ov_test.py \
  --pretrained_model_path ..\test\PaddleOCR-VL \
  --ov_model_path ..\test\ov_paddleocr_vl_model \
  --image_path ./paddle_arch.jpg \
  --task ocr
```

Sample comparison output:

```text
============================================================
📄 Transformers OCR 识别结果:
============================================================
User: OCR:
Assistant: App | AI financial report analysis | AI homework grading | AI contract & bill review | Intelligent document processing | Certificate information extraction | More...
Model Zoo | OCR Series | Doc Parsing Series | Doc Understanding Series | PP-DocBee2 | Table Shibie Seal Shibie Formula Shibie Chart Shibie ... ... | More
Toolkit | PP-OCRv3 | PaddleOCR-VL | PP-DocTranslation | PP-DocTranslation |   |
  | PP-OCRv4 | PP-StructureV3 | PP-ChatOCRv4 | PP-ChatOCRv4 |   |
  | PP-OCRv5 | PP-StructureV3 | PP-ChatOCRv4 | PP-ChatOCRv4 |   |
  | Train | Inference | High Performance Serving | MCP Server |   |
  | by just 1 command | by 3 lines code | by just 1 command | Called by LLMs |   |
Frame work | PaddlePaddle 3.0+ | Intel | KUNLUNXIN | Ascend |   |
Hardware | NVIDIA | Intel | KUNLUNXIN | Ascend |   |
============================================================

⏱️  执行时间统计:
   - 生成时间 (generate): 47.393 秒 (47392.69 毫秒)
   - 解码时间 (decode):   0.005 秒 (4.62 毫秒)
   - 总时间:              47.397 秒 (47397.31 毫秒)
============================================================

============================================================
🔄 正在加载OpenVINO模型...
============================================================
OpenVINO version 
 2026.0.0-20478-59bb607be25

============================================================
📄 openVINO OCR 识别结果:
============================================================
App | AI financial report analysis | AI homework grading | AI contract & bill review | Intelligent document processing | Certificate information extraction | More...       
Model Zoo | OCR Series | Doc Parsing Series | Doc Understanding Series | PP-DocBee2 | Table Shibie Seal Shibie Formula Shibie Chart Shibie ... ... | More
Toolkit | PP-OCRv3 | PaddleOCR-VL | PP-DocTranslation | PP-DocTranslation |   |
  | PP-OCRv4 | PP-StructureV3 | PP-ChatOCRv4 | PP-ChatOCRv4 |   |
  | PP-OCRv5 | PP-StructureV3 | PP-ChatOCRv4 | PP-ChatOCRv4 |   |
  | Train | Inference | High Performance Serving | MCP Server |   |
  | by just 1 command | by 3 lines code | by just 1 command | Called by LLMs |   |
Frame work | PaddlePaddle 3.0+ | Intel | KUNLUNXIN | Ascend |   |
Hardware | NVIDIA | Intel | KUNLUNXIN | Ascend |   |
============================================================

⏱️  执行时间统计:
   - Chat 方法执行时间: 9.471 秒 (9471.47 毫秒)
============================================================
```

---

### 3. Launch the Gradio demo

```bash
python paddleocr_vl_grdio.py
```

This starts an interactive UI where you can initialize the OpenVINO model, upload images, and inspect OCR/LaTeX/Table results with automatic visualization.
