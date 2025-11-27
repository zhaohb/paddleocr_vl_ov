# PaddleOCR-VL + OpenVINO

![PaddleOCR-VL Demo](./gradio_paddleocr_vl.gif)

This repo demonstrates how to convert the [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) model to OpenVINO IR, validate parity between the original PyTorch model and OpenVINO, and launch an interactive Gradio demo.

---

## Update Notes
### 2025/11/25
1. Paddleocr-VL model supports using openvino to accelerate the inference process. Currently only verified on windows.
2. LLM testing shows that enabling INT4 compression is not recommended—it tends to produce incorrect results.
3. **Device Support**: Currently, only GPU and CPU devices are supported. NPU execution is not supported and will result in runtime errors. 

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
  --pretrained_model_path ..\test\PaddleOCR-VL  \
  --ov_model_path ..\test\ov_paddleocr_vl_model \
  --image_path test_images\chart\chart1.png \
  --task chart  \
  --ov_device GPU
```

Sample comparison output:

```text
============================================================
🔄 正在加载Transformers模型...
============================================================
Using a slow image processor as `use_fast` is unset and a slow processor was saved with this model. `use_fast=True` will be the default behavior in v4.52, even if the model was saved with a slow processor. This will result in minor differences in outputs. You'll still be able to use a slow processor with `use_fast=False`.

============================================================
📄 cpu Transformers chart 识别结果:
============================================================
User: Chart Recognition:
Assistant: Quarter | mom change | existing home take rate
1Q22 | -0.04% | 1.64%
2Q22 | -0.2% | 1.41%
3Q22 | 0.2% | 1.59%
4Q22 | -0.1% | 1.5%
1Q23 | -0.1% | 1.38%
2Q23 | 0.0% | 1.41%
3Q23 | 0.0% | 1.44%
4Q23 | -0.1% | 1.29%
1Q24 | -0.03% | 1.26%
2Q24 | 0.02% | 1.29%
3Q24 | 0.0% | 1.3%
4Q24 | -0.1% | 1.2%
1Q25 | -0.01% | 1.18%
============================================================

⏱️  执行时间统计:
   - 生成时间 (generate): 39.446 秒 (39445.81 毫秒)
   - 解码时间 (decode):   0.000 秒 (0.40 毫秒)
   - 总时间:              39.446 秒 (39446.20 毫秒)
============================================================


============================================================
🔄 正在加载OpenVINO模型...
============================================================
OpenVINO version 
 2026.0.0-20478-59bb607be25



============================================================
📄 GPU openVINO chart 识别结果:
============================================================
Quarter | mom change | existing home take rate
1Q22 | -0.04% | 1.64%
2Q22 | -0.2% | 1.41%
3Q22 | 0.2% | 1.59%
4Q22 | -0.1% | 1.5%
1Q23 | -0.1% | 1.38%
2Q23 | 0.0% | 1.41%
3Q23 | 0.0% | 1.44%
4Q23 | -0.1% | 1.29%
1Q24 | -0.03% | 1.26%
2Q24 | 0.02% | 1.29%
3Q24 | 0.0% | 1.3%
4Q24 | -0.1% | 1.2%
1Q25 | -0.01% | 1.18%
============================================================

⏱️  执行时间统计:
   - Chat 方法执行时间: 5.684 秒 (5684.29 毫秒)
============================================================
```

---

### 3. Launch the Gradio demo

```bash
python paddleocr_vl_grdio.py
```

This starts an interactive UI where you can initialize the OpenVINO model, upload images, and inspect OCR/LaTeX/Table results with automatic visualization.
