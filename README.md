# PaddleOCR-VL + OpenVINO

![PaddleOCR-VL Demo](./gradio_paddleocr_vl.gif)

This repository demonstrates how to convert the [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) model to OpenVINO IR, validate parity between the original PyTorch model and OpenVINO, and launch an interactive Gradio demo.

## Table of Contents

- [Features](#features)
- [Important Notes](#important-notes)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Demo](#demo)
- [Command Line Arguments](#command-line-arguments)
- [License](#license)
- [Citation](#citation)

## Features

- ✅ Convert PaddleOCR-VL model to OpenVINO IR format
- ✅ Validate accuracy parity between PyTorch and OpenVINO models
- ✅ Interactive Gradio demo for OCR, Table, Chart, and Formula recognition
- ✅ Support for both GPU and CPU inference

## Important Notes

> **📅 Last Updated: 2025/11/25**

1. **Platform Support**: Currently only verified on Windows
2. **INT4 Compression**: Not recommended - may produce incorrect results
3. **Device Support**: 
   - ✅ Supported: GPU and CPU
   - ❌ Not Supported: NPU (will result in runtime errors)

## Installation

### Environment Setup

```bash
conda create -n paddleocr_vl_ov python=3.12
conda activate paddleocr_vl_ov
pip install -r requirements.txt
```

> **Note**: Replace the Python version above if needed. The `requirements.txt` shipped in this repo already pins the versions verified for both Torch and OpenVINO pipelines.

## Quick Start

### Step 1: Get the OpenVINO IR Model

#### Option 1: Download Pre-converted Model (Recommended) ⭐

If you don't want to convert the model manually, you can directly download the pre-converted OpenVINO IR model from [ModelScope](https://www.modelscope.cn/models/zhaohb/PaddleOCR-Vl-OV).

**Using ModelScope SDK:**
```bash
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('zhaohb/PaddleOCR-Vl-OV')"
```

**Or using git clone:**
```bash
git clone https://www.modelscope.cn/zhaohb/PaddleOCR-Vl-OV.git
```

#### Option 2: Convert to OpenVINO IR Manually

**Step 1.1**: Replace the Hugging Face modeling file

Use the optimized implementation from this repo to overwrite the file inside the model you downloaded from Hugging Face:

```bash
cp modeling_paddleocr_vl.py <PaddleOCR-VL预训练模型路径>/modeling_paddleocr_vl.py
```

**Step 1.2**: Run the conversion script

```bash
python ov_model_convert.py \
  --pretrained_model_path ..\test\PaddleOCR-VL \
  --ov_model_path ..\test\ov_paddleocr_vl_model
```

## Usage

### Step 2: Verify Torch vs. OpenVINO Outputs

Compare the outputs from both PyTorch and OpenVINO models:

```bash
python torch_ov_test.py \
  --pretrained_model_path ..\test\PaddleOCR-VL \
  --ov_model_path ..\test\ov_paddleocr_vl_model \
  --image_path test_images\chart\chart1.png \
  --task chart \
  --ov_device GPU
```

**Available task types:**
- `ocr` - OCR text recognition
- `table` - Table recognition
- `chart` - Chart recognition
- `formula` - Formula recognition

**Run OpenVINO inference only (skip PyTorch):**

```bash
python torch_ov_test.py \
  --ov_model_path ..\test\ov_paddleocr_vl_model \
  --image_path test_images\chart\chart1.png \
  --task chart \
  --skip_torch
```

**Sample comparison output:**

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

## Demo

### Step 3: Launch the Gradio Demo

Start the interactive Gradio demo:

```bash
python paddleocr_vl_grdio.py
```

This launches an interactive web UI where you can:
- Initialize the OpenVINO model
- Upload images for processing
- Inspect OCR/Table/Chart/Formula recognition results
- View automatic visualizations

---

## Command Line Arguments

### `torch_ov_test.py` Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--pretrained_model_path` | str | `./PaddleOCR-VL` | Path to the PyTorch model |
| `--ov_model_path` | str | `./ov_paddleocr_vl_model` | Path to the OpenVINO IR model |
| `--image_path` | str | `./paddle_arch.jpg` | Path to the test image |
| `--task` | str | `ocr` | Task type: `ocr`, `table`, `chart`, or `formula` |
| `--device` | str | `cpu` | PyTorch device: `cpu` or `cuda` |
| `--ov_device` | str | `GPU` | OpenVINO device: `CPU` or `GPU` |
| `--skip_torch` | flag | `False` | Skip PyTorch model testing (OpenVINO only) |

---
