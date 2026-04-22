# PaddleOCR-VL-1.5 OpenVINO

![PaddleOCR-VL APP](./client_app/images/gui.png)

A complete document understanding pipeline based on OpenVINO for PaddleOCR-VL-1.5, supporting document layout detection and Vision Language Model (VLM) inference. Features automatic model downloading for out-of-the-box usage.

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Model Download](#model-download)
- [Contact](#contact)

## 📁 Project Structure

```
paddleocr_vl_ov/
├── client_app/                     # Desktop GUI client (PySide6) - see client_app/README.md
├── paddleocr_vl_openvino/          # Main package
│   ├── paddleocr_vl/               # VLM model related code
│   │   ├── ov_paddleocr_vl.py     # OpenVINO VLM model implementation
│   │   └── image_processing_paddleocr_vl.py  # Image preprocessing
│   ├── paddleocr_vl_pipeline/      # Pipeline implementation
│   │   └── ov_paddleocr_vl_pipeline.py  # Main Pipeline class
│   └── pp_doclayoutv2/            # Layout detection related code
│       └── ov_pp_layoutv2_infer.py  # Layout detection inference
├── ov_pipeline_test.py             # Test script
├── gradio_server.py                # Gradio web interface
├── requirements.txt                # Dependencies list
├── pyproject.toml                  # Package configuration
├── build_wheel.py                  # Build script for whl package
└── README.md                       # This file
```

## 🖥️ Desktop GUI (Optional)

If you want a desktop GUI (task queue + history + settings), see:

- `client_app/README.md`

## 🔧 Installation

### Requirements

- Python 3.10+
- OpenVINO 2025.4+

### Installation Methods

#### Method 1: Install from Wheel Package (Recommended)

The easiest way to install is using the pre-built wheel package:

```bash
# Method 1: Install from GitHub Release (Recommended)
# Direct install from GitHub Release (replace v0.1.0 with the actual release version)
pip install https://github.com/opendatalab/PaddleOCR-VL/releases/download/v0.1.0/paddleocr_vl_openvino-0.1.0-py3-none-any.whl

# Or download manually from GitHub Releases:
# 1. Visit: https://github.com/opendatalab/PaddleOCR-VL/releases
# 2. Download the .whl file from the latest release
# 3. Install locally:
pip install paddleocr_vl_openvino-0.1.0-py3-none-any.whl

# Method 2: Install from local build
# Build and install from source code:
cd paddleocr_vl_ov
python -m build --wheel
pip install dist/paddleocr_vl_openvino-*.whl
```

This will automatically install all required dependencies.

#### Method 2: Build and Install from Source

1. **Clone the repository**

```bash
git clone <repository_url>
cd paddleocr_vl_ov
```

2. **Build the wheel package**

```bash
# Install build tools
pip install --upgrade setuptools wheel build

# Build the package
python -m build --wheel

# Or use the provided build script
python build_wheel.py
```

3. **Install the built package**

```bash
pip install dist/paddleocr_vl_openvino-*.whl
```

#### Method 3: Install in Development Mode

For development, you can install the package in editable mode:

```bash
# Install in development mode
pip install -e .

# This allows you to modify the code without reinstalling
```

**Note:** When installing from wheel package, all dependencies including OpenVINO will be automatically installed.

## 🚀 Quick Start

### Simplest Usage (Automatic Model Download)

```python
from paddleocr_vl_openvino.paddleocr_vl_pipeline import PaddleOCRVL

# Initialize (automatic model download)
pipeline = PaddleOCRVL(
    layout_model_path=None,  # Automatically download layout detection model
    vlm_model_path=None,      # Automatically download VLM model
    vlm_device="GPU", 
    layout_device="CPU",
    layout_precision="fp16",  
    llm_int4_compress=False,  # LLM INT4 quantization compression
    vision_int8_quant=False,  # Vision model INT8 quantization
    llm_int8_compress=False,  # LLM INT8 quantization compression
    llm_int8_quant=False,     # LLM INT8 quantization
)

# Predict
print("Starting recognition...")
output = pipeline.predict("./test_images/paddleocr_vl_demo.png")  

# Process results
for res in output:
    res.print()
    res.save_to_json(save_path="output")
    res.save_to_markdown(save_path="output")
```

**It's that simple!** Models will be automatically downloaded from ModelScope on first run, and cached models will be used directly on subsequent runs.

## 📖 Usage

### Quantization & Compression Options

The pipeline supports various quantization and compression options to optimize model size, memory usage, and inference speed:

- **`llm_int4_compress`** (default: `False`): INT4 quantization compression for LLM model. Significantly reduces model size and memory usage, but may slightly affect accuracy.
- **`vision_int8_quant`** (default: `False`): INT8 quantization for Vision model. Balances accuracy and performance.
- **`llm_int8_compress`** (default: `False`): INT8 quantization compression for LLM model. Reduces model size while maintaining good accuracy.
- **`llm_int8_quant`** (default: `False`): INT8 quantization for LLM model. Improves inference speed with minimal accuracy loss.

**Recommendations:**
- For **maximum accuracy**: Set all quantization options to `False` (default)
- For **balanced performance**: Use settings (`vision_int8_quant=True`, `llm_int8_compress=True`, `llm_int8_quant=True`, `llm_int4_compress=False`)
- For **maximum compression**: Use settings (`llm_int4_compress=True`, `llm_int8_quant=True`, `llm_int8_compress=False`) (smallest model size, but may affect accuracy)

### Method 1: Fully Automatic Download (Recommended)

When model paths are set to `None`, models will be automatically downloaded from ModelScope:

```python
from paddleocr_vl_openvino.paddleocr_vl_pipeline import PaddleOCRVL

pipeline = PaddleOCRVL(
    layout_model_path=None,  # Automatic download
    vlm_model_path=None,     # Automatic download
    vlm_device="GPU", 
    layout_device="CPU",
    layout_precision="fp16",
    llm_int4_compress=False,  # LLM INT4 quantization compression
    vision_int8_quant=False,  # Vision model INT8 quantization
    llm_int8_compress=False,  # LLM INT8 quantization compression
    llm_int8_quant=False,     # LLM INT8 quantization
)
```

### Method 2: Use Existing Models (No Download)

If models already exist, use them directly:

```python
pipeline = PaddleOCRVL(
    layout_model_path="C:/path/to/existing/model.xml",
    vlm_model_path="C:/path/to/existing/vlm_model",
    vlm_device="GPU", 
    layout_device="CPU",
    llm_int4_compress=False,
    vision_int8_quant=False,
    llm_int8_compress=False,
    llm_int8_quant=False,
)
```

### Complete Example

```python
from paddleocr_vl_openvino.paddleocr_vl_pipeline import PaddleOCRVL

# Initialize Pipeline
pipeline = PaddleOCRVL(
    layout_model_path=None,  # Automatically download layout detection model
    vlm_model_path=None,     # Automatically download VLM model
    vlm_device="GPU",        # Use GPU for VLM model
    layout_device="CPU",     # Use CPU for layout detection model
    layout_precision="fp16",
    llm_int4_compress=False,  # LLM INT4 quantization compression (default: False)
    vision_int8_quant=False,  # Vision model INT8 quantization (default: False)
    llm_int8_compress=False,  # LLM INT8 quantization compression (default: False)
    llm_int8_quant=False,     # LLM INT8 quantization (default: False)
)

# Execute prediction
print("Starting recognition...")
output = pipeline.predict("./test_images/paddleocr_vl_demo.png")  

# Process results
for res in output:
    # Print result summary
    res.print()
    
    # Save JSON format results
    res.save_to_json(save_path="output")
    
    # Save Markdown format results
    res.save_to_markdown(save_path="output")
```

## 📚 API Documentation

### `PaddleOCRVL` Class

#### Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `layout_model_path` | `Optional[str]` | `None` | Layout detection model path (.xml file), automatically downloads if `None`. **Note:** If a specific `.xml` file path is provided, the `layout_precision` parameter will be ignored |
| `vlm_model_path` | `Optional[str]` | `None` | VLM model path (directory containing vision.xml, llm_stateful.xml, etc.), automatically downloads if `None` |
| `vlm_device` | `str` | `"CPU"` | VLM model inference device: `"CPU"`, `"GPU"`, `"AUTO"` |
| `layout_device` | `str` | `"CPU"` | Layout detection model inference device: `"CPU"`, `"GPU"`, `"NPU"`, `"AUTO"` |
| `use_layout_detection` | `bool` | `True` | Whether to use layout detection |
| `use_chart_recognition` | `bool` | `False` | Whether to use chart recognition |
| `merge_layout_blocks` | `bool` | `True` | Whether to merge layout blocks |
| `markdown_ignore_labels` | `List[str]` | `None` | List of labels to ignore in Markdown output |
| `cache_dir` | `Optional[str]` | `None` | ModelScope model cache directory, uses default cache directory if `None` |
| `layout_precision` | `str` | `"fp16"` | Layout detection model precision selection: currently only `"fp16"` (DocLayoutV3.xml single-file model). This parameter is kept for backward compatibility and will be ignored. |
| `llm_int4_compress` | `bool` | `False` | Enable LLM INT4 quantization compression (significantly reduces model size and memory usage, may slightly affect accuracy) |
| `vision_int8_quant` | `bool` | `False` | Enable Vision model INT8 quantization (balances accuracy and performance) |
| `llm_int8_compress` | `bool` | `False` | Enable LLM INT8 quantization compression (reduces model size, may slightly affect accuracy) |
| `llm_int8_quant` | `bool` | `False` | Enable LLM INT8 quantization (improves inference speed, may slightly affect accuracy) |

#### `predict` Method

```python
def predict(
    self,
    input: Union[str, List[str], np.ndarray, List[np.ndarray]],
    use_layout_detection: Optional[bool] = None,
    layout_threshold: Optional[Union[float, dict]] = None,
    layout_nms: Optional[bool] = None,
    layout_unclip_ratio: Optional[Union[float, tuple]] = None,
    layout_merge_bboxes_mode: Optional[str] = None,
    max_new_tokens: Optional[int] = None,
    vlm_batch_size: int = 8,
    early_stop_ratio: float = 0.0,
    **kwargs,
) -> List[PaddleOCRVLResult]
```

**Parameter Description:**

- `input`: Input image (file path, list of paths, numpy array, or list of numpy arrays)
- `use_layout_detection`: Whether to use layout detection (overrides initialization setting)
- `layout_threshold`: Layout detection threshold (float or dict, dict format: `{category_id: threshold}`)
- `layout_nms`: Whether to use NMS for deduplication
- `layout_unclip_ratio`: Layout box expansion ratio (float or tuple `(w_ratio, h_ratio)`)
- `layout_merge_bboxes_mode`: Layout box merge mode (`"union"`, `"large"`, `"small"`)
- `max_new_tokens`: Maximum number of tokens to generate for VLM
- `vlm_batch_size`: Number of image blocks processed in a single VLM batch (default: `8`). Increasing this can speed up inference for documents with many blocks, at the cost of higher memory usage.
- `early_stop_ratio`: Batch early-termination ratio (default: `0.0`, disabled). When set to a value such as `0.7`, the batch loop will stop and flush remaining results once 70% of the current batch's slots have finished, reducing tail-latency on uneven batches.

**Return Value:**

Returns `List[PaddleOCRVLResult]`, each result contains:
- `parsing_res_list`: Parsing result list (`PaddleOCRVLBlock` objects)
- `input_path`: Input image path
- `json`: JSON format result
- `img`: Visualization image
- `markdown`: Markdown format result

#### `PaddleOCRVLResult` Class Methods

- `print()`: Print result summary
- `save_to_json(save_path)`: Save JSON format results
- `save_to_img(save_path)`: Save visualization image
- `save_to_markdown(save_path)`: Save Markdown format results

## 📥 Model Download

### Automatic Download (Recommended)

Models will be automatically downloaded from ModelScope on first use, no manual operation required.

### Manual Download

If you need to manually download models, you can use the following methods:

#### PaddleOCR-VL-1.5-ov (Layout + VLM)

Both the layout model and the VLM model are provided under the same ModelScope repository:

**ModelScope**: [PaddleOCR-VL-1.5-ov](https://www.modelscope.cn/models/zhaohb/PaddleOCR-VL-1.5-ov)

Repository structure (subdirectories):

- **Layout (DocLayoutV3-ov)**: `PP-DoclayoutV3-ov/DocLayoutV3.xml`
  - Currently only a single-file model is provided (**no precision variants**)
  - The `layout_precision` parameter is kept for backward compatibility and will be ignored
- **VLM (PaddleOCR-VL-1.5-ov)**: `PaddleOCR-VL-1.5-ov/` (contains `vision.xml`, `llm_stateful.xml`, `llm_embd.xml`, etc.)

```bash
# Using ModelScope SDK
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('zhaohb/PaddleOCR-VL-1.5-ov')"
```

After downloading, set:

- `layout_model_path` to `.../PP-DoclayoutV3-ov/DocLayoutV3.xml` (or the `PP-DoclayoutV3-ov` directory)
- `vlm_model_path` to `.../PaddleOCR-VL-1.5-ov` (directory)

**Windows note:** ModelScope may warn about failing to create symbolic links. This does not affect usage; use the real cache directory path printed in logs.

### Model Caching

Downloaded models are cached in ModelScope's default cache directory (usually `~/.cache/modelscope/hub`). Subsequent runs will directly use cached models without re-downloading.

You can specify a custom cache directory using the `cache_dir` parameter:

```python
pipeline = PaddleOCRVL(
    layout_model_path=None,
    vlm_model_path=None,
    cache_dir="./models_cache",  # Custom cache directory
    vlm_device="GPU",
    layout_device="CPU",
    llm_int4_compress=False,
    vision_int8_quant=False,
    llm_int8_compress=False,
    llm_int8_quant=False,
)
```

### PDF OCR Script

The `pdf_ocr.py` script provides a cross-page dynamic batching OCR pipeline for PDF files. It converts each PDF page to an image and processes them through the PaddleOCR-VL pipeline.

**Basic usage:**

```bash
# Minimal – process a PDF with default settings
python pdf_ocr.py --pdf input.pdf

# Specify output directory and rendering DPI
python pdf_ocr.py --pdf input.pdf --output pdf_output --dpi 150

# Use GPU for VLM, CPU for layout detection
python pdf_ocr.py --pdf input.pdf --device GPU --layout-device CPU

# Tune batch size and enable early-stop
python pdf_ocr.py --pdf input.pdf --vlm-batch-size 40 --early-stop-ratio 0.7
```

**Key arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--pdf` | *(required)* | Input PDF file path |
| `--output` | `pdf_output` | Output directory |
| `--dpi` | `100` | PDF rendering DPI |
| `--device` | `GPU` | VLM inference device (`CPU`/`GPU`/`AUTO`) |
| `--layout-device` | `CPU` | Layout detection device (`CPU`/`GPU`/`NPU`/`AUTO`) |
| `--vlm-batch-size` | `40` | VLM batch size (number of blocks per batch) |
| `--max-new-tokens` | `1024` | Maximum tokens to generate per block |
| `--window-pages` | `10` | Pages per processing window (`0` = all pages at once) |
| `--early-stop-ratio` | `0.0` | Batch early-termination ratio (`0` = disabled, e.g. `0.7` stops when 70% of a batch is done) |
| `--layout-model-path` | `None` | Layout model path (auto-download if `None`) |
| `--vlm-model-path` | `None` | VLM model path (auto-download if `None`) |
| `--layout-threshold` | `0.4` | Layout detection confidence threshold |

Results are saved under `<output>/<pdf_stem>/results/` as per-page `.json` and `.md` files.

### Gradio Server

We provide an interactive Gradio web interface for easy document understanding. Launch the server with:

```bash
python gradio_server.py
```

The server will start at `http://localhost:7860` and automatically open in your browser.

**Usage:**

1. Go to the "Pipeline 配置" tab to initialize the pipeline
2. Upload an image in the "文档识别" tab
3. Configure parameters (layout detection threshold, max tokens, etc.)
4. Click "开始识别" to process the image
5. View results in multiple formats (Markdown, JSON, visualization)


## 📧 Contact

For questions or suggestions, please submit an Issue or Pull Request.
