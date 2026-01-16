# PaddleOCR‑VL OpenVINO Desktop Client (GUI User Guide)

This directory contains a **PySide6** desktop GUI for running `paddleocr_vl_openvino` inference in batch, and for managing outputs and successful task history.

## Quick Start

### 1) Prepare the environment

- **Recommended**: install this project's wheel (or install dependencies in the project root) so that `paddleocr_vl_openvino` can be imported.
- GUI dependencies are listed in `client_app/requirements.txt`:
  - `PySide6`
  - `PyMuPDF` (used to render PDF pages into images)
  - `PyInstaller` (optional; only required when packaging an `.exe`)

### 2) Launch the GUI

Run from the project root `paddleocr_vl_ov/`:

```bash
python client_app/main.py
```

### 3) Smoke Test (optional)

Quickly verifies the GUI can start (without entering the event loop):

```bash
python client_app/main.py --smoke-test
```

## UI Overview

The left navigation has four pages:

- **OCR / Parsing**: import files, manage the task queue, start/stop/rerun, view logs and progress.
- **History**: keeps only *successfully completed* tasks, supports previewing results and deleting output folders.
- **Settings**: configure model paths, devices, precision, thresholds, tokens, quantization/compression switches, and output directory.
- **About**: basic version/info.

## OCR / Parsing (Task Queue)

### Supported inputs

- **Images**: `.png/.jpg/.jpeg/.bmp/.webp`
- **Documents**: `.pdf` (rendered into multiple pages first, then inferred page-by-page)

### Import methods

- **Toolbar “Add Files”**: select one or more files and add them to the queue
- **Drag & drop**: drag files or folders onto “拖拽文件到这里”
  - dragging a folder will recursively collect supported images/PDFs
- **Toolbar/button “Screenshot”**: select a screen region, save as PNG, and add it to the queue automatically

### Task table columns

- **File**: input file path
- **Type**: file extension (png/jpg/pdf…)
- **Task Type**: `ocr/table/chart/formula`
  - **When “Enable layout detection” is ON**: this dropdown is disabled (task type is decided by layout detection results)
  - **When “Enable layout detection” is OFF**: this dropdown is enabled and is used as `prompt_label` for full-image inference
- **Status**: `pending / running / done / error`
  - once a task becomes `done/error`, it will not be overwritten by later runs
- **Output Dir**: the task’s output directory (filled after the run finishes)

### Buttons

- **Start**: runs only the tasks whose status is not `done/error` (i.e., pending tasks)
- **Stop**: requests the background worker thread to stop (completed tasks are not affected)
- **Rerun Selected**: reruns only the selected rows in the table
  - resets selected tasks to `pending` and clears previous error/summary/preview cache fields
- **Delete Selected**: removes the selected tasks from the queue
  - not allowed while running (to avoid worker index mismatch)
- **Clear** (toolbar): clears the entire task queue (not allowed while running)

### Output directory and progress

- The progress label shows `current/total` (only counting the tasks executed in the current run)
- Logs are displayed in the log box below the UI (including messages like “reuse the loaded Pipeline”)

## History

### Recording rules

- **Only successful tasks** are recorded (`done`)
- The history file is stored at: `<output_dir>/_history.jsonl`
  - after you switch the output directory, the History page automatically loads history from that directory

### Preview content

- **Compare (Input vs Markdown)**:
  - left: input preview (for PDF, `pages/page_0001.png` is preferred)
  - right: rendered Markdown preview (offline local assets supported)
- **Output image**: `vis.png` is preferred

### Action buttons

- **Open Output Folder**: open this history task’s output directory in the system file explorer
- **Delete Selected**: deletes the history record and **recursively deletes its output directory** (including `md/json/vis/pages`, etc.)
- **Clear History**: clears the list and deletes `_history.jsonl`

## Settings (Parameters and Recommendations)

Settings affect two kinds of behavior:

- **Initialization config (PipelineInitConfig)**: controls how models are loaded, which device they run on, and whether quantization/compression is enabled.  
  As long as this part does not change, the backend will **reuse the loaded Pipeline** to avoid repeated download/initialization.
- **Inference config (PredictConfig)**: controls thresholds, token limits, and whether to enable layout detection for each run.

### Models and cache

- **layout_model_path**
  - **Meaning**: PP‑DocLayoutV2 OpenVINO model path (typically a `.xml` file or a model directory)
  - **Recommendations**:
    - **If you already have a local model**: point directly to the `.xml` file for maximum stability (avoids auto-download dependency issues)
    - **Leave empty**: the pipeline will auto-download/select a model (requires network and related dependencies)
  - **Important**: if you set a specific `.xml` file path, **`layout_precision` will be ignored** (because the model is fixed).

- **vlm_model_path**
  - **Meaning**: VLM model path (the main OCR‑VL model)
  - **Recommendations**:
    - **Offline / packaged scenarios**: strongly recommend setting a local path to avoid slow/failed downloads on first startup

- **cache_dir**
  - **Meaning**: model download/cache directory
  - **Recommendations**:
    - place it on a drive with enough space and keep the path short (Windows long paths can be problematic)
    - if multiple environments share models, point all of them to a unified cache directory

### Devices and precision

- **vlm_device** (`CPU/GPU/AUTO`)
  - **Recommendation**: prefer `AUTO`; use `CPU` for maximum stability; use `GPU` if the driver/plugins are properly configured

- **layout_device** (`CPU/GPU/NPU/AUTO`)
  - **Recommendations**:
    - for stability: `CPU` or `AUTO`
    - for speed: use `GPU/NPU` only after you confirm OpenVINO device support is available

- **layout_precision** (`fp16 / fp32 / combined_fp16 / combined_fp32`)
  - **Meaning**: selects the layout model variant when `layout_model_path` does not point to a specific `.xml`
  - **Recommendations (general)**:
    - **Prefer GPU/NPU**: `fp16` or `combined_fp16` (faster, lower memory footprint)
    - **Prefer CPU stability/consistency**: `fp32` or `combined_fp32`
    - if you hit layout output format/compatibility issues, try switching between `combined_*` and non-combined variants

### Layout detection and threshold

- **Enable layout detection (use_layout_detection)**
  - **ON**: performs layout analysis/region detection; task type is decided by layout results (the “Task Type” dropdown is disabled)
  - **OFF**: skips layout detection and sends the full image to the VLM; the “Task Type” dropdown becomes effective and is used as `prompt_label`
  - **Recommendations**:
    - documents/papers/images with tables: keep it ON
    - single-block content (only OCR or only formulas): you can turn it OFF and manually set task type

- **layout_threshold**
  - **Meaning**: confidence threshold for layout detection (slider 0.10–1.00)
  - **Recommendations**:
    - default `0.50` is usually sufficient
    - too many missed detections: lower it (e.g., `0.30`)
    - too many false positives: raise it (e.g., `0.60–0.70`)

### Generation length

- **max_new_tokens**
  - **Meaning**: max number of tokens generated by the VLM
  - **Recommendations**:
    - output is truncated: increase it (e.g., 2048/3072/4096)
    - slow or memory-limited: decrease it (e.g., 512/1024)

### Quantization/compression switches (speed vs. quality)

> These switches are part of pipeline initialization. Changing them typically triggers re-initialization (the old model instance cannot be reused).

- **LLM INT4 compression (llm_int4_compress)**
  - **Recommendation**: enable when memory is tight or you prioritize speed; disable if output quality drops noticeably

- **Vision INT8 quantization (vision_int8_quant)**
  - **Recommendation**: enable when it improves speed significantly; if accuracy drops on small text/formulas/details, try disabling it

- **LLM INT8 compression (llm_int8_compress) / LLM INT8 quantization (llm_int8_quant)**
  - **Recommendation**: a balanced option between FP and INT4; if INT4 quality is unacceptable, try INT8

### Output directory

- **Output directory**
  - **Meaning**: root directory for all artifacts and history records
  - **Recommendations**:
    - keep it stable to manage all results in one place
    - after switching it, the History page will read `_history.jsonl` from the new directory

## Artifacts (per task)

Each task creates its own subfolder under the output directory. Common files include:

- `vis.png`: visualization result image (preferred by the History page)
- `result.md` or `<input_filename>.md`: Markdown result (rendered in the History page)
- `result.json`: structured result (for debugging or downstream processing)
- `pages/`: for PDF inputs, rendered page images (e.g., `page_0001.png`)

History file:

- `<output_dir>/_history.jsonl`: one JSON object per line (successful tasks only)

## FAQ

- **Q: Why is the “Task Type” dropdown not editable?**  
  A: You enabled “Enable layout detection”. Turn it off to make task type effective as `prompt_label` for full-image inference.

- **Q: Why doesn’t the model initialize again on subsequent runs?**  
  A: If the initialization-related settings remain unchanged (model paths/devices/precision/quantization, etc.), the backend will reuse the loaded Pipeline.

- **Q: Why does the History page only show the first PDF page as the input preview?**  
  A: By default it uses `pages/page_0001.png` as the input preview; inference runs page-by-page and concatenates results in the Markdown with separators.


