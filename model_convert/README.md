## Manually convert to OpenVINO IR

### PaddleOCR_VL model
This folder contains scripts to convert a PaddleOCR-VL Hugging Face checkpoint into OpenVINO IR.

#### Step 1: Replace the Hugging Face `modeling_*.py`

Use the optimized implementation from this repo to overwrite the modeling file in your downloaded HF model directory.

##### Windows (PowerShell)

```powershell
copy .\modeling_paddleocr_vl.py "<PaddleOCR-VL model path>\modeling_paddleocr_vl.py"
```

##### Linux/macOS

```bash
cp ./modeling_paddleocr_vl.py "<PaddleOCR-VL model path>/modeling_paddleocr_vl.py"
```

#### Step 2: Run the conversion script

##### Windows (PowerShell)

```powershell
python .\ov_model_convert.py `
  --pretrained_model_path ..\test\PaddleOCR-VL `
  --ov_model_path ..\test\ov_paddleocr_vl_model
```

##### Linux

```bash
python ./ov_model_convert.py \
  --pretrained_model_path ../test/PaddleOCR-VL \
  --ov_model_path ../test/ov_paddleocr_vl_model
```

### PP-DocLayout model

This section shows how to convert a PP-DocLayout Paddle model to OpenVINO IR.

#### DocLayout Step 1: Download the model

First, download the corresponding model.

#### DocLayout Step 2: Install Paddle2ONNX

```bash
pip install paddle2onnx
```

#### DocLayout Step 3: Convert Paddle model to ONNX

Convert the Paddle layout model to an ONNX model using the following command:

```bash
paddle2onnx --model_dir PP-DocLayoutV3 \
            --model_filename inference.json \
            --params_filename inference.pdiparams \
            --save_file PP-DocLayoutV3.onnx
```

#### DocLayout Step 4: Convert ONNX to OpenVINO IR

Then convert the ONNX model to OpenVINO IR:

```bash
ovc PP-DocLayoutV3.onnx --output_model DocLayoutV3.xml
```


