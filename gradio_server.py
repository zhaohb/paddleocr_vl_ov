"""
Gradio Server for PaddleOCR-VL OpenVINO Pipeline
基于 OpenVINO 的 PaddleOCR-VL 文档理解 Gradio 界面
"""

import gradio as gr
import os
from pathlib import Path
from paddleocr_vl_openvino.paddleocr_vl_pipeline import PaddleOCRVL
import tempfile
import json
import openvino as ov

# 全局变量存储 pipeline 实例
pipeline = None


def get_available_devices():
    """通过 OpenVINO Core 查询系统可用的推理设备列表"""
    core = ov.Core()
    devices = core.available_devices  # e.g. ['CPU', 'GPU.0', 'GPU.1', 'NPU']
    # 始终保留 AUTO 选项
    if "AUTO" not in devices:
        devices.append("AUTO")
    return devices

# 在导入后立即设置环境变量，避免Gradio初始化时的网络请求
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_SERVER_NAME", "127.0.0.1")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

def initialize_pipeline(
    layout_model_path, 
    vlm_model_path, 
    vlm_device, 
    layout_device,
    llm_int4_compress,
    vision_int8_quant,
    llm_int8_compress,
    llm_int8_quant
):
    """初始化 Pipeline"""
    global pipeline
    try:
        pipeline = PaddleOCRVL(
            layout_model_path=layout_model_path if layout_model_path else None,
            vlm_model_path=vlm_model_path if vlm_model_path else None,
            vlm_device=vlm_device,
            layout_device=layout_device,
            llm_int4_compress=llm_int4_compress,
            vision_int8_quant=vision_int8_quant,
            llm_int8_compress=llm_int8_compress,
            llm_int8_quant=llm_int8_quant,
        )
        return "✅ Pipeline 初始化成功！"
    except Exception as e:
        return f"❌ Pipeline 初始化失败: {str(e)}"

def process_image(image, use_layout_detection, layout_threshold, max_new_tokens):
    """处理上传的图片"""
    global pipeline
    
    if pipeline is None:
        return None, None, None, "❌ 请先初始化 Pipeline！"
    
    if image is None:
        return None, None, None, "❌ 请上传图片！"
    
    try:
        # 保存临时图片
        temp_dir = tempfile.mkdtemp()
        temp_image_path = os.path.join(temp_dir, "temp_image.png")
        image.save(temp_image_path)
        
        # 执行预测（predict 返回生成器，需要转换为列表）
        output_generator = pipeline.predict(
            temp_image_path,
            use_layout_detection=use_layout_detection,
            layout_threshold=layout_threshold,
            max_new_tokens=max_new_tokens,
            vlm_batch_size=30
        )
        
        # 将生成器转换为列表
        output = list(output_generator)
        
        if not output:
            return None, None, None, "❌ 未检测到任何内容"
        
        result = output[0]
        
        # 获取结果
        # 1. Markdown 文本
        markdown_info = result.markdown if hasattr(result, 'markdown') else {}
        markdown_text = markdown_info.get("markdown_texts", "") if isinstance(markdown_info, dict) else ""
        
        # 2. JSON 结果
        json_result = result.json if hasattr(result, 'json') else {}
        # json 返回格式是 {"res": {...}}，提取 res 字段
        json_data = json_result.get("res", json_result) if isinstance(json_result, dict) else json_result
        json_text = json.dumps(json_data, ensure_ascii=False, indent=2)
        
        # 3. 可视化图片
        img_dict = result.img if hasattr(result, 'img') else {}
        # 优先显示 layout_order_res，如果没有则显示第一个可用的图片
        vis_image = None
        if isinstance(img_dict, dict):
            if "layout_order_res" in img_dict and img_dict["layout_order_res"] is not None:
                vis_image = img_dict["layout_order_res"]
            elif img_dict:
                for key, img in img_dict.items():
                    if img is not None:
                        vis_image = img
                        break
        
        # 4. 结果摘要
        summary = f"""
## 处理结果摘要

- **输入路径**: {result.get('input_path', 'N/A')}
- **页面索引**: {result.get('page_index', 'N/A')}
- **页面总数**: {result.get('page_count', 'N/A')}
- **图像尺寸**: {result.get('width', 'N/A')} x {result.get('height', 'N/A')}
- **解析块数量**: {len(result.get('parsing_res_list', []))}
- **表格数量**: {len(result.get('table_res_list', []))}
"""
        
        return vis_image, markdown_text, json_text, summary
        
    except Exception as e:
        import traceback
        error_msg = f"❌ 处理失败: {str(e)}\n\n{traceback.format_exc()}"
        return None, None, None, error_msg

def create_gradio_interface():
    """创建 Gradio 界面"""
    
    with gr.Blocks(title="PaddleOCR-VL OpenVINO Pipeline", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 📄 PaddleOCR-VL OpenVINO Pipeline
        
        基于 OpenVINO 的文档理解系统，支持：
        - 📊 文档布局检测（PP-DocLayoutV2）
        - 🔤 文本识别（OCR）
        - 📋 表格识别
        - 📈 图表识别
        - 🔢 公式识别
        
        **使用说明**：
        1. 首先在"Pipeline 配置"中初始化 Pipeline（模型路径为空则自动下载）
        2. 上传图片并设置参数
        3. 点击"开始识别"查看结果
        """)
        
        with gr.Tab("Pipeline 配置"):
            gr.Markdown("### 初始化 Pipeline")
            with gr.Row():
                layout_model_path = gr.Textbox(
                    label="布局检测模型路径（可选，为空则自动下载）",
                    placeholder="例如: ./pp_doclayoutv2_f16.xml 或留空自动下载",
                    value=""
                )
                vlm_model_path = gr.Textbox(
                    label="VLM 模型路径（可选，为空则自动下载）",
                    placeholder="例如: ./ov_paddleocr_vl_model 或留空自动下载",
                    value=""
                )
            with gr.Row():
                available_devices = get_available_devices()
                def _pick_default(devices):
                    for pref in ["AUTO","GPU","CPU"]:
                        if pref in devices:
                            return pref
                    return devices[0]
                default_device = _pick_default(available_devices)

                vlm_device = gr.Dropdown(
                    choices=available_devices,
                    value=default_device,
                    label="VLM 推理设备"
                )
                layout_device = gr.Dropdown(
                    choices=available_devices,
                    value=default_device,
                    label="布局检测推理设备"
                )
            
            with gr.Accordion("量化/压缩设置", open=False):
                gr.Markdown("""
                **量化/压缩选项说明**：
                - **LLM INT4 压缩**：对 LLM 模型进行 INT4 量化压缩，可大幅减少模型大小和内存占用
                - **Vision INT8 量化**：对视觉模型进行 INT8 量化，平衡精度和性能
                - **LLM INT8 压缩**：对 LLM 模型进行 INT8 量化压缩
                - **LLM INT8 量化**：对 LLM 模型进行 INT8 量化
                
                ⚠️ **注意**：量化可能会略微降低精度，但可以显著提升推理速度和减少内存占用
                """)
                with gr.Row():
                    llm_int4_compress = gr.Checkbox(
                        label="LLM INT4 压缩",
                        value=False,
                        info="对 LLM 模型进行 INT4 量化压缩"
                    )
                    vision_int8_quant = gr.Checkbox(
                        label="Vision INT8 量化",
                        value=True,
                        info="对视觉模型进行 INT8 量化"
                    )
                with gr.Row():
                    llm_int8_compress = gr.Checkbox(
                        label="LLM INT8 压缩",
                        value=True,
                        info="对 LLM 模型进行 INT8 量化压缩"
                    )
                    llm_int8_quant = gr.Checkbox(
                        label="LLM INT8 量化",
                        value=True,
                        info="对 LLM 模型进行 INT8 量化"
                    )
            
            init_btn = gr.Button("初始化 Pipeline", variant="primary")
            init_status = gr.Textbox(label="初始化状态", interactive=False)
            
            init_btn.click(
                fn=initialize_pipeline,
                inputs=[
                    layout_model_path, 
                    vlm_model_path, 
                    vlm_device, 
                    layout_device,
                    llm_int4_compress,
                    vision_int8_quant,
                    llm_int8_compress,
                    llm_int8_quant
                ],
                outputs=init_status
            )
        
        with gr.Tab("文档识别"):
            gr.Markdown("### 上传图片进行识别")
            
            with gr.Row():
                with gr.Column():
                    input_image = gr.Image(
                        label="上传图片",
                        type="pil",
                        sources=["upload", "clipboard"]
                    )
                    
                    with gr.Accordion("高级设置", open=False):
                        use_layout_detection = gr.Checkbox(
                            label="使用布局检测",
                            value=True
                        )
                        layout_threshold = gr.Slider(
                            minimum=0.1,
                            maximum=1.0,
                            value=0.5,
                            step=0.1,
                            label="布局检测阈值"
                        )
                        max_new_tokens = gr.Slider(
                            minimum=256,
                            maximum=4096,
                            value=1024,
                            step=256,
                            label="最大生成 Token 数"
                        )
                    
                    process_btn = gr.Button("开始识别", variant="primary", size="lg")
                
                with gr.Column():
                    vis_image = gr.Image(
                        label="可视化结果",
                        type="pil"
                    )
            
            with gr.Row():
                with gr.Tab("Markdown 结果"):
                    markdown_output = gr.Markdown(
                        label="Markdown 格式结果",
                        value="识别结果将显示在这里..."
                    )
                
                with gr.Tab("JSON 结果"):
                    json_output = gr.Code(
                        label="JSON 格式结果",
                        language="json",
                        value="识别结果将显示在这里..."
                    )
                
                with gr.Tab("结果摘要"):
                    summary_output = gr.Markdown(
                        label="处理摘要",
                        value="处理摘要将显示在这里..."
                    )
            
            process_btn.click(
                fn=process_image,
                inputs=[input_image, use_layout_detection, layout_threshold, max_new_tokens],
                outputs=[vis_image, markdown_output, json_output, summary_output]
            )
        
        with gr.Tab("使用说明"):
            gr.Markdown("""
            ## 📖 使用说明
            
            ### 1. 初始化 Pipeline
            
            - **布局检测模型路径**：PP-DocLayoutV2 模型的路径（.xml 文件），留空则自动从 ModelScope 下载
            - **VLM 模型路径**：PaddleOCR-VL 模型的目录路径，留空则自动从 ModelScope 下载
            - **VLM 推理设备**：选择 VLM 模型运行的设备（CPU/GPU/AUTO）
            - **布局检测推理设备**：选择布局检测模型运行的设备（CPU/GPU/NPU/AUTO）
            
            #### 量化/压缩设置
            
            - **LLM INT4 压缩**：对 LLM 模型进行 INT4 量化压缩，可大幅减少模型大小和内存占用（默认：False）
            - **Vision INT8 量化**：对视觉模型进行 INT8 量化，平衡精度和性能（默认：True）
            - **LLM INT8 压缩**：对 LLM 模型进行 INT8 量化压缩（默认：True）
            - **LLM INT8 量化**：对 LLM 模型进行 INT8 量化（默认：True）
            
            ⚠️ **注意**：量化可能会略微降低精度，但可以显著提升推理速度和减少内存占用。建议根据实际需求调整这些设置。
            
            ### 2. 文档识别
            
            - **上传图片**：支持上传图片文件或从剪贴板粘贴
            - **使用布局检测**：是否启用文档布局检测
            - **布局检测阈值**：布局检测的置信度阈值（0.1-1.0）
            - **最大生成 Token 数**：VLM 模型生成的最大 token 数量（256-4096）
            
            ### 3. 查看结果
            
            - **可视化结果**：显示带检测框和编号的可视化图片
            - **Markdown 结果**：以 Markdown 格式显示识别结果，包含文本、表格、公式等
            - **JSON 结果**：以 JSON 格式显示完整的识别结果数据
            - **结果摘要**：显示处理的基本信息和统计
            
            ### 4. 支持的文档元素
            
            - 📝 文本（Text）
            - 📋 表格（Table）
            - 📊 图表（Chart）
            - 🔢 公式（Formula）
            - 🖼️ 图片（Image）
            - 📑 标题（Title）
            - 📄 段落（Paragraph）
            
            ### 5. 注意事项
            
            - 首次使用需要下载模型，请确保网络连接正常
            - 模型较大，下载可能需要一些时间
            - 建议使用 GPU 设备以获得更好的性能
            - 支持的图片格式：PNG, JPG, JPEG 等常见格式
            """)
    
    return demo

if __name__ == "__main__":
    demo = create_gradio_interface()
    demo.launch(
        server_name="0.0.0.0",  # 允许外部访问
        server_port=7860,        # 端口号
        share=False,             # 是否创建公共链接
        inbrowser=True           # 自动在浏览器中打开
    )

