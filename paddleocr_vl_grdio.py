import gradio as gr
import torch
from PIL import Image
import time
import openvino as ov
from transformers.utils.chat_template_utils import render_jinja_template
from ov_paddleocr_vl import OVPaddleOCRVLForCausalLM
from image_processing_paddleocr_vl import PaddleOCRVLImageProcessor
import requests
from pathlib import Path
from urllib.parse import urlparse
import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端
import io
import base64
import multiprocessing
from multiprocessing import Process, Queue, Manager
import threading

# 在导入后立即设置环境变量，避免Gradio初始化时的网络请求
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("GRADIO_SERVER_NAME", "127.0.0.1")
os.environ.setdefault("NO_PROXY", "127.0.0.1,localhost")
os.environ.setdefault("no_proxy", "127.0.0.1,localhost")

# 全局变量
paddleocr_vl_model = None
my_preprocessor = None
init_process = None  # 初始化进程对象
init_process_pid = None  # 初始化进程ID
init_status_queue = None  # 用于进程间通信的队列
init_params = None  # 保存初始化参数
_accumulated_status_messages = []  # 累积的状态消息，用于保存所有验证步骤信息

# 任务提示词
PROMPTS = {
    "ocr": "OCR:",
    "table": "Table Recognition:",
    "formula": "Formula Recognition:",
    "chart": "Chart Recognition:",
}

# Chat模板（从chat_template.jinja文件读取）
CHAT_TEMPLATE = '''{%- if not add_generation_prompt is defined -%}
    {%- set add_generation_prompt = true -%}
{%- endif -%}
{%- if not cls_token is defined -%}
    {%- set cls_token = "<|begin_of_sentence|>" -%}
{%- endif -%}
{%- if not eos_token is defined -%}
    {%- set eos_token = "</s>" -%}
{%- endif -%}
{%- if not image_token is defined -%}
    {%- set image_token = "<|IMAGE_START|><|IMAGE_PLACEHOLDER|><|IMAGE_END|>" -%}
{%- endif -%}
{{- cls_token -}}
{%- for message in messages -%}
    {%- if message["role"] == "user" -%}
        {{- "User: " -}}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "image" -%}
                {{ image_token }}
            {%- endif -%}
        {%- endfor -%}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "text" -%}
                {{ content["text"] }}
            {%- endif -%}
        {%- endfor -%}
        {{ "\\n" -}}
    {%- elif message["role"] == "assistant" -%}
        {{- "Assistant: " -}}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "text" -%}
                {{ content["text"] }}
            {%- endif -%}
        {%- endfor -%}
        {{ eos_token -}}
    {%- elif message["role"] == "system" -%}
        {%- for content in message["content"] -%}
            {%- if content["type"] == "text" -%}
                {{ content["text"] + "\\n" }}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endfor -%}
{%- if add_generation_prompt -%}
    {{- "Assistant: " -}}
{%- endif -%}'''

def load_chat_template(template_path=None):
    """加载chat模板"""
    global CHAT_TEMPLATE
    if template_path:
        try:
            with open(template_path, 'r', encoding='utf-8') as f:
                CHAT_TEMPLATE = f.read()
            return f"✅ 已从文件加载模板: {template_path}"
        except Exception as e:
            return f"❌ 加载模板失败: {str(e)}，使用默认模板"
    return "使用默认模板"

def _initialize_model_process(
    ov_model_path,
    device_type,
    llm_int4_compress,
    vision_int8_quant,
    llm_int8_quant,
    template_path,
    status_queue,
    result_queue
):
    """
    在独立进程中验证模型可以加载
    注意：由于模型对象无法在进程间共享，此函数主要用于验证模型文件
    实际模型对象需要在主进程中初始化
    """
    import os
    process_id = os.getpid()
    
    try:
        status_queue.put(f"🔄 进程 {process_id} 开始验证模型...")
        
        # 验证模型文件是否存在
        model_path = Path(ov_model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"模型路径不存在: {ov_model_path}")
        
        status_queue.put(f"✅ 模型路径验证通过: {ov_model_path}")
        
        # 验证模型文件
        if llm_int4_compress:
            llm_model_file = model_path / "llm_stateful_int4.xml"
        else:
            llm_model_file = model_path / "llm_stateful.xml"
        
        if not llm_model_file.exists():
            raise FileNotFoundError(f"LLM模型文件不存在: {llm_model_file}")
        
        status_queue.put(f"✅ LLM模型文件验证通过")
        
        # 验证vision模型文件
        if vision_int8_quant:
            vision_model_file = model_path / "vision_int8.xml"
        else:
            vision_model_file = model_path / "vision.xml"
        
        if not vision_model_file.exists():
            raise FileNotFoundError(f"Vision模型文件不存在: {vision_model_file}")
        
        status_queue.put(f"✅ Vision模型文件验证通过")
        
        # 验证chat模板
        if template_path:
            try:
                with open(template_path, 'r', encoding='utf-8') as f:
                    template_content = f.read()
                status_queue.put(f"✅ Chat模板文件验证通过: {template_path}")
            except Exception as e:
                status_queue.put(f"⚠️ Chat模板文件验证失败: {str(e)}，将使用默认模板")
        
        # 尝试读取模型（验证模型文件完整性）
        status_queue.put(f"🔄 正在验证模型文件完整性...")
        core = ov.Core()
        try:
            core.read_model(str(llm_model_file))
            status_queue.put(f"✅ 模型文件完整性验证通过")
        except Exception as e:
            raise Exception(f"模型文件读取失败: {str(e)}")
        
        # 返回成功结果
        result_queue.put({
            'success': True,
            'message': f"✅ 模型验证成功！进程ID: {process_id}",
            'process_id': process_id,
            'ov_model_path': ov_model_path,
            'device_type': device_type,
            'llm_int4_compress': llm_int4_compress,
            'vision_int8_quant': vision_int8_quant,
            'llm_int8_quant': llm_int8_quant,
            'template_path': template_path
        })
        status_queue.put(f"✅ 模型验证完成！进程ID: {process_id}")
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        result_queue.put({
            'success': False,
            'message': f"❌ 模型验证失败: {str(e)}",
            'error': error_detail,
            'process_id': process_id
        })
        status_queue.put(f"❌ 模型验证失败: {str(e)}")

def initialize_model(ov_model_path="./ov_paddleocr_vl_model", 
                     device_type="GPU", 
                     llm_int4_compress=False, 
                     vision_int8_quant=False, 
                     llm_int8_quant=False,
                     template_path=None):
    """
    初始化模型（在独立进程中验证，然后在主进程中初始化）
    返回初始化状态和进程ID
    """
    global paddleocr_vl_model, my_preprocessor, init_process, init_process_pid, init_status_queue, init_params
    
    # 如果已有初始化进程在运行，先等待其完成或终止
    if init_process is not None and init_process.is_alive():
        status, pid = check_init_status()
        return status, str(pid) if pid else "运行中"
    
    try:
        # 保存初始化参数
        init_params = {
            'ov_model_path': ov_model_path,
            'device_type': device_type,
            'llm_int4_compress': llm_int4_compress,
            'vision_int8_quant': vision_int8_quant,
            'llm_int8_quant': llm_int8_quant,
            'template_path': template_path
        }
        
        # 创建进程间通信的队列
        status_queue = Queue()
        result_queue = Queue()
        
        # 创建新进程来验证模型
        init_process = Process(
            target=_initialize_model_process,
            args=(
                ov_model_path,
                device_type,
                llm_int4_compress,
                vision_int8_quant,
                llm_int8_quant,
                template_path,
                status_queue,
                result_queue
            )
        )
        
        # 开始新的验证时，清空之前的累积消息
        global _accumulated_status_messages
        _accumulated_status_messages = []
        
        init_process.start()
        init_process_pid = init_process.pid
        init_status_queue = status_queue
        
        # 启动一个线程来监控进程并在验证成功后初始化模型
        def monitor_and_init():
            global paddleocr_vl_model, my_preprocessor, init_process
            if init_process is not None:
                init_process.join()  # 等待进程完成
                
                # 获取验证结果
                if not result_queue.empty():
                    result = result_queue.get()
                    if result['success']:
                        # 在主进程中初始化模型对象
                        try:
                            # 加载chat模板
                            if result['template_path']:
                                load_chat_template(result['template_path'])
                            
                            # 初始化OpenVINO模型
                            core = ov.Core()
                            llm_infer_list = []
                            vision_infer = []
                            
                            paddleocr_vl_model = OVPaddleOCRVLForCausalLM(
                                core=core,
                                ov_model_path=result['ov_model_path'],
                                device=result['device_type'],
                                llm_int4_compress=result['llm_int4_compress'],
                                vision_int8_quant=result['vision_int8_quant'],
                                llm_int8_quant=result['llm_int8_quant'],
                                llm_infer_list=llm_infer_list,
                                vision_infer=vision_infer
                            )
                            
                            # 初始化图像预处理器
                            my_preprocessor = PaddleOCRVLImageProcessor(
                                resample=3,
                                rescale_factor=0.00392156862745098,
                                image_mean=[0.5, 0.5, 0.5],
                                image_std=[0.5, 0.5, 0.5],
                                min_pixels=147384,
                                max_pixels=2822400,
                                patch_size=14,
                                temporal_patch_size=1,
                                merge_size=2
                            )
                            
                            status_queue.put(f"✅ 模型对象初始化完成！")
                        except Exception as e:
                            import traceback
                            error_detail = traceback.format_exc()
                            status_queue.put(f"❌ 模型对象初始化失败: {str(e)}\n{error_detail}")
        
        monitor_thread = threading.Thread(target=monitor_and_init, daemon=True)
        monitor_thread.start()
        
        status_msg = f"🔄 模型初始化进程已启动，进程ID: {init_process_pid}\n正在后台验证模型，请稍候..."
        return status_msg, str(init_process_pid)
        
    except Exception as e:
        return f"❌ 启动初始化进程失败: {str(e)}", "错误"

def safe_encode_text(text):
    """
    安全编码文本，确保UTF-8编码正确，避免Content-Length错误
    """
    if text is None:
        return ""
    try:
        # 确保是字符串类型
        if not isinstance(text, str):
            text = str(text)
        # 编码为UTF-8，处理任何编码错误
        text = text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
        # 限制长度，避免响应过大
        max_length = 5000  # 限制最大长度为5000字符
        if len(text) > max_length:
            text = text[:max_length] + "\n...(内容已截断)"
        return text
    except Exception as e:
        # 如果编码失败，返回安全的错误消息
        return f"状态信息编码错误: {str(e)}"

# 用于缓存上次状态，避免重复创建相同内容
_last_status_cache = {"text": "", "pid": "", "timestamp": 0}

def check_init_status():
    """
    检查初始化进程的状态
    返回当前状态信息和进程ID
    优化：使用缓存避免重复创建相同内容，减少内存占用
    验证通过的消息（包含✅）会一直保存，不会被刷新
    """
    global init_process, init_process_pid, init_status_queue, _last_status_cache, _accumulated_status_messages
    
    try:
        # 如果没有进程，返回缓存的状态（避免频繁创建新字符串）
        if init_process is None:
            cache_key = "waiting"
            if _last_status_cache.get("key") != cache_key:
                _last_status_cache = {
                    "key": cache_key,
                    "text": safe_encode_text("等待初始化..."),
                    "pid": "未启动",
                    "timestamp": time.time()
                }
            return _last_status_cache["text"], _last_status_cache["pid"]
        
        if init_status_queue is None:
            cache_key = f"started_{init_process_pid}"
            if _last_status_cache.get("key") != cache_key:
                status_msg = f"进程 {init_process_pid} 已启动，等待状态更新..."
                _last_status_cache = {
                    "key": cache_key,
                    "text": safe_encode_text(status_msg),
                    "pid": str(init_process_pid) if init_process_pid else "未启动",
                    "timestamp": time.time()
                }
            return _last_status_cache["text"], _last_status_cache["pid"]
        
        # 收集状态消息（限制数量，避免队列阻塞和内存累积）
        status_messages = []
        max_messages = 10  # 减少到10条消息，降低内存占用
        message_count = 0
        
        # 限制读取时间，避免阻塞
        start_time = time.time()
        timeout = 0.1  # 100ms超时
        
        # 先读取队列中的新消息，追加到累积列表中
        new_messages = []
        try:
            while not init_status_queue.empty():
                try:
                    msg = init_status_queue.get_nowait()
                    if msg:
                        msg_str = str(msg)
                        # 验证通过的消息（包含✅）需要永久保存
                        if '✅' in msg_str or '❌' in msg_str:
                            # 重要消息：追加到累积列表
                            if len(msg_str) > 200:
                                msg_str = msg_str[:200] + "..."
                            _accumulated_status_messages.append(msg_str)
                        new_messages.append(msg_str)
                except Exception:
                    break
        except Exception:
            pass
        
        # 使用累积的消息列表，但限制最大数量避免内存溢出
        max_accumulated = 50  # 最多保存50条累积消息
        if len(_accumulated_status_messages) > max_accumulated:
            # 保留最近的50条消息（保留验证通过的消息）
            _accumulated_status_messages = _accumulated_status_messages[-max_accumulated:]
        
        # 组合累积消息和新消息
        status_messages = _accumulated_status_messages.copy()
        
        # 添加新的非重复消息（非验证通过的消息）
        for msg in new_messages:
            if '✅' not in msg and '❌' not in msg:  # 非验证消息
                if len(msg) > 200:
                    msg = msg[:200] + "..."
                if msg not in status_messages:
                    status_messages.append(msg)
                    message_count += 1
        
        # 检查进程是否还在运行
        try:
            is_alive = init_process.is_alive() if init_process is not None else False
        except Exception:
            is_alive = False
        
        # 生成状态文本
        if status_messages:
            status_text = "\n".join(status_messages)
            if not is_alive:
                status_text += f"\n✅ 进程 {init_process_pid} 已完成"
        else:
            if is_alive:
                status_text = f"🔄 进程 {init_process_pid} 正在运行中..."
            else:
                status_text = f"✅ 进程 {init_process_pid} 已完成"
        
        # 使用缓存避免重复创建相同内容
        cache_key = f"{hash(status_text)}_{init_process_pid}_{is_alive}"
        if _last_status_cache.get("key") != cache_key:
            _last_status_cache = {
                "key": cache_key,
                "text": safe_encode_text(status_text),
                "pid": str(init_process_pid) if init_process_pid else "未启动",
                "timestamp": time.time()
            }
        
        return _last_status_cache["text"], _last_status_cache["pid"]
            
    except Exception as e:
        # 捕获所有异常，返回安全的错误消息
        error_msg = f"检查状态时出错: {str(e)}"
        return safe_encode_text(error_msg), "错误"

def unload_model():
    """
    卸载模型，kill初始化进程并清理所有资源
    返回卸载状态和进程ID
    """
    global paddleocr_vl_model, my_preprocessor, init_process, init_process_pid, init_status_queue, init_params, _last_status_cache
    
    try:
        killed_pid = None
        status_messages = []
        
        # 1. 终止初始化进程（如果存在且正在运行）
        if init_process is not None:
            if init_process.is_alive():
                killed_pid = init_process.pid
                try:
                    # 先尝试优雅终止
                    init_process.terminate()
                    # 等待最多2秒
                    init_process.join(timeout=2)
                    
                    # 如果进程仍在运行，强制kill
                    if init_process.is_alive():
                        init_process.kill()
                        init_process.join(timeout=1)
                        status_messages.append(f"⚠️ 进程 {killed_pid} 已被强制终止")
                    else:
                        status_messages.append(f"✅ 进程 {killed_pid} 已终止")
                except Exception as e:
                    status_messages.append(f"⚠️ 终止进程 {killed_pid} 时出错: {str(e)}")
                    # 尝试强制kill
                    try:
                        if init_process.is_alive():
                            init_process.kill()
                            init_process.join(timeout=1)
                            status_messages.append(f"✅ 进程 {killed_pid} 已被强制终止")
                    except:
                        pass
            else:
                killed_pid = init_process_pid
                status_messages.append(f"ℹ️ 进程 {killed_pid} 已结束")
        
        # 2. 清理模型对象
        if paddleocr_vl_model is not None:
            try:
                # 如果模型有unload方法，调用它
                if hasattr(paddleocr_vl_model, 'unload'):
                    paddleocr_vl_model.unload()
                paddleocr_vl_model = None
                status_messages.append("✅ 模型对象已清理")
            except Exception as e:
                status_messages.append(f"⚠️ 清理模型对象时出错: {str(e)}")
                paddleocr_vl_model = None
        
        # 3. 清理预处理器
        if my_preprocessor is not None:
            my_preprocessor = None
            status_messages.append("✅ 预处理器已清理")
        
        # 4. 清理进程相关变量
        init_process = None
        if init_process_pid is not None:
            old_pid = init_process_pid
            init_process_pid = None
            if killed_pid is None:
                killed_pid = old_pid
        else:
            if killed_pid is None:
                killed_pid = "无"
        
        # 5. 清理队列（彻底清空，避免内存泄漏）
        if init_status_queue is not None:
            # 清空队列，限制尝试次数避免无限循环
            try:
                max_attempts = 100  # 最多尝试100次
                attempt = 0
                while not init_status_queue.empty() and attempt < max_attempts:
                    try:
                        init_status_queue.get_nowait()
                        attempt += 1
                    except:
                        break
                # 确保队列被完全清空
                try:
                    while not init_status_queue.empty():
                        init_status_queue.get_nowait()
                except:
                    pass
            except:
                pass
            init_status_queue = None
            status_messages.append("✅ 状态队列已清理")
        
        # 6. 清理参数
        init_params = None
        
        # 7. 清理状态缓存和累积消息
        _last_status_cache = {"text": "", "pid": "", "timestamp": 0}
        global _accumulated_status_messages
        _accumulated_status_messages = []  # 清空累积的状态消息
        
        # 8. 清理Gradio相关资源
        try:
            # 清理Gradio的内部缓存和会话状态
            # 注意：Gradio的某些资源可能无法直接清理，但我们可以清理我们能控制的部分
            status_messages.append("🔄 正在清理Gradio资源...")
            
            # 清理matplotlib的缓存
            try:
                plt.close('all')  # 关闭所有matplotlib图形
                matplotlib.pyplot.ioff()  # 关闭交互模式
                # 清理matplotlib的后端缓存
                if hasattr(matplotlib.pyplot, 'clear'):
                    matplotlib.pyplot.clear()
            except:
                pass
            
            # 清理PIL图像缓存
            try:
                # PIL图像对象会在垃圾回收时自动清理，这里只是确保没有引用
                pass
            except:
                pass
            
            # 清理pandas缓存
            try:
                # pandas的缓存主要在DataFrame对象中，会在垃圾回收时清理
                pass
            except:
                pass
            
            # 清理torch缓存（如果有）
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()  # 清空CUDA缓存
                torch.cuda.synchronize() if torch.cuda.is_available() else None
            except:
                pass
            
            # 清理OpenVINO缓存（如果有）
            try:
                # OpenVINO的Core对象可能持有缓存
                # 注意：这里不能直接清理，因为core对象可能还在使用
                pass
            except:
                pass
            
            # 清理Python内置缓存
            try:
                import sys
                # 清理模块缓存中的临时对象（谨慎使用）
                # sys.modules 不应该被清理，但可以清理一些临时变量
                pass
            except:
                pass
            
            status_messages.append("✅ Gradio资源清理完成")
        except Exception as e:
            status_messages.append(f"⚠️ 清理Gradio资源时出错: {str(e)}")
        
        # 9. 强制垃圾回收（多次执行，确保彻底清理）
        import gc
        for _ in range(3):  # 执行3次垃圾回收，确保彻底清理
            gc.collect()
        status_messages.append("✅ 内存已清理")
        
        # 生成状态消息
        if status_messages:
            status_text = "\n".join(status_messages)
        else:
            status_text = "✅ 模型已卸载（无运行中的进程）"
        
        status_text = f"🔄 卸载模型完成\n\n{status_text}"
        
        return safe_encode_text(status_text), "未启动"
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"❌ 卸载模型失败: {str(e)}\n\n详细信息:\n{error_detail}", "错误"

def convert_latex_format(text):
    r"""
    将LaTeX格式转换为Gradio Markdown可识别的格式
    - \[...\] -> $$...$$
    - \(...\) -> $...$
    支持多行公式
    """
    if not text:
        return text
    
    # 将 \[...\] 转换为 $$...$$
    # 使用非贪婪匹配，处理多行公式
    # 先处理 \[，再处理 \]
    text = re.sub(r'\\\[', '$$', text)
    text = re.sub(r'\\\]', '$$', text)
    
    # 将 \(...\) 转换为 $...$
    text = re.sub(r'\\\(', '$', text)
    text = re.sub(r'\\\)', '$', text)
    
    # 修复可能出现的 $$ $$ 中间有换行的情况
    # 将 $$...\n...$$ 中的换行替换为空格
    def fix_formula_newlines(match):
        formula = match.group(1)
        # 移除换行，保留空格
        formula = re.sub(r'\n+', ' ', formula)
        formula = re.sub(r'\s+', ' ', formula)
        return f'$${formula.strip()}$$'
    
    # 匹配 $$...$$ 之间的内容（包括换行）
    text = re.sub(r'\$\$(.*?)\$\$', fix_formula_newlines, text, flags=re.DOTALL)
    
    return text

def detect_and_format_latex(text):
    """
    检测文本中的LaTeX公式并格式化
    支持检测常见的数学公式模式
    """
    if not text:
        return text, False
    
    # 首先检查是否已经包含LaTeX格式标记
    has_latex_markers = bool(re.search(r'\\\[|\\\]|\\\(|\\\)|\$\$|\$[^$]+\$', text))
    
    # LaTeX公式的常见模式
    latex_patterns = [
        r'[a-zA-Z]\([^)]*\)\s*=\s*[0-9]+\*?[a-zA-Z0-9\^+\-*/\s]+',  # f(x)=2x^2+2x+3
        r'[a-zA-Z]\([^)]*\)\s*=\s*[a-zA-Z0-9\^+\-*/\s]+',  # f(a)=2a^2+2a+3
        r'\\frac\{[^}]+\}\{[^}]+\}',  # 分数
        r'\\sqrt\{[^}]+\}',  # 根号
        r'\\sum_\{[^}]+\}\^\{[^}]+\}',  # 求和
        r'\\int_\{[^}]+\}\^\{[^}]+\}',  # 积分
        r'[a-zA-Z]\^\{[0-9]+\}',  # 上标 x^{2}
        r'[a-zA-Z]_\{[0-9]+\}',  # 下标 x_{i}
        r'\\cdot',  # 点乘
        r'\\quad',  # 空格
    ]
    
    has_latex = has_latex_markers
    if not has_latex:
        for pattern in latex_patterns:
            if re.search(pattern, text):
                has_latex = True
                break
    
    # 转换LaTeX格式
    if has_latex:
        text = convert_latex_format(text)
    
    return text, has_latex

def needs_table_header(first_cells, has_data_rows=True):
    """
    通用检测：判断表格第一行是否需要添加表头
    
    Args:
        first_cells: 第一行的单元格列表
        has_data_rows: 是否有数据行（至少2行）
    
    Returns:
        bool: 如果需要添加表头返回True，否则返回False
    """
    if len(first_cells) < 2 or not has_data_rows:
        return False
    
    # 数据行的特征模式
    data_like_patterns = [
        r'^\d{4}$',  # 4位年份
        r'^[12]Q\d{2}$',  # 季度格式 1Q22, 2Q23
        r'^\d+\.?\d*%?$',  # 数字或百分比
        r'^\d{4}[-/]\d{1,2}[-/]\d{1,2}$',  # 日期格式
        r'^[A-Z]{2,}\d+$',  # 代码格式如 ABC123
        r'^\d+$',  # 纯数字
    ]
    
    # 表头关键词（如果包含这些词，认为是表头）
    header_keywords = ['项目', '类别', '名称', '类型', 'type', 'category', 
                      'item', 'name', 'label', '标题', 'header', '列', 'column']
    
    # 检查第一行是否包含表头关键词
    has_header_keyword = any(
        keyword.lower() in cell.lower() 
        for cell in first_cells 
        for keyword in header_keywords
    )
    
    # 如果包含表头关键词，不需要添加表头
    if has_header_keyword:
        return False
    
    # 统计第一行中像数据的单元格数量
    data_like_count = 0
    for cell in first_cells:
        # 检查是否匹配数据模式
        if any(re.match(pattern, cell) for pattern in data_like_patterns):
            data_like_count += 1
    
    # 如果80%以上像数据，则需要添加表头
    return data_like_count >= len(first_cells) * 0.8

def format_ocr_result(text):
    """
    格式化OCR识别结果，处理特殊标记
    支持格式：
    - <fcel> 表格单元格标记（格式：<fcel>内容<fcel>）
    - <nl> 换行标记
    - LaTeX数学公式（自动检测并格式化）
    
    注意：格式是 <fcel>内容<fcel>，即开始和结束都是 <fcel>
    只有检测到表格格式时才转换为Markdown表格，否则只清理标记
    """
    if not text:
        return text
    
    # 先替换换行标记
    text = text.replace('<nl>', '\n')
    
    # 检测LaTeX公式
    text, has_latex = detect_and_format_latex(text)
    
    # 检测是否是表格格式（包含多个<fcel>标记）
    # 需要检查是否有多个<fcel>标记，且至少有一行包含多个单元格
    is_table_format = False
    if '<fcel>' in text:
        # 检查是否有多行包含<fcel>标记，或者单行包含多个<fcel>标记
        lines_with_fcel = [line for line in text.split('\n') if '<fcel>' in line]
        if len(lines_with_fcel) > 0:
            # 检查第一行是否有多个<fcel>标记（至少2个，表示有多个单元格）
            first_line_fcel_count = lines_with_fcel[0].count('<fcel>')
            if first_line_fcel_count >= 2:
                is_table_format = True
    
    if is_table_format:
        # 按行分割
        lines = text.split('\n')
        table_rows = []
        
        for line in lines:
            if '<fcel>' in line:
                # 使用正则表达式提取所有 <fcel>内容<fcel> 格式的单元格
                # 格式是 <fcel>内容<fcel>，所以需要匹配 <fcel> 到下一个 <fcel> 之间的内容
                # 使用非贪婪匹配，但需要确保匹配所有单元格
                
                # 方法：找到所有 <fcel> 标记的位置，然后提取每对之间的内容
                fcel_positions = [m.start() for m in re.finditer(r'<fcel>', line)]
                
                if len(fcel_positions) >= 2:
                    row_cells = []
                    # 每两个连续的 <fcel> 之间是一个单元格
                    for i in range(0, len(fcel_positions) - 1):
                        start_pos = fcel_positions[i] + len('<fcel>')
                        end_pos = fcel_positions[i + 1]
                        cell_content = line[start_pos:end_pos].strip()
                        row_cells.append(cell_content)
                    
                    # 如果最后一个 <fcel> 后面还有内容（没有结束的 <fcel>），也提取
                    if len(fcel_positions) > 0:
                        last_fcel_pos = fcel_positions[-1] + len('<fcel>')
                        # 检查最后一个 <fcel> 后面是否还有内容（不是换行符或结束）
                        remaining = line[last_fcel_pos:].strip()
                        # 移除可能的 <nl> 标记
                        remaining = remaining.replace('<nl>', '').strip()
                        if remaining:
                            row_cells.append(remaining)
                    
                    if row_cells:
                        table_rows.append(row_cells)
        
        if table_rows and len(table_rows) > 0:
            # 找到最大列数（用于对齐）
            max_cols = max(len(row) for row in table_rows)
            
            # 转换为Markdown表格
            md_table = ""
            
            # 创建表头（第一行）
            if len(table_rows) > 0:
                header = table_rows[0].copy()
                # 补齐列数
                while len(header) < max_cols:
                    header.append("")
                md_table += "| " + " | ".join(header) + " |\n"
                md_table += "| " + " | ".join(["---"] * max_cols) + " |\n"
                
                # 添加数据行
                for row in table_rows[1:]:
                    row_copy = row.copy()
                    # 确保行长度与最大列数一致
                    while len(row_copy) < max_cols:
                        row_copy.append("")
                    md_table += "| " + " | ".join(row_copy[:max_cols]) + " |\n"
            
            return md_table
    
    # 如果不是表格格式，只清理标记，保持原始文本格式
    # 移除<fcel>标记，但保留其他内容和换行
    text = text.replace('<fcel>', '')
    text = text.replace('</fcel>', '')
    # <nl>已经在开头替换为\n了，这里不需要再处理
    
    # 检测并修复不完整的Markdown表格格式
    # 例如: "| 2017 | 2018 | ..." 或 "2017 | 2018 | ..." (缺少开头|)
    lines = text.split('\n')
    cleaned_lines = []
    prev_empty = False
    
    # 检测是否是表格格式（包含多个|符号）
    pipe_count = sum(line.count('|') for line in lines if line.strip())
    if pipe_count >= 5:  # 至少5个|符号，可能是表格
        # 尝试修复表格格式
        table_lines = []
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                if not prev_empty:
                    cleaned_lines.append('')
                    prev_empty = True
                continue
            
            # 如果包含|符号，可能是表格行
            if '|' in line_stripped:
                # 如果行首没有|，添加一个
                if not line_stripped.startswith('|'):
                    line_stripped = '| ' + line_stripped
                # 如果行尾没有|，添加一个
                if not line_stripped.endswith('|'):
                    line_stripped = line_stripped + ' |'
                table_lines.append(line_stripped)
                prev_empty = False
            else:
                # 如果不是表格行，先处理之前收集的表格行
                if table_lines:
                    # 通用检测：判断第一行是否需要添加表头
                    first_line = table_lines[0]
                    first_cells = [c.strip() for c in first_line.split('|') if c.strip()]
                    needs_header = needs_table_header(first_cells, has_data_rows=len(table_lines) > 1)
                    
                    if needs_header:
                        # 第一行是数据行，需要添加表头
                        # 添加"项目"作为第一列的表头
                        header = "| 项目 | " + " | ".join(first_cells) + " |"
                        separator = "| " + " | ".join(["---"] * (len(first_cells) + 1)) + " |"
                        cleaned_lines.append(header)
                        cleaned_lines.append(separator)
                        # 添加数据行
                        for data_line in table_lines[1:]:
                            cleaned_lines.append(data_line)
                    else:
                        # 标准表格格式，直接添加
                        for table_line in table_lines:
                            cleaned_lines.append(table_line)
                        # 如果第一行不是分隔行，添加分隔行
                        if cleaned_lines and '---' not in cleaned_lines[-1]:
                            first_table_line = table_lines[0]
                            num_cols = first_table_line.count('|') - 1
                            if num_cols > 0:
                                separator = "| " + " | ".join(["---"] * num_cols) + " |"
                                # 在表头后插入分隔行
                                cleaned_lines.insert(-len(table_lines) + 1, separator)
                    
                    table_lines = []
                
                cleaned_lines.append(line_stripped)
                prev_empty = False
        
        # 处理最后的表格行
        if table_lines:
            # 通用检测：判断第一行是否需要添加表头
            first_line = table_lines[0]
            first_cells = [c.strip() for c in first_line.split('|') if c.strip()]
            needs_header = needs_table_header(first_cells, has_data_rows=len(table_lines) > 1)
            
            if needs_header:
                header = "| 项目 | " + " | ".join(first_cells) + " |"
                separator = "| " + " | ".join(["---"] * (len(first_cells) + 1)) + " |"
                cleaned_lines.append(header)
                cleaned_lines.append(separator)
                for data_line in table_lines[1:]:
                    cleaned_lines.append(data_line)
            else:
                for table_line in table_lines:
                    cleaned_lines.append(table_line)
                if cleaned_lines and '---' not in cleaned_lines[-1]:
                    first_table_line = table_lines[0]
                    num_cols = first_table_line.count('|') - 1
                    if num_cols > 0:
                        separator = "| " + " | ".join(["---"] * num_cols) + " |"
                        cleaned_lines.insert(-len(table_lines) + 1, separator)
    else:
        # 不是表格格式，正常处理
        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
                prev_empty = False
            elif not prev_empty:
                cleaned_lines.append('')
                prev_empty = True
    
    result = '\n'.join(cleaned_lines)
    
    # 如果检测到LaTeX公式，尝试格式化
    if has_latex:
        # 首先处理已经存在的 $$...$$ 格式，合并换行
        def fix_multiline_formula(match):
            formula = match.group(1)
            # 移除多余的换行和空白，但保留必要的空格
            formula = re.sub(r'\n+', ' ', formula)
            formula = re.sub(r'\s+', ' ', formula)
            formula = formula.strip()
            return f'$${formula}$$'
        
        # 修复被换行分割的公式：合并 $$...$$ 之间的换行
        result = re.sub(r'\$\$(.*?)\$\$', fix_multiline_formula, result, flags=re.DOTALL)
        
        # 尝试将常见的数学表达式转换为LaTeX格式
        # 例如: f(x)=2x^2+2x+3 -> f(x)=2x^{2}+2x+3
        result = re.sub(r'(\w+)\^(\d+)', r'\1^{\2}', result)  # x^2 -> x^{2}
        result = re.sub(r'(\w+)_(\d+)', r'\1_{\2}', result)  # x_2 -> x_{2}
        
        # 如果公式还没有被 $$ 包围，尝试识别并添加
        if '$$' not in result:
            # 尝试识别公式片段并包围
            lines = result.split('\n')
            formatted_lines = []
            formula_buffer = []
            collecting_formula = False
            
            for line in lines:
                line_stripped = line.strip()
                
                # 检测是否是公式的一部分（包含数学符号或函数表达式）
                is_formula_line = bool(
                    re.search(r'[a-zA-Z]\([^)]+\)\s*=', line_stripped) or  # f(x)=...
                    re.search(r'[a-zA-Z]\([^)]+\)\s*=\s*[0-9]', line_stripped) or  # f(3)=27
                    ('^{' in line_stripped) or  # 上标
                    ('\\cdot' in line_stripped) or  # 点乘
                    ('\\quad' in line_stripped) or  # 空格
                    (re.search(r'[a-zA-Z]\^\{[0-9]+\}', line_stripped))  # x^{2}
                )
                
                # 检测是否是单个字符（可能是被分割的公式片段）
                is_single_char = len(line_stripped) == 1 and line_stripped.isalnum()
                
                if is_formula_line or (is_single_char and collecting_formula):
                    if not collecting_formula:
                        collecting_formula = True
                        formula_buffer = [line_stripped]
                    else:
                        formula_buffer.append(line_stripped)
                else:
                    if collecting_formula and formula_buffer:
                        # 结束公式，合并并添加 $$ 包围
                        formula_text = ' '.join(formula_buffer)
                        # 清理公式文本
                        formula_text = re.sub(r'\s+', ' ', formula_text).strip()
                        if formula_text:
                            formatted_lines.append(f'$${formula_text}$$')
                        formula_buffer = []
                        collecting_formula = False
                    
                    if line_stripped:  # 非空行
                        formatted_lines.append(line)
                    elif not formatted_lines or formatted_lines[-1]:  # 保留空行（如果前一行不为空）
                        formatted_lines.append('')
            
            # 处理最后的公式
            if collecting_formula and formula_buffer:
                formula_text = ' '.join(formula_buffer)
                formula_text = re.sub(r'\s+', ' ', formula_text).strip()
                if formula_text:
                    formatted_lines.append(f'$${formula_text}$$')
            
            if formatted_lines:
                result = '\n'.join(formatted_lines)
    
    return result

def load_image_from_source(image_source):
    """从不同来源加载图片：PIL Image对象、本地路径或URL"""
    if image_source is None:
        return None
    
    # 如果已经是PIL Image对象，直接返回
    if isinstance(image_source, Image.Image):
        return image_source
    
    # 如果是字符串，判断是URL还是本地路径
    if isinstance(image_source, str):
        # 检查是否是URL
        parsed = urlparse(image_source)
        if parsed.scheme in ('http', 'https'):
            # 从URL下载图片
            try:
                response = requests.get(image_source, stream=True, timeout=10)
                response.raise_for_status()
                image = Image.open(response.raw)
                return image
            except Exception as e:
                raise Exception(f"无法从URL加载图片: {str(e)}")
        else:
            # 本地文件路径
            try:
                path = Path(image_source)
                if not path.exists():
                    raise FileNotFoundError(f"文件不存在: {image_source}")
                image = Image.open(image_source)
                return image
            except Exception as e:
                raise Exception(f"无法从本地路径加载图片: {str(e)}")
    
    return image_source

def parse_table_data(text):
    """
    解析表格数据，支持Markdown表格格式
    返回pandas DataFrame
    """
    try:
        # 尝试解析Markdown表格
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        if not lines:
            return None
        
        # 找到表头行（包含 | 的第一行，且不是分隔行）
        header_line = None
        data_start_idx = 0
        
        for i, line in enumerate(lines):
            if '|' in line and '---' not in line and not line.startswith('---'):
                if header_line is None:
                    header_line = line
                    data_start_idx = i + 1
                    break
        
        if header_line is None:
            return None
        
        # 解析表头
        headers = [h.strip() for h in header_line.split('|') if h.strip()]
        
        if not headers:
            return None
        
        # 解析数据行（跳过分隔行）
        data_rows = []
        for line in lines[data_start_idx:]:
            if '|' in line and '---' not in line and not line.startswith('---'):
                row_data = [cell.strip() for cell in line.split('|') if cell.strip()]
                # 允许数据行列数少于表头（补齐空值）
                while len(row_data) < len(headers):
                    row_data.append('')
                if len(row_data) >= len(headers):
                    data_rows.append(row_data[:len(headers)])
        
        if not data_rows:
            return None
        
        # 创建DataFrame
        df = pd.DataFrame(data_rows, columns=headers)
        return df
    except Exception as e:
        print(f"解析表格数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def create_chart_from_table(text, chart_type="line"):
    """
    从表格数据创建图表
    chart_type: "line", "bar", "both"
    返回base64编码的图片
    """
    try:
        df = parse_table_data(text)
        if df is None or df.empty:
            return None
        
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        # 获取第一列作为x轴（通常是时间或类别）
        x_col = df.columns[0]
        x_data = df[x_col].tolist()
        
        # 获取数值列
        numeric_cols = []
        for col in df.columns[1:]:
            try:
                # 尝试转换为数值（处理百分比等格式）
                values = []
                for val in df[col]:
                    val_str = str(val).replace('%', '').strip()
                    try:
                        values.append(float(val_str))
                    except:
                        values.append(0)
                df[col + '_numeric'] = values
                numeric_cols.append(col)
            except:
                continue
        
        if not numeric_cols:
            return None
        
        # 创建图表
        fig, axes = plt.subplots(len(numeric_cols), 1, figsize=(12, 6 * len(numeric_cols)))
        if len(numeric_cols) == 1:
            axes = [axes]
        
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            y_data = df[col + '_numeric'].tolist()
            
            if chart_type in ["line", "both"]:
                ax.plot(x_data, y_data, marker='o', linewidth=2, markersize=6, label=col)
            
            if chart_type in ["bar", "both"]:
                ax.bar(x_data, y_data, alpha=0.6, label=col)
            
            ax.set_xlabel(x_col, fontsize=10)
            ax.set_ylabel(col, fontsize=10)
            ax.set_title(f'{col} 趋势图', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # 转换为base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close()
        
        return f"data:image/png;base64,{img_base64}"
    except Exception as e:
        print(f"创建图表失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def process_ocr(image, image_url_or_path, task_type, max_new_tokens, custom_prompt):
    """处理OCR识别"""
    global paddleocr_vl_model, my_preprocessor
    
    if paddleocr_vl_model is None or my_preprocessor is None:
        return "❌ 请先初始化模型！", None, None
    
    # 确定使用哪个图片源
    image_source = None
    if image is not None:
        image_source = image
    elif image_url_or_path and image_url_or_path.strip():
        image_source = image_url_or_path.strip()
    
    if image_source is None:
        return "❌ 请上传图片、输入图片路径或URL！", None, None
    
    try:
        # 加载图片（支持PIL Image、本地路径或URL）
        loaded_image = load_image_from_source(image_source)
        if loaded_image is None:
            return "❌ 无法加载图片！", None, None
        
        # 准备提示词
        if custom_prompt and custom_prompt.strip():
            prompt_text = custom_prompt.strip()
        else:
            prompt_text = PROMPTS.get(task_type, "OCR:")
        
        # 转换图片为RGB
        image_rgb = loaded_image.convert("RGB")
        
        # 固定调整图片大小为1200x800（与用户代码保持一致）
        target_width = 1200
        target_height = 800
        image_rgb = image_rgb.resize((target_width, target_height), Image.Resampling.LANCZOS)
        
        # 准备消息
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_rgb},
                    {"type": "text", "text": prompt_text},
                ]
            }
        ]
        
        # 使用render_jinja_template处理文本
        text, generation_indices = render_jinja_template(
            conversations=[messages],
            chat_template=CHAT_TEMPLATE,
            add_generation_prompt=True,
            return_tensors="pt",
        )
        
        # 处理图像
        images_info = my_preprocessor(images=image_rgb, return_tensors="pt")
        
        # 处理图像占位符
        if not isinstance(text, list):
            text = [text]
        
        index = 0
        for i in range(len(text)):
            while "<|IMAGE_PLACEHOLDER|>" in text[i]:
                placeholder_count = (
                    images_info['image_grid_thw'][index].prod()
                    // 2
                    // 2
                )
                text[i] = text[i].replace(
                    "<|IMAGE_PLACEHOLDER|>",
                    "<|placeholder|>" * placeholder_count,
                    1,
                )
                index += 1
            text[i] = text[i].replace("<|placeholder|>", "<|IMAGE_PLACEHOLDER|>")
        
        # Tokenize文本
        text_inputs = paddleocr_vl_model.tokenizer(text, return_tensors="pt")
        
        # 准备生成配置
        generation_config = {
            "max_new_tokens": max_new_tokens,
            "do_sample": False,
        }
        
        # 执行OCR识别
        start_time = time.perf_counter()
        response, history = paddleocr_vl_model.chat(
            messages=messages,
            generation_config=generation_config
        )
        elapsed_time = time.perf_counter() - start_time
        
        # 格式化结果（处理特殊标记，包括LaTeX格式转换）
        formatted_response = format_ocr_result(response)
        
        # 检测是否包含LaTeX公式（在格式化后再次检测，因为format_ocr_result已经处理了格式转换）
        formatted_for_detect, has_latex = detect_and_format_latex(formatted_response)
        if has_latex:
            formatted_response = formatted_for_detect
        
        # 判断是否是表格格式（包含Markdown表格）
        # 检测包含多个|符号的行，可能是表格
        lines = [line.strip() for line in formatted_response.split('\n') if line.strip()]
        pipe_count = sum(line.count('|') for line in lines)
        has_separator = '---' in formatted_response
        
        # 如果包含多个|符号（至少5个），可能是表格
        is_table = (formatted_response.strip().startswith('|') and has_separator) or \
                   (pipe_count >= 5 and any('|' in line for line in lines[:3]))  # 前3行中有包含|的行
        
        # 格式化结果文本
        result_text = f"""📄 OCR识别结果:
{formatted_response}

⏱️ 执行时间: {elapsed_time:.3f} 秒 ({elapsed_time*1000:.2f} 毫秒)
"""
        
        # 准备Markdown可视化内容
        if is_table:
            # 表格可视化（只显示表格，不显示图表）
            markdown_content = f"""## 📊 表格可视化

{formatted_response}

---
*执行时间: {elapsed_time:.3f} 秒*
"""
        elif has_latex:
            # 包含LaTeX公式的情况，直接使用格式化后的结果
            # formatted_response 已经包含了正确的 $$...$$ 格式
            markdown_content = f"""## 📐 数学公式识别结果

{formatted_response}

---
*执行时间: {elapsed_time:.3f} 秒*

**提示**: LaTeX公式已自动格式化，如果公式未正确渲染，请检查公式格式是否正确。
"""
        else:
            # 非表格情况，直接显示原始文本（不进行Markdown格式化）
            markdown_content = f"""## 📄 识别结果

{response}

---
*执行时间: {elapsed_time:.3f} 秒*
"""
        
        # 返回：格式化文本、原始结果、Markdown可视化
        return result_text, response, markdown_content
        
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        return f"❌ 识别失败: {str(e)}\n\n详细信息:\n{error_detail}", None, None

# 创建Gradio界面
# 添加异常处理配置，避免响应内容长度错误
with gr.Blocks(
    title="PaddleOCR-VL OCR识别系统",
    analytics_enabled=False
) as demo:
    gr.HTML(
        """
        <style>
        #pid_display textarea,
        #pid_display input {
            color: #111 !important;
            font-weight: 600;
            background-color: #f8f9fb;
        }
        </style>
        """
    )
    gr.Markdown(
        """
        # 🚀 PaddleOCR-VL OCR识别系统
        
        基于OpenVINO的PaddleOCR-VL模型OCR识别系统
        
        ## 使用说明
        1. 首先在"模型设置"中初始化模型
        2. 上传要识别的图片
        3. 选择任务类型或输入自定义提示词
        4. 点击"开始识别"按钮
        """
    )
    
    with gr.Tab("模型设置"):
        with gr.Row():
            with gr.Column():
                ov_model_path_input = gr.Textbox(
                    label="OpenVINO模型路径",
                    value="./ov_paddleocr_vl_model",
                    placeholder="输入OpenVINO模型路径"
                )
                device_type = gr.Dropdown(
                    label="设备类型",
                    choices=["CPU", "GPU"],
                    value="GPU"
                )
                template_path_input = gr.Textbox(
                    label="Chat模板文件路径（可选）",
                    value="",
                    placeholder="留空使用默认模板，或输入模板文件路径"
                )
                llm_int4 = gr.Checkbox(label="LLM INT4压缩", value=False, interactive=False)
                vision_int8 = gr.Checkbox(label="Vision INT8量化", value=False, interactive=False)
                llm_int8 = gr.Checkbox(label="LLM INT8量化", value=False, interactive=False)
                with gr.Row():
                    init_btn = gr.Button("初始化模型", variant="primary")
                    check_status_btn = gr.Button("检查状态", variant="secondary")
                    unload_btn = gr.Button("卸载模型", variant="stop")
            with gr.Column():
                init_status = gr.Textbox(
                    label="初始化状态",
                    value="等待初始化...",
                    interactive=False,
                    lines=5,
                    max_lines=5  # 限制最大行数，避免内容累积
                )
                process_id_display = gr.Textbox(
                    label="进程ID",
                    value="未启动",
                    interactive=False,
                    lines=1,
                    max_lines=1,  # 限制最大行数
                    elem_id="pid_display"
                )
    
    with gr.Tab("OCR识别"):
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(
                    label="上传图片（方式1：直接上传）",
                    type="pil",
                    sources=["upload", "clipboard"],
                )
                image_url_or_path = gr.Textbox(
                    label="图片路径或URL（方式2：输入本地路径或网络URL）",
                    placeholder="例如: ./image.jpg 或 https://example.com/image.png",
                    value="",
                    lines=1
                )
                gr.Markdown("**提示**: 可以使用方式1上传图片，或使用方式2输入本地文件路径或网络图片URL")
                gr.Markdown("**注意**: 图片会自动调整为1200x800尺寸")
                task_type = gr.Dropdown(
                    label="任务类型",
                    choices=["ocr", "table", "formula", "chart"],
                    value="ocr"
                )
                custom_prompt = gr.Textbox(
                    label="自定义提示词（可选）",
                    placeholder="留空则使用默认提示词，例如: OCR: 或 Table Recognition:",
                    lines=2
                )
                max_tokens = gr.Slider(
                    label="最大生成token数",
                    minimum=128,
                    maximum=2048,
                    value=1024,
                    step=128
                )
                recognize_btn = gr.Button("开始识别", variant="primary", size="lg")
            
            with gr.Column():
                markdown_output = gr.Markdown(
                    label="Markdown可视化（表格渲染）",
                    value="等待识别结果...",
                )
                result_output = gr.Textbox(
                    label="识别结果（格式化后文本）",
                    lines=15,
                    interactive=False
                )
                raw_result = gr.Textbox(
                    label="原始结果（未格式化）",
                    lines=8,
                    interactive=True
                )
                gr.Markdown("**提示**: 格式化结果会自动将表格标记转换为Markdown表格格式，并在上方可视化显示。系统会自动识别表格和LaTeX公式并渲染。")
    
    with gr.Tab("使用说明"):
        gr.Markdown(
            """
            ## 📖 使用说明
            
            ### 1. 模型初始化
            - **OpenVINO模型路径**: 转换后的OpenVINO模型路径
            - **设备类型**: 选择CPU或GPU（推荐GPU）
            - **Chat模板文件**: 可选，留空使用默认模板
            - **量化选项**: 根据需要选择是否启用量化以提升性能
            - **自动状态更新**: 系统每2秒自动检查并更新初始化状态，无需手动点击"检查状态"按钮
            - **进程管理**: 
              - 点击"初始化模型"启动独立进程进行模型验证
              - 点击"检查状态"手动查看当前状态
              - 点击"卸载模型"终止进程并清理所有资源
            
            ### 2. OCR识别
            - **上传图片（方式1）**: 支持上传或粘贴图片
            - **图片路径或URL（方式2）**: 
              - 输入本地文件路径，例如: `./image.jpg` 或 `C:/images/test.png`
              - 输入网络图片URL，例如: `https://example.com/image.png`
              - 注意：如果使用方式1上传了图片，方式2会被忽略
            - **图片尺寸**: 图片会自动调整为1200x800尺寸
            - **任务类型**: 
              - `ocr`: 普通文字识别
              - `table`: 表格识别
              - `formula`: 公式识别（支持LaTeX格式）
              - `chart`: 图表识别
            - **自定义提示词**: 可以输入自定义的提示词
            - **最大token数**: 控制生成文本的最大长度
            
            ### 3. 结果查看
            - **识别结果**: 显示完整的识别结果和执行时间
            - **原始结果**: 仅显示识别出的文本内容，可以复制
            - **Markdown可视化**: 自动识别表格和LaTeX公式并渲染
            
            ### 4. 自动可视化功能
            - **LaTeX公式**: 系统会自动检测识别结果中的数学公式并渲染
              - 支持块级公式（`$$...$$`）和行内公式（`$...$`）
              - 自动转换 `\\[...\\]` 格式为 `$$...$$` 格式
            - **表格**: 系统会自动识别表格数据并格式化为Markdown表格显示
              - 自动识别表格格式并转换为Markdown表格
              - 支持年份表格等特殊格式的自动识别和格式化
            
            ## ⚠️ 注意事项
            - 首次使用需要先初始化模型
            - 模型初始化可能需要一些时间
            - 识别时间取决于图片大小和模型配置
            - 本版本使用render_jinja_template和PaddleOCRVLImageProcessor
            - LaTeX公式识别需要模型输出包含正确的数学表达式格式
            """
        )
    
    # 绑定事件
    init_btn.click(
        fn=initialize_model,
        inputs=[ov_model_path_input, device_type, llm_int4, vision_int8, llm_int8, template_path_input],
        outputs=[init_status, process_id_display]
    )
    
    check_status_btn.click(
        fn=check_init_status,
        inputs=[],
        outputs=[init_status, process_id_display]
    )
    
    # 创建卸载模型的包装函数，同时清空所有输出组件
    def unload_model_and_clear_all():
        """卸载模型并清空所有Gradio输出组件和资源"""
        # 卸载模型
        status_text, pid_text = unload_model()
        
        # 清空OCR识别结果
        # 返回所有需要清空的值
        return (
            status_text,  # init_status
            pid_text,     # process_id_display
            "等待识别结果...",  # markdown_output
            "",           # result_output
            ""            # raw_result
        )
    
    unload_btn.click(
        fn=unload_model_and_clear_all,
        inputs=[],
        outputs=[init_status, process_id_display, markdown_output, result_output, raw_result]
    )
    
    recognize_btn.click(
        fn=process_ocr,
        inputs=[image_input, image_url_or_path, task_type, max_tokens, custom_prompt],
        outputs=[result_output, raw_result, markdown_output]
    )
    
    # 添加定时器实现自动轮询状态更新
    # 优化：只在有进程运行时才更新，减少不必要的内存占用
    _last_gc_time = time.time()  # 记录上次垃圾回收时间
    
    def safe_check_status():
        """安全的状态检查包装函数，添加异常处理和内存优化"""
        global init_process, _last_gc_time
        try:
            # 如果没有进程，减少更新频率（每10秒更新一次）
            if init_process is None:
                # 检查缓存时间，避免频繁更新
                cache_age = time.time() - _last_status_cache.get("timestamp", 0)
                if cache_age < 10:
                    # 返回缓存值，不创建新对象
                    return _last_status_cache.get("text", "等待初始化..."), _last_status_cache.get("pid", "未启动")
            
            result = check_init_status()
            
            # 定期清理内存（每30秒清理一次）
            current_time = time.time()
            if current_time - _last_gc_time > 30:
                import gc
                gc.collect()
                _last_gc_time = current_time
            
            return result
        except Exception as e:
            # 如果检查失败，返回安全的错误消息
            error_msg = f"状态检查出错: {str(e)}"
            return safe_encode_text(error_msg), "错误"
    
    # 使用较长的间隔（5秒），减少频率，降低内存占用
    status_timer = gr.Timer(value=5.0, active=True)
    status_timer.tick(
        fn=safe_check_status,
        inputs=[],
        outputs=[init_status, process_id_display]
    )

if __name__ == "__main__":
    import os
    import socket
    
    # 彻底禁用Gradio的网络检查，避免连接超时和403错误
    os.environ["GRADIO_SERVER_NAME"] = "127.0.0.1"
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"
    os.environ["GRADIO_SERVER_PROXY"] = ""
    os.environ["NO_PROXY"] = "127.0.0.1,localhost"
    os.environ["no_proxy"] = "127.0.0.1,localhost"
    # 禁用启动事件检查
    os.environ["GRADIO_SKIP_STARTUP_EVENTS"] = "1"
    # 增加响应大小限制，避免Content-Length错误
    os.environ["GRADIO_MAX_CONTENT_LENGTH"] = "1048576000"  # 100MB
    
    def find_free_port(start_port=7860, max_attempts=10):
        """查找可用端口"""
        for i in range(max_attempts):
            port = start_port + i
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(('127.0.0.1', port))
                    return port
            except OSError:
                continue
        return None
    
    try:
        print("=" * 60)
        print("正在启动PaddleOCR-VL OCR识别系统...")
        print("=" * 60)
        
        # 查找可用端口
        port = find_free_port(7860)
        if port is None:
            print("❌ 无法找到可用端口，请手动指定端口")
            port = 7860
        
        print(f"访问地址: http://127.0.0.1:{port}")
        print("=" * 60)
        
        # 尝试启动，如果失败则尝试其他端口
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                demo.launch(
                    server_name="127.0.0.1",  # 只监听本地
                    server_port=port,          # 端口号
                    share=False,               # 不创建公共链接
                    inbrowser=False,           # 不自动打开浏览器（避免启动事件问题）
                    show_error=True,           # 显示错误信息
                    quiet=False,               # 显示启动信息
                    favicon_path=None,         # 不使用favicon
                    prevent_thread_lock=False,   # 允许在后台运行
                    # 添加这些参数来避免启动事件检查和响应问题
                    max_threads=1,             # 限制线程数
                    # 修复响应内容长度问题
                    max_file_size=None,        # 不限制文件大小（或设置一个较大的值）
                    allowed_paths=None,        # 允许所有路径
                )
                break  # 成功启动
            except Exception as e:
                if attempt < max_attempts - 1:
                    port = find_free_port(port + 1)
                    if port:
                        print(f"尝试端口 {port}...")
                        continue
                raise
        
    except Exception as e:
        print(f"\n❌ 启动失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查端口是否被占用:")
        print("   Windows: netstat -ano | findstr :7860")
        print("   Linux/Mac: lsof -i :7860")
        print("2. 尝试手动指定端口:")
        print("   demo.launch(server_port=7861)")
        print("3. 检查防火墙/代理设置:")
        print("   - 确保没有代理阻止localhost访问")
        print("   - 临时关闭防火墙测试")
        print("4. 设置环境变量后重试:")
        print("   set GRADIO_ANALYTICS_ENABLED=False")
        print("   set NO_PROXY=127.0.0.1,localhost")
        print("5. 如果问题持续，尝试更新Gradio:")
        print("   pip install --upgrade gradio")
        raise

