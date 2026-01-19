from __future__ import annotations

from typing import Dict, Literal, Mapping, Optional


Lang = Literal["zh_CN", "en_US", "zh_TW"]


def normalize_lang(lang: Optional[str]) -> Lang:
    """
    Normalize user input / persisted value into a supported language code.
    """
    if not lang:
        return "zh_CN"
    v = str(lang).strip().replace("-", "_").lower()
    if v in ("zh_cn", "zh_hans", "cn", "zh"):
        return "zh_CN"
    if v in ("en_us", "en", "us"):
        return "en_US"
    if v in ("zh_tw", "zh_hant", "tw", "hk"):
        return "zh_TW"
    return "zh_CN"


# Display names shown in the language combobox (keep stable and explicit).
LANG_DISPLAY: Mapping[Lang, str] = {
    "zh_CN": "简体中文",
    "en_US": "English",
    "zh_TW": "繁體中文",
}


_T: Dict[str, Dict[Lang, str]] = {
    # App / navigation
    "app.title": {
        "zh_CN": "PaddleOCR-VL OpenVINO Client",
        "en_US": "PaddleOCR-VL OpenVINO Client",
        "zh_TW": "PaddleOCR-VL OpenVINO Client",
    },
    "nav.ocr": {"zh_CN": "OCR / 解析", "en_US": "OCR / Parsing", "zh_TW": "OCR / 解析"},
    "nav.history": {"zh_CN": "历史任务", "en_US": "History", "zh_TW": "歷史任務"},
    "nav.settings": {"zh_CN": "设置", "en_US": "Settings", "zh_TW": "設定"},
    "nav.about": {"zh_CN": "关于", "en_US": "About", "zh_TW": "關於"},

    # Toolbar actions
    "action.add_files": {"zh_CN": "添加文件", "en_US": "Add Files", "zh_TW": "新增檔案"},
    "action.clear": {"zh_CN": "清空", "en_US": "Clear", "zh_TW": "清空"},
    "action.screenshot": {"zh_CN": "截图", "en_US": "Screenshot", "zh_TW": "截圖"},
    "action.output_dir": {"zh_CN": "输出目录", "en_US": "Output Dir", "zh_TW": "輸出目錄"},
    "action.about": {"zh_CN": "关于", "en_US": "About", "zh_TW": "關於"},

    # OCR page
    "drop.title": {"zh_CN": "拖拽文件到这里", "en_US": "Drag files here", "zh_TW": "拖曳檔案到這裡"},
    "drop.hint": {
        "zh_CN": "支持：图片（png/jpg/jpeg/webp/bmp），可选 PDF",
        "en_US": "Supported: images (png/jpg/jpeg/webp/bmp), optional PDF",
        "zh_TW": "支援：圖片（png/jpg/jpeg/webp/bmp），可選 PDF",
    },
    "card.task_queue": {"zh_CN": "任务队列", "en_US": "Task Queue", "zh_TW": "任務佇列"},
    "card.logs": {"zh_CN": "日志", "en_US": "Logs", "zh_TW": "日誌"},
    "log.placeholder": {"zh_CN": "日志输出 ...", "en_US": "Logs ...", "zh_TW": "日誌輸出 ..."},
    "progress.label": {"zh_CN": "进度：", "en_US": "Progress:", "zh_TW": "進度："},

    "btn.start": {"zh_CN": "开始", "en_US": "Start", "zh_TW": "開始"},
    "btn.stop": {"zh_CN": "停止", "en_US": "Stop", "zh_TW": "停止"},
    "btn.rerun_selected": {"zh_CN": "重跑所选", "en_US": "Rerun Selected", "zh_TW": "重跑所選"},
    "btn.delete_selected": {"zh_CN": "删除所选", "en_US": "Delete Selected", "zh_TW": "刪除所選"},
    "btn.screenshot": {"zh_CN": "截图", "en_US": "Screenshot", "zh_TW": "截圖"},
    "btn.choose": {"zh_CN": "选择...", "en_US": "Browse...", "zh_TW": "選擇..."},

    "table.file": {"zh_CN": "文件", "en_US": "File", "zh_TW": "檔案"},
    "table.type": {"zh_CN": "类型", "en_US": "Type", "zh_TW": "類型"},
    "table.task_type": {"zh_CN": "任务类型", "en_US": "Task Type", "zh_TW": "任務類型"},
    "table.status": {"zh_CN": "状态", "en_US": "Status", "zh_TW": "狀態"},
    "table.output_dir": {"zh_CN": "输出目录", "en_US": "Output Dir", "zh_TW": "輸出目錄"},

    # Settings page
    "settings.title": {"zh_CN": "设置", "en_US": "Settings", "zh_TW": "設定"},
    "settings.language": {"zh_CN": "界面语言", "en_US": "UI Language", "zh_TW": "介面語言"},
    "ph.layout_model": {"zh_CN": "布局模型路径（留空=自动下载）", "en_US": "Layout model path (empty = auto-download)", "zh_TW": "版面模型路徑（留空=自動下載）"},
    "ph.vlm_model": {"zh_CN": "VLM 模型路径（留空=自动下载）", "en_US": "VLM model path (empty = auto-download)", "zh_TW": "VLM 模型路徑（留空=自動下載）"},
    "ph.cache_dir": {"zh_CN": "ModelScope cache_dir（可选）", "en_US": "ModelScope cache_dir (optional)", "zh_TW": "ModelScope cache_dir（可選）"},
    "settings.use_layout": {"zh_CN": "启用布局检测", "en_US": "Enable layout detection", "zh_TW": "啟用版面偵測"},
    "settings.llm_int4": {"zh_CN": "LLM INT4 压缩", "en_US": "LLM INT4 compression", "zh_TW": "LLM INT4 壓縮"},
    "settings.vision_int8": {"zh_CN": "Vision INT8 量化", "en_US": "Vision INT8 quantization", "zh_TW": "Vision INT8 量化"},
    "settings.llm_int8_compress": {"zh_CN": "LLM INT8 压缩", "en_US": "LLM INT8 compression", "zh_TW": "LLM INT8 壓縮"},
    "settings.llm_int8_quant": {"zh_CN": "LLM INT8 量化", "en_US": "LLM INT8 quantization", "zh_TW": "LLM INT8 量化"},
    "settings.output_dir": {"zh_CN": "输出目录", "en_US": "Output directory", "zh_TW": "輸出目錄"},
    "settings.note": {
        "zh_CN": "提示：若 layout_model_path 指向具体 .xml 文件，则 layout_precision 会被忽略。",
        "en_US": "Note: If layout_model_path points to a specific .xml file, layout_precision will be ignored.",
        "zh_TW": "提示：若 layout_model_path 指向具體 .xml 檔案，則 layout_precision 會被忽略。",
    },

    # History page
    "history.title": {"zh_CN": "历史任务（仅成功）", "en_US": "History (Success Only)", "zh_TW": "歷史任務（僅成功）"},
    "history.open_out": {"zh_CN": "打开输出目录", "en_US": "Open Output Folder", "zh_TW": "開啟輸出目錄"},
    "history.delete_selected": {"zh_CN": "删除所选", "en_US": "Delete Selected", "zh_TW": "刪除所選"},
    "history.clear": {"zh_CN": "清空历史", "en_US": "Clear History", "zh_TW": "清空歷史"},
    "history.col.time": {"zh_CN": "时间", "en_US": "Time", "zh_TW": "時間"},
    "history.col.file": {"zh_CN": "文件", "en_US": "File", "zh_TW": "檔案"},
    "history.col.type": {"zh_CN": "类型", "en_US": "Type", "zh_TW": "類型"},
    "history.col.task_type": {"zh_CN": "任务类型", "en_US": "Task Type", "zh_TW": "任務類型"},
    "history.col.output_dir": {"zh_CN": "输出目录", "en_US": "Output Dir", "zh_TW": "輸出目錄"},
    "history.col.summary": {"zh_CN": "摘要", "en_US": "Summary", "zh_TW": "摘要"},
    "history.detail.placeholder": {"zh_CN": "请选择一条历史任务查看详情…", "en_US": "Select a history task to view details…", "zh_TW": "請選擇一條歷史任務查看詳情…"},
    "history.input_preview": {"zh_CN": "原图预览", "en_US": "Input Preview", "zh_TW": "原圖預覽"},
    "history.output_preview": {"zh_CN": "输出图预览", "en_US": "Output Preview", "zh_TW": "輸出圖預覽"},
    "history.tab.compare": {"zh_CN": "对比（原图 vs Markdown）", "en_US": "Compare (Input vs Markdown)", "zh_TW": "對比（原圖 vs Markdown）"},
    "history.tab.output": {"zh_CN": "输出图", "en_US": "Output Image", "zh_TW": "輸出圖"},

    # About page
    "about.title": {"zh_CN": "关于", "en_US": "About", "zh_TW": "關於"},
    "about.text": {
        "zh_CN": "PaddleOCR-VL OpenVINO 桌面客户端\n\n- 支持拖拽导入 / 批量任务 / 结果预览\n- 导出 result.md / result.json / vis.png\n",
        "en_US": "PaddleOCR-VL OpenVINO Desktop Client\n\n- Drag & drop / batch tasks / preview\n- Export result.md / result.json / vis.png\n",
        "zh_TW": "PaddleOCR-VL OpenVINO 桌面客戶端\n\n- 支援拖曳匯入 / 批次任務 / 結果預覽\n- 匯出 result.md / result.json / vis.png\n",
    },

    # Common dialogs
    "dlg.tip": {"zh_CN": "提示", "en_US": "Tip", "zh_TW": "提示"},
    "dlg.confirm": {"zh_CN": "确认", "en_US": "Confirm", "zh_TW": "確認"},
    "dlg.confirm_delete": {"zh_CN": "确认删除", "en_US": "Confirm Delete", "zh_TW": "確認刪除"},
    "dlg.choose_files": {"zh_CN": "选择文件", "en_US": "Select Files", "zh_TW": "選擇檔案"},
    "dlg.choose_output_dir": {"zh_CN": "选择输出目录", "en_US": "Select Output Directory", "zh_TW": "選擇輸出目錄"},
    "msg.output_dir_missing": {"zh_CN": "输出目录不存在", "en_US": "Output directory does not exist.", "zh_TW": "輸出目錄不存在"},
    "msg.running_stop_first": {"zh_CN": "正在运行中，请先停止。", "en_US": "Running. Please stop first.", "zh_TW": "正在執行，請先停止。"},
    "msg.running_no_delete": {"zh_CN": "正在运行中，无法删除任务。请先停止。", "en_US": "Running. Cannot delete tasks. Please stop first.", "zh_TW": "正在執行，無法刪除任務。請先停止。"},
    "msg.running_no_add": {"zh_CN": "正在运行中，暂不支持添加文件。", "en_US": "Running. Adding files is disabled.", "zh_TW": "正在執行，暫不支援新增檔案。"},
    "msg.running_no_screenshot": {"zh_CN": "正在运行中，无法截图。", "en_US": "Running. Screenshot is disabled.", "zh_TW": "正在執行，無法截圖。"},
    "msg.running": {"zh_CN": "正在运行中。", "en_US": "Running.", "zh_TW": "正在執行。"},
    "msg.select_rows_delete": {"zh_CN": "请先在任务队列表格中选择要删除的任务行。", "en_US": "Select rows in the task table to delete.", "zh_TW": "請先在任務表格中選擇要刪除的任務。"},
    "msg.select_rows_rerun": {"zh_CN": "请先在表格中选择要重跑的任务行。", "en_US": "Select rows in the table to rerun.", "zh_TW": "請先在表格中選擇要重跑的任務。"},
    "msg.no_valid_selected": {"zh_CN": "未选中有效任务。", "en_US": "No valid tasks selected.", "zh_TW": "未選中有效任務。"},
    "msg.select_history_first": {"zh_CN": "请先选择一条历史任务。", "en_US": "Please select a history task first.", "zh_TW": "請先選擇一條歷史任務。"},
    "msg.select_history_delete_first": {"zh_CN": "请先选择要删除的历史任务。", "en_US": "Select history tasks to delete.", "zh_TW": "請先選擇要刪除的歷史任務。"},
    "msg.confirm_clear_history": {"zh_CN": "确定要清空历史记录吗？", "en_US": "Are you sure you want to clear history?", "zh_TW": "確定要清空歷史記錄嗎？"},
    "msg.no_files": {"zh_CN": "请先添加文件。", "en_US": "Please add files first.", "zh_TW": "請先新增檔案。"},
    "msg.no_pending": {"zh_CN": "没有待执行（pending）的任务。", "en_US": "No pending tasks.", "zh_TW": "沒有待執行（pending）的任務。"},
    "msg.confirm_delete_n": {
        "zh_CN": "确定要删除选中的 {n} 个任务吗？",
        "en_US": "Are you sure you want to delete {n} selected tasks?",
        "zh_TW": "確定要刪除選中的 {n} 個任務嗎？",
    },
    # Status labels (display only; internal status stays pending/running/done/error)
    "status.pending": {"zh_CN": "待处理", "en_US": "Pending", "zh_TW": "待處理"},
    "status.running": {"zh_CN": "运行中", "en_US": "Running", "zh_TW": "執行中"},
    "status.done": {"zh_CN": "已完成", "en_US": "Done", "zh_TW": "已完成"},
    "status.error": {"zh_CN": "失败", "en_US": "Error", "zh_TW": "失敗"},

    # MainWindow logs (visible in log box)
    "ui.log.screenshot_done": {
        "zh_CN": "截图完成：{path}",
        "en_US": "Screenshot saved: {path}",
        "zh_TW": "截圖完成：{path}",
    },
    "ui.log.screenshot_canceled": {
        "zh_CN": "截图已取消（ESC 或选区过小）",
        "en_US": "Screenshot canceled (ESC or selection too small)",
        "zh_TW": "截圖已取消（ESC 或選區過小）",
    },
    "ui.log.output_dir": {"zh_CN": "输出目录：{dir}", "en_US": "Output directory: {dir}", "zh_TW": "輸出目錄：{dir}"},
    "ui.log.start_infer": {"zh_CN": "开始推理 ...", "en_US": "Starting inference...", "zh_TW": "開始推理..."},
    "ui.log.stop_req": {"zh_CN": "请求停止 ...", "en_US": "Stop requested...", "zh_TW": "請求停止..."},
    "ui.log.rerun_n": {"zh_CN": "重跑所选任务：{n} 个", "en_US": "Rerunning selected tasks: {n}", "zh_TW": "重跑所選任務：{n} 個"},
    "ui.log.all_done": {"zh_CN": "全部任务结束。", "en_US": "All tasks finished.", "zh_TW": "全部任務結束。"},

    # History detail labels
    "history.detail.time": {"zh_CN": "时间", "en_US": "Time", "zh_TW": "時間"},
    "history.detail.file": {"zh_CN": "文件", "en_US": "File", "zh_TW": "檔案"},
    "history.detail.type": {"zh_CN": "类型", "en_US": "Type", "zh_TW": "類型"},
    "history.detail.task_type": {"zh_CN": "任务类型", "en_US": "Task Type", "zh_TW": "任務類型"},
    "history.detail.output_dir": {"zh_CN": "输出目录", "en_US": "Output Dir", "zh_TW": "輸出目錄"},
    "history.detail.summary": {"zh_CN": "摘要", "en_US": "Summary", "zh_TW": "摘要"},

    # History delete confirmations
    "history.confirm_delete_one": {
        "zh_CN": "确定要删除该历史任务吗？\n\n{input_path}\n\n同时会删除输出目录：\n{output_dir}",
        "en_US": "Delete this history task?\n\n{input_path}\n\nIts output directory will also be deleted:\n{output_dir}",
        "zh_TW": "確定要刪除該歷史任務嗎？\n\n{input_path}\n\n同時會刪除輸出目錄：\n{output_dir}",
    },
    "history.confirm_delete_many": {
        "zh_CN": "确定要删除选中的 {n} 条历史任务吗？\n\n注意：会同时删除每条任务的输出目录。",
        "en_US": "Delete {n} selected history tasks?\n\nNote: the output directory of each task will also be deleted.",
        "zh_TW": "確定要刪除選中的 {n} 條歷史任務嗎？\n\n注意：會同時刪除每條任務的輸出目錄。",
    },
    # Worker logs / errors
    "wk.reuse_pipeline": {
        "zh_CN": "✅ 复用已加载的 Pipeline（模型不再重复初始化）",
        "en_US": "✅ Reusing the loaded Pipeline (no re-initialization)",
        "zh_TW": "✅ 重用已載入的 Pipeline（不再重複初始化）",
    },
    "wk.pre_release_old": {
        "zh_CN": "♻️  配置发生变化：预先释放旧 Pipeline...",
        "en_US": "♻️  Config changed: pre-releasing the previous Pipeline...",
        "zh_TW": "♻️  設定變更：預先釋放舊 Pipeline...",
    },
    "wk.init_pipeline": {
        "zh_CN": "初始化 Pipeline ...（首次/配置变化时可能会下载模型）",
        "en_US": "Initializing Pipeline... (may download models on first run/config change)",
        "zh_TW": "初始化 Pipeline...（首次/設定變更時可能會下載模型）",
    },
    "wk.pipeline_ready": {
        "zh_CN": "✅ Pipeline 初始化完成",
        "en_US": "✅ Pipeline initialized",
        "zh_TW": "✅ Pipeline 初始化完成",
    },
    "wk.pipeline_ready_other": {
        "zh_CN": "✅ Pipeline 已在其他位置初始化，本次直接复用",
        "en_US": "✅ Pipeline was initialized elsewhere; reusing it",
        "zh_TW": "✅ Pipeline 已在其他位置初始化，本次直接重用",
    },
    "wk.local_layout_path": {
        "zh_CN": "📌 使用本地 layout 模型：{path}",
        "en_US": "📌 Using local layout model: {path}",
        "zh_TW": "📌 使用本地 layout 模型：{path}",
    },
    "wk.local_vlm_path": {
        "zh_CN": "📌 使用本地 VLM 模型：{path}",
        "en_US": "📌 Using local VLM model: {path}",
        "zh_TW": "📌 使用本地 VLM 模型：{path}",
    },
    "wk.no_pending": {
        "zh_CN": "没有待执行（pending）的任务，直接结束。",
        "en_US": "No pending tasks. Exiting.",
        "zh_TW": "沒有待執行（pending）的任務，直接結束。",
    },
    "wk.init_failed_prefix": {
        "zh_CN": "❌ Pipeline 初始化失败：",
        "en_US": "❌ Pipeline initialization failed:",
        "zh_TW": "❌ Pipeline 初始化失敗：",
    },

    # Init error classification / hints
    "wk.init_err.modelscope_timeout": {
        "zh_CN": "ModelScope 下载失败：连接 www.modelscope.cn 超时/不可达。",
        "en_US": "ModelScope download failed: connection to www.modelscope.cn timed out/unreachable.",
        "zh_TW": "ModelScope 下載失敗：連線 www.modelscope.cn 超時/不可達。",
    },
    "wk.init_err.local_layout_missing": {
        "zh_CN": "已设置本地 layout_model_path，但路径不存在：{path}",
        "en_US": "Local layout_model_path is set, but the path does not exist: {path}",
        "zh_TW": "已設定本地 layout_model_path，但路徑不存在：{path}",
    },
    "wk.init_err.local_layout_not_xml": {
        "zh_CN": "已设置本地 layout_model_path，但不是 .xml 文件：{path}",
        "en_US": "Local layout_model_path is set, but it is not a .xml file: {path}",
        "zh_TW": "已設定本地 layout_model_path，但不是 .xml 檔案：{path}",
    },
    "wk.init_err.local_layout_no_xml": {
        "zh_CN": "已设置本地 layout_model_path 为目录，但目录中未找到任何 .xml：{dir}",
        "en_US": "Local layout_model_path is a directory, but no .xml files were found: {dir}",
        "zh_TW": "已設定本地 layout_model_path 為目錄，但目錄中未找到任何 .xml：{dir}",
    },
    "wk.init_err.local_layout_no_bin": {
        "zh_CN": "已设置本地 layout 模型，但缺少对应 .bin：xml={xml}  bin={bin}",
        "en_US": "Local layout model is set, but the corresponding .bin is missing: xml={xml}  bin={bin}",
        "zh_TW": "已設定本地 layout 模型，但缺少對應 .bin：xml={xml}  bin={bin}",
    },
    "wk.init_err.local_vlm_missing": {
        "zh_CN": "已设置本地 vlm_model_path，但路径不存在：{path}",
        "en_US": "Local vlm_model_path is set, but the path does not exist: {path}",
        "zh_TW": "已設定本地 vlm_model_path，但路徑不存在：{path}",
    },
    "wk.init_err.local_vlm_not_dir": {
        "zh_CN": "已设置本地 vlm_model_path，但它不是目录：{path}",
        "en_US": "Local vlm_model_path is set, but it is not a directory: {path}",
        "zh_TW": "已設定本地 vlm_model_path，但它不是目錄：{path}",
    },
    "wk.init_err.local_vlm_missing_files": {
        "zh_CN": "已设置本地 VLM 模型目录，但缺少必要文件：{files}\n目录：{dir}",
        "en_US": "Local VLM model directory is missing required files: {files}\nDir: {dir}",
        "zh_TW": "已設定本地 VLM 模型目錄，但缺少必要檔案：{files}\n目錄：{dir}",
    },
    "wk.init_hint.net_check": {
        "zh_CN": "请检查网络是否能访问 modelscope.cn（公司网络/防火墙/需要 VPN 等）。",
        "en_US": "Check network access to modelscope.cn (firewall/corporate network/VPN).",
        "zh_TW": "請檢查網路是否能存取 modelscope.cn（公司網路/防火牆/需要 VPN 等）。",
    },
    "wk.init_hint.proxy_check": {
        "zh_CN": "如需代理，请在系统或环境变量中配置 HTTP(S) 代理后重试。",
        "en_US": "If a proxy is required, configure HTTP(S) proxy in system/env vars and retry.",
        "zh_TW": "如需代理，請在系統或環境變數中設定 HTTP(S) 代理後重試。",
    },
    "wk.init_hint.offline_local_paths": {
        "zh_CN": "离线/不稳定网络：在【设置】里填写本地 layout_model_path（.xml）与 vlm_model_path（目录），避免自动下载。",
        "en_US": "Offline/unstable network: set local layout_model_path (.xml) and vlm_model_path (folder) in Settings to skip auto-download.",
        "zh_TW": "離線/不穩定網路：在【設定】填入本地 layout_model_path（.xml）與 vlm_model_path（資料夾），避免自動下載。",
    },
    "wk.init_hint.pre_download_copy_cache": {
        "zh_CN": "也可在有网环境预下载后拷贝缓存目录（常见：%USERPROFILE%\\.cache\\modelscope\\hub\\models\\...）到目标机器。",
        "en_US": "You can also pre-download on a machine with internet and copy the cache folder (often: %USERPROFILE%\\.cache\\modelscope\\hub\\models\\...).",
        "zh_TW": "也可在有網環境預下載後複製快取目錄（常見：%USERPROFILE%\\.cache\\modelscope\\hub\\models\\...）到目標機器。",
    },
    "wk.stopped": {"zh_CN": "已停止", "en_US": "Stopped.", "zh_TW": "已停止"},
    "wk.processing": {
        "zh_CN": "开始处理：{path}",
        "en_US": "Processing: {path}",
        "zh_TW": "開始處理：{path}",
    },
    "wk.done": {
        "zh_CN": "✅ 完成：{path} -> {out_dir}",
        "en_US": "✅ Done: {path} -> {out_dir}",
        "zh_TW": "✅ 完成：{path} -> {out_dir}",
    },
    "wk.failed": {
        "zh_CN": "❌ 失败：{path}",
        "en_US": "❌ Failed: {path}",
        "zh_TW": "❌ 失敗：{path}",
    },
    "wk.fallback_save": {
        "zh_CN": "⚠️ 使用 save_to_* 失败，回退到直接写盘: {err}",
        "en_US": "⚠️ save_to_* failed; falling back to direct file write: {err}",
        "zh_TW": "⚠️ 使用 save_to_* 失敗，回退到直接寫入檔案: {err}",
    },
    "wk.err.no_content": {
        "zh_CN": "未检测到任何内容",
        "en_US": "No content detected.",
        "zh_TW": "未偵測到任何內容",
    },
    "wk.err.user_stop": {"zh_CN": "用户停止", "en_US": "Stopped by user.", "zh_TW": "使用者停止"},
    "wk.pdf_page": {
        "zh_CN": "  - PDF Page {i}/{n}: {name}",
        "en_US": "  - PDF Page {i}/{n}: {name}",
        "zh_TW": "  - PDF Page {i}/{n}: {name}",
    },
    "wk.err.init_failed_task": {
        "zh_CN": "Pipeline 初始化失败: {err}",
        "en_US": "Pipeline initialization failed: {err}",
        "zh_TW": "Pipeline 初始化失敗: {err}",
    },
    # PDF
    "pdf.err.pymupdf_missing": {
        "zh_CN": "未安装 PyMuPDF，无法处理 PDF。请安装：pip install PyMuPDF",
        "en_US": "PyMuPDF is not installed; cannot process PDF. Install it with: pip install PyMuPDF",
        "zh_TW": "未安裝 PyMuPDF，無法處理 PDF。請安裝：pip install PyMuPDF",
    },
    # Markdown preview (WebEngine)
    "md.err.markdownit_missing": {
        "zh_CN": "Markdown 渲染依赖未加载（markdown-it）。请检查 assets 是否存在或网络策略是否拦截了资源加载。",
        "en_US": "Markdown rendering dependency is not loaded (markdown-it). Check whether assets exist or network policy blocks resource loading.",
        "zh_TW": "Markdown 渲染相依未載入（markdown-it）。請檢查 assets 是否存在或網路策略是否攔截資源載入。",
    },
    "md.hint.perf_mode": {
        "zh_CN": "性能模式已启用：为提升滚动流畅度，已关闭代码高亮/公式渲染/mermaid 渲染。",
        "en_US": "Performance mode is enabled: code highlighting / math rendering / Mermaid rendering are disabled to improve scrolling.",
        "zh_TW": "效能模式已啟用：為提升捲動流暢度，已關閉程式碼高亮/公式渲染/Mermaid 渲染。",
    },
}


def t(key: str, lang: Lang, **kwargs) -> str:
    """
    Translate by key for the given language.
    Falls back to zh_CN and then to the key itself.
    """
    lang = normalize_lang(lang)
    msg = _T.get(key, {}).get(lang) or _T.get(key, {}).get("zh_CN") or key
    if kwargs:
        try:
            return msg.format(**kwargs)
        except Exception:
            return msg
    return msg


