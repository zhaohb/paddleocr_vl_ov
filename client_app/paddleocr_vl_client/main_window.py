from __future__ import annotations

import os
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Set

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QPixmap, QColor, QIcon
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QToolBar,
    QVBoxLayout,
    QWidget,
    QGraphicsDropShadowEffect,
    QStyle,
)

from .types import (
    PipelineInitConfig,
    PredictConfig,
    TaskItem,
    TaskType,
    HistoryTask,
    SUPPORTED_IMAGE_EXTS,
    SUPPORTED_DOC_EXTS,
)
from .worker import InferenceWorker
from .screenshot_overlay import ScreenshotOverlay, ScreenshotResult
from .markdown_preview import MarkdownPreviewWidget
from .pdf_utils import is_pdf


class DropArea(QFrame):
    def __init__(self, on_files, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._on_files = on_files
        self.setAcceptDrops(True)
        self.setObjectName("DropArea")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        self.title = QLabel("拖拽文件到这里")
        self.title.setAlignment(Qt.AlignCenter)
        self.hint = QLabel("支持：图片（png/jpg/jpeg/webp/bmp），可选 PDF")
        self.hint.setAlignment(Qt.AlignCenter)
        self.hint.setStyleSheet("color: #666;")
        layout.addWidget(self.title)
        layout.addWidget(self.hint)

        self.setStyleSheet(
            """
            #DropArea {
                border: 2px dashed #B7B7B7;
                border-radius: 10px;
                background: #FAFAFA;
            }
            #DropArea QLabel {
                color: #111111;
            }
            """
        )

    def dragEnterEvent(self, event):  # noqa: N802
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):  # noqa: N802
        urls = event.mimeData().urls()
        paths = [Path(u.toLocalFile()) for u in urls if u.isLocalFile()]
        self._on_files(paths)


class MainWindow(QMainWindow):
    # 任务表列索引（避免后续加列导致到处改 magic number）
    COL_FILE = 0
    COL_EXT = 1
    COL_TASK_TYPE = 2
    COL_STATUS = 3
    COL_OUT_DIR = 4

    TASK_TYPE_OPTIONS: List[TaskType] = ["ocr", "table", "chart", "formula"]
    def __init__(self) -> None:
        super().__init__()
        # 运行时主入口会再次覆盖（用于打包/多入口一致），这里给一个默认值
        self.setWindowTitle("PaddleOCR-VL OpenVINO Client")
        self.resize(1280, 820)

        self._tasks: List[TaskItem] = []
        self._history: List[HistoryTask] = []
        # 运行过程中不直接删除 tasks（避免 idx 变化导致 worker 混乱），在一次运行结束后统一归档
        self._archive_done_indices: Set[int] = set()
        self._worker: Optional[InferenceWorker] = None
        self._screenshot_overlay: Optional[ScreenshotOverlay] = None
        self._running_task_idx: Optional[int] = None
        self._last_executed_idx: int = -1
        # PDF 预览状态（当前任务的 page 列表与页码）
        self._pdf_preview_pages: List[Path] = []
        self._pdf_preview_page_idx: int = 0
        self._pdf_preview_task_idx: Optional[int] = None

        self._build_ui()
        self._bind_actions()
        self._load_history()

    def _cardify(self, w: QWidget) -> None:
        """
        给容器加轻量阴影 + 圆角（专业软件常见的 card 视觉）。
        """
        w.setObjectName("Card")
        eff = QGraphicsDropShadowEffect(w)
        eff.setBlurRadius(18)
        eff.setOffset(0, 6)
        eff.setColor(QColor(15, 20, 26, 35))
        w.setGraphicsEffect(eff)

    def _apply_design_system(self) -> None:
        """
        全局 UI 风格（不依赖第三方主题库）。
        """
        self.setStyleSheet(
            """
            QMainWindow { background: #0F141A; }

            /* Toolbar */
            QToolBar { background: #0F141A; color: #E6E6E6; border: none; padding: 6px; }
            QToolBar QToolButton { color: #E6E6E6; padding: 6px 10px; border-radius: 10px; }
            QToolBar QToolButton:hover { background: rgba(255,255,255,0.10); }
            QToolBar QToolButton:pressed { background: rgba(255,255,255,0.14); }

            /* Card container */
            QWidget#Card {
                background: #FFFFFF;
                border-radius: 14px;
                border: 1px solid #E2E6EA;
            }

            /* Buttons */
            QPushButton {
                padding: 8px 14px;
                border-radius: 10px;
                background: #1F6FEB;
                color: white;
                border: none;
                font-weight: 600;
            }
            /* 更紧凑的按钮（用于“任务队列”头部等区域） */
            QPushButton[compact="true"] {
                padding: 6px 10px;
                border-radius: 9px;
            }
            QPushButton:hover { background: #175CD3; }
            QPushButton:pressed { background: #1249A9; }
            QPushButton:disabled { background: #9BB9F5; color: #FFFFFF; }
            QPushButton#SecondaryButton { background: #EEF2FF; color: #1F6FEB; border: 1px solid #D0D5DD; }
            QPushButton#SecondaryButton:hover { background: #E8F1FF; }
            QPushButton#DangerButton { background: #D92D20; }
            QPushButton#DangerButton:hover { background: #B42318; }

            /* Inputs */
            QLineEdit, QComboBox, QSpinBox {
                background: #FFFFFF;
                border: 1px solid #D0D5DD;
                border-radius: 10px;
                padding: 7px 10px;
                color: #111111;
            }
            QLineEdit:focus, QComboBox:focus, QSpinBox:focus {
                border: 1px solid #1F6FEB;
            }
            QCheckBox { color: #111111; }

            /* Table */
            QTableWidget { background: #FFFFFF; border-radius: 14px; border: 1px solid #E2E6EA; gridline-color: #EEF2F6; }
            QTableWidget::item { padding: 8px; }
            QTableWidget::item:selected { background: #E8F1FF; color: #111111; }

            /* Tabs */
            QTabWidget::pane { border: none; }
            QTabBar::tab { background: #EDEFF2; color: #111111; padding: 8px 12px; border-radius: 10px; margin-right: 6px; }
            QTabBar::tab:selected { background: #FFFFFF; }

            /* Scrollbar */
            QScrollBar:vertical { background: transparent; width: 10px; margin: 2px; }
            QScrollBar::handle:vertical { background: #D0D5DD; border-radius: 5px; min-height: 20px; }
            QScrollBar::handle:vertical:hover { background: #98A2B3; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
            """
        )

    # ---------------- UI ----------------
    def _build_ui(self) -> None:
        self._apply_design_system()

        # Toolbar
        tb = QToolBar("Main")
        tb.setMovable(False)
        self.addToolBar(tb)

        self.action_add_files = QAction("添加文件", self)
        self.action_clear = QAction("清空", self)
        self.action_screenshot = QAction("截图", self)
        self.action_choose_output = QAction("输出目录", self)
        self.action_build_info = QAction("关于", self)

        # 给工具栏 action 加上系统图标（无需额外资源）
        st = QApplication.style()
        self.action_add_files.setIcon(st.standardIcon(QStyle.SP_FileIcon))
        self.action_clear.setIcon(st.standardIcon(QStyle.SP_TrashIcon))
        self.action_screenshot.setIcon(st.standardIcon(QStyle.SP_DesktopIcon))
        self.action_choose_output.setIcon(st.standardIcon(QStyle.SP_DirIcon))
        self.action_build_info.setIcon(st.standardIcon(QStyle.SP_MessageBoxInformation))

        tb.addAction(self.action_add_files)
        tb.addAction(self.action_clear)
        tb.addAction(self.action_screenshot)
        tb.addSeparator()
        tb.addAction(self.action_choose_output)
        tb.addSeparator()
        tb.addAction(self.action_build_info)

        # 主布局：左侧导航 + 右侧页面
        root_splitter = QSplitter(Qt.Horizontal)
        root_splitter.setChildrenCollapsible(False)

        self.nav = QListWidget()
        self.nav.setFixedWidth(180)
        self.nav.setSpacing(6)
        self.nav.setStyleSheet(
            """
            QListWidget {
                border: none;
                background: #0F141A;
                color: #E6E6E6;
                padding: 10px;
            }
            QListWidget::item {
                padding: 10px 12px;
                border-radius: 8px;
            }
            QListWidget::item:selected {
                background: #1F6FEB;
                color: white;
            }
            QListWidget::item:hover {
                background: rgba(255,255,255,0.08);
            }
            """
        )
        self.nav.addItem(QListWidgetItem("OCR / 解析"))
        self.nav.addItem(QListWidgetItem("历史任务"))
        self.nav.addItem(QListWidgetItem("设置"))
        self.nav.addItem(QListWidgetItem("关于"))
        self.nav.setCurrentRow(0)
        root_splitter.addWidget(self.nav)

        self.pages = QStackedWidget()
        root_splitter.addWidget(self.pages)

        # -------- Page 0: OCR / 解析 --------
        page_ocr = QWidget()
        # 明确页面文字颜色，避免受系统暗色主题 Palette 影响
        page_ocr.setStyleSheet(
            """
            QWidget { background:#F6F7F9; color:#111111; }
            QHeaderView::section {
                background: #FFFFFF;
                color: #111111;
                border: none;
                border-bottom: 1px solid #E2E6EA;
                padding: 8px;
                font-weight: 600;
            }
            QTabBar::tab {
                background: #EDEFF2;
                color: #111111;
                padding: 8px 12px;
                border-radius: 8px;
                margin-right: 6px;
            }
            QTabBar::tab:selected {
                background: #FFFFFF;
                color: #111111;
            }
            """
        )
        center_layout = QVBoxLayout(page_ocr)
        center_layout.setContentsMargins(12, 12, 12, 12)
        center_layout.setSpacing(10)

        # 顶部：导入区（Card）
        drop_card = QWidget()
        drop_card_layout = QVBoxLayout(drop_card)
        drop_card_layout.setContentsMargins(12, 12, 12, 12)
        drop_card_layout.setSpacing(10)
        self.drop_area = DropArea(self._add_paths)
        drop_card_layout.addWidget(self.drop_area)
        self._cardify(drop_card)
        center_layout.addWidget(drop_card)

        # 中部：任务队列（Card）
        task_card = QWidget()
        task_layout = QVBoxLayout(task_card)
        task_layout.setContentsMargins(12, 12, 12, 12)
        task_layout.setSpacing(10)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(8)
        title = QLabel("任务队列")
        title.setStyleSheet("font-weight:700; color:#111;")
        header_row.addWidget(title)
        header_row.addStretch(1)

        self.btn_run = QPushButton("开始")
        self.btn_rerun_selected = QPushButton("重跑所选")
        self.btn_rerun_selected.setObjectName("SecondaryButton")
        self.btn_delete_selected = QPushButton("删除所选")
        self.btn_delete_selected.setObjectName("SecondaryButton")
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setObjectName("DangerButton")
        self.btn_stop.setEnabled(False)
        self.btn_screenshot = QPushButton("截图")
        self.btn_screenshot.setObjectName("SecondaryButton")
        # 头部按钮更紧凑，避免占用过多横向空间
        for b in (self.btn_run, self.btn_rerun_selected, self.btn_delete_selected, self.btn_screenshot, self.btn_stop):
            b.setProperty("compact", True)
        header_row.addWidget(self.btn_run)
        header_row.addWidget(self.btn_rerun_selected)
        header_row.addWidget(self.btn_delete_selected)
        header_row.addWidget(self.btn_screenshot)
        header_row.addWidget(self.btn_stop)
        task_layout.addLayout(header_row)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["文件", "类型", "任务类型", "状态", "输出目录"])
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.verticalHeader().setVisible(False)
        self.table.horizontalHeader().setStretchLastSection(True)
        self.table.setObjectName("TaskTable")
        task_layout.addWidget(self.table, stretch=1)

        footer_row = QHBoxLayout()
        self.progress = QLabel("0/0")
        self.progress.setStyleSheet("color:#667085;")
        footer_row.addWidget(QLabel("进度："))
        footer_row.addWidget(self.progress)
        footer_row.addStretch(1)
        task_layout.addLayout(footer_row)

        self._cardify(task_card)
        center_layout.addWidget(task_card, stretch=1)

        # 底部：日志（Card）
        log_card = QWidget()
        log_layout = QVBoxLayout(log_card)
        log_layout.setContentsMargins(12, 12, 12, 12)
        log_layout.setSpacing(8)
        log_title = QLabel("日志")
        log_title.setStyleSheet("font-weight:700; color:#111;")
        log_layout.addWidget(log_title)
        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setPlaceholderText("日志输出 ...")
        self.log.setMaximumBlockCount(5000)
        self.log.setStyleSheet("border:1px solid #E2E6EA; border-radius:10px; background:#FFFFFF;")
        log_layout.addWidget(self.log, stretch=1)
        self._cardify(log_card)
        center_layout.addWidget(log_card, stretch=0)

        self.pages.addWidget(page_ocr)

        # -------- Page 1: 设置 --------
        self.settings = QWidget()
        self.settings.setStyleSheet("background:#F6F7F9; color:#111111;")
        settings_layout = QVBoxLayout(self.settings)
        settings_layout.setContentsMargins(12, 12, 12, 12)
        settings_layout.setSpacing(10)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.edit_layout_model = QLineEdit()
        self.edit_layout_model.setPlaceholderText("布局模型路径（留空=自动下载）")
        self.edit_vlm_model = QLineEdit()
        self.edit_vlm_model.setPlaceholderText("VLM 模型路径（留空=自动下载）")
        self.edit_cache_dir = QLineEdit()
        self.edit_cache_dir.setPlaceholderText("ModelScope cache_dir（可选）")
        # 去掉系统自带 frame，避免与 QSS 的 border 叠加出现“双层边框”
        for w in (self.edit_layout_model, self.edit_vlm_model, self.edit_cache_dir):
            w.setFrame(False)

        self.combo_vlm_device = QComboBox()
        self.combo_vlm_device.addItems(["CPU", "GPU", "AUTO"])
        self.combo_vlm_device.setCurrentText("GPU")
        self.combo_vlm_device.setFrame(False)

        self.combo_layout_device = QComboBox()
        self.combo_layout_device.addItems(["CPU", "GPU", "NPU", "AUTO"])
        self.combo_layout_device.setCurrentText("GPU")
        self.combo_layout_device.setFrame(False)

        self.combo_layout_precision = QComboBox()
        self.combo_layout_precision.addItems(["fp16", "fp32", "combined_fp16", "combined_fp32"])
        self.combo_layout_precision.setCurrentText("fp16")
        self.combo_layout_precision.setFrame(False)

        self.chk_use_layout = QCheckBox("启用布局检测")
        self.chk_use_layout.setChecked(True)

        self.slider_layout_thresh = QSlider(Qt.Horizontal)
        self.slider_layout_thresh.setMinimum(10)
        self.slider_layout_thresh.setMaximum(100)
        self.slider_layout_thresh.setValue(50)
        self.lbl_layout_thresh = QLabel("0.50")

        self.spin_max_tokens = QSpinBox()
        self.spin_max_tokens.setRange(256, 4096)
        self.spin_max_tokens.setSingleStep(256)
        self.spin_max_tokens.setValue(1024)
        self.spin_max_tokens.setFrame(False)

        self.chk_llm_int4 = QCheckBox("LLM INT4 压缩")
        self.chk_llm_int4.setChecked(False)
        self.chk_vision_int8 = QCheckBox("Vision INT8 量化")
        self.chk_vision_int8.setChecked(True)
        self.chk_llm_int8_compress = QCheckBox("LLM INT8 压缩")
        self.chk_llm_int8_compress.setChecked(True)
        self.chk_llm_int8_quant = QCheckBox("LLM INT8 量化")
        self.chk_llm_int8_quant.setChecked(True)

        self.edit_output_dir = QLineEdit(str((Path.cwd() / "output").resolve()))
        self.edit_output_dir.setFrame(False)
        self.btn_pick_output = QPushButton("选择...")

        form.addRow("layout_model_path", self.edit_layout_model)
        form.addRow("vlm_model_path", self.edit_vlm_model)
        form.addRow("cache_dir", self.edit_cache_dir)
        form.addRow("vlm_device", self.combo_vlm_device)
        form.addRow("layout_device", self.combo_layout_device)
        form.addRow("layout_precision", self.combo_layout_precision)
        form.addRow("", self.chk_use_layout)

        thresh_row = QHBoxLayout()
        thresh_row.addWidget(self.slider_layout_thresh, stretch=1)
        thresh_row.addWidget(self.lbl_layout_thresh)
        thresh_wrap = QWidget()
        thresh_wrap.setLayout(thresh_row)
        form.addRow("layout_threshold", thresh_wrap)
        form.addRow("max_new_tokens", self.spin_max_tokens)

        form.addRow("", self.chk_llm_int4)
        form.addRow("", self.chk_vision_int8)
        form.addRow("", self.chk_llm_int8_compress)
        form.addRow("", self.chk_llm_int8_quant)

        out_row = QHBoxLayout()
        out_row.addWidget(self.edit_output_dir, stretch=1)
        out_row.addWidget(self.btn_pick_output)
        out_wrap = QWidget()
        out_wrap.setLayout(out_row)
        form.addRow("输出目录", out_wrap)

        title = QLabel("设置")
        title.setStyleSheet("font-size:18px;font-weight:600;color:#111;")
        settings_layout.addWidget(title)
        settings_layout.addLayout(form)

        note = QLabel("提示：若 layout_model_path 指向具体 .xml 文件，则 layout_precision 会被忽略。")
        note.setWordWrap(True)
        note.setStyleSheet("color:#666;")
        settings_layout.addWidget(note)
        settings_layout.addStretch(1)

        # -------- Page 2: 历史任务 --------
        history = QWidget()
        history.setStyleSheet("background:#F6F7F9; color:#111111;")
        history_layout = QVBoxLayout(history)
        history_layout.setContentsMargins(12, 12, 12, 12)
        history_layout.setSpacing(10)

        history_title = QLabel("历史任务（仅成功）")
        history_title.setStyleSheet("font-size:18px;font-weight:600;color:#111;")
        history_layout.addWidget(history_title)

        self.history_table = QTableWidget(0, 6)
        self.history_table.setHorizontalHeaderLabels(["时间", "文件", "类型", "任务类型", "输出目录", "摘要"])
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.history_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.history_table.setAlternatingRowColors(True)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.horizontalHeader().setStretchLastSection(True)
        self.history_table.setObjectName("HistoryTable")
        # 历史表格：最多显示 5 行（其余可滚动查看）
        self.history_table.setMinimumHeight(160)
        self.history_table.setMaximumHeight(240)  # 初始化值，后续在 _refresh_history_table 动态调整
        history_layout.addWidget(self.history_table, stretch=0)

        # 操作按钮放在表格下面，避免被下方预览区挤出可视区域
        history_btn_row = QHBoxLayout()
        history_btn_row.setContentsMargins(0, 0, 0, 0)
        history_btn_row.setSpacing(8)
        self.btn_hist_open_out = QPushButton("打开输出目录")
        self.btn_hist_open_out.setObjectName("SecondaryButton")
        self.btn_hist_delete = QPushButton("删除所选")
        # 与“清空历史”保持一致的外观（次级按钮）
        self.btn_hist_delete.setObjectName("SecondaryButton")
        self.btn_hist_clear = QPushButton("清空历史")
        self.btn_hist_clear.setObjectName("SecondaryButton")
        history_btn_row.addWidget(self.btn_hist_open_out)
        history_btn_row.addWidget(self.btn_hist_delete)
        history_btn_row.addWidget(self.btn_hist_clear)
        history_btn_row.addStretch(1)
        history_layout.addLayout(history_btn_row)

        # 任务详情（上） + 预览（下）
        self.hist_detail = QLabel("请选择一条历史任务查看详情…")
        self.hist_detail.setWordWrap(True)
        self.hist_detail.setStyleSheet("color:#344054; background:#FFFFFF; border:1px solid #E2E6EA; border-radius:12px; padding:10px;")
        history_layout.addWidget(self.hist_detail, stretch=0)

        # 对比预览：原图 vs Markdown 可视化（左右）
        self.hist_input_image = QLabel("原图预览")
        self.hist_input_image.setAlignment(Qt.AlignCenter)
        self.hist_input_image.setMinimumHeight(220)
        # 与 Markdown 预览一致：白底卡片风格（避免黑底显得像“图片画布”）
        self.hist_input_image.setStyleSheet("background:#FFFFFF;color:#667085;border:1px solid #E2E6EA;border-radius:10px;")

        self.hist_markdown_preview = MarkdownPreviewWidget()

        compare_wrap = QWidget()
        compare_layout = QHBoxLayout(compare_wrap)
        compare_layout.setContentsMargins(0, 0, 0, 0)
        compare_layout.setSpacing(10)
        compare_layout.addWidget(self.hist_input_image, stretch=1)
        compare_layout.addWidget(self.hist_markdown_preview, stretch=1)

        self.hist_output_image = QLabel("输出图预览")
        self.hist_output_image.setAlignment(Qt.AlignCenter)
        self.hist_output_image.setMinimumHeight(220)
        self.hist_output_image.setStyleSheet("background:#0B0B0B;color:#DDD;border-radius:10px;")

        self.hist_preview_tabs = QTabWidget()
        self.hist_preview_tabs.addTab(compare_wrap, "对比（原图 vs Markdown）")
        self.hist_preview_tabs.addTab(self.hist_output_image, "输出图")
        history_layout.addWidget(self.hist_preview_tabs, stretch=1)

        # pages 顺序与左侧导航一致：0=OCR, 1=History, 2=Settings, 3=About
        self.pages.addWidget(history)
        self.pages.addWidget(self.settings)

        # -------- Page 3: 关于 --------
        about = QWidget()
        about.setStyleSheet("background:#F6F7F9; color:#111111;")
        about_layout = QVBoxLayout(about)
        about_layout.setContentsMargins(20, 20, 20, 20)
        about_layout.setSpacing(10)
        about_title = QLabel("关于")
        about_title.setStyleSheet("font-size:18px;font-weight:600;color:#111;")
        about_layout.addWidget(about_title)
        about_text = QLabel(
            "PaddleOCR-VL OpenVINO Desktop Client\n\n"
            "- 支持拖拽导入 / 批量任务 / 结果预览\n"
            "- 导出 result.md / result.json / vis.png\n"
        )
        about_text.setStyleSheet("color:#333;")
        about_text.setWordWrap(True)
        about_layout.addWidget(about_text)
        about_layout.addStretch(1)
        self.pages.addWidget(about)

        # 全局样式（轻量化，避免侵入控件行为）
        # 全局样式已在 _apply_design_system() 中统一设置

        self.setCentralWidget(root_splitter)

    def _bind_actions(self) -> None:
        self.action_add_files.triggered.connect(self._choose_files)
        self.action_clear.triggered.connect(self._clear_tasks)
        self.action_screenshot.triggered.connect(self._start_screenshot)
        self.action_choose_output.triggered.connect(self._choose_output_dir)
        self.action_build_info.triggered.connect(self._show_about)

        self.btn_pick_output.clicked.connect(self._choose_output_dir)
        self.btn_run.clicked.connect(self._run)
        self.btn_rerun_selected.clicked.connect(self._rerun_selected)
        self.btn_delete_selected.clicked.connect(self._delete_selected_tasks)
        self.btn_stop.clicked.connect(self._stop)
        self.btn_screenshot.clicked.connect(self._start_screenshot)
        self.slider_layout_thresh.valueChanged.connect(self._sync_layout_thresh_label)
        self.table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self.nav.currentRowChanged.connect(self._on_nav_changed)
        self.chk_use_layout.toggled.connect(self._on_use_layout_toggled)
        self.history_table.itemSelectionChanged.connect(self._on_history_selection_changed)
        self.btn_hist_open_out.clicked.connect(self._open_selected_history_output_dir)
        self.btn_hist_delete.clicked.connect(self._delete_selected_history)
        self.btn_hist_clear.clicked.connect(self._clear_history)

    def _on_use_layout_toggled(self, checked: bool) -> None:
        """
        开关布局检测时，控制“任务类型”是否可编辑：
        - 开启布局检测：任务类型由布局模型输出决定，手动 task_type 不生效 -> 禁用下拉框
        - 关闭布局检测：允许手动指定 ocr/table/chart/formula
        """
        for row in range(self.table.rowCount()):
            w = self.table.cellWidget(row, self.COL_TASK_TYPE)
            if isinstance(w, QComboBox):
                w.setEnabled(not checked)

    def _make_task_type_combo(self, row: int, value: TaskType) -> QComboBox:
        combo = QComboBox()
        combo.addItems(list(self.TASK_TYPE_OPTIONS))
        combo.setCurrentText(value)
        combo.setFrame(False)
        # 开启布局检测时禁用（此时手动 task_type 不生效）
        combo.setEnabled(not self.chk_use_layout.isChecked())

        def _on_changed(v: str) -> None:
            if 0 <= row < len(self._tasks):
                try:
                    self._tasks[row].task_type = v  # type: ignore[assignment]
                except Exception:
                    self._tasks[row].task_type = "ocr"  # type: ignore[assignment]

        combo.currentTextChanged.connect(_on_changed)
        return combo

    # ---------------- Actions ----------------
    def _append_log(self, msg: str) -> None:
        self.log.appendPlainText(msg.rstrip())

    def _sync_layout_thresh_label(self) -> None:
        val = self.slider_layout_thresh.value() / 100.0
        self.lbl_layout_thresh.setText(f"{val:.2f}")

    def _choose_files(self) -> None:
        exts = " ".join([f"*{x}" for x in SUPPORTED_IMAGE_EXTS + SUPPORTED_DOC_EXTS])
        paths, _ = QFileDialog.getOpenFileNames(self, "选择文件", str(Path.cwd()), f"Files ({exts})")
        if not paths:
            return
        self._add_paths([Path(p) for p in paths])

    def _choose_output_dir(self) -> None:
        # 输出目录属于“设置”
        self.nav.setCurrentRow(2)
        d = QFileDialog.getExistingDirectory(self, "选择输出目录", self.edit_output_dir.text().strip() or str(Path.cwd()))
        if not d:
            return
        self.edit_output_dir.setText(d)
        self.edit_output_dir.setFocus()
        # 输出目录变更后，历史记录也切到对应目录
        self._load_history()

    def _show_about(self) -> None:
        # 这里直接切到 About 页
        self.nav.setCurrentRow(3)

    def _clear_tasks(self) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "提示", "正在运行中，请先停止。")
            return
        self._tasks.clear()
        self._archive_done_indices.clear()
        self.table.setRowCount(0)
        self.progress.setText("0/0")

    def _delete_selected_tasks(self) -> None:
        """
        删除任务队列中选中的任务。
        约束：
        - 运行中不允许删除（worker 依赖索引，删除会导致错乱）
        """
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "提示", "正在运行中，无法删除任务。请先停止。")
            return

        rows = self.table.selectionModel().selectedRows()
        if not rows:
            QMessageBox.information(self, "提示", "请先在任务队列表格中选择要删除的任务行。")
            return
        indices = sorted({r.row() for r in rows if 0 <= r.row() < len(self._tasks)})
        if not indices:
            return

        msg = f"确定要删除选中的 {len(indices)} 个任务吗？"
        ok = QMessageBox.question(self, "确认删除", msg) == QMessageBox.Yes
        if not ok:
            return

        # 倒序删除，避免索引变化
        for idx in sorted(indices, reverse=True):
            try:
                self._tasks.pop(idx)
            except Exception:
                pass

        # 删除后清理归档索引，避免残留
        self._archive_done_indices.clear()
        self._running_task_idx = None
        self._last_executed_idx = min(self._last_executed_idx, len(self._tasks) - 1)

        self._refresh_table()
        if self._tasks:
            self.progress.setText(f"0/{len(self._tasks)}")
        else:
            self.progress.setText("0/0")

    def _start_screenshot(self) -> None:
        """
        启动截图：全屏框选 → 保存 PNG → 加入任务队列。
        """
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "提示", "正在运行中，无法截图。")
            return

        # 切换到 OCR 页面更符合直觉
        self.nav.setCurrentRow(0)

        from PySide6.QtCore import QTimer

        save_dir = (Path(__file__).resolve().parents[2] / "client_app" / "output" / "_screenshots").resolve()
        self._screenshot_overlay = ScreenshotOverlay(save_dir=save_dir, parent=None)
        self._screenshot_overlay.captured.connect(self._on_screenshot_captured)
        self._screenshot_overlay.canceled.connect(self._on_screenshot_canceled)

        # 先隐藏主窗口，避免截到自己（给系统一小段时间刷新）
        self.setWindowOpacity(0.0)
        self.setEnabled(False)

        def _go():
            if self._screenshot_overlay:
                self._screenshot_overlay.start()

        QTimer.singleShot(150, _go)

    def _restore_after_screenshot(self) -> None:
        self.setEnabled(True)
        self.setWindowOpacity(1.0)
        self.raise_()
        self.activateWindow()

    def _on_screenshot_captured(self, result: ScreenshotResult) -> None:
        self._restore_after_screenshot()
        self._append_log(f"截图完成：{result.saved_path}")
        self._add_paths([result.saved_path])

    def _on_screenshot_canceled(self) -> None:
        self._restore_after_screenshot()
        self._append_log("截图已取消（ESC 或选区过小）")

    def _add_paths(self, paths: List[Path]) -> None:
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "提示", "正在运行中，暂不支持添加文件。")
            return

        for p in paths:
            if p.is_dir():
                # 目录：递归收集图片
                for f in p.rglob("*"):
                    self._maybe_add_file(f)
            else:
                self._maybe_add_file(p)

        self._refresh_table()

    def _maybe_add_file(self, p: Path) -> None:
        if not p.exists():
            return
        suf = p.suffix.lower()
        if suf not in SUPPORTED_IMAGE_EXTS + SUPPORTED_DOC_EXTS:
            return
        self._tasks.append(TaskItem(input_path=p, task_type="ocr"))

    def _refresh_table(self) -> None:
        self.table.setRowCount(len(self._tasks))
        for i, t in enumerate(self._tasks):
            self.table.setItem(i, self.COL_FILE, QTableWidgetItem(str(t.input_path)))
            self.table.setItem(i, self.COL_EXT, QTableWidgetItem(t.input_path.suffix.lower().lstrip(".")))

            # 任务类型：用下拉框便于逐任务指定
            self.table.setCellWidget(i, self.COL_TASK_TYPE, self._make_task_type_combo(i, t.task_type))

            self.table.setItem(i, self.COL_STATUS, QTableWidgetItem(t.status))
            self.table.setItem(i, self.COL_OUT_DIR, QTableWidgetItem(str(t.output_dir) if t.output_dir else ""))
        if self._tasks:
            self.progress.setText(f"0/{len(self._tasks)}")

    def _recompute_task_statuses(self) -> None:
        """
        统一刷新任务状态：
        - pending/running 由队列位置决定（未执行/正在执行）
        - done/error 由任务真实结果决定（成功/失败）
        """
        for i, t in enumerate(self._tasks):
            # 已完成任务永远保持 done/error，不因后续任务启动而被覆盖
            if t.status in ("done", "error") and i != self._running_task_idx:
                self._set_row_status(i)
                st_item = self.table.item(i, self.COL_STATUS)
                if st_item is not None and t.error:
                    st_item.setToolTip(t.error)
                continue

            if self._running_task_idx is not None and i == self._running_task_idx:
                t.status = "running"
            else:
                # 未完成的任务统一显示 pending
                t.status = "pending"

            self._set_row_status(i)

            # 错误信息保留在 tooltip，便于定位
            st_item = self.table.item(i, self.COL_STATUS)
            if st_item is not None and t.error:
                st_item.setToolTip(t.error)

    def _current_init_cfg(self) -> PipelineInitConfig:
        return PipelineInitConfig(
            layout_model_path=self.edit_layout_model.text().strip() or None,
            vlm_model_path=self.edit_vlm_model.text().strip() or None,
            cache_dir=self.edit_cache_dir.text().strip() or None,
            vlm_device=self.combo_vlm_device.currentText(),  # type: ignore[arg-type]
            layout_device=self.combo_layout_device.currentText(),  # type: ignore[arg-type]
            layout_precision=self.combo_layout_precision.currentText(),  # type: ignore[arg-type]
            llm_int4_compress=self.chk_llm_int4.isChecked(),
            vision_int8_quant=self.chk_vision_int8.isChecked(),
            llm_int8_compress=self.chk_llm_int8_compress.isChecked(),
            llm_int8_quant=self.chk_llm_int8_quant.isChecked(),
        )

    def _current_predict_cfg(self) -> PredictConfig:
        return PredictConfig(
            use_layout_detection=self.chk_use_layout.isChecked(),
            layout_threshold=self.slider_layout_thresh.value() / 100.0,
            max_new_tokens=int(self.spin_max_tokens.value()),
        )

    def _run(self) -> None:
        if not self._tasks:
            QMessageBox.information(self, "提示", "请先添加文件。")
            return
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "提示", "正在运行中。")
            return

        # 只执行 pending 的任务；已完成（done/error）的任务保持原状态，不会被新一轮运行重置
        pending_indices = [i for i, t in enumerate(self._tasks) if t.status not in ("done", "error")]
        if not pending_indices:
            QMessageBox.information(self, "提示", "没有待执行（pending）的任务。")
            return

        out_dir = Path(self.edit_output_dir.text().strip() or "output").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)

        self._append_log(f"输出目录：{out_dir}")
        self._append_log("开始推理 ...")

        # 复位 running（若上次被中断，可能残留 running）
        self._running_task_idx = None
        for t in self._tasks:
            if t.status == "running":
                t.status = "pending"
        self._refresh_table()
        self.progress.setText(f"0/{len(pending_indices)}")

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

        pred_cfg = self._current_predict_cfg()
        # 记录本次任务是否启用 layout detection（用于历史任务展示 task_type）
        for i in pending_indices:
            if 0 <= i < len(self._tasks):
                self._tasks[i].used_layout_detection = pred_cfg.use_layout_detection

        self._worker = InferenceWorker(
            tasks=self._tasks,
            output_root=out_dir,
            init_cfg=self._current_init_cfg(),
            pred_cfg=pred_cfg,
            run_indices=None,
        )
        self._worker.signals.log.connect(self._append_log)
        self._worker.signals.progress.connect(self._on_progress)
        self._worker.signals.task_started.connect(self._on_task_started)
        self._worker.signals.task_finished.connect(self._on_task_finished)
        self._worker.signals.task_failed.connect(self._on_task_failed)
        self._worker.signals.preview_ready.connect(self._on_preview_ready)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    def _stop(self) -> None:
        if self._worker and self._worker.isRunning():
            self._append_log("请求停止 ...")
            self._worker.request_stop()
        self.btn_stop.setEnabled(False)
        # UI 层面先清掉 running（后续 worker 结束会更新 last_executed_idx）
        self._running_task_idx = None
        self._recompute_task_statuses()

    def _rerun_selected(self) -> None:
        """
        重新运行选中的 task（只跑所选，不影响其它已完成任务）。
        """
        if self._worker and self._worker.isRunning():
            QMessageBox.warning(self, "提示", "正在运行中，请先停止。")
            return

        rows = self.table.selectionModel().selectedRows()
        if not rows:
            QMessageBox.information(self, "提示", "请先在表格中选择要重跑的任务行。")
            return
        indices = sorted({r.row() for r in rows if 0 <= r.row() < len(self._tasks)})
        if not indices:
            QMessageBox.information(self, "提示", "未选中有效任务。")
            return

        out_dir = Path(self.edit_output_dir.text().strip() or "output").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        self._append_log(f"输出目录：{out_dir}")
        self._append_log(f"重跑所选任务：{len(indices)} 个")

        # 把所选任务恢复到 pending，并清理旧缓存字段（状态/错误/预览内容）
        for idx in indices:
            t = self._tasks[idx]
            t.status = "pending"
            t.error = None
            t.summary = None
            t.markdown_text = None
            t.json_text = None
            t.vis_image_path = None
            self._set_row_status(idx)

        self._running_task_idx = None
        self._recompute_task_statuses()
        self.progress.setText(f"0/{len(indices)}")

        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)

        pred_cfg = self._current_predict_cfg()
        for i in indices:
            if 0 <= i < len(self._tasks):
                self._tasks[i].used_layout_detection = pred_cfg.use_layout_detection

        self._worker = InferenceWorker(
            tasks=self._tasks,
            output_root=out_dir,
            init_cfg=self._current_init_cfg(),
            pred_cfg=pred_cfg,
            run_indices=indices,
        )
        self._worker.signals.log.connect(self._append_log)
        self._worker.signals.progress.connect(self._on_progress)
        self._worker.signals.task_started.connect(self._on_task_started)
        self._worker.signals.task_finished.connect(self._on_task_finished)
        self._worker.signals.task_failed.connect(self._on_task_failed)
        self._worker.signals.preview_ready.connect(self._on_preview_ready)
        self._worker.finished.connect(self._on_worker_finished)
        self._worker.start()

    # ---------------- Worker callbacks ----------------
    def _on_progress(self, current: int, total: int) -> None:
        self.progress.setText(f"{current}/{total}")

    def _on_task_started(self, idx: int) -> None:
        self._running_task_idx = idx
        # running 前面的都视为已执行（即使之前失败）
        self._last_executed_idx = max(self._last_executed_idx, idx - 1)
        self._recompute_task_statuses()

        # 自动选中当前行，便于观察运行进度
        try:
            self.table.blockSignals(True)
            self.table.selectRow(idx)
        finally:
            self.table.blockSignals(False)

    def _on_task_finished(self, idx: int) -> None:
        self._running_task_idx = None
        self._last_executed_idx = max(self._last_executed_idx, idx)
        self._recompute_task_statuses()
        self._set_row_output_dir(idx)
        # 成功任务加入历史（但不立刻从 tasks 删除，避免 worker 索引错乱）
        if 0 <= idx < len(self._tasks) and self._tasks[idx].status == "done":
            self._append_history(self._tasks[idx])
            self._archive_done_indices.add(idx)

    def _on_task_failed(self, idx: int, err: str) -> None:
        # 失败 -> error（错误信息保留到 tooltip + 日志）
        self._tasks[idx].error = err
        self._tasks[idx].status = "error"
        self._running_task_idx = None
        self._last_executed_idx = max(self._last_executed_idx, idx)
        self._recompute_task_statuses()
        self._set_row_output_dir(idx)

    def _set_row_status(self, idx: int) -> None:
        item = self.table.item(idx, self.COL_STATUS)
        if item is None:
            item = QTableWidgetItem()
            self.table.setItem(idx, self.COL_STATUS, item)
        status = self._tasks[idx].status
        item.setText(status)
        item.setTextAlignment(Qt.AlignCenter)

        # 状态 badge（专业软件常见视觉）
        palette = {
            "pending": ("#667085", "#F2F4F7"),
            "running": ("#175CD3", "#E8F1FF"),
            "done": ("#067647", "#ECFDF3"),
            "error": ("#B42318", "#FEF3F2"),
        }
        fg, bg = palette.get(status, ("#344054", "#FFFFFF"))
        item.setForeground(QColor(fg))
        item.setBackground(QColor(bg))

    def _set_row_output_dir(self, idx: int) -> None:
        item = self.table.item(idx, self.COL_OUT_DIR)
        if item is None:
            item = QTableWidgetItem()
            self.table.setItem(idx, self.COL_OUT_DIR, item)
        item.setText(str(self._tasks[idx].output_dir) if self._tasks[idx].output_dir else "")

    def _on_preview_ready(self, idx: int, qimg, md: str, js: str) -> None:
        # OCR/解析页已移除预览区：这里只保留数据落盘与任务状态更新（由 worker 负责）
        # 仍然保留该回调以避免断开信号导致的逻辑分叉。
        return

    def _on_worker_finished(self) -> None:
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._append_log("全部任务结束。")
        self._running_task_idx = None
        self._recompute_task_statuses()
        self._archive_done_tasks()

    def _on_table_selection_changed(self) -> None:
        # OCR/解析页已移除预览区：选中行不再驱动预览刷新
        return

    # NOTE: OCR/解析页已删除预览功能，相关的 PDF 翻页/Markdown 渲染/输入输出预览代码已移除。

    def _on_nav_changed(self, row: int) -> None:
        # 0=OCR, 1=History, 2=Settings, 3=About（pages 添加顺序与 nav 一致）
        if 0 <= row < self.pages.count():
            self.pages.setCurrentIndex(row)

    # ---------------- History ----------------
    def _history_file_path(self) -> Path:
        out_dir = Path(self.edit_output_dir.text().strip() or "output").resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / "_history.jsonl"

    def _load_history(self) -> None:
        self._history.clear()
        p = self._history_file_path()
        if not p.exists():
            self._refresh_history_table()
            return
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    self._history.append(
                        HistoryTask(
                            finished_at=str(obj.get("finished_at", "")),
                            input_path=str(obj.get("input_path", "")),
                            file_type=str(obj.get("file_type", "")),
                            task_type=obj.get("task_type", None),
                            output_dir=str(obj.get("output_dir", "")),
                            summary=str(obj.get("summary", "")),
                        )
                    )
                except Exception:
                    continue
        except Exception:
            # 读失败就当没有历史
            self._history.clear()
        self._refresh_history_table()

    def _save_history_file(self) -> None:
        """
        将当前历史记录完整写回 `_history.jsonl`（用于删除/清空后的落盘）。
        """
        p = self._history_file_path()
        try:
            if not self._history:
                if p.exists():
                    p.unlink()
                return
            with p.open("w", encoding="utf-8") as f:
                for rec in self._history:
                    f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def _append_history(self, task: TaskItem) -> None:
        if not task.output_dir:
            return
        # 仅当关闭 layout detection 时记录 task_type
        task_type = None if task.used_layout_detection else task.task_type
        rec = HistoryTask(
            finished_at=datetime.now().isoformat(timespec="seconds"),
            input_path=str(task.input_path),
            file_type=task.input_path.suffix.lower().lstrip("."),
            task_type=task_type,
            output_dir=str(Path(task.output_dir).resolve()),
            summary=str(task.summary or ""),
        )
        self._history.insert(0, rec)
        try:
            p = self._history_file_path()
            with p.open("a", encoding="utf-8") as f:
                f.write(json.dumps(rec.__dict__, ensure_ascii=False) + "\n")
        except Exception:
            pass
        self._refresh_history_table()

    def _refresh_history_table(self) -> None:
        if not hasattr(self, "history_table"):
            return
        self.history_table.setRowCount(len(self._history))
        for i, r in enumerate(self._history):
            self.history_table.setItem(i, 0, QTableWidgetItem(r.finished_at))
            self.history_table.setItem(i, 1, QTableWidgetItem(r.input_path))
            self.history_table.setItem(i, 2, QTableWidgetItem(r.file_type))
            self.history_table.setItem(i, 3, QTableWidgetItem(r.task_type or ""))
            self.history_table.setItem(i, 4, QTableWidgetItem(r.output_dir))
            self.history_table.setItem(i, 5, QTableWidgetItem(r.summary))
        self._sync_history_table_height()

    def _sync_history_table_height(self) -> None:
        """
        历史表格默认最多显示 5 行，避免页面被表格占满；仍可通过滚动查看更多。
        """
        try:
            header_h = self.history_table.horizontalHeader().height()
            row_h = self.history_table.rowHeight(0) if self.history_table.rowCount() > 0 else 28
            visible = min(5, max(0, self.history_table.rowCount()))
            # 额外 padding 给边框/滚动条
            h = header_h + (row_h * max(visible, 1)) + 10
            self.history_table.setMaximumHeight(max(160, min(h, 320)))
        except Exception:
            pass

    def _on_history_selection_changed(self) -> None:
        rows = self.history_table.selectionModel().selectedRows()
        if not rows:
            self.hist_detail.setText("请选择一条历史任务查看详情…")
            self.hist_markdown_preview.clear()
            self.hist_input_image.clear()
            self.hist_input_image.setText("原图预览")
            self.hist_output_image.clear()
            self.hist_output_image.setText("输出图预览")
            return
        idx = rows[0].row()
        if idx < 0 or idx >= len(self._history):
            return
        rec = self._history[idx]
        out_dir = Path(rec.output_dir)
        if not out_dir.exists():
            self.hist_detail.setText("输出目录不存在")
            self.hist_markdown_preview.clear()
            self.hist_input_image.setText("原图预览")
            self.hist_output_image.setText("输出图预览")
            return

        # 任务详情（上方文本）
        lines = [
            f"时间：{rec.finished_at}",
            f"文件：{rec.input_path}",
            f"类型：{rec.file_type}",
        ]
        if rec.task_type:
            lines.append(f"任务类型：{rec.task_type}")
        lines += [
            f"输出目录：{rec.output_dir}",
            f"摘要：{rec.summary}",
        ]
        self.hist_detail.setText("\n".join(lines))

        # 原图：图片用输入文件；PDF 优先用输出目录 pages/page_0001.png
        input_path = Path(rec.input_path)
        input_preview = input_path
        if input_path.suffix.lower() == ".pdf":
            cand = out_dir / "pages" / "page_0001.png"
            if cand.exists():
                input_preview = cand
        self.hist_input_image.clear()
        self.hist_input_image.setText("原图预览")
        if input_preview.exists():
            pix = QPixmap(str(input_preview))
            if not pix.isNull():
                self.hist_input_image.setPixmap(
                    pix.scaled(self.hist_input_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )

        # 找 md：优先 <stem>.md，其次 result.md
        stem = Path(rec.input_path).stem
        candidates = [out_dir / f"{stem}.md", out_dir / "result.md"]
        md_path = next((c for c in candidates if c.exists()), None)
        if md_path:
            try:
                text = md_path.read_text(encoding="utf-8")
            except Exception:
                text = ""
            self.hist_markdown_preview.set_markdown(text, base_dir=out_dir)
        else:
            self.hist_markdown_preview.clear()

        # 输出图：优先 vis.png
        vis_path = out_dir / "vis.png"
        self.hist_output_image.clear()
        self.hist_output_image.setText("输出图预览")
        if vis_path.exists():
            pix = QPixmap(str(vis_path))
            if not pix.isNull():
                self.hist_output_image.setPixmap(
                    pix.scaled(self.hist_output_image.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                )

    def _open_selected_history_output_dir(self) -> None:
        rows = self.history_table.selectionModel().selectedRows()
        if not rows:
            QMessageBox.information(self, "提示", "请先选择一条历史任务。")
            return
        idx = rows[0].row()
        if idx < 0 or idx >= len(self._history):
            return
        out_dir = Path(self._history[idx].output_dir)
        if out_dir.exists():
            try:
                os.startfile(str(out_dir))  # type: ignore[attr-defined]
            except Exception:
                pass

    def _delete_selected_history(self) -> None:
        """
        删除选中的历史任务：
        - 从历史列表中移除
        - 删除其输出目录（包含 markdown/json/vis/pages 等产物）
        - 重写 _history.jsonl
        """
        rows = self.history_table.selectionModel().selectedRows()
        if not rows:
            QMessageBox.information(self, "提示", "请先选择要删除的历史任务。")
            return
        indices = sorted({r.row() for r in rows if 0 <= r.row() < len(self._history)})
        if not indices:
            return

        if len(indices) == 1:
            rec = self._history[indices[0]]
            msg = f"确定要删除该历史任务吗？\n\n{rec.input_path}\n\n同时会删除输出目录：\n{rec.output_dir}"
        else:
            msg = f"确定要删除选中的 {len(indices)} 条历史任务吗？\n\n注意：会同时删除每条任务的输出目录。"

        ok = QMessageBox.question(self, "确认删除", msg) == QMessageBox.Yes
        if not ok:
            return

        # 先删除输出目录（尽量彻底，但不影响主流程）
        for idx in indices:
            try:
                out_dir = Path(self._history[idx].output_dir)
                if out_dir.exists():
                    shutil.rmtree(out_dir, ignore_errors=True)
            except Exception:
                pass

        # 倒序删除记录，避免索引变化
        for idx in sorted(indices, reverse=True):
            try:
                self._history.pop(idx)
            except Exception:
                pass

        self._save_history_file()
        self._refresh_history_table()

        # 清空详情/预览，避免还显示已删除项
        self.hist_detail.setText("请选择一条历史任务查看详情…")
        self.hist_markdown_preview.clear()
        self.hist_input_image.clear()
        self.hist_input_image.setText("原图预览")
        self.hist_output_image.clear()
        self.hist_output_image.setText("输出图预览")

    def _clear_history(self) -> None:
        if not self._history:
            return
        ok = QMessageBox.question(self, "确认", "确定要清空历史记录吗？") == QMessageBox.Yes
        if not ok:
            return
        self._history.clear()
        try:
            p = self._history_file_path()
            if p.exists():
                p.unlink()
        except Exception:
            pass
        self._refresh_history_table()

    def _archive_done_tasks(self) -> None:
        """
        将本轮运行成功完成的任务从“任务队列”移除，放入“历史任务”。
        为避免 worker 按索引运行时列表变化引起错乱，这里只在一次运行结束后执行。
        """
        if not self._archive_done_indices:
            return
        # 倒序删除，避免索引变化
        for idx in sorted(self._archive_done_indices, reverse=True):
            if 0 <= idx < len(self._tasks) and self._tasks[idx].status == "done":
                try:
                    self._tasks.pop(idx)
                except Exception:
                    pass
        self._archive_done_indices.clear()
        self._refresh_table()


