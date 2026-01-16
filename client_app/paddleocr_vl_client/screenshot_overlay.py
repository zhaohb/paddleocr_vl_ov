from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QPoint, QRect, Qt, Signal
from PySide6.QtGui import QColor, QCursor, QGuiApplication, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QApplication, QRubberBand, QWidget


@dataclass(frozen=True)
class ScreenshotResult:
    pixmap: QPixmap
    saved_path: Path


class ScreenshotOverlay(QWidget):
    """
    全屏透明覆盖层，用于鼠标拖拽选择区域截图。

    说明：
    - 为降低复杂度：只在“当前鼠标所在屏幕”上截图（多屏可再次截图）。
    - 采用 QScreen.grabWindow 截取选区。
    """

    captured = Signal(object)  # ScreenshotResult
    canceled = Signal()

    def __init__(self, save_dir: Path, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._save_dir = save_dir
        self._save_dir.mkdir(parents=True, exist_ok=True)

        self._origin: Optional[QPoint] = None
        self._current: Optional[QPoint] = None
        self._bg: Optional[QPixmap] = None

        self._rubber = QRubberBand(QRubberBand.Rectangle, self)
        self._rubber.hide()

        self.setWindowFlags(
            Qt.WindowStaysOnTopHint
            | Qt.FramelessWindowHint
            | Qt.Tool
        )
        self.setWindowState(Qt.WindowFullScreen)
        # 不依赖透明窗口合成（Windows 上有时会表现为整屏发黑）
        self.setAttribute(Qt.WA_TranslucentBackground, False)
        self.setCursor(Qt.CrossCursor)
        self.setMouseTracking(True)

    def start(self) -> None:
        # 覆盖当前鼠标所在的屏幕（更符合用户预期）
        # PySide6 下 QGuiApplication.cursor() 不存在，使用 QCursor.pos()
        pos = QCursor.pos()
        screen = QGuiApplication.screenAt(pos) or QGuiApplication.primaryScreen()
        if screen is None:  # pragma: no cover
            self.canceled.emit()
            return
        self.setGeometry(screen.geometry())
        # 关键：先抓取一张“冻结屏幕”作为背景，然后再画遮罩与选区
        self._bg = screen.grabWindow(0)
        self.show()
        self.raise_()
        self.activateWindow()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, True)
        # 先画背景（冻结屏幕）
        if self._bg is not None and not self._bg.isNull():
            painter.drawPixmap(self.rect(), self._bg)
        else:
            painter.fillRect(self.rect(), QColor(0, 0, 0, 255))

        # 再画遮罩：只遮住选区之外的区域，选区内保持原图清晰
        if self._origin and self._current:
            r = QRect(self._origin, self._current).normalized()
            shade = QColor(0, 0, 0, 120)
            # top
            painter.fillRect(QRect(0, 0, self.width(), r.top()), shade)
            # bottom (bottom() 是包含边界的坐标，+1 避免遮罩覆盖到选区边界)
            painter.fillRect(QRect(0, r.bottom() + 1, self.width(), self.height() - (r.bottom() + 1)), shade)
            # left
            painter.fillRect(QRect(0, r.top(), r.left(), r.height()), shade)
            # right
            painter.fillRect(QRect(r.right() + 1, r.top(), self.width() - (r.right() + 1), r.height()), shade)

            # 画选区边框（rubber band 之外再描一层，增强对比）
            pen = QPen(QColor("#1F6FEB"))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.drawRect(r)
        else:
            # 未开始选择时，轻微遮罩整个屏幕（不会全黑）
            painter.fillRect(self.rect(), QColor(0, 0, 0, 80))

    def keyPressEvent(self, event):  # noqa: N802
        if event.key() == Qt.Key_Escape:
            self._rubber.hide()
            self.hide()
            self.canceled.emit()
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event):  # noqa: N802
        if event.button() == Qt.LeftButton:
            self._origin = event.pos()
            self._current = event.pos()
            self._rubber.setGeometry(QRect(self._origin, self._current))
            self._rubber.show()
            self.update()

    def mouseMoveEvent(self, event):  # noqa: N802
        if self._origin is None:
            return
        self._current = event.pos()
        self._rubber.setGeometry(QRect(self._origin, self._current).normalized())
        self.update()

    def mouseReleaseEvent(self, event):  # noqa: N802
        if event.button() != Qt.LeftButton or self._origin is None or self._current is None:
            return

        r = QRect(self._origin, self._current).normalized()
        self._rubber.hide()
        self.hide()

        # 过小选区视为取消
        if r.width() < 5 or r.height() < 5:
            self.canceled.emit()
            return

        # grabWindow 使用“屏幕坐标”，因此要把 widget 内坐标转换为全局坐标
        top_left_global = self.mapToGlobal(r.topLeft())
        screen = QGuiApplication.screenAt(top_left_global) or QGuiApplication.primaryScreen()
        if screen is None:  # pragma: no cover
            self.canceled.emit()
            return

        x = top_left_global.x()
        y = top_left_global.y()
        w = r.width()
        h = r.height()

        pix = screen.grabWindow(0, x, y, w, h)
        # 保存为临时 PNG
        ts = int(time.time() * 1000)
        out = self._save_dir / f"screenshot_{ts}.png"
        pix.save(str(out), "PNG")

        self.captured.emit(ScreenshotResult(pixmap=pix, saved_path=out))


