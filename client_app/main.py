import os
import sys
from pathlib import Path


def _ensure_import_path() -> None:
    """
    让 `client_app/` 作为子目录运行时，也能 import 到同级的 `paddleocr_vl_openvino` 包。
    """
    this_file = Path(__file__).resolve()
    project_root = this_file.parent.parent  # .../paddleocr_vl_ov
    sys.path.insert(0, str(project_root))


APP_DISPLAY_NAME = "PaddleOCR-VL OpenVINO Client"
# Windows taskbar / notifications grouping id
APP_USER_MODEL_ID = "hongbo.paddleocr_vl_openvino.client"


def _set_windows_app_user_model_id(app_id: str) -> None:
    if not sys.platform.startswith("win"):
        return
    try:
        import ctypes  # noqa: WPS433 (standard library)

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)  # type: ignore[attr-defined]
    except Exception:
        # Not fatal; app still runs.
        return


def _make_app_icon():
    """
    生成一个简单的内置图标（无需额外文件），用于窗口左上角与任务栏图标。
    """
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPen, QPixmap

    size = 256
    pix = QPixmap(size, size)
    pix.fill(Qt.transparent)

    p = QPainter(pix)
    p.setRenderHint(QPainter.Antialiasing, True)

    # Background
    bg = QColor("#1F6FEB")
    p.setBrush(bg)
    p.setPen(Qt.NoPen)
    p.drawRoundedRect(0, 0, size, size, 56, 56)

    # Border
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(QColor(255, 255, 255, 60), 6))
    p.drawRoundedRect(8, 8, size - 16, size - 16, 52, 52)

    # Text: "OV"
    p.setPen(QPen(QColor("#FFFFFF"), 1))
    font = QFont()
    font.setBold(True)
    font.setPointSize(92)
    p.setFont(font)
    p.drawText(pix.rect(), Qt.AlignCenter, "OV")
    p.end()

    return QIcon(pix)


def main() -> int:
    _ensure_import_path()

    # 避免 Qt / OpenVINO 相关输出被吞（并且让日志能在 UI 中显示）
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")

    from PySide6.QtGui import QColor, QPalette
    from PySide6.QtWidgets import QApplication
    from paddleocr_vl_client.main_window import MainWindow

    _set_windows_app_user_model_id(APP_USER_MODEL_ID)

    app = QApplication(sys.argv)
    app.setApplicationName(APP_DISPLAY_NAME)
    app.setApplicationDisplayName(APP_DISPLAY_NAME)
    app.setOrganizationName("hongbo")
    icon = _make_app_icon()
    app.setWindowIcon(icon)

    # 关键：在 Windows 暗色主题下，Qt 可能给出“浅色字体”的 Palette，
    # 但我们的页面是浅色背景，导致文字/表头/Tab 看不清。这里强制使用浅色 Palette。
    app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window, QColor("#F6F7F9"))
    pal.setColor(QPalette.WindowText, QColor("#111111"))
    pal.setColor(QPalette.Base, QColor("#FFFFFF"))
    pal.setColor(QPalette.AlternateBase, QColor("#F2F4F7"))
    pal.setColor(QPalette.Text, QColor("#111111"))
    pal.setColor(QPalette.Button, QColor("#FFFFFF"))
    pal.setColor(QPalette.ButtonText, QColor("#111111"))
    pal.setColor(QPalette.Highlight, QColor("#1F6FEB"))
    pal.setColor(QPalette.HighlightedText, QColor("#FFFFFF"))
    app.setPalette(pal)

    win = MainWindow()
    win.setWindowTitle(APP_DISPLAY_NAME)
    win.setWindowIcon(icon)
    win.show()
    # 用于自动化/打包前的快速启动检查：不进入事件循环，直接退出
    if "--smoke-test" in sys.argv:
        win.close()
        return 0
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())


