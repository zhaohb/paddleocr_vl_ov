from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QTextBrowser, QWidget, QVBoxLayout

from .i18n import Lang, normalize_lang, t


class MarkdownPreviewWidget(QWidget):
    """
    Markdown 富文本预览组件：
    - 优先使用 QWebEngineView（支持 KaTeX/mermaid/代码高亮等 JS 渲染）
    - 若未安装 PySide6-WebEngine，则降级到 QTextBrowser（基础 Markdown）

    说明：
    - WebEngine 方案为了减少 repo 体积，默认使用 CDN 加载前端库（需要联网）。
      你本身模型自动下载也需要联网，因此一般可接受。
    """

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._base_dir: Optional[Path] = None
        self._engine = "text"  # text|web
        self._lang: Lang = "zh_CN"

        self._text = QTextBrowser()
        self._text.setOpenExternalLinks(True)
        self._text.setStyleSheet("background:#FFFFFF;border-radius:10px;border:1px solid #E2E6EA;")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._text)

        self._web = None
        try:
            from PySide6.QtWebEngineWidgets import QWebEngineView  # type: ignore

            self._web = QWebEngineView()
            layout.removeWidget(self._text)
            self._text.setParent(None)
            layout.addWidget(self._web)
            self._engine = "web"
        except Exception:
            # 降级：仍然可用，只是不支持 LaTeX/mermaid 等 JS 能力
            self._engine = "text"

    def set_lang(self, lang: str) -> None:
        self._lang = normalize_lang(lang)

    def set_markdown(self, md: str, base_dir: Optional[Path] = None) -> None:
        self._base_dir = base_dir
        if self._engine == "web" and self._web is not None:
            self._set_markdown_web(md, base_dir)
        else:
            # QTextBrowser 无法执行 JS，尽量做基础显示
            self._text.setMarkdown(md)

    def clear(self) -> None:
        if self._engine == "web" and self._web is not None:
            self._web.setHtml("<html><body></body></html>")
        else:
            self._text.setHtml("")

    def _set_markdown_web(self, md: str, base_dir: Optional[Path]) -> None:
        """
        使用 markdown-it + KaTeX + mermaid + highlight.js 在浏览器侧渲染。
        """
        # 通过 JSON 注入 md，避免脚本注入问题
        md_json = json.dumps(md)
        # 大文档性能保护：结果过大时，禁用部分重渲染能力，优先保证滚动流畅
        perf_mode = len(md) >= 200_000
        missing_md_it_msg = json.dumps(t("md.err.markdownit_missing", self._lang))
        perf_hint_msg = json.dumps(t("md.hint.perf_mode", self._lang))

        def _app_root() -> Path:
            # 打包后：优先从 PyInstaller 的 _MEIPASS 取资源目录
            meipass = getattr(sys, "_MEIPASS", None)
            if meipass:
                return Path(meipass)
            # 源码运行：.../client_app/paddleocr_vl_client/markdown_preview.py -> parents[1]=client_app
            return Path(__file__).resolve().parents[1]

        def _file_url(p: Path) -> str:
            return QUrl.fromLocalFile(str(p)).toString()

        assets_dir = _app_root() / "assets"
        # baseUrl：用于加载 imgs/xxx.jpg 等相对资源（来自输出目录）
        base_url = QUrl.fromLocalFile(str(base_dir.resolve()) + "/") if base_dir else QUrl()

        # 优先用本地 assets（解决 CDN 失败导致 window.markdownit 不存在的问题）
        md_it_js = _file_url(assets_dir / "markdown-it.min.js")
        katex_js = _file_url(assets_dir / "katex.min.js")
        katex_css = _file_url(assets_dir / "katex.min.css")
        katex_autorender_js = _file_url(assets_dir / "katex-auto-render.min.js")
        hl_js = _file_url(assets_dir / "highlight.min.js")
        hl_css = _file_url(assets_dir / "highlight-github.min.css")
        mermaid_js = _file_url(assets_dir / "mermaid.min.js")

        html_doc = f"""
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Arial, "PingFang SC", "Microsoft YaHei", sans-serif;
      padding: 16px;
      color: #111;
      background: #fff;
    }}
    /* 性能优化：对长页面减少不可见区域的布局/绘制成本（Chromium 支持） */
    img, pre, table, blockquote {{
      content-visibility: auto;
      contain-intrinsic-size: 900px;
    }}
    pre code {{
      display: block;
      padding: 12px;
      border-radius: 10px;
      overflow-x: auto;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      margin: 10px 0;
    }}
    th, td {{
      border: 1px solid #e2e6ea;
      padding: 8px;
      text-align: left;
    }}
    img {{
      max-width: 100%;
      height: auto;
    }}
    blockquote {{
      border-left: 4px solid #1F6FEB;
      margin: 8px 0;
      padding: 4px 12px;
      background: #F6F7F9;
      border-radius: 8px;
    }}
    .perf-hint {{
      position: sticky;
      top: 0;
      z-index: 10;
      margin: 0 0 12px 0;
      padding: 10px 12px;
      border-radius: 10px;
      border: 1px solid #E2E6EA;
      background: #F6F7F9;
      color: #111;
      font-size: 13px;
    }}
  </style>

  <!-- highlight.js -->
  <link rel="stylesheet" href="{hl_css}">
  <script src="{hl_js}"></script>

  <!-- KaTeX -->
  <link rel="stylesheet" href="{katex_css}">
  <script src="{katex_js}"></script>
  <script src="{katex_autorender_js}"></script>

  <!-- markdown-it -->
  <script src="{md_it_js}"></script>

  <!-- Mermaid (```mermaid) -->
  <script src="{mermaid_js}"></script>
</head>
<body>
  <div id="app"></div>
  <script>
    const mdText = {md_json};
    const perfMode = {str(perf_mode).lower()};

    if (typeof window.markdownit !== 'function') {{
      const container = document.getElementById('app');
      container.innerHTML = '<div style=\"color:#B42318;background:#FEE4E2;padding:10px;border-radius:10px;\">' +
        {missing_md_it_msg} +
        '</div><pre style=\"white-space:pre-wrap;\">' + mdText.replace(/[<>&]/g, c => ({{'<':'&lt;','>':'&gt;','&':'&amp;'}}[c])) + '</pre>';
      throw new Error('markdown-it not loaded');
    }}

    const md = window.markdownit({{
      html: true,
      linkify: true,
      typographer: true,
      highlight: function (str, lang) {{
        if (perfMode) {{
          // 性能模式：禁用语法高亮，减少 DOM 复杂度
          return '<pre><code class=\"hljs\">' + md.utils.escapeHtml(str) + '</code></pre>';
        }}
        try {{
          if (lang && hljs.getLanguage(lang)) {{
            return '<pre><code class=\"hljs\">' + hljs.highlight(str, {{language: lang}}).value + '</code></pre>';
          }}
        }} catch (e) {{}}
        return '<pre><code class=\"hljs\">' + md.utils.escapeHtml(str) + '</code></pre>';
      }}
    }});

    let htmlOut = md.render(mdText);

    const container = document.getElementById('app');
    if (perfMode) {{
      const hint = document.createElement('div');
      hint.className = 'perf-hint';
      hint.textContent = {perf_hint_msg};
      container.appendChild(hint);
    }}
    const content = document.createElement('div');
    content.innerHTML = htmlOut;
    container.appendChild(content);

    // 图片懒加载 + 异步解码（对含大量截图的结果滚动提升明显）
    const imgs = container.querySelectorAll('img');
    for (const img of imgs) {{
      try {{
        img.loading = 'lazy';
        img.decoding = 'async';
      }} catch (e) {{}}
    }}

    // KaTeX auto-render for $...$ / $$...$$ / \\(...\\) / \\[...\\]
    if (!perfMode) {{
      try {{
        if (typeof renderMathInElement === 'function') {{
          renderMathInElement(container, {{
            delimiters: [
              {{ left: '$$', right: '$$', display: true }},
              {{ left: '$', right: '$', display: false }},
              {{ left: '\\\\(', right: '\\\\)', display: false }},
              {{ left: '\\\\[', right: '\\\\]', display: true }},
            ],
            throwOnError: false,
          }});
        }}
      }} catch (e) {{}}
    }}

    // Convert mermaid code blocks
    if (!perfMode) {{
      const codes = container.querySelectorAll('code.language-mermaid');
      for (const code of codes) {{
        const pre = code.parentElement;
        const div = document.createElement('div');
        div.className = 'mermaid';
        div.textContent = code.textContent;
        pre.replaceWith(div);
      }}

      try {{
        if (window.mermaid) {{
          mermaid.initialize({{ startOnLoad: false }});
          mermaid.run({{ querySelector: '.mermaid' }});
        }}
      }} catch (e) {{}}
    }}
  </script>
</body>
</html>
        """.strip()

        self._web.setHtml(html_doc, baseUrl=base_url)


