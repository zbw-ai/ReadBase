from pathlib import Path


SOURCE = Path(
    "/private/tmp/ql-resume-preview-v10/"
    "曾柏炜-大模型训练推理Infra高级工程师-原格式版-2026.docx.qlpreview/"
    "Preview.html"
)
OUTPUT = Path("/Users/zengbw/ReadBase/private_resume/tmp/resume-print.html")


def build() -> None:
    html = SOURCE.read_text(encoding="utf-8")
    print_css = """
<style id="codex-print-fixes">
@page { size: 595pt 842pt; margin: 0; }
html, body { margin: 0; padding: 0; width: 595px; background: white; }
body { zoom: 1.3333333333; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
.s1.s1.s1 { overflow: visible; min-height: 842px; }
.tx-page-break {
  break-before: page;
  page-break-before: always;
  margin-top: 8px !important;
}
</style>
"""
    html = html.replace("</head>", print_css + "</head>", 1)
    marker = '<p class="s14"><span class="s12">TX项目训练业务</span>'
    replacement = '<p class="s14 tx-page-break"><span class="s12">TX项目训练业务</span>'
    if marker not in html:
        raise RuntimeError("TX project marker not found in Quick Look preview")
    html = html.replace(marker, replacement, 1)
    html = html.replace('src="Attachment1.png"', 'src="file://' + str(SOURCE.parent / "Attachment1.png") + '"')
    OUTPUT.write_text(html, encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    build()
