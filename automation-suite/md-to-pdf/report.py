from __future__ import annotations

"""Core logic: turn a Markdown report into a branded, print-ready document.

The valuable part is the **report template** - markdown becomes a styled HTML
document with a cover header, consistent typography, table styling, and proper
print/page rules (A4, margins, page numbers). From there:

  - If WeasyPrint's native libraries are available -> a real PDF.
  - If not (e.g. a Mac without pango/cairo) -> we still write the styled HTML, so
    the tool never hard-fails. CI and Colab install the libs and get PDFs.

Pure functions, no CLI - reused by the notebook and mountable as a "Reports" app
on the platform shell.
"""

from datetime import date
from pathlib import Path
from typing import Dict, Tuple

import markdown

REPORT_CSS = """
@page { size: A4; margin: 22mm 18mm 20mm 18mm;
  @bottom-center { content: counter(page) " / " counter(pages);
    font-family: sans-serif; font-size: 9px; color: #999; } }
* { box-sizing: border-box; }
body { font-family: 'Helvetica Neue', Arial, sans-serif; color: #1a1a2e; line-height: 1.55;
  font-size: 11.5px; margin: 0; }
.cover { border-bottom: 3px solid #3d34d6; padding-bottom: 14px; margin-bottom: 22px; }
.cover .kicker { color: #3d34d6; font-weight: 700; letter-spacing: .12em; text-transform: uppercase;
  font-size: 9px; }
.cover h1 { font-size: 26px; margin: 6px 0 4px; letter-spacing: -.01em; }
.cover .sub { color: #6b6b80; font-size: 12px; }
.cover .meta { color: #9a9aae; font-size: 9.5px; margin-top: 8px; }
h2 { font-size: 16px; margin: 20px 0 8px; border-bottom: 1px solid #e7e7f0; padding-bottom: 4px; }
h3 { font-size: 13px; margin: 14px 0 5px; color: #34324f; }
p, li { font-size: 11.5px; }
a { color: #3d34d6; text-decoration: none; }
code { background: #f3f2fc; color: #3d34d6; padding: 1px 4px; border-radius: 3px; font-size: 10.5px; }
pre { background: #1a1830; color: #e8e6ff; padding: 10px 12px; border-radius: 6px; overflow: auto;
  font-size: 10px; line-height: 1.45; }
pre code { background: none; color: inherit; padding: 0; }
table { border-collapse: collapse; width: 100%; margin: 10px 0; font-size: 10.5px; }
th, td { border: 1px solid #e0e0ec; padding: 5px 8px; text-align: left; vertical-align: top; }
th { background: #f3f2fc; color: #3d34d6; }
tr:nth-child(even) td { background: #fafaff; }
blockquote { margin: 10px 0; padding: 6px 12px; background: #f5f5ff; border-left: 3px solid #6c63ff;
  color: #34324f; }
hr { border: 0; border-top: 1px solid #e7e7f0; margin: 16px 0; }
"""


def md_to_report_html(md_text: str, *, title: str = "Report",
                      subtitle: str = "", author: str = "") -> str:
    """Render markdown into a full, styled, print-ready HTML document."""
    body = markdown.markdown(md_text, extensions=["tables", "fenced_code", "sane_lists", "toc"])
    meta_bits = [b for b in (author, date.today().isoformat()) if b]
    meta = " · ".join(meta_bits)
    sub_html = f'<div class="sub">{subtitle}</div>' if subtitle else ""
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>{title}</title><style>{REPORT_CSS}</style></head><body>
<div class="cover"><div class="kicker">Report</div><h1>{title}</h1>{sub_html}
<div class="meta">{meta}</div></div>
{body}
</body></html>"""


def html_to_pdf(html: str, out_path: str) -> None:
    """Write a PDF from HTML using WeasyPrint. Raises if native libs are missing."""
    try:
        from weasyprint import HTML
    except Exception as exc:  # noqa: BLE001 - import-time native lib failure
        raise RuntimeError(
            "WeasyPrint is unavailable (needs system libs pango/cairo). "
            "Install them, or use --html-only. Details: " + str(exc)
        ) from exc
    HTML(string=html).write_pdf(out_path)


def convert(md_path: str, out_path: str | None = None, *,
            title: str | None = None, subtitle: str = "", author: str = "",
            html_only: bool = False) -> Tuple[str, Dict[str, object]]:
    """Convert a Markdown file to a report. Returns (written_path, info).

    Falls back to writing styled HTML if PDF rendering isn't available, so the
    tool always produces an artifact.
    """
    src = Path(md_path)
    md_text = src.read_text()
    title = title or src.stem.replace("_", " ").replace("-", " ").title()
    html = md_to_report_html(md_text, title=title, subtitle=subtitle, author=author)

    out = Path(out_path) if out_path else src.with_suffix(".pdf")
    info: Dict[str, object] = {"title": title, "source": str(src)}

    if html_only or out.suffix.lower() != ".pdf":
        html_out = out.with_suffix(".html")
        html_out.write_text(html)
        info.update(engine="html", fallback=False)
        return str(html_out), info

    try:
        html_to_pdf(html, str(out))
        info.update(engine="weasyprint", fallback=False)
        return str(out), info
    except RuntimeError as exc:
        # Graceful fallback: write the HTML so the user still gets a deliverable.
        html_out = out.with_suffix(".html")
        html_out.write_text(html)
        info.update(engine="html", fallback=True, reason=str(exc))
        return str(html_out), info
