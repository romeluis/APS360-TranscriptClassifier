"""Render assembled HTML to PDF using WeasyPrint."""

import weasyprint


def render_pdf(html_string, output_path):
    """
    Render an assembled HTML string to a PDF file.

    Args:
        html_string: complete HTML document string
        output_path: file path for the output PDF
    """
    doc = weasyprint.HTML(string=html_string)
    doc.write_pdf(output_path)
