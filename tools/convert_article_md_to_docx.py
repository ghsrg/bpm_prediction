"""Convert the EOPKG article Markdown draft to DOCX via Pandoc.

Pandoc is used intentionally because it converts LaTeX math into native Word
OMML equations. python-docx cannot perform that conversion.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import zipfile
import xml.etree.ElementTree as ET
from pathlib import Path


DEFAULT_INPUT = "\u0414\u0440\u0430\u0444\u0442_\u0441\u0442\u0430\u0442\u0442\u0456_4_EOPKG_\u0441\u0442\u0440\u0443\u043a\u0442\u0443\u0440\u043d\u0438\u0439_\u0434\u0440\u0435\u0439\u0444.MD"
W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
WP_NS = "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing"
A_NS = "http://schemas.openxmlformats.org/drawingml/2006/main"
EMU_PER_INCH = 914400
ET.register_namespace("w", W_NS)
ET.register_namespace("wp", WP_NS)
ET.register_namespace("a", A_NS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert article Markdown to DOCX with native Word equations. "
            "Requires pandoc in PATH or --pandoc."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(DEFAULT_INPUT),
        help="Source Markdown article path.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Target DOCX path. Defaults to outputs/article_docx/<input-stem>.docx.",
    )
    parser.add_argument(
        "--pandoc",
        type=Path,
        default=None,
        help="Explicit pandoc executable path. If omitted, PATH is used.",
    )
    parser.add_argument(
        "--reference-doc",
        type=Path,
        default=None,
        help="Optional DOCX style template passed to pandoc --reference-doc.",
    )
    parser.add_argument(
        "--resource-path",
        type=Path,
        action="append",
        default=[],
        help=(
            "Additional resource path for images. Can be passed multiple times. "
            "The input file directory is always included."
        ),
    )
    parser.add_argument(
        "--keep-images",
        action="store_true",
        help=(
            "Keep Markdown image references. By default images are replaced with "
            "text placeholders so SVG/figure issues do not block DOCX export."
        ),
    )
    parser.add_argument(
        "--toc",
        action="store_true",
        help="Ask pandoc to generate a Word table of contents.",
    )
    parser.add_argument(
        "--number-sections",
        action="store_true",
        help="Ask pandoc to number headings. Disabled by default for article drafts.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the preprocessed Markdown file next to the DOCX.",
    )
    parser.add_argument(
        "--no-body-style",
        action="store_true",
        help=(
            "Do not patch DOCX body paragraph styles. By default Normal, "
            "Body Text, and First Paragraph are set to justified text, "
            "0 cm left/right indent, 1.27 cm first-line indent, 0 pt before/after, "
            "and 1.5 line spacing; table-cell paragraphs keep 0 first-line indent."
        ),
    )
    parser.add_argument(
        "--content-width-inches",
        type=float,
        default=6.5,
        help=(
            "Usable page width for full-width figures and tables, in inches. "
            "Default 6.5 matches Letter/A4-like pages with 1 inch margins."
        ),
    )
    return parser.parse_args()


def find_pandoc(explicit_path: Path | None) -> str:
    if explicit_path is not None:
        if explicit_path.exists():
            return str(explicit_path)
        raise FileNotFoundError(f"Pandoc executable was not found: {explicit_path}")

    pandoc = shutil.which("pandoc")
    if pandoc:
        return pandoc

    raise FileNotFoundError(
        "Pandoc was not found in PATH. Install Pandoc or pass --pandoc "
        "C:\\\\path\\\\to\\\\pandoc.exe. Pandoc is required for native Word equations."
    )


def replace_math_fences(markdown: str) -> str:
    """Convert ```math fenced blocks to display math blocks for Pandoc."""

    pattern = re.compile(r"```math\s*\n(.*?)\n```", re.DOTALL | re.IGNORECASE)

    def repl(match: re.Match[str]) -> str:
        body = match.group(1).strip()
        return f"\n$$\n{body}\n$$\n"

    return pattern.sub(repl, markdown)


def replace_images_with_placeholders(markdown: str) -> str:
    """Replace Markdown image references with lightweight placeholders."""

    image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    def repl(match: re.Match[str]) -> str:
        alt = match.group(1).strip() or "figure"
        target = match.group(2).strip()
        return f"**[Figure placeholder: {alt}; source: {target}]**"

    return image_pattern.sub(repl, markdown)


def preserve_display_equation_tags(markdown: str) -> str:
    """Keep equation numbers visible after Pandoc converts math to Word OMML.

    Pandoc's DOCX writer can drop LaTeX ``\tag{n}`` while converting display
    math to native Word equations. Rewriting the tag as visible math text keeps
    the number in the equation object. This is intentionally limited to display
    math blocks so inline formulas are not affected.
    """

    display_pattern = re.compile(r"(?<!\\)\$\$(.*?)(?<!\\)\$\$", re.DOTALL)
    tag_pattern = re.compile(r"\\tag\{([^{}]+)\}")

    def repl(match: re.Match[str]) -> str:
        body = match.group(1)
        tag_match = tag_pattern.search(body)
        if tag_match is None:
            return match.group(0)

        tag = tag_match.group(1).strip()
        body_without_tag = tag_pattern.sub("", body).rstrip()
        visible_tag = rf"\qquad\text{{({tag})}}"
        return f"$${body_without_tag} {visible_tag}\n$$"

    return display_pattern.sub(repl, markdown)


def normalize_markdown(markdown: str, keep_images: bool) -> str:
    markdown = markdown.replace("\ufeff", "")
    markdown = replace_math_fences(markdown)
    markdown = preserve_display_equation_tags(markdown)
    if not keep_images:
        markdown = replace_images_with_placeholders(markdown)
    return markdown


def count_unescaped(text: str, token: str) -> int:
    return len(re.findall(rf"(?<!\\){re.escape(token)}", text))


def collect_math_warnings(markdown: str) -> list[str]:
    warnings: list[str] = []
    display_count = count_unescaped(markdown, "$$")
    if display_count % 2:
        warnings.append("Odd number of display math delimiters ($$).")

    without_display = re.sub(r"(?<!\\)\$\$.*?(?<!\\)\$\$", "", markdown, flags=re.DOTALL)
    inline_count = count_unescaped(without_display, "$")
    if inline_count % 2:
        warnings.append("Odd number of inline math delimiters ($).")

    return warnings


def build_pandoc_command(
    pandoc: str,
    source_md: Path,
    output_docx: Path,
    input_dir: Path,
    resource_paths: list[Path],
    reference_doc: Path | None,
    toc: bool,
    number_sections: bool,
) -> list[str]:
    resource_path = [str(input_dir.resolve())]
    resource_path.extend(str(path.resolve()) for path in resource_paths)

    command = [
        pandoc,
        str(source_md),
        "--from",
        "markdown+tex_math_dollars+tex_math_single_backslash+pipe_tables+raw_tex",
        "--to",
        "docx",
        "--standalone",
        "--wrap=none",
        "--resource-path",
        ";".join(resource_path),
        "--metadata",
        "lang=uk-UA",
        "--output",
        str(output_docx),
    ]

    if reference_doc is not None:
        command.extend(["--reference-doc", str(reference_doc)])
    if toc:
        command.append("--toc")
    if number_sections:
        command.append("--number-sections")

    return command


def qn(local_name: str) -> str:
    return f"{{{W_NS}}}{local_name}"


def qn_ns(namespace: str, local_name: str) -> str:
    return f"{{{namespace}}}{local_name}"


def get_or_create(parent: ET.Element, local_name: str) -> ET.Element:
    child = parent.find(qn(local_name))
    if child is None:
        child = ET.SubElement(parent, qn(local_name))
    return child


def remove_children(parent: ET.Element, local_name: str) -> None:
    for child in list(parent.findall(qn(local_name))):
        parent.remove(child)


def set_table_border(
    borders: ET.Element,
    border_name: str,
    *,
    value: str,
    size: str = "4",
    color: str = "auto",
) -> None:
    border = get_or_create(borders, border_name)
    border.set(qn("val"), value)
    border.set(qn("sz"), size)
    border.set(qn("space"), "0")
    border.set(qn("color"), color)


def patch_style_paragraph_properties(style: ET.Element) -> None:
    ppr = get_or_create(style, "pPr")

    jc = get_or_create(ppr, "jc")
    jc.set(qn("val"), "both")

    ind = get_or_create(ppr, "ind")
    ind.set(qn("left"), "0")
    ind.set(qn("right"), "0")
    ind.set(qn("firstLine"), "720")
    ind.attrib.pop(qn("hanging"), None)

    spacing = get_or_create(ppr, "spacing")
    spacing.set(qn("before"), "0")
    spacing.set(qn("after"), "0")
    spacing.set(qn("line"), "360")
    spacing.set(qn("lineRule"), "auto")


def apply_article_body_styles(docx_path: Path, content_width_inches: float) -> None:
    """Patch common Pandoc body paragraph styles in the generated DOCX."""

    style_ids = {"Normal", "BodyText", "FirstParagraph"}
    with zipfile.ZipFile(docx_path, "r") as source:
        entries = {name: source.read(name) for name in source.namelist()}

    styles_path = "word/styles.xml"
    document_path = "word/document.xml"
    if styles_path not in entries:
        raise RuntimeError(f"{styles_path} was not found in {docx_path}")
    if document_path not in entries:
        raise RuntimeError(f"{document_path} was not found in {docx_path}")

    root = ET.fromstring(entries[styles_path])
    patched = 0
    for style in root.findall(qn("style")):
        style_id = style.get(qn("styleId"))
        style_type = style.get(qn("type"))
        if style_type == "paragraph" and style_id in style_ids:
            patch_style_paragraph_properties(style)
            patched += 1

    if patched == 0:
        raise RuntimeError(
            "No common body paragraph styles were found in word/styles.xml "
            f"(expected one of {sorted(style_ids)})."
        )

    entries[styles_path] = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    entries[document_path] = ET.tostring(
        patch_document_layout(entries[document_path], content_width_inches),
        encoding="utf-8",
        xml_declaration=True,
    )

    with zipfile.ZipFile(docx_path, "w", compression=zipfile.ZIP_DEFLATED) as target:
        for name, content in entries.items():
            target.writestr(name, content)


def paragraph_text(paragraph: ET.Element) -> str:
    return "".join(text.text or "" for text in paragraph.findall(f".//{qn('t')}"))


def has_drawing(paragraph: ET.Element) -> bool:
    return paragraph.find(f".//{qn_ns(WP_NS, 'inline')}") is not None or paragraph.find(
        f".//{qn_ns(WP_NS, 'anchor')}"
    ) is not None


def clear_paragraph_indent(paragraph: ET.Element, align: str | None = None) -> None:
    ppr = get_or_create(paragraph, "pPr")
    ind = get_or_create(ppr, "ind")
    ind.set(qn("left"), "0")
    ind.set(qn("right"), "0")
    ind.set(qn("firstLine"), "0")
    ind.attrib.pop(qn("hanging"), None)
    if align is not None:
        jc = get_or_create(ppr, "jc")
        jc.set(qn("val"), align)


def scale_paragraph_drawings(paragraph: ET.Element, content_width_emu: int) -> None:
    drawings = paragraph.findall(f".//{qn_ns(WP_NS, 'inline')}")
    drawings.extend(paragraph.findall(f".//{qn_ns(WP_NS, 'anchor')}"))

    for drawing in drawings:
        extent = drawing.find(qn_ns(WP_NS, "extent"))
        if extent is None:
            continue
        old_cx = int(extent.get("cx", "0") or "0")
        old_cy = int(extent.get("cy", "0") or "0")
        if old_cx <= 0 or old_cy <= 0:
            continue

        new_cx = content_width_emu
        new_cy = max(1, round(old_cy * (new_cx / old_cx)))
        extent.set("cx", str(new_cx))
        extent.set("cy", str(new_cy))

        for graphic_extent in drawing.findall(f".//{qn_ns(A_NS, 'ext')}"):
            graphic_extent.set("cx", str(new_cx))
            graphic_extent.set("cy", str(new_cy))


def patch_document_layout(document_xml: bytes, content_width_inches: float) -> ET.Element:
    """Normalize table/figure layout for compact DOCX article drafts."""

    root = ET.fromstring(document_xml)
    content_width_emu = round(content_width_inches * EMU_PER_INCH)

    for table in root.findall(f".//{qn('tbl')}"):
        tbl_pr = get_or_create(table, "tblPr")
        tbl_w = get_or_create(tbl_pr, "tblW")
        tbl_w.set(qn("type"), "pct")
        tbl_w.set(qn("w"), "5000")

        tbl_ind = get_or_create(tbl_pr, "tblInd")
        tbl_ind.set(qn("type"), "dxa")
        tbl_ind.set(qn("w"), "0")
        remove_children(tbl_pr, "shd")

        borders = get_or_create(tbl_pr, "tblBorders")
        for border_name in ("top", "bottom", "insideH"):
            set_table_border(borders, border_name, value="single")
        for border_name in ("left", "right", "insideV"):
            set_table_border(borders, border_name, value="nil", size="0", color="auto")

    for table_cell in root.findall(f".//{qn('tc')}"):
        tc_pr = get_or_create(table_cell, "tcPr")
        remove_children(tc_pr, "shd")
        tc_borders = tc_pr.find(qn("tcBorders"))
        if tc_borders is not None:
            for border_name in ("left", "right", "insideV"):
                set_table_border(
                    tc_borders,
                    border_name,
                    value="nil",
                    size="0",
                    color="auto",
                )
        for paragraph in table_cell.findall(qn("p")):
            clear_paragraph_indent(paragraph)

    for paragraph in root.findall(f".//{qn('p')}"):
        text = paragraph_text(paragraph).strip()
        if text.startswith("Рис. ") or text.startswith("Таблиця "):
            clear_paragraph_indent(paragraph)
        if has_drawing(paragraph):
            clear_paragraph_indent(paragraph, align="center")
            scale_paragraph_drawings(paragraph, content_width_emu)

    return root


def main() -> int:
    args = parse_args()

    input_path = args.input.resolve()
    if not input_path.exists():
        print(f"Input Markdown file was not found: {input_path}", file=sys.stderr)
        return 2

    output_path = args.output
    if output_path is None:
        output_path = Path("outputs") / "article_docx" / f"{input_path.stem}.docx"
    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_md = output_path.with_suffix(".pandoc.md")

    try:
        pandoc = find_pandoc(args.pandoc)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    markdown = input_path.read_text(encoding="utf-8-sig")
    prepared = normalize_markdown(markdown, keep_images=args.keep_images)
    temp_md.write_text(prepared, encoding="utf-8", newline="\n")

    for warning in collect_math_warnings(prepared):
        print(f"WARNING: {warning}", file=sys.stderr)

    command = build_pandoc_command(
        pandoc=pandoc,
        source_md=temp_md,
        output_docx=output_path,
        input_dir=input_path.parent,
        resource_paths=args.resource_path,
        reference_doc=args.reference_doc,
        toc=args.toc,
        number_sections=args.number_sections,
    )

    print("Running:", " ".join(command))
    completed = subprocess.run(command, check=False)
    if completed.returncode != 0:
        print(f"Pandoc failed with exit code {completed.returncode}", file=sys.stderr)
        print(f"Preprocessed Markdown kept for inspection: {temp_md}", file=sys.stderr)
        return completed.returncode

    if not args.no_body_style:
        apply_article_body_styles(output_path, args.content_width_inches)

    if not args.keep_temp:
        temp_md.unlink(missing_ok=True)

    print(f"DOCX written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
