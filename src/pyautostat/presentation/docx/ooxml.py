"""Low-level OpenXML (OOXML) helpers for Word document formatting."""

from __future__ import annotations

from typing import Any


def add_page_number_fields(paragraph: Any) -> None:
    """Insert dynamic Page and TotalPages OpenXML field codes into a paragraph."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    # Preceding text
    paragraph.add_run("Page ")

    # Current page number field
    fld_page = OxmlElement("w:fldSimple")
    fld_page.set(qn("w:instr"), "PAGE")
    paragraph._p.append(fld_page)

    # Middle text
    paragraph.add_run(" of ")

    # Total pages field
    fld_numpages = OxmlElement("w:fldSimple")
    fld_numpages.set(qn("w:instr"), "NUMPAGES")
    paragraph._p.append(fld_numpages)


def set_table_header_row(row: Any) -> None:
    """Mark a table row as a repeating header row (tblHeader) across pages."""
    from docx.oxml import OxmlElement

    trPr = row._tr.get_or_add_trPr()
    if trPr.find(trPr.tag.split("}")[0] + "}tblHeader") is None:
        trPr.append(OxmlElement("w:tblHeader"))


def set_row_cant_split(row: Any) -> None:
    """Prevent a table row from breaking across page boundaries (cantSplit)."""
    from docx.oxml import OxmlElement

    trPr = row._tr.get_or_add_trPr()
    if trPr.find(trPr.tag.split("}")[0] + "}cantSplit") is None:
        trPr.append(OxmlElement("w:cantSplit"))


def set_cell_shading(cell: Any, color_hex: str) -> None:
    """Set background fill color on an individual table cell."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    clean_hex = color_hex.lstrip("#").upper()
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), clean_hex)
    tcPr.append(shd)


def set_table_borders(
    table: Any,
    color_hex: str = "CBD5E1",
    sz: str = "4",
    val: str = "single",
) -> None:
    """Apply professional muted borders to a Word table."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    clean_color = color_hex.lstrip("#").upper()
    tblPr = table._tbl.tblPr
    tblBorders = OxmlElement("w:tblBorders")
    for border_name in ("top", "left", "bottom", "right", "insideH"):
        border = OxmlElement(f"w:{border_name}")
        border.set(qn("w:val"), val)
        border.set(qn("w:sz"), sz)
        border.set(qn("w:space"), "0")
        border.set(qn("w:color"), clean_color)
        tblBorders.append(border)

    # Omit vertical inside borders for clean modern table design
    inside_v = OxmlElement("w:insideV")
    inside_v.set(qn("w:val"), "none")
    tblBorders.append(inside_v)

    tblPr.append(tblBorders)


def set_table_cell_margins(
    table: Any,
    top_dxa: int = 120,
    bottom_dxa: int = 120,
    left_dxa: int = 160,
    right_dxa: int = 160,
) -> None:
    """Set default inner padding margins for all cells in a table."""
    from docx.oxml import OxmlElement
    from docx.oxml.ns import qn

    tblPr = table._tbl.tblPr
    tblCellMar = OxmlElement("w:tblCellMar")
    for side, val in (
        ("top", top_dxa),
        ("bottom", bottom_dxa),
        ("left", left_dxa),
        ("right", right_dxa),
    ):
        node = OxmlElement(f"w:{side}")
        node.set(qn("w:w"), str(val))
        node.set(qn("w:type"), "dxa")
        tblCellMar.append(node)
    tblPr.append(tblCellMar)


def set_keep_with_next(paragraph: Any) -> None:
    """Ensure paragraph stays on the same page as the following content."""
    paragraph.paragraph_format.keep_with_next = True
