"""Tests for DOCX figure embedding, captions, placement, zero recalculation, and privacy."""

from __future__ import annotations

import io
import xml.etree.ElementTree as ET
import zipfile
from unittest.mock import patch

import docx
import pandas as pd
from method_factories import (
    make_chi_square_workflow,
    make_linear_regression_workflow,
    make_one_way_anova_workflow,
    make_two_way_anova_workflow,
    make_welch_t_workflow,
)

from pyautostat import (
    AnalysisOptions,
    ResearchAssistant,
    to_docx,
)


def test_docx_default_does_not_embed_figures_or_require_chrome():
    wf = make_welch_t_workflow()

    with patch("pyautostat.presentation.figures.static.check_chrome_available") as mock_chrome:
        docx_bytes = to_docx(wf, include_figures=False)
        mock_chrome.assert_not_called()

        with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
            media_files = [n for n in zf.namelist() if n.startswith("word/media/")]
            assert len(media_files) == 0


def test_docx_no_figure_target_with_include_figures_true_does_not_require_chrome():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    profile = ResearchAssistant(df).profile()

    with patch("pyautostat.presentation.figures.static.check_chrome_available") as mock_chrome:
        docx_bytes = to_docx(profile, include_figures=True)
        mock_chrome.assert_not_called()

        assert docx_bytes.startswith(b"PK")
        with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
            media_files = [n for n in zf.namelist() if n.startswith("word/media/")]
            assert len(media_files) == 0


def test_docx_welch_t_embeds_png_figure_with_caption():
    wf = make_welch_t_workflow()
    docx_bytes = to_docx(wf, include_figures=True)
    assert docx_bytes.startswith(b"PK")

    # Inspect ZIP structure
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
        media_files = [n for n in zf.namelist() if n.startswith("word/media/image")]
        assert len(media_files) >= 1

        img_data = zf.read(media_files[0])
        assert img_data.startswith(b"\x89PNG\r\n\x1a\n")

    # Reopen document via python-docx
    doc = docx.Document(io.BytesIO(docx_bytes))
    captions = [p.text for p in doc.paragraphs if p.text.startswith("Figure ")]
    assert len(captions) >= 1
    assert "Figure 1." in captions[0]

    # Verify editable tables and paragraphs exist
    assert len(doc.tables) >= 1
    assert len(doc.paragraphs) > 5


def test_docx_linear_regression_multiple_figures_and_placement():
    wf = make_linear_regression_workflow()
    docx_bytes = to_docx(wf, include_figures=True)

    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
        media_files = [n for n in zf.namelist() if n.startswith("word/media/image")]
        assert len(media_files) >= 1

    doc = docx.Document(io.BytesIO(docx_bytes))
    captions = [p.text for p in doc.paragraphs if p.text.startswith("Figure ")]
    assert len(captions) >= 1
    for idx, cap in enumerate(captions, start=1):
        assert cap.startswith(f"Figure {idx}.")


def test_docx_chi_square_heatmap_embedding():
    wf = make_chi_square_workflow()
    docx_bytes = to_docx(wf, include_figures=True)

    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
        media_files = [n for n in zf.namelist() if n.startswith("word/media/image")]
        assert len(media_files) >= 1

    doc = docx.Document(io.BytesIO(docx_bytes))
    captions = [
        p.text for p in doc.paragraphs if "Contingency" in p.text or "contingency" in p.text.lower()
    ]
    assert len(captions) >= 1


def test_research_report_to_docx_include_figures():
    wf = make_welch_t_workflow()
    assert wf.report is not None

    docx_bytes = wf.report.to_docx(include_figures=True)
    assert docx_bytes.startswith(b"PK")

    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
        media_files = [n for n in zf.namelist() if n.startswith("word/media/image")]
        assert len(media_files) >= 1

    doc = docx.Document(io.BytesIO(docx_bytes))
    captions = [p.text for p in doc.paragraphs if p.text.startswith("Figure ")]
    assert len(captions) >= 1


def test_docx_figures_zero_recalculation():
    wf = make_welch_t_workflow()

    def exploding_statistic(*args, **kwargs):
        raise RuntimeError("Illegal recalculation during DOCX figure export!")

    with (
        patch("scipy.stats.ttest_ind", side_effect=exploding_statistic),
        patch("scipy.stats.ttest_1samp", side_effect=exploding_statistic),
        patch("pyautostat.execution.execute_specification", side_effect=exploding_statistic),
        patch("pyautostat.execution.execute_selected_method", side_effect=exploding_statistic),
    ):
        docx_bytes = to_docx(wf, include_figures=True)
        assert docx_bytes.startswith(b"PK")


def test_docx_figures_privacy_sentinels_not_leaked():
    sentinel_id = "CONFIDENTIAL_DOCX_ID_XYZ_8888"
    df = pd.DataFrame(
        {
            "group": ["A", "A", "A", "A", "A", "B", "B", "B", "B", "B"],
            "score": [10.0, 11.0, 12.0, 13.0, 14.0, 20.0, 21.0, 22.0, 23.0, 24.0],
            "secret_notes": [f"{sentinel_id}_{i}" for i in range(10)],
        }
    )

    assistant = ResearchAssistant(df)
    wf = assistant.run(
        objective="compare_groups",
        outcome="score",
        predictor="group",
        estimand="mean",
        design="independent",
        variable_types={"score": "continuous", "group": "nominal"},
        options=AnalysisOptions(confidence_level=0.95),
    )

    docx_bytes = to_docx(wf, include_figures=True)
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
        doc_xml = zf.read("word/document.xml").decode("utf-8")
        assert sentinel_id not in doc_xml


def _extract_body_elements(docx_bytes: bytes) -> list[tuple[str, str, bool]]:
    """Return ordered list of (tag, text, has_drawing) for w:body children."""
    with zipfile.ZipFile(io.BytesIO(docx_bytes)) as zf:
        doc_xml = zf.read("word/document.xml").decode("utf-8")
    root = ET.fromstring(doc_xml)
    body = root.find("{http://schemas.openxmlformats.org/wordprocessingml/2006/main}body")
    assert body is not None
    elements = []
    for child in body:
        tag = child.tag.split("}")[-1]
        if tag == "sectPr":
            continue
        text = "".join(child.itertext()).strip()
        has_drawing = any(elem.tag.endswith("drawing") for elem in child.iter())
        elements.append((tag, text, has_drawing))
    return elements


def test_docx_welch_t_structural_placement_ordering():
    """Verify Welch t figure appears after Key Results and before next major section."""
    wf = make_welch_t_workflow()
    docx_bytes = to_docx(wf, include_figures=True)
    elements = _extract_body_elements(docx_bytes)

    key_idx = next(i for i, (tag, text, _) in enumerate(elements) if "Key Results" in text)
    caption_idx = next(
        i for i, (tag, text, _) in enumerate(elements) if text.startswith("Figure 1.")
    )
    drawing_idx = next(i for i, (tag, _, has_draw) in enumerate(elements) if has_draw)

    # Next section heading
    next_sec_idx = next(
        i
        for i, (tag, text, _) in enumerate(elements)
        if i > key_idx and ("Research Design" in text or "Diagnostics" in text)
    )

    assert key_idx < drawing_idx <= caption_idx < next_sec_idx


def test_docx_linear_regression_structural_placement_ordering():
    """Verify OLS coefficient forest appears after coefficients and not at document end."""
    wf = make_linear_regression_workflow()
    docx_bytes = to_docx(wf, include_figures=True)
    elements = _extract_body_elements(docx_bytes)

    coef_tbl_idx = next(
        i for i, (tag, text, _) in enumerate(elements) if tag == "tbl" and "Estimate" in text
    )
    caption_idx = next(
        i for i, (tag, text, _) in enumerate(elements) if text.startswith("Figure 1.")
    )
    next_sec_idx = next(
        i
        for i, (tag, text, _) in enumerate(elements)
        if i > coef_tbl_idx and ("Diagnostics" in text or "Interpretation" in text)
    )

    assert coef_tbl_idx < caption_idx < next_sec_idx


def test_docx_chi_square_structural_placement_ordering():
    """Verify Chi-Square heatmap appears immediately after contingency table."""
    wf = make_chi_square_workflow()
    docx_bytes = to_docx(wf, include_figures=True)
    elements = _extract_body_elements(docx_bytes)

    tbl_idx = next(i for i, (tag, text, _) in enumerate(elements) if tag == "tbl")
    caption_idx = next(
        i for i, (tag, text, _) in enumerate(elements) if text.startswith("Figure 1.")
    )
    next_sec_idx = next(
        i
        for i, (tag, text, _) in enumerate(elements)
        if i > tbl_idx and ("Diagnostics" in text or "Interpretation" in text)
    )

    assert tbl_idx < caption_idx < next_sec_idx


def test_docx_pairwise_forest_structural_placement_ordering():
    """Verify pairwise forest appears after pairwise comparisons table."""
    wf = make_one_way_anova_workflow()
    docx_bytes = to_docx(wf, include_figures=True)
    elements = _extract_body_elements(docx_bytes)

    pairwise_tbl_idx = next(
        i for i, (tag, text, _) in enumerate(elements) if tag == "tbl" and "Contrast" in text
    )
    caption_idx = next(
        i for i, (tag, text, _) in enumerate(elements) if text.startswith("Figure 1.")
    )
    next_sec_idx = next(
        i
        for i, (tag, text, _) in enumerate(elements)
        if i > pairwise_tbl_idx and ("Diagnostics" in text or "Interpretation" in text)
    )

    assert pairwise_tbl_idx < caption_idx < next_sec_idx


def test_docx_cell_profile_structural_placement_ordering():
    """Verify two-way ANOVA cell profile appears near cell summary table."""
    wf = make_two_way_anova_workflow()
    docx_bytes = to_docx(wf, include_figures=True)
    elements = _extract_body_elements(docx_bytes)

    cell_tbl_idx = next(
        i for i, (tag, text, _) in enumerate(elements) if tag == "tbl" and "Mean" in text
    )
    caption_idx = next(
        i for i, (tag, text, _) in enumerate(elements) if text.startswith("Figure 1.")
    )
    next_sec_idx = next(
        i
        for i, (tag, text, _) in enumerate(elements)
        if i > cell_tbl_idx and ("Diagnostics" in text or "Interpretation" in text)
    )

    assert cell_tbl_idx < caption_idx < next_sec_idx
