"""Test that no runtime assert statements exist in pyautostat package source."""

import ast
import subprocess
import sys
from pathlib import Path

import pyautostat


def test_no_production_asserts_in_package_source():
    package_root = Path(pyautostat.__file__).resolve().parent
    python_files = list(package_root.rglob("*.py"))
    assert len(python_files) > 10, "Expected package source files to be found."

    assert_nodes = []
    for file_path in python_files:
        content = file_path.read_text(encoding="utf-8")
        tree = ast.parse(content, filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                assert_nodes.append((file_path.name, node.lineno))

    assert not assert_nodes, (
        f"Found {len(assert_nodes)} assert statements in package source: {assert_nodes}"
    )


def test_optimized_mode_execution():
    """Verify representative flows execute identically under python -O."""
    code = """
import pandas as pd
from pyautostat.research_assistant import ResearchAssistant

df = pd.DataFrame({
    "group": ["A"] * 10 + ["B"] * 10,
    "score": [10.0, 11.0, 12.0, 9.0, 11.0, 10.0, 12.0, 11.0, 9.0, 10.0]
           + [15.0, 14.0, 16.0, 15.0, 17.0, 14.0, 16.0, 15.0, 17.0, 15.0]
})
ra = ResearchAssistant(df)
res = ra.run(
    objective="compare_groups",
    outcome="score",
    predictor="group",
    estimand="mean",
    design="independent"
)
assert res.analysis is not None
assert res.analysis.status.value == "available"
print("OPTIMIZED_OK")
"""
    cmd = [sys.executable, "-O", "-c", code]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert "OPTIMIZED_OK" in result.stdout
