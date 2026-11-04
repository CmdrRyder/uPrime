"""Run the shipped examples, so CI catches one breaking.

The example asserts its own numerical results, so running it to completion is
the whole test: if uprime.core changes shape or return conventions, the
asserts inside the example fire and this test fails.
"""

import os

os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

import runpy
from pathlib import Path

import pytest

# Resolved from this file, never from the working directory, so the test
# passes whether pytest runs from the repo root or anywhere else.
EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples"


def _example_paths():
    if not EXAMPLES_DIR.is_dir():
        return []
    return sorted(EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize(
    "example",
    _example_paths(),
    ids=lambda p: p.name,
)
def test_example_runs(example):
    """Execute the example as __main__; its own asserts do the checking."""
    runpy.run_path(str(example), run_name="__main__")


def test_examples_directory_is_not_empty():
    """Guard against the parametrised test silently collecting nothing."""
    assert _example_paths(), f"no examples found in {EXAMPLES_DIR}"
