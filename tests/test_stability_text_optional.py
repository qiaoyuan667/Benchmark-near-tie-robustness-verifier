"""Numeric stability reproduction must not require undistributed question text."""
import numpy as np
import pandas as pd
import pytest

from family_dif_benchmark_audit.interpretation import item_stability_v3 as stability


@pytest.mark.parametrize('benchmark', stability.MAIN_BENCHMARKS)
def test_numeric_examples_do_not_load_text(monkeypatch, benchmark):
    def unavailable(*args):
        raise FileNotFoundError('No question text provided')
    monkeypatch.setattr(stability, 'load_questions', unavailable)
    questions = stability.example_questions(benchmark, pd.DataFrame(), 3)
    np.testing.assert_array_equal(questions, ['', '', ''])
    with pytest.raises(FileNotFoundError):
        stability.example_questions(benchmark, pd.DataFrame(), 3, include_text=True)


def test_explicit_text_option_preserves_original_loader(monkeypatch):
    expected = np.array(['question A', 'question B'])
    monkeypatch.setattr(stability, 'load_questions', lambda *args: expected)
    assert stability.example_questions('BBH', pd.DataFrame(), 2, include_text=True) is expected
