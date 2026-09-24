"""Public workflow wrappers must preserve the frozen numerical comparisons."""
import importlib.util
from pathlib import Path

import pandas as pd
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def load_tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'tools' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_changed_upstream_is_rejected_before_deserialization(tmp_path):
    prep = load_tool('prepare_inputs')
    path = tmp_path / 'untrusted.pkl'
    path.write_bytes(b'not the pinned upstream file')
    with pytest.raises(ValueError, match='Checksum mismatch'):
        prep.verified(path, '0' * 64)


def test_workflow_keeps_all_five_benchmarks_and_three_diagnostics(tmp_path):
    workflow = load_tool('reproduce')
    cmds = workflow.commands('primary', tmp_path)
    assert [cmd[-1] for cmd in cmds] == list(workflow.KEYS)
    assert len(workflow.commands('diagnostics', tmp_path)) == 3
    assert 'content_replay' in ' '.join(workflow.commands('content', tmp_path)[0])


def test_direct_comparison_recomputed_from_public_benchmark_tables(tmp_path):
    tool = load_tool('summarize_direct_mirt')
    direct, spectral = tmp_path / 'direct', tmp_path / 'spectral'
    direct.mkdir()
    spectral.mkdir()
    for key in tool.LABELS:
        (direct / f'{key}_family_ranking_impact').symlink_to(ROOT / 'results/direct_mirt/per_benchmark' / key)
        (spectral / f'{key}_family_ranking_impact').symlink_to(ROOT / 'results/primary/per_benchmark' / key)
    summary, paired = tool.summarize(direct, spectral)
    pd.testing.assert_frame_equal(summary, pd.read_csv(ROOT / 'results/direct_mirt/summary.csv'),
                                  check_exact=False, rtol=1e-12, atol=1e-12)
    pd.testing.assert_frame_equal(paired, pd.read_csv(ROOT / 'results/direct_mirt/paired_cv_comparison.csv'),
                                  check_exact=False, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('kind', ['primary', 'direct_mirt'])
def test_selection_metadata_matches_final_cv_grid(kind):
    for directory in (ROOT / 'results' / kind / 'per_benchmark').iterdir():
        cv = pd.read_csv(directory / 'dimension_cv.csv')
        selected = pd.read_csv(directory / 'dimension_selection.csv').set_index('audit_half')
        for half, frame in cv.groupby('audit_half'):
            best = frame.loc[frame.mean_log_loss.idxmin()]
            threshold = best.mean_log_loss + best.se_across_validation_models
            choice = int(frame.loc[frame.mean_log_loss <= threshold, 'dimension'].min())
            assert int(selected.loc[half, 'selected_dimension']) == choice
            np.testing.assert_allclose(selected.loc[half, 'one_se_threshold'], threshold, rtol=0, atol=1e-12)
