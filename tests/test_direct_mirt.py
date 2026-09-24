"""Checks for the direct likelihood implementation; optional PyTorch tests."""
import numpy as np
import pytest

torch = pytest.importorskip('torch')
from family_dif_benchmark_audit.core.direct_mirt import fit


def test_fixed_items_remain_fixed_and_training_is_reproducible():
    torch.set_num_threads(1)
    rng = np.random.default_rng(11)
    y = rng.binomial(1, .5, (16, 12)).astype('float32')
    first = fit(y, 2, 'cpu', 7, 200)
    second = fit(y, 2, 'cpu', 7, 200)
    for a, b in zip(first[:3], second[:3]):
        np.testing.assert_array_equal(a, b)
    a, b, _, _ = first
    aa, bb, theta, info = fit(y, 2, 'cpu', 9, 200, fixed=(a, b))
    np.testing.assert_array_equal(aa, a)
    np.testing.assert_array_equal(bb, b)
    assert theta.shape == (12, 2)
    assert info['history'][-1]['objective'] <= info['history'][0]['objective']


def test_no_false_convergence_after_one_optimizer_step():
    y = np.random.default_rng(2).binomial(1, .5, (12, 10)).astype('float32')
    *_, info = fit(y, 1, 'cpu', 3, 1)
    assert info['converged'] is False
