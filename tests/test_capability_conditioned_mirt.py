import numpy as np
from scipy.special import expit

from family_dif_benchmark_audit.core.spectral_mirt import (
    choose_spectral_dimension,
    fit_spectral_mirt_offsets,
    purify_spectral_mirt_anchors,
)


def two_skill_responses(seed=7, n_items=480, n_models=96):
    rng = np.random.default_rng(seed)
    model_coordinates = rng.normal(size=(n_models, 2))
    item_loadings = rng.normal(size=(n_items, 2))
    item_loadings[: n_items // 2, 1] = 0.0
    item_loadings[n_items // 2 :, 0] = 0.0
    intercept = rng.normal(-0.25, 0.4, size=n_items)
    logits = intercept[:, None] + 1.3 * item_loadings @ model_coordinates.T
    return rng.binomial(1, expit(logits)).astype(np.int8)


def test_spectral_offsets_are_finite_and_deterministic():
    responses = two_skill_responses()
    selected = np.arange(len(responses)) % 2 == 0
    first, axes_first, loadings_first = fit_spectral_mirt_offsets(
        responses, selected, dimension=2, seed=19
    )
    second, axes_second, loadings_second = fit_spectral_mirt_offsets(
        responses, selected, dimension=2, seed=19
    )
    assert first.shape == responses.shape
    assert axes_first.shape == (responses.shape[1], 2)
    assert loadings_first.shape == (responses.shape[0], 2)
    assert np.isfinite(first).all()
    assert np.allclose(first, second)
    assert np.allclose(axes_first, axes_second)
    assert np.allclose(loadings_first, loadings_second)


def test_owner_heldout_prediction_detects_more_than_one_axis():
    responses = two_skill_responses(seed=22)
    owners = np.asarray(["owner_%03d" % index for index in range(responses.shape[1])])
    blueprint = np.asarray(
        ["source_%d" % (index % 4) for index in range(responses.shape[0])]
    )
    selected, frame, audit = choose_spectral_dimension(
        responses,
        owners,
        blueprint,
        owner_seed=101,
        svd_seed=102,
    )
    assert selected in set(frame["dimension"])
    assert audit["selected_dimension"] == selected
    assert audit["requested_dimensions"] == [1, 2, 4, 8, 16, 32, 64, 128, 256]
    assert audit["feasible_dimensions"] == [1, 2, 4, 8, 16, 32, 64]
    assert frame["dimension"].max() == 64
    assert frame["selected"].sum() == 1


def test_mirt_anchor_purification_honors_requested_fraction():
    responses = two_skill_responses(seed=31, n_items=240, n_models=80)
    families = np.arange(responses.shape[1], dtype=np.int32) % 5
    sources = np.asarray(
        ["source_%d" % (index % 4) for index in range(responses.shape[0])]
    )
    low = purify_spectral_mirt_anchors(
        responses,
        families,
        sources,
        dimension=2,
        svd_seed=77,
        rounds=1,
        anchor_fraction=0.3,
    )
    high = purify_spectral_mirt_anchors(
        responses,
        families,
        sources,
        dimension=2,
        svd_seed=77,
        rounds=1,
        anchor_fraction=0.7,
    )
    assert low.selected.sum() < high.selected.sum()
    assert np.isclose(low.weights.sum(), len(responses))
    assert np.isclose(high.weights.sum(), len(responses))
