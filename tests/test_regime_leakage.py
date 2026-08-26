"""
test_regime_leakage.py — confirm fit/transform split is leak-free per fold (§21).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

import config as cfg
from regime import RegimeExtractor


def _synthetic_features(n_bars, n_dims=112, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, size=(n_bars, n_dims)).astype(np.float32)


def test_fit_is_never_called_on_test_data():
    extractor = RegimeExtractor(cfg=cfg)
    min_bars = cfg.MIN_REGIME_TRAINING_DAYS * cfg.BARS_PER_EPISODE
    train_feats = _synthetic_features(min_bars + 500, seed=1)
    test_feats = _synthetic_features(300, seed=2)

    train_loadings, train_clusters = extractor.fit(train_feats)
    components_after_fit = extractor._decomposer.components_.copy() if hasattr(extractor._decomposer, "components_") else None

    # transform() must not mutate the fitted decomposer.
    test_loadings, test_clusters = extractor.transform(test_feats)
    components_after_transform = extractor._decomposer.components_.copy() if hasattr(extractor._decomposer, "components_") else None

    if components_after_fit is not None:
        assert np.allclose(components_after_fit, components_after_transform), (
            "transform() must not refit the decomposer — components changed after transform()"
        )

    assert test_loadings.shape[0] == 300
    assert train_loadings.shape[0] == min_bars + 500


def test_transform_before_fit_raises():
    extractor = RegimeExtractor(cfg=cfg)
    try:
        extractor.transform(_synthetic_features(100))
        raised = False
    except AssertionError:
        raised = True
    assert raised, "transform() before fit() must raise, not silently return garbage"


def test_min_data_guard():
    extractor = RegimeExtractor(cfg=cfg)
    too_few = _synthetic_features(cfg.MIN_REGIME_TRAINING_DAYS * cfg.BARS_PER_EPISODE - 10)
    try:
        extractor.fit(too_few)
        raised = False
    except AssertionError:
        raised = True
    assert raised, "fit() must assert on too-short training windows (§11 min-data guard)"


def test_sign_fix_is_consistent_between_fit_and_transform():
    extractor = RegimeExtractor(cfg=cfg)
    min_bars = cfg.MIN_REGIME_TRAINING_DAYS * cfg.BARS_PER_EPISODE
    train_feats = _synthetic_features(min_bars + 200, seed=3)
    train_loadings, _ = extractor.fit(train_feats)
    # Re-transforming the training data itself should reproduce the same
    # sign-fixed loadings (transform re-applies the same _sign_flip as fit).
    reloadings, _ = extractor.transform(train_feats)
    assert np.allclose(train_loadings, reloadings, atol=1e-4), (
        "sign-fixed loadings from fit() and a transform() of the same data must match"
    )


if __name__ == "__main__":
    test_fit_is_never_called_on_test_data()
    test_transform_before_fit_raises()
    test_min_data_guard()
    test_sign_fix_is_consistent_between_fit_and_transform()
    print("test_regime_leakage.py: all tests passed")
