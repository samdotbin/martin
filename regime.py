"""
regime.py — PCA/SVD/ICA + KMeans extractor (§11). Fit/save/load per fold.

Critical fold-safety rule: re-fit separately per walk-forward fold, on that
fold's TRAINING window only. Fitting once on the full history and reusing it
across folds leaks future information into earlier folds.
"""
import numpy as np
from sklearn.decomposition import PCA, FastICA, TruncatedSVD
from sklearn.cluster import KMeans

import config


class RegimeExtractor:
    def __init__(self, method=None, n_components=None, n_clusters=None, cfg=config):
        self.method = method or cfg.REGIME_METHOD
        self.n_components = n_components or cfg.N_REGIME_COMPONENTS
        self.n_clusters = n_clusters or cfg.N_REGIME_CLUSTERS
        # Store only the scalar values this class needs, NOT the config module
        # itself — a module reference isn't picklable, and this object gets
        # pickled alongside every checkpoint (§16, §21: "a checkpoint must
        # never be separated from the regime model it was trained with").
        self._min_regime_training_bars = cfg.MIN_REGIME_TRAINING_DAYS * cfg.BARS_PER_EPISODE
        self._decomposer = None
        self._kmeans = None
        self._sign_flip = None
        self._fitted = False

    def _make_decomposer(self):
        if self.method == "pca":
            return PCA(n_components=self.n_components)
        elif self.method == "svd":
            return TruncatedSVD(n_components=self.n_components)
        elif self.method == "ica":
            return FastICA(n_components=self.n_components, random_state=0, max_iter=1000)
        raise ValueError(f"unknown regime method: {self.method}")

    def fit(self, train_features: np.ndarray):
        """
        train_features: (n_bars, n_pairs * n_features) e.g. (n_train_days*24, 112)
        Fit ONCE per fold, on training data only (never on test data).
        """
        min_bars = self._min_regime_training_bars
        assert len(train_features) >= min_bars, (
            f"fold training window too short to fit a stable regime model "
            f"({len(train_features)} bars < {min_bars} required, §11)"
        )

        self._decomposer = self._make_decomposer()
        loadings = self._decomposer.fit_transform(train_features)

        # Fix SVD/PCA/ICA sign ambiguity: flip each component so its
        # largest-magnitude loading is positive, so the same regime doesn't
        # flip sign across folds fit on different windows.
        self._sign_flip = np.ones(loadings.shape[1])
        for i in range(loadings.shape[1]):
            argmax = np.argmax(np.abs(loadings[:, i]))
            if loadings[argmax, i] < 0:
                self._sign_flip[i] = -1
        loadings = loadings * self._sign_flip

        self._kmeans = KMeans(n_clusters=self.n_clusters, n_init=10, random_state=0)
        cluster_labels = self._kmeans.fit_predict(loadings)
        self._fitted = True
        return loadings, cluster_labels

    def transform(self, features: np.ndarray):
        """Transform-only — never re-fit (train-only fit is enforced by fit()) (§11, §17)."""
        assert self._fitted, "RegimeExtractor.transform() called before fit()"
        loadings = self._decomposer.transform(features) * self._sign_flip
        cluster_labels = self._kmeans.predict(loadings)
        return loadings, cluster_labels

    def embedding(self, loadings: np.ndarray, cluster_labels: np.ndarray) -> np.ndarray:
        """
        Continuous embedding (PCA loadings) AND discrete cluster id (one-hot),
        concatenated — don't discard the continuous signal in favor of only
        the cluster label (§11 step 5).
        """
        onehot = np.zeros((len(cluster_labels), self.n_clusters), dtype=np.float32)
        onehot[np.arange(len(cluster_labels)), cluster_labels] = 1.0
        return np.concatenate([loadings.astype(np.float32), onehot], axis=1)
