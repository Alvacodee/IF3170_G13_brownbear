from itertools import combinations
from typing import Self

import numpy as np

from .base import BaseModel

NO_PARENT = -1


# Source: https://scikit-learn.org/stable/modules/naive_bayes.html#categorical-naive-bayes
class NaiveBayes(BaseModel):
    """Categorical Naive Bayes classifier with additive smoothing.

    Parameters
    ----------
    alpha : float, default=1.0
        Smoothing strength added to every class-category count. Must be
        finite and positive.
    threshold : float or None, default=None
        For two classes, predict the second class when its probability is at
        least ``threshold``. ``None`` predicts the most likely class.

    Notes
    -----
    Input values must be nonnegative integer category codes. Category code 0
    is reserved as the fallback for values that were not seen during fitting.

    Every feature depends on the class only. ``parents_`` stores an optional
    feature parent for each feature, which is always ``NO_PARENT`` here and
    is learned by ``TreeAugmentedNaiveBayes``.
    """

    def __init__(self, alpha: float = 1.0, threshold: float | None = None) -> None:
        """Create a categorical Naive Bayes classifier.

        Parameters
        ----------
        alpha : float, default=1.0
            Smoothing strength for conditional category probabilities.
        threshold : float or None, default=None
            Decision threshold on the second class probability.
        """
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        if threshold is not None and not 0 < threshold < 1:
            raise ValueError("threshold must be None or between 0 and 1")
        self.alpha = float(alpha)
        self.threshold = threshold

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Fit class priors, feature parents, and smoothed conditional probabilities.

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Integer-encoded categorical features. Use a reserved category 0
            for missing or unknown values.
        y : np.ndarray of shape (n_samples,)
            Class labels.
        """
        X = self._validate_X(X)
        encoded_y = self._fit_classes(y, n_samples=len(X))

        self.n_features_in_ = X.shape[1]
        self.n_categories_ = X.max(axis=0) + 1
        self.class_count_ = np.bincount(
            encoded_y, minlength=len(self.classes_)
        ).astype(float)
        self.class_log_prior_ = np.log(self.class_count_ / len(encoded_y))
        self.parents_ = self._learn_parents(X, encoded_y)
        self.feature_log_prob_ = [
            self._conditional_log_prob(X, encoded_y, feature)
            for feature in range(self.n_features_in_)
        ]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict the most likely class, or apply ``threshold`` for two classes."""
        if getattr(self, "threshold", None) is not None and len(self.classes_) == 2:
            return self.classes_[(self.predict_proba(X)[:, 1] >= self.threshold).astype(int)]
        scores = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        """Return normalized class log-probabilities for each row in ``X``."""
        scores = self._joint_log_likelihood(X)
        maximum = scores.max(axis=1, keepdims=True)
        log_normalizer = maximum + np.log(
            np.exp(scores - maximum).sum(axis=1, keepdims=True)
        )
        return scores - log_normalizer

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return normalized class probabilities for each row in ``X``."""
        return np.exp(self.predict_log_proba(X))

    @property
    def edges_(self) -> list[tuple[int, int, float]]:
        """Feature edges as ``(parent, child, weight)``; Naive Bayes has none."""
        self._check_is_fitted()
        return []

    def _learn_parents(self, X: np.ndarray, encoded_y: np.ndarray) -> np.ndarray:
        """Return the feature parent of every feature."""
        return np.full(X.shape[1], NO_PARENT)

    def _conditional_log_prob(
        self, X: np.ndarray, encoded_y: np.ndarray, feature: int
    ) -> np.ndarray:
        """Return log P(feature | parent, class) indexed by (class, parent, category)."""
        n_categories = self.n_categories_[feature]
        shape = (len(self.classes_), self._n_parent_values(feature), n_categories)
        counts = _count(shape, encoded_y, self._parent_values(X, feature), X[:, feature])
        totals = counts.sum(axis=2, keepdims=True)
        return np.log((counts + self.alpha) / (totals + self.alpha * n_categories))

    def _parent_values(self, X: np.ndarray, feature: int) -> np.ndarray:
        parent = self.parents_[feature]
        if parent == NO_PARENT:
            return np.zeros(len(X), dtype=np.int64)
        return X[:, parent]

    def _n_parent_values(self, feature: int) -> int:
        parent = self.parents_[feature]
        return 1 if parent == NO_PARENT else int(self.n_categories_[parent])

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = self._validate_X(X, expected_features=self.n_features_in_)
        X = np.where(X < self.n_categories_, X, 0)

        scores = np.tile(self.class_log_prior_, (len(X), 1))
        for feature, log_prob in enumerate(self.feature_log_prob_):
            scores += log_prob[:, self._parent_values(X, feature), X[:, feature]].T
        return scores

    def _fit_classes(self, y: np.ndarray, n_samples: int) -> np.ndarray:
        """Store the class labels and return ``y`` as class indices."""
        y = np.asarray(y)
        if y.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if len(y) != n_samples:
            raise ValueError("X and y must contain the same number of samples")
        if len(y) == 0:
            raise ValueError("training data must not be empty")
        if np.issubdtype(y.dtype, np.number) and not np.isfinite(y).all():
            raise ValueError("y must contain finite labels")
        if y.dtype == object and any(label is None for label in y):
            raise ValueError("y must not contain missing labels")

        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        return encoded_y

    @staticmethod
    def _validate_X(
        X: np.ndarray, expected_features: int | None = None
    ) -> np.ndarray:
        try:
            values = np.asarray(X, dtype=float)
        except (TypeError, ValueError) as error:
            raise ValueError("X must contain numeric categories") from error

        if values.ndim != 2:
            raise ValueError("X must be a two-dimensional matrix")
        if expected_features is not None and values.shape[1] != expected_features:
            raise ValueError(f"X must contain {expected_features} features")
        if values.shape[1] == 0:
            raise ValueError("X must contain at least one feature")
        if not np.isfinite(values).all():
            raise ValueError("X must contain finite values")
        if (values < 0).any() or (values != np.floor(values)).any():
            raise ValueError("X must contain nonnegative integer categories")
        return values.astype(np.int64)

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "classes_"):
            raise RuntimeError("model must be fitted before prediction")


# Sources: Friedman, Geiger, and Goldszmidt (1997), https://doi.org/10.1023/A:1007465528199
# Chow and Liu (1968), https://doi.org/10.1109/TIT.1968.1054142
class TreeAugmentedNaiveBayes(NaiveBayes):
    """Naive Bayes in which every feature may also depend on one other feature.

    The feature parents form the maximum spanning tree of the conditional
    mutual information I(X_i; X_j | C). Among all structures where each
    feature has the class and at most one other feature as parents, this tree
    maximizes the log-likelihood of the training data. The feature with the
    highest mutual information with the class becomes the root, and every
    edge points away from it.

    Parameters
    ----------
    alpha : float, default=1.0
        Smoothing strength for conditional category probabilities.
    threshold : float or None, default=None
        Decision threshold on the second class probability.
    """

    def _learn_parents(self, X: np.ndarray, encoded_y: np.ndarray) -> np.ndarray:
        n_classes = len(self.classes_)
        self.class_mutual_info_ = np.array([
            _conditional_mutual_information(
                _count((1, n_classes, n_categories), np.zeros_like(encoded_y), encoded_y, X[:, feature])
            )
            for feature, n_categories in enumerate(self.n_categories_)
        ])
        self.conditional_mutual_info_ = np.zeros((X.shape[1], X.shape[1]))
        for first, second in combinations(range(X.shape[1]), 2):
            shape = (n_classes, self.n_categories_[first], self.n_categories_[second])
            weight = _conditional_mutual_information(
                _count(shape, encoded_y, X[:, first], X[:, second])
            )
            self.conditional_mutual_info_[first, second] = weight
            self.conditional_mutual_info_[second, first] = weight

        self.root_ = int(np.argmax(self.class_mutual_info_))
        return _maximum_spanning_tree(self.conditional_mutual_info_, self.root_)

    @property
    def edges_(self) -> list[tuple[int, int, float]]:
        """Feature edges as ``(parent, child, conditional mutual information)``."""
        self._check_is_fitted()
        return [
            (int(parent), child, float(self.conditional_mutual_info_[parent, child]))
            for child, parent in enumerate(self.parents_)
            if parent != NO_PARENT
        ]


def _count(shape: tuple[int, ...], *codes: np.ndarray) -> np.ndarray:
    """Count how often each combination of codes appears."""
    flat = np.ravel_multi_index(codes, shape)
    return np.bincount(flat, minlength=int(np.prod(shape))).reshape(shape).astype(float)


def _conditional_mutual_information(counts: np.ndarray) -> float:
    """Return I(A; B | C) in nats from a count table indexed by (c, a, b)."""
    joint = counts / counts.sum()
    p_c = joint.sum(axis=(1, 2))
    p_ac = joint.sum(axis=2)
    p_bc = joint.sum(axis=1)
    c, a, b = np.nonzero(joint)
    p_abc = joint[c, a, b]
    return float(np.sum(p_abc * np.log(p_abc * p_c[c] / (p_ac[c, a] * p_bc[c, b]))))


# Prim's algorithm on a dense graph: https://doi.org/10.1002/j.1538-7305.1957.tb01515.x
def _maximum_spanning_tree(weights: np.ndarray, root: int) -> np.ndarray:
    """Return the parent of every node in the maximum spanning tree rooted at ``root``."""
    n_nodes = len(weights)
    parents = np.full(n_nodes, NO_PARENT)
    in_tree = np.zeros(n_nodes, dtype=bool)
    in_tree[root] = True
    best_weight = weights[root].astype(float)
    best_link = np.full(n_nodes, root)

    for _ in range(n_nodes - 1):
        node = int(np.argmax(np.where(in_tree, -np.inf, best_weight)))
        in_tree[node] = True
        parents[node] = best_link[node]
        closer = ~in_tree & (weights[node] > best_weight)
        best_weight[closer] = weights[node, closer]
        best_link[closer] = node
    return parents


if __name__ == "__main__":
    X_train = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
    y_train = np.array([0, 0, 1, 1])
    model = NaiveBayes(alpha=1.0).fit(X_train, y_train)

    np.testing.assert_array_equal(model.predict(X_train), y_train)
    np.testing.assert_allclose(model.predict_proba(X_train).sum(axis=1), 1)
    assert np.isfinite(model.predict_log_proba(X_train)).all()

    unknown = np.array([[99, 99]])
    assert model.predict(unknown).shape == (1,)
    print("PASS: categorical likelihoods, predictions, probabilities, and fallback categories")

    # The class is the XOR of both features, so only their joint value is informative.
    X_xor = np.repeat(X_train, 5, axis=0)
    y_xor = np.repeat(np.array([0, 1, 1, 0]), 5)
    naive = NaiveBayes(alpha=1.0).fit(X_xor, y_xor)
    tree = TreeAugmentedNaiveBayes(alpha=1.0).fit(X_xor, y_xor)

    np.testing.assert_allclose(naive.predict_proba(X_train), 0.5)
    np.testing.assert_array_equal(tree.predict(X_train), [0, 1, 1, 0])
    assert [parent for parent, _, _ in tree.edges_] == [tree.root_]
    assert tree.predict(unknown).shape == (1,)
    print("PASS: tree-augmented structure captures dependence between features")
