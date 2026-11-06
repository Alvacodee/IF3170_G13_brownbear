from typing import Self

import numpy as np

from .base import BaseModel


# Source: https://scikit-learn.org/stable/modules/naive_bayes.html#categorical-naive-bayes
class NaiveBayes(BaseModel):
    """Categorical Naive Bayes with additive smoothing."""

    def __init__(self, alpha: float = 1.0) -> None:
        if not np.isfinite(alpha) or alpha <= 0:
            raise ValueError("alpha must be finite and positive")
        self.alpha = float(alpha)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X = self._validate_X(X)
        y = np.asarray(y)

        if y.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if len(y) != len(X):
            raise ValueError("X and y must contain the same number of samples")
        if len(y) == 0:
            raise ValueError("training data must not be empty")
        if np.issubdtype(y.dtype, np.number) and not np.isfinite(y).all():
            raise ValueError("y must contain finite labels")
        if y.dtype == object and any(label is None for label in y):
            raise ValueError("y must not contain missing labels")

        self.classes_, encoded_y = np.unique(y, return_inverse=True)
        self.n_features_in_ = X.shape[1]
        self.class_count_ = np.bincount(
            encoded_y, minlength=len(self.classes_)
        ).astype(float)
        self.class_log_prior_ = np.log(self.class_count_ / len(y))
        self.n_categories_ = X.max(axis=0) + 1
        self.category_count_: list[np.ndarray] = []
        self.feature_log_prob_: list[np.ndarray] = []

        for feature_index, category_count in enumerate(self.n_categories_):
            counts = np.zeros((len(self.classes_), category_count), dtype=float)
            np.add.at(counts, (encoded_y, X[:, feature_index]), 1)
            denominator = (
                self.class_count_[:, np.newaxis] + self.alpha * category_count
            )
            self.category_count_.append(counts)
            self.feature_log_prob_.append(
                np.log((counts + self.alpha) / denominator)
            )

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        scores = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def predict_log_proba(self, X: np.ndarray) -> np.ndarray:
        scores = self._joint_log_likelihood(X)
        maximum = scores.max(axis=1, keepdims=True)
        log_normalizer = maximum + np.log(
            np.exp(scores - maximum).sum(axis=1, keepdims=True)
        )
        return scores - log_normalizer

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.exp(self.predict_log_proba(X))

    def _joint_log_likelihood(self, X: np.ndarray) -> np.ndarray:
        self._check_is_fitted()
        X = self._validate_X(X, expected_features=self.n_features_in_)

        if np.any(X >= self.n_categories_):
            raise ValueError("X contains categories that were not fitted")

        scores = np.tile(self.class_log_prior_, (len(X), 1))
        for feature_index, log_probability in enumerate(self.feature_log_prob_):
            scores += log_probability[:, X[:, feature_index]].T
        return scores

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


if __name__ == "__main__":
    X_train = np.array([[1, 1], [1, 2], [2, 1], [2, 2]])
    y_train = np.array([0, 0, 1, 1])
    model = NaiveBayes(alpha=1.0).fit(X_train, y_train)

    np.testing.assert_array_equal(model.predict(X_train), y_train)
    np.testing.assert_allclose(model.predict_proba(X_train).sum(axis=1), 1)
    assert np.isfinite(model.predict_log_proba(X_train)).all()
    print("PASS: categorical likelihoods, predictions, and probabilities")
