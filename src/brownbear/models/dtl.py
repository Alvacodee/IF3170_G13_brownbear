"""CART classification tree implemented with NumPy and Python built-ins."""

from dataclasses import dataclass
from numbers import Real
from typing import Self

import numpy as np

from .base import BaseModel


# CART dan Gini: https://scikit-learn.org/stable/modules/tree.html
@dataclass
class _Node:
    """A CART node; leaves have ``feature is None``."""

    prediction: object
    feature: int | None = None
    is_numeric: bool = False
    threshold: float | None = None
    category: object = None
    missing_go_to_left: bool = True
    left: "_Node | None" = None
    right: "_Node | None" = None


class DecisionTree(BaseModel):
    """Binary-split CART classifier using Gini impurity.

    Numeric features split at thresholds. Categorical features split one
    category against the rest. At each candidate split, missing values are
    tested on both sides and follow the better direction at prediction time.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
    ) -> None:
        if max_depth is not None and max_depth < 0:
            raise ValueError("max_depth must be nonnegative or None")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")
        if min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be at least 1")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.tree_: _Node | None = None
        self.classes_: np.ndarray | None = None
        self.n_features_in_: int | None = None

    @staticmethod
    def _is_missing(value: object) -> bool:
        if value is None or value is np.ma.masked:
            return True
        try:
            missing = np.isnan(value)
            return bool(missing) if np.ndim(missing) == 0 else False
        except (TypeError, ValueError):
            return False

    @classmethod
    def _as_2d(cls, X: np.ndarray, *, fitting: bool = False) -> np.ndarray:
        # DataFrames retain mixed numeric/categorical columns when converted
        # to object arrays; this also avoids depending on pandas in the model.
        if hasattr(X, "to_numpy"):
            values = X.to_numpy(dtype=object)
        else:
            values = np.asarray(X, dtype=object)
        if values.ndim == 1:
            if fitting:
                values = values.reshape(-1, 1)
            else:
                values = values.reshape(1, -1)
        if values.ndim != 2:
            raise ValueError("X must be a 2D feature matrix")
        return values

    @staticmethod
    def _gini_counts(counts: np.ndarray) -> float:
        total = counts.sum()
        if total == 0:
            return 0.0
        probabilities = counts / total
        return float(1.0 - probabilities @ probabilities)

    @staticmethod
    def _majority(y: np.ndarray) -> object:
        counts: dict[object, int] = {}
        for value in y:
            counts[value] = counts.get(value, 0) + 1
        # Dict insertion order gives a stable first-seen tie break.
        return max(counts, key=counts.__getitem__)

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        values = self._as_2d(X, fitting=True)
        labels = np.asarray(y)
        if labels.ndim != 1 or len(labels) != len(values):
            raise ValueError("y must be one-dimensional and have the same rows as X")
        if len(labels) == 0:
            raise ValueError("Cannot fit a decision tree on an empty dataset")
        if any(self._is_missing(value) for value in labels):
            raise ValueError("y cannot contain missing labels")
        try:
            classes, encoded = np.unique(labels, return_inverse=True)
        except TypeError:
            # Mixed but hashable label types do not necessarily have an order.
            class_to_index: dict[object, int] = {}
            class_list: list[object] = []
            encoded = np.empty(len(labels), dtype=int)
            for i, label in enumerate(labels):
                if label not in class_to_index:
                    class_to_index[label] = len(class_list)
                    class_list.append(label)
                encoded[i] = class_to_index[label]
            classes = np.asarray(class_list, dtype=object)
        self.classes_ = classes
        self.n_features_in_ = values.shape[1]
        self.tree_ = self._build(values, labels, encoded, depth=0)
        return self

    def _build(
        self, X: np.ndarray, y: np.ndarray, encoded_y: np.ndarray, depth: int
    ) -> _Node:
        prediction = self._majority(y)
        node = _Node(prediction=prediction)
        if (
            len(np.unique(encoded_y)) <= 1
            or len(y) < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return node

        split = self._best_split(X, encoded_y)
        if split is None:
            return node
        feature, is_numeric, condition, missing_left, left_mask, gain = split
        if gain <= 1e-15:
            return node

        node.feature = feature
        node.is_numeric = is_numeric
        node.missing_go_to_left = missing_left
        if is_numeric:
            node.threshold = float(condition)
        else:
            node.category = condition
        node.left = self._build(X[left_mask], y[left_mask], encoded_y[left_mask], depth + 1)
        right_mask = ~left_mask
        node.right = self._build(X[right_mask], y[right_mask], encoded_y[right_mask], depth + 1)
        return node

    def _best_split(
        self, X: np.ndarray, encoded_y: np.ndarray
    ) -> tuple[int, bool, object, bool, np.ndarray, float] | None:
        n_rows, n_features = X.shape
        n_classes = len(self.classes_)  # type: ignore[arg-type]
        parent_counts = np.bincount(encoded_y, minlength=n_classes)
        parent_gini = self._gini_counts(parent_counts)
        best: tuple[int, bool, object, bool, np.ndarray, float] | None = None
        best_gain = 0.0

        for feature in range(n_features):
            column = X[:, feature]
            missing = np.fromiter((self._is_missing(v) for v in column), bool, n_rows)
            present_idx = np.flatnonzero(~missing)
            if len(present_idx) < 2:
                continue
            present = column[present_idx]
            is_numeric = all(isinstance(v, Real) and not isinstance(v, complex) for v in present)
            if is_numeric:
                numeric = np.asarray(present, dtype=float)
                if not np.isfinite(numeric).all():
                    raise ValueError("Numeric features must be finite or missing")
                order = np.argsort(numeric, kind="mergesort")
                sorted_values = numeric[order]
                sorted_idx = present_idx[order]
                boundaries = np.flatnonzero(sorted_values[:-1] < sorted_values[1:])
                if len(boundaries) == 0:
                    continue
                # Prefix class counts let every threshold be scored without
                # rescanning all rows for each candidate.
                sorted_one_hot = np.eye(n_classes, dtype=np.int64)[encoded_y[sorted_idx]]
                prefix = np.vstack((np.zeros((1, n_classes), dtype=np.int64), np.cumsum(sorted_one_hot, axis=0)))
                missing_counts = np.bincount(encoded_y[missing], minlength=n_classes)
                present_counts = prefix[-1]
                for boundary in boundaries:
                    left_present = prefix[boundary + 1]
                    right_present = present_counts - left_present
                    lower_value, upper_value = sorted_values[boundary : boundary + 2]
                    threshold = lower_value + (upper_value - lower_value) / 2
                    # Adjacent floating-point values may have no representable
                    # midpoint; the lower observed value preserves the split.
                    if threshold >= upper_value:
                        threshold = lower_value
                    for missing_left in (True, False):
                        left_counts = left_present + (missing_counts if missing_left else 0)
                        right_counts = right_present + (0 if missing_left else missing_counts)
                        left_n, right_n = int(left_counts.sum()), int(right_counts.sum())
                        if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                            continue
                        child_gini = (left_n * self._gini_counts(left_counts) + right_n * self._gini_counts(right_counts)) / n_rows
                        gain = parent_gini - child_gini
                        if gain > best_gain + 1e-15:
                            left_mask = np.zeros(n_rows, dtype=bool)
                            left_mask[sorted_idx[: boundary + 1]] = True
                            left_mask[missing] = missing_left
                            best_gain = gain
                            best = (feature, True, float(threshold), missing_left, left_mask, gain)
                continue

            # CART categorical split: one observed category vs. all others.
            categories: list[object] = []
            for value in present:
                if not any(value == known for known in categories):
                    categories.append(value)
            missing_counts = np.bincount(encoded_y[missing], minlength=n_classes)
            for category in categories:
                category_mask = np.fromiter((value == category for value in present), bool, len(present))
                category_counts = np.bincount(encoded_y[present_idx[category_mask]], minlength=n_classes)
                other_counts = np.bincount(encoded_y[present_idx[~category_mask]], minlength=n_classes)
                for missing_left in (True, False):
                    left_counts = category_counts + (missing_counts if missing_left else 0)
                    right_counts = other_counts + (0 if missing_left else missing_counts)
                    left_n, right_n = int(left_counts.sum()), int(right_counts.sum())
                    if left_n < self.min_samples_leaf or right_n < self.min_samples_leaf:
                        continue
                    child_gini = (left_n * self._gini_counts(left_counts) + right_n * self._gini_counts(right_counts)) / n_rows
                    gain = parent_gini - child_gini
                    if gain > best_gain + 1e-15:
                        left_mask = np.zeros(n_rows, dtype=bool)
                        left_mask[present_idx[category_mask]] = True
                        left_mask[missing] = missing_left
                        best_gain = gain
                        best = (feature, False, category, missing_left, left_mask, gain)
        return best

    def _predict_one(self, row: np.ndarray) -> object:
        node = self.tree_
        while node is not None and node.feature is not None:
            value = row[node.feature]
            if self._is_missing(value):
                go_left = node.missing_go_to_left
            elif node.is_numeric:
                if not isinstance(value, Real):
                    raise ValueError(f"Feature {node.feature} was numeric during fit")
                go_left = float(value) <= node.threshold  # type: ignore[operator]
            else:
                go_left = value == node.category
            node = node.left if go_left else node.right
        if node is None:
            raise RuntimeError("Tree contains an incomplete split")
        return node.prediction

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.tree_ is None or self.n_features_in_ is None:
            raise RuntimeError("Call fit before predict")
        values = self._as_2d(X)
        if values.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {values.shape[1]}")
        return np.asarray([self._predict_one(row) for row in values])
