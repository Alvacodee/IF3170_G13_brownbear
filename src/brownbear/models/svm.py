import warnings
from collections import OrderedDict
from typing import Self

import numpy as np

from .base import BaseModel

TAU = 1e-12


# SMO dengan second-order working set selection.
# Sumber: Fan, Chen, Lin (2005), https://jmlr.org/papers/v6/fan05a.html
class SVM(BaseModel):
    def __init__(
        self,
        kernel: str = "gaussian",
        C: float = 1.0,
        degree: int = 3,
        gamma: float | str = "scale",
        coef0: float = 0.0,
        class_weight: dict | None = None,
        tol: float = 1e-3,
        max_iter: int = 1_000_000,
        cache_mb: int = 512,
    ) -> None:
        if kernel not in ("linear", "polynomial", "gaussian"):
            raise ValueError(f"Unknown kernel: {kernel}")
        self.kernel = kernel
        self.C = C
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.class_weight = class_weight
        self.tol = tol
        self.max_iter = max_iter
        self.cache_mb = cache_mb

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("SVM requires exactly 2 classes")

        signs = np.where(y == self.classes_[1], 1.0, -1.0)
        self.gamma_ = 1 / (X.shape[1] * X.var()) if self.gamma == "scale" else self.gamma
        alpha, self.bias_ = self._solve(X, signs, self._upper_bounds(y))

        self.support_ = np.flatnonzero(alpha > 0)
        self.support_vectors_ = X[self.support_]
        self.dual_coef_ = alpha[self.support_] * signs[self.support_]
        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        batches = np.array_split(X, len(X) // 1024 + 1)
        scores = [self._kernel(batch, self.support_vectors_) @ self.dual_coef_ for batch in batches]
        return np.concatenate(scores) + self.bias_

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.classes_[(self.decision_function(X) > 0).astype(int)]

    def _solve(self, X: np.ndarray, y: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, float]:
        alpha = np.zeros(len(X))
        gradient = -np.ones(len(X))
        diagonal = self._kernel_diagonal(X)
        kernel_row = self._kernel_row_cache(X)
        self.converged_ = False

        for iteration in range(1, self.max_iter + 1):
            self.n_iter_ = iteration
            score = -y * gradient
            can_increase = ((y > 0) & (alpha < upper)) | ((y < 0) & (alpha > 0))
            can_decrease = ((y > 0) & (alpha > 0)) | ((y < 0) & (alpha < upper))

            i = np.where(can_increase, score, -np.inf).argmax()
            max_score, min_score = score[i], score[can_decrease].min()
            if max_score - min_score < self.tol:
                self.converged_ = True
                break

            kernel_i = kernel_row(i)
            candidates = np.flatnonzero(can_decrease & (score < max_score))
            gain = max_score - score[candidates]
            curvature = diagonal[i] + diagonal[candidates] - 2 * kernel_i[candidates]
            curvature[curvature <= 0] = TAU
            best = (-(gain**2) / curvature).argmin()
            j = candidates[best]
            kernel_j = kernel_row(j)

            old_i, old_j = alpha[i], alpha[j]
            bound_i = upper[i] if y[i] > 0 else 0.0
            bound_j = 0.0 if y[j] > 0 else upper[j]
            room_i, room_j = abs(bound_i - old_i), abs(bound_j - old_j)
            step = min(gain[best] / curvature[best], room_i, room_j)

            # Harus tepat di batas: sisa pembulatan bisa membuat solver macet di pasangan yang sama.
            alpha[i] = bound_i if step == room_i else old_i + y[i] * step
            alpha[j] = bound_j if step == room_j else old_j - y[j] * step

            gradient += y * (y[i] * (alpha[i] - old_i) * kernel_i + y[j] * (alpha[j] - old_j) * kernel_j)

        if not self.converged_:
            warnings.warn(f"SVM did not converge within max_iter={self.max_iter}", stacklevel=3)

        score = -y * gradient
        free = (alpha > 0) & (alpha < upper)
        bias = score[free].mean() if free.any() else (max_score + min_score) / 2
        return alpha, bias

    def _upper_bounds(self, y: np.ndarray) -> np.ndarray:
        upper = np.full(len(y), float(self.C))
        for label, weight in (self.class_weight or {}).items():
            upper[y == label] *= weight
        return upper

    def _kernel_row_cache(self, X: np.ndarray):
        # Kernel matrix penuh butuh terlalu banyak memori, jadi barisnya dihitung saat dipakai.
        rows = OrderedDict()
        capacity = max(2, self.cache_mb * 1024**2 // (8 * len(X)))

        def kernel_row(index: int) -> np.ndarray:
            if index in rows:
                rows.move_to_end(index)
            else:
                if len(rows) == capacity:
                    rows.popitem(last=False)
                rows[index] = self._kernel(X[[index]], X)[0]
            return rows[index]

        return kernel_row

    def _kernel(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        if self.kernel == "gaussian":
            distances = (A**2).sum(axis=1, keepdims=True) + (B**2).sum(axis=1) - 2 * A @ B.T
            return np.exp(-self.gamma_ * np.maximum(distances, 0))
        return self._dot_kernel(A @ B.T)

    def _kernel_diagonal(self, X: np.ndarray) -> np.ndarray:
        if self.kernel == "gaussian":
            return np.ones(len(X))
        return self._dot_kernel((X**2).sum(axis=1))

    def _dot_kernel(self, dot: np.ndarray) -> np.ndarray:
        if self.kernel == "linear":
            return dot
        return (self.gamma_ * dot + self.coef0) ** self.degree
