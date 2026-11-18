import warnings
from typing import Self

import numpy as np

from .base import BaseModel


# Optimasi proximal: https://web.stanford.edu/~boyd/papers/prox_algs.html
# Akselerasi (FISTA): https://epubs.siam.org/doi/10.1137/080716542
# Restart adaptif: https://doi.org/10.1007/s10208-013-9150-3
class LogisticRegression(BaseModel):
    """Binary weighted logistic regression with proximal L1/L2 optimization.

    Objective: mean weighted BCE + reg_strength *
    (rho * ||w||_1 + (1-rho) * ||w||_2**2 / 2). The intercept is unpenalized.
    Reference: https://web.stanford.edu/~boyd/papers/prox_algs.html

    ``accelerate=False`` drops the momentum step, which leaves plain proximal
    gradient descent. ``record_path=True`` stores every accepted iterate as
    ``path_`` with columns (intercept, coef_0, coef_1, ...).
    """

    def __init__(self, penalty="l2", reg_strength=1e-3, l1_ratio=.5,
                 class_weight=None, max_iter=5000, tol=1e-6, threshold=.5,
                 accelerate=True, record_path=False):
        self.penalty = penalty
        self.reg_strength = reg_strength
        self.l1_ratio = l1_ratio
        self.class_weight = class_weight
        self.max_iter = max_iter
        self.tol = tol
        self.threshold = threshold
        self.accelerate = accelerate
        self.record_path = record_path

    @property
    def _rho(self):
        return {"l1": 1., "l2": 0., "elasticnet": self.l1_ratio}[self.penalty]

    @staticmethod
    def _sigmoid(z):
        exp = np.exp(-np.abs(z))
        return np.where(z >= 0, 1 / (1 + exp), exp / (1 + exp))

    @staticmethod
    def _soft_threshold(w, amount):
        return np.sign(w) * np.maximum(np.abs(w) - amount, 0)

    @staticmethod
    def _matrix(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or not len(X) or not np.isfinite(X).all():
            raise ValueError("X must be a nonempty finite two-dimensional matrix")
        return X

    def _validate_parameters(self):
        if self.penalty not in ("l1", "l2", "elasticnet"):
            raise ValueError("penalty must be l1, l2, or elasticnet")
        for name, value, lower, upper in [
            ("reg_strength", self.reg_strength, 0, np.inf),
            ("l1_ratio", self.l1_ratio, 0, 1),
            ("threshold", self.threshold, 0, 1),
        ]:
            if not np.isfinite(value) or not lower <= value <= upper:
                raise ValueError(f"Invalid {name}: {value}")
        if self.class_weight not in (None, "balanced"):
            raise ValueError("class_weight must be None or balanced")
        if not isinstance(self.max_iter, (int, np.integer)) or self.max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        if not np.isfinite(self.tol) or self.tol <= 0:
            raise ValueError("tol must be positive and finite")
        for name in ("accelerate", "record_path"):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise TypeError(f"{name} must be a boolean")

    @staticmethod
    def _weights(y, sample_weight):
        weights = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        if (weights.shape != y.shape or not np.isfinite(weights).all()
                or (weights < 0).any() or weights.sum() <= 0):
            raise ValueError("sample_weight must be finite, nonnegative, and have positive total")
        return weights

    def _smooth_gradient(self, X, y, weights, w, b):
        residual = weights / weights.sum() * (self._sigmoid(X @ w + b) - y)
        return (X.T @ residual + self.reg_strength * (1 - self._rho) * w,
                residual.sum())

    def _smooth_loss(self, X, y, normalized_weights, w, b):
        z = X @ w + b
        return (normalized_weights @ (np.logaddexp(0, z) - y * z)
                + self.reg_strength * (1 - self._rho) * (w @ w) / 2)

    def objective(self, X, y, sample_weight=None, *, coef=None, intercept=None):
        """Evaluate the objective with explicit weights (class weights are not reapplied)."""
        X = self._matrix(X)
        y = np.asarray(y, dtype=float)
        if y.shape != (len(X),) or not np.isin(y, [0, 1]).all():
            raise ValueError("y must contain binary labels matching X")
        if coef is None or intercept is None:
            self._require_fitted()
        w = self.coef_ if coef is None else np.asarray(coef, dtype=float)
        b = self.intercept_ if intercept is None else float(intercept)
        weights = self._weights(y, sample_weight)
        return float(self._smooth_loss(X, y, weights / weights.sum(), w, b)
                     + self.reg_strength * self._rho * np.abs(w).sum())

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None) -> Self:
        self._validate_parameters()
        X = self._matrix(X)
        y = np.asarray(y, dtype=float)
        if y.shape != (len(X),) or set(np.unique(y)) != {0., 1.}:
            raise ValueError("y must match X and contain both binary labels 0 and 1")
        weights = self._weights(y, sample_weight).copy()
        if self.class_weight == "balanced":
            weights *= len(y) / (2 * np.bincount(y.astype(int))[y.astype(int)])
        if any(weights[y == label].sum() <= 0 for label in (0, 1)):
            raise ValueError("Both classes must have positive sample weight")
        weights /= weights.sum()
        w, b, step = np.zeros(X.shape[1]), 0., 1.
        extrapolated_w, extrapolated_b, momentum = w.copy(), b, 1.
        smooth = self._smooth_loss(X, y, weights, w, b)
        self.loss_history_ = [float(smooth)]
        path = [np.concatenate(([b], w))]
        self.converged_ = False
        l1 = self.reg_strength * self._rho
        for iteration in range(1, self.max_iter + 1):
            grad_w, grad_b = self._smooth_gradient(X, y, weights, extrapolated_w, extrapolated_b)
            smooth = self._smooth_loss(X, y, weights, extrapolated_w, extrapolated_b)
            step = min(step * 2, 1e6)
            for _ in range(80):
                new_w = self._soft_threshold(extrapolated_w - step * grad_w, step * l1)
                new_b = extrapolated_b - step * grad_b
                dw, db = new_w - extrapolated_w, new_b - extrapolated_b
                new_smooth = self._smooth_loss(X, y, weights, new_w, new_b)
                bound = smooth + grad_w @ dw + grad_b * db + (dw @ dw + db * db) / (2 * step)
                if np.isfinite(new_smooth) and new_smooth <= bound:
                    break
                step *= .5
            else:
                raise FloatingPointError("Backtracking failed to find a finite descent step")
            objective = float(new_smooth + l1 * np.abs(new_w).sum())
            # Restart momentum when extrapolation increases the full objective.
            if objective > self.loss_history_[-1] + 1e-15:
                extrapolated_w, extrapolated_b, momentum = w.copy(), b, 1.
                continue
            mapping_norm = max(np.max(np.abs(dw), initial=0), abs(db)) / step
            new_momentum = (1 + np.sqrt(1 + 4 * momentum ** 2)) / 2 if self.accelerate else 1.
            extrapolated_w = new_w + (momentum - 1) / new_momentum * (new_w - w)
            extrapolated_b = new_b + (momentum - 1) / new_momentum * (new_b - b)
            momentum = new_momentum
            w, b, smooth = new_w, new_b, new_smooth
            self.loss_history_.append(objective)
            path.append(np.concatenate(([b], w)))
            if mapping_norm <= self.tol:
                self.converged_ = True
                break
        self.coef_, self.intercept_ = w, float(b)
        self.classes_ = np.array([0, 1])
        self.n_features_in_ = X.shape[1]
        self.n_iter_ = iteration
        if self.record_path:
            self.path_ = np.array(path)
        if not self.converged_:
            warnings.warn(f"LogisticRegression did not converge in {self.max_iter} iterations",
                          RuntimeWarning, stacklevel=2)
        return self

    def _require_fitted(self):
        if not hasattr(self, "coef_"):
            raise ValueError("Call fit before prediction or objective evaluation")

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        self._require_fitted()
        X = self._matrix(X)
        if X.shape[1] != self.n_features_in_:
            raise ValueError("X has a different feature count from fit")
        return X @ self.coef_ + self.intercept_

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self._sigmoid(self.decision_function(X))
        return np.column_stack([1 - p, p])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not np.isfinite(self.threshold) or not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
