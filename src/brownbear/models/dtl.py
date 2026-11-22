"""Decision Tree Learning varian CART untuk klasifikasi, diimplementasikan dengan NumPy.

Referensi utama:
- L. Breiman, J. Friedman, R. Olshen, C. Stone, *Classification and Regression Trees*,
  Wadsworth, 1984 (split biner, Gini, split kategorikal optimal, cost-complexity pruning).
- Dokumentasi scikit-learn, "Decision Trees": https://scikit-learn.org/stable/modules/tree.html
  (formulasi impurity berbobot dan minimal cost-complexity pruning).
- T. Chen dan C. Guestrin, "XGBoost: A Scalable Tree Boosting System", KDD 2016,
  https://doi.org/10.1145/2939672.2939785 (arah default nilai kosong yang dipelajari saat split).
"""

from numbers import Real
from typing import Self

import numpy as np

from .base import BaseModel

_LEAF = -1
_EPS = 1e-12
_KEEP = object()


class DecisionTree(BaseModel):
    """Pohon klasifikasi CART dengan Gini impurity.

    Parameter
    ---------
    max_depth : int atau None, bawaan=None
        Kedalaman maksimum pohon. ``None`` berarti pohon tumbuh sampai kriteria lain berhenti.
    min_samples_split : int, bawaan=2
        Jumlah baris minimum pada simpul agar simpul tersebut boleh dipecah.
    min_samples_leaf : int, bawaan=1
        Jumlah baris minimum pada setiap daun hasil pemecahan.
    min_impurity_decrease : float, bawaan=0.0
        Penurunan impurity berbobot minimum, ``W_t / W * (G_t - G_anak)``, agar split diterima.
    ccp_alpha : float, bawaan=0.0
        Parameter kompleksitas untuk minimal cost-complexity pruning setelah pohon tumbuh penuh.
    class_weight : None, "balanced", atau dict, bawaan=None
        Bobot kelas pada perhitungan Gini dan distribusi daun.
    threshold : float atau None, bawaan=None
        Ambang peluang kelas kedua untuk klasifikasi biner; ``None`` memakai kelas mayoritas daun.

    Fitur numerik dipecah dengan ambang ``x <= t``. Fitur kategorikal (berisi teks) dipecah
    menjadi dua himpunan kategori. Untuk dua kelas, kategori diurutkan berdasarkan proporsi
    kelas positif sehingga cukup ``k - 1`` kandidat untuk menemukan partisi optimal dari
    ``2^(k-1) - 1`` partisi (Breiman dkk., 1984, Teorema 4.5). Nilai kosong diuji di kedua
    sisi untuk setiap kandidat split dan arah terbaik disimpan sebagai arah default saat
    prediksi (Chen dan Guestrin, 2016). Kategori yang tidak muncul saat training juga
    mengikuti arah default tersebut.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        min_impurity_decrease: float = 0.0,
        ccp_alpha: float = 0.0,
        class_weight: str | dict | None = None,
        threshold: float | None = None,
    ) -> None:
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.ccp_alpha = ccp_alpha
        self.class_weight = class_weight
        self.threshold = threshold
        self._validate_parameters()

    # ------------------------------------------------------------------ validasi & encoding
    def _validate_parameters(self) -> None:
        if self.max_depth is not None and (not isinstance(self.max_depth, (int, np.integer)) or self.max_depth < 0):
            raise ValueError("max_depth must be a nonnegative integer or None")
        if not isinstance(self.min_samples_split, (int, np.integer)) or self.min_samples_split < 2:
            raise ValueError("min_samples_split must be an integer >= 2")
        if not isinstance(self.min_samples_leaf, (int, np.integer)) or self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be an integer >= 1")
        for name in ("min_impurity_decrease", "ccp_alpha"):
            value = getattr(self, name)
            if not isinstance(value, Real) or not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a nonnegative finite number")
        if not (self.class_weight is None or self.class_weight == "balanced" or isinstance(self.class_weight, dict)):
            raise ValueError("class_weight must be None, 'balanced', or a dict")
        if self.threshold is not None and not 0 < self.threshold < 1:
            raise ValueError("threshold must be None or between 0 and 1")

    @staticmethod
    def _is_missing(value: object) -> bool:
        if value is None:
            return True
        try:
            return bool(np.isnan(value))
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _as_object_matrix(X: np.ndarray) -> np.ndarray:
        # DataFrame dikonversi ke object agar kolom numerik dan teks tetap utuh.
        values = X.to_numpy(dtype=object) if hasattr(X, "to_numpy") else np.asarray(X, dtype=object)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if values.ndim != 2 or values.shape[1] == 0:
            raise ValueError("X must be a 2D feature matrix with at least one column")
        return values

    def _encode(self, X: np.ndarray, *, fitting: bool) -> np.ndarray:
        """Mengubah X menjadi matriks float: nilai kosong = NaN, kategori = kode >= 0, kategori baru = -1."""
        values = self._as_object_matrix(X)
        if fitting:
            self.n_features_in_ = values.shape[1]
            self.feature_names_in_ = (
                np.asarray(X.columns, dtype=object) if hasattr(X, "columns") else None
            )
            self.is_categorical_ = np.zeros(values.shape[1], dtype=bool)
            self.categories_: list[np.ndarray | None] = []
        elif values.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {values.shape[1]}")

        encoded = np.empty(values.shape, dtype=np.float64)
        for j in range(values.shape[1]):
            column = values[:, j]
            missing = np.fromiter((self._is_missing(v) for v in column), bool, len(column))
            present = column[~missing]
            if fitting:
                numeric = all(isinstance(v, Real) for v in present)
                self.is_categorical_[j] = not numeric
                categories = None if numeric else np.asarray(sorted({str(v) for v in present}), dtype=object)
                self.categories_.append(categories)
            if not self.is_categorical_[j]:
                if not all(isinstance(v, Real) for v in present):
                    raise ValueError(f"Feature {j} was numeric during fit")
                numbers = np.full(len(column), np.nan)
                numbers[~missing] = np.asarray(present, dtype=np.float64)
                if not np.isfinite(numbers[~missing]).all():
                    raise ValueError("Numeric features must be finite or missing")
                encoded[:, j] = numbers
            else:
                lookup = {category: code for code, category in enumerate(self.categories_[j])}
                codes = np.full(len(column), np.nan)
                codes[~missing] = [lookup.get(str(v), -1) for v in present]
                encoded[:, j] = codes
        return encoded

    def _sample_weights(self, encoded_y: np.ndarray) -> np.ndarray:
        n_classes = len(self.classes_)
        if self.class_weight is None:
            per_class = np.ones(n_classes)
        elif self.class_weight == "balanced":
            # Bobot seimbang: n / (k * n_c), seperti scikit-learn.
            per_class = len(encoded_y) / (n_classes * np.bincount(encoded_y, minlength=n_classes))
        else:
            per_class = np.array([float(self.class_weight.get(label, 1.0)) for label in self.classes_])
            if (per_class <= 0).any() or not np.isfinite(per_class).all():
                raise ValueError("class_weight values must be positive and finite")
        return per_class[encoded_y]

    # ------------------------------------------------------------------ training
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Menumbuhkan pohon CART, lalu memangkasnya jika ``ccp_alpha > 0``."""
        self._validate_parameters()
        labels = np.asarray(y)
        if labels.ndim != 1:
            raise ValueError("y must be one-dimensional")
        if any(self._is_missing(value) for value in labels):
            raise ValueError("y cannot contain missing labels")
        encoded_X = self._encode(X, fitting=True)
        if len(labels) != len(encoded_X) or len(labels) == 0:
            raise ValueError("X and y must have the same nonzero number of rows")
        self.classes_, encoded_y = np.unique(labels, return_inverse=True)
        weights = self._sample_weights(encoded_y)
        # Matriks one-hot berbobot: jumlah kolomnya per simpul = bobot tiap kelas.
        self._Y = np.zeros((len(labels), len(self.classes_)))
        self._Y[np.arange(len(labels)), encoded_y] = weights
        self._X = encoded_X
        self._encoded_y = encoded_y
        self._total_weight = weights.sum()
        self._init_storage()
        self._grow(np.arange(len(labels)), depth=0)
        del self._X, self._Y, self._encoded_y
        self._finalize_storage()
        self.full_tree_n_leaves_ = self.n_leaves_
        if self.ccp_alpha > 0:
            self._prune_in_place(self.ccp_alpha)
        return self

    def _init_storage(self) -> None:
        self._feature, self._threshold, self._missing_left = [], [], []
        self._left, self._right, self._value, self._n_samples = [], [], [], []
        self._impurity, self._depth, self._left_categories = [], [], {}
        self._class_counts = []

    def _finalize_storage(self) -> None:
        self.feature_ = np.asarray(self._feature, dtype=np.int64)
        self.threshold_ = np.asarray(self._threshold, dtype=np.float64)
        self.missing_left_ = np.asarray(self._missing_left, dtype=bool)
        self.children_left_ = np.asarray(self._left, dtype=np.int64)
        self.children_right_ = np.asarray(self._right, dtype=np.int64)
        self.value_ = np.asarray(self._value, dtype=np.float64)
        self.n_node_samples_ = np.asarray(self._n_samples, dtype=np.int64)
        self.impurity_ = np.asarray(self._impurity, dtype=np.float64)
        self.node_depth_ = np.asarray(self._depth, dtype=np.int64)
        # Jumlah baris asli per kelas (tanpa class_weight), untuk ringkasan dan visualisasi.
        self.node_class_counts_ = np.asarray(self._class_counts, dtype=np.int64)
        self.left_categories_ = self._left_categories
        for name in ("_feature", "_threshold", "_missing_left", "_left", "_right", "_value",
                     "_n_samples", "_impurity", "_depth", "_left_categories", "_class_counts"):
            delattr(self, name)

    @staticmethod
    def _gini(counts: np.ndarray) -> np.ndarray:
        """Gini = 1 - sum p_k^2 untuk setiap baris tabel bobot kelas."""
        totals = counts.sum(axis=-1)
        with np.errstate(divide="ignore", invalid="ignore"):
            gini = 1.0 - np.sum(counts**2, axis=-1) / totals**2
        return np.where(totals > 0, gini, 0.0)

    def _new_node(self, indices: np.ndarray, depth: int, counts: np.ndarray, impurity: float) -> int:
        node = len(self._feature)
        self._feature.append(_LEAF)
        self._threshold.append(np.nan)
        self._missing_left.append(True)
        self._left.append(_LEAF)
        self._right.append(_LEAF)
        self._value.append(counts)
        self._n_samples.append(len(indices))
        self._class_counts.append(np.bincount(self._encoded_y[indices], minlength=len(self.classes_)))
        self._impurity.append(impurity)
        self._depth.append(depth)
        return node

    def _grow(self, indices: np.ndarray, depth: int) -> int:
        counts = self._Y[indices].sum(axis=0)
        impurity = float(self._gini(counts))
        node = self._new_node(indices, depth, counts, impurity)
        n = len(indices)
        if (
            impurity <= _EPS
            or n < self.min_samples_split
            or n < 2 * self.min_samples_leaf
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return node
        split = self._best_split(indices, counts, impurity)
        if split is None:
            return node
        feature, threshold, left_set, missing_left, go_left = split
        self._feature[node] = feature
        self._threshold[node] = threshold
        self._missing_left[node] = missing_left
        if left_set is not None:
            self._left_categories[node] = left_set
        self._left[node] = self._grow(indices[go_left], depth + 1)
        self._right[node] = self._grow(indices[~go_left], depth + 1)
        return node

    def _best_split(self, indices: np.ndarray, counts: np.ndarray, impurity: float):
        """Mencari split dengan penurunan Gini berbobot terbesar pada semua fitur."""
        X, Y = self._X[indices], self._Y[indices]
        node_weight = counts.sum()
        min_leaf = self.min_samples_leaf
        best_gain = max(self.min_impurity_decrease * self._total_weight / node_weight, _EPS)
        best = None
        for feature in range(X.shape[1]):
            column = X[:, feature]
            missing = np.isnan(column)
            n_missing = int(missing.sum())
            present = np.flatnonzero(~missing)
            if len(present) == 0:
                continue
            missing_counts = Y[missing].sum(axis=0)
            values = column[present]

            # Kandidat terakhir menaruh semua nilai terisi di kiri, sehingga nilai kosong
            # bisa dipisah sendiri di kanan. Tanpa nilai kosong, kandidat ini gugur oleh min_leaf.
            if self.is_categorical_[feature]:
                codes = values.astype(np.int64)
                n_codes = len(self.categories_[feature])
                per_code = np.zeros((n_codes, Y.shape[1]))
                np.add.at(per_code, codes, Y[present])
                raw_per_code = np.bincount(codes, minlength=n_codes)
                observed = np.flatnonzero(raw_per_code > 0)
                # Urutan Breiman: proporsi bobot kelas positif per kategori.
                share = per_code[observed, -1] / per_code[observed].sum(axis=1)
                order = observed[np.argsort(share, kind="mergesort")]
                left_counts = np.cumsum(per_code[order], axis=0)
                left_n = np.cumsum(raw_per_code[order])
            else:
                order = np.argsort(values, kind="mergesort")
                sorted_values = values[order]
                boundaries = np.flatnonzero(sorted_values[:-1] < sorted_values[1:])
                boundaries = np.append(boundaries, len(present) - 1)
                left_counts = np.cumsum(Y[present][order], axis=0)[boundaries]
                left_n = boundaries + 1

            present_counts = counts - missing_counts
            present_n = len(present)
            for missing_left in (True, False):
                lc = left_counts + (missing_counts if missing_left else 0)
                rc = present_counts - left_counts + (0 if missing_left else missing_counts)
                ln = left_n + (n_missing if missing_left else 0)
                rn = present_n - left_n + (0 if missing_left else n_missing)
                valid = (ln >= min_leaf) & (rn >= min_leaf)
                if not valid.any():
                    continue
                lw, rw = lc.sum(axis=1), rc.sum(axis=1)
                child = (lw * self._gini(lc) + rw * self._gini(rc)) / node_weight
                gain = np.where(valid, impurity - child, -np.inf)
                position = int(np.argmax(gain))
                if gain[position] > best_gain + _EPS:
                    best_gain = gain[position]
                    if self.is_categorical_[feature]:
                        left_set = np.zeros(len(self.categories_[feature]), dtype=bool)
                        left_set[order[: position + 1]] = True
                        codes_or_zero = np.where(missing, 0, column).astype(np.int64)
                        go_left = np.where(missing, missing_left, left_set[codes_or_zero])
                        best = (feature, np.nan, left_set, missing_left, go_left)
                    else:
                        threshold = self._midpoint(sorted_values, boundaries[position])
                        go_left = np.where(missing, missing_left, column <= threshold)
                        best = (feature, threshold, None, missing_left, go_left)
        return best

    @staticmethod
    def _midpoint(sorted_values: np.ndarray, boundary: int) -> float:
        """Threshold di antara dua nilai terurut; ``inf`` untuk split nilai terisi vs kosong."""
        if boundary == len(sorted_values) - 1:
            return np.inf  # sama seperti scikit-learn untuk split terisi vs kosong
        lower, upper = sorted_values[boundary], sorted_values[boundary + 1]
        threshold = lower + (upper - lower) / 2
        # Jika tidak ada titik tengah yang dapat direpresentasikan, pakai nilai bawah.
        return float(lower if threshold >= upper else threshold)

    # ------------------------------------------------------------------ prediksi
    def _check_fitted(self) -> None:
        if not hasattr(self, "value_"):
            raise RuntimeError("Call fit before predict")

    def apply(self, X: np.ndarray) -> np.ndarray:
        """Mengembalikan indeks daun untuk setiap baris; seluruh baris ditelusuri secara vektor."""
        self._check_fitted()
        encoded = self._encode(X, fitting=False)
        nodes = np.zeros(len(encoded), dtype=np.int64)
        active = np.flatnonzero(self.feature_[nodes] != _LEAF)
        while len(active):
            current = nodes[active]
            features = self.feature_[current]
            values = encoded[active, features]
            go_left = np.empty(len(active), dtype=bool)
            unknown = np.isnan(values)
            categorical = self.is_categorical_[features]
            unknown |= categorical & (values < 0)
            numeric = ~categorical & ~unknown
            go_left[numeric] = values[numeric] <= self.threshold_[current[numeric]]
            for position in np.flatnonzero(categorical & ~unknown):
                go_left[position] = self.left_categories_[current[position]][int(values[position])]
            go_left[unknown] = self.missing_left_[current[unknown]]
            nodes[active] = np.where(go_left, self.children_left_[current], self.children_right_[current])
            active = active[self.feature_[nodes[active]] != _LEAF]
        return nodes

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Proporsi bobot kelas pada daun, urut sesuai ``classes_``."""
        leaves = self.apply(X)
        values = self.value_[leaves]
        return values / values.sum(axis=1, keepdims=True)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Kelas mayoritas daun; jika ``threshold`` diisi, kelas kedua dipilih saat peluangnya >= threshold."""
        probabilities = self.predict_proba(X)
        if self.threshold is not None and len(self.classes_) == 2:
            return self.classes_[(probabilities[:, 1] >= self.threshold).astype(int)]
        return self.classes_[np.argmax(probabilities, axis=1)]

    # ------------------------------------------------------------------ ringkasan pohon
    @property
    def n_leaves_(self) -> int:
        self._check_fitted()
        return int(np.sum(self.feature_ == _LEAF))

    @property
    def depth_(self) -> int:
        self._check_fitted()
        return int(self.node_depth_.max())

    @property
    def feature_importances_(self) -> np.ndarray:
        """Total penurunan Gini berbobot per fitur, dinormalisasi agar berjumlah 1."""
        self._check_fitted()
        importances = np.zeros(self.n_features_in_)
        weights = self.value_.sum(axis=1)
        for node in np.flatnonzero(self.feature_ != _LEAF):
            left, right = self.children_left_[node], self.children_right_[node]
            importances[self.feature_[node]] += (
                weights[node] * self.impurity_[node]
                - weights[left] * self.impurity_[left]
                - weights[right] * self.impurity_[right]
            )
        total = importances.sum()
        return importances / total if total > 0 else importances

    def describe_split(self, node: int, feature_names: list[str] | None = None) -> str:
        """Teks kondisi cabang kiri, misalnya ``days_before_arrival <= 120.5``."""
        feature = int(self.feature_[node])
        if feature == _LEAF:
            return "daun"
        names = feature_names if feature_names is not None else self.feature_names_in_
        name = str(names[feature]) if names is not None else f"x[{feature}]"
        if self.is_categorical_[feature]:
            chosen = self.categories_[feature][self.left_categories_[node]]
            return f"{name} in {{{', '.join(map(str, chosen))}}}"
        if np.isinf(self.threshold_[node]):
            return f"{name} terisi"
        return f"{name} <= {self.threshold_[node]:.6g}"

    # ------------------------------------------------------------------ cost-complexity pruning
    # Minimal cost-complexity pruning: Breiman dkk. (1984), bab 3, dan
    # https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning
    # R(t) = (W_t / W) * Gini(t); alpha efektif g(t) = (R(t) - R(T_t)) / (|daun(T_t)| - 1).
    def _subtree_stats(self, is_leaf: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        risk = self.value_.sum(axis=1) / self.value_[0].sum() * self.impurity_
        subtree_risk = risk.copy()
        leaves = np.ones(len(risk))
        parent = np.full(len(risk), _LEAF)
        internal = np.flatnonzero(~is_leaf)
        parent[self.children_left_[internal]] = internal
        parent[self.children_right_[internal]] = internal
        for node in np.argsort(-self.node_depth_, kind="stable"):  # pasca-urutan: anak sebelum induk
            if not is_leaf[node]:
                left, right = self.children_left_[node], self.children_right_[node]
                subtree_risk[node] = subtree_risk[left] + subtree_risk[right]
                leaves[node] = leaves[left] + leaves[right]
        return risk, subtree_risk, leaves, parent

    def _pruning_sequence(self):
        """Menghasilkan (alpha, simpul yang dipangkas, risiko total) secara berurutan (weakest link)."""
        is_leaf = self.feature_ == _LEAF
        alive = np.ones(len(is_leaf), dtype=bool)
        risk, subtree_risk, leaves, parent = self._subtree_stats(is_leaf)
        while not is_leaf[0]:
            candidates = np.flatnonzero(alive & ~is_leaf)
            effective = (risk[candidates] - subtree_risk[candidates]) / (leaves[candidates] - 1)
            node = candidates[int(np.argmin(effective))]
            alpha = max(float(effective.min()), 0.0)
            # Hapus seluruh keturunan simpul, lalu jadikan simpul tersebut daun.
            stack = [self.children_left_[node], self.children_right_[node]]
            while stack:
                child = stack.pop()
                alive[child] = False
                if not is_leaf[child]:
                    stack.extend((self.children_left_[child], self.children_right_[child]))
            is_leaf[node] = True
            delta_risk = risk[node] - subtree_risk[node]
            delta_leaves = leaves[node] - 1
            ancestor = node
            while ancestor != _LEAF:
                subtree_risk[ancestor] += delta_risk
                leaves[ancestor] -= delta_leaves
                ancestor = parent[ancestor]
            yield alpha, node, float(subtree_risk[0])

    def cost_complexity_pruning_path(self) -> dict[str, np.ndarray]:
        """Barisan alpha efektif dan total impurity daun dari pohon yang sudah di-fit."""
        self._check_fitted()
        is_leaf = self.feature_ == _LEAF
        _, subtree_risk, _, _ = self._subtree_stats(is_leaf)
        alphas, impurities = [0.0], [float(subtree_risk[0])]
        for alpha, _, total in self._pruning_sequence():
            alphas.append(alpha)
            impurities.append(total)
        return {"ccp_alphas": np.maximum.accumulate(np.asarray(alphas)), "impurities": np.asarray(impurities)}

    def _prune_in_place(self, alpha: float) -> None:
        pruned = []
        for effective, node, _ in self._pruning_sequence():
            if effective > alpha:
                break
            pruned.append(node)
        for node in pruned:
            self.feature_[node] = _LEAF
        self._compact()

    def pruned(self, alpha: float, threshold: float | None | object = _KEEP) -> "DecisionTree":
        """Salinan pohon yang dipangkas dengan ``alpha`` tanpa training ulang.

        Karena pruning hanya membuang subpohon, satu pohon penuh cukup ditumbuhkan sekali
        lalu dipangkas untuk banyak nilai alpha (berguna untuk cross-validation).
        """
        self._check_fitted()
        copy = object.__new__(DecisionTree)
        copy.__dict__ = {
            key: (value.copy() if isinstance(value, np.ndarray) else
                  dict(value) if isinstance(value, dict) else value)
            for key, value in self.__dict__.items()
        }
        copy.ccp_alpha = float(alpha)
        if threshold is not _KEEP:
            copy.threshold = threshold
        copy._prune_in_place(float(alpha))
        return copy

    def _compact(self) -> None:
        """Membuang simpul yang tidak lagi terjangkau dari akar setelah pruning."""
        order, stack = [], [0]
        while stack:
            node = stack.pop()
            order.append(node)
            if self.feature_[node] != _LEAF:
                stack.extend((self.children_right_[node], self.children_left_[node]))
        order = np.asarray(order)
        mapping = np.full(len(self.feature_), _LEAF)
        mapping[order] = np.arange(len(order))
        is_leaf = self.feature_[order] == _LEAF
        self.children_left_ = np.where(is_leaf, _LEAF, mapping[self.children_left_[order]])
        self.children_right_ = np.where(is_leaf, _LEAF, mapping[self.children_right_[order]])
        self.left_categories_ = {
            int(mapping[node]): mask for node, mask in self.left_categories_.items()
            if mapping[node] != _LEAF and self.feature_[node] != _LEAF
        }
        for name in ("feature_", "threshold_", "missing_left_", "value_", "n_node_samples_",
                     "impurity_", "node_depth_", "node_class_counts_"):
            setattr(self, name, getattr(self, name)[order])


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    numbers = rng.normal(size=(400, 2))
    labels = (numbers[:, 0] + numbers[:, 1] > 0).astype(int)
    tree = DecisionTree().fit(numbers, labels)
    assert (tree.predict(numbers) == labels).all() and tree.n_leaves_ > 2
    path = tree.cost_complexity_pruning_path()
    assert np.all(np.diff(path["ccp_alphas"]) >= 0) and np.all(np.diff(path["impurities"]) >= -1e-12)
    assert tree.pruned(path["ccp_alphas"][-1]).n_leaves_ == 1
    print("PASS: numeric splits and weakest-link pruning path")

    # Kategori biner optimal: {a, c} vs {b, d} tidak bisa didapat dari satu kategori vs sisanya.
    mixed = np.array([["a", 1.0], ["b", np.nan], ["c", 2.0], ["d", np.nan]] * 25, dtype=object)
    target = np.array([1, 0, 1, 0] * 25)
    tree = DecisionTree(max_depth=1).fit(mixed, target)
    assert (tree.predict(mixed) == target).all()
    assert tree.predict(np.array([["NEW", np.nan]], dtype=object)).shape == (1,)
    assert np.allclose(tree.predict_proba(mixed).sum(axis=1), 1)
    print("PASS: optimal categorical split, missing values, and unseen categories")

    # Nilai terisi vs kosong harus bisa menjadi split sendiri, seperti pada scikit-learn.
    target = np.array([0, 0, 1, 1])
    for column in ([1.0, 1.0, np.nan, np.nan], [1.0, 2.0, np.nan, np.nan]):
        values = np.array(column).reshape(-1, 1)
        tree = DecisionTree().fit(values, target)
        assert (tree.predict(values) == target).all() and tree.n_leaves_ == 2
        assert tree.predict(np.array([[5.0]])) == 0
    categories = np.array([["a"], ["b"], [None], [None]], dtype=object)
    assert (DecisionTree().fit(categories, target).predict(categories) == target).all()
    print("PASS: present vs missing split")
