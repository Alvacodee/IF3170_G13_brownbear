from numbers import Integral, Real
from typing import Self

import numpy as np

from .base import BaseModel


class KNN(BaseModel):
    """Klasifikasi data numerik berdasarkan k tetangga terdekat.

    Parameter
    ---------
    k : int, bawaan=5
        Jumlah tetangga; minimal 1 dan maksimal jumlah data latih.
    metric : {"euclidean", "manhattan", "minkowski"}, bawaan="euclidean"
        Jenis jarak untuk menentukan tetangga terdekat.
    weights : {"uniform", "distance"}, bawaan="uniform"
        Voting dengan suara sama besar atau bobot kebalikan jarak.
    p : float, bawaan=2.0
        Pangkat Minkowski yang bernilai hingga dan minimal 1;
        diabaikan untuk jenis jarak lainnya.
    vote_power : float, bawaan=1.0
        Pangkat kebalikan jarak pada ``weights="distance"``.
    threshold : float atau None, bawaan=None
        Ambang peluang kelas positif untuk klasifikasi biner; ``None`` memakai suara terbanyak.
    mismatch_groups : sequence of (list[int], float), bawaan=()
        Kelompok kolom diskret dan penaltinya. Penalti ditambahkan sekali jika ada
        kolom kelompok yang berbeda. Nilai kosong tidak dihitung berbeda.
    tolerances : sequence of (int, float, float), bawaan=()
        Kolom, toleransi, dan penalti. Penalti ditambahkan jika selisih melebihi toleransi.

    Kolom yang dipakai penalti tidak ikut dihitung pada ``metric`` dan boleh berisi NaN.
    """

    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_chunked.html
    _BATCH_SIZE = 128

    def __init__(
        self,
        k: int = 5,
        metric: str = "euclidean",
        weights: str = "uniform",
        p: float = 2.0,
        vote_power: float = 1.0,
        threshold: float | None = None,
        mismatch_groups: tuple = (),
        tolerances: tuple = (),
    ) -> None:
        self.k = k
        self.metric = metric
        self.weights = weights
        self.p = p
        self.vote_power = vote_power
        self.threshold = threshold
        self.mismatch_groups = mismatch_groups
        self.tolerances = tolerances
        self._validate_parameters()

    def _validate_parameters(self) -> None:
        if isinstance(self.k, (bool, np.bool_)) or not isinstance(self.k, Integral) or self.k < 1:
            raise ValueError("k must be a positive integer")
        if self.metric not in ("euclidean", "manhattan", "minkowski"):
            raise ValueError("metric must be 'euclidean', 'manhattan', or 'minkowski'")
        if self.weights not in ("uniform", "distance"):
            raise ValueError("weights must be 'uniform' or 'distance'")
        if (
            isinstance(self.p, (bool, np.bool_))
            or not isinstance(self.p, Real)
            or not np.isfinite(self.p)
            or self.p < 1
        ):
            raise ValueError("p must be a finite number >= 1")
        if not np.isfinite(self.vote_power) or self.vote_power <= 0:
            raise ValueError("vote_power must be a positive finite number")
        if self.threshold is not None and not 0 < self.threshold < 1:
            raise ValueError("threshold must be None or between 0 and 1")
        if any(penalty < 0 for _, penalty in self.mismatch_groups) or any(
            tolerance < 0 or penalty < 0 for _, tolerance, penalty in self.tolerances
        ):
            raise ValueError("Penalties and tolerances must be nonnegative")

    def _penalty_columns(self) -> list[int]:
        columns = [column for group, _ in self.mismatch_groups for column in group]
        return columns + [column for column, _, _ in self.tolerances]

    def _validate_X(self, X: np.ndarray) -> np.ndarray:
        raw = np.asarray(X)
        if np.iscomplexobj(raw):
            raise ValueError("X must contain real numeric features")
        try:
            array = np.asarray(X, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("X must contain numeric features; encode categories first") from exc
        if array.ndim != 2 or array.shape[1] == 0:
            raise ValueError("X must be a 2D array with at least one feature")
        penalty = self._penalty_columns()
        if penalty and max(penalty) >= array.shape[1]:
            raise ValueError("Penalty columns must exist in X")
        if not np.isfinite(np.delete(array, penalty, axis=1)).all() or np.isinf(array).any():
            raise ValueError("X must contain only finite values; impute missing values first")
        return array

    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        """Menyimpan salinan fitur dan label data latih."""
        self._validate_parameters()
        X_array = self._validate_X(X)
        y_array = np.asarray(y)
        if X_array.shape[0] == 0:
            raise ValueError("Training data must contain at least one row")
        if y_array.ndim != 1 or len(y_array) != len(X_array):
            raise ValueError("y must be 1D with one label per row of X")
        if y_array.dtype.kind not in "biufUS":
            raise ValueError("y must contain numeric or string class labels")
        if y_array.dtype.kind in "iuf" and not np.isfinite(y_array).all():
            raise ValueError("y must not contain NaN or infinite labels")
        if self.k > len(X_array):
            raise ValueError("k must not exceed the number of training rows")
        classes, encoded_y = np.unique(y_array, return_inverse=True)
        self.distance_columns_ = np.setdiff1d(np.arange(X_array.shape[1]), self._penalty_columns())
        self.X_train_ = X_array.copy()
        self.y_train_ = y_array.copy()
        self.classes_ = classes
        self._encoded_y = encoded_y
        self.n_features_in_ = X_array.shape[1]
        return self

    def _validate_query(self, X: np.ndarray) -> np.ndarray:
        if not hasattr(self, "X_train_"):
            raise RuntimeError("KNN is not fitted; call fit before prediction")
        self._validate_parameters()
        if self.k > len(self.X_train_):
            raise ValueError("k must not exceed the number of training rows")
        array = self._validate_X(X)
        if array.shape[1] != self.n_features_in_:
            raise ValueError(f"Expected {self.n_features_in_} features, got {array.shape[1]}")
        return array

    def _distances(self, X: np.ndarray) -> np.ndarray:
        """Menghitung jarak data masukan ke seluruh data latih per fitur."""
        exponent = {"euclidean": 2.0, "manhattan": 1.0}.get(self.metric, self.p)
        distances = np.zeros((len(X), len(self.X_train_)), dtype=np.float64)
        difference = np.empty_like(distances)
        with np.errstate(over="raise", invalid="raise"):
            try:
                for feature in self.distance_columns_:
                    np.subtract(X[:, feature, None], self.X_train_[None, :, feature], out=difference)
                    np.abs(difference, out=difference)
                    if exponent != 1:
                        np.power(difference, exponent, out=difference)
                    distances += difference
                if exponent != 1:
                    np.power(distances, 1.0 / exponent, out=distances)
            except FloatingPointError as exc:
                raise ValueError("Distance overflow; scale features before using KNN") from exc
        for columns, penalty in self.mismatch_groups:
            differs = np.zeros(distances.shape, dtype=bool)
            for column in columns:
                query, train = X[:, column, None], self.X_train_[None, :, column]
                differs |= (query != train) & ~np.isnan(query) & ~np.isnan(train)
            distances += penalty * differs
        # Perbandingan dengan NaN bernilai False sehingga nilai kosong tidak diberi penalti.
        for column, tolerance, penalty in self.tolerances:
            distances += penalty * (np.abs(X[:, column, None] - self.X_train_[None, :, column]) > tolerance)
        return distances

    def kneighbors(
        self, X: np.ndarray, n_neighbors: int | None = None, return_distance: bool = True
    ) -> tuple[np.ndarray, np.ndarray] | np.ndarray:
        """Mengembalikan indeks tetangga terdekat dan, jika diminta, jaraknya.

        Jarak yang seri diurutkan sesuai urutan data latih. Perhitungan dibagi
        dalam batch internal untuk membatasi memori matriks jarak.

        Referensi konsep perhitungan jarak secara bertahap:
        https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise_distances_chunked.html
        """
        array = self._validate_query(X)
        count = self.k if n_neighbors is None else n_neighbors
        if (
            isinstance(count, (bool, np.bool_))
            or not isinstance(count, Integral)
            or not 1 <= count <= len(self.X_train_)
        ):
            raise ValueError("n_neighbors must be an integer between 1 and the training size")
        indices = np.empty((len(array), count), dtype=np.intp)
        neighbor_distances = np.empty((len(array), count), dtype=np.float64)
        for start in range(0, len(array), self._BATCH_SIZE):
            stop = min(start + self._BATCH_SIZE, len(array))
            distances = self._distances(array[start:stop])
            nearest = np.argsort(distances, axis=1, kind="stable")[:, :count]
            indices[start:stop] = nearest
            neighbor_distances[start:stop] = np.take_along_axis(distances, nearest, axis=1)
        if return_distance:
            return neighbor_distances, indices
        return indices

    # Bobot jarak: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Menghitung proporsi suara tiap kelas sesuai urutan ``classes_``.

        Pada ``weights="distance"``, suara dibobotkan dengan kebalikan jarak.
        Jika ada tetangga berjarak nol, hanya tetangga tersebut yang memberi suara.

        Referensi voting berbobot jarak:
        https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        """
        array = self._validate_query(X)
        probabilities = np.empty((len(array), len(self.classes_)), dtype=np.float64)
        for start in range(0, len(array), self._BATCH_SIZE):
            stop = min(start + self._BATCH_SIZE, len(array))
            distances, indices = self.kneighbors(array[start:stop])
            labels = self._encoded_y[indices]
            votes = np.ones_like(distances)
            if self.weights == "distance":
                zeros = distances == 0
                exact_match = zeros.any(axis=1)
                votes[exact_match] = zeros[exact_match]
                remaining = ~exact_match
                if remaining.any():
                    votes[remaining] = (distances[remaining, :1] / distances[remaining]) ** self.vote_power
            totals = np.zeros((stop - start, len(self.classes_)), dtype=np.float64)
            for class_index in range(len(self.classes_)):
                totals[:, class_index] = np.sum(votes * (labels == class_index), axis=1)
            probabilities[start:stop] = totals / totals.sum(axis=1, keepdims=True)
        return probabilities

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Memilih kelas dengan suara terbanyak; suara seri memilih kelas terkecil.

        Jika ``threshold`` diisi pada data biner, kelas kedua dipilih saat peluangnya
        minimal sebesar ambang tersebut.
        """
        probabilities = self.predict_proba(X)
        if self.threshold is not None and len(self.classes_) == 2:
            return self.classes_[(probabilities[:, 1] >= self.threshold).astype(int)]
        return self.classes_[np.argmax(probabilities, axis=1)]
