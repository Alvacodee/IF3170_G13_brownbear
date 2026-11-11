"""Shared preprocessing for accommodation booking data."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    FunctionTransformer,
    KBinsDiscretizer,
    OneHotEncoder,
    OrdinalEncoder,
    StandardScaler,
)

CATEGORICAL_COLUMNS = ["meal_package", "room_category", "booking_channel"]
BINARY_COLUMNS = ["needs_parking", "is_returning_guest"]
NUMERIC_COLUMNS = [
    "num_adults", "num_children", "weekend_nights", "weekday_nights",
    "days_before_arrival", "arrival_year", "arrival_month", "arrival_day",
    "prior_cancellations", "prior_completed_bookings", "nightly_rate_idr",
    "special_requests",
]
FEATURE_COLUMNS = NUMERIC_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS


def select_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Select the model features and validate the observed values."""
    if not frame.columns.is_unique:
        raise ValueError("Column names must be unique")
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing features: {sorted(missing)}")
    features = frame.loc[:, FEATURE_COLUMNS].copy()
    features = features.replace(r"^\s*$", np.nan, regex=True)
    for column in NUMERIC_COLUMNS + BINARY_COLUMNS:
        values = pd.to_numeric(features[column], errors="raise").astype(float)
        known = values.dropna()
        if not np.isfinite(known).all() or (known < 0).any() or (known % 1 != 0).any():
            raise ValueError(f"{column} must contain nonnegative finite integers")
        if column in BINARY_COLUMNS and not known.isin([0, 1]).all():
            raise ValueError(f"{column} must contain 0, 1, or missing values")
        features[column] = values
    for column in CATEGORICAL_COLUMNS:
        known = features[column].dropna()
        if not known.map(lambda value: isinstance(value, str)).all():
            raise ValueError(f"{column} must contain text categories")
        values = features[column].astype(object)
        features[column] = values.where(values.notna(), np.nan)
    dates = features[["arrival_year", "arrival_month", "arrival_day"]].rename(
        columns={"arrival_year": "year", "arrival_month": "month", "arrival_day": "day"}
    )
    for column, low, high in [("year", 1, 9999), ("month", 1, 12), ("day", 1, 31)]:
        if not dates[column].dropna().between(low, high).all():
            raise ValueError(f"Arrival date {column} is out of range")
    complete = dates.notna().all(axis=1)
    if pd.to_datetime(dates.loc[complete], errors="coerce").isna().any():
        raise ValueError("Invalid arrival date")
    return features


def split_training_data(frame: pd.DataFrame, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Hold out one fifth of the data without splitting identical feature rows."""
    # Source: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html
    features = select_features(frame)
    target = frame["is_canceled"]
    if target.isna().any() or set(target.unique()) != {0, 1}:
        raise ValueError("Training labels must contain both 0 and 1 without missing values")
    groups = features.groupby(FEATURE_COLUMNS, dropna=False, sort=False).ngroup()
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    return next(splitter.split(features, target, groups))


def make_basic_preprocessor() -> Pipeline:
    """Impute counts and rates with medians, flags with modes, and categories with MISSING."""
    # Source: https://scikit-learn.org/stable/modules/generated/sklearn.impute.SimpleImputer.html
    imputer = ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median", keep_empty_features=True), NUMERIC_COLUMNS),
            ("binary", SimpleImputer(strategy="most_frequent", keep_empty_features=True), BINARY_COLUMNS),
            ("categorical", SimpleImputer(strategy="constant", fill_value="MISSING", keep_empty_features=True), CATEGORICAL_COLUMNS),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")
    return Pipeline([("select", FunctionTransformer(select_features)), ("impute", imputer)])


def make_numeric_preprocessor(scale: bool = True) -> Pipeline:
    """Encode nominal features for KNN and logistic regression."""
    # Source: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OneHotEncoder.html
    numeric = NUMERIC_COLUMNS + BINARY_COLUMNS
    encoded = ColumnTransformer([
        ("numeric", StandardScaler() if scale else "passthrough", numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLUMNS),
    ])
    return Pipeline([("basic", make_basic_preprocessor()), ("encode", encoded)])


def _reserve_unknown_category(values: np.ndarray) -> np.ndarray:
    return values.astype(np.int64) + 1


def make_discrete_preprocessor(n_bins: int = 5) -> Pipeline:
    """Bin continuous features and encode the rest as nonnegative categories."""
    # Sources: https://scikit-learn.org/stable/modules/naive_bayes.html#categorical-naive-bayes
    # https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    binned = ["days_before_arrival", "nightly_rate_idr"]
    remaining = [column for column in NUMERIC_COLUMNS + BINARY_COLUMNS + CATEGORICAL_COLUMNS
                 if column not in binned]
    encoded = ColumnTransformer([
        ("binned", KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile", quantile_method="linear"), binned),
        ("categories", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), remaining),
    ])
    return Pipeline([
        ("basic", make_basic_preprocessor()),
        ("encode", encoded),
        ("reserve_unknown", FunctionTransformer(_reserve_unknown_category)),
    ])


DERIVED_COLUMNS = ["arrival_ordinal", "arrival_weekday", "total_nights", "total_guests", "log_lead", "log_rate"]
FLAGGED_COLUMNS = ["days_before_arrival", "nightly_rate_idr", "special_requests", "meal_package", "room_category"]


def add_svm_features(frame: pd.DataFrame) -> pd.DataFrame:
    features = select_features(frame)
    dates = pd.to_datetime(
        {"year": features["arrival_year"], "month": features["arrival_month"], "day": features["arrival_day"]}
    )
    derived = pd.DataFrame(
        {
            "arrival_ordinal": (dates - pd.Timestamp("2023-01-01")).dt.days,
            "arrival_weekday": dates.dt.day_of_week,
            "total_nights": features["weekend_nights"] + features["weekday_nights"],
            "total_guests": features["num_adults"] + features["num_children"],
            "log_lead": np.log1p(features["days_before_arrival"]),
            "log_rate": np.log1p(features["nightly_rate_idr"]),
        }
    )
    flags = features[FLAGGED_COLUMNS].isna().astype(float).add_prefix("missing_")
    return pd.concat([features, derived, flags], axis=1)


# Source: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
def make_svm_preprocessor() -> Pipeline:
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="MISSING")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    columns = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_COLUMNS + BINARY_COLUMNS + DERIVED_COLUMNS),
            ("categorical", categorical, CATEGORICAL_COLUMNS),
            ("flags", "passthrough", [f"missing_{column}" for column in FLAGGED_COLUMNS]),
        ]
    )
    return Pipeline([("features", FunctionTransformer(add_svm_features)), ("columns", columns)])


# Pemesanan bersaudara: tanggal datang dan lama menginap sama, sedangkan lead time dan harga sedikit berbeda.
SIBLING_KEY = ["arrival_year", "arrival_month", "arrival_day", "weekend_nights", "weekday_nights"]
SIBLING_ATTRIBUTES = [
    "booking_channel", "room_category", "meal_package", "num_adults", "num_children",
    "special_requests", "needs_parking",
]


# Imputasi dari tetangga terdekat: https://doi.org/10.1080/713827181
def impute_from_siblings(
    reference: pd.DataFrame, frame: pd.DataFrame, column: str = "days_before_arrival"
) -> pd.Series:
    """Fill missing values from the most similar training booking with the same arrival and stay."""
    values = frame[column].copy()
    missing = values.isna()
    if not missing.any():
        return values
    known = reference[reference[column].notna()]
    stays = dict(list(known.groupby(SIBLING_KEY)))
    arrivals = dict(list(known.groupby(SIBLING_KEY[:3])))
    fallback = known[column].median()
    for index in frame.index[missing]:
        row = frame.loc[index]
        group = stays.get(tuple(row[SIBLING_KEY]))
        if group is None:
            group = arrivals.get(tuple(row[SIBLING_KEY[:3]]))
        if group is None:
            values[index] = fallback
            continue
        score = np.zeros(len(group))
        for attribute in SIBLING_ATTRIBUTES:
            if pd.notna(row[attribute]):
                score += ((group[attribute] != row[attribute]) & group[attribute].notna()).to_numpy(float)
        if pd.notna(row["nightly_rate_idr"]):
            gap = np.abs(np.log1p(group["nightly_rate_idr"].to_numpy()) - np.log1p(row["nightly_rate_idr"])) / 0.05
            score += np.nan_to_num(gap, nan=2.0).clip(0, 3)
        values[index] = group[column].to_numpy()[np.argmin(score)]
    return values


CATEGORY_LEVELS = {
    "meal_package": ["Breakfast", "Room Only", "Half Board", "Full Board"],
    "room_category": ["Type A", "Type B", "Type C", "Type D", "Type E", "Type F", "Type G"],
    "booking_channel": ["Online Travel Agent", "Offline Agent", "Corporate", "Airline", "Complimentary"],
}
# Bobot hasil coordinate descent pada 5-fold CV; fitur berbobot 0 tidak dipakai.
# Sumber: https://doi.org/10.1007/s10107-015-0892-3
KNN_FEATURE_WEIGHTS = {
    "arrival_day": 0.5, "month_sin": 0.5, "month_cos": 0.3, "days_before_arrival": 1.0,
    "log_lead": 0.3, "booking_ordinal": 1.5, "nightly_rate_idr": 1.0, "log_rate": 1.0,
    "weekend_nights": 0.5, "weekday_nights": 1.0, "num_adults": 0.5, "needs_parking": 1.0,
    "is_returning_guest": 1.0, "prior_cancellations": 1.0, "prior_completed_bookings": 1.0,
    "special_requests": 2.0, "meal_package_Full Board": 1.0,
    "room_category_Type B": 1.0, "room_category_Type C": 0.5, "room_category_Type D": 1.0,
    "room_category_Type E": 1.0, "room_category_Type F": 1.0,
    "booking_channel_Online Travel Agent": 1.0, "booking_channel_Offline Agent": 0.6,
    "booking_channel_Airline": 1.0, "booking_channel_Complimentary": 1.0,
}
# Penalti jarak jika ada atribut kelompok yang berbeda; nilai kosong tidak dihitung berbeda.
KNN_MISMATCH_PENALTIES = {
    ("arrival_year", "arrival_month", "arrival_day"): 0.5,
    ("weekend_nights", "weekday_nights"): 0.5,
    ("num_adults", "num_children"): 0.5,
    ("room_category",): 0.2,
    ("meal_package",): 0.5,
    ("special_requests",): 0.1,
    ("needs_parking",): 0.1,
}
# (kolom, toleransi, penalti): selisih di atas toleransi menambah jarak.
KNN_TOLERANCES = [("days_before_arrival", 15.0, 1.5), ("log_rate", 0.05, 0.3)]


def add_knn_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add arrival, lead-time, and price transforms plus one-hot categories for KNN."""
    dates = pd.to_datetime(
        {"year": features["arrival_year"], "month": features["arrival_month"], "day": features["arrival_day"]}
    )
    month = 2 * np.pi * features["arrival_month"] / 12
    derived = pd.DataFrame({
        "month_sin": np.sin(month),
        "month_cos": np.cos(month),
        "log_lead": np.log1p(features["days_before_arrival"]),
        "booking_ordinal": (dates - pd.Timestamp("2023-01-01")).dt.days - features["days_before_arrival"],
        "log_rate": np.log1p(features["nightly_rate_idr"]),
    })
    onehot = pd.DataFrame({
        f"{column}_{level}": (features[column] == level).astype(float).where(features[column].notna())
        for column, levels in CATEGORY_LEVELS.items() for level in levels
    })
    return pd.concat([features[NUMERIC_COLUMNS + BINARY_COLUMNS], derived, onehot], axis=1).astype(float)


# Jarak campuran numerik dan kategorikal: https://doi.org/10.2307/2528823
class KNNPreprocessor:
    """Build the KNN matrix: weighted standardized features, then raw columns for distance penalties."""

    def __init__(self, feature_weights: dict[str, float] | None = None) -> None:
        self.feature_weights = KNN_FEATURE_WEIGHTS if feature_weights is None else feature_weights

    def fit(self, frame: pd.DataFrame, y: object = None) -> "KNNPreprocessor":
        features = select_features(frame)
        self.reference_ = features[features["days_before_arrival"].notna()].copy()
        self.columns_ = [column for column, weight in self.feature_weights.items() if weight > 0]
        weighted = add_knn_features(self._impute(features))[self.columns_]
        self.median_ = weighted.median()
        filled = weighted.fillna(self.median_)
        self.mean_, self.std_ = filled.mean(), filled.std().replace(0, 1)
        self.weights_ = np.array([self.feature_weights[column] for column in self.columns_])
        penalty_columns = [column for group in KNN_MISMATCH_PENALTIES for column in group]
        self.feature_names_out_ = self.columns_ + [f"raw_{c}" for c in penalty_columns] + [
            f"raw_{column}" for column, _, _ in KNN_TOLERANCES
        ]
        start = len(self.columns_)
        self.mismatch_groups_ = []
        for group, penalty in KNN_MISMATCH_PENALTIES.items():
            self.mismatch_groups_.append((list(range(start, start + len(group))), penalty))
            start += len(group)
        self.tolerances_ = [(start + i, tolerance, penalty) for i, (_, tolerance, penalty) in enumerate(KNN_TOLERANCES)]
        return self

    def _impute(self, features: pd.DataFrame) -> pd.DataFrame:
        return features.assign(days_before_arrival=impute_from_siblings(self.reference_, features))

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        features = select_features(frame)
        weighted = add_knn_features(self._impute(features))[self.columns_].fillna(self.median_)
        scaled = ((weighted - self.mean_) / self.std_).to_numpy() * self.weights_
        raw = features.assign(**{
            column: features[column].map({level: i for i, level in enumerate(levels)})
            for column, levels in CATEGORY_LEVELS.items()
        }).astype({column: float for column in CATEGORY_LEVELS})
        raw["log_rate"] = np.log1p(raw["nightly_rate_idr"])
        penalty = [column for group in KNN_MISMATCH_PENALTIES for column in group]
        penalty += [column for column, _, _ in KNN_TOLERANCES]
        return np.column_stack([scaled, raw[penalty].to_numpy(dtype=float)])

    def fit_transform(self, frame: pd.DataFrame, y: object = None) -> np.ndarray:
        return self.fit(frame).transform(frame)


def make_knn_preprocessor(feature_weights: dict[str, float] | None = None) -> KNNPreprocessor:
    return KNNPreprocessor(feature_weights)


if __name__ == "__main__":
    row = dict.fromkeys(FEATURE_COLUMNS, 0)
    row.update(arrival_year=2024, arrival_month=1, arrival_day=1, meal_package="Breakfast", room_category="Type B", booking_channel="Corporate")
    train = pd.DataFrame([row, {**row, "nightly_rate_idr": 100, "days_before_arrival": 10, "meal_package": "Room Only"}])
    holdout = pd.DataFrame([{**row, "nightly_rate_idr": np.nan, "room_category": "Type NEW"}], index=[17])
    basic = make_basic_preprocessor().fit(train).transform(holdout)
    assert basic.loc[17, "nightly_rate_idr"] == 50
    assert basic.loc[17, "room_category"] == "Type NEW"
    assert make_numeric_preprocessor().fit(train).transform(holdout).shape[0] == 1
    nb = make_discrete_preprocessor().fit(train).transform(holdout)
    assert nb.dtype.kind in "iu" and (nb >= 0).all()
    assert make_svm_preprocessor().fit(train).transform(holdout).shape[0] == 1
    knn_prep = make_knn_preprocessor().fit(train)
    assert np.isfinite(knn_prep.transform(holdout)[:, : len(knn_prep.columns_)]).all()
    print("PASS: train-only imputation and model feature matrices")
