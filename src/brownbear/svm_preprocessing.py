import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .preprocessing import (
    BINARY_COLUMNS,
    CATEGORICAL_COLUMNS,
    NUMERIC_COLUMNS,
    select_features,
)

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
            "arrival_weekday": dates.dt.dayofweek,
            "total_nights": features["weekend_nights"] + features["weekday_nights"],
            "total_guests": features["num_adults"] + features["num_children"],
            "log_lead": np.log1p(features["days_before_arrival"]),
            "log_rate": np.log1p(features["nightly_rate_idr"]),
        }
    )
    flags = features[FLAGGED_COLUMNS].isna().astype(float).add_prefix("missing_")
    return pd.concat([features, derived, flags], axis=1)


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
