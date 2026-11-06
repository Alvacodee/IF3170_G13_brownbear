import pickle
from pathlib import Path
from typing import Self

import numpy as np


class BaseModel:
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> Self:
        with Path(path).open("rb") as f:
            model = pickle.load(f)
        if not isinstance(model, cls):
            raise TypeError(f"Expected {cls.__name__}, got {type(model).__name__}")
        return model
