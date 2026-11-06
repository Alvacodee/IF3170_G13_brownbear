from typing import Self

import numpy as np

from .base import BaseModel


class KNN(BaseModel):
    def fit(self, X: np.ndarray, y: np.ndarray) -> Self:
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError
