from abc import ABC, abstractmethod

import joblib
import numpy as np

from src.config import PATHS, ROOT


class BaseModel(ABC):
    def __init__(self):
        self.model = None
        self.model_name = None
        self.threshold = None

    def fit(self, X_train):
        self._fit(X_train)
        scores = self.anomaly_scores(X_train)
        self.threshold = float(
            np.percentile(scores, self.params["threshold_percentile"])
        )

    @abstractmethod
    def _fit(self, X_train):
        pass

    def predict(self, X):
        if self.threshold is None:
            raise RuntimeError("Threshold is not set. Call fit() before predict().")
        scores = self.anomaly_scores(X)
        return (scores > self.threshold).astype(int)

    @abstractmethod
    def anomaly_scores(self, X):
        pass

    def save(self):
        if self.model is None:
            raise RuntimeError("No model to save. Train the model before saving.")
        joblib.dump(self, ROOT / PATHS["model_dir"] / f"{self.model_name}.pkl")

    def load(self):
        path = ROOT / PATHS["model_dir"] / f"{self.model_name}.pkl"
        if not path.exists():
            raise FileNotFoundError(
                f"Model file not found: {path}\nTrain the model before loading."
            )
        loaded = joblib.load(path)
        self.__dict__.update(loaded.__dict__)
