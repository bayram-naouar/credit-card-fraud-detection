from sklearn.ensemble import IsolationForest

from src.config import MODELS, PREPROCESSING
from src.models.base import BaseModel


class IsolationForestModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        base_params = MODELS["isolation_forest"].copy()
        base_params["random_state"] = PREPROCESSING["random_state"]
        base_params.update(kwargs)
        self.params = base_params.copy()
        self.model_name = "isolation_forest"
        self.threshold = None
        self.model = IsolationForest(
            **{k: v for k, v in base_params.items() if k != "threshold_percentile"}
        )

    def anomaly_scores(self, X):
        return -self.model.decision_function(X)

    def _fit(self, X_train):
        self.model.fit(X_train)
