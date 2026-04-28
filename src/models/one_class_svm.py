from sklearn.svm import OneClassSVM

from src.config import MODELS
from src.models.base import BaseModel


class OneClassSVMModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        kernel = kwargs.get("kernel", MODELS["one_class_svm"]["kernel"])
        base_params = MODELS["one_class_svm"]["kernel_params"][kernel].copy()
        base_params["kernel"] = kernel
        base_params["nu"] = MODELS["one_class_svm"]["nu"]
        base_params["threshold_percentile"] = MODELS["one_class_svm"][
            "threshold_percentile"
        ]
        base_params.update(kwargs)
        self.params = base_params.copy()
        self.model_name = "one_class_svm"
        self.threshold = None
        self.model = OneClassSVM(
            **{k: v for k, v in base_params.items() if k != "threshold_percentile"}
        )

    def anomaly_scores(self, X):
        return -self.model.decision_function(X)

    def _fit(self, X_train):
        self.model.fit(X_train)
