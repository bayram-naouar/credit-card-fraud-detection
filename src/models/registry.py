from src.models.autoencoder import AutoEncoderModel
from src.models.isolation_forest import IsolationForestModel
from src.models.one_class_svm import OneClassSVMModel

# (class, X_file, tuning_sample_rate, final_sample_rate, need_input_dim)
MODEL_REGISTRY = {
    "isolation_forest": (IsolationForestModel, "X_train.npy", 1.0, 1.0, False),
    "one_class_svm": (OneClassSVMModel, "X_train.npy", 0.1, 0.5, False),
    "autoencoder": (AutoEncoderModel, "X_legit.npy", 1.0, 1.0, True),
}
