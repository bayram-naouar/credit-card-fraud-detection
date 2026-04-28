import json
import os

import numpy as np
from sklearn.metrics import precision_score, recall_score

from src.config import PARAM_SPACE, PATHS, PREPROCESSING, ROOT, TRAINING
from src.models.registry import MODEL_REGISTRY


def score(y_true, y_pred):
    recall = recall_score(y_true, y_pred, zero_division=0)
    precision = precision_score(y_true, y_pred, zero_division=0)
    weights = TRAINING["scoring_weights"]
    return weights["recall"] * recall + weights["precision"] * precision


def train(model_name, tune=False):
    model_name = model_name.lower()
    if not model_name in MODEL_REGISTRY:
        raise ValueError(f"Unsupported model: {model_name}")
    model_class, X_file, tuning_sample_rate, final_sample_rate, need_input_dim = (
        MODEL_REGISTRY[model_name]
    )
    try:
        X_train = np.load(ROOT / PATHS["processed_dir"] / X_file)
    except FileNotFoundError:
        raise FileNotFoundError("Run preprocessing first.")
    rng = np.random.default_rng(PREPROCESSING["random_state"])
    if final_sample_rate < 1.0:
        idx_final = rng.choice(
            len(X_train), int(len(X_train) * final_sample_rate), replace=False
        )
        X_train_final = X_train[idx_final]
        print(
            f"[training] Subsampling {final_sample_rate * 100:.0f}% of training data for final fit ({len(X_train_final)} rows)."
        )
    else:
        X_train_final = X_train
    if not tune:
        model = (
            model_class(input_dim=X_train_final.shape[1])
            if need_input_dim
            else model_class()
        )
        print(f"[training] Training {model_name}...")
        model.fit(X_train_final)
        model.save()
        print(f"[training] Model {model_name} trained and saved.")
        return
    X_val = np.load(ROOT / PATHS["processed_dir"] / "X_val.npy")
    y_val = np.load(ROOT / PATHS["processed_dir"] / "y_val.npy")
    print(
        f"[tuning] Sampling {tuning_sample_rate * 100:.1f}% of training data for tuning..."
    )
    idx_train = rng.choice(
        len(X_train), int(len(X_train) * tuning_sample_rate), replace=False
    )
    idx_val = rng.choice(
        len(X_val), int(len(X_val) * tuning_sample_rate), replace=False
    )
    X_train_fit = X_train[idx_train]
    X_val_tune = X_val[idx_val]
    y_val_tune = y_val[idx_val]
    print(f"[tuning] Tuning {model_name} with random search...")
    n_iterations = TRAINING["random_search_iterations"]
    param_space = PARAM_SPACE[model_name]
    best_score = float("-inf")
    best_params = {}
    best_threshold = None
    for i in range(n_iterations):
        param_set = {k: rng.choice(v).item() for k, v in param_space.items()}
        params_display = {
            k: round(v, 4) if isinstance(v, float) else v for k, v in param_set.items()
        }
        print(f"[tuning] Iteration {i + 1}/{n_iterations} | {params_display}")
        model = (
            model_class(X_train_fit.shape[1], **param_set)
            if need_input_dim
            else model_class(**param_set)
        )
        model.fit(X_train_fit)
        scores = model.anomaly_scores(X_val_tune)
        threshold = float(np.percentile(scores, param_set["threshold_percentile"]))
        y_pred = (scores > threshold).astype(int)
        current_score = score(y_val_tune, y_pred)
        if current_score > best_score:
            best_score = current_score
            best_params = param_set
            best_threshold = threshold
            print(f"[tuning] New best → score: {best_score:.4f} | {params_display}")
    print(f"[tuning] Best score: {best_score}, Best params: {best_params}")
    best_params_dir = ROOT / PATHS["best_params_dir"]
    os.makedirs(best_params_dir, exist_ok=True)
    best_params_path = best_params_dir / f"{model_name}_best_params.json"
    if best_params_path.exists():
        print(f"[tuning] Warning: overwriting existing best params for {model_name}")
    with open(best_params_path, "w") as f:
        best_params_serializable = {
            k: v.item() if hasattr(v, "item") else v for k, v in best_params.items()
        }
        json.dump(
            {
                "best_score": best_score,
                "best_threshold": best_threshold,
                "best_params": best_params_serializable,
            },
            f,
            indent=4,
        )
    print(
        f"[tuning] Best parameters saved to {best_params_dir / f'{model_name}_best_params.json'}"
    )
    model = (
        model_class(X_train_final.shape[1], **best_params)
        if need_input_dim
        else model_class(**best_params)
    )
    model.fit(X_train_final)
    model.threshold = best_threshold
    model.save()
    print(f"[tuning] Model {model_name} trained with best parameters and saved.")


def main():
    # train("isolation_forest", True)
    # train("one_class_svm", True)
    train("autoencoder", True)


if __name__ == "__main__":
    main()
