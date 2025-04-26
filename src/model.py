import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from tqdm import tqdm
from itertools import product
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from time import perf_counter
from config import DATA_PROCESSED_DIR, MODELS_DIR

def load_data():
    # Load data
    X_train = np.load(f"{DATA_PROCESSED_DIR}/X_train.npy")
    X_test = np.load(f"{DATA_PROCESSED_DIR}/X_test.npy")
    y_train = np.load(f"{DATA_PROCESSED_DIR}/y_train.npy")
    y_test = np.load(f"{DATA_PROCESSED_DIR}/y_test.npy")
    return X_train, X_test, y_train, y_test

def hyperparameter_tuning(model_class):
    """
    Grid search tuning for IsolationForest and OneClassSVM.
    Always applies fast tuning for OneClassSVM using a subsample of training data.
    """
    # Load processed data
    X_train, X_test, _, y_test = load_data()

    # Define parameter grid
    if model_class == IsolationForest:
        param_grid = {
            "n_estimators": [200, 300, 400, 500],
            "max_samples": [0.5, 0.7, 1.0],
            "contamination": [0.0015, 0.002, 0.0025],
            "bootstrap": [True, False]
        }
    elif model_class == OneClassSVM:
        param_grid = {
            "kernel": ["rbf", "poly", "sigmoid"],
            "nu": [0.01, 0.05, 0.1, 0.5],
            "gamma": ["scale", "auto", 0.01, 0.1, 1.0],
            "degree": [3, 4, 5]  # Only relevant for 'poly'
        }
    else:
        raise ValueError("Unknown model class provided.")

    # Generate unique parameter combinations
    param_list = []
    seen = set()
    for values in product(*param_grid.values()):
        combo = dict(zip(param_grid.keys(), values))

        # Remove 'degree' if not using 'poly' kernel
        if model_class == OneClassSVM and combo.get('kernel') != 'poly':
            combo.pop('degree', None)

        frozen = frozenset(combo.items())
        if frozen not in seen:
            seen.add(frozen)
            param_list.append(combo)

    results = []

    for params in tqdm(param_list, desc="Tuning Progress"):
        start = perf_counter()

        # Use subsample for fast OCSVM training
        if model_class == OneClassSVM:
            sample_size = int(0.25 * len(X_train))
            X_train_sub = X_train[:sample_size]
            model = model_class(**params)
            model.fit(X_train_sub)
        else:
            model = model_class(**params, random_state=42)
            model.fit(X_train)

        # Predict
        y_pred = model.predict(X_test)
        y_pred = [1 if x == -1 else 0 for x in y_pred]

        # Evaluate
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        elapsed = perf_counter() - start
        print(f"\n{elapsed:.2f}s | params: {params} | F1_Score: {f1:.4f}")

        results.append({
            **params,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        })

    # Sort and retrain with best parameters and with full training set
    df_results = pd.DataFrame(results)
    df_sorted = df_results.sort_values(by="f1_score", ascending=False)

    best_params = df_sorted.iloc[0].drop(['precision', 'recall', 'f1_score']).to_dict()
    print("Fitting the model with besst hyperparameters...")
    if model_class == OneClassSVM:
        model = model_class(**best_params)
        model.fit(X_train)
    else:
        model = model_class(**best_params, random_state=42)
        model.fit(X_train)

    return model, df_sorted

def plot_top_n_results(df_results, metric='f1_score', top_n=10):
    df_top = df_results.sort_values(by=metric, ascending=False).head(top_n).copy()

    # Create a short string label for each config (excluding metrics)
    param_cols = [col for col in df_top.columns if col not in ['precision', 'recall', 'f1_score']]
    df_top['params'] = df_top[param_cols].apply(lambda row: ', '.join(f"{k}={v}" for k, v in row.items()), axis=1)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_top, x=metric, y='params', palette='crest')
    plt.xlabel(metric.upper())
    plt.ylabel("Hyperparameters")
    plt.title(f"Top {top_n} Hyperparameter Sets by {metric.upper()}")
    plt.tight_layout()
    plt.show()

def evaluate_model(model, X_test, y_test, plot):
    # Predict
    y_pred = model.predict(X_test)
    # 1 = Fraud / 0 = Legit
    y_pred = [1 if x == -1 else 0 for x in y_pred]
    # Evaluate
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    if plot:
        plot_confusion_matrix(y_test, y_pred)

# Plot confusion matrix
def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

def save_model(model, model_path):
    # Save model and create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def main(model_class, tune=False, save=False, plot=False):
    if model_class == IsolationForest:
        model_path = MODELS_DIR / "isolation_forest.joblib"
    else:
        model_path = MODELS_DIR / "one_class_svm.joblib"
    if tune:
        model, df_sorted = hyperparameter_tuning(model_class)
        if plot:
            plot_top_n_results(df_sorted, top_n=10)
    else:
        if not os.path.exists(model_path):
            raise Exception(f"Model not found at {model_path}")
        model = joblib.load(model_path)
    _, X_test, _, y_test = load_data()
    evaluate_model(model, X_test, y_test, plot)
    if save:
        save_model(model, model_path)