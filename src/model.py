import numpy as np
import pandas as pd

from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score, confusion_matrix

from tqdm import tqdm
from itertools import product
from time import perf_counter

import joblib
import json
import os

from config import DATA_PROCESSED_DIR, MODELS_DIR, PERCENTILE_PATH, AutoEncoderBuilder

import matplotlib.pyplot as plt
import seaborn as sns

def load_data(train=True, test=True, legit=True, fraud=False):
    # Load data
    X_train = np.load(f"{DATA_PROCESSED_DIR}/X_train.npy") if train else None
    y_train = np.load(f"{DATA_PROCESSED_DIR}/y_train.npy") if train else None
    X_test = np.load(f"{DATA_PROCESSED_DIR}/X_test.npy") if test else None
    y_test = np.load(f"{DATA_PROCESSED_DIR}/y_test.npy") if test else None
    X_legit = np.load(f"{DATA_PROCESSED_DIR}/X_legit.npy") if legit else None
    X_fraud = np.load(f"{DATA_PROCESSED_DIR}/X_fraud.npy") if fraud else None
    return X_train, X_test, y_train, y_test, X_legit, X_fraud

def get_param_grid(model_class):
    if model_class == IsolationForest:
        return {
            "n_estimators": [200, 300, 400, 500],
            "max_samples": [0.5, 0.7, 1.0],
            "contamination": [0.0015, 0.002, 0.0025],
            "bootstrap": [True, False]
        }
    elif model_class == OneClassSVM:
        return {
            "kernel": ["rbf", "poly", "sigmoid"],
            "nu": [0.01, 0.05, 0.1, 0.5],
            "gamma": ["scale", "auto", 0.01, 0.1, 1.0],
            "degree": [3, 4, 5]  # Only relevant for 'poly'
        }
    elif model_class == AutoEncoderBuilder:
        return {
            'latent_dim': [2, 4, 8],
            'learning_rate': [0.001, 0.005, 0.01],
            'loss': ['mse', 'mae'],
            'epochs': [10, 20, 30],
            'batch_size': [32, 64, 128],
            'percentile': [95, 97, 99]
        }
    else:
        raise ValueError("Unknown model class provided.")

def hyperparameter_tuning(model_class):
    """
    Grid search tuning for IsolationForest and OneClassSVM.
    Always applies fast tuning for OneClassSVM using a subsample of training data.
    """
    # Define parameter grid
    param_grid = get_param_grid(model_class)

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

    # Load processed data
    X_train, X_test, _, y_test, X_legit, _ = load_data()
    
    results = []
    for params in tqdm(param_list, desc="Tuning Progress"):
        start = perf_counter()

        # Use subsample for fast OCSVM training
        if model_class == OneClassSVM:
            sample_size = int(0.25 * len(X_train))
            X_train_sub = X_train[:sample_size]
            model = model_class(**params)
            model.fit(X_train_sub)
        elif model_class == IsolationForest:
            model = model_class(**params, random_state=42)
            model.fit(X_train)
        # AutoEncoder trains only on legit transactions
        elif model_class == AutoEncoderBuilder:
            model_builder = model_class(X_legit.shape[1],**params)
            model = model_builder()
            model.fit(X_legit, X_legit,
                      epochs=model_builder.epochs,
                      batch_size=model_builder.batch_size,
                      validation_split=0.2,
                      callable=[EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)])

        # Predict
        if model_class == OneClassSVM or model_class == IsolationForest:
            y_pred = model.predict(X_test)
            y_pred = [1 if x == -1 else 0 for x in y_pred]
        elif model_class == AutoEncoderBuilder:
            reconstruction = model.predict(X_test)
            reconstruction_error = ((reconstruction - X_test) ** 2).mean(axis=1)
            threshold = np.percentile(reconstruction_error, model_builder.percentile)
            y_pred = (reconstruction_error > threshold).astype(int)
        else:
            raise ValueError("Unknown model class provided.")

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
    percentile = None
    if model_class == OneClassSVM:
        model = model_class(**best_params)
        model.fit(X_train)
    elif model_class == IsolationForest:
        model = model_class(**best_params, random_state=42)
        model.fit(X_train)
    elif model_class == AutoEncoderBuilder:
        model_builder = model_class(**best_params)
        model = model_builder()
        model.fit(X_train, X_train,
                  epochs=model_builder.epochs,
                  batch_size=model_builder.batch_size,
                  validation_split=0.2)
        percentile = best_params['percentile']
    return model, df_sorted, percentile

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

def evaluate_model(model, X_test, y_test, plot, percentile=None):
    # Predict
    if isinstance(model, IsolationForest) or isinstance(model, OneClassSVM):
        y_pred = model.predict(X_test)
        # 1 = Fraud / 0 = Legit
        y_pred = [1 if x == -1 else 0 for x in y_pred]
    elif isinstance(model, Sequential):
        reconstruction = model.predict(X_test)
        reconstruction_error = ((reconstruction - X_test) ** 2).mean(axis=1)
        threshold = np.percentile(reconstruction_error, percentile)
        y_pred = (reconstruction_error > threshold).astype(int)
    else:
        raise ValueError("Unknown model class provided.")
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

def save_model(model, model_path, percentile=None):
    # Save model and create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    if isinstance(model, Sequential):
        model.save(model_path)
        with open(PERCENTILE_PATH, 'w') as f:
            json.dump({"percentile": percentile}, f)
    else:
        joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def main(model_class, tune=False, save=False, plot=False):
    if model_class == IsolationForest:
        model_path = MODELS_DIR / "isolation_forest.joblib"
    elif model_class == OneClassSVM:
        model_path = MODELS_DIR / "one_class_svm.joblib"
    elif model_class == AutoEncoderBuilder:
        model_path = MODELS_DIR / "auto_encoder.h5"
    else:
        raise ValueError("Unknown model class provided.")
    if tune:
        model, df_sorted, percentile = hyperparameter_tuning(model_class)
        if plot:
            plot_top_n_results(df_sorted, top_n=10)
    else:
        percentile = None
        if not os.path.exists(model_path):
            raise Exception(f"Model not found at: {model_path}")
        if isinstance(model_class, Sequential):
            model = load_model(model_path)
            if os.path.exists(PERCENTILE_PATH):
                with open(PERCENTILE_PATH, 'r') as f:
                    percentile = json.load(f)
            else:
                raise Exception(f"Percentile not found at: {PERCENTILE_PATH}")
        else:
            model = joblib.load(model_path)
    _, X_test, _, y_test = load_data(train=False, test=True, legit=False, fraud=False)
    evaluate_model(model, X_test, y_test, plot, percentile)
    if save:
        save_model(model, model_path, percentile)