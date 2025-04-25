import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from config import DATA_PROCESSED_DIR, MODELS_DIR

def load_data():
    # Load data
    X_train = np.load(f"{DATA_PROCESSED_DIR}/X_train.npy")
    X_test = np.load(f"{DATA_PROCESSED_DIR}/X_test.npy")
    y_train = np.load(f"{DATA_PROCESSED_DIR}/y_train.npy")
    y_test = np.load(f"{DATA_PROCESSED_DIR}/y_test.npy")
    return X_train, X_test, y_train, y_test

def hyperparameter_tuning(model_class,X_train, X_test, y_test):
    # Hyperparameter tuning
    n_estimators = [200, 300, 400, 500]
    max_samples = [0.3, 0.5, 0.7, 1.0]
    contamination = [0.0005, 0.001, 0.0015, 0.002]
    bootstrap = [True, False]

    # Initialize results
    results = []
    # Iterate over hyperparameters
    for ne in n_estimators:
        for ms in max_samples:
            for ct in contamination:
                for bs in bootstrap:
                    model = model_class(n_estimators=ne,
                                            max_samples=ms,
                                            contamination=ct,
                                            bootstrap=bs,
                                            random_state=42)
                    model.fit(X_train)
                    # Predict
                    y_pred = model.predict(X_test)
                    # 1 = Fraud / 0 = Legit
                    y_pred = [1 if x == -1 else 0 for x in y_pred]
                    # Evaluate
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    # Save results
                    results.append({
                        'n_estimators': ne,
                        'max_samples': ms,
                        'contamination': ct,
                        'bootstrap': bs,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1
                    })

    df_results = pd.DataFrame(results)
    # Sort by F1-score (or whichever metric you prefer)
    df_sorted = df_results.sort_values(by='f1_score', ascending=False)
    # Print top results
    print("\nTop Hyperparameter Combinations:")
    print(df_sorted.head(10).to_string(index=False))

    # Using the best parameters to fit the model
    best_params = df_sorted.iloc[0].to_dict()
    exclude = ['precision', 'recall', 'f1_score']
    best_params = {k: v for k, v in best_params.items() if k not in exclude}
    model = model_class(**best_params, random_state=42)
    model.fit(X_train)

    # Visualize top hyperparameters by F1-score
    top_df = df_sorted.head(10).copy()
    top_df['label'] = top_df.apply(lambda row: f"{int(row['n_estimators'])} | {row['max_samples']} | {row['contamination']} | {row['bootstrap']}", axis=1)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_df, x='f1_score', y='label', palette='viridis')
    plt.xlabel("F1 Score")
    plt.ylabel("Hyperparameter Combo")
    plt.title("Top Hyperparameter Combinations by F1 Score")
    plt.tight_layout()
    plt.show()

    return model, df_sorted

def evaluate_model(model, X_test, y_test):
    # Predict
    y_pred = model.predict(X_test)
    # 1 = Fraud / 0 = Legit
    y_pred = [1 if x == -1 else 0 for x in y_pred]
    # Evaluate
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

def save_model(model, model_path):
    # Save model and create directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

def main(model_class, tune=False, save=False):
    X_train, X_test, y_train, y_test = load_data()
    if model_class == IsolationForest:
        model_path = MODELS_DIR / "isolation_forest.joblib"
    else:
        model_path = MODELS_DIR / "one_class_svm.joblib"
    if tune:
        model, df_sorted = hyperparameter_tuning(model_class, X_train, X_test, y_test)
    else:
        if not os.path.exists(model_path):
            raise Exception(f"Model not found at {model_path}")
        model = joblib.load(model_path)
    evaluate_model(model, X_test, y_test)
    if save:
        save_model(model, model_path)