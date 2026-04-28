import os

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

from src.config import PATHS, PREPROCESSING, ROOT

SECONDS_IN_DAY = 86400


def preprocess_data():
    time_col = PREPROCESSING["columns"]["time"]
    amount_col = PREPROCESSING["columns"]["amount"]
    target_col = PREPROCESSING["columns"]["target"]

    print("[preprocessing] Loading dataset...")
    df = pd.read_csv(ROOT / PATHS["raw_data"])

    print("[preprocessing] Checking data quality...")
    print("[preprocessing] Missing values:", df.isnull().sum().sum())
    unexpected = df.dtypes[~df.dtypes.astype(str).isin(["float64", "int64"])]
    if len(unexpected) > 0:
        print("[preprocessing] Unexpected dtypes:\n", unexpected)
    else:
        print("[preprocessing] Dtypes OK")
    print("[preprocessing] Dataset shape:", df.shape)

    print("[preprocessing] Time features engineering...")
    df[time_col + "_sin"] = np.sin(2 * np.pi * df[time_col] / SECONDS_IN_DAY)
    df[time_col + "_cos"] = np.cos(2 * np.pi * df[time_col] / SECONDS_IN_DAY)
    df.drop(columns=[time_col], inplace=True)

    print("[preprocessing] Splitting dataset into train, test and validation sets...")
    random_state, test_size, val_size = (
        PREPROCESSING["random_state"],
        PREPROCESSING["test_size"],
        PREPROCESSING["val_size"],
    )
    train_df, test_df = train_test_split(
        df,
        test_size=test_size + val_size,
        random_state=random_state,
        stratify=df[target_col],
    )
    test_df, val_df = train_test_split(
        test_df,
        test_size=val_size / (test_size + val_size),
        random_state=random_state,
        stratify=test_df[target_col],
    )

    print("[preprocessing] Scaling Amount feature...")
    scaler = RobustScaler().fit(train_df[[amount_col]])
    train_df[[amount_col]] = scaler.transform(train_df[[amount_col]])
    test_df[[amount_col]] = scaler.transform(test_df[[amount_col]])
    val_df[[amount_col]] = scaler.transform(val_df[[amount_col]])
    joblib.dump(scaler, ROOT / PATHS["scaler_path"])

    print("[preprocessing] Saving preprocessed datasets...")
    os.makedirs(ROOT / PATHS["processed_dir"], exist_ok=True)
    X_train, y_train = train_df.drop(columns=[target_col]), train_df[target_col]
    X_test, y_test = test_df.drop(columns=[target_col]), test_df[target_col]
    X_val, y_val = val_df.drop(columns=[target_col]), val_df[target_col]
    X_legit = train_df[train_df[target_col] == 0].drop(columns=[target_col])
    np.save(ROOT / PATHS["processed_dir"] / "X_train.npy", X_train)
    np.save(ROOT / PATHS["processed_dir"] / "y_train.npy", y_train)
    np.save(ROOT / PATHS["processed_dir"] / "X_test.npy", X_test)
    np.save(ROOT / PATHS["processed_dir"] / "y_test.npy", y_test)
    np.save(ROOT / PATHS["processed_dir"] / "X_val.npy", X_val)
    np.save(ROOT / PATHS["processed_dir"] / "y_val.npy", y_val)
    np.save(ROOT / PATHS["processed_dir"] / "X_legit.npy", X_legit)

    print("[preprocessing] Data preprocessing complete.")


def main():
    preprocess_data()


if __name__ == "__main__":
    main()
