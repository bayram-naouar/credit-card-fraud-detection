import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
from config import DATA_RAW, DATA_PROCESSED_DIR

def preprocess():
    #Load data
    df = pd.read_csv(DATA_RAW)
    df.drop(["Time"], axis=1, inplace=True)
    df_legit = df[df["Class"] == 0]
    df_fraud = df[df["Class"] == 1]
    X_legit = df_legit.drop(["Class"], axis=1).to_numpy()
    X_fraud = df_fraud.drop(["Class"], axis=1).to_numpy()
    #Seperate features and target
    X = df.drop(["Class"], axis=1)
    y = df["Class"]

    # In this case, we need to scale only "Amount" feature
    sc = StandardScaler()
    X[:, -1] = sc.fit_transform(X[:, -1].reshape(-1, 1)).flatten()

    #Split data into train and test sets with stratification
    X_tr, X_ts, y_tr, y_ts = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    #Save train and test sets
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    np.save(f"{DATA_PROCESSED_DIR}/X_train.npy", X_tr)
    np.save(f"{DATA_PROCESSED_DIR}/X_test.npy", X_ts)
    np.save(f"{DATA_PROCESSED_DIR}/y_train.npy", y_tr)
    np.save(f"{DATA_PROCESSED_DIR}/y_test.npy", y_ts)
    np.save(f"{DATA_PROCESSED_DIR}/X_legit.npy", X_legit)
    np.save(f"{DATA_PROCESSED_DIR}/X_fraud.npy", X_fraud)