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

    #Seperate features and target
    X = df.drop(["Class"], axis=1)
    y = df["Class"]

    #Scale features using Standard Scaler
    sc = StandardScaler()
    X_sc = sc.fit_transform(X)
    
    #Split data into train and test sets with stratification
    X_tr, X_ts, y_tr, y_ts = train_test_split(X_sc, y, test_size=0.2, random_state=42, stratify=y)

    #Save train and test sets
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
    np.save(f"{DATA_PROCESSED_DIR}/X_train.npy", X_tr)
    np.save(f"{DATA_PROCESSED_DIR}/X_test.npy", X_ts)
    np.save(f"{DATA_PROCESSED_DIR}/y_train.npy", y_tr)
    np.save(f"{DATA_PROCESSED_DIR}/y_test.npy", y_ts)