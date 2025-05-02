# 🕵️‍♂️ Credit Card Fraud Detection using Anomaly Detection

This project applies unsupervised machine learning techniques to detect fraudulent transactions in the widely used [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).

## 📌 Overview

- Built around three anomaly detection algorithms:
  - Isolation Forest
  - One-Class SVM
  - Autoencoder (with reconstruction error + threshold)
- CLI-based design for flexible usage (tuning, saving, plotting)
- Evaluation using F1-score, precision, recall, custom-score and confusion matrix
- Autoencoder tuning with percentile thresholding for fraud separation
- Models are saved/loaded using `.joblib` (or `.h5` for Keras)

## 📁 Project Structure

credit-card-fraud-detection/\
├── data/\
│   ├── processed/\
│   │   ├── X_fraud.npy\
│   │   ├── X_legit.npy\
│   │   ├── X_test.npy\
│   │   ├── X_train.npy\
│   │   ├── y_test.npy\
│   │   └── y_train.npy\
│   └── raw/\
│       └── creditcard.csv\
├── models/\
│   ├── auto_encoder.h5\
│   ├── isolation_forest.joblib\
│   ├── one_class_svm.joblib\
│   └── percentile.json\
├── src/\
│   ├── main.py           # Main entry point with CLI arguments for running the models\
│   ├── model.py          # Contains model loading, training, and evaluation logic\
│   ├── preprocess.py     # Data preprocessing and feature extraction\
│   ├── config.py         # Configuration file for model parameters\
├── notebooks/\
│   └── eda_creditcard.ipynb  # Exploratory Data Analysis (EDA) of the dataset

## 🔧 Setup

To set up the project, clone the repository and install the necessary dependencies:\
```
git clone https://github.com/bayram-naouar/credit-card-fraud-detection.git
cd credit-card-fraud-detection
pip install -r requirements.txt
```

## 🗃️ Data

The dataset used in this project is the Credit Card Fraud Detection dataset available from Kaggle. The raw dataset (creditcard.csv) is stored under data/raw/. After preprocessing, the dataset is split into training and test sets and stored in data/processed/ as .npy files.

### 🗄️ Columns

Time: Time elapsed since the first transaction in the dataset.

V1 to V28: Various anonymized features from PCA transformation.

Amount: The monetary value of the transaction.

Class: Target variable, where 1 represents a fraud transaction and 0 represents a legitimate transaction.

## 🧪 Exploratory Data Analysis (EDA)

Before building any models, a thorough exploratory data analysis was performed to understand the distribution of features and the extent of class imbalance in the dataset. The analysis includes:

Overview of the dataset shape and missing values

Distribution of transaction amounts

Class imbalance visualization (legit vs fraud)

Correlation matrix

Feature-wise behavior across fraud and legit classes

You can view the full EDA notebook here: notebooks/eda_creditcard.ipynb

## 💪 Models

The following anomaly detection models are implemented and trained in this project:

**Isolation Forest**: A tree-based model that isolates observations to detect anomalies.

**One-Class SVM**: A variant of the Support Vector Machine used for anomaly detection.

**Autoencoder (Neural Network)**: A deep learning model used to reconstruct input data and detect anomalies based on reconstruction error.

The models are stored under models/ as joblib (for traditional models) and .h5 (for the Autoencoder model). Additionally, the threshold percentile used for the Autoencoder model is stored in percentile.json.

## 🔧 Command-Line Interface (CLI)

To run the models and evaluate them, you can use the provided CLI in main.py.  
This project includes a CLI interface to run and evaluate different anomaly detection models. One of the following model options must be specified when running the script:  
```
python main.py --auto-encoder       # Use the latest tuned and saved AutoEncoder model
python main.py --isolation-forest   # Use the latest tuned and saved Isolation Forest model
python main.py --one-class-svm      # Use the latest tuned and saved One-Class SVM model
```
You can also pass the following optional arguments:  
* --tune — Perform hyperparameter tuning before evaluation.  
* --save — Save the trained model to disk (used only with --tune)  
* --plot — Show evaluation plots (like confusion matrix).

**Example Usage**  
  
`python main.py --isolation-forest --tune --save --plot`  
  
This will:  
1- Use the AutoEncoder model.  
2- Perform hyperparameter tuning.  
3- Display the top 10 parameters and the confusion matrix using the best parameters for the model.  
4- Save the trained model.

## 🥇 Evaluation Metrics

The following metrics are used to evaluate the models:    
**Precision**: The proportion of positive predictions that are actually correct.  
**Recall**: The proportion of actual positive cases that are correctly identified.  
**F1 Score**: The harmonic mean of precision and recall.  
**Custom Score**: A custom score calculated as following: 0.5 * F1_Score + 0.3 * Recall + 0.2 * Precision  
The model with the best custom score is selected based on hyperparameter tuning.

## 🗒️ Note

The Autoencoder model’s threshold percentile is stored in models/percentile.json and is used to determine if a transaction is fraudulent based on reconstruction error.

The models are trained and evaluated using a split of 80% for training and 20% for testing.

## ⏲️ Future Work

Explore more advanced anomaly detection techniques like Deep One-Class Networks.

Apply ensemble methods to combine the results of multiple models.

Perform model explainability analysis to understand how the models detect fraud.

## 🔓 License

This project is licensed under the MIT License.
