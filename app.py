import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

from src.config import PATHS, ROOT
from src.models.registry import MODEL_REGISTRY

st.set_page_config(page_title="Fraud Detection Playground", layout="wide")

MODEL_DISPLAY = {
    "isolation_forest": "Isolation Forest",
    "one_class_svm": "One-Class SVM",
    "autoencoder": "Autoencoder",
}


@st.cache_resource
def load_model(model_name):
    model_class, _, _, _, need_input_dim = MODEL_REGISTRY[model_name]
    model = model_class(input_dim=1) if need_input_dim else model_class()
    model.load()
    return model


@st.cache_data
def load_test_set():
    X_test = np.load(ROOT / PATHS["processed_dir"] / "X_test.npy")
    y_test = np.load(ROOT / PATHS["processed_dir"] / "y_test.npy")
    return X_test, y_test


@st.cache_data
def compute_scores(model_name):
    model = load_model(model_name)
    X_test, _ = load_test_set()
    return model.anomaly_scores(X_test)


@st.cache_data
def compute_pr_curve(model_name):
    scores = compute_scores(model_name)
    _, y_test = load_test_set()
    precision, recall, _ = precision_recall_curve(y_test, scores)
    return precision, recall, float(auc(recall, precision))


def confusion_matrix_figure(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = confusion_matrix(y_true, y_pred, normalize="true")
    annot = np.array(
        [
            [f"{cm[i][j]}\n({cm_norm[i][j] * 100:.1f}%)" for j in range(2)]
            for i in range(2)
        ]
    )
    labels = ["Legitimate", "Fraud"]
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(
        cm_norm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar=False,
    )
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    fig.tight_layout()
    return fig


def pr_curve_figure(precision, recall, auc_pr, current_recall, current_precision):
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, label=f"AUC-PR = {auc_pr:.3f}")
    ax.scatter(
        [current_recall],
        [current_precision],
        color="red",
        zorder=5,
        label="Current threshold",
    )
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower left")
    fig.tight_layout()
    return fig


st.sidebar.title("Fraud Detection")
st.sidebar.markdown(
    "Anomaly-detection playground for the Kaggle credit-card fraud dataset."
)
selected = st.sidebar.radio(
    "Model",
    options=list(MODEL_DISPLAY.keys()),
    format_func=MODEL_DISPLAY.get,
)

model = load_model(selected)
X_test, y_test = load_test_set()
scores = compute_scores(selected)
precision_curve, recall_curve, auc_pr = compute_pr_curve(selected)
trained_threshold = float(model.threshold)
_, _, _, final_sample_rate, _ = MODEL_REGISTRY[selected]

st.sidebar.markdown("---")
st.sidebar.markdown("**Model info**")
st.sidebar.write(f"Trained on: {int(final_sample_rate * 100)}% of training data")
st.sidebar.write(f"Trained-time threshold: `{trained_threshold:.4f}`")
st.sidebar.write(f"Test-set AUC-PR: `{auc_pr:.3f}`")

st.title("Threshold Playground")
st.markdown(
    f"Adjust the anomaly-score threshold for **{MODEL_DISPLAY[selected]}** and watch the "
    f"precision/recall tradeoff on the held-out test set "
    f"({len(y_test):,} transactions, {int(y_test.sum())} fraud)."
)

score_min = float(scores.min())
score_max = float(scores.max())
threshold = st.slider(
    "Threshold (raw anomaly score)",
    min_value=score_min,
    max_value=score_max,
    value=trained_threshold,
    step=(score_max - score_min) / 1000,
    format="%.4f",
    help=(
        f"Trained-time threshold is {trained_threshold:.4f}. "
        "Drag left → more positives flagged (higher recall, lower precision)."
    ),
)

y_pred = (scores > threshold).astype(int)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Precision", f"{prec:.3f}")
c2.metric("Recall", f"{rec:.3f}")
c3.metric("F1", f"{f1:.3f}")
c4.metric("AUC-PR", f"{auc_pr:.3f}", help="AUC-PR is threshold-independent.")

left, right = st.columns(2)
with left:
    st.pyplot(confusion_matrix_figure(y_test, y_pred))
with right:
    st.pyplot(
        pr_curve_figure(precision_curve, recall_curve, auc_pr, rec, prec)
    )
