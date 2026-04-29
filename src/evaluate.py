import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

from src.config import PATHS, ROOT
from src.models.registry import MODEL_REGISTRY


def evaluate(model_name):
    model_name = model_name.lower()
    model_path = ROOT / PATHS["model_dir"] / f"{model_name}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. Train/Tune the model first."
        )
    model_class, _, _, _, need_input_dim = MODEL_REGISTRY[model_name]
    model = model_class(input_dim=1) if need_input_dim else model_class()
    model.load()
    try:
        X_test = np.load(ROOT / PATHS["processed_dir"] / "X_test.npy")
        y_test = np.load(ROOT / PATHS["processed_dir"] / "y_test.npy")
    except FileNotFoundError:
        raise FileNotFoundError("Run preprocessing first.")
    scores = model.anomaly_scores(X_test)
    y_pred = model.predict(X_test)
    print(f"[evaluating] Classification Report for {model_name}:\n")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    cm_normalized = confusion_matrix(y_test, y_pred, normalize="true")
    print(f"[evaluating] Confusion Matrix for {model_name}:\n")
    print(cm)

    precision, recall, _ = precision_recall_curve(y_test, scores)
    auc_pr = auc(recall, precision)
    print(f"[evaluating] AUC-PR for {model_name}: {auc_pr:.3f}")

    annot = np.array(
        [
            [f"{cm[i][j]}\n({cm_normalized[i][j] * 100:.1f}%)" for j in range(2)]
            for i in range(2)
        ]
    )
    labels = ["Legitimate", "Fraud"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    sns.heatmap(
        cm_normalized,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    axes[1].plot(recall, precision, label=f"AUC-PR = {auc_pr:.3f}")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()

    fig.suptitle(model_name, fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(ROOT / PATHS["assets_dir"] / f"{model_name}.png")
    plt.close(fig)


def main():
    evaluate("isolation_forest")
    evaluate("one_class_svm")
    evaluate("autoencoder")


if __name__ == "__main__":
    main()
