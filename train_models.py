"""Training pipeline for Pregnancy Risk Prediction."""

from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

DATA_PATH = Path("pregnancy_risk_dataset_1015.csv")
OUTPUT_DIR = Path("outputs")
MODEL_DIR = Path("models")


def load_and_prepare_data(path: Path):
    """Load CSV and convert one-hot risk columns to binary target."""
    df = pd.read_csv(path)

    if not {"RiskLevel_Low", "RiskLevel_Medium"}.issubset(df.columns):
        raise ValueError("Dataset must include RiskLevel_Low and RiskLevel_Medium columns.")

    df["RiskLevel"] = df["RiskLevel_Medium"].astype(int)
    df = df.drop(columns=["RiskLevel_Low", "RiskLevel_Medium"])

    X = df.drop(columns=["RiskLevel"])
    y = df["RiskLevel"]
    return X, y


def train_and_evaluate():
    """Train 4 models, print metrics, and export plots + best model."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    X, y = load_and_prepare_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {
        "Logistic Regression": LogisticRegression(C=0.9, max_iter=1200, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=180,
            max_depth=6,
            min_samples_leaf=6,
            random_state=42,
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=140,
            learning_rate=0.08,
            max_depth=2,
            random_state=42,
        ),
        "SVM (RBF)": SVC(C=2.0, gamma=0.2, kernel="rbf", random_state=42),
    }

    results = {}
    confusion_matrices = {}

    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        results[name] = {
            "model": model,
            "accuracy": acc,
            "report": classification_report(y_test, y_pred, target_names=["Low Risk", "Medium Risk"]),
        }
        confusion_matrices[name] = cm

        print(f"\n{'='*75}\n{name}\n{'='*75}")
        print(f"Accuracy: {acc * 100:.2f}%")
        print("Confusion Matrix:")
        print(cm)
        print("Classification Report:")
        print(results[name]["report"])

    # Accuracy comparison bar chart
    plt.figure(figsize=(9, 5))
    model_names = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in model_names]
    bars = plt.bar(model_names, accuracies, color=["#4C78A8", "#72B7B2", "#F58518", "#E45756"])
    plt.ylim(80, 100)
    plt.ylabel("Accuracy (%)")
    plt.title("Pregnancy Risk Model Accuracy Comparison")
    plt.grid(axis="y", alpha=0.2)
    plt.xticks(rotation=15)
    for bar, value in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width() / 2, value + 0.3, f"{value:.1f}%", ha="center")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "model_accuracy_comparison.png", dpi=220)
    plt.close()

    # Confusion matrix heatmaps
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for ax, name in zip(axes, model_names):
        sns.heatmap(
            confusion_matrices[name],
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=["Low", "Medium"],
            yticklabels=["Low", "Medium"],
            ax=ax,
        )
        ax.set_title(name)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle("Confusion Matrix Heatmaps", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "confusion_matrices.png", dpi=220)
    plt.close()

    # Feature importances for tree models
    for tree_model_name in ["Random Forest", "Gradient Boosting"]:
        model = results[tree_model_name]["model"]
        importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=importances.values[:12], y=importances.index[:12], palette="viridis")
        plt.title(f"Top Feature Importances - {tree_model_name}")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"feature_importance_{tree_model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')}.png", dpi=220)
        plt.close()

    # Save best model + scaler
    best_model_name = max(results, key=lambda key: results[key]["accuracy"])
    artifact = {
        "model_name": best_model_name,
        "model": results[best_model_name]["model"],
        "scaler": scaler,
        "feature_columns": list(X.columns),
    }
    joblib.dump(artifact, MODEL_DIR / "best_pregnancy_risk_model.joblib")

    print(f"\nBest model: {best_model_name} ({results[best_model_name]['accuracy']*100:.2f}%)")
    print(f"Saved model to: {MODEL_DIR / 'best_pregnancy_risk_model.joblib'}")


if __name__ == "__main__":
    train_and_evaluate()
