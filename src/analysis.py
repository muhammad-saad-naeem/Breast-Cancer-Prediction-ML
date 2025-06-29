"""
Breast Cancer Diagnosis Prediction
=================================

This module performs an end‑to‑end analysis of the Breast Cancer
Wisconsin Diagnostic dataset using a variety of classical machine
learning algorithms.  The goal is to demonstrate a complete data
science pipeline including exploratory data analysis (EDA), data
preprocessing, model training, cross‑validation and evaluation, and
visualisation.  All plots are saved to the ``plots`` directory so
they can be referenced in a report or README.  For reproducibility
purposes, the script also writes a copy of the dataset to disk.

The dataset used here is provided by the ``sklearn.datasets`` module
and contains 569 samples with 30 numeric features describing
characteristics of cell nuclei present in digitised images of fine
needle aspirates (FNA) of breast masses.  The target variable
indicates whether the tumour is malignant (1) or benign (0).  This
dataset is widely used in the literature for binary classification
tasks, making it ideal for demonstrating statistical and machine
learning techniques.

Example usage from the command line::

    python analysis.py

Running the script will output summary statistics to the terminal,
save several figures to ``plots/``, and print a table of model
accuracies.  See the README.md for additional context and results.
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib

# Set Matplotlib backend to a non‑interactive backend suitable for headless
# environments.  We also set MPLCONFIGDIR and HOME to temporary
# directories to avoid permission issues when Matplotlib tries to write
# caches under the user's home directory.  These environment variables
# should be set *before* importing pyplot.
os.environ.setdefault("MPLCONFIGDIR", "/tmp/.config")
os.environ.setdefault("HOME", "/tmp")

import matplotlib.pyplot as plt  # noqa: E402  import after setting env vars
import seaborn as sns  # noqa: E402
from sklearn.datasets import load_breast_cancer  # noqa: E402
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.svm import SVC  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402


def load_dataset() -> pd.DataFrame:
    """Load the Breast Cancer dataset into a pandas DataFrame.

    Returns
    -------
    df : pd.DataFrame
        DataFrame containing feature columns and the target column ``target``.
    """
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    return df


def save_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Save the dataset to CSV for reproducibility.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset.
    output_path : Path
        Path to write the CSV file to.
    """
    df.to_csv(output_path, index=False)


def describe_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return summary statistics of the numerical features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset.

    Returns
    -------
    stats : pd.DataFrame
        Summary statistics (mean, std, min, max, etc.) for each feature.
    """
    stats = df.describe().T
    return stats


def correlation_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Create and save a correlation heatmap of the feature variables.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset.
    output_dir : Path
        Directory where the heatmap figure will be saved.
    """
    corr = df.drop(columns="target").corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", square=True, cbar_kws={"shrink": 0.7})
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    fig_path = output_dir / "correlation_heatmap.png"
    plt.savefig(fig_path)
    plt.close()


def plot_top_feature_distributions(df: pd.DataFrame, output_dir: Path, top_n: int = 6) -> None:
    """Plot distribution of the top features most correlated with the target.

    This function computes the absolute correlation between each feature and
    the target variable, selects the top ``top_n`` features with the
    strongest correlations, and produces violin plots to visualise the
    distribution of these features across the malignant and benign classes.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the dataset.
    output_dir : Path
        Directory where the plot will be saved.
    top_n : int, optional
        Number of top correlated features to plot, by default 6.
    """
    feature_cols = df.columns.drop("target")
    correlations = df[feature_cols].corrwith(df["target"]).abs()
    top_features = correlations.sort_values(ascending=False).head(top_n).index
    # Melt DataFrame for Seaborn violinplot
    df_melt = df.melt(id_vars="target", value_vars=top_features, var_name="feature", value_name="value")
    plt.figure(figsize=(12, 6))
    sns.violinplot(x="feature", y="value", hue="target", data=df_melt, split=True, inner="quartile")
    plt.title(f"Distribution of Top {top_n} Features by Class")
    plt.xlabel("Feature")
    plt.ylabel("Value")
    plt.legend(title="Target", labels=["Benign (0)", "Malignant (1)"])
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig_path = output_dir / "top_features_violin.png"
    plt.savefig(fig_path)
    plt.close()


def train_models(X: pd.DataFrame, y: pd.Series) -> Dict[str, Dict[str, object]]:
    """Train multiple classifiers and perform cross‑validation.

    This function constructs a dictionary of machine learning pipelines,
    performs 5‑fold stratified cross‑validation for each model, and
    computes the mean accuracy across folds.  It returns a dictionary
    containing the model objects, cross‑validated accuracies, and the
    standard deviation of the scores.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.

    Returns
    -------
    results : dict
        Dictionary mapping model name to a dict with keys ``pipeline``,
        ``cv_scores`` and ``cv_std``.
    """
    models: Dict[str, object] = {
        "Logistic Regression": LogisticRegression(max_iter=500, solver="lbfgs"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "SVM (RBF kernel)": SVC(kernel="rbf", probability=True, random_state=42),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5),
    }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results: Dict[str, Dict[str, object]] = {}
    for name, model in models.items():
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", model),
        ])
        scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        results[name] = {
            "pipeline": pipeline,
            "cv_scores": scores,
            "cv_mean": scores.mean(),
            "cv_std": scores.std(),
        }
    return results


def evaluate_model(model_name: str, pipeline: Pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame, y_train: pd.Series, y_test: pd.Series, output_dir: Path) -> Dict[str, object]:
    """Fit a pipeline on training data, evaluate on test data, and save visualisations.

    Parameters
    ----------
    model_name : str
        Name of the model.
    pipeline : Pipeline
        Preprocessing and classifier pipeline to train.
    X_train : pd.DataFrame
        Training feature matrix.
    X_test : pd.DataFrame
        Test feature matrix.
    y_train : pd.Series
        Training target vector.
    y_test : pd.Series
        Test target vector.
    output_dir : Path
        Directory to save evaluation plots.

    Returns
    -------
    metrics : dict
        Dictionary with evaluation metrics (accuracy, roc_auc, classification report).
    """
    # Fit model
    pipeline.fit(X_train, y_train)
    # Predictions
    y_pred = pipeline.predict(X_test)
    y_proba = None
    try:
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    except Exception:
        # Not all classifiers support predict_proba (e.g., SVM with probability=False)
        pass
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=["Benign", "Malignant"], output_dict=True)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"])
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix: {model_name}")
    plt.tight_layout()
    fig_path = output_dir / f"confusion_matrix_{model_name.replace(' ', '_')}.png"
    plt.savefig(fig_path)
    plt.close()

    # ROC curve
    roc_auc = None
    if y_proba is not None:
        roc_auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.3f})")
        plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve: {model_name}")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_path = output_dir / f"roc_curve_{model_name.replace(' ', '_')}.png"
        plt.savefig(roc_path)
        plt.close()

    metrics = {
        "accuracy": acc,
        "report": report,
        "roc_auc": roc_auc,
    }
    return metrics


def generate_report(results: Dict[str, Dict[str, object]], output_dir: Path) -> pd.DataFrame:
    """Create a summary DataFrame of cross‑validated model performance.

    Parameters
    ----------
    results : dict
        The dictionary returned from :func:`train_models`.
    output_dir : Path
        Directory where the summary CSV will be saved.

    Returns
    -------
    summary_df : pd.DataFrame
        DataFrame summarising the mean and standard deviation of accuracy for each model.
    """
    summary_data: List[Dict[str, object]] = []
    for name, res in results.items():
        summary_data.append({
            "model": name,
            "cv_accuracy_mean": res["cv_mean"],
            "cv_accuracy_std": res["cv_std"],
        })
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / "model_cv_results.csv", index=False)
    return summary_df


def main() -> None:
    """Main routine to orchestrate data analysis and model evaluation."""
    # Configure warnings
    warnings.filterwarnings("ignore")
    # Set output directories
    project_root = Path(__file__).resolve().parent.parent
    plots_dir = project_root / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    data_path = project_root / "breast_cancer_data.csv"

    # Load and save dataset
    df = load_dataset()
    save_dataset(df, data_path)

    # Print basic information and summary statistics
    print("Dataset shape:", df.shape)
    print("Class distribution:")
    print(df["target"].value_counts())
    print("Summary statistics:")
    print(describe_data(df))

    # Exploratory plots
    correlation_heatmap(df, plots_dir)
    plot_top_feature_distributions(df, plots_dir, top_n=6)

    # Prepare data for modelling
    X = df.drop(columns="target")
    y = df["target"]

    # Train models with cross‑validation
    results = train_models(X, y)
    summary_df = generate_report(results, project_root)
    print("\nCross‑validation results:")
    print(summary_df)

    # Train/test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    metrics_table: List[Dict[str, object]] = []
    for name, res in results.items():
        metrics = evaluate_model(
            model_name=name,
            pipeline=res["pipeline"],
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            output_dir=plots_dir,
        )
        metrics_table.append({
            "model": name,
            "test_accuracy": metrics["accuracy"],
            "roc_auc": metrics["roc_auc"],
        })
    metrics_df = pd.DataFrame(metrics_table)
    metrics_df.to_csv(project_root / "model_test_results.csv", index=False)
    print("\nTest set results:")
    print(metrics_df)


if __name__ == "__main__":
    main()