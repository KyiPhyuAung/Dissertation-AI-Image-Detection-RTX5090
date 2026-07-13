"""
metrics.py

Central evaluation module for all dissertation experiments.

This module computes:

- Accuracy
- Precision
- Recall
- Macro / Weighted F1
- Balanced Accuracy
- Matthews Correlation Coefficient
- ROC AUC
- PR AUC

It also stores predictions for later visualization.
"""

from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    ConfusionMatrixDisplay,
)

OUTPUT_DPI = 300


# =============================================================================
# Prediction Container
# =============================================================================

class PredictionRecorder:
    """
    Stores predictions during evaluation.
    """

    def __init__(self):

        self.image_paths = []

        self.generators = []

        self.true_labels = []

        self.predicted_labels = []

        self.prob_real = []

        self.prob_fake = []

    def add(
        self,
        image_path,
        generator,
        true_label,
        predicted_label,
        probability_real,
        probability_fake,
    ):

        self.image_paths.append(image_path)

        self.generators.append(generator)

        self.true_labels.append(int(true_label))

        self.predicted_labels.append(int(predicted_label))

        self.prob_real.append(float(probability_real))

        self.prob_fake.append(float(probability_fake))

    def dataframe(self):

        df = pd.DataFrame({

            "image_path": self.image_paths,

            "generator": self.generators,

            "true_label": self.true_labels,

            "predicted_label": self.predicted_labels,

            "prob_real": self.prob_real,

            "prob_fake": self.prob_fake,

        })

        df["correct"] = (
            df["true_label"] == df["predicted_label"]
        )

        return df


# =============================================================================
# Metric Calculation
# =============================================================================

def compute_metrics(recorder: PredictionRecorder):

    y_true = np.asarray(recorder.true_labels)

    y_pred = np.asarray(recorder.predicted_labels)

    y_prob = np.asarray(recorder.prob_fake)

    metrics = {

        "accuracy":
            accuracy_score(y_true, y_pred),

        "precision":
            precision_score(y_true, y_pred),

        "recall":
            recall_score(y_true, y_pred),

        "macro_f1":
            f1_score(
                y_true,
                y_pred,
                average="macro",
            ),

        "weighted_f1":
            f1_score(
                y_true,
                y_pred,
                average="weighted",
            ),

        "balanced_accuracy":
            balanced_accuracy_score(
                y_true,
                y_pred,
            ),

        "mcc":
            matthews_corrcoef(
                y_true,
                y_pred,
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                y_prob,
            ),

        "pr_auc":
            average_precision_score(
                y_true,
                y_prob,
            ),
    }

    return metrics


# =============================================================================
# Classification Report
# =============================================================================

def classification_dataframe(recorder: PredictionRecorder):

    report = classification_report(

        recorder.true_labels,

        recorder.predicted_labels,

        target_names=[
            "Real",
            "Fake",
        ],

        output_dict=True,

        digits=4,

    )

    return pd.DataFrame(report).transpose()
# =============================================================================
# Save Results
# =============================================================================

def save_predictions_csv(
    recorder: PredictionRecorder,
    output_dir,
):
    """
    Save prediction results for every evaluated image.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = recorder.dataframe()

    csv_path = output_dir / "predictions.csv"

    df.to_csv(csv_path, index=False)

    print(f"Saved predictions: {csv_path}")

    return df


def save_classification_report(
    recorder: PredictionRecorder,
    output_dir,
):
    """
    Save sklearn classification report.
    """

    output_dir = Path(output_dir)

    report = classification_dataframe(recorder)

    report_path = output_dir / "classification_report.csv"

    report.to_csv(report_path)

    print(f"Saved classification report: {report_path}")

    return report


def save_confusion_matrix_csv(
    recorder: PredictionRecorder,
    output_dir,
):
    """
    Save confusion matrix as CSV.
    """

    output_dir = Path(output_dir)

    cm = confusion_matrix(
        recorder.true_labels,
        recorder.predicted_labels,
    )

    df = pd.DataFrame(
        cm,
        index=["Real", "Fake"],
        columns=["Pred Real", "Pred Fake"],
    )

    csv_path = output_dir / "confusion_matrix.csv"

    df.to_csv(csv_path)

    print(f"Saved confusion matrix: {csv_path}")

    return cm


def save_metrics_json(
    metrics,
    output_dir,
):
    """
    Save metrics dictionary as JSON.
    """

    output_dir = Path(output_dir)

    json_path = output_dir / "summary.json"

    with open(
        json_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
        )

    print(f"Saved JSON summary: {json_path}")


def save_metrics_csv(
    metrics,
    output_dir,
):
    """
    Save metrics dictionary as CSV.
    """

    output_dir = Path(output_dir)

    df = pd.DataFrame(
        metrics.items(),
        columns=["Metric", "Value"],
    )

    csv_path = output_dir / "summary.csv"

    df.to_csv(
        csv_path,
        index=False,
    )

    print(f"Saved CSV summary: {csv_path}")


# =============================================================================
# Master Save Function
# =============================================================================

def save_all_results(
    recorder: PredictionRecorder,
    output_dir,
):
    """
    Save every evaluation artifact except plots.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = compute_metrics(recorder)

    save_predictions_csv(
        recorder,
        output_dir,
    )

    save_classification_report(
        recorder,
        output_dir,
    )

    cm = save_confusion_matrix_csv(
        recorder,
        output_dir,
    )

    save_metrics_json(
        metrics,
        output_dir,
    )

    save_metrics_csv(
        metrics,
        output_dir,
    )

    return metrics, cm

# =============================================================================
# Visualization
# =============================================================================

def save_confusion_matrix_plot(
    recorder: PredictionRecorder,
    output_dir,
):
    """
    Save confusion matrix figure.
    """

    output_dir = Path(output_dir)

    cm = confusion_matrix(
        recorder.true_labels,
        recorder.predicted_labels,
    )

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Real", "Fake"],
    )

    fig, ax = plt.subplots(figsize=(6, 6))

    disp.plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )

    plt.title("Confusion Matrix")

    plt.tight_layout()

    plt.savefig(
        output_dir / "confusion_matrix.png",
        dpi=OUTPUT_DPI,
    )

    plt.close()


def save_roc_curve(
    recorder: PredictionRecorder,
    output_dir,
):
    """
    Save ROC curve.
    """

    output_dir = Path(output_dir)

    fpr, tpr, _ = roc_curve(
        recorder.true_labels,
        recorder.prob_fake,
    )

    auc = roc_auc_score(
        recorder.true_labels,
        recorder.prob_fake,
    )

    plt.figure(figsize=(6, 6))

    plt.plot(
        fpr,
        tpr,
        label=f"AUC = {auc:.4f}",
        linewidth=2,
    )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
    )

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_dir / "roc_curve.png",
        dpi=OUTPUT_DPI,
    )

    plt.close()


def save_pr_curve(
    recorder: PredictionRecorder,
    output_dir,
):
    """
    Save Precision-Recall Curve.
    """

    output_dir = Path(output_dir)

    precision, recall, _ = precision_recall_curve(
        recorder.true_labels,
        recorder.prob_fake,
    )

    ap = average_precision_score(
        recorder.true_labels,
        recorder.prob_fake,
    )

    plt.figure(figsize=(6, 6))

    plt.plot(
        recall,
        precision,
        linewidth=2,
        label=f"AP = {ap:.4f}",
    )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title("Precision-Recall Curve")

    plt.legend()

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        output_dir / "pr_curve.png",
        dpi=OUTPUT_DPI,
    )

    plt.close()


# =============================================================================
# Master Evaluation Function
# =============================================================================

def evaluate_predictions(
    recorder: PredictionRecorder,
    output_dir,
):
    """
    Complete evaluation pipeline.

    Returns
    -------
    dict
        Dictionary containing all evaluation metrics.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = recorder.dataframe()

    # Add useful columns
    df["confidence"] = df[
        ["prob_real", "prob_fake"]
    ].max(axis=1)

    df["predicted_class"] = np.where(
        df["predicted_label"] == 0,
        "Real",
        "Fake",
    )

    df.to_csv(
        output_dir / "predictions.csv",
        index=False,
    )

    metrics = compute_metrics(recorder)

    save_classification_report(
        recorder,
        output_dir,
    )

    save_confusion_matrix_csv(
        recorder,
        output_dir,
    )

    save_confusion_matrix_plot(
        recorder,
        output_dir,
    )

    save_roc_curve(
        recorder,
        output_dir,
    )

    save_pr_curve(
        recorder,
        output_dir,
    )

    save_metrics_json(
        metrics,
        output_dir,
    )

    save_metrics_csv(
        metrics,
        output_dir,
    )

    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)

    for key, value in metrics.items():
        print(f"{key:20s}: {value:.4f}")

    print("=" * 70)

    return metrics