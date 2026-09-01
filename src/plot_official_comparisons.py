from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import RESULTS_DIR, FIGURES_DIR

DATASETS = ["tigas", "art_images", "cifake"]
MODELS = ["resnet18", "efficientnet_b0", "convnext_tiny"]

DATASET_NAMES = {
    "tigas": "TIGAS",
    "art_images": "ART_IMAGES",
    "cifake": "CIFAKE",
}
MODEL_NAMES = {
    "resnet18": "ResNet18",
    "efficientnet_b0": "EfficientNet-B0",
    "convnext_tiny": "ConvNeXt-Tiny",
}

FIGURE_OUT = FIGURES_DIR / "official_comparisons"
TABLE_OUT = RESULTS_DIR / "official_comparisons"
FIGURE_OUT.mkdir(parents=True, exist_ok=True)
TABLE_OUT.mkdir(parents=True, exist_ok=True)

DPI = 300

# Used only when old TIGAS runs do not contain readable JSON summaries.
LEGACY_TIGAS = {
    "resnet18": {"accuracy": 0.9740, "macro_f1": 0.9740, "training_minutes": 145.22},
    "efficientnet_b0": {"accuracy": 0.9841, "macro_f1": 0.9841, "training_minutes": 392.71},
    "convnext_tiny": {"accuracy": 0.9905, "macro_f1": 0.9905, "training_minutes": 138.69},
}


def read_json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def find_json(run_dir: Path, names: list[str]):
    for name in names:
        for path in [
            run_dir / name,
            run_dir / "training" / name,
            run_dir / "evaluation" / name,
        ]:
            if path.exists():
                data = read_json(path)
                if data is not None:
                    return path, data

        for path in sorted(run_dir.rglob(name)):
            data = read_json(path)
            if data is not None:
                return path, data

    return None, None


def latest_complete_run(dataset: str, model: str):
    base = RESULTS_DIR / "official" / dataset / model
    if not base.exists():
        return None

    runs = sorted([p for p in base.iterdir() if p.is_dir()], key=lambda p: p.name, reverse=True)

    for run in runs:
        _, metrics = find_json(run, ["summary.json", "overall_test_metrics.json"])
        if metrics and "accuracy" in metrics:
            return run

    return runs[0] if runs else None


def extract_row(dataset: str, model: str):
    run = latest_complete_run(dataset, model)

    row = {
        "dataset": dataset,
        "dataset_display": DATASET_NAMES[dataset],
        "model": model,
        "model_display": MODEL_NAMES[model],
        "accuracy": np.nan,
        "precision": np.nan,
        "recall": np.nan,
        "macro_f1": np.nan,
        "weighted_f1": np.nan,
        "balanced_accuracy": np.nan,
        "mcc": np.nan,
        "roc_auc": np.nan,
        "pr_auc": np.nan,
        "best_epoch": np.nan,
        "best_validation_loss": np.nan,
        "training_minutes": np.nan,
        "experiment_minutes": np.nan,
        "run_directory": str(run) if run else "",
        "metrics_source": "",
        "training_source": "",
        "experiment_source": "",
    }

    if run:
        p, metrics = find_json(run, ["summary.json", "overall_test_metrics.json"])
        if metrics:
            for key in [
                "accuracy", "precision", "recall", "macro_f1", "weighted_f1",
                "balanced_accuracy", "mcc", "roc_auc", "pr_auc"
            ]:
                if key in metrics:
                    row[key] = metrics[key]
            row["metrics_source"] = str(p)

        p, training = find_json(run, ["training_summary.json"])
        if training:
            row["best_epoch"] = training.get("best_epoch", np.nan)
            row["best_validation_loss"] = training.get("best_validation_loss", np.nan)
            if "total_training_seconds" in training:
                row["training_minutes"] = training["total_training_seconds"] / 60.0
            row["training_source"] = str(p)

        p, experiment = find_json(run, ["experiment_summary.json"])
        if experiment:
            if "total_experiment_minutes" in experiment:
                row["experiment_minutes"] = experiment["total_experiment_minutes"]
            elif "total_experiment_seconds" in experiment:
                row["experiment_minutes"] = experiment["total_experiment_seconds"] / 60.0
            row["experiment_source"] = str(p)

    # Old TIGAS official runs may predate the newer result layout.
    if dataset == "tigas":
        fallback = LEGACY_TIGAS[model]

        if pd.isna(row["accuracy"]):
            row["accuracy"] = fallback["accuracy"]
            row["metrics_source"] = "LEGACY_TIGAS_FALLBACK"
            print(f"WARNING: using recorded TIGAS accuracy for {MODEL_NAMES[model]}.")

        if pd.isna(row["macro_f1"]):
            row["macro_f1"] = fallback["macro_f1"]

        if pd.isna(row["training_minutes"]):
            row["training_minutes"] = fallback["training_minutes"]
            row["training_source"] = "LEGACY_TIGAS_FALLBACK"
            print(f"WARNING: using recorded TIGAS training time for {MODEL_NAMES[model]}.")

    return row


def validate(df: pd.DataFrame):
    # New datasets must come from actual official files; no hard-coded fallback.
    new = df[df["dataset"].isin(["art_images", "cifake"])]
    missing = new[new["accuracy"].isna()]

    if not missing.empty:
        text = ", ".join(
            f"{r.dataset_display}/{r.model_display}" for r in missing.itertuples()
        )
        raise RuntimeError(
            f"Missing official result(s): {text}. Finish all official runs first."
        )


def label_bars(ax, suffix=""):
    for container in ax.containers:
        labels = []
        for bar in container:
            h = bar.get_height()
            labels.append("" if pd.isna(h) else f"{h:.2f}{suffix}")
        ax.bar_label(container, labels=labels, padding=3, fontsize=9)


def single_chart(df, dataset, column, title, ylabel, filename, multiplier=1.0, percent=False):
    part = (
        df[df["dataset"] == dataset]
        .set_index("model")
        .reindex(MODELS)
        .reset_index()
    )
    values = part[column].astype(float) * multiplier

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(part["model_display"], values)
    ax.set_title(title)
    ax.set_xlabel("Model")
    ax.set_ylabel(ylabel)

    if percent:
        ax.set_ylim(0, 100)
        label_bars(ax, "%")
    else:
        label_bars(ax)

    fig.tight_layout()
    fig.savefig(FIGURE_OUT / filename, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def grouped_chart(df, column, title, ylabel, filename, multiplier=1.0, percent=False):
    pivot = (
        df.pivot(index="dataset_display", columns="model_display", values=column)
        .reindex(index=[DATASET_NAMES[d] for d in DATASETS])
        .reindex(columns=[MODEL_NAMES[m] for m in MODELS])
        * multiplier
    )

    ax = pivot.plot(kind="bar", figsize=(11, 6), width=0.8)
    ax.set_title(title)
    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel)
    ax.tick_params(axis="x", rotation=0)

    if percent:
        ax.set_ylim(0, 100)
        label_bars(ax, "%")
    else:
        label_bars(ax)

    plt.legend(title="Model")
    plt.tight_layout()
    plt.savefig(FIGURE_OUT / filename, dpi=DPI, bbox_inches="tight")
    plt.close()


def generate_figures(df: pd.DataFrame):
    for dataset in DATASETS:
        name = DATASET_NAMES[dataset]

        single_chart(
            df, dataset, "accuracy",
            f"{name} Official Test Accuracy",
            "Accuracy (%)",
            f"{dataset}_accuracy_comparison.png",
            multiplier=100.0,
            percent=True,
        )

        single_chart(
            df, dataset, "macro_f1",
            f"{name} Official Test Macro F1",
            "Macro F1 (%)",
            f"{dataset}_macro_f1_comparison.png",
            multiplier=100.0,
            percent=True,
        )

        if not df[df["dataset"] == dataset]["training_minutes"].isna().all():
            single_chart(
                df, dataset, "training_minutes",
                f"{name} Model Training Time",
                "Training Time (minutes)",
                f"{dataset}_training_time_comparison.png",
            )

        if not df[df["dataset"] == dataset]["experiment_minutes"].isna().all():
            single_chart(
                df, dataset, "experiment_minutes",
                f"{name} Total Experiment Time",
                "Total Experiment Time (minutes)",
                f"{dataset}_experiment_time_comparison.png",
            )

    grouped_chart(
        df, "accuracy",
        "Official Test Accuracy Across Datasets",
        "Accuracy (%)",
        "all_datasets_accuracy_comparison.png",
        multiplier=100.0,
        percent=True,
    )

    grouped_chart(
        df, "macro_f1",
        "Official Test Macro F1 Across Datasets",
        "Macro F1 (%)",
        "all_datasets_macro_f1_comparison.png",
        multiplier=100.0,
        percent=True,
    )

    grouped_chart(
        df, "training_minutes",
        "Model Training Time Across Datasets",
        "Training Time (minutes)",
        "all_datasets_training_time_comparison.png",
    )

    # Total experiment time is shown only for datasets with all 3 values.
    complete_datasets = []
    for dataset in DATASETS:
        part = df[df["dataset"] == dataset]
        if len(part) == 3 and not part["experiment_minutes"].isna().any():
            complete_datasets.append(dataset)

    if complete_datasets:
        exp_df = df[df["dataset"].isin(complete_datasets)].copy()
        pivot = (
            exp_df.pivot(index="dataset_display", columns="model_display", values="experiment_minutes")
            .reindex(columns=[MODEL_NAMES[m] for m in MODELS])
        )
        ax = pivot.plot(kind="bar", figsize=(11, 6), width=0.8)
        ax.set_title("Total Experiment Time Across Datasets")
        ax.set_xlabel("Dataset")
        ax.set_ylabel("Total Experiment Time (minutes)")
        ax.tick_params(axis="x", rotation=0)
        label_bars(ax)
        plt.legend(title="Model")
        plt.tight_layout()
        plt.savefig(
            FIGURE_OUT / "all_datasets_experiment_time_comparison.png",
            dpi=DPI,
            bbox_inches="tight",
        )
        plt.close()


def save_table(df: pd.DataFrame):
    out = df.copy()
    for col in [
        "accuracy", "precision", "recall", "macro_f1",
        "weighted_f1", "balanced_accuracy", "roc_auc", "pr_auc"
    ]:
        out[f"{col}_percent"] = out[col] * 100.0

    path = TABLE_OUT / "official_model_dataset_comparison.csv"
    out.to_csv(path, index=False)

    display = df[
        [
            "dataset_display", "model_display", "accuracy", "macro_f1", "mcc",
            "best_epoch", "best_validation_loss",
            "training_minutes", "experiment_minutes"
        ]
    ].copy()

    display["accuracy"] = display["accuracy"].map(
        lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "-"
    )
    display["macro_f1"] = display["macro_f1"].map(
        lambda x: f"{x * 100:.2f}%" if pd.notna(x) else "-"
    )

    print("\n" + "=" * 100)
    print("OFFICIAL MULTI-DATASET MODEL COMPARISON")
    print("=" * 100)
    print(display.to_string(index=False))
    print("=" * 100)
    print("Saved table:", path)


def main():
    print("=" * 78)
    print("Official Multi-Dataset Comparison")
    print("=" * 78)

    rows = [
        extract_row(dataset, model)
        for dataset in DATASETS
        for model in MODELS
    ]
    df = pd.DataFrame(rows)

    validate(df)
    save_table(df)
    generate_figures(df)

    print("\nDone.")
    print("Figures:", FIGURE_OUT)
    print("Table  :", TABLE_OUT / "official_model_dataset_comparison.csv")


if __name__ == "__main__":
    main()
