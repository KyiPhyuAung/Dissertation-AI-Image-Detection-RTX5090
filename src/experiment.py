from __future__ import annotations

import argparse
import json
import platform
import time
from contextlib import nullcontext
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import sklearn
import torch
import torch.nn as nn
import torchvision
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from src.config import (
    BATCH_SIZE,
    CHANNELS_LAST,
    CHECKPOINT_DIR,
    CHECKPOINT_METRIC,
    CUDNN_BENCHMARK,
    DEVICE,
    FIGURES_DIR,
    IMAGE_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    NUM_WORKERS,
    PERSISTENT_WORKERS,
    PIN_MEMORY,
    PREFETCH_FACTOR,
    RANDOM_SEED,
    RESULTS_DIR,
    TIGAS_DIR,
    TINY_GENIMAGE_DIR,
    ART_IMAGES_DIR,
    CIFAKE_DIR,
    USE_AMP,
    USE_TF32,
    VALIDATION_FRACTION,
)
from src.genimage_dataset import GenImageDataset
from src.mixed_genimage_dataset import MixedGenImageDataset
from src.binary_folder_dataset import build_art_image_datasets, build_cifake_datasets
from src.models import build_model
from src.tigas_dataset import TIGASDataset
from src.transforms import get_eval_transforms, get_train_transforms
from src.utils import make_dataloader, make_train_validation_subsets, set_random_seed
from src.metrics import (
    PredictionRecorder,
    evaluate_predictions,
)

TINY_TEST_GENERATORS = [
    "imagenet_midjourney",
    "imagenet_ai_0424_sdv5",
    "imagenet_glide",
    "imagenet_ai_0419_biggan",
    "imagenet_ai_0508_adm",
    "imagenet_ai_0424_wukong",
    "imagenet_ai_0419_vqdm",
]

TINY_MIXED_TRAIN_GENERATORS = [
    "imagenet_midjourney",
    "imagenet_ai_0424_sdv5",
    "imagenet_glide",
]

TIGAS_GENERATORS = [
    "ADM", "art001", "art002_1", "art002_2", "art002_3", "art002_4",
    "biggan", "DALLE2", "face", "gaugan", "Glide", "Midjourney",
    "sd14", "sd15_1", "sd15_2", "sd_xl", "stargan", "VQDM", "wuk",
]


def configure_hardware() -> None:
    if DEVICE.type != "cuda":
        return
    torch.backends.cudnn.benchmark = CUDNN_BENCHMARK
    torch.backends.cudnn.deterministic = False
    if USE_TF32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")


def synchronize_device() -> None:
    if DEVICE.type == "cuda":
        torch.cuda.synchronize()


def autocast_context():
    if USE_AMP and DEVICE.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def create_grad_scaler():
    return torch.amp.GradScaler("cuda", enabled=USE_AMP and DEVICE.type == "cuda")


def create_loader(dataset, *, shuffle: bool):
    return make_dataloader(
        dataset=dataset,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        seed=RANDOM_SEED,
        prefetch_factor=PREFETCH_FACTOR,
        persistent_workers=PERSISTENT_WORKERS,
    )


def build_tiny_training_datasets(train_generator: str):
    dataset_class = MixedGenImageDataset if train_generator == "mixed" else GenImageDataset
    common_kwargs = {"root_dir": TINY_GENIMAGE_DIR, "split": "train"}
    if train_generator == "mixed":
        common_kwargs["generators"] = TINY_MIXED_TRAIN_GENERATORS
    else:
        common_kwargs["generator"] = train_generator

    augmented_dataset = dataset_class(**common_kwargs, transform=get_train_transforms())
    deterministic_dataset = dataset_class(**common_kwargs, transform=get_eval_transforms())
    return make_train_validation_subsets(
        training_dataset=augmented_dataset,
        validation_dataset=deterministic_dataset,
        validation_fraction=VALIDATION_FRACTION,
        seed=RANDOM_SEED,
    )


def build_tigas_training_datasets(train_generators: list[str] | None):
    train_dataset = TIGASDataset(
        root_dir=TIGAS_DIR,
        split="train",
        transform=get_train_transforms(),
        generators=train_generators,
    )
    validation_dataset = TIGASDataset(
        root_dir=TIGAS_DIR,
        split="val",
        transform=get_eval_transforms(),
        generators=train_generators,
    )
    return train_dataset, validation_dataset


def build_binary_training_datasets(dataset_name: str):
    if dataset_name == "art_images":
        train_dataset, validation_dataset, _ = build_art_image_datasets(
            get_train_transforms(),
            get_eval_transforms(),
        )
        return (
            train_dataset,
            validation_dataset,
            "deterministic class-stratified 80/10/10 split",
            "all ART_IMAGES samples",
        )

    if dataset_name == "cifake":
        train_dataset, validation_dataset, _ = build_cifake_datasets(
            get_train_transforms(),
            get_eval_transforms(),
        )
        return (
            train_dataset,
            validation_dataset,
            "official CIFAKE train split with deterministic 10% validation holdout",
            "official CIFAKE training set",
        )

    raise ValueError(f"Unsupported binary dataset: {dataset_name}")


def run_training_epoch(model, dataloader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in dataloader:
        images = images.to(DEVICE, non_blocking=PIN_MEMORY)
        labels = labels.to(DEVICE, non_blocking=PIN_MEMORY)
        if CHANNELS_LAST and DEVICE.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += batch_size

    return {"loss": running_loss / total, "accuracy": correct / total}


def run_validation_epoch(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    total = 0
    y_true, y_pred = [], []

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(DEVICE, non_blocking=PIN_MEMORY)
            labels_device = labels.to(DEVICE, non_blocking=PIN_MEMORY)
            if CHANNELS_LAST and DEVICE.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)

            with autocast_context():
                outputs = model(images)
                loss = criterion(outputs, labels_device)

            predicted = outputs.argmax(dim=1)
            batch_size = labels.size(0)
            running_loss += loss.item() * batch_size
            total += batch_size
            y_true.extend(labels.tolist())
            y_pred.extend(predicted.cpu().tolist())

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "loss": running_loss / total,
        "accuracy": accuracy,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
    }


def save_training_figures(history: list[dict], figure_dir: Path) -> None:
    figure_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_loss"], label="Training loss")
    plt.plot(df["epoch"], df["val_loss"], label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "training_validation_loss.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["train_accuracy"] * 100, label="Training accuracy")
    plt.plot(df["epoch"], df["val_accuracy"] * 100, label="Validation accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training and Validation Accuracy")
    plt.ylim(0, 100)
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "training_validation_accuracy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(df["epoch"], df["val_f1_macro"])
    plt.xlabel("Epoch")
    plt.ylabel("Macro F1")
    plt.title("Validation Macro F1")
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(figure_dir / "validation_macro_f1.png", dpi=300)
    plt.close()


def train_model(
    model_name: str,
    dataset_name: str,
    experiment_dir: Path,
    figure_dir: Path,
    train_generator: str | None,
    train_generators: list[str] | None,
    num_epochs: int,
    checkpoint_dir: Path,
):
    if dataset_name == "tigas":
        train_dataset, validation_dataset = build_tigas_training_datasets(train_generators)
        training_description = "all TIGAS generators" if train_generators is None else ", ".join(train_generators)
        validation_description = "official TIGAS validation split"
        checkpoint_tag = "all" if train_generators is None else "-".join(train_generators)
    elif dataset_name in {"art_images", "cifake"}:
        (
            train_dataset,
            validation_dataset,
            validation_description,
            training_description,
        ) = build_binary_training_datasets(dataset_name)
        checkpoint_tag = "all"
    else:
        if train_generator is None:
            raise ValueError("--train-generator is required when --dataset tiny")
        train_dataset, validation_dataset = build_tiny_training_datasets(train_generator)
        training_description = train_generator
        validation_description = "internal deterministic 10% split"
        checkpoint_tag = train_generator

    train_loader = create_loader(train_dataset, shuffle=True)
    validation_loader = create_loader(validation_dataset, shuffle=False)

    model = build_model(model_name).to(DEVICE)
    if CHANNELS_LAST and DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = create_grad_scaler()

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{model_name}_{dataset_name}_{checkpoint_tag}_best.pth"
    best_val_loss = float("inf")
    best_epoch = 0
    history: list[dict] = []

    print("=" * 78)
    print("Controlled Training Experiment")
    print("=" * 78)
    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Training source: {training_description}")
    print(f"Training images: {len(train_dataset):,}")
    print(f"Validation images: {len(validation_dataset):,}")
    print(f"Validation strategy: {validation_description}")
    print("Checkpoint criterion: lowest validation loss")
    print(f"Device: {DEVICE} | AMP: {USE_AMP} | TF32: {USE_TF32}")
    print("=" * 78)

    synchronize_device()
    training_start = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        synchronize_device()
        epoch_start = time.perf_counter()

        train_metrics = run_training_epoch(model, train_loader, criterion, optimizer, scaler)
        validation_metrics = run_validation_epoch(model, validation_loader, criterion)

        synchronize_device()
        epoch_seconds = time.perf_counter() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": validation_metrics["loss"],
            "val_accuracy": validation_metrics["accuracy"],
            "val_precision_macro": validation_metrics["precision_macro"],
            "val_recall_macro": validation_metrics["recall_macro"],
            "val_f1_macro": validation_metrics["f1_macro"],
            "learning_rate": optimizer.param_groups[0]["lr"],
            "epoch_seconds": epoch_seconds,
            "images_per_second": (len(train_dataset) + len(validation_dataset)) / epoch_seconds,
        }
        history.append(row)

        print(
            f"Epoch [{epoch:02d}/{num_epochs}] | "
            f"Train Loss {row['train_loss']:.4f} | Train Acc {row['train_accuracy'] * 100:.2f}% | "
            f"Val Loss {row['val_loss']:.4f} | Val Acc {row['val_accuracy'] * 100:.2f}% | "
            f"Val F1 {row['val_f1_macro']:.4f} | {epoch_seconds:.1f}s | "
            f"{row['images_per_second']:.1f} img/s"
        )

        if validation_metrics["loss"] < best_val_loss:
            best_val_loss = validation_metrics["loss"]
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved new best checkpoint: {checkpoint_path}")

    synchronize_device()
    total_training_seconds = time.perf_counter() - training_start
    pd.DataFrame(history).to_csv(experiment_dir / "training_history.csv", index=False)
    save_training_figures(history, figure_dir)

    best_row = history[best_epoch - 1]
    training_summary = {
        "checkpoint_metric": CHECKPOINT_METRIC,
        "best_epoch": best_epoch,
        "best_validation_loss": best_val_loss,
        "best_validation_accuracy": best_row["val_accuracy"],
        "best_validation_f1_macro": best_row["val_f1_macro"],
        "total_training_seconds": total_training_seconds,
        "average_epoch_seconds": total_training_seconds / num_epochs,
        "checkpoint_path": str(checkpoint_path),
    }
    with (experiment_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(training_summary, file, indent=4)

    print("=" * 78)
    print(
        f"Training finished in {total_training_seconds / 60:.2f} minutes. "
        f"Best epoch: {best_epoch}; best val loss: {best_val_loss:.4f}"
    )
    print("=" * 78)
    return checkpoint_path, training_summary


def calculate_metrics(y_true, y_pred, *, name: str, num_images: int, elapsed: float):
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=[0, 1], zero_division=0
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "test_generator": name,
        "num_images": num_images,
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "real_precision": precision[0],
        "real_recall": recall[0],
        "real_f1": f1[0],
        "fake_precision": precision[1],
        "fake_recall": recall[1],
        "fake_f1": f1[1],
        "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
        "fn": int(cm[1, 0]), "tp": int(cm[1, 1]),
        "evaluation_seconds": elapsed,
        "images_per_second": num_images / elapsed,
    }


def predict_dataset(model, dataset):
    loader = create_loader(dataset, shuffle=False)
    recorder = PredictionRecorder()

    model.eval()
    synchronize_device()
    start = time.perf_counter()

    sample_counter = 0

    with torch.inference_mode():
        for batch in loader:
            if len(batch) == 4:
                images, labels, paths, generators = batch
            else:
                images, labels = batch
                paths = [
                    f"sample_{sample_counter + i}"
                    for i in range(len(labels))
                ]
                generators = ["unknown"] * len(labels)

            images = images.to(
                DEVICE,
                non_blocking=PIN_MEMORY,
            )

            if CHANNELS_LAST and DEVICE.type == "cuda":
                images = images.contiguous(
                    memory_format=torch.channels_last
                )

            with autocast_context():
                outputs = model(images)
                probabilities = torch.softmax(outputs, dim=1)
                predictions = outputs.argmax(dim=1)

            probabilities = probabilities.cpu()
            predictions = predictions.cpu()

            for i in range(len(labels)):
                recorder.add(
                    image_path=str(paths[i]),
                    generator=str(generators[i]),
                    true_label=int(labels[i]),
                    predicted_label=int(predictions[i]),
                    probability_real=float(probabilities[i][0]),
                    probability_fake=float(probabilities[i][1]),
                )

            sample_counter += len(labels)

    synchronize_device()
    elapsed = time.perf_counter() - start

    return recorder, elapsed

def evaluate_tiny(model, train_generator: str):
    results = []
    for generator in TINY_TEST_GENERATORS:
        dataset = GenImageDataset(
            root_dir=TINY_GENIMAGE_DIR,
            generator=generator,
            split="val",
            transform=get_eval_transforms(),
        )
        recorder, elapsed = predict_dataset(model, dataset)

        y_true = recorder.true_labels
        y_pred = recorder.predicted_labels
        row = calculate_metrics(y_true, y_pred, name=generator, num_images=len(dataset), elapsed=elapsed)
        row["train_generator"] = train_generator
        results.append(row)
    return results


def evaluate_tigas(
    model,
    figure_dir: Path,
    evaluation_dir: Path,
):
    results = []
    all_true, all_pred = [], []
    master_recorder = PredictionRecorder()
    for generator in TIGAS_GENERATORS:
        dataset = TIGASDataset(
            root_dir=TIGAS_DIR,
            split="test",
            transform=get_eval_transforms(),
            generators=[generator],
            return_metadata=True,
        )
        recorder, elapsed = predict_dataset(model, dataset)
        master_recorder.image_paths.extend(recorder.image_paths)
        master_recorder.generators.extend(recorder.generators)
        master_recorder.true_labels.extend(recorder.true_labels)
        master_recorder.predicted_labels.extend(recorder.predicted_labels)
        master_recorder.prob_real.extend(recorder.prob_real)
        master_recorder.prob_fake.extend(recorder.prob_fake)
        y_true = recorder.true_labels

        y_pred = recorder.predicted_labels
        all_true.extend(y_true)
        all_pred.extend(y_pred)
        results.append(
            calculate_metrics(y_true, y_pred, name=generator, num_images=len(dataset), elapsed=elapsed)
        )

    overall_cm = confusion_matrix(all_true, all_pred, labels=[0, 1])
    overall = calculate_metrics(
        all_true,
        all_pred,
        name="ALL_TIGAS",
        num_images=len(all_true),
        elapsed=sum(row["evaluation_seconds"] for row in results),
    )

    display = ConfusionMatrixDisplay(overall_cm, display_labels=["Real", "Fake"])
    display.plot(values_format="d")
    plt.title("TIGAS Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figure_dir / "tigas_test_confusion_matrix.png", dpi=300)
    plt.close()

    df = pd.DataFrame(results)
    plt.figure(figsize=(12, 6))
    plt.bar(df["test_generator"], df["accuracy"] * 100)
    plt.xlabel("Generator / Source")
    plt.ylabel("Accuracy (%)")
    plt.title("TIGAS Test Accuracy by Generator")
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figure_dir / "tigas_accuracy_by_generator.png", dpi=300)
    plt.close()

    plt.figure(figsize=(12, 6))
    plt.bar(df["test_generator"], df["fake_recall"] * 100)
    plt.xlabel("Generator / Source")
    plt.ylabel("Fake Recall (%)")
    plt.title("TIGAS Fake Recall by Generator")
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(figure_dir / "tigas_fake_recall_by_generator.png", dpi=300)
    plt.close()

   

    evaluate_predictions(
    master_recorder,
    evaluation_dir,
    )

    return results, overall


def evaluate_binary_dataset(
    model,
    dataset_name: str,
    figure_dir: Path,
    evaluation_dir: Path,
):
    if dataset_name == "art_images":
        _, _, test_dataset = build_art_image_datasets(
            get_train_transforms(),
            get_eval_transforms(),
        )
        display_name = "ART_IMAGES"
    elif dataset_name == "cifake":
        _, _, test_dataset = build_cifake_datasets(
            get_train_transforms(),
            get_eval_transforms(),
        )
        display_name = "CIFAKE"
    else:
        raise ValueError(f"Unsupported binary dataset: {dataset_name}")

    recorder, elapsed = predict_dataset(model, test_dataset)
    overall = calculate_metrics(
        recorder.true_labels,
        recorder.predicted_labels,
        name=display_name,
        num_images=len(test_dataset),
        elapsed=elapsed,
    )

    evaluate_predictions(recorder, evaluation_dir)

    cm = confusion_matrix(
        recorder.true_labels,
        recorder.predicted_labels,
        labels=[0, 1],
    )
    display = ConfusionMatrixDisplay(cm, display_labels=["Real", "Fake"])
    display.plot(values_format="d")
    plt.title(f"{display_name} Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(figure_dir / f"{dataset_name}_test_confusion_matrix.png", dpi=300)
    plt.close()

    with (evaluation_dir / "overall_test_metrics.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(overall, file, indent=4)

    return overall


def collect_environment_info():
    info = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "pytorch_version": torch.__version__,
        "torchvision_version": torchvision.__version__,
        "scikit_learn_version": sklearn.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
    }
    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["gpu_vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
    return info


def run_experiment(
    model_name: str,
    dataset_name: str,
    train_generator: str | None,
    train_generators: list[str] | None,
    num_epochs: int,
):
    set_random_seed(RANDOM_SEED, deterministic=False)
    configure_hardware()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if dataset_name == "tigas":
        source_tag = "all" if train_generators is None else "-".join(train_generators)
    elif dataset_name == "tiny":
        source_tag = train_generator or "unknown"
    else:
        source_tag = "all"

    experiment_name = f"{timestamp}_{model_name}_{dataset_name}_train_{source_tag}"

    is_official = (
        num_epochs == NUM_EPOCHS
        and dataset_name in {"tigas", "art_images", "cifake"}
    )
    run_type = "official" if is_official else "experimental"

    if is_official:
        model_results_dir = RESULTS_DIR / "official" / dataset_name / model_name / experiment_name
        figure_dir = FIGURES_DIR / "official" / dataset_name / model_name / experiment_name
    else:
        model_results_dir = RESULTS_DIR / "experimental" / experiment_name
        figure_dir = FIGURES_DIR / "experimental" / experiment_name

    training_dir = model_results_dir / "training"
    evaluation_dir = model_results_dir / "evaluation"
    checkpoint_dir = CHECKPOINT_DIR / run_type / dataset_name / experiment_name

    training_dir.mkdir(parents=True, exist_ok=True)
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    experiment_dir = training_dir
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    dataset_paths = {
        "tiny": TINY_GENIMAGE_DIR,
        "tigas": TIGAS_DIR,
        "art_images": ART_IMAGES_DIR,
        "cifake": CIFAKE_DIR,
    }

    config = {
        "model": model_name,
        "dataset": dataset_name,
        "dataset_path": str(dataset_paths[dataset_name]),
        "train_generator": train_generator,
        "train_generators": train_generators,
        "device": str(DEVICE),
        "batch_size": BATCH_SIZE,
        "num_workers": NUM_WORKERS,
        "prefetch_factor": PREFETCH_FACTOR,
        "pin_memory": PIN_MEMORY,
        "persistent_workers": PERSISTENT_WORKERS,
        "learning_rate": LEARNING_RATE,
        "optimizer": "Adam",
        "loss_function": "CrossEntropyLoss",
        "num_epochs": num_epochs,
        "image_size": IMAGE_SIZE,
        "random_seed": RANDOM_SEED,
        "split_strategy": (
            "official TIGAS train/val/test"
            if dataset_name == "tigas"
            else "ART_IMAGES deterministic class-stratified 80/10/10"
            if dataset_name == "art_images"
            else "CIFAKE official test; 10% of official train held out for validation"
            if dataset_name == "cifake"
            else "Tiny GenImage deterministic validation holdout"
        ),
        "label_mapping": {"REAL": 0, "FAKE": 1},
        "checkpoint_metric": CHECKPOINT_METRIC,
        "use_amp": USE_AMP,
        "use_tf32": USE_TF32,
        "channels_last": CHANNELS_LAST,
        "cudnn_benchmark": CUDNN_BENCHMARK,
        "environment": collect_environment_info(),
    }
    with (experiment_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(config, file, indent=4)

    experiment_start = time.perf_counter()
    checkpoint_path, training_summary = train_model(
        model_name=model_name,
        dataset_name=dataset_name,
        experiment_dir=experiment_dir,
        figure_dir=figure_dir,
        train_generator=train_generator,
        train_generators=train_generators,
        num_epochs=num_epochs,
        checkpoint_dir=checkpoint_dir,
    )

    model = build_model(model_name).to(DEVICE)
    if CHANNELS_LAST and DEVICE.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE, weights_only=True))

    print("=" * 78)
    print("Final Test Evaluation")
    print("=" * 78)

    if dataset_name == "tigas":
        results, overall = evaluate_tigas(
            model,
            figure_dir,
            evaluation_dir,
        )
        for row in results:
            row.update(
                {
                    "model": model_name,
                    "best_epoch": training_summary["best_epoch"],
                    "train_source": source_tag,
                }
            )
            print(
                f"{row['test_generator']}: Accuracy {row['accuracy'] * 100:.2f}% | "
                f"Macro F1 {row['macro_f1']:.4f} | "
                f"Fake Recall {row['fake_recall'] * 100:.2f}% | "
                f"{row['evaluation_seconds']:.1f}s"
            )
        pd.DataFrame(results).to_csv(
            evaluation_dir / "per_generator_test_results.csv",
            index=False,
        )
        with (evaluation_dir / "overall_test_metrics.json").open(
            "w", encoding="utf-8"
        ) as file:
            json.dump(overall, file, indent=4)

        print("-" * 78)
        print(
            f"OVERALL TIGAS TEST: Accuracy {overall['accuracy'] * 100:.2f}% | "
            f"Macro F1 {overall['macro_f1']:.4f} | "
            f"Fake Recall {overall['fake_recall'] * 100:.2f}%"
        )
    elif dataset_name in {"art_images", "cifake"}:
        overall = evaluate_binary_dataset(
            model,
            dataset_name,
            figure_dir,
            evaluation_dir,
        )
        print("-" * 78)
        print(
            f"OVERALL {dataset_name.upper()} TEST: "
            f"Accuracy {overall['accuracy'] * 100:.2f}% | "
            f"Macro F1 {overall['macro_f1']:.4f} | "
            f"Fake Recall {overall['fake_recall'] * 100:.2f}% | "
            f"{overall['evaluation_seconds']:.1f}s"
        )
    else:
        results = evaluate_tiny(model, train_generator or "unknown")
        for row in results:
            row.update(
                {
                    "model": model_name,
                    "best_epoch": training_summary["best_epoch"],
                }
            )
            print(
                f"{row['test_generator']}: Accuracy {row['accuracy'] * 100:.2f}% | "
                f"Macro F1 {row['macro_f1']:.4f} | "
                f"Fake Recall {row['fake_recall'] * 100:.2f}% | "
                f"{row['evaluation_seconds']:.1f}s"
            )
        pd.DataFrame(results).to_csv(
            experiment_dir / "cross_generator_results.csv",
            index=False,
        )

    total_experiment_seconds = time.perf_counter() - experiment_start
    completion = {
        "total_experiment_seconds": total_experiment_seconds,
        "total_experiment_minutes": total_experiment_seconds / 60,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "training_directory": str(training_dir),
        "evaluation_directory": str(evaluation_dir),
        "figures_directory": str(figure_dir),
    }
    with (experiment_dir / "experiment_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(completion, file, indent=4)

    print("=" * 78)
    print(f"Experiment completed in {total_experiment_seconds / 60:.2f} minutes.")
    print(f"Saved training results to: {training_dir}")
    print(f"Saved evaluation results to: {evaluation_dir}")
    print(f"Saved figures to: {figure_dir}")
    print("=" * 78)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Tiny GenImage, TIGAS, ART_IMAGES, or CIFAKE experiments."
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["resnet18", "efficientnet_b0", "convnext_tiny"],
    )
    parser.add_argument("--dataset", choices=["tiny", "tigas", "art_images", "cifake"], default="tiny")
    parser.add_argument(
        "--train-generator",
        choices=TINY_TEST_GENERATORS + ["mixed"],
    )
    parser.add_argument(
        "--train-generators",
        nargs="+",
        choices=TIGAS_GENERATORS,
        help="Optional TIGAS generator subset. Omit to train on all 19 sources.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override the number of training epochs from config.py.",
    )
    args = parser.parse_args()

    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be at least 1")
    if args.dataset == "tiny" and args.train_generator is None:
        parser.error("--train-generator is required when --dataset tiny")
    if args.dataset != "tiny" and args.train_generator is not None:
        parser.error("--train-generator is only valid with --dataset tiny")
    if args.dataset != "tigas" and args.train_generators is not None:
        parser.error("--train-generators is only valid with --dataset tigas")

    run_experiment(
        model_name=args.model,
        dataset_name=args.dataset,
        train_generator=args.train_generator,
        train_generators=args.train_generators,
        num_epochs=args.epochs if args.epochs is not None else NUM_EPOCHS,
    )
