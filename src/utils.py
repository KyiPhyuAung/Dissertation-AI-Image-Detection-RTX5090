import os
import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


def set_random_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    """Give each DataLoader worker a deterministic NumPy/Python seed."""
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def split_dataset_indices(
    dataset_size: int,
    validation_fraction: float,
    seed: int,
) -> Tuple[list[int], list[int]]:
    """Create deterministic, non-overlapping train and validation indices."""
    if dataset_size < 2:
        raise ValueError("At least two samples are required to create a split.")
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be between 0 and 1.")

    validation_size = max(1, int(round(dataset_size * validation_fraction)))
    validation_size = min(validation_size, dataset_size - 1)

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(dataset_size, generator=generator).tolist()

    validation_indices = indices[:validation_size]
    train_indices = indices[validation_size:]
    return train_indices, validation_indices


def make_train_validation_subsets(
    training_dataset: Dataset,
    validation_dataset: Dataset,
    validation_fraction: float,
    seed: int,
) -> Tuple[Subset, Subset]:
    """
    Apply identical indices to two dataset instances with different transforms.

    The datasets must represent the same underlying files in the same order.
    """
    if len(training_dataset) != len(validation_dataset):
        raise ValueError("Training and validation dataset instances must match in size.")

    train_indices, validation_indices = split_dataset_indices(
        dataset_size=len(training_dataset),
        validation_fraction=validation_fraction,
        seed=seed,
    )

    return (
        Subset(training_dataset, train_indices),
        Subset(validation_dataset, validation_indices),
    )


def make_dataloader(
    dataset: Dataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    seed: int,
    prefetch_factor: int = 2,
    persistent_workers: bool = True,
) -> DataLoader:
    """Build a reproducible DataLoader suitable for Windows and CUDA."""
    generator = torch.Generator().manual_seed(seed)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
        worker_init_fn=seed_worker if num_workers > 0 else None,
        generator=generator,
    )
