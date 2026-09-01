from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset, Subset

from src.config import (
    ART_IMAGES_DIR,
    ART_TRAIN_FRACTION,
    ART_VALIDATION_FRACTION,
    CIFAKE_DIR,
    CIFAKE_VALIDATION_FRACTION,
    RANDOM_SEED,
)


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
}


class BinaryFolderDataset(Dataset):
    """
    Dataset for binary REAL/FAKE folder structures.

    Label convention is kept consistent with the existing project:
        0 = Real
        1 = Fake / AI-generated
    """

    CLASS_TO_LABEL = {
        "REAL": 0,
        "FAKE": 1,
    }

    def __init__(
        self,
        root_dir: str | Path,
        transform=None,
        return_metadata: bool = False,
        source_name: str = "unknown",
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.return_metadata = return_metadata
        self.source_name = source_name
        self.samples: list[tuple[Path, int]] = []

        if not self.root_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory does not exist: {self.root_dir}"
            )

        for class_name, label in self.CLASS_TO_LABEL.items():
            class_dir = self.root_dir / class_name

            if not class_dir.exists():
                raise FileNotFoundError(
                    f"Expected class directory not found: {class_dir}"
                )

            image_paths = sorted(
                path
                for path in class_dir.rglob("*")
                if path.is_file()
                and path.suffix.lower() in SUPPORTED_EXTENSIONS
            )

            self.samples.extend(
                (image_path, label)
                for image_path in image_paths
            )

        if not self.samples:
            raise RuntimeError(
                f"No supported images found in {self.root_dir}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]

        with Image.open(image_path) as image:
            image = image.convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        if self.return_metadata:
            return (
                image,
                label,
                str(image_path),
                self.source_name,
            )

        return image, label


def _make_stratified_split_indices(
    dataset: BinaryFolderDataset,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
):
    """
    Create deterministic class-stratified train/validation/test indices.
    """

    generator = torch.Generator().manual_seed(seed)

    train_indices = []
    validation_indices = []
    test_indices = []

    for label in (0, 1):
        class_indices = [
            index
            for index, (_, sample_label) in enumerate(dataset.samples)
            if sample_label == label
        ]

        permutation = torch.randperm(
            len(class_indices),
            generator=generator,
        ).tolist()

        shuffled_indices = [
            class_indices[i]
            for i in permutation
        ]

        num_train = int(len(shuffled_indices) * train_fraction)
        num_validation = int(
            len(shuffled_indices) * validation_fraction
        )

        train_end = num_train
        validation_end = num_train + num_validation

        train_indices.extend(
            shuffled_indices[:train_end]
        )
        validation_indices.extend(
            shuffled_indices[train_end:validation_end]
        )
        test_indices.extend(
            shuffled_indices[validation_end:]
        )

    return train_indices, validation_indices, test_indices


def build_art_image_datasets(
    train_transform,
    eval_transform,
):
    """
    ART_IMAGES has no official split.

    A deterministic class-stratified 80/10/10 split is created.
    """

    indexing_dataset = BinaryFolderDataset(
        ART_IMAGES_DIR,
        transform=None,
        source_name="ART_IMAGES",
    )

    train_indices, validation_indices, test_indices = (
        _make_stratified_split_indices(
            dataset=indexing_dataset,
            train_fraction=ART_TRAIN_FRACTION,
            validation_fraction=ART_VALIDATION_FRACTION,
            seed=RANDOM_SEED,
        )
    )

    training_dataset = BinaryFolderDataset(
        ART_IMAGES_DIR,
        transform=train_transform,
        source_name="ART_IMAGES",
    )

    validation_dataset = BinaryFolderDataset(
        ART_IMAGES_DIR,
        transform=eval_transform,
        source_name="ART_IMAGES",
    )

    test_dataset = BinaryFolderDataset(
        ART_IMAGES_DIR,
        transform=eval_transform,
        return_metadata=True,
        source_name="ART_IMAGES",
    )

    return (
        Subset(training_dataset, train_indices),
        Subset(validation_dataset, validation_indices),
        Subset(test_dataset, test_indices),
    )


def build_cifake_datasets(
    train_transform,
    eval_transform,
):
    """
    CIFAKE provides official train and test directories.

    The official training set is split deterministically into
    90% training and 10% validation.

    The official test set remains untouched.
    """

    cifake_train_dir = CIFAKE_DIR / "train"
    cifake_test_dir = CIFAKE_DIR / "test"

    indexing_dataset = BinaryFolderDataset(
        cifake_train_dir,
        transform=None,
        source_name="CIFAKE",
    )

    train_indices, validation_indices, remainder_indices = (
        _make_stratified_split_indices(
            dataset=indexing_dataset,
            train_fraction=1.0 - CIFAKE_VALIDATION_FRACTION,
            validation_fraction=CIFAKE_VALIDATION_FRACTION,
            seed=RANDOM_SEED,
        )
    )

    if remainder_indices:
        raise RuntimeError(
            "Unexpected CIFAKE split remainder detected."
        )

    training_dataset = BinaryFolderDataset(
        cifake_train_dir,
        transform=train_transform,
        source_name="CIFAKE",
    )

    validation_dataset = BinaryFolderDataset(
        cifake_train_dir,
        transform=eval_transform,
        source_name="CIFAKE",
    )

    test_dataset = BinaryFolderDataset(
        cifake_test_dir,
        transform=eval_transform,
        return_metadata=True,
        source_name="CIFAKE",
    )

    return (
        Subset(training_dataset, train_indices),
        Subset(validation_dataset, validation_indices),
        test_dataset,
    )