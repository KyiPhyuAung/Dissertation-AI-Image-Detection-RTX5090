from __future__ import annotations

import csv
from pathlib import Path
from typing import Iterable, Optional

from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True
from torch.utils.data import Dataset


class TIGASDataset(Dataset):
    """CSV-backed loader for the TIGAS real-vs-fake image dataset.

    Expected structure::

        TIGAS/
            train/
                annotations01.csv
                images/<generator>/<0_real|1_fake>/...
            val/
                annotations01.csv
                images/<generator>/<0_real|1_fake>/...
            test/
                annotations01.csv
                images/<generator>/<0_real|1_fake>/...

    Labels follow the dataset convention: 0 = real, 1 = fake.
    """

    VALID_SPLITS = {"train", "val", "test"}

    def __init__(
        self,
        root_dir: str |Path,
        split: str,
        transform=None,
        generators: Optional[Iterable[str]] = None,
        return_metadata: bool = False,
    ) -> None:
        split = split.lower()
        if split not in self.VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(self.VALID_SPLITS)}, got {split!r}")

        self.root_dir = Path(root_dir)
        self.split = split
        self.split_dir = self.root_dir / split
        self.annotation_path = self.split_dir / "annotations01.csv"
        self.return_metadata = return_metadata
        self.transform = transform

        if not self.annotation_path.exists():
            raise FileNotFoundError(f"Missing TIGAS annotation file: {self.annotation_path}")

        selected_generators = None
        if generators is not None:
            selected_generators = {name.lower() for name in generators}
            if not selected_generators:
                raise ValueError("generators cannot be an empty collection")

        self.samples: list[tuple[Path, int, str]] = []
        with self.annotation_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            required_columns = {"image_path", "label"}
            if not required_columns.issubset(reader.fieldnames or []):
                raise ValueError(
                    f"{self.annotation_path} must contain columns {sorted(required_columns)}"
                )

            for row_number, row in enumerate(reader, start=2):
                relative_path = Path(row["image_path"].replace("\\", "/"))
                parts = relative_path.parts
                if len(parts) < 3:
                    raise ValueError(
                        f"Invalid image_path at row {row_number}: {row['image_path']!r}"
                    )

                generator = parts[1]
                if selected_generators is not None and generator.lower() not in selected_generators:
                    continue

                try:
                    label = int(row["label"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Invalid label at row {row_number}: {row['label']!r}"
                    ) from exc

                if label not in (0, 1):
                    raise ValueError(f"TIGAS label must be 0 or 1, got {label} at row {row_number}")

                image_path = self.split_dir / relative_path
                self.samples.append((image_path, label, generator))

        if not self.samples:
            filter_text = "all generators" if selected_generators is None else sorted(selected_generators)
            raise ValueError(f"No TIGAS samples found for split={split!r}, generators={filter_text}")

    @property
    def generators(self) -> list[str]:
        return sorted({sample[2] for sample in self.samples}, key=str.lower)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label, _generator = self.samples[index]

        try:
            with Image.open(image_path) as image:
                image = image.convert("RGB")
        except (FileNotFoundError, UnidentifiedImageError, OSError) as exc:
            raise RuntimeError(f"Unable to load TIGAS image: {image_path}") from exc

        if self.transform is not None:
            image = self.transform(image)

        if self.return_metadata:
            return image, label, str(image_path), _generator

        return image, label

    def get_generator(self, index: int) -> str:
        """Return the source/generator folder for a sample index."""
        return self.samples[index][2]
