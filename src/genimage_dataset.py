from pathlib import Path
from PIL import Image

from torch.utils.data import Dataset


class GenImageDataset(Dataset):
    """
    Dataset loader for Tiny GenImage / GenImage.

    Folder structure:

    generator/
        train/
            ai/
            nature/

        val/
            ai/
            nature/
    """

    def __init__(self, root_dir, generator, split="train", transform=None):

        self.transform = transform

        self.dataset_path = (
            Path(root_dir)
            / generator
            / split
        )

        self.samples = []

        class_mapping = {
            "nature": 0,
            "ai": 1
        }

        for class_name, label in class_mapping.items():

            class_dir = self.dataset_path / class_name

            if not class_dir.exists():
                raise FileNotFoundError(class_dir)

            for image_path in sorted(class_dir.rglob("*")):


                if (
                    image_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]
                    and not image_path.name.startswith("._")
                ):

                    self.samples.append(
                        (image_path, label)
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        image_path, label = self.samples[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label