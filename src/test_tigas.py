from src.config import TIGAS_DIR
from src.tigas_dataset import TIGASDataset
from src.transforms import get_eval_transforms


def main():
    for split in ("train", "val", "test"):
        dataset = TIGASDataset(
            root_dir=TIGAS_DIR,
            split=split,
            transform=get_eval_transforms(),
        )
        image, label = dataset[0]
        print(
            f"{split}: {len(dataset):,} images | "
            f"generators={len(dataset.generators)} | "
            f"tensor={tuple(image.shape)} | first_label={label}"
        )


if __name__ == "__main__":
    main()
