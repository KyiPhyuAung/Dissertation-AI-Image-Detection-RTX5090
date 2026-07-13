"""Standalone training entry point using the controlled experiment pipeline."""

import argparse
from datetime import datetime

from src.config import RESULTS_DIR
from src.experiment import TEST_GENERATORS, train_model
from src.utils import set_random_seed
from src.config import RANDOM_SEED


def train(model_name: str, generator: str):
    set_random_seed(RANDOM_SEED, deterministic=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = RESULTS_DIR / f"{timestamp}_{model_name}_train_only_{generator}"
    experiment_dir.mkdir(parents=True, exist_ok=False)

    return train_model(
        model_name=model_name,
        train_generator=generator,
        experiment_dir=experiment_dir,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        required=True,
        choices=["resnet18", "efficientnet_b0", "convnext_tiny"],
    )
    parser.add_argument(
        "--generator",
        required=True,
        choices=TEST_GENERATORS + ["mixed"],
    )
    args = parser.parse_args()
    train(args.model, args.generator)
