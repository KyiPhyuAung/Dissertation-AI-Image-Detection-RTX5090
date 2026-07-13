from __future__ import annotations

import concurrent.futures
import warnings
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageFile
from tqdm import tqdm

from src.config import TIGAS_DIR

ImageFile.LOAD_TRUNCATED_IMAGES = True

warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
warnings.filterwarnings("ignore", message="Palette images with Transparency")


OUTPUT_DIR = Path("results") / "dataset_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SPLITS = ["train", "val", "test"]


def megapixels(width: int, height: int) -> float:
    return (width * height) / 1_000_000


def resolution_category(width: int, height: int) -> str:
    shorter = min(width, height)

    if shorter >= 2160:
        return "4K+"

    if shorter >= 1440:
        return "1440p"

    if shorter >= 1080:
        return "1080p"

    if shorter >= 720:
        return "720p"

    return "<720p"


def scan_image(args):
    image_path, split, generator, label = args

    try:
        with Image.open(image_path) as img:

            width, height = img.size

            return {
                "split": split,
                "generator": generator,
                "label": label,
                "width": width,
                "height": height,
                "pixels": width * height,
                "megapixels": megapixels(width, height),
                "aspect_ratio": round(width / height, 4),
                "format": img.format,
                "resolution_class": resolution_category(width, height),
                "path": str(image_path),
            }

    except Exception as e:

        return {
            "split": split,
            "generator": generator,
            "label": label,
            "width": np.nan,
            "height": np.nan,
            "pixels": np.nan,
            "megapixels": np.nan,
            "aspect_ratio": np.nan,
            "format": "CORRUPTED",
            "resolution_class": "UNKNOWN",
            "path": str(image_path),
            "error": str(e),
        }


def build_file_list():

    files = []

    for split in SPLITS:

        csv_path = TIGAS_DIR / split / "annotations01.csv"

        df = pd.read_csv(csv_path)

        for _, row in df.iterrows():

            relative = Path(row["image_path"])

            image_path = TIGAS_DIR / split / relative

            generator = relative.parts[1]

            label = int(row["label"])

            files.append(
                (
                    image_path,
                    split,
                    generator,
                    label,
                )
            )

    return files


def scan_dataset():

    print("=" * 70)
    print("Scanning TIGAS Dataset")
    print("=" * 70)

    files = build_file_list()

    print(f"Images found : {len(files):,}")

    metadata = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:

        iterator = executor.map(scan_image, files)

        for item in tqdm(iterator, total=len(files)):
            metadata.append(item)

    df = pd.DataFrame(metadata)

    csv_path = OUTPUT_DIR / "image_metadata.csv"

    df.to_csv(csv_path, index=False)

    print(f"\nMetadata saved to:\n{csv_path}")

    return df

# =============================================================================
# Statistics
# =============================================================================

def compute_summary(df):

    clean = df[df["format"] != "CORRUPTED"].copy()

    summary = {
        "Total Images": len(clean),
        "Train Images": (clean["split"] == "train").sum(),
        "Validation Images": (clean["split"] == "val").sum(),
        "Test Images": (clean["split"] == "test").sum(),
        "Real Images": (clean["label"] == 0).sum(),
        "Fake Images": (clean["label"] == 1).sum(),
        "Generators": clean["generator"].nunique(),

        "Min Width": clean["width"].min(),
        "Max Width": clean["width"].max(),
        "Mean Width": round(clean["width"].mean(), 2),
        "Median Width": round(clean["width"].median(), 2),

        "Min Height": clean["height"].min(),
        "Max Height": clean["height"].max(),
        "Mean Height": round(clean["height"].mean(), 2),
        "Median Height": round(clean["height"].median(), 2),

        "Min MP": round(clean["megapixels"].min(), 2),
        "Max MP": round(clean["megapixels"].max(), 2),
        "Mean MP": round(clean["megapixels"].mean(), 2),
        "Median MP": round(clean["megapixels"].median(), 2),
    }

    summary_df = pd.DataFrame(
        summary.items(),
        columns=["Metric", "Value"]
    )

    summary_df.to_csv(
        OUTPUT_DIR / "dataset_summary.csv",
        index=False
    )

    return clean


def compute_resolution_statistics(clean):

    total = len(clean)

    stats = []

    def count_pixels(mp):

        return (clean["megapixels"] >= mp).sum()

    def count_short_side(short_side):

        shorter = np.minimum(clean["width"], clean["height"])

        return (shorter >= short_side).sum()

    thresholds = [
        ("720p", count_short_side(720)),
        ("1080p", count_short_side(1080)),
        ("1440p", count_short_side(1440)),
        ("4K", count_short_side(2160)),
        ("2 MP", count_pixels(2)),
        ("8 MP", count_pixels(8)),
    ]

    for name, count in thresholds:

        stats.append({
            "Threshold": name,
            "Images": count,
            "Percentage": round(count / total * 100, 2)
        })

    pd.DataFrame(stats).to_csv(
        OUTPUT_DIR / "high_resolution_statistics.csv",
        index=False
    )


def generator_statistics(clean):

    rows = []

    for generator, group in clean.groupby("generator"):

        shorter = np.minimum(group["width"], group["height"])

        rows.append({

            "Generator": generator,

            "Images": len(group),

            "Mean Width": round(group["width"].mean(), 2),

            "Mean Height": round(group["height"].mean(), 2),

            "Mean MP": round(group["megapixels"].mean(), 2),

            "Median MP": round(group["megapixels"].median(), 2),

            "Largest MP": round(group["megapixels"].max(), 2),

            "Smallest MP": round(group["megapixels"].min(), 2),

            "1080p %": round(
                (shorter >= 1080).sum() / len(group) * 100,
                2
            ),

            "JPEG": (group["format"] == "JPEG").sum(),

            "PNG": (group["format"] == "PNG").sum()

        })

    generator_df = pd.DataFrame(rows)

    generator_df = generator_df.sort_values(
        "Mean MP",
        ascending=False
    )

    generator_df.to_csv(
        OUTPUT_DIR / "generator_statistics.csv",
        index=False
    )


def class_statistics(clean):

    class_df = (
        clean.groupby("label")
        .agg(
            Images=("label", "count"),
            Mean_MP=("megapixels", "mean"),
            Mean_Width=("width", "mean"),
            Mean_Height=("height", "mean")
        )
    )

    class_df.index = ["Real", "Fake"]

    class_df.to_csv(
        OUTPUT_DIR / "class_statistics.csv"
    )


def file_format_statistics(clean):

    fmt = clean["format"].value_counts()

    fmt = fmt.rename_axis("Format").reset_index(name="Count")

    fmt.to_csv(
        OUTPUT_DIR / "image_formats.csv",
        index=False
    )

    # =============================================================================
# Visualization
# =============================================================================

DPI = 300


def save_histogram(
    data,
    title,
    xlabel,
    filename,
    bins=50,
):

    plt.figure(figsize=(10, 6))

    plt.hist(
        data,
        bins=bins,
        edgecolor="black",
    )

    plt.title(title, fontsize=16)

    plt.xlabel(xlabel, fontsize=14)

    plt.ylabel("Number of Images", fontsize=14)

    plt.grid(alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / filename,
        dpi=DPI,
    )

    plt.close()


def plot_resolution_histograms(clean):

    save_histogram(
        clean["width"],
        "Image Width Distribution",
        "Width (pixels)",
        "width_histogram.png",
    )

    save_histogram(
        clean["height"],
        "Image Height Distribution",
        "Height (pixels)",
        "height_histogram.png",
    )

    save_histogram(
        clean["megapixels"],
        "Megapixel Distribution",
        "Megapixels",
        "megapixel_histogram.png",
    )

    save_histogram(
        clean["pixels"],
        "Resolution Distribution",
        "Pixels",
        "resolution_histogram.png",
    )


def plot_class_distribution(clean):

    counts = clean["label"].value_counts().sort_index()

    plt.figure(figsize=(6, 5))

    plt.bar(
        ["Real", "Fake"],
        counts.values,
    )

    plt.title(
        "Class Distribution",
        fontsize=16,
    )

    plt.ylabel("Images")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "class_distribution.png",
        dpi=DPI,
    )

    plt.close()


def plot_generator_distribution(clean):

    counts = (
        clean["generator"]
        .value_counts()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 7))

    plt.bar(
        counts.index,
        counts.values,
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.ylabel("Images")

    plt.title(
        "Generator Distribution",
        fontsize=16,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "generator_distribution.png",
        dpi=DPI,
    )

    plt.close()


def plot_generator_resolution(clean):

    df = (
        clean.groupby("generator")["megapixels"]
        .mean()
        .sort_values(ascending=False)
    )

    plt.figure(figsize=(12, 7))

    plt.bar(
        df.index,
        df.values,
    )

    plt.xticks(
        rotation=45,
        ha="right",
    )

    plt.ylabel("Average Megapixels")

    plt.title(
        "Average Resolution by Generator",
        fontsize=16,
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "generator_resolution.png",
        dpi=DPI,
    )

    plt.close()


def generate_all_figures(clean):

    print("\nGenerating figures...")

    plot_resolution_histograms(clean)

    plot_class_distribution(clean)

    plot_generator_distribution(clean)

    plot_generator_resolution(clean)

    print("Figures saved.")

# =============================================================================
# Report
# =============================================================================

def generate_report(clean):

    shorter = np.minimum(clean["width"], clean["height"])

    report = []

    report.append("=" * 70)
    report.append("TIGAS DATASET ANALYSIS REPORT")
    report.append("=" * 70)

    report.append("")
    report.append(f"Total Images           : {len(clean):,}")
    report.append(f"Train Images           : {(clean['split']=='train').sum():,}")
    report.append(f"Validation Images      : {(clean['split']=='val').sum():,}")
    report.append(f"Test Images            : {(clean['split']=='test').sum():,}")

    report.append("")
    report.append(f"Real Images            : {(clean['label']==0).sum():,}")
    report.append(f"Fake Images            : {(clean['label']==1).sum():,}")

    report.append("")
    report.append(f"Generators             : {clean['generator'].nunique()}")

    report.append("")
    report.append("Resolution Statistics")
    report.append("-------------------------------")

    report.append(f"Minimum Width          : {clean['width'].min():,.0f}")
    report.append(f"Maximum Width          : {clean['width'].max():,.0f}")

    report.append(f"Minimum Height         : {clean['height'].min():,.0f}")
    report.append(f"Maximum Height         : {clean['height'].max():,.0f}")

    report.append(f"Average Width          : {clean['width'].mean():.2f}")
    report.append(f"Average Height         : {clean['height'].mean():.2f}")

    report.append("")
    report.append(f"Average Megapixels     : {clean['megapixels'].mean():.2f}")
    report.append(f"Median Megapixels      : {clean['megapixels'].median():.2f}")

    report.append("")
    report.append("High Resolution Analysis")
    report.append("-------------------------------")

    report.append(
        f"720p or higher         : {(shorter>=720).sum():,} ({(shorter>=720).mean()*100:.2f}%)"
    )

    report.append(
        f"1080p or higher        : {(shorter>=1080).sum():,} ({(shorter>=1080).mean()*100:.2f}%)"
    )

    report.append(
        f"1440p or higher        : {(shorter>=1440).sum():,} ({(shorter>=1440).mean()*100:.2f}%)"
    )

    report.append(
        f"4K or higher           : {(shorter>=2160).sum():,} ({(shorter>=2160).mean()*100:.2f}%)"
    )

    report.append("")
    report.append(
        f"2 MP or higher         : {(clean['megapixels']>=2).sum():,} ({(clean['megapixels']>=2).mean()*100:.2f}%)"
    )

    report.append(
        f"8 MP or higher         : {(clean['megapixels']>=8).sum():,} ({(clean['megapixels']>=8).mean()*100:.2f}%)"
    )

    report_text = "\n".join(report)

    print()
    print(report_text)

    with open(
        OUTPUT_DIR / "dataset_report.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(report_text)


# =============================================================================
# Main
# =============================================================================

def main():

    print("=" * 70)
    print("TIGAS DATASET EDA")
    print("=" * 70)

    df = scan_dataset()

    clean = compute_summary(df)

    compute_resolution_statistics(clean)

    generator_statistics(clean)

    class_statistics(clean)

    file_format_statistics(clean)

    generate_all_figures(clean)

    generate_report(clean)

    print()
    print("=" * 70)
    print("EDA COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print()
    print(f"Results saved to:\n{OUTPUT_DIR}")


if __name__ == "__main__":
    main()