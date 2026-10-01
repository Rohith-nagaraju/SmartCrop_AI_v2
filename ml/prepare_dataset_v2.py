from pathlib import Path
import random
import shutil


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

SOURCE_DATASET = ROOT / "data" / "disease_dataset"
TOMATO_DATASET = ROOT / "data" / "tomato_raw" / "tomato"
OUTPUT_DATASET = ROOT / "data" / "disease_dataset_v2"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

SEED = 42
random.seed(SEED)

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# Tomato folder -> final class name
TOMATO_CLASSES = {
    "Tomato___Bacterial_spot": "Tomato Bacterial Spot",
    "Tomato___Early_blight": "Tomato Early Blight",
    "Tomato___Late_blight": "Tomato Late Blight",
    "Tomato___Leaf_Mold": "Tomato Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Tomato Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Tomato Spider Mites",
    "Tomato___Target_Spot": "Tomato Target Spot",
    "Tomato___Tomato_mosaic_virus": "Tomato Tomato Mosaic Virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": (
        "Tomato Yellow Leaf Curl Virus"
    ),
}


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def get_images(folder):
    """Return all supported image files recursively."""
    if not folder.exists():
        return []

    return [
        p
        for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
    ]


def copy_images(images, destination):
    """Copy images into destination."""
    destination.mkdir(parents=True, exist_ok=True)

    for index, source in enumerate(images):
        # Prefix avoids filename collisions between datasets.
        destination_name = (
            f"{source.parent.name}_{index:05d}{source.suffix.lower()}"
        )

        shutil.copy2(
            source,
            destination / destination_name,
        )


def copy_existing_dataset():
    """
    Copy the original 6-class dataset exactly as it is.

    The original dataset is never modified.
    """

    print("\n[1/3] Copying existing dataset...")

    for split in ["train", "val", "test"]:

        source_split = SOURCE_DATASET / split
        output_split = OUTPUT_DATASET / split

        if not source_split.exists():
            raise FileNotFoundError(
                f"Missing source split: {source_split}"
            )

        for class_dir in sorted(
            source_split.iterdir()
        ):

            if not class_dir.is_dir():
                continue

            images = get_images(class_dir)

            destination = (
                output_split / class_dir.name
            )

            print(
                f"  {split:5s} | "
                f"{class_dir.name:35s} | "
                f"{len(images):5d} images"
            )

            copy_images(
                images,
                destination,
            )


def prepare_tomato_dataset():
    """
    Add Tomato classes.

    Tomato source:
        train = 1000/class
        val   = 100/class

    We create:
        train = 900/class
        val   = 100/class
        test  = 100/class

    Tomato healthy images are merged into the
    existing Healthy class.
    """

    print("\n[2/3] Preparing Tomato dataset...")

    tomato_train = TOMATO_DATASET / "train"
    tomato_val = TOMATO_DATASET / "val"

    for source_name, final_class in TOMATO_CLASSES.items():

        source_train = tomato_train / source_name
        source_val = tomato_val / source_name

        if not source_train.exists():
            raise FileNotFoundError(
                f"Missing Tomato training class:\n"
                f"{source_train}"
            )

        if not source_val.exists():
            raise FileNotFoundError(
                f"Missing Tomato validation class:\n"
                f"{source_val}"
            )

        train_images = get_images(source_train)
        val_images = get_images(source_val)

        random.shuffle(train_images)

        if len(train_images) < 1000:
            raise ValueError(
                f"{source_name} has only "
                f"{len(train_images)} training images."
            )

        test_images = train_images[:100]
        train_images = train_images[100:1000]

        print(
            f"  {final_class:35s} | "
            f"train={len(train_images):4d} "
            f"val={len(val_images):4d} "
            f"test={len(test_images):4d}"
        )

        copy_images(
            train_images,
            OUTPUT_DATASET / "train" / final_class,
        )

        copy_images(
            val_images,
            OUTPUT_DATASET / "val" / final_class,
        )

        copy_images(
            test_images,
            OUTPUT_DATASET / "test" / final_class,
        )


def prepare_tomato_healthy():
    """
    Merge Tomato healthy images into the existing
    Healthy class.

    Tomato:
        900 train
        100 val
        100 test
    """

    print("\n  Adding Tomato healthy images to Healthy...")

    source_train = (
        TOMATO_DATASET
        / "train"
        / "Tomato___healthy"
    )

    source_val = (
        TOMATO_DATASET
        / "val"
        / "Tomato___healthy"
    )

    train_images = get_images(source_train)
    val_images = get_images(source_val)

    random.shuffle(train_images)

    test_images = train_images[:100]
    train_images = train_images[100:1000]

    copy_images(
        train_images,
        OUTPUT_DATASET / "train" / "Healthy",
    )

    copy_images(
        val_images,
        OUTPUT_DATASET / "val" / "Healthy",
    )

    copy_images(
        test_images,
        OUTPUT_DATASET / "test" / "Healthy",
    )

    print(
        f"  Healthy additions: "
        f"train={len(train_images)} "
        f"val={len(val_images)} "
        f"test={len(test_images)}"
    )


def print_final_counts():
    """Display final dataset statistics."""

    print("\n[3/3] Final dataset counts")
    print("=" * 70)

    for split in ["train", "val", "test"]:

        split_dir = OUTPUT_DATASET / split

        print(f"\n{split.upper()}")

        total = 0

        for class_dir in sorted(
            split_dir.iterdir()
        ):

            if not class_dir.is_dir():
                continue

            count = len(get_images(class_dir))
            total += count

            print(
                f"  {class_dir.name:35s} {count:5d}"
            )

        print(
            f"  {'TOTAL':35s} {total:5d}"
        )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 70)
    print("SmartCrop AI - Dataset V2 Preparation")
    print("=" * 70)

    if OUTPUT_DATASET.exists():
        raise FileExistsError(
            f"\nOutput dataset already exists:\n"
            f"{OUTPUT_DATASET}\n\n"
            f"Delete it manually if you want to rebuild it."
        )

    OUTPUT_DATASET.mkdir(
        parents=True,
        exist_ok=True,
    )

    copy_existing_dataset()

    # Add the 9 disease classes.
    prepare_tomato_dataset()

    # Add Tomato healthy to existing Healthy.
    prepare_tomato_healthy()

    print_final_counts()

    print("\nDataset preparation completed successfully.")
    print(f"Output: {OUTPUT_DATASET}")
    print(f"Random seed: {SEED}")