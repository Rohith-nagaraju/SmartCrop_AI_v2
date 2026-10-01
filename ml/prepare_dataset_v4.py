from pathlib import Path
import shutil
import random
from PIL import Image

# ============================================================
# SmartCrop AI - V4 Dataset Preparation
#
# Existing V2:
#   15 classes
#
# New crops:
#   Potato
#   Grape
#   Pepper
#
# Final V4:
#   21 classes
#
# IMPORTANT:
#   - V2 is NEVER modified.
#   - V4 is created separately.
#   - Healthy classes from new crops are merged
#     into the global Healthy class.
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

V2_DIR = ROOT / "data" / "disease_dataset_v2"

RAW_DIR = (
    ROOT
    / "data"
    / "plantvillage_raw"
    / "PlantVillage"
)

V4_DIR = ROOT / "data" / "disease_dataset_v4"

SEED = 42

random.seed(SEED)


# ============================================================
# Class mapping
# ============================================================

NEW_CLASS_MAPPING = {
    "Grape___Black_rot": "Grape Black Rot",

    "Grape___Esca_(Black_Measles)": "Grape Esca Black Measles",

    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)":
        "Grape Leaf Blight",

    "Pepper,_bell___Bacterial_spot":
        "Pepper Bacterial Spot",

    "Potato___Early_blight":
        "Potato Early Blight",

    "Potato___Late_blight":
        "Potato Late Blight",
}


NEW_HEALTHY_CLASSES = {
    "Grape___healthy",
    "Pepper,_bell___healthy",
    "Potato___healthy",
}


# ============================================================
# Image extensions
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ============================================================
# Utility functions
# ============================================================

def is_image(path: Path):
    return (
        path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def get_images(directory: Path):
    return sorted(
        [
            p
            for p in directory.iterdir()
            if is_image(p)
        ]
    )


def check_image(path: Path):
    """
    Verify that an image can actually be opened.
    """
    try:
        with Image.open(path) as img:
            img.verify()

        return True

    except Exception:
        return False


def safe_copy(
    source: Path,
    destination: Path,
):
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    shutil.copy2(
        source,
        destination,
    )


def unique_destination(
    destination_dir: Path,
    source: Path,
    prefix: str = "",
):
    """
    Generate a destination filename that cannot collide
    with another dataset class.
    """

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    filename = f"{prefix}{source.name}"

    destination = destination_dir / filename

    if not destination.exists():
        return destination

    stem = source.stem
    suffix = source.suffix

    counter = 1

    while True:

        filename = (
            f"{prefix}{stem}_{counter}{suffix}"
        )

        destination = (
            destination_dir / filename
        )

        if not destination.exists():
            return destination

        counter += 1


# ============================================================
# Clean previous V4 dataset
# ============================================================

if V4_DIR.exists():

    print("=" * 70)
    print("Existing V4 dataset found.")
    print("Removing old V4 dataset...")
    print("=" * 70)

    shutil.rmtree(V4_DIR)


for split in ["train", "val", "test"]:

    (V4_DIR / split).mkdir(
        parents=True,
        exist_ok=True,
    )


# ============================================================
# Read existing V2 classes
# ============================================================

print()
print("=" * 70)
print("STEP 1 - Copying existing V2 dataset")
print("=" * 70)


v2_train = V2_DIR / "train"
v2_val = V2_DIR / "val"
v2_test = V2_DIR / "test"


if not V2_DIR.exists():

    raise FileNotFoundError(
        f"V2 dataset not found:\n{V2_DIR}"
    )


v2_classes = sorted(
    [
        p.name
        for p in v2_train.iterdir()
        if p.is_dir()
    ]
)


print()
print("Existing V2 classes:")

for cls in v2_classes:
    print(f"  - {cls}")


# ============================================================
# Copy V2
# ============================================================

for split in ["train", "val", "test"]:

    source_split = V2_DIR / split
    destination_split = V4_DIR / split

    for class_dir in source_split.iterdir():

        if not class_dir.is_dir():
            continue

        destination_class = (
            destination_split / class_dir.name
        )

        destination_class.mkdir(
            parents=True,
            exist_ok=True,
        )

        images = get_images(class_dir)

        for image in images:

            destination = unique_destination(
                destination_class,
                image,
                prefix="v2_",
            )

            safe_copy(
                image,
                destination,
            )


print()
print("V2 dataset copied successfully.")


# ============================================================
# Validate new PlantVillage source
# ============================================================

print()
print("=" * 70)
print("STEP 2 - Validating PlantVillage source")
print("=" * 70)


if not RAW_DIR.exists():

    raise FileNotFoundError(
        f"PlantVillage directory not found:\n{RAW_DIR}"
    )


for split in ["train", "val"]:

    split_dir = RAW_DIR / split

    if not split_dir.exists():

        raise FileNotFoundError(
            f"PlantVillage {split} directory not found:\n"
            f"{split_dir}"
        )


# ============================================================
# Check and copy new classes
# ============================================================

print()
print("=" * 70)
print("STEP 3 - Adding Potato, Grape and Pepper")
print("=" * 70)


corrupted_images = []

new_class_counts = {
    "train": {},
    "val": {},
}


# ------------------------------------------------------------
# Disease classes
# ------------------------------------------------------------

for source_class, target_class in NEW_CLASS_MAPPING.items():

    print()
    print(f"Processing: {source_class}")
    print(f"Target:    {target_class}")

    for split in ["train", "val"]:

        source_dir = (
            RAW_DIR
            / split
            / source_class
        )

        if not source_dir.exists():

            raise FileNotFoundError(
                f"Missing source class:\n{source_dir}"
            )

        destination_dir = (
            V4_DIR
            / split
            / target_class
        )

        destination_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        images = get_images(source_dir)

        valid_count = 0

        for image in images:

            if not check_image(image):

                corrupted_images.append(
                    str(image)
                )

                continue

            destination = unique_destination(
                destination_dir,
                image,
                prefix=f"{source_class}_",
            )

            safe_copy(
                image,
                destination,
            )

            valid_count += 1

        new_class_counts[split][
            target_class
        ] = valid_count

        print(
            f"  {split}: "
            f"{valid_count} valid images"
        )


# ============================================================
# Merge healthy images
# ============================================================

print()
print("=" * 70)
print("STEP 4 - Merging new healthy images")
print("Target class: Healthy")
print("=" * 70)


for source_class in NEW_HEALTHY_CLASSES:

    print()
    print(f"Processing healthy class: {source_class}")

    for split in ["train", "val"]:

        source_dir = (
            RAW_DIR
            / split
            / source_class
        )

        if not source_dir.exists():

            raise FileNotFoundError(
                f"Missing healthy source class:\n"
                f"{source_dir}"
            )

        destination_dir = (
            V4_DIR
            / split
            / "Healthy"
        )

        destination_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        images = get_images(source_dir)

        valid_count = 0

        for image in images:

            if not check_image(image):

                corrupted_images.append(
                    str(image)
                )

                continue

            destination = unique_destination(
                destination_dir,
                image,
                prefix=f"{source_class}_",
            )

            safe_copy(
                image,
                destination,
            )

            valid_count += 1

        print(
            f"  {split}: "
            f"{valid_count} healthy images added"
        )


# ============================================================
# Create test split for NEW classes
# ============================================================

print()
print("=" * 70)
print("STEP 5 - Creating held-out test data")
print("=" * 70)

print(
    "Existing V2 test images are preserved."
)

print(
    "New PlantVillage data already has train/val."
)

print(
    "We will create a new test subset from the "
    "PlantVillage TRAIN portion."
)


# ============================================================
# Move a controlled portion of NEW training images
# to the V4 test set.
#
# IMPORTANT:
# Existing V2 test set remains untouched.
#
# We use approximately 10% of NEW PlantVillage training
# data for testing.
# ============================================================

TEST_FRACTION = 0.10


new_test_counts = {}


# Disease classes + healthy
classes_for_new_test = list(
    NEW_CLASS_MAPPING.values()
) + ["Healthy"]


for target_class in classes_for_new_test:

    source_dir = (
        V4_DIR
        / "train"
        / target_class
    )

    if not source_dir.exists():
        continue

    images = get_images(source_dir)

    # Only test NEW images.
    # V2 images have the v2_ prefix.
    new_images = [
        p
        for p in images
        if not p.name.startswith("v2_")
    ]

    if not new_images:
        continue

    random.shuffle(new_images)

    test_count = max(
        1,
        int(len(new_images) * TEST_FRACTION),
    )

    selected = new_images[:test_count]

    destination_dir = (
        V4_DIR
        / "test"
        / target_class
    )

    destination_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for image in selected:

        destination = unique_destination(
            destination_dir,
            image,
            prefix="pv_",
        )

        shutil.move(
            str(image),
            str(destination),
        )

    new_test_counts[target_class] = test_count

    print(
        f"{target_class}: "
        f"{test_count} new test images"
    )


# ============================================================
# Final validation
# ============================================================

print()
print("=" * 70)
print("STEP 6 - Final dataset verification")
print("=" * 70)


expected_classes = sorted(
    set(v2_classes)
    | set(NEW_CLASS_MAPPING.values())
)


# Healthy must already exist in V2.
if "Healthy" not in expected_classes:

    expected_classes.append("Healthy")


expected_classes = sorted(
    set(expected_classes)
)


print()
print(
    f"Expected total classes: "
    f"{len(expected_classes)}"
)

print(
    "Expected: 21 classes"
)


# ------------------------------------------------------------
# Count images
# ------------------------------------------------------------

for split in ["train", "val", "test"]:

    print()
    print("-" * 70)
    print(f"{split.upper()} COUNTS")
    print("-" * 70)

    split_dir = V4_DIR / split

    total = 0

    actual_classes = []

    for class_dir in sorted(
        split_dir.iterdir()
    ):

        if not class_dir.is_dir():
            continue

        count = len(
            get_images(class_dir)
        )

        actual_classes.append(
            class_dir.name
        )

        total += count

        print(
            f"{class_dir.name:<40} "
            f"{count:>6}"
        )

    print("-" * 70)

    print(
        f"{'TOTAL':<40} "
        f"{total:>6}"
    )


# ============================================================
# Class verification
# ============================================================

actual_classes = sorted(
    [
        p.name
        for p in (V4_DIR / "train").iterdir()
        if p.is_dir()
    ]
)


missing_classes = sorted(
    set(expected_classes)
    - set(actual_classes)
)


extra_classes = sorted(
    set(actual_classes)
    - set(expected_classes)
)


print()
print("=" * 70)
print("CLASS VERIFICATION")
print("=" * 70)


if missing_classes:

    print()
    print("MISSING CLASSES:")

    for cls in missing_classes:
        print(f"  - {cls}")

else:

    print("No missing classes.")


if extra_classes:

    print()
    print("UNEXPECTED CLASSES:")

    for cls in extra_classes:
        print(f"  - {cls}")

else:

    print("No unexpected classes.")


# ============================================================
# Corruption report
# ============================================================

print()
print("=" * 70)
print("IMAGE INTEGRITY")
print("=" * 70)


if corrupted_images:

    print(
        f"Corrupted/unreadable images: "
        f"{len(corrupted_images)}"
    )

    for path in corrupted_images[:20]:
        print(f"  {path}")

    if len(corrupted_images) > 20:

        print(
            f"  ... and "
            f"{len(corrupted_images) - 20} more"
        )

else:

    print(
        "No corrupted images found in the "
        "new PlantVillage classes."
    )


# ============================================================
# Final summary
# ============================================================

print()
print("=" * 70)
print("V4 DATASET PREPARATION COMPLETE")
print("=" * 70)

print()
print(f"Dataset location:")
print(V4_DIR)

print()
print(f"Total classes: {len(actual_classes)}")

print()
print("Classes:")

for cls in actual_classes:
    print(f"  - {cls}")


print()
print("New test images created:")

for cls, count in sorted(
    new_test_counts.items()
):

    print(
        f"  {cls}: {count}"
    )


print()
print(
    "V2 dataset was NOT modified."
)

print(
    "V2 model was NOT modified."
)

print()
print("Next step:")
print(
    "Review the counts above before training V4."
)

print("=" * 70)