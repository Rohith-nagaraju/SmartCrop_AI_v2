from pathlib import Path
import json
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)


# ============================================================
# SmartCrop AI - V4 Disease Model Evaluation
#
# Evaluates:
#   models/disease_model_v4.keras
#
# Against:
#   data/disease_dataset_v4/test
#
# Final model:
#   21 classes
#
# IMPORTANT:
#   V2 is NOT modified.
# ============================================================


ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    ROOT
    / "models"
    / "disease_model_v4.keras"
)

CLASS_NAMES_PATH = (
    ROOT
    / "models"
    / "class_names_v4.json"
)

TEST_DIR = (
    ROOT
    / "data"
    / "disease_dataset_v4"
    / "test"
)

RESULTS_DIR = (
    ROOT
    / "experiments"
    / "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

CLASS_RESULTS_PATH = (
    RESULTS_DIR
    / "disease_v4_class_performance.csv"
)

CONFUSION_MATRIX_PATH = (
    RESULTS_DIR
    / "disease_v4_confusion_matrix.csv"
)

SUMMARY_PATH = (
    RESULTS_DIR
    / "disease_v4_summary.json"
)

CROP_RESULTS_PATH = (
    RESULTS_DIR
    / "disease_v4_crop_performance.csv"
)


IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# ============================================================
# Verify files
# ============================================================

print()
print("=" * 70)
print("SMARTCROP AI - V4 OFFICIAL TEST EVALUATION")
print("=" * 70)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"V4 model not found:\n{MODEL_PATH}"
    )


if not CLASS_NAMES_PATH.exists():

    raise FileNotFoundError(
        f"Class names not found:\n{CLASS_NAMES_PATH}"
    )


if not TEST_DIR.exists():

    raise FileNotFoundError(
        f"Test directory not found:\n{TEST_DIR}"
    )


# ============================================================
# Load class names
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "r",
    encoding="utf-8-sig",
) as f:

    class_names = json.load(f)


num_classes = len(class_names)


print()
print(f"Model: {MODEL_PATH}")
print(f"Test directory: {TEST_DIR}")
print(f"Number of classes: {num_classes}")


if num_classes != 21:

    raise ValueError(
        f"Expected 21 classes, found {num_classes}"
    )


# ============================================================
# Load test dataset
# ============================================================

print()
print("=" * 70)
print("LOADING TEST DATASET")
print("=" * 70)


test_ds = tf.keras.utils.image_dataset_from_directory(
    str(TEST_DIR),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


if test_ds.class_names != class_names:

    print()
    print("Model class order:")
    print(class_names)

    print()
    print("Dataset class order:")
    print(test_ds.class_names)

    raise ValueError(
        "Class ordering mismatch between "
        "class_names_v4.json and test dataset."
    )


# ============================================================
# Count test images
# ============================================================

test_counts = {}

for class_name in class_names:

    class_dir = TEST_DIR / class_name

    test_counts[class_name] = len(
        [
            p
            for p in class_dir.iterdir()
            if p.is_file()
        ]
    )


total_test_images = sum(
    test_counts.values()
)


print()
print(
    f"Total test images: "
    f"{total_test_images}"
)


for class_name in class_names:

    print(
        f"{class_name:<40}"
        f"{test_counts[class_name]:>6}"
    )


# ============================================================
# Load model
# ============================================================

print()
print("=" * 70)
print("LOADING V4 MODEL")
print("=" * 70)


model = tf.keras.models.load_model(
    str(MODEL_PATH)
)


print(
    f"Parameters: "
    f"{model.count_params():,}"
)


# ============================================================
# Prepare dataset
# ============================================================

test_ds = test_ds.prefetch(
    tf.data.AUTOTUNE
)


# ============================================================
# Ground truth
# ============================================================

print()
print("=" * 70)
print("COLLECTING GROUND TRUTH LABELS")
print("=" * 70)


y_true = []

for _, labels in test_ds:

    y_true.extend(
        labels.numpy().tolist()
    )


y_true = np.asarray(y_true)


# ============================================================
# Prediction
# ============================================================

print()
print("=" * 70)
print("RUNNING INFERENCE")
print("=" * 70)


start_time = time.time()


probabilities = model.predict(
    test_ds,
    verbose=1,
)


inference_time = (
    time.time()
    - start_time
)


y_pred = np.argmax(
    probabilities,
    axis=1,
)


# ============================================================
# Basic validation
# ============================================================

if len(y_true) != len(y_pred):

    raise ValueError(
        "Ground truth and prediction lengths differ."
    )


# ============================================================
# Overall metrics
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred,
)

macro_precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

macro_recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

macro_f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0,
)

weighted_precision = precision_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

weighted_recall = recall_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)


latency_ms = (
    inference_time
    / len(y_true)
) * 1000


# ============================================================
# Print overall results
# ============================================================

print()
print("=" * 70)
print("OVERALL V4 TEST RESULTS")
print("=" * 70)

print()
print(
    f"Test images:          {len(y_true)}"
)

print(
    f"Accuracy:             {accuracy * 100:.2f}%"
)

print(
    f"Macro Precision:      {macro_precision * 100:.2f}%"
)

print(
    f"Macro Recall:         {macro_recall * 100:.2f}%"
)

print(
    f"Macro F1:             {macro_f1 * 100:.2f}%"
)

print(
    f"Weighted Precision:   {weighted_precision * 100:.2f}%"
)

print(
    f"Weighted Recall:      {weighted_recall * 100:.2f}%"
)

print(
    f"Weighted F1:          {weighted_f1 * 100:.2f}%"
)

print(
    f"Total inference:      {inference_time:.2f} sec"
)

print(
    f"Latency/image:        {latency_ms:.2f} ms"
)


# ============================================================
# Classification report
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)


# ============================================================
# Per-class results
# ============================================================

class_rows = []


for index, class_name in enumerate(
    class_names
):

    metrics = report[class_name]

    class_rows.append(
        {
            "class": class_name,

            "support": int(
                metrics["support"]
            ),

            "precision": round(
                metrics["precision"],
                6,
            ),

            "recall": round(
                metrics["recall"],
                6,
            ),

            "f1_score": round(
                metrics["f1-score"],
                6,
            ),
        }
    )


class_df = pd.DataFrame(
    class_rows
)


class_df.to_csv(
    CLASS_RESULTS_PATH,
    index=False,
)


# ============================================================
# Print per-class metrics
# ============================================================

print()
print("=" * 70)
print("PER-CLASS PERFORMANCE")
print("=" * 70)

print()

for row in class_rows:

    print(
        f"{row['class']:<40}"
        f"P={row['precision'] * 100:6.2f}% "
        f"R={row['recall'] * 100:6.2f}% "
        f"F1={row['f1_score'] * 100:6.2f}% "
        f"N={row['support']}"
    )


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
)


cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names,
)


cm_df.to_csv(
    CONFUSION_MATRIX_PATH
)


# ============================================================
# Crop mapping
# ============================================================

def get_crop(class_name):

    if class_name.startswith("Grape "):
        return "Grape"

    if class_name.startswith("Pepper "):
        return "Pepper"

    if class_name.startswith("Potato "):
        return "Potato"

    if class_name.startswith("Maize "):
        return "Maize"

    if class_name.startswith("Rice "):
        return "Rice"

    if class_name.startswith("Tomato "):
        return "Tomato"

    if class_name == "Healthy":
        return "Healthy"

    return "Other"


# ============================================================
# Crop-wise disease performance
#
# Healthy is excluded from crop-specific disease accuracy
# because it is a global class shared across crops.
# ============================================================

crop_indices = {
    "Grape": [],
    "Pepper": [],
    "Potato": [],
    "Maize": [],
    "Rice": [],
    "Tomato": [],
}


for index, class_name in enumerate(
    class_names
):

    crop = get_crop(class_name)

    if crop in crop_indices:

        crop_indices[crop].append(index)


crop_rows = []


for crop, indices in crop_indices.items():

    mask = np.isin(
        y_true,
        indices,
    )

    crop_true = y_true[mask]
    crop_pred = y_pred[mask]

    if len(crop_true) == 0:
        continue

    # Local labels for this crop
    mapping = {
        global_index: local_index
        for local_index, global_index
        in enumerate(indices)
    }

    crop_true_local = np.asarray(
        [
            mapping[x]
            for x in crop_true
        ]
    )

    # Predictions outside the crop are treated
    # as incorrect predictions.
    crop_pred_local = np.asarray(
        [
            mapping.get(
                x,
                -1,
            )
            for x in crop_pred
        ]
    )

    crop_accuracy = np.mean(
        crop_true_local
        == crop_pred_local
    )

    crop_macro_precision = precision_score(
        crop_true_local,
        crop_pred_local,
        labels=list(range(len(indices))),
        average="macro",
        zero_division=0,
    )

    crop_macro_recall = recall_score(
        crop_true_local,
        crop_pred_local,
        labels=list(range(len(indices))),
        average="macro",
        zero_division=0,
    )

    crop_macro_f1 = f1_score(
        crop_true_local,
        crop_pred_local,
        labels=list(range(len(indices))),
        average="macro",
        zero_division=0,
    )

    crop_rows.append(
        {
            "crop": crop,

            "test_images": int(
                len(crop_true)
            ),

            "accuracy": round(
                crop_accuracy,
                6,
            ),

            "macro_precision": round(
                crop_macro_precision,
                6,
            ),

            "macro_recall": round(
                crop_macro_recall,
                6,
            ),

            "macro_f1": round(
                crop_macro_f1,
                6,
            ),
        }
    )


crop_df = pd.DataFrame(
    crop_rows
)


crop_df.to_csv(
    CROP_RESULTS_PATH,
    index=False,
)


# ============================================================
# Print crop results
# ============================================================

print()
print("=" * 70)
print("CROP-WISE DISEASE PERFORMANCE")
print("=" * 70)

print()

for row in crop_rows:

    print(
        f"{row['crop']:<12}"
        f"N={row['test_images']:>4} "
        f"Accuracy={row['accuracy'] * 100:6.2f}% "
        f"Macro F1={row['macro_f1'] * 100:6.2f}%"
    )


# ============================================================
# Major confusions
# ============================================================

print()
print("=" * 70)
print("MAJOR CONFUSIONS")
print("=" * 70)


confusions = []


for actual_index in range(
    num_classes
):

    for predicted_index in range(
        num_classes
    ):

        count = cm[
            actual_index,
            predicted_index,
        ]

        if (
            actual_index != predicted_index
            and count > 0
        ):

            confusions.append(
                {
                    "actual":
                        class_names[
                            actual_index
                        ],

                    "predicted":
                        class_names[
                            predicted_index
                        ],

                    "count":
                        int(count),
                }
            )


confusions = sorted(
    confusions,
    key=lambda x: x["count"],
    reverse=True,
)


for item in confusions[:20]:

    print(
        f"{item['actual']:<40}"
        f" -> "
        f"{item['predicted']:<40}"
        f"{item['count']:>4}"
    )


# ============================================================
# Summary JSON
# ============================================================

summary = {

    "model": "EfficientNetB0",

    "version": "V4",

    "num_classes": num_classes,

    "test_images": int(
        len(y_true)
    ),

    "accuracy": float(
        accuracy
    ),

    "macro_precision": float(
        macro_precision
    ),

    "macro_recall": float(
        macro_recall
    ),

    "macro_f1": float(
        macro_f1
    ),

    "weighted_precision": float(
        weighted_precision
    ),

    "weighted_recall": float(
        weighted_recall
    ),

    "weighted_f1": float(
        weighted_f1
    ),

    "inference_seconds": float(
        inference_time
    ),

    "latency_ms_per_image": float(
        latency_ms
    ),

    "model_parameters": int(
        model.count_params()
    ),

    "class_names": class_names,

    "test_class_counts": test_counts,

    "model_path": str(
        MODEL_PATH
    ),

    "test_path": str(
        TEST_DIR
    ),

}


with open(
    SUMMARY_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        summary,
        f,
        indent=2,
    )


# ============================================================
# Final
# ============================================================

print()
print("=" * 70)
print("V4 EVALUATION COMPLETE")
print("=" * 70)

print()
print("Saved results:")

print(
    CLASS_RESULTS_PATH
)

print(
    CONFUSION_MATRIX_PATH
)

print(
    CROP_RESULTS_PATH
)

print(
    SUMMARY_PATH
)

print()
print("V2 remains untouched.")

print("=" * 70)