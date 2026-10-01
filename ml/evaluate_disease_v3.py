import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

# V3 MODEL
MODEL_PATH = ROOT / "models" / "disease_model_v3.keras"
CLASS_NAMES_PATH = ROOT / "models" / "class_names_v3.json"

# SAME HELD-OUT TEST SET USED FOR V2
TEST_DIR = ROOT / "data" / "disease_dataset_v2" / "test"

# RESULTS DIRECTORY
RESULTS_DIR = ROOT / "experiments" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32


# ============================================================
# CHECK FILES
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nDisease V3 model not found:\n{MODEL_PATH}"
    )

if not CLASS_NAMES_PATH.exists():
    raise FileNotFoundError(
        f"\nV3 class names file not found:\n{CLASS_NAMES_PATH}"
    )

if not TEST_DIR.exists():
    raise FileNotFoundError(
        f"\nTest dataset not found:\n{TEST_DIR}"
    )


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r", encoding="utf-8-sig") as f:
    class_names = json.load(f)


print("\n" + "=" * 75)
print("15-CLASS DISEASE MODEL V3 EVALUATION")
print("=" * 75)

print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print(f"\nNumber of classes: {len(class_names)}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading V3 model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("V3 model loaded successfully.")
print(f"Model path: {MODEL_PATH}")


# ============================================================
# LOAD TEST DATASET
# ============================================================

print("\nLoading test dataset...")

test_ds = tf.keras.utils.image_dataset_from_directory(
    TEST_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


# ============================================================
# COLLECT TRUE LABELS
# ============================================================

y_true = []

for _, labels in test_ds:
    y_true.extend(labels.numpy().tolist())

y_true = np.array(y_true)


# ============================================================
# PREDICTIONS
# ============================================================

print("\nRunning V3 predictions...")

start_time = time.perf_counter()

predictions = model.predict(
    test_ds,
    verbose=1
)

end_time = time.perf_counter()

y_prob = predictions
y_pred = np.argmax(y_prob, axis=1)

total_images = len(y_true)

total_time = end_time - start_time

latency_ms = (
    total_time / total_images
) * 1000 if total_images > 0 else 0


# ============================================================
# OVERALL METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
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

weighted_f1 = f1_score(
    y_true,
    y_pred,
    average="weighted",
    zero_division=0,
)


# ============================================================
# PRINT OVERALL RESULTS
# ============================================================

print("\n" + "=" * 75)
print("OVERALL 15-CLASS V3 RESULTS")
print("=" * 75)

print(f"Test images       : {total_images}")
print(f"Accuracy          : {accuracy * 100:.2f}%")
print(f"Macro Precision   : {macro_precision * 100:.2f}%")
print(f"Macro Recall      : {macro_recall * 100:.2f}%")
print(f"Macro F1          : {macro_f1 * 100:.2f}%")
print(f"Weighted F1       : {weighted_f1 * 100:.2f}%")
print(f"Total inference   : {total_time:.2f} seconds")
print(f"Latency/image     : {latency_ms:.2f} ms")


# ============================================================
# FULL CLASSIFICATION REPORT
# ============================================================

report_dict = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    output_dict=True,
    zero_division=0,
)

report_text = classification_report(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
    target_names=class_names,
    zero_division=0,
)

print("\n" + "=" * 75)
print("FULL V3 CLASSIFICATION REPORT")
print("=" * 75)

print(report_text)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = (
    RESULTS_DIR /
    "disease_v3_classification_report.json"
)

with open(report_path, "w", encoding="utf-8") as f:
    json.dump(
        report_dict,
        f,
        indent=4
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=list(range(len(class_names))),
)


# ============================================================
# SAVE CONFUSION MATRIX CSV
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=class_names,
    columns=class_names,
)

cm_csv_path = (
    RESULTS_DIR /
    "disease_v3_confusion_matrix.csv"
)

cm_df.to_csv(cm_csv_path)


# ============================================================
# CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(figsize=(16, 14))

plt.imshow(cm)

plt.title(
    "Disease Model V3 - 15-Class Confusion Matrix"
)

plt.xlabel("Predicted Class")
plt.ylabel("True Class")

plt.xticks(
    range(len(class_names)),
    class_names,
    rotation=90,
)

plt.yticks(
    range(len(class_names)),
    class_names,
)

plt.colorbar()

for i in range(len(class_names)):
    for j in range(len(class_names)):
        plt.text(
            j,
            i,
            str(cm[i, j]),
            ha="center",
            va="center",
        )

plt.tight_layout()

cm_png_path = (
    RESULTS_DIR /
    "disease_v3_confusion_matrix.png"
)

plt.savefig(
    cm_png_path,
    dpi=200,
    bbox_inches="tight",
)

plt.close()


# ============================================================
# DISEASE-ONLY CROP GROUPS
# ============================================================

# IMPORTANT:
# Healthy is intentionally excluded here.
#
# Maize = 3 diseases
# Rice = 2 diseases
# Tomato = 9 diseases
# ============================================================

crop_disease_groups = {

    "Maize": [
        "Maize Blight",
        "Maize Common Rust",
        "Maize Gray Leaf Spot",
    ],

    "Rice": [
        "Rice Bacterial Leaf Blight",
        "Rice Leaf Blast",
    ],

    "Tomato": [
        "Tomato Bacterial Spot",
        "Tomato Early Blight",
        "Tomato Late Blight",
        "Tomato Leaf Mold",
        "Tomato Septoria Leaf Spot",
        "Tomato Spider Mites",
        "Tomato Target Spot",
        "Tomato Tomato Mosaic Virus",
        "Tomato Yellow Leaf Curl Virus",
    ],
}


# ============================================================
# DISEASE-ONLY CROP EVALUATION
# ============================================================

crop_disease_results = {}

print("\n" + "=" * 75)
print("V3 DISEASE-ONLY CROP RESULTS")
print("=" * 75)

for crop_name, crop_class_list in crop_disease_groups.items():

    crop_indices = [
        class_names.index(name)
        for name in crop_class_list
    ]

    # Keep ONLY images whose true label belongs
    # to this crop's disease classes.

    true_mask = np.isin(
        y_true,
        crop_indices
    )

    crop_true = y_true[true_mask]
    crop_pred = y_pred[true_mask]

    if len(crop_true) == 0:
        continue

    crop_accuracy = accuracy_score(
        crop_true,
        crop_pred,
    )

    crop_precision = precision_score(
        crop_true,
        crop_pred,
        labels=crop_indices,
        average="macro",
        zero_division=0,
    )

    crop_recall = recall_score(
        crop_true,
        crop_pred,
        labels=crop_indices,
        average="macro",
        zero_division=0,
    )

    crop_f1 = f1_score(
        crop_true,
        crop_pred,
        labels=crop_indices,
        average="macro",
        zero_division=0,
    )

    crop_report = classification_report(
        crop_true,
        crop_pred,
        labels=crop_indices,
        target_names=crop_class_list,
        output_dict=True,
        zero_division=0,
    )

    crop_disease_results[crop_name] = {

        "test_images": int(len(crop_true)),

        "classes": crop_class_list,

        "accuracy": float(crop_accuracy),

        "macro_precision": float(crop_precision),

        "macro_recall": float(crop_recall),

        "macro_f1": float(crop_f1),

        "classification_report": crop_report,
    }

    print(f"\n{crop_name} — DISEASE ONLY")
    print("-" * 50)

    print(
        f"Test images     : {len(crop_true)}"
    )

    print(
        f"Accuracy        : "
        f"{crop_accuracy * 100:.2f}%"
    )

    print(
        f"Macro Precision : "
        f"{crop_precision * 100:.2f}%"
    )

    print(
        f"Macro Recall    : "
        f"{crop_recall * 100:.2f}%"
    )

    print(
        f"Macro F1        : "
        f"{crop_f1 * 100:.2f}%"
    )


# ============================================================
# TOMATO-ONLY SHORT SUMMARY
# ============================================================

tomato_results = crop_disease_results["Tomato"]

print("\n" + "=" * 75)
print("V3 TOMATO DISEASE-ONLY SUMMARY")
print("=" * 75)

print(
    f"Test images     : "
    f"{tomato_results['test_images']}"
)

print(
    f"Accuracy        : "
    f"{tomato_results['accuracy'] * 100:.2f}%"
)

print(
    f"Macro Precision : "
    f"{tomato_results['macro_precision'] * 100:.2f}%"
)

print(
    f"Macro Recall    : "
    f"{tomato_results['macro_recall'] * 100:.2f}%"
)

print(
    f"Macro F1        : "
    f"{tomato_results['macro_f1'] * 100:.2f}%"
)


# ============================================================
# PER-CLASS SUMMARY
# ============================================================

per_class_results = {}

for class_name in class_names:

    per_class_results[class_name] = {

        "precision": float(
            report_dict[class_name]["precision"]
        ),

        "recall": float(
            report_dict[class_name]["recall"]
        ),

        "f1": float(
            report_dict[class_name]["f1-score"]
        ),

        "support": int(
            report_dict[class_name]["support"]
        ),
    }


# ============================================================
# SAVE METRICS JSON
# ============================================================

metrics = {

    "model": "EfficientNet-B0",

    "version": "V3",

    "num_classes": len(class_names),

    "classes": class_names,

    "test_images": int(total_images),

    "accuracy": float(accuracy),

    "macro_precision": float(macro_precision),

    "macro_recall": float(macro_recall),

    "macro_f1": float(macro_f1),

    "weighted_f1": float(weighted_f1),

    "total_inference_seconds": float(
        total_time
    ),

    "latency_ms_per_image": float(
        latency_ms
    ),

    "crop_disease_only": crop_disease_results,

    "per_class": per_class_results,
}


metrics_path = (
    RESULTS_DIR /
    "disease_v3_metrics_clean.json"
)

with open(metrics_path, "w", encoding="utf-8") as f:
    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# SAVE RAW PREDICTIONS
# ============================================================

prediction_rows = []

for i in range(total_images):

    true_index = int(y_true[i])

    predicted_index = int(y_pred[i])

    prediction_rows.append({

        "image_index": i,

        "true_class": class_names[true_index],

        "predicted_class": class_names[predicted_index],

        "confidence": float(
            np.max(y_prob[i])
        ),

        "correct": bool(
            true_index == predicted_index
        ),
    })


predictions_df = pd.DataFrame(
    prediction_rows
)

predictions_path = (
    RESULTS_DIR /
    "disease_v3_predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 75)
print("FINAL V3 EVALUATION SUMMARY")
print("=" * 75)

print(
    f"Model           : EfficientNet-B0 V3"
)

print(
    f"Classes         : {len(class_names)}"
)

print(
    f"Test images     : {total_images}"
)

print(
    f"Accuracy        : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Macro Precision : "
    f"{macro_precision * 100:.2f}%"
)

print(
    f"Macro Recall    : "
    f"{macro_recall * 100:.2f}%"
)

print(
    f"Macro F1        : "
    f"{macro_f1 * 100:.2f}%"
)

print(
    f"Weighted F1     : "
    f"{weighted_f1 * 100:.2f}%"
)

print(
    f"Latency/image   : "
    f"{latency_ms:.2f} ms"
)

print("\nDisease-only crop results:")

for crop_name, result in crop_disease_results.items():

    print(
        f"  {crop_name}: "
        f"Accuracy={result['accuracy'] * 100:.2f}%, "
        f"Macro F1={result['macro_f1'] * 100:.2f}%"
    )


print("\nSaved V3 files:")

print(
    f"  {report_path}"
)

print(
    f"  {cm_csv_path}"
)

print(
    f"  {cm_png_path}"
)

print(
    f"  {metrics_path}"
)

print(
    f"  {predictions_path}"
)

print("\nV3 evaluation completed successfully.")

print("=" * 75)