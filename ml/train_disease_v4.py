from pathlib import Path
import json
import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# SmartCrop AI - V4 Disease Classification Training
#
# 21 classes:
#   Healthy
#   Maize (3 diseases)
#   Rice (2 diseases)
#   Tomato (9 diseases)
#   Grape (3 diseases)
#   Potato (2 diseases)
#   Pepper (1 disease)
#
# IMPORTANT:
#   - V2 is NOT modified.
#   - V4 uses a separate model file.
# ============================================================


# ============================================================
# Paths
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TRAIN_DIR = ROOT / "data" / "disease_dataset_v4" / "train"
VAL_DIR = ROOT / "data" / "disease_dataset_v4" / "val"

MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "disease_model_v4.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names_v4.json"

EXPERIMENT_DIR = (
    ROOT
    / "experiments"
    / "disease_models"
    / "v4"
)

EXPERIMENT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = (224, 224)

BATCH_SIZE = 32

SEED = 42

STAGE1_EPOCHS = 5

STAGE2_EPOCHS = 7

FINE_TUNE_LAYERS = 40

LEARNING_RATE_STAGE1 = 1e-3

LEARNING_RATE_STAGE2 = 1e-5


# ============================================================
# Reproducibility
# ============================================================

tf.keras.utils.set_random_seed(SEED)

np.random.seed(SEED)


# ============================================================
# Check dataset
# ============================================================

if not TRAIN_DIR.exists():

    raise FileNotFoundError(
        f"Training directory not found:\n{TRAIN_DIR}"
    )


if not VAL_DIR.exists():

    raise FileNotFoundError(
        f"Validation directory not found:\n{VAL_DIR}"
    )


# ============================================================
# Load datasets
# ============================================================

print()
print("=" * 70)
print("SMARTCROP AI - V4 DISEASE MODEL TRAINING")
print("=" * 70)

print()
print("Training directory:")
print(TRAIN_DIR)

print()
print("Validation directory:")
print(VAL_DIR)


train_ds = tf.keras.utils.image_dataset_from_directory(
    str(TRAIN_DIR),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    str(VAL_DIR),
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)


class_names = train_ds.class_names

num_classes = len(class_names)


# ============================================================
# Verify classes
# ============================================================

print()
print("=" * 70)
print("CLASS INFORMATION")
print("=" * 70)

print()
print(f"Number of classes: {num_classes}")

for index, name in enumerate(class_names):

    print(
        f"{index:2d}: {name}"
    )


if num_classes != 21:

    raise ValueError(
        f"Expected 21 classes, "
        f"but found {num_classes}."
    )


if val_ds.class_names != class_names:

    raise ValueError(
        "Training and validation class ordering "
        "does not match."
    )


# ============================================================
# Save class names
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        class_names,
        f,
        indent=2,
    )


print()
print(
    f"Class names saved to:\n"
    f"{CLASS_NAMES_PATH}"
)


# ============================================================
# Count training samples
# ============================================================

print()
print("=" * 70)
print("TRAINING CLASS DISTRIBUTION")
print("=" * 70)


class_counts = {}

for index, class_name in enumerate(class_names):

    class_dir = TRAIN_DIR / class_name

    count = len(
        [
            p
            for p in class_dir.iterdir()
            if p.is_file()
        ]
    )

    class_counts[class_name] = count

    print(
        f"{class_name:<40}"
        f"{count:>6}"
    )


total_training_images = sum(
    class_counts.values()
)

print("-" * 70)

print(
    f"{'TOTAL':<40}"
    f"{total_training_images:>6}"
)


# ============================================================
# Calculate balanced class weights
# ============================================================

print()
print("=" * 70)
print("CLASS WEIGHTS")
print("=" * 70)


labels = []

for class_index, class_name in enumerate(
    class_names
):

    labels.extend(
        [class_index] * class_counts[class_name]
    )


labels = np.asarray(labels)


balanced_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=labels,
)


class_weights = {
    int(index): float(weight)
    for index, weight in enumerate(
        balanced_weights
    )
}


# ------------------------------------------------------------
# Moderate targeted weighting
#
# These classes showed weaker performance in V2:
#   Maize Gray Leaf Spot
#   Rice Bacterial Leaf Blight
#   Rice Leaf Blast
#   Tomato Target Spot
#
# We use moderate multipliers rather than aggressive
# weighting so that overall accuracy is not unnecessarily
# sacrificed.
# ------------------------------------------------------------

TARGETED_BOOSTS = {

    "Maize Gray Leaf Spot": 1.10,

    "Rice Bacterial Leaf Blight": 1.10,

    "Rice Leaf Blast": 1.05,

    "Tomato Target Spot": 1.10,

    "Grape Leaf Blight": 1.05,

    "Potato Early Blight": 1.05,

    "Potato Late Blight": 1.05,

    "Pepper Bacterial Spot": 1.05,
}


for class_name, multiplier in TARGETED_BOOSTS.items():

    if class_name in class_names:

        index = class_names.index(
            class_name
        )

        class_weights[index] *= multiplier


# Normalize maximum weight slightly
# to prevent extreme gradients.

max_weight = max(
    class_weights.values()
)

MAX_WEIGHT = 4.0

if max_weight > MAX_WEIGHT:

    scale = MAX_WEIGHT / max_weight

    for index in class_weights:

        class_weights[index] *= scale


print()

for index, class_name in enumerate(
    class_names
):

    print(
        f"{class_name:<40}"
        f"{class_weights[index]:.4f}"
    )


# ============================================================
# Save class weights
# ============================================================

class_weights_path = (
    EXPERIMENT_DIR
    / "class_weights_v4.json"
)

with open(
    class_weights_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        {
            class_names[index]:
                class_weights[index]
            for index in range(num_classes)
        },
        f,
        indent=2,
    )


# ============================================================
# Performance optimization
# ============================================================

AUTOTUNE = tf.data.AUTOTUNE


train_ds = train_ds.prefetch(
    AUTOTUNE
)

val_ds = val_ds.prefetch(
    AUTOTUNE
)


# ============================================================
# Data augmentation
# ============================================================

data_augmentation = keras.Sequential(
    [

        layers.RandomFlip(
            "horizontal"
        ),

        layers.RandomRotation(
            0.08
        ),

        layers.RandomZoom(
            0.10
        ),

        layers.RandomContrast(
            0.10
        ),

    ],
    name="controlled_augmentation",
)


# ============================================================
# Build EfficientNet-B0
# ============================================================

print()
print("=" * 70)
print("BUILDING EFFICIENTNET-B0")
print("=" * 70)


base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3,
    ),
)


# ------------------------------------------------------------
# Stage 1:
# Freeze the entire ImageNet backbone.
# ------------------------------------------------------------

base_model.trainable = False


inputs = keras.Input(
    shape=(
        IMAGE_SIZE[0],
        IMAGE_SIZE[1],
        3,
    ),
    name="input",
)


x = data_augmentation(
    inputs
)


x = base_model(
    x,
    training=False,
)


x = layers.GlobalAveragePooling2D(
    name="global_average_pooling"
)(x)


x = layers.Dropout(
    0.30,
    name="dropout"
)(x)


outputs = layers.Dense(
    num_classes,
    activation="softmax",
    name="classifier",
)(x)


model = keras.Model(
    inputs,
    outputs,
    name="SmartCrop_EfficientNetB0_V4",
)


# ============================================================
# Stage 1 compilation
# ============================================================

model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=LEARNING_RATE_STAGE1
    ),

    loss=keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy",
    ],
)


print()
print(
    f"Total parameters: "
    f"{model.count_params():,}"
)


# ============================================================
# Callbacks
# ============================================================

best_model_path = (
    EXPERIMENT_DIR
    / "best_v4.keras"
)


callbacks_stage1 = [

    keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=1,
        min_lr=1e-6,
        verbose=1,
    ),

]


# ============================================================
# Stage 1 training
# ============================================================

print()
print("=" * 70)
print("STAGE 1 - FROZEN BACKBONE")
print("=" * 70)

stage1_start = time.time()


history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=STAGE1_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks_stage1,
)


stage1_time = (
    time.time()
    - stage1_start
)


print()
print(
    f"Stage 1 training time: "
    f"{stage1_time:.2f} seconds"
)


# ============================================================
# Stage 2 - Fine tuning
# ============================================================

print()
print("=" * 70)
print("STAGE 2 - FINE TUNING")
print("=" * 70)


base_model.trainable = True


# Freeze all but the final FINE_TUNE_LAYERS
# layers of EfficientNet.

for layer in base_model.layers[
    :-FINE_TUNE_LAYERS
]:

    layer.trainable = False


# Keep BatchNorm layers frozen.
# This improves stability when fine-tuning
# on a moderate-sized dataset.

for layer in base_model.layers:

    if isinstance(
        layer,
        layers.BatchNormalization,
    ):

        layer.trainable = False


trainable_layers = sum(
    1
    for layer in model.layers
    if layer.trainable
)


print()
print(
    f"Trainable model layers: "
    f"{trainable_layers}"
)


model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=LEARNING_RATE_STAGE2
    ),

    loss=keras.losses.SparseCategoricalCrossentropy(),

    metrics=[
        "accuracy",
    ],
)


callbacks_stage2 = [

    keras.callbacks.ModelCheckpoint(
        filepath=str(best_model_path),
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        verbose=1,
    ),

    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        mode="max",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),

    keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),

]


stage2_start = time.time()


history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=STAGE2_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks_stage2,
)


stage2_time = (
    time.time()
    - stage2_start
)


# ============================================================
# Load best model
# ============================================================

print()
print("=" * 70)
print("LOADING BEST V4 MODEL")
print("=" * 70)


if best_model_path.exists():

    model = keras.models.load_model(
        str(best_model_path)
    )

else:

    print(
        "Best checkpoint not found. "
        "Using current model."
    )


# ============================================================
# Save final V4 model
# ============================================================

model.save(
    str(MODEL_PATH)
)


# ============================================================
# Extract training history
# ============================================================

history = {}

for key, values in history1.history.items():

    history.setdefault(
        key,
        [],
    ).extend(
        [
            float(v)
            for v in values
        ]
    )


for key, values in history2.history.items():

    history.setdefault(
        key,
        [],
    ).extend(
        [
            float(v)
            for v in values
        ]
    )


# ============================================================
# Best epoch
# ============================================================

if "val_accuracy" in history:

    best_epoch = (
        int(
            np.argmax(
                history["val_accuracy"]
            )
        )
        + 1
    )

    best_val_accuracy = float(
        max(
            history["val_accuracy"]
        )
    )

else:

    best_epoch = None
    best_val_accuracy = None


# ============================================================
# Save training information
# ============================================================

training_info = {

    "model": "EfficientNetB0",

    "version": "V4",

    "num_classes": num_classes,

    "class_names": class_names,

    "image_size": list(IMAGE_SIZE),

    "batch_size": BATCH_SIZE,

    "seed": SEED,

    "stage1_epochs": STAGE1_EPOCHS,

    "stage2_epochs": STAGE2_EPOCHS,

    "fine_tune_layers": FINE_TUNE_LAYERS,

    "stage1_learning_rate":
        LEARNING_RATE_STAGE1,

    "stage2_learning_rate":
        LEARNING_RATE_STAGE2,

    "stage1_training_seconds":
        stage1_time,

    "stage2_training_seconds":
        stage2_time,

    "total_training_seconds":
        stage1_time + stage2_time,

    "total_parameters":
        int(model.count_params()),

    "best_epoch":
        best_epoch,

    "best_validation_accuracy":
        best_val_accuracy,

    "class_counts":
        class_counts,

    "targeted_class_boosts":
        TARGETED_BOOSTS,

    "model_path":
        str(MODEL_PATH),

}


training_info_path = (
    EXPERIMENT_DIR
    / "training_info_v4.json"
)


with open(
    training_info_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        training_info,
        f,
        indent=2,
    )


# ============================================================
# Save history
# ============================================================

history_path = (
    EXPERIMENT_DIR
    / "training_history_v4.json"
)


with open(
    history_path,
    "w",
    encoding="utf-8",
) as f:

    json.dump(
        history,
        f,
        indent=2,
    )


# ============================================================
# Final output
# ============================================================

print()
print("=" * 70)
print("V4 TRAINING COMPLETE")
print("=" * 70)

print()
print(
    f"Model saved to:"
)

print(
    MODEL_PATH
)

print()
print(
    f"Class names saved to:"
)

print(
    CLASS_NAMES_PATH
)

print()
print(
    f"Training information:"
)

print(
    training_info_path
)

print()
print(
    f"Training history:"
)

print(
    history_path
)

print()
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
    if best_val_accuracy is not None
    else "N/A"
)

print()
print(
    f"Total training time: "
    f"{(stage1_time + stage2_time):.2f} seconds"
)

print()
print(
    "IMPORTANT:"
)

print(
    "V2 model and V2 dataset were NOT modified."
)

print("=" * 70)