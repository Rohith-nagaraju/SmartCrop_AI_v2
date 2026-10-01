from pathlib import Path
import json
import time

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


# ============================================================
# SmartCrop AI - 15-Class Disease Model Training
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = ROOT / "data" / "disease_dataset_v2"
MODEL_DIR = ROOT / "models"

TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"

MODEL_PATH = MODEL_DIR / "disease_model_v2.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names_v2.json"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42

INITIAL_EPOCHS = 5
FINE_TUNE_EPOCHS = 5

AUTOTUNE = tf.data.AUTOTUNE


# ============================================================
# Reproducibility
# ============================================================

tf.keras.utils.set_random_seed(SEED)


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
# Create model directory
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Load datasets
# ============================================================

print("=" * 70)
print("SmartCrop AI - 15-Class Disease Model Training")
print("=" * 70)

print("\nLoading training dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=SEED,
)

print("\nLoading validation dataset...")

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)


# ============================================================
# Class names
# ============================================================

class_names = train_ds.class_names
num_classes = len(class_names)

print("\nClasses:")

for i, name in enumerate(class_names):
    print(f"  {i:2d} : {name}")

print(f"\nNumber of classes: {num_classes}")


# ============================================================
# Verify class names and order
# ============================================================

expected_classes = [
    "Healthy",
    "Maize Blight",
    "Maize Common Rust",
    "Maize Gray Leaf Spot",
    "Rice Bacterial Leaf Blight",
    "Rice Leaf Blast",
    "Tomato Bacterial Spot",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
    "Tomato Septoria Leaf Spot",
    "Tomato Spider Mites",
    "Tomato Target Spot",
    "Tomato Tomato Mosaic Virus",
    "Tomato Yellow Leaf Curl Virus",
]

if class_names != expected_classes:
    raise ValueError(
        "\nUnexpected class order.\n\n"
        f"Expected:\n{expected_classes}\n\n"
        f"Found:\n{class_names}"
    )


# ============================================================
# Count training images
# ============================================================

class_counts = []

print("\nTraining image counts:")

for class_name in class_names:

    class_dir = TRAIN_DIR / class_name

    extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    count = sum(
        1
        for p in class_dir.rglob("*")
        if p.is_file()
        and p.suffix.lower() in extensions
    )

    class_counts.append(count)

    print(
        f"  {class_name:35s}: {count:5d}"
    )


# ============================================================
# Generate labels for class-weight calculation
# ============================================================

labels = []

for class_index, count in enumerate(class_counts):
    labels.extend(
        [class_index] * count
    )

labels = np.array(labels)

classes = np.arange(num_classes)


# ============================================================
# Calculate balanced class weights
# ============================================================

weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=labels,
)

class_weights = {
    int(class_index): float(weight)
    for class_index, weight in zip(
        classes,
        weights
    )
}


print("\nClass weights:")

for i, name in enumerate(class_names):
    print(
        f"  {name:35s}: "
        f"{class_weights[i]:.4f}"
    )


# ============================================================
# Dataset performance
# ============================================================

train_ds = train_ds.prefetch(
    AUTOTUNE
)

val_ds = val_ds.prefetch(
    AUTOTUNE
)


# ============================================================
# Controlled data augmentation
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(
            "horizontal"
        ),

        tf.keras.layers.RandomRotation(
            0.08
        ),

        tf.keras.layers.RandomZoom(
            0.10
        ),

        tf.keras.layers.RandomContrast(
            0.10
        ),
    ],
    name="controlled_augmentation",
)


# ============================================================
# Build EfficientNet-B0
# ============================================================

print("\nBuilding EfficientNet-B0...")

base_model = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)


# Initially freeze the EfficientNet backbone

base_model.trainable = False


# ============================================================
# Build complete model
# ============================================================

inputs = tf.keras.Input(
    shape=(224, 224, 3),
    name="image_input",
)

x = data_augmentation(inputs)


# IMPORTANT:
#
# EfficientNetB0 in this TensorFlow/Keras version contains
# its own input rescaling.
#
# Therefore DO NOT divide images by 255 here.
#
# image_dataset_from_directory provides pixel values
# in the 0-255 range.

x = base_model(
    x,
    training=False
)

x = tf.keras.layers.GlobalAveragePooling2D()(x)

x = tf.keras.layers.Dropout(
    0.30
)(x)

outputs = tf.keras.layers.Dense(
    num_classes,
    activation="softmax",
    name="disease_output",
)(x)


model = tf.keras.Model(
    inputs=inputs,
    outputs=outputs,
    name="smartcrop_efficientnetb0_v2",
)


# ============================================================
# Display model information
# ============================================================

print("\nModel created successfully.")

print(
    f"Total parameters: "
    f"{model.count_params():,}"
)


# ============================================================
# PHASE 1
# Frozen EfficientNet-B0
# ============================================================

print("\n" + "=" * 70)
print("PHASE 1 - FROZEN EFFICIENTNET-B0")
print("=" * 70)


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ],
)


# ============================================================
# Phase 1 callbacks
# ============================================================

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(MODEL_PATH),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)


early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=2,
    mode="max",
    restore_best_weights=True,
    verbose=1,
)


reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=1,
    min_lr=1e-6,
    verbose=1,
)


# ============================================================
# Train Phase 1
# ============================================================

print("\nStarting Phase 1 training...")

start_time = time.time()


history1 = model.fit(
    train_ds,

    validation_data=val_ds,

    epochs=INITIAL_EPOCHS,

    class_weight=class_weights,

    callbacks=[
        checkpoint,
        early_stopping,
        reduce_lr,
    ],
)


phase1_time = time.time() - start_time


print(
    f"\nPhase 1 training time: "
    f"{phase1_time:.2f} seconds"
)


# ============================================================
# PHASE 2
# Fine-tuning
# ============================================================

print("\n" + "=" * 70)
print("PHASE 2 - FINE-TUNING LAST 30 LAYERS")
print("=" * 70)


# Unfreeze EfficientNet

base_model.trainable = True


# Freeze all but last 30 layers

for layer in base_model.layers[:-30]:
    layer.trainable = False


trainable_count = sum(
    1
    for layer in base_model.layers
    if layer.trainable
)


print(
    f"Trainable EfficientNet layers: "
    f"{trainable_count}"
)


# ============================================================
# Compile for fine-tuning
# ============================================================

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ],
)


# ============================================================
# Fine-tuning callbacks
# ============================================================

checkpoint_finetune = tf.keras.callbacks.ModelCheckpoint(
    filepath=str(MODEL_PATH),
    monitor="val_accuracy",
    save_best_only=True,
    mode="max",
    verbose=1,
)


early_stopping_finetune = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy",
    patience=2,
    mode="max",
    restore_best_weights=True,
    verbose=1,
)


reduce_lr_finetune = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=1,
    min_lr=1e-7,
    verbose=1,
)


# ============================================================
# Train Phase 2
# ============================================================

print("\nStarting Phase 2 fine-tuning...")

start_time = time.time()


history2 = model.fit(
    train_ds,

    validation_data=val_ds,

    epochs=FINE_TUNE_EPOCHS,

    class_weight=class_weights,

    callbacks=[
        checkpoint_finetune,
        early_stopping_finetune,
        reduce_lr_finetune,
    ],
)


phase2_time = time.time() - start_time


print(
    f"\nPhase 2 training time: "
    f"{phase2_time:.2f} seconds"
)


# ============================================================
# Load best saved model
# ============================================================

print("\nLoading best model checkpoint...")

best_model = tf.keras.models.load_model(
    str(MODEL_PATH)
)


# ============================================================
# Save final model
# ============================================================

best_model.save(
    str(MODEL_PATH)
)


# ============================================================
# Save class names
# ============================================================

with open(
    CLASS_NAMES_PATH,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        class_names,
        f,
        indent=2,
        ensure_ascii=False,
    )


# ============================================================
# Final validation evaluation
# ============================================================

print("\n" + "=" * 70)
print("FINAL VALIDATION RESULT")
print("=" * 70)


loss, accuracy = best_model.evaluate(
    val_ds,
    verbose=1,
)


print(
    f"\nValidation Loss     : "
    f"{loss:.4f}"
)

print(
    f"Validation Accuracy : "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# Training time
# ============================================================

total_time = (
    phase1_time
    + phase2_time
)


print(
    f"Total Training Time : "
    f"{total_time:.2f} seconds"
)


# ============================================================
# Saved files
# ============================================================

print("\nSaved model:")
print(
    MODEL_PATH
)

print("\nSaved class names:")
print(
    CLASS_NAMES_PATH
)


# ============================================================
# Final confirmation
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    "\nThe original 6-class model was NOT modified."
)

print(
    "\nNew 15-class model:"
)

print(
    f"  {MODEL_PATH}"
)

print(
    "\nNext step: evaluate this model on the "
    "held-out test dataset."
)