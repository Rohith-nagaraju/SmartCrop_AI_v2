"""
Improved EfficientNet-B0 training experiment.

Goal:
- Improve the two weakest classes:
    1. Maize Gray Leaf Spot
    2. Rice Bacterial Leaf Blight
- Preserve the original baseline model.
- Use class-weighted training.
- Use controlled augmentation.
- Save the improved model separately.

Dataset:
data/disease_dataset/{train,val,test}/{class_name}/*.jpg
"""

import json
import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from sklearn.utils.class_weight import compute_class_weight


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ap = argparse.ArgumentParser()
ap.add_argument("--epochs", type=int, default=12)
a = ap.parse_args()

ROOT = Path("data/disease_dataset")
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

train = tf.keras.utils.image_dataset_from_directory(
    ROOT / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=True,
    seed=42,
)

val = tf.keras.utils.image_dataset_from_directory(
    ROOT / "val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode="int",
    shuffle=False,
)

classes = train.class_names
num_classes = len(classes)

print("\nClasses:")
for i, name in enumerate(classes):
    print(i, "->", name)

# ---------------------------------------------------------
# Calculate class weights
# ---------------------------------------------------------

train_counts = []

for class_name in classes:
    class_dir = ROOT / "train" / class_name
    count = len(list(class_dir.glob("*")))
    train_counts.append(count)

print("\nTraining class counts:")
for name, count in zip(classes, train_counts):
    print(f"{name}: {count}")

labels = []

for class_index, count in enumerate(train_counts):
    labels.extend([class_index] * count)

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(num_classes),
    y=np.array(labels),
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(class_weights_array)
}

# ---------------------------------------------------------
# Targeted class weighting
#
# Slightly increase emphasis on the two weakest classes.
# This is intentionally controlled rather than extreme.
# ---------------------------------------------------------

for i, name in enumerate(classes):
    if name == "Maize Gray Leaf Spot":
        class_weights[i] *= 1.20

    if name == "Rice Bacterial Leaf Blight":
        class_weights[i] *= 1.20

print("\nClass weights:")
for i, name in enumerate(classes):
    print(f"{name}: {class_weights[i]:.4f}")

# ---------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------

AUTOTUNE = tf.data.AUTOTUNE

train = train.prefetch(AUTOTUNE)
val = val.prefetch(AUTOTUNE)

# ---------------------------------------------------------
# Controlled augmentation
# ---------------------------------------------------------

augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.12),
        layers.RandomContrast(0.12),
        layers.RandomTranslation(
            height_factor=0.05,
            width_factor=0.05,
        ),
    ],
    name="controlled_augmentation",
)

# ---------------------------------------------------------
# EfficientNet-B0
# ---------------------------------------------------------

base = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(*IMG_SIZE, 3),
)

base.trainable = False

inputs = layers.Input((*IMG_SIZE, 3))

x = augmentation(inputs)

# IMPORTANT:
# EfficientNet-B0 contains its own input rescaling.
# Therefore we intentionally do NOT divide the image by 255.
x = base(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.35)(x)

outputs = layers.Dense(
    num_classes,
    activation="softmax",
)(x)

model = tf.keras.Model(inputs, outputs)

# ---------------------------------------------------------
# Stage 1: frozen backbone
# ---------------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_stage1 = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-6,
    ),
]

print("\n========================================")
print("STAGE 1: Frozen EfficientNet-B0")
print("========================================\n")

model.fit(
    train,
    validation_data=val,
    epochs=a.epochs,
    callbacks=callbacks_stage1,
    class_weight=class_weights,
)

# ---------------------------------------------------------
# Stage 2: fine-tune last 30 layers
# ---------------------------------------------------------

print("\n========================================")
print("STAGE 2: Fine-tuning last 30 layers")
print("========================================\n")

base.trainable = True

for layer in base.layers[:-30]:
    layer.trainable = False

for layer in base.layers[-30:]:
    layer.trainable = True

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=5e-6
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

callbacks_stage2 = [
    tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
    ),
]

model.fit(
    train,
    validation_data=val,
    epochs=6,
    callbacks=callbacks_stage2,
    class_weight=class_weights,
)

# ---------------------------------------------------------
# Save improved model separately
# ---------------------------------------------------------

Path("models").mkdir(exist_ok=True)

output_model = "models/disease_model_improved.keras"
output_classes = "models/class_names_improved.json"
output_weights = "models/class_weights_improved.json"

model.save(output_model)

Path(output_classes).write_text(
    json.dumps(classes, indent=2)
)

Path(output_weights).write_text(
    json.dumps(
        {
            classes[i]: class_weights[i]
            for i in range(num_classes)
        },
        indent=2,
    )
)

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")
print("Saved:", output_model)
print("Saved:", output_classes)
print("Saved:", output_weights)
print("Classes:", classes)