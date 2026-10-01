"""
Train and evaluate image-classification baselines.

Models:
    customcnn
    mobilenetv2
    resnet50

Evaluation:
    Test set only

All models use the same:
    - image size
    - batch size
    - training/validation split
    - number of epochs
    - optimizer
    - loss
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2, ResNet50
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)


# ---------------------------------------------------------
# Arguments
# ---------------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--model",
    required=True,
    choices=["customcnn", "mobilenetv2", "resnet50"],
)

parser.add_argument(
    "--epochs",
    type=int,
    default=5,
)

args = parser.parse_args()


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ROOT = Path("data/disease_dataset")

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
SEED = 42


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

print("\nLoading dataset...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    ROOT / "train",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    ROOT / "val",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    ROOT / "test",
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

class_names = train_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)


# ---------------------------------------------------------
# Data augmentation
# ---------------------------------------------------------

augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ],
    name="augmentation",
)


# ---------------------------------------------------------
# Build model
# ---------------------------------------------------------

print("\nBuilding model:", args.model)


if args.model == "mobilenetv2":

    base = MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )

    base.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))

    x = augmentation(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
    )(x)

    model = Model(inputs, outputs)


elif args.model == "resnet50":

    base = ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
    )

    base.trainable = False

    inputs = layers.Input(shape=(224, 224, 3))

    x = augmentation(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
    )(x)

    model = Model(inputs, outputs)


else:

    # Custom CNN baseline

    inputs = layers.Input(shape=(224, 224, 3))

    x = augmentation(inputs)
    x = layers.Rescaling(1.0 / 255)(x)

    x = layers.Conv2D(
        32,
        3,
        activation="relu",
    )(x)

    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(
        64,
        3,
        activation="relu",
    )(x)

    x = layers.MaxPooling2D()(x)

    x = layers.Conv2D(
        128,
        3,
        activation="relu",
    )(x)

    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
    )(x)

    model = Model(inputs, outputs)


# ---------------------------------------------------------
# Compile
# ---------------------------------------------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


model.summary()


# ---------------------------------------------------------
# Training
# ---------------------------------------------------------

print("\nStarting training...")

start_time = time.perf_counter()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=args.epochs,
    verbose=1,
)

training_seconds = time.perf_counter() - start_time


# ---------------------------------------------------------
# Test evaluation
# ---------------------------------------------------------

print("\nEvaluating on TEST set...")

y_true = np.concatenate(
    [labels.numpy() for _, labels in test_ds]
)

probabilities = model.predict(
    test_ds,
    verbose=1,
)

y_pred = np.argmax(
    probabilities,
    axis=1,
)


# ---------------------------------------------------------
# Metrics
# ---------------------------------------------------------

accuracy = accuracy_score(
    y_true,
    y_pred,
)

precision, recall, macro_f1, _ = (
    precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )
)


# ---------------------------------------------------------
# Inference latency
# ---------------------------------------------------------

sample = next(iter(test_ds))[0][:1]

# Warm-up
model.predict(sample, verbose=0)

start_time = time.perf_counter()

for _ in range(10):
    model.predict(sample, verbose=0)

latency_ms = (
    (time.perf_counter() - start_time)
    / 10
    * 1000
)


# ---------------------------------------------------------
# Results
# ---------------------------------------------------------

parameters = model.count_params()

result = {
    "model": args.model,
    "accuracy": float(accuracy),
    "macro_precision": float(precision),
    "macro_recall": float(recall),
    "macro_f1": float(macro_f1),
    "parameters": int(parameters),
    "train_seconds": float(training_seconds),
    "inference_ms_per_image": float(latency_ms),
}


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

results_dir = Path("experiments/results")
results_dir.mkdir(
    parents=True,
    exist_ok=True,
)

results_file = (
    results_dir / "baselines.jsonl"
)

with results_file.open(
    "a",
    encoding="utf-8",
) as f:
    f.write(
        json.dumps(result) + "\n"
    )


print("\n" + "=" * 60)
print("BASELINE RESULT")
print("=" * 60)

for key, value in result.items():
    print(f"{key}: {value}")

print("=" * 60)

print(
    f"\nSaved to: {results_file}"
)