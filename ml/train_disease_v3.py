import json
import time
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    ReduceLROnPlateau,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

TRAIN_DIR = BASE_DIR / "data" / "disease_dataset_v2" / "train"
VAL_DIR = BASE_DIR / "data" / "disease_dataset_v2" / "val"

V2_MODEL_PATH = BASE_DIR / "models" / "disease_model_v2.keras"
V3_MODEL_PATH = BASE_DIR / "models" / "disease_model_v3.keras"
CLASS_NAMES_PATH = BASE_DIR / "models" / "class_names_v2.json"

RESULTS_DIR = BASE_DIR / "experiments" / "disease_models" / "v3"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

INITIAL_EPOCHS = 4
FINETUNE_EPOCHS = 6

# Fine-tune the last 40 layers instead of the last 30.
FINETUNE_LAST_N = 40


# ============================================================
# CLASS-SPECIFIC WEIGHTS
# ============================================================
#
# Targeted improvement:
#
# Maize:
#   Blight <-> Gray Leaf Spot
#
# Rice:
#   BLB <-> Leaf Blast
#
# Tomato:
#   Target Spot
#   Bacterial Spot
#   Spider Mites
#   Early Blight
#   Late Blight
#   Septoria
#
# We use only a MODERATE boost so that strong classes are
# not unnecessarily disturbed.
# ============================================================

TARGET_CLASS_BOOSTS = {
    "Maize Blight": 1.10,
    "Maize Gray Leaf Spot": 1.15,

    "Rice Bacterial Leaf Blight": 1.15,
    "Rice Leaf Blast": 1.10,

    "Tomato Bacterial Spot": 1.10,
    "Tomato Early Blight": 1.10,
    "Tomato Late Blight": 1.10,
    "Tomato Septoria Leaf Spot": 1.10,
    "Tomato Spider Mites": 1.10,
    "Tomato Target Spot": 1.15,
}


# ============================================================
# LOAD CLASS NAMES
# ============================================================

with open(CLASS_NAMES_PATH, "r", encoding="utf-8-sig") as f:
    class_names = json.load(f)

print()
print("=" * 75)
print("V3 TARGETED 15-CLASS DISEASE MODEL TRAINING")
print("=" * 75)

print()
print("Classes:")
for i, name in enumerate(class_names):
    print(f"{i}: {name}")

print(f"\nNumber of classes: {len(class_names)}")

print()
print("Existing models will NOT be modified.")
print(f"V2 model: {V2_MODEL_PATH}")
print(f"New V3 model: {V3_MODEL_PATH}")


# ============================================================
# DATA AUGMENTATION
# ============================================================
#
# Moderate augmentation is used to improve robustness without
# making the training distribution excessively different.
# ============================================================

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.05, 0.05),
        layers.RandomContrast(0.10),
    ],
    name="controlled_augmentation",
)


# ============================================================
# LOAD DATASETS
# ============================================================

print()
print("=" * 75)
print("LOADING DATA")
print("=" * 75)

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED,
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="int",
    class_names=class_names,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)


# ============================================================
# CALCULATE BALANCED CLASS WEIGHTS
# ============================================================

print()
print("=" * 75)
print("CALCULATING CLASS WEIGHTS")
print("=" * 75)

train_labels = []

for _, labels in train_ds.unbatch():
    train_labels.append(int(labels.numpy()))

train_labels = np.array(train_labels)

balanced_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(class_names)),
    y=train_labels,
)

class_weights = {
    i: float(weight)
    for i, weight in enumerate(balanced_weights)
}


# Apply moderate targeted boosts.
for i, class_name in enumerate(class_names):
    if class_name in TARGET_CLASS_BOOSTS:
        class_weights[i] *= TARGET_CLASS_BOOSTS[class_name]


print("\nFinal V3 class weights:")

for i, name in enumerate(class_names):
    print(f"{i:2d}  {name:35s} {class_weights[i]:.4f}")


# ============================================================
# BUILD MODEL
# ============================================================

print()
print("=" * 75)
print("BUILDING EFFICIENTNET-B0 MODEL")
print("=" * 75)

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)

base_model.trainable = False

inputs = layers.Input(shape=(224, 224, 3))

x = data_augmentation(inputs)

# IMPORTANT:
# Keras EfficientNet includes its own input rescaling.
# Therefore raw 0-255 image values are passed to the model.
x = base_model(x, training=False)

x = layers.GlobalAveragePooling2D()(x)

x = layers.Dropout(0.30)(x)

outputs = layers.Dense(
    len(class_names),
    activation="softmax",
)(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-3
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# ============================================================
# CALLBACKS — INITIAL TRAINING
# ============================================================

checkpoint_initial = RESULTS_DIR / "v3_initial_best.keras"

callbacks_initial = [
    ModelCheckpoint(
        filepath=str(checkpoint_initial),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=1,
        min_lr=1e-6,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
        verbose=1,
    ),
]


# ============================================================
# STAGE 1 — FROZEN BACKBONE
# ============================================================

print()
print("=" * 75)
print("STAGE 1 — FROZEN EFFICIENTNET-B0")
print("=" * 75)

start_time = time.time()

history_initial = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=INITIAL_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks_initial,
)

stage1_time = time.time() - start_time


# ============================================================
# STAGE 2 — TARGETED FINE-TUNING
# ============================================================

print()
print("=" * 75)
print("STAGE 2 — TARGETED FINE-TUNING")
print("=" * 75)

base_model.trainable = True

# Freeze earlier layers.
for layer in base_model.layers[:-FINETUNE_LAST_N]:
    layer.trainable = False

# Keep BatchNorm layers frozen for stable fine-tuning.
for layer in base_model.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False


model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=1e-5
    ),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)


checkpoint_final = RESULTS_DIR / "v3_best.keras"

callbacks_final = [
    ModelCheckpoint(
        filepath=str(checkpoint_final),
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.3,
        patience=2,
        min_lr=1e-7,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True,
        verbose=1,
    ),
]


start_time = time.time()

history_finetune = model.fit(
    train_ds,
    validation_data=val_ds,
    initial_epoch=len(history_initial.history["loss"]),
    epochs=INITIAL_EPOCHS + FINETUNE_EPOCHS,
    class_weight=class_weights,
    callbacks=callbacks_final,
)

stage2_time = time.time() - start_time


# ============================================================
# SAVE FINAL V3 MODEL
# ============================================================

model.save(V3_MODEL_PATH)

# Save class names separately for V3.
v3_class_names_path = BASE_DIR / "models" / "class_names_v3.json"

with open(v3_class_names_path, "w", encoding="utf-8") as f:
    json.dump(class_names, f, indent=2)


# ============================================================
# SAVE TRAINING INFORMATION
# ============================================================

training_info = {
    "model": "EfficientNet-B0",
    "classes": len(class_names),
    "image_size": IMAGE_SIZE,
    "batch_size": BATCH_SIZE,
    "seed": SEED,
    "initial_epochs": INITIAL_EPOCHS,
    "finetune_epochs": FINETUNE_EPOCHS,
    "finetune_last_layers": FINETUNE_LAST_N,
    "target_class_boosts": TARGET_CLASS_BOOSTS,
    "stage1_training_seconds": stage1_time,
    "stage2_training_seconds": stage2_time,
    "total_training_seconds": stage1_time + stage2_time,
    "v2_model_preserved": True,
    "original_6_class_model_preserved": True,
}

with open(
    RESULTS_DIR / "v3_training_info.json",
    "w",
    encoding="utf-8",
) as f:
    json.dump(training_info, f, indent=2)


# ============================================================
# FINAL OUTPUT
# ============================================================

print()
print("=" * 75)
print("V3 TRAINING COMPLETED SUCCESSFULLY")
print("=" * 75)

print()
print("Existing models were NOT modified.")

print()
print("Original 6-class model:")
print(f"  {BASE_DIR / 'models' / 'disease_model.keras'}")

print()
print("V2 15-class model:")
print(f"  {V2_MODEL_PATH}")

print()
print("NEW V3 15-class model:")
print(f"  {V3_MODEL_PATH}")

print()
print("V3 class names:")
print(f"  {v3_class_names_path}")

print()
print(f"Total training time: {stage1_time + stage2_time:.2f} seconds")

print()
print("Next step:")
print("Evaluate V3 on the SAME held-out test dataset.")
print("=" * 75)