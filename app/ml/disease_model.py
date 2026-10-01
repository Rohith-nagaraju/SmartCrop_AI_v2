from pathlib import Path
import json
import os

import numpy as np
from PIL import Image

from ..config import MODEL_DIR, IMAGE_SIZE, UNKNOWN_THRESHOLD


class DiseaseModel:
    """EfficientNet disease classifier with crop-aware output filtering."""

    # ============================================================
    # CROP-SPECIFIC ELIGIBLE CLASSES
    # ============================================================
    #
    # V4 contains 21 total classes:
    # - 1 global Healthy class
    # - 20 disease classes
    #
    # Healthy is shared across all supported crops.
    #
    CROP_CLASSES = {
        "rice": {
            "Healthy",
            "Rice Bacterial Leaf Blight",
            "Rice Leaf Blast",
        },

        "maize": {
            "Healthy",
            "Maize Blight",
            "Maize Common Rust",
            "Maize Gray Leaf Spot",
        },

        "tomato": {
            "Healthy",
            "Tomato Bacterial Spot",
            "Tomato Early Blight",
            "Tomato Late Blight",
            "Tomato Leaf Mold",
            "Tomato Septoria Leaf Spot",
            "Tomato Spider Mites",
            "Tomato Target Spot",
            "Tomato Tomato Mosaic Virus",
            "Tomato Yellow Leaf Curl Virus",
        },

        "grape": {
            "Healthy",
            "Grape Black Rot",
            "Grape Esca Black Measles",
            "Grape Leaf Blight",
        },

        "potato": {
            "Healthy",
            "Potato Early Blight",
            "Potato Late Blight",
        },

        "pepper": {
            "Healthy",
            "Pepper Bacterial Spot",
        },
    }

    def __init__(self):
        self.model = None
        self.classes = []
        self.model_version = "v1"

        # --------------------------------------------------------
        # MODEL SELECTION
        # --------------------------------------------------------
        #
        # Available model versions:
        #
        # v1 = original 6-class model
        # v2 = expanded 15-class model
        # v3 = experimental 15-class model
        # v4 = final 21-class model
        #
        # The version can be selected using:
        #
        # SMARTCROP_MODEL_VERSION=v4
        #
        requested_version = os.getenv(
            "SMARTCROP_MODEL_VERSION",
            "v4",
        ).strip().lower()

        if requested_version == "v4":
            model_filename = "disease_model_v4.keras"
            classes_filename = "class_names_v4.json"
            self.model_version = "v4"

        elif requested_version == "v3":
            model_filename = "disease_model_v3.keras"
            classes_filename = "class_names_v3.json"
            self.model_version = "v3"

        elif requested_version == "v2":
            model_filename = "disease_model_v2.keras"
            classes_filename = "class_names_v2.json"
            self.model_version = "v2"

        else:
            model_filename = "disease_model.keras"
            classes_filename = "class_names.json"
            self.model_version = "v1"

        model_path = MODEL_DIR / model_filename
        classes_path = MODEL_DIR / classes_filename

        # --------------------------------------------------------
        # LOAD MODEL
        # --------------------------------------------------------

        if model_path.exists():
            import tensorflow as tf

            self.model = tf.keras.models.load_model(
                str(model_path)
            )

            # ----------------------------------------------------
            # MODEL WARM-UP
            # ----------------------------------------------------
            #
            # Run one dummy inference during startup so that
            # TensorFlow initialization/tracing does not add
            # unnecessary latency to the first real request.
            #
            # This does NOT modify model weights.
            #

            dummy = np.zeros(
                (
                    1,
                    IMAGE_SIZE[0],
                    IMAGE_SIZE[1],
                    3,
                ),
                dtype=np.float32,
            )

            self.model(
                dummy,
                training=False,
            )

        else:
            print(
                f"WARNING: Disease model file not found: "
                f"{model_path}"
            )

        # --------------------------------------------------------
        # LOAD CLASS NAMES
        # --------------------------------------------------------

        if classes_path.exists():
            with open(
                classes_path,
                "r",
                encoding="utf-8-sig",
            ) as f:
                self.classes = json.load(f)

        else:
            print(
                f"WARNING: Class names file not found: "
                f"{classes_path}"
            )

        print(
            f"Disease model initialized: "
            f"{self.model_version} "
            f"({model_filename})"
        )

        print(
            f"Loaded classes: {len(self.classes)}"
        )

    # ============================================================
    # PREDICTION
    # ============================================================

    def predict(self, image, crop=""):

        if self.model is None:
            return {
                "disease": "MODEL_NOT_TRAINED",
                "confidence": 0.0,
                "uncertain": True,
                "probabilities": {},
                "crop_mismatch": False,
                "model_version": self.model_version,
            }

        # --------------------------------------------------------
        # IMAGE PREPROCESSING
        # --------------------------------------------------------
        #
        # IMPORTANT:
        #
        # EfficientNet-B0 includes its own internal preprocessing.
        #
        # Therefore, DO NOT divide the image by 255 here.
        #
        # Input remains in the 0-255 range.
        #

        x = np.asarray(
            image.convert("RGB").resize(IMAGE_SIZE),
            dtype=np.float32,
        )

        # --------------------------------------------------------
        # MODEL INFERENCE
        # --------------------------------------------------------

        p = self.model(
            x[None, ...],
            training=False,
        ).numpy()[0]

        # --------------------------------------------------------
        # PROBABILITY MAPPING
        # --------------------------------------------------------

        probabilities = {
            self.classes[j]: float(p[j])
            for j in range(len(self.classes))
        }

        crop_key = (
            crop or ""
        ).strip().lower()

        allowed = self.CROP_CLASSES.get(
            crop_key
        )

        # --------------------------------------------------------
        # CROP-AWARE PREDICTION
        # --------------------------------------------------------

        if allowed:

            allowed_indices = [
                i
                for i, name in enumerate(self.classes)
                if name in allowed
            ]

            # ----------------------------------------------------
            # NO MATCHING CLASSES
            # ----------------------------------------------------

            if not allowed_indices:
                return {
                    "disease": "CROP_IMAGE_MISMATCH",
                    "confidence": 0.0,
                    "uncertain": True,
                    "probabilities": probabilities,
                    "crop_mismatch": True,
                    "allowed_probability_mass": 0.0,
                    "model_version": self.model_version,
                }

            # ----------------------------------------------------
            # CHECK PROBABILITY MASS
            # ----------------------------------------------------
            #
            # If less than 30% of the model's probability mass
            # belongs to the selected crop, treat the image as
            # a possible crop mismatch.
            #

            allowed_mass = float(
                np.sum(
                    p[allowed_indices]
                )
            )

            if allowed_mass < 0.30:
                return {
                    "disease": "CROP_IMAGE_MISMATCH",
                    "confidence": round(
                        allowed_mass,
                        6,
                    ),
                    "uncertain": True,
                    "probabilities": probabilities,
                    "crop_mismatch": True,
                    "allowed_probability_mass": allowed_mass,
                    "model_version": self.model_version,
                }

            # ----------------------------------------------------
            # MASK OTHER CROPS
            # ----------------------------------------------------

            masked = np.full_like(
                p,
                -1.0,
            )

            masked[
                allowed_indices
            ] = p[
                allowed_indices
            ]

            i = int(
                np.argmax(masked)
            )

        else:
            # ----------------------------------------------------
            # NO CROP / UNSUPPORTED CROP
            # ----------------------------------------------------
            #
            # Use the complete 21-class output.
            #

            i = int(
                np.argmax(p)
            )

        # --------------------------------------------------------
        # FINAL DIAGNOSIS
        # --------------------------------------------------------

        conf = float(
            p[i]
        )

        disease = self.classes[i]

        uncertain = (
            conf < UNKNOWN_THRESHOLD
        )

        return {
            "disease": (
                "UNCERTAIN"
                if uncertain
                else disease
            ),
            "confidence": conf,
            "uncertain": uncertain,
            "probabilities": probabilities,
            "crop_mismatch": False,
            "model_version": self.model_version,
        }