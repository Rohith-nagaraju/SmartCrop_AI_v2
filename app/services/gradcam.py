import base64
import io

import numpy as np
import tensorflow as tf
from PIL import Image


IMAGE_SIZE = (224, 224)


def find_backbone(model):
    """
    Find the actual EfficientNet-B0 backbone.

    Expected model structure:

        input_2
          ↓
        sequential
          ↓
        efficientnetb0
          ↓
        global_average_pooling2d
          ↓
        dropout
          ↓
        dense
    """

    # Your model has an EfficientNet-B0 layer
    # named exactly "efficientnetb0".
    try:
        return model.get_layer("efficientnetb0")
    except ValueError:
        pass

    # Fallback: search nested models for a 4-D
    # convolutional feature output.
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):
            try:
                if len(layer.output.shape) == 4:
                    return layer
            except Exception:
                continue

    raise ValueError(
        "EfficientNet-B0 backbone not found."
    )


def find_last_conv_layer(backbone):
    """
    Find the final suitable 4-D feature layer
    inside EfficientNet-B0.

    The layer must produce a spatial feature map:
        (batch, height, width, channels)
    """

    candidates = []

    for layer in backbone.layers:
        try:
            shape = layer.output.shape

            if (
                len(shape) == 4
                and shape[1] is not None
                and shape[2] is not None
                and shape[3] is not None
            ):
                candidates.append(layer)

        except Exception:
            continue

    if not candidates:
        raise ValueError(
            "No 4-D feature layer found inside EfficientNet-B0."
        )

    return candidates[-1]


def make_gradcam_heatmap(
    image,
    model,
    class_index=None,
):
    """
    Generate a Grad-CAM heatmap for the disease model.

    The model receives raw 0-255 pixel values because
    preprocessing/rescaling is already part of the
    EfficientNet model.
    """

    # ---------------------------------------------------------
    # 1. Prepare image
    # ---------------------------------------------------------

    img = (
        image
        .convert("RGB")
        .resize(IMAGE_SIZE)
    )

    img_array = np.asarray(
        img,
        dtype=np.float32,
    )

    img_array = np.expand_dims(
        img_array,
        axis=0,
    )

    input_tensor = tf.convert_to_tensor(
        img_array,
        dtype=tf.float32,
    )

    # ---------------------------------------------------------
    # 2. Find EfficientNet-B0
    # ---------------------------------------------------------

    backbone = find_backbone(model)

    target_layer = find_last_conv_layer(
        backbone
    )

    # ---------------------------------------------------------
    # 3. Build the Grad-CAM feature model
    # ---------------------------------------------------------

    grad_model = tf.keras.models.Model(
        inputs=backbone.input,
        outputs=[
            target_layer.output,
            backbone.output,
        ],
    )

    # ---------------------------------------------------------
    # 4. Forward pass through EfficientNet
    # ---------------------------------------------------------

    with tf.GradientTape() as tape:

        tape.watch(input_tensor)

        conv_outputs, backbone_output = (
            grad_model(
                input_tensor,
                training=False,
            )
        )

        # -----------------------------------------------------
        # Reconstruct the classification head.
        #
        # Your model is:
        #
        # efficientnetb0
        #     ↓
        # global_average_pooling2d
        #     ↓
        # dropout
        #     ↓
        # dense
        # -----------------------------------------------------

        x = backbone_output

        backbone_position = None

        for index, layer in enumerate(model.layers):
            if layer.name == backbone.name:
                backbone_position = index
                break

        if backbone_position is None:
            raise ValueError(
                "Could not locate EfficientNet-B0 "
                "inside the main model."
            )

        head_layers = model.layers[
            backbone_position + 1:
        ]

        for layer in head_layers:
            x = layer(
                x,
                training=False,
            )

        predictions = x

        # -----------------------------------------------------
        # Determine predicted class
        # -----------------------------------------------------

        if class_index is None:
            class_index = tf.argmax(
                predictions[0]
            )

        class_index = tf.cast(
            class_index,
            tf.int32,
        )

        class_score = predictions[
            0,
            class_index,
        ]

    # ---------------------------------------------------------
    # 5. Calculate gradients
    # ---------------------------------------------------------

    grads = tape.gradient(
        class_score,
        conv_outputs,
    )

    if grads is None:
        raise ValueError(
            "Grad-CAM gradients could not be calculated."
        )

    # ---------------------------------------------------------
    # 6. Global average pooling of gradients
    # ---------------------------------------------------------

    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2),
    )

    # Remove batch dimension
    conv_outputs = conv_outputs[0]

    # ---------------------------------------------------------
    # 7. Create weighted activation map
    # ---------------------------------------------------------

    heatmap = tf.reduce_sum(
        conv_outputs
        * pooled_grads,
        axis=-1,
    )

    # ReLU
    heatmap = tf.maximum(
        heatmap,
        0,
    )

    # ---------------------------------------------------------
    # 8. Normalize between 0 and 1
    # ---------------------------------------------------------

    max_value = tf.reduce_max(
        heatmap
    )

    if float(max_value.numpy()) > 0:
        heatmap = (
            heatmap / max_value
        )

    return heatmap.numpy()


def overlay_heatmap(
    image,
    heatmap,
    alpha=0.45,
):
    """
    Overlay Grad-CAM heatmap on the original leaf image.
    """

    # Original image
    original = (
        image
        .convert("RGB")
        .resize(IMAGE_SIZE)
    )

    original_array = np.asarray(
        original,
        dtype=np.uint8,
    )

    # Convert heatmap to 0-255
    heatmap_uint8 = np.uint8(
        255 * heatmap
    )

    # Matplotlib colour map
    import matplotlib.cm as cm

    colormap = cm.get_cmap(
        "jet"
    )

    colored_heatmap = colormap(
        heatmap_uint8
    )[:, :, :3]

    colored_heatmap = np.uint8(
        colored_heatmap * 255
    )

    # Convert to PIL and resize
    heatmap_image = (
        Image.fromarray(
            colored_heatmap
        )
        .resize(IMAGE_SIZE)
    )

    heatmap_array = np.asarray(
        heatmap_image,
        dtype=np.float32,
    )

    # Blend
    overlay = (
        alpha * heatmap_array
        + (1 - alpha) * original_array
    )

    overlay = np.clip(
        overlay,
        0,
        255,
    ).astype(
        np.uint8
    )

    return Image.fromarray(
        overlay
    )


def image_to_base64(image):
    """
    Convert PIL image to Base64 JPEG.
    """

    buffer = io.BytesIO()

    image.save(
        buffer,
        format="JPEG",
        quality=90,
    )

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("utf-8")

    return encoded


def generate_gradcam(
    image,
    model,
    class_index=None,
):
    """
    Complete Grad-CAM pipeline.
    """

    heatmap = make_gradcam_heatmap(
        image=image,
        model=model,
        class_index=class_index,
    )

    overlay = overlay_heatmap(
        image=image,
        heatmap=heatmap,
    )

    return {
        "heatmap": heatmap,
        "image_base64": image_to_base64(
            overlay
        ),
    }


def make_gradcam_placeholder(image):
    """
    Backward-compatible wrapper.

    The already-loaded disease model is supplied by
    app.main, so Grad-CAM does not reload the model
    for every API request.
    """

    try:
        model = getattr(
            make_gradcam_placeholder,
            "_model",
            None,
        )

        if model is None:
            raise ValueError(
                "Grad-CAM model reference has not been initialized."
            )

        result = generate_gradcam(
            image=image,
            model=model,
        )

        return result["image_base64"]

    except Exception as e:
        print(
            f"Grad-CAM generation failed: {e}"
        )
        return None