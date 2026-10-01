import numpy as np
from PIL import Image, ImageStat, ImageFilter

def assess_image_quality(image: Image.Image) -> dict:
    img = image.convert('RGB')
    gray = np.asarray(img.convert('L'), dtype=np.float32)
    # Laplacian-like variance without OpenCV dependency.
    lap = (-4*gray + np.roll(gray,1,0)+np.roll(gray,-1,0)+np.roll(gray,1,1)+np.roll(gray,-1,1))
    sharpness = float(np.var(lap))
    mean = float(gray.mean())
    contrast = float(gray.std())
    return {
        'sharpness': round(sharpness, 2),
        'brightness': round(mean, 2),
        'contrast': round(contrast, 2),
        'quality_flag': 'GOOD' if sharpness >= 20 and 35 <= mean <= 220 and contrast >= 15 else 'CHECK_IMAGE'
    }
