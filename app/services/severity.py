import numpy as np
from PIL import Image

def estimate_affected_area(image: Image.Image) -> dict:
    """Prototype affected-area estimate; not a validated agronomic severity measure."""
    a = np.asarray(image.convert("RGB").resize((256, 256)), dtype=np.float32)
    r, g, b = a[:, :, 0], a[:, :, 1], a[:, :, 2]
    lesion = ((r > g * 1.08) & (g > b * 1.15) & (r > 55) & (g > 35)).astype(np.uint8)
    pct = float(lesion.mean() * 100)
    if pct < 5: cls = "MILD"
    elif pct < 15: cls = "MODERATE"
    elif pct < 30: cls = "SEVERE"
    else: cls = "CRITICAL"
    return {"affected_area_percent": round(pct, 2), "severity_class": cls, "method": "prototype_color_heuristic", "validated": False}
