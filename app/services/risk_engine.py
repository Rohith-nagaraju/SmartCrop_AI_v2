"""Transparent environmental and fused crop-health risk calculations."""

def dpi_from_environment(temperature, humidity, rainfall, soil_moisture):
    temp = max(0.0, 1 - abs(25 - temperature) / 20)
    hum = min(max((humidity - 55) / 45, 0), 1)
    rain = min(rainfall / 30, 1)
    wet_soil = min(max((soil_moisture - 45) / 55, 0), 1)
    dpi = 100 * (0.25 * temp + 0.35 * hum + 0.20 * rain + 0.20 * wet_soil)
    return round(dpi, 2)

def fuse_risk(confidence, severity_percent, dpi):
    severity = min(severity_percent / 25, 1) * 100
    score = 0.40 * (confidence * 100) + 0.30 * severity + 0.30 * dpi
    score = round(min(score, 100), 2)
    if score < 25: level = "LOW"
    elif score < 50: level = "MODERATE"
    elif score < 75: level = "HIGH"
    else: level = "CRITICAL"
    return score, level

def risk_decomposition(confidence, severity_percent, temperature, humidity, rainfall, soil_moisture):
    """Return score-point contributions consistent with fuse_risk.

    These are model-score contributions, not causal effects or statistical
    feature importance. Their sum equals the fused risk score before rounding.
    """
    temp_factor = max(0.0, 1 - abs(25 - temperature) / 20)
    hum_factor = min(max((humidity - 55) / 45, 0), 1)
    rain_factor = min(rainfall / 30, 1)
    soil_factor = min(max((soil_moisture - 45) / 55, 0), 1)
    raw = {
        "disease": 0.40 * (confidence * 100),
        "severity": 0.30 * (min(severity_percent / 25, 1) * 100),
        "temperature": 30 * 0.25 * temp_factor,
        "humidity": 30 * 0.35 * hum_factor,
        "rainfall": 30 * 0.20 * rain_factor,
        "soil_moisture": 30 * 0.20 * soil_factor,
    }
    return {k: round(v, 2) for k, v in raw.items()}
