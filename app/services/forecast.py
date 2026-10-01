import math


def clamp(value, minimum=0.0, maximum=100.0):
    """Keep a value within a safe range."""
    return max(minimum, min(maximum, value))


def forecast(environment, current_score, days=5):
    """
    Generate a short-term scenario projection of crop-health risk.

    IMPORTANT:
    This is a prototype scenario projection, not a validated
    weather-driven time-series forecast.

    The projection uses controlled changes in environmental
    variables and the current fused risk score.
    """

    temperature = float(environment["temperature"])
    humidity = float(environment["humidity"])
    rainfall = float(environment["rainfall"])
    soil_moisture = float(environment["soil_moisture"])

    current_score = float(current_score)

    results = []

    for day in range(1, days + 1):

        # Scenario assumptions
        projected_temperature = temperature + (0.3 * day)
        projected_humidity = clamp(humidity + (1.0 * day), 0, 100)

        projected_rainfall = max(
            0.0,
            rainfall + 1.5 * math.sin(day)
        )

        projected_soil_moisture = clamp(
            soil_moisture + (0.8 * day),
            0,
            100
        )

        # Small environmental pressure adjustment.
        #
        # These weights are intentionally modest because this
        # is a scenario projection rather than a trained
        # forecasting model.
        environmental_change = (
            0.08 * (projected_humidity - humidity)
            + 0.04 * (projected_soil_moisture - soil_moisture)
            + 0.03 * projected_rainfall
            - 0.02 * (projected_temperature - temperature)
        )

        projected_score = (
            current_score
            + (2.0 * day)
            + environmental_change
        )

        projected_score = clamp(projected_score)

        results.append(
            {
                "day": day,
                "risk_score": round(projected_score, 2),
                "projection_type": "scenario_projection",
            }
        )

    return results