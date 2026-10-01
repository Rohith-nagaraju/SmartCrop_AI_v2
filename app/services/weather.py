import json
from datetime import datetime
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError


GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"


WEATHER_CODES = {
    0: "Clear sky",

    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",

    45: "Fog",
    48: "Depositing rime fog",

    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",

    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",

    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",

    66: "Light freezing rain",
    67: "Heavy freezing rain",

    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",

    77: "Snow grains",

    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",

    85: "Slight snow showers",
    86: "Heavy snow showers",

    95: "Thunderstorm",

    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


def _get_json(url: str):

    request = Request(
        url,
        headers={
            "User-Agent": "SmartCrop-AI/1.0"
        },
    )

    try:

        with urlopen(request, timeout=15) as response:

            data = response.read().decode("utf-8")

            return json.loads(data)

    except HTTPError as e:

        raise RuntimeError(
            f"Weather service HTTP error: {e.code}"
        )

    except URLError as e:

        raise RuntimeError(
            f"Weather service unavailable: {e.reason}"
        )

    except Exception as e:

        raise RuntimeError(
            f"Weather service error: {e}"
        )


def geocode_location(location: str):

    location = location.strip()

    if not location:

        raise ValueError("Location cannot be empty.")

    params = urlencode(
        {
            "name": location,
            "count": 5,
            "language": "en",
            "format": "json",
        }
    )

    url = f"{GEOCODING_URL}?{params}"

    data = _get_json(url)

    results = data.get("results", [])

    if not results:

        raise ValueError(
            f"Could not find the location '{location}'. "
            "Try entering a city, district or postal code."
        )

    # Prefer the first ranked result returned by Open-Meteo.
    result = results[0]

    return {
        "name": result.get("name"),
        "latitude": float(result["latitude"]),
        "longitude": float(result["longitude"]),
        "country": result.get("country"),
        "country_code": result.get("country_code"),
        "state": result.get("admin1"),
        "city": result.get("name"),
        "timezone": result.get("timezone"),
    }


def _mean(values):

    values = [
        float(v)
        for v in values
        if v is not None
    ]

    if not values:

        return None

    return sum(values) / len(values)


def get_five_day_weather(latitude: float, longitude: float):

    params = urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,

            "forecast_days": 5,

            "timezone": "auto",

            "hourly": ",".join(
                [
                    "temperature_2m",
                    "relative_humidity_2m",
                    "precipitation",
                    "soil_moisture_0_to_1cm",
                    "weather_code",
                ]
            ),

            "daily": ",".join(
                [
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_sum",
                    "precipitation_probability_max",
                    "weather_code",
                ]
            ),
        }
    )

    url = f"{FORECAST_URL}?{params}"

    data = _get_json(url)

    daily = data.get("daily", {})
    hourly = data.get("hourly", {})

    dates = daily.get("time", [])

    daily_max = daily.get("temperature_2m_max", [])
    daily_min = daily.get("temperature_2m_min", [])
    daily_rain = daily.get("precipitation_sum", [])
    daily_rain_probability = daily.get(
        "precipitation_probability_max",
        []
    )
    daily_weather_codes = daily.get(
        "weather_code",
        []
    )

    hourly_times = hourly.get("time", [])
    hourly_temp = hourly.get(
        "temperature_2m",
        []
    )
    hourly_humidity = hourly.get(
        "relative_humidity_2m",
        []
    )
    hourly_rain = hourly.get(
        "precipitation",
        []
    )
    hourly_soil = hourly.get(
        "soil_moisture_0_to_1cm",
        []
    )
    hourly_weather = hourly.get(
        "weather_code",
        []
    )

    forecast = []

    for index, date in enumerate(dates):

        day_temperature = []

        day_humidity = []

        day_soil = []

        day_rain = []

        day_codes = []

        for i, timestamp in enumerate(hourly_times):

            if timestamp.startswith(date):

                if i < len(hourly_temp):

                    day_temperature.append(
                        hourly_temp[i]
                    )

                if i < len(hourly_humidity):

                    day_humidity.append(
                        hourly_humidity[i]
                    )

                if i < len(hourly_soil):

                    day_soil.append(
                        hourly_soil[i]
                    )

                if i < len(hourly_rain):

                    day_rain.append(
                        hourly_rain[i]
                    )

                if i < len(hourly_weather):

                    day_codes.append(
                        hourly_weather[i]
                    )

        code = None

        if index < len(daily_weather_codes):

            code = daily_weather_codes[index]

        if code is None and day_codes:

            code = day_codes[len(day_codes) // 2]

        weather_text = WEATHER_CODES.get(
            int(code) if code is not None else -1,
            "Unknown conditions"
        )

        humidity = _mean(day_humidity)

        soil_moisture = _mean(day_soil)

        # Open-Meteo soil moisture is volumetric water
        # content (m³/m³). Convert to a convenient
        # percentage representation for the existing
        # SmartCrop risk engine.
        if soil_moisture is not None:

            soil_moisture_percent = round(
                soil_moisture * 100,
                2
            )

        else:

            soil_moisture_percent = None

        rainfall = None

        if index < len(daily_rain):

            rainfall = daily_rain[index]

        if rainfall is None:

            rainfall = sum(
                float(v)
                for v in day_rain
                if v is not None
            )

        rain_probability = None

        if index < len(
            daily_rain_probability
        ):

            rain_probability = (
                daily_rain_probability[index]
            )

        forecast.append(
            {
                "date": date,

                "temperature_max": round(
                    float(daily_max[index]),
                    1
                ) if index < len(daily_max)
                else None,

                "temperature_min": round(
                    float(daily_min[index]),
                    1
                ) if index < len(daily_min)
                else None,

                "humidity": round(
                    humidity,
                    1
                ) if humidity is not None
                else None,

                "rainfall": round(
                    float(rainfall),
                    1
                ) if rainfall is not None
                else 0.0,

                "rain_probability": round(
                    float(rain_probability),
                    1
                ) if rain_probability is not None
                else None,

                "soil_moisture": soil_moisture_percent,

                "weather_code": code,

                "weather": weather_text,
            }
        )

    return {
        "latitude": latitude,

        "longitude": longitude,

        "timezone": data.get("timezone"),

        "generated_at": datetime.utcnow().isoformat(),

        "forecast": forecast,
    }