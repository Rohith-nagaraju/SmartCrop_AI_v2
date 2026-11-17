from pathlib import Path

import hashlib
import hmac
import secrets
import io

from typing import Optional

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    Form,
    HTTPException,
    Header,
)

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from PIL import Image


from .database import get_db

from .schemas import (
    FarmerRegister,
    FarmerLogin,
)

from .ml.disease_model import DiseaseModel

from .services.image_quality import (
    assess_image_quality
)

from .services.severity import (
    estimate_affected_area
)

from .services.risk_engine import (
    dpi_from_environment,
    fuse_risk,
    risk_decomposition,
)

from .services.recommender import (
    recommend,
    build_recommendation,
)

from .services.gradcam import (
    make_gradcam_placeholder,
)

from .services.weather import (
    geocode_location,
    get_five_day_weather,
)

from .services.translation_service import (
    SUPPORTED_LANGUAGES,
    translate_recommendation_plan,
)



# ============================================================
# APPLICATION
# ============================================================

app = FastAPI(
    title="SmartCrop AI",
    version="4.0"
)


# ============================================================
# FRONTEND
# ============================================================

FRONTEND = (
    Path(__file__).resolve().parents[1]
    / "frontend"
)


app.mount(
    "/static",
    StaticFiles(directory=FRONTEND),
    name="static"
)


# ============================================================
# DISEASE MODEL
# ============================================================

model = DiseaseModel()


# Share the already-loaded model with Grad-CAM.
try:
    make_gradcam_placeholder._model = model.model
except Exception:
    pass


# ============================================================
# PASSWORD SECURITY
# ============================================================

def hash_password(password: str) -> str:

    salt = secrets.token_bytes(16)

    derived = hashlib.pbkdf2_hmac(
        "sha256",
        password.encode("utf-8"),
        salt,
        120000,
    )

    return (
        salt.hex()
        + ":"
        + derived.hex()
    )


def verify_password(
    password: str,
    stored_password: str
) -> bool:

    try:

        salt_hex, hash_hex = (
            stored_password.split(":")
        )

        salt = bytes.fromhex(
            salt_hex
        )

        expected = bytes.fromhex(
            hash_hex
        )

        actual = hashlib.pbkdf2_hmac(
            "sha256",
            password.encode("utf-8"),
            salt,
            120000,
        )

        return hmac.compare_digest(
            actual,
            expected
        )

    except Exception:
        return False


# ============================================================
# AUTHENTICATION
# ============================================================

def get_farmer_from_token(
    authorization: Optional[str]
):

    if not authorization:

        raise HTTPException(
            status_code=401,
            detail="Login required."
        )

    if not authorization.startswith(
        "Bearer "
    ):

        raise HTTPException(
            status_code=401,
            detail="Invalid authorization header."
        )

    token = authorization[7:].strip()

    if not token:

        raise HTTPException(
            status_code=401,
            detail="Invalid session token."
        )

    with get_db() as db:

        row = db.execute(
            """
            SELECT
                farmers.*
            FROM sessions
            JOIN farmers
                ON farmers.id = sessions.farmer_id
            WHERE sessions.token = ?
            """,
            (token,)
        ).fetchone()

    if row is None:

        raise HTTPException(
            status_code=401,
            detail="Session expired or invalid."
        )

    return row


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return FileResponse(
        FRONTEND / "index.html"
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",

        "model_loaded": (
            model.model is not None
        ),

        "version": "4.0",

        "weather_enabled": True,

        "recommendation_engine":
            "rule_based_crop_health_recommendation_v2",
    }


# ============================================================
# REGISTER
# ============================================================

@app.post("/api/auth/register")
def register(
    data: FarmerRegister
):

    email = (
        data.email
        .strip()
        .lower()
    )

    crop = (
        data.crop
        .strip()
        .lower()
    )

    supported_crops = {
        "rice",
        "maize",
        "tomato",
        "grape",
        "potato",
        "pepper",
    }

    if crop not in supported_crops:

        raise HTTPException(
            status_code=400,
            detail=(
                "Unsupported crop. "
                "Supported crops: rice, maize, tomato, "
                "grape, potato and pepper."
            )
        )

    # --------------------------------------------------------
    # LOCATION GEOCODING
    # --------------------------------------------------------

    try:

        location = geocode_location(
            data.location
        )

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    # --------------------------------------------------------
    # PASSWORD
    # --------------------------------------------------------

    password_hash = hash_password(
        data.password
    )

    # --------------------------------------------------------
    # DATABASE
    # --------------------------------------------------------

    with get_db() as db:

        existing = db.execute(
            """
            SELECT id
            FROM farmers
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

        if existing:

            raise HTTPException(
                status_code=409,
                detail=(
                    "An account with this email "
                    "already exists."
                )
            )


            
        cursor = db.execute(
            """
            INSERT INTO farmers(
                full_name,
                email,
                password_hash,
                farm_name,
                location_name,
                latitude,
                longitude,
                country,
                state,
                city,
                crop,
                field_id
            )
            VALUES(
                ?,?,?,?,?,?,?,?,?,?,?,?
            )
            RETURNING id
            """,
            (
                data.full_name.strip(),
                email,
                password_hash,
                data.farm_name.strip(),
                location["name"],
                location["latitude"],
                location["longitude"],
                location.get("country"),
                location.get("state"),
                location.get("city"),
                crop,
                data.field_id.strip(),
            )
        )

        farmer_id = cursor.fetchone()["id"]


    return {
        "success": True,

        "message":
            "Farmer account created successfully.",

        "farmer_id":
            farmer_id,

        "location":
            location,
    }


# ============================================================
# LOGIN
# ============================================================

@app.post("/api/auth/login")
def login(
    data: FarmerLogin
):

    email = (
        data.email
        .strip()
        .lower()
    )

    with get_db() as db:

        farmer = db.execute(
            """
            SELECT *
            FROM farmers
            WHERE email = ?
            """,
            (email,)
        ).fetchone()

        if farmer is None:

            raise HTTPException(
                status_code=401,
                detail="Invalid email or password."
            )

        if not verify_password(
            data.password,
            farmer["password_hash"]
        ):

            raise HTTPException(
                status_code=401,
                detail="Invalid email or password."
            )

        # ----------------------------------------------------
        # CREATE SESSION
        # ----------------------------------------------------

        token = secrets.token_urlsafe(
            32
        )

        db.execute(
            """
            INSERT INTO sessions(
                token,
                farmer_id
            )
            VALUES(?,?)
            """,
            (
                token,
                farmer["id"]
            )
        )

    return {
        "success": True,

        "token":
            token,

        "farmer": {
            "id":
                farmer["id"],

            "full_name":
                farmer["full_name"],

            "email":
                farmer["email"],

            "farm_name":
                farmer["farm_name"],

            "location":
                farmer["location_name"],

            "latitude":
                farmer["latitude"],

            "longitude":
                farmer["longitude"],

            "country":
                farmer["country"],

            "state":
                farmer["state"],

            "city":
                farmer["city"],

            "crop":
                farmer["crop"],

            "field_id":
                farmer["field_id"],
        }
    }


# ============================================================
# LOGOUT
# ============================================================

@app.post("/api/auth/logout")
def logout(
    authorization: Optional[str] = Header(
        default=None
    )
):

    if (
        authorization
        and authorization.startswith("Bearer ")
    ):

        token = authorization[7:].strip()

        with get_db() as db:

            db.execute(
                """
                DELETE FROM sessions
                WHERE token = ?
                """,
                (token,)
            )

    return {
        "success": True
    }


# ============================================================
# CURRENT FARMER PROFILE
# ============================================================

@app.get("/api/auth/me")
def me(
    authorization: Optional[str] = Header(
        default=None
    )
):

    farmer = get_farmer_from_token(
        authorization
    )

    return {
        "id":
            farmer["id"],

        "full_name":
            farmer["full_name"],

        "email":
            farmer["email"],

        "farm_name":
            farmer["farm_name"],

        "location":
            farmer["location_name"],

        "latitude":
            farmer["latitude"],

        "longitude":
            farmer["longitude"],

        "country":
            farmer["country"],

        "state":
            farmer["state"],

        "city":
            farmer["city"],

        "crop":
            farmer["crop"],

        "field_id":
            farmer["field_id"],
    }


# ============================================================
# LIVE WEATHER
# ============================================================

@app.get("/api/weather")
def weather(
    authorization: Optional[str] = Header(
        default=None
    )
):

    farmer = get_farmer_from_token(
        authorization
    )

    try:

        result = get_five_day_weather(
            farmer["latitude"],
            farmer["longitude"],
        )

    except Exception as e:

        raise HTTPException(
            status_code=503,
            detail=str(e)
        )

    return {
        "location": {
            "name":
                farmer["location_name"],

            "city":
                farmer["city"],

            "state":
                farmer["state"],

            "country":
                farmer["country"],

            "latitude":
                farmer["latitude"],

            "longitude":
                farmer["longitude"],
        },

        **result,
    }


# ============================================================
# ANALYSIS
# ============================================================

@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),

    crop: Optional[str] = Form(
        default=None
    ),

    field_id: Optional[str] = Form(
        default=None
    ),

    growth_stage: str = Form(
        default="Vegetative"
    ),

    authorization: Optional[str] = Header(
        default=None
    ),
):

    # ========================================================
    # AUTHENTICATION
    # ========================================================

    farmer = get_farmer_from_token(
        authorization
    )

    # ========================================================
    # SELECT FARMER CROP / FIELD
    # ========================================================

    selected_crop = (
        crop.strip().lower()
        if crop
        else farmer["crop"]
    )

    selected_field = (
        field_id.strip()
        if field_id
        else farmer["field_id"]
    )

    # Keep the backend response consistent even if the frontend
    # sends an empty growth-stage value.
    growth_stage = (
        (growth_stage or "").strip()
        or "Vegetative"
    )

    supported_crops = {
        "rice",
        "maize",
        "tomato",
        "grape",
        "potato",
        "pepper",
    }

    if selected_crop not in supported_crops:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported crop '{selected_crop}'. "
                "Please select a supported crop."
            )
        )

    # ========================================================
    # READ IMAGE
    # ========================================================

    raw = await image.read()

    if not raw:

        raise HTTPException(
            status_code=400,
            detail="Uploaded image is empty."
        )

    try:

        img = Image.open(
            io.BytesIO(raw)
        ).convert("RGB")

    except Exception as e:

        raise HTTPException(
            status_code=400,
            detail=f"Invalid image: {e}"
        )

    # ========================================================
    # IMAGE QUALITY
    # ========================================================

    try:

        quality = assess_image_quality(
            img
        )

    except Exception as e:

        quality = {
            "status": "unavailable",
            "message": str(e),
        }

    # ========================================================
    # DISEASE MODEL
    # ========================================================

    try:

        pred = model.predict(
            img,
            crop=selected_crop
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Disease model failed: {e}"
        )

    # ========================================================
    # CROP / IMAGE MISMATCH
    # ========================================================

    if pred.get("crop_mismatch"):

        explanation = {

            "status":
                "CROP_IMAGE_MISMATCH",

            "message":
                (
                    "The uploaded image does not provide "
                    "enough probability mass for the selected "
                    "crop. Check the crop selection and upload "
                    "a clear leaf image."
                ),

            "selected_crop":
                selected_crop,

            "allowed_probability_mass":
                round(
                    pred.get(
                        "allowed_probability_mass",
                        0
                    ) * 100,
                    2
                ),
        }

        mismatch_recommendation = (
            "Verify the selected crop and upload "
            "a clear leaf image before making a "
            "crop-health decision."
        )

        mismatch_plan = {

            "summary":
                mismatch_recommendation,

            "priority":
                "VERIFY IMAGE",

            "disease":
                pred["disease"],

            "crop":
                selected_crop,

            "growth_stage":
                growth_stage,

            "disease_category":
                "mismatch",

            "immediate_actions": [
                "Verify the selected crop.",
                "Upload a clear leaf image.",
                "Inspect the plant directly before treatment.",
            ],

            "organic_management": [
                "Maintain normal field sanitation.",
                "Avoid disease-specific treatment until the diagnosis is verified.",
            ],

            "avoid": [
                "Do not apply disease-specific products solely from a mismatched image prediction.",
            ],

            "weather_context": {
                "message":
                    (
                        "Weather information can support "
                        "monitoring but cannot correct an "
                        "incorrect crop-image match."
                    )
            },

            "growth_stage_note":
                None,

            "organic_disclaimer":
                (
                    "Organic-management options are "
                    "decision-support guidance, not a "
                    "product prescription."
                ),

            "recommendation_text":
                mismatch_recommendation,

            "engine":
                "crop_mismatch_safety_rule",
        }

        return {

            "disease":
                pred["disease"],

            "confidence":
                pred["confidence"],

            "uncertain":
                True,

            "crop_mismatch":
                True,

            "severity_percent":
                None,

            "severity_class":
                "NOT_ASSESSED",

            "dpi":
                None,

            "risk_score":
                None,

            "risk_level":
                "NOT_ASSESSED",

            "forecast":
                [],

            "recommendation":
                mismatch_recommendation,

            "recommendation_plan":
                mismatch_plan,

            "explanation":
                explanation,

            "image_quality":
                quality,

            "gradcam":
                make_gradcam_placeholder(
                    img
                ),
        }

    # ========================================================
    # SEVERITY
    # ========================================================

    try:

        sev = estimate_affected_area(
            img
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Severity estimation failed: {e}"
        )

    severity_percent = float(
        sev.get(
            "affected_area_percent",
            0
        )
    )

    severity_class = sev.get(
        "severity_class",
        "MILD"
    )

    # ========================================================
    # LIVE 5-DAY WEATHER
    # ========================================================

    try:
        weather_data = get_five_day_weather(
            farmer["latitude"],
            farmer["longitude"],
        )
        weather_days = weather_data.get("forecast", [])

    except Exception as e:
        # Weather is supplementary to disease classification.
        # Continue the analysis without environmental scoring.
        print(f"Weather unavailable during analysis: {e}")
        weather_days = []

    # ========================================================
    # CURRENT ENVIRONMENT
    # ========================================================

    today = weather_days[0] if weather_days else {}

    # Preserve missing values. Never invent environmental inputs.
    current_temperature = today.get("temperature_max")
    current_humidity = today.get("humidity")
    current_rainfall = today.get("rainfall")
    current_soil = today.get("soil_moisture")

    # ========================================================
    # ENVIRONMENTAL DATA VALIDATION
    # ========================================================

    weather_available = all(
        value is not None
        for value in (
            current_temperature,
            current_humidity,
            current_rainfall,
            current_soil,
        )
    )

    # ========================================================
    # CURRENT ENVIRONMENTAL PRESSURE AND RISK
    # ========================================================

    if weather_available:
        dpi = dpi_from_environment(
            current_temperature,
            current_humidity,
            current_rainfall,
            current_soil,
        )

        score, level = fuse_risk(
            pred["confidence"],
            severity_percent,
            dpi,
        )

        # These are score-point contributions consistent with the
        # current fuse_risk() formula. They are not causal effects
        # or statistical feature-importance values.
        risk_breakdown = risk_decomposition(
            confidence=pred["confidence"],
            severity_percent=severity_percent,
            temperature=current_temperature,
            humidity=current_humidity,
            rainfall=current_rainfall,
            soil_moisture=current_soil,
        )

    else:
        dpi = None
        score = None
        level = None
        risk_breakdown = None

    # Only pass environmental measurements to recommendations
    # when the complete environmental assessment is available.
    recommendation_humidity = (
        current_humidity if weather_available else None
    )
    recommendation_rainfall = (
        current_rainfall if weather_available else None
    )
    recommendation_soil = (
        current_soil if weather_available else None
    )

    # ========================================================
    # RECOMMENDATION ENGINE
    # ========================================================

    if pred.get("uncertain"):

        rec = (
            "Prediction confidence is below the "
            "configured threshold. Capture a clearer "
            "leaf image or request field-expert "
            "verification before acting on the diagnosis."
        )

        recommendation_plan = {

            "summary":
                rec,

            "priority":
                "VERIFY DIAGNOSIS",

            "disease":
                pred["disease"],

            "crop":
                selected_crop,

            "growth_stage":
                growth_stage,

            "disease_category":
                "uncertain",

            "immediate_actions": [

                "Capture a clearer leaf image.",

                "Inspect the affected plant and nearby plants.",

                "Request field-expert verification before treatment.",
            ],

            "organic_management": [

                "Maintain basic field sanitation.",

                "Remove clearly dead material where practical.",

                "Avoid disease-specific treatment until the diagnosis is verified.",
            ],

            "avoid": [

                "Avoid applying disease-specific products solely from an uncertain image prediction.",
            ],

            "weather_context": {

                "humidity":
                    recommendation_humidity,

                "rainfall":
                    recommendation_rainfall,

                "soil_moisture":
                    recommendation_soil,

                "message": (
                    "Weather data is unavailable or incomplete. "
                    "Environmental disease pressure has not been assessed."
                    if not weather_available
                    else
                    "Weather information can support monitoring, "
                    "but does not confirm the disease diagnosis."
                ),
            },

            "growth_stage_note":
                None,

            "organic_disclaimer":
                (
                    "Organic-management options are "
                    "decision-support guidance, not a "
                    "product prescription."
                ),

            "recommendation_text":
                rec,

            "engine":
                "uncertainty_safety_rule",
        }

    else:

        recommendation_plan = build_recommendation(

            disease=
                pred["disease"],

            severity_class=
                severity_class,

            risk_level=
                level,

            humidity=
                recommendation_humidity,

            rainfall=
                recommendation_rainfall,

            soil_moisture=
                recommendation_soil,

            crop=
                selected_crop,

            growth_stage=
                growth_stage,

            confidence=
                pred["confidence"],

            severity_percent=
                severity_percent,

            risk_score=
                score,
        )

        rec = recommendation_plan[
            "recommendation_text"
        ]

    # ========================================================
    # WEATHER-AWARE 5-DAY RISK FORECAST
    # ========================================================

    risk_forecast = []

    for day in weather_days:
        temperature = day.get("temperature_max")
        humidity = day.get("humidity")
        rainfall = day.get("rainfall")
        soil = day.get("soil_moisture")

        day_weather_available = all(
            value is not None
            for value in (
                temperature,
                humidity,
                rainfall,
                soil,
            )
        )

        if day_weather_available:
            day_dpi = dpi_from_environment(
                temperature,
                humidity,
                rainfall,
                soil,
            )

            day_score, day_level = fuse_risk(
                pred["confidence"],
                severity_percent,
                day_dpi,
            )
        else:
            day_dpi = None
            day_score = None
            day_level = None

        risk_forecast.append({
            "date": day.get("date"),
            "temperature_max": temperature,
            "temperature_min": day.get("temperature_min"),
            "humidity": humidity,
            "rainfall": rainfall,
            "rain_probability": day.get("rain_probability"),
            "soil_moisture": soil,
            "weather": day.get("weather"),
            "weather_code": day.get("weather_code"),
            "environmental_pressure": day_dpi,
            "risk_score": day_score,
            "risk_level": day_level,
            "environmental_assessment_status": (
                "ASSESSED" if day_weather_available else "NOT_ASSESSED"
            ),
        })

    # ========================================================
    # EXPLANATION
    # ========================================================

    explanation = {

        "disease_confidence_percent":
            round(
                pred["confidence"] * 100,
                2
            ),

        "estimated_affected_area_percent":
            severity_percent,

        "provisional_severity_band":
            severity_class,

        "environmental_pressure_index":
            dpi,

        "environment": {

            "temperature":
                current_temperature,

            "humidity":
                current_humidity,

            "rainfall":
                current_rainfall,

            "soil_moisture":
                current_soil,
        },

        "location": {

            "name":
                farmer["location_name"],

            "latitude":
                farmer["latitude"],

            "longitude":
                farmer["longitude"],
        },

        "growth_stage":
            growth_stage,

        "forecast_type": (
            "weather_aware_five_day_projection"
            if any(
                day.get("environmental_assessment_status") == "ASSESSED"
                for day in risk_forecast
            )
            else "not_assessed_weather_unavailable"
        ),

        "forecast_note": (
            "The five-day projection combines disease-model "
            "confidence and prototype severity with forecast "
            "environmental conditions. It is a decision-support "
            "projection, not a validated disease outbreak forecast."
            if any(
                day.get("environmental_assessment_status") == "ASSESSED"
                for day in risk_forecast
            )
            else
            "Five-day environmental risk projection was not assessed "
            "because complete weather data is unavailable."
        ),

        "severity_note":
            (
                "Affected area and severity band are "
                "prototype estimates from a color heuristic "
                "and require crop-specific lesion-segmentation "
                "validation for research claims."
            ),

        "soil_moisture_note":
            (
                "Soil moisture is a weather-model proxy and "
                "should not be interpreted as a direct field "
                "sensor measurement."
            ),

        "recommendation_note":
            (
                "Recommendations are rule-based decision-support "
                "guidance derived from disease category, severity, "
                "risk level, weather conditions and crop context."
            ),

        "risk_decomposition":
            risk_breakdown,
    }

    # ========================================================
    # GRAD-CAM
    # ========================================================

    try:

        gradcam = make_gradcam_placeholder(
            img
        )

    except Exception:

        gradcam = None

    # ========================================================
    # SAVE ANALYSIS
    # ========================================================

    with get_db() as db:

        db.execute(
            """
            INSERT INTO analyses(
                farmer_id,
                field_id,
                crop,
                growth_stage,
                temperature,
                humidity,
                rainfall,
                soil_moisture,
                disease,
                confidence,
                uncertain,
                severity,
                severity_class,
                dpi,
                risk_score,
                risk_level,
                recommendation
            )
            VALUES(
                ?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?
            )
            """,
            (

                farmer["id"],

                selected_field,

                selected_crop,

                growth_stage,

                current_temperature,

                current_humidity,

                current_rainfall,

                current_soil,

                pred["disease"],

                pred["confidence"],

                int(
                    pred["uncertain"]
                ),

                severity_percent,

                severity_class,

                dpi,

                score,

                level,

                rec,
            )
        )

    # ========================================================
    # FINAL RESPONSE
    # ========================================================

    return {

        "disease":
            pred["disease"],

        "confidence":
            pred["confidence"],

        "uncertain":
            pred["uncertain"],

        "crop_mismatch":
            False,

        "severity_percent":
            severity_percent,

        "severity_class":
            severity_class,

        "dpi":
            dpi,

        "risk_score":
            score,

        "risk_level":
            level,

        "weather_available": weather_available,

        "environmental_assessment_status": (
            "ASSESSED" if weather_available else "NOT_ASSESSED"
        ),

        "risk_decomposition":
            risk_breakdown,

        "model_version":
            pred.get("model_version"),

        "current_weather": {

            "available": weather_available,

            "temperature":
                current_temperature,

            "humidity":
                current_humidity,

            "rainfall":
                current_rainfall,

            "soil_moisture":
                current_soil,
        },

        "weather_forecast":
            risk_forecast,

        "forecast":
            risk_forecast,

        "recommendation":
            rec,

        "recommendation_plan":
            recommendation_plan,

        "explanation":
            explanation,

        "image_quality":
            quality,

        "gradcam":
            gradcam,
    }


# ============================================================
# ANALYSIS HISTORY
# ============================================================

@app.get("/api/history")
def history(
    authorization: Optional[str] = Header(
        default=None
    )
):

    farmer = get_farmer_from_token(
        authorization
    )

    with get_db() as db:

        rows = db.execute(
            """
            SELECT
                id,
                field_id,
                crop,
                growth_stage,
                temperature,
                humidity,
                rainfall,
                soil_moisture,
                disease,
                confidence,
                uncertain,
                severity,
                severity_class,
                dpi,
                risk_score,
                risk_level,
                recommendation,
                created_at
            FROM analyses
            WHERE farmer_id = ?
            ORDER BY id DESC
            LIMIT 20
            """,
            (
                farmer["id"],
            )
        ).fetchall()

    return {

        "history": [

            dict(row)

            for row in rows
        ]
    }

# ============================================================
# REGIONAL-LANGUAGE RECOMMENDATIONS
# ============================================================

@app.post("/api/recommendation/translate")
def translate_recommendation(
    payload: dict,
    authorization: Optional[str] = Header(default=None),
):
    # Reuse SmartCrop AI's existing farmer authentication.
    get_farmer_from_token(authorization)

    language = payload.get("language")
    plan = payload.get("recommendation_plan")

    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail="Unsupported language. Use en, kn, te, or ta.",
        )

    if not isinstance(plan, dict):
        raise HTTPException(
            status_code=400,
            detail="recommendation_plan must be an object.",
        )

    try:
        translated_plan = translate_recommendation_plan(
            plan,
            language,
        )
    except RuntimeError as exc:
        # Do not expose API keys or internal provider details.
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    return {
        "language": language,
        "language_name": SUPPORTED_LANGUAGES[language],
        "original_plan": plan,
        "translated_plan": translated_plan,
    }

