"""
SmartCrop AI
Crop Health Recommendation Engine

This module converts:
    - disease classification
    - model confidence
    - provisional severity
    - environmental pressure
    - weather conditions
    - growth stage

into structured decision-support guidance.

IMPORTANT:
This is a decision-support / educational recommendation engine.
It is NOT a substitute for field inspection or an agronomist.

Organic recommendations intentionally focus on:
    - sanitation
    - crop hygiene
    - airflow
    - irrigation management
    - removal of heavily affected material
    - biological / organic-compatible approaches

Exact commercial product, formulation, dose and spray interval should
always follow the locally registered product label and applicable
organic-certification rules.
"""

from typing import Any, Dict, List, Optional


# ============================================================
# DISEASE KNOWLEDGE BASE
# ============================================================

DISEASE_PROFILES: Dict[str, Dict[str, Any]] = {

    # --------------------------------------------------------
    # RICE
    # --------------------------------------------------------

    "Rice Bacterial Leaf Blight": {
        "crop": "rice",
        "category": "bacterial",
        "organic": [
            "Remove and safely dispose of severely affected leaves where practical.",
            "Maintain field sanitation and reduce unnecessary leaf injury.",
            "Avoid excessive nitrogen fertilization because lush growth can increase disease pressure.",
            "Use clean planting material and prefer locally recommended resistant varieties.",
            "Avoid unnecessary overhead irrigation and prolonged leaf wetness where irrigation can be managed."
        ],
        "actions": [
            "Inspect surrounding plants for expanding leaf symptoms.",
            "Review nitrogen and irrigation management.",
            "Maintain field sanitation.",
            "Monitor nearby plants during the next few days."
        ],
        "avoid": [
            "Avoid excessive nitrogen application.",
            "Avoid unnecessary mechanical injury to leaves.",
            "Avoid moving contaminated plant material between fields."
        ],
    },

    "Rice Leaf Blast": {
        "crop": "rice",
        "category": "fungal",
        "organic": [
            "Remove heavily affected plant material where practical.",
            "Improve field airflow and avoid unnecessarily prolonged leaf wetness.",
            "Use balanced nutrition and avoid excessive nitrogen.",
            "Use clean seed and locally recommended resistant or tolerant varieties.",
            "Biological or organic-compatible disease-management products may be considered only when locally registered and label-approved."
        ],
        "actions": [
            "Inspect neighboring plants for new blast lesions.",
            "Pay particular attention to humid and wet conditions.",
            "Review nitrogen and irrigation practices.",
            "Increase scouting frequency while environmental pressure remains high."
        ],
        "avoid": [
            "Avoid excessive nitrogen.",
            "Avoid practices that unnecessarily prolong leaf wetness.",
        ],
    },

    # --------------------------------------------------------
    # MAIZE
    # --------------------------------------------------------

    "Maize Blight": {
        "crop": "maize",
        "category": "fungal",
        "organic": [
            "Remove heavily diseased leaves where practical without spreading plant debris.",
            "Maintain field sanitation and manage crop residues appropriately.",
            "Improve airflow through appropriate plant spacing and field management.",
            "Use clean seed and locally recommended tolerant or resistant varieties.",
            "Biological or organic-compatible fungicidal products may be considered if locally registered and appropriate for the crop."
        ],
        "actions": [
            "Scout neighboring maize plants for expanding lesions.",
            "Prioritize field inspection if humidity and rainfall remain high.",
            "Review crop residue and field sanitation.",
            "Monitor disease spread over the next several days."
        ],
        "avoid": [
            "Avoid unnecessary prolonged leaf wetness.",
            "Avoid moving infected plant debris between fields.",
        ],
    },

    "Maize Common Rust": {
        "crop": "maize",
        "category": "fungal",
        "organic": [
            "Remove severely affected leaves where practical.",
            "Maintain balanced crop nutrition.",
            "Use locally recommended resistant or tolerant maize varieties.",
            "Maintain good field hygiene and remove unnecessary disease-bearing debris.",
            "Consider biological or organic-compatible disease-management products only where locally registered."
        ],
        "actions": [
            "Inspect upper and middle leaves for increasing rust pustules.",
            "Increase scouting frequency under humid conditions.",
            "Prioritize monitoring during periods of repeated rainfall."
        ],
        "avoid": [
            "Avoid excessive nitrogen.",
            "Avoid unnecessary movement of contaminated plant material."
        ],
    },

    "Maize Gray Leaf Spot": {
        "crop": "maize",
        "category": "fungal",
        "organic": [
            "Maintain crop-residue management and field sanitation.",
            "Improve airflow where practical.",
            "Use resistant or tolerant varieties recommended for the region.",
            "Maintain balanced nutrition rather than excessive nitrogen.",
            "Organic-compatible biological disease-management products may be considered when locally approved."
        ],
        "actions": [
            "Inspect lower leaves first and monitor upward disease movement.",
            "Increase scouting after humid or rainy periods.",
            "Review residue management and field airflow."
        ],
        "avoid": [
            "Avoid excessive nitrogen.",
            "Avoid unnecessary prolonged leaf wetness."
        ],
    },

    # --------------------------------------------------------
    # TOMATO
    # --------------------------------------------------------

    "Tomato Bacterial Spot": {
        "crop": "tomato",
        "category": "bacterial",
        "organic": [
            "Remove severely affected leaves and fruit where practical.",
            "Use clean planting material.",
            "Keep foliage as dry as practical during irrigation.",
            "Improve spacing and airflow around plants.",
            "Use locally approved biological or organic-compatible products when appropriate."
        ],
        "actions": [
            "Inspect nearby leaves and fruit for new spots.",
            "Improve airflow and sanitation.",
            "Avoid unnecessary overhead irrigation."
        ],
        "avoid": [
            "Avoid working with wet foliage because this can spread plant pathogens.",
            "Avoid moving contaminated plant debris between healthy plants."
        ],
    },

    "Tomato Early Blight": {
        "crop": "tomato",
        "category": "fungal",
        "organic": [
            "Remove lower infected leaves where practical.",
            "Remove fallen infected plant debris.",
            "Use mulch to reduce splash of contaminated soil onto foliage.",
            "Maintain good plant spacing and airflow.",
            "Use locally approved biological or organic-compatible disease-management products when appropriate."
        ],
        "actions": [
            "Inspect lower leaves and nearby plants.",
            "Improve airflow around the canopy.",
            "Reduce leaf wetness where irrigation can be controlled.",
            "Increase scouting after rainfall."
        ],
        "avoid": [
            "Avoid prolonged leaf wetness.",
            "Avoid leaving heavily infected plant debris in the field."
        ],
    },

    "Tomato Late Blight": {
        "crop": "tomato",
        "category": "fungal",
        "organic": [
            "Immediately inspect surrounding plants when symptoms are suspected.",
            "Remove severely infected tissue where practical and dispose of it safely.",
            "Improve airflow and reduce prolonged foliage wetness.",
            "Use healthy planting material and maintain field sanitation.",
            "Organic-compatible disease-management products should only be used according to locally approved labels."
        ],
        "actions": [
            "Increase scouting frequency.",
            "Inspect neighboring plants for rapidly expanding symptoms.",
            "Prioritize management when humidity and rainfall are elevated.",
            "Seek local agronomic verification if symptoms are spreading rapidly."
        ],
        "avoid": [
            "Avoid prolonged leaf wetness.",
            "Avoid moving infected plant material through the field."
        ],
    },

    "Tomato Leaf Mold": {
        "crop": "tomato",
        "category": "fungal",
        "organic": [
            "Remove severely affected leaves where practical.",
            "Increase ventilation around the canopy.",
            "Avoid unnecessary overhead irrigation.",
            "Maintain greenhouse or protected-crop humidity control where applicable.",
            "Use locally approved biological or organic-compatible products if required."
        ],
        "actions": [
            "Inspect the underside of leaves.",
            "Improve airflow.",
            "Reduce excessive humidity where possible."
        ],
        "avoid": [
            "Avoid prolonged high humidity.",
            "Avoid dense unmanaged foliage."
        ],
    },

    "Tomato Septoria Leaf Spot": {
        "crop": "tomato",
        "category": "fungal",
        "organic": [
            "Remove lower infected leaves where practical.",
            "Remove fallen infected debris.",
            "Use mulch to reduce soil splash.",
            "Maintain good spacing and airflow.",
            "Use locally approved biological or organic-compatible products where appropriate."
        ],
        "actions": [
            "Inspect lower foliage regularly.",
            "Increase monitoring after rainfall.",
            "Maintain sanitation around the crop."
        ],
        "avoid": [
            "Avoid overhead irrigation when alternatives are available.",
            "Avoid leaving infected debris around plants."
        ],
    },

    "Tomato Spider Mites Two Spotted Spider Mite": {
        "crop": "tomato",
        "category": "mite",
        "organic": [
            "Inspect leaf undersides for mites and webbing.",
            "Remove severely affected leaves where practical.",
            "Maintain adequate plant-water status and avoid unnecessary plant stress.",
            "Conserve beneficial predatory mites and insects.",
            "Neem or other botanical products should only be used if locally approved and according to the product label."
        ],
        "actions": [
            "Inspect the undersides of leaves.",
            "Check neighboring plants for similar symptoms.",
            "Monitor pest population before choosing a treatment."
        ],
        "avoid": [
            "Avoid unnecessary broad-spectrum pesticide use that can harm beneficial organisms.",
            "Avoid allowing plants to remain severely water-stressed."
        ],
    },

    "Tomato Target Spot": {
        "crop": "tomato",
        "category": "fungal",
        "organic": [
            "Remove severely affected leaves where practical.",
            "Maintain field sanitation.",
            "Improve airflow and canopy ventilation.",
            "Reduce prolonged leaf wetness.",
            "Use locally approved biological or organic-compatible products when appropriate."
        ],
        "actions": [
            "Inspect nearby plants for expanding lesions.",
            "Increase scouting after rainfall.",
            "Maintain good canopy airflow."
        ],
        "avoid": [
            "Avoid prolonged leaf wetness.",
            "Avoid leaving infected debris near healthy plants."
        ],
    },

    "Tomato Mosaic Virus": {
        "crop": "tomato",
        "category": "viral",
        "organic": [
            "Remove and isolate severely symptomatic plants where practical.",
            "Use clean seed and healthy planting material.",
            "Sanitize tools and hands after handling symptomatic plants.",
            "Control weeds that may act as alternate hosts where appropriate.",
            "Use resistant or tolerant varieties recommended locally."
        ],
        "actions": [
            "Inspect neighboring plants for similar mosaic symptoms.",
            "Review sanitation practices.",
            "Consider field-expert confirmation because viral symptoms can resemble nutritional or environmental stress."
        ],
        "avoid": [
            "Do not rely on fungicides for a suspected viral disease.",
            "Avoid moving plant sap between plants through contaminated tools."
        ],
    },

    "Tomato Yellow Leaf Curl Virus": {
        "crop": "tomato",
        "category": "viral",
        "organic": [
            "Remove severely symptomatic plants where practical.",
            "Use healthy planting material.",
            "Manage weeds around the crop.",
            "Conserve beneficial insects.",
            "Monitor and manage insect vectors using locally approved integrated pest-management approaches."
        ],
        "actions": [
            "Inspect plants for characteristic curling and yellowing.",
            "Check for insect-vector activity.",
            "Increase scouting of neighboring plants.",
            "Seek field verification if symptoms are spreading."
        ],
        "avoid": [
            "Do not treat a suspected virus as a fungal disease.",
            "Avoid unnecessary broad-spectrum pesticide applications."
        ],
    },

    # --------------------------------------------------------
    # GRAPE
    # --------------------------------------------------------

    "Grape Black Rot": {
        "crop": "grape",
        "category": "fungal",
        "organic": [
            "Remove infected berries and diseased plant material where practical.",
            "Maintain vineyard sanitation.",
            "Improve canopy airflow and reduce prolonged wetness.",
            "Remove mummified fruit where practical.",
            "Use locally approved biological or organic-compatible products where appropriate."
        ],
        "actions": [
            "Inspect clusters and leaves for expanding symptoms.",
            "Increase monitoring after wet weather.",
            "Maintain vineyard sanitation."
        ],
        "avoid": [
            "Avoid leaving infected fruit or debris near the vines.",
            "Avoid unnecessary prolonged canopy wetness."
        ],
    },

    "Grape Esca Black Measles": {
        "crop": "grape",
        "category": "fungal",
        "organic": [
            "Remove severely affected plant material according to local vineyard-management guidance.",
            "Maintain pruning and sanitation practices.",
            "Use clean planting material.",
            "Monitor affected vines over time because trunk diseases can be difficult to manage after establishment.",
            "Seek local viticulture expertise for persistent or severe symptoms."
        ],
        "actions": [
            "Inspect affected vines individually.",
            "Mark symptomatic vines for follow-up.",
            "Review pruning and sanitation practices.",
            "Seek expert confirmation before removing valuable mature vines."
        ],
        "avoid": [
            "Avoid spreading contaminated pruning tools between vines.",
            "Avoid assuming every leaf symptom is caused by the same disease."
        ],
    },

    "Grape Leaf Blight": {
        "crop": "grape",
        "category": "fungal",
        "organic": [
            "Remove heavily affected leaves where practical.",
            "Maintain vineyard sanitation.",
            "Improve canopy airflow.",
            "Reduce prolonged foliage wetness.",
            "Use locally approved biological or organic-compatible products where appropriate."
        ],
        "actions": [
            "Inspect nearby leaves and clusters.",
            "Increase scouting after rainfall.",
            "Improve canopy ventilation."
        ],
        "avoid": [
            "Avoid excessive canopy density.",
            "Avoid unnecessary prolonged leaf wetness."
        ],
    },

    # --------------------------------------------------------
    # PEPPER
    # --------------------------------------------------------

    "Pepper Bacterial Spot": {
        "crop": "pepper",
        "category": "bacterial",
        "organic": [
            "Remove severely infected leaves and fruit where practical.",
            "Use clean seed and healthy transplants.",
            "Improve spacing and airflow.",
            "Avoid unnecessary overhead irrigation.",
            "Use locally approved biological or organic-compatible products where appropriate."
        ],
        "actions": [
            "Inspect neighboring pepper plants.",
            "Increase sanitation.",
            "Monitor after rainfall or high-humidity periods."
        ],
        "avoid": [
            "Avoid handling wet plants unnecessarily.",
            "Avoid spreading contaminated plant debris."
        ],
    },

    # --------------------------------------------------------
    # POTATO
    # --------------------------------------------------------

    "Potato Early Blight": {
        "crop": "potato",
        "category": "fungal",
        "organic": [
            "Remove severely infected foliage where practical.",
            "Maintain field sanitation and manage infected crop residue.",
            "Maintain balanced plant nutrition.",
            "Use healthy seed tubers.",
            "Use locally approved biological or organic-compatible disease-management products where appropriate."
        ],
        "actions": [
            "Inspect lower and older leaves.",
            "Monitor neighboring plants.",
            "Increase scouting after wet weather."
        ],
        "avoid": [
            "Avoid unnecessary prolonged foliage wetness.",
            "Avoid leaving heavily infected residue unmanaged."
        ],
    },

    "Potato Late Blight": {
        "crop": "potato",
        "category": "fungal",
        "organic": [
            "Inspect the field promptly because symptoms can spread rapidly under favorable conditions.",
            "Remove severely infected material where practical and manage it safely.",
            "Improve airflow and reduce prolonged foliage wetness.",
            "Use healthy planting material.",
            "Organic-compatible products should only be used where locally registered and according to the label."
        ],
        "actions": [
            "Increase field scouting frequency.",
            "Inspect neighboring plants immediately.",
            "Pay particular attention after rainy and humid conditions.",
            "Seek field-expert confirmation when disease is spreading rapidly."
        ],
        "avoid": [
            "Avoid prolonged foliage wetness.",
            "Avoid transporting infected plant material through healthy areas."
        ],
    },
}


# ============================================================
# GENERIC CROP GUIDANCE
# ============================================================

GENERIC_CROP_ACTIONS: Dict[str, List[str]] = {

    "rice": [
        "Inspect nearby plants before making field-level treatment decisions.",
        "Review irrigation and nutrient management.",
        "Maintain field sanitation.",
    ],

    "maize": [
        "Inspect neighboring maize plants for disease spread.",
        "Review crop residue and field sanitation.",
        "Maintain balanced crop nutrition.",
    ],

    "tomato": [
        "Inspect nearby plants and both leaf surfaces.",
        "Maintain canopy airflow and sanitation.",
        "Review irrigation practices.",
    ],

    "grape": [
        "Inspect individual vines and clusters.",
        "Maintain vineyard sanitation.",
        "Review canopy density and airflow.",
    ],

    "potato": [
        "Inspect neighboring plants and lower foliage.",
        "Maintain field sanitation.",
        "Monitor disease closely after wet weather.",
    ],

    "pepper": [
        "Inspect neighboring plants.",
        "Maintain sanitation and airflow.",
        "Review irrigation practices.",
    ],
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def _normalise(value: Optional[str]) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _severity_score(severity_class: Optional[str]) -> int:
    value = _normalise(severity_class)

    mapping = {
        "mild": 1,
        "moderate": 2,
        "severe": 3,
        "critical": 4,
    }

    return mapping.get(value, 1)


def _risk_score(risk_level: Optional[str]) -> int:
    value = _normalise(risk_level)

    mapping = {
        "low": 1,
        "moderate": 2,
        "elevated": 2,
        "high": 3,
        "critical": 4,
    }

    return mapping.get(value, 1)


def _environment_flags(
    humidity: Optional[float],
    rainfall: Optional[float],
    soil_moisture: Optional[float],
) -> Dict[str, bool]:

    humidity_value = float(humidity or 0)
    rainfall_value = float(rainfall or 0)
    soil_value = float(soil_moisture or 0)

    return {
        "high_humidity": humidity_value >= 75,
        "very_high_humidity": humidity_value >= 85,
        "rainy": rainfall_value >= 5,
        "heavy_rain": rainfall_value >= 15,
        "wet_soil": soil_value >= 65,
    }


def _priority_label(
    severity_class: str,
    risk_level: str,
) -> str:

    severity = _severity_score(severity_class)
    risk = _risk_score(risk_level)

    combined = severity + risk

    if combined >= 7:
        return "URGENT FIELD INSPECTION"

    if combined >= 5:
        return "PRIORITY MONITORING"

    if combined >= 3:
        return "MONITOR CLOSELY"

    return "ROUTINE MONITORING"


def _build_weather_message(
    humidity: Optional[float],
    rainfall: Optional[float],
    soil_moisture: Optional[float],
) -> str:

    flags = _environment_flags(
        humidity,
        rainfall,
        soil_moisture,
    )

    messages = []

    if flags["very_high_humidity"]:
        messages.append(
            "High humidity is increasing environmental disease pressure."
        )
    elif flags["high_humidity"]:
        messages.append(
            "Humidity is favorable for moisture-related disease pressure."
        )

    if flags["heavy_rain"]:
        messages.append(
            "Recent rainfall is high, so field scouting should be increased."
        )
    elif flags["rainy"]:
        messages.append(
            "Rainfall may increase leaf-wetness and disease pressure."
        )

    if flags["wet_soil"]:
        messages.append(
            "The weather-model soil-moisture proxy indicates relatively wet conditions."
        )

    if not messages:
        messages.append(
            "Current weather indicators do not show unusually strong moisture pressure."
        )

    return " ".join(messages)


def _select_immediate_actions(
    profile: Dict[str, Any],
    severity_class: str,
    risk_level: str,
    humidity: Optional[float],
    rainfall: Optional[float],
    soil_moisture: Optional[float],
) -> List[str]:

    actions: List[str] = []

    priority = _priority_label(
        severity_class,
        risk_level,
    )

    if priority == "URGENT FIELD INSPECTION":
        actions.append(
            "Inspect the affected area and surrounding plants as soon as practical."
        )

    elif priority == "PRIORITY MONITORING":
        actions.append(
            "Prioritize field inspection and check neighboring plants for spread."
        )

    elif priority == "MONITOR CLOSELY":
        actions.append(
            "Increase scouting frequency and monitor nearby plants."
        )

    else:
        actions.append(
            "Continue routine crop scouting and monitor symptom progression."
        )

    flags = _environment_flags(
        humidity,
        rainfall,
        soil_moisture,
    )

    if (
        flags["high_humidity"]
        or flags["rainy"]
        or flags["wet_soil"]
    ):
        actions.append(
            "Because moisture-related environmental pressure is elevated, "
            "prioritize airflow, sanitation and avoidance of unnecessary leaf wetness."
        )

    actions.extend(
        profile.get(
            "actions",
            []
        )[:3]
    )

    # Remove duplicates while retaining order.
    result = []

    for action in actions:
        if action not in result:
            result.append(action)

    return result[:6]


# ============================================================
# MAIN STRUCTURED RECOMMENDATION
# ============================================================

def build_recommendation(
    disease: str,
    severity_class: str,
    risk_level: str,
    humidity: Optional[float] = None,
    rainfall: Optional[float] = None,
    soil_moisture: Optional[float] = None,
    crop: Optional[str] = None,
    growth_stage: Optional[str] = None,
    confidence: Optional[float] = None,
    severity_percent: Optional[float] = None,
    risk_score: Optional[float] = None,
) -> Dict[str, Any]:

    disease_key = str(disease or "").strip()

    profile = DISEASE_PROFILES.get(
        disease_key
    )

    # --------------------------------------------------------
    # UNKNOWN DISEASE FALLBACK
    # --------------------------------------------------------

    if profile is None:

        profile = {
            "crop": _normalise(crop),
            "category": "unknown",
            "organic": [
                "Maintain field sanitation.",
                "Remove severely affected plant material where practical.",
                "Maintain good crop airflow.",
                "Use healthy planting material.",
                "Seek local agricultural guidance before applying any treatment."
            ],
            "actions": [
                "Inspect neighboring plants.",
                "Monitor symptom progression.",
                "Confirm the diagnosis before selecting a disease-specific treatment."
            ],
            "avoid": [
                "Avoid applying disease-specific chemicals solely from an uncertain image diagnosis."
            ],
        }

    priority = _priority_label(
        severity_class,
        risk_level,
    )

    weather_message = _build_weather_message(
        humidity,
        rainfall,
        soil_moisture,
    )

    immediate_actions = _select_immediate_actions(
        profile,
        severity_class,
        risk_level,
        humidity,
        rainfall,
        soil_moisture,
    )

    organic_actions = list(
        profile.get(
            "organic",
            []
        )
    )

    avoid_actions = list(
        profile.get(
            "avoid",
            []
        )
    )

    # --------------------------------------------------------
    # WEATHER-SPECIFIC ADDITIONS
    # --------------------------------------------------------

    flags = _environment_flags(
        humidity,
        rainfall,
        soil_moisture,
    )

    if flags["very_high_humidity"]:
        organic_actions.insert(
            0,
            "Prioritize canopy ventilation and reduce unnecessary foliage wetness while humidity remains high."
        )

    if flags["heavy_rain"]:
        organic_actions.insert(
            0,
            "After heavy rainfall, inspect for new symptoms and avoid unnecessary movement through wet infected areas."
        )

    # --------------------------------------------------------
    # GROWTH-STAGE GUIDANCE
    # --------------------------------------------------------

    stage = _normalise(
        growth_stage
    )

    stage_note = None

    if stage:
        if stage in {
            "flowering",
            "fruiting",
            "reproductive",
        }:
            stage_note = (
                "The crop is in a reproductive stage, so protect healthy "
                "foliage and developing reproductive structures and verify "
                "any treatment is appropriate for this stage."
            )

        elif stage in {
            "seedling",
            "early growth",
            "vegetative",
        }:
            stage_note = (
                "The crop is in an early growth stage, so prioritize "
                "early detection, sanitation and prevention of disease spread."
            )

        else:
            stage_note = (
                "Consider the current crop growth stage when planning field management."
            )

    # --------------------------------------------------------
    # MAIN SUMMARY
    # --------------------------------------------------------

    if risk_level.upper() == "CRITICAL":
        summary = (
            f"{priority}: {disease}. "
            "Inspect the affected field area promptly and verify the diagnosis "
            "before taking treatment decisions."
        )

    elif risk_level.upper() == "HIGH":
        summary = (
            f"{priority}: {disease}. "
            "Increase scouting and begin disease-management measures supported "
            "by field observations."
        )

    elif risk_level.upper() in {"MODERATE", "ELEVATED"}:
        summary = (
            f"{priority}: {disease}. "
            "Monitor the affected area closely and address environmental and "
            "sanitation factors."
        )

    else:
        summary = (
            f"{priority}: {disease}. "
            "Continue routine scouting and preventive crop-health management."
        )

    # --------------------------------------------------------
    # ORGANIC PLAN DISCLAIMER
    # --------------------------------------------------------

    organic_disclaimer = (
        "Organic-management options are decision-support guidance, not a "
        "product prescription. Use only inputs permitted for the crop, "
        "location and applicable organic-certification system, and follow "
        "the locally registered product label."
    )

    # --------------------------------------------------------
    # COMPACT ACTION SENTENCE
    # --------------------------------------------------------

    recommendation_text = (
        f"{summary} "
        f"{immediate_actions[0]} "
        f"{weather_message}"
    )

    return {
        "summary": summary,

        "priority": priority,

        "disease": disease_key,

        "crop": crop,

        "growth_stage": growth_stage,

        "disease_category": profile.get(
            "category",
            "unknown"
        ),

        "confidence": confidence,

        "severity_percent": severity_percent,

        "severity_class": severity_class,

        "risk_score": risk_score,

        "risk_level": risk_level,

        "weather_context": {
            "humidity": humidity,
            "rainfall": rainfall,
            "soil_moisture": soil_moisture,
            "message": weather_message,
        },

        "immediate_actions": immediate_actions,

        "organic_management": organic_actions,

        "avoid": avoid_actions,

        "growth_stage_note": stage_note,

        "organic_disclaimer": organic_disclaimer,

        "recommendation_text": recommendation_text,

        "engine": "rule_based_crop_health_recommendation_v2",
    }


# ============================================================
# BACKWARD-COMPATIBLE FUNCTION
# ============================================================

def recommend(
    disease: str,
    severity_class: str,
    risk_level: str,
    humidity: Optional[float] = None,
    rainfall: Optional[float] = None,
    soil_moisture: Optional[float] = None,
    crop: Optional[str] = None,
    growth_stage: Optional[str] = None,
    confidence: Optional[float] = None,
    severity_percent: Optional[float] = None,
    risk_score: Optional[float] = None,
) -> str:
    """
    Backward-compatible recommendation function.

    Existing main.py can continue calling:

        recommend(
            disease,
            severity_class,
            risk_level,
            humidity
        )

    Newer code can use build_recommendation() to obtain the
    complete structured recommendation.
    """

    result = build_recommendation(
        disease=disease,
        severity_class=severity_class,
        risk_level=risk_level,
        humidity=humidity,
        rainfall=rainfall,
        soil_moisture=soil_moisture,
        crop=crop,
        growth_stage=growth_stage,
        confidence=confidence,
        severity_percent=severity_percent,
        risk_score=risk_score,
    )

    return result["recommendation_text"]