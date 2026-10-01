from typing import Optional, List, Dict, Any

from pydantic import BaseModel, Field


# ============================================================
# ENVIRONMENT
# ============================================================

class EnvironmentInput(BaseModel):

    temperature: float = Field(
        ...,
        ge=-20,
        le=60
    )

    humidity: float = Field(
        ...,
        ge=0,
        le=100
    )

    rainfall: float = Field(
        ...,
        ge=0
    )

    soil_moisture: float = Field(
        ...,
        ge=0,
        le=100
    )


# ============================================================
# FARMER REGISTRATION
# ============================================================

class FarmerRegister(BaseModel):

    full_name: str = Field(
        ...,
        min_length=2,
        max_length=100
    )

    email: str = Field(
        ...,
        min_length=5,
        max_length=150
    )

    password: str = Field(
        ...,
        min_length=6,
        max_length=100
    )

    farm_name: str = Field(
        ...,
        min_length=2,
        max_length=100
    )

    location: str = Field(
        ...,
        min_length=2,
        max_length=150
    )

    crop: str = Field(
        default="rice"
    )

    field_id: str = Field(
        default="FIELD-001"
    )


# ============================================================
# LOGIN
# ============================================================

class FarmerLogin(BaseModel):

    email: str

    password: str


# ============================================================
# ORGANIC MANAGEMENT
# ============================================================

class OrganicManagement(BaseModel):

    actions: List[str] = []

    avoid: List[str] = []

    disclaimer: Optional[str] = None


# ============================================================
# RECOMMENDATION
# ============================================================

class RecommendationResponse(BaseModel):

    summary: str

    priority: str

    disease: str

    crop: Optional[str] = None

    growth_stage: Optional[str] = None

    disease_category: Optional[str] = None

    immediate_actions: List[str] = []

    organic_management: List[str] = []

    avoid: List[str] = []

    weather_context: Dict[str, Any] = {}

    growth_stage_note: Optional[str] = None

    organic_disclaimer: Optional[str] = None

    recommendation_text: str

    engine: str


# ============================================================
# ANALYSIS RESPONSE
# ============================================================

class AnalysisResponse(BaseModel):

    disease: str

    confidence: float

    uncertain: bool

    severity_percent: Optional[float]

    severity_class: str

    dpi: Optional[float]

    risk_score: Optional[float]

    risk_level: str

    forecast: list

    recommendation: str

    recommendation_plan: Optional[RecommendationResponse] = None

    explanation: dict

    image_quality: dict

    gradcam: Optional[str] = None