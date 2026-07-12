# ============================================================
# schemas/loan_schema.py
# Pydantic Models for Request & Response Validation
# ============================================================

from pydantic import BaseModel, Field, validator
from typing   import Optional, Dict
from enum     import Enum


# ── Enums for fixed-value fields ────────────────────────────

class HomeOwnership(str, Enum):
    OWN      = "own"
    MORTGAGE = "mortgage"
    RENT     = "rent"
    OTHER    = "other"

class VerificationStatus(str, Enum):
    VERIFIED        = "verified"
    SOURCE_VERIFIED = "source verified"
    NOT_VERIFIED    = "not verified"

class ApplicationType(str, Enum):
    INDIVIDUAL = "individual"
    JOINT_APP  = "joint app"

class LoanTerm(int, Enum):
    MONTHS_36 = 36
    MONTHS_60 = 60

class LoanGrade(str, Enum):
    A = "a"
    B = "b"
    C = "c"
    D = "d"
    E = "e"
    F = "f"
    G = "g"

class RiskBand(str, Enum):
    GREEN = "GREEN"
    AMBER = "AMBER"
    RED   = "RED"


# ── Loan Application Request Schema ─────────────────────────

class LoanApplicationRequest(BaseModel):
    """
    Input schema for a new MSME loan application.
    All fields match the cleaned/encoded feature set.
    """

    # Personal & Employment
    annual_income       : float = Field(..., gt=0,      description="Annual income in INR/USD",        example=45000)
    emp_length          : float = Field(..., ge=0, le=10,description="Years at current employer (0–10)",example=3)
    application_type    : ApplicationType = Field(...,   description="Individual or Joint",             example="individual")

    # Loan Details
    loan_amount         : float = Field(..., gt=0,       description="Principal loan amount",           example=15000)
    term                : LoanTerm = Field(...,           description="Loan tenure: 36 or 60 months",   example=36)
    int_rate            : float = Field(..., gt=0, lt=100,description="Interest rate (%)",              example=14.5)
    installment         : float = Field(..., gt=0,        description="Monthly EMI amount",             example=350.0)
    purpose             : str   = Field(...,              description="Reason for loan",                example="car")

    # Credit Profile
    grade               : LoanGrade = Field(...,          description="Loan grade (a–g)",               example="c")
    sub_grade           : str  = Field(...,               description="Sub-grade (e.g. c4)",            example="c4")
    home_ownership      : HomeOwnership = Field(...,      description="Home ownership status",          example="rent")
    verification_status : VerificationStatus = Field(..., description="Document verification status",  example="source verified")

    # Financial Ratios
    dti                 : float = Field(..., ge=0,        description="Debt-to-income ratio (%)",       example=18.5)
    total_acc           : int   = Field(..., ge=0,        description="Total credit accounts",          example=8)
    total_payment       : float = Field(..., ge=0,        description="Total amount paid so far",       example=5000)

    # Location
    address_state       : str   = Field(...,              description="State code (e.g. CA, GA)",       example="CA")

    # Date-derived features (computed externally or passed directly)
    loan_age_months         : Optional[float] = Field(0,  description="Age of loan in months")
    issue_month             : Optional[int]   = Field(1,  description="Month loan was issued (1–12)")
    issue_year              : Optional[int]   = Field(2021,description="Year loan was issued")
    days_since_last_payment : Optional[float] = Field(30, description="Days since last payment")
    days_since_credit_pull  : Optional[float] = Field(30, description="Days since credit was pulled")
    days_to_next_payment    : Optional[float] = Field(15, description="Days until next payment due")

    @validator('sub_grade')
    def validate_sub_grade(cls, v):
        valid = [f"{g}{n}" for g in 'abcdefg' for n in range(1, 6)]
        if v.lower() not in valid:
            raise ValueError(f"sub_grade must be one of {valid[:5]}... etc.")
        return v.lower()

    @validator('address_state')
    def validate_state(cls, v):
        return v.upper()

    class Config:
        use_enum_values = True
        json_schema_extra = {
            "example": {
                "annual_income"           : 45000,
                "emp_length"              : 3,
                "application_type"        : "individual",
                "loan_amount"             : 15000,
                "term"                    : 36,
                "int_rate"                : 14.5,
                "installment"             : 350.0,
                "purpose"                 : "car",
                "grade"                   : "c",
                "sub_grade"               : "c4",
                "home_ownership"          : "rent",
                "verification_status"     : "source verified",
                "dti"                     : 18.5,
                "total_acc"               : 8,
                "total_payment"           : 5000,
                "address_state"           : "CA",
                "loan_age_months"         : 12,
                "issue_month"             : 3,
                "issue_year"              : 2021,
                "days_since_last_payment" : 90,
                "days_since_credit_pull"  : 30,
                "days_to_next_payment"    : 15
            }
        }


# ── Financial Health Card Schema ─────────────────────────────

class HealthCardScores(BaseModel):
    liquidity   : float = Field(..., description="Liquidity score (0–100)")
    solvency    : float = Field(..., description="Solvency score (0–100)")
    growth      : float = Field(..., description="Growth score (0–100)")
    compliance  : float = Field(..., description="Compliance score (0–100)")
    repayment   : float = Field(..., description="Repayment score (0–100)")
    overall     : int   = Field(..., description="Overall score (0–1000)")


# ── SHAP Explanation Schema ──────────────────────────────────

class ShapExplanation(BaseModel):
    top_risk_factors   : Dict[str, float] = Field(..., description="Top features increasing risk")
    top_safety_factors : Dict[str, float] = Field(..., description="Top features decreasing risk")
    base_value         : float            = Field(..., description="SHAP base value")


# ```python
# ── Prediction Response Schema ───────────────────────────────

class LoanPredictionResponse(BaseModel):
    """
    Full prediction response returned by the API.
    """
    # Core Decision
    prediction          : int         = Field(..., description="0 = Good Loan, 1 = Bad Loan")
    decision            : str         = Field(..., description="APPROVE / REJECT / MANUAL REVIEW")
    risk_band           : RiskBand    = Field(..., description="GREEN / AMBER / RED")
    default_probability : float       = Field(..., description="Probability of default (0–1)")
    confidence          : float       = Field(..., description="Model confidence score (%)")

    # Financial Health Card
    health_card         : HealthCardScores

    # SHAP Explanation
    explanation         : ShapExplanation

    # Loan Offer (if approved)
    recommended_amount  : Optional[float] = Field(None, description="Recommended loan amount")
    recommended_rate    : Optional[float] = Field(None, description="Recommended interest rate (%)")
    recommended_term    : Optional[int]   = Field(None, description="Recommended tenure (months)")

    # Metadata
    model_name          : str  = Field(..., description="Model used for prediction")
    model_auc           : float= Field(..., description="Model AUC-ROC score")
    request_id          : str  = Field(..., description="Unique request ID")
    timestamp           : str  = Field(..., description="Prediction timestamp")


# ── Batch Prediction Schema ──────────────────────────────────

class BatchLoanRequest(BaseModel):
    applications : list[LoanApplicationRequest] = Field(
        ...,
        description = "List of loan applications (max 100)",
        max_items   = 100
    )

class BatchPredictionResult(BaseModel):
    index               : int
    prediction          : int
    decision            : str
    risk_band           : str
    default_probability : float
    overall_health_score: int

class BatchLoanResponse(BaseModel):
    total_applications  : int
    approved            : int
    rejected            : int
    manual_review       : int
    results             : list[BatchPredictionResult]
    processing_time_ms  : float


# ── Health Check Schema ──────────────────────────────────────

class HealthCheckResponse(BaseModel):
    status          : str
    model_loaded    : bool
    scaler_loaded   : bool
    model_name      : str
    model_auc       : float
    total_features  : int
    api_version     : str
