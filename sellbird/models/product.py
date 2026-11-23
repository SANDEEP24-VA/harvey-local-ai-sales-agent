"""Product, ICP, and prospect matching data models for multi-product sales."""

from datetime import datetime, timezone
import json
from typing import Any

from pydantic import BaseModel, Field, field_validator


def _utcnow() -> datetime:
    """Naive UTC now (consistent with DB storage; avoids deprecated utcnow)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ProductCreate(BaseModel):
    name: str = Field(..., min_length=1, description="Product or offering name")
    sku: str = Field(default="", description="Unique stock keeping unit or code")
    category: str = Field(default="", description="Product category or vertical")
    description: str = Field(default="", description="Detailed product description")
    pricing_model: str = Field(default="", description="e.g. subscription, usage-based, one-time, tiered")
    pricing_details: str = Field(default="", description="Pricing tier details, starting prices, currency")
    features: list[str] = Field(default_factory=list, description="List of key product capabilities/features")
    benefits: list[str] = Field(default_factory=list, description="Core business outcomes and advantages")
    target_industries: list[str] = Field(default_factory=list, description="Target industry verticals")
    target_customer_sizes: list[str] = Field(default_factory=list, description="e.g. SMB, Mid-Market, Enterprise")
    target_job_titles: list[str] = Field(default_factory=list, description="Key decision maker and champion titles")
    pain_points: list[str] = Field(default_factory=list, description="Customer problems directly addressed")
    value_proposition: str = Field(default="", description="Compelling 1-2 sentence core value proposition")
    competitors: list[str] = Field(default_factory=list, description="Key competitors or alternative solutions")
    objection_responses: dict[str, str] = Field(default_factory=dict, description="Objection trigger -> counter-response mapping")
    entry_offer: str = Field(default="", description="Specific low-friction entry CTA or trial offer")
    is_active: bool = Field(default=True, description="Whether product is active for prospecting and outreach")

    @field_validator("name")
    @classmethod
    def _validate_name(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Product name cannot be empty or whitespace")
        return s


class ProductUpdate(BaseModel):
    name: str | None = None
    sku: str | None = None
    category: str | None = None
    description: str | None = None
    pricing_model: str | None = None
    pricing_details: str | None = None
    features: list[str] | None = None
    benefits: list[str] | None = None
    target_industries: list[str] | None = None
    target_customer_sizes: list[str] | None = None
    target_job_titles: list[str] | None = None
    pain_points: list[str] | None = None
    value_proposition: str | None = None
    competitors: list[str] | None = None
    objection_responses: dict[str, str] | None = None
    entry_offer: str | None = None
    is_active: bool | None = None


class Product(BaseModel):
    id: str = ""
    name: str = ""
    sku: str = ""
    category: str = ""
    description: str = ""
    pricing_model: str = ""
    pricing_details: str = ""
    features: list[str] = Field(default_factory=list)
    benefits: list[str] = Field(default_factory=list)
    target_industries: list[str] = Field(default_factory=list)
    target_customer_sizes: list[str] = Field(default_factory=list)
    target_job_titles: list[str] = Field(default_factory=list)
    pain_points: list[str] = Field(default_factory=list)
    value_proposition: str = ""
    competitors: list[str] = Field(default_factory=list)
    objection_responses: dict[str, str] = Field(default_factory=dict)
    entry_offer: str = ""
    is_active: bool = True
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def features_json(self) -> str:
        return json.dumps(self.features)

    def benefits_json(self) -> str:
        return json.dumps(self.benefits)

    def target_industries_json(self) -> str:
        return json.dumps(self.target_industries)

    def target_customer_sizes_json(self) -> str:
        return json.dumps(self.target_customer_sizes)

    def target_job_titles_json(self) -> str:
        return json.dumps(self.target_job_titles)

    def pain_points_json(self) -> str:
        return json.dumps(self.pain_points)

    def competitors_json(self) -> str:
        return json.dumps(self.competitors)

    def objection_responses_json(self) -> str:
        return json.dumps(self.objection_responses)


# Response alias
ProductResponse = Product


class ProductICP(BaseModel):
    id: str = ""
    product_id: str
    target_industries: list[str] = Field(default_factory=list)
    company_size: str = ""
    revenue_range: str = ""
    job_titles: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)
    business_characteristics: list[str] = Field(default_factory=list)
    pain_points: list[str] = Field(default_factory=list)
    buying_signals: list[str] = Field(default_factory=list)
    exclusion_criteria: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ProspectProductMatch(BaseModel):
    id: str = ""
    prospect_id: str
    product_id: str
    fit_score: float = 0.0
    matching_reasons: list[str] = Field(default_factory=list)
    matching_status: str = "matched"  # 'matched', 'qualified', 'disqualified', 'contacted'
    ai_analysis: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)


class ProductResearchEvidence(BaseModel):
    id: str = ""
    prospect_id: str
    product_id: str
    evidence_source: str = ""
    evidence_url: str = ""
    evidence_description: str = ""
    buying_signal: str = ""
    retrieval_timestamp: datetime = Field(default_factory=_utcnow)
