"""Business profile data models for local company intelligence."""

from datetime import datetime, timezone
import json
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Naive UTC now (consistent with DB storage; avoids deprecated utcnow)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class CompanyBusinessProfile(BaseModel):
    """AI-synthesized business profile for a target company."""
    id: str = ""
    company_id: str
    summary: str = Field(default="", description="Executive summary of the business")
    business_model: str = Field(default="", description="Core business & monetization model")
    core_offerings: list[str] = Field(default_factory=list, description="Primary products or services offered")
    target_markets: list[str] = Field(default_factory=list, description="Target customer segments and markets")
    tech_stack: list[str] = Field(default_factory=list, description="Observed or inferred technologies used")
    operational_scale: str = Field(default="", description="Company size, headcount or market presence")
    raw_profile: dict[str, Any] = Field(default_factory=dict, description="Complete structured profile dictionary")
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def core_offerings_json(self) -> str:
        return json.dumps(self.core_offerings)

    def target_markets_json(self) -> str:
        return json.dumps(self.target_markets)

    def tech_stack_json(self) -> str:
        return json.dumps(self.tech_stack)

    def raw_profile_json(self) -> str:
        return json.dumps(self.raw_profile)
