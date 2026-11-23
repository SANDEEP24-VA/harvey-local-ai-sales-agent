"""Research cache, buying signals, problem statements, and offline queue models."""

from datetime import datetime, timezone
import json
from typing import Any

from pydantic import BaseModel, Field


def _utcnow() -> datetime:
    """Naive UTC now (consistent with DB storage; avoids deprecated utcnow)."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class ResearchCacheItem(BaseModel):
    """Persistent local research cache item for offline re-use."""
    id: str = ""
    url: str
    domain: str = ""
    content_hash: str = ""
    content_type: str = "text/html"
    raw_content: str = ""
    extracted_text: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    status_code: int = 200
    retrieved_at: datetime = Field(default_factory=_utcnow)

    def metadata_json(self) -> str:
        return json.dumps(self.metadata)


class ClassifiedBuyingSignal(BaseModel):
    """Evidence-grounded buying signal classification."""
    id: str = ""
    prospect_id: str
    product_id: str | None = None
    signal_code: str
    category: str = ""
    result: str = ""
    confidence: float = 0.0
    supporting_evidence: str = ""
    source_url: str = ""
    retrieval_timestamp: datetime = Field(default_factory=_utcnow)
    verification_status: str = "unverified"  # 'verified_observation', 'model_inference', 'unverified'
    created_at: datetime = Field(default_factory=_utcnow)


class ProspectProblemStatement(BaseModel):
    """Prospect-specific business problem and product intelligence."""
    id: str = ""
    prospect_id: str
    product_id: str
    business_problem: str = Field(default="", description="Specific business problem identified")
    pain_points: list[str] = Field(default_factory=list, description="Targeted pain points")
    business_impact: str = Field(default="", description="Business cost or impact of inaction")
    product_benefits: list[str] = Field(default_factory=list, description="Relevant product benefits")
    value_proposition: str = Field(default="", description="Tailored value proposition")
    personalisation_angle: str = Field(default="", description="Angle for personalized outreach")
    recommendation_reason: str = Field(default="", description="Why this product is recommended")
    evidence_citations: list[dict[str, Any]] = Field(default_factory=list, description="Traceable citations to prospect data")
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def pain_points_json(self) -> str:
        return json.dumps(self.pain_points)

    def product_benefits_json(self) -> str:
        return json.dumps(self.product_benefits)

    def evidence_citations_json(self) -> str:
        return json.dumps(self.evidence_citations)


class OfflineTask(BaseModel):
    """Offline task queued for asynchronous local processing."""
    id: str = ""
    task_type: str
    payload: dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"  # 'pending', 'processing', 'completed', 'failed'
    retry_count: int = 0
    error_message: str = ""
    created_at: datetime = Field(default_factory=_utcnow)
    updated_at: datetime = Field(default_factory=_utcnow)

    def payload_json(self) -> str:
        return json.dumps(self.payload)
