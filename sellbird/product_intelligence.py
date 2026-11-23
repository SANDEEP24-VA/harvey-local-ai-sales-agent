"""Local AI Product Intelligence using Ollama.

Provides product-specific ICP generation, messaging intelligence, and
prospect-product matching with explainable scoring and traceable evidence.
"""

from datetime import datetime, timezone
import json
import logging
from typing import Any

from sellbird.brain import Brain
from sellbird.models.business_profile import CompanyBusinessProfile
from sellbird.models.product import (
    Product,
    ProductICP,
    ProductResearchEvidence,
    ProspectProductMatch,
)
from sellbird.models.prospect import Prospect
from sellbird.models.research import ClassifiedBuyingSignal, ProspectProblemStatement

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


async def generate_product_icp(brain: Brain, product: Product) -> ProductICP | None:
    """Generate an Ideal Customer Profile (ICP) for a product using local Ollama."""
    features_str = ", ".join(product.features) if product.features else "Not specified"
    benefits_str = ", ".join(product.benefits) if product.benefits else "Not specified"
    industries_str = ", ".join(product.target_industries) if product.target_industries else "Open"
    titles_str = ", ".join(product.target_job_titles) if product.target_job_titles else "Decision makers"
    pains_str = ", ".join(product.pain_points) if product.pain_points else "General productivity"

    prompt = f"""You are a B2B sales strategist. Analyze the product below and generate a detailed, actionable Ideal Customer Profile (ICP).

Product Name: {product.name}
Category: {product.category or 'B2B Solution'}
Description: {product.description}
Pricing: {product.pricing_model} ({product.pricing_details})
Value Proposition: {product.value_proposition}
Features: {features_str}
Benefits: {benefits_str}
Target Industries: {industries_str}
Target Titles: {titles_str}
Pain Points Solved: {pains_str}

Respond ONLY with a JSON object matching this schema:
{{
  "target_industries": ["industry1", "industry2"],
  "company_size": "e.g. 10-250 employees",
  "revenue_range": "e.g. $1M-$20M ARR",
  "job_titles": ["VP Sales", "Head of Growth", "CEO"],
  "technologies": ["CRM", "Salesforce", "HubSpot"],
  "business_characteristics": ["characteristic 1", "characteristic 2"],
  "pain_points": ["pain 1", "pain 2"],
  "buying_signals": ["signal 1", "signal 2"],
  "exclusion_criteria": ["exclusion 1", "exclusion 2"]
}}

Do NOT wrap in markdown explanation. Return ONLY valid JSON."""

    try:
        data = await brain.think_json(
            prompt,
            session_id=f"icp-gen-{product.id}",
            agent="analyst",
            task="generate_product_icp",
        )
    except Exception as e:
        logger.warning(f"Product intelligence: ICP generation failed ({e})")
        return None

    if not isinstance(data, dict):
        logger.warning(f"Product intelligence: ICP generation returned non-dict: {type(data)}")
        return None

    now = _utcnow()
    return ProductICP(
        product_id=product.id,
        target_industries=[str(x) for x in data.get("target_industries", []) if str(x).strip()],
        company_size=str(data.get("company_size", "")).strip(),
        revenue_range=str(data.get("revenue_range", "")).strip(),
        job_titles=[str(x) for x in data.get("job_titles", []) if str(x).strip()],
        technologies=[str(x) for x in data.get("technologies", []) if str(x).strip()],
        business_characteristics=[str(x) for x in data.get("business_characteristics", []) if str(x).strip()],
        pain_points=[str(x) for x in data.get("pain_points", []) if str(x).strip()],
        buying_signals=[str(x) for x in data.get("buying_signals", []) if str(x).strip()],
        exclusion_criteria=[str(x) for x in data.get("exclusion_criteria", []) if str(x).strip()],
        created_at=now,
        updated_at=now,
    )


async def generate_product_intelligence(brain: Brain, product: Product) -> dict[str, Any] | None:
    """Generate problem statements, value props, differentiators, and objection responses."""
    features_str = ", ".join(product.features) if product.features else "Standard features"
    pains_str = ", ".join(product.pain_points) if product.pain_points else "Inefficiency"
    competitors_str = ", ".join(product.competitors) if product.competitors else "Legacy approaches"

    prompt = f"""You are a product marketing and sales positioning expert. Generate core sales intelligence assets for this product.

Product Name: {product.name}
Description: {product.description}
Features: {features_str}
Pain Points Addressed: {pains_str}
Competitors: {competitors_str}
Value Proposition: {product.value_proposition}

Respond ONLY with a JSON object in this format:
{{
  "problem_statements": [
    "Problem statement 1...",
    "Problem statement 2..."
  ],
  "value_propositions": [
    "Value prop 1...",
    "Value prop 2..."
  ],
  "differentiators": [
    "Differentiator 1...",
    "Differentiator 2..."
  ],
  "objection_responses": {{
    "too expensive": "Evidence-backed response...",
    "already have a solution": "Evidence-backed response...",
    "not a priority right now": "Evidence-backed response..."
  }}
}}

Return ONLY valid JSON."""

    try:
        data = await brain.think_json(
            prompt,
            session_id=f"intelligence-gen-{product.id}",
            agent="analyst",
            task="generate_product_intelligence",
        )
    except Exception as e:
        logger.warning(f"Product intelligence: Intelligence generation failed ({e})")
        return None

    if not isinstance(data, dict):
        return None

    return {
        "problem_statements": [str(s) for s in data.get("problem_statements", []) if str(s).strip()],
        "value_propositions": [str(s) for s in data.get("value_propositions", []) if str(s).strip()],
        "differentiators": [str(s) for s in data.get("differentiators", []) if str(s).strip()],
        "objection_responses": {
            str(k): str(v) for k, v in data.get("objection_responses", {}).items()
            if str(k).strip() and str(v).strip()
        },
    }


def _deterministic_heuristic_match(
    prospect: Prospect, product: Product, icp: ProductICP | None
) -> tuple[ProspectProductMatch, list[ProductResearchEvidence]]:
    """Safe fallback matching heuristic when Ollama inference is unavailable or fails."""
    score = 40.0
    reasons = []

    # Title match against product target titles or ICP job titles
    target_titles = [t.lower() for t in product.target_job_titles]
    if icp and icp.job_titles:
        target_titles.extend(t.lower() for t in icp.job_titles)

    p_title = (prospect.title or "").lower()
    if any(t in p_title for t in target_titles):
        score += 25.0
        reasons.append(f"Title '{prospect.title}' matches target personas")
    elif any(kw in p_title for kw in ["head", "vp", "director", "chief", "founder", "lead"]):
        score += 15.0
        reasons.append(f"Decision-maker seniority indicated by title '{prospect.title}'")

    # Industry match
    target_ind = [i.lower() for i in product.target_industries]
    if icp and icp.target_industries:
        target_ind.extend(i.lower() for i in icp.target_industries)

    p_ind = (prospect.industry or "").lower()
    if p_ind and any(i in p_ind or p_ind in i for i in target_ind):
        score += 20.0
        reasons.append(f"Industry '{prospect.industry}' aligns with target vertical")

    # Company size match
    if prospect.company_size and product.target_customer_sizes:
        if any(sz.lower() in prospect.company_size.lower() for sz in product.target_customer_sizes):
            score += 10.0
            reasons.append(f"Company size '{prospect.company_size}' matches ideal tier")

    score = max(5.0, min(95.0, score))
    status = "qualified" if score >= 75 else ("matched" if score >= 50 else "low_fit")

    match = ProspectProductMatch(
        prospect_id=prospect.id,
        product_id=product.id,
        fit_score=score,
        matching_reasons=reasons or ["Baseline profile match"],
        matching_status=status,
        ai_analysis={
            "source": "heuristic_fallback",
            "verified_facts": [
                f"Title: {prospect.title or 'Unknown'}",
                f"Company: {prospect.company or 'Unknown'}",
                f"Industry: {prospect.industry or 'Unknown'}",
            ],
            "inferred_points": ["Automated baseline rule evaluation applied"],
            "missing_information": ["Detailed website evidence not verified"],
        },
    )

    evidence = []
    if prospect.source_url or prospect.source:
        evidence.append(
            ProductResearchEvidence(
                prospect_id=prospect.id,
                product_id=product.id,
                evidence_source=prospect.source or "Prospect record",
                evidence_url=prospect.source_url or "",
                evidence_description=f"Prospect discovered via {prospect.source or 'record'}",
                buying_signal="prospect_discovery",
            )
        )

    return match, evidence


async def match_prospect_product(
    brain: Brain,
    prospect: Prospect,
    product: Product,
    icp: ProductICP | None = None,
    signals: list[dict] | None = None,
    tech_stack: list[str] | None = None,
) -> tuple[ProspectProductMatch, list[ProductResearchEvidence]]:
    """Evaluate prospect against product and ICP, returning fit score, reasoning, and evidence."""
    # Format ICP summary
    icp_summary = ""
    if icp:
        icp_summary = (
            f"Target Industries: {', '.join(icp.target_industries)}\n"
            f"Target Titles: {', '.join(icp.job_titles)}\n"
            f"Company Size: {icp.company_size}\n"
            f"Pain Points: {', '.join(icp.pain_points)}\n"
            f"Buying Signals: {', '.join(icp.buying_signals)}\n"
            f"Exclusion Criteria: {', '.join(icp.exclusion_criteria)}"
        )
    else:
        icp_summary = (
            f"Target Industries: {', '.join(product.target_industries)}\n"
            f"Target Titles: {', '.join(product.target_job_titles)}\n"
            f"Target Sizes: {', '.join(product.target_customer_sizes)}\n"
            f"Pain Points: {', '.join(product.pain_points)}"
        )

    signals_summary = ", ".join([s.get("type", "") for s in (signals or []) if isinstance(s, dict)]) or "None"
    tech_summary = ", ".join(tech_stack or []) or "None"

    prompt = f"""You are an objective B2B prospect qualification agent.
Evaluate if the following prospect is a fit for the specified product.

PRODUCT DETAILS:
Name: {product.name}
Description: {product.description}
Value Proposition: {product.value_proposition}
Pricing: {product.pricing_model} ({product.pricing_details})

PRODUCT ICP:
{icp_summary}

PROSPECT PROFILE:
- Full Name: {prospect.full_name()}
- Job Title: {prospect.title}
- Seniority: {prospect.seniority or 'unknown'}
- Company: {prospect.company}
- Industry: {prospect.industry}
- Company Size: {prospect.company_size}
- Known Tech Stack: {tech_summary}
- Observed Signals: {signals_summary}
- Notes / Website Facts: {prospect.personalization_notes or 'None'}

EVALUATION GUIDELINES:
1. Distinguish strictly between:
   - verified_facts (directly observed in the profile)
   - inferred_points (logical conclusions based on role/industry)
   - missing_information (key data not available)
2. Score criteria (0-100):
   - 80-100 (qualified): Strong persona match, clear pain point fit, high decision authority
   - 60-79 (matched): Good match with minor missing variables
   - 40-59 (low_fit): Partial match, non-ideal persona or adjacent industry
   - 0-39 (disqualified): Violates exclusion criteria or no relevance
3. DO NOT invent website links or facts that were not provided.
4. Extract traceable research evidence items supporting your score.

Respond ONLY with a JSON object in this format:
{{
  "fit_score": 85,
  "status": "qualified",
  "matching_reasons": ["Reason 1", "Reason 2"],
  "analysis": {{
    "verified_facts": ["fact 1", "fact 2"],
    "inferred_points": ["inference 1"],
    "missing_information": ["missing point 1"],
    "key_pain_point": "Primary pain point",
    "recommended_angle": "Suggested outreach hook"
  }},
  "evidence": [
    {{
      "source": "Title Analysis",
      "url": "",
      "description": "Specific rationale...",
      "buying_signal": ""
    }}
  ]
}}

Return ONLY valid JSON."""

    try:
        data = await brain.think_json(
            prompt,
            session_id=f"match-{product.id}-{prospect.id}",
            agent="scout",
            task="match_prospect_product",
        )
    except Exception as e:
        logger.warning(f"Product intelligence: AI matching failed ({e}), using heuristic.")
        return _deterministic_heuristic_match(prospect, product, icp)

    if not isinstance(data, dict) or "fit_score" not in data:
        logger.warning("Product intelligence: AI matching returned invalid JSON, using heuristic.")
        return _deterministic_heuristic_match(prospect, product, icp)

    try:
        raw_score = float(data.get("fit_score", 50))
        fit_score = max(0.0, min(100.0, raw_score))
    except (TypeError, ValueError):
        fit_score = 50.0

    status = str(data.get("status", "matched")).strip().lower()
    if status not in ("qualified", "matched", "low_fit", "disqualified"):
        status = "qualified" if fit_score >= 75 else ("matched" if fit_score >= 50 else "low_fit")

    matching_reasons = [str(r) for r in data.get("matching_reasons", []) if str(r).strip()]
    if not matching_reasons:
        matching_reasons = [f"Evaluated fit score: {int(fit_score)}/100"]

    analysis = data.get("analysis") if isinstance(data.get("analysis"), dict) else {}

    match = ProspectProductMatch(
        prospect_id=prospect.id,
        product_id=product.id,
        fit_score=fit_score,
        matching_reasons=matching_reasons,
        matching_status=status,
        ai_analysis=analysis,
    )

    evidence_items = []
    raw_evidence = data.get("evidence", [])
    if isinstance(raw_evidence, list):
        for ev in raw_evidence:
            if isinstance(ev, dict):
                evidence_items.append(
                    ProductResearchEvidence(
                        prospect_id=prospect.id,
                        product_id=product.id,
                        evidence_source=str(ev.get("source", "AI Research")).strip(),
                        evidence_url=str(ev.get("url", "")).strip(),
                        evidence_description=str(ev.get("description", "")).strip(),
                        buying_signal=str(ev.get("buying_signal", "")).strip(),
                    )
                )

    # Fallback evidence if model did not return any
    if not evidence_items and prospect.source:
        evidence_items.append(
            ProductResearchEvidence(
                prospect_id=prospect.id,
                product_id=product.id,
                evidence_source=prospect.source,
                evidence_url=prospect.source_url or "",
                evidence_description=f"Prospect verified from {prospect.source}",
                buying_signal="prospect_verified",
            )
        )

    return match, evidence_items


def _deterministic_business_profile_fallback(
    company_id: str,
    company_name: str,
    domain: str = "",
    site_text: str = "",
    observations: list[dict] | None = None,
) -> CompanyBusinessProfile:
    """Safe fallback business profile synthesizer when Ollama inference is unavailable or fails."""
    now = _utcnow()
    name = company_name or (domain.split(".")[0].capitalize() if domain else "Target Business")
    summary = f"{name} is an active commercial organization operating under domain {domain or 'verified website'}."

    tech_stack = []
    text_lower = (site_text or "").lower()
    for tech in ["WordPress", "Shopify", "React", "Next.js", "HubSpot", "Salesforce", "Stripe"]:
        if tech.lower() in text_lower:
            tech_stack.append(tech)

    core_offerings = []
    if site_text:
        for line in site_text.splitlines():
            line_str = line.strip()
            if any(k in line_str.lower() for k in ["services", "products", "solutions", "we provide", "we offer"]):
                if len(line_str) < 120 and len(line_str) > 10:
                    core_offerings.append(line_str)
                    if len(core_offerings) >= 4:
                        break
    if not core_offerings:
        core_offerings = [f"{name} core solutions", "B2B client services"]

    return CompanyBusinessProfile(
        company_id=company_id,
        summary=summary,
        business_model="B2B Commercial Services / Product Delivery",
        core_offerings=core_offerings,
        target_markets=[f"Commercial clients for {name}"],
        tech_stack=tech_stack or ["Standard Web Stack"],
        operational_scale="Established operating business",
        raw_profile={
            "source": "deterministic_fallback",
            "domain": domain,
            "observations_count": len(observations or []),
        },
        created_at=now,
        updated_at=now,
    )


async def generate_business_profile(
    brain: Brain,
    company_id: str,
    company_name: str,
    domain: str = "",
    site_text: str = "",
    observations: list[dict] | None = None,
) -> CompanyBusinessProfile:
    """Generate an AI-synthesized business profile for a target company using local Ollama."""
    obs_summary = ", ".join([f"{o.get('code', 'OBS')}: {o.get('detail', '')}" for o in (observations or [])[:10]])
    text_excerpt = (site_text or "")[:4000]

    prompt = f"""You are a commercial corporate analyst. Analyze this business and generate a concise, structured business profile.

COMPANY INFORMATION:
Name: {company_name}
Domain: {domain}
Observed facts & signals: {obs_summary or 'None recorded'}

EXTRACTED WEBSITE TEXT:
\"\"\"
{text_excerpt or 'No website text available.'}
\"\"\"

Respond ONLY with a JSON object in this format:
{{
  "summary": "Executive summary of what the company does (2-3 sentences)",
  "business_model": "e.g. B2B SaaS, Professional Services, Direct-to-Consumer, Agency",
  "core_offerings": ["Offering 1", "Offering 2", "Offering 3"],
  "target_markets": ["Target segment 1", "Target segment 2"],
  "tech_stack": ["Tech 1", "Tech 2"],
  "operational_scale": "e.g. Early-stage, Mid-market (50-200 employees), Enterprise",
  "verified_facts": ["fact 1", "fact 2"],
  "inferred_insights": ["insight 1"]
}}

Return ONLY valid JSON."""

    try:
        data = await brain.think_json(
            prompt,
            session_id=f"business-profile-{company_id}",
            agent="scout",
            task="generate_business_profile",
        )
    except Exception as e:
        logger.warning(f"Product intelligence: Business profile generation failed ({e}), using fallback.")
        return _deterministic_business_profile_fallback(company_id, company_name, domain, site_text, observations)

    if not isinstance(data, dict) or not data.get("summary"):
        logger.warning("Product intelligence: Business profile returned invalid JSON, using fallback.")
        return _deterministic_business_profile_fallback(company_id, company_name, domain, site_text, observations)

    now = _utcnow()
    return CompanyBusinessProfile(
        company_id=company_id,
        summary=str(data.get("summary", "")).strip(),
        business_model=str(data.get("business_model", "Commercial Enterprise")).strip(),
        core_offerings=[str(x) for x in data.get("core_offerings", []) if str(x).strip()],
        target_markets=[str(x) for x in data.get("target_markets", []) if str(x).strip()],
        tech_stack=[str(x) for x in data.get("tech_stack", []) if str(x).strip()],
        operational_scale=str(data.get("operational_scale", "Operating Business")).strip(),
        raw_profile={
            "source": "ollama_inference",
            "verified_facts": [str(x) for x in data.get("verified_facts", []) if str(x).strip()],
            "inferred_insights": [str(x) for x in data.get("inferred_insights", []) if str(x).strip()],
        },
        created_at=now,
        updated_at=now,
    )


def _deterministic_problem_statement_fallback(
    prospect: Prospect,
    product: Product,
    icp: ProductICP | None = None,
    signals: list[ClassifiedBuyingSignal] | None = None,
    research_text: str = "",
) -> ProspectProblemStatement:
    """Safe fallback problem statement generator when Ollama inference is unavailable or fails."""
    now = _utcnow()
    title = prospect.title or "Leader"
    company = prospect.company or "the organization"

    primary_pain = (product.pain_points[0] if product.pain_points else "inefficient manual processes")
    business_problem = f"{company} faces operational friction around {primary_pain.lower()} in their {prospect.industry or 'industry'} workflows."
    pain_points = product.pain_points[:3] if product.pain_points else [
        f"Time-consuming workflows for {title}",
        "Lack of automated, privacy-first local intelligence",
        "Sub-optimal conversion and pipeline velocity",
    ]
    business_impact = f"Unresolved {primary_pain.lower()} leads to higher operational overhead and lost sales opportunities for {company}."
    product_benefits = product.benefits[:3] if product.benefits else [product.value_proposition or f"Automated solution for {product.name}"]
    value_proposition = product.value_proposition or f"{product.name} delivers fast, measurable efficiency gains."
    personalisation_angle = f"Focus on how {title} at {company} can eliminate {primary_pain.lower()} with {product.name}."
    recommendation_reason = f"High alignment between {title}'s responsibilities and {product.name}'s core value proposition."

    citations = []
    if prospect.title:
        citations.append({"source": "Prospect Profile", "quote": f"Job Title: {prospect.title}", "relevance": "Role alignment"})
    if prospect.company:
        citations.append({"source": "Prospect Profile", "quote": f"Company: {prospect.company}", "relevance": "Company context"})
    if signals:
        for s in signals[:2]:
            citations.append({"source": s.signal_code, "quote": s.supporting_evidence, "relevance": s.result})

    return ProspectProblemStatement(
        prospect_id=prospect.id,
        product_id=product.id,
        business_problem=business_problem,
        pain_points=pain_points,
        business_impact=business_impact,
        product_benefits=product_benefits,
        value_proposition=value_proposition,
        personalisation_angle=personalisation_angle,
        recommendation_reason=recommendation_reason,
        evidence_citations=citations,
        created_at=now,
        updated_at=now,
    )


async def generate_prospect_problem_statement(
    brain: Brain,
    prospect: Prospect,
    product: Product,
    icp: ProductICP | None = None,
    signals: list[ClassifiedBuyingSignal] | None = None,
    research_text: str = "",
) -> ProspectProblemStatement:
    """Generate prospect-specific business problems, pain points, impact, and traceable citations using local Ollama."""
    signals_summary = "\n".join(
        [f"- [{s.verification_status}] {s.signal_code}: {s.result} (Evidence: {s.supporting_evidence[:100]})"
         for s in (signals or [])[:8]]
    ) or "None recorded"

    features_str = ", ".join(product.features) if product.features else "Standard features"
    pains_str = ", ".join(product.pain_points) if product.pain_points else "Operational bottlenecks"
    text_excerpt = (research_text or "")[:3000]

    prompt = f"""You are an elite B2B sales strategist. Identify the specific business problem that prospect {prospect.full_name()} ({prospect.title} at {prospect.company}) faces, and map how {product.name} solves it.

PRODUCT INFORMATION:
Product: {product.name}
Description: {product.description}
Value Proposition: {product.value_proposition}
Features: {features_str}
Target Pain Points: {pains_str}

PROSPECT CONTEXT:
Name: {prospect.full_name()}
Title: {prospect.title}
Company: {prospect.company}
Industry: {prospect.industry}
Notes: {prospect.personalization_notes or 'None'}

OBSERVED BUYING SIGNALS:
{signals_summary}

WEB RESEARCH EXCERPT:
\"\"\"
{text_excerpt or 'No additional web text.'}
\"\"\"

CRITICAL GROUNDING RULES:
1. Ground the business problem directly in the prospect's actual role, company, and observed signals.
2. Under "evidence_citations", cite real quotes or attributes from the prospect context or web text.
3. Formulate clear pain points, business impact, product benefits, tailored value proposition, and outreach angle.

Respond ONLY with a JSON object in this format:
{{
  "business_problem": "Specific business problem statement (1-2 sentences)",
  "pain_points": ["Pain point 1", "Pain point 2", "Pain point 3"],
  "business_impact": "Cost or business impact if unaddressed",
  "product_benefits": ["Relevant product benefit 1", "Benefit 2"],
  "value_proposition": "Tailored value proposition for this prospect",
  "personalisation_angle": "Angle for high-conversion cold outreach",
  "recommendation_reason": "Why this product is specifically recommended",
  "evidence_citations": [
    {{
      "source": "Web Research or Profile",
      "quote": "Direct quote or fact",
      "relevance": "Why this supports the problem statement"
    }}
  ]
}}

Return ONLY valid JSON."""

    try:
        data = await brain.think_json(
            prompt,
            session_id=f"problem-statement-{prospect.id}-{product.id}",
            agent="analyst",
            task="generate_problem_statement",
        )
    except Exception as e:
        logger.warning(f"Product intelligence: Problem statement generation failed ({e}), using fallback.")
        return _deterministic_problem_statement_fallback(prospect, product, icp, signals, research_text)

    if not isinstance(data, dict) or not data.get("business_problem"):
        logger.warning("Product intelligence: Problem statement returned invalid JSON, using fallback.")
        return _deterministic_problem_statement_fallback(prospect, product, icp, signals, research_text)

    now = _utcnow()
    citations = []
    for c in data.get("evidence_citations", []):
        if isinstance(c, dict) and c.get("quote"):
            citations.append({
                "source": str(c.get("source", "Prospect Data")).strip(),
                "quote": str(c.get("quote", "")).strip(),
                "relevance": str(c.get("relevance", "")).strip(),
            })

    return ProspectProblemStatement(
        prospect_id=prospect.id,
        product_id=product.id,
        business_problem=str(data.get("business_problem", "")).strip(),
        pain_points=[str(x) for x in data.get("pain_points", []) if str(x).strip()],
        business_impact=str(data.get("business_impact", "")).strip(),
        product_benefits=[str(x) for x in data.get("product_benefits", []) if str(x).strip()],
        value_proposition=str(data.get("value_proposition", "")).strip(),
        personalisation_angle=str(data.get("personalisation_angle", "")).strip(),
        recommendation_reason=str(data.get("recommendation_reason", "")).strip(),
        evidence_citations=citations or _deterministic_problem_statement_fallback(prospect, product, icp, signals, research_text).evidence_citations,
        created_at=now,
        updated_at=now,
    )


OPPORTUNITY_SCORE_FORMULA_VERSION = "v1.0"

SIGNAL_INTENT_WEIGHTS = {
    "HIRING_ROLE": 8.0,
    "OPEN_ROLES": 8.0,
    "RUNNING_GOOGLE_ADS": 7.0,
    "RUNNING_META_ADS": 7.0,
    "NO_WEBSITE": 9.0,
    "DECISION_MAKER_TITLE": 8.0,
    "NO_ONLINE_BOOKING": 6.0,
    "FUNDING_ROUND": 8.0,
    "INCUMBENT_AGENCY": 6.0,
    "YC_BACKED": 7.0,
    "ACCELERATOR": 6.0,
    "TECH_STACK": 5.0,
    "CONTACT_FOUND": 5.0,
    "IN_LOCAL_PACK": 5.0,
    "SERP_RANK": 4.0,
    "MAPS_RANK": 4.0,
    "SITE_PAGE_COUNT": 3.0,
    "BLOG_STALE": 4.0,
    "REVIEW_RATING": 4.0,
    "REVIEW_COUNT": 4.0,
    "NO_SCHEMA_MARKUP": 4.0,
    "UNCLAIMED_LISTING": 5.0,
    "BLOCKS_AI_CRAWLERS": 3.0,
    "REGISTRY_VERIFIED": 4.0,
    "LIKELY_OWNER": 6.0,
    "EMAIL_PATTERN": 3.0,
    "EMAIL_STATUS": 3.0,
    "CONTACT_FORM_URL": 3.0,
    "FOUND_IN_SERP": 2.0,
}

STATUS_MULTIPLIERS = {
    "verified_observation": 1.0,
    "deterministic_heuristic": 0.9,
    "model_inference": 0.7,
    "unverified": 0.3,
}


def calculate_opportunity_score(
    icp_fit: float,
    verified_signals: list[ClassifiedBuyingSignal] | list[dict],
    product: Product,
) -> dict[str, Any]:
    """Calculate versioned, explainable opportunity score combining ICP fit (70%) and verified buying signals (30%).

    If verified buying signals are missing, does NOT penalize missing data as negative intent:
    marks the score as provisional and explains that direct evidence has not yet been observed.
    """
    clamped_icp = max(0.0, min(100.0, float(icp_fit)))
    icp_component = round(0.70 * clamped_icp, 2)

    # Filter signals that apply or are valid
    valid_signals = []
    for s in (verified_signals or []):
        if isinstance(s, ClassifiedBuyingSignal):
            valid_signals.append(s)
        elif isinstance(s, dict) and (s.get("signal_code") or s.get("code")):
            code = s.get("signal_code") or s.get("code")
            valid_signals.append(
                ClassifiedBuyingSignal(
                    prospect_id=str(s.get("prospect_id", "")),
                    product_id=product.id,
                    signal_code=str(code),
                    category=str(s.get("category", "profile")),
                    result=str(s.get("result", "")),
                    confidence=float(s.get("confidence", 0.7)),
                    supporting_evidence=str(s.get("supporting_evidence", s.get("evidence", ""))),
                    source_url=str(s.get("source_url", "")),
                    verification_status=str(s.get("verification_status", "model_inference")),
                )
            )

    # Check if we have verified signals
    if not valid_signals:
        opportunity_score = round(clamped_icp, 2)
        status = "high_priority" if opportunity_score >= 75 else ("moderate_priority" if opportunity_score >= 50 else "low_priority")
        return {
            "opportunity_score": opportunity_score,
            "formula_version": OPPORTUNITY_SCORE_FORMULA_VERSION,
            "is_provisional": True,
            "icp_component": icp_component,
            "signal_component": 0.0,
            "signal_breakdown": [],
            "explanation": (
                f"Provisional opportunity score ({opportunity_score:.1f}/100) based entirely on ICP fit "
                f"({clamped_icp:.1f}/100) because verified web research signals have not yet been observed. "
                f"Missing research is not penalized as negative intent."
            ),
            "status": status,
        }

    raw_signal_points = 0.0
    signal_breakdown = []
    seen_codes = set()

    for sig in valid_signals:
        code = sig.signal_code.upper()
        if code in seen_codes:
            continue
        seen_codes.add(code)

        base_weight = SIGNAL_INTENT_WEIGHTS.get(code, 4.0)
        mult = STATUS_MULTIPLIERS.get(sig.verification_status, 0.7)
        pts = round(base_weight * mult * sig.confidence, 2)
        raw_signal_points += pts
        signal_breakdown.append({
            "signal_code": code,
            "result": sig.result,
            "verification_status": sig.verification_status,
            "confidence": sig.confidence,
            "points_awarded": pts,
            "supporting_evidence": sig.supporting_evidence,
        })

    signal_component = min(30.0, round(raw_signal_points, 2))
    opportunity_score = min(100.0, max(0.0, round(icp_component + signal_component, 2)))
    status = "high_priority" if opportunity_score >= 75 else ("moderate_priority" if opportunity_score >= 50 else "low_priority")

    explanation = (
        f"Opportunity score {opportunity_score:.1f}/100 (Formula {OPPORTUNITY_SCORE_FORMULA_VERSION}): "
        f"70% ICP component ({icp_component:.1f}/70 from {clamped_icp:.1f}% fit) + "
        f"30% Buying Signal component ({signal_component:.1f}/30 from {len(signal_breakdown)} verified signals)."
    )

    return {
        "opportunity_score": opportunity_score,
        "formula_version": OPPORTUNITY_SCORE_FORMULA_VERSION,
        "is_provisional": False,
        "icp_component": icp_component,
        "signal_component": signal_component,
        "signal_breakdown": signal_breakdown,
        "explanation": explanation,
        "status": status,
    }
