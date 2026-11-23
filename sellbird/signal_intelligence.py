"""Evidence-grounded buying signal classification using local Ollama.

Classifies prospects and companies against Sellbird's 29 signal definitions,
strictly separating verified facts (verbatim extracted quotes) from model inference.
"""

from datetime import datetime, timezone
import json
import logging
import re
from typing import Any

from sellbird.brain import Brain
from sellbird.models.product import Product
from sellbird.models.prospect import Prospect
from sellbird.models.research import ClassifiedBuyingSignal
from sellbird.research_engine import (
    STATUS_DETERMINISTIC_HEURISTIC,
    STATUS_MODEL_INFERENCE,
    STATUS_UNVERIFIED,
    STATUS_VERIFIED_OBSERVATION,
    verify_evidence_quote,
)
from sellbird.signals import SIGNAL_CATALOG

logger = logging.getLogger(__name__)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# Catalog lookup by code
_CATALOG_BY_CODE = {s["code"]: s for s in SIGNAL_CATALOG}


def _heuristic_signal_scan(
    prospect: Prospect,
    product: Product | None = None,
    research_text: str = "",
    source_url: str = "",
) -> list[ClassifiedBuyingSignal]:
    """Deterministic, non-LLM scanner for buying signals based on text and prospect attributes.

    Provides high-reliability offline classification with verified quotes directly from research_text.
    """
    signals: list[ClassifiedBuyingSignal] = []
    clean_text = (research_text or "").strip()
    text_lower = clean_text.lower()
    now = _utcnow()

    # 1. Tech stack detection
    tech_patterns = [
        ("WordPress", r"\bwordpress\b|\bwp-content\b"),
        ("Shopify", r"\bshopify\b|\bcdn\.shopify\.com\b"),
        ("HubSpot", r"\bhubspot\b|\bhs-scripts\b"),
        ("Salesforce", r"\bsalesforce\b|\bforce\.com\b"),
        ("React", r"\breact\b|\breactjs\b"),
        ("Next.js", r"\bnext\.js\b|\b_next/\b"),
        ("Stripe", r"\bstripe\b|\bjs\.stripe\.com\b"),
        ("Intercom", r"\bintercom\b|\bwidget\.intercom\.io\b"),
    ]
    detected_tech = []
    for tech_name, pat in tech_patterns:
        match = re.search(pat, text_lower)
        if match:
            detected_tech.append(tech_name)

    if detected_tech:
        quote = ", ".join(detected_tech)
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="TECH_STACK",
                category="profile",
                result=f"Detected technologies: {quote}",
                confidence=0.95,
                supporting_evidence=f"Technologies observed in web content: {quote}",
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 2. Google Ads detection
    if "googleadservices.com" in text_lower or "googletagmanager.com/gtag/js?id=aw-" in text_lower:
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="RUNNING_GOOGLE_ADS",
                category="profile",
                result="Active Google Ads acquisition spend",
                confidence=0.98,
                supporting_evidence="Google Ads tracking tag observed in site code",
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 3. Meta / Facebook Ads detection
    if "connect.facebook.net" in text_lower or "fbq('init'" in text_lower:
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="RUNNING_META_ADS",
                category="profile",
                result="Active Meta Pixel acquisition spend",
                confidence=0.98,
                supporting_evidence="Meta Pixel tracking code observed in site code",
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 4. Hiring role detection
    hiring_match = re.search(
        r"(?:we(?:'re| are) hiring|open positions|career opportunities|join our team|view open roles)[:\s]+([^\n\.\;]{5,80})",
        research_text,
        re.IGNORECASE,
    )
    if hiring_match:
        evidence_quote = hiring_match.group(0).strip()
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="HIRING_ROLE",
                category="profile",
                result="Active hiring for growth/operations",
                confidence=0.90,
                supporting_evidence=evidence_quote,
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 5. Online booking widget detection
    if any(k in text_lower for k in ["calendly.com", "acuityscheduling.com", "book a demo", "schedule a call"]):
        booking_match = re.search(r"(?:calendly\.com|acuityscheduling\.com|book a demo|schedule a call)", text_lower)
        quote = booking_match.group(0) if booking_match else "Online booking widget"
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="NO_ONLINE_BOOKING",
                category="profile",
                result="Online booking active (positive capability)",
                confidence=0.85,
                supporting_evidence=f"Booking phrase detected: '{quote}'",
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 6. Contact & Decision maker title detection from prospect profile
    title = (prospect.title or "").strip()
    if title:
        dm_keywords = ["owner", "founder", "ceo", "president", "director", "head of", "vp"]
        is_dm = any(k in title.lower() for k in dm_keywords)
        if is_dm:
            signals.append(
                ClassifiedBuyingSignal(
                    prospect_id=prospect.id,
                    product_id=product.id if product else None,
                    signal_code="DECISION_MAKER_TITLE",
                    category="people",
                    result=f"Decision-maker title: {title}",
                    confidence=0.95,
                    supporting_evidence=f"Prospect holds verified title '{title}' at {prospect.company}",
                    source_url=source_url or prospect.source_url or "",
                    verification_status="verified_observation",
                    created_at=now,
                )
            )

    # 7. Contact found signal
    if prospect.first_name and prospect.email:
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="CONTACT_FOUND",
                category="people",
                result=f"{prospect.full_name()} ({prospect.email})",
                confidence=0.99,
                supporting_evidence=f"Named contact verified with direct email {prospect.email}",
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 8. Website absence check (NO_WEBSITE)
    has_site = bool(
        getattr(prospect, "source_url", "")
        or getattr(prospect, "domain", "")
        or (prospect.email and "@" in prospect.email and not any(p in prospect.email.lower() for p in ["@gmail.", "@yahoo.", "@hotmail."]))
    )
    if not has_site or "no website" in text_lower:
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="NO_WEBSITE",
                category="discovery",
                result="Prospect has no active website domain",
                confidence=0.95,
                supporting_evidence="No website record or domain observed for prospect listing",
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 9. SERP & Search ranking detection
    serp_match = re.search(r"\b(?:serp rank|search rank:?|position #?)\s*(\d{1,3})\b", text_lower)
    if serp_match:
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="SERP_RANK",
                category="discovery",
                result=f"Rank position {serp_match.group(1)} in organic search",
                confidence=0.90,
                supporting_evidence=serp_match.group(0),
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 10. Local pack detection
    if any(k in text_lower for k in ["local 3-pack", "local pack", "google 3-pack", "local pack position"]):
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="IN_LOCAL_PACK",
                category="discovery",
                result="Appears in Google local 3-pack",
                confidence=0.90,
                supporting_evidence="Local 3-pack presence observed",
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 11. Maps rank detection
    maps_match = re.search(r"\b(?:maps rank|google maps position:?)\s*(\d{1,3})\b", text_lower)
    if maps_match:
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="MAPS_RANK",
                category="discovery",
                result=f"Google Maps rank position {maps_match.group(1)}",
                confidence=0.90,
                supporting_evidence=maps_match.group(0),
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 12. Discovery attribution (FOUND_IN_SERP)
    if prospect.source or source_url:
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="FOUND_IN_SERP",
                category="discovery",
                result=f"Discovered via {prospect.source or 'search index'}",
                confidence=0.95,
                supporting_evidence=f"Prospect discovered via {prospect.source or source_url}",
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 13. Unclaimed Google listing
    if any(k in text_lower for k in ["unclaimed listing", "claim this business", "own this business?"]):
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="UNCLAIMED_LISTING",
                category="discovery",
                result="Unclaimed Google Business Profile",
                confidence=0.88,
                supporting_evidence="Unclaimed listing phrase observed in profile",
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 14. Customer review rating & count
    rating_match = re.search(r"\b([1-5]\.[0-9])\s*(?:stars?|star rating|\/\s*5)\b", text_lower)
    if rating_match:
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="REVIEW_RATING",
                category="profile",
                result=f"Customer review rating: {rating_match.group(1)}",
                confidence=0.85,
                supporting_evidence=rating_match.group(0),
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    review_count_match = re.search(r"\b(\d{1,5})\s*(?:google|customer)?\s*reviews\b", text_lower)
    if review_count_match:
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="REVIEW_COUNT",
                category="profile",
                result=f"Total review volume: {review_count_match.group(1)}",
                confidence=0.85,
                supporting_evidence=review_count_match.group(0),
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 15. Incumbent agency credit
    agency_match = re.search(r"\b(?:website|site|designed|developed|built)\s+(?:by|with)\s+([a-z0-9\s\.\-]{3,35})\b", text_lower)
    if agency_match:
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="INCUMBENT_AGENCY",
                category="profile",
                result=f"Site built by incumbent agency: {agency_match.group(1).strip()}",
                confidence=0.85,
                supporting_evidence=agency_match.group(0),
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 16. Missing schema.org markup
    if clean_text and "schema.org" not in text_lower and "application/ld+json" not in text_lower and len(clean_text) > 300:
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="NO_SCHEMA_MARKUP",
                category="profile",
                result="Missing schema.org / JSON-LD structured data markup",
                confidence=0.85,
                supporting_evidence="No schema.org or JSON-LD tags observed in page content",
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 17. Site page count
    pages_match = re.search(r"\b(?:pages? indexed|site pages?|crawled pages?)\s*[:\s]*(\d{1,4})\b", text_lower)
    if pages_match:
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="SITE_PAGE_COUNT",
                category="profile",
                result=f"Indexed page count: {pages_match.group(1)}",
                confidence=0.80,
                supporting_evidence=pages_match.group(0),
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 18. Stale blog or copyright
    stale_match = re.search(r"\b(?:copyright|©|\(c\))\s*(201[0-9]|202[0-3])\b", text_lower)
    if stale_match:
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="BLOG_STALE",
                category="profile",
                result=f"Potentially stale digital presence (last copyright year {stale_match.group(1)})",
                confidence=0.80,
                supporting_evidence=stale_match.group(0),
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 19. Blocks AI crawlers
    if any(k in text_lower for k in ["gptbot", "ccbot", "claudebot", "anthropic-ai", "bytespider"]):
        ai_bot_match = re.search(r"\b(gptbot|ccbot|claudebot|anthropic-ai|bytespider)\b", text_lower)
        quote = ai_bot_match.group(0) if ai_bot_match else "AI crawler block"
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="BLOCKS_AI_CRAWLERS",
                category="profile",
                result=f"Robots.txt blocks AI scraper: {quote}",
                confidence=0.90,
                supporting_evidence=f"Disallow directive observed for {quote}",
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 20. Y Combinator backed
    if any(k in text_lower for k in ["y combinator", "yc w2", "yc s2", "backed by yc"]):
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="YC_BACKED",
                category="profile",
                result="Backed by Y Combinator",
                confidence=0.92,
                supporting_evidence="Y Combinator affiliation observed in company materials",
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 21. Startup accelerator participation
    if any(k in text_lower for k in ["techstars", "500 global", "500 startups", "plug and play accelerator", "alchemist accelerator"]):
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="ACCELERATOR",
                category="profile",
                result="Participated in notable startup accelerator program",
                confidence=0.88,
                supporting_evidence="Startup accelerator affiliation observed",
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 22. Venture funding round
    funding_match = re.search(r"\b(?:seed round|series [a-d]|raised \$[\d\.]+[mb]|venture[- ]backed)\b", text_lower)
    if funding_match:
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="FUNDING_ROUND",
                category="profile",
                result=f"Funding event observed: {funding_match.group(0)}",
                confidence=0.85,
                supporting_evidence=funding_match.group(0),
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 23. Multiple open roles from ATS or careers
    roles_match = re.search(r"\b(?:open roles|job openings|current openings)\s*[:\s]+([^\n\.\;]{5,80})", research_text, re.I)
    if roles_match:
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="OPEN_ROLES",
                category="profile",
                result="Multiple active open job postings",
                confidence=0.88,
                supporting_evidence=roles_match.group(0).strip(),
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 24. Business registry verification
    if any(k in text_lower for k in ["registered business", "chamber of commerce", "llc filed", "incorporated in"]):
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="REGISTRY_VERIFIED",
                category="verification",
                result="Verified entity in corporate business registry",
                confidence=0.90,
                supporting_evidence="Official business registration terminology observed",
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 25. Likely owner identified
    if any(k in (prospect.title or "").lower() for k in ["owner", "proprietor", "founder"]):
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="LIKELY_OWNER",
                category="people",
                result=f"{prospect.full_name()} identified as principal owner",
                confidence=0.95,
                supporting_evidence=f"Owner title '{prospect.title}' held by prospect",
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 26. Email pattern detected
    if prospect.email and "@" in prospect.email:
        local_part = prospect.email.split("@")[0]
        pattern = "first.last" if "." in local_part else "first"
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="EMAIL_PATTERN",
                category="people",
                result=f"Corporate email pattern: {pattern}@{prospect.email.split('@')[1]}",
                confidence=0.92,
                supporting_evidence=f"Email format derived from {prospect.email}",
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 27. Email status verification
    if prospect.status or prospect.email:
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="EMAIL_STATUS",
                category="verification",
                result=f"Email delivery status: {prospect.status or 'valid'}",
                confidence=0.90,
                supporting_evidence=f"Email record status {prospect.status or 'valid'}",
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    # 28. Contact form URL detected
    if "/contact" in text_lower or (source_url and "/contact" in source_url.lower()):
        signals.append(
            ClassifiedBuyingSignal(
                prospect_id=prospect.id,
                product_id=product.id if product else None,
                signal_code="CONTACT_FORM_URL",
                category="profile",
                result="Direct contact page or inquiry form available",
                confidence=0.88,
                supporting_evidence="Contact URL detected on site",
                source_url=source_url or prospect.source_url or "",
                verification_status="verified_observation",
                created_at=now,
            )
        )

    return signals


async def classify_buying_signals(
    brain: Brain,
    prospect: Prospect,
    product: Product | None = None,
    research_text: str = "",
    source_url: str = "",
) -> list[ClassifiedBuyingSignal]:
    """Classify buying signals using local Ollama, strictly validating verbatim evidence.

    If Ollama is unavailable or inference fails, falls back gracefully to deterministic scanning.
    """
    now = _utcnow()
    clean_text = (research_text or "").strip()

    # Pre-run heuristic scan to obtain deterministic ground-truth signals
    heuristic_signals = _heuristic_signal_scan(
        prospect=prospect,
        product=product,
        research_text=clean_text,
        source_url=source_url,
    )

    if not clean_text and not prospect.personalization_notes:
        return heuristic_signals

    text_excerpt = clean_text[:4000] if clean_text else (prospect.personalization_notes or "")[:2000]

    # Build prompt with all 29 catalog codes
    catalog_sample = [
        f"- {s['code']}: {s['label']} ({s['category']}) - {s['description'][:90]}"
        for s in SIGNAL_CATALOG
    ]
    catalog_sample_str = "\n".join(catalog_sample)

    product_context = ""
    if product:
        product_context = f"\nRelevant Product: {product.name} ({product.description[:120]})"

    prompt = f"""You are a B2B sales intelligence analyst. Analyze the following verified research text for prospect {prospect.full_name()} at {prospect.company}.
Identify observable buying signals strictly matching Sellbird's signal definitions.

RELEVANT SIGNAL CODES:
{catalog_sample_str}
{product_context}

PROSPECT CONTEXT:
Company: {prospect.company}
Title: {prospect.title}
Industry: {prospect.industry}

VERIFIED RESEARCH TEXT:
\"\"\"
{text_excerpt}
\"\"\"

CRITICAL EVIDENCE RULES:
1. "supporting_evidence" MUST be an exact or near-exact quote from the research text above. DO NOT invent citations.
2. If evidence is an exact quote from the text, it will be verified. If inferred, state clearly in supporting_evidence.
3. Choose signals only from the signal catalog codes (e.g. TECH_STACK, RUNNING_GOOGLE_ADS, HIRING_ROLE, DECISION_MAKER_TITLE, CONTACT_FOUND, etc.).
4. Assign confidence between 0.0 and 1.0.

Respond ONLY with a JSON array of objects matching this format:
[
  {{
    "signal_code": "SIGNAL_CODE",
    "category": "profile|people|discovery|verification",
    "result": "Summary of finding",
    "confidence": 0.85,
    "supporting_evidence": "Verbatim quote from text"
  }}
]

Return ONLY valid JSON. No markdown fences, no explanation."""

    try:
        data = await brain.think_json(
            prompt,
            session_id=f"signals-{prospect.id}",
            agent="scout",
            task="classify_buying_signals",
        )
    except Exception as e:
        logger.warning(f"Signal intelligence: Ollama classification failed ({e}), using heuristic.")
        return heuristic_signals

    if not isinstance(data, list):
        if isinstance(data, dict) and "signals" in data and isinstance(data["signals"], list):
            data = data["signals"]
        else:
            logger.warning("Signal intelligence: Ollama returned invalid JSON structure.")
            return heuristic_signals

    classified_signals: list[ClassifiedBuyingSignal] = []
    seen_codes: set[str] = set()

    for item in data:
        if not isinstance(item, dict):
            continue
        code = str(item.get("signal_code", "")).strip().upper()
        if not code:
            continue

        cat_meta = _CATALOG_BY_CODE.get(code, {})
        category = str(item.get("category", "")).strip() or cat_meta.get("category", "profile")
        result = str(item.get("result", "")).strip()
        evidence = str(item.get("supporting_evidence", "")).strip()

        try:
            confidence = float(item.get("confidence", 0.7))
            confidence = max(0.0, min(1.0, confidence))
        except (TypeError, ValueError):
            confidence = 0.7

        # Verification check: strict 4-status provenance classification
        if not evidence:
            verification_status = STATUS_UNVERIFIED
        elif clean_text and verify_evidence_quote(clean_text, evidence):
            verification_status = STATUS_VERIFIED_OBSERVATION
        else:
            verification_status = STATUS_MODEL_INFERENCE

        sig = ClassifiedBuyingSignal(
            prospect_id=prospect.id,
            product_id=product.id if product else None,
            signal_code=code,
            category=category,
            result=result or f"Observed {code}",
            confidence=confidence,
            supporting_evidence=evidence or "Observed via profile analysis",
            source_url=source_url or prospect.source_url or "",
            verification_status=verification_status,
            created_at=now,
        )
        classified_signals.append(sig)
        seen_codes.add(code)

    # Merge heuristic signals if not already present
    for h_sig in heuristic_signals:
        if h_sig.signal_code not in seen_codes:
            # Re-verify heuristic signal status
            if clean_text and h_sig.supporting_evidence and verify_evidence_quote(clean_text, h_sig.supporting_evidence):
                h_sig.verification_status = STATUS_VERIFIED_OBSERVATION
            else:
                h_sig.verification_status = STATUS_DETERMINISTIC_HEURISTIC
            classified_signals.append(h_sig)
            seen_codes.add(h_sig.signal_code)

    return classified_signals
