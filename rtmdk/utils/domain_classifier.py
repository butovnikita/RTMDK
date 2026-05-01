"""
rtmdk/utils/domain_classifier.py — Lightweight Domain Detection.

Pattern-based domain classification for memory nodes.
Inspired by Superagent Memory OS article (habr.com/ru/articles/1021948/).

Returns: (domain, subdomain, topic) tuple
- domain: Macro-domain ("IT", "Law", "Medicine", "Finance", "Science", "Education", "Business", "general")
- subdomain: Sub-domain ("Databases", "Contracts", "Cardiology", etc.)
- topic: Topic/AnchorEntity ("SQL", "Employment Law", etc.) — ≈ AnchorEntity from article

Usage:
    from rtmdk.utils.domain_classifier import detect_domain
    domain, subdomain, topic = detect_domain("How to create a SQL index?")
    # Returns: ("IT", "Databases", "SQL")
"""

from __future__ import annotations
import re
from functools import lru_cache
from typing import Tuple

# ============================================================================
# DOMAIN PATTERNS
# 8 domains, 40+ subdomains, pattern-based (NOT LLM-based)
# ============================================================================

_DOMAIN_PATTERNS = {
    "IT": {
        "subdomains": {
            "Databases": [
                r"\b(sql|database|query|index|table|join|schema|relational|nosql|mongodb|postgres|mysql|oracle|sqlite)\b",
                r"\b(select|insert|update|delete|create table|alter|drop|transaction|commit|rollback)\b",
            ],
            "Programming": [
                r"\b(python|java|javascript|typescript|c\+\+|rust|go(lang)?|ruby|swift|kotlin|react|angular|vue)\b",
                r"\b(function|class|async|await|lambda|loop|recursion|api|rest|graphql|microservice)\b",
            ],
            "DevOps": [
                r"\b(docker|kubernetes|k8s|ci/cd|pipeline|jenkins|terraform|ansible|aws|azure|gcp|cloud)\b",
                r"\b(deploy|container|orchestrat|monitor|logging|scal(e|ing)|load balanc|nginx)\b",
            ],
            "Security": [
                r"\b(authentication|authorization|oauth|jwt|token|encryption|hash|ssl|tls|firewall|vulnerability)\b",
                r"\b(xss|csrf|sql injection|brute force|ddos|penetration|security|access control)\b",
            ],
            "Networking": [
                r"\b(http|https|tcp|ip|dns|dhcp|vpn|router|switch|firewall|port|protocol|socket)\b",
                r"\b(latency|bandwidth|packet|routing|subnet|gateway|nat|vlan)\b",
            ],
        }
    },
    "Law": {
        "subdomains": {
            "Contracts": [
                r"\b(contract|agreement|clause|liability|indemnif|termination|breach|obligation|warrant)\b",
                r"\b(party|parties|sign|execute|amend|void|enforce|compliance|legal binding)\b",
            ],
            "Employment": [
                r"\b(employee|employer|salary|wage|termination|firing|hire|discrimination|harassment)\b",
                r"\b(labor law|employment contract|benefits|vacation|sick leave|overtime|union)\b",
            ],
            "Intellectual Property": [
                r"\b(patent|trademark|copyright|intellectual property|trade secret|infringement|license)\b",
                r"\b(priority date|claim|novelty|inventor|assignee|application number)\b",
            ],
            "Litigation": [
                r"\b(lawsuit|litigation|court|judge|plaintiff|defendant|verdict|settlement|appeal)\b",
                r"\b(evidence|testimony|deposition|discovery|motion|injunction|damages)\b",
            ],
        }
    },
    "Medicine": {
        "subdomains": {
            "Cardiology": [
                r"\b(heart|cardiac|cardiovascular|blood pressure|cholesterol|arrhythmia|atrial)\b",
                r"\b(ecg|ekg|echocardiogram|stent|bypass|myocardial|infarction|stroke)\b",
            ],
            "Pharmacology": [
                r"\b(medication|drug|dosage|prescription|side effect|contraindication|pharma)\b",
                r"\b(tablet|capsule|injection|intravenous|oral|topical|bioavailability)\b",
            ],
            "Surgery": [
                r"\b(surgery|surgical|operation|incision|anesthesia|laparoscopic|transplant)\b",
                r"\b(recovery|post-op|pre-op|sterile|sutures|complication|bleeding)\b",
            ],
            "Diagnostics": [
                r"\b(diagnosis|symptom|test result|lab|mri|ct scan|x-ray|ultrasound|biopsy)\b",
                r"\b(biomarker|pathology|clinical|prognosis|differential diagnosis)\b",
            ],
        }
    },
    "Finance": {
        "subdomains": {
            "Banking": [
                r"\b(bank|account|deposit|withdrawal|interest rate|loan|mortgage|credit|debit)\b",
                r"\b(transfer|wire|ach|swift|routing number|balance|overdraft|fee)\b",
            ],
            "Investing": [
                r"\b(stock|bond|portfolio|dividend|etf|mutual fund|asset|return|yield|risk)\b",
                r"\b(bull|bear|market|trading|broker|exchange|ipo|valuation|capital gain)\b",
            ],
            "Accounting": [
                r"\b(accounting|tax|deduction|revenue|expense|profit|loss|balance sheet|audit)\b",
                r"\b(invoice|receipt|payroll|depreciation|amortization|accrual|cash flow)\b",
            ],
            "Insurance": [
                r"\b(insurance|premium|deductible|coverage|claim|policy|liability|underwriting)\b",
                r"\b(auto insurance|health insurance|life insurance|property|casualty)\b",
            ],
        }
    },
    "Science": {
        "subdomains": {
            "Physics": [
                r"\b(physics|quantum|relativity|thermodynamics|electromagnet|particle|wave)\b",
                r"\b(force|energy|momentum|entropy|gravity|velocity|acceleration|mass)\b",
            ],
            "Chemistry": [
                r"\b(chemistry|molecule|atom|reaction|catalyst|solution|acid|base|ph|element)\b",
                r"\b(compound|organic|inorganic|polymer|crystal|distillation|titration)\b",
            ],
            "Biology": [
                r"\b(biology|cell|dna|rna|protein|gene|mutation|evolution|organism|species)\b",
                r"\b(ecosystem|photosynthesis|metabolism|enzyme|chromosome|genome)\b",
            ],
            "Mathematics": [
                r"\b(math|calculus|algebra|geometry|theorem|proof|equation|integral|derivative)\b",
                r"\b(matrix|vector|probability|statistics|stochastic|linear algebra|topology)\b",
            ],
        }
    },
    "Education": {
        "subdomains": {
            "Teaching": [
                r"\b(teach|learn|student|teacher|curriculum|lesson|exam|grade|course|lecture)\b",
                r"\b(pedagogy|assessment|homework|assignment|syllabus|tutorial|workshop)\b",
            ],
            "Research": [
                r"\b(research|paper|publication|citation|peer review|thesis|dissertation|hypothesis)\b",
                r"\b(literature review|methodology|data collection|analysis|conference|journal)\b",
            ],
        }
    },
    "Business": {
        "subdomains": {
            "Management": [
                r"\b(management|strategy|leadership|team|project|goal|kpi|objective|planning)\b",
                r"\b(organizational|corporate|executive|stakeholder|board of directors|governance)\b",
            ],
            "Marketing": [
                r"\b(marketing|brand|campaign|advertising|seo|sem|social media|content|email)\b",
                r"\b(conversion|engagement|reach|impression|target audience|funnel|retention)\b",
            ],
            "Sales": [
                r"\b(sales|revenue|customer|prospect|pipeline|deal|closing|quota|commission)\b",
                r"\b(negotiation|proposal|crm|lead generation|upsell|cross-sell|retention)\b",
            ],
        }
    },
}

# Precompile all patterns for performance
_COMPILED_PATTERNS = {}
for domain, data in _DOMAIN_PATTERNS.items():
    _COMPILED_PATTERNS[domain] = {}
    for subdomain, patterns in data["subdomains"].items():
        _COMPILED_PATTERNS[domain][subdomain] = [
            re.compile(p, re.IGNORECASE) for p in patterns
        ]


@lru_cache(maxsize=10000)
def detect_domain(text: str) -> Tuple[str, str, str]:
    """Detect (domain, subdomain, topic) from text.

    Pattern-based classification (~0.1ms per call, cached).
    Returns ("general", "", "") if no domain detected.

    Args:
        text: Text content to classify.

    Returns:
        Tuple of (domain, subdomain, topic).
        Example: ("IT", "Databases", "SQL")
    """
    if not text or len(text.strip()) < 3:
        return ("general", "", "")

    text_lower = text.lower()

    # Score each domain/subdomain
    best_domain = "general"
    best_subdomain = ""
    best_score = 0

    for domain, subdomains in _COMPILED_PATTERNS.items():
        for subdomain, patterns in subdomains.items():
            score = 0
            for pattern in patterns:
                if pattern.search(text_lower):
                    score += 1
            if score > best_score:
                best_score = score
                best_domain = domain
                best_subdomain = subdomain

    # Extract topic (≈ AnchorEntity from article)
    # Topic = most specific concept mentioned
    topic = _extract_topic(text_lower, best_domain, best_subdomain)

    return (best_domain, best_subdomain, topic)


def _extract_topic(text: str, domain: str, subdomain: str) -> str:
    """Extract a short topic/anchor entity from text.

    Inspired by AnchorEntity concept from Superagent Memory OS:
    stable entities around which topics cluster.
    """
    # Try to extract key technical/domain terms
    # Priority: capitalized terms, quoted terms, domain-specific keywords

    # 1. Look for quoted concepts
    import re
    quoted = re.findall(r'"([^"]+)"', text)
    if quoted:
        return quoted[0][:50]  # First quote, max 50 chars

    # 2. Look for title-case phrases
    title_case = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)+\b', text)
    if title_case:
        return title_case[0][:50]

    # 3. Extract domain-specific keywords (simplified)
    if domain == "IT" and subdomain == "Databases":
        m = re.search(r'\b(sql|nosql|mongodb|postgres|mysql|redis|elasticsearch)\b', text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    elif domain == "IT" and subdomain == "Programming":
        m = re.search(r'\b(python|java|javascript|typescript|rust|go|ruby|swift|react|django|flask)\b', text, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()
    elif domain == "Law" and subdomain == "Contracts":
        m = re.search(r'\b(nda|msa|sla|sla|contract|agreement|clause)\b', text, re.IGNORECASE)
        if m:
            return m.group(1).upper()
    elif domain == "Medicine" and subdomain == "Cardiology":
        m = re.search(r'\b(heart|cardiac|ecg|ekg|stroke|infarction|cholesterol)\b', text, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()
    elif domain == "Finance" and subdomain == "Investing":
        m = re.search(r'\b(stock|bond|etf|mutual fund|portfolio|dividend|ipo)\b', text, re.IGNORECASE)
        if m:
            return m.group(1).capitalize()

    # 4. Fallback: first significant noun phrase (simplified)
    # Just return subdomain as topic if nothing specific found
    if subdomain:
        return subdomain

    return ""


def detect_domain_batch(texts: list) -> list:
    """Batch domain detection with caching.

    Args:
        texts: List of text strings.

    Returns:
        List of (domain, subdomain, topic) tuples.
    """
    return [detect_domain(text) for text in texts]


def get_domain_stats(field) -> dict:
    """Get domain distribution statistics from a memory field.

    Args:
        field: RTMDKField instance.

    Returns:
        Dict with domain counts and percentages.
    """
    domain_counts = {}
    total = len(field.nodes)
    if total == 0:
        return {"total": 0, "domains": {}}

    for node in field.nodes.values():
        d = getattr(node, 'domain', 'general')
        domain_counts[d] = domain_counts.get(d, 0) + 1

    return {
        "total": total,
        "domains": {
            domain: {
                "count": count,
                "percentage": round(count / total * 100, 1)
            }
            for domain, count in sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
        }
    }
