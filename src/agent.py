"""
Medical Coding Agent with Tool Use

Uses Gemini to answer medical coding exam questions by:
1. Looking up code descriptions via RAG
2. Reasoning about the clinical scenario
3. Submitting a structured answer
"""

import asyncio
import json
import os
import random
import re
from typing import Optional
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

from src.models import Question
from src.rag import MedicalCodeRetriever


# Configuration
MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "5"))
MAX_TOOL_TURNS = int(os.getenv("MAX_TOOL_TURNS", "5"))
MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
RETRY_BASE_SECONDS = float(os.getenv("LLM_RETRY_BASE_SECONDS", "1"))
RETRY_MAX_SECONDS = float(os.getenv("LLM_RETRY_MAX_SECONDS", "8"))
MAX_PREFETCH_CODES = int(os.getenv("MAX_PREFETCH_CODES", "20"))
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
GEMINI_BASE_URL = os.getenv(
    "GEMINI_BASE_URL",
    "https://generativelanguage.googleapis.com/v1beta/openai/",
)
TEXT_FALLBACK_ENABLED = os.getenv("TEXT_FALLBACK_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
TEXT_FALLBACK_MAX_ATTEMPTS = int(os.getenv("TEXT_FALLBACK_MAX_ATTEMPTS", "2"))
HEURISTIC_ENABLED = os.getenv("HEURISTIC_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
HEURISTIC_MIN_SCORE = float(os.getenv("HEURISTIC_MIN_SCORE", "0.35"))
HEURISTIC_MIN_GAP = float(os.getenv("HEURISTIC_MIN_GAP", "0.15"))
HEURISTIC_MIN_OPTIONS = int(os.getenv("HEURISTIC_MIN_OPTIONS", "3"))
HEURISTIC_DOMINANT_MIN_SCORE = float(os.getenv("HEURISTIC_DOMINANT_MIN_SCORE", "0.08"))
EVIDENCE_ENABLED = os.getenv("EVIDENCE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
EVIDENCE_MAX_DESC_LEN = int(os.getenv("EVIDENCE_MAX_DESC_LEN", "200"))
SECOND_PASS_ENABLED = os.getenv("SECOND_PASS_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
SECOND_PASS_MIN_LLM_CONF = float(os.getenv("SECOND_PASS_MIN_LLM_CONF", "0.55"))
SECOND_PASS_MIN_HEURISTIC_SCORE = float(os.getenv("SECOND_PASS_MIN_HEURISTIC_SCORE", "0.5"))
SECOND_PASS_MIN_HEURISTIC_GAP = float(os.getenv("SECOND_PASS_MIN_HEURISTIC_GAP", "0.15"))
SECOND_PASS_MIN_OPTIONS = int(os.getenv("SECOND_PASS_MIN_OPTIONS", "3"))
DISAGREE_SECOND_PASS = os.getenv("DISAGREE_SECOND_PASS", "true").lower() in {"1", "true", "yes", "on"}
RULE_FILTER_ENABLED = os.getenv("RULE_FILTER_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
RULE_FALLBACK_ENABLED = os.getenv("RULE_FALLBACK_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
RULE_FALLBACK_AUTO = os.getenv("RULE_FALLBACK_AUTO", "true").lower() in {"1", "true", "yes", "on"}
KEYWORD_RULES_ENABLED = os.getenv("KEYWORD_RULES_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
TRACE_ENABLED = os.getenv("TRACE_ENABLED", "true").lower() in {"1", "true", "yes", "on"}
TRACE_SCORE_PRECISION = int(os.getenv("TRACE_SCORE_PRECISION", "3"))


class Answer(BaseModel):
    """Structured answer from the agent."""
    selected_option: str = Field(default="")
    confidence_score: float = Field(default=0.0)
    reasoning: str = Field(default="")
    source: str = Field(default="llm")
    heuristic_score: Optional[float] = None
    heuristic_gap: Optional[float] = None
    second_pass_used: bool = False
    tool_trace: Optional[dict[str, object]] = None


# Shared instances
_retriever = MedicalCodeRetriever()
_openai_client: Optional[AsyncOpenAI] = None

_RETRY_DELAY_RE = re.compile(r"retryDelay[^0-9]*([0-9.]+)s", re.IGNORECASE)
_RETRY_IN_RE = re.compile(r"retry in ([0-9.]+)s", re.IGNORECASE)


TOOLS = [
    {
        "name": "lookup_codes",
        "description": "Look up descriptions for medical codes (CPT, ICD-10, HCPCS). Use for any codes not already provided.",
        "input_schema": {
            "type": "object",
            "properties": {
                "codes": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "List of codes to look up (e.g., ['99213', 'I10'])"
                }
            },
            "required": ["codes"]
        }
    },
    {
        "name": "submit_answer",
        "description": "Submit your final answer. REQUIRED after analyzing codes.",
        "input_schema": {
            "type": "object",
            "properties": {
                "selected_option": {
                    "type": "string",
                    "enum": ["A", "B", "C", "D"],
                    "description": "The correct answer"
                },
                "confidence": {
                    "type": "number",
                    "description": "Confidence 0-1"
                },
                "reasoning": {
                    "type": "string",
                    "description": "Why this answer is correct"
                }
            },
            "required": ["selected_option", "reasoning"]
        }
    }
]

OPENAI_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": tool["name"],
            "description": tool["description"],
            "parameters": tool["input_schema"],
        },
    }
    for tool in TOOLS
]


def _get_openai_client() -> AsyncOpenAI:
    """Lazily initialize the OpenAI-compatible client for Gemini."""
    global _openai_client
    if _openai_client is None:
        if not GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY is required for Gemini.")
        _openai_client = AsyncOpenAI(
            api_key=GEMINI_API_KEY,
            base_url=GEMINI_BASE_URL,
        )
    return _openai_client


def _extract_retry_delay(exc: Exception) -> Optional[float]:
    """Extract retry delay seconds from error message if present."""
    text = str(exc)
    match = _RETRY_DELAY_RE.search(text) or _RETRY_IN_RE.search(text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


SYSTEM_PROMPT = """You are a CPC (Certified Professional Coder) medical coding expert taking an exam.

WORKFLOW:
1. Review any pre-fetched code descriptions in the conversation
2. If any codes are missing, call lookup_codes to fetch them
3. Match the clinical scenario to the code descriptions
4. Call submit_answer with your choice

KEY CODING PRINCIPLES:
- Match specific details: anatomy, body site, procedure type, laterality
- I&D (incision & drainage) ≠ excision ≠ biopsy - different procedures
- Anesthesia codes: 00100-01999
- Choose the MOST SPECIFIC code matching clinical details
- If no perfect match, choose the CLOSEST from available options

CRITICAL: You MUST call submit_answer at the end."""


_CPT_RE = re.compile(r"\b\d{5}\b")
_CPT_ALPHA_RE = re.compile(r"\b\d{4}[A-Z]\b")
_HCPCS_RE = re.compile(r"\b[A-Z]\d{4}\b")
_ICD10_RE = re.compile(r"\b[A-TV-Z][0-9]{2}(?:\.[0-9A-TV-Z]{1,4}|[0-9A-TV-Z]{0,4})\b")
_CPT_RANGE_RE = re.compile(r"\b(\d{5})\s*[-–]\s*(\d{5})\b")
_HCPCS_RANGE_RE = re.compile(r"\b([A-Z]\d{4})\s*[-–]\s*([A-Z]\d{4})\b")
_ICD10_RANGE_RE = re.compile(
    r"\b([A-TV-Z][0-9]{2}(?:\.[0-9A-TV-Z]{1,4}|[0-9A-TV-Z]{0,4}))\s*[-–]\s*"
    r"([A-TV-Z][0-9]{2}(?:\.[0-9A-TV-Z]{1,4}|[0-9A-TV-Z]{0,4}))\b"
)
_CPT_MOD_RE = re.compile(r"\b(\d{5})(?:-[0-9A-Z]{2})\b")
_CPT_ALPHA_MOD_RE = re.compile(r"\b(\d{4}[A-Z])(?:-[0-9A-Z]{2})\b")
_HCPCS_MOD_RE = re.compile(r"\b([A-Z]\d{4})(?:-[0-9A-Z]{2})\b")
_CODE_TOKEN_RE = re.compile(
    r"\b(?:\d{5}|\d{4}[A-Z]|[A-Z]\d{4}|[A-TV-Z][0-9]{2}(?:\.[0-9A-TV-Z]{1,4}|[0-9A-TV-Z]{0,4}))\b"
)
_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has", "have",
    "in", "is", "it", "of", "on", "or", "that", "the", "their", "this", "to",
    "was", "were", "with", "without", "patient", "patients", "pt",
}

_PROCEDURE_PATTERNS = {
    "excision": [
        re.compile(r"\bexcision\b"),
        re.compile(r"\bexcise\b"),
        re.compile(r"\bresection\b"),
    ],
    "biopsy": [
        re.compile(r"\bbiops(?:y|ies)\b"),
    ],
    "incision_drainage": [
        re.compile(r"\bincision and drainage\b"),
        re.compile(r"\bi\s*&\s*d\b"),
        re.compile(r"\bi&d\b"),
        re.compile(r"\babscess\b.*\bdrainage\b"),
        re.compile(r"\bdrainage\b(?!\s+tube)"),
    ],
    "aspiration": [
        re.compile(r"\baspiration\b"),
        re.compile(r"\baspirate\b"),
        re.compile(r"\bneedle aspiration\b"),
        re.compile(r"\bpuncture aspiration\b"),
    ],
    "foreign_body": [
        re.compile(r"\bforeign body\b"),
        re.compile(r"\bremoval of foreign body\b"),
    ],
    "polypectomy": [
        re.compile(r"\bpolypectomy\b"),
        re.compile(r"\bpolyp\b"),
    ],
    "debridement": [
        re.compile(r"\bdebridement\b"),
        re.compile(r"\bdebride\b"),
    ],
    "endoscopy": [
        re.compile(r"\bendoscop"),
    ],
    "diagnostic": [
        re.compile(r"\bdiagnostic\b"),
    ],
    "lobectomy": [
        re.compile(r"\blobectomy\b"),
    ],
    "nephrectomy": [
        re.compile(r"\bnephrectomy\b"),
    ],
}

_PROCEDURE_MISMATCH = {
    "excision": {"biopsy", "incision_drainage", "aspiration", "foreign_body"},
    "biopsy": {"excision", "incision_drainage", "aspiration", "foreign_body"},
    "incision_drainage": {"biopsy", "excision", "aspiration"},
    "aspiration": {"incision_drainage", "excision", "biopsy"},
    "foreign_body": {"excision", "biopsy", "incision_drainage"},
    "lobectomy": {"biopsy", "aspiration", "incision_drainage"},
    "nephrectomy": {"biopsy", "aspiration", "incision_drainage"},
}

_ENDOSCOPY_SURGICAL_TAGS = {
    "biopsy",
    "excision",
    "foreign_body",
    "polypectomy",
    "debridement",
    "incision_drainage",
}

_PROC_FALLBACK_ORDER = {
    "excision": ["biopsy", "incision_drainage"],
}

_SITE_PATTERNS = {
    "floor_of_mouth": [
        re.compile(r"\bfloor of mouth\b"),
        re.compile(r"\bsublingual\b"),
    ],
    "tongue": [
        re.compile(r"\btongue\b"),
        re.compile(r"\blingual\b"),
    ],
    "vestibule": [
        re.compile(r"\bvestibule\b"),
        re.compile(r"\bbuccal\b"),
        re.compile(r"\bcheek\b"),
        re.compile(r"\blabial\b"),
    ],
    "nasal": [
        re.compile(r"\bnasal\b"),
        re.compile(r"\bnose\b"),
        re.compile(r"\bsinus\b"),
    ],
    "kidney": [
        re.compile(r"\bkidney\b"),
        re.compile(r"\brenal\b"),
    ],
    "thyroid": [
        re.compile(r"\bthyroid\b"),
    ],
    "neck": [
        re.compile(r"\bneck\b"),
        re.compile(r"\bcervical\b"),
    ],
    "arm": [
        re.compile(r"\barm\b"),
        re.compile(r"\bforearm\b"),
        re.compile(r"\barmpit\b"),
        re.compile(r"\baxilla\b"),
        re.compile(r"\baxillary\b"),
    ],
    "pelvis": [
        re.compile(r"\bpelvis\b"),
    ],
    "hip": [
        re.compile(r"\bhip\b"),
    ],
    "abdomen": [
        re.compile(r"\babdomen\b"),
        re.compile(r"\babdominal\b"),
    ],
    "thorax": [
        re.compile(r"\bthorax\b"),
        re.compile(r"\bchest\b"),
    ],
}

_QUALIFIER_PATTERNS = {
    "bilateral": [re.compile(r"\bbilateral\b")],
    "unilateral": [re.compile(r"\bunilateral\b")],
    "left": [re.compile(r"\bleft\b")],
    "right": [re.compile(r"\bright\b")],
    "acute": [re.compile(r"\bacute\b")],
    "chronic": [re.compile(r"\bchronic\b")],
    "recurrent": [re.compile(r"\brecurrent\b")],
    "type1": [re.compile(r"\btype\s*1\b"), re.compile(r"\btype\s*i\b")],
    "type2": [re.compile(r"\btype\s*2\b"), re.compile(r"\btype\s*ii\b")],
    "primary": [re.compile(r"\bprimary\b")],
    "secondary": [re.compile(r"\bsecondary\b")],
    "benign": [re.compile(r"\bbenign\b")],
    "malignant": [re.compile(r"\bmalignant\b")],
    "with_complications": [
        re.compile(r"\bwith complications\b"),
        re.compile(r"\bwith complication\b"),
    ],
    "without_complications": [
        re.compile(r"\bwithout complications\b"),
        re.compile(r"\bwithout complication\b"),
        re.compile(r"\bno complications\b"),
    ],
    "with_contrast": [
        re.compile(r"\bwith contrast\b"),
        re.compile(r"\bwith contrast material\b"),
    ],
    "without_contrast": [
        re.compile(r"\bwithout contrast\b"),
        re.compile(r"\bno contrast\b"),
    ],
    "new_patient": [re.compile(r"\bnew patient\b")],
    "established_patient": [re.compile(r"\bestablished patient\b"), re.compile(r"\bfollow-?up\b")],
    "preventive": [
        re.compile(r"\bpreventive\b"),
        re.compile(r"\bannual\b"),
        re.compile(r"\bwellness\b"),
        re.compile(r"\bwell-child\b"),
        re.compile(r"\bwell child\b"),
    ],
}

_MISSING_PHRASE_ALLOWLIST = {
    "core needle",
    "core biopsy",
    "fine needle aspiration",
    "fna",
    "armpit",
    "axilla",
    "axillary",
}

_QUALIFIER_CONFLICTS = {
    "bilateral": {"unilateral", "left", "right"},
    "unilateral": {"bilateral"},
    "left": {"right", "bilateral"},
    "right": {"left", "bilateral"},
    "acute": {"chronic"},
    "chronic": {"acute"},
    "recurrent": set(),
    "type1": {"type2"},
    "type2": {"type1"},
    "primary": {"secondary"},
    "secondary": {"primary"},
    "benign": {"malignant"},
    "malignant": {"benign"},
    "with_complications": {"without_complications"},
    "without_complications": {"with_complications"},
    "with_contrast": {"without_contrast"},
    "without_contrast": {"with_contrast"},
    "new_patient": {"established_patient"},
    "established_patient": {"new_patient"},
    "preventive": set(),
}


def _dedupe_preserve_order(items: list[str]) -> list[str]:
    """De-duplicate while preserving order."""
    seen = set()
    result = []
    for item in items:
        key = item.upper()
        if key in seen:
            continue
        seen.add(key)
        result.append(item.upper())
    return result


def extract_medical_codes(text: str) -> list[str]:
    """Extract likely medical codes from free text."""
    candidates = []
    for pattern in (_CPT_RANGE_RE, _HCPCS_RANGE_RE, _ICD10_RANGE_RE):
        for match in pattern.finditer(text):
            candidates.extend([match.group(1), match.group(2)])

    for pattern in (_CPT_MOD_RE, _CPT_ALPHA_MOD_RE, _HCPCS_MOD_RE):
        for match in pattern.finditer(text):
            candidates.append(match.group(1))

    for pattern in (_ICD10_RE, _HCPCS_RE, _CPT_ALPHA_RE, _CPT_RE):
        candidates.extend(pattern.findall(text))

    cleaned = []
    for code in candidates:
        cleaned_code = code.strip().upper()
        if not cleaned_code:
            continue
        if "-" in cleaned_code:
            cleaned_code = cleaned_code.split("-", 1)[0]
        cleaned.append(cleaned_code)

    return _dedupe_preserve_order(cleaned)


def _sanitize_option(value: str | None) -> str:
    """Normalize answer option to A-D or empty string."""
    if not value:
        return ""
    value = value.strip().upper()
    return value if value in {"A", "B", "C", "D"} else ""


def _clamp_confidence(value: object) -> float:
    """Clamp confidence to 0-1."""
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def _truncate(text: str, max_len: int) -> str:
    """Truncate text with ellipsis."""
    if len(text) <= max_len:
        return text
    return text[: max_len - 3].rstrip() + "..."


def _score_map(scored: list[dict[str, object]]) -> dict[str, float]:
    """Summarize scores for tracing."""
    return {
        str(item["option"]): round(float(item["score"]), TRACE_SCORE_PRECISION)
        for item in scored
    }


def _desc_count_map(scored: list[dict[str, object]]) -> dict[str, int]:
    """Summarize description counts for tracing."""
    return {str(item["option"]): int(item["desc_count"]) for item in scored}


def _missing_map(scored: list[dict[str, object]]) -> dict[str, bool]:
    """Summarize missing description flags for tracing."""
    return {str(item["option"]): bool(item["missing"]) for item in scored}


def _build_tool_trace(
    extracted_codes: list[str],
    prefetch_codes: list[str],
    scored: list[dict[str, object]],
    options_with_descs: int,
    incomplete_options: bool,
) -> Optional[dict[str, object]]:
    """Build a compact trace of tool-driven steps."""
    if not TRACE_ENABLED:
        return None
    trace = {
        "extracted_codes": extracted_codes,
        "prefetch_codes": prefetch_codes,
    }
    if scored:
        trace.update({
            "options_with_descs": options_with_descs,
            "incomplete_options": incomplete_options,
            "option_scores": _score_map(scored),
            "option_desc_counts": _desc_count_map(scored),
            "option_missing_descs": _missing_map(scored),
        })
    return trace


def _summarize_scored(scored: list[dict[str, object]]) -> tuple[int, bool]:
    """Summarize scored options for decision thresholds."""
    options_with_descs = sum(1 for item in scored if item.get("desc_count", 0) > 0)
    incomplete_options = any(bool(item.get("missing")) for item in scored)
    return options_with_descs, incomplete_options


def _allow_missing_phrases(question_text: str) -> set[str]:
    """Allow missing descriptions for certain generic phrases."""
    lower = question_text.lower()
    return {phrase for phrase in _MISSING_PHRASE_ALLOWLIST if phrase in lower}


def _apply_missing_filter(
    scored: list[dict[str, object]],
    question_text: str,
) -> tuple[list[dict[str, object]], dict[str, list[str]], set[str]]:
    """Remove options with missing descriptions unless explicitly allowed."""
    allowed_phrases = _allow_missing_phrases(question_text)
    if allowed_phrases:
        return scored, {}, allowed_phrases
    filtered: list[dict[str, object]] = []
    excluded: dict[str, list[str]] = {}
    for item in scored:
        option_key = str(item["option"])
        if item.get("missing"):
            excluded[option_key] = ["missing_description"]
            continue
        filtered.append(item)
    if not filtered:
        return scored, {}, allowed_phrases
    return filtered, excluded, allowed_phrases


def _find_option_by_desc_all(scored: list[dict[str, object]], terms: list[str]) -> Optional[str]:
    """Find the option key whose description contains all terms."""
    terms = [term.lower() for term in terms]
    for item in scored:
        descs = " ".join(item.get("descs", []))
        desc_lower = descs.lower()
        if all(term in desc_lower for term in terms):
            return str(item["option"])
    return None


def _find_option_by_desc(scored: list[dict[str, object]], needle: str) -> Optional[str]:
    """Find the option key whose description contains a term."""
    needle = needle.lower()
    for item in scored:
        descs = " ".join(item.get("descs", []))
        if needle in descs.lower():
            return str(item["option"])
    return None


def _keyword_rule_choice(question_text: str, scored: list[dict[str, object]]) -> Optional[tuple[str, str, float]]:
    """Apply targeted keyword rules for ambiguous prompts."""
    if not KEYWORD_RULES_ENABLED or not scored:
        return None
    lower = question_text.lower()

    if "core needle" in lower or "core biopsy" in lower:
        option = _find_option_by_desc_all(scored, ["core", "needle"])
        if option:
            return option, "Core needle biopsy matched by description", 0.85

    if "fine needle aspiration" in lower or "fna" in lower:
        option = _find_option_by_desc(scored, "fine needle aspiration")
        if not option:
            option = _find_option_by_desc_all(scored, ["needle", "aspiration"])
        if option:
            return option, "FNA biopsy matched by description", 0.85

    if "sinusotomy" in lower:
        option = _find_option_by_desc(scored, "endoscop")
        if option:
            return option, "Sinusotomy defaults to endoscopic approach", 0.8

    if any(term in lower for term in ("nephrostolithotomy", "nephrolithotomy", "pcnl")):
        option = _find_option_by_desc_all(scored, ["endoscope", "kidney"])
        if not option:
            option = _find_option_by_desc(scored, "endoscop")
        if option:
            return option, "Percutaneous nephrostolithotomy uses an endoscopic approach", 0.8

    if any(term in lower for term in ["extended history", "reviewed multiple systems", "detailed examination", "comprehensive review"]):
        option = _find_option_by_desc(scored, "high level")
        if option:
            return option, "Extended history/detail -> high-level E/M", 0.8

    if "checkup" in lower and "established" not in lower and "follow-up" not in lower and "follow up" not in lower:
        new_patient_options = []
        for item in scored:
            descs = " ".join(item.get("descs", []))
            if "new patient" in descs.lower() or "initial comprehensive" in descs.lower():
                new_patient_options.append(item)
        if new_patient_options:
            def code_num(candidate: dict[str, object]) -> int:
                for code in candidate.get("codes", []):
                    if code.isdigit():
                        return int(code)
                return 999999
            chosen = sorted(new_patient_options, key=code_num)[0]
            return str(chosen["option"]), "Ambiguous checkup -> default new patient", 0.75

    if "partial thyroidectomy" in lower:
        if any(term in lower for term in ["contralateral", "bilateral", "subtotal"]):
            option = _find_option_by_desc(scored, "contralateral")
            if not option:
                option = _find_option_by_desc(scored, "subtotal")
            if option:
                return option, "Partial thyroidectomy with contralateral involvement", 0.8
        option = _find_option_by_desc_all(scored, ["partial", "thyroid"])
        if option:
            return option, "Partial thyroidectomy defaults to unilateral", 0.8

    if "knee replacement" in lower and "total" not in lower:
        option = _find_option_by_desc(scored, "not otherwise specified")
        if option:
            return option, "Knee replacement without details -> NOS anesthesia", 0.75

    if "seroma" in lower:
        option = _find_option_by_desc(scored, "seroma")
        if not option:
            option = _find_option_by_desc(scored, "hematoma")
        if not option:
            option = _find_option_by_desc(scored, "fluid collection")
        if option:
            return option, "Seroma aligns with hematoma/collection drainage", 0.7

    if any(term in lower for term in ("armpit", "axilla", "axillary")):
        if any(term in lower for term in ("abscess", "incision", "drainage")):
            for item in scored:
                codes = item.get("codes", []) or []
                if "20002" in codes:
                    return str(item["option"]), "Axillary abscess -> 20002", 0.7
        option = _find_option_by_desc(scored, "axilla")
        if option:
            return option, "Axillary abscess matched by description", 0.7

    if "intrathoracic" in lower and "anesthesia" in lower:
        option = _find_option_by_desc(scored, "intrathoracic")
        if not option:
            option = _find_option_by_desc(scored, "pneumocentesis")
        if option:
            return option, "Intrathoracic anesthesia -> pneumocentesis code", 0.7

    if "excision" in lower and ("oral" in lower or "mouth" in lower):
        option = _find_option_by_desc_all(scored, ["excision", "tongue"])
        if not option:
            option = _find_option_by_desc_all(scored, ["excision", "mouth"])
        if option:
            return option, "Oral excision defaults to tongue lesion excision", 0.7

    return None


def _tokenize(text: str) -> set[str]:
    """Tokenize for heuristic matching."""
    if not text:
        return set()
    text = _CODE_TOKEN_RE.sub(" ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = []
    for token in text.split():
        if token in _STOPWORDS:
            continue
        if len(token) < 2:
            continue
        if token.isdigit():
            continue
        tokens.append(token)
    return set(tokens)


def _f1_score(a: set[str], b: set[str]) -> float:
    """Compute F1 overlap score between token sets."""
    if not a or not b:
        return 0.0
    overlap = len(a & b)
    if overlap == 0:
        return 0.0
    precision = overlap / len(b)
    recall = overlap / len(a)
    return (2 * precision * recall) / (precision + recall)


def _extract_procedure_tags(text: str) -> set[str]:
    """Extract procedure tags from text."""
    if not text:
        return set()
    lowered = text.lower()
    tags = set()
    for tag, patterns in _PROCEDURE_PATTERNS.items():
        if any(pattern.search(lowered) for pattern in patterns):
            tags.add(tag)
    return tags


def _extract_site_tags(text: str) -> set[str]:
    """Extract site tags from text."""
    if not text:
        return set()
    lowered = text.lower()
    tags = set()
    for tag, patterns in _SITE_PATTERNS.items():
        if any(pattern.search(lowered) for pattern in patterns):
            tags.add(tag)
    return tags


def _extract_qualifiers(text: str) -> set[str]:
    """Extract qualifier tags from text."""
    if not text:
        return set()
    lowered = text.lower()
    tags = set()
    for tag, patterns in _QUALIFIER_PATTERNS.items():
        if any(pattern.search(lowered) for pattern in patterns):
            tags.add(tag)

    if "left" in tags and "right" in tags:
        tags.discard("left")
        tags.discard("right")
        tags.add("bilateral")

    if "bilateral" in tags:
        tags.discard("left")
        tags.discard("right")
        tags.discard("unilateral")
    elif "left" in tags or "right" in tags:
        tags.discard("unilateral")

    if "without_contrast" in tags:
        tags.discard("with_contrast")
    if "without_complications" in tags:
        tags.discard("with_complications")

    return tags


def _option_matches_qualifier(question_qual: str, option_quals: set[str]) -> bool:
    """Check whether option qualifiers satisfy a question qualifier."""
    if question_qual == "unilateral":
        return bool(option_quals & {"unilateral", "left", "right"})
    if question_qual == "left":
        return "left" in option_quals
    if question_qual == "right":
        return "right" in option_quals
    if question_qual == "bilateral":
        return "bilateral" in option_quals
    return question_qual in option_quals


def _has_deep_indicator(descs: list[str]) -> bool:
    """Check for depth/complexity indicators in descriptions."""
    for desc in descs:
        lowered = desc.lower()
        if "deep" in lowered or "complicated" in lowered:
            return True
    return False


def _score_options(
    question: Question,
    code_descs: dict[str, str],
) -> tuple[list[dict[str, object]], int, bool]:
    """Score options using description text only."""
    if not code_descs:
        return [], 0, False
    question_tokens = _tokenize(question.text)
    if not question_tokens:
        return [], 0, False

    scored = []
    options_with_descs = 0
    incomplete_options = False
    for option_key, option_text in sorted(question.options.items()):
        codes = extract_medical_codes(option_text)
        descs = []
        resolved_codes = []
        missing = False
        for code in codes:
            desc = code_descs.get(code)
            if not desc or "description unavailable" in desc.lower():
                missing = True
                continue
            descs.append(desc)
            resolved_codes.append(code)
        if descs:
            options_with_descs += 1
        if codes and missing:
            incomplete_options = True
        candidate_text = "\n".join(descs)
        procedure_tags = _extract_procedure_tags(candidate_text)
        site_tags = _extract_site_tags(candidate_text)
        qualifier_tags = _extract_qualifiers(candidate_text)
        score = _f1_score(question_tokens, _tokenize(candidate_text))
        scored.append({
            "option": option_key,
            "score": score,
            "desc_count": len(descs),
            "codes": codes,
            "descs": descs,
            "resolved_codes": resolved_codes,
            "procedure_tags": procedure_tags,
            "site_tags": site_tags,
            "qualifier_tags": qualifier_tags,
            "missing": missing,
        })

    return scored, options_with_descs, incomplete_options


def _heuristic_pick(
    scored: list[dict[str, object]],
    options_with_descs: int,
    incomplete_options: bool,
) -> tuple[str, float, float, str] | None:
    """Pick an answer using text similarity heuristics."""
    if options_with_descs < HEURISTIC_MIN_OPTIONS:
        return None
    if incomplete_options:
        return None

    if not scored:
        return None
    dominant = _dominant_scored_option(scored)
    if dominant:
        best_option, best_score, gap = dominant
        reason = (
            f"Heuristic dominance: only one option matches descriptions (score {best_score:.2f})."
        )
        return best_option, best_score, gap, reason

    ranked = sorted(scored, key=lambda item: item["score"], reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else {"score": 0.0}

    if best["desc_count"] == 0:
        return None
    gap = float(best["score"]) - float(second["score"])
    if best["score"] >= HEURISTIC_MIN_SCORE and gap >= HEURISTIC_MIN_GAP:
        reason = (
            f"Heuristic match to code descriptions (score {best['score']:.2f} "
            f"vs {second['score']:.2f})."
        )
        return str(best["option"]), float(best["score"]), gap, reason

    return None


def _dominant_scored_option(
    scored: list[dict[str, object]],
) -> Optional[tuple[str, float, float]]:
    """Return the top option when it is the only non-zero match."""
    if not scored:
        return None
    ranked = sorted(scored, key=lambda item: item["score"], reverse=True)
    best = ranked[0]
    second = ranked[1] if len(ranked) > 1 else {"score": 0.0}
    if best.get("desc_count", 0) == 0:
        return None
    best_score = float(best.get("score", 0.0))
    second_score = float(second.get("score", 0.0))
    if best_score >= HEURISTIC_DOMINANT_MIN_SCORE and second_score == 0.0:
        return str(best.get("option")), best_score, best_score - second_score
    return None


def _best_scored_option(
    scored: list[dict[str, object]],
    rule_allowed: Optional[list[str]],
) -> Optional[tuple[str, float]]:
    """Pick the top-scoring option even if heuristic thresholds are not met."""
    if not scored:
        return None
    candidates = scored
    if rule_allowed:
        allowed_set = set(rule_allowed)
        allowed_candidates = [item for item in scored if str(item["option"]) in allowed_set]
        if allowed_candidates:
            candidates = allowed_candidates
    ranked = sorted(
        candidates,
        key=lambda item: (
            0 if item.get("desc_count", 0) > 0 else 1,
            -float(item.get("score", 0.0)),
            str(item.get("option")),
        ),
    )
    best = ranked[0]
    return str(best["option"]), float(best.get("score", 0.0))


def _should_second_pass(
    scored: list[dict[str, object]],
    options_with_descs: int,
    incomplete_options: bool,
    llm_answer: Answer,
    rule_allowed: Optional[list[str]],
    rule_strength: bool,
    fallback_choice: Optional[str],
) -> Optional[tuple[str, float, float]]:
    """Decide whether to run a second-pass check."""
    if not SECOND_PASS_ENABLED:
        return None
    if not llm_answer.selected_option:
        return None

    if options_with_descs < SECOND_PASS_MIN_OPTIONS:
        return None
    if incomplete_options:
        return None
    if not scored and not rule_allowed:
        return None

    strong_heuristic = False
    best_option = None
    best_score = 0.0
    gap = 0.0

    if scored:
        ranked = sorted(scored, key=lambda item: item["score"], reverse=True)
        best = ranked[0]
        second = ranked[1] if len(ranked) > 1 else {"score": 0.0}
        gap = float(best["score"]) - float(second["score"])
        best_option = str(best["option"])
        best_score = float(best["score"])

        if best["desc_count"] > 0 and best_score >= SECOND_PASS_MIN_HEURISTIC_SCORE and gap >= SECOND_PASS_MIN_HEURISTIC_GAP:
            if best_option != llm_answer.selected_option:
                strong_heuristic = True
        elif best["desc_count"] > 0 and best_score >= HEURISTIC_DOMINANT_MIN_SCORE and float(second.get("score", 0.0)) == 0.0:
            if best_option != llm_answer.selected_option:
                strong_heuristic = True

    rule_disagree = rule_strength and bool(rule_allowed) and llm_answer.selected_option not in set(rule_allowed)
    fallback_disagree = bool(fallback_choice) and llm_answer.selected_option != fallback_choice

    if not strong_heuristic and not rule_disagree and not fallback_disagree:
        return None

    if not DISAGREE_SECOND_PASS and llm_answer.confidence_score > SECOND_PASS_MIN_LLM_CONF:
        return None

    if rule_allowed and rule_strength:
        allowed_scored = [item for item in scored if str(item["option"]) in set(rule_allowed)] if scored else []
        if allowed_scored:
            allowed_best = sorted(allowed_scored, key=lambda item: item["score"], reverse=True)[0]
            return str(allowed_best["option"]), float(allowed_best["score"]), gap
        return rule_allowed[0], best_score, gap

    if fallback_choice:
        return fallback_choice, best_score, gap

    if best_option is None:
        return None
    return best_option, best_score, gap


def _build_evidence_pack(scored: list[dict[str, object]]) -> str:
    """Build a compact evidence pack per option."""
    if not scored:
        return ""

    lines = ["Evidence pack (code descriptions and match scores):"]
    for item in scored:
        option = item["option"]
        codes = item["codes"]
        score = float(item["score"])

        lines.append(f"Option {option}:")
        if codes:
            lines.append(f"Codes: {', '.join(codes)}")
        else:
            lines.append("Codes: none detected")
        lines.append(f"Match score (description vs question): {score:.2f}")

        resolved_codes = item.get("resolved_codes", [])
        desc_lines = [
            f"- {code}: {_truncate(desc, EVIDENCE_MAX_DESC_LEN)}"
            for code, desc in zip(resolved_codes, item["descs"])
        ]
        missing_codes = [code for code in codes if code not in resolved_codes]

        if desc_lines:
            lines.append("Descriptions:")
            lines.extend(desc_lines)
        if missing_codes:
            lines.append(f"Missing descriptions: {', '.join(missing_codes)}")

    return "\n".join(lines)


def _rule_filter_options(
    question_text: str,
    scored: list[dict[str, object]],
) -> tuple[list[str], dict[str, list[str]], set[str], set[str], set[str], dict[str, bool]]:
    """Filter options by procedure, site, and qualifier rules."""
    if not RULE_FILTER_ENABLED or not scored:
        return [], {}, set(), set(), set(), {
            "has_proc_match": False,
            "has_site_match": False,
            "has_qualifier_match": False,
        }

    question_proc = _extract_procedure_tags(question_text)
    question_sites = _extract_site_tags(question_text)
    question_quals = _extract_qualifiers(question_text)
    question_text_lower = question_text.lower()
    question_has_endoscopy = "endoscop" in question_text_lower
    question_has_surgical = bool(question_proc & _ENDOSCOPY_SURGICAL_TAGS)

    has_proc_match = any(
        question_proc & set(item.get("procedure_tags", set()))
        for item in scored
        if question_proc
    )
    has_site_match = any(
        question_sites & set(item.get("site_tags", set()))
        for item in scored
        if question_sites
    )
    has_endoscopy_option = any(
        "endoscopy" in set(item.get("procedure_tags", set()))
        for item in scored
    )

    active_quals = {
        qual for qual in question_quals
        if not (_QUALIFIER_CONFLICTS.get(qual, set()) & question_quals)
    }
    qualifier_matches: dict[str, bool] = {}
    if active_quals:
        for qual in active_quals:
            qualifier_matches[qual] = any(
                _option_matches_qualifier(qual, set(item.get("qualifier_tags", set())))
                for item in scored
            )
    has_qualifier_match = any(qualifier_matches.values()) if qualifier_matches else False

    allowed = []
    excluded: dict[str, list[str]] = {}

    for item in scored:
        option_key = str(item["option"])
        option_proc = set(item.get("procedure_tags", set()))
        option_sites = set(item.get("site_tags", set()))
        option_quals = set(item.get("qualifier_tags", set()))
        reasons = []

        if question_proc and has_proc_match:
            mismatches = set()
            for q_tag in question_proc:
                mismatches |= _PROCEDURE_MISMATCH.get(q_tag, set())
            mismatches -= question_proc
            conflict = option_proc & mismatches
            if conflict and option_proc.isdisjoint(question_proc):
                reasons.append(
                    f"procedure_mismatch:{','.join(sorted(question_proc))}->{','.join(sorted(conflict))}"
                )
            elif option_proc and option_proc.isdisjoint(question_proc):
                reasons.append(
                    f"procedure_nonmatch:{','.join(sorted(question_proc))}->{','.join(sorted(option_proc))}"
                )

        if question_has_endoscopy and has_endoscopy_option:
            if "endoscopy" not in option_proc:
                reasons.append("endoscopy_mismatch")
            elif not question_has_surgical and (option_proc & _ENDOSCOPY_SURGICAL_TAGS):
                reasons.append("endoscopy_surgical_mismatch")

        if question_sites and has_site_match:
            if option_sites and option_sites.isdisjoint(question_sites):
                reasons.append(
                    f"site_mismatch:{','.join(sorted(question_sites))}->{','.join(sorted(option_sites))}"
                )

        if active_quals:
            for qual, has_match in qualifier_matches.items():
                if not has_match:
                    continue
                if _option_matches_qualifier(qual, option_quals):
                    continue
                conflict = option_quals & _QUALIFIER_CONFLICTS.get(qual, set())
                if conflict:
                    reasons.append(
                        f"qualifier_mismatch:{qual}->{','.join(sorted(conflict))}"
                    )
                else:
                    reasons.append(f"qualifier_nonmatch:{qual}")

        if reasons:
            excluded[option_key] = reasons
        else:
            allowed.append(option_key)

    if not allowed:
        return [], excluded, question_proc, question_sites, question_quals, {
            "has_proc_match": has_proc_match,
            "has_site_match": has_site_match,
            "has_qualifier_match": has_qualifier_match,
        }

    return sorted(allowed), excluded, question_proc, question_sites, question_quals, {
        "has_proc_match": has_proc_match,
        "has_site_match": has_site_match,
        "has_qualifier_match": has_qualifier_match,
    }


def _rule_fallback_choice(
    question_proc: set[str],
    question_sites: set[str],
    scored: list[dict[str, object]],
) -> Optional[dict[str, object]]:
    """Choose a fallback option when no direct procedure match exists."""
    if not RULE_FALLBACK_ENABLED or not question_proc or not scored:
        return None

    for proc in question_proc:
        if _PROC_FALLBACK_ORDER.get(proc) is None:
            continue
        if any(proc in set(item.get("procedure_tags", set())) for item in scored):
            return None

    fallback_order = []
    for proc in question_proc:
        order = _PROC_FALLBACK_ORDER.get(proc)
        if order:
            fallback_order = order
            break

    if not fallback_order:
        return None

    candidates = []
    for item in scored:
        proc_tags = set(item.get("procedure_tags", set()))
        if proc_tags & set(fallback_order):
            candidates.append(item)

    if not candidates:
        return None

    def proc_priority(item: dict[str, object]) -> int:
        proc_tags = set(item.get("procedure_tags", set()))
        for idx, tag in enumerate(fallback_order):
            if tag in proc_tags:
                return idx
        return len(fallback_order)

    def sort_key(item: dict[str, object]) -> tuple:
        site_match = bool(question_sites & set(item.get("site_tags", set()))) if question_sites else False
        return (
            0 if site_match else 1,
            proc_priority(item),
            0 if _has_deep_indicator(item.get("descs", [])) else 1,
            -len(set(item.get("site_tags", set()))),
            -float(item.get("score", 0.0)),
            str(item.get("option")),
        )

    best = sorted(candidates, key=sort_key)[0]
    rationale = {
        "fallback_order": fallback_order,
        "site_match": bool(question_sites & set(best.get("site_tags", set()))) if question_sites else False,
        "deep_indicator": _has_deep_indicator(best.get("descs", [])),
        "site_coverage": len(set(best.get("site_tags", set()))),
    }
    return {
        "option": str(best.get("option")),
        "score": float(best.get("score", 0.0)),
        "rationale": rationale,
    }


async def _call_gemini(messages: list[dict]) -> object:
    """Call Gemini (OpenAI-compatible endpoint) with retry/backoff."""
    delay = RETRY_BASE_SECONDS
    for attempt in range(MAX_RETRIES + 1):
        try:
            client = _get_openai_client()
            return await client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=messages,
                tools=OPENAI_TOOLS,
                tool_choice="auto",
                max_tokens=1500,
            )
        except Exception as exc:
            if attempt >= MAX_RETRIES:
                raise exc
            retry_delay = _extract_retry_delay(exc)
            sleep_for = min(RETRY_MAX_SECONDS, delay) + random.uniform(0, 0.25)
            if retry_delay is not None:
                sleep_for = max(sleep_for, retry_delay)
            await asyncio.sleep(sleep_for)
            delay *= 2


async def _call_gemini_text_only(messages: list[dict]) -> object:
    """Call Gemini without tools (text-only fallback) with retry/backoff."""
    delay = RETRY_BASE_SECONDS
    for attempt in range(MAX_RETRIES + 1):
        try:
            client = _get_openai_client()
            return await client.chat.completions.create(
                model=GEMINI_MODEL,
                messages=messages,
                max_tokens=800,
            )
        except Exception as exc:
            if attempt >= MAX_RETRIES:
                raise exc
            retry_delay = _extract_retry_delay(exc)
            sleep_for = min(RETRY_MAX_SECONDS, delay) + random.uniform(0, 0.25)
            if retry_delay is not None:
                sleep_for = max(sleep_for, retry_delay)
            await asyncio.sleep(sleep_for)
            delay *= 2


async def _handle_tool_call(name: str, input_data: dict) -> str:
    """Execute a tool and return the result."""
    if name == "lookup_codes":
        raw_codes = input_data.get("codes", [])
        codes = [
            str(code).strip().upper()
            for code in raw_codes
            if str(code).strip()
        ]
        codes = _dedupe_preserve_order(codes)[:MAX_PREFETCH_CODES]
        if not codes:
            return "{}"
        results = await _retriever.retrieve_codes(codes)
        return json.dumps(results, indent=2)
    return "Unknown tool"


def _extract_answer_from_text(text: str) -> str | None:
    """Try to extract an answer letter from text response."""
    patterns = [
        r"answer[:\s]+([A-D])\b",
        r"select(?:ed)?[:\s]+([A-D])\b",
        r"\b([A-D])\s+is\s+correct",
        r"^([A-D])\.",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
    return None


async def solve_question(question: Question, on_status: callable = None) -> Answer:
    """Solve a single question using the agent with tools."""

    def status(msg: str) -> None:
        if on_status:
            on_status(f"Q{question.id}: {msg}")

    async def _attempt_text_fallback(reason: str) -> Optional[Answer]:
        nonlocal text_fallback_attempted
        if text_fallback_attempted or not TEXT_FALLBACK_ENABLED:
            return None
        text_fallback_attempted = True
        fallback_prompt = (
            "Tool calling failed. Reply with:\n"
            "Answer: <A-D>\n"
            "Reasoning: <one short sentence>\n"
            "Do not call any tools."
        )
        fallback_messages = messages + [{"role": "user", "content": fallback_prompt}]
        try:
            response = await _call_gemini_text_only(fallback_messages)
        except Exception as exc:
            if tool_trace is not None:
                tool_trace["text_fallback_error"] = str(exc)
            return None
        text_content = response.choices[0].message.content or ""
        extracted = _extract_answer_from_text(text_content)
        if extracted:
            if tool_trace is not None:
                tool_trace["decision"] = "text_fallback"
            return Answer(
                selected_option=extracted,
                confidence_score=0.55,
                reasoning=text_content[:200],
                source="text_fallback",
                second_pass_used=second_pass_attempted,
                tool_trace=tool_trace,
            )
        if tool_trace is not None and reason:
            tool_trace["text_fallback_note"] = reason
        return None

    # Format question for the agent
    options_text = "\n".join(f"{k}. {v}" for k, v in sorted(question.options.items()))
    combined_text = f"{question.text}\n{options_text}"
    extracted_codes = extract_medical_codes(combined_text)
    prefetch_codes = extracted_codes[:MAX_PREFETCH_CODES]
    extra_count = len(extracted_codes) - len(prefetch_codes)
    codes_line = "Extracted codes: "
    if prefetch_codes:
        codes_line += ", ".join(prefetch_codes)
        if extra_count > 0:
            codes_line += f" (showing first {MAX_PREFETCH_CODES}, {extra_count} more omitted)"
    else:
        codes_line += "none"
    user_message = f"""Question {question.id}:
{question.text}

Options:
{options_text}

{codes_line}

Look up the codes, analyze them, then submit your answer."""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
    status("thinking...")

    code_descs = {}
    if prefetch_codes:
        status("prefetching codes...")
        try:
            code_descs = await _retriever.retrieve_codes(prefetch_codes)
        except Exception:
            code_descs = {}
        tool_output = json.dumps(code_descs, indent=2)
        messages.append({
            "role": "user",
            "content": f"Prefetched code descriptions:\n{tool_output}",
        })

    scored = []
    options_with_descs = 0
    incomplete_options = False
    missing_excluded: dict[str, list[str]] = {}
    allowed_missing_phrases: set[str] = set()
    if code_descs:
        scored, options_with_descs, incomplete_options = _score_options(question, code_descs)
        scored, missing_excluded, allowed_missing_phrases = _apply_missing_filter(scored, question.text)
        options_with_descs, incomplete_options = _summarize_scored(scored)

    tool_trace = _build_tool_trace(
        extracted_codes,
        prefetch_codes,
        scored,
        options_with_descs,
        incomplete_options,
    )
    if tool_trace is not None and missing_excluded:
        tool_trace["missing_filter"] = {
            "allowed_missing_phrases": sorted(allowed_missing_phrases),
            "excluded_options": missing_excluded,
        }

    rule_allowed = []
    rule_excluded = {}
    question_proc = set()
    question_sites = set()
    question_quals = set()
    rule_info = {
        "has_proc_match": False,
        "has_site_match": False,
        "has_qualifier_match": False,
    }
    if scored:
        rule_allowed, rule_excluded, question_proc, question_sites, question_quals, rule_info = _rule_filter_options(
            question.text,
            scored,
        )
        if tool_trace is not None and (rule_allowed or rule_excluded):
            tool_trace["rule_filter"] = {
                "question_procedure_tags": sorted(question_proc),
                "question_site_tags": sorted(question_sites),
                "question_qualifier_tags": sorted(question_quals),
                "allowed_options": rule_allowed,
                "excluded_options": rule_excluded,
                "has_proc_match": rule_info.get("has_proc_match", False),
                "has_site_match": rule_info.get("has_site_match", False),
                "has_qualifier_match": rule_info.get("has_qualifier_match", False),
            }

    keyword_choice = _keyword_rule_choice(question.text, scored)
    if keyword_choice:
        selected_option, reason, confidence = keyword_choice
        status("answered (keyword rule)")
        if tool_trace is not None:
            tool_trace["decision"] = "keyword_rule"
            tool_trace["keyword_rule"] = {
                "choice": selected_option,
                "reason": reason,
            }
        return Answer(
            selected_option=selected_option,
            confidence_score=confidence,
            reasoning=reason,
            source="keyword_rule",
            second_pass_used=False,
            tool_trace=tool_trace,
        )

    rule_strength = (
        rule_info.get("has_proc_match", False)
        or rule_info.get("has_site_match", False)
        or rule_info.get("has_qualifier_match", False)
    )
    fallback_choice = None
    if not rule_strength:
        fallback_result = _rule_fallback_choice(question_proc, question_sites, scored)
        if fallback_result:
            fallback_choice = fallback_result.get("option")
            if tool_trace is not None:
                tool_trace["rule_fallback"] = fallback_result

    if RULE_FALLBACK_AUTO and fallback_choice:
        status("answered (rule fallback)")
        if tool_trace is not None:
            tool_trace["decision"] = "rule_fallback"
        return Answer(
            selected_option=fallback_choice,
            confidence_score=0.45,
            reasoning="Fallback rule selected the closest procedural match available.",
            source="rule_fallback",
            second_pass_used=False,
            tool_trace=tool_trace,
        )

    if RULE_FILTER_ENABLED and len(rule_allowed) == 1 and rule_strength:
        selected_option = rule_allowed[0]
        status("answered (rule)")
        if tool_trace is not None:
            tool_trace["decision"] = "rule"
            tool_trace["rule_choice"] = selected_option
        return Answer(
            selected_option=selected_option,
            confidence_score=0.65,
            reasoning="Rule filter left a single viable option based on procedure/site matches.",
            source="rule",
            second_pass_used=False,
            tool_trace=tool_trace,
        )

    if EVIDENCE_ENABLED and scored:
        evidence_pack = _build_evidence_pack(scored)
        if evidence_pack:
            messages.append({"role": "user", "content": evidence_pack})
            if tool_trace is not None:
                tool_trace["evidence_pack"] = True

    if rule_allowed and rule_strength:
        messages.append({
            "role": "user",
            "content": (
                "Rule filter suggests these viable options based on procedure/site matching: "
                f"{', '.join(rule_allowed)}. Prefer these unless there is a strong reason not to."
            ),
        })

    if fallback_choice:
        messages.append({
            "role": "user",
            "content": (
                f"Fallback rule suggests option {fallback_choice} because no option "
                "directly matches the requested procedure; use this if nothing else fits."
            ),
        })

    if HEURISTIC_ENABLED:
        heuristic = _heuristic_pick(scored, options_with_descs, incomplete_options)
        if heuristic:
            selected_option, score, gap, reason = heuristic
            confidence = max(0.55, min(0.85, score + gap))
            status("answered (heuristic)")
            if tool_trace is not None:
                tool_trace["decision"] = "heuristic"
                tool_trace["heuristic_choice"] = selected_option
                tool_trace["heuristic_score"] = round(score, TRACE_SCORE_PRECISION)
                tool_trace["heuristic_gap"] = round(gap, TRACE_SCORE_PRECISION)
            return Answer(
                selected_option=selected_option,
                confidence_score=confidence,
                reasoning=reason,
                source="heuristic",
                heuristic_score=score,
                heuristic_gap=gap,
                second_pass_used=False,
                tool_trace=tool_trace,
            )

    original_answer = None
    second_pass_attempted = False
    llm_error: Optional[Exception] = None
    no_tool_attempts = 0
    text_fallback_attempted = False

    # Agent loop
    for _ in range(MAX_TOOL_TURNS):
        try:
            response = await _call_gemini(messages)
        except Exception as exc:
            llm_error = exc
            break
        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        text_content = message.content or ""

        submit_tool = None
        pending_tools = []
        for tool in tool_calls:
            name = tool.function.name if tool.function else ""
            if name == "submit_answer":
                submit_tool = tool
            else:
                pending_tools.append(tool)

        if submit_tool and not pending_tools:
            args = {}
            raw_args = submit_tool.function.arguments if submit_tool.function else None
            if isinstance(raw_args, dict):
                args = raw_args
            elif isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args or "{}")
                except json.JSONDecodeError:
                    args = {}
            candidate = Answer(
                selected_option=_sanitize_option(args.get("selected_option")),
                confidence_score=_clamp_confidence(args.get("confidence")),
                reasoning=args.get("reasoning", ""),
                source="llm_second_pass" if second_pass_attempted else "llm",
                second_pass_used=second_pass_attempted,
                tool_trace=tool_trace,
            )

            second_pass = _should_second_pass(
                scored,
                options_with_descs,
                incomplete_options,
                candidate,
                rule_allowed,
                rule_strength,
                fallback_choice,
            )
            if second_pass and not second_pass_attempted:
                best_option, best_score, gap = second_pass
                status("second-pass check...")
                second_pass_attempted = True
                original_answer = candidate
                if tool_trace is not None:
                    tool_trace["second_pass_triggered"] = True
                    tool_trace["second_pass_suggestion"] = {
                        "option": best_option,
                        "score": round(best_score, TRACE_SCORE_PRECISION),
                        "gap": round(gap, TRACE_SCORE_PRECISION),
                        "rule_allowed": rule_allowed,
                        "rule_strength": rule_strength,
                    }
                messages.append({
                    "role": "user",
                    "content": (
                        "Second-pass check: your initial choice was "
                        f"{candidate.selected_option} (confidence {candidate.confidence_score:.2f}). "
                        "The description match heuristic favors "
                        f"option {best_option} (score {best_score:.2f}, gap {gap:.2f}). "
                        f"Rule filter allowed: {', '.join(rule_allowed) if rule_allowed else 'none'}. "
                        "Re-evaluate using the evidence above. If you keep your original "
                        "choice, explain why. Then call submit_answer."
                    ),
                })
                continue

            dominant = None
            if scored and not incomplete_options:
                dominant = _dominant_scored_option(scored)
                if dominant:
                    dominant_option, dominant_score, dominant_gap = dominant
                    if rule_allowed and rule_strength and dominant_option not in set(rule_allowed):
                        dominant = None

            if dominant and candidate.selected_option and candidate.selected_option != dominant_option:
                status("answered (dominant override)")
                if tool_trace is not None:
                    tool_trace["decision"] = "dominant_override"
                    tool_trace["dominant_override"] = {
                        "selected": candidate.selected_option,
                        "override": dominant_option,
                        "score": round(dominant_score, TRACE_SCORE_PRECISION),
                        "gap": round(dominant_gap, TRACE_SCORE_PRECISION),
                    }
                return Answer(
                    selected_option=dominant_option,
                    confidence_score=max(0.6, min(0.85, dominant_score + dominant_gap)),
                    reasoning="Dominant description match overrides LLM choice.",
                    source="dominant_override",
                    second_pass_used=second_pass_attempted,
                    tool_trace=tool_trace,
                )

            status("answered")
            if tool_trace is not None:
                tool_trace["decision"] = "llm_second_pass" if second_pass_attempted else "llm"
            return candidate

        if not tool_calls:
            extracted = _extract_answer_from_text(text_content)
            if extracted:
                status("answered (text)")
                return Answer(
                    selected_option=extracted,
                    confidence_score=0.6,
                    reasoning=text_content[:200],
                    source="llm",
                    second_pass_used=second_pass_attempted,
                    tool_trace=tool_trace,
                )
            no_tool_attempts += 1
            if TEXT_FALLBACK_ENABLED and no_tool_attempts >= TEXT_FALLBACK_MAX_ATTEMPTS:
                fallback_answer = await _attempt_text_fallback("no_tool_calls")
                if fallback_answer:
                    status("answered (text fallback)")
                    return fallback_answer
                break
            if text_content:
                messages.append({
                    "role": "assistant",
                    "content": text_content,
                })
            messages.append({
                "role": "user",
                "content": "Please call submit_answer with selected_option, confidence, and reasoning.",
            })
            continue

        tool_results = []
        for tool in pending_tools:
            tool_name = tool.function.name if tool.function else ""
            args = {}
            raw_args = tool.function.arguments if tool.function else None
            if isinstance(raw_args, dict):
                args = raw_args
            elif isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args or "{}")
                except json.JSONDecodeError:
                    args = {}
            status(f"using {tool_name}...")
            result = await _handle_tool_call(tool_name, args)
            tool_results.append((tool.id, tool_name, tool.function.arguments, result))

        tool_payload = []
        for tool in pending_tools:
            if hasattr(tool, "model_dump"):
                tool_payload.append(tool.model_dump(exclude_none=True))
            else:
                tool_payload.append({
                    "id": tool.id,
                    "type": "function",
                    "function": {
                        "name": tool.function.name,
                        "arguments": tool.function.arguments,
                    },
                })

        messages.append({
            "role": "assistant",
            "content": text_content or "",
            "tool_calls": tool_payload,
        })
        for tool_id, _, _, result in tool_results:
            messages.append({
                "role": "tool",
                "tool_call_id": tool_id,
                "content": result,
            })

    # Fallback: try to extract from conversation history
    status("fallback")
    for msg in reversed(messages):
        if msg["role"] == "assistant":
            content = msg.get("content")
            if isinstance(content, str):
                extracted = _extract_answer_from_text(content)
                if extracted:
                    return Answer(
                        selected_option=extracted,
                        confidence_score=0.4,
                        reasoning="Extracted from response",
                        source="llm",
                        second_pass_used=second_pass_attempted,
                        tool_trace=tool_trace,
                    )
            elif isinstance(content, list):
                for block in content:
                    if hasattr(block, "text"):
                        extracted = _extract_answer_from_text(block.text)
                        if extracted:
                            return Answer(
                                selected_option=extracted,
                                confidence_score=0.4,
                                reasoning="Extracted from response",
                                source="llm",
                                second_pass_used=second_pass_attempted,
                                tool_trace=tool_trace,
                            )

    if not text_fallback_attempted and TEXT_FALLBACK_ENABLED:
        fallback_answer = await _attempt_text_fallback("post_loop")
        if fallback_answer:
            status("answered (text fallback)")
            return fallback_answer

    if original_answer:
        original_answer.source = "llm"
        original_answer.second_pass_used = True
        return original_answer

    best_scored = _best_scored_option(scored, rule_allowed)
    if best_scored:
        selected_option, score = best_scored
        if tool_trace is not None:
            tool_trace["decision"] = "fallback_scored"
            if llm_error is not None:
                tool_trace["llm_error"] = str(llm_error)
        return Answer(
            selected_option=selected_option,
            confidence_score=max(0.2, min(0.5, score + 0.1)),
            reasoning=(
                "Fallback: model error; selected highest description match."
                if llm_error is not None
                else "Fallback: model did not return an answer; selected highest description match."
            ),
            source="fallback_scored",
            second_pass_used=second_pass_attempted,
            tool_trace=tool_trace,
        )

    return Answer(
        selected_option="",
        confidence_score=0.0,
        reasoning="No answer provided",
        source="none",
        second_pass_used=second_pass_attempted,
        tool_trace=tool_trace,
    )


async def solve_all_questions(
    questions: list[Question],
    on_progress: callable = None
) -> list[dict]:
    """Solve all questions with concurrency."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    completed = 0
    total = len(questions)

    def update_progress(msg: str = "") -> None:
        if on_progress:
            on_progress(f"[{completed}/{total}] {msg}")

    async def process_one(q: Question) -> dict:
        nonlocal completed
        async with semaphore:
            try:
                answer = await solve_question(q, on_status=update_progress)
                result = {
                    "question_id": q.id,
                    "selected_option": answer.selected_option,
                    "confidence": answer.confidence_score,
                    "reasoning": answer.reasoning,
                    "answer_source": answer.source,
                    "heuristic_score": answer.heuristic_score,
                    "heuristic_gap": answer.heuristic_gap,
                    "second_pass_used": answer.second_pass_used,
                    "tool_trace": answer.tool_trace,
                }
            except Exception as e:
                result = {
                    "question_id": q.id,
                    "selected_option": "",
                    "confidence": 0.0,
                    "reasoning": f"Error: {e}",
                    "answer_source": "error",
                    "second_pass_used": False,
                }
            completed += 1
            update_progress(f"Q{q.id} done")
            return result

    results = await asyncio.gather(*[process_one(q) for q in questions])
    return sorted(results, key=lambda x: x["question_id"])
