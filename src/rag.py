"""
Medical Code Retrieval System (RAG)

Retrieves medical code descriptions from multiple sources:
1. SQLite database (primary)
2. Local flat files (ICD-10, HCPCS, CPT)
3. Supplementary JSON file
"""

import json
import os
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class CodeCache:
    """Centralized cache for all code sources."""
    icd10: dict[str, str] = field(default_factory=dict)
    hcpcs: dict[str, str] = field(default_factory=dict)
    cpt: dict[str, str] = field(default_factory=dict)
    supplement: dict[str, str] = field(default_factory=dict)
    db_conn: Optional[sqlite3.Connection] = None
    db_cache: dict[tuple, str] = field(default_factory=dict)
    loaded: set[str] = field(default_factory=set)


# Global cache instance
_cache = CodeCache()


def _get_codes_dir() -> Path:
    """Get the codes directory path."""
    base_dir = Path(__file__).resolve().parents[1]
    return Path(os.getenv("MEDICAL_CODES_DIR", str(base_dir / "codes")))


def _normalize_code(code: str) -> str:
    """Normalize a code to uppercase alphanumeric only."""
    return re.sub(r"[^A-Z0-9]", "", code.upper())


def _is_cpt_code(code: str) -> bool:
    """Check if a code is a CPT code (5 digits or 4 digits + letter)."""
    normalized = _normalize_code(code)
    return bool(re.fullmatch(r"\d{5}|\d{4}[A-Z]", normalized))


def _classify_code(code: str) -> str:
    """Classify a code into its coding system."""
    normalized = _normalize_code(code)
    if "." in code:
        return "ICD10CM"
    if re.fullmatch(r"[A-Z][0-9]{4}", normalized):
        return "HCPCS"
    if re.fullmatch(r"\d{5}|\d{4}[A-Z]", normalized):
        return "CPT"
    if re.fullmatch(r"[A-Z][0-9]{2}[A-Z0-9]{0,4}", normalized):
        return "ICD10CM"
    return "UNKNOWN"


# =============================================================================
# Loaders - Each loads data into the global cache
# =============================================================================

def _load_sqlite_db() -> None:
    """Load SQLite database connection."""
    if "sqlite" in _cache.loaded:
        return
    _cache.loaded.add("sqlite")

    db_path = Path(os.getenv("CODE_DB_PATH", str(_get_codes_dir() / "codes.sqlite")))
    if db_path.exists():
        try:
            _cache.db_conn = sqlite3.connect(db_path)
        except Exception:
            pass


def _load_icd10_codes() -> None:
    """Load ICD-10-CM codes from flat file."""
    if "icd10" in _cache.loaded:
        return
    _cache.loaded.add("icd10")

    path = _get_codes_dir() / "icd10cm-codes-2026.txt"
    if not path.exists():
        return

    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                parts = line.strip().split(None, 1)
                if len(parts) == 2:
                    code, desc = parts[0].upper(), parts[1].strip()
                    if code and desc:
                        _cache.icd10[code] = desc
    except Exception:
        pass


def _load_hcpcs_codes() -> None:
    """Load HCPCS codes from flat file."""
    if "hcpcs" in _cache.loaded:
        return
    _cache.loaded.add("hcpcs")

    path = _get_codes_dir() / "HCPC2026_JAN_ANWEB_01122026.txt"
    if not path.exists():
        return

    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                if len(line) < 9:
                    continue
                code = line[:5].strip().upper()
                if not re.fullmatch(r"[A-Z][0-9]{4}", code):
                    continue
                # Parse fixed-width format
                offset = 11 if len(line) >= 11 and line[8:11].isdigit() else 8
                desc = line[offset:offset + 80].strip()
                if code and desc:
                    _cache.hcpcs[code] = desc
    except Exception:
        pass


def _load_cpt_codes() -> None:
    """Load CPT codes from optional flat files."""
    if "cpt" in _cache.loaded:
        return
    _cache.loaded.add("cpt")

    codes_dir = _get_codes_dir()
    txt_path = codes_dir / "cpt_codes.txt"
    json_path = codes_dir / "cpt_codes.json"

    if txt_path.exists():
        try:
            with txt_path.open("r", encoding="utf-8", errors="replace") as f:
                for line in f:
                    parts = line.strip().split(None, 1)
                    if len(parts) == 2:
                        code, desc = parts[0].upper(), parts[1].strip()
                        normalized = _normalize_code(code)
                        if normalized and desc:
                            _cache.cpt[normalized] = desc
        except Exception:
            pass

    if json_path.exists():
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                for code, desc in data.items():
                    normalized = _normalize_code(str(code))
                    if normalized and desc:
                        _cache.cpt[normalized] = str(desc)
        except Exception:
            pass


def _load_supplement_codes() -> None:
    """Load supplementary codes from JSON file."""
    if "supplement" in _cache.loaded:
        return
    _cache.loaded.add("supplement")

    path = _get_codes_dir() / "cpt_supplement.json"
    if not path.exists():
        return

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            _cache.supplement = {str(k).upper(): str(v) for k, v in data.items()}
    except Exception:
        pass

# =============================================================================
# Lookup Functions
# =============================================================================

def _lookup_sqlite(code: str) -> Optional[str]:
    """Look up a code in the SQLite database."""
    if not _cache.db_conn:
        return None

    normalized = _normalize_code(code)
    system = _classify_code(code)
    cache_key = (normalized, system)

    if cache_key in _cache.db_cache:
        return _cache.db_cache[cache_key] or None

    try:
        if system != "UNKNOWN":
            row = _cache.db_conn.execute(
                "SELECT description FROM code_best WHERE code_norm=? AND system=? LIMIT 1",
                (normalized, system),
            ).fetchone()
        else:
            row = _cache.db_conn.execute(
                "SELECT description FROM code_best WHERE code_norm=? ORDER BY priority DESC LIMIT 1",
                (normalized,),
            ).fetchone()

        desc = row[0] if row else None
        if desc is None and system == "ICD10CM":
            try:
                row = _cache.db_conn.execute(
                    "SELECT description FROM code_best "
                    "WHERE system=? AND ? LIKE code_norm || '%' "
                    "ORDER BY LENGTH(code_norm) DESC LIMIT 1",
                    (system, normalized),
                ).fetchone()
                desc = row[0] if row else None
            except Exception:
                desc = None
        _cache.db_cache[cache_key] = desc
        return desc
    except Exception:
        return None


def _lookup_flat_files(code: str) -> Optional[str]:
    """Look up a code in the flat file caches."""
    normalized = _normalize_code(code)
    code_upper = code.upper()

    # CPT codes
    if _is_cpt_code(code):
        if normalized in _cache.cpt:
            return _cache.cpt[normalized]
        return None

    # ICD-10 codes
    if _cache.icd10:
        if code_upper in _cache.icd10:
            return _cache.icd10[code_upper]
        if code_upper.replace(".", "") in _cache.icd10:
            return _cache.icd10[code_upper.replace(".", "")]
        prefix_desc = _lookup_icd10_prefix(code)
        if prefix_desc:
            return prefix_desc

    # HCPCS codes (letter + 4 digits)
    if re.fullmatch(r"[A-Z][0-9]{4}", normalized):
        if normalized in _cache.hcpcs:
            return _cache.hcpcs[normalized]

    return None


def _lookup_icd10_prefix(code: str) -> Optional[str]:
    """Try prefix matches for ICD-10 codes."""
    normalized = _normalize_code(code)
    if len(normalized) <= 3:
        return None
    for i in range(len(normalized) - 1, 2, -1):
        prefix = normalized[:i]
        if prefix in _cache.icd10:
            return _cache.icd10[prefix]
        if i > 3:
            dotted = prefix[:3] + "." + prefix[3:]
            if dotted in _cache.icd10:
                return _cache.icd10[dotted]
    return None


def _lookup_supplement(code: str) -> Optional[str]:
    """Look up a code in the supplement file."""
    return _cache.supplement.get(_normalize_code(code))


# =============================================================================
# Main Retriever Class
# =============================================================================

class MedicalCodeRetriever:
    """
    Retrieves medical code descriptions using a multi-source fallback strategy.

    Lookup order:
    1. SQLite database (pre-built from UMLS)
    2. Local flat files (ICD-10, HCPCS)
    3. Supplementary JSON file
    """

    async def retrieve_codes(self, codes: list[str]) -> dict[str, str]:
        """
        Retrieve descriptions for a list of medical codes.

        Args:
            codes: List of code strings (CPT, ICD-10, HCPCS)

        Returns:
            Dict mapping codes to descriptions
        """
        # Load all data sources
        _load_sqlite_db()
        _load_icd10_codes()
        _load_hcpcs_codes()
        _load_cpt_codes()
        _load_supplement_codes()

        results = {}
        missing = []

        # First pass: try local sources
        for code in codes:
            desc = self._lookup_local(code)
            if desc:
                results[code] = desc
            else:
                missing.append(code)

        # Final fallback: unavailable description
        for code in missing:
            results[code] = self._fallback_description(code)

        return results

    def _lookup_local(self, code: str) -> Optional[str]:
        """Try all local lookup sources."""
        # SQLite first (most comprehensive)
        desc = _lookup_sqlite(code)
        if desc:
            return desc

        # Flat files
        desc = _lookup_flat_files(code)
        if desc:
            return desc

        # Supplement
        desc = _lookup_supplement(code)
        if desc:
            return desc

        return None

    def _fallback_description(self, code: str) -> str:
        """Generate a fallback description for unknown codes."""
        normalized = _normalize_code(code)
        if _is_cpt_code(code):
            return f"CPT code {normalized} (description unavailable)"
        return f"Description unavailable for code {normalized}"
