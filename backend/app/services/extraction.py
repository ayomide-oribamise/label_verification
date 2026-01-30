"""Field extraction service using OCR results.

Enhanced brand extraction using:
1. Regex patterns to find brand phrases ending in DISTILLERY/BREWING/WINERY
2. Token-set similarity for order-insensitive matching
3. Fallback to spatial analysis if regex fails
"""

import re
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
import logging

from .ocr import OCRResult, OCRBox, FieldTargetedResult
from ..config import get_settings

logger = logging.getLogger(__name__)


# Standard government warning text (canonical form)
GOVERNMENT_WARNING_CANONICAL = (
    "GOVERNMENT WARNING: (1) ACCORDING TO THE SURGEON GENERAL, WOMEN SHOULD NOT "
    "DRINK ALCOHOLIC BEVERAGES DURING PREGNANCY BECAUSE OF THE RISK OF BIRTH DEFECTS. "
    "(2) CONSUMPTION OF ALCOHOLIC BEVERAGES IMPAIRS YOUR ABILITY TO DRIVE A CAR OR "
    "OPERATE MACHINERY, AND MAY CAUSE HEALTH PROBLEMS."
)

# Brand suffix keywords - brands often end with these
BRAND_SUFFIX_PATTERN = r"(DISTILLERY|DISTILLING|BREWING|BREWERY|WINERY|VINEYARDS?|COMPANY|CO\.?|INC\.?|LLC|CELLARS?|SPIRITS?)"

# Beverage type keywords for class/type detection
# Includes spirits, beer styles, and wine varietals
BEVERAGE_KEYWORDS = {
    # Spirits
    "whiskey", "whisky", "bourbon", "scotch", "rye",
    "vodka", "gin", "rum", "tequila", "mezcal",
    "brandy", "cognac", "armagnac",
    "liqueur", "cordial", "schnapps",
    "sake", "soju", "baijiu",
    
    # Beer styles
    "beer", "ale", "lager", "stout", "porter", "ipa",
    "pilsner", "hefeweizen", "wheat", "saison", "kolsch",
    "amber", "pale ale", "india pale ale",
    
    # Wine types
    "wine", "champagne", "prosecco", "cava", "sparkling",
    "red wine", "white wine", "rose", "rosé",
    
    # Wine varietals (CRITICAL for wine label detection)
    "cabernet", "cabernet sauvignon", "merlot", "pinot noir",
    "pinot", "syrah", "shiraz", "malbec", "zinfandel",
    "chardonnay", "sauvignon blanc", "riesling", "pinot grigio",
    "pinot gris", "moscato", "gewurztraminer", "viognier",
    "tempranillo", "sangiovese", "nebbiolo", "barbera",
    "grenache", "mourvedre", "petite sirah", "carmenere",
    "chenin blanc", "semillon", "gruner veltliner",
    "albarino", "verdejo", "torrontes",
    
    # Dessert wines
    "port", "sherry", "madeira", "vermouth",
    "ice wine", "late harvest", "sauternes",
    
    # Cider
    "cider", "hard cider", "perry",
}

# Wine varietal patterns (for regex matching)
WINE_VARIETAL_PATTERNS = [
    r"cabernet\s+sauvignon",
    r"sauvignon\s+blanc",
    r"pinot\s+noir",
    r"pinot\s+grigio",
    r"pinot\s+gris",
    r"chenin\s+blanc",
    r"gruner\s+veltliner",
    r"late\s+harvest",
    r"india\s+pale\s+ale",
    r"pale\s+ale",
]

# Marketing fluff to ignore when extracting brand
MARKETING_KEYWORDS = {
    "premium", "reserve", "select", "special", "limited",
    "aged", "barrel", "cask", "small batch", "single barrel",
    "handcrafted", "artisan", "craft", "estate", "vintage",
    "original", "classic", "traditional", "authentic",
}

# Stop words to ignore in token matching
BRAND_STOP_WORDS = {"THE", "AND", "OF", "BY"}

# Brand suffixes for candidate scoring
BRAND_SUFFIXES = [
    "DISTILLERY", "DISTILLING", "BREWING", "BREWERY", "WINERY", 
    "VINEYARDS", "VINEYARD", "CELLARS", "CELLAR", "SPIRITS",
    "CO", "COMPANY", "INC", "LLC", "LTD"
]

# =============================================================================
# CANDIDATE-BASED EXTRACTION HELPERS
# =============================================================================
# These functions implement "best match" extraction that's robust to OCR garbling
# by scoring candidates against expected values using similarity + token overlap.
# =============================================================================

def _ocr_normalize(s: str) -> str:
    """
    Aggressive normalization for OCR text comparison.
    Handles common OCR confusions and strips noise.
    """
    if not s:
        return ""
    s = s.upper()
    # Strip punctuation (OCR often adds spurious brackets, periods)
    s = re.sub(r"[^A-Z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    # Common OCR character confusions
    s = s.replace("0", "O")  # zero -> O
    s = s.replace("1", "I")  # one -> I
    s = s.replace("5", "S")  # five -> S
    s = s.replace("8", "B")  # eight -> B
    return s


def _similarity_ratio(a: str, b: str) -> float:
    """Sequence similarity ratio (0-1)."""
    if not a or not b:
        return 0.0
    from difflib import SequenceMatcher
    return SequenceMatcher(None, a, b).ratio()


def _token_overlap(a: str, b: str) -> float:
    """Token set overlap score (0-1). Handles word reordering."""
    ta = set(a.split())
    tb = set(b.split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _contains_brand_suffix(s: str) -> bool:
    """Check if string contains a brand suffix (DISTILLERY, BREWING, etc.)."""
    s_upper = " " + s.upper() + " "
    return any((" " + suf + " ") in s_upper or s_upper.endswith(" " + suf + " ") 
               for suf in BRAND_SUFFIXES)


def _build_brand_candidates(boxes: List['OCRBox'], image_height: int) -> List[str]:
    """
    Build brand candidate strings from OCR boxes.
    
    Candidates include:
    - All boxes in top 45% of image
    - Largest text boxes (brands are usually prominent)
    - Any box containing suffix patterns (DISTILLERY, BREWING, etc.)
    - Line strings from grouping adjacent boxes
    """
    if not boxes:
        return []
    
    # Determine image bounds from boxes
    min_y = min(b.top for b in boxes)
    max_y = max(b.bottom for b in boxes)
    used_h = max(1, max_y - min_y)
    
    # Focus on upper 45% of detected text area
    top_threshold = min_y + int(used_h * 0.45)
    top_boxes = [b for b in boxes if b.center_y <= top_threshold]
    
    if not top_boxes:
        # If no boxes in top region, use all boxes
        top_boxes = boxes[:10]
    
    candidates = []
    
    # 1. Largest boxes (by height) - brands are usually big text
    largest = sorted(top_boxes, key=lambda b: b.height, reverse=True)[:8]
    for b in largest:
        if b.text.strip():
            candidates.append(b.text.strip())
    
    # 2. Boxes containing suffix patterns
    for b in top_boxes:
        if _contains_brand_suffix(b.text) and b.text.strip() not in candidates:
            candidates.append(b.text.strip())
    
    # 3. Build line strings by grouping boxes into reading order
    if top_boxes:
        import numpy as np
        line_h = int(np.median([b.height for b in top_boxes]))
        line_h = max(12, min(60, line_h))
    else:
        line_h = 20
    
    # Sort by line (y bucket) then left-to-right
    sorted_boxes = sorted(top_boxes, key=lambda b: (b.top // line_h, b.left))
    
    # Group into lines
    lines = []
    current_line = []
    current_key = None
    
    for b in sorted_boxes:
        key = b.top // line_h
        if current_key is None or key == current_key:
            current_line.append(b.text)
            current_key = key
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [b.text]
            current_key = key
    
    if current_line:
        lines.append(" ".join(current_line))
    
    # Add lines as candidates
    for line in lines:
        if line.strip() and line.strip() not in candidates:
            candidates.append(line.strip())
    
    # De-duplicate by normalized form
    seen = set()
    unique = []
    for c in candidates:
        n = _ocr_normalize(c)
        if n and len(n) >= 2 and n not in seen:
            seen.add(n)
            unique.append(c)
    
    return unique


def _score_brand_candidate(candidate: str, expected: str) -> Tuple[float, str]:
    """
    Score a brand candidate against expected value.
    
    Returns (score, reason_string).
    
    Scoring components:
    - 65% sequence similarity
    - 25% token overlap (handles word reordering)
    - 6% bonus for suffix (DISTILLERY, BREWING, etc.)
    - 10% bonus if candidate tokens are subset of expected
    - Penalties for OCR garbage (brackets, too many digits)
    """
    cn = _ocr_normalize(candidate)
    exp = _ocr_normalize(expected)
    
    if len(cn) < 2:
        return 0.0, "too_short"
    
    # HARD PENALTY: brackets/noise = definitely not a real brand
    if re.search(r'[\[\]{}<>]', candidate):
        return 0.0, "has_brackets"
    
    # PENALTY: too digit-heavy
    alnum = re.sub(r'[^A-Z0-9]', '', cn)
    if alnum:
        digit_ratio = sum(ch.isdigit() for ch in alnum) / len(alnum)
        if digit_ratio > 0.30:
            return 0.0, "too_many_digits"
    
    # Core similarity scores
    sim = _similarity_ratio(cn, exp)
    tok = _token_overlap(cn, exp)
    
    # Bonus: contains brand suffix
    suffix_bonus = 0.06 if _contains_brand_suffix(cn) else 0.0
    
    # Bonus: candidate tokens are subset of expected
    # e.g., "TOM OLD" matches "OLD TOM DISTILLERY"
    cand_tokens = set(cn.split())
    exp_tokens = set(exp.split())
    subset_bonus = 0.10 if cand_tokens and cand_tokens.issubset(exp_tokens) else 0.0
    
    # Bonus: expected tokens are subset of candidate
    # e.g., "OLD TOM DISTILLERY PREMIUM" contains "OLD TOM DISTILLERY"
    superset_bonus = 0.08 if exp_tokens and exp_tokens.issubset(cand_tokens) else 0.0
    
    score = 0.65 * sim + 0.25 * tok + suffix_bonus + max(subset_bonus, superset_bonus)
    
    reason = f"sim={sim:.2f} tok={tok:.2f} suf={suffix_bonus:.2f} sub={max(subset_bonus, superset_bonus):.2f}"
    
    return score, reason


def pick_best_brand_candidate(
    candidates: List[str],
    expected_brand: str
) -> Tuple[Optional[str], float, str]:
    """
    Pick the best brand candidate given expected value.
    
    Uses similarity scoring that's robust to OCR garbling.
    
    Returns:
        (best_candidate, score, reason)
    """
    if not candidates or not expected_brand:
        return None, 0.0, "no_candidates"
    
    best = None
    best_score = 0.0
    best_reason = "no_match"
    
    for c in candidates:
        score, reason = _score_brand_candidate(c, expected_brand)
        if score > best_score:
            best_score = score
            best = c
            best_reason = reason
    
    return best, best_score, best_reason


def _build_class_type_candidates(
    boxes: List['OCRBox'],
    raw_text: str,
    image_height: int
) -> List[str]:
    """
    Build class/type candidates from OCR results.
    
    Searches for:
    - Boxes containing beverage keywords
    - Multi-word varietal patterns in raw text
    - Lines in upper-middle region (15-60%)
    """
    candidates = []
    
    # 1. Search raw text for multi-word patterns (varietals, beer styles)
    for pattern in WINE_VARIETAL_PATTERNS:
        match = re.search(pattern, raw_text, re.IGNORECASE)
        if match:
            candidates.append(match.group(0).title())
    
    # 2. Search raw text for single keywords
    raw_lower = raw_text.lower()
    for kw in sorted(BEVERAGE_KEYWORDS, key=len, reverse=True):
        if kw in raw_lower:
            # Find the actual text around this keyword
            idx = raw_lower.find(kw)
            # Extract surrounding context
            start = max(0, idx)
            end = min(len(raw_text), idx + len(kw) + 20)
            context = raw_text[start:end].strip()
            # Clean up
            words = context.split()[:4]
            candidate = " ".join(words)
            if candidate and candidate not in candidates:
                candidates.append(candidate)
    
    # 3. Boxes containing beverage keywords
    if boxes:
        min_y = min(b.top for b in boxes)
        max_y = max(b.bottom for b in boxes)
        used_h = max(1, max_y - min_y)
        
        # Focus on upper-middle region (15-60%)
        y_start = min_y + int(used_h * 0.15)
        y_end = min_y + int(used_h * 0.60)
        
        for b in boxes:
            if y_start <= b.center_y <= y_end:
                text_lower = b.text.lower()
                for kw in BEVERAGE_KEYWORDS:
                    if kw in text_lower and b.text.strip() not in candidates:
                        candidates.append(b.text.strip())
                        break
    
    # De-duplicate
    seen = set()
    unique = []
    for c in candidates:
        n = _ocr_normalize(c)
        if n and n not in seen:
            seen.add(n)
            unique.append(c)
    
    return unique


def _score_class_type_candidate(candidate: str, expected: str) -> Tuple[float, str]:
    """
    Score a class/type candidate against expected value.
    
    Similar to brand scoring but with beverage-specific bonuses.
    """
    cn = _ocr_normalize(candidate)
    exp = _ocr_normalize(expected)
    
    if len(cn) < 2:
        return 0.0, "too_short"
    
    sim = _similarity_ratio(cn, exp)
    tok = _token_overlap(cn, exp)
    
    # Bonus: contains a known beverage keyword
    has_keyword = any(kw.upper() in cn for kw in BEVERAGE_KEYWORDS if len(kw) > 3)
    keyword_bonus = 0.08 if has_keyword else 0.0
    
    # Bonus: exact keyword match
    exact_bonus = 0.15 if cn == exp else 0.0
    
    score = 0.60 * sim + 0.30 * tok + keyword_bonus + exact_bonus
    
    reason = f"sim={sim:.2f} tok={tok:.2f} kw={keyword_bonus:.2f}"
    
    return score, reason


def _has_beverage_keyword(text: str) -> bool:
    """
    HARD GATE: class/type must contain a beverage keyword OR match a varietal pattern.
    
    This prevents brand-like text ("Distillery Old Tom...") from being selected
    as class/type when the real type ("KENTUCKY STRAIGHT BOURBON WHISKEY") is present.
    """
    t = " " + text.upper() + " "
    for kw in BEVERAGE_KEYWORDS:
        kw_u = " " + kw.upper() + " "
        if kw_u in t:
            return True
    return False


def _is_contaminated_by_brand(candidate: str, brand_value: Optional[str]) -> bool:
    """
    Reject candidates that are basically brand/marketing lines.
    
    Prevents "Distillery Old Tom Premium Small Batch" from being selected
    as class/type when it's clearly brand text.
    """
    if not candidate:
        return True
    cn = _ocr_normalize(candidate)

    # Contains producer suffix -> likely not class/type
    if _contains_brand_suffix(cn):
        return True

    # Marketing-heavy (2+ fluff words)
    fluff = sum(1 for w in cn.lower().split() if w in MARKETING_KEYWORDS)
    if fluff >= 2:
        return True

    # Overlaps heavily with brand tokens (>= 60%)
    if brand_value:
        bn = _ocr_normalize(brand_value)
        if _token_overlap(cn, bn) >= 0.60:
            return True

    return False


def pick_best_class_type_candidate(
    candidates: List[str],
    expected_class_type: str
) -> Tuple[Optional[str], float, str]:
    """
    Pick the best class/type candidate given expected value.
    
    Returns:
        (best_candidate, score, reason)
    """
    if not candidates or not expected_class_type:
        return None, 0.0, "no_candidates"
    
    best = None
    best_score = 0.0
    best_reason = "no_match"
    
    for c in candidates:
        score, reason = _score_class_type_candidate(c, expected_class_type)
        if score > best_score:
            best_score = score
            best = c
            best_reason = reason
    
    return best, best_score, best_reason


@dataclass
class ExtractedField:
    """Result of extracting a single field."""
    value: Optional[str]
    confidence: float
    source_boxes: List[OCRBox] = field(default_factory=list)
    extraction_method: str = "unknown"
    notes: str = ""
    # Candidates for "best match" rescoring during verification
    candidates: List[str] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """Complete extraction results from a label."""
    brand_name: ExtractedField
    class_type: ExtractedField
    abv_percent: ExtractedField
    net_contents_ml: ExtractedField
    government_warning: ExtractedField
    raw_text: str
    overall_confidence: float


class FieldExtractor:
    """Extracts structured fields from OCR results."""
    
    def __init__(self):
        self.settings = get_settings()
        
        # ABV regex patterns (order matters - more specific first)
        self.abv_patterns = [
            # "45% Alc./Vol." or "45% ALC/VOL" or "45% alc by vol"
            r'(\d+(?:\.\d+)?)\s*%\s*(?:alc\.?/?vol\.?|alc/?vol|alc\s+by\s+vol)',
            # "45% Alcohol by Volume"
            r'(\d+(?:\.\d+)?)\s*%\s*alcohol\s+by\s+volume',
            # "90 Proof" -> convert to 45%
            r'(\d+)\s*proof',
            # Simple percentage at end "45%"
            r'(\d+(?:\.\d+)?)\s*%',
        ]
        
        # Net contents patterns (order matters - more specific first)
        # TIGHTENED: Use word boundaries to prevent garbage matches like "(055..."
        self.net_contents_patterns = [
            # "(355 mL)" exactly inside parentheses - most reliable
            (r'\(\s*(\d{2,4}(?:\.\d+)?)\s*(?:ML|mL|ml)\s*\)', 1.0),
            # "750 mL" with word boundaries (avoid accidental matches)
            (r'\b(\d{2,4}(?:\.\d+)?)\s*(?:ML|mL|ml)\b', 1.0),
            # Centiliters: "75 cL" with boundaries
            (r'\b(\d{1,3}(?:\.\d+)?)\s*(?:CL|cL|cl)\b', 10.0),
            # Liters: "1 L" / "1 liter" with boundaries
            (r'\b(\d+(?:\.\d+)?)\s*(?:L|l|LITER|liter|LITRE|litre)\b', 1000.0),
            # "12 FL OZ" / "12 OZ" - require OZ token boundary, FL optional
            (r'\b(\d{1,3}(?:\.\d+)?)\s*(?:FL\s*)?OZ\b', 29.5735),
        ]
        
        # Common bottle/can sizes for snapping (in mL)
        self.common_sizes_ml = [187, 200, 250, 330, 341, 355, 375, 440, 473, 500, 650, 700, 720, 750, 1000, 1500, 1750]
    
    def extract_all(self, ocr_result: OCRResult) -> ExtractionResult:
        """
        Extract all fields from OCR result.
        
        Args:
            ocr_result: Result from OCR processing
            
        Returns:
            ExtractionResult with all extracted fields
        """
        # Extract each field
        brand = self._extract_brand(ocr_result)
        class_type = self._extract_class_type(ocr_result, brand)
        abv = self._extract_abv(ocr_result)
        net_contents = self._extract_net_contents(ocr_result)
        warning = self._extract_government_warning(ocr_result)
        
        # Calculate overall confidence
        confidences = [
            brand.confidence,
            class_type.confidence,
            abv.confidence,
            net_contents.confidence,
            warning.confidence,
        ]
        # Filter out zero confidences (fields not found)
        valid_confidences = [c for c in confidences if c > 0]
        overall_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.0
        
        return ExtractionResult(
            brand_name=brand,
            class_type=class_type,
            abv_percent=abv,
            net_contents_ml=net_contents,
            government_warning=warning,
            raw_text=ocr_result.raw_text,
            overall_confidence=overall_confidence,
        )
    
    def extract_all_field_targeted(
        self, 
        field_result: FieldTargetedResult,
        ocr_result: OCRResult
    ) -> ExtractionResult:
        """
        Extract all fields using field-targeted OCR results.
        
        This is the enhanced extraction method that uses zone-specific text
        for better accuracy on beer/wine labels.
        
        Args:
            field_result: Result from field-targeted OCR with zone-specific text
            ocr_result: Standard OCR result for box-level analysis
            
        Returns:
            ExtractionResult with all extracted fields
        """
        # Extract brand from brand zone
        brand = self._extract_brand_from_zone(
            field_result.brand_text,
            field_result.brand_confidence,
            ocr_result
        )
        
        # Extract class/type - SEARCH ACROSS BRAND + TYPE ZONES COMBINED
        # Wine varietals often appear in brand zone (upper-mid), not just type zone
        combined_type_text = self._normalize_varietal_text(
            field_result.brand_text + " " + field_result.class_type_text
        )
        class_type = self._extract_class_type(
            ocr_result, 
            brand,
            zone_text=combined_type_text
        )
        
        # Extract ABV from ABV zone
        abv = self._extract_abv_from_zone(
            field_result.abv_text,
            field_result.abv_confidence,
            ocr_result
        )
        
        # Extract net contents from net_contents zone
        net_contents = self._extract_net_contents_from_zone(
            field_result.net_contents_text,
            field_result.net_contents_confidence,
            ocr_result
        )
        
        # Use keyword-based warning detection (from OCR layer)
        # This is more reliable than trying to parse dense warning text
        if field_result.warning_detected:
            warning = ExtractedField(
                value="detected",
                confidence=0.95,
                source_boxes=[],
                extraction_method="keyword_detection",
                notes="Government warning detected via keywords"
            )
        else:
            # Fall back to zone extraction
            warning = self._extract_warning_from_zone(
                field_result.warning_text,
                field_result.warning_confidence,
                ocr_result
            )
        
        # Calculate overall confidence
        confidences = [
            brand.confidence,
            class_type.confidence,
            abv.confidence,
            net_contents.confidence,
            warning.confidence,
        ]
        valid_confidences = [c for c in confidences if c > 0]
        overall_confidence = sum(valid_confidences) / len(valid_confidences) if valid_confidences else 0.0
        
        return ExtractionResult(
            brand_name=brand,
            class_type=class_type,
            abv_percent=abv,
            net_contents_ml=net_contents,
            government_warning=warning,
            raw_text=field_result.combined_raw_text,
            overall_confidence=overall_confidence,
        )
    
    def _normalize_varietal_text(self, text: str) -> str:
        """
        Normalize text for better varietal matching.
        
        OCR often produces:
        - "CABERNETSAUVIGNON" (no space)
        - "CABERNET   SAUVIGNON" (extra spaces)
        - "CABERNET SAUVIGNON NAPA VALLEY" (merged line)
        
        This normalizes for consistent matching.
        """
        if not text:
            return ""
        
        # Collapse multiple spaces
        normalized = re.sub(r'\s+', ' ', text.strip())
        
        # Try to add spaces in common "stuck together" varietals
        varietal_fixes = [
            ("CABERNETSAUVIGNON", "CABERNET SAUVIGNON"),
            ("SAUVIGNONBLANC", "SAUVIGNON BLANC"),
            ("PINOTNOIR", "PINOT NOIR"),
            ("PINOTGRIGIO", "PINOT GRIGIO"),
            ("PINOTGRIS", "PINOT GRIS"),
            ("CHENINBLANC", "CHENIN BLANC"),
            ("INDIAPALEALE", "INDIA PALE ALE"),
            ("PALEALE", "PALE ALE"),
        ]
        
        upper = normalized.upper()
        for stuck, fixed in varietal_fixes:
            if stuck in upper:
                # Replace case-insensitively
                pattern = re.compile(re.escape(stuck), re.IGNORECASE)
                normalized = pattern.sub(fixed, normalized)
        
        return normalized
    
    def _extract_brand_from_zone(
        self,
        zone_text: str,
        zone_confidence: float,
        ocr_result: OCRResult
    ) -> ExtractedField:
        """
        Extract brand using CANDIDATE-BASED scoring.
        
        This is robust to OCR garbling because:
        1. Builds multiple candidates from boxes (large text, suffix patterns, line strings)
        2. Scores each candidate by internal heuristics (suffix bonus, position, size)
        3. Returns best candidate + stores ALL candidates for verification to rescore
        
        During verification, candidates can be rescored against expected value
        using pick_best_brand_candidate() for "best match" selection.
        """
        boxes = ocr_result.boxes if ocr_result else []
        full_raw_text = ocr_result.raw_text if ocr_result else ""
        
        # Determine image height from boxes
        if boxes:
            image_height = max(b.bottom for b in boxes)
        else:
            image_height = 1000  # Default
        
        # Build ALL brand candidates from boxes
        candidates = _build_brand_candidates(boxes, image_height)
        
        # Also add zone text as a candidate if available
        if zone_text and zone_text.strip():
            if zone_text.strip() not in candidates:
                candidates.insert(0, zone_text.strip())
            
            # Add suffix-based extraction from zone text
            suffix_brand = self._extract_brand_with_suffix(zone_text)
            if suffix_brand and suffix_brand not in candidates:
                candidates.insert(0, suffix_brand)
        
        # Add suffix-based extraction from full raw text
        if full_raw_text:
            suffix_brand = self._extract_brand_with_suffix(full_raw_text)
            if suffix_brand and suffix_brand not in candidates:
                candidates.insert(0, suffix_brand)
        
        if not candidates:
            # Absolute fallback
            return self._extract_brand(ocr_result)
        
        # Score candidates by INTERNAL heuristics (no expected value yet)
        # Prefer: suffix patterns > large text > top position
        best_candidate = None
        best_score = 0.0
        best_method = "candidate"
        
        for c in candidates:
            score = 0.0
            cn = _ocr_normalize(c)
            
            if len(cn) < 2:
                continue
            
            # =================================================================
            # HARD REJECTS: Prevent garbage like "[BREW 091" from winning
            # =================================================================
            
            # HARD REJECT: bracket/noise tokens (OCR artifacts)
            if re.search(r'[\[\]{}<>]', c):
                continue
            
            # HARD REJECT: too digit-heavy (brands rarely have many digits)
            alnum = re.sub(r'[^A-Z0-9]', '', cn)
            if alnum:
                digit_ratio = sum(ch.isdigit() for ch in alnum) / len(alnum)
                if digit_ratio > 0.25:
                    continue
            
            # HARD REJECT: fewer than 2 alphabetic tokens
            alpha_tokens = [t for t in cn.split() if re.match(r'^[A-Z]+$', t)]
            if len(alpha_tokens) < 1:
                continue
            
            # =================================================================
            # SCORING (only for valid candidates that passed hard rejects)
            # =================================================================
            
            # Bonus: contains brand suffix (DISTILLERY, BREWING, etc.)
            if _contains_brand_suffix(c):
                score += 0.40
                
            # Bonus: longer candidates (brands are usually multi-word)
            word_count = len(cn.split())
            score += min(0.25, word_count * 0.08)
            
            # Bonus: all-caps or title-case (brand names are styled)
            if c.isupper() or c.istitle():
                score += 0.10
            
            # Penalty: contains numbers (brands rarely have digits)
            if re.search(r'\d', c):
                score -= 0.35  # Increased from -0.15
            
            # Penalty: too short
            if len(cn) < 5:
                score -= 0.10
            
            if score > best_score:
                best_score = score
                best_candidate = c
        
        if not best_candidate and candidates:
            # Fallback to first candidate that doesn't have brackets
            for c in candidates:
                if not re.search(r'[\[\]{}<>]', c):
                    best_candidate = c
                    break
        
        # Calculate confidence based on score
        confidence = min(0.90, max(0.50, 0.50 + best_score))
        
        # Rescue short brands if possible
        if best_candidate:
            best_candidate = self._rescue_short_brand(best_candidate, full_raw_text)
        
        return ExtractedField(
            value=best_candidate,
            confidence=confidence,
            source_boxes=[],
            extraction_method="candidate_scoring",
            notes=f"Best of {len(candidates)} candidates (score={best_score:.2f})",
            candidates=candidates  # Store for verification to rescore
        )
    
    def _extract_abv_from_zone(
        self,
        zone_text: str,
        zone_confidence: float,
        ocr_result: OCRResult
    ) -> ExtractedField:
        """Extract ABV from the ABV zone text."""
        if not zone_text:
            return self._extract_abv(ocr_result)
        
        # Try each ABV pattern on zone text
        for pattern in self.abv_patterns:
            match = re.search(pattern, zone_text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                
                # Convert proof to percentage
                is_proof = 'proof' in pattern.lower()
                if is_proof:
                    value = value / 2
                
                return ExtractedField(
                    value=str(value),
                    confidence=max(zone_confidence, 0.85),
                    source_boxes=[],
                    extraction_method="zone_abv",
                    notes=f"ABV from zone: {match.group(0)}"
                )
        
        # Fall back to standard extraction
        return self._extract_abv(ocr_result)
    
    def _extract_net_contents_from_zone(
        self,
        zone_text: str,
        zone_confidence: float,
        ocr_result: OCRResult
    ) -> ExtractedField:
        """Extract net contents from the net_contents zone text."""
        if not zone_text:
            return self._extract_net_contents(ocr_result)
        
        # Try each net contents pattern on zone text
        for pattern, multiplier in self.net_contents_patterns:
            match = re.search(pattern, zone_text, re.IGNORECASE)
            if match:
                value = float(match.group(1)) * multiplier
                value = round(value, 1)
                
                # PLAUSIBILITY CHECK: reject garbage like 54 mL, 3 mL
                if not self._is_net_contents_plausible(value):
                    logger.debug(f"Rejected implausible net contents: {value} mL from '{match.group(0)}'")
                    continue
                
                # Snap to common sizes (354.9 -> 355)
                value = self._snap_to_common_size(value)
                
                return ExtractedField(
                    value=str(value),
                    confidence=max(zone_confidence, 0.85),
                    source_boxes=[],
                    extraction_method="zone_net_contents",
                    notes=f"Net contents from zone: {match.group(0)}"
                )
        
        # Fall back to standard extraction
        return self._extract_net_contents(ocr_result)
    
    def _is_net_contents_plausible(self, ml: float) -> bool:
        """
        Check if net contents value is plausible for alcohol labels.
        
        Rejects garbage like 3 mL, 54 mL which are clearly OCR errors.
        Valid range: 100 mL (minis) to 2000 mL (large bottles).
        """
        return 100.0 <= ml <= 2000.0
    
    def _snap_to_common_size(self, ml: float) -> float:
        """
        Snap to common bottle/can sizes if within tolerance.
        
        This fixes OCR errors like 354.9 -> 355, 749 -> 750.
        Tolerance: 3% or 15 mL, whichever is larger.
        """
        for common in self.common_sizes_ml:
            tolerance = max(15.0, common * 0.03)
            if abs(ml - common) <= tolerance:
                return float(common)
        return ml
    
    def _extract_warning_from_zone(
        self,
        zone_text: str,
        zone_confidence: float,
        ocr_result: OCRResult
    ) -> ExtractedField:
        """Extract government warning from the warning zone text."""
        if not zone_text:
            return self._extract_government_warning(ocr_result)
        
        # Check for warning keywords in zone text
        warning_indicators = [
            "government warning",
            "surgeon general",
            "birth defects",
            "pregnancy",
            "alcoholic beverages",
            "impairs",
            "machinery",
            "health problems"
        ]
        
        text_lower = zone_text.lower()
        matches = sum(1 for indicator in warning_indicators if indicator in text_lower)
        
        if matches >= 2:  # At least 2 indicators
            return ExtractedField(
                value="detected",
                confidence=min(0.95, zone_confidence + 0.1 * matches),
                source_boxes=[],
                extraction_method="zone_warning",
                notes=f"Government warning detected in zone ({matches} indicators)"
            )
        
        # Fall back to standard extraction
        return self._extract_government_warning(ocr_result)
    
    def _extract_brand(self, ocr_result: OCRResult) -> ExtractedField:
        """
        Extract brand name using multiple strategies.
        
        Strategy (in order):
        1. Regex search for brand ending with DISTILLERY/BREWING/WINERY etc.
        2. Extract from top portion of text (before beverage keywords)
        3. Spatial analysis of top region boxes
        4. Smart fallback (filter stop/fluff words)
        
        Always populates source_boxes for downstream use (e.g., class/type positioning).
        """
        if not ocr_result.raw_text:
            return ExtractedField(
                value=None,
                confidence=0.0,
                extraction_method="none",
                notes="No text detected"
            )
        
        raw_text = ocr_result.raw_text
        boxes = ocr_result.boxes or []
        
        # Strategy 1: Find brand phrase ending with producer keyword (DISTILLERY, BREWING, etc.)
        brand = self._extract_brand_with_suffix(raw_text)
        if brand:
            src_boxes = self._boxes_for_phrase(boxes, brand)
            return ExtractedField(
                value=brand,
                confidence=0.9,
                source_boxes=src_boxes,
                extraction_method="regex_suffix",
                notes="Found brand ending with producer keyword"
            )
        
        # Strategy 2: Extract from top portion of text (before beverage keywords)
        brand = self._extract_brand_before_keywords(raw_text)
        if brand:
            # RESCUE: If we got a short brand but raw_text has suffix-extended version, use it
            brand = self._rescue_short_brand(brand, raw_text)
            src_boxes = self._boxes_for_phrase(boxes, brand)
            return ExtractedField(
                value=brand,
                confidence=0.85,
                source_boxes=src_boxes,
                extraction_method="regex_position",
                notes="Brand extracted before beverage keywords"
            )
        
        # Strategy 3: Spatial analysis using OCR boxes
        if boxes:
            brand, src_boxes = self._extract_brand_spatial(boxes)
            if brand:
                # RESCUE: If we got a short brand but raw_text has suffix-extended version, use it
                brand = self._rescue_short_brand(brand, raw_text)
                src_boxes = self._boxes_for_phrase(boxes, brand)
                return ExtractedField(
                    value=brand,
                    confidence=0.75,
                    source_boxes=src_boxes,
                    extraction_method="spatial",
                    notes="Brand from spatial analysis"
                )
        
        # Fallback: First meaningful tokens (filter stop/fluff)
        brand, src_boxes = self._extract_brand_fallback(raw_text, boxes)
        if brand:
            # RESCUE: If we got a short brand but raw_text has suffix-extended version, use it
            brand = self._rescue_short_brand(brand, raw_text)
            src_boxes = self._boxes_for_phrase(boxes, brand)
        return ExtractedField(
            value=brand,
            confidence=0.5,
            source_boxes=src_boxes,
            extraction_method="fallback",
            notes="Fallback to first meaningful words"
        )
    
    def _boxes_for_phrase(self, boxes: List[OCRBox], phrase: str) -> List[OCRBox]:
        """
        Find OCR boxes that contain tokens from a phrase.
        Used to populate source_boxes for regex-extracted brands.
        """
        if not boxes or not phrase:
            return []
        
        phrase_upper = phrase.upper()
        phrase_tokens = set(phrase_upper.split())
        
        hits = []
        for box in boxes:
            box_text = re.sub(r"\s+", " ", box.text.upper()).strip()
            box_tokens = set(box_text.split())
            # Check if any phrase token appears in this box
            if any(t in box_tokens for t in phrase_tokens):
                hits.append(box)
        
        if not hits:
            return []
        
        # Filter to top region only (brand should be near top)
        if hits:
            min_y = min(b.top for b in boxes)
            max_y = max(b.bottom for b in boxes)
            top_threshold = min_y + (max_y - min_y) * 0.4
            hits = [b for b in hits if b.center_y < top_threshold]
        
        # Sort by position: top-to-bottom, then left-to-right
        hits = sorted(hits, key=lambda b: (b.center_y, b.left))
        
        # Limit to reasonable number of boxes
        return hits[:8]
    
    def _is_plausible_brand(self, brand: str) -> bool:
        """
        Validate that extracted brand is plausible (not beverage type or fluff).
        
        Rejects:
        - "KENTUCKY STRAIGHT BOURBON SPIRITS" (contains beverage keywords)
        - "PREMIUM RESERVE DISTILLING" (mostly marketing fluff)
        - "THE COMPANY" (no meaningful tokens)
        """
        tokens = brand.upper().split()
        tokens_lower = {t.lower() for t in tokens}
        
        # Reject if beverage keyword is inside "brand"
        if any(kw in tokens_lower for kw in BEVERAGE_KEYWORDS):
            return False
        
        # Reject if brand is mostly marketing fluff (2+ fluff words)
        fluff_count = sum(1 for t in tokens_lower if t in MARKETING_KEYWORDS)
        if fluff_count >= 2:
            return False
        
        # Get suffix words to exclude from "meaningful" count
        suffix_words = set()
        for part in re.split(r"[|()]", BRAND_SUFFIX_PATTERN):
            part = part.strip().strip("?").lower()
            if part:
                suffix_words.add(part)
        
        # Require at least 1 meaningful token (not stop word, suffix, or fluff)
        meaningful = [
            t for t in tokens_lower 
            if t not in BRAND_STOP_WORDS 
            and t not in suffix_words 
            and t not in MARKETING_KEYWORDS
            and len(t) > 1
        ]
        
        return len(meaningful) >= 1 and len(tokens) >= 2
    
    def _extract_brand_with_suffix(self, raw_text: str) -> Optional[str]:
        """
        Extract brand phrase ending with DISTILLERY/BREWING/WINERY etc.
        
        Example: "TOM OLD DISTILLERY" from "TOM OLD DISTILLERY Premium Small Batch..."
        
        Uses finditer + scoring to pick the BEST match, not just the first.
        Prefers: longer phrases, earlier in text, DISTILLERY/BREWING over CO/LLC.
        """
        # Normalize text
        text = re.sub(r"\s+", " ", raw_text.upper()).strip()
        
        # Pattern: 1-6 words followed by producer keyword (increased from 0,4 to 0,6)
        # Matches: "OLD TOM DISTILLERY", "THE SOMETHING SOMETHING DISTILLERY", etc.
        pattern = rf"\b([A-Z0-9'&\-]{{2,}}(?:\s+[A-Z0-9'&\-]{{2,}}){{0,6}}\s+{BRAND_SUFFIX_PATTERN})\b"
        
        # Find ALL matches
        matches = list(re.finditer(pattern, text))
        if not matches:
            return None
        
        # Score each match: prefer longer phrases, earlier position, strong suffixes
        STRONG_SUFFIXES = {"DISTILLERY", "DISTILLING", "BREWING", "BREWERY", "WINERY"}
        
        def score_match(m: re.Match) -> tuple:
            phrase = m.group(1).strip()
            tokens = phrase.split()
            last_token = tokens[-1] if tokens else ""
            
            # Scoring tuple (higher is better):
            # 1. Number of tokens (prefer longer)
            # 2. Phrase length (prefer longer)
            # 3. Strong suffix bonus (DISTILLERY > CO)
            # 4. Earlier position (negative start = prefer earlier)
            strong_suffix_bonus = 1 if last_token in STRONG_SUFFIXES else 0
            return (len(tokens), len(phrase), strong_suffix_bonus, -m.start())
        
        # Filter to plausible brands and pick the best
        valid_matches = []
        for match in matches:
            brand = match.group(1).strip()
            if self._is_plausible_brand(brand):
                valid_matches.append((match, brand))
            else:
                logger.debug(f"Rejected implausible brand: {brand}")
        
        if not valid_matches:
            return None
        
        # Pick the best match by score
        best_match, best_brand = max(valid_matches, key=lambda x: score_match(x[0]))
        logger.debug(f"Found brand with suffix: {best_brand} (from {len(valid_matches)} candidates)")
        return best_brand
    
    def _rescue_short_brand(self, brand: str, raw_text: str) -> str:
        """
        Rescue: if we extracted a short brand but raw_text contains suffix-extended version, use it.
        
        Example: "TOM OLD" -> "TOM OLD DISTILLERY" if raw_text contains the full phrase.
        
        This prevents the "first run fails" issue where extraction picks a truncated brand.
        """
        if not brand or len(brand.split()) > 3:
            # Already long enough, don't rescue
            return brand
        
        # Try to find suffix-extended version in raw_text
        upgraded = self._extract_brand_with_suffix(raw_text)
        
        if upgraded:
            brand_upper = brand.upper()
            upgraded_upper = upgraded.upper()
            
            # Check if our short brand is contained in the upgraded version
            if brand_upper in upgraded_upper:
                logger.debug(f"Brand rescue: '{brand}' -> '{upgraded}'")
                return upgraded
        
        return brand
    
    def _extract_brand_before_keywords(self, raw_text: str) -> Optional[str]:
        """
        Extract brand as text before beverage type keywords.
        
        Example: "OLD TOM" from "OLD TOM Kentucky Straight Bourbon Whiskey"
        
        Caps result to MAX_BRAND_TOKENS and stops at obvious markers.
        """
        MAX_BRAND_TOKENS = 6
        STOP_MARKERS = {
            "BATCH", "PROOF", "ML", "M L", "GOVERNMENT", "WARNING", 
            "ALC", "VOL", "ALCOHOL", "CONTAINS", "IMPORTED", "PRODUCED"
        }
        
        text = re.sub(r"\s+", " ", raw_text.upper()).strip()
        
        # Build pattern for beverage keywords
        beverage_pattern = "|".join(kw.upper() for kw in BEVERAGE_KEYWORDS)
        
        # Find text before first beverage keyword
        pattern = rf"^(.+?)\s+(?:KENTUCKY\s+)?(?:STRAIGHT\s+)?(?:{beverage_pattern})"
        
        match = re.search(pattern, text)
        if match:
            brand = match.group(1).strip()
            
            # Tokenize and truncate at stop markers or numbers
            tokens = brand.split()
            cut_tokens = []
            for t in tokens:
                # Stop at markers or tokens starting with digits
                if t in STOP_MARKERS or re.match(r"^\d", t):
                    break
                cut_tokens.append(t)
            
            # Cap at MAX_BRAND_TOKENS
            cut_tokens = cut_tokens[:MAX_BRAND_TOKENS]
            brand = " ".join(cut_tokens).strip()
            
            # Remove marketing fluff from end
            for fluff in MARKETING_KEYWORDS:
                brand = re.sub(rf"\s+{fluff.upper()}$", "", brand, flags=re.IGNORECASE)
            
            # Validate
            if len(brand) >= 3 and self._is_plausible_brand(brand):
                logger.debug(f"Found brand before keywords: {brand}")
                return brand
        
        return None
    
    def _extract_brand_spatial(self, boxes: List[OCRBox]) -> Tuple[Optional[str], List[OCRBox]]:
        """
        Extract brand using spatial analysis of OCR boxes.
        Looks for prominent text in top region, sorted left-to-right.
        
        Returns:
            Tuple of (brand_text, source_boxes)
        """
        if not boxes:
            return None, []
        
        # Get image bounds
        min_y = min(b.top for b in boxes)
        max_y = max(b.bottom for b in boxes)
        image_height = max_y - min_y
        
        # Get boxes in top 35% (where brand usually is)
        top_threshold = min_y + image_height * 0.35
        top_boxes = [b for b in boxes if b.center_y < top_threshold]
        
        if not top_boxes:
            return None, []
        
        # Group by line and sort left-to-right
        lines = self._group_boxes_by_line(top_boxes)
        
        if not lines:
            return None, []
        
        # Find line with largest text (most prominent)
        best_line = max(lines, key=lambda line: sum(b.height for b in line) / len(line))
        
        # Sort boxes left-to-right within the line
        sorted_line = sorted(best_line, key=lambda b: b.left)
        
        # Join text
        brand = " ".join(b.text.strip() for b in sorted_line)
        brand = re.sub(r"\s+", " ", brand).strip()
        
        if len(brand) >= 3:
            return brand, sorted_line
        return None, []
    
    def _extract_brand_fallback(self, raw_text: str, boxes: List[OCRBox]) -> Tuple[Optional[str], List[OCRBox]]:
        """
        Fallback brand extraction: first meaningful tokens (filter stop/fluff).
        
        Better than "first 4 words" which could return "THE PREMIUM RESERVE AGED".
        """
        MAX_FALLBACK_TOKENS = 4
        
        text = re.sub(r"\s+", " ", raw_text.upper()).strip()
        tokens = text.split()
        
        # Filter out stop words and marketing fluff from the beginning
        meaningful = []
        for t in tokens:
            t_lower = t.lower()
            
            # Stop at beverage keywords
            if t_lower in BEVERAGE_KEYWORDS:
                break
            
            # Stop at numbers (likely ABV, proof, or volume)
            if re.match(r"^\d", t):
                break
            
            # Skip stop words and fluff at the beginning
            if not meaningful and (t_lower in BRAND_STOP_WORDS or t_lower in MARKETING_KEYWORDS):
                continue
            
            meaningful.append(t)
            
            if len(meaningful) >= MAX_FALLBACK_TOKENS:
                break
        
        if meaningful:
            brand = " ".join(meaningful)
            src_boxes = self._boxes_for_phrase(boxes, brand)
            return brand, src_boxes
        
        return None, []
    
    def _group_boxes_by_line(self, boxes: List[OCRBox], y_threshold: int = 25) -> List[List[OCRBox]]:
        """
        Group boxes that are on the same horizontal line.
        """
        if not boxes:
            return []
        
        sorted_boxes = sorted(boxes, key=lambda b: b.center_y)
        
        lines = []
        current_line = [sorted_boxes[0]]
        
        for box in sorted_boxes[1:]:
            if abs(box.center_y - current_line[0].center_y) <= y_threshold:
                current_line.append(box)
            else:
                lines.append(current_line)
                current_line = [box]
        
        if current_line:
            lines.append(current_line)
        
        return lines
    
    def _extract_class_type(
        self, 
        ocr_result: OCRResult, 
        brand_result: ExtractedField,
        zone_text: Optional[str] = None
    ) -> ExtractedField:
        """
        Extract class/type using PRIORITY PHRASE SEARCH + CANDIDATE scoring.
        
        Strategy (in order):
        1. FIRST: Search FULL raw text for known long beverage phrases
           (e.g., "KENTUCKY STRAIGHT BOURBON WHISKEY", "INDIA PALE ALE")
           This catches text that may be outside the zone slice.
        2. THEN: Build candidates from zone text and score them
        3. Store ALL candidates for verification to rescore against expected
        """
        boxes = ocr_result.boxes if ocr_result else []
        full_raw_text = ocr_result.raw_text if ocr_result else ""
        search_text = zone_text or full_raw_text
        
        if not search_text and not boxes:
            return ExtractedField(
                value=None,
                confidence=0.0,
                extraction_method="keyword",
                notes="No text detected"
            )
        
        # =================================================================
        # PRIORITY SEARCH: Look for known long phrases in FULL raw text first
        # This catches "KENTUCKY STRAIGHT BOURBON WHISKEY" even if zone slicing missed it
        # =================================================================
        PRIORITY_PHRASES = [
            # Spirits - long descriptive forms
            (r"KENTUCKY\s+STRAIGHT\s+BOURBON\s+WHISKEY", "Kentucky Straight Bourbon Whiskey"),
            (r"STRAIGHT\s+BOURBON\s+WHISKEY", "Straight Bourbon Whiskey"),
            (r"BOURBON\s+WHISKEY", "Bourbon Whiskey"),
            (r"TENNESSEE\s+WHISKEY", "Tennessee Whiskey"),
            (r"SINGLE\s+MALT\s+SCOTCH", "Single Malt Scotch"),
            (r"BLENDED\s+SCOTCH\s+WHISKY", "Blended Scotch Whisky"),
            (r"LONDON\s+DRY\s+GIN", "London Dry Gin"),
            (r"SILVER\s+TEQUILA", "Silver Tequila"),
            (r"REPOSADO\s+TEQUILA", "Reposado Tequila"),
            (r"ANEJO\s+TEQUILA", "Anejo Tequila"),
            # Beer - long forms
            (r"INDIA\s+PALE\s+ALE", "India Pale Ale"),
            (r"IMPERIAL\s+STOUT", "Imperial Stout"),
            (r"DOUBLE\s+IPA", "Double IPA"),
            # Wine varietals (multi-word)
            (r"CABERNET\s+SAUVIGNON", "Cabernet Sauvignon"),
            (r"SAUVIGNON\s+BLANC", "Sauvignon Blanc"),
            (r"PINOT\s+NOIR", "Pinot Noir"),
            (r"PINOT\s+GRIGIO", "Pinot Grigio"),
            (r"CHENIN\s+BLANC", "Chenin Blanc"),
            # Single strong keywords (checked last)
            (r"\bBOURBON\b", "Bourbon"),
            (r"\bWHISKEY\b", "Whiskey"),
            (r"\bWHISKY\b", "Whisky"),
            (r"\bVODKA\b", "Vodka"),
            (r"\bGIN\b", "Gin"),
            (r"\bRUM\b", "Rum"),
            (r"\bTEQUILA\b", "Tequila"),
            (r"\bIPA\b", "IPA"),
            (r"\bALE\b", "Ale"),
            (r"\bLAGER\b", "Lager"),
            (r"\bSTOUT\b", "Stout"),
            (r"\bPORTER\b", "Porter"),
            (r"\bMERLOT\b", "Merlot"),
            (r"\bCHARDONNAY\b", "Chardonnay"),
            (r"\bRIESLING\b", "Riesling"),
            (r"\bZINFANDEL\b", "Zinfandel"),
            (r"\bSYRAH\b", "Syrah"),
            (r"\bSHIRAZ\b", "Shiraz"),
            (r"\bMALBEC\b", "Malbec"),
        ]
        
        # Search FULL raw text for priority phrases (regardless of zone)
        for pattern, canonical in PRIORITY_PHRASES:
            match = re.search(pattern, full_raw_text, re.IGNORECASE)
            if match:
                logger.info(f"Class/type found via priority phrase: '{canonical}' from '{match.group(0)}'")
                return ExtractedField(
                    value=canonical,
                    confidence=0.90,
                    source_boxes=[],
                    extraction_method="priority_phrase",
                    notes=f"Matched priority phrase: {pattern}",
                    candidates=[canonical]
                )
        
        # =================================================================
        # FALLBACK: Candidate-based scoring on zone text
        # =================================================================
        
        # Determine image height from boxes
        image_height = max(b.bottom for b in boxes) if boxes else 1000
        
        # Build candidates from zone text (or full text if no zone)
        candidates = _build_class_type_candidates(boxes, search_text, image_height)
        
        # Also add normalized zone text
        if zone_text and zone_text.strip():
            normalized = self._normalize_varietal_text(zone_text)
            if normalized and normalized not in candidates:
                candidates.append(normalized)
        
        if not candidates:
            return ExtractedField(
                value=None,
                confidence=0.0,
                extraction_method="keyword",
                notes="No beverage keywords found",
                candidates=[]
            )
        
        # Score candidates by INTERNAL heuristics
        best_candidate = None
        best_score = 0.0
        best_method = "candidate"
        
        brand_bottom = 0
        if brand_result and brand_result.source_boxes:
            brand_bottom = brand_result.source_boxes[0].bottom
        
        brand_value = brand_result.value if brand_result else None
        
        for c in candidates:
            score = 0.0
            cn = _ocr_normalize(c)
            c_lower = c.lower()
            
            if len(cn) < 2:
                continue
            
            # =================================================================
            # HARD GATES: Prevent brand/marketing text from being class/type
            # =================================================================
            
            matches_varietal = any(re.search(p, c, re.IGNORECASE) for p in WINE_VARIETAL_PATTERNS)
            
            # HARD GATE: Must contain beverage keyword OR match varietal pattern
            if not matches_varietal and not _has_beverage_keyword(cn):
                continue
            
            # HARD GATE: Reject brand/producer/marketing text
            if _is_contaminated_by_brand(c, brand_value):
                continue
            
            # =================================================================
            # SCORING
            # =================================================================
            
            if matches_varietal:
                score += 0.50
            
            # Bonus: contains strong beverage keyword
            for kw in ["bourbon", "whiskey", "whisky", "vodka", "gin", "rum", 
                       "tequila", "wine", "beer", "ale", "lager", "stout", 
                       "ipa", "cabernet", "chardonnay", "merlot", "pinot"]:
                if kw in c_lower:
                    score += 0.30
                    break
            
            # Bonus: longer phrase (more specific)
            word_count = len(cn.split())
            score += min(0.20, word_count * 0.05)
            
            # Penalty: contains numbers (class/type shouldn't have digits)
            if re.search(r'\d', c):
                score -= 0.20
            
            if score > best_score:
                best_score = score
                best_candidate = c
        
        # If no valid candidate found, DON'T fall back to first candidate
        # (which might be brand text). Return not_found instead.
        
        if best_candidate:
            # Clean up the candidate
            best_candidate = re.sub(r'[^\w\s\'-]', '', best_candidate).strip()
            best_candidate = best_candidate.title()  # Title case for display
            
            # Calculate confidence based on score
            confidence = min(0.90, max(0.50, 0.50 + best_score))
            
            return ExtractedField(
                value=best_candidate,
                confidence=confidence,
                source_boxes=[],
                extraction_method="candidate_scoring",
                notes=f"Best of {len(candidates)} candidates (score={best_score:.2f})",
                candidates=candidates  # Store for verification to rescore
            )
        
        return ExtractedField(
            value=None,
            confidence=0.0,
            extraction_method="keyword",
            notes="No beverage keywords found",
            candidates=candidates
        )
    
    def _extract_abv(self, ocr_result: OCRResult) -> ExtractedField:
        """
        Extract ABV percentage using regex patterns.
        
        Handles:
        - "45%" or "45.5%"
        - "45% Alc./Vol." or "45% ALC/VOL"
        - "90 Proof" -> converts to 45%
        """
        raw_text = ocr_result.raw_text
        
        for i, pattern in enumerate(self.abv_patterns):
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                
                # Convert proof to percentage
                is_proof = 'proof' in pattern.lower()
                if is_proof:
                    value = value / 2
                    method = "regex_proof_conversion"
                else:
                    method = "regex_percentage"
                
                # Find which box contained this
                source_box = self._find_box_containing(
                    ocr_result.boxes, 
                    match.group(0)
                )
                confidence = source_box.confidence if source_box else 0.8
                
                return ExtractedField(
                    value=str(value),
                    confidence=confidence,
                    source_boxes=[source_box] if source_box else [],
                    extraction_method=method,
                    notes=f"Matched: {match.group(0)}" + (" (converted from proof)" if is_proof else "")
                )
        
        return ExtractedField(
            value=None,
            confidence=0.0,
            extraction_method="regex",
            notes="No ABV pattern matched"
        )
    
    def _extract_net_contents(self, ocr_result: OCRResult) -> ExtractedField:
        """
        Extract net contents and normalize to mL.
        
        Handles:
        - "750 mL", "750ml", "750 ML"
        - "75 cL" -> 750 mL
        - "1 L" -> 1000 mL
        - "25.4 FL OZ" -> ~750 mL
        - "12 FL OZ" -> 355 mL
        
        Includes plausibility checks to reject garbage matches like 54 mL.
        """
        raw_text = ocr_result.raw_text
        
        for pattern, multiplier in self.net_contents_patterns:
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                value = float(match.group(1)) * multiplier
                value = round(value, 1)  # Round to 1 decimal
                
                # PLAUSIBILITY CHECK: reject garbage like 54 mL, 3 mL
                if not self._is_net_contents_plausible(value):
                    logger.debug(f"Rejected implausible net contents: {value} mL from '{match.group(0)}'")
                    continue
                
                # Snap to common sizes (354.9 -> 355)
                value = self._snap_to_common_size(value)
                
                source_box = self._find_box_containing(
                    ocr_result.boxes,
                    match.group(0)
                )
                confidence = source_box.confidence if source_box else 0.8
                
                return ExtractedField(
                    value=str(value),
                    confidence=confidence,
                    source_boxes=[source_box] if source_box else [],
                    extraction_method="regex_unit_conversion",
                    notes=f"Matched: {match.group(0)} (normalized to mL)"
                )
        
        return ExtractedField(
            value=None,
            confidence=0.0,
            extraction_method="regex",
            notes="No net contents pattern matched"
        )
    
    def _extract_government_warning(self, ocr_result: OCRResult) -> ExtractedField:
        """
        Detect government warning using canonicalization.
        
        Strategy:
        1. Canonicalize extracted text (uppercase, normalize whitespace)
        2. Look for key phrases: "GOVERNMENT WARNING", "SURGEON GENERAL"
        3. Check if substantial portion of warning is present
        """
        raw_text = ocr_result.raw_text
        
        # Canonicalize: uppercase, normalize whitespace
        canonical_text = self._canonicalize_text(raw_text)
        canonical_warning = self._canonicalize_text(GOVERNMENT_WARNING_CANONICAL)
        
        # Check for key phrases
        has_gov_warning = "GOVERNMENT WARNING" in canonical_text
        has_surgeon_general = "SURGEON GENERAL" in canonical_text
        has_pregnancy = "PREGNANCY" in canonical_text or "BIRTH DEFECTS" in canonical_text
        has_machinery = "MACHINERY" in canonical_text or "DRIVE A CAR" in canonical_text
        
        # Score based on presence of key components
        score = sum([
            has_gov_warning * 0.3,
            has_surgeon_general * 0.25,
            has_pregnancy * 0.25,
            has_machinery * 0.2,
        ])
        
        if score >= 0.5:
            # Find boxes containing warning text
            warning_boxes = []
            for box in ocr_result.boxes:
                box_canonical = self._canonicalize_text(box.text)
                if any(kw in box_canonical for kw in ["GOVERNMENT", "WARNING", "SURGEON", "PREGNANCY"]):
                    warning_boxes.append(box)
            
            # Calculate confidence from box confidences
            if warning_boxes:
                confidence = sum(b.confidence for b in warning_boxes) / len(warning_boxes)
            else:
                confidence = score
            
            # Determine status
            if score >= 0.75:
                status = "detected"
                notes = "Government warning detected"
            else:
                status = "partial"
                notes = "Partial warning text detected - manual review recommended"
            
            return ExtractedField(
                value=status,
                confidence=confidence,
                source_boxes=warning_boxes,
                extraction_method="keyword_canonical",
                notes=notes
            )
        
        return ExtractedField(
            value="not_found",
            confidence=0.0,
            extraction_method="keyword_canonical",
            notes="Warning not detected - manual review required"
        )
    
    def _canonicalize_text(self, text: str) -> str:
        """Canonicalize text for comparison."""
        # Uppercase
        text = text.upper()
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove common OCR artifacts
        text = text.replace('|', 'I')
        # Replace 0 with O only when surrounded by letters (avoids "(1)" → "(I)")
        text = re.sub(r'(?<=[A-Z])0(?=[A-Z])', 'O', text)
        # Fix common OCR joins
        text = re.sub(r'SURGEONGENERAL', 'SURGEON GENERAL', text)
        text = re.sub(r'GOVERNMENTWARNING', 'GOVERNMENT WARNING', text)
        text = re.sub(r'BIRTHDEFECTS', 'BIRTH DEFECTS', text)
        return text.strip()
    
    def _find_box_containing(
        self, 
        boxes: List[OCRBox], 
        text_fragment: str
    ) -> Optional[OCRBox]:
        """Find the OCR box containing a text fragment."""
        text_fragment_lower = text_fragment.lower()
        for box in boxes:
            if text_fragment_lower in box.text.lower():
                return box
        return None


def extract_abv_value(text: str) -> Optional[float]:
    """
    Standalone function to extract ABV from text.
    
    Args:
        text: Raw text to search
        
    Returns:
        ABV as float percentage, or None if not found
    """
    extractor = FieldExtractor()
    
    # Create a minimal OCRResult for the extractor
    from .ocr import OCRResult, OCRBox
    mock_result = OCRResult(
        boxes=[OCRBox(text=text, confidence=1.0, bbox=[[0,0],[100,0],[100,20],[0,20]])],
        raw_text=text,
        average_confidence=1.0
    )
    
    result = extractor._extract_abv(mock_result)
    return float(result.value) if result.value else None


def extract_net_contents_ml(text: str) -> Optional[float]:
    """
    Standalone function to extract net contents in mL.
    
    Args:
        text: Raw text to search
        
    Returns:
        Net contents in mL, or None if not found
    """
    extractor = FieldExtractor()
    
    from .ocr import OCRResult, OCRBox
    mock_result = OCRResult(
        boxes=[OCRBox(text=text, confidence=1.0, bbox=[[0,0],[100,0],[100,20],[0,20]])],
        raw_text=text,
        average_confidence=1.0
    )
    
    result = extractor._extract_net_contents(mock_result)
    return float(result.value) if result.value else None
