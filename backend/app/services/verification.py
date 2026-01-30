"""Verification service for comparing extracted fields against application data."""

import re
from typing import Optional, List
from dataclasses import dataclass
from enum import Enum
import logging

from rapidfuzz import fuzz
from .extraction import ExtractionResult, ExtractedField
from ..config import get_settings

logger = logging.getLogger(__name__)


class VerificationStatus(str, Enum):
    """Status of a field verification."""
    MATCH = "match"
    REVIEW = "review"  
    MISMATCH = "mismatch"
    NOT_FOUND = "not_found"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class FieldVerification:
    """Result of verifying a single field."""
    field_name: str
    status: VerificationStatus
    extracted_value: Optional[str]
    expected_value: Optional[str]
    confidence: float
    message: str
    details: Optional[str] = None


@dataclass
class VerificationResult:
    """Complete verification result."""
    overall_status: VerificationStatus
    fields: List[FieldVerification]
    summary: str
    passed_count: int
    review_count: int
    failed_count: int


class VerificationService:
    """Compares extracted fields against expected application data."""
    
    def __init__(self):
        self.settings = get_settings()
    
    def verify(
        self,
        extraction: ExtractionResult,
        expected_brand: str,
        expected_class_type: Optional[str] = None,
        expected_abv: Optional[float] = None,
        expected_net_contents: Optional[float] = None,
        expected_has_warning: bool = True,
    ) -> VerificationResult:
        """
        Verify extracted fields against expected values.
        
        Args:
            extraction: Result from field extraction
            expected_brand: Expected brand name
            expected_class_type: Expected class/type (optional)
            expected_abv: Expected ABV percentage (optional)
            expected_net_contents: Expected net contents in mL (optional)
            expected_has_warning: Whether warning should be present
            
        Returns:
            VerificationResult with per-field and overall status
        """
        fields = []
        
        # Verify brand name (required)
        brand_result = self._verify_brand(
            extraction.brand_name,
            expected_brand
        )
        fields.append(brand_result)
        
        # Verify class/type (optional)
        if expected_class_type:
            class_result = self._verify_class_type(
                extraction.class_type,
                expected_class_type
            )
            fields.append(class_result)
        
        # Verify ABV (optional)
        if expected_abv is not None:
            abv_result = self._verify_abv(
                extraction.abv_percent,
                expected_abv
            )
            fields.append(abv_result)
        
        # Verify net contents (optional)
        if expected_net_contents is not None:
            net_result = self._verify_net_contents(
                extraction.net_contents_ml,
                expected_net_contents
            )
            fields.append(net_result)
        
        # Verify government warning
        warning_result = self._verify_warning(
            extraction.government_warning,
            expected_has_warning
        )
        fields.append(warning_result)
        
        # Calculate overall status
        passed = sum(1 for f in fields if f.status == VerificationStatus.MATCH)
        review = sum(1 for f in fields if f.status == VerificationStatus.REVIEW)
        failed = sum(1 for f in fields if f.status in [VerificationStatus.MISMATCH, VerificationStatus.NOT_FOUND])
        
        if failed > 0:
            overall_status = VerificationStatus.MISMATCH
        elif review > 0:
            overall_status = VerificationStatus.REVIEW
        else:
            overall_status = VerificationStatus.MATCH
        
        # Generate summary
        summary = self._generate_summary(fields, overall_status)
        
        return VerificationResult(
            overall_status=overall_status,
            fields=fields,
            summary=summary,
            passed_count=passed,
            review_count=review,
            failed_count=failed,
        )
    
    def _verify_brand(
        self,
        extracted: ExtractedField,
        expected: str
    ) -> FieldVerification:
        """
        Verify brand name using token-set similarity + CANDIDATE RESCORING.
        
        This handles cases like:
        - "TOM OLD DISTILLERY" vs "OLD TOM DISTILLERY" (word order swap)
        - "OLD TOM" vs "OLD TOM DISTILLERY" (partial match)
        - OCR garbling: "[BREW 091" vs "MOUNTAIN BREW CO" (rescore candidates)
        
        If initial score is low, RESCORE all candidates against expected
        to find the best match (robust to OCR garbling).
        
        Thresholds:
        - >= 0.85: Match
        - 0.70-0.84: Review
        - < 0.70: Mismatch
        """
        # Import the candidate scoring function
        from .extraction import pick_best_brand_candidate
        
        if not extracted.value:
            # If no value but we have candidates, try to find best match
            if hasattr(extracted, 'candidates') and extracted.candidates:
                best, score, reason = pick_best_brand_candidate(extracted.candidates, expected)
                if best and score >= 0.70:
                    logger.info(f"Brand rescued from candidates: '{best}' (score={score:.2f})")
                    return FieldVerification(
                        field_name="Brand Name",
                        status=VerificationStatus.REVIEW if score < 0.85 else VerificationStatus.MATCH,
                        extracted_value=best,
                        expected_value=expected,
                        confidence=score,
                        message="Brand name found via candidate matching",
                        details=f"Best match from {len(extracted.candidates)} candidates ({reason})"
                    )
            
            return FieldVerification(
                field_name="Brand Name",
                status=VerificationStatus.NOT_FOUND,
                extracted_value=None,
                expected_value=expected,
                confidence=0.0,
                message="Brand name not detected on label",
                details="OCR could not extract brand name. Manual review required."
            )
        
        # Normalize for comparison
        extracted_norm = self._normalize_brand(extracted.value)
        expected_norm = self._normalize_brand(expected)
        
        # Core token overlap (ignores DISTILLERY, BREWING, etc.)
        core_score = self._core_brand_similarity(extracted.value, expected)
        
        # Primary similarity methods - token-based, order-insensitive
        ts = fuzz.token_set_ratio(extracted_norm, expected_norm) / 100.0
        tso = fuzz.token_sort_ratio(extracted_norm, expected_norm) / 100.0
        
        score = max(ts, tso, core_score)
        
        # Only allow partial_ratio to influence score if core token overlap exists
        if core_score >= 0.5:
            pr = fuzz.partial_ratio(extracted_norm, expected_norm) / 100.0
            score = max(score, pr)
        
        # CANDIDATE RESCORING: If score is low, search ALL candidates for better match
        # This is the KEY fix for OCR garbling like "[BREW 091" vs "MOUNTAIN BREW CO"
        best_candidate = extracted.value
        if score < self.settings.brand_match_threshold:
            if hasattr(extracted, 'candidates') and extracted.candidates:
                best, rescore, reason = pick_best_brand_candidate(extracted.candidates, expected)
                if best and rescore > score:
                    logger.info(f"Brand improved via rescore: '{extracted.value}' -> '{best}' "
                               f"(score {score:.2f} -> {rescore:.2f}, {reason})")
                    best_candidate = best
                    score = rescore
        
        logger.debug(f"Brand comparison: '{best_candidate}' vs '{expected}' -> score={score:.2f}")
        
        if score >= self.settings.brand_match_threshold:
            return FieldVerification(
                field_name="Brand Name",
                status=VerificationStatus.MATCH,
                extracted_value=best_candidate,
                expected_value=expected,
                confidence=score,
                message="Brand name matches",
                details=f"Similarity score: {score:.0%}" if score < 1.0 else None
            )
        elif score >= self.settings.brand_review_threshold:
            return FieldVerification(
                field_name="Brand Name",
                status=VerificationStatus.REVIEW,
                extracted_value=best_candidate,
                expected_value=expected,
                confidence=score,
                message="Brand name likely matches - review recommended",
                details=f"'{best_candidate}' vs '{expected}' - similarity {score:.0%}. "
                        "May be a case/punctuation difference."
            )
        else:
            return FieldVerification(
                field_name="Brand Name",
                status=VerificationStatus.MISMATCH,
                extracted_value=best_candidate,
                expected_value=expected,
                confidence=score,
                message="Brand name does not match",
                details=f"Label shows '{best_candidate}' but application states '{expected}'. "
                        f"Similarity: {score:.0%}"
            )
    
    def _normalize_brand(self, text: str) -> str:
        """Normalize brand text for comparison."""
        text = text.upper()
        text = re.sub(r'\s+', ' ', text)
        # Keep apostrophes (O'BRIEN), remove other punctuation
        text = re.sub(r"[^\w\s']", " ", text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    
    def _core_brand_similarity(self, extracted: str, expected: str) -> float:
        """
        Compare core brand tokens, ignoring company suffixes.
        
        This handles "OLD TOM" matching "OLD TOM DISTILLERY" by
        stripping DISTILLERY/BREWING/etc. for comparison only.
        """
        # Suffixes that are optional for matching
        BRAND_SUFFIXES = {
            "DISTILLERY", "DISTILLING", "BREWING", "BREWERY", 
            "WINERY", "VINEYARD", "VINEYARDS", "CELLARS", "CELLAR",
            "COMPANY", "CO", "INC", "LLC", "LTD", "SPIRITS"
        }
        
        def get_core_tokens(text: str) -> set:
            text = text.upper()
            text = re.sub(r"[^\w\s]", " ", text)
            tokens = text.split()
            # Remove brand suffixes
            return {t for t in tokens if t not in BRAND_SUFFIXES and len(t) > 1}
        
        extracted_tokens = get_core_tokens(extracted)
        expected_tokens = get_core_tokens(expected)
        
        if not expected_tokens:
            return 0.0
        
        # Calculate Jaccard-like overlap
        common = extracted_tokens & expected_tokens
        union = extracted_tokens | expected_tokens
        
        if not union:
            return 0.0
        
        overlap_score = len(common) / len(union)
        
        # Bonus if all expected core tokens are found in extracted
        # e.g., expected="OLD TOM" and extracted="OLD TOM DISTILLERY"
        if expected_tokens and expected_tokens <= extracted_tokens:
            overlap_score = max(overlap_score, 0.95)
        
        # Also bonus if all extracted core tokens are found in expected
        # e.g., extracted="OLD TOM" and expected="OLD TOM DISTILLERY"
        if extracted_tokens and extracted_tokens <= expected_tokens:
            overlap_score = max(overlap_score, 0.90)
        
        return overlap_score
    
    # Comprehensive beverage class synonyms - maps canonical name to all variants
    # This enables semantic matching: "IPA" matches "India Pale Ale"
    BEVERAGE_CLASS_SYNONYMS = {
        # === BEER ===
        "IPA": {
            "IPA", "INDIA PALE ALE", "INDIAN PALE ALE"
        },
        "DIPA": {
            "DIPA", "DOUBLE INDIA PALE ALE", "DOUBLE IPA", "IMPERIAL IPA"
        },
        "NEIPA": {
            "NEIPA", "NEW ENGLAND IPA", "NEW ENGLAND INDIA PALE ALE", "HAZY IPA"
        },
        "APA": {
            "APA", "AMERICAN PALE ALE", "PALE ALE"
        },
        "ESB": {
            "ESB", "EXTRA SPECIAL BITTER", "ENGLISH SPECIAL BITTER"
        },
        "STOUT": {
            "STOUT", "IMPERIAL STOUT", "MILK STOUT", "DRY STOUT", "OATMEAL STOUT"
        },
        "PORTER": {
            "PORTER", "ROBUST PORTER", "BALTIC PORTER"
        },
        "PILSNER": {
            "PILSNER", "PILS", "CZECH PILSNER", "GERMAN PILSNER"
        },
        "LAGER": {
            "LAGER", "PALE LAGER", "AMBER LAGER", "DARK LAGER"
        },
        "HELLES": {
            "HELLES", "HELLES LAGER", "GERMAN HELLES"
        },
        "BOCK": {
            "BOCK", "MAIBOCK", "DOPPELBOCK", "EISBOCK"
        },
        "WHEAT BEER": {
            "WHEAT BEER", "HEFEWEIZEN", "WEISSBIER", "WITBIER", "WHITE ALE"
        },
        "SAISON": {
            "SAISON", "FARMHOUSE ALE"
        },
        "SOUR": {
            "SOUR", "SOUR ALE", "GOSE", "BERLINER WEISSE"
        },
        "KOLSCH": {
            "KOLSCH", "KÖLSCH"
        },
        "BITTER": {
            "BITTER", "ENGLISH BITTER"
        },
        "ALE": {
            "ALE", "PALE ALE", "BROWN ALE", "GOLDEN ALE"
        },
        "CRAFT BEER": {
            "CRAFT BEER", "CRAFT ALE", "CRAFT LAGER"
        },
        
        # === SPIRITS (CRITICAL - labels rarely use formal names) ===
        "BOURBON": {
            "BOURBON", "KENTUCKY STRAIGHT BOURBON WHISKEY", "STRAIGHT BOURBON WHISKEY",
            "BOURBON WHISKEY", "KENTUCKY BOURBON", "STRAIGHT BOURBON"
        },
        "RYE": {
            "RYE", "RYE WHISKEY", "STRAIGHT RYE WHISKEY", "RYE WHISKY"
        },
        "TENNESSEE WHISKEY": {
            "TENNESSEE WHISKEY", "TENNESSEE WHISKY"
        },
        "SCOTCH": {
            "SCOTCH", "SCOTCH WHISKY", "SINGLE MALT SCOTCH", "BLENDED SCOTCH",
            "SINGLE MALT", "BLENDED WHISKY"
        },
        "IRISH WHISKEY": {
            "IRISH WHISKEY", "IRISH WHISKY"
        },
        "WHISKEY": {
            "WHISKEY", "WHISKY"
        },
        "VODKA": {
            "VODKA"
        },
        "GIN": {
            "GIN", "LONDON DRY GIN", "DRY GIN"
        },
        "RUM": {
            "RUM", "DARK RUM", "LIGHT RUM", "SPICED RUM", "WHITE RUM", "GOLD RUM"
        },
        "TEQUILA": {
            "TEQUILA", "BLANCO TEQUILA", "REPOSADO TEQUILA", "AÑEJO TEQUILA",
            "SILVER TEQUILA", "GOLD TEQUILA"
        },
        "MEZCAL": {
            "MEZCAL"
        },
        "BRANDY": {
            "BRANDY", "COGNAC", "ARMAGNAC"
        },
        "LIQUEUR": {
            "LIQUEUR", "CORDIAL", "SCHNAPPS"
        },
        
        # === WINE ===
        "WINE": {
            "WINE", "RED WINE", "WHITE WINE", "ROSE WINE", "ROSÉ"
        },
        "CABERNET SAUVIGNON": {
            "CABERNET SAUVIGNON", "CAB SAUV", "CABERNET"
        },
        "CHARDONNAY": {
            "CHARDONNAY", "CHARD"
        },
        "PINOT NOIR": {
            "PINOT NOIR", "PINOT"
        },
        "SAUVIGNON BLANC": {
            "SAUVIGNON BLANC", "SAV BLANC"
        },
        "MERLOT": {
            "MERLOT"
        },
        "SPARKLING WINE": {
            "SPARKLING WINE", "CHAMPAGNE", "PROSECCO", "CAVA", "SPARKLING"
        },
        
        # === OTHER ===
        "MALT BEVERAGE": {
            "MALT BEVERAGE", "FLAVORED MALT BEVERAGE", "FMB"
        },
        "HARD SELTZER": {
            "HARD SELTZER", "SPIKED SELTZER", "SELTZER"
        },
        "CIDER": {
            "CIDER", "HARD CIDER"
        },
    }
    
    # Build reverse lookup: variant -> canonical
    _VARIANT_TO_CANONICAL = None
    
    @classmethod
    def _get_variant_to_canonical(cls) -> dict:
        """Build/get reverse lookup from variant to canonical name."""
        if cls._VARIANT_TO_CANONICAL is None:
            cls._VARIANT_TO_CANONICAL = {}
            for canonical, variants in cls.BEVERAGE_CLASS_SYNONYMS.items():
                for variant in variants:
                    cls._VARIANT_TO_CANONICAL[variant.upper()] = canonical
        return cls._VARIANT_TO_CANONICAL
    
    def _verify_class_type(
        self,
        extracted: ExtractedField,
        expected: str
    ) -> FieldVerification:
        """
        Verify class/type using semantic classification + CANDIDATE RESCORING.
        
        Maps both extracted and expected to canonical forms, then compares.
        This handles:
        - "IPA" matches "India Pale Ale"
        - "Kentucky Straight Bourbon Whiskey" matches "Bourbon"
        - "Cognac" matches "Brandy"
        - "Champagne" matches "Sparkling Wine"
        - OCR garbling: rescore candidates to find best match
        """
        # Import the candidate scoring function
        from .extraction import pick_best_class_type_candidate
        
        if not extracted.value:
            # If no value but we have candidates, try to find best match
            if hasattr(extracted, 'candidates') and extracted.candidates:
                best, score, reason = pick_best_class_type_candidate(extracted.candidates, expected)
                if best and score >= 0.60:
                    logger.info(f"Class/type rescued from candidates: '{best}' (score={score:.2f})")
                    # Check if this matches expected canonically
                    best_canonical = self._get_canonical_class(best)
                    expected_canonical = self._get_canonical_class(expected)
                    
                    if best_canonical and expected_canonical and best_canonical == expected_canonical:
                        return FieldVerification(
                            field_name="Class/Type",
                            status=VerificationStatus.MATCH,
                            extracted_value=best,
                            expected_value=expected,
                            confidence=1.0,
                            message="Class/type matches",
                            details=f"Semantic match: both are '{best_canonical}' (found via candidate search)"
                        )
                    
                    return FieldVerification(
                        field_name="Class/Type",
                        status=VerificationStatus.REVIEW if score < 0.80 else VerificationStatus.MATCH,
                        extracted_value=best,
                        expected_value=expected,
                        confidence=score,
                        message="Class/type found via candidate matching",
                        details=f"Best match from {len(extracted.candidates)} candidates ({reason})"
                    )
            
            return FieldVerification(
                field_name="Class/Type",
                status=VerificationStatus.NOT_FOUND,
                extracted_value=None,
                expected_value=expected,
                confidence=0.0,
                message="Class/type not detected on label",
                details="No beverage type keywords found. Manual review required."
            )
        
        # Map both to canonical forms
        extracted_canonical = self._get_canonical_class(extracted.value)
        expected_canonical = self._get_canonical_class(expected)
        
        logger.debug(f"Class/type: extracted='{extracted.value}' -> '{extracted_canonical}', "
                    f"expected='{expected}' -> '{expected_canonical}'")
        
        # Canonical match = definitive pass
        if extracted_canonical and expected_canonical:
            if extracted_canonical == expected_canonical:
                return FieldVerification(
                    field_name="Class/Type",
                    status=VerificationStatus.MATCH,
                    extracted_value=extracted.value,
                    expected_value=expected,
                    confidence=1.0,
                    message="Class/type matches",
                    details=f"Semantic match: both are '{extracted_canonical}'"
                )
        
        # Fallback to fuzzy matching for edge cases
        extracted_norm = self._normalize_text(extracted.value)
        expected_norm = self._normalize_text(expected)
        
        # Use token set ratio - more lenient for class/type
        score = fuzz.token_set_ratio(extracted_norm, expected_norm) / 100.0
        
        # Also check if key terms are present
        expected_terms = set(expected_norm.split())
        extracted_terms = set(extracted_norm.split())
        common_terms = expected_terms & extracted_terms
        term_overlap = len(common_terms) / len(expected_terms) if expected_terms else 0
        
        # Combine scores
        final_score = max(score, term_overlap)
        
        if final_score >= 0.90:
            return FieldVerification(
                field_name="Class/Type",
                status=VerificationStatus.MATCH,
                extracted_value=extracted.value,
                expected_value=expected,
                confidence=final_score,
                message="Class/type matches",
                details=None
            )
        elif final_score >= 0.70:
            return FieldVerification(
                field_name="Class/Type",
                status=VerificationStatus.REVIEW,
                extracted_value=extracted.value,
                expected_value=expected,
                confidence=final_score,
                message="Class/type partially matches - review recommended",
                details=f"Label shows '{extracted.value}'. Expected '{expected}'."
            )
        else:
            return FieldVerification(
                field_name="Class/Type",
                status=VerificationStatus.MISMATCH,
                extracted_value=extracted.value,
                expected_value=expected,
                confidence=final_score,
                message="Class/type does not match",
                details=f"Label shows '{extracted.value}' but application states '{expected}'."
            )
    
    def _get_canonical_class(self, class_type: str) -> Optional[str]:
        """
        Map a class/type string to its canonical form.
        
        Examples:
        - "India Pale Ale" -> "IPA"
        - "Kentucky Straight Bourbon Whiskey" -> "BOURBON"
        - "Cognac" -> "BRANDY"
        - "Champagne" -> "SPARKLING WINE"
        
        Returns None if no canonical form found.
        """
        if not class_type:
            return None
        
        variant_map = self._get_variant_to_canonical()
        text_upper = class_type.upper().strip()
        
        # Direct lookup
        if text_upper in variant_map:
            return variant_map[text_upper]
        
        # Check if any variant is contained in the text (for longer strings)
        # Sort by length descending to match longer variants first
        for variant in sorted(variant_map.keys(), key=len, reverse=True):
            if variant in text_upper:
                return variant_map[variant]
        
        # Check if text contains any variant
        text_tokens = set(text_upper.split())
        for variant, canonical in variant_map.items():
            variant_tokens = set(variant.split())
            # If all variant tokens are in text, it's a match
            if variant_tokens and variant_tokens <= text_tokens:
                return canonical
        
        return None
    
    def _verify_abv(
        self,
        extracted: ExtractedField,
        expected: float
    ) -> FieldVerification:
        """
        Verify ABV with numeric comparison.
        
        Includes post-processor for common OCR errors:
        - "68" vs "6.8" (missing decimal point)
        - "680" vs "6.8" (OCR reads extra digit)
        
        Tolerance: ±0.5% to account for rounding differences.
        """
        if not extracted.value:
            return FieldVerification(
                field_name="Alcohol Content (ABV)",
                status=VerificationStatus.NOT_FOUND,
                extracted_value=None,
                expected_value=f"{expected}%",
                confidence=0.0,
                message="ABV not detected on label",
                details="No alcohol percentage pattern found. Manual review required."
            )
        
        try:
            extracted_value = float(extracted.value)
        except (ValueError, TypeError):
            return FieldVerification(
                field_name="Alcohol Content (ABV)",
                status=VerificationStatus.REVIEW,
                extracted_value=extracted.value,
                expected_value=f"{expected}%",
                confidence=0.5,
                message="ABV format unclear - review recommended",
                details=f"Extracted '{extracted.value}' could not be parsed as number."
            )
        
        # ABV POST-PROCESSOR: Fix common OCR decimal errors
        # "68" vs "6.8" is extremely common on beer labels
        corrected_value = self._correct_abv_decimal(extracted_value, expected)
        if corrected_value != extracted_value:
            logger.info(f"ABV corrected: {extracted_value}% -> {corrected_value}% (expected {expected}%)")
            extracted_value = corrected_value
        
        difference = abs(extracted_value - expected)
        tolerance = self.settings.abv_tolerance
        
        if difference <= tolerance:
            return FieldVerification(
                field_name="Alcohol Content (ABV)",
                status=VerificationStatus.MATCH,
                extracted_value=f"{extracted_value}%",
                expected_value=f"{expected}%",
                confidence=extracted.confidence,
                message="ABV matches",
                details=f"Label: {extracted_value}%, Application: {expected}%" if difference > 0 else None
            )
        else:
            return FieldVerification(
                field_name="Alcohol Content (ABV)",
                status=VerificationStatus.MISMATCH,
                extracted_value=f"{extracted_value}%",
                expected_value=f"{expected}%",
                confidence=extracted.confidence,
                message="ABV does not match",
                details=f"Label shows {extracted_value}% but application states {expected}%. "
                        f"Difference: {difference}% (tolerance: ±{tolerance}%)"
            )
    
    def _correct_abv_decimal(self, extracted: float, expected: float) -> float:
        """
        Post-processor to fix common ABV OCR errors.
        
        Pattern: OCR reads "68" instead of "6.8" (missing decimal)
        
        Rule (per engineer):
        If extracted >= 20% and expected < 20%:
            try dividing by 10 and by 100
            pick the one closer to expected
        
        This is SAFE because:
        - Real ABV > 20% is rare (some spirits, but they're >40%)
        - Beer/wine is always < 20%
        - If extracted is 68 and expected is 6.8, division by 10 gives exact match
        
        Examples:
        - 68 -> 6.8 (÷10) when expected is 6.8 ✓
        - 680 -> 6.8 (÷100) when expected is 6.8 ✓
        - 45 -> 45 (no change) when expected is 45 (spirits) ✓
        - 12 -> 12 (no change) when expected is 12 (wine) ✓
        """
        # Only apply correction in suspicious range
        if extracted < 20 or expected >= 20:
            return extracted
        
        # Try both corrections
        div_10 = extracted / 10
        div_100 = extracted / 100
        
        # Pick the one closest to expected
        diff_original = abs(extracted - expected)
        diff_10 = abs(div_10 - expected)
        diff_100 = abs(div_100 - expected)
        
        # Only correct if it's a significant improvement
        tolerance = self.settings.abv_tolerance
        
        if diff_10 <= tolerance and diff_10 < diff_original:
            logger.debug(f"ABV decimal correction: {extracted} -> {div_10} (÷10)")
            return div_10
        
        if diff_100 <= tolerance and diff_100 < diff_original:
            logger.debug(f"ABV decimal correction: {extracted} -> {div_100} (÷100)")
            return div_100
        
        return extracted
    
    def _verify_net_contents(
        self,
        extracted: ExtractedField,
        expected: float
    ) -> FieldVerification:
        """
        Verify net contents with numeric comparison.
        
        Must be exact match (after unit normalization).
        """
        if not extracted.value:
            return FieldVerification(
                field_name="Net Contents",
                status=VerificationStatus.NOT_FOUND,
                extracted_value=None,
                expected_value=f"{expected} mL",
                confidence=0.0,
                message="Net contents not detected on label",
                details="No volume/contents pattern found. Manual review required."
            )
        
        try:
            extracted_value = float(extracted.value)
        except (ValueError, TypeError):
            return FieldVerification(
                field_name="Net Contents",
                status=VerificationStatus.REVIEW,
                extracted_value=extracted.value,
                expected_value=f"{expected} mL",
                confidence=0.5,
                message="Net contents format unclear - review recommended",
                details=f"Extracted '{extracted.value}' could not be parsed."
            )
        
        # Allow small tolerance for floating point
        tolerance = 1.0  # 1 mL tolerance
        difference = abs(extracted_value - expected)
        
        if difference <= tolerance:
            return FieldVerification(
                field_name="Net Contents",
                status=VerificationStatus.MATCH,
                extracted_value=f"{extracted_value} mL",
                expected_value=f"{expected} mL",
                confidence=extracted.confidence,
                message="Net contents matches",
                details=None
            )
        else:
            return FieldVerification(
                field_name="Net Contents",
                status=VerificationStatus.MISMATCH,
                extracted_value=f"{extracted_value} mL",
                expected_value=f"{expected} mL",
                confidence=extracted.confidence,
                message="Net contents does not match",
                details=f"Label shows {extracted_value} mL but application states {expected} mL."
            )
    
    def _verify_warning(
        self,
        extracted: ExtractedField,
        expected_present: bool
    ) -> FieldVerification:
        """
        Verify government warning presence.
        
        Strict: warning must be detected if expected.
        """
        warning_detected = extracted.value in ["detected", "partial"]
        
        if not expected_present:
            # Warning not required (unusual but handle it)
            return FieldVerification(
                field_name="Government Warning",
                status=VerificationStatus.NOT_APPLICABLE,
                extracted_value=extracted.value,
                expected_value="Not required",
                confidence=1.0,
                message="Government warning check skipped",
                details="Warning not required per application."
            )
        
        if extracted.value == "detected":
            return FieldVerification(
                field_name="Government Warning",
                status=VerificationStatus.MATCH,
                extracted_value="Present",
                expected_value="Required",
                confidence=extracted.confidence,
                message="Government warning detected",
                details="Required warning text found on label."
            )
        elif extracted.value == "partial":
            return FieldVerification(
                field_name="Government Warning",
                status=VerificationStatus.REVIEW,
                extracted_value="Partial",
                expected_value="Required",
                confidence=extracted.confidence,
                message="Government warning partially detected - review recommended",
                details="Some warning text found but may be incomplete. "
                        "Verify 'GOVERNMENT WARNING:' is in all caps and complete text is present."
            )
        else:
            return FieldVerification(
                field_name="Government Warning",
                status=VerificationStatus.MISMATCH,
                extracted_value="Not found",
                expected_value="Required",
                confidence=0.0,
                message="Government warning not detected",
                details="Required government health warning not found on label. "
                        "All alcohol beverages must display the mandatory warning statement."
            )
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for comparison."""
        # Lowercase
        text = text.lower()
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        # Normalize punctuation
        text = text.replace("'", "'").replace("'", "'")
        text = text.replace(""", '"').replace(""", '"')
        # Remove special characters but keep alphanumeric and spaces
        text = re.sub(r'[^\w\s]', '', text)
        return text.strip()
    
    def _generate_summary(
        self,
        fields: List[FieldVerification],
        overall_status: VerificationStatus
    ) -> str:
        """Generate human-readable summary."""
        if overall_status == VerificationStatus.MATCH:
            return "✅ All fields verified successfully. Label matches application data."
        
        issues = []
        for f in fields:
            if f.status == VerificationStatus.MISMATCH:
                issues.append(f"❌ {f.field_name}: {f.message}")
            elif f.status == VerificationStatus.NOT_FOUND:
                issues.append(f"❌ {f.field_name}: {f.message}")
            elif f.status == VerificationStatus.REVIEW:
                issues.append(f"⚠️ {f.field_name}: {f.message}")
        
        if overall_status == VerificationStatus.MISMATCH:
            header = "❌ Verification failed. Issues found:"
        else:
            header = "⚠️ Review recommended. Potential issues:"
        
        return header + "\n" + "\n".join(issues)
