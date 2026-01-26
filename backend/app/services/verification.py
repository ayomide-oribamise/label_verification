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
        Verify brand name using fuzzy matching.
        
        Thresholds:
        - >= 0.95: Match
        - 0.85-0.94: Review (likely match)
        - < 0.85: Mismatch
        """
        if not extracted.value:
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
        extracted_norm = self._normalize_text(extracted.value)
        expected_norm = self._normalize_text(expected)
        
        # Calculate similarity score
        similarity = fuzz.ratio(extracted_norm, expected_norm) / 100.0
        
        # Also try token sort ratio for word order differences
        token_similarity = fuzz.token_sort_ratio(extracted_norm, expected_norm) / 100.0
        
        # Use the higher score
        score = max(similarity, token_similarity)
        
        if score >= self.settings.brand_match_threshold:
            return FieldVerification(
                field_name="Brand Name",
                status=VerificationStatus.MATCH,
                extracted_value=extracted.value,
                expected_value=expected,
                confidence=score,
                message="Brand name matches",
                details=f"Similarity score: {score:.0%}" if score < 1.0 else None
            )
        elif score >= self.settings.brand_review_threshold:
            return FieldVerification(
                field_name="Brand Name",
                status=VerificationStatus.REVIEW,
                extracted_value=extracted.value,
                expected_value=expected,
                confidence=score,
                message="Brand name likely matches - review recommended",
                details=f"'{extracted.value}' vs '{expected}' - similarity {score:.0%}. "
                        "May be a case/punctuation difference."
            )
        else:
            return FieldVerification(
                field_name="Brand Name",
                status=VerificationStatus.MISMATCH,
                extracted_value=extracted.value,
                expected_value=expected,
                confidence=score,
                message="Brand name does not match",
                details=f"Label shows '{extracted.value}' but application states '{expected}'. "
                        f"Similarity: {score:.0%}"
            )
    
    def _verify_class_type(
        self,
        extracted: ExtractedField,
        expected: str
    ) -> FieldVerification:
        """
        Verify class/type using fuzzy matching.
        
        More lenient than brand - focuses on key terms.
        """
        if not extracted.value:
            return FieldVerification(
                field_name="Class/Type",
                status=VerificationStatus.NOT_FOUND,
                extracted_value=None,
                expected_value=expected,
                confidence=0.0,
                message="Class/type not detected on label",
                details="No beverage type keywords found. Manual review required."
            )
        
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
    
    def _verify_abv(
        self,
        extracted: ExtractedField,
        expected: float
    ) -> FieldVerification:
        """
        Verify ABV with numeric comparison.
        
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
