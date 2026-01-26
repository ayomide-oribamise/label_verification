"""Tests for verification service."""

import pytest
from app.services.verification import VerificationService, VerificationStatus
from app.services.extraction import ExtractionResult, ExtractedField
from app.services.ocr import OCRBox


@pytest.fixture
def service():
    """Create verification service instance."""
    return VerificationService()


def make_extraction_result(
    brand: str = None,
    class_type: str = None,
    abv: str = None,
    net_contents: str = None,
    warning: str = "not_found",
    confidence: float = 0.95
) -> ExtractionResult:
    """Helper to create ExtractionResult for testing."""
    return ExtractionResult(
        brand_name=ExtractedField(
            value=brand,
            confidence=confidence if brand else 0.0,
            extraction_method="test"
        ),
        class_type=ExtractedField(
            value=class_type,
            confidence=confidence if class_type else 0.0,
            extraction_method="test"
        ),
        abv_percent=ExtractedField(
            value=abv,
            confidence=confidence if abv else 0.0,
            extraction_method="test"
        ),
        net_contents_ml=ExtractedField(
            value=net_contents,
            confidence=confidence if net_contents else 0.0,
            extraction_method="test"
        ),
        government_warning=ExtractedField(
            value=warning,
            confidence=confidence if warning != "not_found" else 0.0,
            extraction_method="test"
        ),
        raw_text="test text",
        overall_confidence=confidence
    )


class TestBrandVerification:
    """Test brand name verification."""
    
    def test_exact_match(self, service):
        """Test exact brand match."""
        extraction = make_extraction_result(brand="OLD TOM DISTILLERY")
        result = service.verify(extraction, expected_brand="OLD TOM DISTILLERY")
        
        brand_field = next(f for f in result.fields if f.field_name == "Brand Name")
        assert brand_field.status == VerificationStatus.MATCH
    
    def test_case_insensitive_match(self, service):
        """Test case-insensitive matching."""
        extraction = make_extraction_result(brand="OLD TOM DISTILLERY")
        result = service.verify(extraction, expected_brand="Old Tom Distillery")
        
        brand_field = next(f for f in result.fields if f.field_name == "Brand Name")
        assert brand_field.status == VerificationStatus.MATCH
    
    def test_punctuation_difference(self, service):
        """Test handling of punctuation differences."""
        extraction = make_extraction_result(brand="STONE'S THROW")
        result = service.verify(extraction, expected_brand="Stone's Throw")
        
        brand_field = next(f for f in result.fields if f.field_name == "Brand Name")
        assert brand_field.status == VerificationStatus.MATCH
    
    def test_slight_mismatch_review(self, service):
        """Test that slight mismatches trigger review."""
        extraction = make_extraction_result(brand="OLD TOM DISTILERY")  # Typo
        result = service.verify(extraction, expected_brand="OLD TOM DISTILLERY")
        
        brand_field = next(f for f in result.fields if f.field_name == "Brand Name")
        # Should be review or match depending on threshold
        assert brand_field.status in [VerificationStatus.MATCH, VerificationStatus.REVIEW]
    
    def test_complete_mismatch(self, service):
        """Test complete brand mismatch."""
        extraction = make_extraction_result(brand="JACK DANIELS")
        result = service.verify(extraction, expected_brand="OLD TOM DISTILLERY")
        
        brand_field = next(f for f in result.fields if f.field_name == "Brand Name")
        assert brand_field.status == VerificationStatus.MISMATCH
    
    def test_brand_not_found(self, service):
        """Test when brand not detected."""
        extraction = make_extraction_result(brand=None)
        result = service.verify(extraction, expected_brand="OLD TOM DISTILLERY")
        
        brand_field = next(f for f in result.fields if f.field_name == "Brand Name")
        assert brand_field.status == VerificationStatus.NOT_FOUND


class TestABVVerification:
    """Test ABV verification."""
    
    def test_exact_match(self, service):
        """Test exact ABV match."""
        extraction = make_extraction_result(brand="TEST", abv="45.0")
        result = service.verify(extraction, expected_brand="TEST", expected_abv=45.0)
        
        abv_field = next(f for f in result.fields if f.field_name == "Alcohol Content (ABV)")
        assert abv_field.status == VerificationStatus.MATCH
    
    def test_within_tolerance(self, service):
        """Test ABV within tolerance."""
        extraction = make_extraction_result(brand="TEST", abv="45.3")
        result = service.verify(extraction, expected_brand="TEST", expected_abv=45.0)
        
        abv_field = next(f for f in result.fields if f.field_name == "Alcohol Content (ABV)")
        assert abv_field.status == VerificationStatus.MATCH
    
    def test_outside_tolerance(self, service):
        """Test ABV outside tolerance."""
        extraction = make_extraction_result(brand="TEST", abv="47.0")
        result = service.verify(extraction, expected_brand="TEST", expected_abv=45.0)
        
        abv_field = next(f for f in result.fields if f.field_name == "Alcohol Content (ABV)")
        assert abv_field.status == VerificationStatus.MISMATCH
    
    def test_abv_not_found(self, service):
        """Test when ABV not detected."""
        extraction = make_extraction_result(brand="TEST", abv=None)
        result = service.verify(extraction, expected_brand="TEST", expected_abv=45.0)
        
        abv_field = next(f for f in result.fields if f.field_name == "Alcohol Content (ABV)")
        assert abv_field.status == VerificationStatus.NOT_FOUND


class TestNetContentsVerification:
    """Test net contents verification."""
    
    def test_exact_match(self, service):
        """Test exact net contents match."""
        extraction = make_extraction_result(brand="TEST", net_contents="750.0")
        result = service.verify(extraction, expected_brand="TEST", expected_net_contents=750.0)
        
        net_field = next(f for f in result.fields if f.field_name == "Net Contents")
        assert net_field.status == VerificationStatus.MATCH
    
    def test_mismatch(self, service):
        """Test net contents mismatch."""
        extraction = make_extraction_result(brand="TEST", net_contents="1000.0")
        result = service.verify(extraction, expected_brand="TEST", expected_net_contents=750.0)
        
        net_field = next(f for f in result.fields if f.field_name == "Net Contents")
        assert net_field.status == VerificationStatus.MISMATCH
    
    def test_not_found(self, service):
        """Test when net contents not detected."""
        extraction = make_extraction_result(brand="TEST", net_contents=None)
        result = service.verify(extraction, expected_brand="TEST", expected_net_contents=750.0)
        
        net_field = next(f for f in result.fields if f.field_name == "Net Contents")
        assert net_field.status == VerificationStatus.NOT_FOUND


class TestGovernmentWarningVerification:
    """Test government warning verification."""
    
    def test_warning_detected(self, service):
        """Test when warning is detected."""
        extraction = make_extraction_result(brand="TEST", warning="detected")
        result = service.verify(extraction, expected_brand="TEST", expected_has_warning=True)
        
        warning_field = next(f for f in result.fields if f.field_name == "Government Warning")
        assert warning_field.status == VerificationStatus.MATCH
    
    def test_warning_partial(self, service):
        """Test when warning is partially detected."""
        extraction = make_extraction_result(brand="TEST", warning="partial")
        result = service.verify(extraction, expected_brand="TEST", expected_has_warning=True)
        
        warning_field = next(f for f in result.fields if f.field_name == "Government Warning")
        assert warning_field.status == VerificationStatus.REVIEW
    
    def test_warning_not_found(self, service):
        """Test when warning is not found but required."""
        extraction = make_extraction_result(brand="TEST", warning="not_found")
        result = service.verify(extraction, expected_brand="TEST", expected_has_warning=True)
        
        warning_field = next(f for f in result.fields if f.field_name == "Government Warning")
        assert warning_field.status == VerificationStatus.MISMATCH
    
    def test_warning_not_required(self, service):
        """Test when warning is not required."""
        extraction = make_extraction_result(brand="TEST", warning="not_found")
        result = service.verify(extraction, expected_brand="TEST", expected_has_warning=False)
        
        warning_field = next(f for f in result.fields if f.field_name == "Government Warning")
        assert warning_field.status == VerificationStatus.NOT_APPLICABLE


class TestOverallVerification:
    """Test overall verification results."""
    
    def test_all_pass(self, service):
        """Test when all fields pass."""
        extraction = make_extraction_result(
            brand="OLD TOM DISTILLERY",
            class_type="Kentucky Bourbon",
            abv="45.0",
            net_contents="750.0",
            warning="detected"
        )
        result = service.verify(
            extraction,
            expected_brand="OLD TOM DISTILLERY",
            expected_class_type="Kentucky Bourbon",
            expected_abv=45.0,
            expected_net_contents=750.0,
            expected_has_warning=True
        )
        
        assert result.overall_status == VerificationStatus.MATCH
        assert result.failed_count == 0
        assert "✅" in result.summary
    
    def test_one_failure(self, service):
        """Test when one field fails."""
        extraction = make_extraction_result(
            brand="OLD TOM DISTILLERY",
            abv="40.0",  # Wrong ABV
            warning="detected"
        )
        result = service.verify(
            extraction,
            expected_brand="OLD TOM DISTILLERY",
            expected_abv=45.0,
            expected_has_warning=True
        )
        
        assert result.overall_status == VerificationStatus.MISMATCH
        assert result.failed_count >= 1
        assert "❌" in result.summary
    
    def test_review_needed(self, service):
        """Test when review is needed but no failures."""
        extraction = make_extraction_result(
            brand="OLD TOM DISTILLERY",
            warning="partial"  # Partial detection
        )
        result = service.verify(
            extraction,
            expected_brand="OLD TOM DISTILLERY",
            expected_has_warning=True
        )
        
        assert result.overall_status == VerificationStatus.REVIEW
        assert result.review_count >= 1
        assert "⚠️" in result.summary


class TestPlainEnglishMessages:
    """Test that messages are human-readable."""
    
    def test_brand_match_message(self, service):
        """Test brand match message."""
        extraction = make_extraction_result(brand="OLD TOM DISTILLERY")
        result = service.verify(extraction, expected_brand="OLD TOM DISTILLERY")
        
        brand_field = next(f for f in result.fields if f.field_name == "Brand Name")
        assert "match" in brand_field.message.lower()
    
    def test_brand_mismatch_message(self, service):
        """Test brand mismatch message is descriptive."""
        extraction = make_extraction_result(brand="JACK DANIELS")
        result = service.verify(extraction, expected_brand="OLD TOM DISTILLERY")
        
        brand_field = next(f for f in result.fields if f.field_name == "Brand Name")
        assert "does not match" in brand_field.message.lower()
        # Should mention both values
        assert "JACK DANIELS" in (brand_field.details or brand_field.message)
        assert "OLD TOM DISTILLERY" in (brand_field.details or brand_field.message)
    
    def test_warning_not_found_message(self, service):
        """Test warning not found message is clear."""
        extraction = make_extraction_result(brand="TEST", warning="not_found")
        result = service.verify(extraction, expected_brand="TEST", expected_has_warning=True)
        
        warning_field = next(f for f in result.fields if f.field_name == "Government Warning")
        assert "not detected" in warning_field.message.lower() or "not found" in warning_field.message.lower()


class TestEdgeCases:
    """Test edge cases."""
    
    def test_empty_extraction(self, service):
        """Test with all fields missing."""
        extraction = make_extraction_result(
            brand=None,
            class_type=None,
            abv=None,
            net_contents=None,
            warning="not_found"
        )
        result = service.verify(
            extraction,
            expected_brand="TEST",
            expected_class_type="Bourbon",
            expected_abv=45.0,
            expected_net_contents=750.0,
            expected_has_warning=True
        )
        
        assert result.overall_status == VerificationStatus.MISMATCH
        assert result.failed_count >= 1
    
    def test_optional_fields_not_provided(self, service):
        """Test when optional expected fields are not provided."""
        extraction = make_extraction_result(
            brand="TEST BRAND",
            class_type="Kentucky Bourbon",
            abv="45.0",
            net_contents="750.0",
            warning="detected"
        )
        # Only provide brand
        result = service.verify(extraction, expected_brand="TEST BRAND")
        
        # Should only verify brand and warning
        assert len(result.fields) == 2  # Brand + Warning
        assert result.overall_status == VerificationStatus.MATCH
    
    def test_unicode_in_brand(self, service):
        """Test unicode characters in brand name."""
        extraction = make_extraction_result(brand="CHÂTEAU MARGAUX")
        result = service.verify(extraction, expected_brand="Château Margaux")
        
        brand_field = next(f for f in result.fields if f.field_name == "Brand Name")
        assert brand_field.status == VerificationStatus.MATCH


class TestConfidenceScores:
    """Test confidence score handling."""
    
    def test_high_confidence_match(self, service):
        """Test high confidence match."""
        extraction = make_extraction_result(brand="TEST", confidence=0.98)
        result = service.verify(extraction, expected_brand="TEST")
        
        brand_field = next(f for f in result.fields if f.field_name == "Brand Name")
        assert brand_field.confidence >= 0.95
    
    def test_confidence_in_response(self, service):
        """Test that confidence is included in response."""
        extraction = make_extraction_result(brand="TEST", abv="45.0", confidence=0.90)
        result = service.verify(extraction, expected_brand="TEST", expected_abv=45.0)
        
        for field in result.fields:
            assert hasattr(field, 'confidence')
