"""Tests for field extraction service."""

import pytest
from app.services.extraction import (
    FieldExtractor, 
    extract_abv_value, 
    extract_net_contents_ml,
    GOVERNMENT_WARNING_CANONICAL,
)
from app.services.ocr import OCRResult, OCRBox


@pytest.fixture
def extractor():
    """Create extractor instance."""
    return FieldExtractor()


def make_ocr_result(texts_with_positions: list) -> OCRResult:
    """
    Helper to create OCRResult from list of (text, top, height) tuples.
    
    Args:
        texts_with_positions: List of (text, top_y, height) tuples
        
    Returns:
        OCRResult with boxes at specified positions
    """
    boxes = []
    for text, top, height in texts_with_positions:
        bottom = top + height
        boxes.append(OCRBox(
            text=text,
            confidence=0.95,
            bbox=[[0, top], [200, top], [200, bottom], [0, bottom]]
        ))
    
    raw_text = " ".join(t[0] for t in texts_with_positions)
    return OCRResult(boxes=boxes, raw_text=raw_text, average_confidence=0.95)


class TestABVExtraction:
    """Test ABV extraction patterns."""
    
    def test_simple_percentage(self):
        """Test simple percentage format."""
        assert extract_abv_value("45%") == 45.0
        assert extract_abv_value("12.5%") == 12.5
    
    def test_alc_vol_format(self):
        """Test Alc./Vol. format."""
        assert extract_abv_value("45% Alc./Vol.") == 45.0
        assert extract_abv_value("45% ALC/VOL") == 45.0
        assert extract_abv_value("45% alc by vol") == 45.0
    
    def test_alcohol_by_volume(self):
        """Test full 'Alcohol by Volume' format."""
        assert extract_abv_value("45% Alcohol by Volume") == 45.0
    
    def test_proof_conversion(self):
        """Test proof to percentage conversion."""
        assert extract_abv_value("90 Proof") == 45.0
        assert extract_abv_value("80 proof") == 40.0
        assert extract_abv_value("100 PROOF") == 50.0
    
    def test_combined_format(self):
        """Test label with both percentage and proof."""
        # Should return the percentage (matches first)
        result = extract_abv_value("45% Alc./Vol. (90 Proof)")
        assert result == 45.0
    
    def test_decimal_percentage(self):
        """Test decimal percentage."""
        assert extract_abv_value("12.5% Alc./Vol.") == 12.5
        assert extract_abv_value("5.5%") == 5.5
    
    def test_no_match(self):
        """Test text with no ABV."""
        assert extract_abv_value("Kentucky Straight Bourbon") is None
        assert extract_abv_value("750 mL") is None


class TestNetContentsExtraction:
    """Test net contents extraction patterns."""
    
    def test_milliliters(self):
        """Test milliliter formats."""
        assert extract_net_contents_ml("750 mL") == 750.0
        assert extract_net_contents_ml("750ml") == 750.0
        assert extract_net_contents_ml("750 ML") == 750.0
        assert extract_net_contents_ml("375 mL") == 375.0
    
    def test_centiliters(self):
        """Test centiliter conversion."""
        assert extract_net_contents_ml("75 cL") == 750.0
        assert extract_net_contents_ml("75cl") == 750.0
        assert extract_net_contents_ml("50 CL") == 500.0
    
    def test_liters(self):
        """Test liter conversion."""
        assert extract_net_contents_ml("1 L") == 1000.0
        assert extract_net_contents_ml("1L") == 1000.0
        assert extract_net_contents_ml("1 LITER") == 1000.0
        assert extract_net_contents_ml("1.5 L") == 1500.0
        assert extract_net_contents_ml("1.75 L") == 1750.0
    
    def test_fluid_ounces(self):
        """Test fluid ounce conversion."""
        result = extract_net_contents_ml("25.4 FL OZ")
        assert result is not None
        assert 740 < result < 760  # Should be ~750 mL
        
        result = extract_net_contents_ml("25.4 fl oz")
        assert result is not None
        assert 740 < result < 760
    
    def test_no_match(self):
        """Test text with no net contents."""
        assert extract_net_contents_ml("45% Alc./Vol.") is None
        assert extract_net_contents_ml("Kentucky Bourbon") is None


class TestBrandExtraction:
    """Test brand name extraction."""
    
    def test_top_prominent_text(self, extractor):
        """Test extraction of top prominent text."""
        ocr_result = make_ocr_result([
            ("OLD TOM DISTILLERY", 0, 50),    # Top, large
            ("Kentucky Straight Bourbon Whiskey", 60, 25),  # Below, smaller
            ("45% Alc./Vol.", 100, 15),
            ("750 mL", 120, 15),
        ])
        
        result = extractor.extract_all(ocr_result)
        assert result.brand_name.value == "OLD TOM DISTILLERY"
    
    def test_ignores_beverage_keywords(self, extractor):
        """Test that beverage keywords are not taken as brand."""
        ocr_result = make_ocr_result([
            ("MAKER'S MARK", 0, 50),
            ("Kentucky Bourbon", 60, 30),  # Has keyword, shouldn't be brand
        ])
        
        result = extractor.extract_all(ocr_result)
        assert result.brand_name.value == "MAKER'S MARK"
    
    def test_ignores_abv_as_brand(self, extractor):
        """Test that ABV is not taken as brand."""
        ocr_result = make_ocr_result([
            ("45%", 0, 30),  # Should be ignored
            ("JACK DANIELS", 40, 50),
        ])
        
        result = extractor.extract_all(ocr_result)
        assert "JACK" in result.brand_name.value or "45" not in (result.brand_name.value or "")


class TestClassTypeExtraction:
    """Test class/type extraction."""
    
    def test_keyword_matching(self, extractor):
        """Test extraction via beverage keywords."""
        ocr_result = make_ocr_result([
            ("OLD TOM DISTILLERY", 0, 50),
            ("Kentucky Straight Bourbon Whiskey", 60, 25),
            ("45% Alc./Vol.", 100, 15),
        ])
        
        result = extractor.extract_all(ocr_result)
        assert "Bourbon" in result.class_type.value or "Whiskey" in result.class_type.value
    
    def test_multiple_keywords(self, extractor):
        """Test text with multiple beverage keywords."""
        ocr_result = make_ocr_result([
            ("BRAND NAME", 0, 50),
            ("Single Malt Scotch Whisky", 60, 30),
        ])
        
        result = extractor.extract_all(ocr_result)
        assert result.class_type.value is not None
        # Should contain whisky-related text
        assert any(kw in result.class_type.value.lower() for kw in ["scotch", "whisky", "malt"])


class TestGovernmentWarningExtraction:
    """Test government warning detection."""
    
    def test_full_warning_detected(self, extractor):
        """Test detection of full warning text."""
        warning_text = (
            "GOVERNMENT WARNING: (1) According to the Surgeon General, women should not "
            "drink alcoholic beverages during pregnancy because of the risk of birth defects. "
            "(2) Consumption of alcoholic beverages impairs your ability to drive a car or "
            "operate machinery, and may cause health problems."
        )
        
        ocr_result = make_ocr_result([
            ("BRAND", 0, 50),
            (warning_text, 100, 30),
        ])
        
        result = extractor.extract_all(ocr_result)
        assert result.government_warning.value == "detected"
    
    def test_partial_warning(self, extractor):
        """Test detection of partial warning."""
        ocr_result = make_ocr_result([
            ("BRAND", 0, 50),
            ("GOVERNMENT WARNING: According to the Surgeon General", 100, 20),
        ])
        
        result = extractor.extract_all(ocr_result)
        # Should detect or at least partially match
        assert result.government_warning.value in ["detected", "partial"]
    
    def test_no_warning(self, extractor):
        """Test when warning is not present."""
        ocr_result = make_ocr_result([
            ("BRAND NAME", 0, 50),
            ("Kentucky Bourbon", 60, 30),
            ("45% Alc./Vol.", 100, 15),
        ])
        
        result = extractor.extract_all(ocr_result)
        assert result.government_warning.value == "not_found"
    
    def test_warning_with_ocr_noise(self, extractor):
        """Test warning detection with OCR artifacts."""
        # Simulate OCR joining words
        ocr_result = make_ocr_result([
            ("BRAND", 0, 50),
            ("GOVERNMENTWARNING SURGEONGENERAL PREGNANCY BIRTHDEFECTS", 100, 20),
        ])
        
        result = extractor.extract_all(ocr_result)
        # Should still detect due to canonicalization
        assert result.government_warning.value in ["detected", "partial"]


class TestFullExtraction:
    """Test full extraction pipeline."""
    
    def test_complete_label(self, extractor):
        """Test extraction from a complete label."""
        ocr_result = make_ocr_result([
            ("OLD TOM DISTILLERY", 0, 60),
            ("Premium Small Batch", 70, 20),  # Marketing fluff
            ("Kentucky Straight Bourbon Whiskey", 100, 30),
            ("45% Alc./Vol. (90 Proof)", 140, 20),
            ("750 mL", 170, 20),
            ("GOVERNMENT WARNING: (1) According to the Surgeon General, women should not drink alcoholic beverages during pregnancy...", 200, 15),
        ])
        
        result = extractor.extract_all(ocr_result)
        
        # Check all fields extracted
        assert result.brand_name.value == "OLD TOM DISTILLERY"
        assert "Bourbon" in result.class_type.value or "Whiskey" in result.class_type.value
        assert result.abv_percent.value == "45.0"
        assert result.net_contents_ml.value == "750.0"
        assert result.government_warning.value in ["detected", "partial"]
        
        # Check overall confidence
        assert result.overall_confidence > 0.5
    
    def test_empty_ocr_result(self, extractor):
        """Test handling of empty OCR result."""
        ocr_result = OCRResult.empty()
        result = extractor.extract_all(ocr_result)
        
        assert result.brand_name.value is None
        assert result.class_type.value is None
        assert result.abv_percent.value is None
        assert result.net_contents_ml.value is None
        assert result.overall_confidence == 0.0


class TestConfidenceScoring:
    """Test confidence scoring."""
    
    def test_high_confidence_extraction(self, extractor):
        """Test that good OCR produces high confidence."""
        ocr_result = make_ocr_result([
            ("OLD TOM DISTILLERY", 0, 60),
            ("Kentucky Bourbon", 70, 30),
            ("45% Alc./Vol.", 110, 20),
            ("750 mL", 140, 20),
        ])
        
        result = extractor.extract_all(ocr_result)
        
        # Individual field confidences should be high (from mock 0.95)
        assert result.brand_name.confidence >= 0.9
        assert result.abv_percent.confidence >= 0.8
    
    def test_partial_extraction_confidence(self, extractor):
        """Test confidence when some fields missing."""
        ocr_result = make_ocr_result([
            ("BRAND NAME", 0, 60),
            # Missing class, ABV, net contents
        ])
        
        result = extractor.extract_all(ocr_result)
        
        # Overall confidence is average of found fields only
        # Only brand was found, so confidence reflects just that
        assert result.brand_name.confidence >= 0.9
        # Unfound fields should have 0 confidence
        assert result.abv_percent.confidence == 0.0
        assert result.net_contents_ml.confidence == 0.0


class TestEdgeCases:
    """Test edge cases and unusual inputs."""
    
    def test_very_short_text(self, extractor):
        """Test handling of very short text."""
        ocr_result = make_ocr_result([
            ("AB", 0, 20),
            ("CD", 30, 20),
        ])
        
        result = extractor.extract_all(ocr_result)
        # Should handle gracefully, not crash
        assert result is not None
    
    def test_numbers_only(self, extractor):
        """Test text with only numbers."""
        ocr_result = make_ocr_result([
            ("45", 0, 30),
            ("750", 40, 30),
        ])
        
        result = extractor.extract_all(ocr_result)
        # Should not crash, may not find meaningful data
        assert result is not None
    
    def test_special_characters(self, extractor):
        """Test text with special characters."""
        ocr_result = make_ocr_result([
            ("MAKER'S MARK®", 0, 50),
            ("45% Alc./Vol.", 60, 20),
        ])
        
        result = extractor.extract_all(ocr_result)
        assert result.brand_name.value is not None
        assert "MAKER" in result.brand_name.value
