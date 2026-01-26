"""Field extraction service using OCR results."""

import re
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
import logging

from .ocr import OCRResult, OCRBox
from ..config import get_settings

logger = logging.getLogger(__name__)


# Standard government warning text (canonical form)
GOVERNMENT_WARNING_CANONICAL = (
    "GOVERNMENT WARNING: (1) ACCORDING TO THE SURGEON GENERAL, WOMEN SHOULD NOT "
    "DRINK ALCOHOLIC BEVERAGES DURING PREGNANCY BECAUSE OF THE RISK OF BIRTH DEFECTS. "
    "(2) CONSUMPTION OF ALCOHOLIC BEVERAGES IMPAIRS YOUR ABILITY TO DRIVE A CAR OR "
    "OPERATE MACHINERY, AND MAY CAUSE HEALTH PROBLEMS."
)

# Beverage type keywords for class/type detection
BEVERAGE_KEYWORDS = {
    "whiskey", "whisky", "bourbon", "scotch", "rye",
    "vodka", "gin", "rum", "tequila", "mezcal",
    "brandy", "cognac", "armagnac",
    "wine", "champagne", "prosecco", "cava",
    "beer", "ale", "lager", "stout", "porter", "ipa",
    "liqueur", "cordial", "schnapps",
    "sake", "soju", "baijiu",
}

# Marketing fluff to ignore when extracting brand
MARKETING_KEYWORDS = {
    "premium", "reserve", "select", "special", "limited",
    "aged", "barrel", "cask", "small batch", "single barrel",
    "handcrafted", "artisan", "craft", "estate", "vintage",
    "original", "classic", "traditional", "authentic",
}


@dataclass
class ExtractedField:
    """Result of extracting a single field."""
    value: Optional[str]
    confidence: float
    source_boxes: List[OCRBox] = field(default_factory=list)
    extraction_method: str = "unknown"
    notes: str = ""


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
        
        # Net contents patterns
        self.net_contents_patterns = [
            # Milliliters: "750 mL", "750ml", "750 ML"
            (r'(\d+(?:\.\d+)?)\s*(?:ml|mL|ML)', 1.0),
            # Centiliters: "75 cL", "75cl" -> multiply by 10
            (r'(\d+(?:\.\d+)?)\s*(?:cl|cL|CL)', 10.0),
            # Liters: "1 L", "1L", "1 LITER", "1 liter"
            (r'(\d+(?:\.\d+)?)\s*(?:l|L|liter|LITER|litre|LITRE)(?!\w)', 1000.0),
            # Fluid ounces (US): "25.4 FL OZ", "25.4 fl oz"
            (r'(\d+(?:\.\d+)?)\s*(?:fl\.?\s*oz\.?|FL\.?\s*OZ\.?)', 29.5735),
        ]
    
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
    
    def _extract_brand(self, ocr_result: OCRResult) -> ExtractedField:
        """
        Extract brand name using spatial rules.
        
        Strategy:
        1. Find top-most text blocks with largest height (dominant headline)
        2. Filter out marketing fluff
        3. Return the most prominent candidate
        """
        if not ocr_result.boxes:
            return ExtractedField(
                value=None,
                confidence=0.0,
                extraction_method="spatial",
                notes="No text detected"
            )
        
        # Sort boxes by Y position (top first), then by height (largest first)
        sorted_boxes = sorted(
            ocr_result.boxes,
            key=lambda b: (b.top, -b.height)
        )
        
        # Get top third of the image (where brand usually is)
        min_y = min(b.top for b in ocr_result.boxes)
        max_y = max(b.bottom for b in ocr_result.boxes)
        top_third_threshold = min_y + (max_y - min_y) // 3
        
        # Find candidates in top third with large height
        candidates = []
        for box in sorted_boxes:
            if box.top > top_third_threshold:
                break
            
            text_lower = box.text.lower()
            
            # Skip if it's just marketing fluff
            is_fluff = any(kw in text_lower for kw in MARKETING_KEYWORDS)
            
            # Skip if it contains beverage keywords (likely class/type)
            has_beverage_keyword = any(kw in text_lower for kw in BEVERAGE_KEYWORDS)
            
            # Skip very short text (likely not brand)
            if len(box.text.strip()) < 3:
                continue
            
            # Skip ABV/net contents patterns
            if re.search(r'\d+\s*%|\d+\s*(?:ml|cl|oz|proof)', text_lower):
                continue
                
            if not is_fluff and not has_beverage_keyword:
                candidates.append(box)
        
        if not candidates:
            # Fallback: take largest text in top half
            top_half_threshold = min_y + (max_y - min_y) // 2
            top_boxes = [b for b in sorted_boxes if b.top < top_half_threshold]
            if top_boxes:
                candidates = sorted(top_boxes, key=lambda b: b.height, reverse=True)[:1]
        
        if candidates:
            # Take the one with largest height (most prominent)
            best = max(candidates, key=lambda b: b.height)
            return ExtractedField(
                value=best.text.strip(),
                confidence=best.confidence,
                source_boxes=[best],
                extraction_method="spatial_top_prominent",
                notes="Top-most prominent text"
            )
        
        return ExtractedField(
            value=None,
            confidence=0.0,
            extraction_method="spatial",
            notes="No brand candidate found"
        )
    
    def _extract_class_type(
        self, 
        ocr_result: OCRResult, 
        brand_result: ExtractedField
    ) -> ExtractedField:
        """
        Extract class/type using keyword matching.
        
        Strategy:
        1. Find text containing beverage keywords
        2. Prefer text near (below) the brand
        3. Combine adjacent lines if needed
        """
        if not ocr_result.boxes:
            return ExtractedField(
                value=None,
                confidence=0.0,
                extraction_method="keyword",
                notes="No text detected"
            )
        
        # Find boxes containing beverage keywords
        keyword_boxes = []
        for box in ocr_result.boxes:
            text_lower = box.text.lower()
            matching_keywords = [kw for kw in BEVERAGE_KEYWORDS if kw in text_lower]
            if matching_keywords:
                keyword_boxes.append((box, matching_keywords))
        
        if not keyword_boxes:
            return ExtractedField(
                value=None,
                confidence=0.0,
                extraction_method="keyword",
                notes="No beverage keywords found"
            )
        
        # If we have a brand, prefer class/type below it
        brand_bottom = 0
        if brand_result.source_boxes:
            brand_bottom = brand_result.source_boxes[0].bottom
        
        # Score candidates: prefer those below brand with more keywords
        def score_candidate(item):
            box, keywords = item
            position_score = 1 if box.top >= brand_bottom else 0
            keyword_score = len(keywords)
            height_score = box.height / 100  # Normalize
            return position_score * 2 + keyword_score + height_score
        
        keyword_boxes.sort(key=score_candidate, reverse=True)
        best_box, keywords = keyword_boxes[0]
        
        return ExtractedField(
            value=best_box.text.strip(),
            confidence=best_box.confidence,
            source_boxes=[best_box],
            extraction_method="keyword_match",
            notes=f"Matched keywords: {', '.join(keywords)}"
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
        """
        raw_text = ocr_result.raw_text
        
        for pattern, multiplier in self.net_contents_patterns:
            match = re.search(pattern, raw_text, re.IGNORECASE)
            if match:
                value = float(match.group(1)) * multiplier
                value = round(value, 1)  # Round to 1 decimal
                
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
        text = text.replace('0', 'O')  # Context-dependent, but OK for warning
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
