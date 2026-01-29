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

# Brand suffix keywords - brands often end with these
BRAND_SUFFIX_PATTERN = r"(DISTILLERY|DISTILLING|BREWING|BREWERY|WINERY|VINEYARDS?|COMPANY|CO\.?|INC\.?|LLC|CELLARS?|SPIRITS?)"

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

# Stop words to ignore in token matching
BRAND_STOP_WORDS = {"THE", "AND", "OF", "BY"}


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
        
        # Net contents patterns (order matters - more specific first)
        # Handle OCR artifacts like colons, periods, extra spaces
        self.net_contents_patterns = [
            # Milliliters: "750 mL", "750ml", "750 ML"
            (r'(\d+(?:\.\d+)?)\s*(?:ml|mL|ML)', 1.0),
            # Centiliters: "75 cL", "75cl" -> multiply by 10
            (r'(\d+(?:\.\d+)?)\s*(?:cl|cL|CL)', 10.0),
            # Liters: "1 L", "1L", "1 LITER", "1 liter"
            (r'(\d+(?:\.\d+)?)\s*(?:l|L|liter|LITER|litre|LITRE)(?!\w)', 1000.0),
            # Fluid ounces (US): "25.4 FL OZ", "25.4 fl oz", "12 FL: Oz." (OCR artifacts)
            # More lenient pattern to handle OCR noise (colons, periods, spaces)
            (r'(\d+(?:\.\d+)?)\s*(?:fl|FL)[\s.:;]*(?:oz|OZ|Oz)\.?', 29.5735),
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
                return ExtractedField(
                    value=brand,
                    confidence=0.75,
                    source_boxes=src_boxes,
                    extraction_method="spatial",
                    notes="Brand from spatial analysis"
                )
        
        # Fallback: First meaningful tokens (filter stop/fluff)
        brand, src_boxes = self._extract_brand_fallback(raw_text, boxes)
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
        
        Validates result to avoid false positives like "BOURBON SPIRITS".
        """
        # Normalize text
        text = re.sub(r"\s+", " ", raw_text.upper()).strip()
        
        # Pattern: 1-5 words followed by producer keyword
        # Matches: "OLD TOM DISTILLERY", "JACK DANIEL'S DISTILLERY", etc.
        pattern = rf"\b([A-Z0-9'&\-]{{2,}}(?:\s+[A-Z0-9'&\-]{{2,}}){{0,4}}\s+{BRAND_SUFFIX_PATTERN})\b"
        
        # Find all matches and pick the best one
        for match in re.finditer(pattern, text):
            brand = match.group(1).strip()
            
            # Validate: must be plausible brand (not beverage type or fluff)
            if self._is_plausible_brand(brand):
                logger.debug(f"Found brand with suffix: {brand}")
                return brand
            else:
                logger.debug(f"Rejected implausible brand: {brand}")
        
        return None
    
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
