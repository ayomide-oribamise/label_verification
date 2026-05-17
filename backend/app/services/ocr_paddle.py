"""PaddleOCR adapter for the existing OCR pipeline."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class PaddleOCRBackend:
    """Small adapter that returns the shape the rest of the pipeline expects."""

    def __init__(self):
        from paddleocr import PaddleOCR

        # Model initialization is intentionally done once. In Docker, the model
        # cache is pre-populated during image build to avoid runtime downloads.
        self._ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            use_gpu=False,
            show_log=False,
            det_model_dir=None,
            rec_model_dir=None,
            cls_model_dir=None,
            # Lower threshold helps low-contrast metallic text such as foil.
            det_db_box_thresh=0.3,
            det_db_unclip_ratio=2.0,
        )

    def detect_once(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Run detection/recognition once and return normalized OCR detections."""
        result = self._ocr.ocr(image, cls=True)
        if not result:
            return []

        # PaddleOCR 2.x commonly returns [lines] for one image, while some
        # versions/configs return [[lines]]. Accept both shapes here.
        def is_line(item: Any) -> bool:
            return (
                isinstance(item, (list, tuple))
                and len(item) == 2
                and isinstance(item[1], (list, tuple))
                and len(item[1]) == 2
                and isinstance(item[1][0], str)
            )

        lines = result if is_line(result[0]) else result[0]
        if not lines:
            return []

        boxes: list[dict[str, Any]] = []
        for line in lines:
            bbox, text_result = line
            text, confidence = text_result
            if not text:
                continue

            boxes.append({
                "text": text,
                "confidence": float(confidence),
                "bbox": [[int(point[0]), int(point[1])] for point in bbox],
            })

        return boxes
