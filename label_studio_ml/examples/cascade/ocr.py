"""OCR verification signal for the pre-annotation cascade.

Reads text off a detection crop and fuzzy-matches it against the class's
expected label text (see scripts/build_ocr_expected_text.py). Strong signal
for the ~135 branded/text-bearing classes; deliberately no signal at all for
generic produce/dairy classes where there's nothing distinctive to read.
"""
import json
import logging
import os
from typing import Dict, Optional

import pytesseract
from PIL import Image
from rapidfuzz import fuzz

logger = logging.getLogger(__name__)

# Below this fuzzy-match ratio (0-100 from rapidfuzz), we don't trust the OCR
# read as a match — packaging photos are often blurry/angled/partially
# occluded, so this is deliberately lenient rather than requiring near-exact text.
MATCH_THRESHOLD = 60


def load_expected_text(path: str) -> Dict[str, Optional[str]]:
    """Load {class_name: expected_text_or_None} built by
    scripts/build_ocr_expected_text.py. Returns {} if missing — callers
    should treat that as "no OCR signal available for any class".
    """
    if not os.path.exists(path):
        logger.info(f"No OCR expected-text map found at {path} — OCR signal disabled.")
        return {}
    with open(path) as f:
        return json.load(f)


def extract_text(crop: Image.Image) -> str:
    """Raw OCR text from a crop, whitespace-collapsed. Empty string on any
    failure (corrupt image data, tesseract not installed, etc.) rather than
    raising — this is a best-effort signal, not a required one.
    """
    try:
        text = pytesseract.image_to_string(crop)
        return " ".join(text.split())
    except Exception as e:
        logger.warning(f"OCR extraction failed: {e}")
        return ""


def ocr_match_score(crop: Image.Image, class_name: str, expected_text: Dict[str, Optional[str]]) -> Optional[float]:
    """Fuzzy match ratio (0.0-1.0) between OCR'd crop text and the class's
    expected text. Returns None (no signal, not a rejection) when:
      - the class isn't in expected_text at all
      - the class has no expected text (produce/dairy)
      - OCR extracted no text at all (crop too small/blurry/no text present)
    """
    if class_name not in expected_text or not expected_text[class_name]:
        return None
    text = extract_text(crop)
    if not text:
        return None
    return fuzz.partial_ratio(text.lower(), expected_text[class_name].lower()) / 100.0


def ocr_agrees(crop: Image.Image, class_name: str, expected_text: Dict[str, Optional[str]]) -> Optional[bool]:
    """None (no signal) or bool (does OCR support this class)."""
    score = ocr_match_score(crop, class_name, expected_text)
    if score is None:
        return None
    return score * 100 >= MATCH_THRESHOLD
