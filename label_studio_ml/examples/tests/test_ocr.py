import os
import sys
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cascade.ocr import ocr_agrees, ocr_match_score  # noqa: E402


CROP = Image.new("RGB", (10, 10))


def test_no_expected_text_entry_means_no_signal():
    assert ocr_match_score(CROP, "Apple", {}) is None


def test_class_with_no_expected_text_means_no_signal():
    # e.g. Produce/Dairy classes -> expected_text[class] is None by design
    assert ocr_match_score(CROP, "Apple", {"Apple": None}) is None


def test_empty_ocr_read_means_no_signal():
    with patch("cascade.ocr.extract_text", return_value=""):
        assert ocr_match_score(CROP, "San Pellegrino 6-pack", {"San Pellegrino 6-pack": "San Pellegrino 6-pack"}) is None


def test_matching_text_scores_high_and_agrees():
    expected = {"San Pellegrino 6-pack": "San Pellegrino 6-pack"}
    with patch("cascade.ocr.extract_text", return_value="SAN PELLEGRINO 6 PACK SPARKLING"):
        score = ocr_match_score(CROP, "San Pellegrino 6-pack", expected)
        assert score is not None and score > 0.6
        assert ocr_agrees(CROP, "San Pellegrino 6-pack", expected) is True


def test_unrelated_text_scores_low_and_disagrees():
    expected = {"San Pellegrino 6-pack": "San Pellegrino 6-pack"}
    with patch("cascade.ocr.extract_text", return_value="ORGANIC BANANAS NET WT 1LB"):
        assert ocr_agrees(CROP, "San Pellegrino 6-pack", expected) is False
