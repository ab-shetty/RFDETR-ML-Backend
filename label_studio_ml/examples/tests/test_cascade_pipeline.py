import os
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cascade.pipeline import Decision, verify_detection  # noqa: E402


CROP = Image.new("RGB", (10, 10))
GALLERY = {"Apple": np.ones(4)}


def _verify(confidence, threshold=0.5, in_gallery=True, has_nn=True, floor=0.15):
    return verify_detection(
        crop=CROP,
        class_name="Apple",
        detector_confidence=confidence,
        effective_threshold=threshold,
        expected_text={},
        reference_gallery=GALLERY if in_gallery else {},
        nn_model=object() if has_nn else None,
        cascade_floor=floor,
    )


def test_below_floor_rejects_without_any_signal_work():
    with patch("cascade.pipeline.ocr.ocr_agrees") as mock_ocr, \
         patch("cascade.pipeline.embedding_match.extract_embedding") as mock_emb, \
         patch("cascade.pipeline.gpt_tiebreaker.ask") as mock_gpt:
        assert _verify(confidence=0.1, floor=0.15) == Decision.AUTO_REJECT
    mock_ocr.assert_not_called()
    mock_emb.assert_not_called()
    mock_gpt.assert_not_called()


def test_no_signal_confident_accepts_no_worse_than_threshold_only():
    with patch("cascade.pipeline.ocr.ocr_agrees", return_value=None):
        # class not in gallery -> no embedding, no ocr -> confident -> accept
        assert _verify(confidence=0.8, threshold=0.5, in_gallery=False) == Decision.AUTO_ACCEPT


def test_no_signal_uncertain_drops():
    with patch("cascade.pipeline.ocr.ocr_agrees", return_value=None):
        # uncertain (below threshold) + no verifiable signal -> drop
        assert _verify(confidence=0.3, threshold=0.5, in_gallery=False) == Decision.AUTO_REJECT


def test_embedding_not_computed_when_class_absent_from_gallery():
    with patch("cascade.pipeline.ocr.ocr_agrees", return_value=None), \
         patch("cascade.pipeline.embedding_match.extract_embedding") as mock_emb:
        _verify(confidence=0.8, in_gallery=False)
    mock_emb.assert_not_called()


def test_uncertain_box_promoted_when_signals_confirm():
    # THE recall-recovery case: below threshold, but embedding+OCR both confirm.
    with patch("cascade.pipeline.ocr.ocr_agrees", return_value=True), \
         patch("cascade.pipeline.embedding_match.extract_embedding", return_value=np.ones(4)), \
         patch("cascade.pipeline.embedding_match.embedding_agrees", return_value=True), \
         patch("cascade.pipeline.embedding_match.nearest_classes", return_value=[("Apple", 0.9)]), \
         patch("cascade.pipeline.gpt_tiebreaker.ask") as mock_gpt:
        assert _verify(confidence=0.3, threshold=0.5) == Decision.AUTO_ACCEPT
    mock_gpt.assert_not_called()


def test_confident_box_rejected_when_all_signals_refute():
    # precision case: high confidence but embedding+OCR both say wrong class.
    with patch("cascade.pipeline.ocr.ocr_agrees", return_value=False), \
         patch("cascade.pipeline.embedding_match.extract_embedding", return_value=np.ones(4)), \
         patch("cascade.pipeline.embedding_match.embedding_agrees", return_value=False), \
         patch("cascade.pipeline.embedding_match.nearest_classes", return_value=[("Banana", 0.9)]), \
         patch("cascade.pipeline.gpt_tiebreaker.ask") as mock_gpt:
        assert _verify(confidence=0.8, threshold=0.5) == Decision.AUTO_REJECT
    mock_gpt.assert_not_called()


def test_mixed_signals_call_gpt_accept_when_gpt_confirms():
    with patch("cascade.pipeline.ocr.ocr_agrees", return_value=True), \
         patch("cascade.pipeline.embedding_match.extract_embedding", return_value=np.ones(4)), \
         patch("cascade.pipeline.embedding_match.embedding_agrees", return_value=False), \
         patch("cascade.pipeline.embedding_match.nearest_classes", return_value=[("Banana", 0.9)]), \
         patch("cascade.pipeline.gpt_tiebreaker.ask", return_value="Apple") as mock_gpt:
        assert _verify(confidence=0.3, threshold=0.5) == Decision.AUTO_ACCEPT
    mock_gpt.assert_called_once()


def test_mixed_signals_reject_when_gpt_picks_other():
    with patch("cascade.pipeline.ocr.ocr_agrees", return_value=True), \
         patch("cascade.pipeline.embedding_match.extract_embedding", return_value=np.ones(4)), \
         patch("cascade.pipeline.embedding_match.embedding_agrees", return_value=False), \
         patch("cascade.pipeline.embedding_match.nearest_classes", return_value=[("Banana", 0.9)]), \
         patch("cascade.pipeline.gpt_tiebreaker.ask", return_value="Banana"):
        assert _verify(confidence=0.8, threshold=0.5) == Decision.AUTO_REJECT


def test_mixed_signals_escalate_when_gpt_unsure():
    with patch("cascade.pipeline.ocr.ocr_agrees", return_value=True), \
         patch("cascade.pipeline.embedding_match.extract_embedding", return_value=np.ones(4)), \
         patch("cascade.pipeline.embedding_match.embedding_agrees", return_value=False), \
         patch("cascade.pipeline.embedding_match.nearest_classes", return_value=[("Banana", 0.9)]), \
         patch("cascade.pipeline.gpt_tiebreaker.ask", return_value=None):
        assert _verify(confidence=0.8, threshold=0.5) == Decision.ESCALATE
