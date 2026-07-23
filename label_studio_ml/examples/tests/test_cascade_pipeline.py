import os
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cascade.pipeline import Decision, verify_detection  # noqa: E402


CROP = Image.new("RGB", (10, 10))
GALLERY = {"Apple": np.ones(4)}  # non-empty so nn_model path is exercised


def _verify(**overrides):
    kwargs = dict(
        crop=CROP,
        class_name="Apple",
        detector_confidence=0.8,
        effective_threshold=0.5,
        expected_text={},
        reference_gallery={},
        nn_model=None,
    )
    kwargs.update(overrides)
    return verify_detection(**kwargs)


def test_below_threshold_rejects_without_calling_any_signal():
    with patch("cascade.pipeline.ocr.ocr_agrees") as mock_ocr, \
         patch("cascade.pipeline.embedding_match.extract_embedding") as mock_emb, \
         patch("cascade.pipeline.gpt_tiebreaker.ask") as mock_gpt:
        decision = _verify(detector_confidence=0.3, effective_threshold=0.5)
    assert decision == Decision.AUTO_REJECT
    mock_ocr.assert_not_called()
    mock_emb.assert_not_called()
    mock_gpt.assert_not_called()


def test_no_signal_available_trusts_the_threshold_and_accepts():
    # no OCR expectation for this class, no reference gallery / nn_model -> no signals at all
    with patch("cascade.pipeline.ocr.ocr_agrees", return_value=None), \
         patch("cascade.pipeline.gpt_tiebreaker.ask") as mock_gpt:
        decision = _verify(reference_gallery={}, nn_model=None)
    assert decision == Decision.AUTO_ACCEPT
    mock_gpt.assert_not_called()


def test_all_signals_agree_accepts():
    with patch("cascade.pipeline.ocr.ocr_agrees", return_value=True), \
         patch("cascade.pipeline.embedding_match.extract_embedding", return_value=np.ones(4)), \
         patch("cascade.pipeline.embedding_match.embedding_agrees", return_value=True), \
         patch("cascade.pipeline.embedding_match.nearest_classes", return_value=[("Apple", 0.99)]), \
         patch("cascade.pipeline.gpt_tiebreaker.ask") as mock_gpt:
        decision = _verify(reference_gallery=GALLERY, nn_model=object())
    assert decision == Decision.AUTO_ACCEPT
    mock_gpt.assert_not_called()


def test_all_signals_disagree_rejects():
    with patch("cascade.pipeline.ocr.ocr_agrees", return_value=False), \
         patch("cascade.pipeline.embedding_match.extract_embedding", return_value=np.ones(4)), \
         patch("cascade.pipeline.embedding_match.embedding_agrees", return_value=False), \
         patch("cascade.pipeline.embedding_match.nearest_classes", return_value=[("Banana", 0.9)]), \
         patch("cascade.pipeline.gpt_tiebreaker.ask") as mock_gpt:
        decision = _verify(reference_gallery=GALLERY, nn_model=object())
    assert decision == Decision.AUTO_REJECT
    mock_gpt.assert_not_called()


def test_mixed_signals_calls_gpt_and_accepts_when_gpt_confirms_original_class():
    with patch("cascade.pipeline.ocr.ocr_agrees", return_value=True), \
         patch("cascade.pipeline.embedding_match.extract_embedding", return_value=np.ones(4)), \
         patch("cascade.pipeline.embedding_match.embedding_agrees", return_value=False), \
         patch("cascade.pipeline.embedding_match.nearest_classes", return_value=[("Banana", 0.9)]), \
         patch("cascade.pipeline.gpt_tiebreaker.ask", return_value="Apple") as mock_gpt:
        decision = _verify(reference_gallery=GALLERY, nn_model=object())
    assert decision == Decision.AUTO_ACCEPT
    mock_gpt.assert_called_once()


def test_mixed_signals_rejects_when_gpt_picks_a_different_candidate():
    with patch("cascade.pipeline.ocr.ocr_agrees", return_value=True), \
         patch("cascade.pipeline.embedding_match.extract_embedding", return_value=np.ones(4)), \
         patch("cascade.pipeline.embedding_match.embedding_agrees", return_value=False), \
         patch("cascade.pipeline.embedding_match.nearest_classes", return_value=[("Banana", 0.9)]), \
         patch("cascade.pipeline.gpt_tiebreaker.ask", return_value="Banana"):
        decision = _verify(reference_gallery=GALLERY, nn_model=object())
    assert decision == Decision.AUTO_REJECT


def test_mixed_signals_escalates_when_gpt_is_unsure():
    with patch("cascade.pipeline.ocr.ocr_agrees", return_value=True), \
         patch("cascade.pipeline.embedding_match.extract_embedding", return_value=np.ones(4)), \
         patch("cascade.pipeline.embedding_match.embedding_agrees", return_value=False), \
         patch("cascade.pipeline.embedding_match.nearest_classes", return_value=[("Banana", 0.9)]), \
         patch("cascade.pipeline.gpt_tiebreaker.ask", return_value=None):
        decision = _verify(reference_gallery=GALLERY, nn_model=object())
    assert decision == Decision.ESCALATE
