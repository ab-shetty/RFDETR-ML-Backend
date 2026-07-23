"""Combines the cascade's verification signals into a single accept/reject/
escalate decision per detection, run before a detection becomes a Label
Studio pre-annotation.

Policy (deliberately simple and auditable — see get_decision docstring):
  - Below the per-class threshold -> reject immediately, no need to spend
    OCR/embedding/GPT work on it.
  - No extra signal available for this class at all (no reference-gallery
    entry, no OCR expectation — e.g. most produce/dairy classes) -> trust
    the calibrated per-class threshold alone.
  - Every available signal agrees with the predicted class -> accept.
  - Every available signal disagrees -> reject.
  - Signals disagree with each other -> genuinely ambiguous, worth the
    GPT-5-mini call. GPT resolving to the original class -> accept;
    resolving to a different candidate -> reject (the box is probably
    mislabeled, not just uncertain — relabeling it automatically is a
    reasonable future improvement, not attempted here); GPT unsure -> the
    only path that reaches human review in Label Studio.
"""
import logging
from enum import Enum
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from cascade import embedding_match, gpt_tiebreaker, ocr

logger = logging.getLogger(__name__)

GPT_CANDIDATE_SHORTLIST_SIZE = 3


class Decision(str, Enum):
    AUTO_ACCEPT = "auto_accept"
    AUTO_REJECT = "auto_reject"
    ESCALATE = "escalate"


def verify_detection(
    crop: Image.Image,
    class_name: str,
    detector_confidence: float,
    effective_threshold: float,
    expected_text: Dict[str, Optional[str]],
    reference_gallery: Dict[str, np.ndarray],
    nn_model=None,
) -> Decision:
    if detector_confidence < effective_threshold:
        return Decision.AUTO_REJECT

    ocr_agree = ocr.ocr_agrees(crop, class_name, expected_text)

    emb_agrees = None
    neighbors: List = []
    if nn_model is not None and reference_gallery:
        embedding = embedding_match.extract_embedding(nn_model, crop)
        emb_agrees = embedding_match.embedding_agrees(embedding, reference_gallery, class_name)
        if emb_agrees is not None:
            neighbors = embedding_match.nearest_classes(embedding, reference_gallery, k=GPT_CANDIDATE_SHORTLIST_SIZE)

    signals = [s for s in (emb_agrees, ocr_agree) if s is not None]

    if not signals:
        return Decision.AUTO_ACCEPT
    if all(signals):
        return Decision.AUTO_ACCEPT
    if not any(signals):
        return Decision.AUTO_REJECT

    # Signals disagree with each other -> worth the GPT-5-mini call.
    candidates = [class_name] + [n for n, _ in neighbors if n != class_name]
    candidates = candidates[:GPT_CANDIDATE_SHORTLIST_SIZE] or [class_name]
    gpt_choice = gpt_tiebreaker.ask(crop, candidates)
    if gpt_choice == class_name:
        return Decision.AUTO_ACCEPT
    if gpt_choice is not None:
        return Decision.AUTO_REJECT
    return Decision.ESCALATE
