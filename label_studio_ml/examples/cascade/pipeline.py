"""Combines the cascade's verification signals into a single accept/reject/
escalate decision per detection, run before a detection becomes a Label
Studio pre-annotation.

Two confidence tiers (this is what lets the cascade improve BOTH precision
and recall, instead of only filtering):

  - Confident tier  (detector_confidence >= per-class threshold): RF-DETR is
    already sure. Keep it unless the verification signals actively refute it
    — that removes confident-but-wrong-class false positives (precision).

  - Uncertain tier  (CASCADE_FLOOR <= confidence < per-class threshold):
    RF-DETR sees something but isn't sure — a box the threshold-only system
    would silently drop. Keep it ONLY if a verification signal positively
    confirms it. That recovers real objects RF-DETR under-detected (recall),
    without letting through unverifiable junk.

Cost control: OCR only runs for classes with expected label text; the
embedding forward pass only runs for classes present in the reference
gallery; GPT-5-mini fires only on genuine cheap-signal disagreement. A
detection with no available signal costs almost nothing and falls back to
"trust the threshold" (accept if confident, drop if uncertain) — i.e. the
cascade never makes things worse than threshold-only for those classes.
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


def _gpt_decide(crop, class_name, neighbors):
    candidates = [class_name] + [n for n, _ in neighbors if n != class_name]
    candidates = candidates[:GPT_CANDIDATE_SHORTLIST_SIZE] or [class_name]
    choice = gpt_tiebreaker.ask(crop, candidates)
    if choice == class_name:
        return Decision.AUTO_ACCEPT
    if choice is None:
        return Decision.ESCALATE
    return Decision.AUTO_REJECT


def verify_detection(
    crop: Image.Image,
    class_name: str,
    detector_confidence: float,
    effective_threshold: float,
    expected_text: Dict[str, Optional[str]],
    reference_gallery: Dict[str, np.ndarray],
    nn_model=None,
    cascade_floor: float = 0.15,
) -> Decision:
    if detector_confidence < cascade_floor:
        return Decision.AUTO_REJECT

    confident = detector_confidence >= effective_threshold

    # OCR is already cheap-skipping: returns None without running tesseract
    # for classes that have no expected text.
    ocr_agree = ocr.ocr_agrees(crop, class_name, expected_text)

    # Only pay for the backbone forward pass if this class is in the gallery.
    emb_agree = None
    neighbors: List = []
    if nn_model is not None and class_name in reference_gallery:
        embedding = embedding_match.extract_embedding(nn_model, crop)
        emb_agree = embedding_match.embedding_agrees(embedding, reference_gallery, class_name)
        neighbors = embedding_match.nearest_classes(embedding, reference_gallery, k=GPT_CANDIDATE_SHORTLIST_SIZE)

    signals = [s for s in (emb_agree, ocr_agree) if s is not None]

    if not signals:
        # No way to verify this class -> don't do worse than threshold-only.
        return Decision.AUTO_ACCEPT if confident else Decision.AUTO_REJECT

    if all(signals):
        return Decision.AUTO_ACCEPT           # everything confirms (promotes uncertain boxes)
    if not any(signals):
        # everything refutes: reject in either tier (kills confident false positives too)
        return Decision.AUTO_REJECT

    # Cheap signals disagree with each other -> worth a GPT-5-mini call.
    return _gpt_decide(crop, class_name, neighbors)
