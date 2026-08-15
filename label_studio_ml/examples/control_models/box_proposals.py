"""Propose the rectangles the SKU model does not draw.

The SKU model is good at naming and bad at finding. The relabel delta says so
precisely: of the boxes it proposed, 889 of 890 SKUs came back unchanged, while
the labeller had to ADD 562 boxes it never drew. So the expensive part of
labelling is not correcting names, it is drawing rectangles from scratch.

The class-agnostic facing detector exists for that gap. It answers only "is
there a product facing here", which is the one question in this project the
data can currently support: 1,452 boxes for a single class, rather than a
median of 7 per class across 236.

Measured on the held-out clips (box_detector/eval_boxes.py), at threshold 0.50:

    recall 0.840, precision 0.800
    of 7.1 facings per image, ~6 arrive correct, 1.5 are strays, 1.1 are missed

Proposals are emitted WITHOUT a SKU, on purpose. A box with no taxonomy cannot
be submitted (the Taxonomy control is perRegion required), so every one of them
has to be named by a human before the task closes -- and in this fork, naming a
proposed region is what accepts it. The labeller reads and types instead of
reads, draws, and types.

A previous attempt at proposing extra boxes -- from shelf tags, see
_apply_tag_corrections -- mostly added false positives and was cut back to
correcting names only. This is not that: it is a detector trained on the boxes
themselves, with a measured operating point, rather than an inference from a
price tag's position. The difference is worth keeping in mind if the false
positive rate turns out to be worse in practice than on the test clips, which
came from the same store visit as its training data.

Off unless BOX_PROPOSALS_ENABLED, and a missing checkpoint disables it rather
than failing a prediction: the SKU model's own boxes are still worth having.
"""
import logging
import os
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

BOX_PROPOSALS_ENABLED = os.getenv("BOX_PROPOSALS_ENABLED", "false").lower() in ("1", "true")
BOX_PROPOSALS_CHECKPOINT = os.getenv("BOX_PROPOSALS_CHECKPOINT", "box_detector.pth")
# 0.50 is the cost-optimal operating point when a stray costs ~a third of a
# drawn box; see box_detector/eval_boxes.py in tj-labeling-ops for the sweep
# and why the answer barely moves across plausible cost ratios.
BOX_PROPOSALS_THRESHOLD = float(os.getenv("BOX_PROPOSALS_THRESHOLD", "0.5"))
# Overlap above which a proposal is considered "already covered" by a box the
# SKU model drew, and dropped. Same IoU the eval matches at, so the recall
# number above describes the same notion of "same box".
BOX_PROPOSALS_IOU = float(os.getenv("BOX_PROPOSALS_IOU", "0.5"))

_MISSING_LOGGED = False


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    iy = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def load_detector():
    """The facing detector, or None if it is not installed.

    Goes through ControlModel.get_cached_model so it shares the process-wide
    model cache and the checkpoint-introspecting loader -- this is a 1-class
    nano and the SKU model is a 236-class one, and the loader reads which is
    which from the weights rather than being told.
    """
    global _MISSING_LOGGED
    from control_models.base import MODEL_ROOT, ControlModel

    path = os.path.join(MODEL_ROOT, BOX_PROPOSALS_CHECKPOINT)
    if not os.path.exists(path):
        if not _MISSING_LOGGED:
            logger.warning(
                f"BOX_PROPOSALS_ENABLED is set but {path} is not there — no box "
                f"proposals will be added. Install it with deploy_box_detector.py."
            )
            _MISSING_LOGGED = True
        return None
    model, _names = ControlModel.get_cached_model(BOX_PROPOSALS_CHECKPOINT)
    return model


def propose(image, taken: List[Tuple[float, float, float, float]],
            threshold: Optional[float] = None,
            iou_thresh: Optional[float] = None) -> List[Tuple[float, float, float, float, float]]:
    """Facings the SKU model did not already box, as (x1, y1, x2, y2, score).

    `taken` is the SKU model's boxes in pixel xyxy. Anything overlapping one of
    them is dropped: the SKU model's version carries a name, so it is strictly
    the more useful of the two.
    """
    model = load_detector()
    if model is None:
        return []
    threshold = BOX_PROPOSALS_THRESHOLD if threshold is None else threshold
    iou_thresh = BOX_PROPOSALS_IOU if iou_thresh is None else iou_thresh

    detections = model.predict(image, threshold=threshold)
    out = []
    for i in range(len(detections.xyxy)):
        box = tuple(float(v) for v in detections.xyxy[i].tolist())
        if any(iou(box, t) >= iou_thresh for t in taken):
            continue
        # Later proposals are also checked against earlier ones. The detector
        # does its own NMS, but a box it kept can still overlap one we just
        # accepted once the SKU boxes are mixed in.
        if any(iou(box, (o[0], o[1], o[2], o[3])) >= iou_thresh for o in out):
            continue
        out.append(box + (float(detections.confidence[i]),))
    logger.debug(f"box proposals: {len(out)} added, {len(taken)} already covered")
    return out
