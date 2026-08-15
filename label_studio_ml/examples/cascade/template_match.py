"""Name a crop by keypoint-matching it against labelled crops of known SKUs.

Reaches the boxes nothing else can name: the ones the SKU detector never drew,
which by construction are the ones it did not recognise.

**The bar matters more than the match.** This stage is never right about a SKU
absent from the bank -- there is nothing to match it against, so any name it
emits there is wrong by construction. Measured on the held-out clips, where 166
of 200 facings were SKUs the bank had never seen:

    min matches   in bank: named/right/wrong   absent: named/right/wrong
        12              25 / 25 / 0                 21 / 0 / 21
        18              23 / 23 / 0                  2 / 0 /  2
        25              19 / 19 / 0                  0 / 0 /  0

18 is the default because 12 -> 18 removes 19 wrong names for 2 correct ones,
while 25 removes the last 2 wrong for 4 more correct. Re-check it with
pipeline_dryrun.py when the bank grows; the sweep prints on every run.

A missing bank disables the stage rather than raising: the boxes are still
worth having unnamed.
"""
import logging
import os
import threading

logger = logging.getLogger(__name__)

TEMPLATE_MATCHING_ENABLED = os.getenv("TEMPLATE_MATCHING_ENABLED", "false").lower() in ("1", "true")
TEMPLATE_BANK = os.getenv("TEMPLATE_BANK", "template_bank.npz")
TEMPLATE_MIN_MATCHES = int(os.getenv("TEMPLATE_MIN_MATCHES", "18"))
# Lowe's ratio. Lower is stricter; packaging is full of repeated texture that
# matches everything equally well without it.
TEMPLATE_RATIO = float(os.getenv("TEMPLATE_RATIO", "0.75"))
ORB_FEATURES = int(os.getenv("TEMPLATE_ORB_FEATURES", "200"))
MIN_CROP_PX = 16
MIN_DESCRIPTORS = 8

_bank = None
_bank_lock = threading.Lock()
_missing_logged = False


def _load_bank(model_root):
    """(names, [descriptor matrices]) or None. Loaded once per process."""
    global _bank, _missing_logged
    if _bank is not None:
        return _bank
    with _bank_lock:
        if _bank is not None:
            return _bank
        import numpy as np

        path = os.path.join(model_root, TEMPLATE_BANK)
        if not os.path.exists(path):
            if not _missing_logged:
                logger.warning(
                    f"TEMPLATE_MATCHING_ENABLED is set but {path} is missing — no "
                    f"template naming. Build it with scripts/build_template_bank.py.")
                _missing_logged = True
            return None
        z = np.load(path)
        names = [str(n) for n in z["names"]]
        offsets, descriptors = z["offsets"], z["descriptors"]
        blocks = [descriptors[offsets[i]:offsets[i + 1]] for i in range(len(names))]
        _bank = (names, blocks)
        logger.info(f"Loaded {len(names)} templates over {len(set(names))} SKUs from {path}")
        return _bank


def name_crop(crop, model_root, min_matches=None):
    """(sku_name, n_matches), or (None, 0) if nothing clears the bar."""
    bank = _load_bank(model_root)
    if bank is None:
        return None, 0
    if crop.width < MIN_CROP_PX or crop.height < MIN_CROP_PX:
        return None, 0

    import cv2
    import numpy as np

    names, blocks = bank
    min_matches = TEMPLATE_MIN_MATCHES if min_matches is None else min_matches
    orb = cv2.ORB_create(nfeatures=ORB_FEATURES)
    gray = cv2.cvtColor(np.array(crop.convert("RGB")), cv2.COLOR_RGB2GRAY)
    _kp, des = orb.detectAndCompute(gray, None)
    if des is None or len(des) < MIN_DESCRIPTORS:
        return None, 0

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    best_name, best_n = None, 0
    for name, tdes in zip(names, blocks):
        pairs = matcher.knnMatch(des, tdes, k=2)
        good = sum(1 for m in pairs
                   if len(m) == 2 and m[0].distance < TEMPLATE_RATIO * m[1].distance)
        if good > best_n:
            best_name, best_n = name, good
    if best_n < min_matches:
        return None, best_n
    return best_name, best_n
