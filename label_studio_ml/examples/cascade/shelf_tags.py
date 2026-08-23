"""Reads the price tags off a shelf: one vision call, every tag with a position.

Grocery shelf tags name the product in each slot, independently of anything we
have labelled, which makes them the one naming signal that still works in a
store the detector has never seen. Tesseract cannot read these stylized,
cluttered retail frames; a vision model can.

This module is now just the reader. What used to sit on top of it -- mapping a
read tag to a class through `tag_class_map.json` and correcting SKUs with it --
was removed after being measured against the human labels for two held-out store
visits (`scripts/eval_tag_naming.py`): it resolved half the tags and got three
quarters of those wrong, because the map was built by pairing tags to labelled
boxes on ONE visit and that pairing's noise became majority votes. Worse, it
could only ever name SKUs somebody had already labelled -- the opposite of what
the signal is for.

`cascade/box_naming.py` replaces it, and does not pair tags to boxes at all: it
draws the boxes on the frame and asks what is in each, which is the question the
pairing was a proxy for.

Still used here:
  detect_tags()   vision call -> [{name, x, y}], x/y normalized centers
  normalize_tag() canonical form for tag text (strips prices, uppercases)
  _slot_box()     the slot above a tag, for scripts/bootstrap_zero_data_classes.py
"""
import base64
import io
import json
import logging
import os
from typing import Dict, List

from PIL import Image

logger = logging.getLogger(__name__)

# Overridable so the offline harness (tj-labeling-ops/pipeline_dryrun.py) can
# price and score a different model without editing this file. The default
# stays on the incumbent until a dry run says otherwise.
MODEL = os.getenv("SHELF_TAG_MODEL", "gpt-5.6-luna")
# The gpt-5.6 family renamed the cheapest reasoning setting: "minimal" is
# rejected with a 400 and "none" means what it used to. Picked from the model
# id so switching models does not silently start paying for reasoning, or
# fail every call -- which is exactly what the first Luna dry run did.
REASONING_EFFORT = os.getenv(
    "SHELF_TAG_REASONING_EFFORT",
    "none" if MODEL.startswith("gpt-5.6") else "minimal")
_client = None

# Product slot sits directly above its tag. Measured on labeled frames, the
# product's center is ~0.045 of image height above its tag center, so a slot
# spanning [tag_y - 0.09, tag_y] centers on the product (center ≈ tag_y-0.045)
# and is tall enough to cover height variation. A pre-annotation box the human
# nudges is far cheaper than finding the right SKU from scratch (the tag gives
# that for free).
SLOT_TOP_OFFSET = 0.09     # how far above the tag the slot reaches
SLOT_BOTTOM_OFFSET = 0.0   # slot bottom sits at the tag line
SLOT_HALF_WIDTH = 0.06


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()
    return _client


def _encode(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def detect_tags(image: Image.Image) -> List[Dict]:
    """One vision call -> every shelf tag as {name, x, y} (x,y are 0-1 centers).
    Returns [] on any failure (network/parse) — the proposer just contributes
    nothing in that case, never raises.
    """
    prompt = (
        "This is a grocery cooler with several shelves. Each shelf has light-blue "
        "price tags on its front rail. List EVERY price tag you can see. For each, give "
        "the product NAME text on it (skip the price and fine print) and its approximate "
        "center as fractions of image width/height (0-1). "
        'JSON only: {"tags":[{"name":"...","x":0.0,"y":0.0}]}'
    )
    try:
        r = _get_client().chat.completions.create(
            model=MODEL, max_completion_tokens=1500, reasoning_effort=REASONING_EFFORT,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode(image)}"}},
            ]}],
        )
        raw = (r.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].removeprefix("json").strip()
        tags = json.loads(raw).get("tags", [])
        return [t for t in tags if t.get("name") and "x" in t and "y" in t]
    except Exception as e:
        logger.warning(f"shelf-tag detection failed: {e}")
        return []


def normalize_tag(text: str) -> str:
    """Canonical form for map keys/lookups: upper, no price, collapsed spaces."""
    import re
    text = re.sub(r"\$?\d+[.\-]\d{2}", " ", text)   # strip prices
    text = re.sub(r"[^A-Za-z0-9+& ]", " ", text)
    return " ".join(text.upper().split())


def _slot_box(tag_x: float, tag_y: float, width: int, height: int):
    """Propose a pixel box for the product slot above a tag center."""
    x1 = max(0.0, tag_x - SLOT_HALF_WIDTH) * width
    x2 = min(1.0, tag_x + SLOT_HALF_WIDTH) * width
    y2 = max(0.0, tag_y - SLOT_BOTTOM_OFFSET) * height
    y1 = max(0.0, tag_y - SLOT_TOP_OFFSET) * height
    return [x1, y1, x2, y2]
