"""Shelf-tag proposer — the piece that catches products RF-DETR never boxes.

Grocery shelf price tags name the product in each slot. They're a strong,
independent classification signal, especially for items with no readable
on-package text. But the tag text is the store's abbreviated/marketing
wording ("VANILLA LATTE"), not the catalog class name ("La Colombe® Vanilla
Cold Brew Draft Latte"), so the tag->class mapping is *learned* from labeled
frames rather than fuzzy-matched to the taxonomy.

Findings that shaped this (see the exploration in the PR history):
  - Tesseract can't read these stylized, cluttered retail frames; a
    vision model (gpt-5-mini) reads the tags reliably.
  - One vision call per frame enumerates every tag with a position, which
    is what lets us pair each tag to the product box directly above it.

Pipeline:
  build_tag_class_map.py  (offline) : detect tags on labeled frames, pair
      each to the labeled box above it, aggregate -> tag_class_map.json
  detect_tags()           (runtime) : vision call -> [{name, x, y}]
  lookup_class()          (runtime) : fuzzy-match a read tag to the map
  propose_from_tags()     (runtime) : for tags whose column has no accepted
      detection, emit a box for the slot above the tag with the mapped class
"""
import base64
import io
import json
import logging
import os
from typing import Dict, List, Optional

from PIL import Image
from rapidfuzz import fuzz, process

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
FUZZY_MIN_SCORE = 82       # min rapidfuzz score to accept a tag->map match


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


def load_tag_class_map(path: str) -> Dict[str, str]:
    """Load {normalized_tag_text -> class} (majority class per tag) from the
    JSON produced by build_tag_class_map.py. Returns {} if absent.
    """
    if not os.path.exists(path):
        logger.info(f"No tag->class map at {path} — shelf-tag proposer disabled.")
        return {}
    with open(path) as f:
        raw = json.load(f)
    # raw is {tag: {class: count}} -> pick the majority class per tag
    return {tag: max(counts.items(), key=lambda kv: kv[1])[0] for tag, counts in raw.items()}


def lookup_class(tag_text: str, tag_class_map: Dict[str, str]) -> Optional[str]:
    """Map a freshly-read tag to a class via exact-then-fuzzy match on the
    learned keys. Returns None if nothing clears FUZZY_MIN_SCORE.
    """
    if not tag_class_map:
        return None
    key = normalize_tag(tag_text)
    if key in tag_class_map:
        return tag_class_map[key]
    match = process.extractOne(key, tag_class_map.keys(), scorer=fuzz.token_sort_ratio)
    if match and match[1] >= FUZZY_MIN_SCORE:
        return tag_class_map[match[0]]
    return None


def _slot_box(tag_x: float, tag_y: float, width: int, height: int):
    """Propose a pixel box for the product slot above a tag center."""
    x1 = max(0.0, tag_x - SLOT_HALF_WIDTH) * width
    x2 = min(1.0, tag_x + SLOT_HALF_WIDTH) * width
    y2 = max(0.0, tag_y - SLOT_BOTTOM_OFFSET) * height
    y1 = max(0.0, tag_y - SLOT_TOP_OFFSET) * height
    return [x1, y1, x2, y2]


def propose_from_tags(
    image: Image.Image,
    tag_class_map: Dict[str, str],
    covered_centers: List,
    cover_dist: float = 0.06,
) -> List[Dict]:
    """Emit {class_name, box, tag} proposals for tags whose slot isn't already
    covered by an accepted detection. `covered_centers` is a list of (x,y)
    normalized centers of boxes already kept (from RF-DETR / the cascade).
    """
    if not tag_class_map:
        return []
    width, height = image.size
    proposals = []
    for tag in detect_tags(image):
        cls = lookup_class(tag["name"], tag_class_map)
        if cls is None:
            continue
        tx, ty = float(tag["x"]), float(tag["y"])
        slot_cx, slot_cy = tx, ty - (SLOT_TOP_OFFSET + SLOT_BOTTOM_OFFSET) / 2
        # skip if an accepted detection already sits in this slot
        if any(abs(cx - slot_cx) < cover_dist and abs(cy - slot_cy) < SLOT_TOP_OFFSET
               for cx, cy in covered_centers):
            continue
        proposals.append({"class_name": cls, "box": _slot_box(tx, ty, width, height), "tag": tag["name"]})
    return proposals
