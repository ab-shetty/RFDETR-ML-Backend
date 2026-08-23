"""Name every box in a frame with one vision call, by showing it the boxes.

The SKU model's class head learns names from labelled shelves, so it names a
store it has never seen about as well as it was ever going to: on the two
held-out visits it got 51% of the boxes it found right, and most of that came
from a small high-confidence band. Shelf tags name products independently of any
training data -- the store reprints them when the shelf changes -- but the
earlier attempt to use them read tags in isolation and then worked out which box
each named by geometry, and that pairing became the dominant error: dense
shelves, approximate coordinates, and most wrong names landing one slot over.

Drawing the numbered boxes on the frame removes the pairing step. The question
stops being "where is this tag" and becomes "what is in box 7", which the model
can answer from the tag under the box AND the packaging inside it.

Measured against human labels on both held-out stores (scripts/eval_box_naming.py,
ground-truth boxes, so this is naming in isolation):

    Coppell   87% of boxes named correctly   (geometry pairing: 45%)
    Laguna    73%                            (geometry pairing: 19%)

and against the class head on the same boxes (scripts/eval_naming_gate.py), the
head only catches up above ~0.6 confidence, which is a quarter of its boxes; of
183 disagreements the vision model was right in 160. Hence naming here and the
head kept only as a fallback where this returns nothing (HEAD_FLOOR).

Roughly a cent a frame at gpt-5.6-terra. Terra rather than Luna on purpose: the
tags are often handwritten, which is a capability-tier failure for nano-tier
models (see build_tag_index.py's comment for the same finding on the reader).
"""
import base64
import io
import json
import logging
import os
from typing import Dict, List, Optional

from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

BOX_NAMING_ENABLED = os.getenv("BOX_NAMING_ENABLED", "false").lower() in ["1", "true"]
MODEL = os.getenv("BOX_NAMING_MODEL", "gpt-5.6-terra")
# Below this, the class head's guess is not worth keeping even when this stage
# returns nothing: measured 20-34% correct under 0.4, against a labeller who
# then has to notice it is wrong. An unnamed box is honest work; a wrong name is
# work plus a trap.
HEAD_FLOOR = float(os.getenv("BOX_NAMING_HEAD_FLOOR", "0.8"))
# More boxes than this in one image and the numbers start colliding on dense
# shelves; chunking keeps each rendering legible.
MAX_BOXES_PER_CALL = 40
MAX_SIDE = 1600

PROMPT = (
    "This is a Trader Joe's shelf. Numbered boxes are drawn on it, one per product "
    "facing.\n\nFor EACH numbered box, identify the product and give the row number "
    "from the taxonomy below.\n\nHow to decide:\n"
    "- The price tag on the shelf rail DIRECTLY BELOW a box names that product. It is "
    "the store's abbreviated wording, so match on the product, not on string "
    "similarity: 'VANILLA LATTE' can be 'La Colombe Vanilla Cold Brew Draft Latte'.\n"
    "- Use the packaging in the box itself to confirm, and to choose when the tag is "
    "unreadable or ambiguous.\n"
    "- Answer null if no row is that product, or if you cannot tell. A wrong row "
    "becomes a mislabelled training example; a null is just a box a human still "
    "names.\n"
    "- Flavour and format matter: lemon is not lime, a 4-pack row is not the "
    "single-bottle row.\n\n"
    "TAXONOMY:\n{taxonomy}\n\n"
    'JSON only: {{"boxes":[{{"box":<number>,"row":<row number or null>}}]}}')

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()
    return _client


def draw_boxes(image: Image.Image, boxes: List, numbers: Optional[List[int]] = None) -> Image.Image:
    """A copy of the frame with each box outlined and numbered (pixel xyxy)."""
    img = image.copy().convert("RGB")
    d = ImageDraw.Draw(img)
    H = img.height
    size = max(16, int(H * 0.018))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        d.rectangle((x1, y1, x2, y2), outline=(255, 0, 0), width=max(2, int(H * 0.002)))
        label = str(numbers[i] if numbers else i)
        tw = d.textlength(label, font=font)
        # The number sits on a filled chip: a bare digit over busy packaging is
        # exactly the case the model misreads, and a misread number silently
        # swaps two products' names.
        d.rectangle((x1, y1, x1 + tw + size * 0.6, y1 + size * 1.3), fill=(255, 0, 0))
        d.text((x1 + size * 0.3, y1 + size * 0.1), label, fill=(255, 255, 255), font=font)
    return img


def encode(image: Image.Image, max_side: int = MAX_SIDE) -> str:
    if max(image.size) > max_side:
        s = max_side / max(image.size)
        image = image.resize((int(image.width * s), int(image.height * s)))
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def name_boxes(image: Image.Image, boxes: List, class_names: List[str],
               model: str = MODEL) -> Dict[int, Optional[str]]:
    """{box index -> class name or None} for pixel-xyxy boxes.

    Never raises: a failed call means this stage contributes nothing and the
    caller falls back, rather than a pre-annotation run dying halfway.
    """
    if not boxes or not class_names:
        return {}
    taxonomy = "\n".join(f"{i}: {n}" for i, n in enumerate(class_names))
    out: Dict[int, Optional[str]] = {}
    for start in range(0, len(boxes), MAX_BOXES_PER_CALL):
        chunk = list(range(start, min(start + MAX_BOXES_PER_CALL, len(boxes))))
        # Only this chunk's boxes are drawn -- a rendering carrying boxes we are
        # not asking about is just clutter over the shelf.
        img = draw_boxes(image, [boxes[i] for i in chunk], numbers=list(range(len(chunk))))
        try:
            r = _get_client().chat.completions.create(
                model=model, max_completion_tokens=8000,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": PROMPT.format(taxonomy=taxonomy)},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{encode(img)}"}}]}])
            raw = (r.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].removeprefix("json").strip()
            answer = json.loads(raw).get("boxes", [])
        except Exception as e:
            logger.warning(f"box naming failed for {len(chunk)} boxes: {e}")
            continue
        for item in answer:
            if not isinstance(item, dict):
                continue
            local, row = item.get("box"), item.get("row")
            if not isinstance(local, int) or not 0 <= local < len(chunk):
                continue
            out[chunk[local]] = (class_names[row]
                                 if isinstance(row, int) and 0 <= row < len(class_names)
                                 else None)
    return out
