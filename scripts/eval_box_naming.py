#!/usr/bin/env python3
"""Name a box by showing it to the model, instead of pairing tags by geometry.

`eval_tag_naming.py` reads tags, then works out which product each tag names by
where it sits. That geometry is now the dominant error: most of its wrong names
are classes that ARE in the frame, one slot over. Shelves are dense and the
model's tag coordinates are approximate, so no threshold on "how far above" gets
this right for every facing.

But the pairing only exists because we asked for tags in isolation. Draw the
boxes on the frame and the model can see which tag belongs to which box -- the
question stops being "where is this tag" and becomes "what is in box 7", which
is the question we actually want answered, and it can use the packaging as well
as the tag to answer it.

Boxes here come from ground truth, so this measures naming in isolation. In
production they come from the class-agnostic box detector, which localizes well
across stores -- that is the whole point of splitting localization from naming.

    python3 scripts/eval_box_naming.py --dataset .../rf-detr-combined --split test --limit 20
"""
import argparse
import base64
import io
import json
import os
import sys
from collections import Counter

from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_tag_naming import load_split  # noqa: E402

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


def draw_boxes(image, boxes):
    """A copy of the frame with each box outlined and numbered."""
    img = image.copy().convert("RGB")
    d = ImageDraw.Draw(img)
    W, H = img.size
    size = max(16, int(H * 0.018))
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    for i, (_, x1, y1, x2, y2) in enumerate(boxes):
        px = (x1 * W, y1 * H, x2 * W, y2 * H)
        d.rectangle(px, outline=(255, 0, 0), width=max(2, int(H * 0.002)))
        label = str(i)
        tw = d.textlength(label, font=font)
        # Number goes at the box's top-left, on a filled chip: an unbacked digit
        # over busy packaging is exactly the case the model misreads.
        d.rectangle((px[0], px[1], px[0] + tw + size * 0.6, px[1] + size * 1.3),
                    fill=(255, 0, 0))
        d.text((px[0] + size * 0.3, px[1] + size * 0.1), label, fill=(255, 255, 255), font=font)
    return img


def encode(image, max_side=1600):
    if max(image.size) > max_side:
        s = max_side / max(image.size)
        image = image.resize((int(image.width * s), int(image.height * s)))
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=92)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default=os.getenv("BOX_NAMING_MODEL", "gpt-5.6-terra"))
    ap.add_argument("--limit", type=int, help="frames (they are paid calls)")
    ap.add_argument("--cache", default=os.path.join(HERE, "..", "data", "box_naming.json"))
    ap.add_argument("--misses", type=int, default=8)
    ap.add_argument("--dump-annotated", help="write the first annotated frame here, to check "
                                             "the boxes are legible before paying for a run")
    args = ap.parse_args()

    files, gt, class_names = load_split(args.dataset, args.split)
    files = [f for f in sorted(files) if gt[f]][:args.limit]
    taxonomy = "\n".join(f"{i}: {n}" for i, n in enumerate(class_names))

    if args.dump_annotated:
        f = files[0]
        draw_boxes(Image.open(os.path.join(args.dataset, args.split, f)), gt[f]).save(
            args.dump_annotated)
        print(f"wrote {args.dump_annotated} ({len(gt[f])} boxes)")
        return

    cache_path = os.path.abspath(args.cache)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    from openai import OpenAI
    client = OpenAI()
    stats = Counter()
    misses = []
    for n, fn in enumerate(files, 1):
        key = f"{args.model}|{args.split}/{fn}"
        boxes = gt[fn]
        if key not in cache:
            img = draw_boxes(Image.open(os.path.join(args.dataset, args.split, fn)), boxes)
            r = client.chat.completions.create(
                model=args.model, max_completion_tokens=8000,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": PROMPT.format(taxonomy=taxonomy)},
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{encode(img)}"}}]}])
            raw = (r.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].removeprefix("json").strip()
            try:
                answer = json.loads(raw).get("boxes", [])
            except json.JSONDecodeError:
                print(f"  [{n}/{len(files)}] {fn}: unparseable answer, skipped")
                continue
            cache[key] = answer
            with open(cache_path, "w") as f:
                json.dump(cache, f)
        got = {b["box"]: b.get("row") for b in cache[key] if isinstance(b, dict)}
        for i, box in enumerate(boxes):
            row = got.get(i)
            name = class_names[row] if isinstance(row, int) and 0 <= row < len(class_names) else None
            if name is None:
                stats["unnamed"] += 1
            elif name == box[0]:
                stats["correct"] += 1
            else:
                stats["wrong"] += 1
                if len(misses) < args.misses:
                    misses.append((fn, i, name, box[0]))
        print(f"  [{n}/{len(files)}] {fn}: {len(boxes)} boxes, "
              f"{sum(1 for i in range(len(boxes)) if got.get(i) is not None)} named")

    total = stats["correct"] + stats["wrong"] + stats["unnamed"]
    named = stats["correct"] + stats["wrong"]
    print(f"\n=== {args.split}: {len(files)} frames, {total} boxes, model {args.model}")
    print(f"  named    {named}/{total} ({named / max(total, 1):.0%}) -- the rest answered null")
    print(f"  correct  {stats['correct']}/{max(named, 1)} of named ({stats['correct'] / max(named, 1):.0%})")
    print(f"  yield    {stats['correct']}/{total} boxes named correctly "
          f"({stats['correct'] / max(total, 1):.0%})")
    for fn, i, got_name, truth in misses:
        print(f"    {fn} box {i}: {got_name!r}, truth {truth!r}")


if __name__ == "__main__":
    sys.exit(main())
