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
import json
import os
import sys
from collections import Counter

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "label_studio_ml", "examples"))
from cascade.box_naming import PROMPT, draw_boxes, encode  # noqa: E402
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


def box_iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    if inter <= 0:
        return 0.0
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def px_boxes(rows, size):
    """(name, x1,y1,x2,y2 normalized) rows -> pixel xyxy, what the module takes."""
    W, H = size
    return [(x1 * W, y1 * H, x2 * W, y2 * H) for _, x1, y1, x2, y2 in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--model", default=os.getenv("BOX_NAMING_MODEL", "gpt-5.6-terra"))
    ap.add_argument("--limit", type=int, help="frames (they are paid calls)")
    ap.add_argument("--cache", default=os.path.join(HERE, "..", "data", "box_naming.json"))
    ap.add_argument("--boxes", choices=("gt", "detector"), default="gt",
                    help="gt measures naming alone; detector is the production "
                         "condition -- the class-agnostic facing detector draws the "
                         "boxes, so a missed facing is never named and a stray one is "
                         "named for nothing")
    ap.add_argument("--misses", type=int, default=8)
    ap.add_argument("--dump-annotated", help="write the first annotated frame here, to check "
                                             "the boxes are legible before paying for a run")
    args = ap.parse_args()

    files, gt, class_names = load_split(args.dataset, args.split)
    files = [f for f in sorted(files) if gt[f]][:args.limit]
    if args.boxes == "detector":
        os.environ.setdefault("MODEL_ROOT", os.path.join(
            os.path.dirname(HERE), "label_studio_ml", "examples", "models"))
    taxonomy = "\n".join(f"{i}: {n}" for i, n in enumerate(class_names))

    if args.dump_annotated:
        f = files[0]
        img = Image.open(os.path.join(args.dataset, args.split, f))
        draw_boxes(img, px_boxes(gt[f], img.size)).save(args.dump_annotated)
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
        key = f"{args.model}|{args.boxes}|{args.split}/{fn}"
        src_img = Image.open(os.path.join(args.dataset, args.split, fn)).convert("RGB")
        if args.boxes == "gt":
            boxes = gt[fn]
        else:
            # Detector boxes, then each one labelled with the ground-truth class
            # it overlaps, so the same scoring code works. An unmatched box is a
            # stray: nothing to be right about, and naming it is wasted.
            from control_models.box_proposals import propose
            W, H = src_img.size
            found = propose(src_img, [])
            stats["proposed"] += len(found)
            taken = set()
            boxes = []
            for x1, y1, x2, y2, _score in found:
                nb = (x1 / W, y1 / H, x2 / W, y2 / H)
                best, best_iou = None, 0.5
                for bi, b in enumerate(gt[fn]):
                    if bi in taken:
                        continue
                    o = box_iou(nb, (b[1], b[2], b[3], b[4]))
                    if o >= best_iou:
                        best, best_iou = bi, o
                if best is not None:
                    taken.add(best)
                boxes.append((gt[fn][best][0] if best is not None else None, *nb))
            stats["missed"] += len(gt[fn]) - len(taken)
        if key not in cache:
            img = draw_boxes(src_img, px_boxes(boxes, src_img.size))
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
            if box[0] is None:
                stats["stray"] += 1
            elif name is None:
                stats["unnamed"] += 1
            elif name == box[0]:
                stats["correct"] += 1
            else:
                stats["wrong"] += 1
                if len(misses) < args.misses:
                    misses.append((fn, i, name, box[0]))
        print(f"  [{n}/{len(files)}] {fn}: {len(boxes)} boxes, "
              f"{sum(1 for i in range(len(boxes)) if got.get(i) is not None)} named")

    if args.boxes == "detector":
        gt_total = sum(len(gt[f]) for f in files)
        found = gt_total - stats["missed"]
        print(f"\n=== {args.split}: {len(files)} frames, boxes from the facing detector")
        print(f"  found    {found}/{gt_total} labelled facings ({found / max(gt_total, 1):.0%} "
              f"recall), {stats['stray']} strays out of {stats['proposed']} proposed")
        print(f"  named    {stats['correct']}/{max(found, 1)} of the found facings named "
              f"correctly ({stats['correct'] / max(found, 1):.0%})")
        print(f"  end-to-end {stats['correct']}/{gt_total} labelled facings both found and "
              f"named ({stats['correct'] / max(gt_total, 1):.0%})")
        for fn, i, got_name, truth in misses:
            print(f"    {fn} box {i}: {got_name!r}, truth {truth!r}")
        return

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
