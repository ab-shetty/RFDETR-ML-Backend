#!/usr/bin/env python3
"""Learn the shelf-tag -> class mapping from labeled frames.

For each labeled frame: detect every shelf tag (one vision call), then pair
each tag to the labeled product box directly ABOVE it (same column, nearest
box whose center is above the tag). Record (normalized_tag_text -> class).
Aggregate across all frames into {tag: {class: count}} and write it as
models/tag_class_map.json, consumed by cascade/shelf_tags.py.

Boxes come from the COCO training dataset (see coco_dataset.py), train split
only -- the map is consulted at inference time, so building it from valid or
test would contaminate any later evaluation of shelf-tag correction.

Costs one vision call per frame, so it is not something to re-run casually;
--limit exists for a quick sanity run over a handful of frames.

Usage:
    OPENAI_API_KEY=... python scripts/build_tag_class_map.py \\
        --dataset-dir ~/Datasets/trader-joes/training-data/rf-detr-combined
"""
import argparse
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "label_studio_ml", "examples"))
from cascade.shelf_tags import detect_tags, normalize_tag  # noqa: E402
from coco_dataset import add_dataset_args, iter_labeled_images  # noqa: E402
from PIL import Image  # noqa: E402

# max horizontal offset (fraction of width) for a tag and a product box to be
# considered the same column, and max vertical gap for the box to be "above".
X_TOL = 0.09
Y_MAX_GAP = 0.22


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_args(ap)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "label_studio_ml", "examples", "models", "tag_class_map.json"))
    ap.add_argument("--limit", type=int, default=None, help="cap frames processed (for a quick run)")
    return ap.parse_args()


def associate(tag, boxes):
    """boxes: [(class, cx, cy)]. Return the class of the nearest box directly
    above the tag in the same column, or None."""
    tx, ty = float(tag["x"]), float(tag["y"])
    best, best_gap = None, 1e9
    for cls, bx, by in boxes:
        if by < ty and abs(bx - tx) < X_TOL and (ty - by) < Y_MAX_GAP and (ty - by) < best_gap:
            best, best_gap = cls, ty - by
    return best


def main():
    args = parse_args()
    tag_map = defaultdict(lambda: defaultdict(int))
    n_frames = n_tags = n_paired = 0

    for img_path, width, height, coco_boxes in iter_labeled_images(
            args.dataset_dir, args.splits):
        if args.limit and n_frames >= args.limit:
            break
        # associate() works in normalized coordinates (X_TOL and Y_MAX_GAP are
        # fractions of the frame, and detect_tags returns normalized centers),
        # while COCO stores absolute pixel corners.
        boxes = [(name, (x + w / 2) / width, (y + h / 2) / height)
                 for name, x, y, w, h in coco_boxes]
        if not boxes:
            continue
        image = Image.open(img_path).convert("RGB")
        tags = detect_tags(image)
        n_frames += 1; n_tags += len(tags)
        for tag in tags:
            cls = associate(tag, boxes)
            if cls:
                tag_map[normalize_tag(tag["name"])][cls] += 1
                n_paired += 1
        print(f"  {os.path.basename(img_path)}: {len(tags)} tags, "
              f"running paired={n_paired}", flush=True)

    out = {tag: dict(counts) for tag, counts in tag_map.items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"\nWrote {len(out)} distinct tags to {args.out}")
    print(f"  {n_frames} frames, {n_tags} tags detected, {n_paired} paired to a labeled box")


if __name__ == "__main__":
    main()
