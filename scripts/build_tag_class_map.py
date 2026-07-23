#!/usr/bin/env python3
"""Learn the shelf-tag -> class mapping from labeled frames.

For each labeled frame: detect every shelf tag (one vision call), then pair
each tag to the labeled product box directly ABOVE it (same column, nearest
box whose center is above the tag). Record (normalized_tag_text -> class).
Aggregate across all frames into {tag: {class: count}} and write it as
models/tag_class_map.json, consumed by cascade/shelf_tags.py.

Only standard 5-field YOLO label batches are used (class cx cy w h); the
ad-hoc 9-field polygon batches are skipped (same as build_reference_gallery).

Usage:
    OPENAI_API_KEY=... python scripts/build_tag_class_map.py \\
        --labeling-dir /home/ubuntu/Datasets/trader-joes/labeling/completed \\
        --batches abhishek-1 abhishek-2
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "label_studio_ml", "examples"))
from cascade.shelf_tags import detect_tags, normalize_tag  # noqa: E402
from PIL import Image  # noqa: E402

# max horizontal offset (fraction of width) for a tag and a product box to be
# considered the same column, and max vertical gap for the box to be "above".
X_TOL = 0.09
Y_MAX_GAP = 0.22


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labeling-dir", default="/home/ubuntu/Datasets/trader-joes/labeling/completed")
    ap.add_argument("--batches", nargs="+", default=["abhishek-1", "abhishek-2"])
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "label_studio_ml", "examples", "models", "tag_class_map.json"))
    ap.add_argument("--limit", type=int, default=None, help="cap frames per batch (for a quick run)")
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

    for batch in args.batches:
        bdir = os.path.join(args.labeling_dir, batch)
        classes = [l.strip() for l in open(os.path.join(bdir, "classes.txt")) if l.strip()]
        label_files = sorted(glob.glob(os.path.join(bdir, "labels", "*.txt")))
        if args.limit:
            label_files = label_files[: args.limit]
        print(f"{batch}: {len(label_files)} frames")
        for lf in label_files:
            stem = os.path.splitext(os.path.basename(lf))[0]
            img_path = os.path.join(bdir, "images", stem + ".jpg")
            if not os.path.exists(img_path):
                continue
            boxes = []
            for line in open(lf):
                p = line.split()
                if len(p) != 5:   # skip non-standard (9-field polygon) lines
                    continue
                cid = int(p[0]); cx, cy = float(p[1]), float(p[2])
                if cid < len(classes):
                    boxes.append((classes[cid], cx, cy))
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
            print(f"  {stem}: {len(tags)} tags, running paired={n_paired}", flush=True)

    out = {tag: dict(counts) for tag, counts in tag_map.items()}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f"\nWrote {len(out)} distinct tags to {args.out}")
    print(f"  {n_frames} frames, {n_tags} tags detected, {n_paired} paired to a labeled box")


if __name__ == "__main__":
    main()
