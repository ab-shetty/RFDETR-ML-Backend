#!/usr/bin/env python3
"""Build a per-class embedding reference gallery from already-labeled crops.

Reads completed labeling batches (YOLO-polygon format: images/, labels/,
classes.txt per batch), crops each labeled box, embeds it with the RF-DETR
backbone (see cascade/embedding_match.py), and writes a per-class centroid
to models/reference_gallery.npz. Used at inference time to check whether a
detection's embedding actually looks like the class RF-DETR predicted.

Caveat, same spirit as compute_class_thresholds.py: with only a handful of
labeled batches, some classes will have very few reference crops — a
centroid from 1-2 examples is a rough signal, not a precise one. Classes
below --min-instances are simply omitted from the gallery (embedding_match
already treats "not in gallery" as "no signal" rather than a rejection).

Scope note: only covers completed batches with the images/ + labels/ +
classes.txt layout, standard 5-field YOLO (class cx cy w h) label lines.
clip1.labels/clip2.labels used an ad-hoc 9-field 4-corner-polygon format by
mistake (not a second valid format) — those lines are skipped, not parsed.
Batches exported as a raw Label Studio result.json (clip3.labels,
clip4.labels) aren't parsed here either; add a parser for that format if/when
it's worth the reference examples they'd contribute.

Usage:
    python scripts/build_reference_gallery.py \\
        --checkpoint label_studio_ml/examples/models/checkpoint_best_total.pth \\
        --labeling-dir /home/ubuntu/Datasets/trader-joes/labeling/completed
"""
import argparse
import glob
import logging
import os
import sys
from collections import defaultdict

import numpy as np

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--checkpoint",
        default=os.path.join(os.path.dirname(__file__), "..", "label_studio_ml", "examples", "models", "checkpoint_best_total.pth"),
    )
    ap.add_argument(
        "--labeling-dir",
        default="/home/ubuntu/Datasets/trader-joes/labeling/completed",
        help="Directory containing completed labeling batches",
    )
    ap.add_argument("--out", default=None, help="Output .npz path (default: reference_gallery.npz next to --checkpoint)")
    ap.add_argument("--min-instances", type=int, default=1, help="Classes with fewer crops than this are omitted")
    return ap.parse_args()


def yolo_to_bbox(cx, cy, w, h, width, height):
    """Standard YOLO normalized [center_x, center_y, w, h] -> pixel [x1,y1,x2,y2]."""
    x1 = (cx - w / 2) * width
    y1 = (cy - h / 2) * height
    x2 = (cx + w / 2) * width
    y2 = (cy + h / 2) * height
    return x1, y1, x2, y2


def parse_label_line(parts, width, height):
    """Standard 5-field YOLO (class cx cy w h). Some older batches
    (clip1/clip2) used an ad-hoc 9-field 4-corner polygon format instead —
    that was a labeling-process mistake, not a second valid format, so those
    lines are treated as unrecognized and skipped rather than parsed.
    """
    if len(parts) == 5:
        class_id = int(parts[0])
        cx, cy, w, h = (float(v) for v in parts[1:])
        return class_id, yolo_to_bbox(cx, cy, w, h, width, height)
    return None


def find_yolo_batches(labeling_dir):
    """Batches with the images/ + labels/ + classes.txt layout."""
    batches = []
    for entry in sorted(os.listdir(labeling_dir)):
        batch_dir = os.path.join(labeling_dir, entry)
        if os.path.isdir(os.path.join(batch_dir, "images")) and os.path.isdir(os.path.join(batch_dir, "labels")) and os.path.exists(os.path.join(batch_dir, "classes.txt")):
            batches.append(batch_dir)
    return batches


def main():
    args = parse_args()
    checkpoint_path = os.path.abspath(args.checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_name = os.path.basename(checkpoint_path)

    os.environ["MODEL_ROOT"] = checkpoint_dir
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "label_studio_ml", "examples"))
    from control_models.base import ControlModel  # noqa: E402
    from cascade.embedding_match import extract_embedding, get_backbone_nn_module  # noqa: E402
    from PIL import Image  # noqa: E402

    print(f"Loading model from {checkpoint_path} ...")
    rfdetr_model, _ = ControlModel.load_rfdetr_model(checkpoint_name)
    nn_model = get_backbone_nn_module(rfdetr_model)

    batches = find_yolo_batches(args.labeling_dir)
    if not batches:
        raise SystemExit(f"No YOLO-format labeling batches found under {args.labeling_dir}")
    print(f"Found {len(batches)} usable batches: {[os.path.basename(b) for b in batches]}")

    embeddings_by_class = defaultdict(list)

    for batch_dir in batches:
        with open(os.path.join(batch_dir, "classes.txt")) as f:
            batch_classes = [line.strip() for line in f if line.strip()]

        label_files = sorted(glob.glob(os.path.join(batch_dir, "labels", "*.txt")))
        print(f"  {os.path.basename(batch_dir)}: {len(label_files)} labeled images, {len(batch_classes)} classes")

        for label_file in label_files:
            stem = os.path.splitext(os.path.basename(label_file))[0]
            image_path = os.path.join(batch_dir, "images", stem + ".jpg")
            if not os.path.exists(image_path):
                continue
            image = Image.open(image_path).convert("RGB")
            width, height = image.size

            with open(label_file) as f:
                for line in f:
                    parts = line.split()
                    if not parts:
                        continue
                    parsed = parse_label_line(parts, width, height)
                    if parsed is None:
                        logger.warning(f"Unrecognized label line shape ({len(parts)} fields) in {label_file}, skipping")
                        continue
                    class_id, (x1, y1, x2, y2) = parsed
                    if class_id >= len(batch_classes):
                        continue
                    class_name = batch_classes[class_id]
                    if x2 - x1 < 2 or y2 - y1 < 2:
                        continue
                    crop = image.crop((x1, y1, x2, y2))
                    embedding = extract_embedding(nn_model, crop)
                    if np.any(embedding):
                        embeddings_by_class[class_name].append(embedding)

    class_names = []
    centroids = []
    n_omitted = 0
    for class_name, vectors in sorted(embeddings_by_class.items()):
        if len(vectors) < args.min_instances:
            n_omitted += 1
            continue
        class_names.append(class_name)
        centroids.append(np.mean(np.stack(vectors), axis=0))

    if not class_names:
        raise SystemExit("No classes met --min-instances -- nothing to write.")

    out_path = args.out or os.path.join(checkpoint_dir, "reference_gallery.npz")
    np.savez(out_path, class_names=np.array(class_names), centroids=np.stack(centroids))

    print(f"\nWrote reference gallery for {len(class_names)} classes to {out_path}")
    print(f"  {n_omitted} classes omitted (fewer than {args.min_instances} labeled crop(s))")
    print(f"  Total crops embedded: {sum(len(v) for v in embeddings_by_class.values())}")


if __name__ == "__main__":
    main()
