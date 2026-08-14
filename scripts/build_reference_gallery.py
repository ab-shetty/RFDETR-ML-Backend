#!/usr/bin/env python3
"""Build a per-class embedding reference gallery from already-labeled crops.

Reads the COCO training dataset the model was trained on, crops each labeled
box, embeds it with the RF-DETR backbone (see cascade/embedding_match.py), and
writes a per-class centroid to models/reference_gallery.npz. Used at inference
time to check whether a detection's embedding actually looks like the class
RF-DETR predicted.

**Reads the dataset, not the labeling folder.** This used to scan
labeling/completed/ for batches in the YOLO images/ + labels/ + classes.txt
layout. Two things then went wrong at once, quietly: the export pipeline moved
to COCO and stopped producing that layout, and the last batches that still had
it (abhishek-1, abhishek-2) were archived to to_review_for_deletion. What was
left was clip1/clip2, whose 9-field polygon label lines this never parsed, so
the builder found 101 crops in 3 classes and cheerfully wrote a 3-class gallery
to replace a 115-class one. Reading the dataset removes the whole class of
problem: it is the same COCO that trained the weights and named the classes, so
the gallery cannot drift away from the model it describes.

Train split only, by default. Embedding valid/test crops would make the
cascade's own evaluation meaningless later, and the gallery is not something
you tune against a held-out set anyway.

Caveat, same spirit as compute_class_thresholds.py: at this dataset size some
classes have very few reference crops — a centroid from 1-2 examples is a rough
signal, not a precise one. Classes below --min-instances are omitted, and
embedding_match already treats "not in gallery" as "no signal" rather than as a
rejection.

Usage:
    python scripts/build_reference_gallery.py \\
        --checkpoint label_studio_ml/examples/models/checkpoint_best_total.pth \\
        --dataset-dir ~/Datasets/trader-joes/training-data/rf-detr-combined
"""
import argparse
import logging
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coco_dataset import add_dataset_args, iter_labeled_images, split_summary  # noqa: E402

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--checkpoint",
        default=os.path.join(os.path.dirname(__file__), "..", "label_studio_ml", "examples", "models", "checkpoint_best_total.pth"),
    )
    add_dataset_args(ap)
    ap.add_argument("--out", default=None, help="Output .npz path (default: reference_gallery.npz next to --checkpoint)")
    ap.add_argument("--min-instances", type=int, default=1, help="Classes with fewer crops than this are omitted")
    return ap.parse_args()


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

    print(split_summary(args.dataset_dir, args.splits))

    embeddings_by_class = defaultdict(list)
    missing = []
    for image_path, _width, _height, boxes in iter_labeled_images(
            args.dataset_dir, args.splits, on_missing=missing.append):
        image = Image.open(image_path).convert("RGB")
        for class_name, x, y, w, h in boxes:
            if w < 2 or h < 2:
                continue
            crop = image.crop((x, y, x + w, y + h))
            embedding = extract_embedding(nn_model, crop)
            if np.any(embedding):
                embeddings_by_class[class_name].append(embedding)

    if missing:
        logger.warning(f"{len(missing)} images listed in COCO were not on disk")

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
