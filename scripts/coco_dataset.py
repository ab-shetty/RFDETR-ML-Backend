"""Read the COCO training dataset that the served model was trained on.

Three scripts here need the same thing -- every labeled box, with its class
name and the image it came from -- and all three used to get it by scanning
labeling/completed/ for batches in a YOLO images/ + labels/ + classes.txt
layout. That layout stopped being produced when the export pipeline moved to
COCO, and the last batches that had it were archived, so all three quietly
degraded to reading almost nothing. One of them (build_reference_gallery.py)
was in the deploy path and shipped a 3-class gallery in place of a 115-class
one before anybody noticed.

Reading the dataset instead removes the failure mode rather than patching it:
it is the same COCO that trained the weights and named the classes, so nothing
derived from it can drift away from the model it describes.

**Train split only, by default.** The thresholds, the embedding gallery and the
tag map are all consulted at inference time, so building them from valid or
test would quietly contaminate any later evaluation of the cascade.
"""
import json
import os

DEFAULT_DATASET_DIR = os.path.expanduser(
    "~/Datasets/trader-joes/training-data/rf-detr-combined")


def add_dataset_args(ap, default_splits="train"):
    """The two flags every caller wants, spelled the same way in each."""
    ap.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR,
                    help="COCO dataset the model was trained on (holds train/valid/test)")
    ap.add_argument("--splits", default=default_splits,
                    help="comma-separated splits to read (default: %(default)s)")


def iter_labeled_images(dataset_dir, splits="train", on_missing=None):
    """Yield (image_path, width, height, boxes) for each labeled image.

    boxes is a list of (class_name, x, y, w, h) in absolute pixels, matching
    COCO's own convention. Callers that want normalized coordinates divide by
    the width and height yielded alongside.

    Grouped by image so each file is opened once by the caller rather than once
    per box. Images listed in the COCO but absent from disk are reported
    through on_missing (a callable taking the path) and skipped.
    """
    for split in [s.strip() for s in splits.split(",") if s.strip()]:
        split_dir = os.path.join(dataset_dir, split)
        coco_path = os.path.join(split_dir, "_annotations.coco.json")
        if not os.path.exists(coco_path):
            raise SystemExit(f"no _annotations.coco.json in {split_dir}")
        coco = json.load(open(coco_path))
        images = {im["id"]: im for im in coco["images"]}
        names = {c["id"]: c["name"] for c in coco["categories"]}

        by_image = {}
        for a in coco["annotations"]:
            by_image.setdefault(a["image_id"], []).append(a)

        for image_id, anns in by_image.items():
            im = images.get(image_id)
            if im is None:
                continue
            path = os.path.join(split_dir, im["file_name"])
            if not os.path.exists(path):
                if on_missing:
                    on_missing(path)
                continue
            boxes = []
            for a in anns:
                name = names.get(a["category_id"])
                if name is None:
                    continue
                x, y, w, h = a["bbox"]
                boxes.append((name, x, y, w, h))
            yield path, im["width"], im["height"], boxes


def split_summary(dataset_dir, splits="train"):
    """One line per split, for scripts to print before they start work."""
    lines = []
    for split in [s.strip() for s in splits.split(",") if s.strip()]:
        coco_path = os.path.join(dataset_dir, split, "_annotations.coco.json")
        coco = json.load(open(coco_path))
        populated = len({a["category_id"] for a in coco["annotations"]})
        lines.append(f"  {split}: {len(coco['images'])} images, "
                     f"{len(coco['annotations'])} boxes, "
                     f"{populated} of {len(coco['categories'])} classes present")
    return "\n".join(lines)
