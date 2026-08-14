#!/usr/bin/env python3
"""Find classes with ~no labeled training examples ("blind" classes) — the
ones no amount of threshold/cascade tuning can help, because RF-DETR has
never seen them. Restricted to categories where shelf tags carry useful
product-name text (Beverage/Food); Produce/Dairy items are typically loose
and untagged or generically labeled (matches NO_OCR_CATEGORIES in
build_ocr_expected_text.py) so a tag-based bootstrap can't help them.

Counts come from the COCO training dataset (see coco_dataset.py) -- the train
split specifically, since "blind" means RF-DETR never saw the class, and a
class labeled only in valid is exactly as blind as one labeled nowhere.

Usage:
    python scripts/find_blind_classes.py \\
        --master-list /home/ubuntu/Datasets/master_list.csv \\
        --dataset-dir ~/Datasets/trader-joes/training-data/rf-detr-combined \\
        --out /tmp/blind_classes.json
"""
import argparse
import csv
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coco_dataset import add_dataset_args, iter_labeled_images  # noqa: E402

NO_TAG_CATEGORIES = {"Produce", "Dairy"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master-list", default="/home/ubuntu/Datasets/master_list.csv")
    add_dataset_args(ap)
    ap.add_argument("--min-instances", type=int, default=3, help="classes with fewer labeled boxes than this count as blind")
    ap.add_argument("--out", default="/tmp/blind_classes.json")
    return ap.parse_args()


def count_labeled_instances(dataset_dir, splits):
    counts = defaultdict(int)
    for _path, _w, _h, boxes in iter_labeled_images(dataset_dir, splits):
        for name, *_ in boxes:
            counts[name] += 1
    return counts


def main():
    args = parse_args()
    master = []
    with open(args.master_list) as f:
        for row in csv.DictReader(f):
            name = row["Class Name (str)"].strip()
            category = row["Category"].strip()
            if name:
                master.append((name, category))

    counts = count_labeled_instances(args.dataset_dir, args.splits)

    blind = [
        {"class_name": name, "category": category, "instances": counts.get(name, 0)}
        for name, category in master
        if counts.get(name, 0) < args.min_instances and category not in NO_TAG_CATEGORIES
    ]
    blind.sort(key=lambda b: b["instances"])

    skipped_no_tag = sum(
        1 for name, category in master if counts.get(name, 0) < args.min_instances and category in NO_TAG_CATEGORIES
    )

    with open(args.out, "w") as f:
        json.dump(blind, f, indent=2)

    print(f"{len(master)} total classes")
    print(f"{len(blind)} blind (<{args.min_instances} instances) AND tag-bootstrappable (Beverage/Food) -> {args.out}")
    print(f"{skipped_no_tag} more are blind but in Produce/Dairy -- shelf tags won't help those, skipped")
    for b in blind:
        print(f"  {b['instances']:2d}  {b['class_name']}")


if __name__ == "__main__":
    main()
