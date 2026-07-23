#!/usr/bin/env python3
"""Find classes with ~no labeled training examples ("blind" classes) — the
ones no amount of threshold/cascade tuning can help, because RF-DETR has
never seen them. Restricted to categories where shelf tags carry useful
product-name text (Beverage/Food); Produce/Dairy items are typically loose
and untagged or generically labeled (matches NO_OCR_CATEGORIES in
build_ocr_expected_text.py) so a tag-based bootstrap can't help them.

Usage:
    python scripts/find_blind_classes.py \\
        --master-list /home/ubuntu/Datasets/master_list.csv \\
        --labeling-dir /home/ubuntu/Datasets/trader-joes/labeling/completed \\
        --out /tmp/blind_classes.json
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

NO_TAG_CATEGORIES = {"Produce", "Dairy"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master-list", default="/home/ubuntu/Datasets/master_list.csv")
    ap.add_argument("--labeling-dir", default="/home/ubuntu/Datasets/trader-joes/labeling/completed")
    ap.add_argument("--min-instances", type=int, default=3, help="classes with fewer labeled boxes than this count as blind")
    ap.add_argument("--out", default="/tmp/blind_classes.json")
    return ap.parse_args()


def count_labeled_instances(labeling_dir):
    """Standard 5-field YOLO label lines only (see build_reference_gallery.py
    for why the ad-hoc 9-field batches are excluded)."""
    counts = defaultdict(int)
    for entry in sorted(os.listdir(labeling_dir)):
        batch = os.path.join(labeling_dir, entry)
        classes_path = os.path.join(batch, "classes.txt")
        labels_dir = os.path.join(batch, "labels")
        if not (os.path.isdir(labels_dir) and os.path.exists(classes_path)):
            continue
        with open(classes_path) as f:
            classes = [line.strip() for line in f if line.strip()]
        for label_file in glob.glob(os.path.join(labels_dir, "*.txt")):
            with open(label_file) as f:
                for line in f:
                    parts = line.split()
                    if len(parts) != 5:
                        continue
                    class_id = int(parts[0])
                    if class_id < len(classes):
                        counts[classes[class_id]] += 1
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

    counts = count_labeled_instances(args.labeling_dir)

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
