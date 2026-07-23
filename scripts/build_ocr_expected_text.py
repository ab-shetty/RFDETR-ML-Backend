#!/usr/bin/env python3
"""Build the per-class expected-OCR-text map used by cascade/ocr.py.

Not every class benefits from OCR: "Apple" or "Milk" are visual-category
names, not printed label text, so matching OCR output against them is noise.
Specific branded products ("San Pellegrino 6-pack", "Sparkling Green Tea
with Pineapple") usually do have their name readable on the package, so OCR
is a real signal there.

Heuristic, grounded in the actual taxonomy (see master_list.csv): classes in
the "Produce" and "Dairy" categories are generic product-type names (Apple,
Milk, Yoghurt, ...) -> no expected text. "Beverage" and "Food" categories are
specific named products -> expected text = the class name itself.

Usage:
    python scripts/build_ocr_expected_text.py \\
        --master-list /home/ubuntu/Datasets/master_list.csv
"""
import argparse
import csv
import json
import os

NO_OCR_CATEGORIES = {"Produce", "Dairy"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--master-list", default="/home/ubuntu/Datasets/master_list.csv")
    ap.add_argument(
        "--out",
        default=os.path.join(os.path.dirname(__file__), "..", "label_studio_ml", "examples", "models", "ocr_expected_text.json"),
    )
    return ap.parse_args()


def main():
    args = parse_args()
    with open(args.master_list) as f:
        rows = list(csv.DictReader(f))

    result = {}
    n_with_text = 0
    for row in rows:
        # master_list.csv has at least one known trailing-whitespace entry
        # ("coconut water ") — strip so it matches the class names the model
        # actually uses (see checkpoint_best_total.txt / training COCO categories).
        class_name = row["Class Name (str)"].strip()
        category = row["Category"].strip()
        if category in NO_OCR_CATEGORIES:
            result[class_name] = None
        else:
            result[class_name] = class_name
            n_with_text += 1

    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, sort_keys=True)

    print(f"Wrote {len(result)} classes to {args.out}")
    print(f"  {n_with_text} with expected OCR text (Beverage/Food)")
    print(f"  {len(result) - n_with_text} with no OCR signal (Produce/Dairy)")


if __name__ == "__main__":
    main()
