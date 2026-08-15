#!/usr/bin/env python3
"""Build the ORB template bank used to name boxes the SKU model cannot.

The detector only detects what it recognises, so a facing it failed to box is
one it had already declined to identify -- measured at 7 correct out of 79 when
asked to classify such a crop, against 77% on boxes it found itself. Template
matching reaches those: it named 23 of them with 23 correct on SKUs present in
the bank.

Deliberately per-instance, unlike reference_gallery.npz, which averages a
class's crops into a single centroid. Averaging suits a smooth embedding space
and actively hurts here: what separates two Trader Joe's coffee boxes is a word
and a colour band, and a mean of seven photographs at seven angles is exactly
where that detail goes. Keypoints keep it, and tolerate the scale and rotation
changes that separate one shelf frame from another.

Stored flat -- one concatenated descriptor array plus offsets -- rather than as
an object array of per-crop matrices, so it loads with a plain np.load and no
pickling.

Build from the split the model trained on, never from the split you intend to
score; the two datasets here split the same 173 images differently, and the
default rf-detr-combined/train contains 22 of the 28 box-detector test images.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coco_dataset import add_dataset_args, iter_labeled_images, split_summary  # noqa: E402

MIN_DESCRIPTORS = 8
MIN_CROP_PX = 16


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_dataset_args(ap)
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..", "label_studio_ml", "examples", "models",
        "template_bank.npz"))
    ap.add_argument("--orb-features", type=int, default=200)
    args = ap.parse_args()

    import cv2
    from PIL import Image

    print(split_summary(args.dataset_dir, args.splits))
    orb = cv2.ORB_create(nfeatures=args.orb_features)

    names, blocks, skipped = [], [], 0
    for path, _w, _h, boxes in iter_labeled_images(args.dataset_dir, args.splits):
        image = Image.open(path).convert("RGB")
        for name, x, y, w, h in boxes:
            if w < MIN_CROP_PX or h < MIN_CROP_PX:
                skipped += 1
                continue
            crop = image.crop((x, y, x + w, y + h))
            gray = cv2.cvtColor(np.array(crop), cv2.COLOR_RGB2GRAY)
            _kp, des = orb.detectAndCompute(gray, None)
            if des is None or len(des) < MIN_DESCRIPTORS:
                skipped += 1
                continue
            names.append(name)
            blocks.append(des.astype(np.uint8))

    if not blocks:
        raise SystemExit("no usable crops — is --dataset-dir right?")

    offsets = np.zeros(len(blocks) + 1, dtype=np.int64)
    for i, b in enumerate(blocks):
        offsets[i + 1] = offsets[i] + len(b)
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez(out,
             names=np.array(names),
             offsets=offsets,
             descriptors=np.concatenate(blocks, axis=0))

    print(f"\nWrote {len(names)} templates over {len(set(names))} SKUs to {out}")
    print(f"  {offsets[-1]} descriptors, {os.path.getsize(out)/1048576:.1f} MB")
    print(f"  {skipped} crops skipped (too small, or too few keypoints)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
