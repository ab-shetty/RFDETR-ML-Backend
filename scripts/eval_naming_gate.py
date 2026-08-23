#!/usr/bin/env python3
"""Is there a confidence band where RF-DETR's class head should still name a box?

Box-naming (a vision call over drawn boxes) beats the class head overall on both
held-out stores, but "overall" hides the case that matters: the head is not
uniformly weak. It is weak on the long tail it barely saw and strong on the SKUs
it saw hundreds of times, and its confidence is supposed to say which is which.
If a high-confidence band exists where it wins, that band should keep it.

The decision-relevant number is not each side's accuracy but who is right when
they DISAGREE -- where they agree, whichever names the box gives the same answer
and the choice is free.

Boxes are ground truth, so both namers are graded on the same boxes.

    python3 scripts/eval_naming_gate.py --dataset .../rf-detr-combined \\
        --weights .../checkpoint_best_total.pth --splits valid test
"""
import argparse
import json
import os
import sys
from collections import defaultdict

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from eval_tag_naming import load_split  # noqa: E402

BUCKETS = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]


def ordered_categories(dataset, split):
    """Class names by category id, duplicates kept.

    NOT load_split's name list: that one is sorted and de-duplicated, which is
    fine for scoring a name against a name, and silently wrong here. The head's
    class id is a position in the dataset's category order, so the id has to be
    looked up in that order or every name comes out shifted.
    """
    coco = json.load(open(os.path.join(dataset, split, "_annotations.coco.json")))
    return [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua


def load_model(weights, num_classes):
    import math

    import rfdetr
    import torch

    ckpt = torch.load(weights, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict") or ckpt.get("model") or {}
    n_pos = patch = None
    for key, value in state.items():
        if key.endswith("embeddings.position_embeddings") and getattr(value, "ndim", 0) == 3:
            n_pos = int(value.shape[1])
        elif key.endswith("patch_embeddings.projection.weight") and getattr(value, "ndim", 0) == 4:
            patch = int(value.shape[-1])
    resolution = next((math.isqrt(n) * patch for n in (n_pos, n_pos - 1)
                       if n_pos and patch and math.isqrt(n) ** 2 == n), None)
    model_name = ckpt.get("model_name") or "RFDETRNano"
    del ckpt, state
    print(f"{os.path.basename(weights)}: {model_name}, {num_classes} classes, "
          f"resolution {resolution}")
    return getattr(rfdetr, model_name)(pretrain_weights=weights, num_classes=num_classes,
                                       **({"resolution": resolution} if resolution else {}))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--splits", nargs="+", default=["valid", "test"])
    ap.add_argument("--vision-cache", default=os.path.join(HERE, "..", "data", "box_naming.json"))
    ap.add_argument("--vision-model", default="gpt-5.6-terra")
    ap.add_argument("--floor", type=float, default=0.05, help="detector floor; low on purpose, "
                                                              "the point is to see every band")
    args = ap.parse_args()

    vision = json.load(open(os.path.abspath(args.vision_cache)))
    model = None
    rows = []          # (confidence, head_correct, vision_correct)

    for split in args.splits:
        files, gt, class_names = load_split(args.dataset, split)
        head_names = ordered_categories(args.dataset, split)
        if model is None:
            model = load_model(args.weights, len(head_names))
        done = [f for f in sorted(files)
                if f"{args.vision_model}|{split}/{f}" in vision and gt[f]]
        print(f"{split}: {len(done)} frames with a cached vision naming")
        for n, fn in enumerate(done, 1):
            boxes = gt[fn]
            named = {b["box"]: b.get("row") for b in vision[f"{args.vision_model}|{split}/{fn}"]
                     if isinstance(b, dict)}
            img = Image.open(os.path.join(args.dataset, split, fn)).convert("RGB")
            W, H = img.size
            det = model.predict(img, threshold=args.floor)
            # class-agnostic greedy match, most confident detection first
            order = sorted(range(len(det.confidence)), key=lambda i: -det.confidence[i])
            taken = set()
            for i in order:
                dx1, dy1, dx2, dy2 = [float(v) for v in det.xyxy[i]]
                dbox = (dx1 / W, dy1 / H, dx2 / W, dy2 / H)
                best, best_iou = None, 0.5
                for bi, b in enumerate(boxes):
                    if bi in taken:
                        continue
                    o = iou(dbox, (b[1], b[2], b[3], b[4]))
                    if o >= best_iou:
                        best, best_iou = bi, o
                if best is None:
                    continue
                taken.add(best)
                truth = boxes[best][0]
                row = named.get(best)
                vname = (class_names[row] if isinstance(row, int)
                         and 0 <= row < len(class_names) else None)
                rows.append((float(det.confidence[i]), int(det.class_id[i]), truth,
                             vname == truth, head_names))
            print(f"  [{n}/{len(done)}] {fn}: {len(taken)}/{len(boxes)} GT boxes matched")

    # rfdetr's predict() has returned class ids both 0- and 1-indexed across
    # versions, and an off-by-one reads as a model that is wrong about
    # everything rather than as a bug. Pick the convention the weights actually
    # use instead of assuming one.
    def head_name(conf, cid, truth, vok, names, shift):
        j = cid + shift
        return names[j] if 0 <= j < len(names) else None

    shifts = {sh: sum(head_name(*r, sh) == r[2] for r in rows) for sh in (0, -1, 1)}
    shift = max(shifts, key=shifts.get)
    print(f"\nclass-id convention: shift {shift:+d} "
          f"({', '.join(f'{k:+d}:{v}' for k, v in sorted(shifts.items()))} correct)")
    rows = [(r[0], head_name(*r, shift) == r[2], r[3]) for r in rows]

    print(f"\n=== {len(rows)} ground-truth boxes the detector found and the vision model named")
    print(f"{'confidence':>12} {'n':>5} {'class head':>11} {'box naming':>11} "
          f"{'disagree':>9} {'head wins':>10} {'vision wins':>12}")
    for lo, hi in BUCKETS + [(0.0, 1.01)]:
        sel = [r for r in rows if lo <= r[0] < hi]
        if not sel:
            continue
        dis = [r for r in sel if r[1] != r[2]]
        label = "ALL" if (lo, hi) == (0.0, 1.01) else f"{lo:.1f}-{hi if hi <= 1 else 1:.1f}"
        print(f"{label:>12} {len(sel):>5} "
              f"{sum(r[1] for r in sel) / len(sel):>10.0%} "
              f"{sum(r[2] for r in sel) / len(sel):>10.0%} "
              f"{len(dis):>9} "
              f"{sum(1 for r in dis if r[1]):>10} "
              f"{sum(1 for r in dis if r[2]):>12}")


if __name__ == "__main__":
    sys.exit(main())
