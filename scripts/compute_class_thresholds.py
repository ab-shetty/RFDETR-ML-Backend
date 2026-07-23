#!/usr/bin/env python3
"""Sweep per-class confidence thresholds against a COCO validation split.

The ML backend currently applies one global MODEL_SCORE_THRESHOLD across all
classes, but per-class precision/recall is wildly uneven (some classes at
1.0 precision, others at 0.0) — see models/rfdetr_nano/results.json in the
team's Google Drive. A single cutoff guarantees the weak classes flood every
labeling batch with junk boxes.

This script runs real inference over a validation set, sweeps a grid of
candidate thresholds per class, and picks the one that maximizes F-beta
(beta<1 weights precision higher than recall, since false positives are the
named pain point). Output is a JSON map consumed by control_models/base.py.

Caveat (surfaced in the output, not hidden): with few validation images,
many classes have only 0-2 ground-truth instances. A threshold "optimized"
on 1 example is noise, not signal. Any class with fewer than --min-instances
ground-truth boxes falls back to --default-threshold and is flagged
"insufficient_data": true in the output. Re-run this after every training
run as validation data grows — it's meant to be cheap and repeatable, not a
one-off.

Usage:
    python scripts/compute_class_thresholds.py \\
        --checkpoint label_studio_ml/examples/models/checkpoint_best_total.pth \\
        --valid-dir /home/ubuntu/Datasets/trader-joes/training-data/rf-detr-combined/valid
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        "--checkpoint",
        default=os.path.join(os.path.dirname(__file__), "..", "label_studio_ml", "examples", "models", "checkpoint_best_total.pth"),
        help="Path to the .pth checkpoint. A companion .txt (class names) must sit next to it.",
    )
    ap.add_argument(
        "--valid-dir",
        default="/home/ubuntu/Datasets/trader-joes/training-data/rf-detr-combined/valid",
        help="Directory containing validation images + _annotations.coco.json",
    )
    ap.add_argument("--out", default=None, help="Output JSON path (default: class_thresholds.json next to --checkpoint)")
    ap.add_argument("--min-instances", type=int, default=3, help="Below this many GT boxes, fall back to --default-threshold")
    ap.add_argument("--beta", type=float, default=0.5, help="F-beta weight; <1 favors precision over recall")
    ap.add_argument("--default-threshold", type=float, default=float(os.getenv("MODEL_SCORE_THRESHOLD", 0.5)))
    ap.add_argument("--iou-thresh", type=float, default=0.5, help="IoU cutoff for a prediction to count as matching a GT box")
    ap.add_argument("--score-grid-step", type=float, default=0.05)
    return ap.parse_args()


def iou_matrix(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Vectorized IoU between two sets of [x1,y1,x2,y2] boxes -> (len(a), len(b))."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)))
    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0])
    y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2])
    y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def greedy_match(pred_boxes: np.ndarray, pred_scores: np.ndarray, gt_boxes: np.ndarray, iou_thresh: float):
    """Greedy highest-confidence-first IoU matching. Returns (tp_mask, n_gt_matched)."""
    n_pred = len(pred_boxes)
    if n_pred == 0:
        return np.zeros(0, dtype=bool), 0
    if len(gt_boxes) == 0:
        return np.zeros(n_pred, dtype=bool), 0
    ious = iou_matrix(pred_boxes, gt_boxes)
    order = np.argsort(-pred_scores)
    gt_used = np.zeros(len(gt_boxes), dtype=bool)
    tp = np.zeros(n_pred, dtype=bool)
    for i in order:
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= iou_thresh and not gt_used[j]:
            tp[i] = True
            gt_used[j] = True
    return tp, int(gt_used.sum())


def load_coco_gt(coco_path: str):
    with open(coco_path) as f:
        data = json.load(f)
    cat_id_to_name = {c["id"]: c["name"] for c in data["categories"]}
    img_id_to_file = {im["id"]: im["file_name"] for im in data["images"]}
    gt_by_image = defaultdict(list)  # image_id -> [(class_name, [x1,y1,x2,y2])]
    total_instances = defaultdict(int)
    for ann in data["annotations"]:
        x, y, w, h = ann["bbox"]
        name = cat_id_to_name[ann["category_id"]]
        gt_by_image[ann["image_id"]].append((name, [x, y, x + w, y + h]))
        total_instances[name] += 1
    return img_id_to_file, gt_by_image, total_instances


def main():
    args = parse_args()
    checkpoint_path = os.path.abspath(args.checkpoint)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint_name = os.path.basename(checkpoint_path)

    # base.py reads MODEL_ROOT at import time -> must set before importing it.
    os.environ["MODEL_ROOT"] = checkpoint_dir
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "label_studio_ml", "examples"))
    from control_models.base import ControlModel  # noqa: E402

    from PIL import Image  # noqa: E402

    print(f"Loading model from {checkpoint_path} ...")
    model, class_names = ControlModel.load_rfdetr_model(checkpoint_name)
    if not class_names:
        raise SystemExit(f"No class names found next to {checkpoint_path} (expected a .txt file).")
    print(f"Loaded {len(class_names)} classes.")

    dupes = {name for name in class_names if class_names.count(name) > 1}
    if dupes:
        print(
            f"  WARNING: {len(dupes)} class name(s) appear more than once in the class list: {sorted(dupes)}. "
            f"This is a pre-existing bug in the class taxonomy, not introduced by this script — "
            f"since thresholds (and the existing label_map elsewhere in this backend) are keyed by "
            f"name, duplicate class IDs silently collapse into one entry. Fix at the source "
            f"(master_list.csv / the checkpoint's class list) by renaming or removing the duplicate.",
            file=sys.stderr,
        )

    coco_path = os.path.join(args.valid_dir, "_annotations.coco.json")
    img_id_to_file, gt_by_image, total_instances = load_coco_gt(coco_path)
    print(f"Loaded {sum(total_instances.values())} GT boxes across {len(img_id_to_file)} validation images.")

    score_grid = np.round(np.arange(args.score_grid_step, 0.951, args.score_grid_step), 2)
    floor = float(score_grid[0])

    # counts[class][threshold] -> {tp, fp} accumulated across all images.
    # FN is derived at the end from total_instances - matched, per threshold.
    counts = defaultdict(lambda: defaultdict(lambda: {"tp": 0, "fp": 0}))
    matched_gt = defaultdict(lambda: defaultdict(int))

    for image_id, filename in img_id_to_file.items():
        image_path = os.path.join(args.valid_dir, filename)
        if not os.path.exists(image_path):
            print(f"  WARNING: {image_path} not found, skipping", file=sys.stderr)
            continue
        image = Image.open(image_path).convert("RGB")
        detections = model.predict(image, threshold=floor)

        gt_this_image = gt_by_image.get(image_id, [])
        classes_present = {name for name, _ in gt_this_image} | {
            class_names[int(cid)] for cid in detections.class_id if int(cid) < len(class_names)
        }

        for class_name in classes_present:
            gt_boxes = np.array([box for name, box in gt_this_image if name == class_name]) if gt_this_image else np.zeros((0, 4))
            pred_mask = np.array([
                int(cid) < len(class_names) and class_names[int(cid)] == class_name
                for cid in detections.class_id
            ]) if len(detections.class_id) else np.zeros(0, dtype=bool)
            pred_boxes = np.array(detections.xyxy)[pred_mask] if pred_mask.any() else np.zeros((0, 4))
            pred_scores = np.array(detections.confidence)[pred_mask] if pred_mask.any() else np.zeros(0)

            for thresh in score_grid:
                keep = pred_scores >= thresh
                tp_mask, n_matched = greedy_match(pred_boxes[keep], pred_scores[keep], gt_boxes, args.iou_thresh)
                counts[class_name][thresh]["tp"] += int(tp_mask.sum())
                counts[class_name][thresh]["fp"] += int((~tp_mask).sum())
                matched_gt[class_name][thresh] += n_matched

    results = {}
    beta2 = args.beta ** 2
    for class_name in class_names:
        n_gt = total_instances.get(class_name, 0)
        if n_gt < args.min_instances:
            results[class_name] = {
                "threshold": args.default_threshold,
                "n_gt_instances": n_gt,
                "insufficient_data": True,
            }
            continue

        best = {"threshold": args.default_threshold, "f_beta": -1.0, "precision": 0.0, "recall": 0.0}
        for thresh in score_grid:
            tp = counts[class_name][thresh]["tp"]
            fp = counts[class_name][thresh]["fp"]
            fn = n_gt - matched_gt[class_name][thresh]
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / n_gt if n_gt > 0 else 0.0
            denom = (beta2 * precision) + recall
            f_beta = (1 + beta2) * precision * recall / denom if denom > 0 else 0.0
            # Prefer higher F-beta; on ties prefer the lower threshold (more recall).
            if f_beta > best["f_beta"]:
                best = {"threshold": float(thresh), "f_beta": f_beta, "precision": precision, "recall": recall}

        results[class_name] = {
            "threshold": best["threshold"],
            "n_gt_instances": n_gt,
            "insufficient_data": False,
            "precision": round(best["precision"], 3),
            "recall": round(best["recall"], 3),
            "f_beta": round(best["f_beta"], 3),
        }

    out_path = args.out or os.path.join(checkpoint_dir, "class_thresholds.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, sort_keys=True)

    n_insufficient = sum(1 for r in results.values() if r["insufficient_data"])
    print(f"\nWrote {len(results)} class thresholds to {out_path}")
    print(f"  {n_insufficient} classes fell back to default ({args.default_threshold}) — fewer than {args.min_instances} validation instances.")
    print(f"  {len(results) - n_insufficient} classes got a tuned threshold.")


if __name__ == "__main__":
    main()
