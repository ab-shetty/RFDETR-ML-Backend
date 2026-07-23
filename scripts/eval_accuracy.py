"""Measure pre-annotation accuracy on held-out splits at the DEPLOYED operating
point (per-class thresholds). Reports two views that matter for pre-annotation:

  1. Class-aware: is the box in the right place AND labeled the right class?
     (what a fully-correct pre-annotation needs)
  2. Class-agnostic / localization-only: is there a well-placed box at all,
     regardless of class? (a correctly-placed box with the wrong SKU is still
     useful — the human just fixes the label, which is far cheaper than drawing
     the box from scratch)
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

os.environ.setdefault("MODEL_ROOT", "/app/models")
sys.path.insert(0, "/app")

from control_models.base import load_class_thresholds, MODEL_SCORE_THRESHOLD, ControlModel
from PIL import Image


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = a[:, None, :]; b = b[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0]); y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2]); y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def match(pred_boxes, pred_scores, gt_boxes, iou_t=0.5):
    """Greedy highest-score-first. Returns TP count and matched-GT count."""
    if len(pred_boxes) == 0:
        return 0, 0
    if len(gt_boxes) == 0:
        return 0, 0
    ious = iou_matrix(pred_boxes, gt_boxes)
    order = np.argsort(-pred_scores)
    gt_used = np.zeros(len(gt_boxes), bool)
    tp = 0
    for i in order:
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= iou_t and not gt_used[j]:
            tp += 1; gt_used[j] = True
    return tp, int(gt_used.sum())


def evaluate(split_dir, model, class_names, thresholds):
    coco = json.load(open(os.path.join(split_dir, "_annotations.coco.json")))
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    img_file = {im["id"]: im["file_name"] for im in coco["images"]}
    gt = defaultdict(list)  # img_id -> [(class_name, [x1y1x2y2])]
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        gt[a["image_id"]].append((cat[a["category_id"]], [x, y, x + w, y + h]))

    floor = max(0.05, min(list(thresholds.values()) + [MODEL_SCORE_THRESHOLD]))
    # class-aware and class-agnostic tallies
    ca = {"tp": 0, "fp": 0, "gt": 0}
    cg = {"tp": 0, "fp": 0, "gt": 0}

    for img_id, fname in img_file.items():
        path = os.path.join(split_dir, fname)
        if not os.path.exists(path):
            continue
        image = Image.open(path).convert("RGB")
        det = model.predict(image, threshold=floor)

        # apply deployed per-class thresholds
        keep_boxes, keep_scores, keep_names = [], [], []
        for i in range(len(det.xyxy)):
            cid = int(det.class_id[i])
            if cid >= len(class_names):
                continue
            name = class_names[cid]
            score = float(det.confidence[i])
            if score >= thresholds.get(name, MODEL_SCORE_THRESHOLD):
                keep_boxes.append(det.xyxy[i]); keep_scores.append(score); keep_names.append(name)
        keep_boxes = np.array(keep_boxes) if keep_boxes else np.zeros((0, 4))
        keep_scores = np.array(keep_scores) if keep_scores else np.zeros(0)

        gt_items = gt.get(img_id, [])
        ca["gt"] += len(gt_items); cg["gt"] += len(gt_items)

        # class-agnostic: all kept boxes vs all GT boxes
        gt_all = np.array([b for _, b in gt_items]) if gt_items else np.zeros((0, 4))
        tp_cg, _ = match(keep_boxes, keep_scores, gt_all)
        cg["tp"] += tp_cg; cg["fp"] += len(keep_boxes) - tp_cg

        # class-aware: match per class
        classes = set(keep_names) | {n for n, _ in gt_items}
        tp_ca_img = 0
        for cname in classes:
            pb = np.array([keep_boxes[k] for k in range(len(keep_names)) if keep_names[k] == cname]) if keep_names else np.zeros((0, 4))
            ps = np.array([keep_scores[k] for k in range(len(keep_names)) if keep_names[k] == cname]) if keep_names else np.zeros(0)
            gb = np.array([b for n, b in gt_items if n == cname]) if gt_items else np.zeros((0, 4))
            tp, _ = match(pb, ps, gb)
            tp_ca_img += tp
        ca["tp"] += tp_ca_img
        ca["fp"] += len(keep_boxes) - tp_ca_img

    def pr(d):
        p = d["tp"] / (d["tp"] + d["fp"]) if (d["tp"] + d["fp"]) else 0.0
        r = d["tp"] / d["gt"] if d["gt"] else 0.0
        f = 2 * p * r / (p + r) if (p + r) else 0.0
        return p, r, f

    return ca, cg, pr(ca), pr(cg)


def main():
    model, class_names = ControlModel.load_rfdetr_model("checkpoint_best_total.pth")
    thresholds = load_class_thresholds("/app/models/checkpoint_best_total.pth")
    print(f"{len(thresholds)} per-class thresholds loaded\n")

    for split in ["valid", "test"]:
        split_dir = f"/data/{split}"
        if not os.path.exists(split_dir):
            print(f"(skip {split}: not found)"); continue
        ca, cg, (pa, ra, fa), (pg, rg, fg) = evaluate(split_dir, model, class_names, thresholds)
        print(f"=== {split.upper()} ({ca['gt']} ground-truth boxes) ===")
        print(f"  Class-aware (right box + right SKU):  P={pa:.2f} R={ra:.2f} F1={fa:.2f}   (TP={ca['tp']} FP={ca['fp']} missed={ca['gt']-ca['tp']})")
        print(f"  Box-only  (right box, any SKU):       P={pg:.2f} R={rg:.2f} F1={fg:.2f}   (TP={cg['tp']} FP={cg['fp']} missed={cg['gt']-cg['tp']})")
        print()


if __name__ == "__main__":
    main()
