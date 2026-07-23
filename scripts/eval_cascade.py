"""Compare threshold-only vs threshold+cascade pre-annotation accuracy on the
held-out splits. Shares RF-DETR inference between the two so the only
difference is the filtering policy.
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np

os.environ.setdefault("MODEL_ROOT", "/app/models")
sys.path.insert(0, "/app")

from control_models.base import load_class_thresholds, MODEL_SCORE_THRESHOLD, CASCADE_FLOOR, ControlModel
from cascade.embedding_match import get_backbone_nn_module, load_reference_gallery
from cascade.ocr import load_expected_text
from cascade.pipeline import Decision, verify_detection
from PIL import Image


def iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    a = a[:, None, :]; b = b[None, :, :]
    x1 = np.maximum(a[..., 0], b[..., 0]); y1 = np.maximum(a[..., 1], b[..., 1])
    x2 = np.minimum(a[..., 2], b[..., 2]); y2 = np.minimum(a[..., 3], b[..., 3])
    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
    aa = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    ab = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    u = aa + ab - inter
    return np.where(u > 0, inter / u, 0.0)


def match(pred_boxes, pred_scores, gt_boxes, iou_t=0.5):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return 0
    ious = iou_matrix(pred_boxes, gt_boxes)
    order = np.argsort(-pred_scores)
    used = np.zeros(len(gt_boxes), bool)
    tp = 0
    for i in order:
        j = int(np.argmax(ious[i]))
        if ious[i, j] >= iou_t and not used[j]:
            tp += 1; used[j] = True
    return tp


def score_kept(kept, gt_items):
    """kept: list of (name, box, score). Returns (ca, cg) tally dicts."""
    boxes = np.array([b for _, b, _ in kept]) if kept else np.zeros((0, 4))
    scores = np.array([s for _, _, s in kept]) if kept else np.zeros(0)
    names = [n for n, _, _ in kept]
    gt_all = np.array([b for _, b in gt_items]) if gt_items else np.zeros((0, 4))
    tp_cg = match(boxes, scores, gt_all)
    tp_ca = 0
    for cname in set(names) | {n for n, _ in gt_items}:
        pb = np.array([boxes[k] for k in range(len(names)) if names[k] == cname]) if names else np.zeros((0, 4))
        ps = np.array([scores[k] for k in range(len(names)) if names[k] == cname]) if names else np.zeros(0)
        gb = np.array([b for n, b in gt_items if n == cname]) if gt_items else np.zeros((0, 4))
        tp_ca += match(pb, ps, gb)
    return (
        {"tp": tp_ca, "fp": len(kept) - tp_ca},
        {"tp": tp_cg, "fp": len(kept) - tp_cg},
    )


def pr(tp, fp, gt):
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / gt if gt else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    model, class_names = ControlModel.load_rfdetr_model("checkpoint_best_total.pth")
    thresholds = load_class_thresholds("/app/models/checkpoint_best_total.pth")
    expected_text = load_expected_text("/app/models/ocr_expected_text.json")
    gallery = load_reference_gallery("/app/models/reference_gallery.npz")
    nn_model = get_backbone_nn_module(model)
    print(f"cascade floor={CASCADE_FLOOR}, gallery={len(gallery)} classes, ocr={sum(1 for v in expected_text.values() if v)} classes with text\n")

    for split in ["valid", "test"]:
        d = f"/data/{split}"
        if not os.path.exists(d):
            continue
        coco = json.load(open(os.path.join(d, "_annotations.coco.json")))
        cat = {c["id"]: c["name"] for c in coco["categories"]}
        img_file = {im["id"]: im["file_name"] for im in coco["images"]}
        gt = defaultdict(list)
        for a in coco["annotations"]:
            x, y, w, h = a["bbox"]
            gt[a["image_id"]].append((cat[a["category_id"]], [x, y, x + w, y + h]))
        n_gt = sum(len(v) for v in gt.values())

        agg = {m: {"ca": defaultdict(int), "cg": defaultdict(int)} for m in ("thresh", "cascade")}
        gpt_calls = 0
        for img_id, fname in img_file.items():
            p = os.path.join(d, fname)
            if not os.path.exists(p):
                continue
            image = Image.open(p).convert("RGB")
            det = model.predict(image, threshold=min(CASCADE_FLOOR, 0.15))
            gt_items = gt.get(img_id, [])

            thresh_kept, cascade_kept = [], []
            for i in range(len(det.xyxy)):
                cid = int(det.class_id[i])
                if cid >= len(class_names):
                    continue
                name = class_names[cid]
                score = float(det.confidence[i])
                box = det.xyxy[i]
                eff = thresholds.get(name, MODEL_SCORE_THRESHOLD)
                # threshold-only
                if score >= eff:
                    thresh_kept.append((name, box, score))
                # cascade
                x1, y1, x2, y2 = box.tolist()
                decision = verify_detection(
                    crop=image.crop((x1, y1, x2, y2)), class_name=name,
                    detector_confidence=score, effective_threshold=eff,
                    expected_text=expected_text, reference_gallery=gallery,
                    nn_model=nn_model, cascade_floor=CASCADE_FLOOR,
                )
                if decision != Decision.AUTO_REJECT:
                    cascade_kept.append((name, box, score))

            for mode, kept in (("thresh", thresh_kept), ("cascade", cascade_kept)):
                ca, cg = score_kept(kept, gt_items)
                for k in ("tp", "fp"):
                    agg[mode]["ca"][k] += ca[k]
                    agg[mode]["cg"][k] += cg[k]

        print(f"=== {split.upper()} ({n_gt} ground-truth boxes) ===")
        for mode, label in (("thresh", "threshold-only "), ("cascade", "threshold+cascade")):
            pa, ra, fa = pr(agg[mode]["ca"]["tp"], agg[mode]["ca"]["fp"], n_gt)
            pg, rg, fg = pr(agg[mode]["cg"]["tp"], agg[mode]["cg"]["fp"], n_gt)
            print(f"  [{label}] class-aware P={pa:.2f} R={ra:.2f} F1={fa:.2f}  |  box-only P={pg:.2f} R={rg:.2f} F1={fg:.2f}")
        print()


if __name__ == "__main__":
    main()
