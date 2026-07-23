"""Measure the shelf-tag proposer's recall lift on held-out frames.

Proposed boxes are deliberately approximate (a slot the human nudges), so we
score with a loose center-distance match on (class, location) rather than
strict IoU — the pre-annotation value is 'right product, roughly right place',
which is what saves the human the SKU search. Reports recall for RF-DETR+
threshold alone vs. + shelf-tag proposals.
"""
import json, os, sys
from collections import defaultdict
import numpy as np

os.environ.setdefault("MODEL_ROOT", "/app/models")
sys.path.insert(0, "/app")
from control_models.base import load_class_thresholds, MODEL_SCORE_THRESHOLD, ControlModel
from cascade.shelf_tags import load_tag_class_map, propose_from_tags
from PIL import Image

CENTER_TOL = 0.06  # normalized center distance for a "match"


def centers_match(a, b):
    return abs(a[0] - b[0]) < CENTER_TOL and abs(a[1] - b[1]) < CENTER_TOL


def main():
    model, class_names = ControlModel.load_rfdetr_model("checkpoint_best_total.pth")
    thr = load_class_thresholds("/app/models/checkpoint_best_total.pth")
    tag_map = load_tag_class_map("/app/models/tag_class_map.json")
    print(f"tag map: {len(tag_map)} tags\n")

    for split in ["valid", "test"]:
        d = f"/data/{split}"
        if not os.path.exists(d):
            continue
        coco = json.load(open(f"{d}/_annotations.coco.json"))
        cat = {c["id"]: c["name"] for c in coco["categories"]}
        imgf = {im["id"]: im["file_name"] for im in coco["images"]}
        gt = defaultdict(list)
        for a in coco["annotations"]:
            x, y, w, h = a["bbox"]
            gt[a["image_id"]].append((cat[a["category_id"]], ((x + w / 2), (y + h / 2))))

        n_gt = rf_hit = tag_hit = both_hit = 0
        for img_id, fn in imgf.items():
            p = f"{d}/{fn}"
            if not os.path.exists(p):
                continue
            img = Image.open(p).convert("RGB"); W, H = img.size
            det = model.predict(img, threshold=0.15)
            # RF-DETR kept (class, normalized center)
            rf = []
            for i in range(len(det.xyxy)):
                cid = int(det.class_id[i])
                if cid >= len(class_names):
                    continue
                name = class_names[cid]
                if float(det.confidence[i]) >= thr.get(name, MODEL_SCORE_THRESHOLD):
                    x1, y1, x2, y2 = det.xyxy[i].tolist()
                    rf.append((name, ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H)))
            # shelf-tag proposals (class, normalized center)
            covered = [c for _, c in rf]
            props = []
            for pr in propose_from_tags(img, tag_map, covered):
                bx = pr["box"]; props.append((pr["class_name"], ((bx[0] + bx[2]) / 2 / W, (bx[1] + bx[3]) / 2 / H)))

            for gname, (gx, gy) in gt.get(img_id, []):
                gc = (gx / W, gy / H)
                n_gt += 1
                r = any(n == gname and centers_match(gc, c) for n, c in rf)
                t = any(n == gname and centers_match(gc, c) for n, c in props)
                rf_hit += r
                tag_hit += t
                both_hit += (r or t)

        print(f"=== {split.upper()} ({n_gt} GT boxes, loose center match) ===")
        print(f"  RF-DETR alone           recall = {rf_hit/n_gt:.2f}  ({rf_hit}/{n_gt})")
        print(f"  RF-DETR + shelf-tags    recall = {both_hit/n_gt:.2f}  ({both_hit}/{n_gt})")
        print(f"  (recovered by tags that RF-DETR missed: {both_hit-rf_hit})\n")


if __name__ == "__main__":
    main()
