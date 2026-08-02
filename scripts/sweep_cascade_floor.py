"""Sweep CASCADE_FLOOR and report recall/precision at each value.

CASCADE_FLOOR has been pinned at 0.15 since the recall-recovery work and was
never varied, even though it is the knob that governs exactly the recall/
precision trade that work was optimizing: it sets how far below a class's
threshold the cascade is allowed to reach and promote a detection.

The sweep is cheap because of a property of the pipeline: in
cascade.pipeline.verify_detection, cascade_floor appears in exactly one place,
an early `if detector_confidence < cascade_floor: AUTO_REJECT`. Every other
input to the decision (the confident/uncertain tier split, OCR, embedding
match, the GPT tiebreaker shortlist) is floor-independent. So

    decision(d, floor) = AUTO_REJECT           if conf(d) < floor
                       = decision(d, floor=0)  otherwise

One pass at the lowest floor therefore yields every higher floor by filtering,
instead of re-running the cascade (and re-paying for GPT calls) per value.
Shelf-tag detection is likewise per-image and floor-independent, so it runs
once per image and is reused across the whole sweep.

Usage: python sweep_cascade_floor.py [--splits valid,test] [--floors 0.05,0.1,...]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

DATA_ROOT = os.environ.get(
    "SWEEP_DATA_ROOT", "/home/ubuntu/Datasets/trader-joes/training-data/rf-detr-combined"
)
EXAMPLES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "label_studio_ml", "examples")
EXAMPLES = os.path.abspath(EXAMPLES)
MODEL_ROOT = os.path.join(EXAMPLES, "models")
os.environ.setdefault("MODEL_ROOT", MODEL_ROOT)
sys.path.insert(0, EXAMPLES)

from PIL import Image  # noqa: E402

from control_models.base import MODEL_SCORE_THRESHOLD, ControlModel, load_class_thresholds  # noqa: E402
from cascade.embedding_match import get_backbone_nn_module, load_reference_gallery  # noqa: E402
from cascade.ocr import load_expected_text  # noqa: E402
from cascade.pipeline import Decision, verify_detection  # noqa: E402
from cascade.shelf_tags import detect_tags, load_tag_class_map, lookup_class  # noqa: E402

# Same loose center-match tolerance the earlier evals used, so numbers are
# comparable to the ones recorded in the commit history.
CT = 0.06


def score(kept, gt):
    """Recall over GT objects, precision over kept boxes, loose center match."""
    hit_gt = sum(
        1
        for gname, gc in gt
        if any(n == gname and abs(gc[0] - c[0]) < CT and abs(gc[1] - c[1]) < CT for n, c in kept)
    )
    tp_boxes = sum(
        1
        for n, c in kept
        if any(gn == n and abs(gc[0] - c[0]) < CT and abs(gc[1] - c[1]) < CT for gn, gc in gt)
    )
    return hit_gt, tp_boxes, len(kept)


def collect(split, model, class_names, thr, et, gal, tagm, nn, min_floor, stats):
    """One pass per image at the lowest floor; returns per-image records that
    every floor in the sweep can be derived from."""
    d = os.path.join(DATA_ROOT, split)
    coco = json.load(open(os.path.join(d, "_annotations.coco.json")))
    cat = {c["id"]: c["name"] for c in coco["categories"]}
    imgf = {im["id"]: im["file_name"] for im in coco["images"]}
    gt = defaultdict(list)
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        gt[a["image_id"]].append((cat[a["category_id"]], (x + w / 2, y + h / 2)))

    records = []
    for img_id, fn in imgf.items():
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            continue
        img = Image.open(p).convert("RGB")
        W, H = img.size
        gtn = [(n, (c[0] / W, c[1] / H)) for n, c in gt.get(img_id, [])]

        det = model.predict(img, threshold=min_floor)
        tags = detect_tags(img)  # floor-independent: one vision call per image
        stats["tag_calls"] += 1

        dets = []
        for i in range(len(det.xyxy)):
            cid = int(det.class_id[i])
            if cid >= len(class_names):
                continue
            nm = class_names[cid]
            sc = float(det.confidence[i])
            x1, y1, x2, y2 = det.xyxy[i].tolist()
            c = ((x1 + x2) / 2 / W, (y1 + y2) / 2 / H)
            eff = thr.get(nm, MODEL_SCORE_THRESHOLD)
            # cascade_floor=0 so the decision carries no floor filtering; the
            # sweep applies each floor itself.
            dec = verify_detection(
                crop=img.crop((x1, y1, x2, y2)),
                class_name=nm,
                detector_confidence=sc,
                effective_threshold=eff,
                expected_text=et,
                reference_gallery=gal,
                nn_model=nn,
                cascade_floor=0.0,
            )
            stats["verify_calls"] += 1
            # SKU correction from the shelf tag in this box's column
            near = [t for t in tags if abs(t["x"] - c[0]) < 0.09 and 0 < (t["y"] - c[1]) < 0.13]
            corrected = nm
            if near:
                tcls = lookup_class(min(near, key=lambda t: t["y"] - c[1])["name"], tagm)
                if tcls:
                    corrected = tcls
            dets.append({"name": nm, "corrected": corrected, "conf": sc, "eff": eff, "c": c, "dec": dec})
        records.append({"gt": gtn, "dets": dets})
    return records


def evaluate(records, floor):
    """Derive metrics at `floor` from the single-pass records."""
    agg = {m: {"hit": 0, "tp": 0, "kept": 0} for m in ("rf", "casc", "full")}
    ngt = 0
    for rec in records:
        gtn = rec["gt"]
        ngt += len(gtn)
        rf, casc, full = [], [], []
        for d in rec["dets"]:
            if d["conf"] >= d["eff"]:
                rf.append((d["name"], d["c"]))
            # the only place the floor enters the cascade's decision
            if d["conf"] >= floor and d["dec"] != Decision.AUTO_REJECT:
                casc.append((d["name"], d["c"]))
                full.append((d["corrected"], d["c"]))
        for key, kept in (("rf", rf), ("casc", casc), ("full", full)):
            hit, tp, n = score(kept, gtn)
            agg[key]["hit"] += hit
            agg[key]["tp"] += tp
            agg[key]["kept"] += n
    return agg, ngt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", default="valid,test")
    ap.add_argument("--floors", default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    args = ap.parse_args()
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    floors = [float(f) for f in args.floors.split(",") if f.strip()]
    min_floor = min(floors)

    model, class_names = ControlModel.load_rfdetr_model("checkpoint_best_total.pth")
    thr = load_class_thresholds(os.path.join(MODEL_ROOT, "checkpoint_best_total.pth"))
    et = load_expected_text(os.path.join(MODEL_ROOT, "ocr_expected_text.json"))
    gal = load_reference_gallery(os.path.join(MODEL_ROOT, "reference_gallery.npz"))
    tagm = load_tag_class_map(os.path.join(MODEL_ROOT, "tag_class_map.json"))
    nn = get_backbone_nn_module(model)

    stats = {"tag_calls": 0, "verify_calls": 0}
    for split in splits:
        if not os.path.exists(os.path.join(DATA_ROOT, split)):
            print(f"(skipping {split}: not found)")
            continue
        records = collect(split, model, class_names, thr, et, gal, tagm, nn, min_floor, stats)
        _, ngt = evaluate(records, min_floor)
        nimg = len(records)
        print(f"\n=== {split.upper()} — {nimg} images, {ngt} GT objects, loose center match (CT={CT}) ===")
        print(f"{'floor':>6}  {'cascade R':>9} {'cascade P':>9}  {'+tag R':>7} {'+tag P':>7}  {'kept':>5}")
        for floor in floors:
            agg, ngt = evaluate(records, floor)
            c, f = agg["casc"], agg["full"]
            print(
                f"{floor:>6.2f}  {c['hit']/max(ngt,1):>9.3f} {c['tp']/max(c['kept'],1):>9.3f}"
                f"  {f['hit']/max(ngt,1):>7.3f} {f['tp']/max(f['kept'],1):>7.3f}  {f['kept']:>5}"
            )
        rf = evaluate(records, min_floor)[0]["rf"]
        print(
            f"  baseline RF-DETR+thresholds (floor-independent): "
            f"R={rf['hit']/max(ngt,1):.3f} P={rf['tp']/max(rf['kept'],1):.3f} (kept {rf['kept']})"
        )
    print(f"\nAPI usage: {stats['tag_calls']} tag-reading calls, {stats['verify_calls']} verify_detection calls")


if __name__ == "__main__":
    main()
