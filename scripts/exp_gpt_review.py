"""Experiment: does a holistic GPT-5-mini review pass improve the final set?

Baseline = cascade + shelf-tag SKU-correction. Then GPT sees the frame + the
current boxes (center + label) + the class vocabulary, and returns per-box
verdicts (keep / relabel / remove) plus suggested additions. We measure recall
& precision (loose center match) for baseline vs +GPT-review.

This tests using the model's cognition/scene-reasoning, not just as a per-box
tiebreaker.
"""
import base64, io, json, os, sys
from collections import defaultdict
import numpy as np

os.environ.setdefault("MODEL_ROOT", "/app/models")
sys.path.insert(0, "/app")
from control_models.base import load_class_thresholds, MODEL_SCORE_THRESHOLD, CASCADE_FLOOR, ControlModel
from cascade.embedding_match import get_backbone_nn_module, load_reference_gallery
from cascade.ocr import load_expected_text
from cascade.pipeline import Decision, verify_detection
from cascade.shelf_tags import load_tag_class_map, detect_tags, lookup_class
from PIL import Image
from openai import OpenAI

CT = 0.06
client = OpenAI()


def gpt_review(img, boxes, vocab):
    """boxes: [{'i':int,'x':float,'y':float,'label':str}]. Returns dict with
    'verdicts':[{'i','action','label'}] and 'add':[{'x','y','label'}]."""
    buf = io.BytesIO(); img.convert("RGB").save(buf, format="JPEG", quality=90)
    b = base64.b64encode(buf.getvalue()).decode()
    prompt = (
        "You are reviewing auto-generated product labels on a grocery cooler photo. "
        "Each detection has a center (x,y as 0-1 fractions) and a current product label.\n"
        f"Detections: {json.dumps(boxes)}\n\n"
        "Using what you actually see (products cluster by type; shelf tags name each slot), for EACH "
        "detection decide: keep (label correct), relabel (give the correct label), or remove (nothing "
        "there / duplicate / not a real product). Also list any PROMINENT products clearly visible with "
        "no detection, as additions with approx center.\n"
        "Only use labels from this list (exact strings):\n" + json.dumps(vocab) + "\n\n"
        'JSON only: {"verdicts":[{"i":0,"action":"keep|relabel|remove","label":"<if relabel>"}],'
        '"add":[{"x":0.0,"y":0.0,"label":"..."}]}'
    )
    try:
        r = client.chat.completions.create(model="gpt-5-mini", max_completion_tokens=3000,
            reasoning_effort="minimal",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b}"}}]}])
        raw = (r.choices[0].message.content or "").strip()
        if raw.startswith("```"): raw = raw.split("```")[1].removeprefix("json").strip()
        return json.loads(raw)
    except Exception as e:
        print("  gpt_review fail:", e); return {"verdicts": [], "add": []}


def apply_review(boxes, review, vocab_set):
    out = []
    verd = {v["i"]: v for v in review.get("verdicts", []) if "i" in v}
    for bx in boxes:
        v = verd.get(bx["i"], {"action": "keep"})
        if v["action"] == "remove":
            continue
        label = bx["label"]
        if v["action"] == "relabel" and v.get("label") in vocab_set:
            label = v["label"]
        out.append((label, (bx["x"], bx["y"])))
    for a in review.get("add", []):
        if a.get("label") in vocab_set and "x" in a and "y" in a:
            out.append((a["label"], (float(a["x"]), float(a["y"]))))
    return out


def rp(kept, gt):
    hit = sum(1 for gn, gc in gt if any(n == gn and abs(gc[0]-c[0])<CT and abs(gc[1]-c[1])<CT for n, c in kept))
    tpb = sum(1 for n, c in kept if any(gn == n and abs(gc[0]-c[0])<CT and abs(gc[1]-c[1])<CT for gn, gc in gt))
    return hit, tpb, len(kept)


def main():
    model, cn = ControlModel.load_rfdetr_model("checkpoint_best_total.pth")
    thr = load_class_thresholds("/app/models/checkpoint_best_total.pth")
    et = load_expected_text("/app/models/ocr_expected_text.json")
    gal = load_reference_gallery("/app/models/reference_gallery.npz")
    tagm = load_tag_class_map("/app/models/tag_class_map.json")
    nn = get_backbone_nn_module(model)
    vocab = list(dict.fromkeys(cn)); vocab_set = set(vocab)

    for split in ["valid", "test"]:
        d = f"/data/{split}"
        if not os.path.exists(d): continue
        coco = json.load(open(f"{d}/_annotations.coco.json"))
        cat = {c["id"]: c["name"] for c in coco["categories"]}; imgf = {im["id"]: im["file_name"] for im in coco["images"]}
        gt = defaultdict(list)
        for a in coco["annotations"]:
            x, y, w, h = a["bbox"]; gt[a["image_id"]].append((cat[a["category_id"]], ((x+w/2), (y+h/2))))
        agg = {m: [0, 0, 0] for m in ("base", "gpt")}; ngt = 0
        for img_id, fn in imgf.items():
            p = f"{d}/{fn}"
            if not os.path.exists(p): continue
            img = Image.open(p).convert("RGB"); W, H = img.size
            det = model.predict(img, threshold=CASCADE_FLOOR)
            gtn = [(n, (c[0]/W, c[1]/H)) for n, c in gt.get(img_id, [])]; ngt += len(gtn)
            casc = []
            for i in range(len(det.xyxy)):
                cid = int(det.class_id[i])
                if cid >= len(cn): continue
                nm = cn[cid]; sc = float(det.confidence[i]); x1, y1, x2, y2 = det.xyxy[i].tolist()
                c = ((x1+x2)/2/W, (y1+y2)/2/H); eff = thr.get(nm, MODEL_SCORE_THRESHOLD)
                dec = verify_detection(crop=img.crop((x1, y1, x2, y2)), class_name=nm, detector_confidence=sc,
                    effective_threshold=eff, expected_text=et, reference_gallery=gal, nn_model=nn, cascade_floor=CASCADE_FLOOR)
                if dec != Decision.AUTO_REJECT: casc.append((nm, c))
            # tag-correct
            tags = detect_tags(img); base = []
            for nm, c in casc:
                near = [t for t in tags if abs(t["x"]-c[0]) < 0.09 and 0 < (t["y"]-c[1]) < 0.13]
                if near:
                    tc = lookup_class(min(near, key=lambda t: t["y"]-c[1])["name"], tagm)
                    if tc: nm = tc
                base.append((nm, c))
            # GPT review on top
            boxes = [{"i": i, "x": round(c[0], 3), "y": round(c[1], 3), "label": nm} for i, (nm, c) in enumerate(base)]
            gpt = apply_review(boxes, gpt_review(img, boxes, vocab), vocab_set)
            for k, kept in (("base", base), ("gpt", gpt)):
                h, t, n = rp(kept, gtn); agg[k][0] += h; agg[k][1] += t; agg[k][2] += n
        print(f"=== {split.upper()} ({ngt} GT) ===")
        for k, lab in (("base", "cascade+tag-correct "), ("gpt", "+ GPT review pass   ")):
            h, t, n = agg[k]; print(f"  {lab} recall={h/ngt:.2f} precision={t/max(n,1):.2f} (kept {n})")
        print()


if __name__ == "__main__":
    main()
