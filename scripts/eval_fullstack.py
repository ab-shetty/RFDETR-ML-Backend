"""Full-system recall: RF-DETR alone vs +cascade vs +cascade+shelf-tags.
Loose center-match on (class, location) — the common metric fair to the
shelf-tag proposer's approximate boxes (RF-DETR/cascade boxes are tight and
also pass). Precision reported as (kept boxes that hit a GT) / (kept boxes).
"""
import json, os, sys
from collections import defaultdict
import numpy as np

os.environ.setdefault("MODEL_ROOT", "/app/models")
sys.path.insert(0, "/app")
from control_models.base import load_class_thresholds, MODEL_SCORE_THRESHOLD, CASCADE_FLOOR, ControlModel
from cascade.embedding_match import get_backbone_nn_module, load_reference_gallery
from cascade.ocr import load_expected_text
from cascade.pipeline import Decision, verify_detection
from cascade.shelf_tags import load_tag_class_map, propose_from_tags
from PIL import Image

CT = 0.06

def recall_prec(kept, gt):
    hit_gt = 0
    for gname, gc in gt:
        if any(n == gname and abs(gc[0]-c[0])<CT and abs(gc[1]-c[1])<CT for n, c in kept):
            hit_gt += 1
    tp_boxes = sum(1 for n, c in kept if any(gn==n and abs(gc[0]-c[0])<CT and abs(gc[1]-c[1])<CT for gn, gc in gt))
    r = hit_gt/len(gt) if gt else 0
    p = tp_boxes/len(kept) if kept else 0
    return r, p, hit_gt

def main():
    model, cn = ControlModel.load_rfdetr_model("checkpoint_best_total.pth")
    thr = load_class_thresholds("/app/models/checkpoint_best_total.pth")
    et = load_expected_text("/app/models/ocr_expected_text.json")
    gal = load_reference_gallery("/app/models/reference_gallery.npz")
    tagm = load_tag_class_map("/app/models/tag_class_map.json")
    nn = get_backbone_nn_module(model)

    for split in ["valid", "test"]:
        d=f"/data/{split}"
        if not os.path.exists(d): continue
        coco=json.load(open(f"{d}/_annotations.coco.json"))
        cat={c["id"]:c["name"] for c in coco["categories"]}; imgf={im["id"]:im["file_name"] for im in coco["images"]}
        gt=defaultdict(list)
        for a in coco["annotations"]:
            x,y,w,h=a["bbox"]; gt[a["image_id"]].append((cat[a["category_id"]],((x+w/2),(y+h/2))))
        agg={m:{"r":0,"tpb":0,"kept":0} for m in ("rf","casc","full")}; ngt=0
        for img_id,fn in imgf.items():
            p=f"{d}/{fn}"
            if not os.path.exists(p): continue
            img=Image.open(p).convert("RGB"); W,H=img.size
            det=model.predict(img, threshold=CASCADE_FLOOR)
            gtn=[(n,(c[0]/W,c[1]/H)) for n,c in gt.get(img_id,[])]; ngt+=len(gtn)
            rf, casc = [], []
            for i in range(len(det.xyxy)):
                cid=int(det.class_id[i])
                if cid>=len(cn): continue
                nm=cn[cid]; sc=float(det.confidence[i]); x1,y1,x2,y2=det.xyxy[i].tolist()
                c=((x1+x2)/2/W,(y1+y2)/2/H); eff=thr.get(nm,MODEL_SCORE_THRESHOLD)
                if sc>=eff: rf.append((nm,c))
                dec=verify_detection(crop=img.crop((x1,y1,x2,y2)),class_name=nm,detector_confidence=sc,
                    effective_threshold=eff,expected_text=et,reference_gallery=gal,nn_model=nn,cascade_floor=CASCADE_FLOOR)
                if dec!=Decision.AUTO_REJECT: casc.append((nm,c))
            # full = cascade + shelf-tag proposals for uncovered
            props=[(pr["class_name"],((pr["box"][0]+pr["box"][2])/2/W,(pr["box"][1]+pr["box"][3])/2/H))
                   for pr in propose_from_tags(img, tagm, [c for _,c in casc])]
            full=casc+props
            for key,kept in (("rf",rf),("casc",casc),("full",full)):
                r,pp,hg=recall_prec(kept,gtn); agg[key]["r"]+=hg
                agg[key]["tpb"]+=sum(1 for n,c in kept if any(gn==n and abs(gc[0]-c[0])<CT and abs(gc[1]-c[1])<CT for gn,gc in gtn))
                agg[key]["kept"]+=len(kept)
        print(f"=== {split.upper()} ({ngt} GT) — loose center match ===")
        for key,label in (("rf","RF-DETR + thresholds "),("casc","+ cascade           "),("full","+ cascade + shelf-tags")):
            a=agg[key]; print(f"  {label} recall={a['r']/ngt:.2f}  precision={a['tpb']/max(a['kept'],1):.2f}  (kept {a['kept']})")
        print()

if __name__=="__main__":
    main()
