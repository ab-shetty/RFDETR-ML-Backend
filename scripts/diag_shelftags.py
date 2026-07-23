"""Diagnose why RF-DETR-missed products are/aren't recovered by shelf tags.
For each GT box RF-DETR missed, categorize:
  recovered           : a mapped tag in its column gives the right class
  tag_wrong_class     : a tag is there but maps to a different class
  tag_unmapped        : a tag is there but not in the learned map
  no_tag_nearby       : no tag detected below the product
"""
import json, os, sys
from collections import defaultdict, Counter
import numpy as np

os.environ.setdefault("MODEL_ROOT", "/app/models")
sys.path.insert(0, "/app")
from control_models.base import load_class_thresholds, MODEL_SCORE_THRESHOLD, ControlModel
from cascade.shelf_tags import load_tag_class_map, detect_tags, lookup_class, normalize_tag
from PIL import Image

CT = 0.06

def main():
    model, class_names = ControlModel.load_rfdetr_model("checkpoint_best_total.pth")
    thr = load_class_thresholds("/app/models/checkpoint_best_total.pth")
    tag_map = load_tag_class_map("/app/models/tag_class_map.json")

    cats = Counter()
    for split in ["valid", "test"]:
        d = f"/data/{split}"
        if not os.path.exists(d): continue
        coco = json.load(open(f"{d}/_annotations.coco.json"))
        cat = {c["id"]: c["name"] for c in coco["categories"]}
        imgf = {im["id"]: im["file_name"] for im in coco["images"]}
        gt = defaultdict(list)
        for a in coco["annotations"]:
            x,y,w,h=a["bbox"]; gt[a["image_id"]].append((cat[a["category_id"]],((x+w/2),(y+h/2))))
        for img_id, fn in imgf.items():
            p=f"{d}/{fn}"
            if not os.path.exists(p): continue
            img=Image.open(p).convert("RGB"); W,H=img.size
            det=model.predict(img, threshold=0.15)
            rf=[]
            for i in range(len(det.xyxy)):
                cid=int(det.class_id[i])
                if cid>=len(class_names): continue
                nm=class_names[cid]
                if float(det.confidence[i])>=thr.get(nm,MODEL_SCORE_THRESHOLD):
                    x1,y1,x2,y2=det.xyxy[i].tolist(); rf.append((nm,((x1+x2)/2/W,(y1+y2)/2/H)))
            tags=detect_tags(img)  # [{name,x,y}]
            for gname,(gx,gy) in gt.get(img_id,[]):
                gc=(gx/W, gy/H)
                if any(n==gname and abs(gc[0]-c[0])<CT and abs(gc[1]-c[1])<CT for n,c in rf):
                    continue  # RF-DETR got it
                # find a tag below this product (same column, tag y just below product)
                near=[t for t in tags if abs(t["x"]-gc[0])<0.09 and 0 < (t["y"]-gc[1]) < 0.22]
                if not near:
                    cats["no_tag_nearby"]+=1; continue
                t=min(near,key=lambda t:t["y"]-gc[1])
                cls=lookup_class(t["name"], tag_map)
                if cls is None: cats["tag_unmapped"]+=1
                elif cls==gname: cats["recovered"]+=1
                else: cats["tag_wrong_class"]+=1
    print("MISSED-PRODUCT DIAGNOSIS:", dict(cats))
    tot=sum(cats.values())
    if tot:
        for k,v in cats.most_common(): print(f"  {k:18s} {v:3d}  ({v/tot*100:.0f}%)")

if __name__=="__main__":
    main()
