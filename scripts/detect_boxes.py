#!/usr/bin/env python3
"""Cache RF-DETR's boxes for a folder of frames, so proposals can be SHAPED by
the detector while still being NAMED by the shelf tag.

The two signals are good at different things and were being asked to do each
other's job. A shelf tag tells you what a product is -- the store printed the
name on it -- but its position says nothing about how tall or wide the product
is, so tag-derived boxes were a constant 12%x9% rectangle at a fixed offset,
landing on price cards as often as on bottles. The detector is the reverse: it
puts a tight box around a product but, on a 179-class taxonomy where 65 classes
have never been seen, frequently cannot name it.

So: geometry from here, identity from the tag index.

Detections are kept class-agnostically and at a low score floor on purpose. We
are not using the predicted class at all -- only "there is an object here" --
and a box that is confidently an object but weakly a *particular* SKU is
exactly the case the tag resolves. Filtering by class confidence first would
throw away the boxes we most need.
"""
import argparse
import json
import os
import sys

MODEL_KWARGS = dict(          # must match control_models/base.py:load_rfdetr_model
    num_classes=180,
    resolution=384,
    patch_size=16,
    positional_encoding_size=24,
    num_windows=2,
    dec_layers=2,
    out_feature_indexes=[3, 6, 9, 12],
)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--checkpoint", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "label_studio_ml", "examples", "models", "checkpoint_best_total.pth"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--threshold", type=float, default=0.10,
                    help="low on purpose: we want boxes, not confident classes")
    ap.add_argument("--limit", type=int)
    args = ap.parse_args()

    from PIL import Image
    from rfdetr import RFDETRBase

    print(f"loading {args.checkpoint}")
    model = RFDETRBase(pretrain_weights=args.checkpoint, **MODEL_KWARGS)

    frames = sorted(f for f in os.listdir(args.frames_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if args.limit:
        frames = frames[: args.limit]

    # Resume: this is minutes of GPU-less inference, so a crash partway should
    # not mean starting over.
    out = {}
    if os.path.exists(args.out):
        out = json.load(open(args.out)).get("frames", {})
        print(f"{len(out)} frames already detected; skipping those")

    todo = [f for f in frames if os.path.splitext(f)[0] not in out]
    print(f"{len(frames)} frames, {len(todo)} to run")
    for n, name in enumerate(todo, 1):
        stem = os.path.splitext(name)[0]
        with Image.open(os.path.join(args.frames_dir, name)) as im:
            im = im.convert("RGB")
            W, H = im.size
            det = model.predict(im, threshold=args.threshold)
        boxes = []
        for xyxy, score, cid in zip(det.xyxy, det.confidence, det.class_id):
            x0, y0, x1, y1 = (float(v) for v in xyxy)
            boxes.append({"box": [x0 / W, y0 / H, x1 / W, y1 / H],
                          "score": round(float(score), 4), "class_id": int(cid)})
        out[stem] = {"boxes": boxes, "size": [W, H]}
        if n % 20 == 0 or n == len(todo):
            json.dump({"frames": out}, open(args.out, "w"))
            print(f"  {n}/{len(todo)}")
    json.dump({"frames": out}, open(args.out, "w"))
    total = sum(len(v["boxes"]) for v in out.values())
    print(f"\n{len(out)} frames, {total} boxes "
          f"({total / max(1, len(out)):.1f} per frame) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
