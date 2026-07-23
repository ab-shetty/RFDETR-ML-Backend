#!/usr/bin/env python3
"""Prototype: generate candidate training labels for classes RF-DETR has
never seen ("blind" classes — see find_blind_classes.py), so accuracy on
them can improve without waiting on a manual labeling pass.

The cascade/shelf-tag pipeline already reads shelf tags and maps them to
classes, but tag_class_map.json is *learned* from labeled boxes — a blind
class has none, so it can never appear in that map (see shelf_tags.py's
docstring). This script closes that gap for JUST the discovery step: it
fuzzy-matches freshly-read tag text directly against the blind classes'
catalog names (weaker signal, since catalog names are more verbose/different
from tag wording than the learned map's keys are), proposes the product slot
above the tag as a candidate box (same geometry as propose_from_tags), and
then spends a GPT-5-mini call per candidate on a NARROW, single-candidate
yes/no verification: "is this crop actually <class_name>?" — the same
constrained-role pattern that works well elsewhere in the cascade (see
gpt_tiebreaker.py), not the open-ended multi-box review that (see
exp_gpt_review.py) measurably hurt precision and recall.

Output is NOT auto-merged into training data. It's a manifest + saved crops
for a human to spot-check in minutes rather than label from scratch — the
flywheel this whole system exists to accelerate, aimed specifically at the
classes plain threshold/cascade/shelf-tag-correction tuning cannot reach
(they all require an existing detection to grade; blind classes never
produce one).

Run inside the rfdetr container (needs openai/rapidfuzz/PIL, and cascade/ on
the path):
    docker cp scripts/bootstrap_zero_data_classes.py rfdetr:/app/
    docker cp /tmp/blind_classes.json rfdetr:/data/blind_classes.json
    docker exec -e OPENAI_API_KEY=$OPENAI_API_KEY rfdetr python3 \\
        /app/bootstrap_zero_data_classes.py \\
        --blind-classes /data/blind_classes.json \\
        --frames-dir /data/frames --out-dir /data/bootstrap --max-frames 60
"""
import argparse
import glob
import json
import os
import sys

sys.path.insert(0, "/app")
from cascade.shelf_tags import detect_tags, normalize_tag, _slot_box, SLOT_TOP_OFFSET, SLOT_BOTTOM_OFFSET  # noqa: E402
from PIL import Image  # noqa: E402
from rapidfuzz import fuzz, process  # noqa: E402

FUZZY_MIN_SCORE = 65  # lower than the learned tag_class_map's 82: catalog
                       # names ("Sparkling Tea Violette") drift further from
                       # tag wording than the learned map's own keys do.


def match_tag_to_blind_class(tag_name, blind_names_upper, blind_names):
    """blind_names_upper/blind_names are parallel lists -- match against the
    uppercased forms (normalize_tag() uppercases tag text, and rapidfuzz's
    token_sort_ratio is case-sensitive without an explicit processor, so
    matching against the original Title Case names silently tanks every
    score) and return the original-cased class name."""
    key = normalize_tag(tag_name)
    if not key:
        return None
    match = process.extractOne(key, blind_names_upper, scorer=fuzz.token_sort_ratio)
    if match and match[1] >= FUZZY_MIN_SCORE:
        return blind_names[match[2]], match[1]
    return None


def verify_candidate(crop, class_name):
    """Narrow, single-candidate check -- mirrors gpt_tiebreaker.ask()'s
    constrained-role pattern, deliberately not an open-ended review."""
    import base64
    import io

    from openai import OpenAI

    buf = io.BytesIO()
    crop.convert("RGB").save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    prompt = (
        f'Does this cropped grocery-shelf image show the product "{class_name}"? '
        "Judge only what's visible in the crop (packaging, shape, color -- not the "
        "shelf tag, which may be slightly misaligned with the product above it). "
        'JSON only: {"match": true|false, "reason": "<=10 words"}'
    )
    try:
        r = OpenAI().chat.completions.create(
            model="gpt-5-mini", max_completion_tokens=500, reasoning_effort="minimal",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ]}],
        )
        raw = (r.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].removeprefix("json").strip()
        parsed = json.loads(raw)
        return bool(parsed.get("match")), str(parsed.get("reason", ""))
    except Exception as e:
        return False, f"gpt call failed: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blind-classes", default="/data/blind_classes.json")
    ap.add_argument("--frames-dir", default="/data/frames")
    ap.add_argument("--out-dir", default="/data/bootstrap")
    ap.add_argument("--max-frames", type=int, default=60)
    args = ap.parse_args()

    blind = json.load(open(args.blind_classes))
    blind_names = [b["class_name"] for b in blind]
    blind_names_upper = [normalize_tag(n) for n in blind_names]
    print(f"{len(blind_names)} blind classes to search for")

    frames = sorted(glob.glob(os.path.join(args.frames_dir, "*.jpg")))[: args.max_frames]
    print(f"scanning {len(frames)} frames")

    os.makedirs(args.out_dir, exist_ok=True)
    crops_dir = os.path.join(args.out_dir, "review")
    os.makedirs(crops_dir, exist_ok=True)

    candidates = []
    n_tags_seen = 0
    n_matched = 0
    for fi, path in enumerate(frames):
        stem = os.path.splitext(os.path.basename(path))[0]
        image = Image.open(path).convert("RGB")
        width, height = image.size
        tags = detect_tags(image)
        n_tags_seen += len(tags)
        for ti, tag in enumerate(tags):
            m = match_tag_to_blind_class(tag["name"], blind_names_upper, blind_names)
            if not m:
                continue
            class_name, score = m
            n_matched += 1
            tx, ty = float(tag["x"]), float(tag["y"])
            box = _slot_box(tx, ty, width, height)
            x1, y1, x2, y2 = box
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crop = image.crop((x1, y1, x2, y2))
            confirmed, reason = verify_candidate(crop, class_name)
            status = "confirmed" if confirmed else "rejected"
            print(f"  [{fi+1}/{len(frames)}] {stem} tag={tag['name']!r} -> {class_name} (fuzzy={score}) gpt={status} ({reason})")
            if not confirmed:
                continue
            crop_name = f"{class_name.replace('/', '_')}__{stem}__{ti}.jpg"
            crop.save(os.path.join(crops_dir, crop_name))
            candidates.append({
                "frame": stem,
                "class_name": class_name,
                "box_xyxy": [x1, y1, x2, y2],
                "box_norm": [x1 / width, y1 / height, x2 / width, y2 / height],
                "tag_text": tag["name"],
                "fuzzy_score": score,
                "gpt_reason": reason,
                "crop_file": crop_name,
            })

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(candidates, f, indent=2)

    by_class = {}
    for c in candidates:
        by_class[c["class_name"]] = by_class.get(c["class_name"], 0) + 1

    print(f"\n{n_tags_seen} tags read, {n_matched} matched a blind class name, {len(candidates)} GPT-confirmed")
    print(f"manifest -> {manifest_path}")
    print(f"crops    -> {crops_dir}")
    print(f"\nconfirmed candidates by class ({len(by_class)} of {len(blind_names)} blind classes found):")
    for cls, n in sorted(by_class.items(), key=lambda kv: -kv[1]):
        print(f"  {n:2d}  {cls}")


if __name__ == "__main__":
    main()
