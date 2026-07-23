#!/usr/bin/env python3
"""Prototype: generate candidate training labels for classes RF-DETR has
never seen ("blind" classes — see find_blind_classes.py), so accuracy on
them can improve without waiting on a manual labeling pass.

The cascade/shelf-tag pipeline already reads shelf tags and maps them to
classes, but tag_class_map.json is *learned* from labeled boxes — a blind
class has none, so it can never appear in that map (see shelf_tags.py's
docstring). This script closes that gap for JUST the discovery step: it
fuzzy-matches freshly-read tag text against the blind classes' catalog names
(weaker signal, since catalog names are more verbose/different from tag
wording than the learned map's keys are) to build a short candidate list per
tag, proposes the product slot above the tag as a candidate box (same
geometry as propose_from_tags), and spends one GPT-5-mini call per tag on a
NARROW, shortlist-constrained verification: "which of these <=5 candidates,
if any, does this crop show?" — the same constrained-role pattern that works
well elsewhere in the cascade (see gpt_tiebreaker.py), not the open-ended
multi-box review that (see exp_gpt_review.py) measurably hurt precision and
recall. The shortlist (not a single best-fuzzy-match) matters: confusable
sibling SKUs both match the same tag text well, and only showing GPT one of
them means it can't rule the sibling out -- an early single-candidate
version of this script silently mislabeled a Vanilla Cold Brew crop as its
plain "Cold Brew Coffee Concentrate" sibling for exactly this reason.

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
import csv
import glob
import json
import os
import sys

sys.path.insert(0, "/app")
from cascade.shelf_tags import detect_tags, normalize_tag, _slot_box, SLOT_TOP_OFFSET, SLOT_BOTTOM_OFFSET  # noqa: E402
from PIL import Image  # noqa: E402
from rapidfuzz import fuzz, process  # noqa: E402

FUZZY_MIN_SCORE = 65   # lower than the learned tag_class_map's 82: catalog
                        # names ("Sparkling Tea Violette") drift further from
                        # tag wording than the learned map's own keys do.
SHORTLIST_SIZE = 5      # how many blind classes to offer GPT as candidates


def load_all_class_names(master_list_path):
    names = []
    with open(master_list_path) as f:
        for row in csv.DictReader(f):
            name = row["Class Name (str)"].strip()
            if name:
                names.append(name)
    return names


def shortlist_blind_classes(tag_name, blind_names_upper, blind_names, blind_set, all_names_upper, all_names):
    """blind_names_upper/blind_names are parallel lists -- match against the
    uppercased forms (normalize_tag() uppercases tag text, and rapidfuzz's
    token_sort_ratio is case-sensitive without an explicit processor, so
    matching against the original Title Case names silently tanks every
    score). Returns up to SHORTLIST_SIZE original-cased class names, best
    first, above FUZZY_MIN_SCORE.

    A single best-match is not enough: confusable sibling SKUs ("Cold Brew
    Coffee Concentrate" vs "Vanilla Cold Brew Coffee Concentrate") both
    fuzzy-match the same tag text well, and a yes/no check against only the
    top match can't tell GPT the sibling exists to rule out -- verified
    empirically (see the commit history) where this silently mislabeled a
    Vanilla Cold Brew crop as its plain sibling.

    Just as important, and found the same way (full-run spot check): a tag
    can fuzzy-match a blind class reasonably well while actually naming a
    DIFFERENT, non-blind class RF-DETR already knows -- "PUMPKIN SPICE
    FLAVORED COLD BREW COFFEE CONCENTRATE" shares enough tokens with "Cold
    Brew Coffee Concentrate"/"Vanilla Cold Brew Coffee Concentrate" (both
    blind) to clear FUZZY_MIN_SCORE, even though "Pumpkin Spice Flavored
    Cold Brew Coffee" is itself a real, already-labeled (non-blind) catalog
    class and the true best match by a wide margin. Shown only the blind
    shortlist, GPT confirmed one of the wrong siblings with plausible-
    sounding reasoning instead of correctly matching neither. Fix: find the
    single best match across ALL 179 classes first; if that global-best
    isn't blind, RF-DETR already knows this product -- return [] rather
    than spend a verification call bidding it into the wrong blind bucket.
    """
    key = normalize_tag(tag_name)
    if not key:
        return []
    global_best = process.extractOne(key, all_names_upper, scorer=fuzz.token_sort_ratio)
    if not global_best or global_best[1] < FUZZY_MIN_SCORE:
        return []
    if all_names[global_best[2]] not in blind_set:
        return []
    matches = process.extract(key, blind_names_upper, scorer=fuzz.token_sort_ratio, limit=SHORTLIST_SIZE)
    return [(blind_names[m[2]], m[1]) for m in matches if m[1] >= FUZZY_MIN_SCORE]


def verify_shortlist(crop, candidates):
    """Ask GPT-5-mini to pick which (if any) of the shortlisted candidates
    the crop shows -- forces discrimination between confusable siblings
    instead of a yes/no on one class in isolation (same constrained-role
    pattern as gpt_tiebreaker.ask(), extended to also return a reason for
    the audit trail in the manifest)."""
    import base64
    import io

    from openai import OpenAI

    buf = io.BytesIO()
    crop.convert("RGB").save(buf, format="JPEG", quality=90)
    b64 = base64.b64encode(buf.getvalue()).decode()
    prompt = (
        "This crop is from a grocery-shelf photo. Which of these candidate product "
        f"labels, if any, matches what's actually shown (packaging, shape, color)? "
        f"Candidates: {json.dumps(candidates)}. If more than one candidate is plausible, "
        "pick the one whose visible text/label most specifically matches (e.g. prefer a "
        "flavored variant over its plain sibling if the flavor is visible). "
        'JSON only: {"match": "<exact candidate string>" or null, "reason": "<=10 words"}'
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
        match = parsed.get("match")
        reason = str(parsed.get("reason", ""))
        if match in candidates:
            return match, reason
        if match is not None:
            reason = f"out-of-candidate match {match!r} ignored; {reason}"
        return None, reason
    except Exception as e:
        return None, f"gpt call failed: {e}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blind-classes", default="/data/blind_classes.json")
    ap.add_argument("--master-list", default="/data/master_list.csv", help="full 179-class catalog, for the not-actually-blind guard")
    ap.add_argument("--frames-dir", default="/data/frames")
    ap.add_argument("--out-dir", default="/data/bootstrap")
    ap.add_argument("--max-frames", type=int, default=60)
    args = ap.parse_args()

    blind = json.load(open(args.blind_classes))
    blind_names = [b["class_name"] for b in blind]
    blind_names_upper = [normalize_tag(n) for n in blind_names]
    blind_set = set(blind_names)
    print(f"{len(blind_names)} blind classes to search for")

    all_names = load_all_class_names(args.master_list)
    all_names_upper = [normalize_tag(n) for n in all_names]
    print(f"{len(all_names)} total catalog classes loaded for the not-actually-blind guard")

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
            shortlist = shortlist_blind_classes(tag["name"], blind_names_upper, blind_names, blind_set, all_names_upper, all_names)
            if not shortlist:
                continue
            n_matched += 1
            tx, ty = float(tag["x"]), float(tag["y"])
            box = _slot_box(tx, ty, width, height)
            x1, y1, x2, y2 = box
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            crop = image.crop((x1, y1, x2, y2))
            cand_names = [c for c, _ in shortlist]
            match, reason = verify_shortlist(crop, cand_names)
            status = f"confirmed:{match}" if match else "rejected"
            print(f"  [{fi+1}/{len(frames)}] {stem} tag={tag['name']!r} shortlist={cand_names} gpt={status} ({reason})")
            if not match:
                continue
            crop_name = f"{match.replace('/', '_')}__{stem}__{ti}.jpg"
            crop.save(os.path.join(crops_dir, crop_name))
            candidates.append({
                "frame": stem,
                "class_name": match,
                "box_xyxy": [x1, y1, x2, y2],
                "box_norm": [x1 / width, y1 / height, x2 / width, y2 / height],
                "tag_text": tag["name"],
                "shortlist": shortlist,
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
