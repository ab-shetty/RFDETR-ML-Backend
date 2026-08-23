#!/usr/bin/env python3
"""Can shelf tags name a store's products on their own?

The pre-annotation cascade names a box with RF-DETR's class head, which is
trained on one store visit and generalizes badly to the next. The shelf tag
under the product names it too, and the store prints a new tag whenever it
changes what is on the shelf -- so tags are the one naming signal that does not
go stale when we walk into a new store.

This first measured the learned `tag_class_map.json` -- built by pairing tags to
already-labelled boxes -- and found it resolved 50% of Laguna's tags with 25%
of those right, and 81% of Coppell's with 52% right. Both the map and the stage
that used it are gone; that column is kept only in this note, since the file it
needs no longer exists.

This measures the whole path against held-out human labels, splitting it into
the three places it can fail, because the fix is different for each:

  read     did the vision model see the tag at all
  pair     did the tag land under a product we labelled (geometry)
  resolve  did the tag text turn into a class name
  name     was that class name the right one

Reads are cached to disk, so re-running with a different resolver costs nothing
-- one paid pass over the frames, many analyses of it.

    python3 scripts/eval_tag_naming.py --dataset .../rf-detr-combined --read
    python3 scripts/eval_tag_naming.py --dataset .../rf-detr-combined   # analyse only
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

from PIL import Image
from rapidfuzz import fuzz, process

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "label_studio_ml", "examples"))
from cascade.shelf_tags import MODEL, detect_tags, normalize_tag  # noqa: E402

FUZZY_MIN_SCORE = 82        # the bar the retired learned map used, kept for comparability

# A tag sits on the rail directly below its product. These bound "directly":
# the product's bottom edge must be above the tag and within PAIR_MAX_ABOVE of
# it (a shelf is ~0.15 of frame height, so anything further is the shelf above),
# and the tag's x must fall inside the box, widened by PAIR_X_SLACK for tags
# printed off-centre in their slot.
PAIR_MAX_ABOVE = 0.15
PAIR_X_SLACK = 0.02


def load_split(dataset, split):
    """(images, gt) for a split: gt[file_name] = [(class_name, x1,y1,x2,y2 normalized)]."""
    path = os.path.join(dataset, split, "_annotations.coco.json")
    coco = json.load(open(path))
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    dims = {im["id"]: (im["file_name"], im["width"], im["height"]) for im in coco["images"]}
    gt = defaultdict(list)
    for a in coco["annotations"]:
        fn, W, H = dims[a["image_id"]]
        x, y, w, h = a["bbox"]
        gt[fn].append((cats[a["category_id"]], x / W, y / H, (x + w) / W, (y + h) / H))
    return [d[0] for d in dims.values()], gt, sorted(set(cats.values()))


def assign_tags(tags, boxes):
    """{tag index -> the labelled box it names}, one box per tag and one tag per box.

    A shelf puts a tag under every facing, so the mapping is one-to-one; letting
    two tags claim the same box (an earlier version of this) both loses a real
    pairing and invents a wrong one, and the wrong one then reads as a naming
    error when the resolver was right all along.

    Candidates are ranked by how far the box sits above the tag, tie-broken by
    horizontal offset, and assigned greedily -- the tightest pairs are the ones
    that are unambiguous, so taking them first is what leaves the ambiguous ones
    with fewer ways to go wrong.
    """
    cands = []
    for ti, tag in enumerate(tags):
        tx, ty = float(tag["x"]), float(tag["y"])
        for bi, b in enumerate(boxes):
            _, x1, _, x2, y2 = b
            gap = ty - y2
            if gap < -0.01 or gap > PAIR_MAX_ABOVE:
                continue
            if not (x1 - PAIR_X_SLACK <= tx <= x2 + PAIR_X_SLACK):
                continue
            cands.append((gap + 0.5 * abs((x1 + x2) / 2 - tx), ti, bi))

    out, used_t, used_b = {}, set(), set()
    for _, ti, bi in sorted(cands):
        if ti in used_t or bi in used_b:
            continue
        out[ti] = boxes[bi]
        used_t.add(ti)
        used_b.add(bi)
    return out


def read_all(dataset, splits, cache_path, limit=None):
    """One vision call per frame, cached. Writes after every read: a paid pass
    interrupted halfway keeps everything it already bought."""
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    for split in splits:
        files, _, _ = load_split(dataset, split)
        todo = [f for f in files if f"{split}/{f}" not in cache]
        if limit:
            todo = todo[:limit]
        print(f"{split}: {len(files)} frames, {len(todo)} to read with {MODEL}")
        for i, fn in enumerate(todo, 1):
            img = Image.open(os.path.join(dataset, split, fn)).convert("RGB")
            cache[f"{split}/{fn}"] = {"model": MODEL, "tags": detect_tags(img)}
            with open(cache_path, "w") as f:
                json.dump(cache, f)
            print(f"  [{i}/{len(todo)}] {fn}: {len(cache[f'{split}/{fn}']['tags'])} tags")
    return cache


RESOLVE_PROMPT = (
    "Each line below is the text printed on a Trader Joe's shelf price tag, in the "
    "store's own abbreviated wording. Match each one to the row of our product "
    "taxonomy that names the SAME product.\n\n"
    "Rules:\n"
    "- The tag is abbreviated: 'VANILLA LATTE' can be row 'La Colombe Vanilla Cold "
    "Brew Draft Latte'. Match on the product, not on string similarity.\n"
    "- Answer null when no row is that product. A wrong row is worse than null: it "
    "becomes a mislabelled training example, while a null is just a box a human "
    "still has to name.\n"
    "- Flavour and format matter. 'Lemon Sparkling Water' is not 'Lime Sparkling "
    "Water', and a 4-pack row is not the single-bottle row.\n\n"
    "TAXONOMY:\n{taxonomy}\n\nTAGS:\n{tags}\n\n"
    'JSON only: {{"matches":[{{"tag":<tag number>,"row":<row number or null>}}]}}')

RESOLVE_BATCH = 40


def llm_resolve(tag_texts, class_names, cache_path, model, batch=RESOLVE_BATCH):
    """{normalized tag -> class or None}, resolved against the taxonomy itself.

    This is the piece the learned map cannot do: the map only knows tags that
    have already been paired to a labelled box, so it is empty for every SKU in
    a store nobody has labelled yet. The taxonomy is known in full up front, so
    matching against it directly works on the first visit to a new store.

    Cached per (model, tag text) -- the same tag repeats across frames of the
    same shelf, and across stores.
    """
    from openai import OpenAI

    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    todo = sorted({normalize_tag(t) for t in tag_texts} - {k.split("|", 1)[1]
                                                           for k in cache if k.startswith(model + "|")})
    if todo:
        taxonomy = "\n".join(f"{i}: {n}" for i, n in enumerate(class_names))
        client = OpenAI()
        for start in range(0, len(todo), batch):
            chunk = todo[start:start + batch]
            listing = "\n".join(f"{i}. {t}" for i, t in enumerate(chunk))
            r = client.chat.completions.create(
                model=model, max_completion_tokens=4000,
                messages=[{"role": "user", "content": RESOLVE_PROMPT.format(
                    taxonomy=taxonomy, tags=listing)}])
            raw = (r.choices[0].message.content or "").strip()
            if raw.startswith("```"):
                raw = raw.split("```")[1].removeprefix("json").strip()
            got = {m["tag"]: m.get("row") for m in json.loads(raw).get("matches", [])}
            for i, tag in enumerate(chunk):
                row = got.get(i)
                cache[f"{model}|{tag}"] = (class_names[row]
                                           if isinstance(row, int) and 0 <= row < len(class_names)
                                           else None)
            with open(cache_path, "w") as f:
                json.dump(cache, f, indent=0)
            print(f"  resolved {min(start + batch, len(todo))}/{len(todo)} unique tags")
    return {k.split("|", 1)[1]: v for k, v in cache.items() if k.startswith(model + "|")}


def resolvers(class_names, llm_map=None):
    """{name: fn(tag_text) -> (class_or_None, note)} for the strategies compared."""
    norm_classes = {normalize_tag(c): c for c in class_names}

    def by_fuzzy(text):
        m = process.extractOne(normalize_tag(text), list(norm_classes),
                               scorer=fuzz.token_sort_ratio)
        if not m:
            return None, None
        # The runner-up is reported too: a top-1 that is right but below the bar
        # is a threshold problem, a top-1 that is wrong is a matching problem.
        return (norm_classes[m[0]] if m[1] >= FUZZY_MIN_SCORE else None), norm_classes[m[0]]

    out = {"fuzzy vs master list": by_fuzzy}
    if llm_map is not None:
        out["llm vs master list"] = lambda text: (llm_map.get(normalize_tag(text)), None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--splits", nargs="+", default=["valid", "test"])
    ap.add_argument("--cache", default=os.path.join(HERE, "..", "data", "tag_reads.json"))
    ap.add_argument("--read", action="store_true", help="make the vision calls for any "
                                                        "frame not already cached")
    ap.add_argument("--limit", type=int, help="cap frames read per split (for a cheap probe)")
    ap.add_argument("--resolve-llm", action="store_true",
                    help="also resolve tag text against the taxonomy with a text model "
                         "(cached per tag, so this is a few calls, not one per frame)")
    ap.add_argument("--resolve-model", default=os.getenv("TAG_RESOLVE_MODEL", "gpt-5.6-terra"),
                    help="the reasoning tier matters here: this is a knowledge task (which "
                         "Trader Joe's product is this), not a formatting one")
    ap.add_argument("--resolve-cache", default=os.path.join(HERE, "..", "data",
                                                            "tag_resolutions.json"))
    ap.add_argument("--misses", type=int, default=0, help="print N mis-namings per split")
    args = ap.parse_args()

    cache_path = os.path.abspath(args.cache)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    if args.read:
        cache = read_all(args.dataset, args.splits, cache_path, args.limit)
    elif os.path.exists(cache_path):
        cache = json.load(open(cache_path))
    else:
        raise SystemExit(f"no cached reads at {cache_path} -- run once with --read")

    for split in args.splits:
        files, gt, class_names = load_split(args.dataset, split)
        llm_map = None
        if args.resolve_llm:
            texts = [t["name"] for f in files if f"{split}/{f}" in cache
                     for t in cache[f"{split}/{f}"]["tags"]]
            print(f"\nresolving {len(set(map(normalize_tag, texts)))} unique tags with "
                  f"{args.resolve_model}")
            llm_map = llm_resolve(texts, class_names, os.path.abspath(args.resolve_cache),
                                  args.resolve_model)
        strategies = resolvers(class_names, llm_map)
        read = [f for f in files if f"{split}/{f}" in cache]
        if not read:
            continue

        tags_total = paired = 0
        boxes_total = sum(len(gt[f]) for f in read)
        boxes_tagged = 0
        stats = {k: Counter() for k in strategies}
        misses = {k: [] for k in strategies}

        for fn in read:
            tags = cache[f"{split}/{fn}"]["tags"]
            boxes = gt[fn]
            in_frame = {b[0] for b in boxes}
            tags_total += len(tags)
            assigned = assign_tags(tags, boxes)
            hit_boxes = set()
            for ti, tag in enumerate(tags):
                box = assigned.get(ti)
                if box is None:
                    continue
                paired += 1
                hit_boxes.add(id(box))
                truth = box[0]
                for name, fn_resolve in strategies.items():
                    cls, runner_up = fn_resolve(tag["name"])
                    if cls is None:
                        stats[name]["unresolved"] += 1
                        if runner_up == truth:
                            stats[name]["right_but_below_bar"] += 1
                    elif cls == truth:
                        stats[name]["correct"] += 1
                    else:
                        stats[name]["wrong"] += 1
                        # A name that IS in the frame, just not on this box, is
                        # usually the tag pairing landing one slot over -- a
                        # geometry bug, not a resolution one. Worth separating:
                        # tightening the pairing and improving the matcher are
                        # different work.
                        if cls in in_frame:
                            stats[name]["wrong_but_in_frame"] += 1
                        if len(misses[name]) < args.misses:
                            misses[name].append((tag["name"], cls, truth))
            boxes_tagged += len(hit_boxes)

        print(f"\n=== {split}: {len(read)} frames, {boxes_total} labelled boxes")
        print(f"  read    {tags_total} tags ({tags_total / len(read):.1f}/frame)")
        print(f"  pair    {paired}/{tags_total} tags sit under a labelled box "
              f"({paired / max(tags_total, 1):.0%})")
        print(f"  ceiling {boxes_tagged}/{boxes_total} labelled boxes have a tag "
              f"({boxes_tagged / max(boxes_total, 1):.0%}) -- the most tags alone could name")
        for name in strategies:
            s = stats[name]
            resolved = s["correct"] + s["wrong"]
            print(f"  {name}:")
            print(f"    resolve {resolved}/{paired} paired tags -> a class "
                  f"({resolved / max(paired, 1):.0%})")
            print(f"    name    {s['correct']}/{max(resolved, 1)} of those are the right class "
                  f"({s['correct'] / max(resolved, 1):.0%})")
            print(f"    yield   {s['correct']}/{boxes_total} labelled boxes named correctly "
                  f"({s['correct'] / max(boxes_total, 1):.0%})")
            if s["wrong_but_in_frame"]:
                print(f"    (of the {s['wrong']} wrong, {s['wrong_but_in_frame']} name a class "
                      f"that IS in the frame -- pairing, not resolution)")
            if s["right_but_below_bar"]:
                print(f"    (+{s['right_but_below_bar']} unresolved whose top match was "
                      f"actually right -- threshold, not matching)")
            for read_text, got, truth in misses[name]:
                print(f"      {read_text!r} -> {got!r}, truth {truth!r}")


if __name__ == "__main__":
    sys.exit(main())
