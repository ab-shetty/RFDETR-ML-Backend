#!/usr/bin/env python3
"""Read every shelf tag in the frame pool ONCE and cache it, so frame
selection can score an unlabeled frame without running a vision model.

Why this exists as a standalone artifact rather than a step inside the
cascade: the cascade reads tags at pre-annotation time and throws them away.
That is fine for pre-annotation, but frame *selection* needs to ask "which
classes are plausibly in this frame?" across the whole pool at once, and a
vision call per frame per request would make a person wait minutes for a
folder of images. Reading is slow and one-time; querying should be instant.

The critical difference from cascade/shelf_tags.py's lookup_class(): that maps
tag text through tag_class_map.json, which is *learned from labeled boxes*, so
a class with no labels can never appear in it -- exactly the classes selection
cares about most. Here the master list itself is put in the prompt and the
model that is already reading the frame does the matching, so a class with
zero labels resolves as readily as a well-covered one.

Tags that match NOTHING in the master list are kept too, under "unmatched".
Those are the master list's own gaps: a product on the shelf that the taxonomy
has no row for. See the class-naming SOP before adding rows for them.

Usage (native venv or inside the rfdetr container):
    python3 scripts/build_tag_index.py \\
        --frames-dir ~/Datasets/trader-joes/frames \\
        --master-list ~/Datasets/master_list.csv \\
        --out ~/Datasets/trader-joes/tag_index.json
"""
import argparse
import csv
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

from PIL import Image
from rapidfuzz import fuzz, process

# cascade/ lives under label_studio_ml/examples/ in the repo but is copied to
# /app in the container image, so try both rather than assuming either.
for candidate in ("/app",
                  os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "label_studio_ml", "examples")):
    if os.path.isdir(os.path.join(candidate, "cascade")):
        sys.path.insert(0, candidate)
        break
from cascade.shelf_tags import _encode, _get_client, normalize_tag  # noqa: E402

# Fuzzy scoring is the FALLBACK here, not the primary matcher, and the floor is
# only used to salvage a returned name that is not verbatim from the list.
#
# String similarity cannot do this job. Measured on real tags against the 179
# master list names:
#   token_sort_ratio  VANILLA LATTE -> La Colombe(R) Vanilla Cold Brew ... 49  (a MISS)
#   token_set_ratio   MOZZARELLA TOMATO SALAD -> Tomato ...................100 (nonsense)
#   WRatio            PINEAPPLE PROBIOTIC SHOT -> Apple .................... 90 (nonsense)
# and a bidirectional token-coverage variant scores the true match
# (VANILLA LATTE) and the false one (MOZZARELLA TOMATO SALAD -> Tomato)
# within a few points of each other. The distinction is semantic -- one tag
# abbreviates a long SKU name, the other names a prepared salad that merely
# mentions an ingredient -- so it is handed to the model that is already
# looking at the frame instead.
# gpt-5-mini was reading these HANDWRITTEN tags badly: it returned the single
# boldest word ("SMOOTHIE" for a card reading ORGANIC COCONUT SMOOTHIE), dropped
# brand lines, and sometimes described the tag instead of naming the product
# ("SPARKLING WATER (BLUE OVAL LABEL ON GREEN BOXES)"). That made the taxonomy
# look far more ambiguous than it is -- 30% of tag texts came out <=2 words,
# which was an artifact of the reader, not the store. gpt-5 at the SAME minimal
# effort reads them correctly in ~5s/frame. This pass is offline and cached, so
# accuracy is worth much more than tokens here.
#
# Do NOT raise gpt-5-mini to medium effort as a cheaper fix: it then spends the
# whole completion budget on reasoning and returns empty content.
DEFAULT_MODEL = "gpt-5"

FUZZY_MIN_SCORE = 65
# Where the salvage bar sits is measured, not guessed. Scoring every tag the
# model left unresolved against the master list, the band just below this is
# where wrong matches start: 85 pairs "SPARKLING LEMON STRAWBERRY" with
# "Sparkling Strawberry Juice" and 84 pairs "SPARKLING APPLE CIDER VINEGAR"
# with "Sparkling Apple Cider" (a different drink), while 87 correctly rescues
# "MIGHTY TURMERIC JUICE SHOT" -> "Organic Mighty Turmeric Juice Shot".
# Measured on one 8-frame sample; worth re-checking against a larger index
# before treating 86 as settled.
SALVAGE_MIN_SCORE = 86

_lock = threading.Lock()


def load_class_names(master_list_path):
    names = []
    with open(master_list_path) as f:
        for row in csv.DictReader(f):
            name = (row.get("Class Name (str)") or "").strip()
            cid = (row.get("Class ID (int)") or "").strip()
            if name and cid:
                names.append((name, int(cid)))
    return names


PROMPT_HEAD = (
    "This is a Trader Joe's shelf. The price tags on the front rails are "
    "HANDWRITTEN cards. List EVERY tag you can see. For each tag give:\n"
    "  text  - the COMPLETE product name as written, including smaller or "
    "lighter words above or below the big lettering (brand, 'ORGANIC', a "
    "flavour). Do NOT shorten it to the boldest word: a card reading ORGANIC "
    "COCONUT SMOOTHIE is not 'SMOOTHIE'. Do NOT describe the tag's colour or "
    "position -- the product name only, skipping the price and the "
    "bullet-point blurb.\n"
    "  x, y  - the tag's approximate center as fractions of width/height (0-1)\n"
    "  cls   - the ONE entry from the catalog below that names this same "
    "product, copied EXACTLY, or null if no catalog entry is that product.\n"
    "Match on product identity, not shared words: a tag reading 'MOZZARELLA "
    "TOMATO SALAD' is NOT the catalog's 'Tomato', and 'PINEAPPLE PROBIOTIC "
    "SHOT' is NOT 'Pineapple'. An abbreviation of a longer catalog name IS a "
    "match ('VANILLA LATTE' is the catalog's La Colombe vanilla cold brew "
    "latte). When unsure between siblings, prefer null over guessing.\n\n"
    "CATALOG:\n")
PROMPT_TAIL = '\n\nJSON only: {"tags":[{"text":"...","x":0.0,"y":0.0,"cls":"..."}]}'


def build_prompt(names):
    """Static across every frame, so it sits in the provider's prompt cache
    rather than being re-billed 500 times. Keep the catalog FIRST and the
    per-frame instruction last for that reason.

    This scales to the current 179 classes (~2k tokens). At a few thousand SKUs
    it would not, and the shape would have to become: fuzzy shortlist per tag,
    then a constrained pick from that shortlist -- the pattern already used by
    bootstrap_zero_data_classes.py.
    """
    return PROMPT_HEAD + "\n".join(n for n, _ in names) + PROMPT_TAIL


def resolve(cls_text, raw_text, name_to_id, names_upper, names):
    """Map the model's returned name to a master list row.

    Verbatim is the normal case. Fuzzy is only a salvage path for near-misses
    (stray registered-trademark sign, dropped comma); anything below the floor
    is treated as no match, because a wrong id here silently credits coverage
    to a class that is not on the shelf.
    """
    if cls_text and cls_text in name_to_id:
        return cls_text, name_to_id[cls_text], "exact"
    if cls_text:
        hit = process.extractOne(normalize_tag(cls_text), names_upper,
                                 scorer=fuzz.token_sort_ratio)
        if hit and hit[1] >= FUZZY_MIN_SCORE:
            return names[hit[2]][0], names[hit[2]][1], f"fuzzy:{hit[1]:.0f}"
    # Salvage net for the model returning null on a near-verbatim tag -- it
    # missed "TANGERINE PROBIOTIC SPARKLING BEVERAGE" against the identically
    # worded row 139. token_sort_ratio is well suited to exactly this and only
    # this: the same tag scores 100 here, while the false friends that ruled
    # fuzzy out as the PRIMARY matcher score far below the bar
    # (MOZZARELLA TOMATO SALAD -> Tomato is 47). Deliberately much stricter
    # than FUZZY_MIN_SCORE; this fires only on near-identical wording.
    hit = process.extractOne(normalize_tag(raw_text), names_upper,
                             scorer=fuzz.token_sort_ratio)
    if hit and hit[1] >= SALVAGE_MIN_SCORE:
        return names[hit[2]][0], names[hit[2]][1], f"salvaged:{hit[1]:.0f}"
    return None, None, "unresolved"


def read_tags(image, prompt, model=DEFAULT_MODEL, effort="minimal"):
    """One vision call -> [{text, x, y, cls}]. Returns [] on any failure, the
    same contract as cascade.detect_tags: an indexer that dies on one bad
    response would lose the whole run."""
    try:
        r = _get_client().chat.completions.create(
            model=model, max_completion_tokens=4000,
            reasoning_effort=effort,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url",
                 "image_url": {"url": f"data:image/jpeg;base64,{_encode(image)}"}},
            ]}],
        )
        raw = (r.choices[0].message.content or "").strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1].removeprefix("json").strip()
        return [t for t in json.loads(raw).get("tags", [])
                if t.get("text") and "x" in t and "y" in t]
    except Exception as exc:
        print(f"  tag read failed: {exc}", file=sys.stderr)
        return []


def index_frame(path, prompt, name_to_id, names_upper, names,
                model=DEFAULT_MODEL, effort="minimal"):
    """One vision call -> this frame's entry. Never raises. A frame that reads
    no tags is a legitimate result, which is what `tag_count` distinguishes --
    a pool where many frames read 0 tags means the reader regressed, not that
    the shelves are empty."""
    stem = os.path.splitext(os.path.basename(path))[0]
    try:
        with Image.open(path) as im:
            tags = read_tags(im.convert("RGB"), prompt, model, effort)
    except Exception as exc:                       # unreadable/truncated jpeg
        return stem, {"error": str(exc)[:200], "tags": [], "unmatched": []}

    entry, unmatched, price_only = [], [], 0
    for t in tags:
        name, cid, how = resolve(t.get("cls"), t["text"], name_to_id,
                                 names_upper, names)
        entry.append({"text": t["text"], "x": float(t["x"]), "y": float(t["y"]),
                      "class_name": name, "class_id": cid, "match": how})
        if name is not None:
            continue
        # normalize_tag() strips prices, so an empty key means the reader
        # picked up a price with no product name ("$2.99"). Those are not
        # taxonomy gaps and they swamped the gap report on the first full run
        # -- 904 of 6238 tags, taking 8 of the top 15 "missing" slots.
        if normalize_tag(t["text"]):
            unmatched.append(t["text"])
        else:
            price_only += 1
    return stem, {"tags": entry, "unmatched": unmatched,
                  "tag_count": len(tags), "price_only": price_only}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--master-list", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, help="index at most this many NEW frames")
    ap.add_argument("--workers", type=int, default=8,
                    help="concurrent vision calls; each takes a few seconds")
    ap.add_argument("--checkpoint-every", type=int, default=25)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--reasoning-effort", default="minimal")
    args = ap.parse_args()

    names = load_class_names(args.master_list)
    names_upper = [normalize_tag(n) for n, _ in names]
    name_to_id = dict(names)
    prompt = build_prompt(names)
    print(f"{len(names)} master list classes loaded")

    # Resumable by design: 600 vision calls is long enough that a dropped
    # connection partway through must not mean starting over, and re-running
    # after new frames land should cost only the new frames.
    index = {}
    if os.path.exists(args.out):
        with open(args.out) as f:
            index = json.load(f).get("frames", {})
        print(f"{len(index)} frames already indexed; they will be skipped")

    frames = sorted(p for p in os.listdir(args.frames_dir)
                    if p.lower().endswith((".jpg", ".jpeg", ".png")))
    todo = [os.path.join(args.frames_dir, p) for p in frames
            if os.path.splitext(p)[0] not in index]
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(frames)} frames in pool, {len(todo)} to read")
    if not todo:
        return 0

    def save():
        tmp = args.out + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"meta": {"frames_dir": os.path.abspath(args.frames_dir),
                                "master_list": os.path.abspath(args.master_list),
                                "fuzzy_min_score": FUZZY_MIN_SCORE,
                                "updated": time.strftime("%Y-%m-%dT%H:%M:%S")},
                       "frames": index}, f)
        os.replace(tmp, args.out)      # atomic: a crash mid-write keeps the old index

    done = 0
    start = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        work = lambda p: index_frame(p, prompt, name_to_id, names_upper, names,
                                     args.model, args.reasoning_effort)
        for stem, entry in pool.map(work, todo):
            with _lock:
                index[stem] = entry
                done += 1
                if done % args.checkpoint_every == 0:
                    save()
                    rate = done / (time.time() - start)
                    print(f"  {done}/{len(todo)} frames  ({rate:.1f}/s, "
                          f"~{(len(todo)-done)/rate/60:.0f} min left)")
    save()

    read = sum(e.get("tag_count", 0) for e in index.values())
    blank = sum(1 for e in index.values() if not e.get("tag_count"))
    gaps, matched, classes = {}, 0, set()
    for e in index.values():
        for t in e.get("tags", []):
            if t.get("class_id") is not None:
                matched += 1
                classes.add(t["class_id"])
        for u in e.get("unmatched", []):
            key = normalize_tag(u)     # collapses the "$3.99" variants of one tag
            if key:                    # skips price-only tags in indexes built
                gaps[key] = gaps.get(key, 0) + 1   # before they were filtered
    junk = sum(e.get("price_only", 0) for e in index.values())
    print(f"\n{len(index)} frames indexed, {read} tags read, "
          f"{blank} frames with no readable tag")
    print(f"{matched} tags resolved to {len(classes)} distinct master list classes; "
          f"{junk} price-only tags ignored")
    print(f"{len(gaps)} distinct tag texts matched NOTHING in the master list "
          f"(taxonomy gaps); most frequent:")
    for text, n in sorted(gaps.items(), key=lambda kv: -kv[1])[:15]:
        print(f"  {n:>3}  {text}")
    print(f"\nindex -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
