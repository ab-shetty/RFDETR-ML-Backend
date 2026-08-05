#!/usr/bin/env python3
"""Re-resolve a tag index's class assignments without re-reading any frames.

The vision call that reads shelf tags is the expensive part and its output --
the tag text and position -- does not change when the taxonomy does. Only the
tag -> class decision does. This re-runs that decision over a cached index, so
improving the matching costs seconds instead of another pass over the pool.

The matching order is the point of this script, and it is ordered by how much
each source actually knows:

  1. models/tag_class_map.json -- LEARNED FROM YOUR OWN LABELED FRAMES by
     build_tag_class_map.py, which pairs each shelf tag with the labeled box
     above it. It knows this store's real tag wording, including abbreviations
     no string match would get. It was not consulted at all before, which is
     how "LEMON + STRAWBERRY" became "Sparkling Tea & Lemonade" when the map
     had the right answer.
  2. near-verbatim catalog match on the RAW TAG TEXT. "TANGERINE PROBIOTIC
     SPARKLING BEVERAGE" is character-for-character a master list row; that
     should never lose to a model's guess, and it did -- it came back as
     "Sparkling Strawberry Juice", which is a real catalog name, just not this
     product's. A wrong-but-valid answer is the failure mode a plain
     membership check cannot catch.
  3. whatever the vision model already resolved, kept from the index.

Deliberately NOT retained: the old loose fuzzy fallback on the model's returned
name (accepted anything scoring 65). That is what turned "SPARKLING APPLE CIDER
VINEGAR BEVERAGE" into "Sparkling Pomegrante Punch Beverage".
"""
import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "label_studio_ml", "examples"))
from cascade.shelf_tags import normalize_tag  # noqa: E402
from rapidfuzz import fuzz, process  # noqa: E402

MAP_MIN_SCORE = 82      # same bar cascade/shelf_tags.lookup_class uses on this map
VERBATIM_MIN_SCORE = 86  # measured floor; below this, wrong matches start
# Share of the tag's words that must appear in a learned-map answer. Measured:
# the map lifts accuracy 43%->67% against human boxes, and this guard costs
# 0.2 points there while removing the mispairings the test cannot see, on the
# unlabeled products where proposals are actually used.
MAP_MIN_OVERLAP = 0.34


def load_master(path):
    rows = []
    for row in csv.DictReader(open(path)):
        name = (row.get("Class Name (str)") or "").strip()
        cid = (row.get("Class ID (int)") or "").strip()
        if name and cid:
            rows.append((name, int(cid)))
    return rows


def load_map(path):
    """{normalized tag text -> majority class}, dropping blank keys.

    A blank key appears when a tag read produced only a price; it collects
    whatever classes happened to sit above those tags and would match anything.
    """
    if not os.path.exists(path):
        return {}
    raw = json.load(open(path))
    return {k: max(v.items(), key=lambda kv: kv[1])[0]
            for k, v in raw.items() if k.strip() and v}


def word_overlap(tag_text, cls):
    """Share of the tag's words that appear in the class name.

    This is the guard on the learned map. build_tag_class_map.py pairs a tag
    with the nearest labeled box ABOVE it, so for a product nobody labeled it
    happily pairs the tag with some unrelated product further up the shelf.
    That is how the map came to claim "GTS ALIVE ROOT BEER" is a La Colombe
    Triple Coffee and "MIGHTY TURMERIC JUICE SHOT" is a Guava Paloma.

    Support count does not catch it -- the turmeric error has the same support
    (2) as the correct "LEMON + STRAWBERRY" entry. Shared wording does: a real
    abbreviation keeps the product's words ("VANILLA LATTE" inside "La Colombe
    Vanilla Cold Brew Draft Latte"), a mispairing shares nothing.
    """
    a = set(normalize_tag(tag_text).split())
    b = set(normalize_tag(cls).split())
    return len(a & b) / len(a) if a else 0.0


def resolve(text, tag_map, map_keys, names, names_upper, name_to_id, current):
    key = normalize_tag(text)
    if key:
        cand = None
        if key in tag_map:
            cand, how = tag_map[key], "map:exact"
        else:
            hit = process.extractOne(key, map_keys, scorer=fuzz.token_sort_ratio)
            if hit and hit[1] >= MAP_MIN_SCORE:
                cand, how = tag_map[map_keys[hit[2]]], f"map:{hit[1]:.0f}"
        if cand is not None:
            if word_overlap(text, cand) >= MAP_MIN_OVERLAP:
                return cand, how
            # Mispairing: fall through rather than trust it.
        hit = process.extractOne(key, names_upper, scorer=fuzz.token_sort_ratio)
        if hit and hit[1] >= VERBATIM_MIN_SCORE:
            return names[hit[2]], f"verbatim:{hit[1]:.0f}"
    return current, "kept" if current else "unresolved"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", required=True)
    ap.add_argument("--master-list", default="/home/ubuntu/Datasets/master_list.csv")
    ap.add_argument("--tag-map", default=os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "label_studio_ml", "examples", "models", "tag_class_map.json"))
    ap.add_argument("--out", help="write here instead of in place")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    rows = load_master(args.master_list)
    names = [n for n, _ in rows]
    names_upper = [normalize_tag(n) for n in names]
    name_to_id = dict(rows)
    tag_map = load_map(args.tag_map)
    map_keys = list(tag_map)
    # A map entry naming a class the master list no longer has would silently
    # produce a None id downstream, so surface it here instead.
    stale = {v for v in tag_map.values() if v not in name_to_id}
    print(f"{len(names)} master classes, {len(tag_map)} learned tag texts"
          + (f", {len(stale)} map entries name unknown classes" if stale else ""))

    doc = json.load(open(args.index))
    changed = by_source = 0
    sources, examples = {}, []
    for entry in doc["frames"].values():
        for t in entry.get("tags", []):
            before = t.get("class_name")
            name, how = resolve(t["text"], tag_map, map_keys, names, names_upper,
                                name_to_id, before)
            if name is not None and name not in name_to_id:
                name, how = before, "kept"      # stale map entry, do not use
            sources[how.split(":")[0]] = sources.get(how.split(":")[0], 0) + 1
            if name != before:
                changed += 1
                if len(examples) < 12:
                    examples.append((t["text"][:38], before, name, how))
            t["class_name"] = name
            t["class_id"] = name_to_id.get(name) if name else None
            t["match"] = how
        entry["unmatched"] = [t["text"] for t in entry.get("tags", [])
                              if not t.get("class_name") and normalize_tag(t["text"])]
    print(f"{changed} tag resolutions changed; by source: {sources}")
    for text, before, after, how in examples:
        print(f"  {text:<40} {str(before)[:28]:<30} -> {str(after)[:34]:<36} [{how}]")
    if args.dry_run:
        print("\n[dry run] nothing written")
        return 0
    out = args.out or args.index
    json.dump(doc, open(out, "w"))
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
