import json
import os
import sys
from unittest.mock import patch

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cascade import shelf_tags  # noqa: E402


def test_normalize_tag_strips_price_and_uppercases():
    assert shelf_tags.normalize_tag("Vanilla Latte $2.99") == "VANILLA LATTE"
    assert shelf_tags.normalize_tag("COLD BREW WITH OATMILK $2-69") == "COLD BREW WITH OATMILK"
    assert shelf_tags.normalize_tag("Sparkling Ginger + Lemon") == "SPARKLING GINGER + LEMON"


def test_load_tag_class_map_picks_majority_class(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({
        "VERY GREEN JUICE": {"Very Green Juice": 3},
        "AQUA KEFIR": {"Aqua Kefir Orange Peach Mango": 2, "Something Else": 1},
    }))
    m = shelf_tags.load_tag_class_map(str(p))
    assert m == {"VERY GREEN JUICE": "Very Green Juice", "AQUA KEFIR": "Aqua Kefir Orange Peach Mango"}


def test_load_tag_class_map_missing_returns_empty():
    assert shelf_tags.load_tag_class_map("/nope/x.json") == {}


def test_lookup_class_exact_and_fuzzy():
    m = {"VERY GREEN JUICE": "Very Green Juice", "AQUA KEFIR ORANGE PEACH MANGO": "Aqua Kefir Orange Peach Mango"}
    assert shelf_tags.lookup_class("VERY GREEN JUICE $2.99", m) == "Very Green Juice"
    # minor OCR wording drift still matches via fuzzy
    assert shelf_tags.lookup_class("Aqua Kefir Orange Peach Mango", m) == "Aqua Kefir Orange Peach Mango"


def test_lookup_class_rejects_below_fuzzy_threshold():
    m = {"VERY GREEN JUICE": "Very Green Juice"}
    assert shelf_tags.lookup_class("EGG SALAD", m) is None


def test_propose_from_tags_emits_box_for_uncovered_slot():
    m = {"VERY GREEN JUICE": "Very Green Juice"}
    img = Image.new("RGB", (100, 200))
    with patch("cascade.shelf_tags.detect_tags", return_value=[{"name": "VERY GREEN JUICE $2.99", "x": 0.8, "y": 0.4}]):
        props = shelf_tags.propose_from_tags(img, m, covered_centers=[])
    assert len(props) == 1
    assert props[0]["class_name"] == "Very Green Juice"
    # slot box sits above the tag (y1 < y2 <= tag_y*height)
    x1, y1, x2, y2 = props[0]["box"]
    assert y1 < y2 <= 0.4 * 200


def test_propose_from_tags_skips_slot_already_covered():
    m = {"VERY GREEN JUICE": "Very Green Juice"}
    img = Image.new("RGB", (100, 200))
    # a detection already sits in the slot above the tag (tag y=0.4, slot ~0.31)
    with patch("cascade.shelf_tags.detect_tags", return_value=[{"name": "VERY GREEN JUICE", "x": 0.8, "y": 0.4}]):
        props = shelf_tags.propose_from_tags(img, m, covered_centers=[(0.8, 0.31)])
    assert props == []


def test_propose_from_tags_skips_unmapped_tag():
    m = {"VERY GREEN JUICE": "Very Green Juice"}
    img = Image.new("RGB", (100, 200))
    with patch("cascade.shelf_tags.detect_tags", return_value=[{"name": "MYSTERY ITEM", "x": 0.5, "y": 0.5}]):
        assert shelf_tags.propose_from_tags(img, m, covered_centers=[]) == []


def _corrector_model(tag_class_map):
    from control_models.rectangle_labels import RFDETRRectangleLabelsModel
    return RFDETRRectangleLabelsModel.model_construct(
        class_names=["Apple", "Very Green Juice"],
        tag_class_map=tag_class_map,
        taxonomy_path_map={"Very Green Juice": ["Beverage", "Very Green Juice"], "Apple": ["Produce", "Apple"]},
        taxonomy_from_name="sku", taxonomy_to_name="image", to_name="image",
    )


def test_apply_tag_corrections_fixes_taxonomy_from_tag():
    from unittest.mock import patch
    m = _corrector_model({"VERY GREEN JUICE": "Very Green Juice"})
    # a box mis-classed as Apple, with a Very Green Juice tag in its column
    regions = [
        {"id": "r1", "type": "rectanglelabels", "value": {"x": 75, "y": 30, "width": 10, "height": 10}},
        {"id": "t1", "type": "taxonomy", "parentID": "r1", "value": {"taxonomy": [["Produce", "Apple"]]}},
    ]
    with patch("cascade.shelf_tags.detect_tags", return_value=[{"name": "VERY GREEN JUICE $2.99", "x": 0.8, "y": 0.42}]):
        m._apply_tag_corrections(Image.new("RGB", (100, 100)), regions)
    assert regions[1]["value"]["taxonomy"] == [["Beverage", "Very Green Juice"]]


def test_apply_tag_corrections_noop_without_matching_tag():
    from unittest.mock import patch
    m = _corrector_model({"VERY GREEN JUICE": "Very Green Juice"})
    regions = [
        {"id": "r1", "type": "rectanglelabels", "value": {"x": 5, "y": 5, "width": 10, "height": 10}},
        {"id": "t1", "type": "taxonomy", "parentID": "r1", "value": {"taxonomy": [["Produce", "Apple"]]}},
    ]
    with patch("cascade.shelf_tags.detect_tags", return_value=[{"name": "VERY GREEN JUICE", "x": 0.8, "y": 0.42}]):
        m._apply_tag_corrections(Image.new("RGB", (100, 100)), regions)
    assert regions[1]["value"]["taxonomy"] == [["Produce", "Apple"]]  # far-away tag: unchanged
