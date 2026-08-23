"""Cover the naming stage that overrides the class head.

This stage EDITS and DELETES names other stages produced, on every box in the
frame, which makes its failure mode the quiet kind: a mis-wired fallback does
not raise, it just ships the head's wrong guess into a labeller's queue, or
strips a name that was right. So each of the four cases it distinguishes is
pinned here -- named, declined-but-confident, declined-and-not, and the call
never came back at all, which must not be read as a decline.
"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cascade import box_naming  # noqa: E402
from control_models.rectangle_labels import RFDETRRectangleLabelsModel  # noqa: E402

CLASS_NAMES = ["Apple", "Banana", "San Pellegrino 6-pack"]


def _instance(taxonomy=True):
    return RFDETRRectangleLabelsModel.model_construct(
        control=SimpleNamespace(labels_attrs={"Product": {}}),
        class_names=CLASS_NAMES,
        class_thresholds={},
        model_score_threshold=0.1,
        xml_threshold_override=None,
        label_map={name: "Product" for name in CLASS_NAMES},
        from_name="label",
        to_name="image",
        taxonomy_path_map={n: [n] for n in CLASS_NAMES} if taxonomy else {},
        taxonomy_from_name="sku" if taxonomy else None,
        taxonomy_to_name="image",
        model=MagicMock(),
    )


def _regions(score, named="Apple"):
    """One box at a given detector confidence, with the head's name on it."""
    out = [{
        "id": "r1", "from_name": "label", "to_name": "image", "type": "rectanglelabels",
        "value": {"rectanglelabels": ["Product"], "x": 10, "y": 10, "width": 20, "height": 20},
        "score": score,
    }]
    if named:
        out.append({
            "id": "r1", "from_name": "sku", "to_name": "image", "type": "taxonomy",
            "value": {"taxonomy": [[named]]}, "score": score,
        })
    return out


@pytest.fixture
def image():
    return Image.new("RGB", (100, 100))


def _tax(regions):
    return [r for r in regions if r.get("type") == "taxonomy"]


def test_vision_name_overrides_the_head(image):
    regions = _regions(score=0.95, named="Apple")
    with patch.object(box_naming, "name_boxes", return_value={0: "Banana"}):
        _instance()._apply_box_naming(image, regions, 100, 100)
    assert _tax(regions)[0]["value"]["taxonomy"] == [["Banana"]]


def test_confident_head_name_survives_a_decline(image):
    regions = _regions(score=box_naming.HEAD_FLOOR + 0.05, named="Apple")
    with patch.object(box_naming, "name_boxes", return_value={0: None}):
        _instance()._apply_box_naming(image, regions, 100, 100)
    assert _tax(regions)[0]["value"]["taxonomy"] == [["Apple"]]


def test_unconfident_head_name_is_cleared_on_a_decline(image):
    regions = _regions(score=box_naming.HEAD_FLOOR - 0.05, named="Apple")
    with patch.object(box_naming, "name_boxes", return_value={0: None}):
        _instance()._apply_box_naming(image, regions, 100, 100)
    assert _tax(regions) == []
    # the box itself stays: it is a real facing, it just has no name yet
    assert len(regions) == 1


def test_a_failed_call_is_not_a_decline(image):
    """No answer for a box must leave it exactly as it was.

    A decline means "I looked and could not tell"; an empty answer for that
    index means the request failed. Treating the second as the first would
    silently strip names off every box whenever the API had a bad minute.
    """
    regions = _regions(score=0.2, named="Apple")
    with patch.object(box_naming, "name_boxes", return_value={7: "Banana"}):
        _instance()._apply_box_naming(image, regions, 100, 100)
    assert _tax(regions)[0]["value"]["taxonomy"] == [["Apple"]]


def test_an_unnamed_proposal_gets_a_name(image):
    """Box proposals arrive with no taxonomy region at all -- the stage has to
    add one, not just edit what is there."""
    regions = _regions(score=0.5, named=None)
    with patch.object(box_naming, "name_boxes", return_value={0: "Banana"}):
        _instance()._apply_box_naming(image, regions, 100, 100)
    tax = _tax(regions)
    assert len(tax) == 1
    assert tax[0]["id"] == "r1" and tax[0]["value"]["taxonomy"] == [["Banana"]]


def test_no_taxonomy_control_is_a_no_op(image):
    regions = _regions(score=0.5, named=None)
    with patch.object(box_naming, "name_boxes") as called:
        _instance(taxonomy=False)._apply_box_naming(image, regions, 100, 100)
    called.assert_not_called()
    assert len(regions) == 1


def test_chunking_numbers_boxes_from_zero_in_each_call():
    """Each call draws only its own chunk, numbered from 0, and the answers are
    mapped back to global indices. Off-by-one here swaps products' names."""
    boxes = [(0, 0, 10, 10)] * (box_naming.MAX_BOXES_PER_CALL + 2)
    seen = []

    class _Resp:
        def __init__(self, n):
            self.choices = [SimpleNamespace(message=SimpleNamespace(
                content='{"boxes":[{"box":0,"row":1}]}'))]
            seen.append(n)

    client = MagicMock()
    client.chat.completions.create.side_effect = lambda **kw: _Resp(len(kw["messages"]))
    with patch.object(box_naming, "_get_client", return_value=client):
        out = box_naming.name_boxes(Image.new("RGB", (50, 50)), boxes, CLASS_NAMES)
    assert len(seen) == 2                       # 42 boxes -> two calls
    assert out[0] == "Banana"                   # first box of the first chunk
    assert out[box_naming.MAX_BOXES_PER_CALL] == "Banana"   # first of the second
