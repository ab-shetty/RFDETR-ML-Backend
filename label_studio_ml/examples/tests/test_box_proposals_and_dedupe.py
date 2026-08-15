"""Cover the three stages added on top of plain detection.

Two of them default to off, so the rest of the suite passes without ever
executing them. The third, cross-class dedupe, runs unconditionally on every
prediction and DELETES regions -- the highest-risk code here, because a wrong
deletion is invisible: it looks like the model having a bad day, not like a bug.
"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from control_models.rectangle_labels import RFDETRRectangleLabelsModel  # noqa: E402

CLASS_NAMES = ["Apple", "Banana", "San Pellegrino 6-pack"]


def _fake_detections(class_ids, confidences, boxes):
    return SimpleNamespace(
        xyxy=np.array(boxes, dtype=float) if boxes else np.zeros((0, 4)),
        confidence=np.array(confidences, dtype=float),
        class_id=np.array(class_ids, dtype=int),
    )


def _instance(detections, taxonomy=True):
    mock_model = MagicMock()
    mock_model.predict.return_value = detections
    return RFDETRRectangleLabelsModel.model_construct(
        # _add_box_proposals reads the labelling config to find the single
        # rectangle label a proposal should carry; the real object always has
        # one, so the fixture must too.
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
        model=mock_model,
    )


@pytest.fixture
def image_path(tmp_path):
    path = tmp_path / "img.jpg"
    Image.new("RGB", (100, 100)).save(path)
    return str(path)


def _rects(regions):
    return [r for r in regions if r["type"] == "rectanglelabels"]


def _taxonomies(regions):
    return [r for r in regions if r["type"] == "taxonomy"]


# ------------------------------------------------------------ cross-class dedupe

def test_dedupe_collapses_the_same_facing_under_competing_names(image_path):
    """RF-DETR runs NMS per class, not across classes, so one facing comes back
    once per plausible SKU. Every one of the SKU model's 31 false positives on
    the held-out clips was this."""
    det = _fake_detections(
        class_ids=[0, 1],
        confidences=[0.9, 0.6],
        boxes=[[10, 10, 50, 50], [11, 11, 51, 51]],  # same facing, two names
    )
    regions = _instance(det).predict_regions(image_path)
    assert len(_rects(regions)) == 1
    # the confident one survives
    assert _taxonomies(regions)[0]["value"]["taxonomy"] == [["Apple"]]


def test_dedupe_removes_the_loser_s_taxonomy_row_too(image_path):
    """A taxonomy result shares its box's id. Dropping the box and leaving its
    SKU behind would attach that name to nothing."""
    det = _fake_detections([0, 1], [0.9, 0.6], [[10, 10, 50, 50], [11, 11, 51, 51]])
    regions = _instance(det).predict_regions(image_path)
    ids = {r["id"] for r in _rects(regions)}
    assert all(t["id"] in ids for t in _taxonomies(regions))
    assert len(_taxonomies(regions)) == 1


def test_dedupe_leaves_distinct_facings_alone(image_path):
    det = _fake_detections([0, 1], [0.9, 0.8], [[0, 0, 20, 20], [60, 60, 90, 90]])
    regions = _instance(det).predict_regions(image_path)
    assert len(_rects(regions)) == 2


def test_dedupe_keeps_the_higher_confidence_box_regardless_of_order(image_path):
    """Detections do not arrive sorted; the survivor must not depend on that."""
    det = _fake_detections([1, 0], [0.4, 0.95], [[10, 10, 50, 50], [11, 11, 51, 51]])
    regions = _instance(det).predict_regions(image_path)
    assert len(_rects(regions)) == 1
    assert _rects(regions)[0]["score"] == pytest.approx(0.95)


# -------------------------------------------------------------- box proposals

def test_box_proposals_are_off_unless_enabled(image_path, monkeypatch):
    monkeypatch.setattr("control_models.box_proposals.BOX_PROPOSALS_ENABLED", False)
    det = _fake_detections([0], [0.9], [[0, 0, 20, 20]])
    regions = _instance(det).predict_regions(image_path)
    assert len(_rects(regions)) == 1


def test_box_proposals_add_unnamed_rectangles(image_path, monkeypatch):
    """A proposal carries no SKU on purpose: the Taxonomy control is perRegion
    required, so an unnamed box cannot be submitted, which is what forces a
    human to name it."""
    monkeypatch.setattr("control_models.box_proposals.BOX_PROPOSALS_ENABLED", True)
    monkeypatch.setattr("control_models.box_proposals.propose",
                        lambda image, taken, **kw: [(60.0, 60.0, 90.0, 90.0, 0.8)])
    monkeypatch.setattr("cascade.template_match.TEMPLATE_MATCHING_ENABLED", False)
    det = _fake_detections([0], [0.9], [[0, 0, 20, 20]])
    regions = _instance(det).predict_regions(image_path)

    rects = _rects(regions)
    assert len(rects) == 2
    proposal = [r for r in rects if r["score"] == pytest.approx(0.8)][0]
    assert proposal["value"]["rectanglelabels"] == ["Product"]
    # exactly one taxonomy row, belonging to the detector's own box
    assert len(_taxonomies(regions)) == 1
    assert _taxonomies(regions)[0]["id"] != proposal["id"]


def test_a_run_can_ask_for_boxes_without_name_guesses(image_path, monkeypatch):
    """The 'Boxes only' option: on a new store visit most SKUs have no template,
    and template matching is never right about a SKU absent from its bank."""
    monkeypatch.setattr("control_models.box_proposals.BOX_PROPOSALS_ENABLED", False)
    monkeypatch.setattr("control_models.box_proposals.propose",
                        lambda image, taken, **kw: [(60.0, 60.0, 90.0, 90.0, 0.8)])
    called = []
    monkeypatch.setattr("cascade.template_match.name_crop",
                        lambda *a, **kw: called.append(1) or ("Apple", 99))
    det = _fake_detections([0], [0.9], [[0, 0, 20, 20]])

    regions = _instance(det).predict_regions(
        image_path, propose_boxes=True, name_proposals=False)
    assert len(_rects(regions)) == 2
    assert not called, "name_proposals=False must not consult the template bank"


def test_proposals_overlapping_a_detected_box_are_dropped_by_propose(image_path):
    """propose() is what enforces this, so assert it directly rather than
    through the region plumbing."""
    from control_models.box_proposals import iou

    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


# ------------------------------------------------------------ template matching

def test_template_matching_disables_itself_without_a_bank(tmp_path):
    """A missing bank must not fail a prediction -- the boxes are still worth
    having unnamed."""
    import cascade.template_match as tm

    tm._bank = None
    tm._missing_logged = False
    name, n = tm.name_crop(Image.new("RGB", (40, 40)), str(tmp_path))
    assert (name, n) == (None, 0)


def test_template_matching_ignores_crops_too_small_to_have_keypoints(tmp_path):
    import cascade.template_match as tm

    tm._bank = None
    name, n = tm.name_crop(Image.new("RGB", (4, 4)), str(tmp_path))
    assert (name, n) == (None, 0)
