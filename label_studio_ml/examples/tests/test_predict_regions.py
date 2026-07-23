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


def _instance(class_thresholds, detections, model_score_threshold=0.5, xml_threshold_override=None):
    mock_model = MagicMock()
    mock_model.predict.return_value = detections
    return RFDETRRectangleLabelsModel.model_construct(
        class_names=CLASS_NAMES,
        class_thresholds=class_thresholds,
        model_score_threshold=model_score_threshold,
        xml_threshold_override=xml_threshold_override,
        label_map={name: name for name in CLASS_NAMES},
        from_name="label",
        to_name="image",
        taxonomy_path_map={},
        taxonomy_from_name=None,
        model=mock_model,
    )


@pytest.fixture
def sample_image_path(tmp_path):
    img = Image.new("RGB", (100, 100))
    path = tmp_path / "img.jpg"
    img.save(path)
    return str(path)


def test_per_class_threshold_drops_detection_below_its_own_cutoff(sample_image_path):
    # Apple has a tuned high threshold (historically noisy class); Banana keeps
    # the low global default. Both detections score 0.6 — only Banana should survive.
    detections = _fake_detections(
        class_ids=[0, 1],
        confidences=[0.6, 0.6],
        boxes=[[10, 10, 50, 50], [10, 10, 50, 50]],
    )
    instance = _instance(class_thresholds={"Apple": 0.7}, detections=detections)

    regions = instance.predict_regions(sample_image_path)
    labels = [r["value"]["rectanglelabels"][0] for r in regions if r["type"] == "rectanglelabels"]
    assert labels == ["Banana"]


def test_unknown_class_falls_back_to_global_threshold(sample_image_path):
    # San Pellegrino 6-pack has no tuned threshold -> uses model_score_threshold (0.5).
    detections = _fake_detections(
        class_ids=[2],
        confidences=[0.55],
        boxes=[[10, 10, 50, 50]],
    )
    instance = _instance(class_thresholds={"Apple": 0.9}, detections=detections, model_score_threshold=0.5)

    regions = instance.predict_regions(sample_image_path)
    labels = [r["value"]["rectanglelabels"][0] for r in regions if r["type"] == "rectanglelabels"]
    assert labels == ["San Pellegrino 6-pack"]


def test_model_is_called_with_the_lowest_threshold_as_a_floor(sample_image_path):
    detections = _fake_detections(class_ids=[], confidences=[], boxes=[])
    instance = _instance(class_thresholds={"Apple": 0.2, "Banana": 0.8}, detections=detections)

    instance.predict_regions(sample_image_path)

    called_threshold = instance.model.predict.call_args.kwargs["threshold"]
    assert called_threshold == 0.2
