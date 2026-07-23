import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from control_models.rectangle_labels import RFDETRRectangleLabelsModel  # noqa: E402


def _instance(class_thresholds, model_score_threshold=0.5, xml_threshold_override=None):
    # model_construct bypasses pydantic validation — lets us build a ControlModel
    # subclass instance without a real ControlTag/model/LabelStudioMLBase, since
    # this test only exercises the threshold-resolution methods.
    return RFDETRRectangleLabelsModel.model_construct(
        class_thresholds=class_thresholds,
        model_score_threshold=model_score_threshold,
        xml_threshold_override=xml_threshold_override,
    )


def test_effective_threshold_uses_per_class_value_when_present():
    instance = _instance(class_thresholds={"Apple": 0.3})
    assert instance.get_effective_threshold("Apple") == 0.3


def test_effective_threshold_falls_back_to_global_default_for_unknown_class():
    instance = _instance(class_thresholds={"Apple": 0.3}, model_score_threshold=0.55)
    assert instance.get_effective_threshold("Unknown Class") == 0.55


def test_xml_override_acts_as_a_floor_not_a_replacement():
    # per-class threshold (0.2) is lower than the admin's explicit XML override (0.4)
    # -> override wins, since it's a floor.
    instance = _instance(class_thresholds={"Apple": 0.2}, xml_threshold_override=0.4)
    assert instance.get_effective_threshold("Apple") == 0.4


def test_xml_override_does_not_lower_a_higher_class_threshold():
    # per-class threshold (0.8) is already above the XML floor (0.4) -> class value wins.
    instance = _instance(class_thresholds={"San Pellegrino 6-pack": 0.8}, xml_threshold_override=0.4)
    assert instance.get_effective_threshold("San Pellegrino 6-pack") == 0.8


def test_min_prediction_threshold_is_lowest_of_all_candidates():
    instance = _instance(class_thresholds={"Apple": 0.2, "Banana": 0.8}, model_score_threshold=0.5)
    assert instance.min_prediction_threshold() == 0.2


def test_min_prediction_threshold_has_an_absolute_floor():
    instance = _instance(class_thresholds={"Apple": 0.01}, model_score_threshold=0.5)
    assert instance.min_prediction_threshold() == 0.05


def test_min_prediction_threshold_with_no_class_thresholds_uses_global():
    instance = _instance(class_thresholds={}, model_score_threshold=0.6)
    assert instance.min_prediction_threshold() == 0.6
