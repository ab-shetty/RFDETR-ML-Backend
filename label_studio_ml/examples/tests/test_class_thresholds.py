import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from control_models.base import load_class_thresholds  # noqa: E402


def _write_checkpoint_dir(tmp_path, thresholds_json=None):
    model_path = tmp_path / "checkpoint.pth"
    model_path.write_bytes(b"")
    if thresholds_json is not None:
        (tmp_path / "class_thresholds.json").write_text(json.dumps(thresholds_json))
    return str(model_path)


def test_load_class_thresholds_missing_file_returns_empty(tmp_path):
    model_path = _write_checkpoint_dir(tmp_path)
    assert load_class_thresholds(model_path) == {}


def test_load_class_thresholds_parses_values(tmp_path):
    model_path = _write_checkpoint_dir(tmp_path, {
        "Apple": {"threshold": 0.35, "insufficient_data": False, "n_gt_instances": 12},
        "San Pellegrino 6-pack": {"threshold": 0.9, "insufficient_data": True, "n_gt_instances": 1},
    })
    thresholds = load_class_thresholds(model_path)
    assert thresholds == {"Apple": 0.35, "San Pellegrino 6-pack": 0.9}


def test_load_class_thresholds_shared_across_models_in_same_dir(tmp_path):
    # class_thresholds.json lives next to the model, keyed by class name only —
    # two differently-named checkpoints in the same dir see the same file.
    (tmp_path / "class_thresholds.json").write_text(json.dumps({"Apple": {"threshold": 0.4}}))
    model_a = tmp_path / "modelA.pth"
    model_b = tmp_path / "modelB.pth"
    model_a.write_bytes(b"")
    model_b.write_bytes(b"")
    assert load_class_thresholds(str(model_a)) == {"Apple": 0.4}
    assert load_class_thresholds(str(model_b)) == {"Apple": 0.4}


def test_check_query_rows_passes_when_lengths_match():
    """3900 = 300 queries x 13 groups, checkpoint and build agreeing -- the flat
    slice rfdetr warns about is an identity here."""
    from control_models.base import check_query_rows

    check_query_rows("ckpt.pth", 3900, 3900)
    check_query_rows("ckpt.pth", None, 3900)      # nothing read, nothing to check


def test_check_query_rows_refuses_a_shorter_build():
    """A build with fewer queries would have rfdetr truncate 3900 rows to 1300,
    taking whole groups 0-3 as if they were one query set. Silent otherwise."""
    import pytest

    from control_models.base import check_query_rows

    with pytest.raises(RuntimeError, match="scrambling the query groups"):
        check_query_rows("ckpt.pth", 3900, 1300)
