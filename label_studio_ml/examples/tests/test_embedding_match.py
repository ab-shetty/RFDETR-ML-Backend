import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cascade.embedding_match import embedding_agrees, load_reference_gallery, nearest_classes  # noqa: E402


def test_load_reference_gallery_missing_file_returns_empty():
    assert load_reference_gallery("/nonexistent/path/gallery.npz") == {}


def test_load_reference_gallery_roundtrip(tmp_path):
    path = str(tmp_path / "gallery.npz")
    np.savez(path, class_names=np.array(["Apple", "Banana"]), centroids=np.array([[1.0, 0.0], [0.0, 1.0]]))
    gallery = load_reference_gallery(path)
    assert set(gallery.keys()) == {"Apple", "Banana"}
    assert np.allclose(gallery["Apple"], [1.0, 0.0])


def test_nearest_classes_ranks_by_cosine_similarity():
    gallery = {"Apple": np.array([1.0, 0.0]), "Banana": np.array([0.0, 1.0])}
    top = nearest_classes(np.array([0.9, 0.1]), gallery, k=2)
    assert top[0][0] == "Apple"


def test_nearest_classes_empty_gallery_returns_empty():
    assert nearest_classes(np.array([1.0, 0.0]), {}, k=3) == []


def test_embedding_agrees_true_when_predicted_class_is_nearest():
    gallery = {"Apple": np.array([1.0, 0.0]), "Banana": np.array([0.0, 1.0])}
    assert embedding_agrees(np.array([0.9, 0.1]), gallery, "Apple", top_k=1) is True


def test_embedding_agrees_false_when_predicted_class_is_not_in_top_k():
    gallery = {"Apple": np.array([1.0, 0.0]), "Banana": np.array([0.0, 1.0]), "Cherry": np.array([0.0, 0.99])}
    assert embedding_agrees(np.array([0.0, 1.0]), gallery, "Apple", top_k=1) is False


def test_embedding_agrees_none_when_class_not_in_gallery():
    gallery = {"Banana": np.array([0.0, 1.0])}
    assert embedding_agrees(np.array([1.0, 0.0]), gallery, "Apple") is None
