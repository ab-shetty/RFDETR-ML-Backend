"""Cover what is left of the shelf-tag module: the reader.

The map-based naming this used to feed (lookup_class, propose_from_tags, SKU
correction) was measured against human labels on two held-out store visits and
removed -- see cascade/box_naming.py, which replaces it. The reader itself is
still the cheapest way to find out what is on a shelf, so its two contracts are
pinned here: normalisation, and that a bad response is silence rather than an
exception.
"""
import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cascade import shelf_tags  # noqa: E402


def test_normalize_tag_strips_price_and_uppercases():
    assert shelf_tags.normalize_tag("Vanilla Latte $2.99") == "VANILLA LATTE"
    assert shelf_tags.normalize_tag("COLD BREW WITH OATMILK $2-69") == "COLD BREW WITH OATMILK"
    assert shelf_tags.normalize_tag("Sparkling Ginger + Lemon") == "SPARKLING GINGER + LEMON"


def _client_returning(content):
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
    return client


def test_detect_tags_parses_positions():
    body = '{"tags":[{"name":"VERY GREEN JUICE","x":0.8,"y":0.4}]}'
    with patch.object(shelf_tags, "_get_client", return_value=_client_returning(body)):
        tags = shelf_tags.detect_tags(Image.new("RGB", (50, 50)))
    assert tags == [{"name": "VERY GREEN JUICE", "x": 0.8, "y": 0.4}]


def test_detect_tags_tolerates_a_fenced_answer():
    body = '```json\n{"tags":[{"name":"MANGO","x":0.1,"y":0.2}]}\n```'
    with patch.object(shelf_tags, "_get_client", return_value=_client_returning(body)):
        assert len(shelf_tags.detect_tags(Image.new("RGB", (50, 50)))) == 1


def test_detect_tags_returns_nothing_on_a_bad_answer():
    """A frame with no readable tags and a frame the call failed on both come
    back empty -- callers treat the reader as optional, never as required."""
    with patch.object(shelf_tags, "_get_client", return_value=_client_returning("not json")):
        assert shelf_tags.detect_tags(Image.new("RGB", (50, 50))) == []
