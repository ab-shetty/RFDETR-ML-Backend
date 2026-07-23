import os
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from cascade import gpt_tiebreaker  # noqa: E402


CROP = Image.new("RGB", (10, 10))


def _mock_response(content):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def _patched_client(content):
    client = MagicMock()
    client.chat.completions.create.return_value = _mock_response(content)
    return client


def test_no_candidates_returns_none_without_calling_api():
    with patch("cascade.gpt_tiebreaker._get_client") as mock_get:
        assert gpt_tiebreaker.ask(CROP, []) is None
        mock_get.assert_not_called()


def test_valid_match_within_candidates():
    with patch("cascade.gpt_tiebreaker._get_client", return_value=_patched_client('{"match": "Apple"}')):
        assert gpt_tiebreaker.ask(CROP, ["Apple", "Banana"]) == "Apple"


def test_null_match_returns_none():
    with patch("cascade.gpt_tiebreaker._get_client", return_value=_patched_client('{"match": null}')):
        assert gpt_tiebreaker.ask(CROP, ["Apple", "Banana"]) is None


def test_out_of_candidate_match_treated_as_unresolved():
    with patch("cascade.gpt_tiebreaker._get_client", return_value=_patched_client('{"match": "Cherry"}')):
        assert gpt_tiebreaker.ask(CROP, ["Apple", "Banana"]) is None


def test_empty_content_returns_none_not_crash():
    # This is the real-world failure: a reasoning model can return empty
    # content (finish_reason=length) when the token budget is exhausted by
    # reasoning. Must degrade to None (-> escalate), never raise.
    with patch("cascade.gpt_tiebreaker._get_client", return_value=_patched_client("")):
        assert gpt_tiebreaker.ask(CROP, ["Apple", "Banana"]) is None


def test_none_content_returns_none_not_crash():
    with patch("cascade.gpt_tiebreaker._get_client", return_value=_patched_client(None)):
        assert gpt_tiebreaker.ask(CROP, ["Apple", "Banana"]) is None


def test_markdown_fenced_json_is_parsed():
    with patch("cascade.gpt_tiebreaker._get_client", return_value=_patched_client('```json\n{"match": "Apple"}\n```')):
        assert gpt_tiebreaker.ask(CROP, ["Apple", "Banana"]) == "Apple"


def test_api_exception_returns_none_not_crash():
    client = MagicMock()
    client.chat.completions.create.side_effect = RuntimeError("network down")
    with patch("cascade.gpt_tiebreaker._get_client", return_value=client):
        assert gpt_tiebreaker.ask(CROP, ["Apple", "Banana"]) is None
