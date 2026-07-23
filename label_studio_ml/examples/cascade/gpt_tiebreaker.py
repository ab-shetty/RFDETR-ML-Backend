"""GPT-5-mini vision tiebreaker — the expensive, slow, but smart signal in
the cascade. Deliberately called from only one place (pipeline.py's
"ambiguous" branch), never per-detection unconditionally, to keep API call
volume and latency bounded.

Requires OPENAI_API_KEY in the environment (loaded from ~/.env in this
deployment). The client is constructed lazily so importing this module
doesn't require a key to be present — useful for tests, which mock ask()
directly.
"""
import base64
import io
import json
import logging
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

MODEL = "gpt-5-mini"
MAX_OUTPUT_TOKENS = 50

_client = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI
        _client = OpenAI()  # reads OPENAI_API_KEY from the environment
    return _client


def _encode_crop(crop: Image.Image) -> str:
    buf = io.BytesIO()
    crop.convert("RGB").save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def ask(crop: Image.Image, candidates: List[str]) -> Optional[str]:
    """Ask GPT-5-mini which of `candidates` (if any) the crop shows.

    Returns the chosen class name (must be one of `candidates`), or None if
    the model says none of them match, or if the call fails for any reason
    (network, auth, malformed response) — callers treat None as "still
    unresolved", which routes to human review rather than a wrong auto-accept.
    """
    if not candidates:
        return None

    prompt = (
        "You are verifying an object-detection box cropped from a grocery-store photo. "
        "Which of these candidate product labels, if any, matches what's shown in the image? "
        f"Candidates: {json.dumps(candidates)}. "
        'Respond with strict JSON only: {"match": "<exact candidate string>"} or {"match": null} '
        "if none of the candidates match or the crop is too ambiguous to tell."
    )

    try:
        client = _get_client()
        response = client.chat.completions.create(
            model=MODEL,
            max_completion_tokens=MAX_OUTPUT_TOKENS,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{_encode_crop(crop)}"},
                        },
                    ],
                }
            ],
        )
        raw = response.choices[0].message.content
        parsed = json.loads(raw)
        match = parsed.get("match")
        if match in candidates:
            return match
        if match is not None:
            logger.warning(f"GPT tiebreaker returned an out-of-candidate match {match!r}, treating as unresolved")
        return None
    except Exception as e:
        logger.warning(f"GPT tiebreaker call failed, treating as unresolved: {e}")
        return None
