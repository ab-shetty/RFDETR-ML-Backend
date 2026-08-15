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
import os
from typing import List, Optional

from PIL import Image

logger = logging.getLogger(__name__)

# Overridable for the same reason as shelf_tags.MODEL: the offline harness
# swaps it to measure a candidate model, production default is unchanged.
MODEL = os.getenv("GPT_TIEBREAKER_MODEL", "gpt-5.6-luna")
# gpt-5-mini is a reasoning model: max_completion_tokens covers reasoning
# tokens AND the visible answer. A small budget (e.g. 50) gets entirely
# consumed by reasoning, returning empty content with finish_reason=length.
# reasoning_effort="minimal" keeps reasoning tokens ~0 for this simple
# classification, and the budget leaves ample room for the tiny JSON answer.
MAX_OUTPUT_TOKENS = 2000
# The gpt-5.6 family renamed the cheapest reasoning setting: "minimal" is
# rejected with a 400 and "none" means what it used to. Picked from the model
# id so switching models does not silently start paying for reasoning, or
# fail every call -- which is exactly what the first Luna dry run did.
REASONING_EFFORT = os.getenv(
    "GPT_TIEBREAKER_REASONING_EFFORT",
    "none" if MODEL.startswith("gpt-5.6") else "minimal")

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
            reasoning_effort=REASONING_EFFORT,
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
        raw = (response.choices[0].message.content or "").strip()
        # Defensive: strip a ```json ... ``` fence if the model wraps its answer.
        if raw.startswith("```"):
            raw = raw.split("```")[1].removeprefix("json").strip() if "```" in raw[3:] else raw.strip("`")
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
