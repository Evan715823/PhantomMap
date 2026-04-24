"""
Tolerant parsing of VLM outputs into (answer, bbox, raw) tuples.

The elicitation prompt in prompts.py asks for:
    yes|no\n{"bbox_2d": [x1, y1, x2, y2]}
but real models emit noisier strings: trailing prose, markdown fences,
single quotes, slight key-name variants. This module absorbs that noise
while refusing anything that cannot be made into 4 integers in
image-pixel range.
"""

from __future__ import annotations
import json
import re
from dataclasses import dataclass
from typing import Optional


# Any JSON-like object that looks like a bbox. We capture non-greedy
# to tolerate multiple objects in the same string.
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)

# Fallback: 4 comma-separated numbers in square brackets.
_RAW_BBOX_RE = re.compile(
    r"\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)"
    r"\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]"
)

_YES_RE = re.compile(r"\byes\b", re.IGNORECASE)
_NO_RE = re.compile(r"\bno\b", re.IGNORECASE)


@dataclass
class ParsedOutput:
    """Result of parsing one VLM generation.

    answer: "yes" / "no" / "unknown"
    bbox:   (x1, y1, x2, y2) floats in image-pixel coordinates, or None
    raw:    the original string (kept for error analysis)
    """

    answer: str
    bbox: Optional[tuple[float, float, float, float]]
    raw: str


def _coerce_answer(text: str) -> str:
    """The model's first yes/no wins; default to 'unknown' if neither appears."""
    y = _YES_RE.search(text)
    n = _NO_RE.search(text)
    if y and n:
        return "yes" if y.start() < n.start() else "no"
    if y:
        return "yes"
    if n:
        return "no"
    return "unknown"


def _try_json_bbox(s: str) -> Optional[tuple[float, float, float, float]]:
    """Iterate over {...} substrings; return the first that yields a valid bbox."""
    # Prefer JSON-like dict with any of the common bbox keys first.
    for m in _JSON_OBJ_RE.finditer(s):
        chunk = m.group(0)
        # Normalise single quotes and python-style True/False just in case.
        chunk2 = chunk.replace("'", '"')
        try:
            obj = json.loads(chunk2)
        except Exception:
            continue
        if not isinstance(obj, dict):
            continue
        for key in ("bbox_2d", "bbox", "box", "bounding_box"):
            if key in obj and _is_bbox_list(obj[key]):
                return tuple(float(x) for x in obj[key])  # type: ignore[return-value]
    return None


def _try_raw_bbox(s: str) -> Optional[tuple[float, float, float, float]]:
    m = _RAW_BBOX_RE.search(s)
    if not m:
        return None
    return tuple(float(g) for g in m.groups())  # type: ignore[return-value]


def _is_bbox_list(v) -> bool:
    if not isinstance(v, (list, tuple)):
        return False
    if len(v) != 4:
        return False
    try:
        [float(x) for x in v]
        return True
    except Exception:
        return False


def parse(text: str) -> ParsedOutput:
    """Main entry point: accept a raw VLM string, return structured output."""
    answer = _coerce_answer(text)
    bbox = _try_json_bbox(text) or _try_raw_bbox(text)
    return ParsedOutput(answer=answer, bbox=bbox, raw=text)


def validate_bbox(
    bbox: tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    min_side: int = 4,
) -> Optional[tuple[float, float, float, float]]:
    """Return the bbox clipped to image bounds if it is sane, else None.

    A bbox is sane when: 0 <= x1 < x2 <= W, 0 <= y1 < y2 <= H, and both
    sides are at least min_side pixels. We clip to image bounds first
    (Qwen occasionally overshoots by 1-2 px).
    """
    x1, y1, x2, y2 = bbox
    # Some models emit (y1, x1, y2, x2). We detect and correct only the
    # obvious case where x > W but y <= H and swapping makes it sane.
    if x1 > img_w and y1 <= img_h and x2 > img_w and y2 <= img_h:
        x1, y1, x2, y2 = y1, x1, y2, x2
    # Clip.
    x1 = max(0.0, min(float(img_w), x1))
    y1 = max(0.0, min(float(img_h), y1))
    x2 = max(0.0, min(float(img_w), x2))
    y2 = max(0.0, min(float(img_h), y2))
    # Enforce ordering.
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if (x2 - x1) < min_side or (y2 - y1) < min_side:
        return None
    return (x1, y1, x2, y2)


if __name__ == "__main__":
    # Smoke test.
    cases = [
        'yes\n{"bbox_2d": [10, 20, 100, 200]}',
        'YES.\n```json\n{"bbox_2d": [10, 20, 100, 200]}\n```',
        "no",
        "yes {'bbox': [1,2,3,4]}",
        'yes\n{"box": [5, 6, 50, 60]}',
        "garbage output",
    ]
    for c in cases:
        out = parse(c)
        print(f"IN: {c!r}\n -> answer={out.answer} bbox={out.bbox}\n")
