"""
Download POPE (3 splits) + AMBER + the COCO val2014 image subset
referenced by those benchmarks. Designed to be idempotent and to
skip files that already exist on disk.

Usage (run on Colab or locally):
    python src/download_data.py --out data/
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


POPE_SPLITS = {
    "pope_random": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_random.json",
    "pope_popular": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_popular.json",
    "pope_adversarial": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_adversarial.json",
}

# AMBER's bench file plus the COCO image index.
AMBER_QUERY_URL = (
    "https://raw.githubusercontent.com/junyangwang0410/AMBER/main/data/query/query_discriminative.json"
)

# COCO val2014 images. We only fetch the images referenced by POPE/AMBER
# because the full zip is 6GB. See _coco_image_url.
COCO_VAL2014_BASE = "http://images.cocodataset.org/val2014"


def _download(url: str, dest: Path, chunk: int = 1 << 15) -> None:
    """Stream a URL to dest, skipping if the file already exists and is non-empty."""
    if dest.exists() and dest.stat().st_size > 0:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(tmp, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=dest.name, leave=False
        ) as pbar:
            for data in r.iter_content(chunk):
                f.write(data)
                pbar.update(len(data))
    tmp.replace(dest)


def _coco_image_url(image_id: int) -> str:
    """COCO val2014 files are named COCO_val2014_{image_id:012d}.jpg"""
    return f"{COCO_VAL2014_BASE}/COCO_val2014_{image_id:012d}.jpg"


def download_pope(out_dir: Path) -> list[dict]:
    """Download POPE jsonl for all 3 splits; return the combined records."""
    merged: list[dict] = []
    for split, url in POPE_SPLITS.items():
        dest = out_dir / "pope" / f"{split}.json"
        _download(url, dest)
        # POPE file is JSON-lines despite the .json extension.
        with open(dest, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rec["split"] = split
                merged.append(rec)
    print(f"POPE: {len(merged)} records across {len(POPE_SPLITS)} splits")
    return merged


def download_amber(out_dir: Path) -> list[dict]:
    """Download AMBER discriminative-task query file."""
    dest = out_dir / "amber" / "query_discriminative.json"
    _download(AMBER_QUERY_URL, dest)
    with open(dest, encoding="utf-8") as f:
        data = json.load(f)
    # Attach a split name for bookkeeping.
    for r in data:
        r["split"] = "amber"
    print(f"AMBER: {len(data)} records")
    return data


def download_coco_images(records: list[dict], out_dir: Path) -> None:
    """Download only the COCO val2014 images referenced by the given records.

    Records are expected to carry an "image" or "image_id" field. POPE uses
    "image": "COCO_val2014_000000xxx.jpg"; AMBER uses "image": "AMBER_x.jpg"
    (a separate image set we skip here; user should also fetch the AMBER
    image zip separately if running AMBER in full).
    """
    coco_dir = out_dir / "coco_val2014"
    coco_dir.mkdir(parents=True, exist_ok=True)

    needed: set[int] = set()
    for r in records:
        name = r.get("image") or r.get("image_id") or ""
        if isinstance(name, int):
            needed.add(name)
            continue
        name = str(name)
        if name.startswith("COCO_val2014_") and name.endswith(".jpg"):
            try:
                img_id = int(name.split("_")[-1].split(".")[0])
                needed.add(img_id)
            except ValueError:
                continue

    print(f"COCO val2014: fetching {len(needed)} unique images")
    for img_id in tqdm(sorted(needed), desc="coco"):
        dest = coco_dir / f"COCO_val2014_{img_id:012d}.jpg"
        try:
            _download(_coco_image_url(img_id), dest)
        except requests.HTTPError as e:
            # A handful of images may have been removed upstream; skip.
            print(f"  skip {img_id}: {e}", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data", type=Path, help="output root dir")
    ap.add_argument(
        "--skip-coco", action="store_true", help="only fetch jsonl, skip images"
    )
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    pope = download_pope(args.out)
    amber = download_amber(args.out)

    if not args.skip_coco:
        download_coco_images(pope, args.out)
        # AMBER has its own image set; users should fetch the zip from the
        # AMBER repo README and extract into data/amber/images.
        amber_images = args.out / "amber" / "images"
        if not amber_images.exists():
            print(
                "NOTE: AMBER images are not auto-downloaded. Please grab the "
                "image zip from https://github.com/junyangwang0410/AMBER and "
                f"extract into {amber_images}"
            )


if __name__ == "__main__":
    main()
