#!/usr/bin/env python3
"""Create a self-contained CSV/image subset for transfer to the GPU server."""

import argparse
import csv
import shutil
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-pairs", type=int)
    args = parser.parse_args()

    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
        fields = list(rows[0]) if rows else []
    if not rows or "image_path" not in fields:
        raise ValueError("Input CSV must contain at least one row and an image_path column")

    if args.max_pairs:
        keep = []
        pair_ids = []
        for row in rows:
            pair_id = row.get("pair_id") or row.get("sample_id") or str(len(pair_ids))
            if pair_id not in pair_ids:
                if len(pair_ids) >= args.max_pairs:
                    continue
                pair_ids.append(pair_id)
            if pair_id in pair_ids:
                keep.append(row)
        rows = keep

    images_dir = args.output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    copied = {}
    for row in rows:
        source = Path(row["image_path"].replace("\\", "/"))
        if not source.is_file():
            raise FileNotFoundError(source)
        key = str(source.resolve()).lower()
        if key not in copied:
            destination = images_dir / source.name
            if destination.exists() and destination.stat().st_size != source.stat().st_size:
                destination = images_dir / f"{len(copied):06d}_{source.name}"
            shutil.copy2(source, destination)
            copied[key] = destination.name
        row["image_path"] = f"images/{copied[key]}"

    output_csv = args.output_dir / "corruption_pairs.csv"
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows, {len(copied)} images to {args.output_dir}")


if __name__ == "__main__":
    main()

