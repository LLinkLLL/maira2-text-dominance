#!/usr/bin/env python3
"""Render MAIRA-2 boxes over original images for qualitative review."""

import argparse
import csv
import html
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from image_utils import load_cxr_image


def resolve_image(raw: str, csv_path: Path, image_root: Path | None) -> Path:
    path = Path(raw.replace("\\", "/"))
    attempts = ([image_root / raw, image_root / path.name] if image_root else []) + [csv_path.parent / path, path]
    for attempt in attempts:
        if attempt.is_file():
            return attempt
    raise FileNotFoundError(
        f"Could not resolve image {raw!r}. If results and images are in different "
        "directories, pass --image-root pointing to the transferred pilot directory."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--only-corrupted", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.only_corrupted:
        rows = [row for row in rows if row.get("variant") == "corrupted_negative"]
    if args.limit:
        rows = rows[: args.limit]

    cards, font = [], ImageFont.load_default()
    for index, row in enumerate(rows, 1):
        source = resolve_image(row["image_path"], args.input_csv, args.image_root)
        image = load_cxr_image(source)
        draw = ImageDraw.Draw(image)
        boxes = json.loads(row.get("maira_boxes_original_xyxy") or "[]")
        for box in boxes:
            x1, y1, x2, y2 = box
            draw.rectangle((x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height), fill=None, outline="red", width=max(2, image.width // 300))
        title = f"{row.get('variant','')} | {row.get('tested_claim', row.get('maira_input_claim',''))}"
        draw.rectangle((0, 0, image.width, 24), fill="black")
        draw.text((4, 5), title[:160], fill="white", font=font)
        name = f"{index:04d}_{row.get('negative_type','unknown')}_{row.get('sample_id','')}.jpg"
        image.thumbnail((1200, 1200))
        image.save(args.output_dir / name, quality=92)
        cards.append(f'<article><img src="{html.escape(name)}"><p>{html.escape(title)}</p><small>boxes={len(boxes)}</small></article>')
    page = """<!doctype html><meta charset="utf-8"><title>MAIRA-2 results</title><style>body{font:14px Arial;background:#eee;margin:20px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}article{background:white;padding:8px}img{width:100%;height:auto}</style><h1>MAIRA-2 phrase-grounding results</h1><main>""" + "".join(cards) + "</main>"
    (args.output_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"Rendered {len(cards)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
