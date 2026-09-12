#!/usr/bin/env python3
"""Render MAIRA-2 boxes over original images for qualitative review."""

import argparse
import csv
import html
import json
from collections import defaultdict
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
    parser.add_argument("--paired", action="store_true", help="Show source and corrupted rows side by side")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.only_corrupted and not args.paired:
        rows = [row for row in rows if row.get("variant") == "corrupted_negative"]
    if args.limit:
        rows = rows[: args.limit]

    cards, font = [], ImageFont.load_default()

    def render(row, name, colour):
        source = resolve_image(row["image_path"], args.input_csv, args.image_root)
        image = load_cxr_image(source)
        draw = ImageDraw.Draw(image)
        gt_boxes = json.loads(row.get("gt_boxes_xyxy") or "[]")
        for box in gt_boxes:
            x1, y1, x2, y2 = box
            draw.rectangle((x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height), outline="lime", width=max(2, image.width // 300))
        boxes = json.loads(row.get("maira_boxes_original_xyxy") or "[]")
        for box in boxes:
            x1, y1, x2, y2 = box
            draw.rectangle((x1 * image.width, y1 * image.height, x2 * image.width, y2 * image.height), fill=None, outline=colour, width=max(2, image.width // 300))
        title = f"{row.get('variant','')} | {row.get('tested_claim', row.get('maira_input_claim',''))}"
        draw.rectangle((0, 0, image.width, 24), fill="black")
        draw.text((4, 5), title[:160], fill="white", font=font)
        image.thumbnail((1200, 1200))
        image.save(args.output_dir / name, quality=92)
        return title, len(boxes)

    if args.paired:
        pairs = defaultdict(dict)
        for row in rows:
            pairs[row.get("pair_id", row.get("sample_id", "unknown"))][row.get("variant", "")] = row
        for index, (pair_id, variants) in enumerate(pairs.items(), 1):
            source_key = "source_positive" if "source_positive" in variants else "source_negative_statement"
            if source_key not in variants or "corrupted_negative" not in variants:
                continue
            source_row, corrupted_row = variants[source_key], variants["corrupted_negative"]
            source_name = f"{index:04d}_{pair_id}_source.jpg"
            corrupted_name = f"{index:04d}_{pair_id}_corrupted.jpg"
            source_title, source_count = render(source_row, source_name, "deepskyblue")
            corrupted_title, corrupted_count = render(corrupted_row, corrupted_name, "red")
            kind = corrupted_row.get("negative_type", "unknown")
            cards.append(
                f'<article class="pair"><h2>{html.escape(pair_id)}</h2><b>{html.escape(kind)}</b><div class="panels">'
                f'<section><h3>Original/source</h3><img src="{html.escape(source_name)}"><p>{html.escape(source_title)}</p><small>boxes={source_count}</small></section>'
                f'<section><h3>Corrupted/unsupported</h3><img src="{html.escape(corrupted_name)}"><p>{html.escape(corrupted_title)}</p><small>boxes={corrupted_count}</small></section>'
                '</div></article>'
            )
    else:
        for index, row in enumerate(rows, 1):
            name = f"{index:04d}_{row.get('negative_type','unknown')}_{row.get('sample_id','')}.jpg"
            title, count = render(row, name, "red")
            cards.append(f'<article><img src="{html.escape(name)}"><p>{html.escape(title)}</p><small>boxes={count}</small></article>')
    page = """<!doctype html><meta charset="utf-8"><title>MAIRA-2 results</title><style>body{font:14px Arial;background:#eee;margin:20px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(420px,1fr));gap:14px}article{background:white;padding:10px}img{width:100%;height:auto}.pair{grid-column:span 1}.panels{display:grid;grid-template-columns:1fr 1fr;gap:10px}h2,h3{margin:.3em 0}</style><h1>MAIRA-2 phrase-grounding results</h1><p>Green: source GT. Blue: source prediction. Red: corrupted-claim prediction.</p><main>""" + "".join(cards) + "</main>"
    (args.output_dir / "index.html").write_text(page, encoding="utf-8")
    print(f"Rendered {len(cards)} images to {args.output_dir}")


if __name__ == "__main__":
    main()
