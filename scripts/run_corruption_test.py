#!/usr/bin/env python3
"""Run deterministic MAIRA-2 phrase grounding over a corruption-pair CSV."""

import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


RESULT_FIELDS = [
    "maira_input_claim", "maira_raw_output", "maira_parsed_text",
    "maira_boxes_crop_xyxy", "maira_boxes_original_xyxy", "maira_box_count",
    "maira_output_valid", "maira_predicted_side", "maira_predicted_vertical",
    "maira_error", "maira_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--model-id", default="microsoft/maira-2")
    parser.add_argument("--claim-column", default="tested_claim")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--save-every", type=int, default=10)
    parser.add_argument("--max-new-tokens", type=int, default=150)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def load_rows(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader), list(reader.fieldnames or [])


def write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def resolve_image(row: dict[str, str], csv_path: Path, image_root: Path | None) -> Path:
    raw = row["image_path"].replace("\\", "/")
    candidate = Path(raw)
    attempts = []
    if image_root:
        attempts.extend([image_root / raw, image_root / candidate.name])
    if not candidate.is_absolute():
        attempts.append(csv_path.parent / candidate)
    attempts.append(candidate)
    for attempt in attempts:
        if attempt.is_file():
            return attempt.resolve()
    raise FileNotFoundError(f"Image not found. Tried: {attempts}")


def flatten_prediction(prediction: Any) -> tuple[str, list[list[float]]]:
    if isinstance(prediction, tuple):
        prediction = [prediction]
    texts, boxes = [], []
    for item in prediction if isinstance(prediction, list) else []:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            continue
        text, item_boxes = item
        texts.append(str(text))
        if item_boxes:
            boxes.extend([[float(v) for v in box] for box in item_boxes])
    return " ".join(texts), boxes


def spatial_labels(boxes: list[list[float]]) -> tuple[str, str]:
    if not boxes:
        return "none", "none"
    xs = [(b[0] + b[2]) / 2 for b in boxes]
    ys = [(b[1] + b[3]) / 2 for b in boxes]
    # Standard frontal CXR display: image right corresponds to patient left.
    side = "left" if all(x > 0.5 for x in xs) else "right" if all(x < 0.5 for x in xs) else "mixed"
    vertical = "upper" if all(y < 0.5 for y in ys) else "lower" if all(y > 0.5 for y in ys) else "mixed"
    return side, vertical


def main() -> None:
    args = parse_args()
    rows, input_fields = load_rows(args.input_csv)
    if not rows:
        raise ValueError("Input CSV is empty")
    required = {"image_path", args.claim_column}
    missing = required - set(input_fields)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if args.limit:
        rows = rows[: args.limit]

    fields = input_fields + [field for field in RESULT_FIELDS if field not in input_fields]
    if args.resume and args.output_csv.exists():
        old_rows, _ = load_rows(args.output_csv)
        if len(old_rows) > len(rows):
            raise ValueError("Resume output has more rows than the selected input")
        rows[: len(old_rows)] = old_rows

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()

    for index, row in enumerate(tqdm(rows, total=len(rows), desc="MAIRA-2")):
        if args.resume and (row.get("maira_output_valid") or row.get("maira_error")):
            continue
        started = time.perf_counter()
        row.update({field: "" for field in RESULT_FIELDS})
        claim = row[args.claim_column].strip()
        row["maira_input_claim"] = claim
        try:
            image_path = resolve_image(row, args.input_csv, args.image_root)
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            width, height = image.size
            inputs = processor.format_and_preprocess_phrase_grounding_input(
                frontal_image=image, phrase=claim, return_tensors="pt"
            ).to("cuda")
            with torch.inference_mode():
                output = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    num_beams=1,
                    use_cache=True,
                )
            prompt_length = inputs["input_ids"].shape[-1]
            decoded = processor.decode(output[0][prompt_length:], skip_special_tokens=True)
            parsed = processor.convert_output_to_plaintext_or_grounded_sequence(decoded)
            parsed_text, crop_boxes = flatten_prediction(parsed)
            original_boxes = [
                list(processor.adjust_box_for_original_image_size(tuple(box), width, height))
                for box in crop_boxes
            ]
            side, vertical = spatial_labels(original_boxes)
            row.update({
                "maira_raw_output": decoded,
                "maira_parsed_text": parsed_text,
                "maira_boxes_crop_xyxy": json.dumps(crop_boxes),
                "maira_boxes_original_xyxy": json.dumps(original_boxes),
                "maira_box_count": len(original_boxes),
                "maira_output_valid": True,
                "maira_predicted_side": side,
                "maira_predicted_vertical": vertical,
            })
        except Exception as error:  # Preserve progress and diagnose individual samples.
            row["maira_output_valid"] = False
            row["maira_error"] = f"{type(error).__name__}: {error}"
            print(f"\nRow {index} failed: {row['maira_error']}", file=sys.stderr)
            if isinstance(error, torch.cuda.OutOfMemoryError):
                write_rows(args.output_csv, rows[: index + 1], fields)
                raise
        finally:
            row["maira_seconds"] = round(time.perf_counter() - started, 4)
        if (index + 1) % args.save_every == 0:
            write_rows(args.output_csv, rows[: index + 1], fields)

    write_rows(args.output_csv, rows, fields)
    print(f"Saved {len(rows)} rows to {args.output_csv}")


if __name__ == "__main__":
    main()
