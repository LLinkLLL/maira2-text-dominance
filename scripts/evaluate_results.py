#!/usr/bin/env python3
"""Summarise model-independent MAIRA-2 text-dominance metrics."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes"}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with args.input_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    groups = defaultdict(list)
    for row in rows:
        groups[(row.get("negative_type", "unknown"), row.get("variant", "unknown"))].append(row)

    descriptive = []
    for (negative_type, variant), group in sorted(groups.items()):
        valid = [row for row in group if truthy(row.get("maira_output_valid", ""))]
        with_box = [row for row in valid if int(row.get("maira_box_count") or 0) > 0]
        descriptive.append({
            "negative_type": negative_type,
            "variant": variant,
            "support_label": group[0].get("support_label", ""),
            "n": len(group),
            "valid_output_rate": len(valid) / len(group),
            "box_rate_all": len(with_box) / len(group),
            "box_rate_valid": len(with_box) / len(valid) if valid else "",
            "mean_box_count_valid": sum(int(r.get("maira_box_count") or 0) for r in valid) / len(valid) if valid else "",
        })

    pairs = defaultdict(dict)
    for row in rows:
        pairs[row.get("pair_id", row.get("sample_id", ""))][row.get("variant", "")] = row
    paired = []
    for pair_id, variants in pairs.items():
        source_key = "source_positive" if "source_positive" in variants else "source_negative_statement"
        if source_key not in variants or "corrupted_negative" not in variants:
            continue
        source, corrupted = variants[source_key], variants["corrupted_negative"]
        source_box = int(source.get("maira_box_count") or 0) > 0
        corrupted_box = int(corrupted.get("maira_box_count") or 0) > 0
        changed_to = corrupted.get("changed_to", "").lower()
        negative_type = corrupted.get("negative_type", "")
        spatial_field = "maira_predicted_side" if negative_type == "laterality_flip" else "maira_predicted_vertical"
        text_following = (
            corrupted.get(spatial_field, "").lower() == changed_to
            if negative_type in {"laterality_flip", "vertical_location_flip"} and changed_to
            else ""
        )
        paired.append({
            "pair_id": pair_id,
            "negative_type": negative_type,
            "source_variant": source_key,
            "source_has_box": source_box,
            "corrupted_has_box": corrupted_box,
            "both_have_box": source_box and corrupted_box,
            "text_following": text_following,
            "source_side": source.get("maira_predicted_side", ""),
            "corrupted_side": corrupted.get("maira_predicted_side", ""),
            "source_vertical": source.get("maira_predicted_vertical", ""),
            "corrupted_vertical": corrupted.get("maira_predicted_vertical", ""),
        })

    pair_summary = []
    for negative_type in sorted({row["negative_type"] for row in paired}):
        group = [row for row in paired if row["negative_type"] == negative_type]
        applicable = [row for row in group if row["text_following"] != ""]
        pair_summary.append({
            "negative_type": negative_type,
            "n_pairs": len(group),
            "source_box_rate": sum(row["source_has_box"] for row in group) / len(group),
            "incorrect_claim_box_rate": sum(row["corrupted_has_box"] for row in group) / len(group),
            "both_box_rate": sum(row["both_have_box"] for row in group) / len(group),
            "text_following_n": len(applicable),
            "text_following_rate": sum(row["text_following"] for row in applicable) / len(applicable) if applicable else "",
        })

    write_csv(args.output_dir / "descriptive_statistics.csv", descriptive)
    write_csv(args.output_dir / "paired_results.csv", paired)
    write_csv(args.output_dir / "paired_statistics.csv", pair_summary)
    (args.output_dir / "summary.json").write_text(
        json.dumps({"descriptive": descriptive, "paired": pair_summary}, indent=2), encoding="utf-8"
    )
    print(json.dumps(pair_summary, indent=2))


if __name__ == "__main__":
    main()

