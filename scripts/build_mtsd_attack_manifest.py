#!/usr/bin/env python3
"""Select a balanced MTSD attack set from clean SAM 3 successes."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/mtsd/subset/metadata/manifest.csv"
        ),
    )
    parser.add_argument(
        "--clean-results",
        type=Path,
        default=Path(
            "data/mtsd/results/clean_generic_full.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "data/mtsd/subset/metadata/"
            "attack_manifest_20_per_class.csv"
        ),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "data/mtsd/results/"
            "attack_manifest_summary.csv"
        ),
    )
    parser.add_argument("--per-class", type=int, default=20)
    parser.add_argument("--seed", type=int, default=2026)

    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    args = parse_args()

    manifest_rows = read_csv(args.manifest)
    clean_rows = read_csv(args.clean_results)

    manifest_by_sample = {
        row["sample_id"]: row
        for row in manifest_rows
    }

    clean_by_sample = {}
    duplicate_samples = set()

    for row in clean_rows:
        if row["prompt_type"] != "generic":
            continue

        sample_id = row["sample_id"]

        if sample_id in clean_by_sample:
            duplicate_samples.add(sample_id)

        clean_by_sample[sample_id] = row

    if duplicate_samples:
        raise RuntimeError(
            f"Duplicate clean results for "
            f"{len(duplicate_samples)} samples"
        )

    eligible_by_class: defaultdict[str, list[dict[str, str]]]
    eligible_by_class = defaultdict(list)

    missing_manifest_rows = 0

    for sample_id, clean_row in clean_by_sample.items():
        if clean_row["success_iou_050"] != "1":
            continue

        manifest_row = manifest_by_sample.get(sample_id)

        if manifest_row is None:
            missing_manifest_rows += 1
            continue

        merged = dict(manifest_row)

        merged.update(
            {
                "attack_prompt": clean_row["prompt"],
                "clean_detection_count": clean_row[
                    "detection_count"
                ],
                "clean_best_matching_index": clean_row[
                    "best_matching_index"
                ],
                "clean_best_matching_score": clean_row[
                    "best_matching_score"
                ],
                "clean_best_box_iou": clean_row[
                    "best_box_iou"
                ],
                "clean_pred_xmin": clean_row[
                    "best_pred_xmin"
                ],
                "clean_pred_ymin": clean_row[
                    "best_pred_ymin"
                ],
                "clean_pred_xmax": clean_row[
                    "best_pred_xmax"
                ],
                "clean_pred_ymax": clean_row[
                    "best_pred_ymax"
                ],
                "clean_gt_relative_area": clean_row[
                    "gt_relative_area"
                ],
            }
        )

        eligible_by_class[
            manifest_row["class_label"]
        ].append(merged)

    selected_rows = []
    summary_rows = []

    labels = sorted(
        eligible_by_class,
        key=lambda label: int(
            eligible_by_class[label][0]["class_rank"]
        ),
    )

    for label in labels:
        candidates = sorted(
            eligible_by_class[label],
            key=lambda row: row["sample_id"],
        )

        if len(candidates) < args.per_class:
            raise RuntimeError(
                f"{label} has only {len(candidates)} "
                f"clean successes; requested {args.per_class}"
            )

        rng = random.Random(f"{args.seed}:{label}")
        rng.shuffle(candidates)

        selected = candidates[: args.per_class]

        for attack_number, row in enumerate(
            selected,
            start=1,
        ):
            output_row = dict(row)
            output_row["attack_sample_number"] = attack_number
            output_row["attack_id"] = (
                f"class{int(row['class_rank']):02d}_"
                f"attack{attack_number:03d}_"
                f"{row['image_id']}"
            )
            selected_rows.append(output_row)

        summary_rows.append(
            {
                "class_rank": candidates[0]["class_rank"],
                "class_label": label,
                "fine_prompt": candidates[0]["fine_prompt"],
                "clean_successes_available": len(candidates),
                "selected_for_attack": len(selected),
            }
        )

    selected_rows.sort(
        key=lambda row: (
            int(row["class_rank"]),
            int(row["attack_sample_number"]),
        )
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.summary.parent.mkdir(parents=True, exist_ok=True)

    output_fields = list(selected_rows[0].keys())

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=output_fields,
        )
        writer.writeheader()
        writer.writerows(selected_rows)

    with args.summary.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "class_rank",
                "class_label",
                "fine_prompt",
                "clean_successes_available",
                "selected_for_attack",
            ],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print("Clean generic result rows:", len(clean_by_sample))
    print(
        "Total clean successes:",
        sum(len(rows) for rows in eligible_by_class.values()),
    )
    print("Classes:", len(labels))
    print("Selected attack targets:", len(selected_rows))
    print("Missing manifest rows:", missing_manifest_rows)

    print("\nSelected per class:")

    for row in summary_rows:
        print(
            f"{int(row['selected_for_attack']):>3} selected "
            f"from {int(row['clean_successes_available']):>3} "
            f"clean successes  {row['class_label']}"
        )

    print("\nAttack manifest:")
    print(args.output.resolve())
    print("Summary:")
    print(args.summary.resolve())


if __name__ == "__main__":
    main()
