#!/usr/bin/env python3
"""Summarize the full clean MTSD benchmark and plot failure by class."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median

import matplotlib.pyplot as plt


def wilson_interval(
    failures: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0

    rate = failures / total
    denominator = 1.0 + z**2 / total

    center = (
        rate + z**2 / (2.0 * total)
    ) / denominator

    margin = (
        z
        * math.sqrt(
            rate * (1.0 - rate) / total
            + z**2 / (4.0 * total**2)
        )
        / denominator
    )

    return max(0.0, center - margin), min(1.0, center + margin)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/mtsd/results/clean_generic_full.csv"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mtsd_top15.json"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "data/mtsd/results/"
            "clean_generic_class_summary.csv"
        ),
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=Path(
            "data/mtsd/results/"
            "clean_generic_failure_rate.png"
        ),
    )
    parser.add_argument(
        "--expected-per-class",
        type=int,
        default=200,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.input.open(encoding="utf-8") as handle:
        all_rows = list(csv.DictReader(handle))

    rows = [
        row
        for row in all_rows
        if row["prompt_type"] == "generic"
    ]

    config = json.loads(
        args.config.read_text(encoding="utf-8")
    )

    display_names = {
        entry["label"]: entry["prompt"]
        for entry in config["classes"]
    }

    duplicate_keys = set()
    seen_keys = set()

    for row in rows:
        key = (row["sample_id"], row["prompt_type"])

        if key in seen_keys:
            duplicate_keys.add(key)

        seen_keys.add(key)

    if duplicate_keys:
        raise RuntimeError(
            f"Found {len(duplicate_keys)} duplicate result rows"
        )

    grouped = defaultdict(list)

    for row in rows:
        grouped[row["class_label"]].append(row)

    if len(grouped) != 15:
        raise RuntimeError(
            f"Expected 15 classes, found {len(grouped)}"
        )

    summary_rows = []

    for label, class_rows in grouped.items():
        total = len(class_rows)

        if total != args.expected_per_class:
            raise RuntimeError(
                f"{label} has {total} rows; "
                f"expected {args.expected_per_class}"
            )

        successes_050 = sum(
            int(row["success_iou_050"])
            for row in class_rows
        )
        successes_025 = sum(
            int(row["success_iou_025"])
            for row in class_rows
        )

        failures_050 = total - successes_050

        no_detections = sum(
            int(row["detection_count"]) == 0
            for row in class_rows
        )

        failure_rate = failures_050 / total
        ci_low, ci_high = wilson_interval(
            failures_050,
            total,
        )

        best_ious = [
            float(row["best_box_iou"])
            for row in class_rows
        ]

        matched_scores = [
            float(row["best_matching_score"])
            for row in class_rows
        ]

        summary_rows.append(
            {
                "class_rank": int(class_rows[0]["class_rank"]),
                "class_label": label,
                "display_name": display_names.get(label, label),
                "total": total,
                "successes_iou_050": successes_050,
                "failures_iou_050": failures_050,
                "failure_rate_iou_050": failure_rate,
                "failure_ci_low_95": ci_low,
                "failure_ci_high_95": ci_high,
                "success_rate_iou_025": successes_025 / total,
                "no_detection_count": no_detections,
                "no_detection_rate": no_detections / total,
                "mean_best_box_iou": mean(best_ious),
                "median_best_box_iou": median(best_ious),
                "mean_matching_score": mean(matched_scores),
            }
        )

    summary_rows.sort(key=lambda row: row["class_rank"])

    args.output_csv.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(summary_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    overall_total = len(rows)
    overall_successes = sum(
        int(row["success_iou_050"])
        for row in rows
    )
    overall_failures = overall_total - overall_successes
    overall_failure_rate = overall_failures / overall_total

    print("Generic-prompt clean benchmark")
    print("Rows:", overall_total)
    print("Classes:", len(summary_rows))
    print("Successful detections:", overall_successes)
    print("Failures:", overall_failures)
    print(
        "Overall failure rate:",
        f"{overall_failure_rate:.3%}",
    )

    print("\nPer-class failure rate:")

    for row in sorted(
        summary_rows,
        key=lambda item: item["failure_rate_iou_050"],
        reverse=True,
    ):
        print(
            f"{row['failure_rate_iou_050']:6.1%}  "
            f"({row['failures_iou_050']:3}/"
            f"{row['total']:3})  "
            f"{row['display_name']}"
        )

    plot_rows = sorted(
        summary_rows,
        key=lambda item: item["failure_rate_iou_050"],
    )

    labels = [
        row["display_name"]
        for row in plot_rows
    ]
    rates = [
        row["failure_rate_iou_050"]
        for row in plot_rows
    ]

    lower_errors = [
        row["failure_rate_iou_050"]
        - row["failure_ci_low_95"]
        for row in plot_rows
    ]
    upper_errors = [
        row["failure_ci_high_95"]
        - row["failure_rate_iou_050"]
        for row in plot_rows
    ]

    figure, axis = plt.subplots(figsize=(10, 8))

    bars = axis.barh(
        labels,
        rates,
        xerr=[lower_errors, upper_errors],
        capsize=3,
    )

    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel("Clean failure rate at box IoU < 0.50")
    axis.set_ylabel("MTSD sign class")
    axis.set_title(
        "SAM 3 Clean Failure Rate by Traffic-Sign Class\n"
        "Generic prompt: “traffic sign”; 200 targets per class"
    )
    axis.grid(axis="x", alpha=0.25)

    axis.bar_label(
        bars,
        labels=[f"{rate:.1%}" for rate in rates],
        padding=4,
        fontsize=8,
    )

    figure.tight_layout()
    figure.savefig(
        args.output_plot,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)

    print("\nSaved summary:")
    print(args.output_csv.resolve())
    print("Saved plot:")
    print(args.output_plot.resolve())


if __name__ == "__main__":
    main()
