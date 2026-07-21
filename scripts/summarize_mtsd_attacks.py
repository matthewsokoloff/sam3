#!/usr/bin/env python3
"""Summarize and plot the full MTSD adversarial attack sweep."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


MODE_ORDER = [
    "standard",
    "additive",
    "smooth_additive",
]

MODE_NAMES = {
    "standard": "Standard PGD",
    "additive": "Additive PGD",
    "smooth_additive": "Smooth additive PGD",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "data/mtsd/results/attack_full_5eps.csv"
        ),
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/mtsd_top15.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/mtsd/results/attack_full_summary"
        ),
    )
    parser.add_argument(
        "--expected-targets",
        type=int,
        default=300,
    )
    parser.add_argument(
        "--class-plot-epsilon",
        type=float,
        default=8.0,
    )

    return parser.parse_args()


def wilson_interval(
    successes: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0

    rate = successes / total
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

    return (
        max(0.0, center - margin),
        min(1.0, center + margin),
    )


def mean_interval(
    values: list[float],
) -> tuple[float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0

    value_mean = mean(values)

    if len(values) < 2:
        return value_mean, value_mean, value_mean

    margin = 1.96 * stdev(values) / math.sqrt(len(values))

    return (
        value_mean,
        value_mean - margin,
        value_mean + margin,
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: list[dict],
) -> None:
    if not rows:
        raise RuntimeError(f"No rows available for {path}")

    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize_group(
    rows: list[dict[str, str]],
) -> dict[str, float | int]:
    total = len(rows)

    successes_050 = sum(
        int(row["attack_success_iou_050"])
        for row in rows
    )
    successes_025 = sum(
        int(row["attack_success_iou_025"])
        for row in rows
    )

    success_rate = successes_050 / total
    ci_low, ci_high = wilson_interval(
        successes_050,
        total,
    )

    scores = [
        float(row["adversarial_best_matching_score"])
        for row in rows
    ]
    ious = [
        float(row["adversarial_best_box_iou"])
        for row in rows
    ]
    psnr_values = [
        float(row["psnr_db"])
        for row in rows
        if math.isfinite(float(row["psnr_db"]))
    ]
    runtimes = [
        float(row["runtime_seconds"])
        for row in rows
    ]
    raw_linf = [
        float(row["raw_linf_pixels"])
        for row in rows
    ]
    quantized_linf = [
        float(row["quantized_linf_pixels"])
        for row in rows
    ]

    score_mean, score_ci_low, score_ci_high = (
        mean_interval(scores)
    )

    no_detections = sum(
        int(row["adversarial_detection_count"]) == 0
        for row in rows
    )

    return {
        "total": total,
        "attack_successes_iou_050": successes_050,
        "attack_success_rate_iou_050": success_rate,
        "attack_success_ci_low_95": ci_low,
        "attack_success_ci_high_95": ci_high,
        "attack_success_rate_iou_025": (
            successes_025 / total
        ),
        "zero_detection_count": no_detections,
        "zero_detection_rate": no_detections / total,
        "mean_adversarial_box_iou": mean(ious),
        "median_adversarial_box_iou": median(ious),
        "mean_adversarial_matching_score": score_mean,
        "score_ci_low_95": score_ci_low,
        "score_ci_high_95": score_ci_high,
        "mean_psnr_db": mean(psnr_values),
        "mean_runtime_seconds": mean(runtimes),
        "mean_raw_linf_pixels": mean(raw_linf),
        "maximum_raw_linf_pixels": max(raw_linf),
        "mean_quantized_linf_pixels": mean(
            quantized_linf
        ),
    }


def validate_rows(
    rows: list[dict[str, str]],
    expected_targets: int,
) -> tuple[list[float], list[str]]:
    expected_modes = set(MODE_ORDER)
    modes = {row["mode"] for row in rows}

    if modes != expected_modes:
        raise RuntimeError(
            f"Expected modes {sorted(expected_modes)}, "
            f"found {sorted(modes)}"
        )

    epsilons = sorted(
        {float(row["epsilon_pixels"]) for row in rows}
    )

    keys = set()
    duplicate_keys = []

    for row in rows:
        key = (
            row["attack_id"],
            row["mode"],
            float(row["epsilon_pixels"]),
        )

        if key in keys:
            duplicate_keys.append(key)

        keys.add(key)

    if duplicate_keys:
        raise RuntimeError(
            f"Found {len(duplicate_keys)} duplicate conditions"
        )

    attack_ids = sorted(
        {row["attack_id"] for row in rows}
    )

    if len(attack_ids) != expected_targets:
        raise RuntimeError(
            f"Expected {expected_targets} attack targets, "
            f"found {len(attack_ids)}"
        )

    expected_rows = (
        expected_targets
        * len(MODE_ORDER)
        * len(epsilons)
    )

    if len(rows) != expected_rows:
        raise RuntimeError(
            f"Expected {expected_rows} rows, "
            f"found {len(rows)}"
        )

    grouped_counts = defaultdict(int)

    for row in rows:
        grouped_counts[
            (
                row["mode"],
                float(row["epsilon_pixels"]),
            )
        ] += 1

    for mode in MODE_ORDER:
        for epsilon in epsilons:
            count = grouped_counts[(mode, epsilon)]

            if count != expected_targets:
                raise RuntimeError(
                    f"{mode}, epsilon={epsilon:g}: "
                    f"expected {expected_targets}, found {count}"
                )

    return epsilons, attack_ids


def make_success_plot(
    summary_rows: list[dict],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5.5))

    for mode in MODE_ORDER:
        rows = [
            row
            for row in summary_rows
            if row["mode"] == mode
        ]
        rows.sort(key=lambda row: row["epsilon_pixels"])

        x_values = [
            row["epsilon_pixels"]
            for row in rows
        ]
        y_values = [
            row["attack_success_rate_iou_050"]
            for row in rows
        ]

        lower_errors = [
            row["attack_success_rate_iou_050"]
            - row["attack_success_ci_low_95"]
            for row in rows
        ]
        upper_errors = [
            row["attack_success_ci_high_95"]
            - row["attack_success_rate_iou_050"]
            for row in rows
        ]

        axis.errorbar(
            x_values,
            y_values,
            yerr=[lower_errors, upper_errors],
            label=MODE_NAMES[mode],
            capsize=3,
        )

    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel(
        "Perturbation budget epsilon (8-bit pixel units)"
    )
    axis.set_ylabel(
        "Conditional attack success rate at box IoU < 0.50"
    )
    axis.set_title(
        "SAM 3 Attack Success on Cleanly Detected MTSD Signs"
    )
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def make_score_plot(
    summary_rows: list[dict],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5.5))

    for mode in MODE_ORDER:
        rows = [
            row
            for row in summary_rows
            if row["mode"] == mode
        ]
        rows.sort(key=lambda row: row["epsilon_pixels"])

        x_values = [
            row["epsilon_pixels"]
            for row in rows
        ]
        y_values = [
            row["mean_adversarial_matching_score"]
            for row in rows
        ]

        lower_errors = [
            row["mean_adversarial_matching_score"]
            - row["score_ci_low_95"]
            for row in rows
        ]
        upper_errors = [
            row["score_ci_high_95"]
            - row["mean_adversarial_matching_score"]
            for row in rows
        ]

        axis.errorbar(
            x_values,
            y_values,
            yerr=[lower_errors, upper_errors],
            label=MODE_NAMES[mode],
            capsize=3,
        )

    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel(
        "Perturbation budget epsilon (8-bit pixel units)"
    )
    axis.set_ylabel(
        "Mean adversarial matching confidence"
    )
    axis.set_title(
        "SAM 3 Adversarial Confidence versus Perturbation Budget"
    )
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def make_psnr_plot(
    summary_rows: list[dict],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8, 5.5))

    for mode in MODE_ORDER:
        rows = [
            row
            for row in summary_rows
            if row["mode"] == mode
        ]
        rows.sort(key=lambda row: row["mean_psnr_db"])

        x_values = [
            row["mean_psnr_db"]
            for row in rows
        ]
        y_values = [
            row["attack_success_rate_iou_050"]
            for row in rows
        ]

        axis.plot(
            x_values,
            y_values,
            label=MODE_NAMES[mode],
        )

        for row in rows:
            axis.annotate(
                f"{row['epsilon_pixels']:g}/255",
                (
                    row["mean_psnr_db"],
                    row["attack_success_rate_iou_050"],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=8,
            )

    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Mean PSNR (dB)")
    axis.set_ylabel(
        "Conditional attack success rate at box IoU < 0.50"
    )
    axis.set_title(
        "Attack Effectiveness versus Image Distortion"
    )
    axis.grid(alpha=0.25)
    axis.legend()

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def make_class_plot(
    class_rows: list[dict],
    mode: str,
    epsilon: float,
    output_path: Path,
) -> None:
    rows = [
        row
        for row in class_rows
        if row["mode"] == mode
        and math.isclose(
            row["epsilon_pixels"],
            epsilon,
        )
    ]

    rows.sort(
        key=lambda row: row[
            "attack_success_rate_iou_050"
        ]
    )

    labels = [
        row["display_name"]
        for row in rows
    ]
    rates = [
        row["attack_success_rate_iou_050"]
        for row in rows
    ]

    figure, axis = plt.subplots(figsize=(10, 8))
    bars = axis.barh(labels, rates)

    axis.set_xlim(0.0, 1.0)
    axis.set_xlabel(
        "Conditional attack success rate at box IoU < 0.50"
    )
    axis.set_ylabel("MTSD sign class")
    axis.set_title(
        f"{MODE_NAMES[mode]} by Sign Class, "
        f"epsilon={epsilon:g}/255"
    )
    axis.grid(axis="x", alpha=0.25)

    axis.bar_label(
        bars,
        labels=[f"{rate:.0%}" for rate in rates],
        padding=4,
        fontsize=8,
    )

    figure.tight_layout()
    figure.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(args.input)

    epsilons, attack_ids = validate_rows(
        rows,
        args.expected_targets,
    )

    config = json.loads(
        args.config.read_text(encoding="utf-8")
    )

    display_names = {
        entry["label"]: entry["prompt"]
        for entry in config["classes"]
    }

    condition_groups = defaultdict(list)
    class_groups = defaultdict(list)

    for row in rows:
        mode = row["mode"]
        epsilon = float(row["epsilon_pixels"])
        label = row["class_label"]

        condition_groups[(mode, epsilon)].append(row)
        class_groups[(mode, epsilon, label)].append(row)

    condition_summary = []

    for mode in MODE_ORDER:
        for epsilon in epsilons:
            summary = summarize_group(
                condition_groups[(mode, epsilon)]
            )

            condition_summary.append(
                {
                    "mode": mode,
                    "mode_display_name": MODE_NAMES[mode],
                    "epsilon_pixels": epsilon,
                    **summary,
                }
            )

    class_summary = []

    for mode in MODE_ORDER:
        for epsilon in epsilons:
            labels = sorted(
                {
                    label
                    for current_mode, current_epsilon, label
                    in class_groups
                    if current_mode == mode
                    and current_epsilon == epsilon
                }
            )

            for label in labels:
                group = class_groups[
                    (mode, epsilon, label)
                ]
                summary = summarize_group(group)

                class_summary.append(
                    {
                        "mode": mode,
                        "mode_display_name": MODE_NAMES[mode],
                        "epsilon_pixels": epsilon,
                        "class_rank": int(group[0]["class_rank"]),
                        "class_label": label,
                        "display_name": display_names.get(
                            label,
                            label,
                        ),
                        **summary,
                    }
                )

    condition_csv = (
        args.output_dir / "condition_summary.csv"
    )
    class_csv = (
        args.output_dir / "class_summary.csv"
    )

    write_csv(condition_csv, condition_summary)
    write_csv(class_csv, class_summary)

    success_plot = (
        args.output_dir
        / "attack_success_vs_epsilon.png"
    )
    score_plot = (
        args.output_dir
        / "adversarial_confidence_vs_epsilon.png"
    )
    psnr_plot = (
        args.output_dir
        / "psnr_vs_attack_success.png"
    )

    make_success_plot(
        condition_summary,
        success_plot,
    )
    make_score_plot(
        condition_summary,
        score_plot,
    )
    make_psnr_plot(
        condition_summary,
        psnr_plot,
    )

    for mode in MODE_ORDER:
        class_plot_path = (
            args.output_dir
            / (
                f"class_attack_success_"
                f"{mode}_epsilon_"
                f"{args.class_plot_epsilon:g}.png"
            )
        )

        make_class_plot(
            class_summary,
            mode,
            args.class_plot_epsilon,
            class_plot_path,
        )

    text_path = args.output_dir / "summary.txt"

    lines = [
        "MTSD adversarial attack sweep",
        f"Rows: {len(rows)}",
        f"Attack targets: {len(attack_ids)}",
        f"Epsilons: {epsilons}",
        "",
        (
            "Attack success is conditional on SAM 3 "
            "successfully detecting the target in the clean image."
        ),
        "",
    ]

    for mode in MODE_ORDER:
        lines.append(MODE_NAMES[mode])

        for row in condition_summary:
            if row["mode"] != mode:
                continue

            lines.append(
                "  "
                f"epsilon={row['epsilon_pixels']:g}/255  "
                f"ASR={row['attack_success_rate_iou_050']:.1%}  "
                f"zero_detection="
                f"{row['zero_detection_rate']:.1%}  "
                f"mean_IoU="
                f"{row['mean_adversarial_box_iou']:.3f}  "
                f"mean_score="
                f"{row['mean_adversarial_matching_score']:.3f}  "
                f"PSNR={row['mean_psnr_db']:.2f} dB"
            )

        lines.append("")

    total_runtime = sum(
        float(row["runtime_seconds"])
        for row in rows
    )

    lines.append(
        f"Total recorded attack runtime: "
        f"{total_runtime / 3600:.2f} hours"
    )

    text_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(text_path.read_text(encoding="utf-8"))

    print("Saved:")
    print(condition_csv.resolve())
    print(class_csv.resolve())
    print(success_plot.resolve())
    print(score_plot.resolve())
    print(psnr_plot.resolve())


if __name__ == "__main__":
    main()
