#!/usr/bin/env python3
"""Run resumable SAM 3 attacks on clean-success MTSD targets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3.model.sam3_image_processor import Sam3Processor
from sam3_attack import AttackConfig, build_model, run_attack
from sam3_attack.io import load_image, tensor_to_pil


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}

OUTPUT_FIELDS = [
    "attack_id",
    "sample_id",
    "image_id",
    "class_rank",
    "class_label",
    "prompt",
    "mode",
    "epsilon_pixels",
    "step_size_pixels",
    "steps",
    "restarts",
    "seed",
    "smooth_kernel_size",
    "smooth_sigma",
    "clean_best_box_iou",
    "clean_best_matching_score",
    "adversarial_detection_count",
    "adversarial_best_matching_index",
    "adversarial_best_box_iou",
    "adversarial_best_matching_score",
    "adversarial_highest_score",
    "attack_success_iou_025",
    "attack_success_iou_050",
    "raw_linf_pixels",
    "quantized_linf_pixels",
    "quantized_mean_absolute_pixels",
    "quantized_mean_added_pixels",
    "quantized_negative_pixel_count",
    "psnr_db",
    "best_objective",
    "best_restart",
    "runtime_seconds",
    "saved_adversarial_path",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(
            "data/mtsd/subset/metadata/"
            "attack_manifest_20_per_class.csv"
        ),
    )
    parser.add_argument(
        "--images",
        type=Path,
        default=Path("data/mtsd/subset/images"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/sam3/sam3.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/mtsd/results/attack_sweep.csv"),
    )
    parser.add_argument(
        "--image-output",
        type=Path,
        default=Path("data/mtsd/results/attack_examples"),
    )

    parser.add_argument(
        "--modes",
        nargs="+",
        choices=("standard", "additive", "smooth_additive"),
        default=("standard", "additive", "smooth_additive"),
    )
    parser.add_argument(
        "--epsilons",
        nargs="+",
        type=float,
        default=(1.0, 4.0, 8.0),
    )

    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument(
        "--amp",
        choices=("auto", "bf16", "fp16", "none"),
        default="auto",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--smooth-kernel-size", type=int, default=31)
    parser.add_argument("--smooth-sigma", type=float, default=7.0)

    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=1,
        help="Use the first N attack targets per class; 0 means all.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum selected images; 0 means unlimited.",
    )
    parser.add_argument(
        "--save-images",
        type=int,
        default=45,
        help="Maximum adversarial PNGs to save.",
    )

    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)

    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def select_rows(
    rows: list[dict[str, str]],
    samples_per_class: int,
    limit: int,
) -> list[dict[str, str]]:
    if samples_per_class > 0:
        counts: defaultdict[str, int] = defaultdict(int)
        selected = []

        for row in rows:
            label = row["class_label"]

            if counts[label] >= samples_per_class:
                continue

            selected.append(row)
            counts[label] += 1

        rows = selected

    if limit > 0:
        rows = rows[:limit]

    return rows


def build_image_index(image_dir: Path) -> dict[str, Path]:
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)

    return {
        path.stem: path
        for path in image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    }


def condition_key(
    attack_id: str,
    mode: str,
    epsilon: float,
    step_size: float,
    steps: int,
    restarts: int,
) -> tuple[str, str, str, str, str, str]:
    return (
        attack_id,
        mode,
        f"{epsilon:g}",
        f"{step_size:g}",
        str(steps),
        str(restarts),
    )


def load_completed(
    path: Path,
) -> set[tuple[str, str, str, str, str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()

    completed = set()

    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            completed.add(
                condition_key(
                    row["attack_id"],
                    row["mode"],
                    float(row["epsilon_pixels"]),
                    float(row["step_size_pixels"]),
                    int(row["steps"]),
                    int(row["restarts"]),
                )
            )

    return completed


def stable_seed(
    base_seed: int,
    attack_id: str,
    mode: str,
    epsilon: float,
) -> int:
    text = f"{attack_id}:{mode}:{epsilon:g}".encode("utf-8")
    digest = hashlib.sha256(text).digest()
    offset = int.from_bytes(digest[:4], byteorder="little")
    return base_seed + offset % 1_000_000


def mask_to_box(
    mask: torch.Tensor,
) -> tuple[float, float, float, float] | None:
    positions = torch.nonzero(mask.bool(), as_tuple=False)

    if positions.numel() == 0:
        return None

    ymin = int(positions[:, 0].min().item())
    ymax = int(positions[:, 0].max().item()) + 1
    xmin = int(positions[:, 1].min().item())
    xmax = int(positions[:, 1].max().item()) + 1

    return float(xmin), float(ymin), float(xmax), float(ymax)


def box_area(box: tuple[float, float, float, float]) -> float:
    xmin, ymin, xmax, ymax = box
    return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)


def box_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_width = max(
        0.0,
        min(ax2, bx2) - max(ax1, bx1),
    )
    intersection_height = max(
        0.0,
        min(ay2, by2) - max(ay1, by1),
    )
    intersection = intersection_width * intersection_height

    union = box_area(box_a) + box_area(box_b) - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


def evaluate_adversarial(
    processor: Sam3Processor,
    image: Image.Image,
    prompt: str,
    gt_box: tuple[float, float, float, float],
) -> dict[str, float | int | str]:
    with torch.inference_mode():
        state = processor.set_image(image)
        result = processor.set_text_prompt(
            prompt=prompt,
            state=state,
        )

    masks = result["masks"].detach().cpu()
    scores = result["scores"].detach().float().cpu().reshape(-1)

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    if masks.ndim == 2:
        masks = masks.unsqueeze(0)

    detection_count = int(scores.numel())

    if detection_count == 0:
        return {
            "adversarial_detection_count": 0,
            "adversarial_best_matching_index": "",
            "adversarial_best_box_iou": 0.0,
            "adversarial_best_matching_score": 0.0,
            "adversarial_highest_score": 0.0,
            "attack_success_iou_025": 1,
            "attack_success_iou_050": 1,
        }

    best_index = None
    best_iou = 0.0
    best_score = 0.0

    for index in range(detection_count):
        predicted_box = mask_to_box(masks[index])
        current_iou = (
            0.0
            if predicted_box is None
            else box_iou(gt_box, predicted_box)
        )
        current_score = float(scores[index].item())

        if (
            current_iou > best_iou
            or (
                current_iou == best_iou
                and current_score > best_score
            )
        ):
            best_index = index
            best_iou = current_iou
            best_score = current_score

    return {
        "adversarial_detection_count": detection_count,
        "adversarial_best_matching_index": (
            "" if best_index is None else best_index
        ),
        "adversarial_best_box_iou": best_iou,
        "adversarial_best_matching_score": best_score,
        "adversarial_highest_score": float(scores.max().item()),
        "attack_success_iou_025": int(best_iou < 0.25),
        "attack_success_iou_050": int(best_iou < 0.50),
    }


def image_quality(
    clean: Image.Image,
    adversarial: Image.Image,
) -> dict[str, float | int]:
    clean_array = np.asarray(clean, dtype=np.uint8).astype(np.int16)
    adversarial_array = (
        np.asarray(adversarial, dtype=np.uint8).astype(np.int16)
    )

    difference = adversarial_array - clean_array
    absolute = np.abs(difference)

    mse = float(
        np.mean(
            (
                difference.astype(np.float32) / 255.0
            )
            ** 2
        )
    )

    psnr = (
        float("inf")
        if mse == 0.0
        else 10.0 * math.log10(1.0 / mse)
    )

    return {
        "quantized_linf_pixels": float(absolute.max()),
        "quantized_mean_absolute_pixels": float(absolute.mean()),
        "quantized_mean_added_pixels": float(
            np.maximum(difference, 0).mean()
        ),
        "quantized_negative_pixel_count": int(
            (difference < 0).sum()
        ),
        "psnr_db": psnr,
    }


def log_error(
    path: Path,
    attack_id: str,
    mode: str,
    epsilon: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + "=" * 80 + "\n")
        handle.write(
            f"attack_id={attack_id} "
            f"mode={mode} epsilon={epsilon:g}\n"
        )
        handle.write(traceback.format_exc())


def main() -> None:
    args = parse_args()

    rows = select_rows(
        read_csv(args.manifest),
        args.samples_per_class,
        args.limit,
    )
    image_index = build_image_index(args.images)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.image_output.mkdir(parents=True, exist_ok=True)

    errors_path = args.output.with_name(
        args.output.stem + "_errors.log"
    )

    completed = load_completed(args.output)

    total_conditions = (
        len(rows)
        * len(args.modes)
        * len(args.epsilons)
    )

    print("Selected images:", len(rows))
    print("Modes:", list(args.modes))
    print("Epsilons:", list(args.epsilons))
    print("Total conditions:", total_conditions)
    print("Already completed:", len(completed))

    print("\nLoading SAM 3...")
    model = build_model(args.checkpoint)

    processor = Sam3Processor(
        model,
        confidence_threshold=args.threshold,
    )

    print("GPU:", torch.cuda.get_device_name(0))
    print("SAM 3 loaded.\n")

    write_header = (
        not args.output.exists()
        or args.output.stat().st_size == 0
    )

    saved_count = len(
        list(args.image_output.rglob("*.png"))
    )
    completed_this_run = 0
    condition_number = 0

    with args.output.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as output_handle:
        writer = csv.DictWriter(
            output_handle,
            fieldnames=OUTPUT_FIELDS,
        )

        if write_header:
            writer.writeheader()
            output_handle.flush()

        for row in rows:
            attack_id = row["attack_id"]
            image_id = row["image_id"]
            prompt = row["attack_prompt"]

            image_path = image_index.get(image_id)

            if image_path is None:
                print("MISSING IMAGE:", image_id)
                continue

            clean_tensor, clean_pil = load_image(image_path)

            gt_box = (
                float(row["xmin"]),
                float(row["ymin"]),
                float(row["xmax"]),
                float(row["ymax"]),
            )

            for mode in args.modes:
                for epsilon in args.epsilons:
                    condition_number += 1

                    key = condition_key(
                        attack_id,
                        mode,
                        epsilon,
                        args.step_size,
                        args.steps,
                        args.restarts,
                    )

                    if key in completed:
                        continue

                    seed = stable_seed(
                        args.seed,
                        attack_id,
                        mode,
                        epsilon,
                    )

                    config = AttackConfig(
                        epsilon_pixels=epsilon,
                        step_size_pixels=args.step_size,
                        steps=args.steps,
                        restarts=args.restarts,
                        confidence_threshold=args.threshold,
                        amp=args.amp,
                        seed=seed,
                        perturbation_mode=mode,
                        smooth_kernel_size=args.smooth_kernel_size,
                        smooth_sigma=args.smooth_sigma,
                    )

                    print(
                        f"\n[{condition_number}/{total_conditions}] "
                        f"{attack_id} mode={mode} "
                        f"epsilon={epsilon:g}/255",
                        flush=True,
                    )

                    try:
                        torch.cuda.synchronize()
                        start_time = time.perf_counter()

                        result = run_attack(
                            model,
                            clean_tensor,
                            [prompt],
                            config,
                        )

                        adversarial_pil = tensor_to_pil(
                            result.adversarial_image
                        )

                        evaluation = evaluate_adversarial(
                            processor,
                            adversarial_pil,
                            prompt,
                            gt_box,
                        )

                        torch.cuda.synchronize()
                        runtime_seconds = (
                            time.perf_counter() - start_time
                        )

                        quality = image_quality(
                            clean_pil,
                            adversarial_pil,
                        )

                        saved_path = ""

                        if saved_count < args.save_images:
                            epsilon_tag = (
                                f"{epsilon:g}".replace(".", "p")
                            )
                            path = (
                                args.image_output
                                / mode
                                / f"epsilon_{epsilon_tag}"
                                / f"{attack_id}.png"
                            )
                            path.parent.mkdir(
                                parents=True,
                                exist_ok=True,
                            )
                            adversarial_pil.save(path)
                            saved_path = str(path.resolve())
                            saved_count += 1

                        output_row = {
                            "attack_id": attack_id,
                            "sample_id": row["sample_id"],
                            "image_id": image_id,
                            "class_rank": row["class_rank"],
                            "class_label": row["class_label"],
                            "prompt": prompt,
                            "mode": mode,
                            "epsilon_pixels": epsilon,
                            "step_size_pixels": args.step_size,
                            "steps": args.steps,
                            "restarts": args.restarts,
                            "seed": seed,
                            "smooth_kernel_size": (
                                args.smooth_kernel_size
                            ),
                            "smooth_sigma": args.smooth_sigma,
                            "clean_best_box_iou": row[
                                "clean_best_box_iou"
                            ],
                            "clean_best_matching_score": row[
                                "clean_best_matching_score"
                            ],
                            **evaluation,
                            "raw_linf_pixels": float(
                                result.delta.abs().max().item()
                                * 255.0
                            ),
                            **quality,
                            "best_objective": (
                                result.best_objective
                            ),
                            "best_restart": result.restart,
                            "runtime_seconds": runtime_seconds,
                            "saved_adversarial_path": saved_path,
                        }

                        writer.writerow(output_row)
                        output_handle.flush()

                        completed.add(key)
                        completed_this_run += 1

                        print(
                            "attack_success="
                            f"{evaluation['attack_success_iou_050']} "
                            "adv_iou="
                            f"{evaluation['adversarial_best_box_iou']:.3f} "
                            "adv_score="
                            f"{evaluation['adversarial_best_matching_score']:.3f} "
                            f"PSNR={quality['psnr_db']:.2f} dB "
                            f"time={runtime_seconds:.1f}s",
                            flush=True,
                        )

                    except torch.cuda.OutOfMemoryError:
                        print(
                            f"CUDA OUT OF MEMORY: {attack_id} "
                            f"{mode} epsilon={epsilon:g}. "
                            "Stopping this process."
                        )
                        log_error(
                            errors_path,
                            attack_id,
                            mode,
                            epsilon,
                        )
                        raise

                    except Exception:
                        print(
                            f"FAILED: {attack_id} "
                            f"{mode} epsilon={epsilon:g}"
                        )
                        log_error(
                            errors_path,
                            attack_id,
                            mode,
                            epsilon,
                        )

                    finally:
                        torch.cuda.empty_cache()

    print("\nAttack sweep pass finished.")
    print("Completed this run:", completed_this_run)
    print("Results:", args.output.resolve())
    print("Errors:", errors_path.resolve())


if __name__ == "__main__":
    main()
