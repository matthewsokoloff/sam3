#!/usr/bin/env python3
"""Evaluate clean SAM 3 traffic-sign detection on the MTSD subset."""

from __future__ import annotations

import argparse
import csv
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")

OUTPUT_FIELDS = [
    "sample_id",
    "image_id",
    "split",
    "class_rank",
    "class_label",
    "prompt_type",
    "prompt",
    "confidence_threshold",
    "image_width",
    "image_height",
    "gt_xmin",
    "gt_ymin",
    "gt_xmax",
    "gt_ymax",
    "gt_box_area",
    "gt_relative_area",
    "detection_count",
    "best_matching_index",
    "best_matching_score",
    "best_box_iou",
    "best_pred_xmin",
    "best_pred_ymin",
    "best_pred_xmax",
    "best_pred_ymax",
    "highest_score",
    "highest_score_box_iou",
    "success_iou_025",
    "success_iou_050",
    "clean_failure_iou_050",
    "image_encode_seconds",
    "prompt_seconds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/mtsd/subset/metadata/manifest.csv"),
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
        default=Path("data/mtsd/results/clean_predictions.csv"),
    )
    parser.add_argument(
        "--overlays",
        type=Path,
        default=Path("data/mtsd/results/clean_overlays"),
    )
    parser.add_argument(
        "--prompt-mode",
        choices=("fine", "generic", "both"),
        default="both",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=0,
        help="Use the first N samples per class; 0 means all samples.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of images after class sampling; 0 means unlimited.",
    )
    parser.add_argument(
        "--save-overlays",
        type=int,
        default=30,
        help="Maximum number of prompt-result overlays to save.",
    )

    return parser.parse_args()


def read_manifest(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest not found: {path}")

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
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    index = {}

    for path in image_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            index[path.stem] = path

    return index


def load_completed(path: Path) -> set[tuple[str, str]]:
    if not path.is_file() or path.stat().st_size == 0:
        return set()

    completed = set()

    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            sample_id = row.get("sample_id")
            prompt_type = row.get("prompt_type")

            if sample_id and prompt_type:
                completed.add((sample_id, prompt_type))

    return completed


def prompt_specs(
    row: dict[str, str],
    prompt_mode: str,
) -> list[tuple[str, str]]:
    prompts = []

    if prompt_mode in {"fine", "both"}:
        prompts.append(("fine", row["fine_prompt"]))

    if prompt_mode in {"generic", "both"}:
        prompts.append(("generic", row["generic_prompt"]))

    return prompts


def clamp_box(
    box: tuple[float, float, float, float],
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    xmin, ymin, xmax, ymax = box

    xmin = min(max(xmin, 0.0), float(width))
    ymin = min(max(ymin, 0.0), float(height))
    xmax = min(max(xmax, 0.0), float(width))
    ymax = min(max(ymax, 0.0), float(height))

    return xmin, ymin, xmax, ymax


def box_area(box: tuple[float, float, float, float]) -> float:
    xmin, ymin, xmax, ymax = box
    return max(0.0, xmax - xmin) * max(0.0, ymax - ymin)


def box_iou(
    box_a: tuple[float, float, float, float],
    box_b: tuple[float, float, float, float],
) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    intersection_width = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    intersection_height = max(0.0, min(ay2, by2) - max(ay1, by1))
    intersection = intersection_width * intersection_height

    union = box_area(box_a) + box_area(box_b) - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


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


def normalize_predictions(
    result: dict,
) -> tuple[torch.Tensor, torch.Tensor]:
    if "masks" not in result or "scores" not in result:
        raise KeyError(
            f"Expected masks and scores. Result keys: {list(result.keys())}"
        )

    masks = result["masks"].detach().cpu()
    scores = result["scores"].detach().float().cpu().reshape(-1)

    if scores.numel() == 0:
        return masks, scores

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    if masks.ndim == 2:
        masks = masks.unsqueeze(0)

    if masks.ndim != 3:
        raise RuntimeError(
            f"Unexpected mask shape: {tuple(masks.shape)}"
        )

    if masks.shape[0] != scores.numel():
        raise RuntimeError(
            "Mask/score count mismatch: "
            f"{masks.shape[0]} masks and {scores.numel()} scores"
        )

    return masks.bool(), scores


def evaluate_predictions(
    masks: torch.Tensor,
    scores: torch.Tensor,
    gt_box: tuple[float, float, float, float],
) -> dict[str, float | int | str]:
    detection_count = int(scores.numel())

    if detection_count == 0:
        return {
            "detection_count": 0,
            "best_matching_index": "",
            "best_matching_score": 0.0,
            "best_box_iou": 0.0,
            "best_pred_xmin": "",
            "best_pred_ymin": "",
            "best_pred_xmax": "",
            "best_pred_ymax": "",
            "highest_score": 0.0,
            "highest_score_box_iou": 0.0,
            "success_iou_025": 0,
            "success_iou_050": 0,
            "clean_failure_iou_050": 1,
            "_best_mask": None,
        }

    best_index = None
    best_score = 0.0
    best_iou = 0.0
    best_box = None

    highest_score_index = int(scores.argmax().item())
    highest_score = float(scores[highest_score_index].item())
    highest_score_iou = 0.0

    for index in range(detection_count):
        predicted_box = mask_to_box(masks[index])

        if predicted_box is None:
            current_iou = 0.0
        else:
            current_iou = box_iou(gt_box, predicted_box)

        current_score = float(scores[index].item())

        if index == highest_score_index:
            highest_score_iou = current_iou

        if (
            current_iou > best_iou
            or (
                current_iou == best_iou
                and current_score > best_score
            )
        ):
            best_index = index
            best_score = current_score
            best_iou = current_iou
            best_box = predicted_box

    success_025 = int(best_iou >= 0.25)
    success_050 = int(best_iou >= 0.50)

    return {
        "detection_count": detection_count,
        "best_matching_index": (
            "" if best_index is None else best_index
        ),
        "best_matching_score": best_score,
        "best_box_iou": best_iou,
        "best_pred_xmin": "" if best_box is None else best_box[0],
        "best_pred_ymin": "" if best_box is None else best_box[1],
        "best_pred_xmax": "" if best_box is None else best_box[2],
        "best_pred_ymax": "" if best_box is None else best_box[3],
        "highest_score": highest_score,
        "highest_score_box_iou": highest_score_iou,
        "success_iou_025": success_025,
        "success_iou_050": success_050,
        "clean_failure_iou_050": 1 - success_050,
        "_best_mask": (
            None if best_index is None else masks[best_index]
        ),
    }


def save_overlay(
    image: Image.Image,
    gt_box: tuple[float, float, float, float],
    predicted_box: tuple[float, float, float, float] | None,
    predicted_mask: torch.Tensor | None,
    output_path: Path,
) -> None:
    image_array = np.asarray(image).copy()

    if predicted_mask is not None:
        mask_array = predicted_mask.numpy().astype(np.uint8)

        if mask_array.shape != (image.height, image.width):
            mask_image = Image.fromarray(mask_array * 255)
            mask_image = mask_image.resize(
                image.size,
                resample=Image.Resampling.NEAREST,
            )
            mask_array = (
                np.asarray(mask_image).astype(np.uint8) > 0
            )
        else:
            mask_array = mask_array.astype(bool)

        overlay = image_array.astype(np.float32)
        overlay[mask_array, 0] = (
            0.55 * overlay[mask_array, 0] + 0.45 * 255.0
        )
        overlay[mask_array, 1] *= 0.55
        overlay[mask_array, 2] *= 0.55
        image_array = np.clip(overlay, 0, 255).astype(np.uint8)

    output = Image.fromarray(image_array)
    draw = ImageDraw.Draw(output)

    line_width = max(2, round(min(image.size) / 500))

    draw.rectangle(
        gt_box,
        outline="lime",
        width=line_width,
    )

    if predicted_box is not None:
        draw.rectangle(
            predicted_box,
            outline="red",
            width=line_width,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.save(output_path)


def log_error(
    path: Path,
    sample_id: str,
    prompt_type: str,
    prompt: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n" + "=" * 80 + "\n")
        handle.write(
            f"sample_id={sample_id} "
            f"prompt_type={prompt_type} "
            f"prompt={prompt!r}\n"
        )
        handle.write(traceback.format_exc())


def main() -> None:
    args = parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}"
        )

    all_rows = read_manifest(args.manifest)
    rows = select_rows(
        all_rows,
        args.samples_per_class,
        args.limit,
    )
    image_index = build_image_index(args.images)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.overlays.mkdir(parents=True, exist_ok=True)

    errors_path = args.output.with_name(
        args.output.stem + "_errors.log"
    )

    completed = load_completed(args.output)

    print("Manifest rows selected:", len(rows))
    print("Already completed prompt evaluations:", len(completed))
    print("Confidence threshold:", args.threshold)
    print("Prompt mode:", args.prompt_mode)

    print("\nLoading SAM 3...")

    model = build_sam3_image_model(
        checkpoint_path=str(args.checkpoint),
        load_from_HF=False,
        device="cuda",
        eval_mode=True,
        enable_segmentation=True,
        enable_inst_interactivity=False,
        compile=False,
    )

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

    overlay_count = len(list(args.overlays.glob("*.jpg")))
    completed_this_run = 0
    image_count = 0

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

        with torch.inference_mode():
            for row_number, row in enumerate(rows, start=1):
                sample_id = row["sample_id"]
                image_id = row["image_id"]

                pending_prompts = [
                    spec
                    for spec in prompt_specs(row, args.prompt_mode)
                    if (sample_id, spec[0]) not in completed
                ]

                if not pending_prompts:
                    continue

                image_path = image_index.get(image_id)

                if image_path is None:
                    print(f"MISSING IMAGE: {image_id}")
                    continue

                try:
                    with Image.open(image_path) as opened:
                        image = opened.convert("RGB")

                    width, height = image.size

                    gt_box = clamp_box(
                        (
                            float(row["xmin"]),
                            float(row["ymin"]),
                            float(row["xmax"]),
                            float(row["ymax"]),
                        ),
                        width,
                        height,
                    )

                    gt_area = box_area(gt_box)
                    relative_area = (
                        gt_area / float(width * height)
                        if width > 0 and height > 0
                        else 0.0
                    )

                    torch.cuda.synchronize()
                    encode_start = time.perf_counter()
                    state = processor.set_image(image)
                    torch.cuda.synchronize()
                    encode_seconds = time.perf_counter() - encode_start

                except Exception:
                    print(f"FAILED IMAGE ENCODE: {sample_id}")

                    for prompt_type, prompt in pending_prompts:
                        log_error(
                            errors_path,
                            sample_id,
                            prompt_type,
                            prompt,
                        )

                    continue

                for prompt_type, prompt in pending_prompts:
                    try:
                        torch.cuda.synchronize()
                        prompt_start = time.perf_counter()

                        result = processor.set_text_prompt(
                            prompt=prompt,
                            state=state,
                        )

                        torch.cuda.synchronize()
                        prompt_seconds = (
                            time.perf_counter() - prompt_start
                        )

                        masks, scores = normalize_predictions(result)

                        metrics = evaluate_predictions(
                            masks,
                            scores,
                            gt_box,
                        )

                        best_mask = metrics.pop("_best_mask")

                        predicted_box = None

                        if metrics["best_pred_xmin"] != "":
                            predicted_box = (
                                float(metrics["best_pred_xmin"]),
                                float(metrics["best_pred_ymin"]),
                                float(metrics["best_pred_xmax"]),
                                float(metrics["best_pred_ymax"]),
                            )

                        output_row = {
                            "sample_id": sample_id,
                            "image_id": image_id,
                            "split": row["split"],
                            "class_rank": row["class_rank"],
                            "class_label": row["class_label"],
                            "prompt_type": prompt_type,
                            "prompt": prompt,
                            "confidence_threshold": args.threshold,
                            "image_width": width,
                            "image_height": height,
                            "gt_xmin": gt_box[0],
                            "gt_ymin": gt_box[1],
                            "gt_xmax": gt_box[2],
                            "gt_ymax": gt_box[3],
                            "gt_box_area": gt_area,
                            "gt_relative_area": relative_area,
                            **metrics,
                            "image_encode_seconds": encode_seconds,
                            "prompt_seconds": prompt_seconds,
                        }

                        writer.writerow(output_row)
                        output_handle.flush()

                        completed.add((sample_id, prompt_type))
                        completed_this_run += 1

                        if overlay_count < args.save_overlays:
                            overlay_path = args.overlays / (
                                f"{sample_id}_{prompt_type}.jpg"
                            )

                            save_overlay(
                                image,
                                gt_box,
                                predicted_box,
                                best_mask,
                                overlay_path,
                            )

                            overlay_count += 1

                        print(
                            f"[{row_number:04d}/{len(rows):04d}] "
                            f"{prompt_type:<7} "
                            f"IoU={metrics['best_box_iou']:.3f} "
                            f"score={metrics['best_matching_score']:.3f} "
                            f"detections={metrics['detection_count']:<3} "
                            f"{row['class_label']}",
                            flush=True,
                        )

                    except Exception:
                        print(
                            f"FAILED PROMPT: "
                            f"{sample_id} {prompt_type} {prompt!r}"
                        )

                        log_error(
                            errors_path,
                            sample_id,
                            prompt_type,
                            prompt,
                        )

                image_count += 1

                if image_count % 20 == 0:
                    torch.cuda.empty_cache()

    print("\nBenchmark pass finished.")
    print("Prompt evaluations completed this run:", completed_this_run)
    print("Results:", args.output.resolve())
    print("Overlays:", args.overlays.resolve())
    print("Errors:", errors_path.resolve())


if __name__ == "__main__":
    main()
