#!/usr/bin/env python3
"""Evaluate clean and adversarial PNGs using SAM 3's official processor."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor


def binary_iou(mask_a: torch.Tensor, mask_b: torch.Tensor) -> float:
    mask_a = mask_a.bool()
    mask_b = mask_b.bool()

    intersection = torch.logical_and(mask_a, mask_b).sum().float()
    union = torch.logical_or(mask_a, mask_b).sum().float()

    if union.item() == 0:
        return 1.0

    return float((intersection / union).item())


def save_overlay(
    image: Image.Image,
    mask: torch.Tensor,
    output_path: Path,
) -> None:
    image_array = np.asarray(image).astype(np.float32) / 255.0
    mask_array = mask.detach().cpu().numpy().squeeze().astype(bool)

    overlay = image_array.copy()

    # Red overlay on the segmented region.
    overlay[mask_array, 0] = (
        0.55 * overlay[mask_array, 0] + 0.45
    )
    overlay[mask_array, 1] *= 0.55
    overlay[mask_array, 2] *= 0.55

    output = Image.fromarray(
        np.clip(overlay * 255.0, 0, 255).astype(np.uint8)
    )
    output.save(output_path)


def run_prediction(
    processor: Sam3Processor,
    image_path: Path,
    prompt: str,
) -> tuple[Image.Image, torch.Tensor, torch.Tensor]:
    image = Image.open(image_path).convert("RGB")

    state = processor.set_image(image)
    result = processor.set_text_prompt(
        prompt=prompt,
        state=state,
    )

    masks = result["masks"].detach().cpu()
    scores = result["scores"].detach().float().cpu().reshape(-1)

    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    return image, masks.bool(), scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clean", type=Path, required=True)
    parser.add_argument("--adversarial", type=Path, required=True)
    parser.add_argument("--prompt", default="car")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("checkpoints/sam3/sam3.pt"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/evaluation"),
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

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
        confidence_threshold=0.5,
    )

    clean_image, clean_masks, clean_scores = run_prediction(
        processor,
        args.clean,
        args.prompt,
    )

    adversarial_image, adversarial_masks, adversarial_scores = run_prediction(
        processor,
        args.adversarial,
        args.prompt,
    )

    if clean_scores.numel() == 0:
        raise RuntimeError("No clean detection was returned")

    clean_index = int(clean_scores.argmax().item())
    clean_mask = clean_masks[clean_index]
    clean_score = float(clean_scores[clean_index].item())

    best_adversarial_iou = 0.0
    best_adversarial_score = 0.0
    best_adversarial_index = None
    best_adversarial_mask = torch.zeros_like(clean_mask)

    for index in range(adversarial_scores.numel()):
        current_iou = binary_iou(
            clean_mask,
            adversarial_masks[index],
        )

        if current_iou > best_adversarial_iou:
            best_adversarial_iou = current_iou
            best_adversarial_score = float(
                adversarial_scores[index].item()
            )
            best_adversarial_index = index
            best_adversarial_mask = adversarial_masks[index]

    save_overlay(
        clean_image,
        clean_mask,
        args.output / "clean_overlay.png",
    )

    save_overlay(
        adversarial_image,
        best_adversarial_mask,
        args.output / "adversarial_overlay.png",
    )

    metrics = {
        "prompt": args.prompt,
        "clean_score": clean_score,
        "clean_detection_count": int(clean_scores.numel()),
        "adversarial_detection_count": int(
            adversarial_scores.numel()
        ),
        "adversarial_matching_index": best_adversarial_index,
        "adversarial_matching_score": best_adversarial_score,
        "clean_adversarial_mask_iou": best_adversarial_iou,
    }

    with (args.output / "metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(metrics, handle, indent=2)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
