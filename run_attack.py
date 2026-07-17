#!/usr/bin/env python3
"""Command-line entry point for one image or a directory of images."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from sam3_attack import AttackConfig, build_model, run_attack
from sam3_attack.io import find_images, load_image, save_result

ROAD_PROMPTS = [
    "car",
    "truck",
    "bus",
    "motorcycle",
    "bicycle",
    "person",
    "traffic light",
    "traffic sign",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Attack all SAM 3 instances matching one or more prompts."
    )
    parser.add_argument("--input", type=Path, required=True, help="Image or folder")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/sam3/sam3.pt"))
    parser.add_argument("--output", type=Path, default=Path("outputs"))
    parser.add_argument("--prompt", action="append", dest="prompts", help="Repeat for many prompts")
    parser.add_argument("--road-prompts", action="store_true", help="Use the built-in road-object list")
    parser.add_argument("--epsilon", type=float, default=8.0, help="L-infinity budget in 8-bit pixels")
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--amp", choices=("auto", "bf16", "fp16", "none"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prompts = ROAD_PROMPTS if args.road_prompts else (args.prompts or ["car"])
    images = find_images(args.input)
    if not images:
        raise RuntimeError(f"No supported images found in {args.input}")

    config = AttackConfig(
        epsilon_pixels=args.epsilon,
        step_size_pixels=args.step_size,
        steps=args.steps,
        restarts=args.restarts,
        confidence_threshold=args.threshold,
        amp=args.amp,
        seed=args.seed,
    )

    # Load the 848M-parameter model once, then reuse it for every image.
    model = build_model(args.checkpoint)

    for image_path in images:
        print(f"\n=== {image_path.name} ===")
        try:
            clean_tensor, clean_pil = load_image(image_path)
            result = run_attack(model, clean_tensor, prompts, config)
            output_dir = args.output / image_path.stem
            save_result(
                output_dir,
                clean_pil,
                result,
                metadata={
                    "image": str(image_path.resolve()),
                    "prompts_requested": prompts,
                    "epsilon_pixels": config.epsilon_pixels,
                    "step_size_pixels": config.step_size_pixels,
                    "steps": config.steps,
                    "restarts": config.restarts,
                    "seed": config.seed,
                    "torch_version": torch.__version__,
                    "gpu": torch.cuda.get_device_name(0),
                },
            )
            print(f"Saved to {output_dir}")
        except RuntimeError as exc:
            # A bad image or prompt should not stop a whole folder experiment.
            print(f"FAILED {image_path.name}: {exc}")
        finally:
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
